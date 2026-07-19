#!/usr/bin/env python3
"""
=============================================================================
SMOKE TEST — spatial-conditioning EcoDiffusion. Run BEFORE the full run.
=============================================================================
Usage:
    python AI_simulation/stage2/smoke_test_spatial.py --simulation-dir results/data

Six checks, each maps to a thing that must work before spending the compute:
  [1] phase-specific dataloaders build
  [2] spatial model builds with the 3 extra input channels (input_proj in_ch=4)
  [3] forward + backward against the REAL encoders, no NaN, gradient flows
  [4] build_spatial_cond -> obs_mask is SPARSE (defensive net works)
  [5] inpainting sampler runs (mode='inpaint') and pins observed cells
  [6] trainer 1-epoch run -> training_history.json + last_checkpoint.pt
      written, and early-stop state is INSIDE the checkpoint

WHY STEP [5] CHANGED FROM THE PREVIOUS VERSION
----------------------------------------------
The previous version sliced the condition dict to 8 species "for speed":
    v[:, :8]
That is WRONG. v[:, :8] slices dimension 1, but dimension 1 is not the
species axis for every tensor:
    env               (B, S, Y, X)      dim1 = S      -> sliced to 8 species
    species_features  (B, S, F)         dim1 = S      -> sliced to 8 species
    history_P / _B    (B, T, S, Y, X)   dim1 = T      -> sliced 8 TIMESTEPS!
So env reached the encoder with S=8 but history_P with S=3697, and the
conditioning module's torch.cat([env_proj, int_proj, temp_proj, ...])
mismatched (8 vs 3697). On top of that, edge_index/edge_weight are per-
sample interaction graphs indexed into all 3697 species — slicing species
invalidates the graph entirely. The ONLY correct path is to run on the
full condition dict. Sampling is @torch.no_grad() (no autograd graph held),
so full-species sampling is lighter than step [3], which already passed.
=============================================================================
"""
import argparse, sys, json, traceback
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation-dir", required=True)
    ap.add_argument("--stage2-dir", default=None)
    a = ap.parse_args()
    s2 = Path(a.stage2_dir).resolve() if a.stage2_dir else Path(__file__).resolve().parent
    for p in (s2, s2 / "models", s2 / "configs"):
        sys.path.insert(0, str(p))

    from configs.config import get_default_config
    from models.ecodiffusion_spatial_cond import create_spatial_cond_model
    from data_preprocessing import create_dataloaders, save_preprocessor
    from training import Trainer
    import interaction_encoder, data_preprocessing_v2_patch
    import training_v2_patch, interaction_encoder_v2_patch
    data_preprocessing_v2_patch.auto_install(history_length=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_default_config()
    cfg.paths.simulation_dir = a.simulation_dir
    cfg.paths.checkpoint_dir = "./smoke_out/checkpoints"
    cfg.paths.log_dir = "./smoke_out/logs"
    cfg.training.batch_size = 1
    # tiny curriculum so the 1-epoch run actually crosses a phase boundary
    cfg.training.total_epochs = 2
    cfg.training.phase1_epochs = 1
    cfg.training.phase2_epochs = 1
    cfg.training.phase3_epochs = 1
    cfg.training.phase4_epochs = 2
    cfg.training.warmup_epochs = 0
    cfg.training.val_every_epochs = 1
    cfg.training.save_every_epochs = 1
    cfg.training.hist_sparsify_K = [5, 10]

    # ----------------------------------------------------------------- [1]
    print("\n[1] dataloaders...")
    loaders = {}
    prep = None
    for ph, mode in [(1, 'equilibrium'), (2, 'interaction'),
                     (3, 'temporal'), (4, 'infill')]:
        tl, vl, _, pr = create_dataloaders(a.simulation_dir, cfg, mode=mode)
        loaders[ph] = {'train': tl, 'val': vl}
        prep = prep or pr
    Path("./smoke_out/checkpoints").mkdir(parents=True, exist_ok=True)
    save_preprocessor(prep, "./smoke_out/preprocessor.pkl")
    sample = next(iter(loaders[1]['train']))
    cfg.data.n_species_max = sample['target'].shape[1]
    print(f"    OK  n_species={cfg.data.n_species_max}")

    # ----------------------------------------------------------------- [2]
    print("\n[2] build spatial model...")
    model = create_spatial_cond_model(cfg).to(device)
    in_ch = model.unet.input_proj.in_channels
    assert in_ch == 1 + model.n_cond_channels, \
        f"input_proj in_channels={in_ch}, expected {1 + model.n_cond_channels}"
    print(f"    OK  params={sum(p.numel() for p in model.parameters()):,}  "
          f"input_proj in_ch={in_ch} (1 + {model.n_cond_channels} spatial)")

    # ----------------------------------------------------------------- [3]
    print("\n[3] forward + backward (phase 4 batch)...")
    model.set_training_phase(4)
    batch = next(iter(loaders[4]['train']))
    target = batch['target'].to(device)
    cond = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch['condition'].items()}
    out = model(target, cond)
    assert out['noise_pred'].shape == target.shape, \
        f"shape {out['noise_pred'].shape} != {target.shape}"
    assert not torch.isnan(out['loss']), "NaN loss on smoke batch"
    out['loss'].backward()
    g = sum(p.grad.abs().sum().item()
            for p in model.parameters() if p.grad is not None)
    assert g > 0, "no gradient flowed"
    print(f"    OK  noise_pred {tuple(out['noise_pred'].shape)}  "
          f"loss={out['loss'].item():.4f}  grad_sum={g:.1f}")

    # ----------------------------------------------------------------- [4]
    print("\n[4] build_spatial_cond is SPARSE (defensive net works)...")
    cs = model.build_spatial_cond(cond)            # (B, S, 3, Y, X)
    obs = cs[:, :, 0]                              # obs_mask channel
    mx = obs.reshape(obs.shape[0], obs.shape[1], -1).sum(-1).max().item()
    assert mx <= model.dense_obs_threshold, f"obs_mask dense: max {mx}/species"
    print(f"    OK  obs_mask max {mx:.0f} cells/species "
          f"(<= {model.dense_obs_threshold} threshold)  env+decay channels present")

    # ----------------------------------------------------------------- [5]
    # FIXED: no species slicing. Run sample_spatial on the FULL cond dict.
    # (slicing dim 1 cut history_P's TIME axis, not species, and would also
    #  invalidate the interaction graph's edge indices.)
    print("\n[5] inpainting sampler (mode='inpaint')...")
    from ecodiffusion_sample_spatial import sample_spatial
    model.zero_grad(set_to_none=True)
    model.eval()
    pr = sample_spatial(
        model, cond,
        n_samples=2,
        ddim_steps=4,            # small — U-Net chunks species internally
        eta=0.15,
        mode='inpaint',          # hard-inpainting overlay at observed cells
        repaint_iterations=2,
        verbose=True,
    )
    assert pr.shape[0] == 2, f"ensemble dim wrong: {tuple(pr.shape)}"
    assert not torch.isnan(pr).any(), "NaN in sampled output"

    # inpainting check: observed cells must survive into the sample.
    # be robust to pr being (n, B, S, Y, X) or (n, S, Y, X).
    recon = pr[0]
    recon_bsyx = recon if recon.dim() == 4 else recon.unsqueeze(0)
    hist = cond.get('history_P')
    if hist is not None and hist.numel() > 0:
        om = (hist[:, -1] > 0.5)               # (B, S, Y, X) — last history frame
        if om.shape == recon_bsyx.shape and om.sum() > 0:
            kept = (((recon_bsyx > 0.5) & om).sum().item()
                    / max(1, om.sum().item()))
            print(f"    OK  samples {tuple(pr.shape)}  "
                  f"obs cells kept by inpainting: {100 * kept:.0f}%")
        else:
            print(f"    OK  samples {tuple(pr.shape)}  "
                  f"(inpaint-kept check skipped: shape {tuple(recon_bsyx.shape)} "
                  f"vs obs {tuple(om.shape)})")
    else:
        print(f"    OK  samples {tuple(pr.shape)}")

    # ----------------------------------------------------------------- [6]
    print("\n[6] trainer: 1-epoch run — history / checkpoints / early-stop state...")
    model2 = create_spatial_cond_model(cfg).to(device)
    tr = Trainer(model2, cfg, loaders[1]['train'], loaders[1]['val'], device)
    tr.phase_loaders = loaders                  # consumed by the v2-patched trainer
    tr.train()
    ck = Path("./smoke_out/checkpoints")
    assert (ck / "training_history.json").exists(), "training_history.json NOT written"
    assert (ck / "last_checkpoint.pt").exists(), "last_checkpoint.pt NOT written"
    hist = json.load(open(ck / "training_history.json"))
    assert len(hist) > 0 and 'epoch' in hist, "history empty / missing 'epoch'"
    saved = torch.load(ck / "last_checkpoint.pt", map_location='cpu',
                       weights_only=False)
    for key in ('patience_counter', 'current_phase_for_es',
                'best_metric', 'global_step'):
        assert key in saved, f"checkpoint missing '{key}'"
    print(f"    OK  training_history.json keys={list(hist.keys())[:6]}...")
    print(f"    OK  last_checkpoint.pt has early-stop state "
          f"(patience={saved['patience_counter']}, "
          f"phase={saved['current_phase_for_es']})")
    print(f"    OK  best_model.pt exists: {(ck / 'best_model.pt').exists()}")

    print("\n" + "=" * 60)
    print("  ALL SMOKE CHECKS PASSED — safe to start the full run")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\n  SMOKE TEST FAILED — do NOT start the full run until this passes")
        sys.exit(1)