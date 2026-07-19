#!/usr/bin/env python3
"""
=============================================================================
DIAGNOSE_SPATIAL_USE.PY  —  is the trained model actually USING the spatial
                            conditioning channels, or has it learned to
                            ignore them?
=============================================================================

WHY THIS EXISTS
---------------
The smoke test proved that the 3 spatial conditioning channels (obs_mask,
env, obs_decay) are WIRED INTO the U-Net's input_proj. The 79-epoch
training_history showed val_diffusion plateau and AUC oscillating across all
4 phases — symptoms of "model has learned to denoise but not to use the new
spatial information."  Hypothesis 1: at high diffusion timesteps t, the
noisy x_t channel has std ~1 while the bounded spatial channels are in
[0,1]; the input Conv2d can simply down-weight the spatial channels and
satisfy the eps-prediction objective from x_t alone.

This script TESTS that hypothesis directly. It is NOT a post-hoc analysis
of saved NPZ files (use diagnose_recon_axel_map.py for that). It runs the
LIVE trained model on validation worlds under four conditions:

    FULL        : normal build_spatial_cond (all 3 channels carry signal)
    NO_OBS      : zero channel 0 (obs_mask) AND channel 2 (obs_decay)
                  -> model loses access to WHERE the K observations are
    NO_ENV      : zero channel 1 (env)
                  -> model loses access to per-species suitability
    NO_SPATIAL  : zero ALL 3 channels
                  -> model has only x_t and the per-species FiLM vector

For each condition we compute:
    Jaccard@0.5 vs truth on the full grid
    AUC at UNOBSERVED cells vs truth  <-- Axel's metric
    |pred - pred_FULL|.mean()         <-- decisive: how much does the
                                          output even CHANGE when channels
                                          are masked?

CRITICAL: we sample with mode='extrapolate' so the inpainting overlay
(which would pin obs cells to ground truth regardless of model behavior)
does NOT confound the ablation. With inpaint_strength=0.0, the ONLY way
observations can influence the prediction is through the spatial channels.
That isolation is the point of the diagnostic.

VERDICT LOGIC
-------------
  |pred_NO_SPATIAL - pred_FULL|.mean() < 0.02   AND
  Jaccard(NO_SPATIAL) ~= Jaccard(FULL)
       -> MODEL IGNORES SPATIAL CHANNELS. Hypothesis 1 confirmed.
          Fix: zero-init gated injection in input_proj (ControlNet-style).

  Jaccard(NO_OBS) << Jaccard(FULL)
       -> Model DOES use obs channels. The Jaccard~0 failure has a
          different cause (data, loss, training regime).

  Mixed signal
       -> Need to look at the numbers and think.

USAGE
-----
    python diagnose_spatial_use.py \
        --checkpoint  stage2_outputs_spatial_newest/checkpoints/best_model.pt \
        --simulation-dir results/data \
        --output-dir ./diagnose_out \
        --n-worlds 5 --n-samples 4 --ddim-steps 25
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="path to best_model.pt or last_checkpoint.pt")
    p.add_argument("--simulation-dir", required=True,
                   help="results/data — the IBM .npz worlds")
    p.add_argument("--stage2-dir", default=None,
                   help="path to AI_simulation/stage2 (default: relative to this file)")
    p.add_argument("--output-dir", default="./diagnose_out")
    p.add_argument("--n-worlds", type=int, default=5,
                   help="number of validation worlds to diagnose")
    p.add_argument("--n-samples", type=int, default=4,
                   help="ensemble size per condition")
    p.add_argument("--ddim-steps", type=int, default=25,
                   help="DDIM steps (25 is enough for diagnostic)")
    p.add_argument("--eta", type=float, default=0.15)
    p.add_argument("--device", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------
def make_masking_build_fn(model, original_build, zero_channels):
    """
    Return a callable that replaces model.build_spatial_cond. It calls the
    original method and zeros out the specified channels of cond_spatial
    (shape (B, S, 3, Y, X), channel layout [obs_mask, env, obs_decay]).
    """
    def patched_build(condition):
        cs = original_build(condition)         # (B, S, 3, Y, X)
        if zero_channels:
            cs = cs.clone()
            for ch in zero_channels:
                cs[:, :, ch] = 0.0
        return cs
    return patched_build


# --------------------------------------------------------------------------
def jaccard_at_threshold(pred, truth, thr=0.5, mask=None):
    """
    pred  : (S, Y, X) float in [0,1]
    truth : (S, Y, X) {0,1}
    mask  : optional (S, Y, X) bool — only consider these cells
    Returns mean Jaccard over species that have at least one truth-cell.
    """
    pb = (pred > thr)
    tb = (truth > 0.5)
    if mask is not None:
        pb = pb & mask
        tb = tb & mask
    # per species
    inter = (pb & tb).reshape(pb.shape[0], -1).sum(-1).astype(np.float64)
    union = (pb | tb).reshape(pb.shape[0], -1).sum(-1).astype(np.float64)
    # only species with truth presence
    valid = (tb.reshape(pb.shape[0], -1).sum(-1) > 0)
    if valid.sum() == 0:
        return 0.0
    j = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return float(j[valid].mean())


def auc_at_unobs(pred, truth, obs_mask):
    """
    AUC computed ONLY at unobserved cells (where obs_mask is 0).
    pred, truth, obs_mask : (S, Y, X)
    Per-species AUC averaged over species where there is at least one
    positive and one negative at the unobserved cells.
    """
    from sklearn.metrics import roc_auc_score
    S = pred.shape[0]
    aucs = []
    for s in range(S):
        unobs = (obs_mask[s] < 0.5)
        if unobs.sum() < 5:
            continue
        y = truth[s][unobs].astype(int).ravel()
        p = pred[s][unobs].astype(float).ravel()
        if y.sum() < 1 or y.sum() == len(y):
            continue
        try:
            aucs.append(roc_auc_score(y, p))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.0


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    s2 = (Path(args.stage2_dir).resolve()
          if args.stage2_dir else Path(__file__).resolve().parent)
    for p in (s2, s2 / "models", s2 / "configs"):
        sys.path.insert(0, str(p))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- imports happen AFTER sys.path is set --------------------------
    from configs.config import get_default_config
    from models.ecodiffusion_spatial_cond import create_spatial_cond_model
    from ecodiffusion_sample_spatial import sample_spatial
    from data_preprocessing import create_dataloaders
    import data_preprocessing_v2_patch
    import interaction_encoder_v2_patch  # noqa: F401  — installs the patch
    data_preprocessing_v2_patch.auto_install(history_length=10)

    # ---- build phase-4 ('infill') dataloader: this is the realistic regime
    print(f"\n  Building phase-4 (infill) val dataloader...")
    cfg = get_default_config()
    cfg.paths.simulation_dir = args.simulation_dir
    cfg.training.batch_size = 1
    cfg.training.hist_sparsify_K = [5, 10, 20]
    _, val_loader, _, _ = create_dataloaders(args.simulation_dir, cfg,
                                             mode='infill')
    print(f"  val worlds available: {len(val_loader)}")

    # ---- build & load model -------------------------------------------
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    sample_batch = next(iter(val_loader))
    cfg.data.n_species_max = sample_batch['target'].shape[1]
    model = create_spatial_cond_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.set_training_phase(4)
    print(f"    epoch: {ckpt.get('epoch', '?')}   "
          f"best_metric: {ckpt.get('best_metric', '?'):.4f}   "
          f"params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    input_proj in_channels = {model.unet.input_proj.in_channels} "
          f"(1 + {model.n_cond_channels} spatial)")

    # ---- the four ablation conditions ---------------------------------
    conditions = {
        'FULL':       [],         # no channels zeroed
        'NO_OBS':     [0, 2],     # zero obs_mask and obs_decay
        'NO_ENV':     [1],        # zero env channel
        'NO_SPATIAL': [0, 1, 2],  # zero all three
    }

    original_build = model.build_spatial_cond
    results = []                       # list of dicts, one per (world, condition)

    # ---- loop over worlds × conditions --------------------------------
    val_iter = iter(val_loader)
    n_worlds_done = 0
    t0 = time.time()
    for world_idx in range(args.n_worlds):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        target = batch['target'].to(device)            # (B, S, Y, X)
        cond = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch['condition'].items()}

        # build the unmasked spatial cond once, save obs_mask for evaluation
        with torch.no_grad():
            cs_full = original_build(cond)             # (B, S, 3, Y, X)
        obs_mask_real = cs_full[:, :, 0].cpu().numpy() # (B, S, Y, X)

        truth_np = target.cpu().numpy()                # (B, S, Y, X)

        preds_by_cond = {}
        for cond_name, channels_to_zero in conditions.items():
            # monkey-patch build_spatial_cond
            model.build_spatial_cond = make_masking_build_fn(
                model, original_build, channels_to_zero)
            try:
                with torch.no_grad():
                    samples = sample_spatial(
                        model, cond,
                        n_samples=args.n_samples,
                        ddim_steps=args.ddim_steps,
                        eta=args.eta,
                        mode='extrapolate',   # CRITICAL: no inpainting overlay
                        verbose=False,
                    )                          # (n_samples, B, S, Y, X)
            finally:
                model.build_spatial_cond = original_build

            ens_mean = samples.mean(dim=0).cpu().numpy()   # (B, S, Y, X)
            preds_by_cond[cond_name] = ens_mean

        # compute metrics for each condition on this world (use batch 0)
        full_pred = preds_by_cond['FULL'][0]               # (S, Y, X)
        truth_b = truth_np[0]
        obs_b = obs_mask_real[0]

        for cond_name, ens_mean in preds_by_cond.items():
            pred = ens_mean[0]                              # (S, Y, X)
            jac = jaccard_at_threshold(pred, truth_b, thr=0.5)
            auc_un = auc_at_unobs(pred, truth_b, obs_b)
            diff_full = float(np.mean(np.abs(pred - full_pred)))
            n_cells = int((pred > 0.5).sum())
            mean_prob = float(pred.mean())
            results.append({
                'world':       world_idx,
                'condition':   cond_name,
                'jaccard':     jac,
                'auc_unobs':   auc_un,
                'diff_full':   diff_full,
                'cells>0.5':   n_cells,
                'mean_prob':   mean_prob,
            })

        n_worlds_done += 1
        dt = time.time() - t0
        print(f"\n  world {world_idx + 1}/{args.n_worlds} done  "
              f"(elapsed {dt:.0f}s, ~{dt/n_worlds_done:.0f}s/world)")
        print(f"    {'cond':<12} {'jaccard':>8} {'auc_unobs':>10}"
              f" {'|d FULL|':>10} {'cells>.5':>10}")
        for cond_name in conditions:
            r = [r for r in results
                 if r['world'] == world_idx and r['condition'] == cond_name][0]
            print(f"    {r['condition']:<12} {r['jaccard']:>8.4f}"
                  f" {r['auc_unobs']:>10.4f} {r['diff_full']:>10.5f}"
                  f" {r['cells>0.5']:>10,}")

    # ---- summarise across worlds --------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY (mean across worlds)")
    print("=" * 72)
    print(f"  {'condition':<12} {'jaccard':>8} {'auc_unobs':>10}"
          f" {'|d FULL|':>10}")
    means = {}
    for cond_name in conditions:
        rs = [r for r in results if r['condition'] == cond_name]
        m_jac = np.mean([r['jaccard'] for r in rs])
        m_auc = np.mean([r['auc_unobs'] for r in rs])
        m_diff = np.mean([r['diff_full'] for r in rs])
        means[cond_name] = {'jaccard': m_jac, 'auc_unobs': m_auc,
                            'diff_full': m_diff}
        print(f"  {cond_name:<12} {m_jac:>8.4f} {m_auc:>10.4f}"
              f" {m_diff:>10.5f}")

    # ---- verdict -------------------------------------------------------
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    j_full = means['FULL']['jaccard']
    j_no_obs = means['NO_OBS']['jaccard']
    j_no_env = means['NO_ENV']['jaccard']
    j_no_sp = means['NO_SPATIAL']['jaccard']
    d_no_obs = means['NO_OBS']['diff_full']
    d_no_env = means['NO_ENV']['diff_full']
    d_no_sp = means['NO_SPATIAL']['diff_full']

    # Decisive number: does the model output even CHANGE when channels masked?
    print(f"\n  Decisive number — how much does the prediction change when")
    print(f"  spatial channels are zeroed?")
    print(f"    |pred - FULL|  (all 3 zeroed)  = {d_no_sp:.5f}")
    print(f"    |pred - FULL|  (obs zeroed)    = {d_no_obs:.5f}")
    print(f"    |pred - FULL|  (env zeroed)    = {d_no_env:.5f}")
    print(f"  Interpretation:")
    print(f"    < 0.01   model is essentially ignoring those channels")
    print(f"    < 0.03   weak influence")
    print(f"    > 0.05   meaningful influence")

    print(f"\n  Jaccard FULL = {j_full:.4f}")
    print(f"  Jaccard delta when channels masked:")
    print(f"    FULL - NO_OBS     = {j_full - j_no_obs:+.4f}")
    print(f"    FULL - NO_ENV     = {j_full - j_no_env:+.4f}")
    print(f"    FULL - NO_SPATIAL = {j_full - j_no_sp:+.4f}")

    print()
    if d_no_sp < 0.01 and abs(j_full - j_no_sp) < 0.005:
        print("  >>> MODEL IGNORES THE SPATIAL CHANNELS <<<")
        print("      Hypothesis 1 CONFIRMED. The U-Net learned to denoise from")
        print("      x_t alone and treats obs_mask/env/obs_decay as noise.")
        print()
        print("      Most likely cause: noisy x_t (std ~1 at high t) dominates")
        print("      the bounded spatial channels ([0,1]) in the input Conv2d.")
        print("      The optimiser found that ignoring spatial channels is a")
        print("      local optimum that satisfies the eps-prediction loss.")
        print()
        print("      Fix: zero-initialised GATED injection in input_proj —")
        print("      keep x_t through its own Conv2d (standard initialisation),")
        print("      add the 3 spatial channels through a SEPARATE Conv2d whose")
        print("      output is initially zero. Spatial influence starts at 0")
        print("      and the model learns to dial it up where it helps. This is")
        print("      ControlNet's mechanism, used precisely because input")
        print("      concatenation under heavy noise has this failure mode.")
    elif d_no_sp > 0.05 and (j_full - j_no_sp) > 0.01:
        print("  >>> Model DOES use spatial channels. Bug is elsewhere. <<<")
        print("      Spatial channels measurably influence the output AND")
        print("      Jaccard with FULL is meaningfully better than NO_SPATIAL.")
        print("      Look at:")
        print("      - data pipeline (is the sparse obs_mask placed correctly?)")
        print("      - loss function (does it reward spatial precision?)")
        print("      - K=5/10/20 sparsification (too sparse to learn from?)")
    else:
        print("  AMBIGUOUS: weak signal. Interpret the numbers above directly.")
        print("  If diff_full < 0.02 but Jaccard differs slightly, the spatial")
        print("  channels have small influence — still warrants the zero-init")
        print("  gated injection fix, just less dramatically.")

    # ---- save raw results ---------------------------------------------
    out = {'args': vars(args), 'means': means, 'per_world': results}
    json_path = out_dir / "diagnostic_spatial_use.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  results saved -> {json_path}")


if __name__ == "__main__":
    main()