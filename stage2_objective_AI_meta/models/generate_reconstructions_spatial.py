#!/usr/bin/env python3
"""
=============================================================================
GENERATE RECONSTRUCTIONS — SPATIAL CONDITIONING + Fix A+B
=============================================================================

Identical inputs, outputs, and CLI to generate_reconstructions_v7_inpaint.py.
The SAME NPZ schema (`samples`, `mean`, `noisy_input`) is produced, so every
downstream evaluation/figure script that read v7 NPZs reads these unchanged.

CHANGES FROM v7 (surgical, two imports + one assert + two defaults)
-------------------------------------------------------------------
  v7  : from models.ecodiffusion             import create_fixed_model
  HERE: from models.ecodiffusion_spatial_cond import create_spatial_cond_model

  v7  : from ecodiffusion_sample_v7_inpaint  import sample_v7
  HERE: from ecodiffusion_sample_spatial      import sample_spatial

  v7  : --eta default 0.5
  HERE: --eta default 0.15  (matches sample_spatial default)

  v7  : --output-dir default ./reconstructions_v7_inpaint
  HERE: --output-dir default ./reconstructions_spatial

Plus a defensive assert that input_proj.in_channels == 4 (the spatial model's
signature: 1 noisy + 3 spatial channels). Fires if the wrong checkpoint is
loaded, before any sampling work is done.

USAGE
-----
  Copy ecodiffusion_sample_spatial.py to AI_simulation/stage2/models/
  (this script copies it automatically if it isn't there).

    python generate_reconstructions_spatial.py \\
        --stage2-dir       AI_simulation/stage2 \\
        --checkpoint       ./stage2_outputs_SPATIAL_FIXAB_partial/checkpoints/FIXAB_epoch109_BACKUP.pt \\
        --truth-npz        ./results/data/<world>.npz \\
        --output-dir       ./reconstructions_spatial/<world_stem> \\
        --fixed-budgets    5 10 \\
        --n-ensemble       8 \\
        --mode             inpaint \\
        --ddim-steps       50 \\
        --eta              0.15

DOWNSTREAM COMPATIBILITY
------------------------
Output keys are identical to v7:
   *_samples.npz -> samples, mean, noisy_input, sample_mode, n_ensemble
   *.npz         -> mean, noisy_input, sample_mode, n_ensemble

These downstream scripts therefore run UNCHANGED on this script's output:
  compute_ensemble_coverage_stratified.py
  multi_world_v7_evaluation.py
  multi_world_2x_evaluation.py
  make_figure1_honest_map.py  (and axel_distribution_tests_inpaint.py helper)
  diagnose_recon_axel_map.py
=============================================================================
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def find_stage2_dir(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).resolve())
    here = Path(__file__).resolve().parent
    candidates.extend([here.parent, here, here.parent.parent])
    cwd = Path.cwd()
    for k in range(4):
        candidates.append(cwd if k == 0 else
                          (cwd.parents[k - 1] if len(cwd.parents) >= k else cwd))
        candidates.append(cwd / "AI_simulation" / "stage2")
        candidates.append(cwd / "stage2")
    seen, unique = set(), []
    for c in candidates:
        cr = c.resolve()
        if cr not in seen:
            seen.add(cr); unique.append(cr)
    for c in unique:
        if (c / "models" / "ecodiffusion.py").exists() \
           and (c / "configs" / "config.py").exists():
            print(f"  ✓ Stage 2 source tree: {c}")
            sys.path.insert(0, str(c))
            return c
    return None


def sparsify_history_fixed_budget(Pt, budget, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    T, S, Y, X = Pt.shape
    Pt_sparse = np.zeros_like(Pt)
    last = Pt[-1]
    sparse_last = np.zeros_like(last)
    for s in range(S):
        occupied = np.argwhere(last[s] > 0)
        if len(occupied) == 0:
            continue
        n_keep = min(budget, len(occupied))
        chosen = rng.choice(len(occupied), size=n_keep, replace=False)
        for idx in chosen:
            y, x = occupied[idx]
            sparse_last[s, y, x] = 1.0
    Pt_sparse[-1] = sparse_last
    return Pt_sparse


def build_condition(npz_data, device, species_subset, sparse_Pt):
    import torch
    Y, X = int(npz_data["Y"]), int(npz_data["X"])
    P = np.asarray(npz_data["P_last_final"])
    S_data = P.shape[0]
    if species_subset is None:
        species_subset = np.arange(S_data)
    S = len(species_subset)
    B = 1

    env_raw = np.asarray(npz_data["ENV_r_field"]).astype(np.float32)
    if env_raw.ndim == 2:
        env_raw = np.broadcast_to(env_raw[None], (S, Y, X)).copy()
    elif env_raw.ndim == 3 and env_raw.shape[0] >= S_data:
        env_raw = env_raw[species_subset]
    env_t = torch.from_numpy(env_raw[np.newaxis].copy()).float().to(device)

    y_grid = np.broadcast_to(
        np.arange(Y, dtype=np.float32).reshape(1, Y, 1), (B, Y, X)).copy()
    x_grid = np.broadcast_to(
        np.arange(X, dtype=np.float32).reshape(1, 1, X), (B, Y, X)).copy()
    y_coords_t = torch.from_numpy(y_grid).float().to(device)
    x_coords_t = torch.from_numpy(x_grid).float().to(device)

    pv = np.asarray(npz_data.get("prevalence_final", np.zeros(S_data)))[species_subset]
    sp_feats = np.zeros((B, S, 8), dtype=np.float32)
    for s in range(S):
        v = float(pv[s])
        sp_feats[0, s, 0] = v
        sp_feats[0, s, 1] = float(np.log1p(v * 400) / 8.0) if v > 0 else 0.0
        sp_feats[0, s, 2] = float(v > 0)
        sp_feats[0, s, 3] = 1.0 / (1.0 + v * 100) if v > 0 else 0
        sp_feats[0, s, 4] = float(v < 0.05)
        sp_feats[0, s, 5] = float(v >= 0.05)
        sp_feats[0, s, 6] = np.log1p(v)
        sp_feats[0, s, 7] = float(s) / max(S - 1, 1)
    species_features_t = torch.from_numpy(sp_feats).float().to(device)

    edge_index_t = None
    if "C_topk_idx" in npz_data:
        ctk = np.asarray(npz_data["C_topk_idx"])
        edge_list = []
        subset_list = species_subset.tolist() if hasattr(species_subset, 'tolist') \
            else list(species_subset)
        subset_set = set(subset_list)
        old_to_new = {old: new for new, old in enumerate(subset_list)}
        for new_s, old_s in enumerate(subset_list):
            if old_s < ctk.shape[0]:
                for neighbor in ctk[old_s]:
                    if int(neighbor) in subset_set:
                        edge_list.append([new_s, old_to_new[int(neighbor)]])
        edge_index_t = (
            torch.tensor(edge_list, dtype=torch.long, device=device).T
            if edge_list else
            torch.empty(2, 0, dtype=torch.long, device=device)
        )

    Pt_chunk = sparse_Pt[:, species_subset, :, :]
    history_t = torch.from_numpy(Pt_chunk[np.newaxis].copy()).float().to(device)

    cond = {
        "env": env_t, "y_coords": y_coords_t, "x_coords": x_coords_t,
        "species_features": species_features_t, "history_P": history_t,
    }
    if edge_index_t is not None:
        cond["edge_index"] = edge_index_t
        cond["edge_weight"] = None
    return cond


def run_inference_v7(model, npz_data, device, sparse_Pt,
                     n_ensemble=8, chunk_size=200,
                     ddim_steps=50, eta=0.5,
                     mode='inpaint', repaint_iterations=2,
                     verbose=False):
    """Inference using v7 inpainting sampler."""
    import torch
    from ecodiffusion_sample_spatial   import sample_spatial

    P = np.asarray(npz_data["P_last_final"])
    S = P.shape[0]
    Y, X = int(npz_data["Y"]), int(npz_data["X"])

    all_preds = np.zeros((S, Y, X), dtype=np.float32)
    all_samples = np.zeros((n_ensemble, S, Y, X), dtype=np.float32)

    if hasattr(model, "set_training_phase"):
        model.set_training_phase(4)
    model.eval()

    n_chunks = (S + chunk_size - 1) // chunk_size
    with torch.no_grad():
        for ci, chunk_start in enumerate(range(0, S, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, S)
            chunk_indices = np.arange(chunk_start, chunk_end)
            chunk_S = len(chunk_indices)
            print(f"      chunk {ci+1}/{n_chunks}: species "
                  f"[{chunk_start}:{chunk_end}] (n={chunk_S}), "
                  f"ensemble={n_ensemble}, mode={mode}", flush=True)

            condition = build_condition(npz_data, device, chunk_indices, sparse_Pt)

            preds = sample_spatial(
                model=model, condition=condition,
                n_samples=n_ensemble, ddim_steps=ddim_steps, eta=eta,
                mode=mode, repaint_iterations=repaint_iterations,
                verbose=(verbose and ci == 0),
            )  # (n_ensemble, B=1, chunk_S, Y, X)

            preds_np = preds.squeeze(1).detach().cpu().numpy()
            preds_np = np.clip(preds_np, 0, 1).astype(np.float32)

            all_preds[chunk_start:chunk_end] = preds_np.mean(axis=0)
            all_samples[:, chunk_start:chunk_end, :, :] = preds_np

    return all_preds, all_samples


def load_model(checkpoint_path, truth_npz_path, device):
    import torch
    print(f"\n  Loading checkpoint: {checkpoint_path}")
    from models.ecodiffusion_spatial_cond import create_spatial_cond_model
    from configs.config import get_default_config

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    config = get_default_config()
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    detected_n = None
    for k, v in state_dict.items():
        if 'env_encoder' in k and 'species_embed' in k:
            detected_n = int(v.shape[0])
            config.data.n_species_max = detected_n
            break
    if not detected_n or config.data.n_species_max == 0:
        with np.load(truth_npz_path, allow_pickle=True) as d:
            detected_n = int(np.asarray(d["P_last_final"]).shape[0])
            config.data.n_species_max = detected_n

    model = create_spatial_cond_model(config)
    # SAFETY: the spatial model has 4 input channels (1 noisy + 3 spatial).
    # If a non-spatial checkpoint is loaded by mistake this assert fires
    # BEFORE any sampling work is done.
    assert model.unet.input_proj.in_channels == 4, (
        f'expected EcoDiffusionSpatial (input_proj.in_channels == 4), got '
        f'{model.unet.input_proj.in_channels}. Did you load a non-spatial checkpoint?'
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device); model.eval()
    print(f"  ✓ Model loaded: "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--truth-npz", required=True)
    p.add_argument("--output-dir", default="./reconstructions_spatial")
    p.add_argument("--fixed-budgets", type=int, nargs="*", default=[5, 10])
    p.add_argument("--mode", default="inpaint",
                   choices=["inpaint", "soft_inpaint", "extrapolate"])
    p.add_argument("--repaint-iterations", type=int, default=2,
                   help="RePaint refinement iterations per step (1-5)")
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.15)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("\n  " + "=" * 70)
    print("  generate_reconstructions_v7  —  INPAINTING-GUIDED")
    print("  " + "=" * 70)

    stage2_dir = find_stage2_dir(args.stage2_dir)
    if stage2_dir is None:
        print("  ✗ Could not find Stage 2 source tree.")
        return 1

    # Make sample_spatial importable
    sample_spatial_src = Path(__file__).parent / "ecodiffusion_sample_spatial.py"
    sample_spatial_dst = stage2_dir / "models" / "ecodiffusion_sample_spatial.py"
    if sample_spatial_src.exists() and not sample_spatial_dst.exists():
        import shutil
        shutil.copy(sample_spatial_src, sample_spatial_dst)
        print(f"  ✓ copied sample_spatial to {sample_spatial_dst}")

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(stage2_dir / "models"))

    import torch
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device: {device}")
    print(f"  mode: {args.mode}, repaint iterations: {args.repaint_iterations}")

    truth_path = Path(args.truth_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out: {out_dir.resolve()}")

    npz_data = dict(np.load(truth_path, allow_pickle=True))
    Pt_full = np.asarray(npz_data["P_t"]).astype(np.float32)
    print(f"  S={Pt_full.shape[1]}  T={Pt_full.shape[0]}")

    model = load_model(args.checkpoint, truth_path, device)

    for K in args.fixed_budgets:
        print(f"\n  K = {K} per present species")
        sparse_Pt = sparsify_history_fixed_budget(
            Pt_full, K, rng_seed=args.rng_seed)
        n_obs = int(sparse_Pt[-1].sum())
        print(f"    observed cells: {n_obs}")

        save_path = out_dir / f"recon_fixed_b{K}.npz"
        preds, samples = run_inference_v7(
            model, npz_data, device, sparse_Pt,
            n_ensemble=args.n_ensemble, chunk_size=args.chunk_size,
            ddim_steps=args.ddim_steps, eta=args.eta,
            mode=args.mode, repaint_iterations=args.repaint_iterations,
            verbose=args.verbose,
        )

        # Save mean
        np.savez_compressed(
            save_path,
            mean=preds.astype(np.float32),
            noisy_input=sparse_Pt[-1].astype(np.float32),
            sample_mode=str(args.mode),
            n_ensemble=int(args.n_ensemble),
        )
        sz_mb = save_path.stat().st_size / 1024 / 1024
        print(f"      ✓ saved {save_path.name}  shape={preds.shape}  "
              f"size={sz_mb:.1f} MB  mean={preds.mean():.4f}  "
              f"frac>0.5={(preds > 0.5).mean():.4f}")

        # Save samples
        samples_path = save_path.parent / (save_path.stem + '_samples.npz')
        np.savez_compressed(
            samples_path,
            samples=samples.astype(np.float32),
            mean=preds.astype(np.float32),
            noisy_input=sparse_Pt[-1].astype(np.float32),
            sample_mode=str(args.mode),
            n_ensemble=int(args.n_ensemble),
        )
        sz_mb = samples_path.stat().st_size / 1024 / 1024
        print(f"      ✓ saved {samples_path.name}  shape={samples.shape}  "
              f"size={sz_mb:.1f} MB  (samples for ensemble figure)")

    print(f"\n  All SPATIAL reconstructions written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())