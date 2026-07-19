#!/usr/bin/env python3
"""
=============================================================================
GENERATE RECONSTRUCTIONS — v6 EXTRAPOLATION-CAPABLE  (with ensemble samples)
=============================================================================

This is a CORRECTED replacement for generate_reconstructions_v6_map_axel.py.

Two changes from the previous (broken) version:

  1. The samples-save block now lives OUTSIDE the chunk loop, after
     mean_pred has been computed for all chunks. The previous version
     tried to save before mean_pred existed, causing UnboundLocalError.

  2. We accumulate samples across chunks in a single (n_ensemble, S, Y, X)
     buffer and save it once per condition, in a sibling NPZ file named
     <name>_samples.npz alongside the existing <name>.npz file.

Output files per condition:
   recon_fixed_b5.npz          — ensemble mean (existing format, unchanged)
   recon_fixed_b5_samples.npz  — all 8 ensemble samples (new)

USAGE (unchanged from before)
-----------------------------
    python generate_reconstructions_v6_map_axel.py \\
        --stage2-dir       AI_simulation/stage2 \\
        --checkpoint       ./stage2_outputs_new/checkpoints/best_model.pt \\
        --truth-npz        ./results/data/<world>.npz \\
        --output-dir       ./reconstructions_v2_phase4_stage2_map_axel \\
        --fixed-budgets    5 \\
        --mode             extrapolate \\
        --n-ensemble       8 \\
        --verbose
=============================================================================
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np


# ──────────────────────────────────────────────────────────────────
# STAGE 2 SOURCE TREE LOCATION
# ──────────────────────────────────────────────────────────────────

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
    print("  ✗ Could not find Stage 2 source tree. Pass --stage2-dir.")
    return None


# ──────────────────────────────────────────────────────────────────
# OBSERVATION-MODEL SPARSIFIERS
# ──────────────────────────────────────────────────────────────────

def sparsify_history_poisson_v5(Pt, B_last, body_mass, p_obs, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    Pt_sparse = np.zeros_like(Pt)
    N_individuals = B_last.astype(np.float64) / float(body_mass)
    expected_obs = N_individuals * float(p_obs)
    observed_count = rng.poisson(expected_obs)
    Pt_sparse[-1] = (observed_count >= 1).astype(np.float32)
    return Pt_sparse


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


# ──────────────────────────────────────────────────────────────────
# CONDITIONING BUILDER
# ──────────────────────────────────────────────────────────────────

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

    if sparse_Pt is not None:
        Pt_chunk = sparse_Pt[:, species_subset, :, :]
    else:
        Pt = np.asarray(npz_data["P_t"]).astype(np.float32)
        Pt_chunk = Pt[:, species_subset, :, :]
    history_t = torch.from_numpy(Pt_chunk[np.newaxis].copy()).float().to(device)

    cond = {
        "env": env_t, "y_coords": y_coords_t, "x_coords": x_coords_t,
        "species_features": species_features_t, "history_P": history_t,
    }
    if edge_index_t is not None:
        cond["edge_index"] = edge_index_t
        cond["edge_weight"] = None
    return cond


# ──────────────────────────────────────────────────────────────────
# v6 INFERENCE — accumulates ensemble mean AND samples
# ──────────────────────────────────────────────────────────────────

def run_inference_v6(model, npz_data, device, sparse_Pt,
                     n_ensemble=8, chunk_size=200,
                     ddim_steps=50, eta=0.5,
                     mode='extrapolate', guidance_w=0.0,
                     verbose=False, return_samples=True):
    """
    Run inference using sample_v6 (mode='extrapolate' by default).

    Returns:
        all_preds   : (S, Y, X)               ensemble mean
        all_samples : (n_ensemble, S, Y, X)   all individual samples
                      (None if return_samples=False)
    """
    import torch
    from ecodiffusion_sample_v6_map_axel import sample_v6

    P = np.asarray(npz_data["P_last_final"])
    S = P.shape[0]
    Y, X = int(npz_data["Y"]), int(npz_data["X"])

    all_preds = np.zeros((S, Y, X), dtype=np.float32)
    all_samples = (np.zeros((n_ensemble, S, Y, X), dtype=np.float32)
                   if return_samples else None)

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

            preds = sample_v6(
                model=model, condition=condition,
                n_samples=n_ensemble, ddim_steps=ddim_steps, eta=eta,
                mode=mode, guidance_w=guidance_w,
                verbose=(verbose and ci == 0),
            )  # (n_ensemble, 1, chunk_S, Y, X)

            # Squeeze the batch dim, move to CPU once, convert to fp32 in [0,1]
            preds_np = preds.squeeze(1).detach().cpu().numpy()  # (n_ens, chunk_S, Y, X)
            preds_np = np.clip(preds_np, 0, 1).astype(np.float32)

            # Mean across ensemble → goes into all_preds
            all_preds[chunk_start:chunk_end] = preds_np.mean(axis=0)

            # Each individual sample → goes into all_samples
            if all_samples is not None:
                all_samples[:, chunk_start:chunk_end, :, :] = preds_np

    return all_preds, all_samples


# ──────────────────────────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, truth_npz_path, device):
    import torch
    print(f"\n  Loading checkpoint: {checkpoint_path}")
    from models.ecodiffusion import create_fixed_model
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
        print(f"  ℹ n_species_max from truth NPZ: {detected_n}")
    else:
        print(f"  ℹ n_species_max from checkpoint: {detected_n}")

    model = create_fixed_model(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"  ⚠ {len(missing)} missing keys")
    if unexpected: print(f"  ⚠ {len(unexpected)} unexpected keys")
    model = model.to(device); model.eval()
    print(f"  ✓ Model loaded: "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model


# ──────────────────────────────────────────────────────────────────
# OUTPUT NAMING / SAVING
# ──────────────────────────────────────────────────────────────────

def poisson_filename(p_obs):
    s = f"{p_obs:.6f}".rstrip('0').rstrip('.').replace('.', 'p')
    if s == '' or s == '0':  s = '0'
    return f"recon_poisson_p{s}.npz"


def fixed_filename(K):
    return f"recon_fixed_b{int(K)}.npz"


def save_recon_mean(predictions, output_path, condition_meta, noisy_input=None):
    save_dict = {
        'mean': predictions.astype(np.float32),
        'condition_type':  str(condition_meta.get("type", "")),
        'condition_value': float(condition_meta.get("value", -1.0)),
        'n_ensemble':      int(condition_meta.get("n_ensemble", -1)),
        'body_mass':       float(condition_meta.get("body_mass", 0.0)),
        'ddim_steps':      int(condition_meta.get("ddim_steps", 50)),
        'eta':             float(condition_meta.get("eta", 0.5)),
        'rng_seed':        int(condition_meta.get("rng_seed", 42)),
        'sample_mode':     str(condition_meta.get("mode", "extrapolate")),
    }
    if noisy_input is not None:
        save_dict['noisy_input'] = noisy_input.astype(np.float32)
    np.savez_compressed(output_path, **save_dict)
    sz = output_path.stat().st_size / 1024
    print(f"      ✓ saved {output_path.name}  shape={predictions.shape}  "
          f"size={sz:.0f} KB  mean={predictions.mean():.4f}  "
          f"frac>0.5={(predictions > 0.5).mean():.4f}")


def save_recon_samples(samples, mean_pred, output_path,
                       condition_meta, noisy_input=None):
    """Save the (n_ensemble, S, Y, X) samples buffer as a sibling file."""
    save_dict = {
        'samples': samples.astype(np.float32),
        'mean':    mean_pred.astype(np.float32),
        'condition_type':  str(condition_meta.get("type", "")),
        'condition_value': float(condition_meta.get("value", -1.0)),
        'n_ensemble':      int(samples.shape[0]),
        'sample_mode':     str(condition_meta.get("mode", "extrapolate")),
    }
    if noisy_input is not None:
        save_dict['noisy_input'] = noisy_input.astype(np.float32)
    np.savez_compressed(output_path, **save_dict)
    sz = output_path.stat().st_size / 1024
    print(f"      ✓ saved {output_path.name}  shape={samples.shape}  "
          f"size={sz:.0f} KB  (samples for ensemble figure)")


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--truth-npz", required=True)
    p.add_argument("--output-dir", default="./reconstructions_v6")
    p.add_argument("--poisson-p", type=float, nargs="*",
                   default=[0.0001, 0.0005, 0.001])
    p.add_argument("--fixed-budgets", type=int, nargs="*",
                   default=[5, 10, 20, 50])
    p.add_argument("--mode", default="extrapolate",
                   choices=["echo", "extrapolate", "guided"])
    p.add_argument("--guidance-w", type=float, default=0.0)
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-save-samples", action="store_true",
                   help="Skip saving the per-sample npz "
                        "(saves disk space if you only need the mean)")
    args = p.parse_args()

    print("\n  " + "=" * 70)
    print("  generate_reconstructions_v6  —  EXTRAPOLATION-CAPABLE  (samples)")
    print("  " + "=" * 70)

    stage2_dir = find_stage2_dir(args.stage2_dir)
    if stage2_dir is None:
        return 1

    sample_v6_src = Path(__file__).parent / "ecodiffusion_sample_v6_map_axel.py"
    sample_v6_dst = stage2_dir / "models" / "ecodiffusion_sample_v6_map_axel.py"
    if sample_v6_src.exists() and not sample_v6_dst.exists():
        import shutil
        shutil.copy(sample_v6_src, sample_v6_dst)
        print(f"  ✓ copied sample_v6 to {sample_v6_dst}")

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(stage2_dir / "models"))

    import torch
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device: {device}")

    truth_path = Path(args.truth_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  truth: {truth_path.name}")
    print(f"  out:   {out_dir.resolve()}")
    print(f"  save samples: {not args.no_save_samples}")

    npz_data = dict(np.load(truth_path, allow_pickle=True))
    Pt_full = np.asarray(npz_data["P_t"]).astype(np.float32)
    B_last = np.asarray(npz_data["B_last"]).astype(np.float32)
    body_mass = float(npz_data.get("BODY_MASS", 1e-4))
    S, Y, X = B_last.shape
    print(f"  S={S}  grid={Y}x{X}  T={Pt_full.shape[0]}  body_mass={body_mass:g}")

    model = load_model(args.checkpoint, truth_path, device)

    return_samples = not args.no_save_samples

    # ── Poisson regime ────────────────────────────────────────
    print("\n  " + "=" * 70)
    print("  Poisson v5 (corrected p_obs schedule)")
    print("  " + "=" * 70)
    for p_obs in args.poisson_p:
        Np = (1.0 / body_mass) * p_obs
        p_detect = 1 - np.exp(-Np)
        print(f"\n  p_obs = {p_obs:g}  →  N×p ≈ {Np:.2f}  P(detect|present) ≈ {p_detect:.4f}")
        if p_detect > 0.99:
            print(f"    ⚠  WARNING: this p_obs saturates.")
        sparse_Pt = sparsify_history_poisson_v5(
            Pt_full, B_last, body_mass, p_obs, rng_seed=args.rng_seed)
        n_obs = int(sparse_Pt[-1].sum())
        n_truth = int((Pt_full[-1] > 0).sum())
        print(f"    observed cells: {n_obs} / {n_truth} "
              f"({100 * n_obs / max(1, n_truth):.1f}% of truth)")

        save_path = out_dir / poisson_filename(p_obs)
        preds, samples = run_inference_v6(
            model, npz_data, device, sparse_Pt,
            n_ensemble=args.n_ensemble, chunk_size=args.chunk_size,
            ddim_steps=args.ddim_steps, eta=args.eta,
            mode=args.mode, guidance_w=args.guidance_w,
            verbose=args.verbose, return_samples=return_samples,
        )
        meta = {
            "type": "poisson_v5", "value": p_obs, "n_ensemble": args.n_ensemble,
            "body_mass": body_mass, "ddim_steps": args.ddim_steps,
            "eta": args.eta, "rng_seed": args.rng_seed, "mode": args.mode,
        }
        save_recon_mean(preds, save_path, meta, noisy_input=sparse_Pt[-1])
        if samples is not None:
            samples_path = save_path.parent / (save_path.stem + '_samples.npz')
            save_recon_samples(samples, preds, samples_path, meta,
                               noisy_input=sparse_Pt[-1])

    # ── Fixed-budget regime ───────────────────────────────────
    print("\n  " + "=" * 70)
    print("  Fixed budget per species")
    print("  " + "=" * 70)
    for K in args.fixed_budgets:
        print(f"\n  K = {K} per present species")
        sparse_Pt = sparsify_history_fixed_budget(
            Pt_full, K, rng_seed=args.rng_seed)
        n_obs = int(sparse_Pt[-1].sum())
        n_present = int((Pt_full[-1].sum(axis=(1, 2)) > 0).sum())
        print(f"    observed cells: {n_obs} across {n_present} present species")

        save_path = out_dir / fixed_filename(K)
        preds, samples = run_inference_v6(
            model, npz_data, device, sparse_Pt,
            n_ensemble=args.n_ensemble, chunk_size=args.chunk_size,
            ddim_steps=args.ddim_steps, eta=args.eta,
            mode=args.mode, guidance_w=args.guidance_w,
            verbose=args.verbose, return_samples=return_samples,
        )
        meta = {
            "type": "fixed_budget", "value": K, "n_ensemble": args.n_ensemble,
            "body_mass": body_mass, "ddim_steps": args.ddim_steps,
            "eta": args.eta, "rng_seed": args.rng_seed, "mode": args.mode,
        }
        save_recon_mean(preds, save_path, meta, noisy_input=sparse_Pt[-1])
        if samples is not None:
            samples_path = save_path.parent / (save_path.stem + '_samples.npz')
            save_recon_samples(samples, preds, samples_path, meta,
                               noisy_input=sparse_Pt[-1])

    print(f"\n  All v6 reconstructions written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())