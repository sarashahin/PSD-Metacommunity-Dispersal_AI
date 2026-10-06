#!/usr/bin/env python3
"""
=============================================================================
RUN_ABLATION_V7.PY  —  Axel's "remove-each-predictor" study, on the new
                        v7 inpainting retrained model
=============================================================================

WHAT THIS DOES (one-line)
-------------------------
For each ablation variant in {FULL, NO_HISTORY, NO_NETWORK, NO_ENV,
NO_SPECIES_FEATS}, regenerates the v7-inpainting reconstruction on a
single world. Saves one NPZ per variant. Equivalent to running
generate_reconstructions_v7_inpaint.py 5 times with different inputs
zeroed out, but in a single pass over the model.

WHY EACH VARIANT
----------------
Axel's transcript: "you don't need the past history, or you don't need
the information on the network, or you don't need the suitability of
each patch for each species. You can do without. and see, okay, what
is really the crucial information here that you need to make this
prediction?"

  FULL              all conditioning (= baseline, sanity check)
  NO_HISTORY        history_P zeroed → no obs, no temporal context, no
                    inpainting overlay (obs_mask becomes zeros). Tests:
                    can the model predict without ANY observations?
  NO_NETWORK        edge_index emptied → GNN runs but with no edges.
                    Tests: does the species-interaction graph help?
  NO_ENV            env field zeroed (uniform "neutral" environment).
                    Tests: does environmental suitability help?
  NO_SPECIES_FEATS  species_features zeroed (no prevalence prior).
                    Tests: does species-level metadata help?

CRUCIAL DETAIL
--------------
For NO_HISTORY we ZERO history_P — we do NOT set it to None. Setting
to None would trip the v7 sampler's fallback path to v6 sampling, which
is a different SAMPLING algorithm entirely. Zeroing keeps v7's pipeline
unchanged but feeds it empty observations. That isolates the "removing
the predictor" effect from the "changing the sampler" effect.

USAGE
-----
  python run_ablation_v7.py \\
      --stage2-dir         AI_simulation/stage2 \\
      --checkpoint         ./stage2_outputs_new/checkpoints/best_model.pt \\
      --truth-npz          ./results/data/<world>.npz \\
      --output-dir         ./ablation_v7_world5 \\
      --K                  5 \\
      --variants           FULL NO_HISTORY NO_NETWORK NO_ENV NO_SPECIES_FEATS \\
      --n-ensemble         8 \\
      --verbose

The output directory will contain:
  recon_FULL_b5.npz             recon_FULL_b5_samples.npz
  recon_NO_HISTORY_b5.npz       recon_NO_HISTORY_b5_samples.npz
  recon_NO_NETWORK_b5.npz       recon_NO_NETWORK_b5_samples.npz
  recon_NO_ENV_b5.npz           recon_NO_ENV_b5_samples.npz
  recon_NO_SPECIES_FEATS_b5.npz recon_NO_SPECIES_FEATS_b5_samples.npz
  ablation_run_log.txt
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------
# Stage-2 source-tree resolver  (lifted verbatim from generate_recon...)
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# History sparsification  (matches generate_reconstructions_v7_inpaint.py)
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# Build condition dict  (matches generate_reconstructions_v7_inpaint.py)
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# THE CORE OF THIS SCRIPT — applying each ablation
# -----------------------------------------------------------------------

VARIANTS = [
    'FULL',
    'NO_HISTORY',
    'NO_NETWORK',
    'NO_ENV',
    'NO_SPECIES_FEATS',
]


def apply_ablation(condition, variant, verbose=False):
    """
    Return a new condition dict with the named input zeroed/emptied.

    Note: we ZERO history_P rather than setting it to None — see header.
    The shape is preserved so the temporal_encoder still runs, just on
    empty content; obs_mask = (history[:,-1] > 0.5) becomes all-zeros so
    the inpainting overlay does nothing (model has no anchor cells).
    """
    import torch

    if variant == 'FULL':
        return dict(condition)

    new_cond = dict(condition)

    if variant == 'NO_HISTORY':
        if new_cond.get('history_P') is not None:
            new_cond['history_P'] = torch.zeros_like(new_cond['history_P'])
        if verbose:
            print("    [ablation] history_P zeroed (no obs, no temporal "
                  "context, no inpainting anchor)")

    elif variant == 'NO_NETWORK':
        device = new_cond['env'].device
        new_cond['edge_index'] = torch.empty(
            2, 0, dtype=torch.long, device=device)
        new_cond['edge_weight'] = None
        if verbose:
            print("    [ablation] edge_index emptied (GNN sees isolated "
                  "species)")

    elif variant == 'NO_ENV':
        new_cond['env'] = torch.zeros_like(new_cond['env'])
        if verbose:
            print("    [ablation] env field zeroed (uniform neutral "
                  "environment)")

    elif variant == 'NO_SPECIES_FEATS':
        if new_cond.get('species_features') is not None:
            new_cond['species_features'] = torch.zeros_like(
                new_cond['species_features'])
        if verbose:
            print("    [ablation] species_features zeroed (no prevalence "
                  "prior)")

    else:
        raise ValueError(f"Unknown ablation variant: {variant}")

    return new_cond


# -----------------------------------------------------------------------
# Inference per variant
# -----------------------------------------------------------------------

def run_inference_one_variant(
    model, npz_data, device, sparse_Pt, variant,
    n_ensemble=8, chunk_size=200,
    ddim_steps=50, eta=0.5,
    mode='inpaint', repaint_iterations=2,
    verbose=False,
):
    """Inference using v7 inpainting sampler with ablation applied."""
    import torch
    from ecodiffusion_sample_v7_inpaint import sample_v7

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
            if verbose:
                print(f"      chunk {ci+1}/{n_chunks}: species "
                      f"[{chunk_start}:{chunk_end}] (n={chunk_S}), "
                      f"variant={variant}", flush=True)

            cond_full = build_condition(
                npz_data, device, chunk_indices, sparse_Pt)
            condition = apply_ablation(
                cond_full, variant,
                verbose=(verbose and ci == 0))

            preds = sample_v7(
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


# -----------------------------------------------------------------------
# Model loader (matches generate_reconstructions_v7_inpaint.py)
# -----------------------------------------------------------------------

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

    model = create_fixed_model(config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device); model.eval()
    print(f"  ✓ Model loaded: "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model


# -----------------------------------------------------------------------
# Sanity check — runs after each variant. CHEAP (just looks at NPZ).
# -----------------------------------------------------------------------

def sanity_check_recon(preds, samples, sparse_last, variant, log_lines):
    """Light validation that the saved arrays are well-formed."""
    msgs = [f"\n  [SANITY {variant}]"]
    msgs.append(f"    pred shape : {preds.shape}    "
                f"min={preds.min():.4f}  max={preds.max():.4f}  "
                f"mean={preds.mean():.5f}")
    msgs.append(f"    sample shape : {samples.shape}    "
                f"all_finite={np.isfinite(samples).all()}")
    n_obs = int(sparse_last.sum())
    n_high = int((preds > 0.5).sum())
    msgs.append(f"    obs cells={n_obs:,}   pred>0.5={n_high:,}")

    if variant == 'NO_HISTORY':
        # We expect very few obs cells reproduced (no inpainting anchor)
        # because history_P was zeroed → obs_mask was zero
        echo = int(((preds > 0.5) & (sparse_last > 0.5)).sum())
        msgs.append(f"    [check] echo at obs cells = {echo} "
                    f"(expected ≈ 0 because obs_mask was zeroed; "
                    f"any nonzero is from learned priors)")
    elif variant == 'FULL':
        echo = int(((preds > 0.5) & (sparse_last > 0.5)).sum())
        msgs.append(f"    [check] echo at obs cells = {echo} / {n_obs} "
                    f"(expected ≈ {n_obs} from inpainting overlay)")

    text = "\n".join(msgs)
    print(text)
    log_lines.append(text)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--truth-npz", required=True)
    p.add_argument("--output-dir", default="./ablation_v7")
    p.add_argument("--K", type=int, default=5,
                   help="Observation budget per species (default 5)")
    p.add_argument("--variants", nargs="+", default=VARIANTS,
                   choices=VARIANTS,
                   help="Subset of ablation variants to run")
    p.add_argument("--mode", default="inpaint",
                   choices=["inpaint", "soft_inpaint", "extrapolate"])
    p.add_argument("--repaint-iterations", type=int, default=2)
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("\n  " + "=" * 70)
    print("  run_ablation_v7  —  Axel-style remove-each-predictor study")
    print("  " + "=" * 70)
    print(f"  Variants: {', '.join(args.variants)}")
    print(f"  K (observations/species): {args.K}")
    print(f"  Ensemble size: {args.n_ensemble}")

    stage2_dir = find_stage2_dir(args.stage2_dir)
    if stage2_dir is None:
        print("  ✗ Could not find Stage 2 source tree.")
        return 1

    # Make sample_v7 importable from stage2/models if not already there
    sample_v7_dst = stage2_dir / "models" / "ecodiffusion_sample_v7_inpaint.py"
    if not sample_v7_dst.exists():
        sample_v7_src = Path(__file__).parent / "ecodiffusion_sample_v7_inpaint.py"
        if sample_v7_src.exists():
            import shutil
            shutil.copy(sample_v7_src, sample_v7_dst)
            print(f"  ✓ copied sample_v7 to {sample_v7_dst}")

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(stage2_dir / "models"))

    import torch
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device: {device}")

    truth_path = Path(args.truth_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out:    {out_dir.resolve()}")

    npz_data = dict(np.load(truth_path, allow_pickle=True))
    Pt_full = np.asarray(npz_data["P_t"]).astype(np.float32)
    S_data = Pt_full.shape[1]
    print(f"  S={S_data}  T={Pt_full.shape[0]}")

    # Sparsify ONCE — same K observations used by all variants
    # so differences across variants are due ONLY to ablation, not to
    # different observation samples.
    print(f"\n  sparsifying history to K={args.K} (seed={args.rng_seed})...")
    sparse_Pt = sparsify_history_fixed_budget(
        Pt_full, args.K, rng_seed=args.rng_seed)
    n_obs = int(sparse_Pt[-1].sum())
    print(f"    observed cells: {n_obs}")

    model = load_model(args.checkpoint, truth_path, device)

    # ── Run each variant ─────────────────────────────────────────────
    log_lines = [f"Ablation run on world: {truth_path.name}",
                 f"K={args.K}, ensemble={args.n_ensemble}",
                 f"variants={args.variants}",
                 f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    timings = {}
    for variant in args.variants:
        print(f"\n  {'─' * 68}")
        print(f"  VARIANT: {variant}")
        print(f"  {'─' * 68}")
        t0 = time.time()
        preds, samples = run_inference_one_variant(
            model, npz_data, device, sparse_Pt, variant,
            n_ensemble=args.n_ensemble, chunk_size=args.chunk_size,
            ddim_steps=args.ddim_steps, eta=args.eta,
            mode=args.mode, repaint_iterations=args.repaint_iterations,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        timings[variant] = elapsed
        log_lines.append(f"\n  {variant} elapsed: {elapsed:.1f}s")

        # Save mean
        save_path = out_dir / f"recon_{variant}_b{args.K}.npz"
        np.savez_compressed(
            save_path,
            mean=preds.astype(np.float32),
            noisy_input=sparse_Pt[-1].astype(np.float32),
            sample_mode=str(args.mode),
            n_ensemble=int(args.n_ensemble),
            ablation_variant=variant,
            K=int(args.K),
        )
        sz_mb = save_path.stat().st_size / 1024 / 1024
        msg = (f"  ✓ saved {save_path.name}  shape={preds.shape}  "
               f"size={sz_mb:.1f} MB  mean={preds.mean():.4f}  "
               f"frac>0.5={(preds > 0.5).mean():.4f}")
        print(msg); log_lines.append(msg)

        # Save samples
        samples_path = out_dir / f"recon_{variant}_b{args.K}_samples.npz"
        np.savez_compressed(
            samples_path,
            samples=samples.astype(np.float32),
            mean=preds.astype(np.float32),
            noisy_input=sparse_Pt[-1].astype(np.float32),
            sample_mode=str(args.mode),
            n_ensemble=int(args.n_ensemble),
            ablation_variant=variant,
            K=int(args.K),
        )
        sz_mb = samples_path.stat().st_size / 1024 / 1024
        msg = f"  ✓ saved {samples_path.name}  size={sz_mb:.1f} MB"
        print(msg); log_lines.append(msg)

        # Sanity check this variant
        sanity_check_recon(preds, samples, sparse_Pt[-1], variant, log_lines)

    # Write log
    log_path = out_dir / "ablation_run_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n  ✓ log written to {log_path}")

    # Summary
    print(f"\n  {'=' * 70}")
    print(f"  ABLATION RUN COMPLETE  ({truth_path.name})")
    print(f"  {'=' * 70}")
    for v, t in timings.items():
        print(f"    {v:<22} {t:>7.1f}s")
    print(f"\n  Next steps:")
    print(f"    1. python validate_ablation.py "
          f"--ablation-dir {out_dir} --truth-npz {truth_path} "
          f"--K {args.K}")
    print(f"    2. python visualize_ablation_results.py "
          f"--metrics-csv {out_dir}/ablation_metrics.csv")
    print(f"    3. (optional) make_figure1_honest_map.py per variant "
          f"NPZ for visual comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())