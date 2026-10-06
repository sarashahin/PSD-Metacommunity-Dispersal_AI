#!/usr/bin/env python3
"""
=============================================================================
GENERATE RECONSTRUCTIONS — PROPORTIONAL (Bernoulli-p) OBSERVATION
=============================================================================
Thin wrapper over generate_reconstructions_spatial.py. It reuses that script's
model loading, conditioning and sampling UNCHANGED, and only swaps the
observation rule:

    fixed-K       : keep K random occupied cells per species   (the original)
    proportional  : keep each occupied cell with probability p  (this script)

Output schema is identical (samples, mean, noisy_input), so the posterior
figure and every other downstream script read the result unchanged. Run it
exactly like the fixed-K generator, with --obs-prob instead of --fixed-budgets.
=============================================================================
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Reuse the tested spatial pipeline verbatim.
from generate_reconstructions_spatial import (
    find_stage2_dir, load_model, run_inference_v7,
)


def sparsify_history_proportional(Pt, prob, rng_seed=42, min_obs=1):
    """Observe each occupied cell of the LAST frame independently with
    probability `prob` (Bernoulli / Poisson thinning). Mirrors
    sparsify_history_fixed_budget exactly, but with a per-cell probability
    instead of a fixed count. Guarantees >= min_obs observed cells for any
    present species so the inpainting sampler always has a seed
    (set min_obs=0 for pure Bernoulli)."""
    rng = np.random.default_rng(rng_seed)
    T, S, Y, X = Pt.shape
    Pt_sparse = np.zeros_like(Pt)
    last = Pt[-1]
    sparse_last = np.zeros_like(last)
    for s in range(S):
        occupied = np.argwhere(last[s] > 0)
        if len(occupied) == 0:
            continue
        keep = rng.random(len(occupied)) < prob
        if keep.sum() < min_obs:
            idx = rng.choice(len(occupied), size=min(min_obs, len(occupied)), replace=False)
            keep = np.zeros(len(occupied), dtype=bool); keep[idx] = True
        for idx in np.argwhere(keep).ravel():
            y, x = occupied[idx]
            sparse_last[s, y, x] = 1.0
    Pt_sparse[-1] = sparse_last
    return Pt_sparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--truth-npz", required=True)
    p.add_argument("--output-dir", default="./reconstructions_proportional")
    p.add_argument("--obs-prob", type=float, nargs="+", default=[0.10],
                   help="one or more per-cell observation probabilities, e.g. 0.10 0.30")
    p.add_argument("--obs-min", type=int, default=1,
                   help="min observed cells per present species (0 = pure Bernoulli)")
    p.add_argument("--mode", default="inpaint",
                   choices=["inpaint", "soft_inpaint", "extrapolate"])
    p.add_argument("--repaint-iterations", type=int, default=2)
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.15)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("\n  " + "=" * 70)
    print("  generate_reconstructions_proportional  —  Bernoulli-p observation")
    print("  " + "=" * 70)

    stage2_dir = find_stage2_dir(args.stage2_dir)
    if stage2_dir is None:
        print("  could not find Stage 2 source tree."); return 1

    sample_src = Path(__file__).parent / "ecodiffusion_sample_spatial.py"
    sample_dst = stage2_dir / "models" / "ecodiffusion_sample_spatial.py"
    if sample_src.exists() and not sample_dst.exists():
        import shutil; shutil.copy(sample_src, sample_dst)
        print(f"  copied sample_spatial to {sample_dst}")

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(stage2_dir / "models"))

    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device: {device}   mode: {args.mode}")

    truth_path = Path(args.truth_npz)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    npz_data = dict(np.load(truth_path, allow_pickle=True))
    Pt_full = np.asarray(npz_data["P_t"]).astype(np.float32)
    S = Pt_full.shape[1]
    present = int((Pt_full[-1].reshape(S, -1).sum(axis=1) > 0).sum())
    print(f"  S={S}  T={Pt_full.shape[0]}  present species={present}")
    print(f"  out: {out_dir.resolve()}")

    model = load_model(args.checkpoint, truth_path, device)

    for prob in args.obs_prob:
        print(f"\n  proportional p = {prob:.2f}  (min_obs = {args.obs_min})")
        sparse_Pt = sparsify_history_proportional(
            Pt_full, prob, rng_seed=args.rng_seed, min_obs=args.obs_min)
        n_obs = int(sparse_Pt[-1].sum())
        print(f"    observed cells: {n_obs}  (~{n_obs / max(present, 1):.1f} per present species)")

        preds, samples = run_inference_v7(
            model, npz_data, device, sparse_Pt,
            n_ensemble=args.n_ensemble, chunk_size=args.chunk_size,
            ddim_steps=args.ddim_steps, eta=args.eta,
            mode=args.mode, repaint_iterations=args.repaint_iterations,
            verbose=args.verbose,
        )

        tag = f"prop_p{prob:.2f}"
        np.savez_compressed(
            out_dir / f"recon_{tag}.npz",
            mean=preds.astype(np.float32),
            noisy_input=sparse_Pt[-1].astype(np.float32),
            sample_mode=str(args.mode), n_ensemble=int(args.n_ensemble),
            obs_prob=np.float32(prob))
        samples_path = out_dir / f"recon_{tag}_samples.npz"
        np.savez_compressed(
            samples_path,
            samples=samples.astype(np.float32),
            mean=preds.astype(np.float32),
            noisy_input=sparse_Pt[-1].astype(np.float32),
            sample_mode=str(args.mode), n_ensemble=int(args.n_ensemble),
            obs_prob=np.float32(prob))
        print(f"      saved {samples_path.name}  shape={samples.shape}  "
              f"mean={preds.mean():.4f}  frac>0.5={(preds > 0.5).mean():.4f}")

    print(f"\n  done -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())