#!/usr/bin/env python3
"""
=============================================================================
RUN_ABLATION_SPATIAL.PY  —  remove-each-predictor study on the DEPLOYED
                            spatial-conditioning model (Fix A+B, epoch 149)
=============================================================================
WHY THIS REPLACES run_ablation_v7.py
------------------------------------
run_ablation_v7.py targets the OLD v7 architecture:
    from models.ecodiffusion import create_fixed_model      (1-channel U-Net)
    from ecodiffusion_sample_v7_inpaint import sample_v7    (3-arg unet call)
The deployed model is EcoDiffusionSpatial:
    create_spatial_cond_model                               (4-channel U-Net)
    sample_spatial                                          (4-arg unet call)
Loading a spatial checkpoint into create_fixed_model with strict=False would
succeed silently and produce meaningless output. This script reuses the tested
spatial pipeline verbatim (including its input_proj.in_channels == 4 assert),
and only inserts the ablation between build_condition() and sample_spatial().

VARIANTS (all share the SAME sparse observations, so the only difference
between arms is the ablated input)
  FULL              all conditioning
  NO_HISTORY        history_P zeroed. Records reach the model through the
                    history channel, so this also empties spatial channel 0
                    (obs_mask) and channel 2 (obs_decay), and disables the
                    inpainting overlay. It therefore tests prediction with NO
                    occurrence data at all -- interpret it that way.
  NO_NETWORK        edge_index emptied (GNN sees isolated species)
  NO_ENV            env zeroed. This propagates to BOTH the environmental
                    encoder AND spatial channel 1, so the ablation is complete.
  NO_SPECIES_FEATS  species_features zeroed

We ZERO tensors rather than setting them to None: None would change which
code path the sampler takes, confounding "removed the predictor" with
"changed the algorithm".

USAGE
-----
  python run_ablation_spatial.py \
      --stage2-dir AI_simulation/stage2 \
      --checkpoint ./stage2_outputs_new/checkpoints/best_model.pt \
      --truth-npz  ./results/data/data_eval_unseen/<world>_training.npz \
      --output-dir ./ablation_spatial/<world> \
      --K 5 --n-ensemble 8 --ddim-steps 50 --eta 0.15

  # proportional instead of fixed budget:
  #   --obs-prob 0.10   (overrides --K)
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the tested spatial pipeline verbatim.
from generate_reconstructions_spatial import (
    find_stage2_dir, load_model, build_condition,
    sparsify_history_fixed_budget,
)

VARIANTS = ['FULL', 'NO_HISTORY', 'NO_NETWORK', 'NO_ENV', 'NO_SPECIES_FEATS',
            'PERM_ENV', 'PERM_SPECIES_FEATS']


def sparsify_history_proportional(Pt, prob, rng_seed=42, min_obs=1):
    """Bernoulli thinning of the last frame (mirrors the proportional generator)."""
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
            idx = rng.choice(len(occupied),
                             size=min(min_obs, len(occupied)), replace=False)
            keep = np.zeros(len(occupied), dtype=bool)
            keep[idx] = True
        for idx in np.argwhere(keep).ravel():
            y, x = occupied[idx]
            sparse_last[s, y, x] = 1.0
    Pt_sparse[-1] = sparse_last
    return Pt_sparse


def apply_ablation(condition, variant, verbose=False):
    """Return a copy of `condition` with the named input zeroed/emptied."""
    import torch

    if variant == 'FULL':
        return dict(condition)

    c = dict(condition)

    if variant == 'NO_HISTORY':
        if c.get('history_P') is not None:
            c['history_P'] = torch.zeros_like(c['history_P'])
        if verbose:
            print("    [ablation] history_P zeroed -> no records, no obs_mask, "
                  "no obs_decay, no inpainting anchor")

    elif variant == 'NO_NETWORK':
        dev = c['env'].device
        c['edge_index'] = torch.empty(2, 0, dtype=torch.long, device=dev)
        c['edge_weight'] = None
        if verbose:
            print("    [ablation] edge_index emptied -> isolated species")

    elif variant == 'NO_ENV':
        c['env'] = torch.zeros_like(c['env'])
        if verbose:
            print("    [ablation] env zeroed -> encoder AND spatial channel 1")

    elif variant == 'NO_SPECIES_FEATS':
        if c.get('species_features') is not None:
            c['species_features'] = torch.zeros_like(c['species_features'])
        if verbose:
            print("    [ablation] species_features zeroed")

    elif variant == 'PERM_ENV':
        # keep the field on-distribution, break the species-field pairing
        e = c['env'].clone()
        perm = torch.randperm(e.shape[1], generator=torch.Generator().manual_seed(0))
        c['env'] = e[:, perm]

    elif variant == 'PERM_SPECIES_FEATS':
        f = c['species_features'].clone()
        perm = torch.randperm(f.shape[1], generator=torch.Generator().manual_seed(1))
        c['species_features'] = f[:, perm]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return c




def run_variant(model, npz_data, device, sparse_Pt, variant,
                n_ensemble=8, chunk_size=200, ddim_steps=50, eta=0.15,
                mode='inpaint', repaint_iterations=2, verbose=False):
    """Mirror of generate_reconstructions_spatial.run_inference_v7, with the
    ablation applied to the condition dict before sampling."""
    import torch
    from ecodiffusion_sample_spatial import sample_spatial

    S = np.asarray(npz_data["P_last_final"]).shape[0]
    Y, X = int(npz_data["Y"]), int(npz_data["X"])
    all_preds = np.zeros((S, Y, X), dtype=np.float32)
    all_samples = np.zeros((n_ensemble, S, Y, X), dtype=np.float32)

    if hasattr(model, "set_training_phase"):
        model.set_training_phase(4)
    model.eval()

    n_chunks = (S + chunk_size - 1) // chunk_size
    with torch.no_grad():
        for ci, start in enumerate(range(0, S, chunk_size)):
            end = min(start + chunk_size, S)
            idx = np.arange(start, end)
            if verbose:
                print(f"      chunk {ci+1}/{n_chunks}: [{start}:{end}] "
                      f"variant={variant}", flush=True)

            cond = build_condition(npz_data, device, idx, sparse_Pt)
            cond = apply_ablation(cond, variant, verbose=(verbose and ci == 0))

            preds = sample_spatial(
                model=model, condition=cond, n_samples=n_ensemble,
                ddim_steps=ddim_steps, eta=eta, mode=mode,
                repaint_iterations=repaint_iterations,
                verbose=(verbose and ci == 0))

            p = np.clip(preds.squeeze(1).detach().cpu().numpy(), 0, 1).astype(np.float32)
            all_preds[start:end] = p.mean(axis=0)
            all_samples[:, start:end, :, :] = p

    return all_preds, all_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--truth-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--obs-prob", type=float, default=None,
                   help="if set, use proportional Bernoulli-p instead of fixed K")
    p.add_argument("--obs-min", type=int, default=1)
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--mode", default="inpaint",
                   choices=["inpaint", "soft_inpaint", "extrapolate"])
    p.add_argument("--repaint-iterations", type=int, default=2)
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.15,
                   help="MUST match the deployed reconstructions (0.15)")
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("\n  " + "=" * 70)
    print("  run_ablation_spatial  —  remove-each-predictor (spatial model)")
    print("  " + "=" * 70)
    print(f"  variants : {', '.join(args.variants)}")
    print(f"  sampling : eta={args.eta}, steps={args.ddim_steps}, "
          f"ensemble={args.n_ensemble}, mode={args.mode}")

    stage2_dir = find_stage2_dir(args.stage2_dir)
    if stage2_dir is None:
        print("  could not find Stage 2 source tree.")
        return 1

    src = Path(__file__).parent / "ecodiffusion_sample_spatial.py"
    dst = stage2_dir / "models" / "ecodiffusion_sample_spatial.py"
    if src.exists() and not dst.exists():
        import shutil
        shutil.copy(src, dst)
        print(f"  copied sample_spatial -> {dst}")

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(stage2_dir / "models"))

    import torch
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device   : {device}")

    truth_path = Path(args.truth_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_data = dict(np.load(truth_path, allow_pickle=True))
    if "P_t" not in npz_data:
        print(f"  ERROR: {truth_path.name} has no 'P_t' array "
              f"(keys: {sorted(npz_data.keys())[:8]}...).")
        print("  Use the *_training.npz worlds; *_dataset.npz files lack P_t.")
        return 1
    Pt_full = np.asarray(npz_data["P_t"]).astype(np.float32)
    S = Pt_full.shape[1]
    present = int((Pt_full[-1].reshape(S, -1).sum(axis=1) > 0).sum())
    print(f"  world    : {truth_path.name}")
    print(f"  S={S}  T={Pt_full.shape[0]}  present={present}")

    # ONE sparsification shared by every variant -> differences are the
    # ablation, never a different draw of observations.
    if args.obs_prob is not None:
        sparse_Pt = sparsify_history_proportional(
            Pt_full, args.obs_prob, rng_seed=args.rng_seed, min_obs=args.obs_min)
        tag, obs_desc = f"p{args.obs_prob:.2f}", f"p={args.obs_prob}"
    else:
        sparse_Pt = sparsify_history_fixed_budget(
            Pt_full, args.K, rng_seed=args.rng_seed)
        tag, obs_desc = f"b{args.K}", f"K={args.K}"
    n_obs = int(sparse_Pt[-1].sum())
    print(f"  obs rule : {obs_desc}  ->  {n_obs} cells "
          f"(~{n_obs / max(present,1):.1f} per present species)")

    model = load_model(args.checkpoint, truth_path, device)   # asserts 4 channels

    log = [f"world: {truth_path.name}", f"obs: {obs_desc}, cells={n_obs}",
           f"eta={args.eta}, steps={args.ddim_steps}, ens={args.n_ensemble}",
           f"variants: {args.variants}",
           f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    for v in args.variants:
        print(f"\n  {'-' * 68}\n  VARIANT: {v}\n  {'-' * 68}")
        t0 = time.time()
        preds, samples = run_variant(
            model, npz_data, device, sparse_Pt, v,
            n_ensemble=args.n_ensemble, chunk_size=args.chunk_size,
            ddim_steps=args.ddim_steps, eta=args.eta, mode=args.mode,
            repaint_iterations=args.repaint_iterations, verbose=args.verbose)
        dt = time.time() - t0

        common = dict(mean=preds.astype(np.float32),
                      noisy_input=sparse_Pt[-1].astype(np.float32),
                      sample_mode=str(args.mode),
                      n_ensemble=int(args.n_ensemble),
                      ablation_variant=v, eta=np.float32(args.eta))
        np.savez_compressed(out_dir / f"recon_{v}_{tag}.npz", **common)
        np.savez_compressed(out_dir / f"recon_{v}_{tag}_samples.npz",
                            samples=samples.astype(np.float32), **common)

        # per-variant sanity: how much did the model add beyond the records?
        obs_b = sparse_Pt[-1] > 0.5
        hi = preds > 0.5
        msg = (f"  saved recon_{v}_{tag}_samples.npz  "
               f"mean={preds.mean():.4f}  cells>0.5={int(hi.sum())}  "
               f"of which at obs={int((hi & obs_b).sum())}  "
               f"({dt:.1f}s)")
        print(msg)
        log.append(msg)

    (out_dir / "ablation_run_log.txt").write_text("\n".join(log))
    print(f"\n  done -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())