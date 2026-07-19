#!/usr/bin/env python3
"""
=============================================================================
DIAGNOSE_SPATIAL_USE_STRATIFIED.PY  —  the decisive root-cause diagnostic
=============================================================================

WHY THIS EXISTS — what the previous diagnostic could not tell us
----------------------------------------------------------------
diagnose_spatial_use.py reported `auc_unobs = 0.354` on the Fix A+B
checkpoint. Baseline was 0.356. That looks like Fix A+B did nothing. But
`auc_unobs` is averaged over ALL species, and your test worlds have a
known property:

  ~75% of present species have range size 1-2 cells.
  At K=5 observations, K >= range for these species. There IS NO
  extrapolation problem for them — the observations ARE the range.

For these narrow-range species, "the model should predict presence at the
K observed cells and nothing else" is the CORRECT behaviour, and the
copy-the-observations shortcut is the right answer. A perfect extrapolator
would do exactly the same.

So the headline `auc_unobs = 0.35` aggregates two completely different
regimes: (a) species where extrapolation is impossible, and (b) species
where extrapolation is the entire task. The aggregate is dominated by (a),
where copying scores roughly 0.5 (random ranking of unobserved absence
cells), and (b) — the species we actually care about — is invisible in the
average.

This diagnostic STRATIFIES auc_unobs by truth range size:
    Trivial (range <= K)         : copying IS correct; baseline expectation
    Sparse  (K < range <= 2K)    : modest extrapolation needed
    Real    (range > 2K)         : genuine wide-range reconstruction task
    Wide    (range > 4K)         : Axel's "many observations, large range"

If `auc_unobs` is high in Trivial and low in Real/Wide → Fix A+B failed to
fix extrapolation, and the architectural fix (zero-init gated injection,
Fix D) is justified.

If `auc_unobs` rises monotonically with range OR is high across all
strata → Fix A+B helped more than the headline suggested, and the next
step is a hyperparameter pass (Fix C — raise extrap_weight to 1.0/2.0).

CRITICAL DESIGN CHOICES (why this differs from your existing script)
--------------------------------------------------------------------
1. IN-MEMORY, NO RECONSTRUCTIONS. We run sample_spatial in this script,
   on a few worlds, in extrapolate mode (mode='extrapolate'). The existing
   compute_ensemble_coverage_stratified.py needs finished recon NPZ files,
   which you don't have for the Fix A+B model yet. This one creates
   predictions on the fly and computes auc_unobs directly.

2. mode='extrapolate' (NOT 'inpaint'). With inpainting on, observed cells
   are pinned to truth — auc_unobs at unobserved cells could be inflated
   by the model's ability to NOT touch the pinned cells. We need to see
   what the model predicts when it has only the spatial channels as
   evidence, no inpainting overlay.

3. RESTRICT TO UNOBSERVED CELLS. The metric is auc_unobs — AUC computed
   only at cells where obs_mask = 0. This is the actual extrapolation
   question.

4. PER-SPECIES, THEN STRATIFIED AVERAGE. Compute auc_unobs per species,
   then average within each stratum. Avoids the "narrow-species majority
   drags the average" failure mode.

USAGE
-----
    python diagnose_spatial_use_stratified.py \
        --checkpoint stage2_outputs_SPATIAL_REAL/checkpoints/last_checkpoint.pt \
        --simulation-dir results/data \
        --output-dir   ./diagnose_stratified \
        --n-worlds     5  \
        --n-samples    4  \
        --K-obs        5

Runtime: ~30-50 minutes (5 worlds, 4-sample ensemble, 25 DDIM steps each).
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="best_model.pt or last_checkpoint.pt of the model under test")
    p.add_argument("--simulation-dir", required=True)
    p.add_argument("--stage2-dir", default=None)
    p.add_argument("--output-dir", default="./diagnose_stratified")
    p.add_argument("--n-worlds", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=4,
                   help="ensemble size for sample_spatial")
    p.add_argument("--ddim-steps", type=int, default=25)
    p.add_argument("--eta", type=float, default=0.15)
    p.add_argument("--K-obs", type=int, default=5,
                   help="observation budget used by training; "
                        "drives the stratum boundaries (K, 2K, 4K)")
    return p.parse_args()


# ---------------------------------------------------------------------------
def auc_per_species_at_unobs(pred, truth, obs_mask):
    """
    Per-species AUC at unobserved cells.
    Inputs: numpy arrays of shape (S, Y, X).
    Returns dict: {species_idx: auc} for species where AUC is defined.
    """
    from sklearn.metrics import roc_auc_score
    out = {}
    S = pred.shape[0]
    for s in range(S):
        unobs = (obs_mask[s] < 0.5)
        if unobs.sum() < 5:
            continue
        y = truth[s][unobs].astype(int).ravel()
        p = pred[s][unobs].astype(float).ravel()
        # need both classes present
        if y.sum() < 1 or y.sum() == len(y):
            continue
        try:
            out[s] = float(roc_auc_score(y, p))
        except Exception:
            continue
    return out


def jaccard_per_species(pred, truth, thr=0.5):
    """Per-species Jaccard, full grid."""
    out = {}
    for s in range(pred.shape[0]):
        if truth[s].sum() == 0:
            continue
        pb = (pred[s] > thr)
        tb = (truth[s] > 0.5)
        inter = (pb & tb).sum()
        union = (pb | tb).sum()
        out[s] = float(inter / max(1, union))
    return out


def stratify(species_metric, truth_ranges, K):
    """
    Group per-species metric dict by truth range strata.
    Returns list of (stratum_label, n_species, mean, std, ci95) rows.
    """
    strata = [
        ("Trivial  (range <= K)",       lambda r: 1 <= r <= K),
        ("Sparse   (K < r <= 2K)",      lambda r: K < r <= 2 * K),
        ("Real     (2K < r <= 4K)",     lambda r: 2 * K < r <= 4 * K),
        ("Wide     (r > 4K)",           lambda r: r > 4 * K),
        ("Meaningful (r > K)  AGG",     lambda r: r > K),
        ("ALL species         AGG",     lambda r: r >= 1),
    ]
    rows = []
    for label, cond in strata:
        vals = [species_metric[s] for s, r in truth_ranges.items()
                if s in species_metric and cond(r)]
        if not vals:
            rows.append((label, 0, float('nan'), float('nan'), float('nan')))
            continue
        a = np.array(vals)
        mean = float(a.mean())
        std = float(a.std()) if len(a) > 1 else 0.0
        ci95 = 1.96 * std / max(1, np.sqrt(len(a)))
        rows.append((label, len(a), mean, std, ci95))
    return rows


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    s2 = (Path(args.stage2_dir).resolve()
          if args.stage2_dir else Path(__file__).resolve().parent)
    for p in (s2, s2 / "models", s2 / "configs"):
        sys.path.insert(0, str(p))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- imports happen AFTER sys.path is set ------------------------
    from configs.config import get_default_config
    from models.ecodiffusion_spatial_cond import create_spatial_cond_model
    from ecodiffusion_sample_spatial import sample_spatial
    from data_preprocessing import create_dataloaders
    import data_preprocessing_v2_patch
    import interaction_encoder_v2_patch  # noqa: F401
    data_preprocessing_v2_patch.auto_install(history_length=10)

    # ---- data + model ------------------------------------------------
    print(f"\n  Building phase-4 (infill) val dataloader...")
    cfg = get_default_config()
    cfg.paths.simulation_dir = args.simulation_dir
    cfg.training.batch_size = 1
    cfg.training.hist_sparsify_K = [args.K_obs, 10, 20]
    _, val_loader, _, _ = create_dataloaders(args.simulation_dir, cfg,
                                             mode='infill')

    sample_batch = next(iter(val_loader))
    cfg.data.n_species_max = sample_batch['target'].shape[1]
    model = create_spatial_cond_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.set_training_phase(4)
    print(f"\n  Loaded checkpoint: {args.checkpoint}")
    print(f"    epoch        = {ckpt.get('epoch', '?')}")
    print(f"    best_metric  = {ckpt.get('best_metric', 0.0):.4f}")
    print(f"    params       = {sum(p.numel() for p in model.parameters()):,}")
    print(f"    K_obs (strat)= {args.K_obs}  "
          f"(strata: <=K, K<r<=2K, 2K<r<=4K, r>4K)")

    # ---- collect per-species metrics across worlds ------------------
    all_auc_unobs = {}      # (world_idx, species_idx) -> auc
    all_jaccard   = {}
    all_ranges    = {}      # (world_idx, species_idx) -> truth range size

    val_iter = iter(val_loader)
    t0 = time.time()
    for w in range(args.n_worlds):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        target = batch['target'].to(device)          # (B, S, Y, X)
        cond = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch['condition'].items()}

        # extract the exact obs_mask the model sees
        with torch.no_grad():
            cs = model.build_spatial_cond(cond)      # (B, S, 3, Y, X)
        obs_mask_real = cs[:, :, 0].cpu().numpy()    # (B, S, Y, X)

        # FULL prediction in extrapolate mode (no inpainting confound)
        with torch.no_grad():
            samples = sample_spatial(
                model, cond,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                eta=args.eta,
                mode='extrapolate',
                verbose=False,
            )                                        # (n, B, S, Y, X)
        ens_mean = samples.mean(dim=0).cpu().numpy() # (B, S, Y, X)

        pred = ens_mean[0]                            # (S, Y, X)
        truth_b = target.cpu().numpy()[0]            # (S, Y, X)
        obs_b = obs_mask_real[0]                      # (S, Y, X)

        # per-species AUC at unobserved cells
        auc_dict = auc_per_species_at_unobs(pred, truth_b, obs_b)
        jac_dict = jaccard_per_species(pred, truth_b, thr=0.5)

        for s, val in auc_dict.items():
            all_auc_unobs[(w, s)] = val
        for s, val in jac_dict.items():
            all_jaccard[(w, s)] = val
        for s in range(truth_b.shape[0]):
            r = int((truth_b[s] > 0.5).sum())
            if r > 0:
                all_ranges[(w, s)] = r

        n_with_auc = sum(1 for k in all_auc_unobs if k[0] == w)
        elapsed = time.time() - t0
        print(f"\n  world {w + 1}/{args.n_worlds}  "
              f"({elapsed:.0f}s, ~{elapsed/(w+1):.0f}s/world)  "
              f"species_with_auc={n_with_auc}")

    # ---- stratified summary -----------------------------------------
    print("\n" + "=" * 78)
    print(f"  STRATIFIED auc_unobs  (K={args.K_obs})")
    print("=" * 78)
    print(f"  {'stratum':<32} {'n_species':>10} {'mean':>8}"
          f" {'std':>8} {'95% CI':>10}")
    print(f"  {'-'*32} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    auc_rows = stratify(all_auc_unobs, all_ranges, args.K_obs)
    for label, n, mean, std, ci in auc_rows:
        if n == 0:
            print(f"  {label:<32} {'(none)':>10}")
            continue
        marker = ""
        if not np.isnan(mean):
            if mean < 0.5:    marker = "  BELOW random"
            elif mean < 0.6:  marker = "  weak"
            elif mean < 0.7:  marker = "  moderate"
            else:             marker = "  strong"
        print(f"  {label:<32} {n:>10} {mean:>8.4f}"
              f" {std:>8.4f} {'±'+f'{ci:.4f}':>10}{marker}")

    print("\n" + "=" * 78)
    print(f"  STRATIFIED Jaccard (full grid, thr=0.5)")
    print("=" * 78)
    print(f"  {'stratum':<32} {'n_species':>10} {'mean':>8}"
          f" {'std':>8} {'95% CI':>10}")
    print(f"  {'-'*32} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    jac_rows = stratify(all_jaccard, all_ranges, args.K_obs)
    for label, n, mean, std, ci in jac_rows:
        if n == 0:
            print(f"  {label:<32} {'(none)':>10}")
            continue
        print(f"  {label:<32} {n:>10} {mean:>8.4f}"
              f" {std:>8.4f} {'±'+f'{ci:.4f}':>10}")

    # ---- the verdict ------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  ROOT-CAUSE VERDICT")
    print("=" * 78)

    # find auc_unobs for Trivial and for Wide
    auc_by_label = {row[0]: row[2] for row in auc_rows if row[1] > 0}
    n_by_label = {row[0]: row[1] for row in auc_rows if row[1] > 0}

    trivial_label = "Trivial  (range <= K)"
    wide_label    = "Wide     (r > 4K)"
    real_label    = "Real     (2K < r <= 4K)"
    meaningful_label = "Meaningful (r > K)  AGG"

    auc_trivial = auc_by_label.get(trivial_label, float('nan'))
    auc_wide    = auc_by_label.get(wide_label, float('nan'))
    auc_real    = auc_by_label.get(real_label, float('nan'))
    auc_meaningful = auc_by_label.get(meaningful_label, float('nan'))

    n_trivial = n_by_label.get(trivial_label, 0)
    n_wide    = n_by_label.get(wide_label, 0)
    n_real    = n_by_label.get(real_label, 0)
    n_meaningful = n_by_label.get(meaningful_label, 0)

    print(f"\n  auc_unobs by stratum:")
    print(f"    Trivial (r<=K)         = {auc_trivial:.4f}  (n={n_trivial})  "
          f"copy is correct here, expect ~0.5 if model copies, higher if env helps")
    print(f"    Real    (2K<r<=4K)     = {auc_real:.4f}  (n={n_real})  "
          f"genuine extrapolation needed")
    print(f"    Wide    (r>4K)         = {auc_wide:.4f}  (n={n_wide})  "
          f"Axel's scenario — must extrapolate")
    print(f"    Meaningful (r>K) AGG   = {auc_meaningful:.4f}  (n={n_meaningful})  "
          f"the metric for reporting")

    print(f"\n  Diagnosis:")
    if np.isnan(auc_wide) or n_wide < 5:
        print(f"    Not enough wide-range species ({n_wide}) in these {args.n_worlds} worlds")
        print(f"    to draw a conclusion about extrapolation. Try --n-worlds 10 or")
        print(f"    sample more worlds with --K-obs lower (e.g. K_obs=3) to enlarge")
        print(f"    the 'wide' stratum relative to K.")
    elif auc_wide >= 0.7:
        print(f"    >>> The model EXTRAPOLATES on wide-range species ({auc_wide:.3f}). <<<")
        print(f"    The headline `auc_unobs = 0.354` was being dragged down by the")
        print(f"    trivial-range majority. Fix A+B is more successful than the")
        print(f"    aggregate suggested. Next step: report the stratified result")
        print(f"    to Axel; do NOT change the architecture.")
    elif auc_wide >= 0.55:
        print(f"    >>> The model PARTIALLY extrapolates ({auc_wide:.3f}). <<<")
        print(f"    Fix A+B is working but not strongly. The next intervention is")
        print(f"    a HYPERPARAMETER pass — raise extrap_weight 0.5 -> 1.5 and")
        print(f"    presence_weight 20 -> 50 in fix_a_b_extrapolation.install_all().")
        print(f"    Retrain from scratch (~60h). DO NOT yet change the architecture.")
    else:
        print(f"    >>> The model DOES NOT extrapolate even on wide-range species")
        print(f"        ({auc_wide:.3f} is at-or-below random). <<<")
        print(f"    Fix A imposed pressure (verified by train_extrapolation key)")
        print(f"    but the model could not reduce it through its current input")
        print(f"    representation. Cause: the U-Net's input Conv2d cannot")
        print(f"    disentangle env (clean, bounded) from noisy x_t (std~1 at")
        print(f"    high t) — env channel is treated as noise.")
        print(f"")
        print(f"    Architectural fix justified: zero-init gated injection")
        print(f"    (ControlNet's mechanism). x_t through its normal Conv2d,")
        print(f"    spatial channels through a SEPARATE Conv2d initialised to")
        print(f"    zero — forces the model to learn spatial-channel influence")
        print(f"    from zero, no more 'env-is-noise' local minimum.")

    # ---- save ---------------------------------------------------------
    out = {
        "args": vars(args),
        "checkpoint_epoch": int(ckpt.get('epoch', -1)),
        "checkpoint_best_metric": float(ckpt.get('best_metric', 0.0)),
        "auc_unobs_strata": [
            {"stratum": r[0], "n_species": r[1], "mean": r[2],
             "std": r[3], "ci95": r[4]}
            for r in auc_rows
        ],
        "jaccard_strata": [
            {"stratum": r[0], "n_species": r[1], "mean": r[2],
             "std": r[3], "ci95": r[4]}
            for r in jac_rows
        ],
    }
    json_path = out_dir / "stratified_diagnostic.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  results saved -> {json_path}")


if __name__ == "__main__":
    main()