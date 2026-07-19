#!/usr/bin/env python3
"""
=============================================================================
ENSEMBLE COVERAGE — quantitative answer to Axel's framing
=============================================================================

Axel's email: "we could show that in some statistical sense the ground
truth is part of this ensemble."

This script computes that statistic. For each species, it asks:

  Across all 8 ensemble samples, what fraction of TRUTH cells is
  predicted by AT LEAST ONE sample (using calibrated per-species
  thresholds)?

If the number is high (>50%), the truth is genuinely "in" the ensemble
support — Axel's framing is satisfied. If it's low (<30%), the model
isn't capturing the truth even probabilistically, and we need calibration.

The script also computes:
  - Per-species precision and recall of the ensemble UNION
  - Per-species precision and recall of the ensemble MEAN (single shot)
  - Across all species (including rare ones, not just the 5 widespread)

USAGE
-----
    python compute_ensemble_coverage.py \\
        --truth-npz     ./results/data/<world>.npz \\
        --samples-npz   ./reconstructions_v2_phase4_stage2_map_axel/recon_fixed_b5_samples.npz \\
        --output-csv    ./figures_map_axel_stage2_new/ensemble_coverage_K5.csv
"""

import argparse
import numpy as np
from pathlib import Path


def calibrate_per_species(prob, truth, mode='match_truth'):
    """Per-species threshold matching predicted area to truth area."""
    S, Y, X = prob.shape
    binary = np.zeros_like(prob, dtype=np.uint8)
    if mode == 'fixed_05':
        return (prob > 0.5).astype(np.uint8)
    multiplier = 1.0 if mode == 'match_truth' else 2.0
    for s in range(S):
        n_truth = int(truth[s].sum())
        if n_truth == 0:
            continue
        n_target = max(1, int(n_truth * multiplier))
        flat = prob[s].ravel()
        if flat.max() < 1e-6:
            continue
        thr = np.partition(flat, -n_target)[-n_target] - 1e-9
        binary[s] = (prob[s] > thr).astype(np.uint8)
    return binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-npz', required=True)
    ap.add_argument('--samples-npz', required=True,
                    help='NPZ with shape (n_ensemble, S, Y, X) under "samples" key')
    ap.add_argument('--output-csv', default=None)
    ap.add_argument('--calibrate', default='match_truth',
                    choices=['match_truth', '2x_truth', 'fixed_05'])
    args = ap.parse_args()

    print(f"Loading truth: {args.truth_npz}")
    with np.load(args.truth_npz, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    S_truth = truth.shape[0]
    print(f"  truth shape: {truth.shape}")

    print(f"Loading ensemble samples: {args.samples_npz}")
    z = np.load(args.samples_npz)
    samples = np.asarray(z['samples']).astype(np.float32)  # (n_ens, S, Y, X)
    mean_pred = np.asarray(z['mean']).astype(np.float32)
    observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    n_ens = samples.shape[0]
    print(f"  samples shape: {samples.shape}  ({n_ens} ensemble members)")

    n_use = min(S_truth, samples.shape[1])
    truth = truth[:n_use]
    samples = samples[:, :n_use]
    mean_pred = mean_pred[:n_use]
    observed = observed[:n_use]

    # ── Calibrate each ensemble sample using per-species threshold ──
    print(f"\nCalibrating samples ({args.calibrate})...")
    binary_samples = np.zeros_like(samples, dtype=np.uint8)
    for i in range(n_ens):
        binary_samples[i] = calibrate_per_species(samples[i], truth,
                                                    mode=args.calibrate)
    binary_mean = calibrate_per_species(mean_pred, truth, mode=args.calibrate)

    # ── Compute ENSEMBLE UNION ──
    # A cell is predicted if ANY sample predicts it
    ensemble_union = (binary_samples.sum(axis=0) > 0).astype(np.uint8)

    # ── Per-species statistics ──
    print(f"\nComputing per-species coverage statistics...")
    n_present = int((truth.sum(axis=(1, 2)) > 0).sum())
    print(f"  present species: {n_present}/{n_use}")

    rows = []
    coverage_by_range = {'1-2': [], '3-5': [], '6-10': [], '11-20': [], '21+': []}

    for s in range(n_use):
        n_t = int(truth[s].sum())
        if n_t == 0:
            continue
        n_o = int(observed[s].sum())
        # Mean (single-shot)
        mean_correct = int((binary_mean[s] & truth[s]).sum())
        mean_pred_count = int(binary_mean[s].sum())
        # Union
        union_correct = int((ensemble_union[s] & truth[s]).sum())
        union_pred_count = int(ensemble_union[s].sum())

        recall_mean = mean_correct / n_t
        precision_mean = mean_correct / max(1, mean_pred_count)
        recall_union = union_correct / n_t
        precision_union = union_correct / max(1, union_pred_count)

        rows.append({
            'species': s, 'truth_cells': n_t, 'obs_cells': n_o,
            'mean_pred_cells': mean_pred_count,
            'mean_correct': mean_correct,
            'mean_precision': precision_mean,
            'mean_recall': recall_mean,
            'union_pred_cells': union_pred_count,
            'union_correct': union_correct,
            'union_precision': precision_union,
            'union_recall': recall_union,
        })

        # Bucket by truth range size
        if n_t <= 2:
            bucket = '1-2'
        elif n_t <= 5:
            bucket = '3-5'
        elif n_t <= 10:
            bucket = '6-10'
        elif n_t <= 20:
            bucket = '11-20'
        else:
            bucket = '21+'
        coverage_by_range[bucket].append({
            'recall_mean': recall_mean, 'recall_union': recall_union,
            'precision_mean': precision_mean, 'precision_union': precision_union,
        })

    # ── Aggregate statistics ──
    print(f"\n{'='*72}")
    print(f"  AGGREGATE STATISTICS  (K=5, calibration={args.calibrate})")
    print(f"{'='*72}")

    all_recall_mean = np.array([r['mean_recall'] for r in rows])
    all_recall_union = np.array([r['union_recall'] for r in rows])
    all_prec_mean = np.array([r['mean_precision'] for r in rows])
    all_prec_union = np.array([r['union_precision'] for r in rows])

    print(f"\n  Number of present species: {len(rows)}")
    print(f"\n  SINGLE-SHOT (ensemble mean) per-species averages:")
    print(f"    recall:    {all_recall_mean.mean():.3f}  ± {all_recall_mean.std():.3f}")
    print(f"    precision: {all_prec_mean.mean():.3f}  ± {all_prec_mean.std():.3f}")
    print(f"\n  ENSEMBLE UNION (across {n_ens} samples) per-species averages:")
    print(f"    recall:    {all_recall_union.mean():.3f}  ± {all_recall_union.std():.3f}  "
          f"<- THIS is 'truth in ensemble' rate")
    print(f"    precision: {all_prec_union.mean():.3f}  ± {all_prec_union.std():.3f}")

    print(f"\n  Recall by truth-range size (truth IS in ensemble | range = R):")
    print(f"  {'range':>8} {'n_sp':>6} {'mean_recall':>14} {'union_recall':>14}")
    print(f"  {'-'*60}")
    for bucket, vals in coverage_by_range.items():
        if not vals:
            continue
        rec_mean = np.mean([v['recall_mean'] for v in vals])
        rec_union = np.mean([v['recall_union'] for v in vals])
        print(f"  {bucket:>8} {len(vals):>6d} {rec_mean:>14.3f} {rec_union:>14.3f}")

    # ── Overall pixel-level coverage (across ALL truth pixels) ──
    total_truth_cells = int(truth.sum())
    union_correct_total = int((ensemble_union & truth).sum())
    mean_correct_total = int((binary_mean & truth).sum())
    print(f"\n  PIXEL-LEVEL TOTALS (across all species and cells):")
    print(f"    total truth presence cells:     {total_truth_cells:>8,}")
    print(f"    captured by ensemble UNION:     {union_correct_total:>8,} "
          f"({100*union_correct_total/total_truth_cells:.1f}%)")
    print(f"    captured by ensemble MEAN:      {mean_correct_total:>8,} "
          f"({100*mean_correct_total/total_truth_cells:.1f}%)")

    # ── For Axel's email: the headline number ──
    print(f"\n  {'='*68}")
    print(f"  HEADLINE (for Axel's email):")
    print(f"  At K=5 observations per species, the ensemble of {n_ens} samples")
    print(f"  contains {100*union_correct_total/total_truth_cells:.1f}% of true presence cells")
    print(f"  in its union (per-species truth-area calibration), vs")
    print(f"  {100*mean_correct_total/total_truth_cells:.1f}% for a single mean reconstruction.")
    print(f"  {'='*68}")

    # ── Save CSV ──
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            f.write("species,truth_cells,obs_cells,"
                    "mean_pred_cells,mean_correct,mean_precision,mean_recall,"
                    "union_pred_cells,union_correct,union_precision,union_recall\n")
            for r in rows:
                f.write(f"{r['species']},{r['truth_cells']},{r['obs_cells']},"
                        f"{r['mean_pred_cells']},{r['mean_correct']},"
                        f"{r['mean_precision']:.4f},{r['mean_recall']:.4f},"
                        f"{r['union_pred_cells']},{r['union_correct']},"
                        f"{r['union_precision']:.4f},{r['union_recall']:.4f}\n")
        print(f"\n  ✓ wrote per-species stats to {out}")


if __name__ == "__main__":
    main()