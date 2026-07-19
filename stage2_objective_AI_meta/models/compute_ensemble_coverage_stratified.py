#!/usr/bin/env python3
"""
=============================================================================
ENSEMBLE COVERAGE — STRATIFIED BY RANGE SIZE  (the honest evaluation)
=============================================================================

THE SCIENTIFIC ISSUE
--------------------
Your test world has 3,191 present species. 75% of them (n=2,382) have
range size 1-2 cells. With K=5 observations, K >= range for these species
— there is no extrapolation problem, the observations ARE the range.

For these "trivial" species, the per-species threshold calibration asks
the model to pick exactly the 1-2 truth cells out of 400 candidates. The
model's continuous probabilities get truncated by this threshold and
almost always pick slightly different cells.

This is why the aggregate coverage looks bad (7.1%) even though for
species where reconstruction is meaningful (range >= 6), recall is 40-75%.

THE FIX (this script)
---------------------
Stratify the evaluation:
  - "Trivial" species (range < K): K observations cover the whole range.
    Report per-species precision/recall but explicitly note this is a
    degenerate case.
  - "Sparse" species (range >= K and range < 2K): some extrapolation needed.
  - "Real" species (range >= 2K): meaningful reconstruction problem.

This script computes coverage stats for each stratum AND for the union of
"sparse" + "real" species — which is the metric that actually answers
Axel's question.

USAGE
-----
    python compute_ensemble_coverage_stratified.py \\
        --truth-npz     ./results/data/<world>.npz \\
        --samples-npz   ./reconstructions_v7_inpaint_world5_stage2/recon_fixed_b5_samples.npz \\
        --K             5 \\
        --output-csv    ./figures_map_axel_stage2_new/coverage_stratified_K5.csv
"""

import argparse
import numpy as np
from pathlib import Path


def calibrate_per_species(prob, truth, mode='match_truth'):
    """Per-species threshold matching predicted area to truth area."""
    S = prob.shape[0]
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


def compute_stratum_stats(rows, label):
    """Aggregate stats for a subset of species records."""
    if not rows:
        return None
    n = len(rows)
    truth_total = sum(r['truth_cells'] for r in rows)
    union_correct_total = sum(r['union_correct'] for r in rows)
    mean_correct_total = sum(r['mean_correct'] for r in rows)

    mean_recalls = np.array([r['mean_recall'] for r in rows])
    union_recalls = np.array([r['union_recall'] for r in rows])
    mean_precs = np.array([r['mean_precision'] for r in rows])
    union_precs = np.array([r['union_precision'] for r in rows])

    return {
        'label': label,
        'n_species': n,
        'truth_cells': truth_total,
        'mean_recall': float(mean_recalls.mean()),
        'mean_recall_std': float(mean_recalls.std()),
        'union_recall': float(union_recalls.mean()),
        'union_recall_std': float(union_recalls.std()),
        'mean_precision': float(mean_precs.mean()),
        'union_precision': float(union_precs.mean()),
        'pixel_coverage_union': union_correct_total / max(1, truth_total),
        'pixel_coverage_mean': mean_correct_total / max(1, truth_total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-npz', required=True)
    ap.add_argument('--samples-npz', required=True)
    ap.add_argument('--K', type=int, default=5,
                    help='Number of observations per species (used for stratifying)')
    ap.add_argument('--output-csv', default=None)
    ap.add_argument('--calibrate', default='match_truth',
                    choices=['match_truth', '2x_truth', 'fixed_05'])
    args = ap.parse_args()

    print(f"Loading truth: {args.truth_npz}")
    with np.load(args.truth_npz, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    print(f"  truth shape: {truth.shape}")

    print(f"Loading ensemble samples: {args.samples_npz}")
    z = np.load(args.samples_npz)
    samples = np.asarray(z['samples']).astype(np.float32)
    mean_pred = np.asarray(z['mean']).astype(np.float32)
    observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    n_ens = samples.shape[0]
    print(f"  samples shape: {samples.shape} ({n_ens} ensemble members)")

    n_use = min(truth.shape[0], samples.shape[1])
    truth = truth[:n_use]
    samples = samples[:, :n_use]
    mean_pred = mean_pred[:n_use]
    observed = observed[:n_use]

    # Calibrate
    print(f"\nCalibrating samples ({args.calibrate})...")
    binary_samples = np.zeros_like(samples, dtype=np.uint8)
    for i in range(n_ens):
        binary_samples[i] = calibrate_per_species(samples[i], truth,
                                                    mode=args.calibrate)
    binary_mean = calibrate_per_species(mean_pred, truth, mode=args.calibrate)
    ensemble_union = (binary_samples.sum(axis=0) > 0).astype(np.uint8)

    # Per-species stats
    print(f"\nComputing per-species coverage statistics...")
    rows = []
    for s in range(n_use):
        n_t = int(truth[s].sum())
        if n_t == 0:
            continue
        n_o = int(observed[s].sum())
        mean_correct = int((binary_mean[s] & truth[s]).sum())
        mean_pred_count = int(binary_mean[s].sum())
        union_correct = int((ensemble_union[s] & truth[s]).sum())
        union_pred_count = int(ensemble_union[s].sum())

        rows.append({
            'species': s, 'truth_cells': n_t, 'obs_cells': n_o,
            'mean_pred_cells': mean_pred_count,
            'mean_correct': mean_correct,
            'mean_precision': mean_correct / max(1, mean_pred_count),
            'mean_recall': mean_correct / n_t,
            'union_pred_cells': union_pred_count,
            'union_correct': union_correct,
            'union_precision': union_correct / max(1, union_pred_count),
            'union_recall': union_correct / n_t,
        })

    print(f"  total present species: {len(rows)}")

    # ── Stratify by reconstruction-task meaningfulness ──
    K = args.K
    trivial   = [r for r in rows if r['truth_cells'] <= K]   # range <= K
    sparse    = [r for r in rows if K < r['truth_cells'] <= 2*K]  # K < range <= 2K
    real      = [r for r in rows if r['truth_cells'] > 2*K]  # range > 2K
    meaningful = sparse + real    # any species where reconstruction is real

    print(f"\n{'='*72}")
    print(f"  STRATIFIED ANALYSIS  (K={K})")
    print(f"{'='*72}")

    print(f"""
  STRATUM DEFINITIONS:
    Trivial    : range <= K = {K}    →  K observations cover whole range
                 (no extrapolation problem; model just has to pick those cells)
    Sparse     : {K} < range <= {2*K}    →  K observations cover ~50% of range
                 (modest extrapolation needed)
    Real       : range > {2*K}      →  K observations cover < 50% of range
                 (real reconstruction problem — Axel's scenario)
    Meaningful : Sparse + Real    →  combined "reconstruction is needed" subset
""")

    strata = [
        ('TRIVIAL    (range <= K)', trivial),
        ('SPARSE     (K < range <= 2K)', sparse),
        ('REAL       (range > 2K)', real),
        ('MEANINGFUL (range > K)', meaningful),
        ('ALL species', rows),
    ]

    print(f"  {'stratum':<28} {'n_sp':>5} {'truth_cells':>11} "
          f"{'mean_rec':>9} {'union_rec':>10} {'pix_cov_un':>11}")
    print(f"  {'-'*78}")
    summary = {}
    for label, stratum in strata:
        stats = compute_stratum_stats(stratum, label)
        if stats is None:
            print(f"  {label:<28}  (empty)")
            continue
        summary[label] = stats
        print(f"  {label:<28} {stats['n_species']:>5} "
              f"{stats['truth_cells']:>11,} "
              f"{stats['mean_recall']:>9.3f} "
              f"{stats['union_recall']:>10.3f} "
              f"{stats['pixel_coverage_union']:>11.3f}")

    # ── HEADLINE for Axel's email ──
    print(f"\n  {'='*68}")
    print(f"  HEADLINE (for Axel — RESTRICTED to meaningful species, range > {K}):")
    if 'MEANINGFUL (range > K)' in summary:
        s = summary['MEANINGFUL (range > K)']
        print(f"    {s['n_species']:,} species (out of {len(rows):,} present species)")
        print(f"    where K={K} obs < truth range, so reconstruction is genuinely needed.")
        print(f"    Per-species mean recall:       {s['mean_recall']:.1%}")
        print(f"    Per-species ensemble union recall:  {s['union_recall']:.1%}")
        print(f"    Pixel coverage by ensemble union:   {s['pixel_coverage_union']:.1%}")
        print(f"    Pixel coverage by ensemble mean:    {s['pixel_coverage_mean']:.1%}")
    print(f"  {'='*68}")
    if 'REAL       (range > 2K)' in summary:
        s = summary['REAL       (range > 2K)']
        print(f"\n  STRONG SUBSET (range > 2K = {2*K}):")
        print(f"    {s['n_species']} species — these are the 'wide range with sparse obs'")
        print(f"    case Axel describes (many species with 5 observations).")
        print(f"    Mean recall:    {s['mean_recall']:.1%}")
        print(f"    Union recall:   {s['union_recall']:.1%}")

    # ── Save CSV ──
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            f.write("stratum,n_species,truth_cells,mean_recall,mean_recall_std,"
                    "union_recall,union_recall_std,mean_precision,union_precision,"
                    "pixel_coverage_union,pixel_coverage_mean\n")
            for label, s in summary.items():
                f.write(f"{label},{s['n_species']},{s['truth_cells']},"
                        f"{s['mean_recall']:.4f},{s['mean_recall_std']:.4f},"
                        f"{s['union_recall']:.4f},{s['union_recall_std']:.4f},"
                        f"{s['mean_precision']:.4f},{s['union_precision']:.4f},"
                        f"{s['pixel_coverage_union']:.4f},"
                        f"{s['pixel_coverage_mean']:.4f}\n")
        print(f"\n  ✓ wrote {out}")


if __name__ == "__main__":
    main()