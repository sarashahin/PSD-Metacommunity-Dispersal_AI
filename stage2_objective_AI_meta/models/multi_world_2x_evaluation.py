#!/usr/bin/env python3
"""
=============================================================================
MULTI-WORLD 2X_TRUTH COVERAGE AGGREGATION
=============================================================================

Companion script to multi_world_v7_evaluation.py. Where that script ran
1x calibration (match_truth = predicted area equals truth area), this one
runs 2x calibration (predicted area = 2× truth area) and aggregates across
all 10 worlds.

WHY 2X CALIBRATION MATTERS
--------------------------
With K=5 observations on a 25-cell range, the model genuinely cannot pin
down the exact range edges from spatial information alone. A 2× envelope
acknowledges this and asks "does the predicted region CONTAIN the truth?"
rather than "do the predicted cells EXACTLY match the truth?"

This corresponds directly to Axel's email framing: "ground truth is part
of this ensemble". The 2× envelope IS the ensemble support.

In SDM literature this is called "potential range" (where species could be
given environmental suitability) vs "realized range" (where it actually is).
Both are scientifically valid; reporting both gives a fuller picture.

USAGE
-----
    python multi_world_2x_evaluation.py \\
        --wide-range-csv  ./figures_map_axel_stage2_new/wide_range_species.csv \\
        --recon-dir-pattern  './reconstructions_v7_inpaint_{world_stem}_stage2' \\
        --truth-dir       ./results/data \\
        --K               5 \\
        --top-n-worlds    10 \\
        --output-csv      ./figures_map_axel_stage2_new/multi_world_K5_2x_summary.csv

The output CSV has the same schema as multi_world_K5_summary.csv but with
2x calibration applied.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def calibrate_per_species(prob, truth, mode='match_truth'):
    S = prob.shape[0]
    binary = np.zeros_like(prob, dtype=np.uint8)
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


def evaluate_world(truth_path, samples_path, K, calibrate='2x_truth'):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)

    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    mean_pred = np.asarray(z['mean']).astype(np.float32)
    n_ens = samples.shape[0]

    n_use = min(truth.shape[0], samples.shape[1])
    truth = truth[:n_use]
    samples = samples[:, :n_use]
    mean_pred = mean_pred[:n_use]

    binary_samples = np.zeros_like(samples, dtype=np.uint8)
    for i in range(n_ens):
        binary_samples[i] = calibrate_per_species(samples[i], truth,
                                                    mode=calibrate)
    binary_mean = calibrate_per_species(mean_pred, truth, mode=calibrate)
    ensemble_union = (binary_samples.sum(axis=0) > 0).astype(np.uint8)

    rows = []
    for s in range(n_use):
        n_t = int(truth[s].sum())
        if n_t == 0:
            continue
        n_correct_mean = int((binary_mean[s] & truth[s]).sum())
        n_correct_union = int((ensemble_union[s] & truth[s]).sum())
        rows.append({
            'truth_cells': n_t,
            'mean_correct': n_correct_mean,
            'mean_recall': n_correct_mean / n_t,
            'union_correct': n_correct_union,
            'union_recall': n_correct_union / n_t,
        })

    meaningful = [r for r in rows if r['truth_cells'] > K]
    real = [r for r in rows if r['truth_cells'] > 2 * K]

    def agg(stratum):
        if not stratum:
            return None
        truth_total = sum(r['truth_cells'] for r in stratum)
        return {
            'n_species': len(stratum),
            'truth_cells': truth_total,
            'mean_recall': float(np.mean([r['mean_recall'] for r in stratum])),
            'union_recall': float(np.mean([r['union_recall'] for r in stratum])),
            'pix_cov_mean': sum(r['mean_correct'] for r in stratum) / max(1, truth_total),
            'pix_cov_union': sum(r['union_correct'] for r in stratum) / max(1, truth_total),
        }

    return {
        'meaningful': agg(meaningful),
        'real': agg(real),
        'all_present': agg(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide-range-csv', required=True)
    ap.add_argument('--recon-dir-pattern', required=True)
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--top-n-worlds', type=int, default=10)
    ap.add_argument('--calibrate', default='2x_truth',
                    choices=['match_truth', '2x_truth'])
    ap.add_argument('--output-csv', required=True)
    args = ap.parse_args()

    world_species_count = defaultdict(int)
    with open(args.wide_range_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            world_species_count[row['world']] += 1

    top_worlds = sorted(world_species_count.items(),
                         key=lambda x: -x[1])[:args.top_n_worlds]
    print(f"Top {args.top_n_worlds} worlds:")
    for w, n in top_worlds:
        print(f"  {n:>3} sp — {w}")

    results = []
    truth_dir = Path(args.truth_dir)

    print(f"\nEvaluating each world with K={args.K} v7 inpainting samples "
          f"({args.calibrate})...")
    for world_name, n_wide in top_worlds:
        truth_path = truth_dir / world_name
        world_stem = world_name.replace('.npz', '')
        recon_dir = Path(args.recon_dir_pattern.format(world_stem=world_stem))
        samples_path = recon_dir / f'recon_fixed_b{args.K}_samples.npz'

        if not samples_path.exists():
            print(f"  ⚠ skip {world_name}: samples missing")
            continue

        try:
            stats = evaluate_world(truth_path, samples_path, args.K,
                                    calibrate=args.calibrate)
            mn = stats['meaningful']
            world_result = {'world': world_name, 'n_wide_range_in_csv': n_wide}
            if mn:
                world_result.update({f'meaningful_{k}': v for k, v in mn.items()})
            if stats['real']:
                world_result.update({f'real_{k}': v
                                     for k, v in stats['real'].items()})
            results.append(world_result)
            if mn:
                print(f"  ✓ {world_name[:60]}...")
                print(f"      meaningful: {mn['n_species']} sp, "
                      f"mean_recall={mn['mean_recall']:.3f}, "
                      f"union_recall={mn['union_recall']:.3f}, "
                      f"pix_cov_union={mn['pix_cov_union']:.3f}")
        except Exception as e:
            print(f"  ✗ {world_name}: {e}")

    # Aggregate
    print(f"\n{'='*72}")
    print(f"  POPULATION SUMMARY  (K={args.K}, calibration={args.calibrate})")
    print(f"{'='*72}")

    if results:
        n_worlds = len(results)
        all_meaningful_n = sum(r.get('meaningful_n_species', 0) for r in results)
        mean_recalls = [r['meaningful_mean_recall'] for r in results
                        if 'meaningful_mean_recall' in r]
        union_recalls = [r['meaningful_union_recall'] for r in results
                         if 'meaningful_union_recall' in r]
        pix_means = [r['meaningful_pix_cov_mean'] for r in results
                     if 'meaningful_pix_cov_mean' in r]
        pix_unions = [r['meaningful_pix_cov_union'] for r in results
                      if 'meaningful_pix_cov_union' in r]

        print(f"\n  Worlds: {n_worlds}, total meaningful species: {all_meaningful_n}")
        print(f"  mean_recall:    {np.mean(mean_recalls):.1%} ± {np.std(mean_recalls):.1%}")
        print(f"  union_recall:   {np.mean(union_recalls):.1%} ± {np.std(union_recalls):.1%}")
        print(f"  pix_cov_mean:   {np.mean(pix_means):.1%} ± {np.std(pix_means):.1%}")
        print(f"  pix_cov_union:  {np.mean(pix_unions):.1%} ± {np.std(pix_unions):.1%}")

    # Save CSV with same schema as 1x summary
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if results:
        keys = sorted({k for r in results for k in r.keys()})
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
    print(f"\n  ✓ wrote {out}")


if __name__ == "__main__":
    main()