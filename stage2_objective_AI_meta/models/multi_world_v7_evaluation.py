#!/usr/bin/env python3
"""
=============================================================================
MULTI-WORLD V7 INFERENCE + STRATIFIED COVERAGE
=============================================================================

For PhD-quality Objective 2 evaluation, single-world numbers are not
sufficient. This script:

  1. Reads your wide_range_species.csv to identify the worlds with the
     most wide-range species (i.e. where reconstruction is meaningful).

  2. For each such world, checks if v7 inpainting inference has been run.
     If not, it skips that world (you'd need to run inference separately
     — instructions printed at the end).

  3. Computes the stratified coverage analysis on each world.

  4. Aggregates the results into a population-level table:
     - Total meaningful species across all worlds
     - Distribution of recall across worlds
     - Per-K analysis (K=5 and K=10)

  5. Outputs a master CSV with one row per world, plus a summary line.

USAGE
-----
    python multi_world_v7_evaluation.py \\
        --wide-range-csv  ./figures_map_axel_stage2_new/wide_range_species.csv \\
        --recon-dir-pattern  './reconstructions_v7_inpaint_{world_stem}_stage2' \\
        --truth-dir       ./results/data \\
        --K               5 \\
        --top-n-worlds    10 \\
        --output-csv      ./figures_map_axel_stage2_new/multi_world_K5_summary.csv

The recon-dir-pattern uses {world_stem} which gets replaced with the world
filename without the .npz extension. So if the pattern is:
    ./reconstructions_v7_inpaint_{world_stem}_stage2
And a world filename is:
    pool22510000_batcha_ls10p0_vr0p001_thr3p0_env123_grid20x20_dr2em08_ld0p06_training.npz
The script looks in:
    ./reconstructions_v7_inpaint_pool22510000_..._training_stage2/

Adjust the pattern to match your actual directory naming.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def evaluate_world(truth_path, samples_path, K, calibrate='match_truth'):
    """Run stratified analysis on one world. Returns dict of per-stratum stats."""
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

    # Calibrate
    binary_samples = np.zeros_like(samples, dtype=np.uint8)
    for i in range(n_ens):
        binary_samples[i] = calibrate_per_species(samples[i], truth, mode=calibrate)
    binary_mean = calibrate_per_species(mean_pred, truth, mode=calibrate)
    ensemble_union = (binary_samples.sum(axis=0) > 0).astype(np.uint8)

    # Per-species
    rows = []
    for s in range(n_use):
        n_t = int(truth[s].sum())
        if n_t == 0:
            continue
        n_pred_mean = int(binary_mean[s].sum())
        n_correct_mean = int((binary_mean[s] & truth[s]).sum())
        n_pred_union = int(ensemble_union[s].sum())
        n_correct_union = int((ensemble_union[s] & truth[s]).sum())
        rows.append({
            'truth_cells': n_t,
            'mean_correct': n_correct_mean,
            'mean_recall': n_correct_mean / n_t,
            'union_correct': n_correct_union,
            'union_recall': n_correct_union / n_t,
        })

    # Stratify
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
    ap.add_argument('--recon-dir-pattern', required=True,
                    help='Pattern with {world_stem} placeholder')
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--top-n-worlds', type=int, default=10)
    ap.add_argument('--calibrate', default='match_truth')
    ap.add_argument('--output-csv', required=True)
    args = ap.parse_args()

    # Load wide-range CSV, count wide-range species per world
    world_species_count = defaultdict(int)
    with open(args.wide_range_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            world_species_count[row['world']] += 1

    # Sort worlds by wide-range species count
    top_worlds = sorted(world_species_count.items(),
                         key=lambda x: -x[1])[:args.top_n_worlds]
    print(f"Top {args.top_n_worlds} worlds by wide-range species count:")
    for w, n in top_worlds:
        print(f"  {n:>3} species — {w}")

    # Run evaluation on each world
    results = []
    missing_worlds = []
    truth_dir = Path(args.truth_dir)

    print(f"\nEvaluating each world with K={args.K} v7 inpainting samples...")
    for world_name, n_wide in top_worlds:
        truth_path = truth_dir / world_name
        world_stem = world_name.replace('.npz', '')

        # Build expected recon dir from pattern
        recon_dir = Path(args.recon_dir_pattern.format(world_stem=world_stem))
        samples_path = recon_dir / f'recon_fixed_b{args.K}_samples.npz'

        if not samples_path.exists():
            print(f"  ⚠ {world_name}: samples file missing at {samples_path}")
            missing_worlds.append((world_name, str(samples_path)))
            continue

        try:
            stats = evaluate_world(truth_path, samples_path, args.K,
                                    calibrate=args.calibrate)
            world_result = {
                'world': world_name,
                'n_wide_range_in_csv': n_wide,
                **{f'meaningful_{k}': v for k, v in (stats['meaningful'] or {}).items()},
                **{f'real_{k}': v for k, v in (stats['real'] or {}).items()},
            }
            results.append(world_result)
            mn = stats['meaningful']
            if mn:
                print(f"  ✓ {world_name[:60]}...")
                print(f"      meaningful: {mn['n_species']} sp, "
                      f"mean_recall={mn['mean_recall']:.3f}, "
                      f"pix_cov_mean={mn['pix_cov_mean']:.3f}")
        except Exception as e:
            print(f"  ✗ {world_name}: error {e}")

    # Aggregate across worlds
    print(f"\n{'='*72}")
    print(f"  POPULATION-LEVEL SUMMARY  (K={args.K})")
    print(f"{'='*72}")

    if not results:
        print("\n  No results — run inference on the worlds first.")
    else:
        n_worlds = len(results)
        all_meaningful_n = sum(r.get('meaningful_n_species', 0) for r in results)
        all_meaningful_truth = sum(r.get('meaningful_truth_cells', 0) for r in results)

        # Cross-world aggregates
        mean_recalls = [r['meaningful_mean_recall'] for r in results
                        if 'meaningful_mean_recall' in r]
        union_recalls = [r['meaningful_union_recall'] for r in results
                         if 'meaningful_union_recall' in r]
        pix_cov_means = [r['meaningful_pix_cov_mean'] for r in results
                         if 'meaningful_pix_cov_mean' in r]
        pix_cov_unions = [r['meaningful_pix_cov_union'] for r in results
                          if 'meaningful_pix_cov_union' in r]

        print(f"\n  Worlds evaluated: {n_worlds}")
        print(f"  Total meaningful species across worlds: {all_meaningful_n:,}")
        print(f"  Total meaningful truth cells: {all_meaningful_truth:,}")
        print(f"\n  Per-species mean recall:    "
              f"{np.mean(mean_recalls):.1%} ± {np.std(mean_recalls):.1%}  "
              f"[range {min(mean_recalls):.1%} – {max(mean_recalls):.1%}]")
        print(f"  Per-species union recall:   "
              f"{np.mean(union_recalls):.1%} ± {np.std(union_recalls):.1%}")
        print(f"  Pixel coverage by mean:     "
              f"{np.mean(pix_cov_means):.1%} ± {np.std(pix_cov_means):.1%}")
        print(f"  Pixel coverage by union:    "
              f"{np.mean(pix_cov_unions):.1%} ± {np.std(pix_cov_unions):.1%}")

        # Save CSV
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

    # Print missing worlds with how to run them
    if missing_worlds:
        print(f"\n{'='*72}")
        print(f"  MISSING INFERENCE FOR {len(missing_worlds)} WORLDS")
        print(f"{'='*72}")
        print(f"  Run these to complete the multi-world evaluation:\n")
        for world_name, expected_path in missing_worlds[:3]:
            print(f"  python AI_simulation/stage2/models/generate_reconstructions_v7_inpaint.py \\")
            print(f"      --stage2-dir       AI_simulation/stage2 \\")
            print(f"      --checkpoint       ./stage2_outputs_new/checkpoints/best_model.pt \\")
            print(f"      --truth-npz        ./results/data/{world_name} \\")
            print(f"      --output-dir       {Path(expected_path).parent} \\")
            print(f"      --fixed-budgets    {args.K} \\")
            print(f"      --mode             inpaint \\")
            print(f"      --repaint-iterations 2 \\")
            print(f"      --n-ensemble       8")
            print()
        if len(missing_worlds) > 3:
            print(f"  ... and {len(missing_worlds) - 3} more (same pattern)")


if __name__ == "__main__":
    main()