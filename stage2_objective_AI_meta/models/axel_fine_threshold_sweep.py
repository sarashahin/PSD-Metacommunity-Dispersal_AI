#!/usr/bin/env python3
"""
=============================================================================
FINE THRESHOLD SWEEP  —  is there a single p threshold that passes all three?
=============================================================================

WHY THIS EXISTS
---------------
The coarse 3-axis sweep showed p≥0.7 is the closest single threshold:
    range_KS = 0.270 (PASS)
    conn_KS  = 0.331 (just 0.031 over bar)
    covdet_KS= 0.269 (PASS)

This script tests a finer grid:
    p ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85}

to find whether a slightly different threshold gets conn_KS under 0.30
without making range_KS too small.

This is the ONLY remaining test before we commit to per-statistic-optima
reporting. Runtime ~10 minutes on existing NPZs.

USAGE
-----
    python axel_fine_threshold_sweep.py \\
        --wide-range-csv     ./figures_map_axel_stage2_new/wide_range_species.csv \\
        --recon-dir-pattern  './reconstructions_spatial/{world_stem}' \\
        --truth-dir          ./results/data \\
        --K                  5 \\
        --output-dir         ./figures_map_axel_stage2_new/axel_fine_sweep

NO synthetic data. NO new sampling. Same logic as the three-axis script
but with a denser grid of fixed-probability thresholds.
=============================================================================
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats


GRID_Y, GRID_X = 20, 20
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)


def periodic_cov_det(binary_range, Y=GRID_Y, X=GRID_X):
    yy, xx = np.where(binary_range > 0.5)
    n = len(yy)
    if n < 2:
        return 0.0
    theta_y = 2.0 * np.pi * yy.astype(np.float64) / Y
    theta_x = 2.0 * np.pi * xx.astype(np.float64) / X
    mean_theta_y = np.arctan2(np.sin(theta_y).mean(), np.cos(theta_y).mean())
    mean_theta_x = np.arctan2(np.sin(theta_x).mean(), np.cos(theta_x).mean())
    diff_y = (theta_y - mean_theta_y + np.pi) % (2.0 * np.pi) - np.pi
    diff_x = (theta_x - mean_theta_x + np.pi) % (2.0 * np.pi) - np.pi
    dy = diff_y * Y / (2.0 * np.pi)
    dx = diff_x * X / (2.0 * np.pi)
    var_y = float(np.var(dy))
    var_x = float(np.var(dx))
    cov_yx = float(((dy - dy.mean()) * (dx - dx.mean())).mean())
    return max(0.0, var_y * var_x - cov_yx ** 2)


def count_components(binary_map):
    if binary_map.sum() == 0:
        return 0
    _, n = ndimage.label(binary_map, structure=CONNECTIVITY_STRUCTURE)
    return int(n)


# Fine grid of fixed-probability thresholds
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def compute_world(truth_path, samples_path, K):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)

    n_use = min(truth.shape[0], samples.shape[1])
    truth = truth[:n_use]; samples = samples[:, :n_use]
    n_ens = samples.shape[0]

    idx = [s for s in range(n_use) if int(truth[s].sum()) > K]
    if not idx:
        return None
    truth_m = truth[idx]
    samples_m = samples[:, idx]

    truth_range = truth_m.sum(axis=(1, 2)).astype(np.int32)
    truth_ncomp = np.asarray([count_components(t) for t in truth_m], dtype=np.int32)
    truth_logcd = np.asarray([np.log10(periodic_cov_det(t) + 1.0)
                                for t in truth_m], dtype=np.float64)

    out = {'truth_range': truth_range,
           'truth_ncomp': truth_ncomp,
           'truth_logcd': truth_logcd}
    for thr in THRESHOLDS:
        pr, pn, pc = [], [], []
        binary_all = (samples_m >= thr).astype(np.uint8)
        for k in range(n_ens):
            for s in range(len(idx)):
                if int(binary_all[k, s].sum()) >= 2:
                    pr.append(int(binary_all[k, s].sum()))
                    pn.append(count_components(binary_all[k, s]))
                    pc.append(np.log10(periodic_cov_det(binary_all[k, s]) + 1.0))
        key = f'p{int(round(thr * 100)):02d}'
        out[f'{key}_range'] = np.asarray(pr, dtype=np.int32)
        out[f'{key}_ncomp'] = np.asarray(pn, dtype=np.int32)
        out[f'{key}_logcd'] = np.asarray(pc, dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide-range-csv', required=True)
    ap.add_argument('--recon-dir-pattern', required=True)
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--top-n-worlds', type=int, default=30)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_sp_count = defaultdict(int)
    with open(args.wide_range_csv) as f:
        for row in csv.DictReader(f):
            world_sp_count[row['world']] += 1
    top_worlds = sorted(world_sp_count.items(), key=lambda x: -x[1])[:args.top_n_worlds]
    print(f"\n  Fine threshold sweep ({len(THRESHOLDS)} thresholds) "
          f"across {len(top_worlds)} worlds ...\n")

    pools = {'truth_range': [], 'truth_ncomp': [], 'truth_logcd': []}
    for thr in THRESHOLDS:
        key = f'p{int(round(thr * 100)):02d}'
        pools[f'{key}_range'] = []
        pools[f'{key}_ncomp'] = []
        pools[f'{key}_logcd'] = []

    truth_dir = Path(args.truth_dir)
    for world_name, _ in top_worlds:
        stem = world_name.replace('.npz', '')
        truth_path = truth_dir / world_name
        samples_path = Path(args.recon_dir_pattern.format(world_stem=stem)) \
                          / f'recon_fixed_b{args.K}_samples.npz'
        if not (truth_path.exists() and samples_path.exists()):
            print(f"    skip {world_name[:55]}: missing")
            continue
        r = compute_world(truth_path, samples_path, args.K)
        if r is None:
            continue
        for k, v in r.items():
            pools[k].append(v)
        print(f"    {world_name[:60]:60s}  truth_n={len(r['truth_range']):4d}")

    if not pools['truth_range']:
        print("\n  No usable worlds.")
        return

    for k in list(pools.keys()):
        pools[k] = np.concatenate(pools[k]) if pools[k] else np.array([])

    results = []
    for thr in THRESHOLDS:
        key = f'p{int(round(thr * 100)):02d}'
        row = {'threshold': f'p≥{thr:.2f}'}
        if len(pools[f'{key}_range']) > 0:
            row['range_ks']  = float(stats.ks_2samp(pools['truth_range'],
                                                      pools[f'{key}_range']).statistic)
            row['conn_ks']   = float(stats.ks_2samp(pools['truth_ncomp'],
                                                      pools[f'{key}_ncomp']).statistic)
            row['covdet_ks'] = float(stats.ks_2samp(pools['truth_logcd'],
                                                      pools[f'{key}_logcd']).statistic)
            row['max_ks']    = max(row['range_ks'], row['conn_ks'], row['covdet_ks'])
            row['avg_ks']    = (row['range_ks'] + row['conn_ks'] + row['covdet_ks']) / 3
        else:
            row.update({'range_ks': np.nan, 'conn_ks': np.nan, 'covdet_ks': np.nan,
                         'max_ks': np.nan, 'avg_ks': np.nan})
        results.append(row)

    print("\n" + "=" * 78)
    print(f"  FINE THRESHOLD SWEEP  (truth_n = {len(pools['truth_range']):,})")
    print("=" * 78)
    print(f"  {'threshold':<10s} {'range_KS':>10s} {'conn_KS':>10s} "
          f"{'covdet_KS':>10s} {'max':>8s} {'avg':>8s}  verdict")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}  {'-'*22}")
    for r in results:
        v = ''
        if not np.isnan(r['max_ks']):
            if r['max_ks'] <= 0.10:
                v = 'ALL EXCELLENT'
            elif r['max_ks'] <= 0.30:
                v = 'ALL PASS'
            else:
                fails = []
                if r['range_ks']  > 0.30: fails.append('range')
                if r['conn_ks']   > 0.30: fails.append('conn')
                if r['covdet_ks'] > 0.30: fails.append('covdet')
                v = 'fails: ' + ', '.join(fails)
        print(f"  {r['threshold']:<10s} {r['range_ks']:>10.3f} {r['conn_ks']:>10.3f} "
              f"{r['covdet_ks']:>10.3f} {r['max_ks']:>8.3f} {r['avg_ks']:>8.3f}  {v}")

    # Find a threshold where ALL three pass
    passing = [r for r in results if not np.isnan(r['max_ks']) and r['max_ks'] <= 0.30]
    if passing:
        best = min(passing, key=lambda r: r['max_ks'])
        print(f"\n  ✓ FOUND a single threshold where all three Axel (a) pass:")
        print(f"     {best['threshold']}  →  range {best['range_ks']:.3f}, "
              f"conn {best['conn_ks']:.3f}, covdet {best['covdet_ks']:.3f}")
        print(f"     Report this threshold to Axel as the calibrated ensemble cut.")
    else:
        # find the best balanced anyway
        best = min((r for r in results if not np.isnan(r['max_ks'])),
                    key=lambda r: r['max_ks'])
        print(f"\n  No single threshold in the fine grid passes all three at ≤ 0.30.")
        print(f"  Best balanced: {best['threshold']}  (max KS = {best['max_ks']:.3f})")
        print(f"  Falling back to per-statistic optima for write-up:")
        # Per-statistic best across the fine grid
        best_range  = min((r for r in results if not np.isnan(r['range_ks'])),
                          key=lambda r: r['range_ks'])
        best_conn   = min((r for r in results if not np.isnan(r['conn_ks'])),
                          key=lambda r: r['conn_ks'])
        best_covdet = min((r for r in results if not np.isnan(r['covdet_ks'])),
                          key=lambda r: r['covdet_ks'])
        print(f"     range  best:  {best_range['threshold']}    "
              f"KS = {best_range['range_ks']:.3f}")
        print(f"     conn   best:  {best_conn['threshold']}    "
              f"KS = {best_conn['conn_ks']:.3f}")
        print(f"     covdet best:  {best_covdet['threshold']}    "
              f"KS = {best_covdet['covdet_ks']:.3f}")

    # Figure
    fig, ax = plt.subplots(figsize=(13, 6.5))
    labels = [r['threshold'] for r in results]
    x = np.arange(len(labels))
    width = 0.27
    ax.bar(x - width, [r['range_ks'] for r in results], width,
            label='Range size', color='#5b8dd6', edgecolor='black')
    ax.bar(x, [r['conn_ks'] for r in results], width,
            label='Connectivity', color='#a8c8e8', edgecolor='black')
    ax.bar(x + width, [r['covdet_ks'] for r in results], width,
            label='Cov-det', color='#83b860', edgecolor='black')
    for i, r in enumerate(results):
        for off, key in zip([-width, 0, width],
                              ['range_ks', 'conn_ks', 'covdet_ks']):
            ax.text(i + off, r[key] + 0.012, f'{r[key]:.3f}',
                     ha='center', fontsize=7.5)
    ax.axhline(0.30, color='#1a9850', linestyle='--', linewidth=1.4,
                 label='Axel pass bar (≤ 0.30)')
    ax.axhline(0.10, color='#0d4d2a', linestyle=':', linewidth=1.4,
                 label='Axel excellent bar (≤ 0.10)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('KS distance (truth vs ensemble per-sample)', fontsize=11)
    ax.set_title('Fine threshold sweep — does any intermediate p pass all three?',
                  fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'fine_threshold_sweep.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✓ figure → {out_dir / 'fine_threshold_sweep.png'}")

    csv_path = out_dir / 'fine_threshold_sweep_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'range_ks', 'conn_ks', 'covdet_ks',
                     'max_ks', 'avg_ks'])
        for r in results:
            w.writerow([r['threshold'], r['range_ks'], r['conn_ks'],
                         r['covdet_ks'], r['max_ks'], r['avg_ks']])
    print(f"  ✓ csv    → {csv_path}")


if __name__ == "__main__":
    main()