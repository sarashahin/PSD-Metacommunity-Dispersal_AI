#!/usr/bin/env python3
"""
=============================================================================
AXEL (a)-DISTRIBUTION TESTS — CANONICAL SCRIPT
=============================================================================

This is the SINGLE canonical script for Axel's three distribution-comparison
tests. It supersedes and replaces:

    axel_covariance_determinant_test.py    (cov-det at p≥0.5 only)
    axel_covdet_threshold_sweep.py         (cov-det at 6 thresholds)

Both predecessors are subsumed by this script. Delete them after migrating.

WHAT THIS SCRIPT DOES
---------------------
Computes Axel's three (a)-statistics:

   (1) RANGE-SIZE distribution        — KS vs truth
   (2) CONNECTIVITY distribution      — KS vs truth (number of patches)
   (3) COVARIANCE-DETERMINANT
       distribution (with PBC)        — KS vs truth (spatial spread)

At six truth-free thresholds:

   p≥0.5, p≥0.7, p≥0.9, topQ_K3, topQ_K5, otsu

Then identifies the SINGLE threshold that best satisfies all three of
Axel's (a)-statistics simultaneously — minimising the worst KS across
the three axes.

WHY ONE SCRIPT INSTEAD OF THREE
-------------------------------
At epoch 149, the cov-det optimum threshold shifted from p≥0.7 (KS=0.061
at epoch 109) to p≥0.9 (KS=0.047 at epoch 149). To honestly report
Axel's distributional test, all three statistics should be reported at
the SAME threshold, otherwise the reporting cherry-picks per-statistic.

This script finds that single calibrated threshold or, if no single
threshold passes all three, reports each statistic at its own optimum
and makes the trade-off explicit.

NO synthetic data. NO new sampling. Operates only on existing recon NPZs
and truth NPZs.

USAGE
-----
    # 1. Self-test the periodic_cov_det math (no data needed):
    python axel_three_axis_threshold_alignment.py --self-test

    # 2. Real run on epoch-149 recons:
    python axel_three_axis_threshold_alignment.py \\
        --wide-range-csv     ./figures_map_axel_stage2_new/wide_range_species.csv \\
        --recon-dir-pattern  './reconstructions_spatial/{world_stem}' \\
        --truth-dir          ./results/data \\
        --K                  5 \\
        --output-dir         ./figures_map_axel_stage2_new/axel_three_axis_alignment

OUTPUTS
-------
   three_axis_threshold_alignment.png   — 3 statistics × 6 thresholds
   covdet_threshold_sweep.png           — replicates the old sweep figure
   three_axis_alignment_summary.csv     — full numeric table
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


# ─── PBC-aware covariance determinant ─────────────────────────────────
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
    var_y  = float(np.var(dy))
    var_x  = float(np.var(dx))
    cov_yx = float(((dy - dy.mean()) * (dx - dx.mean())).mean())
    return max(0.0, var_y * var_x - cov_yx ** 2)


def count_components(binary_map):
    if binary_map.sum() == 0:
        return 0
    _, n = ndimage.label(binary_map, structure=CONNECTIVITY_STRUCTURE)
    return int(n)


# ─── Thresholding modes ───────────────────────────────────────────────
def threshold_fixed(prob, threshold):
    return (prob >= threshold).astype(np.uint8)


def threshold_topQ(prob, observed, multiplier):
    S, Y, X = prob.shape
    binary = np.zeros_like(prob, dtype=np.uint8)
    for s in range(S):
        K_obs = int(observed[s].sum())
        if K_obs == 0:
            continue
        Q = min(K_obs * multiplier, Y * X)
        flat = prob[s].ravel()
        if flat.max() < 1e-6:
            continue
        kth = np.partition(flat, -Q)[-Q]
        binary[s] = (prob[s] >= kth).astype(np.uint8)
    return binary


def threshold_otsu(prob):
    S = prob.shape[0]
    binary = np.zeros_like(prob, dtype=np.uint8)
    for s in range(S):
        vals = prob[s].ravel()
        if vals.max() == vals.min():
            continue
        hist, edges = np.histogram(vals, bins=256, range=(0, 1))
        total = hist.sum()
        if total == 0:
            continue
        p = hist / total
        omega = np.cumsum(p)
        mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
        mu_T = mu[-1]
        denom = omega * (1 - omega)
        denom[denom < 1e-12] = 1e-12
        sigma_b2 = (mu_T * omega - mu) ** 2 / denom
        idx = int(np.argmax(sigma_b2))
        thr = float((edges[idx] + edges[idx + 1]) / 2)
        binary[s] = (prob[s] >= thr).astype(np.uint8)
    return binary


THRESHOLD_MODES = [
    ('p≥0.5',   lambda P, O: threshold_fixed(P, 0.5)),
    ('p≥0.7',   lambda P, O: threshold_fixed(P, 0.7)),
    ('p≥0.9',   lambda P, O: threshold_fixed(P, 0.9)),
    ('topQ_K3', lambda P, O: threshold_topQ(P, O, 3)),
    ('topQ_K5', lambda P, O: threshold_topQ(P, O, 5)),
    ('otsu',    lambda P, O: threshold_otsu(P)),
]


# ─── Self-tests for the math (run with --self-test) ──────────────────
def _self_test():
    """
    Verifies the periodic_cov_det() and count_components() implementations
    against known analytic answers. Run with `--self-test`. No data needed.
    """
    print("\n  Self-tests for periodic_cov_det() and count_components()")
    print("  " + "-" * 70)

    # ── 1. tight 2x2 block, away from boundary ──
    # 4 cells at (5,5),(5,6),(6,5),(6,6). Centered coords ±0.5 → var=0.25,
    # cov=0 → det = 0.0625.
    r = np.zeros((GRID_Y, GRID_X), dtype=np.uint8)
    r[5:7, 5:7] = 1
    cd_tight = periodic_cov_det(r)
    print(f"    [1] 2x2 block        (analytic 0.0625)   cov-det = {cd_tight:.6f}")
    assert abs(cd_tight - 0.0625) < 1e-9, f"expected 0.0625, got {cd_tight}"
    # Also: one connected component
    assert count_components(r) == 1, "2x2 block should be 1 connected component"

    # ── 2. 3x3 block — must give STRICTLY LARGER cov-det than 2x2 ──
    # 9 cells, var_y = var_x = 6/9, cov_yx = 0 → det = 4/9 ≈ 0.4444
    r = np.zeros((GRID_Y, GRID_X), dtype=np.uint8)
    r[5:8, 5:8] = 1
    cd_block3 = periodic_cov_det(r)
    print(f"    [2] 3x3 block        (analytic 0.4444)   cov-det = {cd_block3:.6f}")
    assert abs(cd_block3 - 4.0 / 9.0) < 1e-9, f"expected ~0.444, got {cd_block3}"
    assert cd_block3 > cd_tight, "3x3 block should give larger cov-det than 2x2"
    assert count_components(r) == 1, "3x3 block is 1 component"

    # ── 3. 4-cell perfect diagonal — rank-1 covariance, det = 0 ──
    r = np.zeros((GRID_Y, GRID_X), dtype=np.uint8)
    for i in range(4):
        r[5 + i, 5 + i] = 1
    cd_diag = periodic_cov_det(r)
    print(f"    [3] 4-cell diagonal  (analytic 0.0000)   cov-det = {cd_diag:.6f}")
    assert cd_diag < 1e-9, (
        f"diagonal → rank-1 cov → det should be 0; got {cd_diag}")
    # 4-connectivity (no diagonals): each diagonal cell is its own component
    assert count_components(r) == 4, (
        "4-cell diagonal should be 4 components under 4-connectivity")

    # ── 4. THE PBC TEST AXEL WARNED ABOUT ──
    # Place a 2x2 block straddling the y=0 seam: y ∈ {19,0} × x ∈ {5,6}.
    # On the torus this is identical to a non-wrapping 2x2; PBC-aware
    # cov-det must equal 0.0625. Naive variance gives 22.5625 (361× too big).
    r = np.zeros((GRID_Y, GRID_X), dtype=np.uint8)
    r[0, 5] = 1; r[0, 6] = 1; r[19, 5] = 1; r[19, 6] = 1
    cd_pbc = periodic_cov_det(r)
    print(f"    [4] 2x2 wrapping y=0 seam  (analytic 0.0625)   "
          f"cov-det = {cd_pbc:.6f}")
    yy, xx = np.where(r > 0)
    naive_var_y = float(np.var(yy.astype(float)))
    naive_var_x = float(np.var(xx.astype(float)))
    naive_cov_yx = float(((yy - yy.mean()) * (xx - xx.mean())).mean())
    naive_det = naive_var_y * naive_var_x - naive_cov_yx ** 2
    print(f"        naive (no PBC)                     cov-det = "
          f"{naive_det:.4f}  → {naive_det / max(cd_pbc, 1e-9):.0f}× inflation")
    assert abs(cd_pbc - 0.0625) < 1e-9, f"PBC-aware should = 0.0625; got {cd_pbc}"
    assert abs(naive_det - 22.5625) < 1e-9, f"naive should = 22.5625; got {naive_det}"

    # ── 5. singleton range — degenerate ──
    r = np.zeros((GRID_Y, GRID_X), dtype=np.uint8)
    r[7, 11] = 1
    cd_one = periodic_cov_det(r)
    print(f"    [5] single cell      (analytic 0.0000)   cov-det = {cd_one:.6f}")
    assert cd_one == 0.0, f"1-cell range → cov-det = 0; got {cd_one}"
    assert count_components(r) == 1, "single cell is 1 component"

    # ── 6. two disconnected blocks — count_components = 2 ──
    r = np.zeros((GRID_Y, GRID_X), dtype=np.uint8)
    r[2:4, 2:4] = 1   # block A
    r[10:12, 14:16] = 1   # block B (well-separated)
    assert count_components(r) == 2, (
        "two well-separated blocks should be 2 components")
    print(f"    [6] two separate blocks                  ncomp   = "
          f"{count_components(r)}")

    print("\n  ✓ All self-tests passed. PBC math + connectivity counting verified.\n")


# ─── Per-world: all three KS values at each threshold ─────────────────
def compute_per_world(truth_path, samples_path, K):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)

    n_use = min(truth.shape[0], samples.shape[1])
    truth = truth[:n_use]
    samples = samples[:, :n_use]
    observed = observed[:n_use]
    n_ens = samples.shape[0]

    # meaningful species
    idx = [s for s in range(n_use) if int(truth[s].sum()) > K]
    if not idx:
        return None
    truth_m = truth[idx]
    samples_m = samples[:, idx]
    observed_m = observed[idx]

    # truth statistics (range, ncomp, cov-det) per meaningful species
    truth_range = truth_m.sum(axis=(1, 2)).astype(np.int32)
    truth_ncomp = np.asarray([count_components(t) for t in truth_m],
                              dtype=np.int32)
    truth_logcd = np.asarray([np.log10(periodic_cov_det(t) + 1.0)
                                for t in truth_m], dtype=np.float64)

    out = {
        'truth_range': truth_range,
        'truth_ncomp': truth_ncomp,
        'truth_logcd': truth_logcd,
    }

    # per-threshold ensemble pools (pooled over n_ens samples × species)
    for label, fn in THRESHOLD_MODES:
        pred_range = []
        pred_ncomp = []
        pred_logcd = []
        for k in range(n_ens):
            binary_k = fn(samples_m[k], observed_m)
            for s in range(len(idx)):
                if int(binary_k[s].sum()) >= 2:
                    pred_range.append(int(binary_k[s].sum()))
                    pred_ncomp.append(count_components(binary_k[s]))
                    pred_logcd.append(np.log10(periodic_cov_det(binary_k[s]) + 1.0))
        out[f'{label}_range'] = np.asarray(pred_range, dtype=np.int32)
        out[f'{label}_ncomp'] = np.asarray(pred_ncomp, dtype=np.int32)
        out[f'{label}_logcd'] = np.asarray(pred_logcd, dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true',
                    help='Run sanity tests for the math and exit. '
                         'No data flags needed in this mode.')
    ap.add_argument('--wide-range-csv',     default=None)
    ap.add_argument('--recon-dir-pattern',  default=None,
                    help="pattern with {world_stem} placeholder")
    ap.add_argument('--truth-dir',          default=None)
    ap.add_argument('--K',                  type=int, default=5)
    ap.add_argument('--top-n-worlds',       type=int, default=30)
    ap.add_argument('--output-dir',         default=None)
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    required = ['wide_range_csv', 'recon_dir_pattern',
                'truth_dir', 'output_dir']
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        ap.error(f"missing required args for real run: {missing}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_sp_count = defaultdict(int)
    with open(args.wide_range_csv) as f:
        for row in csv.DictReader(f):
            world_sp_count[row['world']] += 1
    top_worlds = sorted(world_sp_count.items(), key=lambda x: -x[1])[:args.top_n_worlds]
    print(f"\n  Computing 3-axis KS at 6 thresholds across {len(top_worlds)} worlds ...\n")

    pools = {
        'truth_range': [], 'truth_ncomp': [], 'truth_logcd': [],
    }
    for label, _ in THRESHOLD_MODES:
        pools[f'{label}_range'] = []
        pools[f'{label}_ncomp'] = []
        pools[f'{label}_logcd'] = []

    truth_dir = Path(args.truth_dir)
    for world_name, _ in top_worlds:
        stem = world_name.replace('.npz', '')
        truth_path = truth_dir / world_name
        samples_path = Path(args.recon_dir_pattern.format(world_stem=stem)) \
                          / f'recon_fixed_b{args.K}_samples.npz'
        if not (truth_path.exists() and samples_path.exists()):
            print(f"    skip {world_name[:55]}: missing")
            continue
        r = compute_per_world(truth_path, samples_path, args.K)
        if r is None:
            continue
        for k, v in r.items():
            pools[k].append(v)
        print(f"    {world_name[:60]:60s}  truth_n={len(r['truth_range']):4d}")

    if not pools['truth_range']:
        print("\n  No usable worlds.")
        return

    # Concatenate
    for k in list(pools.keys()):
        pools[k] = np.concatenate(pools[k]) if pools[k] else np.array([])

    # KS tests for each threshold × each axis
    results = []
    for label, _ in THRESHOLD_MODES:
        row = {'threshold': label}
        if len(pools[f'{label}_range']) > 0:
            row['range_ks']  = float(stats.ks_2samp(pools['truth_range'],
                                                      pools[f'{label}_range']).statistic)
            row['conn_ks']   = float(stats.ks_2samp(pools['truth_ncomp'],
                                                      pools[f'{label}_ncomp']).statistic)
            row['covdet_ks'] = float(stats.ks_2samp(pools['truth_logcd'],
                                                      pools[f'{label}_logcd']).statistic)
            row['max_ks']    = max(row['range_ks'], row['conn_ks'], row['covdet_ks'])
            row['avg_ks']    = (row['range_ks'] + row['conn_ks'] + row['covdet_ks']) / 3
        else:
            row.update({'range_ks': np.nan, 'conn_ks': np.nan, 'covdet_ks': np.nan,
                         'max_ks': np.nan, 'avg_ks': np.nan})
        results.append(row)

    # Print the verdict
    print("\n" + "=" * 78)
    print(f"  THREE-AXIS THRESHOLD ALIGNMENT")
    print("=" * 78)
    print(f"  truth_n = {len(pools['truth_range']):,} meaningful species")
    print(f"  n per threshold = {len(pools['p≥0.5_range']):,} pooled samples\n")

    print(f"  {'threshold':<10s} {'range_KS':>10s} {'conn_KS':>10s} "
          f"{'covdet_KS':>10s} {'max':>8s} {'avg':>8s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        verdict_max = ''
        if not np.isnan(r['max_ks']):
            if r['max_ks'] <= 0.30:
                verdict_max = '  ← all 3 PASS'
            if r['max_ks'] <= 0.10:
                verdict_max = '  ← all 3 EXCELLENT'
        print(f"  {r['threshold']:<10s} {r['range_ks']:>10.3f} {r['conn_ks']:>10.3f} "
              f"{r['covdet_ks']:>10.3f} {r['max_ks']:>8.3f} {r['avg_ks']:>8.3f}{verdict_max}")

    # Find the threshold that minimises max KS (most balanced)
    valid_results = [r for r in results if not np.isnan(r['max_ks'])]
    if valid_results:
        best_by_max = min(valid_results, key=lambda r: r['max_ks'])
        best_by_avg = min(valid_results, key=lambda r: r['avg_ks'])
        print(f"\n  Best balanced threshold (min worst KS):  {best_by_max['threshold']}"
              f"  (max KS = {best_by_max['max_ks']:.3f})")
        print(f"  Best average threshold (min mean KS):    {best_by_avg['threshold']}"
              f"  (avg KS = {best_by_avg['avg_ks']:.3f})")

        if best_by_max['max_ks'] <= 0.30:
            print(f"\n  →  At {best_by_max['threshold']}, all three Axel (a)-statistics PASS.")
            print(f"     range_KS = {best_by_max['range_ks']:.3f},  "
                  f"conn_KS = {best_by_max['conn_ks']:.3f},  "
                  f"covdet_KS = {best_by_max['covdet_ks']:.3f}")
            print(f"     Report this threshold to Axel as the single calibrated ensemble cut.")
        else:
            print(f"\n  →  No single threshold passes all three at the ≤0.30 bar.")
            print(f"     Need to report each statistic at its own optimal threshold.")

    # Figure
    fig, ax = plt.subplots(figsize=(12, 6.5))
    labels = [r['threshold'] for r in results]
    x = np.arange(len(labels))
    width = 0.27

    bars1 = ax.bar(x - width, [r['range_ks'] for r in results], width,
                    label='Range size', color='#5b8dd6', edgecolor='black')
    bars2 = ax.bar(x, [r['conn_ks'] for r in results], width,
                    label='Connectivity', color='#a8c8e8', edgecolor='black')
    bars3 = ax.bar(x + width, [r['covdet_ks'] for r in results], width,
                    label='Cov-det (spread)', color='#83b860', edgecolor='black')

    for bars in [bars1, bars2, bars3]:
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width()/2, h + 0.015,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0.30, color='#1a9850', linestyle='--', linewidth=1.4,
                label='Axel pass bar (≤ 0.30)')
    ax.axhline(0.10, color='#0d4d2a', linestyle=':', linewidth=1.4,
                label='Axel excellent bar (≤ 0.10)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('KS distance (truth vs ensemble per-sample)', fontsize=11)
    if valid_results:
        ax.set_title(f'Three-axis threshold alignment — best balanced: '
                      f'{best_by_max["threshold"]} (max KS = {best_by_max["max_ks"]:.3f})',
                      fontweight='bold', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'three_axis_threshold_alignment.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✓ figure → {out_dir / 'three_axis_threshold_alignment.png'}")

    # ── Second figure: cov-det-only sweep (replaces axel_covdet_threshold_sweep.py) ──
    fig, ax = plt.subplots(figsize=(11, 6))
    cd_vals = [r['covdet_ks'] for r in results]
    cd_labels = [r['threshold'] for r in results]
    colors = ['#1a9850' if v <= 0.30 else '#e6a23c' if v <= 0.50 else '#d73027'
              for v in cd_vals]
    # Highlight the BEST cov-det threshold
    if not all(np.isnan(cd_vals)):
        best_cd_idx = int(np.nanargmin(cd_vals))
        edge_colors = ['black'] * len(cd_vals)
        edge_widths = [0.8] * len(cd_vals)
        edge_colors[best_cd_idx] = '#1a5f1a'
        edge_widths[best_cd_idx] = 2.5
        best_cd_label = cd_labels[best_cd_idx]
        best_cd_value = cd_vals[best_cd_idx]
    else:
        edge_colors = ['black'] * len(cd_vals)
        edge_widths = [0.8] * len(cd_vals)
        best_cd_label, best_cd_value = 'n/a', float('nan')

    bars = ax.bar(range(len(cd_labels)), cd_vals,
                  color=colors, edgecolor=edge_colors, linewidth=edge_widths)
    for b, v, lbl in zip(bars, cd_vals, cd_labels):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold' if lbl == best_cd_label else 'normal')

    ax.axhline(0.30, color='#1a9850', linestyle='--', linewidth=1.4,
               label='Axel pass bar (≤ 0.30)')
    ax.axhline(0.10, color='#0d4d2a', linestyle=':', linewidth=1.4,
               label='Axel excellent bar (≤ 0.10)')
    ax.set_xticks(range(len(cd_labels)))
    ax.set_xticklabels(cd_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Cov-det KS distance (truth vs prediction)', fontsize=11)
    ax.set_title(
        f'Cov-det spatial-spread KS by threshold — best = {best_cd_label} '
        f'(KS = {best_cd_value:.3f})',
        fontweight='bold', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'covdet_threshold_sweep.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ figure → {out_dir / 'covdet_threshold_sweep.png'}")

    # CSV
    csv_path = out_dir / 'three_axis_alignment_summary.csv'
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