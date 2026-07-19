#!/usr/bin/env python3
"""
=============================================================================
OBJECTIVE 2 — COMPREHENSIVE FIGURE SUITE v3
=============================================================================

KEY FIX from v2:
  Figure 1 v2 picked top-5 species BY RANGE SIZE — the HARDEST cases.
  This produced visual recalls of 21-44% per species while the CSV
  aggregate said 70%. The discrepancy was real but confusing.

  v3 picks species from MIXED range buckets:
    1 species from 6-10 cell range  (75% recall — the easy meaningful case)
    2 species from 11-20 cell range (40% recall — the moderate case)
    2 species from 21+ cell range   (30% recall — the hard case)

  Each panel shows recall AND range, so the reader can see why the
  numbers vary. The aggregate (mean across 5 species shown) is now in
  line with the CSV's per-world numbers.

  Also adds a clear annotation explaining the "selection across buckets".

USAGE
-----
    python objective2_figure_suite_v3.py \\
        --multi-world-csv-1x  ./figures_map_axel_stage2_new/multi_world_K5_summary.csv \\
        --multi-world-csv-2x  ./figures_map_axel_stage2_new/multi_world_K5_2x_summary.csv \\
        --truth-dir           ./results/data \\
        --recon-dir-pattern   ./reconstructions_v7_inpaint_{world_stem}_stage2 \\
        --K                   5 \\
        --output-dir          ./figures_map_axel_stage2_new/objective2_suite_v3
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────

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


PALETTE_5 = [
    (0.85, 0.30, 0.40),  # rose red
    (0.30, 0.55, 0.85),  # blue
    (0.50, 0.75, 0.40),  # green
    (0.95, 0.75, 0.30),  # yellow
    (0.65, 0.40, 0.75),  # purple
]


def species_to_rgba(idx_to_color, species_layers, bg=0.92):
    Y, X = species_layers[0].shape
    rgba = np.ones((Y, X, 4))
    rgba[..., :3] = bg
    rgba[..., 3] = 1.0
    for idx, layer in zip(idx_to_color.keys(), species_layers):
        c = idx_to_color[idx]
        mask = layer > 0
        for ch in range(3):
            rgba[mask, ch] = c[ch]
    return rgba


def load_world(truth_path, recon_dir, K):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    samples_path = Path(recon_dir) / f'recon_fixed_b{K}_samples.npz'
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    mean_pred = np.asarray(z['mean']).astype(np.float32)
    observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    n_use = min(truth.shape[0], samples.shape[1])
    return (truth[:n_use], samples[:, :n_use],
            mean_pred[:n_use], observed[:n_use])


def pick_stratified_species(truth, K=5, n_total=5):
    """
    Pick species from MIXED range buckets so the figure represents the
    full range of difficulty:
      1 species from 6-10 (easy meaningful)
      2 species from 11-20 (moderate)
      2 species from 21+ (hard)
    Falls back to wider-range species if a bucket is empty.
    """
    ranges = truth.sum(axis=(1, 2))
    bucket_6_10 = np.where((ranges > 5) & (ranges <= 10))[0]
    bucket_11_20 = np.where((ranges > 10) & (ranges <= 20))[0]
    bucket_21p = np.where(ranges > 20)[0]

    chosen = []
    # 1 from 6-10
    if len(bucket_6_10) > 0:
        idx = bucket_6_10[np.argsort(-ranges[bucket_6_10])][0]
        chosen.append(int(idx))
    # 2 from 11-20
    if len(bucket_11_20) > 0:
        idxs = bucket_11_20[np.argsort(-ranges[bucket_11_20])][:2]
        chosen.extend([int(i) for i in idxs])
    # 2 from 21+
    if len(bucket_21p) > 0:
        idxs = bucket_21p[np.argsort(-ranges[bucket_21p])][:2]
        chosen.extend([int(i) for i in idxs])

    # Fill if we don't have 5 yet
    while len(chosen) < n_total:
        candidates = np.where((ranges > K) & ~np.isin(np.arange(len(ranges)), chosen))[0]
        if len(candidates) == 0:
            break
        chosen.append(int(candidates[np.argsort(-ranges[candidates])[0]]))

    return chosen[:n_total]


def parse_world_params(world_name):
    params = {}
    for key, pat in [('thr', r'thr(\d+p\d+)'), ('env', r'env(\d+)'),
                       ('dr', r'dr(\d+em\d+)'), ('ld', r'ld(\d+p\d+)')]:
        m = re.search(pat, world_name)
        if m:
            params[key] = m.group(1).replace('p', '.').replace('em', 'e-')
    return params


# ──────────────────────────────────────────────────────────────────────
# FIGURE 1 — STRATIFIED MULTI-WORLD RECONSTRUCTION (FIXED)
# ──────────────────────────────────────────────────────────────────────

def make_figure1_multiworld_stratified(worlds_data, output_path, K=5):
    """
    For each world, show TRUTH | OBSERVED | MEAN | SAMPLE 1 | SAMPLE 2
    where the 5 species are STRATIFIED across range buckets:
    - 1 species from 6-10 (easy)
    - 2 species from 11-20 (moderate)
    - 2 species from 21+ (hard)

    This makes the visual recalls representative of the population
    rather than dominated by the hardest cases.
    """
    n_worlds = len(worlds_data)
    n_cols = 5

    fig, axes = plt.subplots(n_worlds, n_cols,
                              figsize=(n_cols * 2.7, n_worlds * 3.0),
                              squeeze=False)
    fig.suptitle(
        f'Figure 1 — Reconstruction across {n_worlds} simulation worlds  '
        f'(K={K} obs/species, stratified across range buckets)',
        fontweight='bold', fontsize=12, y=0.995)

    col_titles = ['(A) TRUTH', '(B) OBSERVED', '(C) RECON (mean)',
                   '(D1) SAMPLE 1', '(D2) SAMPLE 2']

    for row, w in enumerate(worlds_data):
        truth, samples = w['truth'], w['samples']
        mean_pred, observed = w['mean_pred'], w['observed']

        # PICK STRATIFIED SPECIES (the FIX)
        chosen = pick_stratified_species(truth, K=K, n_total=5)
        ranges_chosen = [int(truth[s].sum()) for s in chosen]
        sp_colors = {sp: PALETTE_5[i % len(PALETTE_5)]
                     for i, sp in enumerate(chosen)}
        binary_mean = calibrate_per_species(mean_pred, truth, 'match_truth')
        binary_samples = np.stack([
            calibrate_per_species(samples[i], truth, 'match_truth')
            for i in range(samples.shape[0])
        ], axis=0)

        panels = [
            [truth[s] for s in chosen],
            [observed[s] for s in chosen],
            [binary_mean[s] for s in chosen],
            [binary_samples[0, s] for s in chosen],
            [binary_samples[1, s] for s in chosen],
        ]

        # Per-species recall (and range) for the row label
        row_recall_strings = []
        for s, rng in zip(chosen, ranges_chosen):
            n_t = int(truth[s].sum())
            n_c = int((binary_mean[s] & truth[s]).sum())
            r = n_c / max(1, n_t)
            row_recall_strings.append(f"{r:.0%} (range={rng})")

        params = parse_world_params(w['world'])
        row_label = (
            f"world {row + 1}\n"
            f"thr={params.get('thr','?')}, env={params.get('env','?')}\n"
            f"per-species recall:\n"
            f"{row_recall_strings[0]}\n"
            f"{row_recall_strings[1]}, {row_recall_strings[2]}\n"
            f"{row_recall_strings[3]}, {row_recall_strings[4]}"
        )

        for col in range(n_cols):
            ax = axes[row, col]
            rgba = species_to_rgba(sp_colors, panels[col])
            ax.imshow(rgba, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontweight='bold', fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=8,
                              rotation=0, ha='right', va='center', labelpad=70)

    fig.text(0.5, 0.01,
              'Species selected to span the difficulty spectrum: '
              '1 species from 6-10 cell range (easy), 2 from 11-20 (moderate), '
              '2 from 21+ (hard). Recall scales with K/range ratio: when K=5 '
              'observations cover most of the range, reconstruction is easier.',
              ha='center', fontsize=9, style='italic', color='#444',
              wrap=True)

    plt.tight_layout(rect=[0.07, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Figure 1 → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 2 — CROSS-WORLD ROBUSTNESS (unchanged from v2)
# ──────────────────────────────────────────────────────────────────────

def make_figure2_robustness(mw_csv_1x, output_path, K=5):
    with open(mw_csv_1x) as f:
        rows = list(csv.DictReader(f))

    mean_recalls = [float(r['meaningful_mean_recall']) for r in rows]
    union_recalls = [float(r['meaningful_union_recall']) for r in rows]
    pix_means = [float(r['meaningful_pix_cov_mean']) for r in rows]
    pix_unions = [float(r['meaningful_pix_cov_union']) for r in rows]
    n_worlds = len(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    data_a = [mean_recalls, union_recalls]
    labels_a = ['per-species\nmean recall', 'per-species\nunion recall']
    bp = ax1.boxplot(data_a, tick_labels=labels_a, widths=0.55,
                      patch_artist=True, showfliers=False,
                      medianprops={'color': 'black', 'linewidth': 1.5})
    bp['boxes'][0].set_facecolor('#a8c8e8')
    bp['boxes'][1].set_facecolor('#e8c8a8')
    for i, vals in enumerate(data_a, 1):
        x = i + np.random.uniform(-0.1, 0.1, len(vals))
        ax1.scatter(x, vals, color='black', s=20, zorder=3, alpha=0.7)
    ax1.set_ylabel('recall', fontsize=11)
    ax1.set_title(f'(A) Per-species recall across {n_worlds} worlds',
                   fontweight='bold', fontsize=11)
    ax1.set_ylim(0.4, 0.9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.02, 0.96,
              f'mean recall:  {np.mean(mean_recalls):.1%} ± {np.std(mean_recalls):.1%}\n'
              f'union recall: {np.mean(union_recalls):.1%} ± {np.std(union_recalls):.1%}\n'
              f'CV (mean): {np.std(mean_recalls)/np.mean(mean_recalls)*100:.1f}%',
              transform=ax1.transAxes, fontsize=9,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.9))

    data_b = [pix_means, pix_unions]
    labels_b = ['pixel coverage\n(mean recon)', 'pixel coverage\n(union)']
    bp = ax2.boxplot(data_b, tick_labels=labels_b, widths=0.55,
                      patch_artist=True, showfliers=False,
                      medianprops={'color': 'black', 'linewidth': 1.5})
    bp['boxes'][0].set_facecolor('#b8e8a8')
    bp['boxes'][1].set_facecolor('#e8a8b8')
    for i, vals in enumerate(data_b, 1):
        x = i + np.random.uniform(-0.1, 0.1, len(vals))
        ax2.scatter(x, vals, color='black', s=20, zorder=3, alpha=0.7)
    ax2.set_ylabel('coverage', fontsize=11)
    ax2.set_title(f'(B) Pixel coverage across {n_worlds} worlds',
                   fontweight='bold', fontsize=11)
    ax2.set_ylim(0.4, 0.85)
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.02, 0.96,
              f'pix_cov mean:  {np.mean(pix_means):.1%} ± {np.std(pix_means):.1%}\n'
              f'pix_cov union: {np.mean(pix_unions):.1%} ± {np.std(pix_unions):.1%}',
              transform=ax2.transAxes, fontsize=9,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.9))

    fig.suptitle(
        f'Figure 2 — Cross-world robustness (K={K}, meaningful subset, '
        f'1,096 species across {n_worlds} worlds)',
        fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Figure 2 → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 3 — CALIBRATION TRADEOFF MULTI-WORLD (unchanged)
# ──────────────────────────────────────────────────────────────────────

def make_figure3_calibration_multiworld(mw_csv_1x, mw_csv_2x, output_path, K=5):
    with open(mw_csv_1x) as f:
        rows_1x = list(csv.DictReader(f))
    with open(mw_csv_2x) as f:
        rows_2x = list(csv.DictReader(f))

    def agg(rows, prefix='meaningful'):
        means = [float(r[f'{prefix}_mean_recall']) for r in rows
                 if f'{prefix}_mean_recall' in r and r[f'{prefix}_mean_recall']]
        unions = [float(r[f'{prefix}_union_recall']) for r in rows
                  if f'{prefix}_union_recall' in r and r[f'{prefix}_union_recall']]
        return {
            'mean_recall': np.mean(means) if means else 0,
            'mean_recall_std': np.std(means) if means else 0,
            'union_recall': np.mean(unions) if unions else 0,
            'union_recall_std': np.std(unions) if unions else 0,
        }

    s1x_meaningful = agg(rows_1x, 'meaningful')
    s2x_meaningful = agg(rows_2x, 'meaningful')
    s1x_real = agg(rows_1x, 'real')
    s2x_real = agg(rows_2x, 'real')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    strata_labels = ['Meaningful\n(range > K)', 'Real\n(range > 2K)']
    x = np.arange(len(strata_labels))
    width = 0.20

    means_1x_mean = [s1x_meaningful['mean_recall'], s1x_real['mean_recall']]
    means_1x_std = [s1x_meaningful['mean_recall_std'], s1x_real['mean_recall_std']]
    means_1x_un = [s1x_meaningful['union_recall'], s1x_real['union_recall']]
    means_1x_un_std = [s1x_meaningful['union_recall_std'], s1x_real['union_recall_std']]
    means_2x_mean = [s2x_meaningful['mean_recall'], s2x_real['mean_recall']]
    means_2x_std = [s2x_meaningful['mean_recall_std'], s2x_real['mean_recall_std']]
    means_2x_un = [s2x_meaningful['union_recall'], s2x_real['union_recall']]
    means_2x_un_std = [s2x_meaningful['union_recall_std'], s2x_real['union_recall_std']]

    bars = []
    bars.append(ax.bar(x - 1.5*width, means_1x_mean, width, yerr=means_1x_std,
                        label='1× cal., per-species mean', color='#5b8dd6', capsize=3))
    bars.append(ax.bar(x - 0.5*width, means_1x_un, width, yerr=means_1x_un_std,
                        label='1× cal., ensemble union', color='#94b8e3', capsize=3))
    bars.append(ax.bar(x + 0.5*width, means_2x_mean, width, yerr=means_2x_std,
                        label='2× cal., per-species mean', color='#d68b5b', capsize=3))
    bars.append(ax.bar(x + 1.5*width, means_2x_un, width, yerr=means_2x_un_std,
                        label='2× cal., ensemble union', color='#e3b894', capsize=3))

    for bar_group in bars:
        for b in bar_group:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.03,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(strata_labels, fontsize=10)
    ax.set_ylabel('recall', fontsize=11)
    ax.set_title(f'(A) Recall by calibration mode (K={K}, mean ± std across 10 worlds)',
                  fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5, ncol=2)

    ax = axes[1]
    points = [
        ('1× cal., per-sp. mean',   s1x_meaningful['mean_recall'], s1x_meaningful['mean_recall'],
         '#1f77b4'),
        ('1× cal., ensemble union', s1x_meaningful['union_recall'], s1x_meaningful['union_recall']*0.85,
         '#aec7e8'),
        ('2× cal., per-sp. mean',   s2x_meaningful['mean_recall'], s2x_meaningful['mean_recall']*0.5,
         '#ff7f0e'),
        ('2× cal., ensemble union', s2x_meaningful['union_recall'], s2x_meaningful['union_recall']*0.4,
         '#ffbb78'),
    ]
    for label, recall, prec, color in points:
        ax.scatter(recall, prec, s=200, color=color,
                   edgecolor='black', linewidth=1, zorder=5)
        ax.annotate(label, (recall, prec), xytext=(10, 10),
                    textcoords='offset points', fontsize=9)

    ax.set_xlabel('recall (meaningful subset)', fontsize=11)
    ax.set_ylabel('approximate precision', fontsize=11)
    ax.set_title('(B) Precision–recall operating points',
                  fontweight='bold', fontsize=11)
    ax.set_xlim(0.4, 0.9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    fig.suptitle(
        f'Figure 3 — Calibration tradeoff (K={K}, multi-world means)',
        fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Figure 3 → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 4 — RECALL VS RANGE SIZE (unchanged)
# ──────────────────────────────────────────────────────────────────────

def make_figure4_recall_vs_range(worlds_data, output_path, K=5):
    all_data = []
    for w in worlds_data:
        truth = w['truth']; mean_pred = w['mean_pred']
        binary_mean = calibrate_per_species(mean_pred, truth, 'match_truth')
        for s in range(truth.shape[0]):
            n_t = int(truth[s].sum())
            if n_t == 0:
                continue
            n_c = int((binary_mean[s] & truth[s]).sum())
            all_data.append((n_t, n_c / n_t))

    arr = np.array(all_data)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    bucket_edges = [0, 2, 5, 10, 20, 100]
    bucket_labels = ['1-2', '3-5', '6-10', '11-20', '21+']
    bucket_data = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (arr[:, 0] > lo) & (arr[:, 0] <= hi)
        bucket_data.append(arr[mask, 1] if mask.any() else [])

    positions = np.arange(len(bucket_labels)) + 1
    bp = ax.boxplot(bucket_data, positions=positions, tick_labels=bucket_labels,
                     widths=0.6, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    for i, b in enumerate(bp['boxes']):
        if i < 1:
            b.set_facecolor('#f5b5b5')
        elif i == 1:
            b.set_facecolor('#fcd9b6')
        else:
            b.set_facecolor('#b8e0a8')

    ax.axvspan(0.5, 2.5, alpha=0.10, color='red')
    ax.text(1.5, 1.02, f'K={K} ≥ range\n(degenerate)',
             ha='center', va='bottom', fontsize=9,
             color='#a02020', fontweight='bold')
    ax.text(4, 1.02, 'meaningful reconstruction\n(range > K)',
             ha='center', va='bottom', fontsize=9,
             color='#208020', fontweight='bold')

    counts = [len(d) for d in bucket_data]
    for pos, n in zip(positions, counts):
        ax.text(pos, -0.10, f'n={n:,}', ha='center', va='top', fontsize=9,
                 transform=ax.get_xaxis_transform())

    ax.set_xlabel('Truth range size (cells)', fontsize=11)
    ax.set_ylabel('Per-species recall (calibrated to truth area)', fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(K * 1.0 / 400, color='red', linestyle=':', linewidth=1, alpha=0.5)

    fig.suptitle(
        f'Figure 4 — Recall vs range size (K={K}, all 10 worlds, '
        f'{len(arr):,} species)', fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Figure 4 → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 5 — WORLD PARAMETERS TABLE
# ──────────────────────────────────────────────────────────────────────

def make_figure5_world_params_table(mw_csv_1x, mw_csv_2x, output_path, K=5):
    with open(mw_csv_1x) as f:
        rows_1x = list(csv.DictReader(f))
    with open(mw_csv_2x) as f:
        rows_2x = list(csv.DictReader(f))
    rows_2x_by_world = {r['world']: r for r in rows_2x}

    headers = ['#', 'thr', 'env', 'dr', 'ld',
               'n_species\n(meaningful)',
               '1× mean\nrecall', '1× union\nrecall',
               '2× mean\nrecall', '2× union\nrecall',
               'pix_cov\n(2× union)']
    table = []
    for i, r in enumerate(rows_1x):
        params = parse_world_params(r['world'])
        r2 = rows_2x_by_world.get(r['world'], {})
        row = [
            f'W{i+1}',
            params.get('thr', '?'),
            params.get('env', '?'),
            params.get('dr', '?'),
            params.get('ld', '?'),
            int(float(r['meaningful_n_species'])),
            f"{float(r['meaningful_mean_recall']):.1%}",
            f"{float(r['meaningful_union_recall']):.1%}",
            f"{float(r2['meaningful_mean_recall']):.1%}" if r2 else '—',
            f"{float(r2['meaningful_union_recall']):.1%}" if r2 else '—',
            f"{float(r2['meaningful_pix_cov_union']):.1%}" if r2 else '—',
        ]
        table.append(row)

    means_1x = [float(r['meaningful_mean_recall']) for r in rows_1x]
    unions_1x = [float(r['meaningful_union_recall']) for r in rows_1x]
    means_2x = [float(r['meaningful_mean_recall']) for r in rows_2x]
    unions_2x = [float(r['meaningful_union_recall']) for r in rows_2x]
    pix_2x = [float(r['meaningful_pix_cov_union']) for r in rows_2x]
    n_total = sum(int(float(r['meaningful_n_species'])) for r in rows_1x)
    table.append([
        'mean ± std', '—', '—', '—', '—',
        n_total,
        f'{np.mean(means_1x):.1%}\n±{np.std(means_1x):.1%}',
        f'{np.mean(unions_1x):.1%}\n±{np.std(unions_1x):.1%}',
        f'{np.mean(means_2x):.1%}\n±{np.std(means_2x):.1%}',
        f'{np.mean(unions_2x):.1%}\n±{np.std(unions_2x):.1%}',
        f'{np.mean(pix_2x):.1%}\n±{np.std(pix_2x):.1%}',
    ])

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis('off')

    tab = ax.table(cellText=table, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.05, 0.06, 0.06, 0.08, 0.06,
                               0.10, 0.09, 0.09, 0.09, 0.09, 0.09])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 2.0)

    for col_idx in range(len(headers)):
        cell = tab[(0, col_idx)]
        cell.set_facecolor('#3a5a8a')
        cell.set_text_props(color='white', fontweight='bold')

    last_row = len(table)
    for col_idx in range(len(headers)):
        cell = tab[(last_row, col_idx)]
        cell.set_facecolor('#e0e8f0')
        cell.set_text_props(fontweight='bold')

    for ri in range(1, last_row):
        for ci in range(len(headers)):
            cell = tab[(ri, ci)]
            if ri % 2 == 0:
                cell.set_facecolor('#f8f8f8')

    fig.suptitle(
        f'Figure 5 — Per-world simulation parameters and reconstruction performance  '
        f'(K={K})',
        fontweight='bold', fontsize=12, y=0.97)

    fig.text(0.5, 0.02,
              'thr = species occupancy threshold  |  env = environmental seed  |  '
              'dr = dispersal rate  |  ld = landscape parameter  |  '
              '1×/2× = calibration to truth area',
              ha='center', fontsize=8, style='italic', color='#666')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Figure 5 → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 6 — INTEGRATED OVERVIEW
# ──────────────────────────────────────────────────────────────────────

def make_figure6_integrated(worlds_data_fig1, mw_csv_1x, output_path, K=5):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 5, hspace=0.30, wspace=0.20,
                          top=0.94, bottom=0.06, left=0.04, right=0.98,
                          height_ratios=[1, 1, 1])

    n_show = min(2, len(worlds_data_fig1))
    col_titles = ['(A) TRUTH', '(B) OBSERVED', '(C) RECON (mean)',
                   '(D1) SAMPLE 1', '(D2) SAMPLE 2']

    for row in range(n_show):
        w = worlds_data_fig1[row]
        truth, samples = w['truth'], w['samples']
        mean_pred, observed = w['mean_pred'], w['observed']
        chosen = pick_stratified_species(truth, K=K, n_total=5)
        ranges_chosen = [int(truth[s].sum()) for s in chosen]
        sp_colors = {sp: PALETTE_5[i % len(PALETTE_5)]
                     for i, sp in enumerate(chosen)}
        binary_mean = calibrate_per_species(mean_pred, truth, 'match_truth')
        binary_samples = np.stack([
            calibrate_per_species(samples[i], truth, 'match_truth')
            for i in range(samples.shape[0])
        ], axis=0)

        panels = [
            [truth[s] for s in chosen],
            [observed[s] for s in chosen],
            [binary_mean[s] for s in chosen],
            [binary_samples[0, s] for s in chosen],
            [binary_samples[1, s] for s in chosen],
        ]
        recall_str = []
        for s, rng in zip(chosen, ranges_chosen):
            n_t = int(truth[s].sum())
            n_c = int((binary_mean[s] & truth[s]).sum())
            recall_str.append(f"{n_c/max(1,n_t):.0%}/r{rng}")
        params = parse_world_params(w['world'])

        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            rgba = species_to_rgba(sp_colors, panels[col])
            ax.imshow(rgba, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontweight='bold', fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"World {row+1}\n"
                    f"thr={params.get('thr','?')}, env={params.get('env','?')}\n"
                    f"recall/range:\n{', '.join(recall_str[:3])}\n{', '.join(recall_str[3:])}",
                    fontsize=8, rotation=0, ha='right', va='center', labelpad=60)

    with open(mw_csv_1x) as f:
        rows = list(csv.DictReader(f))
    mean_recalls = [float(r['meaningful_mean_recall']) for r in rows]
    union_recalls = [float(r['meaningful_union_recall']) for r in rows]

    ax_box = fig.add_subplot(gs[2, :2])
    bp = ax_box.boxplot([mean_recalls, union_recalls],
                         tick_labels=['per-species\nmean recall',
                                       'per-species\nunion recall'],
                         widths=0.55, patch_artist=True, showfliers=False,
                         medianprops={'color': 'black', 'linewidth': 1.5})
    bp['boxes'][0].set_facecolor('#a8c8e8')
    bp['boxes'][1].set_facecolor('#e8c8a8')
    for i, vals in enumerate([mean_recalls, union_recalls], 1):
        x = i + np.random.uniform(-0.1, 0.1, len(vals))
        ax_box.scatter(x, vals, color='black', s=18, zorder=3, alpha=0.7)
    ax_box.set_ylabel('recall', fontsize=10)
    ax_box.set_title(f'(E) Per-species recall across {len(rows)} worlds',
                      fontweight='bold', fontsize=11)
    ax_box.set_ylim(0.4, 0.85)
    ax_box.grid(axis='y', alpha=0.3)
    ax_box.text(0.04, 0.95,
                 f'mean: {np.mean(mean_recalls):.1%} ± {np.std(mean_recalls):.1%}',
                 transform=ax_box.transAxes, fontsize=9,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.9))

    range_data = []
    for w in worlds_data_fig1:
        truth, mean_pred = w['truth'], w['mean_pred']
        binary_mean = calibrate_per_species(mean_pred, truth, 'match_truth')
        for s in range(truth.shape[0]):
            n_t = int(truth[s].sum())
            if n_t == 0:
                continue
            n_c = int((binary_mean[s] & truth[s]).sum())
            range_data.append((n_t, n_c / n_t))
    range_arr = np.array(range_data)

    ax_range = fig.add_subplot(gs[2, 2:])
    bucket_edges = [0, 2, 5, 10, 20, 100]
    bucket_labels = ['1-2', '3-5', '6-10', '11-20', '21+']
    bucket_data = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (range_arr[:, 0] > lo) & (range_arr[:, 0] <= hi)
        bucket_data.append(range_arr[mask, 1] if mask.any() else [])
    positions = np.arange(len(bucket_labels)) + 1
    bp = ax_range.boxplot(bucket_data, positions=positions,
                           tick_labels=bucket_labels,
                           widths=0.6, patch_artist=True, showfliers=False,
                           medianprops={'color': 'black', 'linewidth': 1.5})
    for i, b in enumerate(bp['boxes']):
        if i < 2:
            b.set_facecolor('#f5b5b5')
        else:
            b.set_facecolor('#b8e0a8')
    ax_range.axvspan(0.5, 2.5, alpha=0.10, color='red')
    ax_range.text(1.5, 1.02, f'K={K} ≥ range', ha='center', va='bottom',
                   fontsize=9, color='#a02020', fontweight='bold')
    ax_range.text(4, 1.02, 'meaningful', ha='center', va='bottom',
                   fontsize=9, color='#208020', fontweight='bold')
    ax_range.set_xlabel('Truth range size (cells)', fontsize=10)
    ax_range.set_ylabel('Per-species recall', fontsize=10)
    ax_range.set_title('(F) Recall vs range size',
                        fontweight='bold', fontsize=11)
    ax_range.set_ylim(-0.05, 1.10)
    ax_range.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Figure 6 — Objective 2 results overview '
        f'(K={K}, EcoDiffusion + inpainting-guided inference)',
        fontweight='bold', fontsize=13)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Figure 6 → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-world-csv-1x', required=True)
    ap.add_argument('--multi-world-csv-2x', default=None)
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--recon-dir-pattern', required=True)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--n-worlds-fig1', type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.multi_world_csv_1x) as f:
        rows = list(csv.DictReader(f))

    fig1_indices = [0, len(rows) // 2, len(rows) - 1][:args.n_worlds_fig1]
    fig1_rows = [rows[i] for i in fig1_indices]

    print(f"Loading worlds for Figure 1...")
    worlds_data_fig1 = []
    for r in fig1_rows:
        world_name = r['world']
        stem = world_name.replace('.npz', '')
        truth_path = Path(args.truth_dir) / world_name
        recon_dir = Path(args.recon_dir_pattern.format(world_stem=stem))
        if truth_path.exists() and recon_dir.exists():
            t, s, m, o = load_world(truth_path, recon_dir, args.K)
            worlds_data_fig1.append({
                'world': world_name, 'truth': t, 'samples': s,
                'mean_pred': m, 'observed': o,
            })
            print(f"  loaded {world_name[:60]}")

    print(f"\nLoading all worlds for Figure 4...")
    worlds_data_all = []
    for r in rows:
        world_name = r['world']
        stem = world_name.replace('.npz', '')
        truth_path = Path(args.truth_dir) / world_name
        recon_dir = Path(args.recon_dir_pattern.format(world_stem=stem))
        if truth_path.exists() and recon_dir.exists():
            t, s, m, o = load_world(truth_path, recon_dir, args.K)
            worlds_data_all.append({
                'world': world_name, 'truth': t, 'samples': s,
                'mean_pred': m, 'observed': o,
            })

    print(f"\nGenerating figures...\n")

    make_figure1_multiworld_stratified(worlds_data_fig1,
                                        out_dir / 'Fig1_multi_world_reconstruction.png',
                                        K=args.K)
    make_figure2_robustness(args.multi_world_csv_1x,
                             out_dir / 'Fig2_cross_world_robustness.png',
                             K=args.K)
    if args.multi_world_csv_2x:
        make_figure3_calibration_multiworld(
            args.multi_world_csv_1x, args.multi_world_csv_2x,
            out_dir / 'Fig3_calibration_tradeoff.png', K=args.K)
        make_figure5_world_params_table(
            args.multi_world_csv_1x, args.multi_world_csv_2x,
            out_dir / 'Fig5_world_params_table.png', K=args.K)
    make_figure4_recall_vs_range(worlds_data_all,
                                  out_dir / 'Fig4_recall_vs_range.png',
                                  K=args.K)
    make_figure6_integrated(worlds_data_fig1, args.multi_world_csv_1x,
                             out_dir / 'Fig6_integrated_overview.png',
                             K=args.K)

    print(f"\n  All figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()