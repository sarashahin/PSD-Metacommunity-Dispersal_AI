#!/usr/bin/env python3
"""
=============================================================================
ABLATION_VISUALIZE_ALL.PY  —  unified figure-generation script
=============================================================================

WHAT THIS REPLACES
------------------
Three previous scripts, now consolidated into one for organisational sanity:
    visualize_ablation_results.py   →  panels A1, A2, A3, A4 below
    ablation_per_variant_maps.py    →  panel B
    ablation_summary_table.py       →  panel C
    ablation_ecology_figures.py     →  panels D1, D2, D3

Run once. Get every figure for the manuscript.

WHAT IT NEEDS
-------------
1. The ablation NPZs produced by run_ablation_v7.py (one set per world).
2. The metrics CSV produced by validate_ablation_v2.py (1× and/or 2×).
3. The truth NPZ(s) — IBM simulation worlds — for per-species heatmaps and
   community metrics.

SUPPORTS BOTH MODES (auto-detected)
-----------------------------------
Single-world  (one ablation directory + one truth NPZ + one metrics CSV)
Multi-world   (--ablation-dir-pattern + --wide-range-csv + --truth-dir +
                --top-n-worlds)

ALL OUTPUTS GO TO ONE DIRECTORY
-------------------------------
You specify --output-dir; this script writes:

    fig_A1_summary_bars.png           ← 4-panel headline metrics
    fig_A2_stratified.png             ← strata × variants comparison
    fig_A3_inpainting_diagnostic.png  ← echo-at-obs + fill-in sanity
    fig_A4_per_world_heatmap.png      ← multi-world only: world × variant
    fig_B_per_variant_maps.png        ← per-variant per-species heatmaps
    fig_C_summary_table.png           ← publication-ready table + ranking
    fig_D1_per_species_delta.png      ← FULL recall vs ablation recall scatter
    fig_D2_recall_vs_range.png        ← recall curves overlaid by variant
    fig_D3_richness_betadiv.png       ← community-level metrics
    fig_E_axel_three_map_demo.png     ← Axel's exact 6-panel demonstration:
                                         TRUTH | OBSERVED | RECON_MEAN |
                                         SAMPLE_1 | SAMPLE_2 | SAMPLE_3

USAGE — SINGLE-WORLD
--------------------
    python ablation_visualize_all.py \\
        --metrics-csv     ./ablation_v7_world5_stage2_inpaint/ablation_metrics_1x_v2.csv \\
        --ablation-dir    ./ablation_v7_world5_stage2_inpaint \\
        --truth-npz       ./results/data/<world5>.npz \\
        --K               5 \\
        --output-dir      ./ablation_v7_world5_stage2_inpaint/figures_all_1x

USAGE — MULTI-WORLD
-------------------
    python ablation_visualize_all.py \\
        --metrics-csv            ./ablation_v7_multi/ablation_metrics_multiworld_v2.csv \\
        --ablation-dir-pattern   './ablation_v7_{world_stem}_stage2_inpaint' \\
        --wide-range-csv         ./figures_map_axel_stage2_new/wide_range_species.csv \\
        --truth-dir              ./results/data \\
        --top-n-worlds           3 \\
        --K                      5 \\
        --output-dir             ./ablation_v7_multi/figures_all_1x

DEPENDENCIES
------------
matplotlib, numpy. sklearn optional (for AUC, falls back gracefully).
=============================================================================
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# CONSTANTS
# =============================================================================

VARIANT_ORDER = ['FULL', 'NO_HISTORY', 'NO_NETWORK', 'NO_ENV', 'NO_SPECIES_FEATS']
ABLATION_VARIANTS = [v for v in VARIANT_ORDER if v != 'FULL']

VARIANT_COLOURS = {
    'FULL':              '#2E7D32',
    'NO_HISTORY':        '#C62828',
    'NO_NETWORK':        '#6A1B9A',
    'NO_ENV':            '#1565C0',
    'NO_SPECIES_FEATS':  '#757575',
}
VARIANT_LABELS_SHORT = {
    'FULL':              'FULL\n(baseline)',
    'NO_HISTORY':        'no\nhistory',
    'NO_NETWORK':        'no\nnetwork',
    'NO_ENV':            'no\nenv',
    'NO_SPECIES_FEATS':  'no species\nfeats',
}
VARIANT_LABELS_FULL = {
    'FULL':              'FULL (baseline)',
    'NO_HISTORY':        'NO_HISTORY',
    'NO_NETWORK':        'NO_NETWORK',
    'NO_ENV':            'NO_ENV',
    'NO_SPECIES_FEATS':  'NO_SPECIES_FEATS',
}

CATEGORY_COLOURS = {
    'CRITICAL':    '#C62828',
    'MODERATE':    '#E67E22',
    'NEGLIGIBLE':  '#7F8C8D',
    'BENEFICIAL':  '#27AE60',
    'BASELINE':    '#2E7D32',
    'UNKNOWN':     '#BDC3C7',
}

BAND_COLOURS_HEX = {
    'easy':     '#1f558e',
    'moderate': '#9e6420',
    'hard':     '#8e1d2a',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def read_metrics_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if v in ('', None):
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(r)
    worlds = {r['world'] for r in rows}
    return rows, len(worlds) > 1


def aggregate_by_variant(rows):
    """For each variant, collect (mean, std, n) for every metric."""
    by_var = defaultdict(list)
    for r in rows:
        by_var[r['variant']].append(r)
    metric_keys = [
        'meaningful_mean_recall', 'meaningful_union_recall',
        'meaningful_pix_cov_mean', 'meaningful_pix_cov_union',
        'real_mean_recall', 'real_union_recall',
        'real_pix_cov_union',
        'all_present_mean_recall', 'all_present_union_recall',
        'auc_overall', 'auc_unobs',
        'echo_at_obs', 'fillin_cells', 'n_obs',
    ]
    agg = {}
    for variant, rs in by_var.items():
        vals = {}
        for k in metric_keys:
            xs = [float(r[k]) for r in rs
                  if r.get(k) is not None]
            vals[k] = ((float(np.mean(xs)), float(np.std(xs)), len(xs))
                       if xs else (None, None, 0))
        # category + max drop (from validate_ablation_v2.py output)
        cats = [r.get('effect_category') for r in rs
                if r.get('effect_category')]
        vals['effect_category'] = (cats[0] if cats else
                                    ('BASELINE' if variant == 'FULL'
                                     else 'UNKNOWN'))
        drops = [float(r['max_relative_drop_pct'])
                 for r in rs
                 if r.get('max_relative_drop_pct') not in (None, '')]
        vals['max_drop'] = (
            (float(np.mean(drops)), float(np.std(drops)))
            if drops else (None, None)
        )
        vals['interpretation'] = next(
            (r['interpretation'] for r in rs
             if r.get('interpretation')), '')
        agg[variant] = vals
    return agg


def load_truth(truth_path):
    with np.load(truth_path, allow_pickle=True) as td:
        return (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)


def load_variant_npz(ablation_dir, variant, K):
    p = Path(ablation_dir) / f'recon_{variant}_b{K}_samples.npz'
    if not p.exists():
        return None
    z = np.load(p)
    return {
        'mean':     np.asarray(z['mean']).astype(np.float32),
        'samples':  np.asarray(z['samples']).astype(np.float32),
        'observed': (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8),
    }


def calibrate_per_species(prob, truth, multiplier=1.0):
    S = prob.shape[0]
    binary = np.zeros_like(prob, dtype=np.uint8)
    for s in range(S):
        n_t = int(truth[s].sum())
        if n_t == 0:
            continue
        n_target = max(1, int(n_t * multiplier))
        flat = prob[s].ravel()
        if flat.max() < 1e-6:
            continue
        thr = np.partition(flat, -n_target)[-n_target] - 1e-9
        binary[s] = (prob[s] > thr).astype(np.uint8)
    return binary


def per_species_recall(mean_pred, truth):
    S = mean_pred.shape[0]
    out = np.full(S, np.nan)
    binary = calibrate_per_species(mean_pred, truth)
    for s in range(S):
        n_t = int(truth[s].sum())
        if n_t == 0:
            continue
        out[s] = float((binary[s] & truth[s]).sum()) / n_t
    return out


# =============================================================================
# FIGURE A1 — SUMMARY BAR CHART (4 panels)
# =============================================================================

def fig_summary_bars(agg, output_path, is_multiworld):
    metrics = [
        ('meaningful_mean_recall',
         'Per-species mean recall\n(meaningful: range > K)'),
        ('meaningful_union_recall',
         'Per-species ensemble-union recall\n(meaningful: range > K)'),
        ('meaningful_pix_cov_union',
         'Pixel coverage by ensemble union\n(meaningful: range > K)'),
        ('auc_unobs',
         'AUC at unobserved cells\n(extrapolation)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    variants = [v for v in VARIANT_ORDER if v in agg]
    x_pos = np.arange(len(variants))

    for ax, (key, title) in zip(axes, metrics):
        means = [agg[v][key][0] if agg[v][key][0] is not None else 0
                 for v in variants]
        stds = [agg[v][key][1] if agg[v][key][1] is not None else 0
                for v in variants]
        colours = [VARIANT_COLOURS[v] for v in variants]

        bars = ax.bar(x_pos, means,
                      yerr=stds if is_multiworld else None,
                      capsize=4, color=colours,
                      edgecolor='black', linewidth=0.7)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(means) * 0.02 if max(means) > 0 else 0.005),
                    f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([VARIANT_LABELS_SHORT[v] for v in variants],
                           fontsize=9)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if 'FULL' in variants:
            full_idx = variants.index('FULL')
            ax.axhline(means[full_idx], color=VARIANT_COLOURS['FULL'],
                       linestyle='--', linewidth=0.8, alpha=0.5)

    suptitle = 'A1. Ablation summary'
    if is_multiworld:
        n_w = max(agg[v]['meaningful_mean_recall'][2] for v in variants)
        suptitle += f' — {n_w} worlds (mean ± SD)'
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE A2 — STRATIFIED COMPARISON
# =============================================================================

def fig_stratified(agg, output_path, is_multiworld):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    variants = [v for v in VARIANT_ORDER if v in agg]
    n_v = len(variants)

    strata = [
        ('all_present', 'ALL present species'),
        ('meaningful',  'MEANINGFUL (range > K)'),
        ('real',        'REAL (range > 2K)'),
    ]
    width = 0.25
    for i, (stratum, label) in enumerate(strata):
        key = f'{stratum}_mean_recall'
        means = [agg[v][key][0] if agg[v][key][0] is not None else 0
                 for v in variants]
        stds = [agg[v][key][1] if agg[v][key][1] is not None else 0
                for v in variants]
        offset = (i - 1) * width
        axes[0].bar(np.arange(n_v) + offset, means, width,
                    yerr=stds if is_multiworld else None, capsize=3,
                    label=label, edgecolor='black', linewidth=0.5,
                    alpha=0.85)
    axes[0].set_xticks(np.arange(n_v))
    axes[0].set_xticklabels([VARIANT_LABELS_SHORT[v] for v in variants],
                            fontsize=9)
    axes[0].set_ylabel('Per-species mean recall', fontsize=10)
    axes[0].set_title('Mean recall by stratum', fontsize=11)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    pix_strata = [('meaningful', 'MEANINGFUL'), ('real', 'REAL')]
    width = 0.35
    for i, (stratum, label) in enumerate(pix_strata):
        key = f'{stratum}_pix_cov_union'
        means = [agg[v][key][0] if agg[v][key][0] is not None else 0
                 for v in variants]
        stds = [agg[v][key][1] if agg[v][key][1] is not None else 0
                for v in variants]
        offset = (i - 0.5) * width
        axes[1].bar(np.arange(n_v) + offset, means, width,
                    yerr=stds if is_multiworld else None, capsize=3,
                    label=label, edgecolor='black', linewidth=0.5, alpha=0.85)
    axes[1].set_xticks(np.arange(n_v))
    axes[1].set_xticklabels([VARIANT_LABELS_SHORT[v] for v in variants],
                            fontsize=9)
    axes[1].set_ylabel('Pixel coverage by ensemble union', fontsize=10)
    axes[1].set_title('Pixel coverage by stratum', fontsize=11)
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.suptitle('A2. Stratified ablation — predictor importance grows '
                  'with reconstruction difficulty',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE A3 — INPAINTING DIAGNOSTIC
# =============================================================================

def fig_inpainting_diagnostic(agg, output_path, is_multiworld):
    variants = [v for v in VARIANT_ORDER if v in agg]
    n_v = len(variants)

    n_obs = [agg[v]['n_obs'][0] or 0 for v in variants]
    echo = [agg[v]['echo_at_obs'][0] or 0 for v in variants]
    fillin = [agg[v]['fillin_cells'][0] or 0 for v in variants]
    echo_frac = [e / max(1, no) for e, no in zip(echo, n_obs)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bars = axes[0].bar(np.arange(n_v), echo_frac,
                       color=[VARIANT_COLOURS[v] for v in variants],
                       edgecolor='black', linewidth=0.7)
    axes[0].axhline(1.0, color='black', linestyle=':', linewidth=0.8,
                    label='Perfect inpainting (100%)')
    axes[0].axhline(0.0, color='red', linestyle=':', linewidth=0.8,
                    label='No inpainting (0%)')
    axes[0].set_xticks(np.arange(n_v))
    axes[0].set_xticklabels([VARIANT_LABELS_SHORT[v] for v in variants],
                            fontsize=9)
    axes[0].set_ylabel('Fraction of obs cells reproduced\n(echo / n_obs)',
                       fontsize=10)
    axes[0].set_title('Inpainting verification', fontsize=11)
    axes[0].set_ylim(0, 1.15)
    for bar, ef in zip(bars, echo_frac):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f'{ef:.2f}', ha='center', fontsize=9)
    axes[0].legend(fontsize=8, loc='center right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    bars = axes[1].bar(np.arange(n_v), fillin,
                       color=[VARIANT_COLOURS[v] for v in variants],
                       edgecolor='black', linewidth=0.7)
    axes[1].set_xticks(np.arange(n_v))
    axes[1].set_xticklabels([VARIANT_LABELS_SHORT[v] for v in variants],
                            fontsize=9)
    axes[1].set_ylabel('Cells predicted (>0.5) beyond observations',
                       fontsize=10)
    axes[1].set_title('Predicted volume', fontsize=11)
    for bar, fi in zip(bars, fillin):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (max(fillin) * 0.01 if max(fillin) > 0 else 1),
                     f'{int(fi):,}', ha='center', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.suptitle('A3. Diagnostic: inpainting + extrapolation behaviour per variant',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE A4 — PER-WORLD HEATMAP (multi-world only)
# =============================================================================

def fig_per_world_heatmap(rows, output_path):
    by_world = defaultdict(dict)
    for r in rows:
        by_world[r['world']][r['variant']] = r
    worlds = sorted(by_world.keys())
    variants = [v for v in VARIANT_ORDER
                if any(v in by_world[w] for w in worlds)]

    M = np.full((len(worlds), len(variants)), np.nan)
    for i, w in enumerate(worlds):
        for j, v in enumerate(variants):
            r = by_world[w].get(v)
            if r and r.get('meaningful_mean_recall') is not None:
                M[i, j] = float(r['meaningful_mean_recall'])

    fig, ax = plt.subplots(
        figsize=(max(8, len(variants) * 1.3), max(4, len(worlds) * 0.5)))
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=max(0.3, np.nanmax(M)))
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS_SHORT[v].replace('\n', ' ')
                        for v in variants], rotation=30, ha='right')
    ax.set_yticks(np.arange(len(worlds)))
    short_w = [w[:50] + '...' if len(w) > 53 else w for w in worlds]
    ax.set_yticklabels(short_w, fontsize=8)
    for i in range(len(worlds)):
        for j in range(len(variants)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}',
                        ha='center', va='center', fontsize=8,
                        color='black' if M[i, j] > 0.3 else 'white')
    ax.set_title('A4. Per-world meaningful mean recall',
                 fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, label='mean recall')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE B — PER-VARIANT PER-SPECIES HEATMAPS
# =============================================================================

def select_species_for_b(truth, full_mean, n_per_band=2):
    ranges = truth.reshape(truth.shape[0], -1).sum(axis=1)
    recalls = per_species_recall(full_mean, truth)
    bands = [
        ('easy', 6, 10, 'top'),
        ('moderate', 11, 20, 'median'),
        ('hard', 21, 1000, 'bottom'),
    ]
    picks = []
    for label, lo, hi, mode in bands:
        in_band = np.where((ranges >= lo) & (ranges <= hi)
                           & np.isfinite(recalls))[0]
        if len(in_band) == 0:
            continue
        sorted_in = in_band[np.argsort(recalls[in_band])]
        n = min(n_per_band, len(in_band))
        if mode == 'top':
            chosen = sorted_in[-n:][::-1]
        elif mode == 'bottom':
            chosen = sorted_in[:n]
        else:
            mid = len(sorted_in) // 2
            chosen = sorted_in[max(0, mid - n // 2):
                                max(0, mid - n // 2) + n]
        for s in chosen:
            picks.append({
                'species_idx': int(s),
                'range':       int(ranges[s]),
                'full_recall': float(recalls[s]),
                'band':        label,
            })
    return picks


def binary_to_rgba(binary, colour, alpha_fg=0.95):
    h, w = binary.shape
    out = np.ones((h, w, 4), dtype=np.float32)
    out[..., :3] = 1.0
    out[..., 3] = 1.0
    mask = binary > 0
    out[mask, 0] = colour[0]
    out[mask, 1] = colour[1]
    out[mask, 2] = colour[2]
    out[mask, 3] = alpha_fg
    return out


def hex_to_rgb(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def fig_per_variant_maps(truth, observed, variant_data, picks, output_path,
                          K=5, world_label=None):
    if not picks or 'FULL' not in variant_data:
        print(f"  ⚠ skipping fig_B (no picks or no FULL data)")
        return
    n_rows = len(picks)
    cols = ['TRUTH', 'OBSERVED'] + VARIANT_ORDER
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.0 * n_cols, 2.4 * n_rows),
                              squeeze=False)
    title = ('B. Per-variant per-species reconstruction maps  '
             '(rows = species spanning difficulty bands; '
             'cols = ablation variants)')
    if world_label:
        title += f'\nWorld: {world_label}  |  K={K}'
    fig.suptitle(title, fontweight='bold', fontsize=12, y=0.997)

    for row_idx, pick in enumerate(picks):
        s = pick['species_idx']
        rng = pick['range']
        band = pick['band']
        band_rgb = hex_to_rgb(BAND_COLOURS_HEX[band])

        ax = axes[row_idx, 0]
        ax.imshow(binary_to_rgba(truth[s], band_rgb), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title('TRUTH', fontweight='bold', fontsize=10)

        ax = axes[row_idx, 1]
        ax.imshow(binary_to_rgba(observed[s], band_rgb),
                  interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(f'OBSERVED (K={K})', fontweight='bold', fontsize=10)

        for var_idx, variant in enumerate(VARIANT_ORDER):
            ax = axes[row_idx, 2 + var_idx]
            d = variant_data.get(variant)
            if d is None:
                ax.text(0.5, 0.5, '(missing)', ha='center', va='center',
                        transform=ax.transAxes, color='red')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            mean_pred = d['mean'][s]
            ax.imshow(mean_pred, cmap='Blues', vmin=0, vmax=1,
                      interpolation='nearest')
            for yy in range(truth.shape[1]):
                for xx in range(truth.shape[2]):
                    if truth[s, yy, xx] > 0:
                        ax.add_patch(mpatches.Rectangle(
                            (xx - 0.5, yy - 0.5), 1, 1,
                            edgecolor='red', facecolor='none', linewidth=1.0))
            obs_cells = np.argwhere(observed[s] > 0)
            for (yy, xx) in obs_cells:
                ax.add_patch(mpatches.Circle(
                    (xx, yy), 0.28, facecolor='yellow',
                    edgecolor='black', linewidth=0.6, zorder=5))
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(VARIANT_LABELS_SHORT[variant].replace('\n', ' '),
                             fontweight='bold', fontsize=9)

        axes[row_idx, 0].set_ylabel(
            f"sp #{s}\n[{band.upper()}]\n"
            f"range={rng}\nFULL recall={pick['full_recall']:.0%}",
            fontsize=8, rotation=0, ha='right', va='center', labelpad=46,
            color=BAND_COLOURS_HEX[band], fontweight='bold',
        )

    fig.text(0.5, 0.005,
             'Heatmap = predicted probability (Blues, 0=light → 1=dark). '
             'Red outlines = true presence cells. Yellow circles = K observation cells.',
             ha='center', fontsize=8, style='italic', color='#444')
    plt.tight_layout(rect=[0.04, 0.025, 1, 0.965])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE C — PUBLICATION SUMMARY TABLE
# =============================================================================

def fig_summary_table(agg, output_path, is_multiworld):
    fig = plt.figure(figsize=(15, 7.5))
    ax_table = fig.add_axes([0.04, 0.42, 0.93, 0.50])
    ax_table.axis('off')

    headers = ['Variant', 'Effect', 'Max Δ %',
               'mean_rec\n(meaningful)',
               'union_rec\n(meaningful)',
               'pix_cov_union\n(meaningful)',
               'AUC at\nunobserved',
               'Interpretation']
    col_widths = [0.10, 0.10, 0.08, 0.10, 0.10, 0.11, 0.10, 0.31]

    x = 0; y = 0.92; row_h = 0.10
    for i, h in enumerate(headers):
        ax_table.text(x + col_widths[i] / 2, y, h,
                      ha='center', va='center',
                      fontsize=9, fontweight='bold',
                      transform=ax_table.transAxes)
        x += col_widths[i]
    y -= row_h

    for variant in VARIANT_ORDER:
        if variant not in agg:
            continue
        d = agg[variant]
        cat = d['effect_category']
        bg = CATEGORY_COLOURS.get(cat, '#EEEEEE')

        ax_table.add_patch(plt.Rectangle(
            (0, y - row_h / 2 - 0.012), 1, row_h,
            facecolor=bg, alpha=0.18, transform=ax_table.transAxes))

        def fmt(metric_key):
            v = d.get(metric_key)
            if v is None or v[0] is None:
                return '—'
            mean, std, _ = v
            if is_multiworld and std is not None and std > 0.001:
                return f'{mean:.3f}\n±{std:.3f}'
            return f'{mean:.3f}'

        max_drop_mean, max_drop_std = d['max_drop']
        if max_drop_mean is None:
            max_drop_str = '—'
        else:
            max_drop_str = f'{max_drop_mean:+.1f}%'
            if is_multiworld and max_drop_std is not None:
                max_drop_str += f'\n±{max_drop_std:.1f}'

        cells = [
            variant.replace('_', '\n', 1),
            cat,
            max_drop_str,
            fmt('meaningful_mean_recall'),
            fmt('meaningful_union_recall'),
            fmt('meaningful_pix_cov_union'),
            fmt('auc_unobs'),
            (d.get('interpretation') or '')[:120],
        ]

        x = 0
        for i, c in enumerate(cells):
            colour = CATEGORY_COLOURS[cat] if i == 1 else 'black'
            weight = 'bold' if i in (0, 1, 2) else 'normal'
            fontsize = 8 if i == 7 else 9
            ha = 'left' if i == 7 else 'center'
            x_text = x + (0.005 if i == 7 else col_widths[i] / 2)
            ax_table.text(x_text, y, c,
                          ha=ha, va='center',
                          fontsize=fontsize, color=colour, fontweight=weight,
                          transform=ax_table.transAxes, wrap=True)
            x += col_widths[i]
        y -= row_h

    ax_table.text(0.5, 1.02, 'C. Predictor importance — ablation summary',
                   fontsize=13, fontweight='bold', ha='center',
                   transform=ax_table.transAxes)

    ax_bar = fig.add_axes([0.10, 0.06, 0.85, 0.30])
    ablations = [v for v in VARIANT_ORDER if v != 'FULL' and v in agg]
    drops = [agg[v]['max_drop'][0] or 0 for v in ablations]
    drop_stds = [agg[v]['max_drop'][1] or 0 for v in ablations]
    cats = [agg[v]['effect_category'] for v in ablations]
    colours = [CATEGORY_COLOURS[c] for c in cats]

    y_pos = np.arange(len(ablations))
    bars = ax_bar.barh(y_pos, drops,
                        xerr=drop_stds if is_multiworld else None,
                        capsize=4, color=colours,
                        edgecolor='black', linewidth=0.7)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ablations, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Max relative drop in metrics (%)\n'
                       '(positive = ablation hurts; negative = better without it)',
                       fontsize=10)
    ax_bar.axvline(0, color='black', linewidth=0.8)
    ax_bar.axvline(20, color='#E67E22', linestyle='--', linewidth=0.8,
                    alpha=0.6, label='MODERATE threshold (20%)')
    ax_bar.axvline(50, color='#C62828', linestyle='--', linewidth=0.8,
                    alpha=0.6, label='CRITICAL threshold (50%)')
    for bar, dv, c in zip(bars, drops, cats):
        x_b = bar.get_width()
        ha_b = 'left' if x_b >= 0 else 'right'
        offset = 1 if x_b >= 0 else -1
        ax_bar.text(x_b + offset, bar.get_y() + bar.get_height() / 2,
                    f'  [{c}] {dv:+.1f}%',
                    va='center', ha=ha_b, fontsize=9,
                    color=CATEGORY_COLOURS[c], fontweight='bold')
    ax_bar.set_title('Predictor importance ranking',
                      fontsize=12, fontweight='bold', loc='left')
    ax_bar.legend(loc='lower right', fontsize=8)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.grid(axis='x', alpha=0.3)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE D1 — PER-SPECIES DELTA SCATTER
# =============================================================================

def fig_per_species_delta(truth, full_recall, variant_recalls,
                           output_path, K=5):
    ranges = truth.reshape(truth.shape[0], -1).sum(axis=1)
    bands = [
        ('easy (range 6-10)',     6, 10),
        ('moderate (range 11-20)', 11, 20),
        ('hard (range 21+)',       21, 1000),
    ]
    fig, axes = plt.subplots(
        len(ABLATION_VARIANTS), len(bands),
        figsize=(4.0 * len(bands), 3.0 * len(ABLATION_VARIANTS)),
        squeeze=False,
    )
    for row_idx, variant in enumerate(ABLATION_VARIANTS):
        v_rec = variant_recalls.get(variant)
        if v_rec is None:
            continue
        for col_idx, (band_label, lo, hi) in enumerate(bands):
            ax = axes[row_idx, col_idx]
            mask = ((ranges >= lo) & (ranges <= hi)
                    & np.isfinite(full_recall) & np.isfinite(v_rec))
            n = int(mask.sum())
            if n == 0:
                ax.set_visible(False)
                continue
            ax.scatter(full_recall[mask], v_rec[mask],
                       s=14, alpha=0.55,
                       color=VARIANT_COLOURS[variant], edgecolor='none')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.6,
                    label='y = x (no effect)')
            delta_mean = float(np.mean(v_rec[mask] - full_recall[mask]))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            if row_idx == len(ABLATION_VARIANTS) - 1:
                ax.set_xlabel('FULL recall', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f'{variant}\nrecall', fontsize=9,
                              color=VARIANT_COLOURS[variant], fontweight='bold')
            if row_idx == 0:
                ax.set_title(f'{band_label}  (n={n})', fontsize=10)
            ax.text(0.05, 0.95, f'Δmean = {delta_mean:+.2f}',
                     transform=ax.transAxes, fontsize=9, va='top',
                     color=VARIANT_COLOURS[variant], fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.85,
                                edgecolor='none'))
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.suptitle(
        f'D1. Per-species ablation effect by range-size band  (K={K})\n'
        f'Each point = one species. y=x dashed = "ablation has no effect". '
        f'Below the line = ablation hurt this species.',
        fontsize=12, fontweight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE D2 — RECALL VS RANGE SIZE (per variant)
# =============================================================================

def fig_recall_vs_range(truth, variant_recalls, output_path, K=5):
    ranges = truth.reshape(truth.shape[0], -1).sum(axis=1)
    bin_edges = np.array([0, 5, 10, 15, 20, 25, 30, 50, 100, 1000])
    bin_centres = [(bin_edges[i] + bin_edges[i+1]) / 2
                    for i in range(len(bin_edges) - 1)]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bin_n_species = []
    for i in range(len(bin_edges) - 1):
        in_bin = (ranges >= bin_edges[i]) & (ranges < bin_edges[i+1])
        bin_n_species.append(int(in_bin.sum()))

    for variant in VARIANT_ORDER:
        v_rec = variant_recalls.get(variant)
        if v_rec is None:
            continue
        means = []; stds = []
        for i in range(len(bin_edges) - 1):
            in_bin = ((ranges >= bin_edges[i])
                      & (ranges < bin_edges[i+1])
                      & np.isfinite(v_rec))
            if in_bin.sum() == 0:
                means.append(np.nan); stds.append(np.nan)
            else:
                means.append(float(np.mean(v_rec[in_bin])))
                stds.append(float(np.std(v_rec[in_bin])))
        means = np.array(means); stds = np.array(stds)
        valid = ~np.isnan(means)
        ax.errorbar(np.array(bin_centres)[valid], means[valid],
                    yerr=stds[valid],
                    color=VARIANT_COLOURS[variant], label=variant,
                    marker='o', markersize=5,
                    linewidth=2 if variant == 'FULL' else 1.3,
                    alpha=0.95 if variant == 'FULL' else 0.8, capsize=3)

    for bc, n in zip(bin_centres, bin_n_species):
        if n > 0:
            ax.annotate(f'n={n}', (bc, -0.06),
                         fontsize=8, ha='center', color='#666',
                         annotation_clip=False)

    ax.axvline(K, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
               label=f'K = {K}')
    ax.axvline(2 * K, color='grey', linestyle='--', linewidth=0.8, alpha=0.4,
               label=f'2K = {2*K}')
    ax.set_xlabel('True range size (cells)', fontsize=11)
    ax.set_ylabel('Per-species recall (mean ± SD across species in bin)',
                   fontsize=11)
    ax.set_xscale('log')
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(['5', '10', '20', '50', '100'])
    ax.set_xlim(2, 200)
    ax.set_ylim(-0.1, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f'D2. Predictor importance vs species range size  (K={K})',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE D3 — RICHNESS + BETA DIVERSITY
# =============================================================================

def compute_richness_per_cell(binary):
    return binary.sum(axis=0)


def compute_sorensen_per_pair(binary, max_pairs=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    S, Y, X = binary.shape
    flat = binary.reshape(S, Y * X)
    has_species = flat.sum(axis=0) > 0
    valid_cells = np.where(has_species)[0]
    if len(valid_cells) < 2:
        return None
    n_pairs = min(max_pairs,
                   len(valid_cells) * (len(valid_cells) - 1) // 2)
    pairs = rng.choice(len(valid_cells), size=(n_pairs, 2), replace=True)
    dissims = []
    for i, j in pairs:
        if i == j:
            continue
        ci = valid_cells[i]; cj = valid_cells[j]
        a = flat[:, ci].astype(int)
        b = flat[:, cj].astype(int)
        intersection = int((a & b).sum())
        a_only = int(a.sum()) - intersection
        b_only = int(b.sum()) - intersection
        denom = 2 * intersection + a_only + b_only
        if denom == 0:
            continue
        dissims.append((a_only + b_only) / denom)
    return float(np.mean(dissims)) if dissims else None


def fig_richness_betadiv(truth, variant_data, output_path, K=5):
    rng = np.random.default_rng(42)
    truth_rich = compute_richness_per_cell(truth).ravel()
    truth_sor = compute_sorensen_per_pair(truth, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for variant in VARIANT_ORDER:
        d = variant_data.get(variant)
        if d is None:
            continue
        bin_pred = calibrate_per_species(d['mean'], truth)
        pred_rich = compute_richness_per_cell(bin_pred).ravel()
        if pred_rich.std() > 0 and truth_rich.std() > 0:
            corr = float(np.corrcoef(truth_rich, pred_rich)[0, 1])
        else:
            corr = 0.0
        jitter = rng.normal(0, 0.15, len(truth_rich))
        axes[0].scatter(truth_rich + jitter, pred_rich + jitter,
                         s=10, alpha=0.35, color=VARIANT_COLOURS[variant],
                         label=f'{variant}  r={corr:+.2f}',
                         edgecolor='none')
    max_r = float(truth_rich.max()) * 1.1
    axes[0].plot([0, max_r], [0, max_r], 'k--', linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel('True species richness per cell', fontsize=11)
    axes[0].set_ylabel('Predicted species richness per cell', fontsize=11)
    axes[0].set_title('A. Per-cell species richness', fontsize=11,
                       fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=8, framealpha=0.9)
    axes[0].grid(alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    sors = {'TRUTH': truth_sor}
    for variant in VARIANT_ORDER:
        d = variant_data.get(variant)
        if d is None:
            continue
        bin_pred = calibrate_per_species(d['mean'], truth)
        sors[variant] = compute_sorensen_per_pair(bin_pred, rng=rng)

    labels = list(sors.keys())
    values = [sors[k] if sors[k] is not None else 0 for k in labels]
    colours = ['black'] + [VARIANT_COLOURS[l] for l in labels[1:]]
    bars = axes[1].bar(np.arange(len(labels)), values, color=colours,
                        edgecolor='black', linewidth=0.7, alpha=0.85)
    if truth_sor is not None:
        axes[1].axhline(truth_sor, color='black', linestyle='--',
                        linewidth=0.8, alpha=0.5,
                        label='Truth dissimilarity')
    for bar, v in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f'{v:.3f}', ha='center', fontsize=9)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel('Mean Sørensen dissimilarity', fontsize=11)
    axes[1].set_title('B. Community beta-diversity (Sørensen)',
                       fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].legend(loc='lower right', fontsize=8)

    fig.suptitle(f'D3. Community-level reconstruction under ablation  (K={K})',
                  fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# FIGURE E — AXEL'S THREE-MAP DEMONSTRATION (post-ablation)
# =============================================================================
# Exactly the figure Axel asked for in the meeting transcript:
#   "I would want to see the three three maps next to each other.
#    The first is the actual distribution... Next picture is the simulated
#    observations... and this is what my AI algorithm reconstructed."
# Plus the email follow-up: "the AI would generate a new plausible
# distribution consistent with the point observations with each run
# (so, it would generate what is called a statistical 'ensemble')".
#
# This figure produces 6 columns per species:
#   TRUTH | OBSERVED (K=5) | RECON MEAN | SAMPLE 1 | SAMPLE 2 | SAMPLE 3
# Showing both the deterministic mean reconstruction AND three ensemble
# samples drawn from the diffusion model — directly demonstrating that
# the ground truth is "part of the ensemble" in Axel's words.
# =============================================================================

def fig_E_axel_three_map_demo(truth, observed, full_data, picks,
                               output_path, K=5, world_label=None):
    """
    Axel's requested figure: TRUTH | OBSERVED | RECON | 3 ensemble samples.

    truth      : (S, Y, X) ground truth (binary)
    observed   : (S, Y, X) K observations (binary)
    full_data  : dict with 'mean' (S, Y, X) and 'samples' (n_ens, S, Y, X)
    picks      : list of dicts with species_idx, range, full_recall, band
    """
    if not picks:
        print(f"  ⚠ skipping fig_E (no picks)")
        return
    if 'samples' not in full_data:
        print(f"  ⚠ skipping fig_E (no samples in FULL data)")
        return
    if full_data['samples'].shape[0] < 3:
        print(f"  ⚠ skipping fig_E (need ≥3 ensemble members, got "
              f"{full_data['samples'].shape[0]})")
        return

    n_rows = len(picks)
    cols = ['TRUTH', f'OBSERVED\n(K={K})', 'RECON MEAN',
            'SAMPLE 1', 'SAMPLE 2', 'SAMPLE 3']
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.0 * n_cols, 2.2 * n_rows),
                              squeeze=False)

    title = (f'E. Axel three-map demonstration  —  '
             f'TRUTH | OBSERVED (K={K}) | RECON | 3 ensemble samples')
    if world_label:
        title += f'\nWorld: {world_label}'
    fig.suptitle(title, fontweight='bold', fontsize=12, y=0.998)

    for row_idx, pick in enumerate(picks):
        s = pick['species_idx']
        rng = pick['range']
        band = pick['band']
        band_rgb = hex_to_rgb(BAND_COLOURS_HEX[band])

        # Col 0: TRUTH
        ax = axes[row_idx, 0]
        ax.imshow(binary_to_rgba(truth[s], band_rgb), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title('TRUTH', fontweight='bold', fontsize=10)

        # Col 1: OBSERVED
        ax = axes[row_idx, 1]
        ax.imshow(binary_to_rgba(observed[s], band_rgb),
                  interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(f'OBSERVED (K={K})', fontweight='bold', fontsize=10)

        # Col 2: RECON MEAN
        ax = axes[row_idx, 2]
        mean_pred = full_data['mean'][s]
        ax.imshow(mean_pred, cmap='Blues', vmin=0, vmax=1,
                  interpolation='nearest')
        for yy in range(truth.shape[1]):
            for xx in range(truth.shape[2]):
                if truth[s, yy, xx] > 0:
                    ax.add_patch(mpatches.Rectangle(
                        (xx - 0.5, yy - 0.5), 1, 1,
                        edgecolor='red', facecolor='none', linewidth=1.0))
        obs_cells = np.argwhere(observed[s] > 0)
        for (yy, xx) in obs_cells:
            ax.add_patch(mpatches.Circle(
                (xx, yy), 0.28, facecolor='yellow',
                edgecolor='black', linewidth=0.6, zorder=5))
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title('RECON MEAN', fontweight='bold', fontsize=10)

        # Cols 3-5: ensemble samples (3 of 8)
        # We deliberately show samples 0, 3, 6 (spread out) rather than
        # 0, 1, 2 (which would all be very correlated)
        sample_indices = [0, 3, 6] if full_data['samples'].shape[0] >= 7 \
            else list(range(min(3, full_data['samples'].shape[0])))
        for j, sample_idx in enumerate(sample_indices):
            ax = axes[row_idx, 3 + j]
            sample_pred = full_data['samples'][sample_idx, s]
            ax.imshow(sample_pred, cmap='Blues', vmin=0, vmax=1,
                      interpolation='nearest')
            for yy in range(truth.shape[1]):
                for xx in range(truth.shape[2]):
                    if truth[s, yy, xx] > 0:
                        ax.add_patch(mpatches.Rectangle(
                            (xx - 0.5, yy - 0.5), 1, 1,
                            edgecolor='red', facecolor='none', linewidth=1.0))
            for (yy, xx) in obs_cells:
                ax.add_patch(mpatches.Circle(
                    (xx, yy), 0.28, facecolor='yellow',
                    edgecolor='black', linewidth=0.6, zorder=5))
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f'SAMPLE {j+1}', fontweight='bold', fontsize=10)

        # Y-axis label
        axes[row_idx, 0].set_ylabel(
            f"sp #{s}\n[{band.upper()}]\n"
            f"range={rng}\nrecall={pick['full_recall']:.0%}",
            fontsize=8, rotation=0, ha='right', va='center', labelpad=46,
            color=BAND_COLOURS_HEX[band], fontweight='bold',
        )

    fig.text(0.5, 0.005,
             'Heatmaps = predicted probability (Blues, 0=light → 1=dark). '
             'Red outlines = true presence cells. Yellow circles = K observation cells. '
             'Three ensemble samples show stochastic variation; truth should be '
             'visible as plausible across multiple samples.',
             ha='center', fontsize=8, style='italic', color='#444')
    plt.tight_layout(rect=[0.04, 0.025, 1, 0.965])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# =============================================================================
# MAIN  —  orchestrates everything
# =============================================================================

def process_one_world(truth_path, ablation_dir, K, variants):
    truth = load_truth(truth_path)
    variant_data = {}
    variant_recalls = {}
    for v in variants:
        d = load_variant_npz(ablation_dir, v, K)
        if d is None:
            print(f"  ⚠ {v} missing in {ablation_dir}")
            continue
        variant_data[v] = d
        variant_recalls[v] = per_species_recall(d['mean'], truth)
    return truth, variant_data, variant_recalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics-csv', required=True,
                    help='CSV from validate_ablation_v2.py (1× or 2×)')
    # Single-world mode
    ap.add_argument('--ablation-dir', default=None)
    ap.add_argument('--truth-npz', default=None)
    # Multi-world mode
    ap.add_argument('--ablation-dir-pattern', default=None)
    ap.add_argument('--wide-range-csv', default=None)
    ap.add_argument('--truth-dir', default=None)
    ap.add_argument('--top-n-worlds', type=int, default=3)
    # Common
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--variants', nargs='+', default=VARIANT_ORDER)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--n-per-band', type=int, default=2,
                    help='species per band for fig B (default 2)')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  ABLATION_VISUALIZE_ALL  →  {out_dir}")
    print(f"{'='*70}")

    # ── 1. Read metrics CSV ────────────────────────────────────────────
    print(f"\nReading metrics: {args.metrics_csv}")
    rows, is_multiworld = read_metrics_csv(args.metrics_csv)
    print(f"  rows={len(rows)}  multi-world={is_multiworld}")
    agg = aggregate_by_variant(rows)

    # ── 2. Figures from CSV alone (don't need NPZs) ────────────────────
    print(f"\n[CSV-driven figures]")
    fig_summary_bars(agg, out_dir / 'fig_A1_summary_bars.png', is_multiworld)
    fig_stratified(agg, out_dir / 'fig_A2_stratified.png', is_multiworld)
    fig_inpainting_diagnostic(
        agg, out_dir / 'fig_A3_inpainting_diagnostic.png', is_multiworld)
    if is_multiworld:
        fig_per_world_heatmap(rows, out_dir / 'fig_A4_per_world_heatmap.png')
    fig_summary_table(agg, out_dir / 'fig_C_summary_table.png', is_multiworld)

    # ── 3. NPZ-driven figures: load truth + variant data ────────────────
    print(f"\n[NPZ-driven figures]")

    if args.ablation_dir and args.truth_npz:
        truth, variant_data, variant_recalls = process_one_world(
            args.truth_npz, args.ablation_dir, args.K, args.variants)
        if 'FULL' not in variant_data:
            print("  ✗ FULL variant missing; cannot produce B/D figures")
            return 1
        observed = variant_data['FULL']['observed']
        full_mean = variant_data['FULL']['mean']
        full_recall = variant_recalls['FULL']
        v_rec_others = {v: r for v, r in variant_recalls.items()
                         if v != 'FULL'}

        picks = select_species_for_b(truth, full_mean, args.n_per_band)
        for p in picks:
            print(f"    sp #{p['species_idx']:>4}  [{p['band']:<8}]  "
                  f"range={p['range']:>3}  FULL_recall={p['full_recall']:.2f}")

        fig_per_variant_maps(truth, observed, variant_data, picks,
                              out_dir / 'fig_B_per_variant_maps.png',
                              K=args.K,
                              world_label=Path(args.truth_npz).stem)
        fig_per_species_delta(truth, full_recall, v_rec_others,
                               out_dir / 'fig_D1_per_species_delta.png',
                               K=args.K)
        fig_recall_vs_range(truth, variant_recalls,
                             out_dir / 'fig_D2_recall_vs_range.png', K=args.K)
        fig_richness_betadiv(truth, variant_data,
                              out_dir / 'fig_D3_richness_betadiv.png',
                              K=args.K)
        fig_E_axel_three_map_demo(
            truth, observed, variant_data['FULL'], picks,
            out_dir / 'fig_E_axel_three_map_demo.png',
            K=args.K, world_label=Path(args.truth_npz).stem)

    elif args.ablation_dir_pattern and args.wide_range_csv \
            and args.truth_dir:
        # Pool across worlds
        world_count = defaultdict(int)
        with open(args.wide_range_csv) as f:
            for r in csv.DictReader(f):
                world_count[r['world']] += 1
        top_worlds = sorted(world_count.items(),
                             key=lambda x: -x[1])[:args.top_n_worlds]

        pooled_truth = []
        pooled_recalls = defaultdict(list)
        first_world_truth = None
        first_variant_data = {}
        first_observed = None
        first_full_mean = None
        first_world_label = None

        for world_name, _ in top_worlds:
            stem = world_name.replace('.npz', '')
            ad = Path(args.ablation_dir_pattern.format(world_stem=stem))
            tp = Path(args.truth_dir) / world_name
            if not ad.exists():
                print(f"  ⚠ skip {world_name}: dir missing")
                continue
            print(f"\n  --- {world_name[:55]} ---")
            tr, vd, vr = process_one_world(tp, ad, args.K, args.variants)
            pooled_truth.append(tr)
            for v, rec in vr.items():
                pooled_recalls[v].append(rec)
            if first_world_truth is None:
                first_world_truth = tr
                first_variant_data = vd
                first_observed = vd.get('FULL', {}).get('observed')
                first_full_mean = vd.get('FULL', {}).get('mean')
                first_world_label = stem

        if not pooled_truth:
            print("  ✗ no worlds processed")
            return 1

        truth_concat = np.concatenate(pooled_truth, axis=0)
        v_recalls_concat = {v: np.concatenate(rs)
                             for v, rs in pooled_recalls.items()}
        if 'FULL' not in v_recalls_concat:
            print("  ✗ FULL missing; cannot produce D figures")
            return 1
        full_recall_concat = v_recalls_concat['FULL']
        v_rec_others = {v: r for v, r in v_recalls_concat.items()
                         if v != 'FULL'}

        # Figs D1, D2 use POOLED data across worlds for statistical robustness
        fig_per_species_delta(
            truth_concat, full_recall_concat, v_rec_others,
            out_dir / 'fig_D1_per_species_delta.png', K=args.K)
        fig_recall_vs_range(
            truth_concat, v_recalls_concat,
            out_dir / 'fig_D2_recall_vs_range.png', K=args.K)

        # Figs B, D3 use the FIRST world only
        # (community structure isn't well-defined across worlds)
        if first_world_truth is not None and first_variant_data:
            picks = select_species_for_b(
                first_world_truth, first_full_mean, args.n_per_band)
            for p in picks:
                print(f"    sp #{p['species_idx']:>4}  [{p['band']:<8}]  "
                      f"range={p['range']:>3}  "
                      f"FULL_recall={p['full_recall']:.2f}")
            fig_per_variant_maps(
                first_world_truth, first_observed, first_variant_data, picks,
                out_dir / 'fig_B_per_variant_maps.png',
                K=args.K, world_label=first_world_label)
            fig_richness_betadiv(
                first_world_truth, first_variant_data,
                out_dir / 'fig_D3_richness_betadiv.png', K=args.K)
            if 'FULL' in first_variant_data:
                fig_E_axel_three_map_demo(
                    first_world_truth, first_observed,
                    first_variant_data['FULL'], picks,
                    out_dir / 'fig_E_axel_three_map_demo.png',
                    K=args.K, world_label=first_world_label)

    else:
        print("  ✗ Specify either:")
        print("      --ablation-dir AND --truth-npz   (single-world)")
        print("      OR")
        print("      --ablation-dir-pattern, --wide-range-csv, --truth-dir "
              "(multi-world)")
        return 1

    print(f"\n{'='*70}")
    print(f"  All figures written to: {out_dir.resolve()}")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())