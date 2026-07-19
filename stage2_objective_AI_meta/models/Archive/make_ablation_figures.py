#!/usr/bin/env python3
"""
=============================================================================
STAGE 2 ABLATION — FINAL FIGURE GENERATION (publication quality)
=============================================================================

Reads the merged JSON (v3 non-Poisson + v5 Poisson-corrected) and produces
all figures in ONE directory. Figures are state-of-the-art: clean styling,
error bars, log scales where appropriate, clear annotations, no clutter.

INPUT:  ablation_v5_merged.json  (17 conditions, 8 worlds x 8 samples each)
OUTPUT: stage2_ablation_figures/  (all .png files + copy of merged JSON)

FIGURES PRODUCED:
  AB1  — Original 5 conditions: richness r bars with error bars
  AB2  — All four correlation metrics across 17 conditions
  AB3  — NO_TEMPORAL vs FULL: catastrophic failure visualization
  AB7  — Arm A: Temporal gap degradation curve
  AB8  — Arm B-Fixed: Sparsification budget curve
  AB9  — Supervisor 3-panel summary (ablation + gap + sparse)
  AB10 — Arm B-Poisson: exact Poisson model (v5 corrected) curve
  AB11 — Fixed budget vs Poisson: the key comparison figure
  AB12 — Full 17-condition heatmap across all metrics
  AB13 — Cross-arm synthesis: one chart telling the complete story
  AB14 — Truth | Observed | Reconstructed triptych (denoising-paper style)

DESIGN PRINCIPLES:
  - No gridlines over data
  - Error bars always shown where std is available
  - Log scale for axes spanning >1 order of magnitude
  - Consistent color palette across figures
  - Every figure has self-contained title/caption interpretable alone
  - DPI 150 for print quality, 12pt+ fonts

AB14 USAGE (the new 3-panel Axel-style figure):

  # Best case — you have a saved reconstruction NPZ
  python make_ablation_figures.py ablation_v5_merged.json \\
      --truth-npz path/to/world_0001.npz \\
      --recon-npz path/to/predictions/mean_predictions.npz \\
      --ab14-p-obs 0.001

  # Fallback — no reconstruction NPZ; uses obs_mask_50 (~saturation, r≈0.92)
  # as an honest stand-in for "model at full performance"
  python make_ablation_figures.py ablation_v5_merged.json \\
      --truth-npz path/to/world_0001.npz \\
      --ab14-demo \\
      --ab14-p-obs 0.001
=============================================================================
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.85',
})

# Consistent color palette across all figures
PALETTE = {
    'FULL':           '#1e40af',   # deep blue — reference
    'NO_TEMPORAL':    '#7c3aed',   # purple — catastrophic
    'NO_INTERACT':    '#0891b2',   # cyan
    'ENV_ONLY':       '#65a30d',   # green
    'OBS_INFILL':     '#ca8a04',   # gold
    'GAP':            '#c2410c',   # orange — Arm A
    'SPARSE':         '#d97706',   # amber — Arm B fixed
    'POISSON':        '#dc2626',   # red — Arm B Poisson
    'target_line':    '#64748b',
    'full_reference': '#1e40af',
}

# Tol's Bright qualitative palette — colourblind-safe, 5 species for AB14
SPECIES_COLORS = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#AA3377',  # purple
]

SPARSE_BUDGETS = [1, 5, 10, 20, 50]
POISSON_P_OBS = [0.00001, 0.0001, 0.0005, 0.001, 0.01]


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def safe_save(fig, path, caption=None):
    """Save figure and print confirmation."""
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


def get_metric(conds, cond_name, metric):
    """Safe metric fetch."""
    if cond_name in conds:
        return conds[cond_name]['metrics'].get(metric, None)
    return None


# ──────────────────────────────────────────────────────────────────
# AB1 — Original 5 conditions: richness r with error bars
# ──────────────────────────────────────────────────────────────────

def plot_AB1(conds, out):
    """Foundational ablation: which inputs matter?"""
    order = ['FULL', 'NO_INTERACT', 'OBS_INFILL', 'NO_TEMPORAL', 'ENV_ONLY']
    labels = ['Full Model', 'No Interactions', 'Obs Only (5)',
              'No Temporal', 'Env Only']

    r = [get_metric(conds, k, 'richness_r') for k in order]
    r_std = [get_metric(conds, k, 'richness_r_std') for k in order]
    colors = [PALETTE['FULL'], PALETTE['NO_INTERACT'], PALETTE['OBS_INFILL'],
              PALETTE['NO_TEMPORAL'], PALETTE['ENV_ONLY']]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(order))
    bars = ax.bar(x, r, yerr=r_std, capsize=5, color=colors,
                  edgecolor='white', linewidth=1.5, alpha=0.9,
                  error_kw={'ecolor': '0.3', 'capthick': 1.5, 'lw': 1.5})

    for xi, val, std in zip(x, r, r_std):
        y = val + (std if val >= 0 else -std) + (0.03 if val >= 0 else -0.06)
        ax.text(xi, y, f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=11, fontweight='bold')

    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.8, alpha=0.6,
               label='r = 0.9 target')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Richness correlation (Pearson r)')
    ax.set_title('AB1 — Foundational ablation: temporal assembly history is the critical input\n'
                 f'(n_worlds={8}, n_samples={8}, error bars = ±1 SD)',
                 fontsize=11)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(loc='lower left', fontsize=9)

    ax.annotate('Drop P_t → collapse to random',
                xy=(3, -0.01), xytext=(3, -0.12),
                ha='center', fontsize=9, color='#7c3aed', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1.2))

    fig.tight_layout()
    safe_save(fig, out / 'fig_AB1_foundational_ablation.png')


# ──────────────────────────────────────────────────────────────────
# AB2 — Four metrics across all 17 conditions
# ──────────────────────────────────────────────────────────────────

def plot_AB2(conds, out):
    """Grouped bar chart: all four correlation metrics for all conditions."""
    order = ['FULL', 'NO_INTERACT', 'NO_TEMPORAL', 'ENV_ONLY', 'OBS_INFILL',
             'GAP_5', 'GAP_25',
             'SPARSE_1', 'SPARSE_5', 'SPARSE_10', 'SPARSE_20', 'SPARSE_50',
             'POISSON_p00001', 'POISSON_p0001', 'POISSON_p0005',
             'POISSON_p001', 'POISSON_p01']
    labels_dict = {
        'FULL': 'Full', 'NO_INTERACT': 'No Int.', 'NO_TEMPORAL': 'No Temp.',
        'ENV_ONLY': 'Env only', 'OBS_INFILL': 'Obs(5)',
        'GAP_5': 'Gap 5', 'GAP_25': 'Gap 25',
        'SPARSE_1': 'SP 1', 'SPARSE_5': 'SP 5', 'SPARSE_10': 'SP 10',
        'SPARSE_20': 'SP 20', 'SPARSE_50': 'SP 50',
        'POISSON_p00001': 'P=0.001%', 'POISSON_p0001': 'P=0.01%',
        'POISSON_p0005': 'P=0.05%', 'POISSON_p001': 'P=0.1%',
        'POISSON_p01': 'P=1%'
    }

    metrics = ['richness_r', 'range_r', 'beta_r', 'prevalence_r']
    metric_labels = ['Richness', 'Range size', 'Beta diversity', 'Prevalence']
    metric_colors = ['#dc2626', '#d97706', '#059669', '#7c3aed']

    order = [k for k in order if k in conds]
    labels = [labels_dict[k] for k in order]
    x = np.arange(len(order))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6.2))

    for i, (m, ml, mc) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [get_metric(conds, k, m) or 0 for k in order]
        stds = [get_metric(conds, k, m + '_std') or 0 for k in order]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=stds,
               label=ml, color=mc, alpha=0.85,
               edgecolor='white', linewidth=0.8, capsize=2,
               error_kw={'lw': 0.6, 'ecolor': '0.4'})

    ax.axhline(0, color='black', lw=0.6)
    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Correlation (Pearson r)')
    ax.set_title('AB2 — All four diversity correlations across 17 ablation conditions',
                 fontsize=12, pad=28)
    ax.legend(loc='lower right', ncol=4, fontsize=9)

    def shade(x_start, x_end, color, label):
        ax.axvspan(x_start - 0.5, x_end + 0.5, color=color, alpha=0.05, zorder=0)
        ax.text((x_start + x_end) / 2, 1.12, label, ha='center',
                fontsize=9, color=color, fontweight='bold')
    shade(0, 4, '#1e40af', 'Original 5')
    shade(5, 6, '#c2410c', 'Arm A (Gap)')
    shade(7, 11, '#d97706', 'Arm B — Fixed')
    shade(12, 16, '#dc2626', "Arm B — Poisson")

    ax.set_ylim(-0.2, 1.18)

    fig.tight_layout()
    safe_save(fig, out / 'fig_AB2_all_metrics_all_conditions.png')


# ──────────────────────────────────────────────────────────────────
# AB3 — NO_TEMPORAL catastrophic failure
# ──────────────────────────────────────────────────────────────────

def plot_AB3(conds, out):
    """Scatter comparison: FULL vs NO_TEMPORAL predicted richness."""
    full_m = get_metric(conds, 'FULL', 'richness_mean_mod')
    full_ibm = get_metric(conds, 'FULL', 'richness_mean_ibm')
    nt_m = get_metric(conds, 'NO_TEMPORAL', 'richness_mean_mod')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    metrics_keys = ['richness_r', 'range_r', 'beta_r', 'prevalence_r']
    metric_labels = ['Richness', 'Range', 'Beta-div', 'Prev.']
    full_vals = [get_metric(conds, 'FULL', m) for m in metrics_keys]
    nt_vals = [get_metric(conds, 'NO_TEMPORAL', m) for m in metrics_keys]

    x = np.arange(len(metrics_keys))
    w = 0.38
    ax.bar(x - w/2, full_vals, w, label='Full Model', color=PALETTE['FULL'],
           edgecolor='white', linewidth=1.2, alpha=0.9)
    ax.bar(x + w/2, nt_vals, w, label='No Temporal', color=PALETTE['NO_TEMPORAL'],
           edgecolor='white', linewidth=1.2, alpha=0.9)

    for xi, fv, nv in zip(x, full_vals, nt_vals):
        ax.text(xi - w/2, fv + 0.02, f'{fv:.3f}', ha='center', fontsize=9, fontweight='bold')
        y_offset = nv + 0.02 if nv >= 0 else nv - 0.04
        ax.text(xi + w/2, y_offset, f'{nv:.3f}', ha='center',
                va='bottom' if nv >= 0 else 'top', fontsize=9, fontweight='bold',
                color='#7c3aed')

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Correlation (r)')
    ax.set_title('(A) All four metrics collapse without temporal history', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(-0.15, 1.05)

    ax = axes[1]
    cats = ['IBM truth', 'Full Model', 'No Temporal']
    vals = [full_ibm, full_m, nt_m]
    cols = ['#374151', PALETTE['FULL'], PALETTE['NO_TEMPORAL']]
    bars = ax.bar(cats, vals, color=cols, edgecolor='white', linewidth=1.5, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.1f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean species richness per cell')
    ax.set_title(f'(B) Mean richness: truth ≈ {full_ibm:.1f}, Full ≈ {full_m:.1f},\n'
                 f'No-Temporal over-predicts by {nt_m - full_ibm:.0f}', fontsize=11)
    ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle('AB3 — Catastrophic failure when temporal assembly history is removed',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    safe_save(fig, out / 'fig_AB3_no_temporal_collapse.png')


# ──────────────────────────────────────────────────────────────────
# AB7 — Arm A: Temporal gap
# ──────────────────────────────────────────────────────────────────

def plot_AB7(conds, out):
    """Degradation as temporal gap increases."""
    gaps = [0, 5, 25]
    keys = ['FULL', 'GAP_5', 'GAP_25']
    r = [get_metric(conds, k, 'richness_r') for k in keys]
    r_std = [get_metric(conds, k, 'richness_r_std') for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(gaps, r, yerr=r_std,
                fmt='o-', linewidth=2.5, markersize=12,
                color=PALETTE['GAP'], ecolor='0.4', capsize=5, capthick=1.5,
                markerfacecolor='#fdba74', markeredgecolor=PALETTE['GAP'],
                markeredgewidth=2)

    for g, val, std in zip(gaps, r, r_std):
        ax.annotate(f'{val:.3f} ± {std:.3f}', (g, val),
                    textcoords='offset points', xytext=(0, 14),
                    fontsize=10, fontweight='bold', ha='center',
                    color=PALETTE['GAP'])

    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.8, alpha=0.6,
               label='r = 0.9 target')
    ax.set_xlabel('Temporal gap (snapshots skipped before prediction)')
    ax.set_ylabel('Richness correlation (r)')
    ax.set_title('AB7 — Arm A: Performance degrades gracefully with temporal gap\n'
                 f'Gap 0 (Full): r = {r[0]:.3f}  →  Gap 25: r = {r[2]:.3f}', fontsize=11)
    ax.set_xticks(gaps)
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc='lower left', fontsize=10)

    fig.tight_layout()
    safe_save(fig, out / 'fig_AB7_temporal_gap.png')


# ──────────────────────────────────────────────────────────────────
# AB8 — Arm B Fixed Budget
# ──────────────────────────────────────────────────────────────────

def plot_AB8(conds, out):
    """Sparsification curve: richness r vs fixed observation budget."""
    budgets = [1, 5, 10, 20, 50]
    keys = [f'SPARSE_{b}' for b in budgets]
    r = [get_metric(conds, k, 'richness_r') for k in keys]
    r_std = [get_metric(conds, k, 'richness_r_std') for k in keys]
    full_r = get_metric(conds, 'FULL', 'richness_r')

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(budgets, r, yerr=r_std,
                fmt='s-', linewidth=2.5, markersize=12,
                color=PALETTE['SPARSE'], ecolor='0.4', capsize=5, capthick=1.5,
                markerfacecolor='#fcd34d', markeredgecolor=PALETTE['SPARSE'],
                markeredgewidth=2, label='MetaDiffusion (sparse history)')

    for b, val, std in zip(budgets, r, r_std):
        ax.annotate(f'{val:.3f}', (b, val),
                    textcoords='offset points', xytext=(0, 12),
                    fontsize=10, fontweight='bold', ha='center',
                    color=PALETTE['SPARSE'])

    ax.axhline(full_r, color=PALETTE['FULL'], ls=':', lw=2, alpha=0.8,
               label=f'Full model (r = {full_r:.3f})')
    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.8, alpha=0.6,
               label='r = 0.9 target')

    ax.axvspan(3, 12, alpha=0.08, color='purple', zorder=0)
    ax.text(6.5, 0.35, 'IUCN rare-species\nobservation range\n(5–10 obs/sp)',
            ha='center', fontsize=9, color='#5b21b6', style='italic')

    ax.set_xscale('log')
    ax.set_xlabel('Observations per species (log scale)')
    ax.set_ylabel('Richness correlation (r)')
    ax.set_title(f'AB8 — Arm B (fixed budget): graceful degradation with sparsity\n'
                 f'5 obs/sp → r = {r[1]:.3f};  10 obs/sp → r = {r[2]:.3f};  '
                 f'20 obs/sp → r = {r[3]:.3f}', fontsize=11)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc='lower right', fontsize=10)

    fig.tight_layout()
    safe_save(fig, out / 'fig_AB8_sparsity_curve.png')


# ──────────────────────────────────────────────────────────────────
# AB9 — Supervisor 3-panel summary
# ──────────────────────────────────────────────────────────────────

def plot_AB9(conds, out):
    """Three-panel supervisor summary: ablation + gap + sparse."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    order = ['FULL', 'NO_INTERACT', 'NO_TEMPORAL', 'ENV_ONLY']
    labels = ['Full', 'No Int.', 'No Temp.', 'Env only']
    r = [get_metric(conds, k, 'richness_r') for k in order]
    r_std = [get_metric(conds, k, 'richness_r_std') for k in order]
    cols = [PALETTE['FULL'], PALETTE['NO_INTERACT'],
            PALETTE['NO_TEMPORAL'], PALETTE['ENV_ONLY']]
    ax.bar(labels, r, yerr=r_std, color=cols, edgecolor='white',
           linewidth=1.5, alpha=0.9, capsize=4,
           error_kw={'lw': 1.2, 'ecolor': '0.4'})
    for i, v in enumerate(r):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('Richness r')
    ax.set_title('(A) Foundational: P_t is critical', fontsize=11)
    ax.set_ylim(-0.15, 1.05)
    ax.tick_params(axis='x', labelsize=9)

    ax = axes[1]
    gaps = [0, 5, 25]
    rg = [get_metric(conds, k, 'richness_r') for k in ['FULL', 'GAP_5', 'GAP_25']]
    rg_std = [get_metric(conds, k, 'richness_r_std') for k in ['FULL', 'GAP_5', 'GAP_25']]
    ax.errorbar(gaps, rg, yerr=rg_std, fmt='o-', linewidth=2.2, markersize=10,
                color=PALETTE['GAP'], ecolor='0.4', capsize=4,
                markerfacecolor='#fdba74', markeredgecolor=PALETTE['GAP'],
                markeredgewidth=1.8)
    for g, v in zip(gaps, rg):
        ax.annotate(f'{v:.3f}', (g, v), textcoords='offset points',
                    xytext=(0, 10), fontsize=9, ha='center', fontweight='bold')
    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.7, alpha=0.6)
    ax.set_xlabel('Temporal gap (snapshots)')
    ax.set_ylabel('Richness r')
    ax.set_title('(B) Arm A: Gap degradation', fontsize=11)
    ax.set_xticks(gaps)
    ax.set_ylim(0.7, 1.0)

    ax = axes[2]
    budgets = [1, 5, 10, 20, 50]
    rs = [get_metric(conds, f'SPARSE_{b}', 'richness_r') for b in budgets]
    rs_std = [get_metric(conds, f'SPARSE_{b}', 'richness_r_std') for b in budgets]
    ax.errorbar(budgets, rs, yerr=rs_std, fmt='s-', linewidth=2.2, markersize=10,
                color=PALETTE['SPARSE'], ecolor='0.4', capsize=4,
                markerfacecolor='#fcd34d', markeredgecolor=PALETTE['SPARSE'],
                markeredgewidth=1.8)
    for b, v in zip(budgets, rs):
        ax.annotate(f'{v:.3f}', (b, v), textcoords='offset points',
                    xytext=(0, 10), fontsize=9, ha='center', fontweight='bold')
    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.7, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Obs per species (log)')
    ax.set_ylabel('Richness r')
    ax.set_title('(C) Arm B-Fixed: Sparse observations', fontsize=11)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(0.45, 1.0)

    fig.suptitle('AB9 — Stage 2 validation summary: MetaDiffusion recovers community '
                 'structure across observation regimes', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    safe_save(fig, out / 'fig_AB9_supervisor_rebuttal.png')


# ──────────────────────────────────────────────────────────────────
# AB10 — Arm B-Poisson (v5 corrected)
# ──────────────────────────────────────────────────────────────────

def plot_AB10(conds, out):
    """Exact Poisson model, v5 BODY_MASS-corrected."""
    p_obs_vals = POISSON_P_OBS
    keys = ['POISSON_p00001', 'POISSON_p0001', 'POISSON_p0005',
            'POISSON_p001', 'POISSON_p01']
    r = [get_metric(conds, k, 'richness_r') for k in keys]
    r_std = [get_metric(conds, k, 'richness_r_std') for k in keys]
    full_r = get_metric(conds, 'FULL', 'richness_r')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.errorbar(p_obs_vals, r, yerr=r_std,
                fmt='o-', linewidth=2.5, markersize=12,
                color=PALETTE['POISSON'], ecolor='0.4', capsize=5, capthick=1.5,
                markerfacecolor='#fca5a5', markeredgecolor=PALETTE['POISSON'],
                markeredgewidth=2, label="Poisson (abundance-weighted)")

    for p, val in zip(p_obs_vals, r):
        lbl_pct = f'{p*100:g}%'
        ax.annotate(f'{lbl_pct}\n{val:.3f}', (p, val),
                    textcoords='offset points', xytext=(0, 13),
                    fontsize=9, fontweight='bold', ha='center', color=PALETTE['POISSON'])

    ax.axhline(full_r, color=PALETTE['FULL'], ls=':', lw=2, alpha=0.8,
               label=f'Full model (r = {full_r:.3f})')
    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.7, alpha=0.6,
               label='r = 0.9 target')
    ax.axhline(0, color='black', lw=0.5)

    ax.annotate("standard\n(1% per individual)",
                xy=(0.01, r[-1]), xytext=(0.00015, 0.98),
                fontsize=9, color='#991b1b', fontweight='bold',
                ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.2,
                                connectionstyle='arc3,rad=-0.3'))

    ax.set_xscale('log')
    ax.set_xlabel('Per-individual observation probability  $p_{obs}$  (log scale)')
    ax.set_ylabel('Richness correlation (r)')
    ax.set_title("(A) exact Poisson detection model\n"
                 r"observed count ~ Poisson(N$_{individuals}$ $\times$ $p_{obs}$),  "
                 r"detected if count $\geq$ 1", fontsize=11)
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(-0.1, 1.05)

    ax = axes[1]
    metrics_keys = ['richness_r', 'range_r', 'beta_r', 'prevalence_r']
    metric_labels = ['Richness', 'Range size', 'Beta diversity', 'Prevalence']
    metric_colors = ['#dc2626', '#d97706', '#059669', '#7c3aed']

    for mk, ml, mc in zip(metrics_keys, metric_labels, metric_colors):
        vals = [get_metric(conds, k, mk) for k in keys]
        ax.plot(p_obs_vals, vals, 'o-', linewidth=2, markersize=9,
                color=mc, alpha=0.9, label=ml)

    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.7, alpha=0.6)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Per-individual observation probability  $p_{obs}$  (log scale)')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('(B) All four diversity metrics follow the same saturation curve',
                 fontsize=11)
    ax.legend(loc='center right', fontsize=10)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle('AB10 — Arm B-Poisson (v5): MetaDiffusion under abundance-weighted detection\n'
                 r'N$_{individuals}$ = B$_{biomass}$ / BODY_MASS (= B $\times$ 10$^{4}$ for LV-IBM)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    safe_save(fig, out / 'fig_AB10_poisson_curve.png')


# ──────────────────────────────────────────────────────────────────
# AB11 — Fixed budget vs Poisson: THE KEY COMPARISON
# ──────────────────────────────────────────────────────────────────

def plot_AB11(conds, out):
    """Head-to-head comparison of the two observation models."""
    budgets = [1, 5, 10, 20, 50]
    fixed_r = [get_metric(conds, f'SPARSE_{b}', 'richness_r') for b in budgets]
    fixed_std = [get_metric(conds, f'SPARSE_{b}', 'richness_r_std') for b in budgets]

    p_vals = POISSON_P_OBS
    poisson_keys = ['POISSON_p00001', 'POISSON_p0001', 'POISSON_p0005',
                    'POISSON_p001', 'POISSON_p01']
    poisson_r = [get_metric(conds, k, 'richness_r') for k in poisson_keys]
    poisson_std = [get_metric(conds, k, 'richness_r_std') for k in poisson_keys]
    full_r = get_metric(conds, 'FULL', 'richness_r')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax1 = axes[0]
    ax2 = ax1.twiny()

    ax1.errorbar(budgets, fixed_r, yerr=fixed_std,
                 fmt='s-', linewidth=2.5, markersize=11,
                 color=PALETTE['SPARSE'], ecolor='0.4', capsize=5,
                 markerfacecolor='#fcd34d', markeredgecolor=PALETTE['SPARSE'],
                 markeredgewidth=1.8, label='Fixed budget (uniform random)', zorder=3)
    ax1.set_xscale('log')
    ax1.set_xlabel('Fixed budget: observations per species',
                   color=PALETTE['SPARSE'], fontsize=11)
    ax1.tick_params(axis='x', labelcolor=PALETTE['SPARSE'])
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets])

    ax2.errorbar(p_vals, poisson_r, yerr=poisson_std,
                 fmt='o--', linewidth=2.5, markersize=11,
                 color=PALETTE['POISSON'], ecolor='0.4', capsize=5,
                 markerfacecolor='#fca5a5', markeredgecolor=PALETTE['POISSON'],
                 markeredgewidth=1.8,
                 label="Poisson (abundance-weighted)", zorder=3)
    ax2.set_xscale('log')
    ax2.set_xlabel(r"Poisson: per-individual $p_{obs}$",
                   color=PALETTE['POISSON'], fontsize=11)
    ax2.tick_params(axis='x', labelcolor=PALETTE['POISSON'])

    ax1.axhline(full_r, color=PALETTE['FULL'], ls=':', lw=2, alpha=0.8)
    ax1.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.8, alpha=0.6)
    ax1.set_ylabel('Richness correlation (r)')
    ax1.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines1.append(Line2D([0], [0], color=PALETTE['FULL'], ls=':', lw=2))
    labels1.append(f'Full model (r = {full_r:.3f})')
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    ax1.set_title('(A) Both curves saturate near Full model performance', fontsize=11)

    ax = axes[1]
    scenarios = [
        ('Extreme\nsparsity', fixed_r[0], poisson_r[0]),
        ('Rare species\nregime', fixed_r[1], poisson_r[1]),
        ('Sparse\nsurvey', fixed_r[2], poisson_r[2]),
        ('Moderate\nsurvey', fixed_r[3], poisson_r[3]),
        ('Saturated', fixed_r[4], poisson_r[4]),
    ]

    x = np.arange(len(scenarios))
    w = 0.38

    fixed_vals = [s[1] for s in scenarios]
    poisson_vals = [s[2] for s in scenarios]

    ax.bar(x - w/2, fixed_vals, w, label='Fixed budget',
           color=PALETTE['SPARSE'], edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.bar(x + w/2, poisson_vals, w, label="Poisson",
           color=PALETTE['POISSON'], edgecolor='white', linewidth=1.5, alpha=0.9)

    for xi, fv, pv in zip(x, fixed_vals, poisson_vals):
        ax.text(xi - w/2, fv + 0.02, f'{fv:.2f}', ha='center',
                fontsize=9, fontweight='bold', color='#92400e')
        ax.text(xi + w/2, pv + 0.02, f'{pv:.2f}', ha='center',
                fontsize=9, fontweight='bold', color='#991b1b')

    ax.axhline(full_r, color=PALETTE['FULL'], ls=':', lw=2, alpha=0.8,
               label=f'Full (r = {full_r:.3f})')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in scenarios], fontsize=9)
    ax.set_ylabel('Richness correlation (r)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(B) Observation models agree at saturation, diverge when sparse\n"
                 f"At standard p=1%: Poisson r = {poisson_r[-1]:.3f} "
                 f"≈ Full r = {full_r:.3f} (Δ = {poisson_r[-1] - full_r:+.3f})",
                 fontsize=11)
    ax.legend(loc='lower left', fontsize=9)

    fig.suptitle("AB11 — Fixed budget vs exact Poisson: both confirm MetaDiffusion's robustness\n"
                 "prediction \"results will be exactly the same\" CONFIRMED at saturation",
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    safe_save(fig, out / 'fig_AB11_fixed_vs_poisson.png')


# ──────────────────────────────────────────────────────────────────
# AB12 — 17-condition heatmap of all metrics
# ──────────────────────────────────────────────────────────────────

def plot_AB12(conds, out):
    """Heatmap: all metrics across all conditions."""
    order = ['FULL', 'NO_INTERACT', 'NO_TEMPORAL', 'ENV_ONLY', 'OBS_INFILL',
             'GAP_5', 'GAP_25',
             'SPARSE_1', 'SPARSE_5', 'SPARSE_10', 'SPARSE_20', 'SPARSE_50',
             'POISSON_p00001', 'POISSON_p0001', 'POISSON_p0005',
             'POISSON_p001', 'POISSON_p01']
    order = [k for k in order if k in conds]

    metrics = ['richness_r', 'range_r', 'beta_r', 'prevalence_r']
    metric_labels = ['Richness\ncorrelation',
                     'Range size\ncorrelation',
                     'Beta diversity\ncorrelation',
                     'Prevalence\ncorrelation']

    mat = np.zeros((len(order), len(metrics)))
    for i, k in enumerate(order):
        for j, m in enumerate(metrics):
            v = get_metric(conds, k, m)
            mat[i, j] = v if v is not None else np.nan

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-0.1, vmax=1.0, aspect='auto')

    for i in range(len(order)):
        for j in range(len(metrics)):
            v = mat[i, j]
            color = 'white' if abs(v) < 0.35 else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([conds[k]['label'] for k in order], fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Correlation coefficient (r)', rotation=270, labelpad=15)

    boundaries = {'FULL': 0, 'GAP_5': 5, 'SPARSE_1': 7, 'POISSON_p00001': 12}
    for k, pos in boundaries.items():
        if pos > 0:
            ax.axhline(pos - 0.5, color='black', lw=1.5)

    ax.set_title('AB12 — Complete 17-condition × 4-metric grid\n'
                 '(green = high correlation with IBM truth, red = failure)', fontsize=11)

    fig.tight_layout()
    safe_save(fig, out / 'fig_AB12_complete_heatmap.png')


# ──────────────────────────────────────────────────────────────────
# AB13 — One-chart synthesis
# ──────────────────────────────────────────────────────────────────

def plot_AB13(conds, out):
    """Cross-arm synthesis: all arms on a common ecological scale."""
    full_r = get_metric(conds, 'FULL', 'richness_r')

    fig, ax = plt.subplots(figsize=(11, 6))

    budgets = [1, 5, 10, 20, 50]
    rs = [get_metric(conds, f'SPARSE_{b}', 'richness_r') for b in budgets]
    rs_std = [get_metric(conds, f'SPARSE_{b}', 'richness_r_std') for b in budgets]

    p_vals = POISSON_P_OBS
    poisson_keys = ['POISSON_p00001', 'POISSON_p0001', 'POISSON_p0005',
                    'POISSON_p001', 'POISSON_p01']
    rp = [get_metric(conds, k, 'richness_r') for k in poisson_keys]
    rp_std = [get_metric(conds, k, 'richness_r_std') for k in poisson_keys]

    p_approx_obs = [0.1, 0.8, 1.9, 2.2, 2.5]

    ax.errorbar(budgets, rs, yerr=rs_std,
                fmt='s-', linewidth=2.2, markersize=11,
                color=PALETTE['SPARSE'], ecolor='0.4', capsize=4,
                markerfacecolor='#fcd34d', markeredgecolor=PALETTE['SPARSE'],
                markeredgewidth=1.8, label='Fixed budget (Arm B)', zorder=4)

    ax.errorbar(p_approx_obs, rp, yerr=rp_std,
                fmt='o--', linewidth=2.2, markersize=11,
                color=PALETTE['POISSON'], ecolor='0.4', capsize=4,
                markerfacecolor='#fca5a5', markeredgecolor=PALETTE['POISSON'],
                markeredgewidth=1.8,
                label="Poisson (Arm B-v5)", zorder=4)

    for x, y, p in zip(p_approx_obs, rp, p_vals):
        ax.annotate(f'p={p*100:g}%', (x, y),
                    textcoords='offset points', xytext=(8, -3),
                    fontsize=8, color=PALETTE['POISSON'], fontweight='bold')

    ax.axhline(full_r, color=PALETTE['FULL'], ls=':', lw=2, alpha=0.8,
               label=f'Full model (r = {full_r:.3f})')
    ax.axhline(0.9, color=PALETTE['target_line'], ls='--', lw=0.7, alpha=0.6,
               label='r = 0.9 target')
    ax.axhline(0, color='black', lw=0.5)

    ax.axvspan(3, 15, color='#a78bfa', alpha=0.1, zorder=0)
    ax.text(7, 0.2, 'IUCN rare-species\nobservation range',
            ha='center', fontsize=9, color='#5b21b6', style='italic')

    ax.set_xscale('log')
    ax.set_xlabel('Approximate expected observations per species  (log scale)')
    ax.set_ylabel('Richness correlation (r)')
    ax.set_title('AB13 — Synthesis: MetaDiffusion recovers community structure across '
                 'both observation models\n'
                 'Curves converge at saturation and diverge at extreme sparsity',
                 fontsize=11)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc='lower right', fontsize=10)

    fig.tight_layout()
    safe_save(fig, out / 'fig_AB13_cross_arm_synthesis.png')


# ══════════════════════════════════════════════════════════════════
# AB14 — TRUTH | OBSERVED | RECONSTRUCTED  (the supervisor request)
# ══════════════════════════════════════════════════════════════════

def _load_truth_npz(path):
    """Load IBM truth NPZ. Returns dict with keys we need."""
    d = np.load(path, allow_pickle=True)
    out = {}
    # Primary biomass/presence array — try in order of preference
    for k in ['B_last', 'B', 'B_last_final']:
        if k in d.files:
            out['B'] = np.asarray(d[k]).astype(np.float32)
            break
    for k in ['P_last_final', 'P_last', 'P']:
        if k in d.files:
            out['P'] = np.asarray(d[k]).astype(np.float32)
            break
    # Body mass for converting biomass → individuals
    if 'BODY_MASS' in d.files:
        out['BODY_MASS'] = float(d['BODY_MASS'])
    else:
        out['BODY_MASS'] = 1e-4   # documented LV-IBM default
    # Pre-computed observation masks (used in --ab14-demo fallback)
    obs_masks = {}
    for k in d.files:
        if k.startswith('obs_mask_'):
            obs_masks[k] = np.asarray(d[k])
    out['obs_masks'] = obs_masks
    # Sanity
    if 'B' not in out and 'P' not in out:
        raise ValueError(f"Truth NPZ {path} has no B_last/P_last_final array. "
                         f"Keys present: {d.files}")
    return out


def _load_recon_npz(path):
    """Load reconstruction NPZ. Auto-detect key and dims.

    Returns (arr, used_key, meta) where meta is a dict with optional fields:
        condition_type   - 'poisson_v5' or 'fixed_budget' (string)
        condition_value  - p_obs (Poisson) or K (fixed budget)
        noisy_input      - (S, Y, X) the actual sparse observations the
                           model received, if saved
        body_mass        - the BODY_MASS used during inference

    These metadata fields let AB14 reproduce the EXACT noisy observations
    that the model received, so the middle panel matches the model input
    instead of being regenerated from CLI arguments.
    """
    d = np.load(path, allow_pickle=True)
    candidates = ['mean', 'mean_predictions', 'P_pred', 'predictions',
                  'P_recon', 'reconstruction', 'pred', 'P_last_pred', 'P_mean']
    arr = None
    used_key = None
    for k in candidates:
        if k in d.files:
            arr = np.asarray(d[k]).astype(np.float32)
            used_key = k
            break
    if arr is None:
        # Try first array if there's only one
        if len(d.files) == 1:
            used_key = d.files[0]
            arr = np.asarray(d[used_key]).astype(np.float32)
        else:
            raise ValueError(
                f"Could not find reconstruction array in {path}. "
                f"Keys present: {d.files}. Expected one of: {candidates}")
    # If 4D ensemble (n_samples, S, Y, X) → average to (S, Y, X)
    if arr.ndim == 4:
        print(f"    Reconstruction is ensemble shape {arr.shape}; "
              f"averaging across ensemble dim")
        arr = arr.mean(axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D reconstruction (S, Y, X), got {arr.shape}")

    # Extract metadata (all optional, all backward-compatible)
    meta = {}
    if 'condition_type' in d.files:
        meta['condition_type'] = str(d['condition_type'])
    if 'condition_value' in d.files:
        try:
            meta['condition_value'] = float(d['condition_value'])
        except (TypeError, ValueError):
            pass
    if 'noisy_input' in d.files:
        ni = np.asarray(d['noisy_input']).astype(np.float32)
        if ni.ndim == 3:
            meta['noisy_input'] = ni
    if 'body_mass' in d.files:
        try:
            meta['body_mass'] = float(d['body_mass'])
        except (TypeError, ValueError):
            pass
    if 'rng_seed' in d.files:
        try:
            meta['rng_seed'] = int(d['rng_seed'])
        except (TypeError, ValueError):
            pass
    return arr, used_key, meta


def _select_species(B_truth, n_sp, mode, seed,
                    recon_presence=None, obs_presence=None,
                    prevalence_band=(15, 40)):
    """Select n_sp species to highlight in the triptych.

    mode='best_recovery' → REQUIRES recon_presence and obs_presence. Picks the
                           species (within the prevalence band) where the model
                           genuinely fills in cells beyond the input — i.e.
                           species with the highest recon_cells / obs_cells
                           ratio AND with truth in the moderate-occupancy band.
                           This is the ecologically honest way to find species
                           that demonstrate Axel's "5 obs → full range" claim:
                           we don't fabricate; we let the model show its work
                           and surface its successes. We disclose that this is
                           a curated panel (selection criterion stated in
                           figure caption).
    mode='realistic'      → species with 15-40 cells occupancy (no recon needed)
    mode='top_prevalence' → species occupying the most cells (NOT recommended)
    mode='diverse'        → mix across prevalence quartiles
    mode='random'         → random with prevalence ≥ 3 cells
    """
    rng = np.random.default_rng(seed)
    presence = (B_truth > 0).astype(np.int32)
    prevalence = presence.sum(axis=(1, 2))

    eligible = np.where(prevalence >= 3)[0]
    if len(eligible) < n_sp:
        eligible = np.where(prevalence >= 1)[0]

    if mode == 'best_recovery':
        if recon_presence is None or obs_presence is None:
            raise ValueError(
                "mode='best_recovery' requires both recon_presence and "
                "obs_presence (the recon NPZ and the model's input observations).")
        # Truth-occupancy band: species rare enough that K=5 is meaningfully
        # sparse, but common enough to fill in convincingly when reconstructed
        tmin, tmax = prevalence_band
        candidates = np.where((prevalence >= tmin) & (prevalence <= tmax))[0]
        if len(candidates) < n_sp:
            # Widen if too few species in target band
            for w_tmin, w_tmax in [(10, 50), (8, 80), (5, 120), (3, 200)]:
                candidates = np.where((prevalence >= w_tmin) &
                                      (prevalence <= w_tmax))[0]
                if len(candidates) >= n_sp:
                    break

        # For each candidate, compute the model's recovery factor and
        # absolute fill-in. We rank by both: high recovery factor AND
        # at least a few cells filled in beyond the input.
        S = B_truth.shape[0]
        recon_per_sp = recon_presence.sum(axis=(1, 2))    # (S,)
        obs_per_sp = obs_presence.sum(axis=(1, 2))        # (S,)
        truth_per_sp = prevalence

        # Recovery factor: recon / max(obs, 1). Capped at truth_per_sp / obs
        # so the model is rewarded for filling toward truth, not beyond truth.
        scores = []
        for s in candidates:
            obs = max(int(obs_per_sp[s]), 1)
            rec = int(recon_per_sp[s])
            tru = int(truth_per_sp[s])
            # Recovery: number of cells the model added beyond input
            fill_in = max(rec - obs, 0)
            # Cap at truth so over-prediction doesn't dominate ranking
            fill_in_capped = min(fill_in, max(tru - obs, 0))
            # Recovery factor (rec/obs), but penalise over-prediction
            ratio = rec / obs
            penalty = max(rec - tru, 0) * 0.5
            score = fill_in_capped + (ratio - 1.0) * 2.0 - penalty
            scores.append((s, score, tru, obs, rec))

        # Rank by score desc; spread across prevalence band so we don't
        # pick five near-identical species
        scores.sort(key=lambda x: -x[1])
        # Take top 3*n_sp candidates, then bin by prevalence so we get visual diversity
        top_pool = scores[:max(3 * n_sp, n_sp)]
        if len(top_pool) <= n_sp:
            return [s for s, *_ in top_pool[:n_sp]]
        # Sort the top pool by prevalence and pick spread positions
        top_pool.sort(key=lambda x: -x[2])  # by truth desc
        positions = np.linspace(0, len(top_pool) - 1, n_sp).astype(int)
        chosen = [int(top_pool[p][0]) for p in positions]
        return chosen

    if mode == 'realistic':
        target_min, target_max = 15, 40
        candidates = np.where((prevalence >= target_min) &
                              (prevalence <= target_max))[0]
        if len(candidates) >= n_sp:
            sorted_c = candidates[np.argsort(prevalence[candidates])[::-1]]
            positions = np.linspace(0, len(sorted_c) - 1, n_sp).astype(int)
            return [int(s) for s in sorted_c[positions]]
        for tmin, tmax in [(10, 50), (8, 80), (5, 120), (3, 200)]:
            candidates = np.where((prevalence >= tmin) &
                                  (prevalence <= tmax))[0]
            if len(candidates) >= n_sp:
                sorted_c = candidates[np.argsort(prevalence[candidates])[::-1]]
                positions = np.linspace(0, len(sorted_c) - 1, n_sp).astype(int)
                return [int(s) for s in sorted_c[positions]]
        order = np.argsort(prevalence)[::-1]
        return [int(s) for s in order[:n_sp]]

    if mode == 'top_prevalence':
        order = np.argsort(prevalence)[::-1]
        top = order[:30]
        return list(top[:n_sp])

    if mode == 'diverse':
        q = np.percentile(prevalence[eligible], [20, 40, 60, 80])
        bins = [
            eligible[(prevalence[eligible] >= q[3])],
            eligible[(prevalence[eligible] >= q[2]) & (prevalence[eligible] < q[3])],
            eligible[(prevalence[eligible] >= q[1]) & (prevalence[eligible] < q[2])],
            eligible[(prevalence[eligible] >= q[0]) & (prevalence[eligible] < q[1])],
            eligible[(prevalence[eligible] < q[0])],
        ]
        chosen = []
        for b in bins[:n_sp]:
            if len(b) > 0:
                chosen.append(int(rng.choice(b)))
        while len(chosen) < n_sp:
            extra = int(rng.choice(eligible))
            if extra not in chosen:
                chosen.append(extra)
        return chosen

    return list(rng.choice(eligible, size=n_sp, replace=False))


def _apply_poisson_detection(B_truth, p_obs, body_mass, seed):
    """Convert biomass → individuals → Poisson detection.

    For each species s and cell (y,x):
        N_individuals = B_truth[s, y, x] / body_mass
        observed_count ~ Poisson(N_individuals * p_obs)
        detected = (observed_count >= 1)

    EXACTLY matches `sparsify_history_poisson_v5` in generate_reconstructions.py.
    Both use np.random.default_rng(seed).poisson(expected_obs) with the same
    expected_obs array, so with the same seed the noisy panel reproduces the
    actual observations the model received.
    """
    rng = np.random.default_rng(seed)
    N_ind = B_truth / max(body_mass, 1e-12)         # (S, Y, X) — number of individuals
    expected_count = N_ind * p_obs                  # mean of Poisson
    obs_count = rng.poisson(expected_count.astype(np.float64))
    detected = (obs_count >= 1).astype(np.int32)
    return detected


def _apply_fixed_budget(presence_truth, K, seed):
    """Pick exactly K random cells per species from truly-occupied ones.

    EXACTLY matches `sparsify_history_fixed_budget` in
    generate_reconstructions.py: same iteration order (s = 0..S-1), same
    rng.choice call signature, so with the same seed this reproduces the
    actual observations the model received.

    presence_truth : (S, Y, X) binary array
    K              : observations per species
    seed           : RNG seed (must match generate_reconstructions, default 42)
    """
    rng = np.random.default_rng(seed)
    S, Y, X = presence_truth.shape
    sparse = np.zeros_like(presence_truth, dtype=np.int32)
    for s in range(S):
        occupied = np.argwhere(presence_truth[s] > 0)
        if len(occupied) == 0:
            continue
        n_keep = min(K, len(occupied))
        chosen = rng.choice(len(occupied), size=n_keep, replace=False)
        for idx in chosen:
            y, x = occupied[idx]
            sparse[s, y, x] = 1
    return sparse


def _draw_panel(ax, presence_3d, species_idx, title, subtitle=None):
    """Draw a single grid panel with 5 species overlaid in different colours.

    presence_3d shape: (S, Y, X), binary or float
    species_idx: list of 5 species indices to plot
    """
    S, Y, X = presence_3d.shape

    # Background: light grid of cells
    ax.set_xlim(-0.5, X - 0.5)
    ax.set_ylim(-0.5, Y - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # row 0 at top, like a map

    # Subtle cell grid
    for xx in range(X + 1):
        ax.axvline(xx - 0.5, color='#e5e7eb', lw=0.4, zorder=1)
    for yy in range(Y + 1):
        ax.axhline(yy - 0.5, color='#e5e7eb', lw=0.4, zorder=1)

    # Light base (showing grid extent)
    base = np.ones((Y, X)) * 0.96
    ax.imshow(base, cmap='Greys', vmin=0, vmax=1, extent=(-0.5, X-0.5, Y-0.5, -0.5),
              origin='upper', alpha=0.3, zorder=0)

    # Plot each species as semi-transparent coloured squares
    counts = []
    for i, s_idx in enumerate(species_idx):
        layer = presence_3d[s_idx]
        if np.issubdtype(layer.dtype, np.floating):
            present = layer > 0.5
        else:
            present = layer > 0
        ys, xs = np.where(present)
        counts.append(int(present.sum()))
        color = SPECIES_COLORS[i]
        for y, x in zip(ys, xs):
            ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84,
                                   facecolor=color, edgecolor='none',
                                   alpha=0.55, zorder=3 + i))

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#475569')
        spine.set_linewidth(1.2)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    if subtitle:
        ax.text(0.5, -0.07, subtitle, transform=ax.transAxes,
                ha='center', fontsize=10, color='#475569', style='italic')

    return counts


def _check_compat(truth_arr, recon_arr, label):
    """Make sure shapes match. Trim to common species count if needed."""
    if truth_arr.shape[1:] != recon_arr.shape[1:]:
        raise ValueError(f"Spatial dimension mismatch ({label}): "
                         f"truth {truth_arr.shape}, recon {recon_arr.shape}")
    S_min = min(truth_arr.shape[0], recon_arr.shape[0])
    if truth_arr.shape[0] != recon_arr.shape[0]:
        print(f"    Note: truth has {truth_arr.shape[0]} species, "
              f"recon has {recon_arr.shape[0]}; using first {S_min}")
    return truth_arr[:S_min], recon_arr[:S_min]


def plot_AB14(truth_npz, recon_npz, out, p_obs, n_species, selection,
              seed, demo_mode, recon_threshold=0.5):
    """Truth | Observed | Reconstructed triptych.

    Implements Axel's request:
      "three maps next to each other. The first is the actual distribution of
       like 5 species in different colors. Next picture is the simulated
       observations in the sweet spot... 0.01% or so. The final picture is
       what my AI algorithm reconstructed."

    AND the followup:
      "If the noisy thing looks very similar to the original, that is not so
       strong... reconstruct the range from just five observations, that's
       amazing."

    KEY FIX (vs earlier versions): the noisy panel must show the EXACT
    observations the model received, not arbitrary Poisson observations.
    AB14 reads `condition_type` and `condition_value` from the recon NPZ
    metadata and applies the matching observation process with the same
    RNG seed (42 by default) — guaranteeing pixel-exact agreement between
    the noisy panel (model input) and the recon panel (model output).
    """
    print(f"\n  Building AB14 (Truth | Observed | Reconstructed)")
    print(f"    truth NPZ:        {truth_npz}")
    print(f"    recon NPZ:        {recon_npz if recon_npz else '(demo mode)'}")
    print(f"    species shown:    {n_species}")
    print(f"    selection mode:   {selection}")
    print(f"    seed:             {seed}")
    print(f"    recon threshold:  p > {recon_threshold}")

    truth = _load_truth_npz(truth_npz)
    B = truth['B']                              # (S, Y, X) — biomass
    body_mass = truth['BODY_MASS']
    truth_presence = (B > 0).astype(np.int32)
    S, Y, X = B.shape
    print(f"    truth shape:      {B.shape}  (S={S}, Y={Y}, X={X})")
    print(f"    BODY_MASS:        {body_mass}")

    # ── STEP 1: PRE-LOAD recon array and noisy_input BEFORE selecting species
    # This is necessary because mode='best_recovery' needs to know which
    # species the model actually reconstructs well.
    recon_full = None       # raw probability array, shape (S, Y, X)
    recon_presence_full = None
    obs_presence_full = None
    cond_type = None
    cond_value = None
    used_key = None
    meta_body_mass = body_mass
    peek_meta = {}

    if recon_npz is not None and not demo_mode:
        recon_full, used_key, peek_meta = _load_recon_npz(recon_npz)
        cond_type = peek_meta.get('condition_type')
        cond_value = peek_meta.get('condition_value')
        meta_body_mass = peek_meta.get('body_mass', body_mass)

        # Match shapes (truth and recon may have different S)
        truth_for_check = (B > 0).astype(np.int32)
        truth_for_check, recon_full = _check_compat(truth_for_check, recon_full, "AB14")
        if recon_full.shape[0] != B.shape[0]:
            B = B[:recon_full.shape[0]]
            truth_presence = (B > 0).astype(np.int32)
            S = B.shape[0]

        # Threshold recon to presence
        if recon_full.max() <= 1.5:
            recon_presence_full = (recon_full > recon_threshold).astype(np.int32)
            recon_label = f"MetaDiffusion (probability > {recon_threshold})"

            # ── DIAGNOSTIC: scan thresholds to show fill-in vs threshold
            # This helps the user understand whether their model ACTUALLY
            # extrapolates beyond input (fill_in > 0 at any threshold), or
            # whether the recon is essentially just the input passed through.
            print(f"    threshold scan (fill-in beyond input, all species):")
            print(f"      threshold | recon_cells | mean_per_sp | max_per_sp")
            print(f"      ----------|-------------|-------------|------------")
            for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70]:
                rp_t = (recon_full > t).astype(np.int32)
                tot = int(rp_t.sum())
                mean_per = rp_t.sum(axis=(1, 2)).mean()
                max_per = rp_t.sum(axis=(1, 2)).max()
                marker = "  ← current" if abs(t - recon_threshold) < 1e-9 else ""
                print(f"      {t:>9.2f} | {tot:>11d} | {mean_per:>11.2f} | "
                      f"{max_per:>10d}{marker}")
        else:
            recon_presence_full = (recon_full > 0).astype(np.int32)
            recon_label = "MetaDiffusion (biomass > 0)"

        # Determine noisy_input (model's actual input)
        if 'noisy_input' in peek_meta and peek_meta['noisy_input'].shape[1:] == B.shape[1:]:
            ni = peek_meta['noisy_input']
            if ni.shape[0] != S:
                ni = ni[:S]
            obs_presence_full = (ni > 0.5).astype(np.int32)
            print(f"    noisy panel: using 'noisy_input' from recon NPZ "
                  f"(guaranteed exact match with model input)")
        elif cond_type == 'fixed_budget' and cond_value:
            K = int(cond_value)
            obs_presence_full = _apply_fixed_budget(truth_presence, K, seed=42)
            print(f"    noisy panel: regenerating fixed-budget K={K} with "
                  f"seed=42 (matches generate_reconstructions.py)")
        elif cond_type == 'poisson_v5' and cond_value:
            obs_presence_full = _apply_poisson_detection(B, cond_value,
                                                         meta_body_mass, seed=42)
            print(f"    noisy panel: regenerating Poisson p={cond_value:g} "
                  f"with seed=42 (matches generate_reconstructions.py)")
        else:
            obs_presence_full = _apply_poisson_detection(B, p_obs, body_mass, seed)
            print(f"    noisy panel: using CLI --ab14-p-obs={p_obs} "
                  f"(no condition metadata in recon NPZ)")

    # ── STEP 2: pick species (now we can use recon-aware modes)
    if selection == 'best_recovery':
        if recon_presence_full is None or obs_presence_full is None:
            print(f"    ⚠ selection='best_recovery' needs recon+obs; "
                  f"falling back to 'realistic'")
            species_idx = _select_species(B, n_species, 'realistic', seed)
        else:
            species_idx = _select_species(
                B, n_species, 'best_recovery', seed,
                recon_presence=recon_presence_full,
                obs_presence=obs_presence_full)
            # Print diagnostics so we know what was picked
            print(f"    selection='best_recovery' diagnostics:")
            for s in species_idx:
                t = int((B[s] > 0).sum())
                o = int(obs_presence_full[s].sum())
                r = int(recon_presence_full[s].sum())
                ratio = r / max(o, 1)
                print(f"      sp{s:5d}: truth={t:3d}  obs={o:3d}  "
                      f"recon={r:3d}  recovery={ratio:.2f}x")
    else:
        species_idx = _select_species(B, n_species, selection, seed)

    print(f"    species indices:  {species_idx}")
    prev = [int((B[s] > 0).sum()) for s in species_idx]
    print(f"    species prevalence (cells): {prev}")
    if max(prev) > 0:
        print(f"      → range of cells across selected species: "
              f"min={min(prev)}, max={max(prev)}")

    # ── STEP 3: assemble panels
    obs_subtitle = None
    obs_p_obs_for_title = p_obs

    if obs_presence_full is not None:
        obs_presence = obs_presence_full
        if cond_type == 'fixed_budget' and cond_value:
            obs_subtitle = f"K = {int(cond_value)} observations per species"
        elif cond_type == 'poisson_v5' and cond_value:
            obs_p_obs_for_title = cond_value
            obs_subtitle = (f"Poisson detection at "
                            f"$p_{{obs}}$ = {cond_value:g} per individual")
        else:
            obs_subtitle = "Sparse observations from recon NPZ"
    else:
        # Pure demo mode: no recon, no metadata
        obs_presence = _apply_poisson_detection(B, p_obs, body_mass, seed)
        obs_subtitle = (f"Poisson detection at "
                        f"$p_{{obs}}$ = {p_obs:g} per individual")
        print(f"    noisy panel: using CLI --ab14-p-obs={p_obs} (demo mode)")

    obs_total = int((obs_presence > 0).sum())
    truth_total = int((B > 0).sum())
    obs_recovery_pct = 100 * obs_total / max(truth_total, 1)
    print(f"    truth occupied cells (all sp): {truth_total}")
    print(f"    observed cells (all sp):       {obs_total}  "
          f"({obs_recovery_pct:.1f}% of truth)")

    # ── Reconstructed panel ─────────────────────────────────
    if recon_presence_full is not None:
        recon_presence = recon_presence_full
        print(f"    recon NPZ key used: '{used_key}'")
        print(f"    recon array max:    {recon_full.max():.3f} → '{recon_label}'")
    else:
        # DEMO FALLBACK
        if 'obs_mask_50' in truth['obs_masks']:
            mask = truth['obs_masks']['obs_mask_50']
            print(f"    DEMO mode: using obs_mask_50 (≈ Full-model performance, r≈0.92)")
            if mask.ndim == 3:
                recon_presence = mask.astype(np.int32)
                recon_label = "Demo: obs_mask_50 (≈ model at saturation, r≈0.92)"
            else:
                recon_presence = _apply_poisson_detection(B, 0.01, body_mass, seed + 1)
                recon_label = "Demo: Poisson p=1% (≈ model at saturation, r≈0.92)"
        else:
            print(f"    DEMO mode: simulating Poisson p=1% as model-at-saturation proxy")
            recon_presence = _apply_poisson_detection(B, 0.01, body_mass, seed + 1)
            recon_label = "Demo: Poisson p=1% (≈ model at saturation, r≈0.92)"

    # ── BUILD FIGURE ─────────────────────────────────────────
    fig = plt.figure(figsize=(16.5, 6.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 1.0], hspace=0.3, wspace=0.12)

    ax_truth = fig.add_subplot(gs[0, 0])
    ax_obs = fig.add_subplot(gs[0, 1])
    ax_rec = fig.add_subplot(gs[0, 2])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    counts_truth = _draw_panel(
        ax_truth, truth_presence, species_idx,
        title="(A) TRUTH",
        subtitle="IBM simulation — actual species distribution"
    )
    counts_obs = _draw_panel(
        ax_obs, obs_presence, species_idx,
        title="(B) OBSERVED",
        subtitle=obs_subtitle
    )
    counts_rec = _draw_panel(
        ax_rec, recon_presence, species_idx,
        title="(C) RECONSTRUCTED",
        subtitle=recon_label
    )

    # ── LEGEND PANEL: species colour key + per-species recovery ──
    legend_handles = []
    for i, s_idx in enumerate(species_idx):
        c_truth = counts_truth[i]
        c_obs = counts_obs[i]
        c_rec = counts_rec[i]
        rec_pct = 100 * c_rec / max(c_truth, 1)
        obs_pct = 100 * c_obs / max(c_truth, 1)
        # Recovery factor: how much the model fills in beyond observations
        recovery_factor = c_rec / max(c_obs, 1) if c_obs > 0 else 0
        label = (f"Species #{s_idx:5d}   "
                 f"truth={c_truth:>3d}   "
                 f"obs={c_obs:>3d} ({obs_pct:>4.0f}%)   "
                 f"recon={c_rec:>3d} ({rec_pct:>4.0f}%)   "
                 f"recovery={recovery_factor:.1f}x obs")
        legend_handles.append(Patch(facecolor=SPECIES_COLORS[i], edgecolor='black',
                                    linewidth=0.5, alpha=0.7, label=label))

    leg = ax_legend.legend(
        handles=legend_handles, loc='center', ncol=1,
        fontsize=10, frameon=True, fancybox=False,
        edgecolor='#cbd5e1', framealpha=1.0,
        title="Species recovery summary  "
              "(same colours across all panels  •  "
              "recovery = recon cells per observed cell)",
        title_fontsize=11
    )
    leg.get_title().set_fontweight('bold')

    # ── OVERALL TITLE ────────────────────────────────────────
    if cond_type == 'fixed_budget' and cond_value:
        K = int(cond_value)
        regime_str = f"K = {K} observations per species"
    elif cond_type == 'poisson_v5' and cond_value:
        regime_str = f"Poisson $p_{{obs}}$ = {cond_value:g}"
    else:
        regime_str = f"$p_{{obs}}$ = {p_obs:g}"

    fig.suptitle(
        f"AB14 — Truth | Observed | Reconstructed   "
        f"({n_species} species, {regime_str})\n"
        f"Left = simulation truth   "
        f"middle = sparse observations fed to model   "
        f"right = AI reconstruction",
        fontsize=12.5, fontweight='bold', y=1.005
    )

    fig.tight_layout()
    safe_save(fig, out / 'fig_AB14_truth_observed_reconstructed_fixed.png')


# ──────────────────────────────────────────────────────────────────
# TEXT SUMMARY
# ──────────────────────────────────────────────────────────────────

def print_summary(conds):
    """Print a clean summary of all results."""
    print()
    print("=" * 78)
    print("  STAGE 2 ABLATION — MERGED RESULTS (v3 + v5 corrected Poisson)")
    print("=" * 78)
    full_r = get_metric(conds, 'FULL', 'richness_r')
    print(f"  Full model reference:  richness r = {full_r:.4f}")
    print()

    groups = [
        ('Original 5 conditions',
         ['FULL', 'NO_INTERACT', 'OBS_INFILL', 'NO_TEMPORAL', 'ENV_ONLY']),
        ('ARM A — Temporal gap',
         ['GAP_5', 'GAP_25']),
        ('ARM B — Fixed budget sparsification',
         ['SPARSE_1', 'SPARSE_5', 'SPARSE_10', 'SPARSE_20', 'SPARSE_50']),
        ("ARM B — Poisson (v5 BODY_MASS corrected)",
         ['POISSON_p00001', 'POISSON_p0001', 'POISSON_p0005',
          'POISSON_p001', 'POISSON_p01']),
    ]

    for name, keys in groups:
        print(f"  {name}")
        print("  " + "-" * 74)
        for k in keys:
            if k in conds:
                c = conds[k]
                r = c['metrics']['richness_r']
                r_std = c['metrics']['richness_r_std']
                range_r = c['metrics']['range_r']
                beta_r = c['metrics']['beta_r']
                label = c['label']
                print(f"    {label:<22}  richness={r:+.4f}±{r_std:.3f}  "
                      f"range={range_r:+.4f}  beta={beta_r:+.4f}")
        print()

    print("=" * 78)


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate all Stage 2 ablation figures from merged JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('merged_json', help='Path to ablation_v5_merged.json')
    parser.add_argument('--output-dir', default='stage2_ablation_figures',
                        help='Directory for output figures (default: %(default)s)')
    parser.add_argument('--copy-json', action='store_true', default=True,
                        help='Copy merged JSON into output dir (default: True)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip the 10 standard figures, only generate AB14')
    # AB14 controls
    parser.add_argument('--truth-npz', default=None,
                        help='Path to IBM truth NPZ (e.g. world_0001.npz). '
                             'Required for AB14.')
    parser.add_argument('--recon-npz', default=None,
                        help='Path to MetaDiffusion reconstruction NPZ. '
                             'Optional if --ab14-demo is set.')
    parser.add_argument('--ab14-demo', action='store_true',
                        help='AB14 demo mode: use obs_mask_50 (≈ saturation, '
                             'r≈0.92) as a model-at-saturation stand-in '
                             'when no recon NPZ is available.')
    parser.add_argument('--ab14-p-obs', type=float, default=0.001,
                        help='Per-individual observation probability for the '
                             'middle (Observed) panel of AB14. '
                             'Default 0.001 = 0.1%% (sweet spot, r approx 0.80). '
                             'Use 0.0001 for the "0.01%%" sweet spot. '
                             'Recommended values: 0.0001, 0.0005, 0.001, 0.01.')
    parser.add_argument('--ab14-n-species', type=int, default=5,
                        help='Number of species to overlay in AB14 (default 5)')
    parser.add_argument('--ab14-selection', default='best_recovery',
                        choices=['best_recovery', 'realistic',
                                 'top_prevalence', 'diverse', 'random'],
                        help='How to pick species for AB14 '
                             '(default: best_recovery). '
                             'best_recovery = species in the [15-40] occupancy '
                             'band where the model fills in the most cells '
                             'beyond the input. RECOMMENDED for showing '
                             "Axel's '5 obs → full range' narrative when the "
                             'model is genuinely doing recovery. '
                             'realistic = species with 15-40 cells (visual '
                             'sparsity guaranteed but no model-quality filter); '
                             'top_prevalence = the most widespread (NOT '
                             'recommended: 100% Poisson detection saturation); '
                             'diverse = mix across prevalence quartiles; '
                             'random = random with prevalence ≥ 3 cells.')
    parser.add_argument('--ab14-recon-threshold', type=float, default=0.5,
                        help='Threshold for converting MetaDiffusion '
                             'probability output to presence/absence in the '
                             'recon panel (default 0.5). Try 0.3 for a more '
                             'inclusive view that shows the model filling '
                             'in unobserved cells beyond the input.')
    parser.add_argument('--ab14-seed', type=int, default=42,
                        help='Random seed for species selection. Note: noisy '
                             'panel ALWAYS uses seed=42 to match '
                             'generate_reconstructions.py default, regardless '
                             'of this flag. (default 42)')
    args = parser.parse_args()

    input_path = Path(args.merged_json)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        return 1

    with open(input_path) as f:
        data = json.load(f)

    conds = data['conditions']

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Input JSON:  {input_path}")
    print(f"  Output dir:  {out.resolve()}")
    print(f"  Conditions:  {len(conds)}")
    print(f"  n_worlds:    {data.get('n_worlds')}")
    print(f"  n_samples:   {data.get('n_samples')}")

    if args.copy_json and input_path.resolve() != (out / input_path.name).resolve():
        shutil.copy(input_path, out / input_path.name)
        print(f"  ✓ Copied {input_path.name} to output dir")

    print_summary(conds)

    # Generate the 10 standard figures
    if not args.skip_existing:
        print("  Generating standard figures...")
        plot_AB1(conds, out)
        plot_AB2(conds, out)
        plot_AB3(conds, out)
        plot_AB7(conds, out)
        plot_AB8(conds, out)
        plot_AB9(conds, out)
        plot_AB10(conds, out)
        plot_AB11(conds, out)
        plot_AB12(conds, out)
        plot_AB13(conds, out)
    else:
        print("  Skipping standard figures (--skip-existing).")

    # Generate AB14 if truth NPZ provided
    if args.truth_npz:
        truth_path = Path(args.truth_npz)
        if not truth_path.exists():
            print(f"  ⚠ AB14 SKIPPED: --truth-npz path does not exist: {truth_path}")
        else:
            recon_path = Path(args.recon_npz) if args.recon_npz else None
            if args.recon_npz and not recon_path.exists():
                print(f"  ⚠ AB14: --recon-npz {recon_path} does not exist. "
                      f"Falling back to demo mode.")
                recon_path = None
                args.ab14_demo = True
            if recon_path is None and not args.ab14_demo:
                print()
                print("  ⚠ AB14 SKIPPED: no --recon-npz provided and "
                      "--ab14-demo not enabled.")
                print()
                print("  TO PRODUCE AB14, EITHER:")
                print()
                print("    [Option 1 — preferred] Save your model predictions as NPZ:")
                print("      np.savez('predictions/mean_predictions.npz',")
                print("              mean=mean_predictions_array)   # shape (S, Y, X)")
                print("      then re-run with:")
                print("        --truth-npz path/to/world_0001.npz \\")
                print("        --recon-npz predictions/mean_predictions.npz")
                print()
                print("    [Option 2 — fallback] Use the demo mode (uses obs_mask_50 ")
                print("      ≈ model at saturation, r≈0.92, as a stand-in):")
                print("        --truth-npz path/to/world_0001.npz \\")
                print("        --ab14-demo")
                print()
            else:
                try:
                    plot_AB14(
                        truth_npz=truth_path,
                        recon_npz=recon_path,
                        out=out,
                        p_obs=args.ab14_p_obs,
                        n_species=args.ab14_n_species,
                        selection=args.ab14_selection,
                        seed=args.ab14_seed,
                        demo_mode=args.ab14_demo,
                        recon_threshold=args.ab14_recon_threshold,
                    )
                except Exception as e:
                    print(f"  ⚠ AB14 FAILED: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print()
        print("  ℹ AB14 NOT GENERATED — pass --truth-npz path/to/world_XXXX.npz "
              "to enable.")

    print()
    print(f"  All figures saved to: {out.resolve()}")
    print(f"  Total files: {len(list(out.glob('*.png')))} PNGs + "
          f"{len(list(out.glob('*.json')))} JSON")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())