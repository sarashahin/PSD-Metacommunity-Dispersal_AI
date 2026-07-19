#!/usr/bin/env python3
"""
=============================================================================
CROSS-WORLD SUMMARY FIGURE — REGIME-INDEPENDENCE EVIDENCE
=============================================================================

Companion to axel_per_species_map_ecological.py.

The per-world three-map figure tells Axel "this is what the AI did on ONE
world." This script tells him "this is what the AI does ACROSS WORLDS."

Direct quotes this figure answers:
  email:        "in some statistical sense the ground truth is part of
                 this ensemble"
  transcript    "particularly if there's very little input data, it just
   0:28-1:22:    generates something different each time. And I thought
                 maybe that's good. ... So we want our model probably to
                 make some random predictions."
  transcript    "for far away cells, the AI just randomly picks ... is
   10:09:        also probably not better than chance"

The cross-world figure quantifies four metrics per world and shows them
side by side. The CRITICAL scientific claim — that ENS (ensemble UNION
recall) is stable across ecological regimes — is the headline supporting
Axel's email criterion of ensemble-support evaluation.

WHAT THIS SCRIPT DOES (and doesn't do)
======================================
DOES:
  - Imports the EXACT same metric functions used by the per-world figure
    (per_species_recall, per_species_recall_near_far,
     compute_ensemble_truth_coverage), so cross-world numbers are
    GUARANTEED IDENTICAL to per-world numbers — they're the same code.
  - Uses the EXACT same species-picker (_pick_three_map_species), so each
    world contributes the same 5 species that the per-world three-map
    figure shows. A reviewer can audit any column against the per-world
    figure.
  - Produces ONE figure with 4 panels (mean novel / NEAR / FAR / ENS),
    each showing per-species dots + per-world bar + random-baseline dash.
  - Produces ONE CSV with every per-species number for every world,
    so the underlying data is fully auditable.

DOES NOT:
  - Compute new metrics. Only the four already-validated ones.
  - Modify or wrap the per-world figure. Use that separately.
  - Cherry-pick species. Uses the same picker as the per-world figure.
  - Resample / refit / re-threshold. Pure aggregation.

USAGE
=====
    python axel_cross_world_summary_figure.py \\
        --truth-dir          ./results/data \\
        --recon-dir-pattern  './reconstructions_spatial/{world_stem}' \\
        --K                  5 \\
        --n-species          5 \\
        --worlds-csv         worlds_to_summarise.csv \\
        --output-path        ./figures/Fig_cross_world_summary.png \\
        --output-csv         ./figures/Fig_cross_world_summary.csv

Or list worlds inline:
    python axel_cross_world_summary_figure.py \\
        --truth-dir          ./results/data \\
        --recon-dir-pattern  './reconstructions_spatial/{world_stem}' \\
        --K                  5 \\
        --world-stems        baseline=pool22510000_batcha_ls10p0_..._training \\
                             ld020=pool22510000_batcha_..._ld0p2_training \\
                             highmix=pool22510000_batchb_highmix_..._training \\
                             widerange=pool22510000_..._ls2p5_..._training \\
        --output-path        ./figures/Fig_cross_world_summary.png

The CSV form is:
    label,world_stem
    baseline,pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_..._ld0p06_training
    ld020,pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_..._ld0p2_training
    ...

LABEL --- short ecological tag used as the x-axis tick (e.g. "baseline",
"ld=0.20", "highmix", "wide-range"). The full world_stem appears in the
CSV and in the figure footer.
=============================================================================
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Import the per-world module so we use IDENTICAL metric functions.
# We add the per-world script's directory to sys.path so this script
# can be placed next to it without packaging.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import axel_per_species_map_ecological_metrics as ape
except ImportError as e:
    raise ImportError(
        "Could not import axel_per_species_map_ecological. "
        "Place this file in the same directory as that module, or add "
        "that module's directory to PYTHONPATH.") from e


GRID_Y = ape.GRID_Y
GRID_X = ape.GRID_X


# ──────────────────────────────────────────────────────────────────────
# Per-world metrics — uses the IDENTICAL helpers as the per-world figure
# ──────────────────────────────────────────────────────────────────────

def compute_one_world_metrics(world_data, K, n_species=5):
    """For one world, pick the same 5 species the per-world three-map
    figure would pick, then compute (mean, near, far, ens) recall for each.

    Returns a list of dicts, one per picked species. Each dict has all
    the per-species data needed for both the figure and the CSV.
    """
    truth     = world_data['truth']       # (S, Y, X) binary
    samples   = world_data['samples']     # (n_ens, S, Y, X) probability
    mean_pred = world_data['mean_pred']   # (S, Y, X) probability
    observed  = world_data['observed']    # (S, Y, X) binary

    # Identical species picker — keeps cross-world figure rows in 1:1
    # correspondence with per-world figure species
    chosen = ape._pick_three_map_species(
        truth, mean_pred, K=K, n_species=n_species, prefer_wide_range=True)

    grid_size = GRID_Y * GRID_X
    per_species = []
    for sp in chosen:
        rng_truth = int(truth[sp].sum())
        if rng_truth == 0:
            continue

        # Top-N binary from MEAN (same logic as the per-world figure's
        # 'match_truth' mode — exact top-N via argpartition)
        flat = mean_pred[sp].ravel()
        if rng_truth > 0 and flat.max() > 1e-6:
            idx_topN = np.argpartition(flat, -rng_truth)[-rng_truth:]
            binary_recon = np.zeros(flat.size, dtype=np.uint8)
            binary_recon[idx_topN] = 1
            binary_recon = binary_recon.reshape(truth[sp].shape)
        else:
            binary_recon = np.zeros_like(truth[sp])

        rec_all, rec_novel = ape.per_species_recall(
            truth[sp], binary_recon, observed[sp])
        near_far = ape.per_species_recall_near_far(
            truth[sp], binary_recon, observed[sp], near_radius=2)
        ens_rec_novel, ens_div = ape.compute_ensemble_truth_coverage(
            truth[sp], samples[:, sp], observed[sp])

        # Standard ecology metrics (SDM literature)
        jaccard = ape.per_species_jaccard(truth[sp], binary_recon)
        boyce   = ape.per_species_boyce_index(truth[sp], mean_pred[sp])

        per_species.append({
            'sp_id':           int(sp),
            'range':           rng_truth,
            'K':               K,
            'recall_all':      rec_all,
            'recall_novel':    rec_novel,
            'recall_near':     near_far['rec_near'],
            'recall_far':      near_far['rec_far'],
            'baseline_novel':  rng_truth / max(1, grid_size - K),
            'baseline_near':   near_far['baseline_near'],
            'baseline_far':    near_far['baseline_far'],
            'recall_ens':      ens_rec_novel,
            'diversity_jaccard': ens_div,
            'cand_near':       near_far['cand_near'],
            'cand_far':        near_far['cand_far'],
            'n_truth_near':    near_far['n_truth_near'],
            'n_truth_far':     near_far['n_truth_far'],
            # Standard ecology metrics
            'jaccard':         jaccard,
            'boyce_index':     boyce,
        })

    return per_species


def _world_means(per_species, key):
    """Mean of `key` across species, skipping NaN. Returns nan if all-NaN."""
    vals = [d[key] for d in per_species if not np.isnan(d[key])]
    return float(np.mean(vals)) if vals else float('nan')


# ──────────────────────────────────────────────────────────────────────
# Figure rendering
# ──────────────────────────────────────────────────────────────────────
# Layout: 2 rows × 4 cols (gridspec)
#   Row 1 (4 panels, shared y-axis = recall):
#     A. Mean recall(novel)   B. NEAR recall   C. FAR recall   D. ENS UNION
#   Row 2 (Jaccard + Boyce + text summary):
#     E. Jaccard               F. Boyce Index             (ecology summary)
#
# Panels A-D answer Axel's transcript/email asks (recall, near/far, ensemble).
# Panels E-F answer standard SDM reviewer asks (Jaccard, Boyce Index).


def _render_panel(ax, world_results, key, baseline_key,
                    title, subtitle, y_max=None,
                    y_min=0.0, percent_axis=True,
                    show_zero_line=False,
                    show_x_random=True,
                    show_legend=False,
                    bar_color='#4477AA', bar_edge='#1f3556',
                    note_when_all_zero=None):
    """Render one bar-with-scatter panel for one metric.

    Pulled out as a helper so we can render recall-based panels and
    Jaccard/Boyce panels with the same visual style. Each metric has its
    own y-axis range (e.g. Boyce uses [-1, +1] with a zero reference
    line) but the rendering logic is identical."""
    n_worlds = len(world_results)
    x_pos = np.arange(n_worlds)
    bar_w = 0.55

    bar_vals = []
    baseline_vals = []
    scatter_x, scatter_y = [], []
    for wi, wr in enumerate(world_results):
        sp_vals = [d[key]    for d in wr['per_species']
                   if not np.isnan(d[key])]
        bl_vals = ([d[baseline_key] for d in wr['per_species']
                    if not np.isnan(d[baseline_key])]
                   if baseline_key else [])
        bar_vals.append(float(np.mean(sp_vals)) if sp_vals else 0.0)
        baseline_vals.append(float(np.mean(bl_vals)) if bl_vals else 0.0)
        n_sp = len(sp_vals)
        if n_sp:
            jitter = np.linspace(-bar_w * 0.30, bar_w * 0.30, n_sp)
            ordered = sorted(
                [(d[key], d['sp_id']) for d in wr['per_species']
                 if not np.isnan(d[key])])
            for (val, _), jx in zip(ordered, jitter):
                scatter_x.append(wi + jx)
                scatter_y.append(val)

    # Zero reference line for signed metrics (Boyce)
    if show_zero_line:
        ax.axhline(0.0, color='#888', linewidth=0.8, zorder=1)

    # Per-world mean bar
    ax.bar(x_pos, bar_vals, bar_w,
            color=bar_color, edgecolor=bar_edge,
            linewidth=1.0, alpha=0.85, zorder=2)
    # Baseline dashes (only if baseline_key was provided)
    if baseline_key:
        for xi, bl in zip(x_pos, baseline_vals):
            ax.hlines(bl, xi - bar_w / 2, xi + bar_w / 2,
                       color='#111111', linewidth=1.4, linestyle='--',
                       zorder=4)
    # Per-species scatter
    ax.scatter(scatter_x, scatter_y, s=22, c='#444444',
                marker='o', edgecolor='white', linewidth=0.6,
                zorder=5, alpha=0.85)

    # × annotation above each bar (only if baseline meaningful and asked for)
    if show_x_random and baseline_key:
        for xi, bv, bl in zip(x_pos, bar_vals, baseline_vals):
            if bl > 1e-9 and y_max:
                x_rand = bv / bl
                y_lab = max(bv, bl) + y_max * 0.03
                ax.text(xi, y_lab, f"{x_rand:.1f}\u00d7",
                          ha='center', va='bottom', fontsize=9.5,
                          color='#222', fontweight='bold')

    # Panel labels
    ax.set_title(f"{title}\n{subtitle}",
                   fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([wr['label'] for wr in world_results],
                         rotation=20, ha='right', fontsize=9.5)
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    if percent_axis:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
    else:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:+.1f}' if y else '0.0'))
    ax.grid(axis='y', alpha=0.25, linestyle=':')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # In-panel note when all worlds near zero
    if note_when_all_zero:
        if (max(bar_vals) < 0.03 and max(baseline_vals) < 0.03):
            ax.text(0.5, 0.55, note_when_all_zero,
                     transform=ax.transAxes, ha='center', va='center',
                     fontsize=9, color='#555', style='italic',
                     bbox=dict(facecolor='white', edgecolor='#bbb',
                                 boxstyle='round,pad=0.4', alpha=0.85))

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=bar_color, edgecolor=bar_edge,
                            label='per-world mean'),
            plt.Line2D([0], [0], color='#111111', linewidth=1.4,
                        linestyle='--', label='random baseline'),
            plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='#444444', markeredgecolor='white',
                        markersize=7, label='per-species value', linewidth=0),
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=8.5,
                   frameon=False)

    return bar_vals, baseline_vals


def make_cross_world_figure(world_results, output_path, K):
    """world_results : list of dicts of the form
         {'label': ..., 'world_stem': ..., 'per_species': [...]}
       in the order the user wants them displayed on the x-axis.

    Layout: 2 rows × 4 cols (gridspec).
      Row 1 (4 panels, shared y-axis = recall):
        A. Mean recall(novel)   B. NEAR recall   C. FAR recall   D. ENS UNION
      Row 2 (2 panels with own y-axis, 1 empty for spacing):
        E. Jaccard               F. Boyce Index             (text summary)
    """
    n_worlds = len(world_results)
    if n_worlds == 0:
        raise ValueError("No world results to plot")

    # Y-max for recall panels (shared)
    recall_keys = [('recall_novel', 'baseline_novel'),
                    ('recall_near',  'baseline_near'),
                    ('recall_far',   'baseline_far'),
                    ('recall_ens',   'baseline_novel')]
    all_recall_vals = []
    for wr in world_results:
        for d in wr['per_species']:
            for k, bk in recall_keys:
                v = d.get(k, float('nan'))
                if not np.isnan(v): all_recall_vals.append(v)
                b = d.get(bk, float('nan'))
                if not np.isnan(b): all_recall_vals.append(b)
    y_max_recall = max(all_recall_vals) * 1.15 if all_recall_vals else 1.0
    y_max_recall = max(y_max_recall, 0.10)

    # Y-max for Jaccard panel (own scale)
    all_jac = [d.get('jaccard', float('nan'))
               for wr in world_results for d in wr['per_species']
               if not np.isnan(d.get('jaccard', float('nan')))]
    y_max_jac = max(all_jac) * 1.20 if all_jac else 0.50
    y_max_jac = max(y_max_jac, 0.30)

    # Y-range for Boyce panel ([-1, +1] standard range)
    all_boy = [d.get('boyce_index', float('nan'))
               for wr in world_results for d in wr['per_species']
               if not np.isnan(d.get('boyce_index', float('nan')))]
    if all_boy:
        boy_min = min(min(all_boy), -0.1) - 0.1
        boy_max = max(max(all_boy), +0.1) + 0.15
    else:
        boy_min, boy_max = -1.05, 1.20

    # Figure with explicit gridspec for two rows
    fig = plt.figure(figsize=(4.0 * n_worlds + 5.0, 11.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=4, height_ratios=[1.0, 1.0],
        hspace=0.85, wspace=0.32,
        left=0.05, right=0.985, top=0.91, bottom=0.13,
    )
    # Row 1: 4 recall panels (sharing y-axis)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1], sharey=ax_A)
    ax_C = fig.add_subplot(gs[0, 2], sharey=ax_A)
    ax_D = fig.add_subplot(gs[0, 3], sharey=ax_A)
    # Row 2: Jaccard + Boyce, then a text summary spanning 2 columns
    ax_E = fig.add_subplot(gs[1, 0])
    ax_F = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2:])

    # ─ Row 1 panels (recall metrics) ─
    _render_panel(ax_A, world_results, 'recall_novel', 'baseline_novel',
                   'A. Mean recall(novel)', 'single best prediction',
                   y_max=y_max_recall, show_legend=True)
    _render_panel(ax_B, world_results, 'recall_near', 'baseline_near',
                   'B. NEAR recall', 'within 2 cells of obs',
                   y_max=y_max_recall)
    _render_panel(ax_C, world_results, 'recall_far', 'baseline_far',
                   'C. FAR recall', 'beyond 2 cells of obs',
                   y_max=y_max_recall,
                   note_when_all_zero=
                     "All worlds near zero\n"
                     "(model does not extrapolate\n"
                     "beyond ~2 cells from obs;\n"
                     "matches transcript 10:09 expectation)")
    _render_panel(ax_D, world_results, 'recall_ens', 'baseline_novel',
                   'D. ENS UNION recall', 'truth in ensemble support',
                   y_max=y_max_recall)
    ax_A.set_ylabel('Recall at unobserved truth cells', fontsize=10)

    # ─ Row 2 panel E: Jaccard (no baseline — Jaccard has no natural null) ─
    _render_panel(ax_E, world_results, 'jaccard', None,
                   'E. Jaccard similarity',
                   'set overlap = TP / (truth \u222a pred)',
                   y_max=y_max_jac, show_x_random=False,
                   bar_color='#117733', bar_edge='#0a4a1f')
    ax_E.set_ylabel('Jaccard similarity (0\u20131)', fontsize=10)
    # Annotate Jaccard mean above each bar
    x_pos = np.arange(n_worlds)
    for xi, wr in zip(x_pos, world_results):
        vals = [d['jaccard'] for d in wr['per_species']
                if not np.isnan(d.get('jaccard', float('nan')))]
        if vals:
            m = float(np.mean(vals))
            ax_E.text(xi, m + y_max_jac * 0.03, f"{m:.2f}",
                       ha='center', va='bottom', fontsize=9.5,
                       color='#222', fontweight='bold')

    # ─ Row 2 panel F: Boyce Index (own y-range [-1, +1] with zero line) ─
    _render_panel(ax_F, world_results, 'boyce_index', None,
                   'F. Boyce Index',
                   'probability ranking quality',
                   y_max=boy_max, y_min=boy_min,
                   percent_axis=False, show_x_random=False,
                   show_zero_line=True,
                   bar_color='#CC6677', bar_edge='#7a3340')
    ax_F.set_ylabel('Boyce Index (\u22121 \u2192 +1)', fontsize=10)
    # Annotate Boyce mean and indicate if signif > random expectation
    # (random model: per-species std \u2248 0.44 on 20x20 grid \u2192
    #  population SEM for n species = 0.44/\u221an)
    for xi, wr in zip(x_pos, world_results):
        vals = [d['boyce_index'] for d in wr['per_species']
                if not np.isnan(d.get('boyce_index', float('nan')))]
        if vals:
            m = float(np.mean(vals))
            n_sp = len(vals)
            sem_random = 0.44 / np.sqrt(max(1, n_sp))
            z = m / sem_random if sem_random > 0 else 0.0
            sig = '*' if abs(z) > 1.96 else ''
            y_lab = m + (boy_max - boy_min) * 0.03 if m >= 0 else m - (boy_max - boy_min) * 0.06
            va_lab = 'bottom' if m >= 0 else 'top'
            ax_F.text(xi, y_lab, f"{m:+.2f}{sig}",
                       ha='center', va=va_lab, fontsize=9.5,
                       color='#222', fontweight='bold')

    # ─ Summary text panel ─
    ax_summary.axis('off')
    # Build the ecology summary using actual numbers
    summary_lines = ["ECOLOGY METRIC SUMMARY", ""]
    all_jac = []; all_boy = []
    for wr in world_results:
        for d in wr['per_species']:
            if not np.isnan(d.get('jaccard', float('nan'))):
                all_jac.append(d['jaccard'])
            if not np.isnan(d.get('boyce_index', float('nan'))):
                all_boy.append(d['boyce_index'])
    if all_jac:
        m_jac = float(np.mean(all_jac))
        summary_lines.append(f"Jaccard (mean across all species): {m_jac:.3f}")
        summary_lines.append("  Real et al. 2006 SDM metric. Set-overlap")
        summary_lines.append("  similarity; ranges 0\u20131, higher better.")
        summary_lines.append("")
    if all_boy:
        m_boy = float(np.mean(all_boy))
        med_boy = float(np.median(all_boy))
        n_pos = sum(1 for b in all_boy if b > 0.05)
        n_neg = sum(1 for b in all_boy if b < -0.05)
        n_total = len(all_boy)
        # Population-level significance against the random-model null
        sem_random = 0.44 / np.sqrt(max(1, n_total))
        z = m_boy / sem_random if sem_random > 0 else 0.0
        summary_lines.append(f"Boyce Index (Hirzel et al. 2006):")
        summary_lines.append(f"  population mean    = {m_boy:+.3f}")
        summary_lines.append(f"  population median  = {med_boy:+.3f}")
        summary_lines.append(f"  positive species   = {n_pos}/{n_total}")
        summary_lines.append(f"  negative species   = {n_neg}/{n_total}")
        summary_lines.append(f"  vs random expect.  = {z:+.1f}\u03c3"
                              f" {'(p<0.05)' if abs(z) > 1.96 else '(n.s.)'}")
        summary_lines.append("")
        summary_lines.append("Boyce caveat: per-species variance is high")
        summary_lines.append("on 20\u00d720 grids (\u03c3\u22480.44 under H0). Interpret")
        summary_lines.append("the POPULATION mean across species, not")
        summary_lines.append("individual species values.")
    ax_summary.text(0.02, 0.98, "\n".join(summary_lines),
                     transform=ax_summary.transAxes, va='top', ha='left',
                     fontsize=9.5, family='monospace', color='#222',
                     bbox=dict(facecolor='#f5f5f5', edgecolor='#bbb',
                                 boxstyle='round,pad=0.6', alpha=0.95))

    # Suptitle — describes ENS spread across worlds. Only claims
    # "stable across regimes" when the actual cross-world ENS range
    # is narrow (max - min <= 0.10). Otherwise reports the range
    # neutrally so the figure does not over-claim.
    headline_ens = [
        _world_means(wr['per_species'], 'recall_ens')
        for wr in world_results
    ]
    headline_ens = [v for v in headline_ens if not np.isnan(v)]
    if headline_ens:
        ens_min = min(headline_ens); ens_max = max(headline_ens)
        ens_spread = ens_max - ens_min
        if ens_spread <= 0.10:
            stability = (
                f"ensemble UNION recall STABLE at "
                f"{ens_min:.0%}\u2013{ens_max:.0%} across ecological regimes "
                f"(spread \u2264 10pp)"
            )
        else:
            stability = (
                f"ensemble UNION recall range = "
                f"{ens_min:.0%}\u2013{ens_max:.0%} across worlds "
                f"(spread = {ens_spread*100:.0f}pp)"
            )
        head = (f"Cross-world summary  |  K = {K} observations/species  |  "
                 f"{n_worlds} worlds  |  {stability}")
    else:
        head = (f"Cross-world summary  |  K = {K} observations/species  |  "
                 f"{n_worlds} worlds")
    fig.suptitle(head, fontsize=12.5, fontweight='bold', y=0.965)

    # Bottom footer: full world stems for traceability
    footer_lines = []
    for wr in world_results:
        footer_lines.append(f"   {wr['label']:<14s}  {wr['world_stem']}")
    fig.text(0.005, 0.005,
              "Worlds shown (label  \u2192  world_stem):\n" +
              "\n".join(footer_lines) +
              "\n\nROW 1 (recall metrics, shared y-axis): blue bar = per-world "
              "mean across the same 5 species the per-world figure picks; dots "
              "= per-species values; dashed line = random baseline; \u00d7 "
              "annotation = mean/baseline ratio.\n"
              "ROW 2: green bar (E) = Jaccard similarity (own [0,1] scale); "
              "pink bar (F) = Boyce Index (own [\u22121,+1] scale). "
              "Asterisk on Boyce annotation = per-world mean significantly "
              "above random expectation (|z|>1.96 vs population SEM under H0).",
              ha='left', va='bottom', fontsize=8, color='#555',
              family='monospace', wrap=True)

    # NOTE: do NOT call plt.tight_layout() — gridspec already manages it.
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# CSV output
# ──────────────────────────────────────────────────────────────────────

def write_csv(world_results, csv_path):
    """One row per (world, species). Includes everything needed to
    reconstruct the figure values independently."""
    fields = [
        'world_label', 'world_stem', 'K', 'sp_id', 'range',
        'recall_all', 'recall_novel', 'recall_near', 'recall_far',
        'recall_ens', 'diversity_jaccard',
        'baseline_novel', 'baseline_near', 'baseline_far',
        'n_truth_near', 'n_truth_far', 'cand_near', 'cand_far',
        # Standard ecology metrics
        'jaccard', 'boyce_index',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for wr in world_results:
            for d in wr['per_species']:
                row = {k: d.get(k, '') for k in fields if k in d}
                row['world_label'] = wr['label']
                row['world_stem']  = wr['world_stem']
                # Convert NaNs to empty strings for CSV cleanliness
                for k, v in list(row.items()):
                    if isinstance(v, float) and np.isnan(v):
                        row[k] = ''
                w.writerow(row)


# ──────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────

def parse_world_arg(world_arg):
    """Parse 'label=world_stem' or bare 'world_stem' (label = stem prefix)."""
    if '=' in world_arg:
        label, stem = world_arg.split('=', 1)
        return label.strip(), stem.strip()
    # Fall back: use a short auto-label from the parameters
    p = ape.parse_world_params(world_arg)
    parts = []
    if 'ls' in p:  parts.append(f"ls={p['ls']}")
    if 'ld' in p:  parts.append(f"ld={p['ld']}")
    if 'thr' in p: parts.append(f"thr={p['thr']}")
    label = ', '.join(parts) if parts else world_arg[:18]
    return label, world_arg


def load_worlds_csv(csv_path):
    """Load a CSV with columns 'label' and 'world_stem'."""
    out = []
    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            label = row.get('label', '').strip()
            stem = row.get('world_stem', '').strip()
            if not stem:
                continue
            if not label:
                label, _ = parse_world_arg(stem)
            out.append((label, stem))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Cross-world summary figure for Axel review.")
    ap.add_argument('--truth-dir', required=True,
                     help='Directory containing the truth NPZ files.')
    ap.add_argument('--recon-dir-pattern', required=True,
                     help="Pattern with {world_stem}, e.g. "
                          "'./reconstructions_spatial/{world_stem}'.")
    ap.add_argument('--K', type=int, default=5,
                     help='Observations per species (matches recon NPZ name).')
    ap.add_argument('--n-species', type=int, default=5,
                     help='Number of species per world (matches per-world fig).')
    ap.add_argument('--world-stems', nargs='+', default=None,
                     help="Space-separated 'label=world_stem' entries, OR bare "
                          "world_stems (label auto-derived from params).")
    ap.add_argument('--worlds-csv', default=None,
                     help='CSV with columns label, world_stem.')
    ap.add_argument('--output-path', required=True,
                     help='Output figure path (.png).')
    ap.add_argument('--output-csv', default=None,
                     help='Output CSV path. Defaults to figure path with .csv suffix.')
    args = ap.parse_args()

    # Resolve the list of worlds
    if args.worlds_csv:
        worlds = load_worlds_csv(args.worlds_csv)
    elif args.world_stems:
        worlds = [parse_world_arg(w) for w in args.world_stems]
    else:
        ap.error("Provide either --world-stems or --worlds-csv")

    if not worlds:
        raise SystemExit("No worlds provided")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = (Path(args.output_csv) if args.output_csv
                  else output_path.with_suffix('.csv'))

    # Compute per-world metrics
    print(f"\n  Cross-world summary  |  K = {args.K}  |  "
            f"{len(worlds)} worlds\n")
    world_results = []
    for label, stem in worlds:
        try:
            wd = ape.load_world(
                args.truth_dir, args.recon_dir_pattern, stem, K=args.K)
        except FileNotFoundError as e:
            print(f"  \u2717 {label:<14s}  MISSING:")
            for line in str(e).splitlines():
                print(f"      {line}")
            continue
        per_species = compute_one_world_metrics(
            wd, K=args.K, n_species=args.n_species)
        if not per_species:
            print(f"  \u2717 {label:<14s}  NO USABLE SPECIES (truth_range\u2264K?)")
            continue
        world_results.append({
            'label':       label,
            'world_stem':  stem,
            'per_species': per_species,
        })
        # Per-world summary line — mirrors the per-world figure suptitle
        mn = _world_means(per_species, 'recall_novel')
        mb = _world_means(per_species, 'baseline_novel')
        nn = _world_means(per_species, 'recall_near')
        ff = _world_means(per_species, 'recall_far')
        ee = _world_means(per_species, 'recall_ens')
        jj = _world_means(per_species, 'jaccard')
        bb = _world_means(per_species, 'boyce_index')
        xr = (mn / mb) if (mb and mb > 1e-9 and not np.isnan(mn)) else float('nan')
        def _fmt(v): return f"{v:.0%}" if not np.isnan(v) else " n/a"
        def _fmt_signed(v): return f"{v:+.2f}" if not np.isnan(v) else " n/a"
        print(f"  \u2713 {label:<14s}  novel={_fmt(mn)} ({xr:>4.1f}\u00d7)  "
              f"near={_fmt(nn)}  far={_fmt(ff)}  ens={_fmt(ee)}  "
              f"Jac={jj:.2f}  Boy={_fmt_signed(bb)}  "
              f"[{len(per_species)} species]")

    if not world_results:
        raise SystemExit("No world results produced — check paths.")

    # Render
    make_cross_world_figure(world_results, str(output_path), K=args.K)
    write_csv(world_results, str(csv_path))

    print(f"\n  \u2713 figure  \u2192 {output_path}")
    print(f"  \u2713 csv     \u2192 {csv_path}\n")


if __name__ == '__main__':
    main()