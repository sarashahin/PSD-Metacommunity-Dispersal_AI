#!/usr/bin/env python3
"""
=============================================================================
RECALL vs OBSERVATIONS  —  MEE paper figure (rewrite)
=============================================================================
Axel's request:
    "Recall (far/near, and similar) as a function of (a) number of
     observations and/or (b) percentage observed: demonstrating that AI
     predicts true locations to some extent, and that prediction makes
     good use of sampled presence data."

WHAT CHANGED vs the previous version (and why)
----------------------------------------------
1. ENS baseline corrected. The ENS series is the UNION of the S ensemble
   samples, so its random baseline must be the union of S random top-N
   draws:  bl_ens = 1 - (1 - R/cand)^S  (cand = unobserved cells).
   The old code used R/cand (a single draw), which under-stated chance and
   disagreed with the per-species map figure ("chance 54%"). Fixed here so
   both figures use the same definition.
2. Readable layout. Three schemes x three series = 9 overlapping lines was
   unreadable. Now a 2x2 grid:
      row 1  NEAR / FAR / ENS pooled over all schemes   -> "does it work"
      row 2  one clean line per scheme (K=5 / p=0.10 / p=0.30) -> "K vs prop"
      col A  vs number of observations   col B  vs % of range observed
   Each panel has at most 3 data lines.
3. Baselines are faint dotted lines (one legend proxy), not a thicket.
4. No baked-in bottom caption and no world-string title (put text in the
   LaTeX caption). Panel tags (A)-(D) in bold, per MEE.
5. Vector PDF (crisp at any size) + 300-dpi PNG. Fonts >= 8 pt at final size.

Keep this file in the same folder as axel_per_species_map_ecological.py so
the recall functions match the map figure exactly.

USAGE (see bottom of this file for a full runnable block):
    python recall_vs_observations.py \
      --truth-dir ./results/data/data_eval_unseen \
      --world-stems STEM1 STEM2 ... \
      --labels "K=5" "p=0.10" "p=0.30" \
      --recon-dir-patterns './reconstructions_proportional_k5_n50/{world_stem}' \
                           './reconstructions_proportional_n50/{world_stem}' \
                           './reconstructions_proportional_p30_n50/{world_stem}' \
      --recon-filenames recon_fixed_b5_samples.npz \
                        recon_prop_p0.10_samples.npz \
                        recon_prop_p0.30_samples.npz \
      --layout full --x both --include-ens --scheme-metric near \
      --output ./results/Fig_recall_vs_observations \
      --csv ./results/recall_vs_observations.csv
=============================================================================
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reuse the identical logic used by the per-world map figures.
from axel_per_species_map_ecological import (
    load_world,
    per_species_recall_near_far,
    compute_ensemble_truth_coverage,
    GRID_Y, GRID_X,
)

# --- binning ----------------------------------------------------------------
OBS_BINS = [(1, 1, '1'), (2, 2, '2'), (3, 4, '3\u20134'), (5, 9, '5\u20139'),
            (10, 10**9, '10+')]
PCT_BINS = [(0, 10, '\u226410'), (10, 20, '10\u201320'), (20, 30, '20\u201330'),
            (30, 50, '30\u201350'), (50, 100.01, '>50')]

# Paul Tol colourblind-safe
C_NEAR, C_FAR, C_ENS = '#4477AA', '#EE6677', '#228833'
SCHEME_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']
C_BASE = '#777777'

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'lines.solid_capstyle': 'round',
    'figure.dpi': 300, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # keep text editable in vector output
})


# ---------------------------------------------------------------------------
def topN_binary(mean_sp, n):
    """Model binary = the n highest-probability cells (n = true range size)."""
    flat = mean_sp.ravel()
    if n <= 0 or flat.max() < 1e-6:
        return np.zeros_like(mean_sp, dtype=np.uint8)
    idx = np.argpartition(flat, -n)[-n:]
    b = np.zeros(flat.size, dtype=np.uint8)
    b[idx] = 1
    return b.reshape(mean_sp.shape)


def ens_union_baseline(range_N, n_obs, n_samples, grid=GRID_Y * GRID_X):
    """Random chance for the ENSEMBLE UNION recall on novel cells.

    Each of `n_samples` random draws marks `range_N` cells among the
    `cand = grid - n_obs` unobserved cells. For one novel truth cell the
    chance a single draw hits it is p = range_N / cand; the chance the
    UNION of S independent draws hits it is 1 - (1 - p)^S. This matches the
    'chance' printed on the per-species map figure.
    """
    cand = max(1, grid - n_obs)
    p_single = min(1.0, range_N / cand)
    return 1.0 - (1.0 - p_single) ** max(1, n_samples)


def collect_rows(truth_dir, stems, label, pattern, filename, near_radius):
    """One row per species: record count, % observed, NEAR/FAR/ENS recall and
    matched random baselines (ENS baseline uses the union-of-S formula)."""
    rows = []
    grid = GRID_Y * GRID_X
    for stem in stems:
        w = load_world(truth_dir, pattern, stem, recon_filename=filename)
        truth, samples, mean, obs = (w['truth'], w['samples'],
                                     w['mean_pred'], w['observed'])
        S = int(samples.shape[0])                     # ensemble size
        for sp in range(truth.shape[0]):
            R = int(truth[sp].sum())
            if R <= 0:
                continue
            n_obs = int(obs[sp].sum())
            recon = topN_binary(mean[sp], R)
            nf = per_species_recall_near_far(truth[sp], recon, obs[sp],
                                             near_radius=near_radius)
            ens_rec, _ = compute_ensemble_truth_coverage(truth[sp],
                                                         samples[:, sp], obs[sp])
            rows.append(dict(
                regime=label, world=stem, sp=sp, range_N=R, n_obs=n_obs,
                pct_obs=100.0 * n_obs / R, n_samples=S,
                rec_near=nf['rec_near'], rec_far=nf['rec_far'], rec_ens=ens_rec,
                bl_near=nf['baseline_near'], bl_far=nf['baseline_far'],
                bl_ens=ens_union_baseline(R, n_obs, S, grid)))
    return rows


def binned_mean(rows, key, bins, field):
    """Mean of `field` within each bin of `key` (NaNs skipped)."""
    ys, ns = [], []
    for lo, hi, _ in bins:
        vals = [r[field] for r in rows
                if lo <= r[key] <= hi and r[field] == r[field]]   # NaN-safe
        ys.append(np.mean(vals) if vals else np.nan)
        ns.append(len(vals))
    return np.array(ys), np.array(ns)


# --- panels -----------------------------------------------------------------
def _style(ax, xlabels, xlabel, tag):
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('recall on unobserved true cells  [%]')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', lw=0.4, alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.015, 0.98, tag, transform=ax.transAxes, fontweight='bold',
            fontsize=11, va='top', ha='left')


def panel_pooled(ax, rows, key, bins, xlabel, tag, include_ens, show_n):
    x = np.arange(len(bins))
    labels = [b[2] for b in bins]
    series = [('rec_near', 'bl_near', 'NEAR (\u22642 cells)', C_NEAR, 'o'),
              ('rec_far',  'bl_far',  'FAR (>2 cells)',       C_FAR,  's')]
    if include_ens:
        series.append(('rec_ens', 'bl_ens', 'ENS (union)', C_ENS, '^'))
    for field, blf, lab, c, mk in series:
        y, _ = binned_mean(rows, key, bins, field)
        yb, _ = binned_mean(rows, key, bins, blf)
        ax.plot(x, y * 100, marker=mk, color=c, lw=2.2, ms=6, label=lab, zorder=3)
        ax.plot(x, yb * 100, ls=':', color=c, lw=1.2, alpha=0.7, zorder=2)
    if show_n:
        _, nc = binned_mean(rows, key, bins, 'rec_near')
        for i, nn in enumerate(nc):
            if nn:
                ax.annotate(f'n={nn}', (x[i], 1.5), ha='center', va='bottom',
                            fontsize=6.5, color='#999')
    _style(ax, labels, xlabel, tag)
    handles, labs = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], ls=':', color=C_BASE, lw=1.2))
    labs.append('random baseline')
    ax.legend(handles, labs, frameon=False, loc='upper right', handlelength=1.6)


def panel_by_scheme(ax, rows, key, bins, xlabel, tag, metric, regimes):
    x = np.arange(len(bins))
    labels = [b[2] for b in bins]
    field = {'near': 'rec_near', 'far': 'rec_far', 'ens': 'rec_ens'}[metric]
    blf   = {'near': 'bl_near',  'far': 'bl_far',  'ens': 'bl_ens'}[metric]
    mname = {'near': 'NEAR', 'far': 'FAR', 'ens': 'ENS'}[metric]
    for ri, rlabel in enumerate(regimes):
        rr = [r for r in rows if r['regime'] == rlabel]
        c = SCHEME_COLORS[ri % len(SCHEME_COLORS)]
        y, _ = binned_mean(rr, key, bins, field)
        ax.plot(x, y * 100, marker='o', color=c, lw=2.2, ms=6, label=rlabel, zorder=3)
    yb, _ = binned_mean(rows, key, bins, blf)             # one pooled baseline
    ax.plot(x, yb * 100, ls=':', color=C_BASE, lw=1.2, alpha=0.8, zorder=2,
            label='random baseline')
    _style(ax, labels, xlabel, tag)
    ax.legend(frameon=False, loc='upper right', title=f'{mname} by scheme',
              title_fontsize=8, handlelength=1.6)


# --- figure builder ---------------------------------------------------------
def build_figure(rows, regimes, layout, x, include_ens, scheme_metric,
                 show_n, title):
    cols = []
    if x in ('both', 'obs'):
        cols.append(('n_obs', OBS_BINS, 'number of observations per species'))
    if x in ('both', 'pct'):
        cols.append(('pct_obs', PCT_BINS, '% of true range observed'))

    row_kinds = {'full': ['pooled', 'by-scheme'],
                 'pooled': ['pooled'],
                 'by-scheme': ['by-scheme']}[layout]

    nrows, ncols = len(row_kinds), len(cols)
    fw = 6.9 if ncols == 2 else 3.7
    fh = 3.15 * nrows + 0.15
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)

    tag = iter([f'({c})' for c in 'ABCDEFGH'])
    for r, kind in enumerate(row_kinds):
        for c, (key, bins, xlabel) in enumerate(cols):
            ax = axes[r][c]
            if kind == 'pooled':
                panel_pooled(ax, rows, key, bins, xlabel, next(tag),
                             include_ens, show_n)
            else:
                panel_by_scheme(ax, rows, key, bins, xlabel, next(tag),
                                scheme_metric, regimes)

    if title:
        fig.suptitle(title, fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97 if title else 1.0])
    return fig


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['regime', 'world', 'sp', 'range_N', 'n_obs', 'pct_obs', 'n_samples',
              'rec_near', 'rec_far', 'rec_ens', 'bl_near', 'bl_far', 'bl_ens']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in fields}
            for k in ('rec_near', 'rec_far', 'rec_ens', 'bl_near', 'bl_far',
                      'bl_ens', 'pct_obs'):
                v = out[k]
                out[k] = '' if (v != v) else round(float(v), 4)   # NaN -> ''
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--world-stems', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True,
                    help="scheme labels, e.g. 'K=5' 'p=0.10' 'p=0.30'")
    ap.add_argument('--recon-dir-patterns', nargs='+', required=True,
                    help="parallel to --labels; each contains {world_stem}")
    ap.add_argument('--recon-filenames', nargs='+', required=True,
                    help="parallel to --labels; exact samples filename per scheme")
    ap.add_argument('--near-radius', type=int, default=2)
    ap.add_argument('--x', choices=['both', 'obs', 'pct'], default='both')
    ap.add_argument('--layout', choices=['full', 'pooled', 'by-scheme'],
                    default='full',
                    help="full=2x2 (pooled + by-scheme); pooled or by-scheme = 1 row")
    ap.add_argument('--scheme-metric', choices=['near', 'far', 'ens'],
                    default='near', help="which recall the by-scheme row compares")
    ap.add_argument('--include-ens', dest='include_ens', action='store_true',
                    default=True, help="show ENS union in the pooled row (default on)")
    ap.add_argument('--no-ens', dest='include_ens', action='store_false')
    ap.add_argument('--show-n', action='store_true',
                    help="annotate per-bin species counts on the pooled row")
    ap.add_argument('--title', default='',
                    help="optional on-figure title (default none; put it in the caption)")
    ap.add_argument('--by-regime', action='store_true',
                    help="deprecated alias; forces --layout by-scheme")
    ap.add_argument('--output', required=True, help="base path; writes .pdf and .png")
    ap.add_argument('--csv', default=None, help="optional per-species CSV dump")
    args = ap.parse_args()

    if args.by_regime and args.layout == 'full':
        args.layout = 'by-scheme'

    n = len(args.labels)
    if not (len(args.recon_dir_patterns) == n and len(args.recon_filenames) == n):
        ap.error("--labels, --recon-dir-patterns and --recon-filenames "
                 "must be the same length")

    rows = []
    for label, pattern, filename in zip(args.labels, args.recon_dir_patterns,
                                        args.recon_filenames):
        r = collect_rows(args.truth_dir, args.world_stems, label, pattern,
                         filename, args.near_radius)
        print(f"  {label}: {len(r)} species from {len(args.world_stems)} world(s)")
        rows += r
    if not rows:
        ap.error("no species collected \u2014 check paths/filenames")

    fig = build_figure(rows, args.labels, args.layout, args.x, args.include_ens,
                       args.scheme_metric, args.show_n, args.title)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = out.with_suffix('.png'), out.with_suffix('.pdf')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')          # vector
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')  # raster
    plt.close(fig)
    print(f"  \u2713 saved -> {pdf}")
    print(f"  \u2713 saved -> {png}")
    if args.csv:
        write_csv(args.csv, rows)
        print(f"  \u2713 saved -> {args.csv}  ({len(rows)} rows)")

    # console summary: pooled recall by observation bin (the headline trend)
    for lab, field in [('NEAR', 'rec_near'), ('FAR', 'rec_far'),
                       ('ENS', 'rec_ens')]:
        y, nc = binned_mean(rows, 'n_obs', OBS_BINS, field)
        cells = "  ".join(
            f"{OBS_BINS[i][2]}:{'' if y[i] != y[i] else f'{y[i]*100:.0f}%'}"
            f"(n={nc[i]})" for i in range(len(OBS_BINS)))
        print(f"    {lab:>4} by obs -> {cells}")


if __name__ == '__main__':
    main()