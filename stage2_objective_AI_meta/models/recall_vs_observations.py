#!/usr/bin/env python3
"""
=============================================================================
RECALL vs OBSERVATIONS  —  Axel's paper figure
=============================================================================
Axel's request:
    "Recall (far/near, and similar) as a function of (a) number of
     observations and/or (b) percentage observed: demonstrating that AI
     predicts true locations to some extent, and that prediction makes
     good use of sampled presence data."

This pools every species across ALL supplied sampling regimes (e.g. K=5,
proportional 10%, proportional 30%) and plots recall on the UNOBSERVED true
cells as a function of:
    (a) the number of observations a species had, and
    (b) the percentage of its true range that was observed,
against the random baseline. Three recall series are shown:
    NEAR  recall on novel cells within `near_radius` of an observation
          (local spatial autocorrelation)
    FAR   recall on novel cells beyond that radius (true extrapolation)
    ENS   recall under the ensemble union of all samples at top-N ("and similar")

It reuses the EXACT recall functions from axel_per_species_map_ecological.py,
so the numbers match the per-world map figures. Keep both files in the same
folder.

Default pools all regimes onto one trend (the "records govern recall"
message). --by-regime instead draws one NEAR+FAR line per regime so the
three sampling schemes can be contrasted directly.

USAGE (three regimes, both x-axes, PNG + PDF):
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
      --x both --include-ens \
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

# Reuse the identical logic used by the per-world map figures.
from axel_per_species_map_ecological import (
    load_world,
    per_species_recall_near_far,
    compute_ensemble_truth_coverage,
    GRID_Y, GRID_X,
)

OBS_BINS = [(1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'), (5, 9, '5-9'), (10, 10**9, '10+')]
PCT_BINS = [(0, 10, '\u226410'), (10, 20, '10-20'), (20, 30, '20-30'),
            (30, 50, '30-50'), (50, 100.01, '>50')]

C_NEAR, C_FAR, C_ENS = '#4477AA', '#EE6677', '#228833'
REGIME_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']


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


def collect_rows(truth_dir, stems, label, pattern, filename, near_radius):
    """One row per species: its record count, % observed, and NEAR/FAR/ENS
    recall with matched random baselines."""
    rows = []
    grid = GRID_Y * GRID_X
    for stem in stems:
        w = load_world(truth_dir, pattern, stem, recon_filename=filename)
        truth, samples, mean, obs = (w['truth'], w['samples'],
                                     w['mean_pred'], w['observed'])
        for sp in range(truth.shape[0]):
            R = int(truth[sp].sum())
            if R <= 0:
                continue
            n_obs = int(obs[sp].sum())
            recon = topN_binary(mean[sp], R)
            nf = per_species_recall_near_far(truth[sp], recon, obs[sp],
                                             near_radius=near_radius)
            ens_rec, _ = compute_ensemble_truth_coverage(truth[sp], samples[:, sp], obs[sp])
            rows.append(dict(
                regime=label, world=stem, sp=sp, range_N=R, n_obs=n_obs,
                pct_obs=100.0 * n_obs / R,
                rec_near=nf['rec_near'], rec_far=nf['rec_far'], rec_ens=ens_rec,
                bl_near=nf['baseline_near'], bl_far=nf['baseline_far'],
                bl_ens=R / max(1, grid - n_obs)))
    return rows


def binned_mean(rows, key, bins, field):
    """Mean of `field` within each bin of `key` (NaNs skipped)."""
    ys, ns = [], []
    for lo, hi, _ in bins:
        vals = [r[field] for r in rows
                if lo <= r[key] <= hi and r[field] == r[field]]  # NaN-safe
        ys.append(np.mean(vals) if vals else np.nan)
        ns.append(len(vals))
    return np.array(ys), np.array(ns)


def draw_panel(ax, rows, key, bins, include_ens, by_regime, regimes, xlabel):
    x = np.arange(len(bins))
    labels = [b[2] for b in bins]
    if by_regime:
        for ri, rlabel in enumerate(regimes):
            rr = [r for r in rows if r['regime'] == rlabel]
            c = REGIME_COLORS[ri % len(REGIME_COLORS)]
            yn, _ = binned_mean(rr, key, bins, 'rec_near')
            yf, _ = binned_mean(rr, key, bins, 'rec_far')
            ax.plot(x, yn * 100, '-o', color=c, lw=2.0, ms=6, label=f'NEAR {rlabel}')
            ax.plot(x, yf * 100, '--s', color=c, lw=2.0, ms=6, label=f'FAR {rlabel}')
    else:
        series = [('rec_near', 'bl_near', 'NEAR (\u22642 cells)', C_NEAR, 'o'),
                  ('rec_far',  'bl_far',  'FAR (>2 cells)',       C_FAR,  's')]
        if include_ens:
            series.append(('rec_ens', 'bl_ens', 'ENS (any sample)', C_ENS, '^'))
        for field, blf, lab, c, mk in series:
            y, _ = binned_mean(rows, key, bins, field)
            yb, _ = binned_mean(rows, key, bins, blf)
            ax.plot(x, y * 100, marker=mk, color=c, lw=2.3, ms=7, label=lab)
            ax.plot(x, yb * 100, ls=':', color=c, lw=1.3, alpha=0.75)
        _, ncount = binned_mean(rows, key, bins, 'rec_far')
        for i, nn in enumerate(ncount):
            if nn:
                ax.annotate(f'n={nn}', (x[i], 2.0), ha='center', va='bottom',
                            fontsize=7, color='#888')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('recall on unobserved true cells   [%]')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, ncol=1)
    ax.spines[['top', 'right']].set_visible(False)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['regime', 'world', 'sp', 'range_N', 'n_obs', 'pct_obs',
              'rec_near', 'rec_far', 'rec_ens', 'bl_near', 'bl_far', 'bl_ens']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in fields}
            for k in ('rec_near', 'rec_far', 'rec_ens', 'bl_near', 'bl_far', 'bl_ens', 'pct_obs'):
                v = out[k]
                out[k] = '' if (v != v) else round(float(v), 4)  # NaN -> ''
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--world-stems', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True,
                    help="regime labels, e.g. 'K=5' 'p=0.10' 'p=0.30'")
    ap.add_argument('--recon-dir-patterns', nargs='+', required=True,
                    help="parallel to --labels; each with {world_stem}")
    ap.add_argument('--recon-filenames', nargs='+', required=True,
                    help="parallel to --labels; exact samples filename per regime")
    ap.add_argument('--near-radius', type=int, default=2)
    ap.add_argument('--x', choices=['obs', 'pct', 'both'], default='both')
    ap.add_argument('--include-ens', action='store_true',
                    help="add the ensemble-union recall line (Axel's 'and similar')")
    ap.add_argument('--by-regime', action='store_true',
                    help="draw one NEAR+FAR line per regime instead of pooling")
    ap.add_argument('--output', required=True, help="base path; writes .png and .pdf")
    ap.add_argument('--csv', default=None, help="optional per-species CSV dump")
    args = ap.parse_args()

    n = len(args.labels)
    if not (len(args.recon_dir_patterns) == n and len(args.recon_filenames) == n):
        ap.error("--labels, --recon-dir-patterns and --recon-filenames must be the same length")

    rows = []
    for label, pattern, filename in zip(args.labels, args.recon_dir_patterns, args.recon_filenames):
        r = collect_rows(args.truth_dir, args.world_stems, label, pattern, filename, args.near_radius)
        print(f"  {label}: {len(r)} species from {len(args.world_stems)} world(s)")
        rows += r
    if not rows:
        ap.error("no species collected — check paths/filenames")

    panels = [('n_obs', OBS_BINS, 'number of observations per species')] if args.x == 'obs' else \
             [('pct_obs', PCT_BINS, '% of true range observed')] if args.x == 'pct' else \
             [('n_obs', OBS_BINS, 'number of observations per species'),
              ('pct_obs', PCT_BINS, '% of true range observed')]

    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.2), squeeze=False)
    for ax, (key, bins, xlabel) in zip(axes[0], panels):
        draw_panel(ax, rows, key, bins, args.include_ens, args.by_regime, args.labels, xlabel)

    fig.suptitle("AI recall on unobserved cells rises with the amount of presence data",
                 fontweight='bold', fontsize=13)
    fig.text(0.5, 0.005,
             "Recall on truth cells the model was NOT shown, pooled across "
             f"{', '.join(args.labels)}. NEAR = within {args.near_radius} cells of an "
             "observation; FAR = beyond that (true extrapolation); "
             + ("ENS = any ensemble sample; " if args.include_ens else "")
             + "dotted = random baseline. Recall above the baseline means the model "
               "predicts true locations; recall rising with data means it uses the records.",
             ha='center', va='bottom', fontsize=8, style='italic', color='#555', wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = out.with_suffix('.png'), out.with_suffix('.pdf')
    plt.savefig(png, dpi=180, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  \u2713 saved -> {png}")
    print(f"  \u2713 saved -> {pdf}")
    if args.csv:
        write_csv(args.csv, rows)
        print(f"  \u2713 saved -> {args.csv}  ({len(rows)} rows)")

    # console summary: pooled recall by observation bin (the headline trend)
    for lab, field in [('NEAR', 'rec_near'), ('FAR', 'rec_far')] + \
                      ([('ENS', 'rec_ens')] if args.include_ens else []):
        y, nc = binned_mean(rows, 'n_obs', OBS_BINS, field)
        cells = "  ".join(f"{OBS_BINS[i][2]}:{'' if y[i] != y[i] else f'{y[i]*100:.0f}%'}(n={nc[i]})"
                          for i in range(len(OBS_BINS)))
        print(f"    {lab:>4} by obs -> {cells}")


if __name__ == '__main__':
    main()