#!/usr/bin/env python3
"""
=============================================================================
POOLED PIT / RANK HISTOGRAM  —  ensemble calibration check
=============================================================================
For every species, posterior_per_species.py recorded the PERCENTILE of the
true statistic within that species' ensemble (the share of samples below the
truth: columns patches_pctile, spread_pctile). Pooled over many species, a
well-calibrated ensemble gives a UNIFORM (flat) histogram - the test Axel
described ("the percentage of sample values below the true value, over many
iterations, will be evenly distributed if the method is perfect").

How to read the shape:
  flat         -> calibrated: the ensemble's spread honestly brackets the truth
  left-heavy   -> truth usually BELOW the samples: model OVER-predicts the statistic
                  (e.g. ranges drawn more fragmented / more spread than reality)
  right-heavy  -> truth usually ABOVE the samples: model UNDER-predicts it
  U-shaped     -> ensemble too NARROW (over-confident): truth often outside the spread
  dome-shaped  -> ensemble too WIDE (under-confident)

Statistics: connected patches and spatial spread. Range size has no percentile
here, because the top-N rule holds every sample at the true range size.
=============================================================================
"""

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

NICE = {'patches': 'connected patches', 'spread': 'spatial spread',
        'range': 'range size'}


def read_column(csv_paths, stat):
    col = f"{stat}_pctile"
    vals, n_ens = [], []
    for p in csv_paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                v = row.get(col, "")
                if v not in ("", None):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
                ne = row.get("n_ens", "")
                if ne:
                    try:
                        n_ens.append(int(float(ne)))
                    except ValueError:
                        pass
    return np.asarray(vals, float), (min(n_ens) if n_ens else None)


def panel(ax, vals, title, bins):
    if vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha='center', va='center'); ax.set_title(title)
        return
    below  = float(np.mean(vals < 0.05))
    within = float(np.mean((vals >= 0.05) & (vals <= 0.95)))
    above  = float(np.mean(vals > 0.95))
    ax.axvspan(0.05, 0.95, color='#DDE7F0', alpha=0.6, zorder=0)   # ensemble support band
    ax.hist(vals, bins=np.linspace(0, 1, bins + 1),
            color='#4477AA', alpha=0.75, edgecolor='white', zorder=2)
    ax.axhline(vals.size / bins, ls='--', color='#CC3311', lw=1.6,
               label='uniform (calibrated)', zorder=3)
    try:
        _, pval = stats.kstest(vals, 'uniform')
        sub = f"KS vs uniform p = {pval:.3f}"
    except Exception:
        sub = ""
    ax.set_title(f"{title}\n"
                 f"truth WITHIN ensemble (5\u201395%): {within:.0%}   "
                 f"(below {below:.0%}, above {above:.0%})\n"
                 f"n = {vals.size}   mean pctile = {vals.mean():.2f}   {sub}",
                 fontsize=9)
    ax.set_xlabel("percentile of truth within ensemble")
    ax.set_ylabel("number of species")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc='upper center')
    ax.spines[['top', 'right']].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description="Pooled PIT histogram (ensemble calibration)")
    ap.add_argument('--csv', nargs='+', required=True,
                    help="metrics CSV(s) or glob(s) from posterior_per_species.py (same regime)")
    ap.add_argument('--stats', nargs='+', default=['patches', 'spread'])
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--label', default='', help="regime label for the title, e.g. 'p=0.10' or 'K=5'")
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    paths = []
    for c in args.csv:
        hits = sorted(glob.glob(c))
        paths += hits if hits else [c]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        print("no CSV files found"); return
    print(f"  pooling {len(paths)} CSV file(s)")

    fig, axes = plt.subplots(1, len(args.stats),
                             figsize=(5.2 * len(args.stats), 4.7), squeeze=False)
    min_ne = None
    for j, stat in enumerate(args.stats):
        vals, ne = read_column(paths, stat)
        if ne is not None:
            min_ne = ne if min_ne is None else min(min_ne, ne)
        panel(axes[0, j], vals, NICE.get(stat, stat), args.bins)
        if vals.size:
            _, p = stats.kstest(vals, 'uniform')
            within = float(np.mean((vals >= 0.05) & (vals <= 0.95)))
            print(f"  {stat:>8}: n={vals.size}  within[5,95]={within:.0%}  "
                  f"mean pctile={vals.mean():.3f}  KS-uniform p={p:.3f}")

    title = "Calibration check: is the truth's percentile uniform across species?"
    if args.label:
        title += f"   [{args.label}]"
    fig.suptitle(title, fontsize=12.5, fontweight='bold')

    foot = ("Percentile of the TRUE statistic within each species' ensemble (share of samples below the "
            "truth). Shaded band (5\u201395%) = the ensemble's support: a species there has its truth "
            "bracketed by the reconstructions. Left-heavy = reconstructions more fragmented / more spread "
            "than reality (over-predicts the statistic); right-heavy = under-predicts; flat = calibrated.")
    if min_ne is not None and min_ne < 20:
        foot += (f"  NOTE: ensembles have only {min_ne} samples, so percentiles are coarse - "
                 "regenerate with ~100 samples for a clean test.")
    fig.text(0.5, 0.005, foot, ha='center', fontsize=8, style='italic', color='#555', wrap=True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.92])
    plt.savefig(args.output, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved -> {args.output}")


if __name__ == "__main__":
    main()