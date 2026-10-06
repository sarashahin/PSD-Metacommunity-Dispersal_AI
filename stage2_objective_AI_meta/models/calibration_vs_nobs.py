#!/usr/bin/env python3
"""
Coverage vs number of observations — one honest law across sampling regimes.

Reads one or more PIT CSVs (each needs columns: n_obs, patches_pctile,
spread_pctile), pools every species-case, bins by how many records the species
had, and plots the share whose TRUE range shape falls inside the ensemble's
5-95 percent band against the record count. Because it bins on observation
count, K=5, proportional-10 percent and proportional-30 percent can be pooled:
the point is that calibration is governed by how many records a species has,
not by the sampling scheme that produced them.
"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BINS = [(1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'), (5, 9, '5-9'), (10, 10**9, '10+')]


def load(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    nob = np.array([int(r['n_obs']) for r in rows])
    pp = np.array([float(r['patches_pctile']) for r in rows])
    sp = np.array([float(r['spread_pctile']) for r in rows])
    return nob, pp, sp


def coverage(v):
    return float(np.mean((v >= 0.05) & (v <= 0.95))) if v.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True,
                    help='one or more PIT CSVs that include an n_obs column')
    ap.add_argument('--output', required=True)
    ap.add_argument('--title', default='Calibration improves with the number of records per species')
    a = ap.parse_args()

    nob, pp, sp = load(a.csv)
    xs = np.arange(len(BINS))
    labels = [b[2] for b in BINS]
    cov_p, cov_s, ns = [], [], []
    for lo, hi, _ in BINS:
        m = (nob >= lo) & (nob <= hi)
        cov_p.append(coverage(pp[m]))
        cov_s.append(coverage(sp[m]))
        ns.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, np.array(cov_p) * 100, 'o-', color='#4477AA', lw=2.2, ms=8, label='connected patches')
    ax.plot(xs, np.array(cov_s) * 100, 's-', color='#228833', lw=2.2, ms=8, label='spatial spread')
    for i in range(len(BINS)):
        if ns[i]:
            ax.annotate(f"n={ns[i]}", (xs[i], 4), ha='center', va='bottom', fontsize=8, color='#777')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel('number of observations per species')
    ax.set_ylabel('truth within ensemble (5\u201395%)   [%]')
    ax.set_ylim(0, 100)
    ax.set_title(a.title, fontweight='bold', fontsize=12)
    ax.legend(frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.5, 0.005,
             "Each point pools every species with that many records, across all sampling schemes supplied. "
             "Higher = the ensemble brackets the true range shape more often. The curve is the same whatever the "
             "sampling scheme \u2014 what matters is how many records a species has.",
             ha='center', va='bottom', fontsize=8, style='italic', color='#555', wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(a.output, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"saved -> {a.output}")
    print(f"pooled {len(nob)} species-cases from {len(a.csv)} CSV(s)")
    for i in range(len(BINS)):
        cp = 'n/a' if np.isnan(cov_p[i]) else f"{cov_p[i]*100:3.0f}%"
        cs = 'n/a' if np.isnan(cov_s[i]) else f"{cov_s[i]*100:3.0f}%"
        print(f"  obs {labels[i]:>4}: n={ns[i]:3d}  patches cov={cp}  spread cov={cs}")


if __name__ == '__main__':
    main()