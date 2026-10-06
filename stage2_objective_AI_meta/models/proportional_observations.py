#!/usr/bin/env python3
"""
=============================================================================
PROPORTIONAL (POISSON / BERNOULLI) OBSERVATION — DIAGNOSTIC FIGURE
=============================================================================
Purpose
-------
Illustrate the TWO observation extremes for the paper:
  - fixed-K        : every species gets the same number of records (K).
  - proportional   : each truly-occupied cell is observed with probability p,
                     so abundant species get many records and rare species few.
The real observation process lies between the two; showing the method works at
both brackets the realistic case (the exact percentage is not important).

This script ONLY makes the diagnostic figure (observation counts under the two
schemes) and, optionally, saves the proportional masks. It does NOT make
reconstructions. The actual p=10% reconstructions come from the modified
generation script (generate_reconstructions_proportional.py), which builds the
SAME Bernoulli observations internally, so the two stay consistent.
=============================================================================
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_truth(truth_path, truth_key=None):
    """Return (binary S,Y,X truth, key_used). Auto-detects the truth key and
    validates shape, so stray / differently-keyed .npz files do not crash the
    run. Falls back to the last timestep of P_t if needed."""
    with np.load(truth_path, allow_pickle=True) as td:
        files = list(td.files)
        arr, used = None, None
        for k in ([truth_key] if truth_key else []) + ['P_last_final', 'P_final', 'P']:
            if k and k in files:
                a = np.asarray(td[k])
                if a.ndim == 3:
                    arr, used = a, k; break
        if arr is None and 'P_t' in files:
            a = np.asarray(td['P_t'])
            if a.ndim == 4:
                arr, used = a[-1], 'P_t[-1]'
    if arr is None:
        raise KeyError(f"no 3-D truth array found; keys present: {files}")
    return (arr > 0.5).astype(np.uint8), used


def proportional_mask(truth_world, p, rng, min_obs=1):
    """Each true cell kept with probability p. Tops up to min_obs cells for any
    present species (so the sampler always has a seed). min_obs=0 = pure Bernoulli."""
    S = truth_world.shape[0]
    mask = np.zeros_like(truth_world)
    counts = np.zeros(S, dtype=int)
    for s in range(S):
        cells = np.argwhere(truth_world[s] > 0)
        if len(cells) == 0:
            continue
        keep = rng.random(len(cells)) < p
        if keep.sum() < min_obs:
            idx = rng.choice(len(cells), size=min(min_obs, len(cells)), replace=False)
            keep = np.zeros(len(cells), dtype=bool); keep[idx] = True
        for (y, x) in cells[keep]:
            mask[s, y, x] = 1
        counts[s] = int(keep.sum())
    return mask, counts


def main():
    ap = argparse.ArgumentParser(description="Diagnostic: fixed-K vs proportional observation counts")
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--world-stems', default=None,
                    help="comma-separated stems (no .npz). Omit to use every match of --pattern")
    ap.add_argument('--pattern', default='*.npz', help="glob within truth-dir (e.g. 'pool*_training.npz')")
    ap.add_argument('--truth-key', default=None, help="force a truth key (else auto-detect)")
    ap.add_argument('--prob', type=float, default=0.10)
    ap.add_argument('--min-obs', type=int, default=1)
    ap.add_argument('--K-compare', type=int, default=5)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--save-masks', action='store_true', help="also write per-world mask .npz (not needed for generation)")
    ap.add_argument('--fig-path', default=None)
    ap.add_argument('--seed', type=int, default=20240)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig_path) if args.fig_path else out_dir / f'obs_count_extremes_p{args.prob:.2f}.png'

    if args.world_stems:
        stems = [s.strip() for s in args.world_stems.split(',') if s.strip()]
    else:
        stems = sorted(Path(p).stem for p in glob.glob(str(truth_dir / args.pattern)))
    if not stems:
        print(f"no files matching {args.pattern} in {truth_dir}"); return

    all_N, all_prop, all_fixed = [], [], []
    skipped = []
    print(f"\nProportional sampling  p = {args.prob:.2f}   (min_obs = {args.min_obs})")
    print(f"{'world':<46} {'species':>7} {'mean N':>7} {'mean prop-obs':>13} {'fixed-K':>8}")
    for stem in stems:
        tp = truth_dir / f'{stem}.npz'
        if not tp.exists():
            continue
        try:
            truth, key_used = load_truth(tp, truth_key=args.truth_key)
        except KeyError as e:
            skipped.append((stem, str(e))); continue
        if truth.ndim != 3 or truth.shape[0] == 0:
            skipped.append((stem, f"bad shape {truth.shape}")); continue
        wrng = np.random.default_rng(args.seed + abs(hash(stem)) % (2**31))
        mask, counts = proportional_mask(truth, args.prob, wrng, min_obs=args.min_obs)
        if args.save_masks:
            np.savez(out_dir / f'{stem}__obsmask_p{args.prob:.2f}.npz',
                     obs_mask=mask.astype(np.uint8), prob=np.float32(args.prob), world_stem=stem)
        N = truth.sum(axis=(1, 2)).astype(int)
        present = N > 0
        all_N += list(N[present]); all_prop += list(counts[present])
        all_fixed += list(np.minimum(args.K_compare, N[present]))
        print(f"{stem[:44]:<46} {int(present.sum()):>7} {N[present].mean():>7.1f} "
              f"{counts[present].mean():>13.1f} {args.K_compare:>8}")

    if skipped:
        print(f"\n  skipped {len(skipped)} non-truth file(s):")
        for s, why in skipped[:10]:
            print(f"    - {s[:60]}  ({why[:70]})")
    if not all_N:
        print("\n  no usable truth files found - nothing to plot."); return

    all_N = np.array(all_N); all_prop = np.array(all_prop); all_fixed = np.array(all_fixed)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    a = ax[0]
    a.scatter(all_N, all_fixed, s=22, color='#CC6677', alpha=0.7,
              label=f'fixed-K (K={args.K_compare})', edgecolor='white', lw=0.4)
    a.scatter(all_N, all_prop, s=22, color='#4477AA', alpha=0.7,
              label=f'proportional (p={args.prob:.0%})', edgecolor='white', lw=0.4)
    xs = np.linspace(all_N.min(), all_N.max(), 50)
    a.plot(xs, args.prob * xs, '--', color='#4477AA', lw=1.2)
    a.set_xlabel('true range size (occupied cells)'); a.set_ylabel('observations for that species')
    a.set_title('Observations vs range size', fontsize=11, fontweight='bold')
    a.legend(fontsize=9, loc='upper left'); a.spines[['top', 'right']].set_visible(False)

    b = ax[1]
    hi = int(max(all_prop.max(), all_fixed.max())) + 1
    bins = np.arange(-0.5, hi + 1.5, 1)
    b.hist(all_fixed, bins=bins, color='#CC6677', alpha=0.55, label=f'fixed-K (K={args.K_compare})')
    b.hist(all_prop, bins=bins, color='#4477AA', alpha=0.55, label=f'proportional (p={args.prob:.0%})')
    b.set_xlabel('observations per species'); b.set_ylabel('number of species')
    b.set_title('Distribution of observation counts', fontsize=11, fontweight='bold')
    b.legend(fontsize=9); b.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f"Two observation extremes (n = {len(all_N)} species across {len(set(stems))-len(skipped)} worlds)",
                 fontsize=12.5, fontweight='bold')
    fig.text(0.5, 0.005,
             "Fixed-K gives every species the same number of records (red). Proportional sampling observes each true "
             "cell with probability p, so abundant species get many records and rare species few (blue). The real "
             "observation process lies between these extremes; covering both brackets the realistic case.",
             ha='center', fontsize=8, style='italic', color='#555', wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(fig_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n  diagnostic figure -> {fig_path}")
    print(f"  proportional obs: median {np.median(all_prop):.0f}, "
          f"range {all_prop.min()}-{all_prop.max()}; species with 0 obs: {int(np.sum(all_prop==0))}")
    print("  NOTE: the actual p-sampling RECONSTRUCTIONS come from "
          "generate_reconstructions_proportional.py (same Bernoulli rule).")


if __name__ == "__main__":
    main()