#!/usr/bin/env python3
"""
=============================================================================
SELECT WIDE-RANGE SPECIES — for honest AB14 figures
=============================================================================

THE PROBLEM
-----------
Your single test world has only 14 species with ranges >10 cells. 75% of
species occupy 1-2 cells. With K=5 observations on a 1-2 cell species,
"reconstruction" is mathematically degenerate. This makes the AB14 figure
hard to interpret.

THE SOLUTION
------------
Across your 22 test worlds (saved by data_preprocessing during training),
find species with substantial ranges (15-50 cells) — i.e., where K=5
observations represent 10-30% of the true range and reconstruction is
genuinely meaningful. Use those species for AB14 figures.

WHAT THIS GIVES YOU
-------------------
For wide-range species, the ensemble's per-species recall is 33-45% (from
the coverage stats you just computed). At that level, the AB14 figure
shows real reconstruction: the truth has ~25 cells, observations show
5 cells, the ensemble union covers ~12 of the 25 truth cells. This is
the "truth is part of the ensemble" demonstration Axel asked for.

USAGE
-----
    python select_wide_range_worlds.py \\
        --truth-dir   ./results/data/ \\
        --min-range   15 \\
        --max-range   50 \\
        --n-species   5 \\
        --output-csv  ./figures_map_axel_stage2_new/wide_range_species.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def find_wide_range_in_world(npz_path, min_range, max_range, top_k=10):
    """
    Find species with truth ranges in [min_range, max_range] cells.
    Returns list of (species_idx, n_cells) tuples sorted by range desc.
    """
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            P = np.asarray(d['P_last_final'])
    except Exception as e:
        print(f"  ⚠ failed to load {npz_path.name}: {e}", file=sys.stderr)
        return []

    if P.ndim == 2:
        # Already (S, n_cells)
        ranges = (P > 0.5).sum(axis=1)
    else:
        # (S, Y, X)
        ranges = (P > 0.5).sum(axis=(1, 2))

    valid = (ranges >= min_range) & (ranges <= max_range)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return []

    sorted_idx = valid_idx[np.argsort(-ranges[valid_idx])]
    return [(int(s), int(ranges[s])) for s in sorted_idx[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-dir', required=True,
                    help='Directory containing IBM simulation .npz files')
    ap.add_argument('--min-range', type=int, default=15,
                    help='Minimum number of presence cells (default 15)')
    ap.add_argument('--max-range', type=int, default=50,
                    help='Maximum number of presence cells (default 50)')
    ap.add_argument('--n-species', type=int, default=5,
                    help='How many wide-range species to select per world')
    ap.add_argument('--output-csv', default=None)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    npz_files = sorted(truth_dir.glob('*_training.npz'))
    print(f"Scanning {len(npz_files)} world files in {truth_dir}")
    print(f"Looking for species with range in [{args.min_range}, "
          f"{args.max_range}] cells\n")

    all_finds = []
    n_with_wide = 0
    for f in npz_files:
        finds = find_wide_range_in_world(f, args.min_range, args.max_range,
                                          top_k=args.n_species)
        if finds:
            n_with_wide += 1
            print(f"  {f.name}: found {len(finds)} wide-range species")
            for s_idx, rg in finds:
                print(f"    species {s_idx:>5}: range = {rg:>3} cells")
            all_finds.append({'world': f.name, 'species': finds})

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  Worlds scanned:                {len(npz_files)}")
    print(f"  Worlds with wide-range species: {n_with_wide}")
    print(f"  Total wide-range species found: "
          f"{sum(len(f['species']) for f in all_finds)}")

    # Find the world with the MOST wide-range species — best for figure
    best_world = max(all_finds, key=lambda x: len(x['species']),
                     default=None)
    if best_world:
        print(f"\n  RECOMMENDED WORLD FOR AB14:")
        print(f"    {best_world['world']}")
        print(f"    Has {len(best_world['species'])} wide-range species:")
        for s_idx, rg in best_world['species'][:args.n_species]:
            print(f"      species {s_idx} (range = {rg})")

    # Save CSV
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['world', 'species_idx', 'truth_range_cells'])
            for record in all_finds:
                for s_idx, rg in record['species']:
                    w.writerow([record['world'], s_idx, rg])
        print(f"\n  ✓ wrote {out}")


if __name__ == "__main__":
    main()