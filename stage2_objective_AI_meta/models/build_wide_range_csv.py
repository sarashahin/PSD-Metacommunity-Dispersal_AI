#!/usr/bin/env python3
"""build_wide_range_csv.py
Scans inference world NPZs and emits the wide_range_species.csv that
multi_world_v7_evaluation.py expects.

USAGE
-----
    python build_wide_range_csv.py \\
        --manifest  ./figures_map_axel_stage2_new/inference_worlds_manifest.csv \\
        --truth-dir ./results/data \\
        --K         5 \\
        --output    ./figures_map_axel_stage2_new/wide_range_species.csv
"""

import argparse
import csv
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)

    with open(args.manifest) as f:
        manifest_rows = [r for r in csv.DictReader(f) if r.get("found") == "yes"]

    out_rows = []
    for r in manifest_rows:
        fname = r["filename"]
        fpath = truth_dir / fname
        if not fpath.exists():
            print(f"  ⚠ skip (missing): {fname}")
            continue

        try:
            with np.load(fpath, allow_pickle=True) as d:
                P = np.asarray(d["P_last_final"])
        except Exception as e:
            print(f"  ✗ error reading {fname}: {e}")
            continue

        # P shape: (S, Y, X)
        truth_bin = (P > 0.5).astype(np.uint8)
        ranges = truth_bin.reshape(truth_bin.shape[0], -1).sum(axis=1)
        n_wide = int((ranges > args.K).sum())
        n_present = int((ranges > 0).sum())

        # Emit one row per wide-range species (so multi_world_v7_evaluation
        # counting via defaultdict[world] += 1 works correctly)
        for s_idx in np.where(ranges > args.K)[0]:
                    out_rows.append({
                        "world": fname,
                        "species_idx": int(s_idx),
                        "range_size": int(ranges[s_idx]),
                        "batch": r.get("batch", r.get("scenario", "")),
                        "concern": r.get("concern", r.get("regime", "")),
                    })

        print(f"  ✓ {fname[:60]}:  present={n_present}, wide(>{args.K})={n_wide}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        if out_rows:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
        else:
            f.write("world,species_idx,range_size,batch,concern\n")

    n_worlds = len({r["world"] for r in out_rows})
    print(f"\n  ✓ Wrote {out}")
    print(f"     {len(out_rows)} wide-range species across {n_worlds} worlds")


if __name__ == "__main__":
    main()