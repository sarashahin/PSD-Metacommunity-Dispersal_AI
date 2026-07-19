#!/usr/bin/env python3
"""make_regime_manifests.py — adapt the eval manifest for build_wide_range_csv.py
and the figure scripts, which expect a 'found' column == 'yes'. Splits the eval
manifest by regime so in-distribution (24) and extrapolation (6) are reported
SEPARATELY. Writes <out-dir>/eval_manifest_indist.csv and _extrap.csv."""
import argparse, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.manifest)))
    fields = rows[0].keys()
    out_fields = list(fields) + (["found"] if "found" not in fields else [])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for regime, fn in (("in_dist", "eval_manifest_indist.csv"),
                       ("extrap",  "eval_manifest_extrap.csv")):
        sel = [r for r in rows if r["regime"] == regime]
        with open(out / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=out_fields)
            w.writeheader()
            for r in sel:
                rr = dict(r); rr["found"] = "yes"; w.writerow(rr)
        counts[fn] = len(sel)
        print(f"  ✓ {out/fn}: {len(sel)} worlds (found=yes)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())