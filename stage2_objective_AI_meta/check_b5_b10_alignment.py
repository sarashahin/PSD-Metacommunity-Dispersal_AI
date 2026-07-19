#!/usr/bin/env python3
"""check_b5_b10_alignment.py — confirm K=5 and K=10 reconstructions are aligned.
For every world in the eval manifest, load recon_fixed_b5_samples.npz and
recon_fixed_b10_samples.npz and confirm they have the SAME species count (S)
and grid (Y,X), and that both match the truth P_last_final. Fails loudly on any
mismatch so the K5-vs-K10 comparison figure is built on aligned arrays."""
import argparse, csv, sys
from pathlib import Path
import numpy as np

def shp(samples_path):
    z = np.load(samples_path)
    s = np.asarray(z['samples'])          # (n_ensemble, S, Y, X)
    return s.shape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--recon-dir-pattern", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.manifest)))
    ok = bad = 0
    for r in rows:
        stem = r["filename"][:-4] if r["filename"].endswith(".npz") else r["filename"]
        rd = Path(args.recon_dir_pattern.format(world_stem=stem))
        b5 = rd / "recon_fixed_b5_samples.npz"
        b10 = rd / "recon_fixed_b10_samples.npz"
        tp = Path(args.truth_dir) / r["filename"]
        problems = []
        if not b5.exists(): problems.append("missing b5")
        if not b10.exists(): problems.append("missing b10")
        if not problems:
            n5, S5, Y5, X5 = shp(b5)
            n10, S10, Y10, X10 = shp(b10)
            with np.load(tp, allow_pickle=True) as td:
                St, Yt, Xt = np.asarray(td["P_last_final"]).shape
            if (S5, Y5, X5) != (S10, Y10, X10):
                problems.append(f"S/Y/X differ b5={(S5,Y5,X5)} b10={(S10,Y10,X10)}")
            if (S5, Y5, X5) != (St, Yt, Xt):
                problems.append(f"recon vs truth differ recon={(S5,Y5,X5)} truth={(St,Yt,Xt)}")
        if problems:
            bad += 1; print(f"  MISMATCH {stem[:55]}: {'; '.join(problems)}")
        else:
            ok += 1
    print(f"\n  {ok} / {ok+bad} worlds: K5 and K10 aligned (same S, Y, X; match truth)")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())