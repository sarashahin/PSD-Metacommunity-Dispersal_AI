#!/usr/bin/env python3
"""
=============================================================================
author_clean_split.py  —  build a DELIBERATE, auditable train/val/test split
=============================================================================

WHY THIS EXISTS
---------------
The previous split was an accident of np.random.permutation over whatever
files happened to validate that run, and it was never saved. That is how 29
of 30 evaluation worlds ended up in training.

This script does the opposite, deliberately:
  1. You name the worlds you WANT to evaluate on (the held-out / test set).
  2. Those EXACT worlds are placed in 'test'.
  3. Everything else is split into train / val (val drawn deterministically
     with a fixed seed so it is reproducible).
  4. The result is written to split_manifest.csv — the frozen artifact that
     data_preprocessing_FROZEN_SPLIT.py reads and ENFORCES.
  5. It then reports whether the TRAINING set still covers the parameter
     ranges (ls, vr, thr, env, dr, ld) present in your worlds — so holding
     out your eval worlds does not strip a whole regime from training.

This script does NOT touch the model and does NOT run training. It only
produces (and sanity-checks) the manifest.

USAGE
-----
  # List eval worlds explicitly (filenames, with or without .npz):
  python author_clean_split.py \
      --sim-dir   ./results/data \
      --eval-list \
        pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr5em08_ld0p06_training \
        pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr5em08_ld0p2_training \
        pool22510000_batchb_highmix_ls10p0_vr0p002_thr3p0_env123_grid20x20_dr5em08_ld0p2_training \
        pool22510000_batcha_ls2p5_vr0p004_thr3p0_env456_grid20x20_dr1em07_ld0p2_training \
      --val-ratio 0.083 \
      --seed      42 \
      --out       ./split_audit_clean/split_manifest.csv

  # Or read eval worlds from a file (one filename per line):
  python author_clean_split.py --sim-dir ./results/data \
      --eval-file my_eval_worlds.txt --out ./split_audit_clean/split_manifest.csv

VALIDATION FILTER NOTE
----------------------
This script lists worlds with the SAME glob the trainer uses ('*_training.npz'
by default) and applies the SAME 20x20 validity check, so the manifest covers
exactly the files training will see. If any of your named eval worlds fail the
20x20 check or are not found, the script STOPS and tells you — it will not
silently drop them.
=============================================================================
"""

import argparse
import csv
import re
import sys
from pathlib import Path
import numpy as np


def grid_shape_fast(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        if 'Y' in d and 'X' in d:
            return (int(d['Y']), int(d['X']))
        if 'P_last_final' in d:
            a = np.asarray(d['P_last_final'])
            if a.ndim == 3:
                return (a.shape[1], a.shape[2])
        if 'P_t' in d:
            a = np.asarray(d['P_t'])
            if a.ndim == 4:
                return (a.shape[2], a.shape[3])
    raise ValueError(f"cannot determine grid shape for {Path(npz_path).name}")


def parse_params(stem):
    """Pull (ls, vr, thr, env, dr, ld) out of a world filename stem, so we can
    report training-set parameter coverage. Returns a dict of strings (kept as
    found, e.g. '10p0', '5em08') so we don't lose precision."""
    def grab(pat):
        m = re.search(pat, stem)
        return m.group(1) if m else None
    return {
        'ls':  grab(r'_ls([0-9p]+)_'),
        'vr':  grab(r'_vr([0-9p]+)_'),
        'thr': grab(r'_thr([0-9p]+)_'),
        'env': grab(r'_env([0-9]+)_'),
        'dr':  grab(r'_dr([0-9em]+)_'),
        'ld':  grab(r'_ld([0-9p]+)_'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--glob", default="*_training.npz")
    ap.add_argument("--eval-list", nargs="*", default=None)
    ap.add_argument("--eval-file", default=None,
                    help="text file, one eval world filename per line")
    ap.add_argument("--val-ratio", type=float, default=0.083)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sim = Path(args.sim_dir)

    # ---- 1. list + validity-filter all worlds (same as trainer) ----
    files = sorted(sim.glob(args.glob))
    if not files:
        raise SystemExit(f"No files match {args.glob} in {sim}")
    valid, skipped = [], []
    for f in files:
        try:
            if grid_shape_fast(str(f)) == (20, 20):
                valid.append(f.name)
            else:
                skipped.append(f.name)
        except Exception as e:
            skipped.append(f.name)
            print(f"  skip {f.name}: {e}")
    print(f"  glob '{args.glob}': {len(files)} files, "
          f"{len(valid)} valid 20x20, {len(skipped)} skipped")
    valid_set = set(valid)

    # ---- 2. gather the requested eval worlds ----
    eval_names = []
    if args.eval_file:
        eval_names += [ln.strip() for ln in open(args.eval_file)
                       if ln.strip()]
    if args.eval_list:
        eval_names += list(args.eval_list)
    if not eval_names:
        raise SystemExit("No eval worlds given (use --eval-list or --eval-file)")

    def norm(n):
        return n if n.endswith(".npz") else n + ".npz"
    eval_names = [norm(n) for n in eval_names]

    # STOP if any requested eval world is missing or invalid — never silent.
    missing = [n for n in eval_names if n not in valid_set]
    if missing:
        print("\n  ERROR: these requested eval worlds are NOT in the valid "
              "20x20 file set:")
        for n in missing:
            print(f"     {n}")
        raise SystemExit("Fix the eval list (typo? wrong stem? not 20x20?) "
                         "and re-run. Refusing to author a partial manifest.")

    test_set = set(eval_names)
    rest = [n for n in valid if n not in test_set]   # preserves sorted order

    # ---- 3. split the rest into train / val (val drawn with fixed seed) ----
    n_rest = len(rest)
    n_val = int(round(len(valid) * args.val_ratio))   # val as frac of TOTAL
    n_val = min(n_val, n_rest)                         # safety
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_rest)
    val_idx = set(perm[:n_val].tolist())
    val_names   = [rest[i] for i in range(n_rest) if i in val_idx]
    train_names = [rest[i] for i in range(n_rest) if i not in val_idx]

    print(f"  → test (held out, your choice): {len(test_set)}")
    print(f"  → val  (random, seed={args.seed}): {len(val_names)}")
    print(f"  → train:                          {len(train_names)}")

    # ---- 4. write the frozen manifest ----
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "split"])
        for n in train_names: w.writerow([n, "train"])
        for n in val_names:   w.writerow([n, "val"])
        for n in sorted(test_set): w.writerow([n, "test"])
    print(f"  ✓ wrote frozen manifest → {out}")

    # ---- 5. training-set parameter coverage report ----
    # For each parameter, list the values present in TRAIN and flag any value
    # that appears in TEST but NOT in TRAIN (a regime held out entirely).
    def values_for(names, key):
        vals = set()
        for n in names:
            v = parse_params(n[:-4] if n.endswith(".npz") else n)[key]
            if v is not None:
                vals.add(v)
        return vals

    print("\n  TRAINING-SET PARAMETER COVERAGE")
    print("  (values in TEST but absent from TRAIN = a regime held out whole)")
    any_gap = False
    for key in ['ls', 'vr', 'thr', 'env', 'dr', 'ld']:
        tr_vals = values_for(train_names, key)
        te_vals = values_for(sorted(test_set), key)
        gap = te_vals - tr_vals
        flag = ""
        if gap:
            any_gap = True
            flag = f"   <-- TEST-only values not in train: {sorted(gap)}"
        print(f"    {key:<4} train={sorted(tr_vals)}{flag}")
    if any_gap:
        print("\n  NOTE: at least one parameter value appears only in the test "
              "set. That is sometimes intentional (true extrapolation to an "
              "unseen regime), but if you wanted the model to have SEEN that "
              "regime during training, adjust your eval list.")
    else:
        print("\n  ✓ every parameter value in the test set also appears in "
              "training — no regime is held out wholesale.")

    return 0


if __name__ == "__main__":
    sys.exit(main())