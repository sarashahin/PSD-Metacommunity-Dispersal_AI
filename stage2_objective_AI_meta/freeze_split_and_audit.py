#!/usr/bin/env python3
"""
=============================================================================
freeze_split_and_audit.py  —  FIX FOR THE EVAL/TRAIN LEAKAGE
=============================================================================

WHAT WENT WRONG
---------------
The train/val/test split in data_preprocessing.create_dataloaders is computed
ON THE FLY every run:

    npz_files   = sorted(sim_path.glob(config.paths.npz_pattern))   # 251 files
    valid_paths = [f for f in npz_files if grid == (20,20)]
    np.random.seed(config.training.seed)                             # seed=42
    indices     = np.random.permutation(len(valid_paths))
    train = indices[:n_train]; val = [...]; test = indices[n_train+n_val:]

That is deterministic ONLY while the directory contents and filenames are
byte-for-byte identical. It is never written down. So when the 30 evaluation
worlds were chosen (by ecological interest, from the inference manifest),
nothing checked them against the test split — and 25 of 30 turned out to be
TRAIN worlds. Metrics on those 25 are contaminated by memorisation.

WHAT THIS SCRIPT DOES
---------------------
1. Recomputes the EXACT split the training used (same glob, same validity
   filter, same seed, same slice math as data_preprocessing.py 736-749).
2. WRITES it to disk as split_manifest.csv  (filename,split) — a frozen,
   auditable artifact. This is the thing that should have existed all along.
3. Audits a list of evaluation worlds against the frozen split and prints
   exactly which are LEAKED (in train/val) and which are CLEAN (in test).
4. Writes eval_world_audit.csv with the per-world verdict.

This script CHANGES NOTHING about the model or the existing figures. It only
produces the two manifests + a verdict. The actual code change that PREVENTS
recurrence is in patch_create_dataloaders_frozen_split.py (separate file).

USAGE
-----
  # from PSD_Dispersal_pool/ (same dir you run training from)
  python freeze_split_and_audit.py \
      --sim-dir       ./results/data \
      --stage2-dir    AI_simulation/stage2 \
      --eval-manifest figures_map_axel_stage2_new/inference_worlds_manifest_REV2.csv \
      --out-dir       ./split_audit

  # If you have no eval manifest handy, pass the eval worlds directly:
  python freeze_split_and_audit.py --sim-dir ./results/data \
      --stage2-dir AI_simulation/stage2 \
      --eval-list world1.npz world2.npz ... \
      --out-dir ./split_audit
=============================================================================
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np


def load_config(stage2_dir):
    """Import the project's real config so we use the SAME ratios/seed/pattern
    the training used — never hard-coded paraphrases."""
    sys.path.insert(0, str(Path(stage2_dir).resolve()))
    from configs.config import get_default_config
    return get_default_config()


def get_grid_shape_fast(npz_path):
    """Cheap grid-shape probe mirroring get_world_metadata_fast's intent:
    we only need to know whether the world is 20x20 so the validity filter
    matches training. We read Y and X if present, else infer from P_last_final.
    """
    with np.load(npz_path, allow_pickle=True) as d:
        if 'Y' in d and 'X' in d:
            return (int(d['Y']), int(d['X']))
        # fall back to the last-frame presence array shape
        if 'P_last_final' in d:
            arr = np.asarray(d['P_last_final'])
            if arr.ndim == 3:           # (S, Y, X)
                return (arr.shape[1], arr.shape[2])
        if 'P_t' in d:
            arr = np.asarray(d['P_t'])
            if arr.ndim == 4:           # (T, S, Y, X)
                return (arr.shape[2], arr.shape[3])
    raise ValueError(f"cannot determine grid shape for {npz_path}")


def reproduce_split(valid_paths, cfg):
    """EXACT reproduction of data_preprocessing.create_dataloaders 736-749."""
    n_total = len(valid_paths)
    n_train = int(n_total * cfg.data.train_ratio)
    n_val   = int(n_total * cfg.data.val_ratio)
    if n_total >= 3 and n_val == 0:
        n_val = 1
        n_train = n_train - 1 if n_train > 1 else n_train
    np.random.seed(cfg.training.seed)
    indices = np.random.permutation(n_total)
    train = [valid_paths[i] for i in indices[:n_train]]
    val   = [valid_paths[i] for i in indices[n_train:n_train + n_val]]
    test  = [valid_paths[i] for i in indices[n_train + n_val:]]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True,
                    help="directory of *_training.npz worlds (results/data)")
    ap.add_argument("--stage2-dir", required=True,
                    help="path containing configs/config.py")
    ap.add_argument("--eval-manifest", default=None,
                    help="CSV with 'filename' + (optional) 'found' columns")
    ap.add_argument("--eval-list", nargs="*", default=None,
                    help="explicit list of eval world filenames (alt. to manifest)")
    ap.add_argument("--out-dir", default="./split_audit")
    args = ap.parse_args()

    cfg = load_config(args.stage2_dir)
    sim_path = Path(args.sim_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Recompute the validity-filtered, sorted file list (= training) ----
    npz_files = sorted(sim_path.glob(cfg.paths.npz_pattern))
    if not npz_files:
        raise SystemExit(f"No files match {cfg.paths.npz_pattern} in {sim_path}")
    print(f"  glob '{cfg.paths.npz_pattern}' → {len(npz_files)} files")

    valid_paths = []
    skipped = 0
    for f in npz_files:
        try:
            if get_grid_shape_fast(str(f)) == (20, 20):
                valid_paths.append(str(f))
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"    skip {f.name}: {e}")
    print(f"  valid 20x20 worlds: {len(valid_paths)}  (skipped {skipped})")

    # ---- 2. Reproduce the split and FREEZE it to disk ----
    train, val, test = reproduce_split(valid_paths, cfg)
    print(f"  split: {len(train)} train / {len(val)} val / {len(test)} test "
          f"(seed={cfg.training.seed})")

    split_csv = out_dir / "split_manifest.csv"
    with open(split_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "split"])
        for p in train: w.writerow([Path(p).name, "train"])
        for p in val:   w.writerow([Path(p).name, "val"])
        for p in test:  w.writerow([Path(p).name, "test"])
    print(f"  ✓ froze split → {split_csv}")

    train_names = {Path(p).name for p in train}
    val_names   = {Path(p).name for p in val}
    test_names  = {Path(p).name for p in test}

    # ---- 3. Gather the evaluation world list ----
    eval_names = []
    if args.eval_manifest:
        with open(args.eval_manifest) as fh:
            for r in csv.DictReader(fh):
                fn = r.get("filename")
                if not fn:
                    continue
                # honour a 'found' column if present (yes/true/1)
                if "found" in r and str(r["found"]).lower() not in (
                        "yes", "true", "1", ""):
                    continue
                eval_names.append(fn)
    elif args.eval_list:
        eval_names = list(args.eval_list)
    else:
        print("  (no eval manifest/list given — split frozen, audit skipped)")
        return 0

    # normalise: ensure .npz suffix for comparison
    def norm(n):
        return n if n.endswith(".npz") else n + ".npz"
    eval_names = [norm(n) for n in eval_names]

    # ---- 4. Audit ----
    rows = []
    n_leak = n_clean = n_unknown = 0
    for n in eval_names:
        if n in train_names:
            verdict = "LEAKED_train"; n_leak += 1
        elif n in val_names:
            verdict = "LEAKED_val"; n_leak += 1
        elif n in test_names:
            verdict = "CLEAN_test"; n_clean += 1
        else:
            verdict = "NOT_IN_ANY_SPLIT"; n_unknown += 1
        rows.append((n, verdict))

    audit_csv = out_dir / "eval_world_audit.csv"
    with open(audit_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "verdict"])
        w.writerows(rows)

    print(f"\n  EVAL AUDIT  ({len(eval_names)} worlds)")
    print(f"    CLEAN (in test split):   {n_clean}")
    print(f"    LEAKED (train/val):      {n_leak}")
    if n_unknown:
        print(f"    NOT in any split:        {n_unknown}  "
              f"(filename mismatch or new file — investigate)")
    print(f"  ✓ wrote per-world verdict → {audit_csv}")

    print("\n  CLEAN worlds you CAN evaluate on:")
    for n, v in rows:
        if v == "CLEAN_test":
            print(f"     {n}")
    if n_clean == 0:
        print("     (none — see remediation note in script header)")

    return 0


if __name__ == "__main__":
    sys.exit(main())