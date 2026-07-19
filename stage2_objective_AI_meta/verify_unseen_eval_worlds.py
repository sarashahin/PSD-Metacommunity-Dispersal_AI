#!/usr/bin/env python3
"""
=============================================================================
verify_unseen_eval_worlds.py
   — confirm the generated eval worlds are present, valid, and provably UNSEEN
=============================================================================
Run this AFTER sweep_worlds_eval30.py finishes. It checks, for every world in
the eval manifest:
  (A) the .npz exists in --sim-dir,
  (B) it is a valid 20x20 world with the arrays inference needs
      (P_last_final = truth labels; ENV_r_field = conditioning; at least one
       obs_mask_* = sparse observations; P_t present),
  (C) it is NOVEL: its parameter combo is NOT in the pre-training snapshot, and
      its env seed is NOT among the pre-training env seeds — i.e. the model
      could not have trained on it.
It also classifies each world in_dist vs extrap (by whether every value is in
the trained grid) and prints a clean PASS/FAIL summary.

USAGE
-----
  python verify_unseen_eval_worlds.py \
      --sim-dir   ./results/data \
      --manifest  ./eval_unseen_manifest.csv \
      --snapshot  ./pretraining_pool_combos.csv \
      --need-budget 5
=============================================================================
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

TRAINED = {  # the parameter VALUES the model trained on (slug strings)
    "ls":  {"2p5", "5p0", "10p0"},
    "vr":  {"0p001", "0p002", "0p004"},
    "thr": {"1p0", "3p0", "5p0"},
    "dr":  {"2em08", "5em08", "1em07"},
    "ld":  {"0p0", "0p06", "0p12", "0p2"},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--need-budget", type=int, default=5,
                    help="require obs_mask_<budget> to be present (K for inference)")
    ap.add_argument("--training-dir", default=None,
                    help="if given, confirm NO eval world file is present here "
                         "(physical separation from training data)")
    args = ap.parse_args()
    sim = Path(args.sim_dir)

    snap = set()
    snap_envs = set()
    with open(args.snapshot) as fh:
        for r in csv.DictReader(fh):
            snap.add((r["ls"], r["vr"], r["thr"], r["env"], r["dr"], r["ld"]))
            snap_envs.add(r["env"])

    rows = list(csv.DictReader(open(args.manifest)))
    print(f"  verifying {len(rows)} eval worlds against {len(snap)} "
          f"pre-training combos (pool env seeds={sorted(snap_envs)})\n")

    # The manifest stores NUMERIC values (e.g. ls=2.5, dr=1e-07) while the
    # snapshot + TRAINED sets use the filename SLUG form (2p5, 1em07). Convert
    # numeric -> slug so comparisons are like-for-like. env is an int either way.
    def _slug(x):
        return str(x).replace('.', 'p').replace('-', 'm')

    n_ok = 0
    problems = []
    for r in rows:
        fn = r["filename"]; p = sim / fn
        combo = (_slug(r["ls"]), _slug(r["vr"]), _slug(r["thr"]),
                 _slug(r["env"]), _slug(r["dr"]), _slug(r["ld"]))
        issues = []

        # (A) exists
        if not p.exists():
            issues.append("MISSING file")
        else:
            # (B) valid + has inference arrays
            try:
                with np.load(p, allow_pickle=True) as d:
                    keys = set(d.files)
                    Y = int(d["Y"]) if "Y" in keys else None
                    X = int(d["X"]) if "X" in keys else None
                    if "P_last_final" not in keys:
                        issues.append("no P_last_final (truth)")
                    else:
                        shp = np.asarray(d["P_last_final"]).shape
                        if len(shp) == 3:
                            Y = Y or shp[1]; X = X or shp[2]
                    if (Y, X) != (20, 20):
                        issues.append(f"grid {Y}x{X} != 20x20")
                    if "ENV_r_field" not in keys:
                        issues.append("no ENV_r_field (conditioning)")
                    if f"obs_mask_{args.need_budget}" not in keys:
                        have = sorted(k for k in keys if k.startswith("obs_mask_"))
                        issues.append(f"no obs_mask_{args.need_budget} "
                                      f"(have {have})")
                    if "P_t" not in keys:
                        issues.append("no P_t")
            except Exception as e:
                issues.append(f"unreadable: {e}")

        # (C) novelty
        if combo in snap:
            issues.append("NOT NOVEL — combo in pre-training snapshot")
        if combo[3] in snap_envs:
            issues.append(f"env seed {combo[3]} was in pre-training pool")

        # in_dist vs extrap classification
        is_extrap = (combo[0] not in TRAINED["ls"] or combo[1] not in TRAINED["vr"]
                     or combo[2] not in TRAINED["thr"] or combo[4] not in TRAINED["dr"]
                     or combo[5] not in TRAINED["ld"])
        regime = "extrap" if is_extrap else "in_dist"
        if regime != r.get("regime", regime):
            issues.append(f"regime mismatch: manifest={r.get('regime')} computed={regime}")

        if issues:
            problems.append((fn, issues))
        else:
            n_ok += 1

    print(f"  PASS: {n_ok}/{len(rows)} worlds present, valid for inference, and novel")
    if problems:
        print(f"  FAIL: {len(problems)} world(s) have problems:")
        for fn, iss in problems:
            print(f"     {fn}")
            for i in iss:
                print(f"        - {i}")
        return 1

    # optional: confirm physical separation from the training directory
    if args.training_dir:
        tdir = Path(args.training_dir)
        leaked_here = [r["filename"] for r in rows if (tdir / r["filename"]).exists()]
        if leaked_here:
            print(f"  FAIL: {len(leaked_here)} eval world(s) are ALSO present in "
                  f"the training dir {tdir} (not separated):")
            for n in leaked_here[:10]:
                print(f"     {n}")
            return 1
        print(f"  ✓ separation confirmed: no eval world is present in {tdir}")

    print("\n  ALL eval worlds verified: unseen, valid, ready for reconstruction.")
    # scenario / regime breakdown
    from collections import Counter
    print("  scenario:", dict(Counter(r["scenario"] for r in rows)))
    print("  regime  :", dict(Counter(r["regime"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())