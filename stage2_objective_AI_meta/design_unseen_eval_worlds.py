#!/usr/bin/env python3
"""
=============================================================================
design_unseen_eval_worlds.py  (v3)
   — design a CLEAN, UNSEEN evaluation set in a SEPARATE directory
=============================================================================

WHY THIS NEEDS NO RETRAINING AND HAS NO LEAKAGE
-----------------------------------------------
The model trained on an (unknown, now-lost) SUBSET of the worlds in
results/data. Any world whose full parameter combination (ls,vr,thr,env,dr,ld)
is NOT among those files cannot have been in the training split — it did not
exist when the model trained. The simplest guaranteed-novel axis is the
ENVIRONMENT SEED: the training pool has only env=123/456, so worlds with NEW
seeds (789, 2024) are fresh draws of the environment from the SAME process.

DIRECTORY SEPARATION (important safety property)
------------------------------------------------
The generated eval worlds are written to results/data/data_eval_unseen/, a
SUBDIRECTORY of results/data. The training loader globs results/data with a
NON-RECURSIVE pattern (Path.glob("*_training.npz")), which does NOT descend
into subdirectories. So eval worlds in data_eval_unseen/ are INVISIBLE to
training and can never accidentally re-enter the training pool.

THE 30-WORLD DESIGN (editable below in WORLD_DESIGN)
---------------------------------------------------
24 IN-DISTRIBUTION worlds (trained parameter values, NEW env seeds) across six
ecological regimes, + 6 EXTRAPOLATION worlds (NEW parameter values beyond the
trained grid, clearly labelled regime='extrap' and reported SEPARATELY):
  hardband        ls=2.5, vr=0.004           rugged env, narrow suitable bands
  connectivity    ls=10,  vr=0.001, ld=0.2   smooth env, strong long-distance link
  wide_range      thr=3,  dr=1e-07, ld=0.2   broad occupancy via high dispersal
  high_dispersal  dr=1e-07, ld=0.2           maximum movement
  low_dispersal   dr=2e-08, ld=0.0           minimal movement, fragmented ranges
  LDD             ld=0.2, dr=2e-08           connectivity by long jumps, low diffusion
  *_extrap        values beyond trained grid (ls=1/20, vr=0.008, dr=2e-07/1e-08, ld=0.3)

WHAT THIS SCRIPT DOES (it does NOT run the simulator)
-----------------------------------------------------
  1. reads existing worlds in --sim-dir, records their parameter combos,
  2. VERIFIES every designed world is novel + unique,
  3. predicts each output filename EXACTLY as run_all_rps builds it,
  4. writes:
       --out-sweep      (default scripts/sweep_worlds_eval30.py) — generates
                        ONLY these worlds and MOVES them into the eval dir,
       --out-manifest   (default <eval-dir>/eval_unseen_manifest.csv),
       --out-snapshot   (default <eval-dir>/pretraining_pool_combos.csv),
  5. prints a design report.
=============================================================================
"""
import argparse, csv, re, sys
from pathlib import Path

def slug(x):
    if x is None: return "NA"
    return str(x).replace('.', 'p').replace('-', 'm')

def build_stem(ls, vr, thr, env, dr, ld, pool="22510000", extra="batchA", Y=20, X=20):
    base = f"pool{pool}_{extra}"
    tag = (f"{base}_ls{slug(ls)}_vr{slug(vr)}_thr{slug(thr)}_env{slug(env)}"
           f"_grid{Y}x{X}_dr{slug(dr)}_ld{slug(ld)}")
    return tag.lower()

def combo_slug(ls, vr, thr, env, dr, ld):
    return (slug(ls), slug(vr), slug(thr), slug(env), slug(dr), slug(ld))

def parse_combo_from_name(name):
    s = name[:-4] if name.endswith(".npz") else name
    def g(p):
        m = re.search(p, s); return m.group(1) if m else None
    return (g(r'_ls([0-9p]+|na)_'), g(r'_vr([0-9p]+|na)_'), g(r'_thr([0-9p]+|na)_'),
            g(r'_env([0-9]+|na)_'), g(r'_dr([0-9em]+)_'), g(r'_ld([0-9p]+)_'))

# =============================================================================
# THE DESIGN — edit here. Each row: (scenario, regime, ls, vr, thr, env, dr, ld)
# All in_dist values are in the trained grid; all env seeds (789/2024) are NEW.
# All extrap rows use >=1 value OUTSIDE the trained grid (and a NEW env seed).
# =============================================================================
WORLD_DESIGN = [
    # ---- HARDBAND (rugged env, narrow bands): ls=2.5, vr=0.004 ----
    ("hardband",       "in_dist", 2.5,  0.004, 1.0, 789,  1e-07, 0.12),
    ("hardband",       "in_dist", 2.5,  0.004, 1.0, 2024, 1e-07, 0.12),
    ("hardband",       "in_dist", 2.5,  0.004, 3.0, 789,  1e-07, 0.12),
    ("hardband",       "in_dist", 2.5,  0.004, 3.0, 2024, 5e-08, 0.12),
    # ---- CONNECTIVITY (smooth env + long-distance link): ls=10, vr=0.001, ld=0.2 ----
    ("connectivity",   "in_dist", 10.0, 0.001, 1.0, 789,  5e-08, 0.2),
    ("connectivity",   "in_dist", 10.0, 0.001, 1.0, 2024, 5e-08, 0.2),
    ("connectivity",   "in_dist", 10.0, 0.001, 3.0, 789,  1e-07, 0.2),
    ("connectivity",   "in_dist", 10.0, 0.001, 3.0, 2024, 1e-07, 0.2),
    # ---- WIDE_RANGE (broad occupancy via high dispersal): thr=3, dr=1e-07, ld=0.2 ----
    ("wide_range",     "in_dist", 2.5,  0.004, 3.0, 789,  1e-07, 0.2),
    ("wide_range",     "in_dist", 5.0,  0.002, 3.0, 789,  1e-07, 0.2),
    ("wide_range",     "in_dist", 5.0,  0.002, 3.0, 2024, 1e-07, 0.2),
    ("wide_range",     "in_dist", 2.5,  0.002, 3.0, 2024, 1e-07, 0.2),
    # ---- HIGH_DISPERSAL (maximum movement): dr=1e-07, ld=0.2 ----
    ("high_dispersal", "in_dist", 10.0, 0.002, 1.0, 789,  1e-07, 0.2),
    ("high_dispersal", "in_dist", 10.0, 0.002, 1.0, 2024, 1e-07, 0.2),
    ("high_dispersal", "in_dist", 5.0,  0.002, 1.0, 789,  1e-07, 0.2),
    ("high_dispersal", "in_dist", 10.0, 0.004, 3.0, 2024, 1e-07, 0.2),
    # ---- LOW_DISPERSAL (minimal movement, fragmented): dr=2e-08, ld=0.0 ----
    ("low_dispersal",  "in_dist", 2.5,  0.004, 1.0, 789,  2e-08, 0.0),
    ("low_dispersal",  "in_dist", 5.0,  0.002, 1.0, 789,  2e-08, 0.0),
    ("low_dispersal",  "in_dist", 10.0, 0.001, 1.0, 2024, 2e-08, 0.0),
    ("low_dispersal",  "in_dist", 5.0,  0.002, 3.0, 2024, 2e-08, 0.0),
    # ---- LDD (connectivity by long jumps, low diffusion): ld=0.2, dr=2e-08 ----
    ("LDD",            "in_dist", 10.0, 0.001, 1.0, 789,  2e-08, 0.2),
    ("LDD",            "in_dist", 10.0, 0.001, 1.0, 2024, 2e-08, 0.2),
    ("LDD",            "in_dist", 5.0,  0.002, 1.0, 789,  2e-08, 0.2),
    ("LDD",            "in_dist", 5.0,  0.001, 3.0, 2024, 2e-08, 0.2),
    # ---- EXTRAPOLATION (NEW values beyond trained grid; report SEPARATELY) ----
    ("hardband_extrap",        "extrap", 1.0,  0.004, 1.0, 789, 1e-07, 0.12),  # ls=1.0 < trained min 2.5
    ("variance_extrap",        "extrap", 2.5,  0.008, 3.0, 789, 1e-07, 0.2),   # vr=0.008 > trained max
    ("ultra_high_disp_extrap", "extrap", 10.0, 0.002, 1.0, 789, 2e-07, 0.2),   # dr=2e-07 > trained max
    ("ultra_LDD_extrap",       "extrap", 10.0, 0.001, 1.0, 789, 5e-08, 0.3),   # ld=0.3 > trained max
    ("ultra_low_disp_extrap",  "extrap", 5.0,  0.002, 1.0, 789, 1e-08, 0.0),   # dr=1e-08 < trained min
    ("smooth_extrap",          "extrap", 20.0, 0.001, 1.0, 789, 5e-08, 0.2),   # ls=20 > trained max
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True,
                    help="directory of EXISTING training worlds (results/data)")
    ap.add_argument("--glob", default="*_training.npz")
    ap.add_argument("--pool", default="22510000")
    ap.add_argument("--world-tag-extra", default="batchA")
    ap.add_argument("--eval-dir", default="results/data/data_eval_unseen",
                    help="where the sweep will MOVE generated worlds")
    ap.add_argument("--raw-dir", default="results/data",
                    help="where run_all_rps initially writes (before the move)")
    ap.add_argument("--in-dist-only", action="store_true",
                    help="drop the 6 extrapolation worlds (keep 24 in-dist)")
    ap.add_argument("--out-sweep", default="scripts/sweep_worlds_eval30.py")
    ap.add_argument("--out-manifest", default=None)
    ap.add_argument("--out-snapshot", default=None)
    args = ap.parse_args()

    eval_dir = args.eval_dir.rstrip("/")
    out_manifest = args.out_manifest or f"{eval_dir}/eval_unseen_manifest.csv"
    out_snapshot = args.out_snapshot or f"{eval_dir}/pretraining_pool_combos.csv"

    sim = Path(args.sim_dir)
    existing_files = sorted(sim.glob(args.glob))
    existing_combos, existing_envs = set(), set()
    for f in existing_files:
        c = parse_combo_from_name(f.name)
        existing_combos.add(c)
        if c[3] not in (None, "na"):
            existing_envs.add(c[3])
    print(f"  existing pool: {len(existing_files)} files, "
          f"{len(existing_combos)} distinct combos, env seeds={sorted(existing_envs)}")

    design = [w for w in WORLD_DESIGN if not (args.in_dist_only and w[1] == "extrap")]

    # novelty + uniqueness + filename prediction
    seen, rows = set(), []
    for (scen, regime, ls, vr, thr, env, dr, ld) in design:
        cs = combo_slug(ls, vr, thr, env, dr, ld)
        fname = build_stem(ls, vr, thr, env, dr, ld,
                           pool=args.pool, extra=args.world_tag_extra) + "_training.npz"
        if slug(env) in existing_envs:
            raise SystemExit(f"env seed {env} already in training pool — pick a new seed "
                             f"(pool has {sorted(existing_envs)})")
        if cs in seen:
            raise SystemExit(f"INTERNAL DUPLICATE combo in design: {fname}")
        if cs in existing_combos:
            raise SystemExit(f"NOT NOVEL — combo already in training pool: {fname}")
        seen.add(cs)
        rows.append(dict(scenario=scen, regime=regime, ls=ls, vr=vr, thr=thr,
                         env=env, dr=dr, ld=ld, filename=fname))

    # representatives: first in_dist world per scenario (for the 4/6-panel figures)
    reps = {}
    for r in rows:
        if r["regime"] == "in_dist" and r["scenario"] not in reps:
            reps[r["scenario"]] = id(r)
    for r in rows:
        r["representative"] = "yes" if reps.get(r["scenario"]) == id(r) else "no"

    n_in = sum(1 for r in rows if r["regime"] == "in_dist")
    n_ex = sum(1 for r in rows if r["regime"] == "extrap")
    from collections import Counter
    print(f"  designed {len(rows)} NEW worlds: {n_in} in-distribution, {n_ex} extrapolation"
          f" — ALL novel, ALL unique")
    print("  in-dist scenarios:",
          {k: v for k, v in sorted(Counter(r['scenario'] for r in rows
                                            if r['regime']=='in_dist').items())})
    if n_ex:
        print("  extrapolation    :",
              {k: v for k, v in sorted(Counter(r['scenario'] for r in rows
                                               if r['regime']=='extrap').items())})

    # write snapshot of pre-existing combos (for the verifier)
    Path(out_snapshot).parent.mkdir(parents=True, exist_ok=True)
    with open(out_snapshot, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["ls", "vr", "thr", "env", "dr", "ld"])
        for c in sorted(existing_combos, key=lambda t: tuple("" if x is None else x for x in t)):
            w.writerow(c)
    print(f"  ✓ snapshot of existing combos → {out_snapshot}")

    # write eval manifest
    Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "scenario", "regime", "representative",
                    "ls", "vr", "thr", "env", "dr", "ld"])
        for r in rows:
            w.writerow([r["filename"], r["scenario"], r["regime"], r["representative"],
                        r["ls"], r["vr"], r["thr"], r["env"], r["dr"], r["ld"]])
    print(f"  ✓ eval manifest → {out_manifest}")
    print("  representatives (one per in-dist scenario; pick 4 or use all for figures):")
    for r in rows:
        if r["representative"] == "yes":
            print(f"     [{r['scenario']}] {r['filename']}")

    # write the sweep script (placeholder substitution — no brace escaping)
    combos_py = ",\n".join(
        f"    ({r['ls']!r}, {r['vr']!r}, {r['thr']!r}, {r['env']!r}, "
        f"{r['dr']!r}, {r['ld']!r}),   # {r['scenario']} [{r['regime']}]"
        for r in rows)
    sweep = (SWEEP_TEMPLATE
             .replace("__POOL__", args.pool)
             .replace("__EXTRA__", args.world_tag_extra)
             .replace("__RAW_DIR__", args.raw_dir)
             .replace("__EVAL_DIR__", eval_dir)
             .replace("__COMBOS__", combos_py))
    Path(args.out_sweep).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_sweep, "w") as fh:
        fh.write(sweep)
    print(f"  ✓ sweep script → {args.out_sweep}")
    print("\n  NEXT (run from the PROJECT ROOT so run_all_rps.py + results/ resolve):")
    print(f"     python {args.out_sweep} --limit 1     # smoke test ONE world")
    print(f"     python {args.out_sweep}                # full {len(rows)}-world run")
    print(f"  Worlds will be MOVED into: {eval_dir}/  (kept OUT of {args.raw_dir})")
    return 0


SWEEP_TEMPLATE = r'''# -----------------------------------------------------
# sweep_worlds_eval30.py  (AUTO-GENERATED by design_unseen_eval_worlds.py)
# -----------------------------------------------------
# Generates UNSEEN evaluation worlds at parameter combinations NOT in the
# training pool (NEW env seeds), with IDENTICAL IBM settings to sweep_worlds.py.
# After each world is generated, its output files are MOVED out of the raw
# results/data directory into the separate eval directory, so the training
# pool is never polluted.
#
# RUN FROM THE PROJECT ROOT (PSD_Dispersal_pool/) so that run_all_rps.py and
# results/ resolve correctly:
#     python scripts/sweep_worlds_eval30.py --limit 1     # smoke test
#     python scripts/sweep_worlds_eval30.py                # full run
# -----------------------------------------------------
import subprocess, os, sys, argparse, shlex, shutil
from pathlib import Path
from multiprocessing import Process


def get_gpu_list():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [g.strip() for g in cvd.split(",") if g.strip() != ""]
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        return [str(i) for i in range(n)]
    except Exception:
        return ["0"]


# IDENTICAL base command to sweep_worlds.py — do not change.
base = [
    "python", "run_all_rps.py",
    "--pool", "__POOL__",
    "--skip-psd", "--skip-ode",
    "--tmax", "10000", "--record", "200", "--no-movie",
    "--ibm-frac-multi", "0.05",
    "--ibm-window-steps", "1000",
    "--ibm-max-attempts", "10000",
    "--ibm-richness-cap", "None",
    "--save-env-field",
    "--ibm-record-mode", "full",
    "--ibm-F-sat", "None",
    "--ibm-max-rounds", "None",
    "--save-every-rounds", "500",
    "--save-every-seconds", "600",
    "--C-topk", "16",
    "--coocc-sample", "4096",
    "--world-tag-extra", "__EXTRA__",
    "--fp16-time-series",
    "--obs-budgets", "1,5,10,20,50,100",
]

# (ls, vr, thr, env_seed, dr, ld) — env seeds are NEW (not in training pool)
EVAL_COMBOS = [
__COMBOS__
]

RAW_DIR = Path("__RAW_DIR__")          # where run_all_rps writes first
EVAL_DIR = Path("__EVAL_DIR__")        # where we move the world (separate dir)


def _slug(x):
    if x is None:
        return "NA"
    return str(x).replace('.', 'p').replace('-', 'm')


def build_stem(ls, vr, thr, env, dr, ld):
    base_tag = "pool__POOL___" + "__EXTRA__"
    tag = (base_tag
           + "_ls" + _slug(ls) + "_vr" + _slug(vr) + "_thr" + _slug(thr)
           + "_env" + _slug(env) + "_grid20x20"
           + "_dr" + _slug(dr) + "_ld" + _slug(ld))
    return tag.lower()


def move_outputs(combo, gpu_id):
    ls, vr, thr, env, dr, ld = combo
    stem = build_stem(ls, vr, thr, env, dr, ld)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    moved = []
    for suffix in ("_training.npz", "_dataset.npz"):
        src = RAW_DIR / (stem + suffix)
        if src.exists():
            dst = EVAL_DIR / src.name
            shutil.move(str(src), str(dst))
            moved.append(dst.name)
    if moved:
        print("[GPU " + str(gpu_id) + "] moved -> " + str(EVAL_DIR) + ": "
              + ", ".join(moved), flush=True)
    else:
        print("[GPU " + str(gpu_id) + "] WARNING: no output found to move for "
              + stem, flush=True)
    return stem


def make_cmd(ls, vr, thr, seed, dr, ld):
    return base + [
        "--env-length-scale",        str(ls),
        "--env-var-r",               str(vr),
        "--env-seed-field",          str(seed),
        "--ibm-detection-threshold", str(thr),
        "--disp-rate",               str(dr),
        "--ldd-prob",                str(ld),
    ]


def run_one(cmd, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print("[GPU " + str(gpu_id) + "] >>> "
          + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def worker(gpu_id, items):
    for combo, cmd in items:
        try:
            run_one(cmd, gpu_id)
            move_outputs(combo, gpu_id)     # move OUT of results/data
        except subprocess.CalledProcessError as e:
            print("[GPU " + str(gpu_id) + "] [ERROR] " + str(e), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N combos (smoke test)")
    parser.add_argument("--raw-out-dir", default=None,
                        help="override where run_all_rps writes (default RAW_DIR)")
    parser.add_argument("--eval-out-dir", default=None,
                        help="override where worlds are moved (default EVAL_DIR)")
    args = parser.parse_args()
    global RAW_DIR, EVAL_DIR
    if args.raw_out_dir:
        RAW_DIR = Path(args.raw_out_dir)
    if args.eval_out_dir:
        EVAL_DIR = Path(args.eval_out_dir)

    combos = EVAL_COMBOS if args.limit is None else EVAL_COMBOS[:args.limit]
    print("[INFO] eval worlds to generate: " + str(len(combos)))
    print("[INFO] raw dir : " + str(RAW_DIR))
    print("[INFO] eval dir: " + str(EVAL_DIR))
    for i, c in enumerate(combos):
        ls, vr, thr, seed, dr, ld = c
        print("  [" + str(i + 1) + "] ls=" + str(ls) + " vr=" + str(vr)
              + " thr=" + str(thr) + " env=" + str(seed)
              + " dr=" + ("%.0e" % dr) + " ld=" + str(ld))
    cmds = [make_cmd(*p) for p in combos]
    items = list(zip(combos, cmds))
    gpus = get_gpu_list()
    print("[INFO] Using GPUs: " + str(gpus))
    chunks = [items[i::len(gpus)] for i in range(len(gpus))]
    procs = []
    for gpu_id, chunk in zip(gpus, chunks):
        if not chunk:
            continue
        p = Process(target=worker, args=(gpu_id, chunk), daemon=False)
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    print("[INFO] All eval worlds finished (moved into " + str(EVAL_DIR) + ").")


if __name__ == "__main__":
    os.environ.setdefault("USE_GPU", "1")
    main()
'''

if __name__ == "__main__":
    sys.exit(main())