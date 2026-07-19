# -----------------------------------------------------
# sweep_worlds_missing5.py
# -----------------------------------------------------
# Generates the 5 missing inference worlds (hardband × 2, connectivity × 2,
# wide_range × 1) using IDENTICAL IBM settings to your original sweep.
# Only the parameter combinations differ.
#
# These 5 worlds correspond to the missing entries in
# inference_worlds_manifest.csv:
#
#   hardband_01      ls=2.5,vr=0.004,thr=1,env=123,dr=1e-07,ld=0.12
#   hardband_02      ls=2.5,vr=0.004,thr=1,env=456,dr=1e-07,ld=0.12
#   connectivity_01  ls=10, vr=0.001,thr=1,env=123,dr=5e-08,ld=0.20
#   connectivity_02  ls=10, vr=0.001,thr=1,env=456,dr=5e-08,ld=0.20
#   wide_range_01    ls=2.5,vr=0.004,thr=3,env=456,dr=1e-07,ld=0.20
#
# Run with the same harness as sweep_worlds.py:
#   python sweep_worlds_missing5.py
# Or, if you want to dry-run first:
#   python sweep_worlds_missing5.py --limit 1
# -----------------------------------------------------
import itertools, subprocess, os, sys, argparse, shlex
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
    "--pool", "22510000",
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
    "--world-tag-extra", "batchA",
    "--fp16-time-series",
    "--obs-budgets", "1,5,10,20,50,100",
]


# Explicit list — each tuple is (ls, vr, thr, seed, dr, ld)
# Matches the 5 missing worlds reported by pick_inference_worlds.py.
MISSING_COMBOS = [
    (2.5,  0.004, 1.0, 123, 1e-07, 0.12),   # hardband_01
    (2.5,  0.004, 1.0, 456, 1e-07, 0.12),   # hardband_02
    (10.0, 0.001, 1.0, 123, 5e-08, 0.20),   # connectivity_01
    (10.0, 0.001, 1.0, 456, 5e-08, 0.20),   # connectivity_02
    (2.5,  0.004, 3.0, 456, 1e-07, 0.20),   # wide_range_01
]


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
    print(f"[GPU {gpu_id}] >>> {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def worker(gpu_id, cmds):
    for cmd in cmds:
        try:
            run_one(cmd, gpu_id)
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] [ERROR] {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N combos (for smoke tests)")
    args = parser.parse_args()

    combos = MISSING_COMBOS
    if args.limit is not None:
        combos = combos[:args.limit]
    print(f"[INFO] Total worlds to generate: {len(combos)}")
    for i, c in enumerate(combos):
        ls, vr, thr, seed, dr, ld = c
        print(f"  [{i+1}] ls={ls}, vr={vr}, thr={thr}, env={seed}, "
              f"dr={dr:.0e}, ld={ld}")

    cmds = [make_cmd(*p) for p in combos]
    gpus = get_gpu_list()
    print(f"[INFO] Using GPUs: {gpus}")
    chunks = [cmds[i::len(gpus)] for i in range(len(gpus))]

    procs = []
    for gpu_id, chunk in zip(gpus, chunks):
        if not chunk:
            continue
        p = Process(target=worker, args=(gpu_id, chunk), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[INFO] All missing worlds finished.")


if __name__ == "__main__":
    os.environ.setdefault("USE_GPU", "1")
    main()