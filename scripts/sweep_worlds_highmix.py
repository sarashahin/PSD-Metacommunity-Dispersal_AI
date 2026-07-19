# -----------------------------------------------------
# sweep_worlds_highmix.py  (extra high-mixing worlds)
# -----------------------------------------------------
import itertools, subprocess, os, argparse, shlex
from multiprocessing import Process
from pathlib import Path

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

# ---------- EXTRA high-mixing parameter block ------------------------------
LENS_HIGH  = [5, 10]
VARS_HIGH  = [0.002]
THRS_HIGH  = [3]
SEEDS      = [123, 456]
DRS_HIGH   = [5e-08, 1e-07]
LDD_HIGH   = [0.06, 0.12, 0.2]

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
    # IMPORTANT: NEW TAG to avoid overwriting existing batchA files
    "--world-tag-extra", "batchB_highmix",
    "--fp16-time-series",
    "--obs-budgets", "1,5,10,20,50,100",
]

def make_cmd(ls, vr, thr, seed, dr, ld):
    return base + [
        "--env-length-scale", str(ls),
        "--env-var-r",        str(vr),
        "--env-seed-field",   str(seed),
        "--ibm-detection-threshold", str(thr),
        "--disp-rate", str(dr),
        "--ldd-prob",  str(ld),
    ]

def run_one(cmd, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"[GPU {gpu_id}] >>> {cmd_str}", flush=True)
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
                        help="Run only the first N worlds (for smoke tests)")
    args = parser.parse_args()

    combos = list(itertools.product(
        LENS_HIGH, VARS_HIGH, THRS_HIGH, SEEDS, DRS_HIGH, LDD_HIGH
    ))
    if args.limit is not None:
        combos = combos[:args.limit]
    print(f"[INFO] Total extra worlds: {len(combos)}")

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
    print("[INFO] All extra worlds finished.")

if __name__ == "__main__":
    os.environ.setdefault("USE_GPU", "1")
    main()
