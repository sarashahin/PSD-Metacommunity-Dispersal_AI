# -----------------------------------------------------
# sweep_worlds.py  (GPU-aware, one worker per GPU)
# -----------------------------------------------------
import itertools, subprocess, os, sys, argparse, shlex
from multiprocessing import Process
from pathlib import Path

# ---------- NEW: detect visible GPUs safely ------------------------------  
def get_gpu_list():
    # Respect pre-set CUDA_VISIBLE_DEVICES if present
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [g.strip() for g in cvd.split(",") if g.strip() != ""]
    # Fallback: try CuPy to count devices
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        return [str(i) for i in range(n)]
    except Exception:
        return ["0"]  # single GPU

# ---------- parameters ----------------------------------------------------
# LENS  = [2.5, 5, 10]
# VARS  = [0.001, 0.002, 0.004]
# THRS  = [1, 3, 5]
# SEEDS = [123,456]
# DRS   = [2e-08, 5e-08]
# LDD   = [0.0, 0.06]
LENS = [2.5,5, 10]
VARS  = [0.001,0.002, 0.004]
THRS  = [1,3,5]
SEEDS      = [123, 456]
DRS   = [2e-08, 5e-08, 1e-07]
LDD   = [0.0, 0.06, 0.12, 0.2]


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
    "--fp16-time-series",                        # ←  save IBM_B as float16 in training file
    "--obs-budgets", "1,5,10,20,50,100",         # ←  richer observation budgets
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
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)     # ← NEW: pin to one GPU
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
                        help="Run only the first N worlds (for smoke tests)")  # ← NEW
    args = parser.parse_args()

    combos = list(itertools.product(LENS, VARS, THRS, SEEDS, DRS, LDD))
    if args.limit is not None:
        combos = combos[:args.limit]
    print(f"[INFO] Total worlds: {len(combos)}")

    cmds = [make_cmd(*p) for p in combos]

    gpus = get_gpu_list()
    print(f"[INFO] Using GPUs: {gpus}")
    # round-robin split across GPUs
    chunks = [cmds[i::len(gpus)] for i in range(len(gpus))]

    procs = []
    for gpu_id, chunk in zip(gpus, chunks):
        if not chunk: continue
        p = Process(target=worker, args=(gpu_id, chunk), daemon=False)
        p.start()
        procs.append(p)

    for p in procs: p.join()
    print("[INFO] All worlds finished.")

if __name__ == "__main__":
    # safer default
    os.environ.setdefault("USE_GPU", "1")
    main()





