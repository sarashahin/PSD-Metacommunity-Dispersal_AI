#!/usr/bin/env python3
"""range_estimator_diagnostic_v2.py  (adds EXPECTED_COUNT, truth-free self-calibration)

Per-bucket range-size KS under honest, non-tautological predicted-range estimators:
  per_sample_p080 : pooled per-member cell counts at p>=thr            (current figure)
  mean_p050       : (ensemble mean >= 0.5) cell count                  (single best guess)
  ENS_UNION       : cell in ANY sample at p>=thr  (overshoots — kept for reference)
  EXPECTED_COUNT  : N = round(sum of ensemble-mean prob over grid); range = N
                    (self-consistent E[occupied]=Σp; scales with range; NO truth used)
Only range-SIZE KS is reported (that is the failing statistic). All estimators are truth-free.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

BUCKETS = [("EASY", 6, 10), ("MODERATE", 11, 20), ("HARD", 21, 10**9)]

def per_species(truth_path, samples_path, thr, K):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td["P_last_final"]) > 0.5).astype(np.uint8)
    z = np.load(samples_path); samples = np.asarray(z["samples"]).astype(np.float32)
    n_use = min(truth.shape[0], samples.shape[1]); truth = truth[:n_use]; samples = samples[:, :n_use]
    n_ens = samples.shape[0]
    idx = [s for s in range(n_use) if int(truth[s].sum()) > K]
    if not idx: return None
    truth_m = truth[idx]; samp_m = samples[:, idx]
    tr = truth_m.sum(axis=(1, 2)).astype(int)
    binar = (samp_m >= thr).astype(np.uint8)
    persamp = binar.sum(axis=(2, 3))
    mean_prob = samp_m.mean(axis=0)                              # (n_sp, Y, X)
    exp_count = np.clip(np.round(mean_prob.sum(axis=(1, 2))), 1, None).astype(int)
    return {"truth": tr,
            "per_sample_p080": persamp.reshape(-1),
            "mean_p050": (mean_prob >= 0.5).sum(axis=(1, 2)).astype(int),
            "ENS_UNION": binar.max(axis=0).sum(axis=(1, 2)).astype(int),
            "EXPECTED_COUNT": exp_count,
            "per_sample_srctr": np.repeat(tr[None, :], n_ens, axis=0).reshape(-1)}

def buck(v, src, lo, hi): return v[(src >= lo) & (src <= hi)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--recon-dir-pattern", required=True); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.80); ap.add_argument("--output-csv", default=None)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.manifest)) if r.get("found", "yes") == "yes"]
    ests = ["per_sample_p080", "mean_p050", "ENS_UNION", "EXPECTED_COUNT"]
    pool = {k: [] for k in ["truth"] + ests}; ps_src = []
    for r in rows:
        stem = r["filename"][:-4] if r["filename"].endswith(".npz") else r["filename"]
        sp = Path(a.recon_dir_pattern.format(world_stem=stem)) / f"recon_fixed_b{a.K}_samples.npz"
        tp = Path(a.truth_dir) / r["filename"]
        if not (sp.exists() and tp.exists()): print(f"  skip: {stem[:55]}"); continue
        d = per_species(tp, sp, a.threshold, a.K)
        if d is None: continue
        for k in pool: pool[k].append(d[k])
        ps_src.append(d["per_sample_srctr"])
    for k in pool: pool[k] = np.concatenate(pool[k]) if pool[k] else np.array([])
    ps_src = np.concatenate(ps_src) if ps_src else np.array([]); truth = pool["truth"]
    out = []
    print(f"\n  Range-size KS by bucket  (K={a.K}, thr={a.threshold}, {len(truth)} meaningful species)\n")
    hdr = f"  {'bucket':9s} {'n_truth':>7s} | " + " | ".join(f"{e:>15s}" for e in ests)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, lo, hi in BUCKETS:
        t = buck(truth, truth, lo, hi)
        if len(t) < 5: print(f"  {name:9s} {len(t):7d} |  (insufficient)"); continue
        cells = []
        for e in ests:
            src = ps_src if e == "per_sample_p080" else truth
            p = buck(pool[e], src, lo, hi)
            ks = ks_2samp(t, p).statistic if len(p) >= 5 else float("nan"); cells.append(ks)
            out.append({"bucket": name, "estimator": e, "n_truth": len(t), "n_pred": int(len(p)),
                        "ks_range": round(float(ks), 4)})
        print(f"  {name:9s} {len(t):7d} | " + " | ".join(f"{c:15.3f}" for c in cells))
    cells = []
    for e in ests:
        p = pool[e]; ks = ks_2samp(truth, p).statistic if len(p) >= 5 else float("nan"); cells.append(ks)
        out.append({"bucket": "POOLED", "estimator": e, "n_truth": len(truth), "n_pred": int(len(p)),
                    "ks_range": round(float(ks), 4)})
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'POOLED':9s} {len(truth):7d} | " + " | ".join(f"{c:15.3f}" for c in cells))
    if a.output_csv:
        with open(a.output_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["bucket","estimator","n_truth","n_pred","ks_range"])
            w.writeheader(); w.writerows(out)
        print(f"\n  wrote {a.output_csv}")
    return 0
if __name__ == "__main__": sys.exit(main())