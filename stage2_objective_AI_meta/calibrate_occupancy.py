#!/usr/bin/env python3
"""calibrate_occupancy.py — fix the diffuse over-prediction with a HONEST global calibration.

Fits a single monotone map  raw_prob -> P(occupied)  (binned isotonic / reliability-diagram
calibration) on a CALIBRATION manifest, applies it to a DISJOINT apply manifest, and reports
range-size KS per bucket BEFORE vs AFTER calibration. Calibration uses only the *aggregate*
occupancy frequency of raw probabilities — never per-species truth — so size match on the
held-out set is a real generalization result, not a tautology.

Estimator after calibration: EXPECTED_COUNT  (N = round(Σ calibrated p); top-N cells).
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ks_2samp

BUCKETS = [("EASY", 6, 10), ("MODERATE", 11, 20), ("HARD", 21, 10**9)]
NB = 200  # calibration bins over [0,1]

def world_mean_prob(samples_path):
    z = np.load(samples_path)
    return np.asarray(z["samples"]).astype(np.float32).mean(axis=0)   # (S, Y, X)

def truth_bin(truth_path):
    with np.load(truth_path, allow_pickle=True) as td:
        return (np.asarray(td["P_last_final"]) > 0.5).astype(np.uint8)

def accumulate_reliability(rows, truth_dir, pat, K, n_acc, pos_acc):
    """Bin raw probs; accumulate per-bin count and #occupied. Memory-bounded (NB bins)."""
    edges = np.linspace(0.0, 1.0, NB + 1)
    for r in rows:
        stem = r["filename"][:-4] if r["filename"].endswith(".npz") else r["filename"]
        sp = Path(pat.format(world_stem=stem)) / f"recon_fixed_b{K}_samples.npz"
        tp = Path(truth_dir) / r["filename"]
        if not (sp.exists() and tp.exists()): continue
        mp = world_mean_prob(sp); tb = truth_bin(tp)
        n_use = min(mp.shape[0], tb.shape[0]); mp = mp[:n_use]; tb = tb[:n_use]
        b = np.clip(np.digitize(mp.ravel(), edges) - 1, 0, NB - 1)
        np.add.at(n_acc, b, 1); np.add.at(pos_acc, b, tb.ravel())

def fit_isotonic(n_acc, pos_acc):
    centers = (np.linspace(0, 1, NB + 1)[:-1] + np.linspace(0, 1, NB + 1)[1:]) / 2
    m = n_acc > 0
    freq = np.zeros(NB); freq[m] = pos_acc[m] / n_acc[m]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(centers[m], freq[m], sample_weight=n_acc[m])
    return iso

def per_species_sizes(rows, truth_dir, pat, K, iso=None):
    """Return dict bucket->(truth sizes, predicted sizes) using EXPECTED_COUNT on (calibrated) prob."""
    truth_all, pred_all = [], []
    for r in rows:
        stem = r["filename"][:-4] if r["filename"].endswith(".npz") else r["filename"]
        sp = Path(pat.format(world_stem=stem)) / f"recon_fixed_b{K}_samples.npz"
        tp = Path(truth_dir) / r["filename"]
        if not (sp.exists() and tp.exists()): continue
        mp = world_mean_prob(sp); tb = truth_bin(tp)
        n_use = min(mp.shape[0], tb.shape[0]); mp = mp[:n_use]; tb = tb[:n_use]
        idx = [s for s in range(n_use) if int(tb[s].sum()) > K]
        if not idx: continue
        cp = mp.copy()
        if iso is not None:
            cp = iso.predict(mp.ravel()).reshape(mp.shape).astype(np.float32)
        for s in idx:
            truth_all.append(int(tb[s].sum()))
            pred_all.append(max(1, int(round(float(cp[s].sum())))))
    return np.array(truth_all), np.array(pred_all)

def ks_by_bucket(truth, pred, tag, out):
    print(f"\n  {tag}")
    for name, lo, hi in BUCKETS:
        m = (truth >= lo) & (truth <= hi)
        if m.sum() < 5: continue
        ks = ks_2samp(truth[m], pred[m]).statistic
        out.append({"set": tag, "bucket": name, "n": int(m.sum()), "ks_range": round(float(ks), 4)})
        print(f"    {name:9s} n={int(m.sum()):4d}  range-size KS = {ks:.3f}")
    ks = ks_2samp(truth, pred).statistic
    out.append({"set": tag, "bucket": "POOLED", "n": int(len(truth)), "ks_range": round(float(ks), 4)})
    print(f"    {'POOLED':9s} n={len(truth):4d}  range-size KS = {ks:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-manifest", required=True)
    ap.add_argument("--apply-manifest", required=True)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--recon-dir-pattern", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--output-csv", default=None)
    a = ap.parse_args()
    calib = [r for r in csv.DictReader(open(a.calib_manifest)) if r.get("found", "yes") == "yes"]
    apply = [r for r in csv.DictReader(open(a.apply_manifest)) if r.get("found", "yes") == "yes"]
    n_acc = np.zeros(NB); pos_acc = np.zeros(NB)
    accumulate_reliability(calib, a.truth_dir, a.recon_dir_pattern, a.K, n_acc, pos_acc)
    iso = fit_isotonic(n_acc, pos_acc)
    out = []
    t0, p0 = per_species_sizes(apply, a.truth_dir, a.recon_dir_pattern, a.K, iso=None)
    ks_by_bucket(t0, p0, "BEFORE (raw Σp)", out)
    t1, p1 = per_species_sizes(apply, a.truth_dir, a.recon_dir_pattern, a.K, iso=iso)
    ks_by_bucket(t1, p1, "AFTER (calibrated Σp)", out)
    if a.output_csv:
        with open(a.output_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["set", "bucket", "n", "ks_range"])
            w.writeheader(); w.writerows(out)
        print(f"\n  wrote {a.output_csv}")
    return 0
if __name__ == "__main__": sys.exit(main())