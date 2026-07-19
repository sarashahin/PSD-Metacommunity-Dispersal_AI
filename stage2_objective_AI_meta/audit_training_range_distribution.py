#!/usr/bin/env python3
"""audit_training_range_distribution.py
Decide — with certainty, not a guess — whether the wide-range (HARD) failure is a
DATA problem (training pool thin in wide-range species) or an INFORMATION-LIMIT
problem (training already covers wide ranges; 5-10 obs simply can't pin them down).

It pools the per-species range sizes of EVERY species across all training worlds and
compares that distribution to the eval-truth distribution. The decisive quantities:
  (1) ABSOLUTE count of HARD (range>=21) species the model saw in training
      — "did it see enough examples" — this is range-K-independent.
  (2) TAIL COVERAGE: does the training range distribution reach as far as the eval
      HARD ranges? If training never contained a 40-cell range, the model cannot learn one.
  (3) SHAPE: KS + bucket fractions, training vs eval meaningful species.

Training glob is NON-recursive (results/data/*_training.npz), so eval worlds living in
results/data/data_eval_unseen/ are excluded automatically. Restricted to the eval grid
(20x20) for an apples-to-apples comparison; other grids are counted and skipped.
"""
import argparse, csv, glob, os, sys
import numpy as np
from scipy.stats import ks_2samp

BUCKETS = [("EASY", 6, 10), ("MODERATE", 11, 20), ("HARD", 21, 10**9)]

def world_ranges(path, grid):
    try:
        with np.load(path, allow_pickle=True) as d:
            if "P_last_final" not in d.files: return None, "no_P_last_final"
            P = np.asarray(d["P_last_final"])
    except Exception as e:
        return None, f"load_error:{e}"
    if P.ndim != 3: return None, f"ndim={P.ndim}"
    if tuple(P.shape[1:]) != tuple(grid): return None, f"grid={P.shape[1:]}"
    r = (P > 0.5).reshape(P.shape[0], -1).sum(1).astype(int)
    return r, "ok"

def collect_train(train_dir, grid):
    files = sorted(glob.glob(os.path.join(train_dir, "*_training.npz")))
    allr, nw, skip_grid, skip_other = [], 0, 0, 0
    for i, f in enumerate(files):
        r, why = world_ranges(f, grid)
        if r is None:
            if why.startswith("grid"): skip_grid += 1
            else: skip_other += 1
            continue
        allr.append(r); nw += 1
        if (i + 1) % 50 == 0: print(f"    ...scanned {i+1}/{len(files)} training files")
    return (np.concatenate(allr) if allr else np.array([], int)), nw, skip_grid, skip_other, len(files)

def collect_eval(manifest, truth_dir, grid):
    rows = [r for r in csv.DictReader(open(manifest)) if r.get("found", "yes") == "yes"]
    allr, nw = [], 0
    for r in rows:
        f = os.path.join(truth_dir, r["filename"])
        if not os.path.exists(f): continue
        rr, why = world_ranges(f, grid)
        if rr is None: continue
        allr.append(rr); nw += 1
    return (np.concatenate(allr) if allr else np.array([], int)), nw

def buckets_of(ranges, K):
    meaningful = ranges[ranges > K]
    out = {}
    for name, lo, hi in BUCKETS:
        out[name] = int(((meaningful >= lo) & (meaningful <= hi)).sum())
    out["_meaningful"] = int(len(meaningful))
    out["_present"] = int((ranges > 0).sum())
    out["_total"] = int(len(ranges))
    return out, meaningful

def pct(a, q): return float(np.percentile(a, q)) if len(a) else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--eval-truth-dir", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--grid", type=int, nargs=2, default=[20, 20])
    ap.add_argument("--output-csv", default=None)
    a = ap.parse_args()
    grid = tuple(a.grid)

    print(f"\n  Scanning training pool: {a.train_dir}/*_training.npz  (non-recursive)")
    tr, ntw, sg, so, nf = collect_train(a.train_dir, grid)
    print(f"  Training: {ntw} worlds used, {sg} skipped (other grid), {so} skipped (other), {nf} files seen")
    ev, nev = collect_eval(a.eval_manifest, a.eval_truth_dir, grid)
    print(f"  Eval:     {nev} worlds")

    tb, tr_m = buckets_of(tr, a.K)
    eb, ev_m = buckets_of(ev, a.K)
    if tb["_meaningful"] < 5 or eb["_meaningful"] < 5:
        print("  Not enough meaningful species to compare."); return 1

    print(f"\n  Meaningful species (range > K={a.K}):  train={tb['_meaningful']:,}   eval={eb['_meaningful']:,}")
    print(f"  {'bucket':9s} | {'train n':>9s} {'train %':>8s} | {'eval n':>8s} {'eval %':>7s} | train/eval %-ratio")
    print("  " + "-" * 70)
    rows = []
    for name, lo, hi in BUCKETS:
        tfrac = tb[name] / tb["_meaningful"]; efrac = eb[name] / eb["_meaningful"]
        ratio = (tfrac / efrac) if efrac > 0 else float("inf")
        print(f"  {name:9s} | {tb[name]:9,d} {tfrac:7.1%} | {eb[name]:8,d} {efrac:6.1%} | {ratio:6.2f}x")
        rows.append({"bucket": name, "train_n": tb[name], "train_frac": round(tfrac, 4),
                     "eval_n": eb[name], "eval_frac": round(efrac, 4), "frac_ratio": round(ratio, 3)})

    # tail coverage + KS
    ks = ks_2samp(tr_m, ev_m).statistic
    t_p99, t_max = pct(tr_m, 99), int(tr_m.max())
    e_p95H = pct(ev_m[ev_m >= 21], 95) if (ev_m >= 21).sum() else float("nan")
    e_max = int(ev_m.max())
    print("  " + "-" * 70)
    print(f"  range KS (train vs eval meaningful) = {ks:.3f}")
    print(f"  training tail:  99th pct = {t_p99:.0f} cells,  max = {t_max} cells")
    print(f"  eval HARD tail: 95th pct(of HARD) = {e_p95H:.0f} cells,  eval max = {e_max} cells")

    # ---- decisive verdict ----
    # The verdict rests on REPRESENTATIVENESS, not raw count: (1) does training reach
    # as wide as eval (tail), and (2) is the wide fraction comparable (not <half eval)?
    # Absolute count is reported as context only — a matched distribution with a covered
    # tail means the model saw a representative sample even if the count is modest.
    hard_n_train = tb["HARD"]
    tail_covers = (t_p99 >= (e_p95H if not np.isnan(e_p95H) else 1e9)) and (t_max >= 0.9 * e_max)
    thin_frac = ((tb["HARD"] / tb["_meaningful"]) < 0.5 * (eb["HARD"] / eb["_meaningful"])) if eb["HARD"] else False
    data_thin = (not tail_covers) or thin_frac
    print("\n  VERDICT")
    print(f"    • {hard_n_train:,} HARD (>=21) species seen across the whole training pool (context).")
    if not tail_covers:
        print(f"    • TAIL GAP: training tops out near {t_max} cells but eval needs up to {e_max} — "
              f"the model was rarely/never shown ranges that wide.")
    else:
        print(f"    • TAIL OK: training ranges reach as wide as eval (max {t_max} vs {e_max}).")
    if thin_frac:
        print(f"    • Training is proportionally THIN in wide species (<half the eval HARD fraction).")
    else:
        print(f"    • Wide-species FRACTION is comparable to eval (not proportionally thin).")
    if hard_n_train < 100 and not data_thin:
        print(f"    • Note: only {hard_n_train} HARD examples — representative but few; enrichment optional.")
    print()
    if data_thin:
        print("    => DATA is a plausible lever: the training pool under-represents wide-range species.")
        print("       Targeted enrichment (more worlds in wide-range regimes: high dispersal / high")
        print("       connectance / smooth env) has a real mechanism and is worth trying BEFORE a retrain.")
        verdict = "DATA_CANDIDATE"
    else:
        print("    => Training ALREADY covers the wide-range regime in count and tail. The HARD-bucket")
        print("       limit is therefore INFORMATIONAL (5-10 obs under-determine wide ranges); more data")
        print("       of the same kind will not help. Report it as the information limit.")
        verdict = "INFORMATION_LIMIT"

    if a.output_csv:
        with open(a.output_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["bucket","train_n","train_frac","eval_n","eval_frac","frac_ratio"])
            w.writeheader(); w.writerows(rows)
            w.writerow({}); 
        with open(a.output_csv, "a") as fh:
            fh.write(f"# range_KS_train_vs_eval,{ks:.4f}\n")
            fh.write(f"# train_p99,{t_p99:.1f},train_max,{t_max}\n")
            fh.write(f"# eval_HARD_p95,{e_p95H:.1f},eval_max,{e_max}\n")
            fh.write(f"# train_HARD_count,{hard_n_train}\n")
            fh.write(f"# verdict,{verdict}\n")
        print(f"\n  wrote {a.output_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())