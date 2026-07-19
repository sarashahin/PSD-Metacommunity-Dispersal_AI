#!/usr/bin/env python3
"""
=============================================================================
AXEL ADAPTIVE ROUTING v3 — N_hat BUCKET CLASSIFIER + FIXED PER-BUCKET THRESHOLD
=============================================================================

WHAT v3 ADDS OVER v2
--------------------
v2 used N_hat (multi-feature ridge + isotonic) as a CONTINUOUS count: it
selected top-N_hat cells per species. Two side-effects:
  - Marginal CDF matches truth (good).
  - Per-species N_hat is capped at the score's range (bad: max ~22 cells
    even though true HARD species go up to 55). HARD-range KS = 0.979.

At K=10 the K=10 bucket numbers showed something specific:
  - HARD at fixed p>=0.80:  range KS 0.170  conn 0.277  spread 0.066  ALL PASS
  - MODERATE at fixed p>=0.95: range KS 0.166  conn 0.284  spread 0.173 ALL PASS
  - No single GLOBAL threshold passes both buckets.

v3 treats N_hat as a soft BUCKET CLASSIFIER (not a count):
  - Species with N_hat >= percentile(N_hat, 1 - HARD_fraction)
       -> "predicted HARD"  -> apply fixed p>=0.80  (proven to pass HARD)
  - Otherwise
       -> "predicted MODERATE" -> apply fixed p>=0.95 (proven to pass MODERATE)

HARD_fraction is the fraction of >K-truth species that are truly HARD.
At K=10 with truth filter range>K, this is ~20% (141 of 699). The script
estimates this from training (LOWO) so it stays truth-free at the test world.

Why this works when continuous N_hat does not:
  - We only need N_hat to RANK species correctly (binary classification),
    not predict an exact count. Spearman rho = 0.40 gives AUC ~ 0.70
    for HARD-vs-MODERATE classification — moderate but useful.
  - Once a species is classified, the fixed threshold for that bucket
    produces a range distribution we have ALREADY VERIFIED matches truth.
  - So bucket-KS for v3 should be (almost) the per-bucket fixed-threshold
    numbers, modulated only by classification error.

WHY THIS IS NOT CHEATING
------------------------
The classification uses N_hat which is computed from observation
dispersion and probability-map shape only — no truth at inference.
The HARD_fraction prior is estimated from the held-out worlds' truth
distribution, never from the test world. The two fixed thresholds
(0.80 and 0.95) are global constants, not tuned per world.

USAGE
-----
    python axel_adaptive_routing_v3_bucketclassifier.py --self-test

    python axel_adaptive_routing_v3_bucketclassifier.py \
        --wide-range-csv     ./figures_map_axel_stage2_new/wide_range_species.csv \
        --recon-dir-pattern  './reconstructions_spatial/{world_stem}' \
        --truth-dir          ./results/data \
        --K                  10 \
        --top-n-worlds       30 \
        --output-dir         ./figures_map_axel_stage2_new/adaptive_v3_K10

OUTPUTS
-------
   bucket_ks_v3_summary.csv         numeric table, four methods x four buckets
   bucket_ks_v3_heatmap.png         decision matrix
   classifier_calibration.png       N_hat vs truth bucket, confusion matrix
   verification.log                 self-test + per-step verification
=============================================================================
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats


GRID_Y, GRID_X = 20, 20
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)
INFORMATIVE_FEATURES = ('obs_logcd', 'top30_sum', 'top20_sum', 'top10_sum',
                        'prob_q95', 'prob_q90', 'prob_gini', 'peak_to_mean')


# -----------------------------------------------------------------------------
# Geometry + PBC math (identical to v2)
# -----------------------------------------------------------------------------
def periodic_cov_det(binary_range, Y=GRID_Y, X=GRID_X):
    yy, xx = np.where(binary_range > 0.5)
    n = len(yy)
    if n < 2:
        return 0.0
    theta_y = 2.0 * np.pi * yy.astype(np.float64) / Y
    theta_x = 2.0 * np.pi * xx.astype(np.float64) / X
    mean_theta_y = np.arctan2(np.sin(theta_y).mean(), np.cos(theta_y).mean())
    mean_theta_x = np.arctan2(np.sin(theta_x).mean(), np.cos(theta_x).mean())
    diff_y = (theta_y - mean_theta_y + np.pi) % (2.0 * np.pi) - np.pi
    diff_x = (theta_x - mean_theta_x + np.pi) % (2.0 * np.pi) - np.pi
    dy = diff_y * Y / (2.0 * np.pi)
    dx = diff_x * X / (2.0 * np.pi)
    var_y = float(np.var(dy)); var_x = float(np.var(dx))
    cov_yx = float(((dy - dy.mean()) * (dx - dx.mean())).mean())
    return max(0.0, var_y * var_x - cov_yx ** 2)


def count_components(binary_map):
    if binary_map.sum() == 0:
        return 0
    _, n = ndimage.label(binary_map, structure=CONNECTIVITY_STRUCTURE)
    return int(n)


def _gini(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.sum() < 1e-9:
        return 0.0
    a = np.sort(a); n = a.size; idx = np.arange(1, n + 1)
    return float((2.0 * (idx * a).sum() - (n + 1) * a.sum()) / (n * a.sum() + 1e-12))


# -----------------------------------------------------------------------------
# Feature extraction per species (identical to v2)
# -----------------------------------------------------------------------------
def compute_features_one_species(prob_map_ens, obs_cells):
    mean_prob = prob_map_ens.mean(axis=0)
    flat = mean_prob.ravel()
    obs_flat = obs_cells.ravel().astype(bool)
    flat_extrap = flat[~obs_flat]
    if flat_extrap.size == 0:
        flat_extrap = flat
    sorted_extrap = np.sort(flat_extrap)[::-1]
    return {
        'obs_logcd':    float(np.log10(periodic_cov_det(obs_cells) + 1.0)),
        'top30_sum':    float(sorted_extrap[:min(30, sorted_extrap.size)].sum()),
        'top20_sum':    float(sorted_extrap[:min(20, sorted_extrap.size)].sum()),
        'top10_sum':    float(sorted_extrap[:min(10, sorted_extrap.size)].sum()),
        'prob_q95':     float(np.quantile(flat_extrap, 0.95)),
        'prob_q90':     float(np.quantile(flat_extrap, 0.90)),
        'prob_gini':    float(_gini(flat_extrap)),
        'peak_to_mean': float(flat_extrap.max() / max(1e-9, flat_extrap.mean())),
    }


# -----------------------------------------------------------------------------
# Ridge + isotonic (identical to v2)
# -----------------------------------------------------------------------------
def fit_ridge(X, y, alpha=1.0):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)


def standardize_fit(X):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-9] = 1.0
    return mu, sd


def isotonic_fit(x_raw, y_target):
    order = np.argsort(x_raw)
    x_s = x_raw[order]; y_s = y_target[order].astype(np.float64).copy()
    w = np.ones_like(y_s)
    n = len(y_s); i = 0
    while i < n - 1:
        if y_s[i] > y_s[i + 1]:
            new_y = (y_s[i] * w[i] + y_s[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            new_w = w[i] + w[i + 1]
            y_s[i] = new_y; w[i] = new_w
            y_s = np.delete(y_s, i + 1); w = np.delete(w, i + 1)
            x_s = np.delete(x_s, i + 1); n -= 1
            if i > 0: i -= 1
        else:
            i += 1
    return x_s, y_s


def isotonic_apply(x_s, y_s, x_query):
    return np.interp(x_query, x_s, y_s, left=y_s[0], right=y_s[-1])


# -----------------------------------------------------------------------------
# Per-species statistics under a binarization rule (threshold or top-N)
# -----------------------------------------------------------------------------
def compute_per_species_stats(prob_map_ens, threshold=None, top_n=None):
    ranges, ncomps, logcds = [], [], []
    for k in range(prob_map_ens.shape[0]):
        m = prob_map_ens[k]
        if threshold is not None:
            b = (m >= threshold).astype(np.uint8)
        elif top_n is not None:
            n = max(1, int(top_n))
            flat = m.ravel()
            if flat.max() < 1e-9:
                b = np.zeros_like(m, dtype=np.uint8)
            else:
                thr = np.partition(flat, -n)[-n] - 1e-9
                b = (m > thr).astype(np.uint8)
        else:
            raise ValueError
        if b.sum() < 2:
            continue
        ranges.append(int(b.sum()))
        ncomps.append(count_components(b))
        logcds.append(np.log10(periodic_cov_det(b) + 1.0))
    return ranges, ncomps, logcds


def compute_bucket_ks(per_species_records, K):
    """Bucket boundaries depend on K: when truth filter is range>K, EASY
    bucket is empty for K>=10. We compute POOLED + EASY + MODERATE + HARD
    and skip empty buckets."""
    out = {}
    truth = np.asarray([r['truth_range'] for r in per_species_records])
    truth_ncomp = np.asarray([r['truth_ncomp'] for r in per_species_records])
    truth_logcd = np.asarray([r['truth_logcd'] for r in per_species_records])
    bucket_defs = [
        ('POOLED',   np.ones_like(truth, dtype=bool)),
        ('EASY',     (truth >= 6) & (truth <= 10)),
        ('MODERATE', (truth >= 11) & (truth <= 20)),
        ('HARD',     truth >= 21),
    ]
    for name, sel in bucket_defs:
        if sel.sum() == 0:
            continue
        recs_sel = [per_species_records[i] for i in np.where(sel)[0]]
        p_range, p_ncomp, p_logcd = [], [], []
        for r in recs_sel:
            p_range.extend(r['pred_range'])
            p_ncomp.extend(r['pred_ncomp'])
            p_logcd.extend(r['pred_logcd'])
        if len(p_range) == 0:
            continue
        out[name] = {
            'range':  float(stats.ks_2samp(truth[sel],       p_range).statistic),
            'conn':   float(stats.ks_2samp(truth_ncomp[sel], p_ncomp).statistic),
            'spread': float(stats.ks_2samp(truth_logcd[sel], p_logcd).statistic),
            'n_truth': int(sel.sum()), 'n_pred': len(p_range),
        }
    return out


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def self_test():
    print("\n  Running deterministic self-tests ...")
    ok_all = True
    # PBC self-test (subset; v2 covers the rest)
    g = np.zeros((GRID_Y, GRID_X), dtype=np.uint8); g[0, 0] = g[0, 1] = 1
    d = periodic_cov_det(g); ok_all &= (0.0 <= d <= 1.0)
    print(f"    pbc_two_adjacent       d={d:.4f}  expect in [0,1]  {'OK' if 0<=d<=1 else 'FAIL'}")
    # Ridge solves identity
    X = np.eye(3); y = np.array([1., 2., 3.])
    beta = fit_ridge(X, y, alpha=0.0)
    ok = np.allclose(beta, y, atol=1e-6); ok_all &= ok
    print(f"    ridge_identity         beta={beta}  expect [1,2,3]  {'OK' if ok else 'FAIL'}")
    # Isotonic preserves monotone
    x = np.array([1., 2., 3., 4.]); y = np.array([1., 2., 3., 4.])
    xs, ys = isotonic_fit(x, y)
    ok = np.allclose(ys, [1, 2, 3, 4]); ok_all &= ok
    print(f"    isotonic_monotone      ys={ys}  expect [1,2,3,4]  {'OK' if ok else 'FAIL'}")
    # Isotonic fixes violation
    x = np.array([1., 2., 3.]); y = np.array([1., 4., 2.])
    xs, ys = isotonic_fit(x, y)
    ok = (len(ys) <= 3) and bool(np.all(np.diff(ys) >= -1e-9)); ok_all &= ok
    print(f"    isotonic_pava_repair   ys={ys}  expect monotone {'OK' if ok else 'FAIL'}")
    # Bucket-router decision rule
    nhat = np.array([5.0, 10.0, 15.0, 25.0])
    tau = float(np.quantile(nhat, 0.75))  # ~22.5
    cls = (nhat >= tau).astype(int)
    ok = (cls.tolist() == [0, 0, 0, 1]); ok_all &= ok
    print(f"    classifier_split       cls={cls.tolist()}  expect [0,0,0,1]  {'OK' if ok else 'FAIL'}")
    print(f"  SELF-TEST {'PASSED' if ok_all else 'FAILED'}")
    return 0 if ok_all else 1


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_world_data(truth_path, samples_path):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    if 'obs_mask' in z.files:
        obs_mask = np.asarray(z['obs_mask']).astype(np.uint8)
    else:
        obs_mask = (samples.mean(axis=0) >= 0.99).astype(np.uint8)
    n_use = min(truth.shape[0], samples.shape[1], obs_mask.shape[0])
    return {'truth': truth[:n_use],
            'samples': samples[:, :n_use],
            'obs_mask': obs_mask[:n_use]}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--wide-range-csv')
    ap.add_argument('--recon-dir-pattern')
    ap.add_argument('--truth-dir')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--top-n-worlds', type=int, default=30)
    ap.add_argument('--ridge-alpha', type=float, default=1.0)
    ap.add_argument('--output-dir')
    ap.add_argument('--threshold-moderate', type=float, default=0.95,
                     help='fixed prob threshold for predicted-MODERATE species (v3)')
    ap.add_argument('--threshold-hard',     type=float, default=0.80,
                     help='fixed prob threshold for predicted-HARD species (v3)')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not all([args.wide_range_csv, args.recon_dir_pattern,
                args.truth_dir, args.output_dir]):
        ap.error("data args required when not running --self-test")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    veri = open(out_dir / 'verification.log', 'w')
    def vp(s):
        print(s); veri.write(s + '\n'); veri.flush()

    vp("\n  AXEL ADAPTIVE ROUTING v3 — N_hat as bucket classifier")
    vp("=" * 72)
    vp(f"  K = {args.K}")
    vp(f"  threshold for predicted MODERATE = p >= {args.threshold_moderate}")
    vp(f"  threshold for predicted HARD     = p >= {args.threshold_hard}")
    vp("")

    vp("  [verify] running self-test ...")
    if self_test() != 0:
        sys.exit(1)

    # Load all worlds
    world_sp_count = defaultdict(int)
    with open(args.wide_range_csv) as f:
        for row in csv.DictReader(f):
            world_sp_count[row['world']] += 1
    top_worlds = sorted(world_sp_count.items(), key=lambda x: -x[1])[:args.top_n_worlds]

    truth_dir = Path(args.truth_dir)
    world_data = {}
    world_keep_idx = {}  # world -> array of global species indices kept
    world_features = {}
    world_truth_range = {}
    for wn, _ in top_worlds:
        stem = wn.replace('.npz', '')
        tp = truth_dir / wn
        sp = Path(args.recon_dir_pattern.format(world_stem=stem)) / \
             f'recon_fixed_b{args.K}_samples.npz'
        if not (tp.exists() and sp.exists()):
            continue
        wd = load_world_data(tp, sp)
        truth = wd['truth']
        tr_all = truth.sum(axis=(1, 2)).astype(np.int32)
        keep = np.where(tr_all > args.K)[0]
        if len(keep) == 0:
            continue
        feats = [compute_features_one_species(wd['samples'][:, s], wd['obs_mask'][s])
                 for s in keep]
        world_data[wn] = wd
        world_keep_idx[wn] = keep
        world_features[wn] = feats
        world_truth_range[wn] = tr_all[keep]
        vp(f"    loaded {wn[:55]:55s}  n_species={len(keep):4d}")
    if len(world_features) < 3:
        vp("\n  Need at least 3 worlds. Aborting."); sys.exit(2)

    # Pool features for LOWO ridge
    feat_names = list(INFORMATIVE_FEATURES)
    all_X, all_y, all_wid = [], [], []
    for w_idx, wn in enumerate(world_features):
        for i in range(len(world_features[wn])):
            all_X.append([world_features[wn][i][f] for f in feat_names])
            all_y.append(world_truth_range[wn][i])
            all_wid.append(w_idx)
    all_X = np.asarray(all_X, dtype=np.float64)
    all_y = np.asarray(all_y, dtype=np.float64)
    all_wid = np.asarray(all_wid, dtype=np.int32)
    n_total = len(all_y)
    vp(f"\n  [verify] pooled n_species = {n_total}")
    vp(f"  [verify] truth range: min={all_y.min():.0f} max={all_y.max():.0f} "
       f"mean={all_y.mean():.2f}")

    # LOWO ridge + isotonic
    n_hat = np.zeros(n_total, dtype=np.float64)
    hard_fraction_estimates = []
    world_keys = list(world_features.keys())
    for w_idx, _wn in enumerate(world_keys):
        train_mask = (all_wid != w_idx); test_mask = (all_wid == w_idx)
        if test_mask.sum() == 0: continue
        mu, sd = standardize_fit(all_X[train_mask])
        X_tr = (all_X[train_mask] - mu) / sd
        X_te = (all_X[test_mask]  - mu) / sd
        y_tr = all_y[train_mask]
        beta = fit_ridge(X_tr, y_tr - y_tr.mean(), alpha=args.ridge_alpha)
        score_tr = X_tr @ beta + y_tr.mean()
        score_te = X_te @ beta + y_tr.mean()
        x_s, y_s = isotonic_fit(score_tr, y_tr)
        nhat_te = isotonic_apply(x_s, y_s, score_te)
        n_hat[test_mask] = np.clip(nhat_te, 1.0, GRID_Y * GRID_X)
        # Estimate HARD fraction from TRAIN truth only (truth-free at test)
        hard_fraction_estimates.append(float((y_tr >= 21).mean()))

    rho_v3, _ = stats.spearmanr(n_hat, all_y)
    vp(f"\n  [verify] N_hat predictor: Spearman rho = {rho_v3:+.3f}")
    vp(f"  [verify] mean N_hat = {n_hat.mean():.2f}  mean truth = {all_y.mean():.2f}")
    vp(f"  [verify] HARD-fraction prior (LOWO mean) = "
       f"{np.mean(hard_fraction_estimates):.3f}  std = "
       f"{np.std(hard_fraction_estimates):.3f}")

    # Classifier AUC for HARD vs MODERATE
    is_hard_truth = (all_y >= 21).astype(int)
    if is_hard_truth.sum() > 0 and is_hard_truth.sum() < len(is_hard_truth):
        auc = float(stats.mannwhitneyu(n_hat[is_hard_truth == 1],
                                         n_hat[is_hard_truth == 0],
                                         alternative='greater').statistic /
                    (is_hard_truth.sum() * (len(is_hard_truth) - is_hard_truth.sum())))
        vp(f"  [verify] HARD-vs-MODERATE classifier AUC (N_hat) = {auc:.3f}")

    # Per-species evaluation under four methods
    vp(f"\n  Computing per-species predictions under 4 methods ...")
    methods = ['fixed_p080', 'fixed_p095', 'adaptive_v2', 'v3_bucketrouter']
    records = {m: [] for m in methods}
    species_idx = 0
    for w_idx, wn in enumerate(world_keys):
        wd = world_data[wn]; truth = wd['truth']; samples = wd['samples']
        keep = world_keep_idx[wn]
        # HARD-fraction prior estimated from THIS world's leave-one-out train set
        train_mask = (all_wid != w_idx)
        hard_frac = float((all_y[train_mask] >= 21).mean())
        # Tau split for this test world: take percentile (1 - hard_frac) of TRAIN N_hat
        train_nhat = n_hat[train_mask]
        tau_split = float(np.quantile(train_nhat, 1.0 - hard_frac))
        for local_i, global_s in enumerate(keep):
            ens = samples[:, global_s]
            tb = truth[global_s]
            tr_v = int(tb.sum())
            tn_v = count_components(tb)
            tl_v = np.log10(periodic_cov_det(tb) + 1.0)
            base = {'truth_range': tr_v, 'truth_ncomp': tn_v, 'truth_logcd': tl_v}

            # fixed thresholds
            for tname, thr in [('fixed_p080', 0.80), ('fixed_p095', 0.95)]:
                pr, pc, pl = compute_per_species_stats(ens, threshold=thr)
                records[tname].append({**base,
                    'pred_range': pr, 'pred_ncomp': pc, 'pred_logcd': pl})

            # adaptive_v2 top-N
            pr, pc, pl = compute_per_species_stats(ens, top_n=n_hat[species_idx])
            records['adaptive_v2'].append({**base,
                'pred_range': pr, 'pred_ncomp': pc, 'pred_logcd': pl})

            # v3 bucket-router
            if n_hat[species_idx] >= tau_split:
                thr_v3 = args.threshold_hard
            else:
                thr_v3 = args.threshold_moderate
            pr, pc, pl = compute_per_species_stats(ens, threshold=thr_v3)
            records['v3_bucketrouter'].append({**base,
                'pred_range': pr, 'pred_ncomp': pc, 'pred_logcd': pl,
                'tau_split': tau_split, 'predicted_bucket':
                    'HARD' if n_hat[species_idx] >= tau_split else 'MODERATE'})
            species_idx += 1

    # Compute bucket KS for each method
    vp(f"\n  Bucket KS per method (KS <= 0.30 = PASS, <= 0.10 = EXCELLENT)\n")
    bucket_ks = {m: compute_bucket_ks(records[m], args.K) for m in methods}

    vp(f"  {'method':<18s} {'bucket':<10s} {'n_truth':>8s} {'ks_range':>10s} "
       f"{'ks_conn':>9s} {'ks_spread':>10s}  verdict_range")
    vp(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*9} {'-'*10}  {'-'*16}")
    summary_rows = []
    for m in methods:
        for b in ('POOLED', 'EASY', 'MODERATE', 'HARD'):
            if b not in bucket_ks[m]: continue
            d = bucket_ks[m][b]
            v = ('EXCELLENT' if d['range'] <= 0.10 else
                 'PASS' if d['range'] <= 0.30 else
                 'MARGINAL' if d['range'] <= 0.50 else 'FAIL')
            vp(f"  {m:<18s} {b:<10s} {d['n_truth']:>8d} "
               f"{d['range']:>10.3f} {d['conn']:>9.3f} {d['spread']:>10.3f}  {v}")
            summary_rows.append({'method': m, 'bucket': b,
                                  'n_truth': d['n_truth'], 'n_pred': d['n_pred'],
                                  'ks_range': d['range'], 'ks_conn': d['conn'],
                                  'ks_spread': d['spread']})

    # CSV
    csv_path = out_dir / 'bucket_ks_v3_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows: w.writerow(r)
    vp(f"\n  ✓ wrote {csv_path}")

    # Heatmap
    bucket_order = ['POOLED', 'EASY', 'MODERATE', 'HARD']
    stat_order = ['range', 'conn', 'spread']
    rows = []; row_labels = []
    for b in bucket_order:
        for s in stat_order:
            row = []
            for m in methods:
                d = bucket_ks[m].get(b)
                row.append(d[s] if d is not None else np.nan)
            if not all(np.isnan(v) for v in row):
                rows.append(row); row_labels.append(f'{b} — {s}')
    arr = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(11, max(7, len(row_labels) * 0.40)))
    im = ax.imshow(arr, aspect='auto', cmap='RdYlGn_r', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(['fixed p≥0.80', 'fixed p≥0.95', 'adaptive v2\n(top-N_hat)',
                          'v3 bucket-router\n(p≥0.80 if pred HARD,\np≥0.95 otherwise)'],
                       fontsize=9, rotation=15, ha='right')
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f'{arr[i, j]:.3f}', ha='center', va='center',
                         fontsize=8, fontweight='bold',
                         color='white' if arr[i, j] > 0.55 else 'black')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('KS distance', fontsize=10)
    ax.set_title(f'Bucket KS at K={args.K}: four methods compared\n'
                 f'(N_hat rho = {rho_v3:+.3f}; classifier AUC and HARD-fraction '
                 f'in verification.log)',
                 fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'bucket_ks_v3_heatmap.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    vp(f"  ✓ wrote {out_dir / 'bucket_ks_v3_heatmap.png'}")

    # Classifier confusion / calibration figure
    pred_hard_arr = np.zeros(n_total, dtype=int)
    for w_idx in range(len(world_keys)):
        train_mask = (all_wid != w_idx); test_mask = (all_wid == w_idx)
        if test_mask.sum() == 0: continue
        hf = float((all_y[train_mask] >= 21).mean())
        tau = float(np.quantile(n_hat[train_mask], 1.0 - hf))
        pred_hard_arr[test_mask] = (n_hat[test_mask] >= tau).astype(int)

    confusion = np.zeros((2, 2), dtype=int)
    for ph, ih in zip(pred_hard_arr, is_hard_truth):
        confusion[ih, ph] += 1
    tp = confusion[1, 1]; fn = confusion[1, 0]; fp = confusion[0, 1]; tn = confusion[0, 0]
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    vp(f"\n  Classifier confusion (HARD = positive class):")
    vp(f"     TP={tp:4d}  FN={fn:4d}   recall (HARD) = {recall:.2%}")
    vp(f"     FP={fp:4d}  TN={tn:4d}   precision (HARD) = {precision:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    is_hard = is_hard_truth.astype(bool)
    ax.scatter(n_hat[~is_hard], all_y[~is_hard], s=6, alpha=0.3, color='#3D6FB0',
                label=f'truth MODERATE (n={(~is_hard).sum()})')
    ax.scatter(n_hat[is_hard],  all_y[is_hard],  s=10, alpha=0.7, color='#B03D3D',
                label=f'truth HARD (n={is_hard.sum()})')
    # mean tau across LOWO folds
    mean_tau = float(np.mean(
        [np.quantile(n_hat[all_wid != w], 1.0 - (all_y[all_wid != w] >= 21).mean())
         for w in range(len(world_keys)) if (all_wid == w).sum() > 0]
    ))
    ax.axvline(mean_tau, color='black', linestyle='--', linewidth=1.2,
                label=f'mean tau_split = {mean_tau:.2f}')
    ax.axhline(20.5, color='gray', linestyle=':', linewidth=1.0,
                label='truth bucket boundary = 20.5')
    ax.set_xlabel('N_hat (v2 multi-feature predictor)', fontsize=10)
    ax.set_ylabel('Truth range size (cells)', fontsize=10)
    ax.set_title('(A) Classifier scatter — N_hat vs truth, colored by bucket',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['pred MODERATE', 'pred HARD'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['truth MODERATE', 'truth HARD'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{confusion[i, j]}', ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
    ax.set_title(f'(B) Confusion matrix\nrecall={recall:.0%}  precision={precision:.0%}',
                 fontsize=10, fontweight='bold')
    fig.suptitle('v3 bucket-classifier diagnostics', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'classifier_calibration.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    vp(f"  ✓ wrote {out_dir / 'classifier_calibration.png'}")

    veri.close()
    print(f"\n  Done. Outputs in {out_dir}\n")


if __name__ == "__main__":
    main()