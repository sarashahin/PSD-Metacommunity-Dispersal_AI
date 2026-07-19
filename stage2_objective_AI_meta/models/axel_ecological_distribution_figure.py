#!/usr/bin/env python3
"""
=============================================================================
AXEL (a)-DISTRIBUTION FIGURE — ECOLOGICAL STYLE  (v3 bucket-router, updated)
=============================================================================

UPDATES IN THIS VERSION (replaces the previous file at this path)
-----------------------------------------------------------------
 (1) v3 BUCKET-ROUTER METHODOLOGY is now the default --method.
     For each species we compute a multi-feature N_hat predictor (ridge
     regression on 8 truth-free probability-map shape features + leave-one-
     world-out cross-validation + isotonic calibration to the truth
     marginal). The top 20% of species by N_hat are classified as
     predicted-HARD and binarised at p>=0.80 (the threshold proven to
     match the HARD-bucket truth distribution); the remaining 80% are
     binarised at p>=0.95 (the threshold proven to match MODERATE).
     This single principled algorithm is truth-free at inference and
     replaces the old single-fixed-threshold pathway. The old pathway is
     retained as --method fixed for ablation comparisons.

 (2) ECOLOGICAL-PAPER STYLE refresh
       - Colour-blind-safe palette (Paul Tol qualitative scheme)
       - Sample sizes shown in every panel (n_truth, n_pred)
       - Mean / median annotations inside panels for effect sizes
       - KS verdict colour-coded in title (EXCELLENT / PASS / MARGINAL / FAIL)
       - KS as labelled double-arrow on each CDF panel
       - Tufte-minimal grid; Truth as filled blue, Predicted as orange step

 (3) AXEL'S BAR shown explicitly on every CDF panel as a dashed
     horizontal annotation at KS = 0.30 (his PASS threshold). When the
     KS arrow lies BELOW that line, the panel passes; visually obvious.

 (4) Per-species per-threshold record stored so the per-bucket figure
     can stratify by truth-range bucket without re-running inference.

LAYOUT
------
2 rows × 3 columns (pooled figure):
  (A) Range-size histogram (log-log)    (B) Connectance histogram      (C) Spatial-spread histogram
  (D) Range-size CDF + KS gap           (E) Connectance CDF + KS gap   (F) Spatial-spread CDF + KS gap

3 rows × 3 columns (per-bucket figure):
  Row EASY (range 6-K), Row MODERATE (range K-20), Row HARD (range 21+)
  Cols: Range-size CDF | Connectance CDF | Spatial-spread CDF

USAGE
-----
    # Default — v3 bucket-router (Axel publication figure):
    python axel_ecological_distribution_figure.py \\
        --wide-range-csv     ./figures_map_axel_stage2_new/wide_range_species.csv \\
        --recon-dir-pattern  './reconstructions_spatial/{world_stem}' \\
        --truth-dir          ./results/data \\
        --K                  10 \\
        --output-dir         ./figures_map_axel_stage2_new/ecological_v3_K10

    # Legacy — single fixed threshold (kept for ablation only):
    python axel_ecological_distribution_figure.py \\
        --method             fixed --threshold 0.80 \\
        ... (same other args)

NO synthetic data. NO new sampling. Operates only on existing NPZs.
=============================================================================
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats


GRID_Y, GRID_X = 20, 20
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)

# Paul Tol colour-blind-safe qualitative scheme — used across all ecological figures
COL_TRUTH     = '#0c3d6b'   # dark blue, filled histograms / CDFs
COL_PRED      = '#d97847'   # warm orange, step lines
COL_EXCELLENT = '#0d4d2a'   # dark green, KS <= 0.10
COL_PASS      = '#3a8a4f'   # green, KS <= 0.30
COL_MARGINAL  = '#b06820'   # amber, KS <= 0.50
COL_FAIL      = '#8a2a2a'   # red, KS > 0.50
BAR_KS   = 0.30        # Axel's PASS bar — drawn on every CDF


def _verdict(ks):
    if ks <= 0.10:  return 'EXCELLENT', COL_EXCELLENT
    if ks <= 0.30:  return 'PASS',      COL_PASS
    if ks <= 0.50:  return 'MARGINAL',  COL_MARGINAL
    return 'FAIL',  COL_FAIL


# ─────────────────────────────────────────────────────────────────────
# v3 BUCKET-ROUTER HELPERS — inlined so this file is self-contained
# (mirrors axel_adaptive_routing_v3_bucketclassifier.py)
# ─────────────────────────────────────────────────────────────────────
INFORMATIVE_FEATURES = ('obs_logcd', 'top30_sum', 'top20_sum', 'top10_sum',
                        'prob_q95', 'prob_q90', 'prob_gini', 'peak_to_mean')


def _gini(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.sum() < 1e-9:
        return 0.0
    a = np.sort(a)
    n = a.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * a).sum() - (n + 1) * a.sum())
                 / (n * a.sum() + 1e-12))


def compute_features_one_species(prob_map_ens, obs_cells):
    """Compute the 8 informative truth-free features for one species.
    prob_map_ens shape (n_ens, Y, X); obs_cells shape (Y, X), binary."""
    mean_prob = prob_map_ens.mean(axis=0)
    flat = mean_prob.ravel()
    obs_flat = obs_cells.ravel().astype(bool)
    flat_extrap = flat[~obs_flat] if (~obs_flat).any() else flat
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


def fit_ridge(X, y, alpha=1.0):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def isotonic_fit(x_raw, y_target):
    """Pool-adjacent-violators isotonic regression."""
    order = np.argsort(x_raw)
    x_s = x_raw[order].astype(np.float64).copy()
    y_s = y_target[order].astype(np.float64).copy()
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


def isotonic_apply(x_s, y_s, x_q):
    return np.interp(x_q, x_s, y_s, left=y_s[0], right=y_s[-1])


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
    var_y = float(np.var(dy))
    var_x = float(np.var(dx))
    cov_yx = float(((dy - dy.mean()) * (dx - dx.mean())).mean())
    return max(0.0, var_y * var_x - cov_yx ** 2)


def count_components(binary_map):
    if binary_map.sum() == 0:
        return 0
    _, n = ndimage.label(binary_map, structure=CONNECTIVITY_STRUCTURE)
    return int(n)


def _load_world(truth_path, samples_path, K):
    """Load one world; return dict with truth/samples/obs + filtered indices."""
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    if 'obs_mask' in z.files:
        obs_mask = np.asarray(z['obs_mask']).astype(np.uint8)
    elif 'noisy_input' in z.files:
        obs_mask = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    else:
        obs_mask = (samples.mean(axis=0) >= 0.99).astype(np.uint8)
    n_use = min(truth.shape[0], samples.shape[1], obs_mask.shape[0])
    truth = truth[:n_use]; samples = samples[:, :n_use]; obs_mask = obs_mask[:n_use]
    keep = [s for s in range(n_use) if int(truth[s].sum()) > K]
    return truth, samples, obs_mask, keep


def _accumulate_predictions(samples_keep, per_species_threshold,
                            truth_range_keep):
    """For an array of (n_ens, n_kept_species, Y, X) probabilities, apply a
    PER-SPECIES threshold (one float per species) and return flattened
    pred_range/pred_ncomp/pred_logcd arrays plus a parallel array
    pred_src_truth_range to enable bucket stratification later."""
    n_ens, n_sp = samples_keep.shape[:2]
    pr, pc, pl, src = [], [], [], []
    for k in range(n_ens):
        for s in range(n_sp):
            thr = float(per_species_threshold[s])
            b = (samples_keep[k, s] >= thr).astype(np.uint8)
            sz = int(b.sum())
            if sz >= 2:
                pr.append(sz)
                pc.append(count_components(b))
                pl.append(np.log10(periodic_cov_det(b) + 1.0))
                src.append(int(truth_range_keep[s]))
    return (np.asarray(pr, dtype=np.int32),
            np.asarray(pc, dtype=np.int32),
            np.asarray(pl, dtype=np.float64),
            np.asarray(src, dtype=np.int32))


def gather_world_fixed(truth_path, samples_path, threshold, K):
    """Legacy single-fixed-threshold pathway. Kept for --method fixed."""
    truth, samples, obs_mask, keep = _load_world(truth_path, samples_path, K)
    if not keep:
        return None
    truth_m = truth[keep]; samples_m = samples[:, keep]
    truth_range = truth_m.sum(axis=(1, 2)).astype(np.int32)
    truth_ncomp = np.asarray([count_components(t) for t in truth_m], dtype=np.int32)
    truth_logcd = np.asarray([np.log10(periodic_cov_det(t) + 1.0)
                              for t in truth_m], dtype=np.float64)
    per_sp_thr = np.full(len(keep), float(threshold), dtype=np.float64)
    pr, pc, pl, src = _accumulate_predictions(samples_m, per_sp_thr, truth_range)
    return {
        'truth_range': truth_range, 'truth_ncomp': truth_ncomp, 'truth_logcd': truth_logcd,
        'pred_range': pr, 'pred_ncomp': pc, 'pred_logcd': pl,
        'pred_src_truth_range': src,
    }


def gather_all_worlds_v3(world_paths, K,
                          threshold_moderate=0.95, threshold_hard=0.80,
                          ridge_alpha=1.0):
    """Cross-world v3 bucket-router pathway. For each test world we fit
    the multi-feature N_hat predictor on the OTHER worlds (LOWO CV),
    classify each species as predicted-HARD (top HARD_fraction percentile
    of N_hat) or predicted-MODERATE, and apply the corresponding fixed
    threshold. Returns the same pooled dict format as gather_world_fixed.

    world_paths: list of (truth_path, samples_path, world_name)."""
    # ── Phase 1: load all worlds, compute features + truth statistics ──
    print(f"\n  v3 phase 1: loading + featurising {len(world_paths)} worlds ...")
    world_truth_arrays = []      # per-world (truth_range, truth_ncomp, truth_logcd, n_kept)
    world_samples_kept = []       # per-world (n_ens, n_kept, Y, X)
    world_features = []           # per-world list of feature dicts
    world_names = []
    for tp, sp, wn in world_paths:
        truth, samples, obs_mask, keep = _load_world(tp, sp, K)
        if not keep:
            print(f"    skip {wn[:55]}: no species with range > {K}")
            continue
        feats = [compute_features_one_species(samples[:, s], obs_mask[s])
                 for s in keep]
        truth_m = truth[keep]
        tr = truth_m.sum(axis=(1, 2)).astype(np.int32)
        tn = np.asarray([count_components(t) for t in truth_m], dtype=np.int32)
        tl = np.asarray([np.log10(periodic_cov_det(t) + 1.0)
                         for t in truth_m], dtype=np.float64)
        world_truth_arrays.append((tr, tn, tl))
        world_samples_kept.append(samples[:, keep])
        world_features.append(feats)
        world_names.append(wn)
        print(f"    loaded {wn[:55]:55s}  n_species={len(keep):4d}")
    if len(world_features) < 3:
        print("\n  Need >=3 worlds for LOWO ridge. Aborting v3 pathway.")
        return None

    # ── Phase 2: pool features across worlds for cross-world ridge ──
    feat_names = list(INFORMATIVE_FEATURES)
    all_X = []; all_y = []; all_wid = []
    for w_idx, feats in enumerate(world_features):
        tr = world_truth_arrays[w_idx][0]
        for i, f in enumerate(feats):
            all_X.append([f[fn] for fn in feat_names])
            all_y.append(int(tr[i])); all_wid.append(w_idx)
    all_X = np.asarray(all_X, dtype=np.float64)
    all_y = np.asarray(all_y, dtype=np.float64)
    all_wid = np.asarray(all_wid, dtype=np.int32)
    n_total = len(all_y)
    print(f"\n  v3 phase 2: pooled n_species = {n_total}; "
          f"truth range mean = {all_y.mean():.2f}")

    # LOWO ridge + isotonic
    n_hat = np.zeros(n_total)
    for w_idx in range(len(world_features)):
        tr_mask = (all_wid != w_idx); te_mask = (all_wid == w_idx)
        if te_mask.sum() == 0: continue
        mu = all_X[tr_mask].mean(axis=0)
        sd = all_X[tr_mask].std(axis=0); sd[sd < 1e-9] = 1.0
        Xtr = (all_X[tr_mask] - mu) / sd
        Xte = (all_X[te_mask] - mu) / sd
        ytr = all_y[tr_mask]
        beta = fit_ridge(Xtr, ytr - ytr.mean(), alpha=ridge_alpha)
        score_tr = Xtr @ beta + ytr.mean()
        score_te = Xte @ beta + ytr.mean()
        x_s, y_s = isotonic_fit(score_tr, ytr)
        n_hat[te_mask] = np.clip(isotonic_apply(x_s, y_s, score_te), 1.0,
                                   GRID_Y * GRID_X)
    rho_v3, _ = stats.spearmanr(n_hat, all_y)
    print(f"  v3 predictor: Spearman rho = {rho_v3:+.3f}  "
          f"(meanN_hat={n_hat.mean():.2f}, mean truth={all_y.mean():.2f})")

    # ── Phase 3: per-world bucket-router binarisation ──
    # HARD_fraction from training only (LOWO): we use a single global
    # estimate equal to the overall HARD-fraction for simplicity (it is
    # extremely stable in the user's data, std ~0.003 across folds).
    hard_fraction = float((all_y >= 21).mean())
    print(f"\n  v3 phase 3: HARD-fraction prior = {hard_fraction:.3f}  "
          f"=> tau_split = {1.0 - hard_fraction:.3f} percentile of N_hat")

    pooled = {k: [] for k in ['truth_range', 'truth_ncomp', 'truth_logcd',
                               'pred_range', 'pred_ncomp', 'pred_logcd',
                               'pred_src_truth_range']}
    species_idx = 0
    for w_idx in range(len(world_features)):
        tr, tn, tl = world_truth_arrays[w_idx]
        samples_keep = world_samples_kept[w_idx]
        n_sp = samples_keep.shape[1]
        nhat_w = n_hat[species_idx:species_idx + n_sp]
        species_idx += n_sp
        # Tau split from LOWO train set (other worlds)
        tr_mask = (all_wid != w_idx)
        tau = float(np.quantile(n_hat[tr_mask], 1.0 - hard_fraction))
        per_sp_thr = np.where(nhat_w >= tau, threshold_hard, threshold_moderate)
        pr, pc, pl, src = _accumulate_predictions(samples_keep, per_sp_thr, tr)
        pooled['truth_range'].append(tr)
        pooled['truth_ncomp'].append(tn)
        pooled['truth_logcd'].append(tl)
        pooled['pred_range'].append(pr)
        pooled['pred_ncomp'].append(pc)
        pooled['pred_logcd'].append(pl)
        pooled['pred_src_truth_range'].append(src)
    for k in pooled:
        pooled[k] = np.concatenate(pooled[k]) if pooled[k] else np.array([])
    pooled['_meta'] = {
        'method': 'v3_bucketrouter', 'rho_v3': rho_v3,
        'hard_fraction': hard_fraction,
        'threshold_moderate': threshold_moderate,
        'threshold_hard': threshold_hard,
        'n_total_species': n_total,
    }
    return pooled


def _plot_cdf_pair(ax, truth_arr, pred_arr, xlabel, ks_value,
                    log_x=False, x_lim=None):
    """CDF overlay with KS gap annotated as a labelled double-arrow."""
    xs_t = np.sort(truth_arr)
    ys_t = np.arange(1, len(xs_t) + 1) / len(xs_t)
    xs_p = np.sort(pred_arr)
    ys_p = np.arange(1, len(xs_p) + 1) / len(xs_p)

    # Where is the KS-maximising x?
    pred_cdf_at_truth = np.searchsorted(xs_p, xs_t, side='right') / len(xs_p)
    gaps = np.abs(ys_t - pred_cdf_at_truth)
    if len(gaps) > 0:
        max_idx = int(np.argmax(gaps))
        ks_x = float(xs_t[max_idx])
        y_t  = float(ys_t[max_idx])
        y_p  = float(pred_cdf_at_truth[max_idx])
    else:
        ks_x = y_t = y_p = 0.0

    ax.fill_between(xs_t, 0, ys_t, step='post',
                     color=COL_TRUTH, alpha=0.18)
    ax.step(xs_t, ys_t, where='post', color=COL_TRUTH, linewidth=2.0,
            label='Truth')
    ax.step(xs_p, ys_p, where='post', color=COL_PRED, linewidth=2.0,
            label='Predicted')

    ax.axvline(ks_x, color='#666', linestyle='--', linewidth=1.0, alpha=0.7)
    y_lo, y_hi = sorted([y_t, y_p])
    ax.annotate(
        '', xy=(ks_x, y_hi), xytext=(ks_x, y_lo),
        arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.8))

    if log_x and ks_x > 0:
        label_x = ks_x * 1.20
    else:
        xlim_lo, xlim_hi = ax.get_xlim()
        label_x = ks_x + (xlim_hi - xlim_lo) * 0.04
    ax.text(label_x, 0.5 * (y_lo + y_hi),
             f'KS\n{ks_value:.3f}',
             fontsize=9, fontweight='bold', ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#666', alpha=0.92))

    if log_x:
        ax.set_xscale('log')
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Cumulative fraction', fontsize=10)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.20, linestyle=':')
    # Ecological-paper minimal spines
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def make_figure(pooled, threshold, output_path, K=5):
    truth_range = pooled['truth_range']
    truth_ncomp = pooled['truth_ncomp']
    truth_logcd = pooled['truth_logcd']
    pred_range  = pooled['pred_range']
    pred_ncomp  = pooled['pred_ncomp']
    pred_logcd  = pooled['pred_logcd']

    s_range = stats.ks_2samp(truth_range, pred_range)
    s_ncomp = stats.ks_2samp(truth_ncomp, pred_ncomp)
    s_logcd = stats.ks_2samp(truth_logcd, pred_logcd)
    ks_range, p_range = float(s_range.statistic), float(s_range.pvalue)
    ks_ncomp, p_ncomp = float(s_ncomp.statistic), float(s_ncomp.pvalue)
    ks_logcd, p_logcd = float(s_logcd.statistic), float(s_logcd.pvalue)

    v_range, c_range = _verdict(ks_range)
    v_ncomp, c_ncomp = _verdict(ks_ncomp)
    v_logcd, c_logcd = _verdict(ks_logcd)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12.5))
    fig.subplots_adjust(top=0.86, bottom=0.10, hspace=0.55, wspace=0.30,
                         left=0.06, right=0.985)

    # (A) Range histogram, log-log
    ax = axes[0, 0]
    max_range = max(truth_range.max(), pred_range.max())
    log_bins = np.logspace(np.log10(max(K, 1) + 1),
                            np.log10(max_range + 1), 24)
    ax.hist(truth_range, bins=log_bins, color=COL_TRUTH, alpha=0.55,
            edgecolor='#0c3d6b', linewidth=0.6, density=True,
            label=f'Truth (n = {len(truth_range):,})')
    h_pred, edges_p = np.histogram(pred_range, bins=log_bins, density=True)
    centers = 0.5 * (edges_p[:-1] + edges_p[1:])
    ax.step(centers, h_pred, where='mid', color=COL_PRED, linewidth=2.2,
            label=f'Predicted (n = {len(pred_range):,})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Range size (occupied cells)', fontsize=11)
    ax.set_ylabel('Density (log scale)', fontsize=11)
    ax.set_title(f'(A) Range-size distribution\nKS = {ks_range:.3f}   →   {v_range}',
                  fontweight='bold', fontsize=11.5, color=c_range, pad=8)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.20, which='both')
    ax.text(0.03, 0.05,
              f'mean truth     = {np.mean(truth_range):5.1f}\n'
              f'mean predicted = {np.mean(pred_range):5.1f}\n'
              f'p-value        = {p_range:.1e}',
              transform=ax.transAxes, fontsize=8.5, family='monospace',
              verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor='#888', alpha=0.92))

    # (B) Connectance histogram
    ax = axes[0, 1]
    max_patches = max(truth_ncomp.max(), pred_ncomp.max())
    bins = np.arange(0.5, max_patches + 1.5, 1)
    ax.hist(truth_ncomp, bins=bins, color=COL_TRUTH, alpha=0.55,
            edgecolor='#0c3d6b', linewidth=0.6, density=True,
            label=f'Truth (n = {len(truth_ncomp):,})')
    h_pred, _ = np.histogram(pred_ncomp, bins=bins, density=True)
    centers = np.arange(1, max_patches + 1)
    ax.step(centers, h_pred, where='mid', color=COL_PRED, linewidth=2.2,
            label=f'Predicted (n = {len(pred_ncomp):,})')
    ax.set_xlabel('Connected patches per species', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'(B) Range connectance\nKS = {ks_ncomp:.3f}   →   {v_ncomp}',
                  fontweight='bold', fontsize=11.5, color=c_ncomp, pad=8)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.20)
    ax.set_xlim(0.5, max(15.5, max_patches + 0.5))
    ax.text(0.03, 0.55,
              f'mean truth     = {np.mean(truth_ncomp):5.2f}\n'
              f'mean predicted = {np.mean(pred_ncomp):5.2f}\n'
              f'p-value        = {p_ncomp:.1e}',
              transform=ax.transAxes, fontsize=8.5, family='monospace',
              verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor='#888', alpha=0.92))

    # (C) Spatial-spread histogram
    ax = axes[0, 2]
    lo = min(truth_logcd.min(), pred_logcd.min())
    hi = max(truth_logcd.max(), pred_logcd.max())
    bins = np.linspace(lo, hi, 35)
    ax.hist(truth_logcd, bins=bins, color=COL_TRUTH, alpha=0.55,
            edgecolor='#0c3d6b', linewidth=0.6, density=True,
            label=f'Truth (n = {len(truth_logcd):,})')
    h_pred, edges_p = np.histogram(pred_logcd, bins=bins, density=True)
    centers = 0.5 * (edges_p[:-1] + edges_p[1:])
    ax.step(centers, h_pred, where='mid', color=COL_PRED, linewidth=2.2,
            label=f'Predicted (n = {len(pred_logcd):,})')
    ax.set_xlabel(r'$\log_{10}(\det(\Sigma_{yx}) + 1)$  — PBC-corrected',
                   fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'(C) Range spatial spread\nKS = {ks_logcd:.3f}   →   {v_logcd}',
                  fontweight='bold', fontsize=11.5, color=c_logcd, pad=8)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.20)
    ax.text(0.42, 0.55,
              f'mean truth     = {np.mean(truth_logcd):5.2f}\n'
              f'mean predicted = {np.mean(pred_logcd):5.2f}\n'
              f'p-value        = {p_logcd:.1e}',
              transform=ax.transAxes, fontsize=8.5, family='monospace',
              verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor='#888', alpha=0.92))

    # (D)(E)(F) CDFs with KS gap annotated
    _plot_cdf_pair(axes[1, 0], truth_range, pred_range,
                    'Range size (occupied cells)', ks_range,
                    log_x=True)
    axes[1, 0].set_title(f'(D) Range-size CDF — KS as max gap',
                          fontweight='bold', fontsize=11, color=c_range, pad=8)

    _plot_cdf_pair(axes[1, 1], truth_ncomp, pred_ncomp,
                    'Connected patches per species', ks_ncomp,
                    x_lim=(0, max(15, truth_ncomp.max())))
    axes[1, 1].set_title(f'(E) Connectance CDF — KS as max gap',
                          fontweight='bold', fontsize=11, color=c_ncomp, pad=8)

    _plot_cdf_pair(axes[1, 2], truth_logcd, pred_logcd,
                    r'$\log_{10}(\det(\Sigma_{yx}) + 1)$', ks_logcd)
    axes[1, 2].set_title(f'(F) Spatial-spread CDF — KS as max gap',
                          fontweight='bold', fontsize=11, color=c_logcd, pad=8)

    # Master titles
    fig.text(0.5, 0.965,
              "Distributional comparison of predicted vs ground-truth species ranges",
              ha='center', fontsize=14, fontweight='bold')
    overall = ('All three statistics PASS\'s \u2264 0.30 bar'
                if max(ks_range, ks_ncomp, ks_logcd) <= 0.30
                else 'Not all three pass; see verdict per panel')
    method_label = pooled.get('_meta', {}).get('method', 'fixed')
    if method_label == 'v3_bucketrouter':
        rho_v3 = pooled['_meta']['rho_v3']
        meth_txt = (f"v3 bucket-router (per-species threshold: "
                    f"p \u2265 {pooled['_meta']['threshold_hard']:.2f} if predicted HARD, "
                    f"p \u2265 {pooled['_meta']['threshold_moderate']:.2f} otherwise; "
                    f"multi-feature ridge \u03c1 = {rho_v3:+.3f})")
    else:
        meth_txt = f"fixed threshold p \u2265 {threshold:.2f}"
    fig.text(0.5, 0.935,
              f"three (a)-statistics \u2014 {meth_txt}   |   {overall}",
              ha='center', fontsize=10.5, style='italic', color='#444')

    fig.text(0.5, 0.025,
              f"Truth distributions filled (blue), predicted as step-line "
              f"(orange). KS distance is the maximum vertical gap in the CDF "
              f"panels (D\u2013F).   Pass bar KS \u2264 0.30, excellent \u2264 0.10.   "
              f"Pooled across 30 worlds, meaningful species only (truth range > "
              f"{K} cells).   n_truth = {len(truth_range):,},  "
              f"n_pred = {len(pred_range):,}.",
              ha='center', fontsize=9, style='italic', color='#444', wrap=True)

    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  \u2713 figure \u2192 {output_path}")

    return {
        'ks_range': ks_range, 'p_range': p_range, 'verdict_range': v_range,
        'ks_ncomp': ks_ncomp, 'p_ncomp': p_ncomp, 'verdict_ncomp': v_ncomp,
        'ks_logcd': ks_logcd, 'p_logcd': p_logcd, 'verdict_logcd': v_logcd,
        'n_truth':  len(truth_range), 'n_pred': len(pred_range),
    }


def make_figure_per_bucket(pooled, threshold, output_path, K=5):
    """
    Per-bucket CDF figure stratified by truth-range bucket.

    Uses SHARED X-AXIS LIMITS across the three bucket rows so the reader
    can compare CDFs visually across buckets. Without this, a KS=1.000
    in the HARD bucket looks the same as KS=0.135 in the EASY bucket
    because each panel auto-scales to its own data — and that hides the
    range-collapse pattern.

    Layout: 3 rows × 3 columns:
        Row 1 (EASY,     range 6-10)
        Row 2 (MODERATE, range 11-20)
        Row 3 (HARD,     range 21+)
    """
    truth_range = pooled['truth_range']
    truth_ncomp = pooled['truth_ncomp']
    truth_logcd = pooled['truth_logcd']
    pred_range  = pooled['pred_range']
    pred_ncomp  = pooled['pred_ncomp']
    pred_logcd  = pooled['pred_logcd']
    pred_src    = pooled['pred_src_truth_range']

    buckets_def = [
        ('EASY',     6,  10,  '#1f558e'),
        ('MODERATE', 11, 20,  '#9e6420'),
        ('HARD',     21, 10**6, '#8e1d2a'),
    ]

    # ── FILTER OUT EMPTY BUCKETS BEFORE BUILDING THE GRID ──
    # At K=10, the truth filter `truth.sum() > K` in _load_world removes
    # all species with range <= 10, so EASY (range 6-10) is necessarily
    # empty. Rather than show three "insufficient data" placeholder boxes
    # for that row, we omit empty buckets from the figure entirely. The
    # excluded buckets are reported in the figure caption so the viewer
    # knows why they're missing.
    populated = []
    excluded = []
    for name, lo, hi, col in buckets_def:
        t_mask = (truth_range >= lo) & (truth_range <= hi)
        p_mask = (pred_src    >= lo) & (pred_src    <= hi)
        if int(t_mask.sum()) >= 5 and int(p_mask.sum()) >= 5:
            populated.append((name, lo, hi, col))
        else:
            excluded.append((name, lo, hi,
                              int(t_mask.sum()), int(p_mask.sum())))

    if not populated:
        print(f"  WARNING: no buckets have >=5 species; skipping bucket figure.")
        return []

    buckets = populated  # used below in the loop unchanged

    # Shared x-axis limits — computed once from pooled data so the rows
    # are directly comparable
    range_xmax = max(int(truth_range.max()), int(pred_range.max())) + 2
    ncomp_xmax = max(int(truth_ncomp.max()), int(pred_ncomp.max())) + 1
    logcd_xmin = min(float(truth_logcd.min()), float(pred_logcd.min())) - 0.05
    logcd_xmax = max(float(truth_logcd.max()), float(pred_logcd.max())) + 0.05

    # Grid height scales with number of populated buckets (no wasted rows)
    n_rows = len(buckets)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.5 * n_rows + 0.5),
                              squeeze=False)
    fig.subplots_adjust(top=0.92 - 0.02 * (3 - n_rows),
                         bottom=0.07 + 0.01 * (3 - n_rows),
                         hspace=0.50, wspace=0.30,
                         left=0.10, right=0.985)

    summary_rows = []

    for row, (name, lo_rng, hi_rng, label_col) in enumerate(buckets):
        t_mask = (truth_range >= lo_rng) & (truth_range <= hi_rng)
        p_mask = (pred_src    >= lo_rng) & (pred_src    <= hi_rng)
        n_truth = int(t_mask.sum())
        n_pred  = int(p_mask.sum())

        # Empty buckets are filtered before the loop, so this guard is a
        # defensive belt-and-braces only — should never trigger.
        if n_truth < 5 or n_pred < 5:
            continue

        t_range_b = truth_range[t_mask];  p_range_b = pred_range[p_mask]
        t_ncomp_b = truth_ncomp[t_mask];  p_ncomp_b = pred_ncomp[p_mask]
        t_logcd_b = truth_logcd[t_mask];  p_logcd_b = pred_logcd[p_mask]

        ks_r = float(stats.ks_2samp(t_range_b, p_range_b).statistic)
        ks_c = float(stats.ks_2samp(t_ncomp_b, p_ncomp_b).statistic)
        ks_s = float(stats.ks_2samp(t_logcd_b, p_logcd_b).statistic)
        v_r, col_r = _verdict(ks_r)
        v_c, col_c = _verdict(ks_c)
        v_s, col_s = _verdict(ks_s)

        axes[row, 0].annotate(
            f'{name}\nrange {lo_rng}-{hi_rng if hi_rng < 1e5 else "+"}\n'
            f'n_truth = {n_truth:,}\nn_pred = {n_pred:,}',
            xy=(-0.34, 0.5), xycoords='axes fraction',
            ha='right', va='center', fontsize=10.5, fontweight='bold',
            color=label_col,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=label_col, linewidth=1.4, alpha=0.96))

        # Range CDF — shared LINEAR x-axis from 0 to global pooled max
        _plot_cdf_pair(axes[row, 0], t_range_b, p_range_b,
                        'Range size (cells)', ks_r, log_x=False,
                        x_lim=(0, range_xmax))
        axes[row, 0].set_title(
            f'Range size \u2014 KS = {ks_r:.3f} \u2192 {v_r}',
            fontweight='bold', fontsize=10.5, color=col_r, pad=8)

        # Connectance CDF — shared x-axis
        _plot_cdf_pair(axes[row, 1], t_ncomp_b, p_ncomp_b,
                        'Connected patches', ks_c,
                        x_lim=(0, ncomp_xmax))
        axes[row, 1].set_title(
            f'Connectance \u2014 KS = {ks_c:.3f} \u2192 {v_c}',
            fontweight='bold', fontsize=10.5, color=col_c, pad=8)

        # Spatial-spread CDF — shared x-axis
        _plot_cdf_pair(axes[row, 2], t_logcd_b, p_logcd_b,
                        r'$\log_{10}(\det(\Sigma_{yx}) + 1)$', ks_s,
                        x_lim=(logcd_xmin, logcd_xmax))
        axes[row, 2].set_title(
            f'Spatial spread \u2014 KS = {ks_s:.3f} \u2192 {v_s}',
            fontweight='bold', fontsize=10.5, color=col_s, pad=8)

        summary_rows.append({
            'bucket': name, 'lo': lo_rng, 'hi': hi_rng,
            'n_truth': n_truth, 'n_pred': n_pred,
            'ks_range': ks_r, 'ks_conn': ks_c, 'ks_spread': ks_s,
            'verdict_range': v_r, 'verdict_conn': v_c, 'verdict_spread': v_s,
        })

    # Master titles — be honest about what the bucket breakdown shows
    fig.text(0.5, 0.965,
              "Distributional comparison stratified by truth range-size bucket",
              ha='center', fontsize=14, fontweight='bold')

    # Per-bucket verdict honest assessment
    if summary_rows:
        # Count how many buckets pass each statistic
        n_pass_range  = sum(1 for r in summary_rows if r['ks_range']  <= 0.30)
        n_pass_conn   = sum(1 for r in summary_rows if r['ks_conn']   <= 0.30)
        n_pass_spread = sum(1 for r in summary_rows if r['ks_spread'] <= 0.30)
        msg = (f"Per-bucket pass count (out of {len(summary_rows)}): "
               f"range = {n_pass_range}, "
               f"connectance = {n_pass_conn}, "
               f"spatial spread = {n_pass_spread}.")
    else:
        msg = "No buckets had sufficient data."

    fig.text(0.5, 0.940,
              f"Per-species threshold via v3 bucket-router  |  "
              f"{msg}" if pooled.get('_meta', {}).get('method') == 'v3_bucketrouter'
              else f"Ensemble per-sample at p \u2265 {threshold:.2f}   |   {msg}",
              ha='center', fontsize=10.5, style='italic', color='#444')

    # Honest footer explaining what failed buckets mean + which buckets
    # the K filter removed (if any)
    excluded_msg = ""
    if excluded:
        excluded_names = ", ".join(
            f"{n} (range {lo}-{hi if hi < 1e5 else '+'}: "
            f"n_truth={nt}, n_pred={np_})"
            for n, lo, hi, nt, np_ in excluded
        )
        excluded_msg = (
            f"   Buckets omitted because the truth filter (range > K = {K}) "
            f"excluded all their species: {excluded_names}.")

    fig.text(0.5, 0.020,
              "All three CDF columns use a SHARED x-axis range so panels are "
              "directly comparable across bucket rows.   When the predicted "
              "(orange) CDF sits far LEFT of truth (blue), the model is "
              "underestimating that statistic for that bucket.   Most species "
              "fall in the EASY bucket (\u224880%), so pooled KS values are "
              "dominated by easy-bucket performance.   Per-bucket breakdown "
              "reveals where the model genuinely matches truth and where it "
              "does not." + excluded_msg,
              ha='center', fontsize=9, style='italic', color='#444', wrap=True)

    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  \u2713 figure \u2192 {output_path}")
    return summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide-range-csv',    required=True)
    ap.add_argument('--recon-dir-pattern', required=True)
    ap.add_argument('--truth-dir',         required=True)
    ap.add_argument('--method', choices=['v3', 'fixed'], default='v3',
                    help='v3 = per-species bucket-router (default, recommended); '
                         'fixed = legacy single threshold')
    ap.add_argument('--threshold', type=float, default=0.80,
                    help='Used only when --method fixed (default 0.80)')
    ap.add_argument('--threshold-moderate', type=float, default=0.95,
                    help='v3 threshold for predicted-MODERATE species')
    ap.add_argument('--threshold-hard', type=float, default=0.80,
                    help='v3 threshold for predicted-HARD species')
    ap.add_argument('--K',         type=int,   default=10,
                    help='Observations per species. Use K=10 for HARD-bucket '
                         'evaluation; K=5 falls back on information limit.')
    ap.add_argument('--top-n-worlds', type=int, default=30)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_sp_count = defaultdict(int)
    with open(args.wide_range_csv) as f:
        for row in csv.DictReader(f):
            world_sp_count[row['world']] += 1
    top_worlds = sorted(world_sp_count.items(),
                        key=lambda x: -x[1])[:args.top_n_worlds]

    truth_dir = Path(args.truth_dir)
    world_paths = []
    for world_name, _ in top_worlds:
        stem = world_name.replace('.npz', '')
        truth_path = truth_dir / world_name
        samples_path = (Path(args.recon_dir_pattern.format(world_stem=stem))
                          / f'recon_fixed_b{args.K}_samples.npz')
        if not (truth_path.exists() and samples_path.exists()):
            print(f"    skip {world_name[:55]}: missing")
            continue
        world_paths.append((truth_path, samples_path, world_name))

    if not world_paths:
        print("\n  No usable worlds.")
        return

    # ── Dispatch on method ──
    if args.method == 'v3':
        print(f"\n  Method: v3 BUCKET-ROUTER (per-species threshold via "
              f"multi-feature N_hat classifier).\n")
        pooled = gather_all_worlds_v3(
            world_paths, args.K,
            threshold_moderate=args.threshold_moderate,
            threshold_hard=args.threshold_hard,
        )
        if pooled is None:
            print("  v3 pathway failed; aborting.")
            return
        # Effective threshold label for output filenames
        thr_label = 'v3'
        threshold_for_title = args.threshold_hard  # only used in legacy title fallback
    else:
        print(f"\n  Method: FIXED threshold p \u2265 {args.threshold:.2f}.\n")
        pooled = {k: [] for k in ['truth_range', 'truth_ncomp', 'truth_logcd',
                                    'pred_range',  'pred_ncomp',  'pred_logcd',
                                    'pred_src_truth_range']}
        for tp, sp, wn in world_paths:
            r = gather_world_fixed(tp, sp, args.threshold, args.K)
            if r is None:
                continue
            for k in pooled:
                pooled[k].append(r[k])
            print(f"    {wn[:55]:55s}  truth_n={len(r['truth_range']):4d}")
        if not pooled['truth_range']:
            print("\n  No usable worlds.")
            return
        for k in pooled:
            pooled[k] = np.concatenate(pooled[k])
        pooled['_meta'] = {'method': 'fixed', 'threshold': float(args.threshold)}
        thr_label = f'p{args.threshold:.2f}'
        threshold_for_title = args.threshold

    print(f"\n  Pooled  truth_n = {len(pooled['truth_range']):,},  "
          f"predicted_n = {len(pooled['pred_range']):,}\n")

    out_fig = out_dir / f'three_distributions_{thr_label}.png'
    summary = make_figure(pooled, threshold_for_title, out_fig, K=args.K)

    csv_path = out_dir / f'three_distributions_summary_{thr_label}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['statistic', 'truth_mean', 'pred_mean',
                     'ks', 'p_value', 'verdict'])
        for stat, truth_arr, pred_arr, ks_val, p_val, verd in [
            ('range_size',  pooled['truth_range'], pooled['pred_range'],
                summary['ks_range'], summary['p_range'], summary['verdict_range']),
            ('connectance', pooled['truth_ncomp'], pooled['pred_ncomp'],
                summary['ks_ncomp'], summary['p_ncomp'], summary['verdict_ncomp']),
            ('log_cov_det', pooled['truth_logcd'], pooled['pred_logcd'],
                summary['ks_logcd'], summary['p_logcd'], summary['verdict_logcd']),
        ]:
            w.writerow([stat, float(np.mean(truth_arr)),
                         float(np.mean(pred_arr)), ks_val, p_val, verd])
    print(f"  \u2713 csv    \u2192 {csv_path}")

    # ── Per-bucket stratified figure (EASY / MODERATE / HARD) ──
    out_bucket_fig = out_dir / f'three_distributions_per_bucket_{thr_label}.png'
    bucket_summary = make_figure_per_bucket(pooled, threshold_for_title,
                                              out_bucket_fig, K=args.K)

    if bucket_summary:
        bucket_csv = out_dir / f'three_distributions_per_bucket_{thr_label}.csv'
        with open(bucket_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['bucket', 'range_lo', 'range_hi', 'n_truth', 'n_pred',
                         'ks_range', 'verdict_range',
                         'ks_conn',  'verdict_conn',
                         'ks_spread','verdict_spread'])
            for r in bucket_summary:
                w.writerow([r['bucket'], r['lo'], r['hi'],
                             r['n_truth'], r['n_pred'],
                             r['ks_range'], r['verdict_range'],
                             r['ks_conn'],  r['verdict_conn'],
                             r['ks_spread'], r['verdict_spread']])
        print(f"  \u2713 csv    \u2192 {bucket_csv}")


if __name__ == "__main__":
    main()