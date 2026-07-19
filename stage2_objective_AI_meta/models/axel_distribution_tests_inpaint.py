#!/usr/bin/env python3
"""
=============================================================================
axel_distribution_tests_inpaint.py   (REVISION 5 + REV6 ensemble — bug-fixed)
=============================================================================

REV5 fixes three defects exposed by REV4 results:

  (1) F1 arithmetic anomaly: F1 is computed from POOLED counts (TP, FP, FN)
      over species, the correct way to summarise classification metrics
      across a set. Per-species F1 is also reported, but the headline
      number is the pooled F1.

  (2) Empty hard-band panels: panels G/H/I get an explanatory text
      annotation stating n and the ecological reason when n < MIN_N.

  (3) Best-threshold display: adds a "BEST EcoDiff" column showing the
      truth-free threshold with highest pooled F1 per band.

REV6 ADDITION (bug-fixed in this file):

  The MEAN prediction collapses to the K observed cells under averaging
  (8 high-variance samples average to a sharp, near-empty map). The
  model's real capability lives in the ENSEMBLE. REV6 adds
  `compute_novel_cell_ensemble_PRF1`, which evaluates novel-cell P/R/F1
  on three ensemble summaries — union, majority, per-sample-pooled —
  each thresholded TRUTH-FREE. A pooled cross-world summary is printed
  at the end so the ensemble F1 can be compared directly against the
  Smooth-MATCHED baseline (the comparison Axel will ask for).

  BUG-FIX NOTE: in the first REV6 edit, two blocks were pasted before the
  variables they wrote into existed:
    - `out['ensemble_novel_*'] = ...` ran before `out = {...}` was defined
      in compute_axel_metrics_for_world  -> UnboundLocalError: out
    - `e05 = m['ensemble_novel_p05']` ran before the loop that creates `m`
      in run_axel_distribution_tests    -> UnboundLocalError: m
  Both blocks are now placed AFTER their dependencies. No logic changed.

Drop-in compatible: same `run_axel_distribution_tests(...)` signature.
Reads only real .npz data. No synthetic data anywhere.
"""

import csv
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats


# =============================================================================
#  CONFIG
# =============================================================================
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)

BAND_DEGENERATE = (1, 5)
BAND_EASY       = (6, 10)
BAND_MODERATE   = (11, 20)
BAND_HARD       = (21, 10000)

MIN_N_PER_BAND        = 5
HARD_BAND_POWER_FLOOR = 30

SAMPLE_THRESHOLDS = {
    'p>=0.5':  ('fixed',   0.5),
    'p>=0.7':  ('fixed',   0.7),
    'p>=0.9':  ('fixed',   0.9),
    'topQ_K3': ('topQ',    3),
    'topQ_K5': ('topQ',    5),
    'otsu':    ('otsu',    None),
}


# =============================================================================
#  A. TRUTH-FREE THRESHOLDS
# =============================================================================

def threshold_fixed(prob, threshold=0.5):
    return (prob >= threshold).astype(np.uint8)


def threshold_per_sample_topQ(prob, observed, multiplier=3):
    S, Y, X = prob.shape
    binary = np.zeros_like(prob, dtype=np.uint8)
    for s in range(S):
        K_obs = int(observed[s].sum())
        if K_obs == 0:
            continue
        Q = min(K_obs * multiplier, Y * X)
        flat = prob[s].ravel()
        if flat.max() < 1e-6:
            continue
        # BUG-FIX: previous code used `thr = partition[-Q] - 1e-9` then
        # `prob > thr` to select the top-Q inclusively. `prob` is float32;
        # at values ~0.88 the float32 ULP (~6e-8) is larger than 1e-9, so
        # `kth - 1e-9 == kth` exactly and the comparison silently dropped
        # the Q-th cell (selecting Q-1, or fewer with near-ties). Using
        # `>= kth` is exact and dtype-independent: it selects exactly Q
        # cells when values are distinct, or all tied boundary cells.
        kth = np.partition(flat, -Q)[-Q]
        binary[s] = (prob[s] >= kth).astype(np.uint8)
    return binary


def _otsu_threshold_1d(values):
    if values.size == 0 or values.max() == values.min():
        return values.max() if values.size else 0.5
    hist, edges = np.histogram(values, bins=256, range=(0, 1))
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mu_T = mu[-1]
    denom = omega * (1 - omega)
    denom[denom < 1e-12] = 1e-12
    sigma_b2 = (mu_T * omega - mu) ** 2 / denom
    idx = int(np.argmax(sigma_b2))
    return float((edges[idx] + edges[idx + 1]) / 2)


def threshold_otsu_per_species(prob):
    S = prob.shape[0]
    binary = np.zeros_like(prob, dtype=np.uint8)
    for s in range(S):
        thr = _otsu_threshold_1d(prob[s].ravel())
        binary[s] = (prob[s] >= thr).astype(np.uint8)
    return binary


def _calibrate_per_species_topn_LEGACY(prob, truth):
    S = prob.shape[0]
    binary = np.zeros_like(prob, dtype=np.uint8)
    for s in range(S):
        n_truth = int(truth[s].sum())
        if n_truth == 0:
            continue
        flat = prob[s].ravel()
        if flat.max() < 1e-6:
            continue
        thr = np.partition(flat, -n_truth)[-n_truth] - 1e-9
        binary[s] = (prob[s] > thr).astype(np.uint8)
    return binary


def apply_sample_threshold(prob, observed, mode, param):
    if mode == 'fixed':
        return threshold_fixed(prob, param)
    elif mode == 'topQ':
        return threshold_per_sample_topQ(prob, observed, param)
    elif mode == 'otsu':
        return threshold_otsu_per_species(prob)
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")


# =============================================================================
#  B. CHEAT BASELINES
# =============================================================================

def cheat_all_cells_occupied(shape_like):
    return np.ones_like(shape_like, dtype=np.uint8)


def cheat_only_observations(observed):
    return observed.astype(np.uint8)


def cheat_smoothed_observations(observed, sigma=2.0, threshold=0.3):
    S, Y, X = observed.shape
    pred = np.zeros((S, Y, X), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
    for s in range(S):
        obs_cells = np.argwhere(observed[s] > 0)
        if len(obs_cells) == 0:
            continue
        d2 = np.full((Y, X), np.inf)
        for (oy, ox) in obs_cells:
            d2 = np.minimum(d2, (yy - oy) ** 2 + (xx - ox) ** 2)
        prob = np.exp(-d2 / (sigma ** 2))
        pred[s] = (prob >= threshold).astype(np.uint8)
    return pred


def cheat_smoothed_observations_matched(observed, target_pred):
    """Smoothed baseline calibrated to predict the same number of cells per
    species as `target_pred` (model). Truth-free."""
    S, Y, X = observed.shape
    pred = np.zeros((S, Y, X), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
    for s in range(S):
        obs_cells = np.argwhere(observed[s] > 0)
        if len(obs_cells) == 0:
            continue
        n_target = int(target_pred[s].sum())
        if n_target == 0:
            continue
        d2 = np.full((Y, X), np.inf)
        for (oy, ox) in obs_cells:
            d2 = np.minimum(d2, (yy - oy) ** 2 + (xx - ox) ** 2)
        n_target = min(n_target, Y * X)
        flat = -d2.ravel()
        thr = np.partition(flat, -n_target)[-n_target] - 1e-9
        pred[s] = ((-d2) > thr).astype(np.uint8)
    return pred


def cheat_random_matched_density(target_pred, seed=42):
    """Random predictions matched to `target_pred`'s density per species.
    REV5 FIX: matched to MODEL density (truth-free), not truth density."""
    rng = np.random.default_rng(seed)
    S, Y, X = target_pred.shape
    pred = np.zeros((S, Y, X), dtype=np.uint8)
    for s in range(S):
        n_t = int(target_pred[s].sum())
        if n_t == 0:
            continue
        idx = rng.choice(Y * X, size=min(n_t, Y * X), replace=False)
        for c in idx:
            pred[s, c // X, c % X] = 1
    return pred


# =============================================================================
#  C. SINGLE-PREDICTION DISTRIBUTION TESTS
# =============================================================================

def count_connected_components(binary_map):
    if binary_map.sum() == 0:
        return 0
    _, n = ndimage.label(binary_map, structure=CONNECTIVITY_STRUCTURE)
    return int(n)


def compute_range_size_distribution(truth_bin, pred_bin, min_range=6):
    tr_all = truth_bin.sum(axis=(1, 2))
    pr_all = pred_bin.sum(axis=(1, 2))
    mask = tr_all >= min_range
    tr = tr_all[mask].astype(np.int32)
    pr = pr_all[mask].astype(np.int32)
    if len(tr) < MIN_N_PER_BAND:
        return _empty_range(tr, pr)
    ks = stats.ks_2samp(tr, pr)
    delta = pr - tr
    return {
        'truth_ranges': tr, 'pred_ranges': pr, 'delta': delta,
        'ks_statistic': float(ks.statistic), 'ks_pvalue': float(ks.pvalue),
        'truth_median': float(np.median(tr)), 'pred_median': float(np.median(pr)),
        'truth_mean': float(np.mean(tr)),    'pred_mean': float(np.mean(pr)),
        'mean_abs_delta': float(np.mean(np.abs(delta))),
        'fraction_within_one_cell': float(np.mean(np.abs(delta) <= 1)),
        'n': len(tr),
    }


def _empty_range(t, p):
    return {
        'truth_ranges': t, 'pred_ranges': p, 'delta': np.array([], np.int32),
        'ks_statistic': np.nan, 'ks_pvalue': np.nan,
        'truth_median': np.nan, 'pred_median': np.nan,
        'truth_mean': np.nan, 'pred_mean': np.nan,
        'mean_abs_delta': np.nan, 'fraction_within_one_cell': np.nan,
        'n': len(t),
    }


def compute_connectivity_distribution(truth_bin, pred_bin, min_range=6):
    tr = truth_bin.sum(axis=(1, 2))
    mask = tr >= min_range
    t_nc, p_nc = [], []
    for s in np.where(mask)[0]:
        t_nc.append(count_connected_components(truth_bin[s]))
        p_nc.append(count_connected_components(pred_bin[s]))
    t_nc = np.asarray(t_nc, dtype=np.int32)
    p_nc = np.asarray(p_nc, dtype=np.int32)
    if len(t_nc) < MIN_N_PER_BAND:
        return {
            'truth_ncomp': t_nc, 'pred_ncomp': p_nc,
            'ks_statistic': np.nan, 'ks_pvalue': np.nan,
            'truth_mean': np.nan, 'pred_mean': np.nan,
            'fragmentation_ratio': np.nan, 'n': len(t_nc),
        }
    ks = stats.ks_2samp(t_nc, p_nc)
    tm, pm = float(np.mean(t_nc)), float(np.mean(p_nc))
    return {
        'truth_ncomp': t_nc, 'pred_ncomp': p_nc,
        'ks_statistic': float(ks.statistic), 'ks_pvalue': float(ks.pvalue),
        'truth_mean': tm, 'pred_mean': pm,
        'fragmentation_ratio': pm / max(tm, 1e-9),
        'n': len(t_nc),
    }


# =============================================================================
#  D. ENSEMBLE DISTRIBUTION SWEEP
# =============================================================================

def compute_ensemble_distribution_at_threshold(truth_bin, samples_prob,
                                                  observed_bin, mode, param,
                                                  min_range=6):
    n_ens, S, Y, X = samples_prob.shape
    tr_all = truth_bin.sum(axis=(1, 2))
    mask = tr_all >= min_range
    if mask.sum() < MIN_N_PER_BAND:
        return None, None

    samples_bin = np.zeros_like(samples_prob, dtype=np.uint8)
    for i in range(n_ens):
        samples_bin[i] = apply_sample_threshold(
            samples_prob[i], observed_bin, mode, param)

    tr = tr_all[mask].astype(np.int32)
    pr_pool, t_nc, p_nc_pool = [], [], []
    for s_idx in np.where(mask)[0]:
        t_nc.append(count_connected_components(truth_bin[s_idx]))
        for i in range(n_ens):
            pr_pool.append(int(samples_bin[i, s_idx].sum()))
            p_nc_pool.append(count_connected_components(samples_bin[i, s_idx]))
    pr_pool = np.asarray(pr_pool, dtype=np.int32)
    t_nc = np.asarray(t_nc, dtype=np.int32)
    p_nc_pool = np.asarray(p_nc_pool, dtype=np.int32)

    ks_r = stats.ks_2samp(tr, pr_pool)
    ks_c = stats.ks_2samp(t_nc, p_nc_pool)

    return {
        'truth_ranges': tr, 'pred_ranges_pooled': pr_pool,
        'n_truth': len(tr), 'n_pred_pooled': len(pr_pool),
        'n_ensembles': n_ens,
        'ks_statistic': float(ks_r.statistic), 'ks_pvalue': float(ks_r.pvalue),
        'truth_mean': float(np.mean(tr)), 'pred_mean': float(np.mean(pr_pool)),
        'truth_median': float(np.median(tr)),
        'pred_median': float(np.median(pr_pool)),
    }, {
        'truth_ncomp': t_nc, 'pred_ncomp_pooled': p_nc_pool,
        'ks_statistic': float(ks_c.statistic), 'ks_pvalue': float(ks_c.pvalue),
        'truth_mean': float(np.mean(t_nc)),
        'pred_mean': float(np.mean(p_nc_pool)),
        'fragmentation_ratio': float(np.mean(p_nc_pool)) / max(float(np.mean(t_nc)), 1e-9),
    }


# =============================================================================
#  E. NOVEL-CELL METRICS — POOLED (REV5 FIX)
# =============================================================================
# REV5 KEY FIX: F1 is computed from POOLED counts across species (TP, FP, FN),
# not as the mean of per-species F1. The per-species mean over-weights species
# with few novel cells. Pooled F1 is the standard classification summary.

def compute_novel_cell_pooled_counts(truth_bin, pred_bin, observed_bin, K,
                                        min_range=None):
    """
    For each species:
        TP_s = |pred ∩ truth ∩ ¬obs|
        FP_s = |pred ∩ ¬truth ∩ ¬obs|     (predicted novel, but not true)
        FN_s = |truth ∩ ¬pred ∩ ¬obs|     (true novel, but not predicted)
    These are POOLED across species within a band to compute band P/R/F1.
    """
    if min_range is None:
        min_range = K + 1
    S = truth_bin.shape[0]
    tr_all = truth_bin.sum(axis=(1, 2)).astype(np.int32)

    TP = np.zeros(S, dtype=np.int64)
    FP = np.zeros(S, dtype=np.int64)
    FN = np.zeros(S, dtype=np.int64)
    n_novel_truth = np.zeros(S, dtype=np.int32)
    n_novel_pred = np.zeros(S, dtype=np.int32)

    for s in range(S):
        t = truth_bin[s].astype(bool)
        o = observed_bin[s].astype(bool)
        p = pred_bin[s].astype(bool)
        not_o = ~o
        novel_t = t & not_o
        novel_p = p & not_o
        TP[s] = int((novel_p & novel_t).sum())
        FP[s] = int((novel_p & ~novel_t).sum())
        FN[s] = int((novel_t & ~novel_p).sum())
        n_novel_truth[s] = int(novel_t.sum())
        n_novel_pred[s] = int(novel_p.sum())

    def band_PRF1(lo, hi):
        m = (tr_all >= lo) & (tr_all <= hi) & (n_novel_truth > 0)
        n_species = int(m.sum())
        if n_species < MIN_N_PER_BAND:
            return {'n_species': n_species, 'precision': np.nan,
                    'recall': np.nan, 'f1': np.nan,
                    'TP_sum': int(TP[m].sum()) if n_species > 0 else 0,
                    'FP_sum': int(FP[m].sum()) if n_species > 0 else 0,
                    'FN_sum': int(FN[m].sum()) if n_species > 0 else 0,
                    'reported': False,
                    'insufficient_power': n_species < HARD_BAND_POWER_FLOOR}
        tp_sum = int(TP[m].sum())
        fp_sum = int(FP[m].sum())
        fn_sum = int(FN[m].sum())
        prec = tp_sum / max(1, tp_sum + fp_sum)
        rec = tp_sum / max(1, tp_sum + fn_sum)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            'n_species': n_species,
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'TP_sum': tp_sum, 'FP_sum': fp_sum, 'FN_sum': fn_sum,
            'reported': True,
            'insufficient_power': n_species < HARD_BAND_POWER_FLOOR,
        }

    meaningful = (tr_all >= min_range) & (n_novel_truth > 0)
    tp_sum = int(TP[meaningful].sum())
    fp_sum = int(FP[meaningful].sum())
    fn_sum = int(FN[meaningful].sum())
    prec = tp_sum / max(1, tp_sum + fp_sum)
    rec = tp_sum / max(1, tp_sum + fn_sum)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        'TP_per_species': TP, 'FP_per_species': FP, 'FN_per_species': FN,
        'n_novel_truth_per_species': n_novel_truth,
        'n_novel_pred_per_species': n_novel_pred,
        'band_easy':     band_PRF1(*BAND_EASY),
        'band_moderate': band_PRF1(*BAND_MODERATE),
        'band_hard':     band_PRF1(*BAND_HARD),
        'overall_precision': float(prec),
        'overall_recall': float(rec),
        'overall_f1': float(f1),
        'overall_TP': tp_sum, 'overall_FP': fp_sum, 'overall_FN': fn_sum,
        'n_meaningful_with_novel': int(meaningful.sum()),
    }


# -----------------------------------------------------------------------------
#  REV6 ADDITION — ENSEMBLE-BASED NOVEL-CELL METRICS
# -----------------------------------------------------------------------------
# The MEAN prediction collapses to the K obs cells under averaging (8
# high-variance samples average to a sharp, near-empty map). The model's
# real capability is in the ENSEMBLE. This function evaluates novel-cell
# P/R/F1 on three ensemble summaries, each thresholded TRUTH-FREE upstream:
#   - 'union'      : a cell is predicted if ANY of the n_ens samples predicts it
#   - 'majority'   : predicted if >= half the samples predict it
#   - 'per_sample' : pool TP/FP/FN across (species x n_ens samples)
#
# Defined in Section E (with the other novel-cell metrics) so it exists
# before compute_axel_metrics_for_world (Section G) calls it.

def compute_novel_cell_ensemble_PRF1(truth_bin, samples_bin, observed_bin, K,
                                       min_range=None):
    """
    truth_bin    : (S, Y, X)        binary truth
    samples_bin  : (n_ens, S, Y, X) binary, each sample ALREADY thresholded
                   truth-free (e.g. p>=0.5 per sample, or per-sample topQ)
    observed_bin : (S, Y, X)        binary K-observation mask

    Returns pooled P/R/F1 per band + overall for each summary mode.
    Band keys carry 'TP_sum'/'FP_sum'/'FN_sum'; 'overall' carries 'TP'/'FP'/'FN'.
    """
    if min_range is None:
        min_range = K + 1
    n_ens, S, Y, X = samples_bin.shape
    tr_all = truth_bin.sum(axis=(1, 2)).astype(np.int32)

    union    = (samples_bin.sum(axis=0) > 0).astype(np.uint8)            # (S,Y,X)
    majority = (samples_bin.sum(axis=0) >= (n_ens / 2.0)).astype(np.uint8)

    def pooled_counts(pred_bin_SYX):
        """pred_bin_SYX: (S,Y,X). Returns per-species TP/FP/FN/nnt arrays."""
        TP = np.zeros(S, np.int64); FP = np.zeros(S, np.int64)
        FN = np.zeros(S, np.int64); nnt = np.zeros(S, np.int32)
        for s in range(S):
            t = truth_bin[s].astype(bool); o = observed_bin[s].astype(bool)
            p = pred_bin_SYX[s].astype(bool)
            nt = t & ~o; npd = p & ~o
            TP[s] = int((npd & nt).sum())
            FP[s] = int((npd & ~nt).sum())
            FN[s] = int((nt & ~npd).sum())
            nnt[s] = int(nt.sum())
        return TP, FP, FN, nnt

    def per_sample_pooled_counts():
        """Pool over (species x samples)."""
        TP = np.zeros(S, np.int64); FP = np.zeros(S, np.int64)
        FN = np.zeros(S, np.int64); nnt = np.zeros(S, np.int32)
        for s in range(S):
            t = truth_bin[s].astype(bool); o = observed_bin[s].astype(bool)
            nt = t & ~o
            nnt[s] = int(nt.sum())
            for i in range(n_ens):
                p = samples_bin[i, s].astype(bool)
                npd = p & ~o
                TP[s] += int((npd & nt).sum())
                FP[s] += int((npd & ~nt).sum())
                FN[s] += int((nt & ~npd).sum())
        return TP, FP, FN, nnt

    def band_prf1(TP, FP, FN, nnt, lo, hi):
        m = (tr_all >= lo) & (tr_all <= hi) & (nnt > 0)
        n = int(m.sum())
        if n < MIN_N_PER_BAND:
            return {'n_species': n, 'precision': np.nan, 'recall': np.nan,
                    'f1': np.nan,
                    'TP_sum': int(TP[m].sum()) if n else 0,
                    'FP_sum': int(FP[m].sum()) if n else 0,
                    'FN_sum': int(FN[m].sum()) if n else 0,
                    'reported': False}
        tp, fp, fn = int(TP[m].sum()), int(FP[m].sum()), int(FN[m].sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {'n_species': n, 'precision': float(prec), 'recall': float(rec),
                'f1': float(f1), 'TP_sum': tp, 'FP_sum': fp, 'FN_sum': fn,
                'reported': True}

    def overall(TP, FP, FN, nnt):
        m = (tr_all >= min_range) & (nnt > 0)
        tp, fp, fn = int(TP[m].sum()), int(FP[m].sum()), int(FN[m].sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
                'TP': tp, 'FP': fp, 'FN': fn, 'n_species': int(m.sum())}

    results = {}
    for mode, counts in [
        ('union',      pooled_counts(union)),
        ('majority',   pooled_counts(majority)),
        ('per_sample', per_sample_pooled_counts()),
    ]:
        TP, FP, FN, nnt = counts
        results[mode] = {
            'band_easy':     band_prf1(TP, FP, FN, nnt, *BAND_EASY),
            'band_moderate': band_prf1(TP, FP, FN, nnt, *BAND_MODERATE),
            'band_hard':     band_prf1(TP, FP, FN, nnt, *BAND_HARD),
            'overall':       overall(TP, FP, FN, nnt),
        }
    return results


# =============================================================================
#  F. LIFT (carryover)
# =============================================================================

def compute_trivial_baseline_lift(truth_bin, pred_bin, observed_bin, K,
                                    min_range=None):
    if min_range is None:
        min_range = K + 1
    S = truth_bin.shape[0]
    tr = truth_bin.sum(axis=(1, 2)).astype(np.int32)
    mr = np.zeros(S, dtype=np.float32)
    trv = np.zeros(S, dtype=np.float32)
    valid = np.zeros(S, dtype=bool)
    for s in range(S):
        n_t = int(tr[s])
        if n_t == 0:
            continue
        mr[s] = int((pred_bin[s].astype(bool) & truth_bin[s].astype(bool)).sum()) / n_t
        trv[s] = int((observed_bin[s].astype(bool) & truth_bin[s].astype(bool)).sum()) / n_t
        valid[s] = True
    lifts = mr - trv
    meaningful = valid & (tr >= min_range)
    return {
        'overall_lift_mean': (float(lifts[meaningful].mean())
                                if meaningful.any() else np.nan),
        'n_meaningful': int(meaningful.sum()),
    }


# =============================================================================
#  G. PER-WORLD AGGREGATION
# =============================================================================

def compute_axel_metrics_for_world(truth, samples, mean_pred, observed, K=5):
    pred_05 = threshold_fixed(mean_pred, 0.5)
    pred_topQ_K3 = threshold_per_sample_topQ(mean_pred, observed, multiplier=3)
    pred_otsu = threshold_otsu_per_species(mean_pred)
    pred_topn_legacy = _calibrate_per_species_topn_LEGACY(mean_pred, truth)

    cheat_all = cheat_all_cells_occupied(truth)
    cheat_smooth_raw = cheat_smoothed_observations(observed)
    cheat_smooth_matched = cheat_smoothed_observations_matched(observed, pred_05)
    # REV5 FIX: random matched to MODEL density (truth-free)
    cheat_random = cheat_random_matched_density(pred_05, seed=42)

    minr = K + 1

    # ── Build the per-world metrics dict FIRST ───────────────────────────
    out = {
        'mean_p05': {
            'range': compute_range_size_distribution(truth, pred_05, minr),
            'connectivity': compute_connectivity_distribution(truth, pred_05, minr),
            'lift': compute_trivial_baseline_lift(truth, pred_05, observed, K),
            'novel': compute_novel_cell_pooled_counts(truth, pred_05, observed, K),
        },
        'mean_topQ_K3': {
            'range': compute_range_size_distribution(truth, pred_topQ_K3, minr),
            'connectivity': compute_connectivity_distribution(truth, pred_topQ_K3, minr),
            'novel': compute_novel_cell_pooled_counts(truth, pred_topQ_K3, observed, K),
        },
        'mean_otsu': {
            'range': compute_range_size_distribution(truth, pred_otsu, minr),
            'connectivity': compute_connectivity_distribution(truth, pred_otsu, minr),
            'novel': compute_novel_cell_pooled_counts(truth, pred_otsu, observed, K),
        },
        'mean_topN_LEGACY': {
            'range': compute_range_size_distribution(truth, pred_topn_legacy, minr),
            'novel': compute_novel_cell_pooled_counts(truth, pred_topn_legacy, observed, K),
        },
        'ensemble_sweep': {},
        'cheat_all_cells': {
            'range': compute_range_size_distribution(truth, cheat_all, minr),
            'novel': compute_novel_cell_pooled_counts(truth, cheat_all, observed, K),
        },
        'cheat_smooth_unmatched': {
            'range': compute_range_size_distribution(truth, cheat_smooth_raw, minr),
            'novel': compute_novel_cell_pooled_counts(truth, cheat_smooth_raw, observed, K),
        },
        'cheat_smooth_MATCHED': {
            'range': compute_range_size_distribution(truth, cheat_smooth_matched, minr),
            'novel': compute_novel_cell_pooled_counts(truth, cheat_smooth_matched, observed, K),
        },
        'cheat_random_matched': {
            'range': compute_range_size_distribution(truth, cheat_random, minr),
            'novel': compute_novel_cell_pooled_counts(truth, cheat_random, observed, K),
        },
    }

    # ── REV6: ensemble novel-cell metrics ───────────────────────────────
    # BUG-FIX: this block now runs AFTER `out` is defined (it writes into
    # `out`). Threshold each of the n_ens samples truth-free, then evaluate
    # the union / majority / per-sample summaries.
    n_ens = samples.shape[0]
    samples_05 = np.zeros_like(samples, dtype=np.uint8)
    samples_topQ = np.zeros_like(samples, dtype=np.uint8)
    for i in range(n_ens):
        samples_05[i]   = threshold_fixed(samples[i], 0.5)
        samples_topQ[i] = threshold_per_sample_topQ(samples[i], observed,
                                                    multiplier=3)

    out['ensemble_novel_p05']  = compute_novel_cell_ensemble_PRF1(
        truth, samples_05, observed, K)
    out['ensemble_novel_topQ'] = compute_novel_cell_ensemble_PRF1(
        truth, samples_topQ, observed, K)

    # ── Ensemble distribution sweep (range/connectivity KS) ─────────────
    for label, (mode, param) in SAMPLE_THRESHOLDS.items():
        r, c = compute_ensemble_distribution_at_threshold(
            truth, samples, observed, mode, param, minr)
        out['ensemble_sweep'][label] = {'range': r, 'connectivity': c,
                                          'mode': mode, 'param': param}
    return out


# =============================================================================
#  H. HELPERS
# =============================================================================

def _fmt(x, default='n/a'):
    if x is None:
        return default
    try:
        if np.isnan(x):
            return default
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.4f}"


def _deterministic_jitter(n, amp=0.08):
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-amp, amp, n)


def _pool_band_PRF1_across_worlds(per_world_results, method_key, band_key):
    """Pool TP/FP/FN counts across worlds for a method × band, then compute
    P/R/F1 from the totals. This is the correct way to aggregate."""
    tp_total, fp_total, fn_total, n_species = 0, 0, 0, 0
    for r in per_world_results:
        if method_key not in r or 'novel' not in r[method_key]:
            continue
        b = r[method_key]['novel'][band_key]
        tp_total += b['TP_sum']
        fp_total += b['FP_sum']
        fn_total += b['FN_sum']
        n_species += b['n_species']
    if tp_total + fp_total == 0 and tp_total + fn_total == 0:
        return {'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
                'n_species': n_species,
                'TP': tp_total, 'FP': fp_total, 'FN': fn_total}
    prec = tp_total / max(1, tp_total + fp_total)
    rec = tp_total / max(1, tp_total + fn_total)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
            'n_species': n_species,
            'TP': tp_total, 'FP': fp_total, 'FN': fn_total}


def _pool_ensemble_across_worlds(per_world_results, ens_key, mode):
    """Pool ensemble novel-cell TP/FP/FN across worlds for one summary mode
    ('union'/'majority'/'per_sample'), for each band + overall.

    Band sub-dicts carry TP_sum/FP_sum/FN_sum; 'overall' carries TP/FP/FN.
    """
    pooled = {}
    for band in ['band_easy', 'band_moderate', 'band_hard', 'overall']:
        tp = fp = fn = n = 0
        for r in per_world_results:
            if ens_key not in r:
                continue
            e = r[ens_key][mode][band]
            if band == 'overall':
                tp += e['TP']; fp += e['FP']; fn += e['FN']
                n += e['n_species']
            else:
                tp += e['TP_sum']; fp += e['FP_sum']; fn += e['FN_sum']
                n += e['n_species']
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        pooled[band] = {'precision': prec, 'recall': rec, 'f1': f1,
                        'TP': tp, 'FP': fp, 'FN': fn, 'n_species': n}
    return pooled


# =============================================================================
#  I. FIGURE 13 — distribution tests (truth-free mean prediction)
# =============================================================================

def make_fig13_truth_free(per_world_results, output_path, K=5):
    tr = np.concatenate([r['mean_p05']['range']['truth_ranges']
                          for r in per_world_results
                          if r['mean_p05']['range']['n'] >= MIN_N_PER_BAND])
    pr = np.concatenate([r['mean_p05']['range']['pred_ranges']
                          for r in per_world_results
                          if r['mean_p05']['range']['n'] >= MIN_N_PER_BAND])
    tnc = np.concatenate([r['mean_p05']['connectivity']['truth_ncomp']
                            for r in per_world_results
                            if r['mean_p05']['connectivity']['n'] >= MIN_N_PER_BAND])
    pnc = np.concatenate([r['mean_p05']['connectivity']['pred_ncomp']
                            for r in per_world_results
                            if r['mean_p05']['connectivity']['n'] >= MIN_N_PER_BAND])

    n_worlds = len(per_world_results)

    ks_r = stats.ks_2samp(tr, pr)
    ks_c = stats.ks_2samp(tnc, pnc)
    frag = float(pnc.mean()) / max(float(tnc.mean()), 1e-9)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    max_r = max(int(tr.max()), int(pr.max()))
    bins = np.arange(K, max_r + 2)
    ax.hist(tr, bins=bins, alpha=0.55, color='#2c7fb8', edgecolor='black',
             linewidth=0.4, label='Truth')
    ax.hist(pr, bins=bins, alpha=0.55, color='#fc8d59', edgecolor='black',
             linewidth=0.4, label='Pred (mean, p≥0.5)')
    ax.set_xlabel('Range size (cells)', fontsize=11)
    ax.set_ylabel('Species count', fontsize=11)
    ax.set_title('(A) RANGE-SIZE  (MEAN, truth-free p≥0.5)',
                  fontweight='bold', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.97, 0.97,
              f'KS = {ks_r.statistic:.3f}\np  = {ks_r.pvalue:.2e}\n'
              f'truth med = {np.median(tr):.1f}\n'
              f'pred  med = {np.median(pr):.1f}\n'
              f'truth mean = {np.mean(tr):.2f}\n'
              f'pred  mean = {np.mean(pr):.2f}',
              transform=ax.transAxes, fontsize=8.5,
              verticalalignment='top', horizontalalignment='right',
              family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    ax = axes[0, 1]
    x_t = np.sort(tr); y_t = np.arange(1, len(x_t)+1)/len(x_t)
    x_p = np.sort(pr); y_p = np.arange(1, len(x_p)+1)/len(x_p)
    ax.step(x_t, y_t, where='post', color='#2c7fb8', linewidth=2, label='Truth')
    ax.step(x_p, y_p, where='post', color='#fc8d59', linewidth=2, label='Pred')
    ax.set_xlabel('Range size (cells)', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title(f'(B) RANGE-SIZE CDF  (KS = {ks_r.statistic:.3f})',
                  fontweight='bold', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ks_p05 = [r['mean_p05']['range']['ks_statistic'] for r in per_world_results
                if not np.isnan(r['mean_p05']['range'].get('ks_statistic', np.nan))]
    ks_topQ = [r['mean_topQ_K3']['range']['ks_statistic'] for r in per_world_results
                 if not np.isnan(r['mean_topQ_K3']['range'].get('ks_statistic', np.nan))]
    ks_otsu = [r['mean_otsu']['range']['ks_statistic'] for r in per_world_results
                if not np.isnan(r['mean_otsu']['range'].get('ks_statistic', np.nan))]
    ks_legacy = [r['mean_topN_LEGACY']['range']['ks_statistic']
                   for r in per_world_results
                   if not np.isnan(r['mean_topN_LEGACY']['range'].get('ks_statistic', np.nan))]
    data = [ks_p05, ks_topQ, ks_otsu, ks_legacy]
    labels = ['p≥0.5\n(primary)', 'top-Q\n(Q=3K_obs)', 'Otsu\nper-species',
              'top-N\n(USES TRUTH —\nfor reference)']
    bp = ax.boxplot(data, positions=[1, 2, 3, 4], tick_labels=labels,
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    for b, c in zip(bp['boxes'], ['#83b860', '#a8c8e8', '#fdb462', '#fb8072']):
        b.set_facecolor(c)
    for i, vals in enumerate(data, 1):
        x = i + _deterministic_jitter(len(vals))
        ax.scatter(x, vals, color='black', s=22, zorder=3, alpha=0.8)
    ax.set_ylabel('Range-size KS distance', fontsize=11)
    ax.set_title('(C) MEAN prediction — threshold mode comparison',
                  fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.97,
              f'p≥0.5:  {np.mean(ks_p05):.3f}\n'
              f'topQ:   {np.mean(ks_topQ):.3f}\n'
              f'Otsu:   {np.mean(ks_otsu):.3f}\n'
              f'top-N:  {np.mean(ks_legacy):.3f}',
              transform=ax.transAxes, fontsize=8,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    ax = axes[1, 0]
    max_nc = max(int(tnc.max()), int(pnc.max()))
    bins = np.arange(0, max_nc + 2) - 0.5
    ax.hist(tnc, bins=bins, alpha=0.55, color='#2c7fb8', edgecolor='black',
             linewidth=0.4, label='Truth')
    ax.hist(pnc, bins=bins, alpha=0.55, color='#fc8d59', edgecolor='black',
             linewidth=0.4, label='Pred')
    ax.set_xlabel('Patches per species', fontsize=11)
    ax.set_ylabel('Species count', fontsize=11)
    ax.set_title('(D) CONNECTIVITY', fontweight='bold', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.97, 0.97,
              f'KS = {ks_c.statistic:.3f}\np  = {ks_c.pvalue:.2e}\n'
              f'truth mean = {tnc.mean():.2f}\n'
              f'pred  mean = {pnc.mean():.2f}\n'
              f'fragmentation = {frag:.2f}\n'
              f'{"over-frag" if frag > 1.1 else ("under-frag" if frag < 0.9 else "MATCH")}',
              transform=ax.transAxes, fontsize=8.5,
              verticalalignment='top', horizontalalignment='right',
              family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    ax = axes[1, 1]
    x_t = np.sort(tnc); y_t = np.arange(1, len(x_t)+1)/len(x_t)
    x_p = np.sort(pnc); y_p = np.arange(1, len(x_p)+1)/len(x_p)
    ax.step(x_t, y_t, where='post', color='#2c7fb8', linewidth=2, label='Truth')
    ax.step(x_p, y_p, where='post', color='#fc8d59', linewidth=2, label='Pred')
    ax.set_xlabel('Patches per species', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title(f'(E) CONNECTIVITY CDF  (KS = {ks_c.statistic:.3f})',
                  fontweight='bold', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ks_smooth = [r['cheat_smooth_unmatched']['range']['ks_statistic']
                  for r in per_world_results
                  if not np.isnan(r['cheat_smooth_unmatched']['range'].get('ks_statistic', np.nan))]
    ks_smooth_m = [r['cheat_smooth_MATCHED']['range']['ks_statistic']
                     for r in per_world_results
                     if not np.isnan(r['cheat_smooth_MATCHED']['range'].get('ks_statistic', np.nan))]
    ks_rand = [r['cheat_random_matched']['range']['ks_statistic']
                for r in per_world_results
                if not np.isnan(r['cheat_random_matched']['range'].get('ks_statistic', np.nan))]
    ks_all = [r['cheat_all_cells']['range']['ks_statistic']
               for r in per_world_results
               if not np.isnan(r['cheat_all_cells']['range'].get('ks_statistic', np.nan))]
    data = [ks_p05, ks_smooth_m, ks_smooth, ks_rand, ks_all]
    labels = ['EcoDiff\np≥0.5', 'Smooth\nMATCHED', 'Smooth\nraw',
              'Random\nmatched', 'All\ncells']
    bp = ax.boxplot(data, positions=range(1, 6), tick_labels=labels,
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    for b, c in zip(bp['boxes'], ['#83b860', '#fdcb74', '#fdb462',
                                     '#a8c8e8', '#fb8072']):
        b.set_facecolor(c)
    ax.set_ylabel('Range-size KS distance', fontsize=11)
    ax.set_title('(F) Cheat baselines — Axel 6:51',
                  fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Figure 13 — Distribution tests (REV5, MEAN prediction, truth-free)\n'
        f'K={K}, {len(tr):,} meaningful species, {n_worlds} worlds',
        fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 13 → {output_path}")
    return {
        'range_ks_p05': float(ks_r.statistic),
        'conn_ks_p05': float(ks_c.statistic),
        'fragmentation_p05': frag,
        'n_pooled': len(tr),
    }


# =============================================================================
#  J. FIGURE 14 — NOVEL-CELL P/R/F1, POOLED, with BEST EcoDiff column
# =============================================================================

def make_fig14_novel_PRF1(per_world_results, output_path, K=5):
    """REV5: POOLED P/R/F1 (TP+FP+FN summed across species, then ratios).
    Includes a 'BEST EcoDiff' column = best truth-free EcoDiff threshold per band.
    Empty bands get text annotation instead of blank panel."""

    method_specs = [
        ('EcoDiff p≥0.5', 'mean_p05', '#2c7fb8'),
        ('EcoDiff topQ',  'mean_topQ_K3', '#5b8dd6'),
        ('EcoDiff Otsu',  'mean_otsu', '#a8c8e8'),
        ('BEST EcoDiff',  '__BEST__', '#1a5f1a'),    # filled in below
        ('Smooth MATCHED', 'cheat_smooth_MATCHED', '#fdb462'),
        ('Smooth raw',    'cheat_smooth_unmatched', '#fc8d59'),
        ('Random matched', 'cheat_random_matched', '#bbbbbb'),
    ]

    bands = [('easy 6-10', 'band_easy'),
             ('moderate 11-20', 'band_moderate'),
             ('hard 21+', 'band_hard')]
    metrics = ['precision', 'recall', 'f1']

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    for row, (band_label, band_key) in enumerate(bands):
        # Compute pooled P/R/F1 for each method in this band
        eco_keys = ['mean_p05', 'mean_topQ_K3', 'mean_otsu']
        eco_pooled = {k: _pool_band_PRF1_across_worlds(per_world_results, k, band_key)
                       for k in eco_keys}
        # BEST EcoDiff = highest F1 among EcoDiff thresholds (truth-free)
        best_eco_key = max(eco_keys, key=lambda k:
                           (eco_pooled[k]['f1'] if not np.isnan(eco_pooled[k]['f1']) else -1))
        best_eco_label = best_eco_key.replace('mean_', '')

        # Build full method table
        rows_data = {}
        for label, key, _ in method_specs:
            if key == '__BEST__':
                rows_data[label] = eco_pooled[best_eco_key].copy()
                rows_data[label]['_best_label'] = best_eco_label
            else:
                rows_data[label] = _pool_band_PRF1_across_worlds(
                    per_world_results, key, band_key)

        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            method_labels = [m[0] for m in method_specs]
            method_colors = [m[2] for m in method_specs]
            vals = [rows_data[ml].get(metric, np.nan) for ml in method_labels]
            ns = [rows_data[ml].get('n_species', 0) for ml in method_labels]

            # Empty-band handling
            if all(np.isnan(v) for v in vals):
                ax.axis('off')
                ax.text(0.5, 0.5,
                          f'({chr(65+row*3+col)}) {band_label} — novel-cell {metric.upper()}\n\n'
                          f'INSUFFICIENT POWER\n\n'
                          f'n species pooled = {ns[0]} < {MIN_N_PER_BAND}\n\n'
                          f'Ecological note (REV5):\n'
                          f'IBM produces few large-range species\n'
                          f'on 20×20 grids.  Need 40×40 or different\n'
                          f'IBM parameters for power in this band.',
                          transform=ax.transAxes, ha='center', va='center',
                          fontsize=10, family='monospace',
                          bbox=dict(boxstyle='round', facecolor='#fff0f0',
                                    edgecolor='#cc6666', alpha=0.92))
                continue

            x = np.arange(len(method_labels))
            bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in vals],
                           color=method_colors, edgecolor='black', linewidth=0.6)
            # Highlight BEST column
            bars[3].set_edgecolor('#1a5f1a')
            bars[3].set_linewidth(2.5)

            for b, v, n in zip(bars, vals, ns):
                if not np.isnan(v):
                    flag = '⚠' if n < HARD_BAND_POWER_FLOOR else ''
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                            f'{v:.2f}\nn={n}{flag}', ha='center', va='bottom',
                            fontsize=7.5)
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=35, ha='right', fontsize=7.5)
            ax.set_ylim(0, max(0.5, max([v for v in vals if not np.isnan(v)] + [0.1]) * 1.25))
            ax.set_ylabel(metric, fontsize=10)
            best_note = (f"  [BEST = {rows_data['BEST EcoDiff']['_best_label']}]"
                          if not np.isnan(rows_data['BEST EcoDiff'].get(metric, np.nan)) else '')
            ax.set_title(f'({chr(65+row*3+col)}) {band_label}  —  novel-cell {metric.upper()}{best_note}',
                          fontweight='bold', fontsize=9.5)
            ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Figure 14 — Novel-cell PRECISION × RECALL × F1 (REV5, POOLED)\n'
        f'K={K} obs cells excluded from numerator AND denominator. '
        f'F1 computed from TP/FP/FN summed across species (standard).  '
        f'⚠ = n < {HARD_BAND_POWER_FLOOR}.',
        fontweight='bold', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 14 → {output_path}")

    # Return overall pooled metrics for printing
    overall = {}
    for label, key, _ in method_specs:
        if key == '__BEST__':
            continue
        tp, fp, fn = 0, 0, 0
        for r in per_world_results:
            if key in r and 'novel' in r[key]:
                tp += r[key]['novel']['overall_TP']
                fp += r[key]['novel']['overall_FP']
                fn += r[key]['novel']['overall_FN']
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        overall[label] = {'precision': prec, 'recall': rec, 'f1': f1,
                          'TP': tp, 'FP': fp, 'FN': fn}
    return overall


# =============================================================================
#  K. FIGURE 15 — ENSEMBLE SWEEP
# =============================================================================

def make_fig15_ensemble_sweep(per_world_results, output_path, K=5):
    threshold_labels = list(SAMPLE_THRESHOLDS.keys())
    pooled = {}
    for label in threshold_labels:
        truth_r_list, pred_r_list = [], []
        truth_nc_list, pred_nc_list = [], []
        for r in per_world_results:
            d = r['ensemble_sweep'][label]
            if d['range'] is None or d['connectivity'] is None:
                continue
            truth_r_list.append(d['range']['truth_ranges'])
            pred_r_list.append(d['range']['pred_ranges_pooled'])
            truth_nc_list.append(d['connectivity']['truth_ncomp'])
            pred_nc_list.append(d['connectivity']['pred_ncomp_pooled'])
        if not truth_r_list:
            continue
        tr = np.concatenate(truth_r_list); pr = np.concatenate(pred_r_list)
        tnc = np.concatenate(truth_nc_list); pnc = np.concatenate(pred_nc_list)
        ks_r = stats.ks_2samp(tr, pr)
        ks_c = stats.ks_2samp(tnc, pnc)
        pooled[label] = {
            'tr': tr, 'pr': pr, 'tnc': tnc, 'pnc': pnc,
            'ks_r': float(ks_r.statistic), 'p_r': float(ks_r.pvalue),
            'ks_c': float(ks_c.statistic), 'p_c': float(ks_c.pvalue),
            'pred_mean': float(np.mean(pr)),
            'pred_med': float(np.median(pr)),
            'frag': float(np.mean(pnc)) / max(float(np.mean(tnc)), 1e-9),
        }

    best = min(pooled.keys(), key=lambda k: pooled[k]['ks_r']) if pooled else None

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    ax = axes[0, 0]
    labels_plot = [l for l in threshold_labels if l in pooled]
    ks_r_vals = [pooled[l]['ks_r'] for l in labels_plot]
    pred_means = [pooled[l]['pred_mean'] for l in labels_plot]
    truth_mean = float(np.mean(pooled[labels_plot[0]]['tr'])) if labels_plot else 0
    colors_bar = ['#83b860' if l == best else '#a8c8e8' for l in labels_plot]
    bars = ax.bar(range(len(labels_plot)), ks_r_vals, color=colors_bar,
                    edgecolor='black', linewidth=0.6)
    for b, v, pm in zip(bars, ks_r_vals, pred_means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'KS={v:.3f}\nmean={pm:.1f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=35, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Range-size KS distance', fontsize=11)
    ax.set_title(f'(A) Ensemble range KS by threshold\n'
                  f'truth mean = {truth_mean:.1f}  —  GREEN = BEST truth-free',
                  fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    if best:
        d = pooled[best]
        max_r = max(int(d['tr'].max()), int(d['pr'].max()))
        bins = np.arange(0, max_r + 2)
        ax.hist(d['tr'], bins=bins, alpha=0.55, color='#2c7fb8',
                 edgecolor='black', linewidth=0.4, density=True,
                 label=f'Truth ({len(d["tr"]):,})')
        ax.hist(d['pr'], bins=bins, alpha=0.55, color='#fc8d59',
                 edgecolor='black', linewidth=0.4, density=True,
                 label=f'Samples @ {best} ({len(d["pr"]):,})')
        ax.set_xlabel('Range size (cells)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'(B) BEST: {best}\n'
                      f'KS = {d["ks_r"]:.3f}  (p = {d["p_r"]:.2e})',
                      fontweight='bold', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 2]
    if 'p>=0.5' in pooled:
        d = pooled['p>=0.5']
        max_r = max(int(d['tr'].max()), int(d['pr'].max()))
        bins = np.arange(0, max_r + 2)
        ax.hist(d['tr'], bins=bins, alpha=0.55, color='#2c7fb8',
                 edgecolor='black', linewidth=0.4, density=True, label='Truth')
        ax.hist(d['pr'], bins=bins, alpha=0.55, color='#fc8d59',
                 edgecolor='black', linewidth=0.4, density=True,
                 label='Samples @ p≥0.5')
        ax.set_xlabel('Range size (cells)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'(C) p≥0.5 — over-confident samples\n'
                      f'KS = {d["ks_r"]:.3f}  (shows why p≥0.5 wrong for samples)',
                      fontweight='bold', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    if pooled:
        ks_c_vals = [pooled[l]['ks_c'] for l in labels_plot]
        frags = [pooled[l]['frag'] for l in labels_plot]
        bars = ax.bar(range(len(labels_plot)), ks_c_vals,
                       color=['#83b860' if l == best else '#a8c8e8'
                              for l in labels_plot],
                       edgecolor='black', linewidth=0.6)
        for b, v, fr in zip(bars, ks_c_vals, frags):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'KS={v:.3f}\nfrag={fr:.2f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(labels_plot)))
        ax.set_xticklabels(labels_plot, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Connectivity KS', fontsize=11)
        ax.set_title('(D) Ensemble connectivity KS',
                      fontweight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    if best and best in pooled:
        d = pooled[best]
        max_nc = max(int(d['tnc'].max()), int(d['pnc'].max()))
        bins = np.arange(0, max_nc + 2) - 0.5
        ax.hist(d['tnc'], bins=bins, alpha=0.55, color='#2c7fb8',
                 edgecolor='black', linewidth=0.4, density=True, label='Truth')
        ax.hist(d['pnc'], bins=bins, alpha=0.55, color='#fc8d59',
                 edgecolor='black', linewidth=0.4, density=True,
                 label=f'Samples @ {best}')
        ax.set_xlabel('Patches per species', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'(E) Connectivity @ {best}\n'
                      f'KS = {d["ks_c"]:.3f}  frag = {d["frag"]:.2f}',
                      fontweight='bold', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 2]
    ax.axis('off')
    if pooled and best:
        d_best = pooled[best]
        d_05 = pooled.get('p>=0.5', None)
        txt = (
            "AXEL'S QUESTION (50:31–51:25)\n"
            "─" * 50 + "\n\n"
            "Q: Do statistics across multiple plausible\n"
            "    samples match the truth distribution?\n\n"
            "REV5 SWEEP — TRUTH-FREE THRESHOLDS\n"
            "─" * 50 + "\n"
        )
        for label in labels_plot:
            d = pooled[label]
            mark = '  ← BEST' if label == best else ''
            txt += (f"  {label:<10s}  range_KS = {d['ks_r']:.3f}  "
                    f"mean = {d['pred_mean']:.1f}{mark}\n")
        txt += "\n" + "─" * 50 + "\n"
        txt += "FINDING\n"
        txt += "─" * 50 + "\n"
        if d_05:
            txt += (f"At p≥0.5: each sample predicts ~{d_05['pred_mean']:.0f}\n"
                    f"cells (truth ~{np.mean(d_05['tr']):.0f}). Diffusion samples\n"
                    f"are over-confident on uncertain cells.\n\n")
        txt += (f"Best truth-free threshold: {best}\n"
                f"  range KS  = {d_best['ks_r']:.3f}\n"
                f"  conn KS   = {d_best['ks_c']:.3f}\n"
                f"  frag      = {d_best['frag']:.2f}\n\n")
        txt += "─" * 50 + "\n"
        txt += "HONEST READING\n"
        txt += "─" * 50 + "\n"
        txt += ("Per-sample range distribution does NOT\n"
                "fully match truth at any threshold. The\n"
                "MEAN prediction is sharper but collapses\n"
                "to the K obs cells. The ENSEMBLE union\n"
                "(see REV6 pooled summary in console) is\n"
                "the object that recovers most truth cells.\n"
                "Improving per-sample sharpness requires\n"
                "noise-schedule tuning or stricter\n"
                "training-time mask sparsity.")
    ax.text(0.0, 0.98, txt, transform=ax.transAxes, fontsize=8.8,
              verticalalignment='top', family='monospace', linespacing=1.30)

    fig.suptitle(
        f'Figure 15 — Ensemble sweep (REV5, truth-free thresholds)\n'
        f'Per-sample (8/species) pooled across worlds. K={K}',
        fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 15 → {output_path}")

    return {
        'best_threshold': best,
        'best_range_ks': pooled[best]['ks_r'] if best else np.nan,
        'best_conn_ks': pooled[best]['ks_c'] if best else np.nan,
        'all_pooled': {l: {'ks_r': pooled[l]['ks_r'],
                            'ks_c': pooled[l]['ks_c'],
                            'pred_mean': pooled[l]['pred_mean']}
                        for l in pooled},
    }


# =============================================================================
#  L. CSV
# =============================================================================

def save_axel_distribution_csv(per_world_results, world_names, output_path):
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['world',
                   'p05_range_ks', 'p05_conn_ks', 'p05_frag',
                   # Pooled-counts novel P/R/F1 per band, p05
                   'p05_novel_pre_easy', 'p05_novel_rec_easy', 'p05_novel_F1_easy',
                   'p05_novel_TP_easy', 'p05_novel_FP_easy', 'p05_novel_FN_easy',
                   'p05_novel_pre_mod', 'p05_novel_rec_mod', 'p05_novel_F1_mod',
                   'p05_novel_TP_mod', 'p05_novel_FP_mod', 'p05_novel_FN_mod',
                   'p05_novel_pre_hard', 'p05_novel_rec_hard', 'p05_novel_F1_hard',
                   'p05_novel_TP_hard', 'p05_novel_FP_hard', 'p05_novel_FN_hard',
                   'p05_novel_overall_pre', 'p05_novel_overall_rec', 'p05_novel_overall_F1',
                   # topQ, otsu, legacy
                   'topQ_range_ks', 'topQ_novel_overall_F1',
                   'otsu_range_ks', 'otsu_novel_overall_F1',
                   'topN_LEG_range_ks',
                   # ensemble sweep
                   'ens_range_ks_p05', 'ens_range_ks_p07', 'ens_range_ks_p09',
                   'ens_range_ks_topQK3', 'ens_range_ks_topQK5', 'ens_range_ks_otsu',
                   # REV6 ensemble novel-cell F1 (overall), p>=0.5 thresholding
                   'ens_novel_F1_union_p05', 'ens_novel_F1_majority_p05',
                   'ens_novel_F1_persample_p05',
                   'ens_novel_rec_union_p05', 'ens_novel_pre_union_p05',
                   # REV6 ensemble novel-cell F1 (overall), topQ thresholding
                   'ens_novel_F1_union_topQ', 'ens_novel_F1_persample_topQ',
                   # cheats
                   'smooth_MATCHED_range_ks', 'smooth_MATCHED_novel_F1',
                   'smooth_raw_range_ks',     'smooth_raw_novel_F1',
                   'random_range_ks',         'random_novel_F1',
                   'all_cells_range_ks',
                   ]
        w.writerow(header)
        for r, wn in zip(per_world_results, world_names):
            row = [wn]
            mp = r['mean_p05']; mq = r['mean_topQ_K3']; mo = r['mean_otsu']
            ml = r['mean_topN_LEGACY']
            row += [_fmt(mp['range']['ks_statistic']),
                     _fmt(mp['connectivity']['ks_statistic']),
                     _fmt(mp['connectivity']['fragmentation_ratio']),
                     ]
            for bk in ['band_easy', 'band_moderate', 'band_hard']:
                b = mp['novel'][bk]
                row += [_fmt(b['precision']), _fmt(b['recall']), _fmt(b['f1']),
                         b['TP_sum'], b['FP_sum'], b['FN_sum']]
            row += [_fmt(mp['novel']['overall_precision']),
                     _fmt(mp['novel']['overall_recall']),
                     _fmt(mp['novel']['overall_f1']),
                     _fmt(mq['range']['ks_statistic']),
                     _fmt(mq['novel']['overall_f1']),
                     _fmt(mo['range']['ks_statistic']),
                     _fmt(mo['novel']['overall_f1']),
                     _fmt(ml['range']['ks_statistic']),
                     ]
            for label in ['p>=0.5', 'p>=0.7', 'p>=0.9',
                          'topQ_K3', 'topQ_K5', 'otsu']:
                d = r['ensemble_sweep'][label]['range']
                row.append(_fmt(d['ks_statistic']) if d else 'n/a')
            # REV6 ensemble novel-cell metrics (overall)
            e05 = r['ensemble_novel_p05']
            etq = r['ensemble_novel_topQ']
            row += [_fmt(e05['union']['overall']['f1']),
                     _fmt(e05['majority']['overall']['f1']),
                     _fmt(e05['per_sample']['overall']['f1']),
                     _fmt(e05['union']['overall']['recall']),
                     _fmt(e05['union']['overall']['precision']),
                     _fmt(etq['union']['overall']['f1']),
                     _fmt(etq['per_sample']['overall']['f1']),
                     ]
            row += [_fmt(r['cheat_smooth_MATCHED']['range']['ks_statistic']),
                     _fmt(r['cheat_smooth_MATCHED']['novel']['overall_f1']),
                     _fmt(r['cheat_smooth_unmatched']['range']['ks_statistic']),
                     _fmt(r['cheat_smooth_unmatched']['novel']['overall_f1']),
                     _fmt(r['cheat_random_matched']['range']['ks_statistic']),
                     _fmt(r['cheat_random_matched']['novel']['overall_f1']),
                     _fmt(r['cheat_all_cells']['range']['ks_statistic']),
                     ]
            w.writerow(row)
    print(f"  ✓ CSV → {output_path}")


# =============================================================================
#  M. ENTRY POINT
# =============================================================================

def run_axel_distribution_tests(worlds_loaded, all_results, out_dir, K=5,
                                  calibrate_fn=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("  AXEL DISTRIBUTION TESTS  (Fig 13, 14, 15)  — REV5 + REV6 ensemble")
    print("=" * 72)
    print(f"  Fig 13:  MEAN @ truth-free p≥0.5  (range/conn dist tests)")
    print(f"  Fig 14:  novel P/R/F1, POOLED counts (TP/FP/FN summed)")
    print(f"  Fig 15:  ENSEMBLE sweep over 6 truth-free thresholds")
    print(f"  REV6:    ENSEMBLE union/majority/per-sample novel-cell F1")
    print(f"  Worlds: {len(worlds_loaded)}, K={K}")
    print()

    # ── Per-world metrics ────────────────────────────────────────────────
    # BUG-FIX: the REV6 ensemble print uses `m`, so it must be INSIDE this
    # loop (after `m` is computed), not before it.
    per_world_results = []
    world_names = []
    for w in worlds_loaded:
        m = compute_axel_metrics_for_world(
            w['truth'], w['samples'], w['mean_pred'], w['observed'], K=K)
        per_world_results.append(m)
        world_names.append(w['world'])

        mp = m['mean_p05']
        eco_f1 = mp['novel']['overall_f1']
        sm_f1 = m['cheat_smooth_MATCHED']['novel']['overall_f1']
        e05 = m['ensemble_novel_p05']           # REV6 — now `m` exists
        print(
            f"  {w['world'][:48]}:\n"
            f"    Fig 13:  range_KS={mp['range']['ks_statistic']:.3f}, "
            f"conn_KS={mp['connectivity']['ks_statistic']:.3f}, "
            f"frag={mp['connectivity']['fragmentation_ratio']:.2f}\n"
            f"    Fig 14:  novel_F1 (model MEAN p≥0.5) = {eco_f1:.3f}  "
            f"vs  smooth_MATCHED = {sm_f1:.3f}\n"
            f"    REV6:    ensemble novel_F1 (p≥0.5)  "
            f"union={e05['union']['overall']['f1']:.3f}  "
            f"majority={e05['majority']['overall']['f1']:.3f}  "
            f"per_sample={e05['per_sample']['overall']['f1']:.3f}"
        )

    save_axel_distribution_csv(per_world_results, world_names,
                                 out_dir / 'axel_distribution_tests_REV5.csv')

    s13 = make_fig13_truth_free(per_world_results,
                                  out_dir / 'Fig13_distribution_tests.png', K=K)
    s14 = make_fig14_novel_PRF1(per_world_results,
                                   out_dir / 'Fig14_trivial_baseline_lift.png', K=K)
    s15 = make_fig15_ensemble_sweep(per_world_results,
                                       out_dir / 'Fig15_ensemble_distribution.png', K=K)

    # ── REV6: pool ensemble novel-cell counts across all worlds ─────────
    ens_p05 = {mode: _pool_ensemble_across_worlds(per_world_results,
                                                  'ensemble_novel_p05', mode)
               for mode in ['union', 'majority', 'per_sample']}
    ens_topQ = {mode: _pool_ensemble_across_worlds(per_world_results,
                                                   'ensemble_novel_topQ', mode)
                for mode in ['union', 'majority', 'per_sample']}

    print("\n  POOLED RESULTS (REV5):")
    print(f"    Fig 13 MEAN p≥0.5 range KS: {s13['range_ks_p05']:.3f}  "
          f"(conn = {s13['conn_ks_p05']:.3f}, frag = {s13['fragmentation_p05']:.2f})")
    print(f"    Fig 14 OVERALL pooled F1 (MEAN prediction):")
    for label, vals in s14.items():
        print(f"        {label:<18s} P={vals['precision']:.3f}  "
              f"R={vals['recall']:.3f}  F1={vals['f1']:.3f}  "
              f"(TP={vals['TP']}, FP={vals['FP']}, FN={vals['FN']})")
    print(f"    Fig 15 ENSEMBLE sweep:")
    for label, vals in s15['all_pooled'].items():
        mark = '  ← BEST' if label == s15['best_threshold'] else ''
        print(f"        {label:<10s} range_KS = {vals['ks_r']:.3f}  "
              f"mean = {vals['pred_mean']:.1f}{mark}")

    # ── REV6 headline: ensemble novel-cell F1 — the correct object ──────
    print("\n  POOLED RESULTS (REV6 — ENSEMBLE novel-cell P/R/F1):")
    print(f"    {'thresholding':<14s} {'mode':<11s} {'band':<14s}"
          f" {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>8s} {'FP':>8s} {'FN':>8s}  n")
    for tag, ens in [('p>=0.5', ens_p05), ('topQ_K3', ens_topQ)]:
        for mode in ['union', 'majority', 'per_sample']:
            for band in ['band_easy', 'band_moderate', 'band_hard', 'overall']:
                v = ens[mode][band]
                print(f"    {tag:<14s} {mode:<11s} {band:<14s}"
                      f" {v['precision']:7.3f} {v['recall']:7.3f}"
                      f" {v['f1']:7.3f} {v['TP']:8d} {v['FP']:8d}"
                      f" {v['FN']:8d}  {v['n_species']}")

    # ── Direct comparison — the question Axel will ask ──────────────────
    mean_f1          = s14['EcoDiff p≥0.5']['f1']
    smooth_matched_f1 = s14['Smooth MATCHED']['f1']
    union_f1         = ens_p05['union']['overall']['f1']
    union_rec        = ens_p05['union']['overall']['recall']
    union_pre        = ens_p05['union']['overall']['precision']
    persample_f1     = ens_p05['per_sample']['overall']['f1']
    print("\n  ── HEADLINE COMPARISON  (overall, novel-cell F1) ──")
    print(f"    EcoDiff MEAN p≥0.5         F1 = {mean_f1:.3f}")
    print(f"    EcoDiff ENSEMBLE union     F1 = {union_f1:.3f}  "
          f"(R = {union_rec:.3f}, P = {union_pre:.3f})")
    print(f"    EcoDiff ENSEMBLE per-samp  F1 = {persample_f1:.3f}")
    print(f"    Smooth MATCHED (baseline)  F1 = {smooth_matched_f1:.3f}")
    if union_f1 > smooth_matched_f1:
        print(f"    → ENSEMBLE UNION BEATS the smoothing baseline "
              f"(+{union_f1 - smooth_matched_f1:.3f})")
    else:
        print(f"    → ensemble union still below smoothing "
              f"({union_f1 - smooth_matched_f1:+.3f})  "
              f"— union recall is high ({union_rec:.2f}); precision is the limiter")

    return per_world_results, s13, s14, s15


# =============================================================================
#  N. SELF-TESTS
# =============================================================================

if __name__ == "__main__":
    print("axel_distribution_tests_inpaint.py REV5 + REV6 — self-tests")
    rng = np.random.default_rng(42)

    # T1: identical → high p
    t = np.zeros((100, 20, 20), dtype=np.uint8)
    p = np.zeros((100, 20, 20), dtype=np.uint8)
    for s in range(100):
        n = int(rng.integers(6, 25))
        idx = rng.choice(400, size=n, replace=False)
        for c in idx: t[s, c // 20, c % 20] = 1
        idx2 = rng.choice(400, size=n, replace=False)
        for c in idx2: p[s, c // 20, c % 20] = 1
    rt = compute_range_size_distribution(t, p, 6)
    assert rt['ks_pvalue'] > 0.05, "T1 FAIL"
    print(f"T1 (identical range dist): p={rt['ks_pvalue']:.3f}  PASS")

    # T2: pooled P/R/F1 math.
    # 2 species, range 6 each, K=2 obs in each. POOLED novel: TP=4, FP=2, FN=4.
    # NOTE: this is checked on the `overall_*` keys, not `band_easy` — the band
    # path correctly returns nan for n_species < MIN_N_PER_BAND (=5), which is
    # intended behaviour (drives the "INSUFFICIENT POWER" panels in Fig 14).
    # The `overall_*` path has no minimum-n gate and verifies the same
    # pooled-count arithmetic.
    truth = np.zeros((2, 5, 5), dtype=np.uint8)
    obs   = np.zeros((2, 5, 5), dtype=np.uint8)
    pred  = np.zeros((2, 5, 5), dtype=np.uint8)
    for c in [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)]: truth[0,c[0],c[1]] = 1
    for c in [(0,0),(0,1)]: obs[0,c[0],c[1]] = 1
    for c in [(0,2),(0,3),(2,2)]: pred[0,c[0],c[1]] = 1
    for c in [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)]: truth[1,c[0],c[1]] = 1
    for c in [(0,0),(0,1)]: obs[1,c[0],c[1]] = 1
    for c in [(1,0),(1,1),(3,3)]: pred[1,c[0],c[1]] = 1
    # precision = 4/6, recall = 4/8 = 0.5, F1 = 4/7
    nv = compute_novel_cell_pooled_counts(truth, pred, obs, K=2, min_range=6)
    assert nv['overall_TP'] == 4 and nv['overall_FP'] == 2 and nv['overall_FN'] == 4, \
        f"T2 counts fail: TP={nv['overall_TP']} FP={nv['overall_FP']} FN={nv['overall_FN']}"
    assert abs(nv['overall_precision'] - 4/6) < 1e-6, f"P fail: {nv['overall_precision']}"
    assert abs(nv['overall_recall'] - 0.5) < 1e-6, f"R fail: {nv['overall_recall']}"
    assert abs(nv['overall_f1'] - 4/7) < 1e-6, f"F1 fail: {nv['overall_f1']}"
    # Also confirm the band path correctly returns nan for n=2 < MIN_N_PER_BAND
    assert np.isnan(nv['band_easy']['precision']), \
        "T2: band_easy should be nan for n<MIN_N_PER_BAND"
    assert nv['band_easy']['TP_sum'] == 4, \
        f"T2: band_easy TP_sum should still be 4, got {nv['band_easy']['TP_sum']}"
    print(f"T2 (pooled P/R/F1, overall): P=0.667, R=0.500, F1={4/7:.3f}  PASS")
    print(f"T2 (band path nan-gate for n<5, counts still pooled)  PASS")

    # T3: empty pred -> F1 = 0 not NaN
    truth_empty_pred = np.zeros((10, 10, 10), dtype=np.uint8)
    for s in range(10):
        truth_empty_pred[s, 0, :7] = 1
    obs_T3 = np.zeros((10, 10, 10), dtype=np.uint8)
    for s in range(10):
        obs_T3[s, 0, 0] = 1
        obs_T3[s, 0, 1] = 1
    pred_empty = np.zeros((10, 10, 10), dtype=np.uint8)
    nv2 = compute_novel_cell_pooled_counts(truth_empty_pred, pred_empty, obs_T3,
                                              K=2, min_range=6)
    assert nv2['overall_f1'] == 0.0, f"Empty pred should give F1=0, got {nv2['overall_f1']}"
    print(f"T3 (empty pred -> F1=0): F1={nv2['overall_f1']}  PASS")

    # T4: topQ multiplier
    prob = rng.uniform(0, 1, (1, 5, 5)).astype(np.float32)
    obs_one = np.zeros((1, 5, 5), dtype=np.uint8); obs_one[0, 0, 0] = 1
    bin_topQ = threshold_per_sample_topQ(prob, obs_one, multiplier=3)
    assert bin_topQ.sum() == 3, f"T4 FAIL: expected 3 cells, got {bin_topQ.sum()}"
    print(f"T4 (topQ density): expected 3 cells, got {bin_topQ.sum()}  PASS")

    # T5: REV6 ensemble union novel-cell P/R/F1
    # 1 species, range 6, K=2 obs. 2 samples.
    #   novel_truth = {(0,2),(0,3),(1,0),(1,1)}            -> 4
    #   sample 0 predicts (0,2),(0,3),(2,2)
    #   sample 1 predicts (1,0),(3,3)
    #   union novel = {(0,2),(0,3),(2,2),(1,0),(3,3)}
    #     TP = {(0,2),(0,3),(1,0)} = 3 ; FP = {(2,2),(3,3)} = 2 ; FN = {(1,1)} = 1
    truth5 = np.zeros((1, 5, 5), dtype=np.uint8)
    obs5   = np.zeros((1, 5, 5), dtype=np.uint8)
    for c in [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)]: truth5[0, c[0], c[1]] = 1
    for c in [(0,0),(0,1)]: obs5[0, c[0], c[1]] = 1
    samples5 = np.zeros((2, 1, 5, 5), dtype=np.uint8)
    for c in [(0,2),(0,3),(2,2)]: samples5[0, 0, c[0], c[1]] = 1
    for c in [(1,0),(3,3)]:       samples5[1, 0, c[0], c[1]] = 1
    ens5 = compute_novel_cell_ensemble_PRF1(truth5, samples5, obs5, K=2, min_range=3)
    uo = ens5['union']['overall']
    assert uo['TP'] == 3 and uo['FP'] == 2 and uo['FN'] == 1, \
        f"T5 union counts fail: {uo}"
    assert abs(uo['precision'] - 0.6) < 1e-6, f"T5 P fail: {uo['precision']}"
    assert abs(uo['recall'] - 0.75) < 1e-6, f"T5 R fail: {uo['recall']}"
    # per_sample pooled: s0 TP=2 FP=1 FN=2 ; s1 TP=1 FP=1 FN=3 -> TP=3 FP=2 FN=5
    po = ens5['per_sample']['overall']
    assert po['TP'] == 3 and po['FP'] == 2 and po['FN'] == 5, \
        f"T5 per_sample counts fail: {po}"
    print(f"T5 (ensemble union): TP=3 FP=2 FN=1 P=0.600 R=0.750  PASS")
    print(f"T5 (ensemble per_sample): TP=3 FP=2 FN=5  PASS")

    print("\nAll REV5 + REV6 self-tests PASSED.")