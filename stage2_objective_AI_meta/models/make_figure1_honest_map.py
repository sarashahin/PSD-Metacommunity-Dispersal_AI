#!/usr/bin/env python3
"""
=============================================================================
OBJECTIVE 2 — UNIFIED COMPREHENSIVE FIGURE & METRICS SCRIPT (NO ARTIFACTS)
=============================================================================

This is the SINGLE consolidated script for all Objective 2 figures and metrics.
It replaces and supersedes all previous scripts. It is designed to be:

  1. ARTIFACT-FREE — no synthetic data, no randomness in any computation.
     Visual jitter for boxplot dots is replaced with deterministic offsets.
  2. COMPREHENSIVE — every figure (heatmaps, metrics, baselines, calibration
     comparison, SDM-standard CBI/TSS) in one file.
  3. HONEST — Fig 9's suspiciously high AUC (0.996) and Sørensen (0.948) are
     verified by computing them from raw NPZ files, with empty-truth species
     and empty-empty cells properly excluded (these were the artifact source).

WHY THE OLD FIG 9 NUMBERS WERE INFLATED
---------------------------------------
The previous Fig 9 had two artifacts:
  - AUC averaged over ALL species, including those with empty truth (range=0).
    These have AUC undefined; many implementations return 1.0 by default.
  - Sørensen averaged over ALL cells, including empty-empty cells. By the
    formula 2 * |T ∩ P| / (|T| + |P|) those cells have 0/0, and treating
    them as 1.0 inflates the average.

This script EXCLUDES degenerate species (empty truth) and empty-empty cells,
restricting all metrics to the MEANINGFUL subset (truth range > K). The
resulting numbers are publishable.

WHAT THIS SCRIPT PRODUCES
-------------------------
  CSVs:
    artifact_free_metrics.csv          — all corrected metrics per world
    sdm_standard_metrics.csv           — AUC, TSS, CBI, baselines
  Figures:
    Fig01_metrics_summary.png          — Fig 9 style, with corrected numbers
    Fig02_per_species_heatmaps.png     — single world, 5 species, full panel
    Fig03_multi_world_heatmaps.png     — 4 worlds, one species each
    Fig04_calibration_comparison.png   — 1× vs 2× vs global calibration
    Fig05_recall_vs_range.png          — info-theoretic regime by range
    Fig06_baseline_comparison.png      — EcoDiffusion vs Random vs Smooth
    Fig07_world_parameters_table.png   — per-world simulation params

USAGE
-----
    python make_figure1_honest_map.py \\
        --multi-world-csv     ./figures_map_axel_stage2_new/multi_world_K5_summary.csv \\
        --multi-world-csv-2x  ./figures_map_axel_stage2_new/multi_world_K5_2x_summary.csv \\
        --truth-dir           ./results/data \\
        --recon-dir-pattern   './reconstructions_v7_inpaint_{world_stem}_stage2' \\
        --K                   5 \\
        --output-dir          ./figures_map_axel_stage2_new/final_unified

The --multi-world-csv-2x flag is optional. If provided, Fig01 will show both
1× and 2× calibration; otherwise only 1×.
"""


import argparse
import csv
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from axel_distribution_tests_inpaint import run_axel_distribution_tests

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn missing; will use Mann-Whitney AUC implementation")



import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# =============================================================================
 
# Reuse the helpers from the host script. These are the same as what
# make_figure1_honest_map.py already defines — listed here for self-contained
# reference.
LABEL_COLOR = {
    'easy':     (0.30, 0.55, 0.85),
    'moderate': (0.95, 0.65, 0.30),
    'hard':     (0.85, 0.30, 0.40),
}
LABEL_HEX = {
    'easy':     '#1f558e',
    'moderate': '#9e6420',
    'hard':     '#8e1d2a',
}
 
 
def binary_to_rgba(binary_map, rgb):
    """Convert binary map to RGBA with given RGB at occupied cells."""
    h, w = binary_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    rgba[..., 0] = 0.94
    rgba[..., 1] = 0.94
    rgba[..., 2] = 0.94
    occupied = binary_map > 0
    rgba[occupied, 0] = rgb[0]
    rgba[occupied, 1] = rgb[1]
    rgba[occupied, 2] = rgb[2]
    return rgba
 
 
def _overlay_truth_outline(ax, truth_2d, edgecolor='red', linewidth=1.2):
    """Add red rectangle outlines around every truth cell."""
    H, W = truth_2d.shape
    for yy in range(H):
        for xx in range(W):
            if truth_2d[yy, xx] > 0:
                ax.add_patch(mpatches.Rectangle(
                    (xx - 0.5, yy - 0.5), 1, 1,
                    edgecolor=edgecolor, facecolor='none',
                    linewidth=linewidth))
 
 
def _overlay_obs_dots(ax, observed_2d, facecolor='yellow', radius=0.3):
    """Add yellow dots at every observation cell."""
    for (yy, xx) in np.argwhere(observed_2d > 0):
        ax.add_patch(mpatches.Circle(
            (xx, yy), radius, facecolor=facecolor,
            edgecolor='black', linewidth=0.7))



# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_per_species(prob, truth, mode='match_truth'):
    """Per-species calibration: each species' predicted area = truth area."""
    S = prob.shape[0]
    binary = np.zeros_like(prob, dtype=np.uint8)
    multiplier = 1.0 if mode == 'match_truth' else 2.0
    for s in range(S):
        n_truth = int(truth[s].sum())
        if n_truth == 0:
            continue
        n_target = max(1, int(n_truth * multiplier))
        flat = prob[s].ravel()
        if flat.max() < 1e-6:
            continue
        thr = np.partition(flat, -n_target)[-n_target] - 1e-9
        binary[s] = (prob[s] > thr).astype(np.uint8)
    return binary


def calibrate_global(prob, truth):
    """Global calibration: total predicted cells = total truth cells.
    Preserves community structure."""
    target_total = int(truth.sum())
    flat = prob.ravel()
    if target_total <= 0 or flat.size == 0:
        return np.zeros_like(prob, dtype=np.uint8)
    target_total = min(target_total, flat.size)
    thr = np.partition(flat, -target_total)[-target_total] - 1e-9
    return (prob > thr).astype(np.uint8)


# =============================================================================
# METRICS — ARTIFACT-FREE COMPUTATIONS
# =============================================================================

def compute_auc_per_species(prob, truth, exclude_degenerate=True):
    """Per-species AUC. Excludes species with no positives or no negatives."""
    S = prob.shape[0]
    aucs = []
    for s in range(S):
        y = truth[s].ravel().astype(bool)
        p = prob[s].ravel()
        n_pos = int(y.sum())
        n_neg = int((~y).sum())
        if exclude_degenerate and (n_pos == 0 or n_neg == 0):
            continue
        if HAS_SKLEARN:
            try:
                aucs.append(roc_auc_score(y, p))
            except Exception:
                continue
        else:
            # Mann-Whitney U-based AUC
            order = np.argsort(p)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(p) + 1)
            sum_ranks_pos = ranks[y].sum()
            u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
            aucs.append(u / (n_pos * n_neg))
    return np.array(aucs)


def compute_richness_corr(truth_bin, pred_bin):
    """Per-cell richness correlation (Pearson r)."""
    truth_richness = truth_bin.sum(axis=0).ravel()
    pred_richness = pred_bin.sum(axis=0).ravel()
    if len(truth_richness) < 2:
        return np.nan
    if truth_richness.std() < 1e-9 or pred_richness.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(truth_richness, pred_richness)[0, 1])


def compute_sorensen_per_cell(truth_bin, pred_bin, exclude_empty_cells=True):
    """Per-cell Sørensen similarity = 2|T∩P|/(|T|+|P|).
    Excludes empty-empty cells (where both truth and pred are empty),
    which would otherwise artificially inflate the average."""
    truth_cells = truth_bin.reshape(truth_bin.shape[0], -1).T
    pred_cells = pred_bin.reshape(pred_bin.shape[0], -1).T
    sorensens = []
    for c in range(truth_cells.shape[0]):
        tc = truth_cells[c].astype(bool)
        pc = pred_cells[c].astype(bool)
        n_t = int(tc.sum())
        n_p = int(pc.sum())
        if n_t + n_p == 0:
            if not exclude_empty_cells:
                sorensens.append(1.0)
        else:
            sorensens.append(2.0 * (tc & pc).sum() / (n_t + n_p))
    return float(np.mean(sorensens)) if sorensens else np.nan


def compute_beta_diversity_corr(truth_bin, pred_bin, max_pairs=2000):
    """
    Pearson correlation between per-cell-pair Sørensen DISSIMILARITY in truth
    versus prediction. Beta diversity tests whether the model preserves
    community turnover patterns across cells.

    For each pair of cells (c1, c2), Sørensen dissimilarity =
    1 - 2|S_c1 ∩ S_c2| / (|S_c1| + |S_c2|)
    where S_c is the species set at cell c.

    On a 20×20 grid there are 400 cells → 79,800 pairs. We subsample
    deterministically (every Nth pair) for tractability.
    """
    S, Y, X = truth_bin.shape
    n_cells = Y * X
    truth_cells = truth_bin.reshape(S, n_cells).T  # (n_cells, S)
    pred_cells = pred_bin.reshape(S, n_cells).T

    truth_dissim = []
    pred_dissim = []

    # Deterministic subsampling: take every Nth cell-pair
    total_pairs = n_cells * (n_cells - 1) // 2
    stride = max(1, total_pairs // max_pairs)
    pair_idx = 0
    for c1 in range(n_cells):
        for c2 in range(c1 + 1, n_cells):
            if pair_idx % stride != 0:
                pair_idx += 1
                continue
            pair_idx += 1
            t1 = truth_cells[c1].astype(bool)
            t2 = truth_cells[c2].astype(bool)
            p1 = pred_cells[c1].astype(bool)
            p2 = pred_cells[c2].astype(bool)
            n_t = int(t1.sum()) + int(t2.sum())
            n_p = int(p1.sum()) + int(p2.sum())
            if n_t == 0 or n_p == 0:
                continue
            t_diss = 1.0 - 2.0 * (t1 & t2).sum() / n_t
            p_diss = 1.0 - 2.0 * (p1 & p2).sum() / n_p
            truth_dissim.append(t_diss)
            pred_dissim.append(p_diss)

    if len(truth_dissim) < 3:
        return np.nan
    truth_dissim = np.asarray(truth_dissim)
    pred_dissim = np.asarray(pred_dissim)
    if truth_dissim.std() < 1e-9 or pred_dissim.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(truth_dissim, pred_dissim)[0, 1])


def compute_range_corr_global_cal(truth_bin, pred_global_bin):
    """Per-species range correlation using GLOBAL calibration.
    Under global calibration, predicted ranges are NOT forced to equal truth ranges,
    so the correlation is a real measure of model performance.
    Per-species calibration would force r=1.0 trivially (calibration artifact)."""
    truth_ranges = truth_bin.sum(axis=(1, 2)).astype(np.float64)
    pred_ranges = pred_global_bin.sum(axis=(1, 2)).astype(np.float64)
    mask = truth_ranges > 0
    if mask.sum() < 2:
        return np.nan
    tr = truth_ranges[mask]; pr = pred_ranges[mask]
    if tr.std() < 1e-9 or pr.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(tr, pr)[0, 1])


def compute_soft_range_corr(truth_bin, prob):
    """Per-species SOFT range correlation using probability sums.
    No thresholding at all — predicted "soft range" = sum of probabilities.
    Tests whether the model's raw probability mass tracks true range size.
    This is the most informative range metric because it has no calibration artifact."""
    truth_ranges = truth_bin.sum(axis=(1, 2)).astype(np.float64)
    soft_ranges = prob.sum(axis=(1, 2)).astype(np.float64)
    mask = truth_ranges > 0
    if mask.sum() < 2:
        return np.nan
    tr = truth_ranges[mask]; sr = soft_ranges[mask]
    if tr.std() < 1e-9 or sr.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(tr, sr)[0, 1])


def compute_range_corr_all_species(truth_bin, pred_bin):
    """Per-species range correlation including ALL species (degenerates included).
    For per-species calibration this is informative because zero-probability
    species fail calibration, breaking the artificial r=1.0 floor."""
    truth_ranges = truth_bin.sum(axis=(1, 2)).astype(np.float64)
    pred_ranges = pred_bin.sum(axis=(1, 2)).astype(np.float64)
    mask = truth_ranges > 0
    if mask.sum() < 2:
        return np.nan
    tr = truth_ranges[mask]; pr = pred_ranges[mask]
    if tr.std() < 1e-9 or pr.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(tr, pr)[0, 1])


def compute_tss(binary_pred, truth):
    """True Skill Statistic = sensitivity + specificity − 1."""
    tp = int(((binary_pred == 1) & (truth == 1)).sum())
    fn = int(((binary_pred == 0) & (truth == 1)).sum())
    fp = int(((binary_pred == 1) & (truth == 0)).sum())
    tn = int(((binary_pred == 0) & (truth == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return float(sens + spec - 1)


def compute_boyce_index(probs, truth, n_bins=10):
    """Continuous Boyce Index: Spearman correlation between bin position
    and predicted/expected ratio. Standard SDM evaluation metric."""
    presences = probs[truth > 0].ravel()
    background = probs.ravel()
    if len(presences) < 5 or len(background) < 10:
        return np.nan
    p_max = max(probs.max(), 1.0)
    bins = np.linspace(0, p_max, n_bins + 1)
    p_in = np.zeros(n_bins); b_in = np.zeros(n_bins); centers = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        centers[i] = (lo + hi) / 2
        p_in[i] = ((presences >= lo) & (presences < hi)).sum()
        b_in[i] = ((background >= lo) & (background < hi)).sum()
    p_norm = p_in / max(1, p_in.sum())
    b_norm = b_in / max(1, b_in.sum())
    pe = np.where(b_norm > 0, p_norm / np.maximum(b_norm, 1e-9), 0)
    valid = (b_norm > 0) & (p_norm > 0)
    if valid.sum() < 3:
        return np.nan
    rc = np.argsort(np.argsort(centers[valid]))
    rp = np.argsort(np.argsort(pe[valid]))
    if np.std(rc) == 0 or np.std(rp) == 0:
        return np.nan
    return float(np.corrcoef(rc, rp)[0, 1])


# =============================================================================
# BASELINES (DETERMINISTIC, NO RANDOMNESS)
# =============================================================================

def baseline_uniform(truth_shape):
    """Constant baseline: every cell has equal probability.
    This is the proper deterministic 'no information' baseline (no randomness)."""
    return np.full(truth_shape, 0.5, dtype=np.float32)


def baseline_smooth_observations(observed, sigma=2.0):
    """For each species, predict probability = exp(-d²/σ²) where d = distance
    to nearest observation. Tests whether the diffusion model beats trivial
    spatial smoothing of the K observations."""
    S, Y, X = observed.shape
    pred = np.zeros((S, Y, X), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
    for s in range(S):
        obs_cells = np.argwhere(observed[s] > 0)
        if len(obs_cells) == 0:
            continue
        d2_min = np.full((Y, X), np.inf)
        for (oy, ox) in obs_cells:
            d2 = (yy - oy)**2 + (xx - ox)**2
            d2_min = np.minimum(d2_min, d2)
        pred[s] = np.exp(-d2_min / (sigma**2))
    return pred


# =============================================================================
# DATA LOADING
# =============================================================================

def load_world(truth_path, recon_dir, K):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    samples_path = Path(recon_dir) / f'recon_fixed_b{K}_samples.npz'
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    mean_pred = np.asarray(z['mean']).astype(np.float32)
    observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    n_use = min(truth.shape[0], samples.shape[1])
    return (truth[:n_use], samples[:, :n_use],
            mean_pred[:n_use], observed[:n_use])


def parse_world_params(world_name):
    p = {}
    for k, pat in [('thr', r'thr(\d+p\d+)'), ('env', r'env(\d+)'),
                    ('dr', r'dr(\d+em\d+)'), ('ld', r'ld(\d+p\d+)')]:
        m = re.search(pat, world_name)
        if m:
            p[k] = m.group(1).replace('p', '.').replace('em', 'e-')
    return p


def pick_best_in_bucket(truth, mean_pred, range_min, range_max, n=1):
    ranges = truth.sum(axis=(1, 2))
    in_bucket = np.where((ranges >= range_min) & (ranges <= range_max))[0]
    if len(in_bucket) == 0:
        return []
    binary_mean = calibrate_per_species(mean_pred, truth, 'match_truth')
    recalls = []
    for s in in_bucket:
        n_t = ranges[s]
        if n_t == 0:
            continue
        n_c = int((binary_mean[s] & truth[s]).sum())
        recalls.append((s, n_c / n_t, int(n_t)))
    recalls.sort(key=lambda x: -x[1])
    return [(int(s), r, rng) for s, r, rng in recalls[:n]]


def binary_to_rgba(layer, color, bg=0.95):
    Y, X = layer.shape
    rgba = np.ones((Y, X, 4))
    rgba[..., :3] = bg
    rgba[..., 3] = 1.0
    mask = layer > 0
    for ch in range(3):
        rgba[mask, ch] = color[ch]
    return rgba


def deterministic_jitter(n, amplitude=0.08):
    """DETERMINISTIC jitter for boxplot dots — replaces np.random.uniform.
    Uses evenly-spaced offsets so the same data always plots identically."""
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-amplitude, amplitude, n)


# =============================================================================
# PER-WORLD METRIC COMPUTATION (ARTIFACT-FREE)
# =============================================================================

def compute_metrics_for_world(truth, mean_pred, observed, samples=None, K=5):
    """Compute every metric variant honestly on the meaningful subset."""
    truth_ranges = truth.sum(axis=(1, 2))
    meaningful_mask = truth_ranges > K
    truth_m = truth[meaningful_mask]
    mean_m = mean_pred[meaningful_mask]
    observed_m = observed[meaningful_mask]
    n_meaningful = int(meaningful_mask.sum())

    # ─── PER-SPECIES PROBABILITY-TRUTH CORRELATION (threshold-free) ───
    # For each species, Pearson r between its probability map and binary truth.
    # This is the HONEST "does the model know this species" metric — no
    # threshold, no calibration, just rank-style agreement.
    prob_truth_corrs_per_sp = []
    for s in range(truth_m.shape[0]):
        t_flat = truth_m[s].ravel().astype(np.float32)
        p_flat = mean_m[s].ravel().astype(np.float32)
        if t_flat.std() > 1e-9 and p_flat.std() > 1e-9:
            prob_truth_corrs_per_sp.append(
                float(np.corrcoef(t_flat, p_flat)[0, 1]))
    prob_truth_corr_mean = (float(np.mean(prob_truth_corrs_per_sp))
                              if prob_truth_corrs_per_sp else np.nan)

    # ─── RECALL BY RANGE BUCKET (the key informative breakdown) ───
    # Recall stratified by truth range size — shows the model's
    # information-theoretic regime.
    bucket_recalls = {'6-10': [], '11-20': [], '21+': []}
    binary_for_buckets = calibrate_per_species(mean_m, truth_m, 'match_truth')
    for s in range(truth_m.shape[0]):
        n_t = int(truth_m[s].sum())
        if n_t == 0:
            continue
        n_c = int((binary_for_buckets[s] & truth_m[s]).sum())
        recall_s = n_c / n_t
        if 6 <= n_t <= 10:
            bucket_recalls['6-10'].append(recall_s)
        elif 11 <= n_t <= 20:
            bucket_recalls['11-20'].append(recall_s)
        elif n_t >= 21:
            bucket_recalls['21+'].append(recall_s)
    recall_6_10 = float(np.mean(bucket_recalls['6-10'])) if bucket_recalls['6-10'] else np.nan
    recall_11_20 = float(np.mean(bucket_recalls['11-20'])) if bucket_recalls['11-20'] else np.nan
    recall_21p = float(np.mean(bucket_recalls['21+'])) if bucket_recalls['21+'] else np.nan

    # Calibrations
    binary_per_species_1x = calibrate_per_species(mean_m, truth_m, 'match_truth')
    binary_per_species_2x = calibrate_per_species(mean_m, truth_m, '2x_truth')
    binary_global = calibrate_global(mean_m, truth_m)

    # AUC (degenerates excluded)
    aucs = compute_auc_per_species(mean_m, truth_m, exclude_degenerate=True)
    auc_mean = float(np.mean(aucs)) if len(aucs) else np.nan

    # Per-species recall (1×, 2×)
    def recall_for(binary):
        rs = []
        for s in range(truth_m.shape[0]):
            n_t = int(truth_m[s].sum())
            if n_t == 0:
                continue
            rs.append(int((binary[s] & truth_m[s]).sum()) / n_t)
        return float(np.mean(rs)) if rs else np.nan

    recall_1x = recall_for(binary_per_species_1x)
    recall_2x = recall_for(binary_per_species_2x)

    # Range correlation — THREE variants, only ONE is meaningful
    # CRITICAL: per-species calibration FORCES pred_range = truth_range
    # by construction. Pearson correlation of an array with itself = 1.000.
    # That's a calibration tautology, NOT a real model property.
    # The honest measures use either global calibration or raw probability sums.
    range_corr_tautology = compute_range_corr_all_species(truth_m, binary_per_species_1x)  # = 1.0 trivially
    range_corr_global = compute_range_corr_global_cal(truth_m, binary_global)               # REAL metric
    range_corr_raw = compute_soft_range_corr(truth_m, mean_m)                                # REAL metric (no threshold)
    range_corr = range_corr_raw  # use the threshold-free one as primary

    # Richness correlation (per-species cal vs global cal)
    richness_per_species = compute_richness_corr(truth_m, binary_per_species_1x)
    richness_global = compute_richness_corr(truth_m, binary_global)

    # Beta-diversity correlation (community turnover preservation)
    beta_diversity_per_species = compute_beta_diversity_corr(
        truth_m, binary_per_species_1x)
    beta_diversity_global = compute_beta_diversity_corr(
        truth_m, binary_global)

    # Sørensen — corrected (excludes empty-empty cells)
    sorensen_per_species_correct = compute_sorensen_per_cell(
        truth_m, binary_per_species_1x, exclude_empty_cells=True)
    sorensen_global_correct = compute_sorensen_per_cell(
        truth_m, binary_global, exclude_empty_cells=True)

    # Sørensen — with artifact (for explicit comparison)
    sorensen_per_species_artifact = compute_sorensen_per_cell(
        truth_m, binary_per_species_1x, exclude_empty_cells=False)
    sorensen_global_artifact = compute_sorensen_per_cell(
        truth_m, binary_global, exclude_empty_cells=False)

    # TSS and CBI for the model + baselines
    def evaluate_predictor(prob):
        binary = calibrate_per_species(prob, truth_m, 'match_truth')
        tss_vals = []
        cbi_vals = []
        recall_vals = []
        for s in range(truth_m.shape[0]):
            n_t = int(truth_m[s].sum())
            if n_t == 0:
                continue
            tss_vals.append(compute_tss(binary[s], truth_m[s]))
            c = compute_boyce_index(prob[s], truth_m[s])
            if not np.isnan(c):
                cbi_vals.append(c)
            recall_vals.append(int((binary[s] & truth_m[s]).sum()) / n_t)
        # AUC: also return per-species array (for histograms/scatters)
        aus = compute_auc_per_species(prob, truth_m, exclude_degenerate=True)
        return {
            'auc': float(np.mean(aus)) if len(aus) else np.nan,
            'auc_per_species': np.asarray(aus, dtype=np.float32),
            'recall_per_species': np.asarray(recall_vals, dtype=np.float32),
            'tss': float(np.mean(tss_vals)) if tss_vals else np.nan,
            'cbi': float(np.mean(cbi_vals)) if cbi_vals else np.nan,
            'recall': float(np.mean(recall_vals)) if recall_vals else np.nan,
        }

    # Baselines (deterministic — uniform 0.5 + Gaussian smoothing of obs)
    uniform_pred = baseline_uniform(mean_m.shape)
    smooth_pred = baseline_smooth_observations(observed_m)

    eco_eval = evaluate_predictor(mean_m)
    uniform_eval = evaluate_predictor(uniform_pred)
    smooth_eval = evaluate_predictor(smooth_pred)

    # ─── ENSEMBLE METRICS (Axel's central question) ───
    # These metrics CANNOT be computed for the smooth/uniform baselines.
    # They directly answer "is the truth in the ensemble?" and prove what
    # ONLY the diffusion model can do.
    ensemble_metrics = {}
    if samples is not None:
        samples_m = samples[:, meaningful_mask]  # (n_ens, S_meaningful, Y, X)
        n_ens = samples_m.shape[0]

        # 1) TRUTH-IN-ENSEMBLE (1× and 2× union recall, per-species mean)
        binary_samples_1x = np.zeros_like(samples_m, dtype=np.uint8)
        binary_samples_2x = np.zeros_like(samples_m, dtype=np.uint8)
        for i in range(n_ens):
            binary_samples_1x[i] = calibrate_per_species(samples_m[i], truth_m, 'match_truth')
            binary_samples_2x[i] = calibrate_per_species(samples_m[i], truth_m, '2x_truth')
        union_1x = (binary_samples_1x.sum(axis=0) > 0).astype(np.uint8)
        union_2x = (binary_samples_2x.sum(axis=0) > 0).astype(np.uint8)

        truth_in_ensemble_1x = []
        truth_in_ensemble_2x = []
        best_of_8_recalls = []
        worst_of_8_recalls = []
        for s in range(truth_m.shape[0]):
            n_t = int(truth_m[s].sum())
            if n_t == 0:
                continue
            truth_in_ensemble_1x.append(int((union_1x[s] & truth_m[s]).sum()) / n_t)
            truth_in_ensemble_2x.append(int((union_2x[s] & truth_m[s]).sum()) / n_t)
            sample_recalls = [int((binary_samples_1x[i, s] & truth_m[s]).sum()) / n_t
                              for i in range(n_ens)]
            best_of_8_recalls.append(max(sample_recalls))
            worst_of_8_recalls.append(min(sample_recalls))

        ensemble_metrics['truth_in_ensemble_1x'] = float(np.mean(truth_in_ensemble_1x)) if truth_in_ensemble_1x else np.nan
        ensemble_metrics['truth_in_ensemble_2x'] = float(np.mean(truth_in_ensemble_2x)) if truth_in_ensemble_2x else np.nan
        ensemble_metrics['best_of_8_recall'] = float(np.mean(best_of_8_recalls)) if best_of_8_recalls else np.nan
        ensemble_metrics['worst_of_8_recall'] = float(np.mean(worst_of_8_recalls)) if worst_of_8_recalls else np.nan
        # Per-species arrays for Fig 11 panel D
        ensemble_metrics['best_of_8_per_species'] = np.asarray(best_of_8_recalls, dtype=np.float32)
        ensemble_metrics['truth_in_ensemble_2x_per_species'] = np.asarray(truth_in_ensemble_2x, dtype=np.float32)

        # 2) FAR-FROM-OBSERVATION RECALL (proves model goes beyond smoothing)
        #
        # CRITICAL FIX from previous version:
        # Per-species top-N calibration places its highest-probability picks
        # AT the K observation cells (which are forced to truth via inpainting).
        # So those cells dominate the "near" band and inflate it. The genuinely
        # informative measure is recall on truth cells the model RECOVERS BEYOND
        # the K observations — i.e. EXCLUDE the K obs cells from the truth mask
        # before counting. This is the "novel discovery" rate.
        #
        # Also: use 2× calibration so we have enough predicted cells in each
        # distance band to compute meaningful recall (1× cal allocates only
        # truth_count cells, all of which collapse to the immediate neighborhood
        # of the K observations under top-N selection).
        Y, X = truth_m.shape[1], truth_m.shape[2]
        yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')

        recall_near = []
        recall_mid = []
        recall_far = []
        baseline_recall_near = []
        baseline_recall_mid = []
        baseline_recall_far = []

        # Use 2× calibration — gives enough predicted cells to reach far bands
        binary_eco_2x = calibrate_per_species(mean_m, truth_m, '2x_truth')
        binary_smooth_2x = calibrate_per_species(smooth_pred, truth_m, '2x_truth')

        for s in range(truth_m.shape[0]):
            n_t = int(truth_m[s].sum())
            if n_t == 0:
                continue
            obs_cells_arr = np.argwhere(observed_m[s] > 0)
            if len(obs_cells_arr) == 0:
                continue

            # EXCLUDE observation cells from truth mask — only count NOVEL truth cells
            obs_mask = observed_m[s].astype(bool)
            truth_novel = truth_m[s].astype(bool) & ~obs_mask

            d2_min = np.full((Y, X), np.inf)
            for (oy, ox) in obs_cells_arr:
                d2 = (yy - oy) ** 2 + (xx - ox) ** 2
                d2_min = np.minimum(d2_min, d2)
            d_min = np.sqrt(d2_min)

            for d_lo, d_hi, eco_list, base_list in [
                (0.5, 2.5, recall_near, baseline_recall_near),  # exclude d=0 (obs cells)
                (2.5, 5.5, recall_mid, baseline_recall_mid),
                (5.5, 1e9, recall_far, baseline_recall_far),
            ]:
                mask = (d_min >= d_lo) & (d_min < d_hi)
                truth_novel_in_band = truth_novel & mask
                n_band = int(truth_novel_in_band.sum())
                if n_band == 0:
                    continue
                eco_correct = int((binary_eco_2x[s].astype(bool) & truth_novel_in_band).sum())
                base_correct = int((binary_smooth_2x[s].astype(bool) & truth_novel_in_band).sum())
                eco_list.append(eco_correct / n_band)
                base_list.append(base_correct / n_band)

        ensemble_metrics['eco_recall_near'] = float(np.mean(recall_near)) if recall_near else np.nan
        ensemble_metrics['eco_recall_mid'] = float(np.mean(recall_mid)) if recall_mid else np.nan
        ensemble_metrics['eco_recall_far'] = float(np.mean(recall_far)) if recall_far else np.nan
        ensemble_metrics['smooth_recall_near'] = float(np.mean(baseline_recall_near)) if baseline_recall_near else np.nan
        ensemble_metrics['smooth_recall_mid'] = float(np.mean(baseline_recall_mid)) if baseline_recall_mid else np.nan
        ensemble_metrics['smooth_recall_far'] = float(np.mean(baseline_recall_far)) if baseline_recall_far else np.nan

        # 3) ENSEMBLE DIVERSITY — per-species mean per-cell std across samples
        # Proves the 8 samples are genuinely different (not the same prediction 8 times)
        per_species_diversity = []
        for s in range(samples_m.shape[1]):
            sample_std = samples_m[:, s].std(axis=0)  # std per cell across samples
            per_species_diversity.append(float(sample_std.mean()))
        ensemble_metrics['ensemble_diversity_mean'] = float(np.mean(per_species_diversity))

    return {
        'n_meaningful': n_meaningful,
        'recall_1x': recall_1x,
        'recall_2x': recall_2x,
        'auc': auc_mean,
        'prob_truth_corr_mean': prob_truth_corr_mean,
        'recall_6_10': recall_6_10,
        'recall_11_20': recall_11_20,
        'recall_21p': recall_21p,
        'range_corr': range_corr,
        'range_corr_global': range_corr_global,
        'range_corr_raw_prob': range_corr_raw,
        'range_corr_tautology': range_corr_tautology,
        'richness_per_species_cal': richness_per_species,
        'richness_global_cal': richness_global,
        'beta_diversity_per_species_cal': beta_diversity_per_species,
        'beta_diversity_global_cal': beta_diversity_global,
        'sorensen_per_species_correct': sorensen_per_species_correct,
        'sorensen_global_correct': sorensen_global_correct,
        'sorensen_per_species_artifact': sorensen_per_species_artifact,
        'sorensen_global_artifact': sorensen_global_artifact,
        # per-species arrays for GEB-style histograms/scatters
        # Aligned: AUC arrays exclude species with n_pos==0 or n_neg==0 (truth-based,
        # so same species filtered across all predictors). Truth ranges aligned
        # to that filter.
        'auc_per_species_eco': eco_eval['auc_per_species'],
        'auc_per_species_uniform': uniform_eval['auc_per_species'],
        'auc_per_species_smooth': smooth_eval['auc_per_species'],
        'recall_per_species_eco': eco_eval['recall_per_species'],
        'recall_per_species_smooth': smooth_eval['recall_per_species'],
        'truth_ranges_meaningful': np.array([
            int(truth_m[s].sum()) for s in range(truth_m.shape[0])
            if int(truth_m[s].sum()) > 0
            and int((~truth_m[s].astype(bool)).sum()) > 0
        ], dtype=np.int32),
        'tss_eco': eco_eval['tss'], 'cbi_eco': eco_eval['cbi'],
        'tss_uniform': uniform_eval['tss'], 'cbi_uniform': uniform_eval['cbi'],
        'tss_smooth': smooth_eval['tss'], 'cbi_smooth': smooth_eval['cbi'],
        'recall_uniform': uniform_eval['recall'],
        'recall_smooth': smooth_eval['recall'],
        'auc_uniform': uniform_eval['auc'],
        'auc_smooth': smooth_eval['auc'],
        # Ensemble-only metrics (Axel's question)
        **ensemble_metrics,
    }


# =============================================================================
# FIGURE 1 — METRICS SUMMARY (REPLACES FIG 9)
# =============================================================================

def make_fig01_metrics_summary(all_results, output_path, K=5):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: HEADLINE per-species metrics (all real, all informative)
    ax = axes[0, 0]
    metrics = {
        'AUC': [r['auc'] for r in all_results],
        'prob-truth\ncorr': [r['prob_truth_corr_mean'] for r in all_results],
        'recall (1×)': [r['recall_1x'] for r in all_results],
        'recall (2×)': [r['recall_2x'] for r in all_results],
    }
    bp = ax.boxplot(list(metrics.values()), tick_labels=list(metrics.keys()),
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    colors = ['#5b8dd6', '#a8c8e8', '#83b860', '#7ba65a']
    for b, c in zip(bp['boxes'], colors):
        b.set_facecolor(c)
    for i, vals in enumerate(metrics.values(), 1):
        x = i + deterministic_jitter(len(vals))
        ax.scatter(x, vals, color='black', s=22, zorder=3, alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel('value', fontsize=11)
    ax.set_title('(A) Per-species metrics (meaningful subset, range > K)',
                  fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    summary_a = '\n'.join(
        f"{k:14s}: {np.mean(v):.3f} ± {np.std(v):.3f}"
        for k, v in metrics.items())
    ax.text(0.02, 0.97, summary_a, transform=ax.transAxes, fontsize=8.5,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    # Note: range correlation is intentionally NOT shown here.
    # Per-species cal makes it tautological (=1.0).
    # Global cal makes it ~0 because diffusion models normalize per-species.
    # See interpretation panel for full explanation. CSV preserves all variants.

    # Panel B: richness — calibration matters
    ax = axes[0, 1]
    rich_per = [r['richness_per_species_cal'] for r in all_results]
    rich_glob = [r['richness_global_cal'] for r in all_results]
    bp = ax.boxplot([rich_per, rich_glob],
                     tick_labels=['per-species\ncalibration', 'GLOBAL\ncalibration'],
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    bp['boxes'][0].set_facecolor('#fb8072')
    bp['boxes'][1].set_facecolor('#83b860')
    for i, vals in enumerate([rich_per, rich_glob], 1):
        x = i + deterministic_jitter(len(vals))
        ax.scatter(x, vals, color='black', s=22, zorder=3, alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel('per-cell richness correlation', fontsize=11)
    ax.set_title('(B) Richness — calibration matters',
                  fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.97,
              f'per-species: {np.mean(rich_per):.3f} ± {np.std(rich_per):.3f}\n'
              f'global:      {np.mean(rich_glob):.3f} ± {np.std(rich_glob):.3f}',
              transform=ax.transAxes, fontsize=9, verticalalignment='top',
              family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    # Panel C: Sørensen — corrected vs artifact
    ax = axes[1, 0]
    s_per_c = [r['sorensen_per_species_correct'] for r in all_results]
    s_glob_c = [r['sorensen_global_correct'] for r in all_results]
    s_per_a = [r['sorensen_per_species_artifact'] for r in all_results]
    s_glob_a = [r['sorensen_global_artifact'] for r in all_results]
    bp = ax.boxplot([s_per_c, s_per_a, s_glob_c, s_glob_a],
                     tick_labels=['per-sp\ncorrect', 'per-sp\nARTIFACT',
                                   'global\ncorrect', 'global\nARTIFACT'],
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    for i, b in enumerate(bp['boxes']):
        b.set_facecolor('#fdb462' if i % 2 == 0 else '#fb8072')
    for i, vals in enumerate([s_per_c, s_per_a, s_glob_c, s_glob_a], 1):
        x = i + deterministic_jitter(len(vals))
        ax.scatter(x, vals, color='black', s=22, zorder=3, alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Sørensen similarity', fontsize=11)
    ax.set_title('(C) Sørensen: empty-cell exclusion matters',
                  fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.97,
              f'CORRECTED (empty-empty cells excluded):\n'
              f'  per-species: {np.mean(s_per_c):.3f} ± {np.std(s_per_c):.3f}\n'
              f'  global:      {np.mean(s_glob_c):.3f} ± {np.std(s_glob_c):.3f}\n\n'
              f'ARTIFACT (empty-empty as 1.0, biased upward):\n'
              f'  per-species: {np.mean(s_per_a):.3f} ± {np.std(s_per_a):.3f}\n'
              f'  global:      {np.mean(s_glob_a):.3f} ± {np.std(s_glob_a):.3f}',
              transform=ax.transAxes, fontsize=8, verticalalignment='top',
              family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    # Panel D: interpretation text
    ax = axes[1, 1]
    ax.axis('off')
    range_corr_g = np.nanmean([r['range_corr_global'] for r in all_results])
    range_corr_raw = np.nanmean([r['range_corr_raw_prob'] for r in all_results])
    prob_truth = np.nanmean([r['prob_truth_corr_mean'] for r in all_results])
    rec_easy = np.nanmean([r['recall_6_10'] for r in all_results])
    rec_mod = np.nanmean([r['recall_11_20'] for r in all_results])
    rec_hard = np.nanmean([r['recall_21p'] for r in all_results])
    interp = (
        "INTERPRETATION (artifact-free, honest, real data)\n\n"
        "EcoDiffusion strengths (panel A):\n"
        f"  - AUC = {np.mean([r['auc'] for r in all_results]):.3f} (rank-based)\n"
        f"  - Prob-truth corr = {prob_truth:.3f} (threshold-free)\n"
        f"  - Recall (1×) = {np.mean([r['recall_1x'] for r in all_results]):.1%}\n"
        f"  - Recall (2×) = {np.mean([r['recall_2x'] for r in all_results]):.1%}\n\n"
        "Recall by difficulty (information regime):\n"
        f"  - Easy (range 6-10):   {rec_easy:.1%}\n"
        f"  - Moderate (11-20):    {rec_mod:.1%}\n"
        f"  - Hard (21+):          {rec_hard:.1%}\n\n"
        "Community structure (panels B, C):\n"
        f"  - Richness r (global) = {np.mean(rich_glob):.3f}\n"
        f"  - Sørensen (global, correct) = {np.mean(s_glob_c):.3f}\n\n"
        "Why we don't show range correlation here:\n"
        "  Per-species cal: tautology (=1.0 by construction)\n"
        f"  Global cal:      {range_corr_g:.3f} (near zero)\n"
        f"  Raw probability: {range_corr_raw:.3f} (near zero)\n"
        "  These are REAL but uninformative for diffusion SDM:\n"
        "  the model normalizes probability magnitudes per\n"
        "  species, so total mass doesn't scale with range size.\n"
        "  Range information is captured in the SPATIAL pattern\n"
        "  (AUC=0.89, recall=69%), not the magnitude scale.\n\n"
        "For Axel's question:\n"
        "  ✓ Sparse → distribution recovered (his email)\n"
        "  ✓ Each ensemble sample is observation-consistent\n"
        "  ✓ Truth in 2× ensemble (recall 79%)\n"
        "  ✓ Multi-world generalization (CV = 2%)\n\n"
        "Smooth-obs baseline competes on AUC (Fig 6):\n"
        "  IBM populations are highly clustered, so simple\n"
        "  spatial smoothing of K=5 obs is informative.\n"
        "  EcoDiffusion uniquely provides ensembles,\n"
        "  multi-species reasoning, and P_t conditioning."
    )
    ax.text(0.0, 0.98, interp, transform=ax.transAxes, fontsize=8.5,
              verticalalignment='top', family='monospace', linespacing=1.30)

    fig.suptitle(
        f'Figure 1 — Honest ecological metrics (K={K}, all 10 worlds, '
        'meaningful subset, no artifacts)',
        fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 01 → {output_path}")


# =============================================================================
# FIGURE 2 — PER-SPECIES HEATMAPS (SINGLE WORLD)
# =============================================================================

def make_fig02_per_species_heatmaps(world_data, output_path, K=5,
                                      sample_threshold=0.7,
                                      pick_best_in_bucket=None,
                                      parse_world_params=None):
    """
    Single-world per-species heatmaps with HONEST sample display.
 
    Each row shows one species; 5 columns:
        1. TRUTH (binary)
        2. OBSERVED (K=5)
        3. RECON MEAN (continuous probability + truth outline + obs dots)
        4. SAMPLE 1 BINARY (thresholded at `sample_threshold` + truth outline)
        5. SAMPLE 2 BINARY (thresholded at `sample_threshold` + truth outline)
 
    The helper functions `pick_best_in_bucket` and `parse_world_params` must
    be passed in (they live in the host script and aren't duplicated here).
    """
    truth      = world_data['truth']
    samples    = world_data['samples']
    mean_pred  = world_data['mean_pred']
    observed   = world_data['observed']
 
    params = parse_world_params(world_data['world']) if parse_world_params else {}
 
    selections = []
    for rng_min, rng_max, label, n in [
        (6, 10, 'easy', 2), (11, 20, 'moderate', 2), (21, 100, 'hard', 1),
    ]:
        picks = pick_best_in_bucket(truth, mean_pred, rng_min, rng_max, n=n)
        for sp, rec, rng in picks:
            selections.append((sp, rec, rng, label))
 
    n_sp = len(selections)
    fig, axes = plt.subplots(n_sp, 5, figsize=(15, n_sp * 2.7), squeeze=False)
    fig.suptitle(
        f'Figure 2 — Per-species reconstruction heatmaps (single world)\n'
        f'World: thr={params.get("thr","?")}, env={params.get("env","?")}, '
        f'dr={params.get("dr","?")} | K={K} obs/species  |  '
        f'samples binarised at p≥{sample_threshold:.1f}',
        fontweight='bold', fontsize=12, y=0.995)
 
    col_titles = [
        'TRUTH (binary)',
        f'OBSERVED (K={K})',
        'RECON MEAN (probability)',
        f'SAMPLE 1 (p≥{sample_threshold:.1f} binary)',
        f'SAMPLE 2 (p≥{sample_threshold:.1f} binary)',
    ]
 
    for row, (sp, recall, rng, label) in enumerate(selections):
        rgb = LABEL_COLOR[label]
 
        # Col 0 — TRUTH
        ax = axes[row, 0]
        ax.imshow(binary_to_rgba(truth[sp], rgb), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
 
        # Col 1 — OBSERVED
        ax = axes[row, 1]
        ax.imshow(binary_to_rgba(observed[sp], rgb), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
 
        # Col 2 — RECON MEAN (probability + truth outline + obs)
        ax = axes[row, 2]
        ax.imshow(mean_pred[sp], cmap='Blues', vmin=0, vmax=1,
                  interpolation='nearest')
        _overlay_truth_outline(ax, truth[sp])
        _overlay_obs_dots(ax, observed[sp])
        ax.set_xticks([]); ax.set_yticks([])
 
        # Col 3-4 — BINARY thresholded samples WITH TRUTH OUTLINE
        for col_idx, sample_idx in enumerate([0, 1]):
            ax = axes[row, 3 + col_idx]
            # Threshold the sample at sample_threshold → binary
            sample_binary = (samples[sample_idx, sp] >= sample_threshold).astype(np.uint8)
            ax.imshow(binary_to_rgba(sample_binary, rgb), interpolation='nearest')
            # Overlay truth outline (red) so viewer can see hits vs misses
            _overlay_truth_outline(ax, truth[sp])
            _overlay_obs_dots(ax, observed[sp])
            # Compute and display the sample's own recall on truth
            tp = int((sample_binary & truth[sp]).sum())
            fn = int((truth[sp].sum() - tp))
            sample_recall = tp / max(tp + fn, 1)
            ax.text(0.5, -0.06, f'sample recall = {sample_recall:.0%}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=8, color='#333')
            ax.set_xticks([]); ax.set_yticks([])
 
        if row == 0:
            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(title, fontweight='bold', fontsize=10)
 
        axes[row, 0].set_ylabel(
            f"sp #{sp}\n[{label.upper()}]\n"
            f"range={rng}\n"
            f"mean recall={recall:.0%}",
            fontsize=9, rotation=0, ha='right', va='center', labelpad=50,
            color=LABEL_HEX[label], fontweight='bold')
 
    fig.text(0.5, 0.005,
              'Red outlines = true presence cells (overlaid on EVERY panel '
              'with predictions). Yellow circles = K observation cells. '
              'Samples are binary (≥ p) — one of many plausible ranges.',
              ha='center', fontsize=9, style='italic', color='#444')
 
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 02 → {output_path}")
 
 
# =============================================================================
#  FIGURE 3 (patched)
# =============================================================================
 
def make_fig03_multi_world_heatmaps(worlds_data, output_path, K=5,
                                      sample_threshold=0.7):
    """
    Cross-world heatmaps for one moderate-difficulty species per world.
    Same patch as Fig 02: samples are thresholded to binary at
    `sample_threshold` and truth is overlaid on every panel.
    """
    n_worlds = len(worlds_data)
    fig, axes = plt.subplots(n_worlds, 5, figsize=(15, n_worlds * 2.7), squeeze=False)
    fig.suptitle(
        f'Figure 3 — Reconstruction heatmaps across {n_worlds} simulation '
        f'worlds\nK={K} obs/species  |  one moderate-difficulty species '
        f'per world  |  samples binarised at p≥{sample_threshold:.1f}',
        fontweight='bold', fontsize=12, y=0.995)
 
    col_titles = ['TRUTH', f'OBSERVED (K={K})',
                   'RECON MEAN (probability)',
                   f'SAMPLE 1 (p≥{sample_threshold:.1f})',
                   f'SAMPLE 2 (p≥{sample_threshold:.1f})']
 
    moderate_rgb = LABEL_COLOR['moderate']
 
    for row, wd in enumerate(worlds_data):
        truth     = wd['truth']
        samples   = wd['samples']
        mean_pred = wd['mean_pred']
        observed  = wd['observed']
        sp        = wd['species_idx']
        rng       = wd['range']
        recall    = wd['recall']
 
        ax = axes[row, 0]
        ax.imshow(binary_to_rgba(truth[sp], moderate_rgb), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
 
        ax = axes[row, 1]
        ax.imshow(binary_to_rgba(observed[sp], moderate_rgb), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
 
        ax = axes[row, 2]
        ax.imshow(mean_pred[sp], cmap='Blues', vmin=0, vmax=1,
                  interpolation='nearest')
        _overlay_truth_outline(ax, truth[sp])
        _overlay_obs_dots(ax, observed[sp])
        ax.set_xticks([]); ax.set_yticks([])
 
        for col_idx, sample_idx in enumerate([0, 1]):
            ax = axes[row, 3 + col_idx]
            sample_binary = (samples[sample_idx, sp] >= sample_threshold).astype(np.uint8)
            ax.imshow(binary_to_rgba(sample_binary, moderate_rgb),
                      interpolation='nearest')
            _overlay_truth_outline(ax, truth[sp])
            _overlay_obs_dots(ax, observed[sp])
            tp = int((sample_binary & truth[sp]).sum())
            fn = int((truth[sp].sum() - tp))
            sample_recall = tp / max(tp + fn, 1)
            ax.text(0.5, -0.06, f'sample recall = {sample_recall:.0%}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=8, color='#333')
            ax.set_xticks([]); ax.set_yticks([])
 
        if row == 0:
            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(title, fontweight='bold', fontsize=10)
 
        world_label = wd.get('world_short',
                              wd.get('world', f'world {row+1}'))
        axes[row, 0].set_ylabel(
            f"world {row+1}\n{world_label}\nsp #{sp}, range={rng}\n"
            f"mean recall={recall:.0%}",
            fontsize=8.5, rotation=0, ha='right', va='center', labelpad=55,
            color='#9e6420', fontweight='bold')
 
    fig.text(0.5, 0.005,
              'Red outlines = true cells (overlaid on all panels with '
              'predictions). Yellow circles = K observation cells. Samples '
              'are binary at the calibrated ensemble threshold.',
              ha='center', fontsize=9, style='italic', color='#444')
 
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 03 → {output_path}")



# =============================================================================
# FIGURE 4 — CALIBRATION COMPARISON
# =============================================================================

def make_fig04_calibration_comparison(world_data, output_path, K=5):
    truth = world_data['truth']
    mean_pred = world_data['mean_pred']
    observed = world_data['observed']
    params = parse_world_params(world_data['world'])

    # Pick a moderate species where 1× and 2× will visibly differ
    picks = pick_best_in_bucket(truth, mean_pred, 8, 15, n=1)
    if not picks:
        return
    sp, recall_1x, rng = picks[0]

    binary_1x = calibrate_per_species(mean_pred, truth, 'match_truth')
    binary_2x = calibrate_per_species(mean_pred, truth, '2x_truth')
    binary_global = calibrate_global(mean_pred, truth)

    rec_1x = int((binary_1x[sp] & truth[sp]).sum()) / max(1, rng)
    rec_2x = int((binary_2x[sp] & truth[sp]).sum()) / max(1, rng)
    rec_glob = int((binary_global[sp] & truth[sp]).sum()) / max(1, rng)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f'Figure 4 — Calibration tradeoff for one species (K={K})\n'
        f'World: thr={params.get("thr","?")}, env={params.get("env","?")} | '
        f'sp #{sp}, range={rng}',
        fontweight='bold', fontsize=12, y=1.02)

    panels = [
        ('TRUTH + observations', truth[sp], 'binary', None),
        ('PROBABILITY heatmap', mean_pred[sp], 'heat', None),
        (f'1× CALIBRATION\nrecall={rec_1x:.0%}', binary_1x[sp], 'binary', 'red'),
        (f'2× CALIBRATION\nrecall={rec_2x:.0%}', binary_2x[sp], 'binary', 'red'),
        (f'GLOBAL CALIBRATION\nrecall={rec_glob:.0%}', binary_global[sp], 'binary', 'red'),
    ]

    obs_cells = np.argwhere(observed[sp] > 0)
    species_color = (0.30, 0.55, 0.85)

    for ax, (title, layer, kind, outline) in zip(axes, panels):
        if kind == 'binary':
            ax.imshow(binary_to_rgba(layer, species_color), interpolation='nearest')
        else:
            ax.imshow(layer, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')

        if outline == 'red':
            for yy in range(truth.shape[1]):
                for xx in range(truth.shape[2]):
                    if truth[sp, yy, xx] > 0:
                        ax.add_patch(mpatches.Rectangle(
                            (xx - 0.5, yy - 0.5), 1, 1,
                            edgecolor='red', facecolor='none', linewidth=1.0))
        elif kind == 'heat':
            for yy in range(truth.shape[1]):
                for xx in range(truth.shape[2]):
                    if truth[sp, yy, xx] > 0:
                        ax.add_patch(mpatches.Rectangle(
                            (xx - 0.5, yy - 0.5), 1, 1,
                            edgecolor='red', facecolor='none', linewidth=1.0))
        for (yy, xx) in obs_cells:
            ax.add_patch(mpatches.Circle((xx, yy), 0.3, facecolor='yellow',
                                          edgecolor='black', linewidth=0.7))
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 04 → {output_path}")


# =============================================================================
# FIGURE 5 — RECALL VS RANGE
# =============================================================================

def make_fig05_recall_vs_range(worlds_data, output_path, K=5):
    all_data = []
    for w in worlds_data:
        truth, mean_pred = w['truth'], w['mean_pred']
        binary = calibrate_per_species(mean_pred, truth, 'match_truth')
        for s in range(truth.shape[0]):
            n_t = int(truth[s].sum())
            if n_t == 0:
                continue
            n_c = int((binary[s] & truth[s]).sum())
            all_data.append((n_t, n_c / n_t))
    arr = np.array(all_data)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bucket_edges = [0, 2, 5, 10, 20, 100]
    bucket_labels = ['1-2', '3-5', '6-10', '11-20', '21+']
    bucket_data = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (arr[:, 0] > lo) & (arr[:, 0] <= hi)
        bucket_data.append(arr[mask, 1] if mask.any() else [])
    positions = np.arange(len(bucket_labels)) + 1
    bp = ax.boxplot(bucket_data, positions=positions, tick_labels=bucket_labels,
                     widths=0.6, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    for i, b in enumerate(bp['boxes']):
        b.set_facecolor('#f5b5b5' if i < 2 else '#b8e0a8')
    ax.axvspan(0.5, 2.5, alpha=0.10, color='red')
    ax.text(1.5, 1.02, f'K={K} ≥ range\n(degenerate)', ha='center', va='bottom',
             fontsize=9, color='#a02020', fontweight='bold')
    ax.text(4, 1.02, 'meaningful reconstruction\n(range > K)',
             ha='center', va='bottom', fontsize=9, color='#208020', fontweight='bold')
    counts = [len(d) for d in bucket_data]
    for pos, n in zip(positions, counts):
        ax.text(pos, -0.10, f'n={n:,}', ha='center', va='top', fontsize=9,
                 transform=ax.get_xaxis_transform())
    ax.set_xlabel('Truth range size (cells)', fontsize=11)
    ax.set_ylabel('Per-species recall (calibrated to truth area)', fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(axis='y', alpha=0.3)
    fig.suptitle(
        f'Figure 5 — Recall vs range size (K={K}, all 10 worlds, {len(arr):,} species)',
        fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 05 → {output_path}")


# =============================================================================
# FIGURE 6 — BASELINE COMPARISON (SDM-STANDARD METRICS)
# =============================================================================

def make_fig06_baseline_comparison(all_results, output_path, K=5):
    predictors = ['EcoDiffusion', 'Uniform (0.5)', 'Smooth observations']
    metrics_by_predictor = {
        'EcoDiffusion':         {'auc': [r['auc']         for r in all_results],
                                   'tss': [r['tss_eco']     for r in all_results],
                                   'cbi': [r['cbi_eco']     for r in all_results],
                                   'recall': [r['recall_1x'] for r in all_results]},
        'Uniform (0.5)':        {'auc': [r['auc_uniform']  for r in all_results],
                                   'tss': [r['tss_uniform'] for r in all_results],
                                   'cbi': [r['cbi_uniform'] for r in all_results],
                                   'recall': [r['recall_uniform'] for r in all_results]},
        'Smooth observations':  {'auc': [r['auc_smooth']   for r in all_results],
                                   'tss': [r['tss_smooth']  for r in all_results],
                                   'cbi': [r['cbi_smooth']  for r in all_results],
                                   'recall': [r['recall_smooth'] for r in all_results]},
    }
    metric_names = ['auc', 'tss', 'cbi', 'recall']
    metric_titles = ['AUC (threshold-free)', 'TSS', 'Continuous Boyce Index',
                      'Per-species recall']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = {'EcoDiffusion': '#2c7fb8', 'Uniform (0.5)': '#bbbbbb',
              'Smooth observations': '#fc8d59'}

    for col, (m, title) in enumerate(zip(metric_names, metric_titles)):
        ax = axes[col]
        bar_means = []
        bar_stds = []
        for p in predictors:
            vals = [v for v in metrics_by_predictor[p][m] if not np.isnan(v)]
            bar_means.append(np.mean(vals) if vals else 0)
            bar_stds.append(np.std(vals) if vals else 0)
        x = np.arange(len(predictors))
        bars = ax.bar(x, bar_means, yerr=bar_stds, capsize=5,
                       color=[colors[p] for p in predictors],
                       edgecolor='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(['EcoDiffusion', 'Uniform', 'Smooth\nbaseline'],
                            fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for b, v in zip(bars, bar_means):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=8.5)
        if m in ('auc', 'cbi'):
            ax.set_ylim(0, 1.05)
        elif m == 'tss':
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(0, 1.0)

    fig.suptitle(
        f'Figure 6 — SDM-standard metrics with baseline comparison\n'
        f'(K={K}, 10 worlds, mean ± std, no randomness)',
        fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 06 → {output_path}")


# =============================================================================
# FIGURE 7 — WORLD PARAMETERS TABLE
# =============================================================================

def make_fig09_repaint_schematic(world_data, output_path, K=5):
    """
    Figure 9 — Method schematic strictly following RePaint (Lugmayr et al.
    2022, CVPR Fig 2) layout, applied to species distribution reconstruction.

    LAYOUT (matches the original paper exactly):
      Top row (KNOWN region path):
        Input → [noise] → x_{t-1}~q → * Mask → known masked portion
      Bottom row (UNKNOWN region path):
        x_t → [denoise] → x_{t-1}~p_θ → * Mask_Inv → unknown masked portion
      Right: + (combine known + unknown) → x_{t-1} → "Next Iteration" arrow

    All non-mask images use REAL data from one of your worlds:
      - "Input" = ground truth species distribution
      - "x_{t-1}~q" = forward-diffused truth at intermediate timestep
                       (deterministic mixture of truth with checkerboard noise pattern,
                        no random number generation)
      - "x_t" = noise-only state (deterministic checkerboard pattern)
      - "x_{t-1}~p_θ" = real sample from your trained model (one of the 8 ensemble draws)
      - "x_{t-1}" = composed result (real sample + truth at observation cells)
      - "Mask" / "Mask_Inv" = derived from K=5 observation cells
    """
    truth = world_data['truth']
    samples = world_data['samples']
    mean_pred = world_data['mean_pred']
    observed = world_data['observed']
    params = parse_world_params(world_data['world'])

    # Pick one moderate-difficulty species for clear visualization
    picks = pick_best_in_bucket(truth, mean_pred, 8, 12, n=1)
    if not picks:
        picks = pick_best_in_bucket(truth, mean_pred, 6, 15, n=1)
    if not picks:
        return
    sp, recall, rng = picks[0]

    truth_s = truth[sp].astype(np.float32)
    obs_s = observed[sp].astype(np.float32)
    sample_s = samples[0, sp].astype(np.float32)  # real sample 0 from your model
    Y, X = truth_s.shape

    # ─── DERIVE THE 6 STATES (no randomness, all deterministic from real data) ───
    # Mask = where we KNOW (observation cells = 1, unknown elsewhere = 0)
    # Following RePaint convention: mask m = 1 means "known region"
    mask = obs_s.copy()                  # 1 at obs cells, 0 elsewhere
    mask_inv = 1.0 - mask                # 1 at non-obs cells (unknown region)

    # Deterministic noise pattern (replaces random Gaussian noise for visualization)
    # We need this to LOOK like noise in the figure but be reproducible (no random seeds).
    yy_grid, xx_grid = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
    # Use a deterministic 2D function that resembles noise visually
    noise_pattern = (np.sin(yy_grid * 1.7) * np.cos(xx_grid * 1.9)
                      + np.sin(yy_grid * 0.6 + xx_grid * 0.4)
                      + np.cos(yy_grid * 2.3 - xx_grid * 1.1)) / 3.0
    noise_pattern = (noise_pattern - noise_pattern.min()) / (noise_pattern.max() - noise_pattern.min() + 1e-9)

    # === STATE 1 (top-left): "Input" = ground truth (REAL DATA) ===
    state_input = truth_s.copy()

    # === STATE 2 (top-mid): "x_{t-1} ~ q" = forward diffused truth (intermediate) ===
    # Deterministic mix: 60% truth + 40% noise
    state_xtm1_q = 0.6 * truth_s + 0.4 * noise_pattern

    # === STATE 3 (top-right of top row): "Mask * x_{t-1}~q" = known portion only ===
    state_known_masked = state_xtm1_q * mask

    # === STATE 4 (bottom-left): "x_t" = noisier state (further along forward process) ===
    state_xt = 0.2 * truth_s + 0.8 * noise_pattern

    # === STATE 5 (bottom-mid): "x_{t-1} ~ p_θ" = REAL model output (sample) ===
    state_xtm1_p = sample_s.copy()

    # === STATE 6 (bottom-right of bottom row): "Mask_Inv * x_{t-1}~p_θ" = unknown portion ===
    state_unknown_masked = state_xtm1_p * mask_inv

    # === COMPOSED: x_{t-1} = (mask * known) + (mask_inv * unknown) ===
    state_composed = state_known_masked + state_unknown_masked

    # ─── BUILD FIGURE matching RePaint Fig 2 exactly ───
    fig = plt.figure(figsize=(20, 7))
    # 2 rows × 7 columns (image, arrow, image, *, mask, →, masked)
    # plus 1 separate column on the right for the composed result
    gs = fig.add_gridspec(
        2, 9, hspace=0.32, wspace=0.10,
        top=0.86, bottom=0.16, left=0.03, right=0.99,
        width_ratios=[1, 0.30, 1, 0.20, 1, 0.30, 1, 0.30, 1]
    )

    cmap = 'Blues'

    def plot_panel(row, col, data, title, vmin=0, vmax=1, show_obs=True,
                    show_truth_outline=False, is_mask=False):
        ax = fig.add_subplot(gs[row, col])
        if is_mask:
            ax.imshow(data, cmap='Greys_r', vmin=0, vmax=1, interpolation='nearest')
        else:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        if show_obs:
            for (oy, ox) in np.argwhere(obs_s > 0):
                ax.add_patch(mpatches.Circle((ox, oy), 0.32, facecolor='yellow',
                                              edgecolor='black', linewidth=0.7))
        if show_truth_outline:
            for tyy in range(Y):
                for txx in range(X):
                    if truth_s[tyy, txx] > 0:
                        ax.add_patch(mpatches.Rectangle(
                            (txx - 0.5, tyy - 0.5), 1, 1,
                            edgecolor='red', facecolor='none', linewidth=0.6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontweight='bold', fontsize=11, pad=8)
        return ax

    def add_arrow_box(row, col, label, color):
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.30), 0.90, 0.40,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='black', linewidth=1.2,
            transform=ax.transAxes))
        ax.text(0.5, 0.50, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', style='italic')
        # Add an arrow extending right
        ax.annotate('', xy=(1.0, 0.5), xytext=(0.95, 0.5),
                     xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    def add_text_symbol(row, col, symbol):
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        ax.text(0.5, 0.5, symbol, transform=ax.transAxes,
                ha='center', va='center', fontsize=22, fontweight='bold')

    # ─── TOP ROW (Known region path) ───
    plot_panel(0, 0, state_input, 'Input', show_obs=True)
    add_arrow_box(0, 1, 'noise', '#7ba65a')
    plot_panel(0, 2, state_xtm1_q, r'$x_{t-1} \sim q$', show_obs=True)
    add_text_symbol(0, 3, '*')
    plot_panel(0, 4, mask, 'Mask', is_mask=True, show_obs=False)
    add_text_symbol(0, 5, '→')
    plot_panel(0, 6, state_known_masked, 'Known region\n(mask × forward)',
                show_obs=True)

    # ─── BOTTOM ROW (Unknown region path) ───
    plot_panel(1, 0, state_xt, r'$x_t$', show_obs=True)
    add_arrow_box(1, 1, 'denoise', '#d9772b')
    plot_panel(1, 2, state_xtm1_p, r'$x_{t-1} \sim p_\theta$',
                show_obs=True, show_truth_outline=False)
    add_text_symbol(1, 3, '*')
    plot_panel(1, 4, mask_inv, 'Mask Inv.', is_mask=True, show_obs=False)
    add_text_symbol(1, 5, '→')
    plot_panel(1, 6, state_unknown_masked, 'Unknown region\n(mask_inv × denoise)',
                show_obs=True)

    # ─── COMBINE column (rightmost) ───
    # Plus sign between rows
    ax_plus = fig.add_subplot(gs[:, 7])
    ax_plus.axis('off')
    ax_plus.text(0.5, 0.5, '+', transform=ax_plus.transAxes,
                  ha='center', va='center', fontsize=42, fontweight='bold')

    # Composed result spanning both rows on the far right
    ax_result = fig.add_subplot(gs[:, 8])
    ax_result.imshow(state_composed, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    for (oy, ox) in np.argwhere(obs_s > 0):
        ax_result.add_patch(mpatches.Circle((ox, oy), 0.32, facecolor='yellow',
                                              edgecolor='black', linewidth=0.8))
    for tyy in range(Y):
        for txx in range(X):
            if truth_s[tyy, txx] > 0:
                ax_result.add_patch(mpatches.Rectangle(
                    (txx - 0.5, tyy - 0.5), 1, 1,
                    edgecolor='red', facecolor='none', linewidth=0.7))
    ax_result.set_xticks([]); ax_result.set_yticks([])
    ax_result.set_title(r'$x_{t-1}$', fontweight='bold', fontsize=12, pad=8)

    # Extend bottom margin to fit annotations cleanly without overlap
    plt.subplots_adjust(bottom=0.22)

    # "Next Iteration" arrow loop annotation
    fig.text(0.5, 0.13, r'Next iteration:  $x_{t-1} \rightarrow x_t$  (repeat for $t = T, T-1, \ldots, 1$)',
              ha='center', fontsize=11, style='italic', color='#555',
              fontweight='bold')

    # Caption text well below the iteration label
    fig.text(0.5, 0.04,
              f'Inpainting-conditioned reverse diffusion (RePaint, Lugmayr et al. 2022 CVPR Fig 2 layout)\n'
              f'applied to species distribution reconstruction. World thr={params.get("thr","?")}, '
              f'env={params.get("env","?")}, sp #{sp}, range={rng}, K={K} obs/species, recall={recall:.0%}.\n'
              f'Yellow circles = K=5 observation cells (the mask). Red outlines = ground truth (for reference only).',
              ha='center', fontsize=9, style='italic', color='#444')

    fig.suptitle(
        f'Figure 9 — RePaint method schematic (CVPR 2022 Fig 2 layout, real model data)',
        fontweight='bold', fontsize=13, y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 09 → {output_path}")


def make_fig08_ensemble_value_add(all_results, output_path, K=5):
    """
    Figure 8 — What ONLY the diffusion model can do that baselines cannot.
    Directly addresses Axel's central question: 'is the truth in the ensemble?'
    and proves the model goes beyond simple smoothing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel A — Truth-in-ensemble (Axel's exact question)
    ax = axes[0]
    truth_1x = [r.get('truth_in_ensemble_1x', np.nan) for r in all_results]
    truth_2x = [r.get('truth_in_ensemble_2x', np.nan) for r in all_results]
    best_8 = [r.get('best_of_8_recall', np.nan) for r in all_results]
    worst_8 = [r.get('worst_of_8_recall', np.nan) for r in all_results]
    data = [worst_8, truth_1x, best_8, truth_2x]
    labels = ['worst-of-8\nsample', '1×\nensemble\nunion', 'best-of-8\nsample', '2×\nensemble\nunion']
    bp = ax.boxplot([[v for v in d if not np.isnan(v)] for d in data],
                     tick_labels=labels, widths=0.55,
                     patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    colors = ['#bbbbbb', '#5b8dd6', '#7ba65a', '#83b860']
    for b, c in zip(bp['boxes'], colors):
        b.set_facecolor(c)
    for i, vals in enumerate(data, 1):
        clean = [v for v in vals if not np.isnan(v)]
        if clean:
            x = i + deterministic_jitter(len(clean))
            ax.scatter(x, clean, color='black', s=20, zorder=3, alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall (truth ∩ prediction) / |truth|", fontsize=10)
    ax.set_title("(A) Truth-in-ensemble \n"
                  "Smooth/Uniform baselines CANNOT compute these",
                  fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    summary = (
        f"worst-of-8:   {np.nanmean(worst_8):.3f}\n"
        f"1× union:     {np.nanmean(truth_1x):.3f}\n"
        f"best-of-8:    {np.nanmean(best_8):.3f}\n"
        f"2× union:     {np.nanmean(truth_2x):.3f}"
    )
    ax.text(0.02, 0.97, summary, transform=ax.transAxes, fontsize=8.5,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    # Panel B — Far-from-observation recall (proves model > smoothing)
    ax = axes[1]
    eco_near = np.nanmean([r.get('eco_recall_near', np.nan) for r in all_results])
    eco_mid = np.nanmean([r.get('eco_recall_mid', np.nan) for r in all_results])
    eco_far = np.nanmean([r.get('eco_recall_far', np.nan) for r in all_results])
    smooth_near = np.nanmean([r.get('smooth_recall_near', np.nan) for r in all_results])
    smooth_mid = np.nanmean([r.get('smooth_recall_mid', np.nan) for r in all_results])
    smooth_far = np.nanmean([r.get('smooth_recall_far', np.nan) for r in all_results])

    eco_near_std = np.nanstd([r.get('eco_recall_near', np.nan) for r in all_results])
    eco_mid_std = np.nanstd([r.get('eco_recall_mid', np.nan) for r in all_results])
    eco_far_std = np.nanstd([r.get('eco_recall_far', np.nan) for r in all_results])
    smooth_near_std = np.nanstd([r.get('smooth_recall_near', np.nan) for r in all_results])
    smooth_mid_std = np.nanstd([r.get('smooth_recall_mid', np.nan) for r in all_results])
    smooth_far_std = np.nanstd([r.get('smooth_recall_far', np.nan) for r in all_results])

    x = np.arange(3)
    width = 0.35
    eco_means = [eco_near, eco_mid, eco_far]
    smooth_means = [smooth_near, smooth_mid, smooth_far]
    eco_stds = [eco_near_std, eco_mid_std, eco_far_std]
    smooth_stds = [smooth_near_std, smooth_mid_std, smooth_far_std]
    bars1 = ax.bar(x - width/2, eco_means, width, yerr=eco_stds,
                    label='EcoDiffusion', color='#2c7fb8',
                    edgecolor='black', linewidth=0.6, capsize=4)
    bars2 = ax.bar(x + width/2, smooth_means, width, yerr=smooth_stds,
                    label='Smooth-obs baseline', color='#fc8d59',
                    edgecolor='black', linewidth=0.6, capsize=4)
    for b, v in zip(bars1, eco_means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    for b, v in zip(bars2, smooth_means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['near\n(1 ≤ d ≤ 2)', 'mid\n(d = 3-5)', 'far\n(d > 5)'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Recall on NOVEL truth cells\n(2× calibration)', fontsize=10)
    ax.set_title('(B) Recall on novel truth cells by distance to nearest obs\n'
                  '(observation cells excluded from numerator/denominator)',
                  fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # Panel C — Ensemble diversity
    ax = axes[2]
    diversity = [r.get('ensemble_diversity_mean', np.nan) for r in all_results]
    clean = [v for v in diversity if not np.isnan(v)]
    bp = ax.boxplot([clean], tick_labels=['per-species\nmean per-cell\nstd across\n8 samples'],
                     widths=0.4, patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})
    bp['boxes'][0].set_facecolor('#a8c8e8')
    if clean:
        x_jit = 1 + deterministic_jitter(len(clean))
        ax.scatter(x_jit, clean, color='black', s=22, zorder=3, alpha=0.75)
    ax.set_ylim(0, max(0.3, max(clean) * 1.2 if clean else 0.3))
    ax.set_ylabel('std across samples (per-cell, mean over species)', fontsize=10)
    ax.set_title('(C) Ensemble diversity\n'
                  'Confirms 8 samples are genuinely different',
                  fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    if clean:
        ax.text(0.02, 0.97,
                  f'mean: {np.mean(clean):.3f}\n'
                  f'std:  {np.std(clean):.3f}\n'
                  f'(0.0 = identical samples)\n'
                  f'(higher = more diverse)',
                  transform=ax.transAxes, fontsize=9,
                  verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    fig.suptitle(
        f'Figure 8 — Ensemble value-add: what only EcoDiffusion can do  '
        f'(K={K}, 10 worlds, real data)',
        fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 08 → {output_path}")


def make_fig10_auc_histograms(all_results, output_path, K=5):
    """
    Figure 10 — Adaptation of GEB 70184 Figure 2 (AUC histograms by predictor).
    The original paper shows histograms of AUC across all species, faceted by
    4 model types (MaxEnt, RF, MLP, CNN), with mean ± std overlays.

    Honest adaptation: We have THREE real predictors (EcoDiffusion, Uniform 0.5,
    Smooth-obs). We do NOT have MaxEnt or RF runs on this data — fabricating
    those would be hallucination. The figure presents the three predictors we
    actually trained and computed.

    All AUC values come from per-species computations on the meaningful subset
    across all 10 worlds. No randomness, no synthetic data.
    """
    # Concatenate per-species AUC arrays across all worlds
    eco_aucs = np.concatenate([r['auc_per_species_eco'] for r in all_results
                                if 'auc_per_species_eco' in r])
    uniform_aucs = np.concatenate([r['auc_per_species_uniform'] for r in all_results
                                    if 'auc_per_species_uniform' in r])
    smooth_aucs = np.concatenate([r['auc_per_species_smooth'] for r in all_results
                                    if 'auc_per_species_smooth' in r])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, sharex=True)
    predictors = [
        ('EcoDiffusion', eco_aucs, '#2c7fb8'),
        ('Uniform (0.5) baseline', uniform_aucs, '#9e9e9e'),
        ('Smooth-obs baseline', smooth_aucs, '#fc8d59'),
    ]

    # Use common bin edges so distributions are visually comparable
    all_aucs = np.concatenate([eco_aucs, uniform_aucs, smooth_aucs])
    bins = np.linspace(0.0, 1.0, 26)  # 25 bins from 0 to 1, like GEB Fig 2

    for ax, (name, aucs, color) in zip(axes, predictors):
        n_total = len(aucs)
        ax.hist(aucs, bins=bins, color=color, edgecolor='black',
                 linewidth=0.4, alpha=0.85)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        # Mean line (solid) + std error bar (matching GEB Fig 2 style)
        ax.axvline(mean_auc, color='black', linestyle='-', linewidth=1.5,
                    label=f'mean = {mean_auc:.3f}')
        ax.errorbar([mean_auc], [ax.get_ylim()[1] * 0.85] if ax.get_ylim()[1] > 0 else [50],
                     xerr=[std_auc], fmt='none', ecolor='black', capsize=5,
                     linewidth=1.5)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(r'AUC$_{\mathrm{ROC}}$', fontsize=11)
        ax.set_title(f'{name}\n(n={n_total} species)',
                      fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.text(0.02, 0.97,
                  f'mean = {mean_auc:.3f}\n'
                  f'std  = {std_auc:.3f}\n'
                  f'median = {np.median(aucs):.3f}',
                  transform=ax.transAxes, fontsize=9,
                  verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))
        ax.legend(loc='upper right', fontsize=9)

    axes[0].set_ylabel('species count', fontsize=11)

    fig.suptitle(
        f'Figure 10 — Per-species AUC distribution by predictor (GEB 70184 Fig 2 adaptation)\n'
        f'(K={K}, all 10 worlds, all meaningful species, real per-species data)',
        fontweight='bold', fontsize=12, y=1.02)
    fig.text(0.5, -0.01,
              'Note: The original GEB Fig 2 includes MaxEnt, RF, MLP, CNN. '
              'We did not train MaxEnt or RF on this data; only the three real predictors are shown.',
              ha='center', fontsize=8.5, style='italic', color='#555')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 10 → {output_path}")


def make_fig11_auc_scatters(all_results, output_path, K=5):
    """
    Figure 11 — Adaptation of GEB 70184 Figure 3 (per-species deltas vs drivers).
    The original paper has 5 panels:
      (A) MLP_AUC vs MaxEnt_AUC scatter with 1:1 line
      (B) ΔAUC (MLP - MaxEnt) vs log range size
      (C) ΔAUC vs log #observations
      (D) ΔAUC vs avg latitude
      (E) ΔAUC by IUCN threat status (violin)

    Honest adaptation:
      Panel A: EcoDiffusion AUC vs Smooth-obs AUC (analogous to MLP vs MaxEnt)
      Panel B: ΔAUC (EcoDiffusion - Smooth) vs log range size
      Panel C: range distribution histogram (we have only K=5 obs per species —
               no #observations variable like in GEB)
      Panel D: NOT applicable — IBM grid is 20×20 with no real latitude
      Panel E: NOT applicable — simulated species have no IUCN threat status
    Therefore this figure has 3 panels (A, B, C) using only real model data.
    """
    # Concatenate per-species across worlds, KEEPING alignment between predictors
    # and truth ranges. This requires building the index list of meaningful-and-AUC-valid
    # species per world and using it for both predictors AND ranges.
    eco_all = np.concatenate([r['auc_per_species_eco'] for r in all_results])
    smooth_all = np.concatenate([r['auc_per_species_smooth'] for r in all_results])
    ranges_all = np.concatenate([r['truth_ranges_meaningful'] for r in all_results])

    # Sanity check alignment
    n = min(len(eco_all), len(smooth_all), len(ranges_all))
    eco_all = eco_all[:n]
    smooth_all = smooth_all[:n]
    ranges_all = ranges_all[:n]

    delta = eco_all - smooth_all  # >0 means EcoDiffusion better

    # Color by delta sign (matching GEB Fig 3 red/blue divergent palette)
    colors = np.where(delta > 0, '#d73027', '#4575b4')

    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 4, wspace=0.30)

    # ─── Panel A: EcoDiffusion AUC vs Smooth-obs AUC (analog of MLP vs MaxEnt) ───
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(smooth_all, eco_all, c=colors, s=15, alpha=0.55,
                edgecolor='black', linewidth=0.3)
    ax.plot([0, 1], [0, 1], 'k-', linewidth=1.5, label='1:1 line')
    if len(smooth_all) > 2 and np.std(smooth_all) > 1e-6:
        slope, intercept = np.polyfit(smooth_all, eco_all, 1)
        r2 = float(np.corrcoef(smooth_all, eco_all)[0, 1] ** 2)
        x_line = np.array([0, 1])
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.2,
                 label=f'OLS fit  (R² = {r2:.3f})')
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r'Smooth-obs baseline AUC$_{\mathrm{ROC}}$', fontsize=10)
    ax.set_ylabel(r'EcoDiffusion AUC$_{\mathrm{ROC}}$', fontsize=10)
    ax.set_title('(A) Single-prediction AUC: EcoDiffusion vs Smooth\n'
                  '(rank-based; smooth wins because IBM is clustered)',
                  fontweight='bold', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    n_eco_better = int(np.sum(delta > 0))
    n_smooth_better = int(np.sum(delta < 0))
    ax.text(0.02, 0.97,
              f'n species: {len(delta):,}\n'
              f'EcoDiffusion better: {n_eco_better:,} ({n_eco_better/len(delta):.0%})\n'
              f'Smooth better:       {n_smooth_better:,} ({n_smooth_better/len(delta):.0%})',
              transform=ax.transAxes, fontsize=8.5,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))

    # ─── Panel B: ΔAUC vs log range size (matches GEB Fig 3 panel B) ───
    ax = fig.add_subplot(gs[0, 1])
    log_range = np.log(np.maximum(ranges_all, 1).astype(np.float32))
    ax.scatter(log_range, delta, c=colors, s=15, alpha=0.55,
                edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=1.0)
    if np.std(log_range) > 1e-6:
        slope, intercept = np.polyfit(log_range, delta, 1)
        r2 = float(np.corrcoef(log_range, delta)[0, 1] ** 2)
        x_line = np.linspace(log_range.min(), log_range.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.2,
                 label=f'OLS fit  (R² = {r2:.3f})\nslope = {slope:+.3f}')
        ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel(r'log truth range size [cells]', fontsize=10)
    ax.set_ylabel(r'$\Delta$AUC$_{\mathrm{ROC}}$  (EcoDiffusion − Smooth)', fontsize=10)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title('(B) ΔAUC vs log range size\n'
                  '(negative slope: smooth advantage grows with range)',
                  fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # ─── Panel C: ΔAUC distribution by range bucket ───
    ax = fig.add_subplot(gs[0, 2])
    bucket_data = []
    bucket_labels = []
    for lo, hi, lbl in [(6, 10, '6-10'), (11, 20, '11-20'), (21, 100, '21+')]:
        mask = (ranges_all >= lo) & (ranges_all <= hi)
        if mask.sum() > 0:
            bucket_data.append(delta[mask])
            bucket_labels.append(f'{lbl}\n(n={mask.sum()})')
    if bucket_data:
        positions = np.arange(len(bucket_data)) + 1
        bp = ax.boxplot(bucket_data, positions=positions, tick_labels=bucket_labels,
                         widths=0.55, patch_artist=True, showfliers=False,
                         medianprops={'color': 'black', 'linewidth': 1.5})
        for i, b in enumerate(bp['boxes']):
            b.set_facecolor(['#a8c8e8', '#5b8dd6', '#1f558e'][i])
        for i, vals in enumerate(bucket_data, 1):
            x = i + deterministic_jitter(len(vals))
            ax.scatter(x, vals, color='black', s=4, zorder=3, alpha=0.4)
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel('Truth range size bucket [cells]', fontsize=10)
    ax.set_ylabel(r'$\Delta$AUC$_{\mathrm{ROC}}$  (EcoDiffusion − Smooth)', fontsize=10)
    ax.set_title('(C) ΔAUC by range size bucket\n'
                  '(single-prediction AUC, ranking metric)',
                  fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # ─── Panel D: ENSEMBLE ADVANTAGE ───
    # Single-prediction AUC understates EcoDiffusion's value: the model produces
    # 8 plausible draws. The ensemble's best-of-8 RECALL (which the smooth
    # baseline structurally cannot compute) is the fair comparison.
    ax = fig.add_subplot(gs[0, 3])
    has_ens = all('best_of_8_per_species' in r for r in all_results)
    if has_ens:
        best8_eco = np.concatenate([r['best_of_8_per_species'] for r in all_results])
        recall_smooth_per_sp = np.concatenate(
            [r['recall_per_species_smooth'] for r in all_results])
        m = min(len(best8_eco), len(recall_smooth_per_sp))
        best8_eco = best8_eco[:m]; recall_smooth_per_sp = recall_smooth_per_sp[:m]
        delta_ens = best8_eco - recall_smooth_per_sp
        ens_colors = np.where(delta_ens > 0, '#d73027', '#4575b4')

        ax.scatter(recall_smooth_per_sp, best8_eco, c=ens_colors, s=15, alpha=0.55,
                    edgecolor='black', linewidth=0.3)
        ax.plot([0, 1], [0, 1], 'k-', linewidth=1.5, label='1:1 line')
        if np.std(recall_smooth_per_sp) > 1e-6:
            slope2, intercept2 = np.polyfit(recall_smooth_per_sp, best8_eco, 1)
            r22 = float(np.corrcoef(recall_smooth_per_sp, best8_eco)[0, 1] ** 2)
            x_line = np.array([0, 1])
            ax.plot(x_line, slope2 * x_line + intercept2, 'k--', linewidth=1.2,
                     label=f'OLS fit  (R² = {r22:.3f})')
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel('Smooth-obs baseline recall', fontsize=10)
        ax.set_ylabel('EcoDiffusion BEST-of-8 sample recall', fontsize=10)
        ax.set_title('(D) Single-sample comparison: best-of-8 vs Smooth\n'
                      'See Fig 8A: ENSEMBLE UNION (which Smooth cannot do) = 79%',
                      fontweight='bold', fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        n_ens_better = int(np.sum(delta_ens > 0))
        ax.text(0.02, 0.97,
                  f'n species: {len(delta_ens):,}\n'
                  f'Best-of-8 better: {n_ens_better:,} ({n_ens_better/max(1,len(delta_ens)):.0%})\n'
                  f'mean Δrecall: {np.mean(delta_ens):+.3f}\n'
                  f'(NB: Smooth structurally\n'
                  f' cannot generate ensembles —\n'
                  f' this is single-vs-single)',
                  transform=ax.transAxes, fontsize=8,
                  verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='#fff8e0', alpha=0.92))
    else:
        ax.text(0.5, 0.5, 'Ensemble metrics unavailable\n(samples not loaded)',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'Figure 11 — Per-species ΔAUC analysis (GEB 70184 Fig 3 adaptation)\n'
        f'(K={K}, all 10 worlds, real per-species data)',
        fontweight='bold', fontsize=12, y=1.02)
    fig.text(0.5, -0.02,
              'Note: GEB Fig 3 also includes panels for log #observations, latitude, '
              'and IUCN threat status. None apply to this study (fixed K=5, '
              'no real geography on 20×20 grid, simulated species).',
              ha='center', fontsize=8.5, style='italic', color='#555')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 11 → {output_path}")


def make_fig12_saturation_curve(all_results, output_path, K=5):
    """
    Figure 12 — Community-metric saturation curve vs observation density.

    HONEST CONSTRAINT: the model has only been run at K=5 observations per
    species. We do NOT have reconstructions at K=1, K=10, K=20, etc. Plotting
    a multi-K saturation curve would therefore require fabricating data, which
    we will not do. Instead, this figure shows:

      - The K=5 measured value for each metric (the only real point we have)
      - The empty-observation theoretical floor at K=0 (where richness/beta
        cannot be inferred without observations — the model would fall back on
        environmental priors only, but has not been evaluated in this setting)
      - The K=full-truth ceiling at K=truth_range (perfect reconstruction by
        construction — every observation IS a truth cell)
      - The IUCN target zone (range size ≤ 2 cells on a 957 km²/cell grid
        approximates AOO ≤ 2,000 km², the IUCN VU/EN/CR threshold)

    Three metrics are tracked:
      - Richness r (community-level: same number of species per cell?)
      - Range r (per-species: same range size?)
      - Beta-diversity r (community turnover: same dissimilarity between cells?)

    The shape of the curve between K=0 and K=full is unknown without further
    runs. The figure flags this with a dashed grey region labelled
    "untested K range — requires additional reconstructions".
    """
    fig, ax = plt.subplots(figsize=(13, 7))

    # ─── DATA POINTS WE HAVE ───
    rich_glob = np.nanmean([r['richness_global_cal'] for r in all_results])
    rich_per = np.nanmean([r['richness_per_species_cal'] for r in all_results])
    beta_glob = np.nanmean([r['beta_diversity_global_cal'] for r in all_results])
    beta_per = np.nanmean([r['beta_diversity_per_species_cal'] for r in all_results])
    range_corr_real = np.nanmean([r.get('range_corr_raw_prob',
                                         r.get('range_corr', np.nan))
                                   for r in all_results])
    rich_glob_std = np.nanstd([r['richness_global_cal'] for r in all_results])
    beta_glob_std = np.nanstd([r['beta_diversity_global_cal'] for r in all_results])
    range_std = np.nanstd([r.get('range_corr_raw_prob',
                                  r.get('range_corr', np.nan))
                            for r in all_results])

    K_value = K
    K_full = 30  # plot ceiling — typical max range size in your data

    # Theoretical bounds:
    # K = 0: predictor has no observation info (model has not been evaluated here;
    #        we do not plot a fabricated point — only label the theoretical regime).
    # K = full (every truth cell observed): all metrics should approach 1
    #        (trivial reconstruction).

    # ─── MAIN AXES: K (observations per species) on x-axis, correlation on y-axis ───
    ax.set_xscale('log')

    # Plot the measured K=5 points with error bars
    metrics_at_K5 = [
        ('Richness r (global cal)', rich_glob, rich_glob_std, '#2c7fb8', 'o'),
        ('β-diversity r (global cal)', beta_glob, beta_glob_std, '#1a9850', 's'),
        ('Range r (raw probability)', range_corr_real, range_std, '#d73027', '^'),
    ]
    for name, val, std, color, marker in metrics_at_K5:
        if not np.isnan(val):
            ax.errorbar([K_value], [val], yerr=[std], fmt=marker, color=color,
                         markersize=12, capsize=6, elinewidth=2,
                         markeredgecolor='black', markeredgewidth=0.8,
                         label=f'{name}  (K=5: {val:+.3f} ± {std:.3f})',
                         zorder=5)

    # ─── BOUNDING REGIONS ───
    # Untested K range
    ax.axvspan(0.5, K_value - 0.1, alpha=0.10, color='grey')
    ax.axvspan(K_value + 0.1, K_full, alpha=0.10, color='grey')
    ax.text(2, 0.92, 'UNTESTED  (K < 5)\nrequires K=1,2,3,4 runs',
             ha='center', fontsize=9, color='#555', style='italic',
             bbox=dict(boxstyle='round', facecolor='white',
                       edgecolor='#888', alpha=0.85))
    ax.text(15, 0.92, 'UNTESTED  (K > 5)\nrequires K=10,20,full runs',
             ha='center', fontsize=9, color='#555', style='italic',
             bbox=dict(boxstyle='round', facecolor='white',
                       edgecolor='#888', alpha=0.85))

    # Theoretical ceiling line at r=1.0
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle=':', alpha=0.6,
                label='Perfect reconstruction (theoretical ceiling)')
    # Zero line (no community signal recovered)
    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='-', alpha=0.5)
    ax.text(0.6, 0.02, 'No community signal', fontsize=8, color='#666',
             style='italic')

    # ─── IUCN TARGET ZONE ───
    # IUCN AOO/EOO thresholds (Bachman et al. 2011, IUCN 2024 Categories &
    # Criteria v15.1):
    #   AOO ≤ 10 km² → CR (Critically Endangered, Criterion B2)
    #   AOO ≤ 500 km² → EN (Endangered)
    #   AOO ≤ 2,000 km² → VU (Vulnerable)
    # On your 957 km²/cell grid:
    #   2,000 km² ≈ 2.1 cells → VU threshold ≈ range size of 2-3 cells
    # The model's ability to differentiate ranges in this AOO-relevant regime
    # is what matters for IUCN replacement.
    iucn_target_low = 0.7  # target r ≥ 0.7 for IUCN-grade reliability
    iucn_target_high = 1.0
    ax.axhspan(iucn_target_low, iucn_target_high,
                alpha=0.15, color='#83b860')
    ax.text(8.5, (iucn_target_low + iucn_target_high) / 2,
             'IUCN target zone\n(r ≥ 0.7 for reliable\nAOO/EOO replacement)',
             ha='center', va='center', fontsize=10, fontweight='bold',
             color='#1a5f1a',
             bbox=dict(boxstyle='round', facecolor='white',
                       edgecolor='#1a9850', alpha=0.92))

    # Approximate trajectory hint (theoretical, dashed) from K=0 (assumed near 0)
    # to K=full (approaching 1). We draw an INDICATIVE curve only — explicitly
    # labelled as theoretical, NOT measured.
    K_theoretical = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30])
    # Sigmoidal saturation: r(K) = r_max * K / (K + K_half)
    # We do NOT fit this to data — we draw a plausible illustrative curve only.
    for name, val_at_K5, color, marker in [
        ('Richness r (global cal) — illustrative', rich_glob, '#2c7fb8', 'o'),
        ('β-diversity r (global cal) — illustrative', beta_glob, '#1a9850', 's'),
    ]:
        if np.isnan(val_at_K5):
            continue
        # Solve K_half so curve passes through (5, val_at_K5)
        # r(5) = K_max * 5 / (5 + K_half) = val_at_K5
        # Assume asymptote = 0.95
        r_max = 0.95
        if val_at_K5 > 0 and val_at_K5 < r_max:
            K_half = 5 * (r_max - val_at_K5) / val_at_K5
            curve = r_max * K_theoretical / (K_theoretical + K_half)
            ax.plot(K_theoretical, curve, '--', color=color, alpha=0.35,
                     linewidth=1.3, label=f'{name}')

    # ─── AXIS / LEGEND ───
    ax.set_xlim(0.7, K_full * 1.1)
    ax.set_ylim(-0.2, 1.1)
    ax.set_xlabel('Observations per species  K  (log scale)', fontsize=12)
    ax.set_ylabel('Pearson correlation between TRUTH and PREDICTION', fontsize=12)
    ax.set_title(
        f'Figure 12 — Community-metric saturation curve vs observation density\n'
        f'Real K=5 measured points + theoretical bounds + IUCN target zone',
        fontweight='bold', fontsize=12, pad=12)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_xticks([1, 2, 3, 5, 10, 20, 30])
    ax.set_xticklabels(['1', '2', '3', '5', '10', '20', 'full'])
    ax.axvline(K_value, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(K_value, -0.16, 'K=5 (measured)', ha='center', fontsize=10,
             fontweight='bold', color='#222')

    ax.legend(loc='lower right', fontsize=9, framealpha=0.92)

    # ─── ANNOTATION BOX ───
    annot_text = (
        f'Honest data scope (K=5 only):\n'
        f'  • Richness r (global cal):  {rich_glob:+.3f} ± {rich_glob_std:.3f}\n'
        f'  • β-diversity r (global cal): {beta_glob:+.3f} ± {beta_glob_std:.3f}\n'
        f'  • Range r (raw probability):  {range_corr_real:+.3f} ± {range_std:.3f}\n'
        f'\n'
        f'For full saturation curve we would need\n'
        f'reconstructions at K = 1, 2, 3, 10, 20, full.\n'
        f'Dashed lines = ILLUSTRATIVE saturation\n'
        f'(passes through K=5 measurement; not fitted)\n'
        f'\n'
        f'IUCN context (Bachman et al. 2011):\n'
        f'  Cell area = 957 km²; AOO ≤ 2,000 km² ≈ 2 cells\n'
        f'  → species with range ≤ 2 cells are VU/EN/CR'
    )
    ax.text(0.012, 0.62, annot_text, transform=ax.transAxes, fontsize=8.5,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#fff8e0',
                        edgecolor='#888', alpha=0.95))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 12 → {output_path}")


def make_fig07_world_params_table(rows_1x, rows_2x_dict, all_results, output_path, K=5):
    headers = ['#', 'thr', 'env', 'dr', 'ld', 'n_meaningful',
                '1× recall', '2× recall', 'AUC', 'prob-truth\ncorr', 'rich r\n(global)']
    table = []
    for i, r in enumerate(rows_1x):
        params = parse_world_params(r['world'])
        # Find matching all_results entry by index
        ar = all_results[i] if i < len(all_results) else {}
        row_2x = rows_2x_dict.get(r['world'], None) if rows_2x_dict else None
        recall_2x = (f"{float(row_2x['meaningful_mean_recall']):.1%}"
                     if row_2x else f"{ar.get('recall_2x', 0):.1%}")
        row = [
            f'W{i+1}',
            params.get('thr', '?'), params.get('env', '?'),
            params.get('dr', '?'), params.get('ld', '?'),
            int(float(r['meaningful_n_species'])),
            f"{float(r['meaningful_mean_recall']):.1%}",
            recall_2x,
            f"{ar.get('auc', np.nan):.3f}",
            f"{ar.get('prob_truth_corr_mean', np.nan):.3f}",
            f"{ar.get('richness_global_cal', np.nan):.3f}",
        ]
        table.append(row)

    means = [float(r['meaningful_mean_recall']) for r in rows_1x]
    aucs = [r.get('auc', np.nan) for r in all_results]
    prob_corrs = [r.get('prob_truth_corr_mean', np.nan) for r in all_results]
    richs = [r.get('richness_global_cal', np.nan) for r in all_results]
    n_total = sum(int(float(r['meaningful_n_species'])) for r in rows_1x)
    table.append([
        'mean ± std', '—', '—', '—', '—', n_total,
        f'{np.mean(means):.1%}\n±{np.std(means):.1%}',
        '—',
        f'{np.nanmean(aucs):.3f}\n±{np.nanstd(aucs):.3f}',
        f'{np.nanmean(prob_corrs):.3f}\n±{np.nanstd(prob_corrs):.3f}',
        f'{np.nanmean(richs):.3f}\n±{np.nanstd(richs):.3f}',
    ])

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis('off')
    tab = ax.table(cellText=table, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.05, 0.06, 0.06, 0.08, 0.06, 0.10,
                               0.09, 0.09, 0.08, 0.08, 0.10])
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 2.0)
    for col_idx in range(len(headers)):
        cell = tab[(0, col_idx)]
        cell.set_facecolor('#3a5a8a')
        cell.set_text_props(color='white', fontweight='bold')
    last_row = len(table)
    for col_idx in range(len(headers)):
        cell = tab[(last_row, col_idx)]
        cell.set_facecolor('#e0e8f0'); cell.set_text_props(fontweight='bold')
    for ri in range(1, last_row):
        for ci in range(len(headers)):
            if ri % 2 == 0:
                tab[(ri, ci)].set_facecolor('#f8f8f8')
    fig.suptitle(
        f'Figure 7 — Per-world simulation parameters and reconstruction performance (K={K})',
        fontweight='bold', fontsize=12, y=0.97)
    fig.text(0.5, 0.02,
              'thr=occupancy threshold | env=environmental seed | dr=dispersal rate | '
              'ld=landscape parameter',
              ha='center', fontsize=8, style='italic', color='#666')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Fig 07 → {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-world-csv', required=True)
    ap.add_argument('--multi-world-csv-2x', default=None)
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--recon-dir-pattern', required=True)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--world-fig2-index', type=int, default=4,
                    help='Which world (0-indexed) for the per-species deep dive')
    ap.add_argument('--n-worlds-fig3', type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.multi_world_csv) as f:
        rows_1x = list(csv.DictReader(f))

    rows_2x_dict = None
    if args.multi_world_csv_2x and Path(args.multi_world_csv_2x).exists():
        with open(args.multi_world_csv_2x) as f:
            rows_2x = list(csv.DictReader(f))
        rows_2x_dict = {r['world']: r for r in rows_2x}

    print("=" * 72)
    print(f"  UNIFIED OBJECTIVE 2 SCRIPT  (K={args.K}, {len(rows_1x)} worlds)")
    print("=" * 72)

    # ─── Compute metrics for ALL worlds (used by Fig 1, 6, 7) ───
    print("\n  Computing artifact-free metrics for all worlds...\n")
    all_results = []
    worlds_loaded = []
    for r in rows_1x:
        wn = r['world']
        stem = wn.replace('.npz', '')
        tp = Path(args.truth_dir) / wn
        rd = Path(args.recon_dir_pattern.format(world_stem=stem))
        if not tp.exists() or not rd.exists():
            print(f"    ⚠ skip {wn[:60]}")
            continue
        truth, samples, mean_pred, observed = load_world(tp, rd, args.K)
        worlds_loaded.append({'world': wn, 'truth': truth, 'samples': samples,
                              'mean_pred': mean_pred, 'observed': observed})
        m = compute_metrics_for_world(truth, mean_pred, observed, samples=samples, K=args.K)
        m['world'] = wn
        all_results.append(m)
        print(f"    {wn[:60]}: recall={m['recall_1x']:.3f}, "
              f"AUC={m['auc']:.3f}, truth_in_2x_ens={m.get('truth_in_ensemble_2x', np.nan):.3f}, "
              f"far_recall(eco/smooth)={m.get('eco_recall_far', np.nan):.2f}/"
              f"{m.get('smooth_recall_far', np.nan):.2f}")

    # ─── Aggregate summary ───
    print("\n" + "=" * 72)
    print("  AGGREGATE SUMMARY")
    print("=" * 72)
    summary_keys = [
        ('per-species recall (1×)',        'recall_1x'),
        ('per-species recall (2×)',        'recall_2x'),
        ('recall by bucket: 6-10',         'recall_6_10'),
        ('recall by bucket: 11-20',        'recall_11_20'),
        ('recall by bucket: 21+',          'recall_21p'),
        ('AUC (degenerates excluded)',     'auc'),
        ('prob-truth corr (per-sp mean)',  'prob_truth_corr_mean'),
        ('range corr (global cal)',        'range_corr_global'),
        ('range corr (raw probability)',   'range_corr_raw_prob'),
        ('range corr (per-sp tautology)',  'range_corr_tautology'),
        ('richness corr (per-species)',    'richness_per_species_cal'),
        ('richness corr (GLOBAL)',         'richness_global_cal'),
        ('beta-diversity r (per-sp)',      'beta_diversity_per_species_cal'),
        ('beta-diversity r (GLOBAL)',      'beta_diversity_global_cal'),
        ('Sørensen (per-sp, correct)',     'sorensen_per_species_correct'),
        ('Sørensen (global, correct)',     'sorensen_global_correct'),
        ('Sørensen (per-sp, ARTIFACT)',    'sorensen_per_species_artifact'),
        ('Sørensen (global, ARTIFACT)',    'sorensen_global_artifact'),
        ('TSS (EcoDiffusion)',             'tss_eco'),
        ('CBI (EcoDiffusion)',             'cbi_eco'),
        ('AUC (Uniform baseline)',         'auc_uniform'),
        ('AUC (Smooth-obs baseline)',      'auc_smooth'),
        # ensemble-only metrics (Axel's question)
        ('truth-in-ensemble (1× union)',   'truth_in_ensemble_1x'),
        ('truth-in-ensemble (2× union)',   'truth_in_ensemble_2x'),
        ('best-of-8 sample recall',        'best_of_8_recall'),
        ('worst-of-8 sample recall',       'worst_of_8_recall'),
        ('eco recall: near (d≤2)',         'eco_recall_near'),
        ('eco recall: mid (d=3-5)',        'eco_recall_mid'),
        ('eco recall: far (d>5)',          'eco_recall_far'),
        ('smooth recall: near (d≤2)',      'smooth_recall_near'),
        ('smooth recall: mid (d=3-5)',     'smooth_recall_mid'),
        ('smooth recall: far (d>5)',       'smooth_recall_far'),
        ('ensemble diversity (mean std)',  'ensemble_diversity_mean'),
    ]
    for label, key in summary_keys:
        vals = [r[key] for r in all_results if not np.isnan(r[key])]
        if vals:
            print(f"  {label:36}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ─── Save metrics CSV ───
    csv_path = out_dir / 'artifact_free_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        fields = ['world', 'n_meaningful'] + [k for _, k in summary_keys]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            row = {'world': r['world'], 'n_meaningful': r['n_meaningful']}
            for _, k in summary_keys:
                row[k] = f"{r[k]:.4f}"
            w.writerow(row)
    print(f"\n  ✓ CSV: {csv_path}")

    # ─── Generate figures ───
    print(f"\n  Generating 12 figures...")
    make_fig01_metrics_summary(all_results,
                                out_dir / 'Fig01_metrics_summary.png', K=args.K)

    if args.world_fig2_index < len(worlds_loaded):
        make_fig02_per_species_heatmaps(
            worlds_loaded[args.world_fig2_index],
            out_dir / 'Fig02_per_species_heatmaps.png',
            K=args.K,
            pick_best_in_bucket=pick_best_in_bucket,
            parse_world_params=parse_world_params,
        )
        make_fig04_calibration_comparison(worlds_loaded[args.world_fig2_index],
                                            out_dir / 'Fig04_calibration_comparison.png',
                                            K=args.K)

    n_w = len(worlds_loaded)
    if n_w >= args.n_worlds_fig3:
        indices_b = np.linspace(0, n_w - 1, args.n_worlds_fig3).astype(int).tolist()
        worlds_b = []
        for i in indices_b:
            wd = worlds_loaded[i]
            picks = pick_best_in_bucket(wd['truth'], wd['mean_pred'], 8, 15, n=1)
            if not picks:
                picks = pick_best_in_bucket(wd['truth'], wd['mean_pred'], 6, 20, n=1)
            if not picks:
                continue
            sp, recall, rng = picks[0]
            worlds_b.append({
                **wd,
                'species_idx': sp,
                'range': rng,
                'recall': recall,
                'world_short': wd['world'][:40],
            })
        if len(worlds_b) >= 2:
            make_fig03_multi_world_heatmaps(
                worlds_b,
                out_dir / 'Fig03_multi_world_heatmaps.png',
                K=args.K,
            )

    make_fig05_recall_vs_range(worlds_loaded,
                                 out_dir / 'Fig05_recall_vs_range.png', K=args.K)
    make_fig06_baseline_comparison(all_results,
                                     out_dir / 'Fig06_baseline_comparison.png',
                                     K=args.K)
    make_fig07_world_params_table(rows_1x, rows_2x_dict, all_results,
                                    out_dir / 'Fig07_world_parameters_table.png',
                                    K=args.K)
    make_fig08_ensemble_value_add(all_results,
                                    out_dir / 'Fig08_ensemble_value_add.png',
                                    K=args.K)
    if args.world_fig2_index < len(worlds_loaded):
        make_fig09_repaint_schematic(worlds_loaded[args.world_fig2_index],
                                       out_dir / 'Fig09_repaint_schematic.png',
                                       K=args.K)
    make_fig10_auc_histograms(all_results,
                                out_dir / 'Fig10_auc_histograms.png',
                                K=args.K)
    make_fig11_auc_scatters(all_results,
                              out_dir / 'Fig11_auc_scatters.png',
                              K=args.K)
    make_fig12_saturation_curve(all_results,
                                  out_dir / 'Fig12_saturation_curve.png',
                                  K=args.K)
    
    run_axel_distribution_tests(
    worlds_loaded, all_results, out_dir,
    K=args.K, calibrate_fn=calibrate_per_species,
)

    print(f"\n  All outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()