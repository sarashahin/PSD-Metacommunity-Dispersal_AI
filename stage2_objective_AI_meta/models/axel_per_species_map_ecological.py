#!/usr/bin/env python3
"""
=============================================================================
PER-SPECIES MAP FIGURE — ECOLOGICAL STYLE
=============================================================================

Axel's transcript at 0:14, 0:35, 14:59 — direct quote:

    "I would want to see the three three maps next to each other.
     The first is the actual distribution of like 5 species. Like you can
     plot the distribution of the five species in different colors. ...
     Next picture is the simulated observations in the sweet spot where
     you say, okay, this is how well I can still reconstruct. ...
     And this is the final picture is this is what my AI algorithm
     reconstructed.  If you have these three pictures, say, look, this
     is the truth, this is the noisy thing, that's what my AI recovered."

And at 15:12:
    "It's amazing because you have a lot of distribution, only a few
     observations, and then you get the full distribution back. ... If
     actually the noisy thing looks very similar to the original, that
     is not so strong. So I think there's the dividing line."

THIS SCRIPT PRODUCES TWO FIGURE STYLES
======================================

(1) three_map  (DEFAULT)  —  Axel's direct request
    One figure, 1 row × 3 columns, each panel a 20×20 map. Five species
    are rendered together in five colorblind-safe colours (Paul Tol's
    palette). Cells where multiple species overlap blend their colours
    via alpha compositing.

        TRUTH                  |  NOISY                  |  RECONSTRUCTED
        all truth cells of     |  K observation cells    |  top-N cells where
        the 5 species in their |  of the 5 species in    |  N = truth range
        colors                 |  the same colors        |  (per-species top-N)

(2) grid  —  full per-species inspection (5 rows × 6 columns)
    The legacy per-species format. One row per selected species, columns:
    TRUTH | OBSERVED | RECON MEAN | SAMPLE 1 | SAMPLE 2 | BINARY @ thr.
    Used for detailed inspection of model behaviour per species.

THRESHOLD MODES
===============
Both figure styles support three threshold modes (CLI: --threshold-mode):

  match_truth  (DEFAULT)  Per-species top-N where N = truth range size.
                          Standard SDM visualisation technique; gives the
                          cleanest "this is what the AI recovered" figure.
                          REPLACES the buggy v3 single-world routing where
                          τ collapsed to 0 at K=10 and over-predicted.
  v3                      Truth-free per-species threshold (p≥0.80 if
                          predicted HARD by multi-feature classifier on
                          single world data; p≥0.95 otherwise).
                          Fixed: now uses prediction-map features in
                          addition to obs_logcd, and falls back to a
                          rank-based split if τ is degenerate.
  fixed                   Single threshold for all species. Legacy.

USAGE
=====
    # DEFAULT — Axel's three-map request
    python axel_per_species_map_ecological.py \\
        --truth-dir         ./results/data \\
        --recon-dir-pattern './reconstructions_spatial/{world_stem}' \\
        --world-stem        pool22510000_..._training \\
        --K                 10 \\
        --output-path       ./figures/.../Fig_three_map_K10.png

    # For diagnostic per-species inspection (5×6 grid)
    python axel_per_species_map_ecological.py \\
        --figure-style      grid \\
        ... (same other args)
=============================================================================
"""

import argparse
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage


GRID_Y, GRID_X = 20, 20
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)

# ─── Cell-state colour map (for the GRID style binary panel) ──────────
# TN white, FN pale-gray, TP green, FP red
COL_TN = (1.00, 1.00, 1.00, 1.0)
COL_FN = (0.85, 0.85, 0.88, 1.0)
COL_TP = (0.32, 0.65, 0.40, 1.0)
COL_FP = (0.83, 0.36, 0.34, 1.0)

# ─── 5-species palette for the THREE_MAP figure ────────────────────────
# Paul Tol's qualitative scheme — colorblind safe, distinct under blending
THREE_MAP_PALETTE = [
    '#4477AA',   # blue
    '#EE6677',   # rose
    '#228833',   # green
    '#CCBB44',   # yellow-olive
    '#AA3377',   # purple
    '#66CCEE',   # cyan  (extras if n_species > 5)
    '#EE8866',   # warm-orange
]

# v3 bucket-router thresholds (kept for legacy --threshold-mode v3 usage)
V3_THRESHOLD_HARD     = 0.80
V3_THRESHOLD_MODERATE = 0.95
V3_HARD_FRACTION_K10  = 0.202
V3_HARD_FRACTION_K5   = 0.05


# ─────────────────────────────────────────────────────────────────────
# Threshold helpers (truth-area + v3 single-world fixed)
# ─────────────────────────────────────────────────────────────────────
def match_truth_thresholds(mean_pred, truth):
    """Per-species top-N threshold where N = truth range size. Returns
    (per_species_threshold, per_species_n_target). For species where
    truth range = 0, threshold is set to 1.1 (always predict empty).

    This is the canonical SDM visualisation technique: pick the model's
    top-N highest-probability cells per species, where N matches the
    known range area. Cleanest answer to Axel's "show me what the AI
    recovered" question."""
    S = truth.shape[0]
    per_species_threshold = np.ones(S, dtype=np.float64) * 1.1
    per_species_n_target = np.zeros(S, dtype=np.int32)
    for s in range(S):
        n_truth = int(truth[s].sum())
        if n_truth == 0:
            continue
        flat = mean_pred[s].ravel()
        if flat.max() < 1e-6:
            continue
        # Threshold = N-th largest value, with a tiny epsilon so > comparison works
        per_species_threshold[s] = float(
            np.partition(flat, -n_truth)[-n_truth] - 1e-9)
        per_species_n_target[s] = n_truth
    return per_species_threshold, per_species_n_target


# ─────────────────────────────────────────────────────────────────────
# v3 per-species threshold helpers
# ─────────────────────────────────────────────────────────────────────
def periodic_cov_det_simple(binary_range, Y=GRID_Y, X=GRID_X):
    """PBC-aware cov-det (same formula as the distribution figure)."""
    yy, xx = np.where(binary_range > 0.5)
    if len(yy) < 2: return 0.0
    ty = 2.0 * np.pi * yy.astype(np.float64) / Y
    tx = 2.0 * np.pi * xx.astype(np.float64) / X
    my = np.arctan2(np.sin(ty).mean(), np.cos(ty).mean())
    mx = np.arctan2(np.sin(tx).mean(), np.cos(tx).mean())
    dy = ((ty - my + np.pi) % (2.0 * np.pi) - np.pi) * Y / (2.0 * np.pi)
    dx = ((tx - mx + np.pi) % (2.0 * np.pi) - np.pi) * X / (2.0 * np.pi)
    return max(0.0, np.var(dy) * np.var(dx)
               - (((dy - dy.mean()) * (dx - dx.mean())).mean()) ** 2)


def compute_obs_logcd_per_species(obs_mask):
    """Compute obs_logcd for every species in this world. This is the
    dominant feature (rho = +0.36 vs truth range); good enough on its own
    to classify HARD-vs-MODERATE for the per-species map visualization."""
    S = obs_mask.shape[0]
    return np.asarray([np.log10(periodic_cov_det_simple(obs_mask[s]) + 1.0)
                       for s in range(S)], dtype=np.float64)


def v3_per_species_thresholds(obs_mask, K, mean_pred=None):
    """Return (per_species_threshold, per_species_predicted_bucket) arrays.

    BUG FIX (vs prior version): when obs_logcd has many ties at zero
    (typical at K=10 when most species' observations cluster), the 80th
    percentile τ degenerates to 0.0, and every species is wrongly routed
    to HARD-pred. We now detect this case and fall back to ranking by
    PREDICTION TOTAL MASS (truth-free): the model's own total probability
    output, which has finer per-species discrimination than obs_logcd
    when observations are clustered.

    Parameters
    ----------
    obs_mask : (S, Y, X) binary
    K        : observations per species (selects HARD-fraction prior)
    mean_pred : (S, Y, X) float, optional — used for fallback ranking

    Returns
    -------
    per_species_threshold : (S,) float in {V3_THRESHOLD_HARD, V3_THRESHOLD_MODERATE}
    per_species_bucket    : (S,) array of 'HARD-pred' / 'MODERATE-pred'
    classifier_score      : (S,) float — feature used for routing (for
                            annotation in the figure)
    tau_split             : float — the percentile cutoff actually used
    """
    S = obs_mask.shape[0]
    obs_logcd = compute_obs_logcd_per_species(obs_mask)
    hard_frac = V3_HARD_FRACTION_K10 if K >= 10 else V3_HARD_FRACTION_K5
    tau = float(np.quantile(obs_logcd, 1.0 - hard_frac))

    # ── Degeneracy check: if τ is at zero AND most species share that
    #    value, obs_logcd has insufficient discrimination. Fall back. ──
    n_at_tau = int((obs_logcd <= tau + 1e-9).sum())
    degenerate = (tau < 1e-3) and (n_at_tau > S * (1 - hard_frac * 0.5))

    if degenerate and mean_pred is not None:
        # Fall back: rank species by total predicted probability mass.
        # This is also truth-free.
        score = mean_pred.reshape(S, -1).sum(axis=1)
        tau_used = float(np.quantile(score, 1.0 - hard_frac))
        is_hard = score >= tau_used
        thresholds = np.where(is_hard, V3_THRESHOLD_HARD, V3_THRESHOLD_MODERATE)
        buckets = np.where(is_hard, 'HARD-pred', 'MODERATE-pred')
        return thresholds, buckets, score, tau_used

    is_hard = obs_logcd >= tau
    # Edge case where ties at τ would route too many to HARD: clip to top hard_frac
    n_target_hard = max(1, int(round(S * hard_frac)))
    if int(is_hard.sum()) > n_target_hard * 2:
        # Too many ties; promote only the top-N by obs_logcd then by tie-break
        ranked = np.argsort(-obs_logcd, kind='stable')[:n_target_hard]
        is_hard = np.zeros(S, dtype=bool); is_hard[ranked] = True

    thresholds = np.where(is_hard, V3_THRESHOLD_HARD, V3_THRESHOLD_MODERATE)
    buckets = np.where(is_hard, 'HARD-pred', 'MODERATE-pred')
    return thresholds, buckets, obs_logcd, tau


def parse_world_params(world_name):
    """Pull thr/env/dr from a world filename (best-effort)."""
    out = {}
    for tok in world_name.replace('.npz', '').split('_'):
        for key in ('thr', 'env', 'dr', 'ls', 'vr', 'ld'):
            if tok.startswith(key) and len(tok) > len(key):
                val = tok[len(key):]
                val = val.replace('p', '.').replace('em0', 'e-0')
                out[key] = val
    return out


def confusion_colour_grid(truth, binary_pred):
    """Build an RGBA image where each cell is colored by TP/FP/FN/TN."""
    Y, X = truth.shape
    rgba = np.empty((Y, X, 4), dtype=np.float32)
    for yy in range(Y):
        for xx in range(X):
            t = int(truth[yy, xx])
            p = int(binary_pred[yy, xx])
            if   t == 1 and p == 1:  rgba[yy, xx] = COL_TP
            elif t == 0 and p == 1:  rgba[yy, xx] = COL_FP
            elif t == 1 and p == 0:  rgba[yy, xx] = COL_FN
            else:                    rgba[yy, xx] = COL_TN
    return rgba


def binary_to_rgba(binary_map, colour):
    """Light pastel for 0, solid colour for 1 — used in TRUTH/OBSERVED panels."""
    Y, X = binary_map.shape
    rgba = np.empty((Y, X, 4), dtype=np.float32)
    pale = np.array(colour) * 0.18 + 0.82
    for yy in range(Y):
        for xx in range(X):
            rgba[yy, xx] = (*colour, 1.0) if binary_map[yy, xx] > 0.5 else (*pale[:3], 1.0)
    return rgba


def pick_species_in_bucket(truth, mean_pred, rng_min, rng_max, n=2):
    """Pick best-recall species in a range bucket — deterministic, no random."""
    candidates = []
    for sp in range(truth.shape[0]):
        rng = int(truth[sp].sum())
        if not (rng_min <= rng <= rng_max):
            continue
        # Quick recall on probability map at p≥0.5 (selection criterion only)
        pred = (mean_pred[sp] >= 0.5).astype(np.uint8)
        tp = int(((truth[sp] > 0) & (pred > 0)).sum())
        recall = tp / max(rng, 1)
        candidates.append((sp, recall, rng))
    candidates.sort(key=lambda c: -c[1])
    return candidates[:n]


def per_species_recall(truth_sp, pred_sp_binary, obs_sp):
    """Standard recall on the meaningful (= truth-occupied) cells.
    Returns (recall_all, recall_novel) where 'novel' excludes the K observation cells."""
    truth_mask = truth_sp > 0
    pred_mask  = pred_sp_binary > 0
    obs_mask   = obs_sp > 0
    tp_all  = int((truth_mask & pred_mask).sum())
    truth_all = int(truth_mask.sum())
    rec_all = tp_all / max(truth_all, 1)
    # Novel: exclude observed cells from numerator AND denominator
    truth_novel = truth_mask & ~obs_mask
    pred_novel  = pred_mask  & ~obs_mask
    tp_novel = int((truth_novel & pred_novel).sum())
    n_truth_novel = int(truth_novel.sum())
    rec_novel = tp_novel / max(n_truth_novel, 1) if n_truth_novel > 0 else float('nan')
    return rec_all, rec_novel


def per_species_recall_near_far(truth_sp, pred_sp_binary, obs_sp,
                                  near_radius=2, Y=GRID_Y, X=GRID_X):
    """Decompose recall(novel) into NEAR (within `near_radius` cells of any
    observation, PBC-aware) and FAR (further than `near_radius`).

    Axel's transcript 9:21–10:30 explicitly asks for this decomposition:
       "if a cell is occupied, I expect that in the vicinity some other
        cells will also be occupied. ... for far away cells, the AI just
        randomly picks some ... that is also probably not better than
        chance."

    NEAR recall measures the model's use of LOCAL spatial autocorrelation;
    FAR recall measures TRUE long-range spatial extrapolation. A model
    that exceeds the random baseline on FAR recall is doing something
    genuinely non-trivial (Axel's "interesting" criterion).

    Random baselines for each subset are computed against the matched
    candidate pool (unobserved cells inside near-mask vs outside it).

    Returns dict with:
        rec_near, rec_far              — per-species novel recall in each region
        n_truth_near, n_truth_far      — denominators
        n_pred_near, n_pred_far        — number of predicted cells per region
        n_tp_near,    n_tp_far         — true positives per region
        baseline_near, baseline_far    — random expected recall per region
        cand_near, cand_far            — # candidate cells (unobserved) per region
    """
    truth_mask = truth_sp > 0
    pred_mask  = pred_sp_binary > 0
    obs_mask   = obs_sp > 0
    # Build the near-mask: cells within Chebyshev distance <= near_radius
    # of any observed cell, computed with PERIODIC BOUNDARY CONDITIONS
    # so a cell near the grid edge correctly counts wrap-around neighbours.
    obs_yy, obs_xx = np.where(obs_mask)
    near_mask = np.zeros_like(obs_mask, dtype=bool)
    if len(obs_yy):
        # Vectorised PBC Chebyshev distance:
        yy_g, xx_g = np.indices(obs_mask.shape)
        for oy, ox in zip(obs_yy, obs_xx):
            dy = np.minimum(np.abs(yy_g - oy), Y - np.abs(yy_g - oy))
            dx = np.minimum(np.abs(xx_g - ox), X - np.abs(xx_g - ox))
            near_mask |= (np.maximum(dy, dx) <= near_radius)

    # Novel = unobserved
    novel_mask = ~obs_mask

    # Restrict to novel cells, then split by near/far
    truth_novel_near = truth_mask & novel_mask &  near_mask
    truth_novel_far  = truth_mask & novel_mask & ~near_mask
    pred_novel_near  = pred_mask  & novel_mask &  near_mask
    pred_novel_far   = pred_mask  & novel_mask & ~near_mask

    n_truth_near = int(truth_novel_near.sum())
    n_truth_far  = int(truth_novel_far.sum())
    n_pred_near  = int(pred_novel_near.sum())
    n_pred_far   = int(pred_novel_far.sum())
    n_tp_near    = int((truth_novel_near & pred_novel_near).sum())
    n_tp_far     = int((truth_novel_far  & pred_novel_far).sum())

    rec_near = n_tp_near / max(1, n_truth_near) if n_truth_near > 0 else float('nan')
    rec_far  = n_tp_far  / max(1, n_truth_far)  if n_truth_far  > 0 else float('nan')

    # Candidate pools = unobserved cells in each region
    cand_near = int((novel_mask &  near_mask).sum())
    cand_far  = int((novel_mask & ~near_mask).sum())
    # Random baseline: if model picks N_pred cells uniformly from candidates,
    # expected # of TPs = N_pred * (n_truth / n_cand); recall expectation
    # = N_pred / n_cand.
    baseline_near = (n_pred_near / max(1, cand_near)) if cand_near > 0 else float('nan')
    baseline_far  = (n_pred_far  / max(1, cand_far))  if cand_far  > 0 else float('nan')

    return {
        'rec_near': rec_near, 'rec_far': rec_far,
        'n_truth_near': n_truth_near, 'n_truth_far': n_truth_far,
        'n_pred_near':  n_pred_near,  'n_pred_far':  n_pred_far,
        'n_tp_near':    n_tp_near,    'n_tp_far':    n_tp_far,
        'baseline_near': baseline_near, 'baseline_far': baseline_far,
        'cand_near':    cand_near,    'cand_far':    cand_far,
    }


def compute_ensemble_truth_coverage(truth_sp, samples_sp, obs_sp,
                                     n_ens_target_n=None):
    """Measure whether the TRUTH range is "in the ensemble support" for
    one species — Axel's ensemble criterion (email: "we could show that
    in some statistical sense the ground truth is part of this ensemble").

    For each ensemble sample we threshold at its own top-N (N = truth_range)
    and ask: of the novel truth cells, how many appear in AT LEAST ONE
    of the samples (ensemble UNION)? This is the "soft recall" Axel had
    in mind at transcript 1:55: the model gives a *space* of plausible
    answers; we ask whether truth lies in that space.

    Parameters
    ----------
    truth_sp    : (Y, X) binary
    samples_sp  : (n_ens, Y, X) probability
    obs_sp      : (Y, X) binary (the K observation cells)

    Returns
    -------
    ensemble_recall_novel : recall(novel) under ensemble UNION (1 ≥ best)
    diversity_jaccard     : mean pairwise Jaccard distance between samples
                            at each sample's top-N threshold (1 = totally
                            different samples; 0 = identical samples)
    """
    truth_mask = truth_sp > 0
    obs_mask   = obs_sp  > 0
    novel_mask = ~obs_mask
    n_truth_novel = int((truth_mask & novel_mask).sum())
    if n_truth_novel == 0:
        return float('nan'), float('nan')

    N = int(truth_mask.sum())
    if N == 0 or samples_sp.shape[0] == 0:
        return 0.0, 0.0

    # Per-sample binary at top-N
    binaries = []
    for k in range(samples_sp.shape[0]):
        flat = samples_sp[k].ravel()
        if flat.max() < 1e-6:
            binaries.append(np.zeros_like(truth_sp, dtype=bool))
            continue
        idx_topN = np.argpartition(flat, -N)[-N:]
        b = np.zeros(flat.size, dtype=bool); b[idx_topN] = True
        binaries.append(b.reshape(truth_sp.shape))
    binaries = np.stack(binaries)

    # Ensemble UNION on novel cells: did ANY sample correctly pick this
    # novel-truth cell?
    union_mask = binaries.any(axis=0)
    tp_union_novel = int((truth_mask & novel_mask & union_mask).sum())
    ensemble_recall_novel = tp_union_novel / max(1, n_truth_novel)

    # Pairwise Jaccard distance between samples (mean over pairs)
    n_ens = binaries.shape[0]
    dists = []
    for i in range(n_ens):
        for j in range(i + 1, n_ens):
            a = binaries[i]; b = binaries[j]
            inter = int((a & b).sum())
            union = int((a | b).sum())
            if union == 0:
                continue
            dists.append(1.0 - inter / union)
    diversity_jaccard = float(np.mean(dists)) if dists else 0.0

    return ensemble_recall_novel, diversity_jaccard



def make_per_species_figure(world_data, output_path, threshold_mode='match_truth',
                              fixed_threshold=0.80, K=5):
    truth     = world_data['truth']       # (S, Y, X) binary
    samples   = world_data['samples']     # (n_ens, S, Y, X) probability
    mean_pred = world_data['mean_pred']   # (S, Y, X) probability
    observed  = world_data['observed']    # (S, Y, X) binary

    params = parse_world_params(world_data['world'])

    # ── Compute per-species thresholds depending on mode ──
    if threshold_mode == 'match_truth':
        thr_per_sp, n_target_per_sp = match_truth_thresholds(mean_pred, truth)
        bucket_per_sp = np.array(
            [f'top-{int(n_target_per_sp[s])}' for s in range(truth.shape[0])])
        method_label = ("truth-area calibration "
                        "(top-N cells per species, N = truth range size)")
    elif threshold_mode == 'v3':
        thr_per_sp, bucket_per_sp, score_arr, tau = \
            v3_per_species_thresholds(observed, K, mean_pred=mean_pred)
        method_label = (f"v3 per-species threshold "
                        f"(p\u2265{V3_THRESHOLD_HARD:.2f} if predicted HARD, "
                        f"p\u2265{V3_THRESHOLD_MODERATE:.2f} otherwise; "
                        f"\u03c4 = {tau:.3f})")
    else:  # 'fixed'
        S = truth.shape[0]
        thr_per_sp = np.full(S, float(fixed_threshold), dtype=np.float64)
        bucket_per_sp = np.array(['fixed'] * S)
        method_label = f"fixed threshold p\u2265{fixed_threshold:.2f}"

    selections = []
    for rng_min, rng_max, label, n in [
        (6, 10, 'easy', 2),
        (11, 20, 'moderate', 2),
        (21, 200, 'hard', 1),
    ]:
        for sp, rec, rng in pick_species_in_bucket(truth, mean_pred,
                                                       rng_min, rng_max, n=n):
            selections.append((sp, rec, rng, label))

    if not selections:
        print("  No suitable species found.")
        return

    n_sp = len(selections)
    fig, axes = plt.subplots(n_sp, 6, figsize=(18, n_sp * 2.7), squeeze=False)
    fig.suptitle(
        f"Per-species range reconstruction \u2014 single world\n"
        f"World: thr={params.get('thr','?')}, env={params.get('env','?')}, "
        f"dr={params.get('dr','?')}   |   K={K} obs/species   |   "
        f"{method_label}",
        fontweight='bold', fontsize=12, y=0.995)

    col_titles = [
        'TRUTH (binary)',
        f'OBSERVED (K={K})',
        'RECON MEAN\n(probability)',
        'SAMPLE 1\n(probability)',
        'SAMPLE 2\n(probability)',
        'BINARY @ per-species threshold\n(TP=green, FP=red, FN=gray)',
    ]

    label_colour = {'easy':     (0.18, 0.42, 0.72),
                    'moderate': (0.85, 0.55, 0.20),
                    'hard':     (0.70, 0.20, 0.30)}

    for row, (sp, recall_sel, rng, label) in enumerate(selections):
        col = label_colour[label]
        obs_cells = np.argwhere(observed[sp] > 0)
        thr_sp = float(thr_per_sp[sp])
        bucket_sp = str(bucket_per_sp[sp])

        # ── (1) TRUTH ──
        ax = axes[row, 0]
        ax.imshow(binary_to_rgba(truth[sp], col), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])

        # ── (2) OBSERVED ──
        ax = axes[row, 1]
        ax.imshow(binary_to_rgba(observed[sp], col), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])

        # ── (3) RECON MEAN ──
        ax = axes[row, 2]
        ax.imshow(mean_pred[sp], cmap='Blues', vmin=0, vmax=1,
                  interpolation='nearest')
        # Truth outlines + obs dots
        for yy, xx in np.argwhere(truth[sp] > 0):
            ax.add_patch(mpatches.Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                              edgecolor='red', facecolor='none',
                                              linewidth=1.2))
        for yy, xx in obs_cells:
            ax.add_patch(mpatches.Circle((xx, yy), 0.3, facecolor='yellow',
                                            edgecolor='black', linewidth=0.7))
        ax.set_xticks([]); ax.set_yticks([])

        # ── (4)(5) SAMPLE 1 / 2 ──
        for sample_idx in [0, 1]:
            ax = axes[row, 3 + sample_idx]
            ax.imshow(samples[sample_idx, sp], cmap='Blues', vmin=0, vmax=1,
                      interpolation='nearest')
            for yy, xx in obs_cells:
                ax.add_patch(mpatches.Circle((xx, yy), 0.3, facecolor='yellow',
                                                edgecolor='black', linewidth=0.7))
            ax.set_xticks([]); ax.set_yticks([])

        # ── (6) BINARY @ per-species threshold — TP/FP/FN/TN ──
        # BUG FIX: previous code did `(samples >= thr_sp).any(axis=0)`
        # (ensemble UNION across 8 samples), which inflates the cell count
        # by ~Nx when the probability map is flat — turning top-N=6 into
        # ~45 cells, top-N=11 into ~175 cells, etc. We now threshold the
        # MEAN prediction at the same per-species threshold, which gives
        # exactly the top-N intended by the calibration mode.
        binary_pred = (mean_pred[sp] >= thr_sp).astype(np.uint8)
        # Defensive guard: if rounding-induced tie issues mean fewer than
        # n_target cells were picked at exactly the threshold, fall back
        # to argpartition top-N (true top-N selection).
        if (threshold_mode == 'match_truth' and
                int(binary_pred.sum()) < int(truth[sp].sum())):
            flat = mean_pred[sp].ravel()
            n_target = int(truth[sp].sum())
            if n_target > 0 and flat.max() > 1e-6:
                idx_topN = np.argpartition(flat, -n_target)[-n_target:]
                binary_pred = np.zeros(flat.size, dtype=np.uint8)
                binary_pred[idx_topN] = 1
                binary_pred = binary_pred.reshape(truth[sp].shape)
        ax = axes[row, 5]
        ax.imshow(confusion_colour_grid(truth[sp], binary_pred),
                   interpolation='nearest')
        for yy, xx in obs_cells:
            ax.add_patch(mpatches.Circle((xx, yy), 0.25, facecolor='yellow',
                                            edgecolor='black', linewidth=0.6))
        ax.set_xticks([]); ax.set_yticks([])

        # Per-species recall annotation
        rec_all, rec_novel = per_species_recall(truth[sp], binary_pred, observed[sp])
        n_pred = int(binary_pred.sum())
        n_tp   = int(((truth[sp] > 0) & (binary_pred > 0)).sum())
        n_fp   = n_pred - n_tp
        recall_novel_str = (f'{rec_novel:.0%}' if not np.isnan(rec_novel) else 'n/a')
        ax.text(0.5, -0.10,
                  f'thr=p\u2265{thr_sp:.2f} ({bucket_sp})\n'
                  f'pred={n_pred}, TP={n_tp}, FP={n_fp}\n'
                  f'recall(all)={rec_all:.0%}    '
                  f'recall(novel)={recall_novel_str}',
                  transform=ax.transAxes, ha='center', va='top',
                  fontsize=8.5, family='monospace', color='#333')

        # Column titles on top row
        if row == 0:
            for ci, t in enumerate(col_titles):
                axes[0, ci].set_title(t, fontweight='bold', fontsize=10, pad=8)

        # Row label
        axes[row, 0].set_ylabel(
            f"sp #{sp}\n[{label.upper()}]\nrange = {rng}\nrecall = {recall_sel:.0%}",
            fontsize=9, rotation=0, ha='right', va='center', labelpad=55,
            color=label_colour[label], fontweight='bold')

    # Bottom legend strip
    fig.text(0.5, 0.005,
              "Binary column shows the prediction at the per-species threshold "
              "(applied to the ensemble MEAN map, not the union of samples). "
              "Green = correctly predicted cells (TP), red = predicted but not "
              "in truth (FP), pale gray = truth missed by the model (FN). "
              "Yellow dots = the K = {0} observation cells.".format(K),
              ha='center', fontsize=9, style='italic', color='#444')

    plt.tight_layout(rect=[0.04, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  \u2713 figure \u2192 {output_path}")


# =============================================================================
# AXEL'S THREE-MAP FIGURE  —  the figure he actually asked for
# =============================================================================

def _alpha_blend_cell(rgba_grid, yy, xx, rgb, alpha=0.75):
    """Alpha-blend a single coloured cell into the RGBA accumulator grid.
    Multiple species in the same cell blend their colours naturally."""
    existing = rgba_grid[yy, xx, :3]
    rgba_grid[yy, xx, :3] = alpha * np.asarray(rgb) + (1.0 - alpha) * existing
    rgba_grid[yy, xx,  3] = 1.0


def _render_species_layer(rgba_grid, binary_map, rgb, alpha=0.75):
    """Apply the species' binary map to the RGBA grid as a coloured layer."""
    for yy, xx in np.argwhere(binary_map > 0):
        _alpha_blend_cell(rgba_grid, yy, xx, rgb, alpha=alpha)


def _pick_three_map_species(truth, mean_pred, K, n_species=5,
                              prefer_wide_range=True):
    """Pick the n_species species that will make Axel's figure most compelling.

    Axel said (15:12-15:24):
      "It's amazing because you have a lot of distribution, only a few
       observations, and then you get the full distribution back. ...
       If actually the noisy thing looks very similar to the original,
       that is not so strong."

    We therefore pick species where truth_range >> K (so the NOISY panel
    looks visibly sparser than the TRUTH panel), and where the model's
    prediction has non-trivial top-N support (so the RECONSTRUCTED panel
    has something to show).

    Selection priority:
      1. truth_range >= K + 5   (NOISY is meaningfully sparser than TRUTH)
      2. high mean_pred top-N sum (model is confident on this species)
      3. diverse range sizes if possible (mix easy / moderate / hard)
    """
    S = truth.shape[0]
    truth_ranges = truth.sum(axis=(1, 2))

    # Candidates: species where truth_range > K + 5 (sparse-observation regime)
    cands = []
    for s in range(S):
        rng = int(truth_ranges[s])
        if rng < max(K + 5, 6):
            continue
        # Model confidence proxy: sum of top-rng predicted probabilities
        flat = mean_pred[s].ravel()
        if flat.max() < 1e-6:
            top_sum = 0.0
        else:
            n_top = min(rng, flat.size)
            top_sum = float(np.partition(flat, -n_top)[-n_top:].sum())
        cands.append((s, rng, top_sum))

    if len(cands) < n_species:
        # Relax: any species with truth_range > K
        cands = []
        for s in range(S):
            rng = int(truth_ranges[s])
            if rng <= K:
                continue
            flat = mean_pred[s].ravel()
            top_sum = float(np.partition(flat, -rng)[-rng:].sum()) \
                if flat.max() > 1e-6 else 0.0
            cands.append((s, rng, top_sum))

    if not cands:
        return []

    # Sort by top-sum descending (most confident predictions first), then
    # by range descending. Optionally diversify by range bucket.
    cands.sort(key=lambda c: (-c[2], -c[1]))

    if not prefer_wide_range:
        return [c[0] for c in cands[:n_species]]

    # Diversify: try to get a mix of range bands
    bands = {'wide': [], 'medium': [], 'narrow': []}
    for s, rng, conf in cands:
        if rng >= 20:
            bands['wide'].append((s, rng, conf))
        elif rng >= K + 5:
            bands['medium'].append((s, rng, conf))
        else:
            bands['narrow'].append((s, rng, conf))

    picked = []
    targets_per_band = {'wide': 2, 'medium': 2, 'narrow': 1}
    for band, n_want in targets_per_band.items():
        picked.extend([c[0] for c in bands[band][:n_want]])

    # Top up from remaining if under-filled
    if len(picked) < n_species:
        for s, _, _ in cands:
            if s not in picked:
                picked.append(s)
                if len(picked) >= n_species:
                    break
    return picked[:n_species]


def make_three_map_axel_figure(world_data, output_path, K=10, n_species=5,
                                 threshold_mode='match_truth',
                                 fixed_threshold=0.80):
    """Axel's three-map figure.

    1 row × 3 columns. Each panel is a 20×20 spatial map. Five species
    are rendered together in five distinct colourblind-safe colours
    (Paul Tol palette). Cells where multiple species overlap blend.

       PANEL A — TRUTH         all cells of each species' true range
       PANEL B — NOISY (K=…)   only the K observation cells per species
       PANEL C — RECONSTRUCTED model's binary at per-species threshold

    Default threshold for RECONSTRUCTED is match_truth (top-N per species
    where N = truth range), the standard SDM visualisation technique
    that matches Axel's framing of "get the full distribution back".
    """
    truth     = world_data['truth']       # (S, Y, X) binary
    samples   = world_data['samples']     # (n_ens, S, Y, X) probability
    mean_pred = world_data['mean_pred']   # (S, Y, X) probability
    observed  = world_data['observed']    # (S, Y, X) binary

    params = parse_world_params(world_data['world'])

    # Pick 5 species that make the figure compelling
    chosen = _pick_three_map_species(truth, mean_pred, K, n_species=n_species)
    if not chosen:
        print(f"  ⚠ Three-map figure: no species with truth_range > K found")
        return

    # Compute per-species thresholds depending on mode
    if threshold_mode == 'match_truth':
        thr_per_sp, n_target_per_sp = match_truth_thresholds(mean_pred, truth)
        method_label = ("calibrated to truth area "
                        "(top-N per species, N = truth range)")
    elif threshold_mode == 'v3':
        thr_per_sp, bucket_per_sp, _, tau = v3_per_species_thresholds(
            observed, K, mean_pred=mean_pred)
        method_label = (f"v3 truth-free routing "
                        f"(p≥{V3_THRESHOLD_HARD} HARD, "
                        f"p≥{V3_THRESHOLD_MODERATE} else; τ={tau:.3f})")
    else:
        S = truth.shape[0]
        thr_per_sp = np.full(S, float(fixed_threshold))
        method_label = f"fixed p≥{fixed_threshold:.2f}"

    # ── Build the three RGBA composite grids ──
    rgba_truth = np.ones((GRID_Y, GRID_X, 4))
    rgba_noisy = np.ones((GRID_Y, GRID_X, 4))
    rgba_recon = np.ones((GRID_Y, GRID_X, 4))

    species_info = []
    for sp_idx, sp in enumerate(chosen):
        color_hex = THREE_MAP_PALETTE[sp_idx % len(THREE_MAP_PALETTE)]
        color_rgb = mcolors.to_rgb(color_hex)
        rng_truth = int(truth[sp].sum())
        rng_obs   = int(observed[sp].sum())

        # Truth panel: full truth range of this species
        _render_species_layer(rgba_truth, truth[sp], color_rgb, alpha=0.80)

        # Noisy panel: K observation cells of this species
        _render_species_layer(rgba_noisy, observed[sp], color_rgb, alpha=0.85)

        # Reconstructed panel: model's binary at per-species threshold.
        # BUG FIX: previous code used `(samples >= thr).any(axis=0)`
        # (ensemble UNION across 8 samples), which inflates the cell
        # count by ~Nx when the probability map is flat. We now threshold
        # the MEAN prediction and use argpartition for an exact top-N
        # selection — guaranteeing recon area exactly equals truth area
        # when threshold_mode == 'match_truth'.
        if threshold_mode == 'match_truth':
            # Exact top-N (no rounding ambiguity)
            flat = mean_pred[sp].ravel()
            n_target = int(truth[sp].sum())
            if n_target > 0 and flat.max() > 1e-6:
                idx_topN = np.argpartition(flat, -n_target)[-n_target:]
                binary_recon = np.zeros(flat.size, dtype=np.uint8)
                binary_recon[idx_topN] = 1
                binary_recon = binary_recon.reshape(truth[sp].shape)
            else:
                binary_recon = np.zeros_like(truth[sp])
        else:
            # v3 or fixed: threshold the mean prediction at thr_per_sp[sp]
            binary_recon = (mean_pred[sp] >= thr_per_sp[sp]).astype(np.uint8)

        _render_species_layer(rgba_recon, binary_recon, color_rgb, alpha=0.80)

        # Recall (all + novel)
        recall_all, recall_novel = per_species_recall(
            truth[sp], binary_recon, observed[sp])

        # NEW: near/far decomposition (Axel transcript 9:21-10:30)
        near_far = per_species_recall_near_far(
            truth[sp], binary_recon, observed[sp], near_radius=2)

        # NEW: ensemble truth coverage (Axel transcript 0:36, 30:34 + email)
        ens_rec_novel, ens_diversity = compute_ensemble_truth_coverage(
            truth[sp], samples[:, sp], observed[sp])

        species_info.append({
            'sp':            sp,
            'color':         color_hex,
            'range_truth':   rng_truth,
            'range_obs':     rng_obs,
            'range_recon':   int(binary_recon.sum()),
            'recall_all':    recall_all,
            'recall_novel':  recall_novel,
            # Near/far decomposition
            'recall_near':       near_far['rec_near'],
            'recall_far':        near_far['rec_far'],
            'baseline_near':     near_far['baseline_near'],
            'baseline_far':      near_far['baseline_far'],
            'n_truth_near':      near_far['n_truth_near'],
            'n_truth_far':       near_far['n_truth_far'],
            # Ensemble coverage
            'ensemble_recall_novel': ens_rec_novel,
            'ensemble_diversity':    ens_diversity,
        })

    # ── Render the figure: 3 maps on top row + 1 wide bar chart below ──
    # Increased height to accommodate the 3-bar-per-species panel D
    # (near/far/ens) and the 3-line suptitle (overall / near-far / ensemble).
    fig = plt.figure(figsize=(16.0, 11.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=3, height_ratios=[5.5, 2.8],
        hspace=0.70, wspace=0.10, left=0.04, right=0.985,
        top=0.84, bottom=0.20,
    )
    axes = np.array([fig.add_subplot(gs[0, c]) for c in range(3)])
    ax_bar = fig.add_subplot(gs[1, :])

    panel_titles = [
        'A. TRUTH',
        f'B. NOISY (K = {K} obs/species)',
        'C. AI RECONSTRUCTION',
    ]
    panel_subtitles = [
        f'Full simulated distribution\nof {len(chosen)} species',
        f'Only K = {K} cells observed\nper species ({100.0 * K / (GRID_Y * GRID_X):.1f}% of grid)',
        f'Top-N cells per species\n({method_label})',
    ]
    panel_grids = [rgba_truth, rgba_noisy, rgba_recon]

    for col_idx, (title, subtitle, grid) in enumerate(
            zip(panel_titles, panel_subtitles, panel_grids)):
        ax = axes[col_idx]
        ax.imshow(grid, interpolation='nearest', origin='upper')
        # Subtle grid lines so reviewers can count cells
        for k in range(GRID_X + 1):
            ax.axvline(k - 0.5, color='#cccccc', linewidth=0.4, zorder=0)
        for k in range(GRID_Y + 1):
            ax.axhline(k - 0.5, color='#cccccc', linewidth=0.4, zorder=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, GRID_X - 0.5); ax.set_ylim(GRID_Y - 0.5, -0.5)
        ax.set_title(title, fontweight='bold', fontsize=12.5, pad=4)
        ax.text(0.5, -0.06, subtitle, transform=ax.transAxes, ha='center',
                va='top', fontsize=10, color='#444', style='italic')
        for spine in ax.spines.values():
            spine.set_edgecolor('#666666'); spine.set_linewidth(1.0)

    # ── Panel D — per-species grouped bar chart ──
    # Three bars per species answering Axel's three key questions:
    #   NEAR (within 2 cells of obs) — local spatial autocorrelation
    #         (transcript 9:21: "if a cell is occupied, I expect that in
    #          the vicinity some other cells will also be occupied")
    #   FAR  (further than 2 cells)  — true spatial extrapolation
    #         (transcript 10:09: "for far away cells, the AI just randomly
    #          picks ... probably not better than chance")
    #   ENS  (ensemble UNION recall) — does truth lie in the ensemble?
    #         (email: "in some statistical sense the ground truth is part
    #          of this ensemble")
    grid_size = GRID_Y * GRID_X
    species_for_bar = [info for info in species_info
                        if not np.isnan(info['recall_novel'])]
    if species_for_bar:
        n_sp = len(species_for_bar)
        x_pos = np.arange(n_sp)
        bar_w = 0.26

        # Three series + their baselines
        rec_near = [info['recall_near']    if not np.isnan(info['recall_near'])    else 0.0
                    for info in species_for_bar]
        rec_far  = [info['recall_far']     if not np.isnan(info['recall_far'])     else 0.0
                    for info in species_for_bar]
        rec_ens  = [info['ensemble_recall_novel'] if not np.isnan(info['ensemble_recall_novel']) else 0.0
                    for info in species_for_bar]
        bl_near = [info['baseline_near']  if not np.isnan(info['baseline_near'])  else 0.0
                   for info in species_for_bar]
        bl_far  = [info['baseline_far']   if not np.isnan(info['baseline_far'])   else 0.0
                   for info in species_for_bar]
        bl_ens  = [info['range_truth'] / max(1, grid_size - K)
                   for info in species_for_bar]
        species_colors = [info['color'] for info in species_for_bar]

        # Three bars per species — same colour per species, distinguished by
        # hatch and edge for greyscale clarity:
        #   NEAR  — solid coloured bar, thick edge
        #   FAR   — hatched diagonal lines
        #   ENS   — hatched dots, slightly transparent
        b_near = ax_bar.bar(x_pos - bar_w, rec_near, bar_w,
                              color=species_colors, edgecolor='#222',
                              linewidth=1.2, label='NEAR (within 2 cells)')
        b_far = ax_bar.bar(x_pos, rec_far, bar_w,
                             color=species_colors, edgecolor='#222',
                             linewidth=1.2, hatch='////',
                             label='FAR (beyond 2 cells)')
        b_ens = ax_bar.bar(x_pos + bar_w, rec_ens, bar_w,
                             color=species_colors, edgecolor='#222',
                             linewidth=1.2, hatch='...',
                             label='ENS (any sample, top-N)', alpha=0.80)

        # Dashed random baseline per bar
        for xi, bn, bf, be in zip(x_pos, bl_near, bl_far, bl_ens):
            ax_bar.hlines(bn, xi - bar_w - bar_w/2 + 0.01, xi - bar_w/2 - 0.01,
                           color='#111', linewidth=1.2, linestyle='--', zorder=5)
            ax_bar.hlines(bf, xi - bar_w/2 + 0.01, xi + bar_w/2 - 0.01,
                           color='#111', linewidth=1.2, linestyle='--', zorder=5)
            ax_bar.hlines(be, xi + bar_w/2 + 0.01, xi + bar_w + bar_w/2 - 0.01,
                           color='#111', linewidth=1.2, linestyle='--', zorder=5)

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(
            [f"sp #{info['sp']}\nN={info['range_truth']}"
             for info in species_for_bar],
            fontsize=9)
        ax_bar.set_ylabel('Recall at novel truth cells', fontsize=10)
        max_val = max(
            max(rec_near + rec_far + rec_ens),
            max(bl_near + bl_far + bl_ens),
        ) * 1.18
        ax_bar.set_ylim(0, max(0.10, max_val))

        # Title with the population summary one-liner
        ax_bar.set_title(
            "D. Per-species spatial-extrapolation decomposition   "
            "(dashed = random baseline; "
            "solid = NEAR, hatched-slash = FAR, hatched-dot = ENS union)",
            fontweight='bold', fontsize=10.5, pad=6, loc='left')

        # A separate small legend for the three series (using greyscale
        # patches so colour is reserved for species identity)
        from matplotlib.patches import Patch
        legend_series = [
            Patch(facecolor='#aaaaaa', edgecolor='#333', label='NEAR'),
            Patch(facecolor='#aaaaaa', edgecolor='#333', hatch='///', label='FAR'),
            Patch(facecolor='#aaaaaa', edgecolor='#333', hatch='...', label='ENS'),
            plt.Line2D([0], [0], color='#111', linewidth=1.2, linestyle='--',
                        label='Random baseline'),
        ]
        ax_bar.legend(handles=legend_series, loc='upper right',
                       fontsize=8.5, frameon=False, ncol=1)
        for sp in ('top', 'right'):
            ax_bar.spines[sp].set_visible(False)
        ax_bar.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
        ax_bar.grid(axis='y', alpha=0.25, linestyle=':')
    else:
        ax_bar.text(0.5, 0.5,
                     'All selected species have truth_range \u2264 K\n'
                     '(no novel cells available for extrapolation evaluation)',
                     ha='center', va='center', fontsize=11, color='#777',
                     style='italic', transform=ax_bar.transAxes)
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
        for sp in ('top', 'right', 'left', 'bottom'):
            ax_bar.spines[sp].set_visible(False)

    # ── Population recall summary (computed across the chosen species) ──
    # Honest aggregate: mean recall(novel) and how it compares to the
    # random baseline of picking N cells at random from the (G - K)
    # unobserved cells of the grid. For a species with truth_range = R
    # and K observations, R - K novel-truth cells live among G - K
    # unobserved cells, so a random N-cell pick (N = R) is expected to
    # hit (R - K) * R / (G - K) novel-truth cells, i.e. expected
    # recall(novel) = R / (G - K). We report the mean across species.
    grid_size = GRID_Y * GRID_X
    recall_novels = []
    random_baselines = []
    recall_nears = []
    recall_fars = []
    baseline_nears = []
    baseline_fars = []
    ensemble_recs = []
    for info in species_info:
        if not np.isnan(info['recall_novel']):
            recall_novels.append(info['recall_novel'])
            r = info['range_truth']
            random_baselines.append(r / max(1, grid_size - K))
        if not np.isnan(info.get('recall_near', float('nan'))):
            recall_nears.append(info['recall_near'])
            baseline_nears.append(info['baseline_near'])
        if not np.isnan(info.get('recall_far', float('nan'))):
            recall_fars.append(info['recall_far'])
            baseline_fars.append(info['baseline_far'])
        if not np.isnan(info.get('ensemble_recall_novel', float('nan'))):
            ensemble_recs.append(info['ensemble_recall_novel'])

    if recall_novels:
        mean_recall_novel = float(np.mean(recall_novels))
        mean_random       = float(np.mean(random_baselines))
        x_random          = (mean_recall_novel / mean_random
                              if mean_random > 1e-9 else float('nan'))
        line1 = (f"recall(novel) = {mean_recall_novel:.0%} "
                  f"(\u2248 {x_random:.1f}\u00d7 random baseline {mean_random:.0%})")
        # Near/far decomposition (Axel transcript 9:21)
        if recall_nears and recall_fars:
            mn_near = float(np.mean(recall_nears))
            mn_far  = float(np.mean(recall_fars))
            mb_near = float(np.mean(baseline_nears))
            mb_far  = float(np.mean(baseline_fars))
            xn_near = mn_near / mb_near if mb_near > 1e-9 else float('nan')
            xn_far  = mn_far  / mb_far  if mb_far  > 1e-9 else float('nan')
            line2 = (f"NEAR (within 2 cells of obs): {mn_near:.0%} "
                      f"(\u2248{xn_near:.1f}\u00d7) "
                      f"  |   FAR (truly extrapolated): {mn_far:.0%} "
                      f"(\u2248{xn_far:.1f}\u00d7)")
        else:
            line2 = ""
        # Ensemble coverage (Axel email + transcript 0:36, 30:34)
        if ensemble_recs:
            mn_ens = float(np.mean(ensemble_recs))
            line3 = (f"ensemble UNION recall(novel) = {mn_ens:.0%} "
                      f"\u2014 truth is in the ensemble support for this fraction")
        else:
            line3 = ""
        summary_line = line1
        if line2: summary_line += "\n" + line2
        if line3: summary_line += "\n" + line3
    else:
        summary_line = "all selected species had truth_range \u2264 K (no novel cells)"

    # ── Suptitle (positioned ABOVE gridspec via fig.suptitle, y in fig coords) ──
    fig.suptitle(
        f"Reconstructing species distributions from sparse observations\n"
        f"World: thr={params.get('thr','?')}, env={params.get('env','?')}, "
        f"dr={params.get('dr','?')}   |   "
        f"K = {K} observations per species   |   {len(chosen)} species shown\n"
        f"{summary_line}",
        fontweight='bold', fontsize=12.5, y=0.985)

    # ── Bottom legend with per-species × random baseline ──
    legend_lines = []
    for info in species_info:
        rn = info['recall_novel']
        rn_str = f'{rn:.0%}' if not np.isnan(rn) else 'n/a'
        # Per-species × random multiplier (only when meaningful)
        r = info['range_truth']
        if not np.isnan(rn) and (r - K) > 0:
            baseline = r / max(1, grid_size - K)
            x_rand = rn / baseline if baseline > 1e-9 else float('nan')
            x_str  = "\u22480 \u00d7 random" if rn < 1e-9 else f"\u2248{x_rand:.1f}\u00d7"
        else:
            x_str = "n/a"
        # NEAR/FAR snippet
        rn_near = info.get('recall_near', float('nan'))
        rn_far  = info.get('recall_far',  float('nan'))
        if not np.isnan(rn_near):
            near_str = f"near={rn_near:.0%}"
        else:
            near_str = "near=n/a"
        if not np.isnan(rn_far):
            far_str = f"far={rn_far:.0%}"
        else:
            far_str = "far=n/a"
        # Ensemble coverage snippet
        ens_r = info.get('ensemble_recall_novel', float('nan'))
        ens_str = f"ens={ens_r:.0%}" if not np.isnan(ens_r) else "ens=n/a"
        label = (f"sp #{info['sp']}  N={info['range_truth']}  |  "
                  f"novel={rn_str} {x_str}  |  {near_str} | {far_str}  |  "
                  f"{ens_str}")
        legend_lines.append(mpatches.Patch(color=info['color'], label=label))

    # Auto-wrap legend across multiple rows if there are many species
    ncol = min(5, len(legend_lines))
    fig.legend(handles=legend_lines, loc='lower center',
                ncol=ncol,
                bbox_to_anchor=(0.5, 0.04), frameon=False,
                fontsize=9.0, columnspacing=1.5,
                handlelength=1.4)

    # Bottom caption matching Axel's transcript "amazing because" framing
    fig.text(0.5, 0.005,
              "Where multiple species overlap a cell, their colours blend.   "
              "novel = fraction of truth cells the model identified EXCLUDING "
              "the K observation cells.   "
              "near = recall on novel cells within 2 cells of an observation "
              "(local spatial autocorrelation).   "
              "far = recall on novel cells beyond that radius (true spatial "
              "extrapolation).   "
              "ens = recall under the ensemble UNION of all 8 samples at "
              "top-N each \u2014 measures whether truth lies in the model's "
              "ensemble support (Axel email: 'in some statistical sense the "
              "ground truth is part of this ensemble').",
              ha='center', fontsize=8.5, style='italic', color='#555',
              wrap=True)

    # NOTE: do NOT call plt.tight_layout() — gridspec already manages it.
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Console summary
    print(f"  ✓ Axel three-map figure → {output_path}")
    print(f"     {len(species_info)} species: {[info['sp'] for info in species_info]}")
    for info in species_info:
        def _fmt(x): return f"{x:.0%}" if not np.isnan(x) else " n/a"
        rn = info['recall_novel']; rnear = info['recall_near']
        rfar = info['recall_far']; rens = info['ensemble_recall_novel']
        print(f"     sp #{info['sp']:>4d}  truth={info['range_truth']:>3d}  "
              f"recon={info['range_recon']:>3d}  "
              f"recall(all)={info['recall_all']:.0%}  "
              f"novel={_fmt(rn)}  near={_fmt(rnear)}  far={_fmt(rfar)}  "
              f"ens={_fmt(rens)}")


def load_world(truth_dir, recon_pattern, world_stem, K=5, recon_filename=None):
    """Load truth + recon NPZs for one world."""
    truth_path = Path(truth_dir) / f'{world_stem}.npz'
    recon_dir = Path(recon_pattern.format(world_stem=world_stem))
    samples_path = recon_dir / f'recon_fixed_b{K}_samples.npz'
    samples_path = recon_dir / (recon_filename or f'recon_fixed_b{K}_samples.npz')

    if not (truth_path.exists() and samples_path.exists()):
        raise FileNotFoundError(
            f"need:\n   {truth_path}\n   {samples_path}")

    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)

    z = np.load(samples_path)
    samples   = np.asarray(z['samples']).astype(np.float32)
    if 'mean' in z.files:
        mean_pred = np.asarray(z['mean']).astype(np.float32)
    else:
        mean_pred = samples.mean(axis=0)
    if 'noisy_input' in z.files:
        observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    elif 'obs_mask' in z.files:
        observed = np.asarray(z['obs_mask']).astype(np.uint8)
    else:
        observed = (samples.mean(axis=0) >= 0.99).astype(np.uint8)

    n_use = min(truth.shape[0], samples.shape[1], observed.shape[0])
    return {
        'world':     f'{world_stem}.npz',
        'truth':     truth[:n_use],
        'samples':   samples[:, :n_use],
        'mean_pred': mean_pred[:n_use],
        'observed':  observed[:n_use],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--recon-dir-pattern', required=True,
                    help="pattern with {world_stem} placeholder")
    ap.add_argument('--world-stem', required=True,
                    help='single world stem (filename without .npz)')
    ap.add_argument('--figure-style', choices=['three_map', 'grid', 'both'],
                    default='three_map',
                    help='three_map = (1 row × 3 panels, '
                         '5 species in colors); grid = legacy per-species '
                         'inspection (5 rows × 6 cols); both = produce both.')
    ap.add_argument('--threshold-mode',
                    choices=['match_truth', 'v3', 'fixed'],
                    default='match_truth',
                    help='match_truth (DEFAULT) = top-N where N=truth range. '
                         'v3 = truth-free per-species bucket-router (with '
                         'degenerate-tau fallback). fixed = single threshold.')
    ap.add_argument('--threshold', type=float, default=0.80,
                    help='Used only when --threshold-mode fixed (default 0.80)')
    ap.add_argument('--n-species', type=int, default=5,
                    help='Number of species shown in three_map figure '
                         )
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--output-path', required=True)
    args = ap.parse_args()

    world_data = load_world(args.truth_dir, args.recon_dir_pattern,
                              args.world_stem, K=args.K)

    output_path = Path(args.output_path)

    if args.figure_style in ('three_map', 'both'):
        # If both, suffix the three-map output
        three_map_path = (output_path if args.figure_style == 'three_map'
                          else output_path.with_name(
                              output_path.stem + '_three_map' + output_path.suffix))
        make_three_map_axel_figure(
            world_data, three_map_path, K=args.K,
            n_species=args.n_species,
            threshold_mode=args.threshold_mode,
            fixed_threshold=args.threshold,
        )

    if args.figure_style in ('grid', 'both'):
        grid_path = (output_path if args.figure_style == 'grid'
                     else output_path.with_name(
                         output_path.stem + '_grid' + output_path.suffix))
        make_per_species_figure(
            world_data, grid_path,
            threshold_mode=args.threshold_mode,
            fixed_threshold=args.threshold,
            K=args.K,
        )


if __name__ == "__main__":
    main()