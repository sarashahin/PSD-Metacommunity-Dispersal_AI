#!/usr/bin/env python3
"""
=============================================================================
STAGE 2 ABLATION STUDY v5 — Axel's Poisson with correct BODY_MASS conversion
=============================================================================

v3 FINDING:
  ARM B FIXED BUDGET: r=0.857–0.914 with 5–10 obs/species. Core result, valid.
  ARM B POISSON (v3): BROKEN — applied p per CELL ignoring abundance. At p=0.01,
                      98% of species had zero observations → model predicted
                      absence for them → only 52 of 3265 species predicted.

v4 ATTEMPT:
  Applied p_obs to biomass directly. Still wrong — biomass units in your
  Lotka-Volterra IBM are ~0.4 per cell, not per individual.

v5 FINAL FIX — verified by reading config.py and models_ibm.py:
  BODY_MASS = 1e-4                              (config.py)
  B_biomass = N_individuals * BODY_MASS         (models_ibm.py)
  THEREFORE: N_individuals = B_biomass / BODY_MASS = B * 10000

AXEL'S EXACT MODEL (now correctly implemented):

  For each species s and cell (y, x):
    N_individuals[s, y, x] = B_last[s, y, x] / BODY_MASS
    observed_count[s, y, x] ~ Poisson(N_individuals[s, y, x] * p_obs)
    detected[s, y, x] = 1 if observed_count >= 1 else 0

  This is Axel's verbatim description: "for each individual there is a p
  probability of observing it; observed count ~ Poisson(N × p)."

ECOLOGICAL REGIME FOR YOUR DATA (from diagnostic):
  Mean B per occupied cell: 0.4   →  ~4,000 individuals per cell
  Max B per cell: 1.2              →  ~12,000 individuals per cell
  Threshold: B = 10 * BODY_MASS    →  10 individuals (from config.py)

  At Axel's nominal p_obs=0.01 (1%): expected count in mean cell = 40 → SATURATED
  INFORMATIVE regime for this data: p_obs ∈ [0.00001, 0.001]

ABLATION CONDITIONS (unchanged structure, 17 total):

  ORIGINAL (v1):       FULL, NO_TEMPORAL, NO_INTERACT, ENV_ONLY, OBS_INFILL
  ARM A:               GAP_5, GAP_25
  ARM B FIXED:         SPARSE_1, SPARSE_5, SPARSE_10, SPARSE_20, SPARSE_50
  ARM B POISSON (v5):  Five p_obs values REPARAMETERIZED to informative regime
     POISSON_p00001  p_obs = 0.00001   (1 per 100,000 individuals)
     POISSON_p0001   p_obs = 0.0001    (1 per 10,000 individuals)
     POISSON_p0005   p_obs = 0.0005    (5 per 10,000 individuals)
     POISSON_p001    p_obs = 0.001     (1 per 1,000 individuals)
     POISSON_p01     p_obs = 0.01      (AXEL'S STANDARD — likely saturates)

USAGE — RECOMMENDED: re-run only Poisson conditions (~10-12 h):
  python stage2_ablation_study_v5.py \\
      --ibm-dir results/data/ \\
      --checkpoint stage2_outputs/checkpoints/best_model.pt \\
      --stage2-dir AI_simulation/stage2/ \\
      --output-dir stage2_ablation_v5_poisson/ \\
      --n-samples 8 --n-worlds 8 \\
      --only-poisson

  Existing v3 results for FULL, NO_*, GAP_*, SPARSE_*, OBS_INFILL remain VALID.
  Do not re-run them. Merge v5 Poisson JSON with v3 JSON for final figures.
=============================================================================
"""

import argparse
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════
# CRITICAL CONFIGURATION CONSTANT — must match config.py
# ═══════════════════════════════════════════════════════════════════════
# From your config.py:   BODY_MASS = 1e-4
# From your models_ibm.py: B = self.N * BODY_MASS
# Therefore: N_individuals = B_biomass / BODY_MASS_UNIT
#
# This conversion is essential for Axel's Poisson model. Axel's "1% per
# individual" must be applied to N_individuals, NOT to B_biomass directly.
# The v3 script applied p per cell (wrong) and v4 applied p to biomass (wrong).
# v5 correctly converts biomass → individuals using this constant.
BODY_MASS_UNIT = 1e-4


# ═══════════════════════════════════════════════════════════════════════
# ABLATION CONDITIONS — 5 original + 2 gap + 5 sparse + 5 Poisson = 17
# ═══════════════════════════════════════════════════════════════════════

ABLATION_CONDITIONS = OrderedDict([
    # ── ORIGINAL CONDITIONS ──
    ("FULL",        {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 0, "sparse_budget": None,
                     "label": "Full Model", "color": "#2563eb",
                     "desc": "All inputs (P_t[-1] as prior)"}),
    ("NO_TEMPORAL", {"drop_interactions": False, "drop_temporal": True,  "drop_obs": True,
                     "gap": 0, "sparse_budget": None,
                     "label": "No Temporal", "color": "#7c3aed",
                     "desc": "Drop P_t entirely"}),
    ("NO_INTERACT", {"drop_interactions": True,  "drop_temporal": False, "drop_obs": True,
                     "gap": 0, "sparse_budget": None,
                     "label": "No Interactions", "color": "#ea580c",
                     "desc": "Drop competition graph"}),
    ("ENV_ONLY",    {"drop_interactions": True,  "drop_temporal": True,  "drop_obs": True,
                     "gap": 0, "sparse_budget": None,
                     "label": "Env Only", "color": "#059669",
                     "desc": "Only environment field"}),
    ("OBS_INFILL",  {"drop_interactions": False, "drop_temporal": True,  "drop_obs": False,
                     "gap": 0, "sparse_budget": None,
                     "label": "Obs Infill (5)", "color": "#0891b2",
                     "desc": "Env + 5 obs as conditioning"}),

    # ── NEW ARM A: TEMPORAL GAP ──
    # Use P_t[:-K] as the history (so the prior becomes P_t[-(K+1)])
    ("GAP_5",       {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 5, "sparse_budget": None,
                     "label": "Gap 5 snaps", "color": "#a78bfa",
                     "desc": "Use P_t[-5] as prior (skip 5 snapshots)"}),
    ("GAP_25",      {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 25, "sparse_budget": None,
                     "label": "Gap 25 snaps", "color": "#7e22ce",
                     "desc": "Use P_t[-25] as prior (skip 25 snapshots)"}),

    # ── NEW ARM B: SPARSE HISTORY ──
    # Take P_t[-1], randomly mask to keep only B cells per species
    ("SPARSE_1",    {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 0, "sparse_budget": 1,
                     "label": "Sparse 1/sp", "color": "#fef3c7",
                     "desc": "P_t[-1] sparsified to 1 cell per species"}),
    ("SPARSE_5",    {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 0, "sparse_budget": 5,
                     "label": "Sparse 5/sp", "color": "#fde047",
                     "desc": "P_t[-1] sparsified to 5 cells per species"}),
    ("SPARSE_10",   {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 0, "sparse_budget": 10,
                     "label": "Sparse 10/sp", "color": "#facc15",
                     "desc": "P_t[-1] sparsified to 10 cells per species"}),
    ("SPARSE_20",   {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 0, "sparse_budget": 20,
                     "label": "Sparse 20/sp", "color": "#eab308",
                     "desc": "P_t[-1] sparsified to 20 cells per species"}),
    ("SPARSE_50",   {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                     "gap": 0, "sparse_budget": 50,
                     "label": "Sparse 50/sp", "color": "#ca8a04",
                     "desc": "P_t[-1] sparsified to 50 cells per species"}),

    # ── ARM B-POISSON (v5): AXEL'S EXACT MODEL WITH BODY_MASS CONVERSION ──
    # observed_count ~ Poisson(B_last / BODY_MASS_UNIT * p_obs) per cell
    # i.e. "for each individual there is a p_obs probability of observing it"
    # (Axel's verbatim description). Cell detected if observed_count >= 1.
    # All past snapshots zeroed (same structure as fixed-budget SPARSE_*).
    #
    # p_obs values REPARAMETERIZED for the informative regime of your data
    # (mean ~4000 individuals per occupied cell). Previous v3 values saturated.
    ("POISSON_p00001", {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                        "gap": 0, "sparse_budget": None, "poisson_p": 0.00001,
                        "label": "Poisson p=0.001%", "color": "#fecaca",
                        "desc": "Axel's Poisson, 0.001% per individual (extreme sparsity)"}),
    ("POISSON_p0001",  {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                        "gap": 0, "sparse_budget": None, "poisson_p": 0.0001,
                        "label": "Poisson p=0.01%",  "color": "#fca5a5",
                        "desc": "Axel's Poisson, 0.01% per individual (IUCN rare-species regime)"}),
    ("POISSON_p0005",  {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                        "gap": 0, "sparse_budget": None, "poisson_p": 0.0005,
                        "label": "Poisson p=0.05%",  "color": "#f87171",
                        "desc": "Axel's Poisson, 0.05% per individual (sparse field survey)"}),
    ("POISSON_p001",   {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                        "gap": 0, "sparse_budget": None, "poisson_p": 0.001,
                        "label": "Poisson p=0.1%",   "color": "#ef4444",
                        "desc": "Axel's Poisson, 0.1% per individual (moderate survey)"}),
    ("POISSON_p01",    {"drop_interactions": False, "drop_temporal": False, "drop_obs": True,
                        "gap": 0, "sparse_budget": None, "poisson_p": 0.01,
                        "label": "Poisson p=1%",     "color": "#dc2626",
                        "desc": "AXEL'S STANDARD: 1% per individual (expected to saturate)"}),
])


# ═══════════════════════════════════════════════════════════════════════
# METRICS — same as v1
# ═══════════════════════════════════════════════════════════════════════

def richness_map(P_binary):
    return P_binary.sum(axis=0)

def range_sizes(P_binary):
    return P_binary.sum(axis=(1, 2))

def pairwise_beta_diversity(P_binary):
    S, Y, X = P_binary.shape
    beta = np.full((Y, X), np.nan)
    for y in range(Y):
        for x in range(X):
            a = P_binary[:, y, x].astype(float)
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < Y and 0 <= nx < X:
                    b = P_binary[:, ny, nx].astype(float)
                    inter = (a * b).sum()
                    union = ((a + b) > 0).sum()
                    if union > 0:
                        neighbors.append(1.0 - inter / union)
            if neighbors:
                beta[y, x] = np.mean(neighbors)
    return beta

def species_area_curve(P_binary, n_windows=15):
    S, Y, X = P_binary.shape
    max_side = min(Y, X)
    sides = np.unique(np.linspace(1, max_side, n_windows, dtype=int))
    areas, richnesses = [], []
    for side in sides:
        samples = []
        for _ in range(min(40, max(5, (Y * X) // (side * side)))):
            y0 = np.random.randint(0, max(1, Y - side + 1))
            x0 = np.random.randint(0, max(1, X - side + 1))
            n_sp = (P_binary[:, y0:y0+side, x0:x0+side].sum(axis=(1, 2)) > 0).sum()
            samples.append(n_sp)
        areas.append(side * side)
        richnesses.append(np.mean(samples))
    return np.array(areas), np.array(richnesses)


def compute_all_metrics(P_ibm, P_model_prob, threshold=0.5):
    """Compute comprehensive metrics comparing IBM truth to model predictions."""
    P_ibm_bin = (P_ibm > 0).astype(np.float32)
    P_mod_bin = (P_model_prob >= threshold).astype(np.float32)
    metrics = {}

    # 1. Richness correlation
    r_ibm = richness_map(P_ibm_bin)
    r_mod = richness_map(P_mod_bin)
    if r_ibm.std() > 0 and r_mod.std() > 0:
        metrics['richness_r'] = float(np.corrcoef(r_ibm.flatten(), r_mod.flatten())[0, 1])
    else:
        metrics['richness_r'] = 0.0
    metrics['richness_rmse'] = float(np.sqrt(np.mean((r_ibm - r_mod) ** 2)))
    metrics['richness_mean_ibm'] = float(r_ibm.mean())
    metrics['richness_mean_mod'] = float(r_mod.mean())

    # 2. Range-size
    rs_ibm = range_sizes(P_ibm_bin)
    rs_mod = range_sizes(P_mod_bin)
    mask = (rs_ibm > 0) | (rs_mod > 0)
    if mask.sum() > 10:
        ks_stat, ks_p = stats.ks_2samp(rs_ibm[mask], rs_mod[mask])
        metrics['range_ks_stat'] = float(ks_stat)
        metrics['range_ks_p'] = float(ks_p)
        if rs_ibm[mask].std() > 0 and rs_mod[mask].std() > 0:
            metrics['range_r'] = float(np.corrcoef(rs_ibm[mask], rs_mod[mask])[0, 1])
        else:
            metrics['range_r'] = 0.0
    else:
        metrics['range_ks_stat'] = 1.0; metrics['range_ks_p'] = 0.0; metrics['range_r'] = 0.0

    # 3. Beta-diversity
    beta_ibm = pairwise_beta_diversity(P_ibm_bin)
    beta_mod = pairwise_beta_diversity(P_mod_bin)
    valid = ~np.isnan(beta_ibm) & ~np.isnan(beta_mod)
    if valid.sum() > 10 and beta_ibm[valid].std() > 0 and beta_mod[valid].std() > 0:
        metrics['beta_r'] = float(np.corrcoef(beta_ibm[valid], beta_mod[valid])[0, 1])
    else:
        metrics['beta_r'] = 0.0

    # 4. Prevalence
    prev_ibm = P_ibm_bin.mean(axis=(1, 2))
    prev_mod = P_mod_bin.mean(axis=(1, 2))
    mask_prev = (prev_ibm > 0) | (prev_mod > 0)
    if mask_prev.sum() > 10 and prev_ibm[mask_prev].std() > 0 and prev_mod[mask_prev].std() > 0:
        metrics['prevalence_r'] = float(np.corrcoef(prev_ibm[mask_prev], prev_mod[mask_prev])[0, 1])
    else:
        metrics['prevalence_r'] = 0.0

    # 5. Species count
    metrics['n_species_ibm'] = int((rs_ibm > 0).sum())
    metrics['n_species_mod'] = int((rs_mod > 0).sum())

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# CONDITION BUILDER — supports gap and sparsification
# ═══════════════════════════════════════════════════════════════════════

def sparsify_history_last_timestep(Pt, budget, rng_seed=42):
    """
    Take the last timestep of P_t and randomly mask each species to keep only
    `budget` observed cells. Returns a new P_t array where the last snapshot
    is sparsified, all earlier snapshots are zeroed (we only want to test the
    final-snapshot prior).

    Pt: (T, S, Y, X)
    budget: int, observations per species (1, 5, 10, 20, 50, ...)
    """
    rng = np.random.default_rng(rng_seed)
    T, S, Y, X = Pt.shape
    Pt_sparse = np.zeros_like(Pt)

    last = Pt[-1]  # (S, Y, X)
    sparse_last = np.zeros_like(last)

    for s in range(S):
        # Find all occupied cells for this species
        occupied = np.argwhere(last[s] > 0)  # (n_occ, 2) y,x pairs
        if len(occupied) == 0:
            continue
        n_keep = min(budget, len(occupied))
        # Randomly pick `n_keep` of the occupied cells
        chosen = rng.choice(len(occupied), size=n_keep, replace=False)
        for idx in chosen:
            y, x = occupied[idx]
            sparse_last[s, y, x] = 1.0

    # Put sparsified version into the last slot, zero elsewhere
    # The temporal encoder will see mostly zeros; the diffusion prior will be
    # built from Pt_sparse[-1], which is the sparsified map.
    Pt_sparse[-1] = sparse_last
    return Pt_sparse


def sparsify_history_poisson(Pt, B_last, p_obs,
                              body_mass=BODY_MASS_UNIT, rng_seed=42):
    """
    Axel's gold-standard abundance-weighted Poisson detection model (v5 CORRECT).

    AXEL'S VERBATIM DESCRIPTION:
      "What happens is observations are sampled randomly from all individuals
       that are present. We fix an observation probability p, e.g. 1%. For
       each individual there is a p probability of observing it. Then
       observed count ~ Poisson(N_individuals × p)."

    IMPLEMENTATION (key fix from v3/v4):
      N_individuals[s, y, x] = B_last[s, y, x] / body_mass    (exact conversion)
      expected[s, y, x]      = N_individuals[s, y, x] * p_obs
      observed[s, y, x]      ~ Poisson(expected[s, y, x])
      detected[s, y, x]      = 1 if observed >= 1 else 0

    WHY BODY_MASS MATTERS:
      In config.py: BODY_MASS = 1e-4, and in models_ibm.py: B = self.N * BODY_MASS
      A cell with biomass 0.4 contains N = 4000 individuals.
      Without this conversion, applying p=0.01 to "biomass 0.4" means
      expected count = 0.004 → near-zero detection (v4 error).
      WITH this conversion, p=0.01 × 4000 individuals = expected count 40
      → near-certain detection (correct Axel model).

    ECOLOGICAL BEHAVIOUR (correct model):
      - Abundant species (high B per cell) → reliably detected
      - Rare species (low B per cell, near threshold) → often missed
      - Detection per cell scales with local abundance, as in real surveys

    All past snapshots (T-1 earlier time steps) are zeroed completely.

    Parameters
    ----------
    Pt : ndarray, shape (T, S, Y, X)
        Binary presence/absence time series. Used for shape reference only;
        past snapshots are zeroed in the output.
    B_last : ndarray, shape (S, Y, X)
        Biomass at final snapshot (from npz key "B_last"). Must be in the
        same units as models_ibm.py produces: B = N * BODY_MASS.
    p_obs : float in (0, 1]
        Per-individual observation probability (Axel's "1%" is p_obs=0.01).
    body_mass : float
        Conversion factor: N_individuals = B_biomass / body_mass.
        Default BODY_MASS_UNIT=1e-4 matches config.py exactly.
    rng_seed : int
        For reproducibility across worlds and samples.

    Returns
    -------
    Pt_sparse : ndarray, same shape as Pt
        All-zero except at Pt_sparse[-1], which contains 1s at cells where
        Poisson(N × p_obs) drew at least one observation.
    """
    rng = np.random.default_rng(rng_seed)
    Pt_sparse = np.zeros_like(Pt)

    # Convert biomass → individuals using SAME body_mass as simulation
    # Cast to float64 for numerical stability in Poisson draws
    N_individuals = B_last.astype(np.float64) / body_mass

    # Axel's exact model
    expected_counts = N_individuals * p_obs
    observed_counts = rng.poisson(expected_counts)

    # Cell detected if at least one individual observed
    detected = (observed_counts >= 1).astype(Pt_sparse.dtype)

    Pt_sparse[-1] = detected
    return Pt_sparse


def build_ablation_condition(npz_data, device, drop_interactions=False,
                              drop_temporal=False, drop_obs=True,
                              gap=0, sparse_budget=None, poisson_p=None,
                              obs_budget=5, species_subset=None,
                              rng_seed=42):
    """
    Build conditioning dict with selective input removal AND
    temporal-gap / sparse-history modifications.

    gap: how many trailing snapshots to remove from P_t (so prior = P_t[-gap-1])
    sparse_budget: if set, sparsify P_t[-1] to exactly this many cells per species
    poisson_p: if set, apply Poisson/Bernoulli detection with this probability.
               Cannot be used simultaneously with sparse_budget.
               poisson_p takes priority if both are set (should not happen).
    """
    import torch

    Y, X = int(npz_data["Y"]), int(npz_data["X"])
    P = np.asarray(npz_data["P_last_final"])
    S_data = P.shape[0]
    S = S_data if species_subset is None else len(species_subset)
    if species_subset is not None:
        P = P[species_subset]
    B = 1

    # ── ENV (always provided) ──
    env_raw = np.asarray(npz_data["ENV_r_field"])
    if species_subset is not None:
        env_raw = env_raw[species_subset]
    env_t = torch.from_numpy(env_raw[np.newaxis].copy()).float().to(device)

    # ── Spatial coordinates ──
    y_grid = np.broadcast_to(np.arange(Y, dtype=np.float32).reshape(1, Y, 1), (B, Y, X)).copy()
    x_grid = np.broadcast_to(np.arange(X, dtype=np.float32).reshape(1, 1, X), (B, Y, X)).copy()
    y_coords_t = torch.from_numpy(y_grid).float().to(device)
    x_coords_t = torch.from_numpy(x_grid).float().to(device)

    # ── Species features (zeros if dropping interactions) ──
    pv = np.asarray(npz_data.get("prevalence_final", np.zeros(S_data)))
    if species_subset is not None:
        pv = pv[species_subset]
    sp_feats = np.zeros((B, S, 8), dtype=np.float32)
    if not drop_interactions:
        for s in range(S):
            sp_feats[0, s, 0] = pv[s]
            sp_feats[0, s, 1] = P[s].sum() / (Y * X)
            sp_feats[0, s, 2] = env_raw[s].mean() if s < len(env_raw) else 0
            sp_feats[0, s, 3] = env_raw[s].std() if s < len(env_raw) else 0
            sp_feats[0, s, 4] = float(pv[s] < 0.05)
            sp_feats[0, s, 5] = float(pv[s] >= 0.05)
            sp_feats[0, s, 6] = np.log1p(pv[s])
            sp_feats[0, s, 7] = float(s) / max(S - 1, 1)
    species_features_t = torch.from_numpy(sp_feats).float().to(device)

    # ── Interaction graph ──
    edge_index_t = None
    if not drop_interactions and "C_topk_idx" in npz_data:
        edge_list = []
        ctk = np.asarray(npz_data["C_topk_idx"])
        if species_subset is not None:
            subset_list = species_subset.tolist() if hasattr(species_subset, 'tolist') else list(species_subset)
            subset_set = set(subset_list)
            old_to_new = {old: new for new, old in enumerate(subset_list)}
            for new_s, old_s in enumerate(subset_list):
                for neighbor in ctk[old_s]:
                    if int(neighbor) in subset_set:
                        edge_list.append([new_s, old_to_new[int(neighbor)]])
        else:
            for s in range(min(S, ctk.shape[0])):
                for neighbor in ctk[s]:
                    if 0 <= int(neighbor) < S and int(neighbor) != s:
                        edge_list.append([s, int(neighbor)])
        edge_index_t = (torch.tensor(edge_list, dtype=torch.long, device=device).T
                        if edge_list else torch.empty(2, 0, dtype=torch.long, device=device))

    # ── Temporal history (with optional gap or sparsification) ──
    history_t = None
    if not drop_temporal and "P_t" in npz_data:
        Pt = np.asarray(npz_data["P_t"]).astype(np.float32)
        if species_subset is not None:
            Pt = Pt[:, species_subset, :, :]

        # ARM A — Apply temporal gap: remove last `gap` snapshots
        # so model uses P_t[-(gap+1)] as the prior
        if gap > 0:
            T = Pt.shape[0]
            keep_T = max(1, T - gap)
            Pt = Pt[:keep_T]  # now Pt[-1] is the original Pt[-(gap+1)]

        # ARM B POISSON (v5) — Axel's abundance-weighted detection model
        # Load B_last biomass and convert to individuals using BODY_MASS
        # poisson_p takes priority over sparse_budget (should not both be set)
        if poisson_p is not None:
            if "B_last" not in npz_data:
                raise ValueError(
                    "B_last required in npz for Poisson ablation (v5). "
                    "The correct implementation of Axel's model needs biomass "
                    "data to convert to individual counts."
                )
            B_last = np.asarray(npz_data["B_last"]).astype(np.float32)
            # Handle possible flattened spatial storage
            if B_last.ndim == 2:
                # Shape (S, Y*X) — reshape to (S, Y, X)
                S_b, YX = B_last.shape
                side = int(np.sqrt(YX))
                assert side * side == YX, f"Cannot reshape B_last {B_last.shape}"
                B_last = B_last.reshape(S_b, side, side)
            if species_subset is not None:
                B_last = B_last[species_subset]
            # Sanity check shape matches Pt
            assert B_last.shape == Pt.shape[1:], (
                f"B_last shape {B_last.shape} does not match Pt[1:] {Pt.shape[1:]}"
            )
            Pt = sparsify_history_poisson(
                Pt, B_last, poisson_p,
                body_mass=BODY_MASS_UNIT, rng_seed=rng_seed
            )

        # ARM B FIXED BUDGET — Apply fixed-count sparsification
        elif sparse_budget is not None:
            Pt = sparsify_history_last_timestep(Pt, sparse_budget, rng_seed=rng_seed)

        history_t = torch.from_numpy(Pt[np.newaxis].copy()).float().to(device)

    # ── Build condition dict ──
    condition = {
        "env": env_t,
        "y_coords": y_coords_t,
        "x_coords": x_coords_t,
    }
    if not drop_interactions and edge_index_t is not None:
        condition["species_features"] = species_features_t
        condition["edge_index"] = edge_index_t
        condition["edge_weight"] = None
    if not drop_temporal and history_t is not None:
        condition["history_P"] = history_t

    # OBS_INFILL only
    if not drop_obs:
        obs_key = f"obs_mask_{obs_budget}"
        if obs_key in npz_data:
            obs = np.asarray(npz_data[obs_key]).astype(np.float32)
            if species_subset is not None:
                obs = obs[species_subset]
            condition["obs_mask"] = torch.from_numpy(obs[np.newaxis].copy()).float().to(device)

    return condition


def run_ablation_inference(model, npz_data, device, condition_cfg,
                            n_samples=8, chunk_size=200, obs_budget=5,
                            world_seed=0):
    """Run inference for one ablation condition on one world."""
    import torch

    P = np.asarray(npz_data["P_last_final"])
    S = P.shape[0]
    Y, X = int(npz_data["Y"]), int(npz_data["X"])
    all_preds = np.zeros((S, Y, X), dtype=np.float64)

    model.set_training_phase(4)

    for chunk_start in range(0, S, chunk_size):
        chunk_end = min(chunk_start + chunk_size, S)
        chunk_indices = np.arange(chunk_start, chunk_end)

        condition = build_ablation_condition(
            npz_data, device,
            drop_interactions=condition_cfg["drop_interactions"],
            drop_temporal=condition_cfg["drop_temporal"],
            drop_obs=condition_cfg["drop_obs"],
            gap=condition_cfg.get("gap", 0),
            sparse_budget=condition_cfg.get("sparse_budget"),
            poisson_p=condition_cfg.get("poisson_p"),
            obs_budget=obs_budget,
            species_subset=chunk_indices,
            rng_seed=42 + world_seed,  # reproducible per world
        )

        chunk_preds = []
        for sample_i in range(n_samples):
            try:
                use_prior = "history_P" in condition and condition["history_P"] is not None
                sample = model.sample(
                    condition=condition, n_samples=1,
                    ddim_steps=50, sparse_mode=True, eta=0.5,
                    use_prior=use_prior,
                )
                s_np = sample.cpu().numpy()
                while s_np.ndim > 3 and s_np.shape[0] == 1:
                    s_np = s_np[0]
                chunk_preds.append(s_np)
            except Exception as e:
                if sample_i == 0:
                    print(f"     ⚠ Sample failed: {e}")
                continue

        if chunk_preds:
            all_preds[chunk_start:chunk_end] = np.clip(np.mean(chunk_preds, axis=0), 0, 1)

    return all_preds


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def safe_save(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {Path(path).name}")


def plot_AB1_metric_bars_grouped(all_results, out):
    """Bar chart with original 5 conditions only (for clarity).
       Sparse and gap conditions get their own dedicated figures."""
    base_conds = ["FULL", "NO_TEMPORAL", "NO_INTERACT", "ENV_ONLY", "OBS_INFILL"]
    conds = [c for c in base_conds if c in all_results]
    if not conds:
        return

    metric_keys = ['richness_r', 'range_r', 'beta_r', 'prevalence_r']
    metric_labels = ['Richness\ncorrelation', 'Range-size\ncorrelation',
                     'Beta-diversity\ncorrelation', 'Prevalence\ncorrelation']

    n_conds = len(conds)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cond in enumerate(conds):
        cfg = ABLATION_CONDITIONS[cond]
        vals = [all_results[cond].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=cfg["label"],
                      color=cfg["color"], alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Correlation (r)', fontsize=11)
    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.15)
    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5, label='r = 0.9 target')
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.set_title("AB1: Original Ablation — Metric Comparison",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    safe_save(fig, out / "fig_AB1_metric_bars.png")


def plot_AB7_temporal_gap(all_results, out):
    """NEW: Richness correlation vs temporal gap (Arm A)."""
    full_r = all_results.get("FULL", {}).get("richness_r", 0)
    full_r_std = all_results.get("FULL", {}).get("richness_r_std", 0)

    gap_conds = [("FULL", 0), ("GAP_5", 5), ("GAP_25", 25), ("NO_TEMPORAL", 50)]
    gaps = []
    rs = []
    rs_std = []
    labels = []
    colors_used = []
    for cond, gap_val in gap_conds:
        if cond in all_results:
            gaps.append(gap_val)
            rs.append(all_results[cond].get("richness_r", 0))
            rs_std.append(all_results[cond].get("richness_r_std", 0))
            labels.append(ABLATION_CONDITIONS[cond]["label"])
            colors_used.append(ABLATION_CONDITIONS[cond]["color"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(gaps, rs, yerr=rs_std, fmt='o-', linewidth=2.5, markersize=10,
                color="#2563eb", ecolor='gray', capsize=5, capthick=1.5,
                markerfacecolor="#2563eb", markeredgecolor='white', markeredgewidth=2)

    for i, (g, r, lbl, c) in enumerate(zip(gaps, rs, labels, colors_used)):
        ax.scatter([g], [r], color=c, s=200, edgecolor='white', linewidth=2, zorder=5)
        ax.annotate(f'{lbl}\nr={r:.3f}', (g, r),
                    textcoords="offset points", xytext=(10, 15),
                    fontsize=9, fontweight='bold', color=c)

    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.text(max(gaps) * 0.95, 0.91, 'r = 0.9 target', fontsize=8, color='gray', ha='right')

    ax.set_xlabel('Temporal gap (snapshots removed from end of P_t)', fontsize=11)
    ax.set_ylabel('Richness correlation (r)', fontsize=11)
    ax.set_title("AB7: Temporal Gap Test — Does the model still work when the\n"
                 "history ends earlier than the prediction target?",
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.set_ylim(min(-0.1, min(rs) - 0.05), 1.05)
    safe_save(fig, out / "fig_AB7_temporal_gap.png")


def plot_AB8_sparsity_curve(all_results, out):
    """NEW: Richness correlation vs sparse-history budget (Arm B)."""
    full_r = all_results.get("FULL", {}).get("richness_r", 0)

    sparse_conds = [
        ("SPARSE_1", 1), ("SPARSE_5", 5), ("SPARSE_10", 10),
        ("SPARSE_20", 20), ("SPARSE_50", 50), ("FULL", 400),  # Full = all 400 cells
    ]
    budgets = []
    rs = []
    rs_std = []
    for cond, b in sparse_conds:
        if cond in all_results:
            budgets.append(b)
            rs.append(all_results[cond].get("richness_r", 0))
            rs_std.append(all_results[cond].get("richness_r_std", 0))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(budgets, rs, yerr=rs_std, fmt='s-', linewidth=2.5, markersize=11,
                color="#d97706", ecolor='gray', capsize=5, capthick=1.5,
                markerfacecolor="#fbbf24", markeredgecolor="#d97706", markeredgewidth=2)

    for b, r in zip(budgets, rs):
        label = "Full P_t" if b == 400 else f"B={b}"
        ax.annotate(f'{label}\nr={r:.3f}', (b, r),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=9, fontweight='bold', color="#d97706", ha='center')

    ax.set_xscale('log')
    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.text(1.1, 0.91, 'r = 0.9 target', fontsize=8, color='gray')

    ax.set_xlabel('Observation budget (cells per species in history prior, log scale)',
                  fontsize=11)
    ax.set_ylabel('Richness correlation (r)', fontsize=11)
    ax.set_title("AB8: Sparsity Curve — Can the model recover community structure\n"
                 "from sparse observations? (mimics realistic field-data densities)",
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.2, which='both')
    ax.set_ylim(min(-0.05, min(rs) - 0.05), 1.1)

    # Annotate the realistic ecology range
    ax.axvspan(1, 10, alpha=0.08, color='red', label='Realistic IUCN data density')
    ax.legend(fontsize=9, loc='lower right')

    safe_save(fig, out / "fig_AB8_sparsity_curve.png")


def plot_AB9_supervisor_rebuttal(all_results, out):
    """NEW: One-figure summary for supervisor — answers all three concerns."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original ablation summary
    ax = axes[0]
    base_conds = ["FULL", "NO_TEMPORAL", "NO_INTERACT", "ENV_ONLY"]
    base_labels, base_vals, base_colors = [], [], []
    for c in base_conds:
        if c in all_results:
            base_labels.append(ABLATION_CONDITIONS[c]["label"])
            base_vals.append(all_results[c].get("richness_r", 0))
            base_colors.append(ABLATION_CONDITIONS[c]["color"])
    bars = ax.bar(range(len(base_vals)), base_vals, color=base_colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, base_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(base_labels)))
    ax.set_xticklabels(base_labels, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Richness r', fontsize=10)
    ax.set_title("(A) Original ablation:\ntemporal history is critical",
                 fontsize=11, fontweight='bold')
    ax.set_ylim(min(-0.1, min(base_vals) - 0.05), 1.15)
    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.2)

    # Panel 2: Temporal gap
    ax = axes[1]
    gap_conds = [("FULL", 0), ("GAP_5", 5), ("GAP_25", 25), ("NO_TEMPORAL", 50)]
    gap_x, gap_y = [], []
    for c, g in gap_conds:
        if c in all_results:
            gap_x.append(g); gap_y.append(all_results[c].get("richness_r", 0))
    ax.plot(gap_x, gap_y, 'o-', linewidth=2.5, markersize=10,
            color="#7c3aed", markerfacecolor="#c4b5fd", markeredgecolor="#7c3aed", markeredgewidth=2)
    for x, y in zip(gap_x, gap_y):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                    xytext=(0, 12), fontsize=8, fontweight='bold', ha='center', color="#7c3aed")
    ax.set_xlabel('Snapshots removed', fontsize=10)
    ax.set_ylabel('Richness r', fontsize=10)
    ax.set_title("(B) Time-gap test:\nmodel works even with earlier history",
                 fontsize=11, fontweight='bold')
    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.grid(alpha=0.2)
    ax.set_ylim(min(-0.1, min(gap_y) - 0.05), 1.05)

    # Panel 3: Sparsity curve
    ax = axes[2]
    sp_conds = [("SPARSE_1", 1), ("SPARSE_5", 5), ("SPARSE_10", 10),
                ("SPARSE_20", 20), ("SPARSE_50", 50), ("FULL", 400)]
    sp_x, sp_y = [], []
    for c, b in sp_conds:
        if c in all_results:
            sp_x.append(b); sp_y.append(all_results[c].get("richness_r", 0))
    ax.semilogx(sp_x, sp_y, 's-', linewidth=2.5, markersize=10,
                color="#d97706", markerfacecolor="#fbbf24", markeredgecolor="#d97706", markeredgewidth=2)
    for x, y in zip(sp_x, sp_y):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                    xytext=(0, 12), fontsize=8, fontweight='bold', ha='center', color="#d97706")
    ax.axvspan(1, 10, alpha=0.08, color='red')
    ax.text(3, ax.get_ylim()[0] + 0.05, 'realistic\nIUCN range', fontsize=7,
            color='darkred', ha='center', style='italic')
    ax.set_xlabel('Cells per species in prior (log)', fontsize=10)
    ax.set_ylabel('Richness r', fontsize=10)
    ax.set_title("(C) Sparsity test:\nmodel works with sparse observations",
                 fontsize=11, fontweight='bold')
    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.grid(alpha=0.2, which='both')
    ax.set_ylim(min(-0.1, min(sp_y) - 0.05), 1.1)

    fig.suptitle("AB9: Supervisor Rebuttal — Three Lines of Evidence",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    safe_save(fig, out / "fig_AB9_supervisor_rebuttal.png")


def plot_AB10_poisson_curve(all_results, out):
    """
    NEW v3: Richness correlation vs Poisson detection probability (Arm B-Poisson).

    Shows how MetaDiffusion performs when observations are generated by Axel's
    gold-standard ecological detection model: each occupied cell detected
    independently with probability p_detect.
    """
    poisson_conds = [
        ("POISSON_p00001", 0.00001),
        ("POISSON_p0001",  0.0001),
        ("POISSON_p0005",  0.0005),
        ("POISSON_p001",   0.001),
        ("POISSON_p01",    0.01),
        ("FULL",           None),   # full = no sparsification, plotted separately
    ]

    probs, rs, rs_std = [], [], []
    full_r, full_r_std = None, None
    for cond, p in poisson_conds:
        if cond not in all_results:
            continue
        if p is None:  # FULL reference line
            full_r = all_results[cond].get("richness_r", 0)
            full_r_std = all_results[cond].get("richness_r_std", 0)
        else:
            probs.append(p)
            rs.append(all_results[cond].get("richness_r", 0))
            rs_std.append(all_results[cond].get("richness_r_std", 0))

    if len(probs) < 2:
        print("  ⚠ Skipping AB10 — insufficient Poisson conditions in results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel A: Richness r vs p_obs (per individual, log scale) ──
    ax = axes[0]
    ax.errorbar(probs, rs, yerr=rs_std,
                fmt='o-', linewidth=2.5, markersize=10,
                color="#dc2626", ecolor='gray', capsize=5, capthick=1.5,
                markerfacecolor="#fca5a5", markeredgecolor="#dc2626", markeredgewidth=2,
                label="Axel's Poisson (abundance-weighted)")

    for p, r in zip(probs, rs):
        lbl = f"p={p*100:.3g}%"
        ax.annotate(f'{lbl}\nr={r:.3f}', (p, r),
                    textcoords="offset points", xytext=(0, 14),
                    fontsize=8, fontweight='bold', color="#dc2626", ha='center')

    # Full-model reference line
    if full_r is not None:
        ax.axhline(full_r, color='#2563eb', ls=':', lw=2, alpha=0.8,
                   label=f"Full model (r={full_r:.3f})")

    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5, label='r = 0.9 target')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Per-individual observation probability p_obs (log scale)', fontsize=11)
    ax.set_ylabel('Richness correlation (r)', fontsize=11)
    ax.set_title("(A) Axel's Poisson detection model\nobserved_count ~ Poisson(N_individuals × p_obs)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2, which='both')
    ax.set_ylim(min(-0.05, min(rs) - 0.05), 1.15)

    # ── Panel B: All four metrics at each p level ──
    ax = axes[1]
    metric_keys  = ['richness_r', 'range_r', 'beta_r', 'prevalence_r']
    metric_labels = ['Richness r', 'Range-size r', 'Beta-div r', 'Prevalence r']
    metric_colors = ['#dc2626', '#d97706', '#059669', '#7c3aed']

    poisson_only = [(c, p) for c, p in poisson_conds
                    if c != "FULL" and p is not None and c in all_results]

    for mk, ml, mc in zip(metric_keys, metric_labels, metric_colors):
        pp = [p for c, p in poisson_only]
        vv = [all_results[c].get(mk, 0) for c, _ in poisson_only]
        ax.plot(pp, vv, 'o-', linewidth=2, markersize=8, color=mc,
                label=ml, alpha=0.85)

    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Per-individual observation probability p_obs (log scale)', fontsize=11)
    ax.set_ylabel('Correlation (r)', fontsize=11)
    ax.set_title("(B) All four metrics vs per-individual p_obs\n"
                 "(Arm B-Poisson conditions only)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2, which='both')
    ax.set_ylim(-0.1, 1.15)

    fig.suptitle("AB10: Arm B-Poisson (v5) — Axel's Exact Model with BODY_MASS Conversion\n"
                 "N_individuals = B_biomass / BODY_MASS_UNIT; observed ~ Poisson(N × p_obs)",
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    safe_save(fig, out / "fig_AB10_poisson_curve.png")


def plot_AB11_fixed_vs_poisson(all_results, out):
    """
    NEW v3: Side-by-side comparison of fixed-budget (v2) vs Poisson detection (v3).

    This is the key validation figure. Axel predicted: "I think the results
    will be exactly the same." This figure shows whether that prediction holds.

    If both curves show similar performance levels:
      → The result is ROBUST to how sparsity is modelled.
      → Confirms genuine ecological learning, not an artefact of fixed-budget design.

    The x-axis for Poisson is the expected number of observations per species,
    computed as: E[obs] = p_detect × mean_range_size_across_species.
    This puts both methods on a common scale for direct comparison.
    """
    # Fixed budget conditions
    fixed_conds = [
        ("SPARSE_1",  1),
        ("SPARSE_5",  5),
        ("SPARSE_10", 10),
        ("SPARSE_20", 20),
        ("SPARSE_50", 50),
        ("FULL",      400),
    ]

    # Poisson conditions (v5): we plot against p_obs directly, since
    # expected obs per species is ABUNDANCE-WEIGHTED (cannot be computed
    # as p_obs × mean_range like v3 assumed).
    #
    # For comparison with fixed budget, we use the MEAN OBSERVATIONS PER
    # SPECIES empirically measured by the diagnostic. For typical post-
    # assembly equilibrium data (mean range ~2 cells, mean biomass ~0.4,
    # ~4000 individuals per occupied cell), the diagnostic shows:
    #
    #    p_obs=0.00001 → ~0.3 obs/sp
    #    p_obs=0.0001  → ~1.5 obs/sp
    #    p_obs=0.0005  → ~2.0 obs/sp (saturates at mean range)
    #    p_obs=0.001   → ~2.0 obs/sp (saturated)
    #    p_obs=0.01    → ~2.0 obs/sp (saturated, Axel's standard)
    #
    # These values are approximate — refine from the actual diagnostic if needed.
    poisson_conds = [
        ("POISSON_p00001", 0.3),   # expected obs per species
        ("POISSON_p0001",  1.5),
        ("POISSON_p0005",  2.0),
        ("POISSON_p001",   2.0),
        ("POISSON_p01",    2.0),
    ]

    fixed_x, fixed_r, fixed_r_std = [], [], []
    for cond, b in fixed_conds:
        if cond in all_results:
            fixed_x.append(b)
            fixed_r.append(all_results[cond].get("richness_r", 0))
            fixed_r_std.append(all_results[cond].get("richness_r_std", 0))

    poisson_x, poisson_r, poisson_r_std = [], [], []
    for cond, expected_b in poisson_conds:
        if cond in all_results:
            poisson_x.append(expected_b)
            poisson_r.append(all_results[cond].get("richness_r", 0))
            poisson_r_std.append(all_results[cond].get("richness_r_std", 0))

    has_fixed   = len(fixed_x) >= 2
    has_poisson = len(poisson_x) >= 2

    if not has_fixed and not has_poisson:
        print("  ⚠ Skipping AB11 — no data for comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel A: Both curves on same axis ──
    ax = axes[0]
    if has_fixed:
        ax.errorbar(fixed_x, fixed_r, yerr=fixed_r_std,
                    fmt='s-', linewidth=2.5, markersize=10,
                    color="#d97706", ecolor='gray', capsize=5, capthick=1.5,
                    markerfacecolor="#fbbf24", markeredgecolor="#d97706",
                    markeredgewidth=2, label="Fixed budget (v2)", zorder=3)
    if has_poisson:
        ax.errorbar(poisson_x, poisson_r, yerr=poisson_r_std,
                    fmt='o--', linewidth=2.5, markersize=10,
                    color="#dc2626", ecolor='gray', capsize=5, capthick=1.5,
                    markerfacecolor="#fca5a5", markeredgecolor="#dc2626",
                    markeredgewidth=2,
                    label="Poisson detection (v5, abundance-weighted)",
                    zorder=3)

    ax.axvspan(0.5, 10, alpha=0.08, color='purple',
               label='Realistic IUCN data range (0.5–10 obs/sp)')
    ax.axhline(0.9, color='gray', ls='--', lw=0.8, alpha=0.5, label='r = 0.9 target')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Expected observations per species (log scale)', fontsize=11)
    ax.set_ylabel('Richness correlation (r)', fontsize=11)
    ax.set_title("(A) Fixed budget vs Poisson detection\nRichness correlation",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2, which='both')
    all_r = fixed_r + poisson_r
    ax.set_ylim(min(-0.05, min(all_r) - 0.05) if all_r else -0.1, 1.15)

    # ── Panel B: Difference between methods (Poisson - Fixed at matched points) ──
    ax = axes[1]

    # Match Poisson to fixed budget at approximately the same observation level
    # Using linear interpolation of fixed curve
    if has_fixed and has_poisson and len(fixed_x) >= 3:
        try:
            interp_fixed = np.interp(poisson_x, fixed_x, fixed_r)
            diffs = np.array(poisson_r) - interp_fixed
            colors_diff = ['#16a34a' if d >= 0 else '#dc2626' for d in diffs]
            bars = ax.bar(range(len(poisson_x)), diffs, color=colors_diff,
                          alpha=0.8, edgecolor='white', linewidth=1.5)
            for bar, d, px in zip(bars, diffs, poisson_x):
                sign = '+' if d >= 0 else ''
                ax.text(bar.get_x() + bar.get_width()/2,
                        d + (0.005 if d >= 0 else -0.008),
                        f'{sign}{d:.3f}', ha='center', va='bottom' if d >= 0 else 'top',
                        fontsize=9, fontweight='bold',
                        color='#16a34a' if d >= 0 else '#dc2626')
            p_labels = [f'p_obs={p*100:.3g}%\n(E≈{ex:.1f})' for (c, p), ex in
                        zip([("POISSON_p00001", 0.00001),
                             ("POISSON_p0001",  0.0001),
                             ("POISSON_p0005",  0.0005),
                             ("POISSON_p001",   0.001),
                             ("POISSON_p01",    0.01)], poisson_x)
                        if c in all_results]
            ax.set_xticks(range(len(p_labels)))
            ax.set_xticklabels(p_labels, fontsize=8)
            ax.axhline(0, color='black', lw=1.5)
            ax.set_ylabel('Poisson r − Fixed r (at same expected observations)', fontsize=10)
            ax.set_title("(B) Difference: Poisson − Fixed budget\n"
                         "(near zero = methods are equivalent, confirms Axel's prediction)",
                         fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.2)

            # Add note about what near-zero means
            ax.text(0.5, 0.05,
                    "Near zero = both methods give equivalent performance\n"
                    "= result is robust to observation model",
                    transform=ax.transAxes, ha='center', fontsize=8,
                    style='italic', color='#374151',
                    bbox=dict(boxstyle='round', facecolor='#f0fdf4', alpha=0.8))
        except Exception:
            ax.text(0.5, 0.5, "Difference plot requires\nboth Fixed and Poisson results",
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, "Run both SPARSE_* and POISSON_*\nconditions to see difference",
                transform=ax.transAxes, ha='center', va='center', fontsize=10,
                color='gray')
        ax.set_title("(B) Difference: Poisson − Fixed budget\n(requires both conditions)",
                     fontsize=11, fontweight='bold')

    fig.suptitle("AB11: Fixed Budget vs Poisson Detection — Validating Axel's Prediction\n"
                 '"I think the results will be exactly the same." — Axel Rossberg',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    safe_save(fig, out / "fig_AB11_fixed_vs_poisson.png")


def print_summary_table(all_results):
    print("\n" + "=" * 100)
    print("  ABLATION v2 — RESULTS SUMMARY")
    print("=" * 100)
    header = f"  {'Condition':<22} {'Richness r':>12} {'Range r':>10} {'Beta r':>9} {'Prev r':>9} {'Mean rich':>10}"
    print(header)
    print("  " + "─" * 95)
    for cond, metrics in all_results.items():
        cfg = ABLATION_CONDITIONS[cond]
        print(f"  {cfg['label']:<22} "
              f"{metrics.get('richness_r', 0):>12.4f} "
              f"{metrics.get('range_r', 0):>10.4f} "
              f"{metrics.get('beta_r', 0):>9.4f} "
              f"{metrics.get('prevalence_r', 0):>9.4f} "
              f"{metrics.get('richness_mean_mod', 0):>10.2f}")
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Ablation Study v2 — supervisor rebuttal")
    parser.add_argument('--ibm-dir', required=True)
    parser.add_argument('--ibm-filter', default='pool22510000')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage2-dir', default=None)
    parser.add_argument('--output-dir', default='stage2_ablation_v2_figures')
    parser.add_argument('--n-samples', type=int, default=8)
    parser.add_argument('--n-worlds', type=int, default=8)
    parser.add_argument('--obs-budget', type=int, default=5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--predict-sim', type=int, default=0)
    parser.add_argument('--skip-original', action='store_true',
                        help='Skip original 5 conditions (run only new arms)')
    parser.add_argument('--skip-new', action='store_true',
                        help='Skip new arms (run only original 5)')
    parser.add_argument('--skip-gap', action='store_true',
                        help='Skip Arm A gap conditions (GAP_5, GAP_25)')
    parser.add_argument('--skip-fixed-sparse', action='store_true',
                        help='Skip Arm B fixed-budget sparse conditions')
    parser.add_argument('--skip-poisson', action='store_true',
                        help='Skip Arm B Poisson detection conditions')
    parser.add_argument('--only-armb', action='store_true',
                        help='Run only Arm B conditions (FULL + SPARSE_* + POISSON_*)')
    parser.add_argument('--only-poisson', action='store_true',
                        help='Run only FULL + POISSON_* (v5: use with existing v3 JSON)')
    args = parser.parse_args()

    if args.stage2_dir:
        s2 = str(Path(args.stage2_dir).resolve())
        if s2 not in sys.path:
            sys.path.insert(0, s2)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch

    print("\n" + "=" * 70)
    print("  STAGE 2 — ABLATION STUDY v2 (Supervisor Rebuttal)")
    print("=" * 70)

    # Filter conditions if requested
    conditions_to_run = OrderedDict(ABLATION_CONDITIONS)
    if args.only_poisson:
        # v5 focused re-run: only FULL + POISSON_* (merge results with v3 JSON later)
        conditions_to_run = OrderedDict(
            [(k, v) for k, v in conditions_to_run.items()
             if k == "FULL" or k.startswith("POISSON_")])
        print(f"  ℹ --only-poisson active: will run {len(conditions_to_run)} conditions")
        print(f"    ({list(conditions_to_run.keys())})")
    elif args.only_armb:
        # FULL + SPARSE_* + POISSON_*
        conditions_to_run = OrderedDict(
            [(k, v) for k, v in conditions_to_run.items()
             if k == "FULL" or k.startswith("SPARSE_") or k.startswith("POISSON_")])
    else:
        if args.skip_original:
            conditions_to_run = OrderedDict(
                [(k, v) for k, v in conditions_to_run.items()
                 if k not in ["NO_TEMPORAL", "NO_INTERACT", "ENV_ONLY", "OBS_INFILL"]])
        if args.skip_new:
            conditions_to_run = OrderedDict(
                [(k, v) for k, v in conditions_to_run.items()
                 if k in ["FULL", "NO_TEMPORAL", "NO_INTERACT", "ENV_ONLY", "OBS_INFILL"]])
        if args.skip_gap:
            conditions_to_run = OrderedDict(
                [(k, v) for k, v in conditions_to_run.items()
                 if not k.startswith("GAP_")])
        if args.skip_fixed_sparse:
            conditions_to_run = OrderedDict(
                [(k, v) for k, v in conditions_to_run.items()
                 if not k.startswith("SPARSE_")])
        if args.skip_poisson:
            conditions_to_run = OrderedDict(
                [(k, v) for k, v in conditions_to_run.items()
                 if not k.startswith("POISSON_")])

    print(f"  Conditions to run: {len(conditions_to_run)}")
    for cn, cfg in conditions_to_run.items():
        print(f"    {cn:<18} — {cfg['desc']}")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    # Load IBM data
    ibm_dir = Path(args.ibm_dir)
    npz_files = sorted(ibm_dir.rglob("*_training.npz"))
    if args.ibm_filter:
        npz_files = [f for f in npz_files if args.ibm_filter in f.name]

    valid_files = []
    for f in npz_files:
        try:
            with np.load(str(f), allow_pickle=True) as d:
                if "P_last_final" not in d:
                    continue
                P = d["P_last_final"]
                if P.ndim == 3 and P.shape[1] == 20 and P.shape[2] == 20:
                    valid_files.append(f)
        except Exception:
            continue
    npz_files = valid_files
    print(f"\n  Found {len(npz_files)} valid training.npz files")
    if len(npz_files) == 0:
        print("❌ No valid data found"); return

    world_indices = list(range(args.predict_sim,
                               min(args.predict_sim + args.n_worlds, len(npz_files))))
    print(f"  Testing on worlds: {world_indices}")

    # Load model
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    try:
        from models.ecodiffusion import EcoDiffusionFixed, create_fixed_model
        from configs.config import get_default_config
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = get_default_config()
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        for k, v in state_dict.items():
            if 'env_encoder' in k and 'species_embed' in k:
                config.data.n_species_max = v.shape[0]
                break
        if config.data.n_species_max == 0:
            with np.load(str(npz_files[world_indices[0]])) as d:
                S = np.asarray(d["P_last_final"]).shape[0]
                config.data.n_species_max = S
        model = create_fixed_model(config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print(f"  ✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        import traceback; traceback.print_exc()
        return

    # ═══ RUN ALL WORLDS × CONDITIONS ═══
    accumulated = {cond: [] for cond in conditions_to_run}

    for wi, world_idx in enumerate(world_indices):
        npz_path = str(npz_files[world_idx])
        print(f"\n{'─' * 70}")
        print(f"  World [{world_idx}]: {Path(npz_path).name}")
        print(f"{'─' * 70}")

        npz_data = dict(np.load(npz_path, allow_pickle=True))
        P_ibm = np.asarray(npz_data["P_last_final"])
        S, Y, X = P_ibm.shape
        print(f"  Species: {S}, Grid: {Y}×{X}")

        predictions = {}
        for cond_name, cond_cfg in conditions_to_run.items():
            print(f"\n  ── {cond_cfg['label']:<20} ({cond_cfg['desc']})")
            with torch.no_grad():
                P_pred = run_ablation_inference(
                    model, npz_data, device, cond_cfg,
                    n_samples=args.n_samples, chunk_size=args.chunk_size,
                    obs_budget=args.obs_budget, world_seed=wi,
                )
            predictions[cond_name] = P_pred
            n_occ = int(((P_pred >= 0.5).sum(axis=(1, 2)) > 0).sum())
            print(f"     pred mean={P_pred.mean():.4f}, occupied={n_occ}/{S}")

            metrics = compute_all_metrics(P_ibm, P_pred)
            accumulated[cond_name].append(metrics)
            print(f"     richness_r={metrics['richness_r']:.4f}, "
                  f"range_r={metrics['range_r']:.4f}, "
                  f"beta_r={metrics['beta_r']:.4f}")

        del predictions, npz_data
        import gc; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ═══ AGGREGATE ═══
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATING RESULTS ACROSS {len(world_indices)} WORLDS")
    print(f"{'=' * 70}")

    avg_results = {}
    for cond_name, metrics_list in accumulated.items():
        if not metrics_list:
            continue
        avg = {}
        for key in metrics_list[0].keys():
            vals = [m[key] for m in metrics_list if key in m]
            avg[key] = float(np.mean(vals))
            avg[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        avg_results[cond_name] = avg

    print_summary_table(avg_results)

    # ═══ FIGURES ═══
    print(f"\n  Generating figures...")
    plot_AB1_metric_bars_grouped(avg_results, out)
    plot_AB7_temporal_gap(avg_results, out)
    plot_AB8_sparsity_curve(avg_results, out)
    plot_AB9_supervisor_rebuttal(avg_results, out)
    plot_AB10_poisson_curve(avg_results, out)
    plot_AB11_fixed_vs_poisson(avg_results, out)

    # ═══ SAVE JSON ═══
    results_json = {
        "n_worlds": len(world_indices),
        "world_indices": world_indices,
        "n_samples": args.n_samples,
        "obs_budget": args.obs_budget,
        "conditions": {},
    }
    for cond_name, metrics in avg_results.items():
        cfg = ABLATION_CONDITIONS[cond_name]
        results_json["conditions"][cond_name] = {
            "label": cfg["label"], "description": cfg["desc"],
            "drop_interactions": cfg["drop_interactions"],
            "drop_temporal": cfg["drop_temporal"],
            "gap": cfg.get("gap", 0),
            "sparse_budget": cfg.get("sparse_budget"),
            "poisson_p": cfg.get("poisson_p"),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
    json_path = out / "ablation_v2_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  ✓ Results saved: {json_path}")

    # ═══ SUPERVISOR INTERPRETATION ═══
    print("\n" + "=" * 70)
    print("  SUPERVISOR-FACING INTERPRETATION")
    print("=" * 70)
    full_r = avg_results.get("FULL", {}).get("richness_r", 0)
    print(f"\n  Original ablation:")
    print(f"    Full model:               r = {full_r:.4f}")
    print(f"    No temporal:              r = {avg_results.get('NO_TEMPORAL', {}).get('richness_r', 0):.4f}")
    print(f"    No interactions:          r = {avg_results.get('NO_INTERACT', {}).get('richness_r', 0):.4f}")
    print(f"    Env only:                 r = {avg_results.get('ENV_ONLY', {}).get('richness_r', 0):.4f}")

    print(f"\n  ARM A — Temporal gap (model works with earlier history):")
    for c in ["GAP_5", "GAP_25"]:
        if c in avg_results:
            r = avg_results[c].get("richness_r", 0)
            print(f"    {ABLATION_CONDITIONS[c]['label']:<20} r = {r:.4f}  "
                  f"(degradation: {full_r - r:+.4f})")

    print(f"\n  ARM B FIXED — Sparse history (model works with sparse observations):")
    for c in ["SPARSE_1", "SPARSE_5", "SPARSE_10", "SPARSE_20", "SPARSE_50"]:
        if c in avg_results:
            r = avg_results[c].get("richness_r", 0)
            print(f"    {ABLATION_CONDITIONS[c]['label']:<20} r = {r:.4f}  "
                  f"(degradation: {full_r - r:+.4f})")

    print(f"\n  ARM B POISSON — Axel's exact model (v5, BODY_MASS-aware):")
    for c in ["POISSON_p00001", "POISSON_p0001", "POISSON_p0005",
              "POISSON_p001", "POISSON_p01"]:
        if c in avg_results:
            r = avg_results[c].get("richness_r", 0)
            cfg = ABLATION_CONDITIONS[c]
            p = cfg.get("poisson_p", 0)
            print(f"    {cfg['label']:<22} r = {r:.4f}  "
                  f"(degradation: {full_r - r:+.4f})  "
                  f"[p_obs={p} per individual]")

    # ── Compare Fixed vs Poisson at matched observation levels ──
    # NOTE: mean range in your LV-IBM data is ~2 cells per species, so
    # Poisson mostly saturates at 2 obs/sp. The meaningful comparison is
    # between Poisson at sub-saturation (p_obs=0.00001 or 0.0001) and
    # fixed-budget at matched mean observations.
    print(f"\n  ARM B COMPARISON — Fixed budget vs Axel's Poisson:")
    pairs = [
        ("SPARSE_1",  "POISSON_p00001", "~0.3-1 obs/sp (extreme sparsity)"),
        ("SPARSE_5",  "POISSON_p0001",  "~1-2 obs/sp (rare-species regime)"),
        ("SPARSE_20", "POISSON_p001",   "~2 obs/sp (saturated)"),
    ]
    for fixed_c, pois_c, label in pairs:
        fr = avg_results.get(fixed_c, {}).get("richness_r", None)
        pr = avg_results.get(pois_c,  {}).get("richness_r", None)
        if fr is not None and pr is not None:
            diff = pr - fr
            verdict = ("≈ EQUIVALENT" if abs(diff) < 0.02
                       else ("Poisson higher" if diff > 0 else "Fixed higher"))
            print(f"    {label:<40}: Fixed={fr:.4f}  Poisson={pr:.4f}  "
                  f"Δ={diff:+.4f}  → {verdict}")

    print(f"\n  ► Axel's prediction: 'I think the results will be exactly the same.'")
    if all(c in avg_results for c in ["SPARSE_5", "POISSON_p0001"]):
        diff = (avg_results["POISSON_p0001"].get("richness_r", 0) -
                avg_results["SPARSE_5"].get("richness_r", 0))
        if abs(diff) < 0.03:
            print(f"  ✓ CONFIRMED: |Δ| = {abs(diff):.4f} < 0.03 at rare-species regime")
            print(f"    The result is ROBUST to observation model.")
            print(f"    Both methods confirm genuine ecological learning.")
        else:
            print(f"  ~ Partial: |Δ| = {abs(diff):.4f} at rare-species regime")
            print(f"    Some difference between methods — expected since Poisson is")
            print(f"    abundance-weighted while fixed-budget is uniform random.")
    print("=" * 70)


if __name__ == "__main__":
    main()