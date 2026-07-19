#!/usr/bin/env python3
"""
===============================================================================
POISSON DETECTION MODEL — CORRECTED VERSION
===============================================================================

ROOT CAUSE OF v3 BUG:
  v3 script applied p_detect per CELL directly. This ignored that ecological
  observation probability is per INDIVIDUAL, not per cell. With mean range ~2
  cells per species in the data, this created a degenerate regime where most
  species received zero observations.

  At p=0.01, approximately 98% of species had zero detections and the model
  predicted absence for them. Only 52 of 3265 species were predicted present.

AXEL'S ACTUAL MODEL:
  "For each individual, 1% observation probability. Observed count per
   species per cell ~ Poisson(N_individuals × p_obs). A cell is detected
   if observed count >= 1."

CORRECT IMPLEMENTATION — three strategies depending on available data:

  STRATEGY A (BEST — uses biomass):
    If B_last_final (biomass per species per cell) is available:
      expected_count[s,y,x] = B[s,y,x] * p_obs
      detected[s,y,x] = 1 if Poisson(expected_count) >= 1
    This is Axel's exact gold-standard model.

  STRATEGY B (FALLBACK — uses range-weighted p):
    If only binary P_t is available:
      For each species, scale p by range_size so that expected observations
      per species = target_obs (e.g., 5, 10, 20), matching fixed-budget levels.
      effective_p[s] = min(1.0, target_obs / range_size[s])
      detected[s,y,x] = 1 if (P[s,y,x] > 0) AND (random() < effective_p[s])
    This matches fixed-budget behavior while preserving the per-cell
    stochasticity of detection. Species-specific p reflects that rare species
    are proportionally harder to survey completely.

  STRATEGY C (NAIVE — broken, DO NOT USE):
    Uniform p per cell (what v3 did). This collapses rare species entirely.

USAGE:
  Add these functions to stage2_ablation_study_v3.py, replacing the existing
  sparsify_history_poisson function. Replace the POISSON_* conditions in
  ABLATION_CONDITIONS with the new ones below.
===============================================================================
"""

import numpy as np


# ==============================================================================
# STRATEGY A — BIOMASS-WEIGHTED POISSON (AXEL'S EXACT MODEL)
# ==============================================================================

def sparsify_history_poisson_biomass(Pt, B_t, p_obs, rng_seed=42):
    """
    Axel's exact gold-standard ecological detection model.

    For each species s and each cell (y,x), the observed count is drawn from
    Poisson(B[s,y,x] * p_obs) where B is biomass (proxy for individual count).
    The cell is detected if observed count >= 1.

    This captures the correct ecological behavior:
      - Abundant species (high biomass) are reliably detected
      - Rare species (low biomass) are frequently missed
      - Detection probability per cell scales with local abundance

    Pt: (T, S, Y, X) binary presence/absence
    B_t: (T, S, Y, X) biomass per species per cell (can be None)
    p_obs: float in (0, 1) — observation probability per individual
    """
    rng = np.random.default_rng(rng_seed)
    Pt_sparse = np.zeros_like(Pt)

    B_last = B_t[-1]                      # (S, Y, X) biomass at final snapshot
    expected_counts = B_last * p_obs      # (S, Y, X) expected observations
    # Poisson draw per cell
    observed_counts = rng.poisson(expected_counts)
    # Detected if count >= 1
    detected = (observed_counts >= 1).astype(np.float32)

    Pt_sparse[-1] = detected
    return Pt_sparse


# ==============================================================================
# STRATEGY B — RANGE-WEIGHTED POISSON (FALLBACK IF NO BIOMASS)
# ==============================================================================

def sparsify_history_poisson_range_weighted(Pt, target_obs_per_sp, rng_seed=42):
    """
    Range-weighted detection: each species receives approximately `target_obs_per_sp`
    observations in expectation, regardless of range size.

    For a species with range_size R, set effective p = min(1, target_obs / R).
    Then each occupied cell is detected independently with that probability.

    This produces the correct expected observation count per species while
    preserving per-cell detection stochasticity. It also avoids the degenerate
    regime where rare species get zero observations.

    This is the correct comparison point for fixed-budget SPARSE_B conditions:
      target_obs = B gives the same expected observations per species, but
      with stochastic per-cell detection instead of fixed count.

    Pt: (T, S, Y, X) binary presence/absence
    target_obs_per_sp: float — expected observations per present species
    """
    rng = np.random.default_rng(rng_seed)
    T, S, Y, X = Pt.shape
    Pt_sparse = np.zeros_like(Pt)

    last = Pt[-1]                          # (S, Y, X)
    sparse_last = np.zeros_like(last)

    for s in range(S):
        occupied = np.argwhere(last[s] > 0)
        n_occ = len(occupied)
        if n_occ == 0:
            continue
        # Species-specific detection probability
        # Capped at 1.0 to avoid oversampling rare species
        effective_p = min(1.0, target_obs_per_sp / n_occ)
        # Bernoulli per occupied cell
        detected = rng.random(n_occ) < effective_p
        for idx, is_detected in enumerate(detected):
            if is_detected:
                y, x = occupied[idx]
                sparse_last[s, y, x] = 1.0

    Pt_sparse[-1] = sparse_last
    return Pt_sparse


# ==============================================================================
# DISPATCH FUNCTION — chooses strategy based on available data
# ==============================================================================

def sparsify_history_poisson_correct(Pt, B_t=None, p_obs=None,
                                     target_obs_per_sp=None, rng_seed=42):
    """
    Dispatch to correct Poisson strategy based on available data.

    Priority:
      1. If B_t (biomass) and p_obs provided → Strategy A (Axel's exact model)
      2. Elif target_obs_per_sp provided → Strategy B (range-weighted)
      3. Else → error

    This replaces the broken uniform-p Poisson in v3.
    """
    if B_t is not None and p_obs is not None:
        return sparsify_history_poisson_biomass(Pt, B_t, p_obs, rng_seed)
    elif target_obs_per_sp is not None:
        return sparsify_history_poisson_range_weighted(
            Pt, target_obs_per_sp, rng_seed)
    else:
        raise ValueError(
            "Must provide either (B_t, p_obs) for biomass-weighted Poisson "
            "or target_obs_per_sp for range-weighted Poisson"
        )


# ==============================================================================
# UPDATED ABLATION CONDITIONS
# ==============================================================================

POISSON_CONDITIONS_CORRECTED = {
    # Strategy A conditions (require biomass) — run these if B_last_final exists
    "POISSON_BIO_p001": {
        "drop_interactions": False, "drop_temporal": False, "drop_obs": True,
        "gap": 0, "sparse_budget": None,
        "poisson_p": 0.01, "poisson_strategy": "biomass",
        "label": "Poisson(biomass) p=1%",
        "color": "#fca5a5",
        "desc": "Biomass-weighted Poisson, p_obs=1% per individual"
    },
    "POISSON_BIO_p005": {
        "drop_interactions": False, "drop_temporal": False, "drop_obs": True,
        "gap": 0, "sparse_budget": None,
        "poisson_p": 0.05, "poisson_strategy": "biomass",
        "label": "Poisson(biomass) p=5%",
        "color": "#f87171",
        "desc": "Biomass-weighted Poisson, p_obs=5% per individual"
    },
    "POISSON_BIO_p010": {
        "drop_interactions": False, "drop_temporal": False, "drop_obs": True,
        "gap": 0, "sparse_budget": None,
        "poisson_p": 0.10, "poisson_strategy": "biomass",
        "label": "Poisson(biomass) p=10%",
        "color": "#ef4444",
        "desc": "Biomass-weighted Poisson, p_obs=10% per individual"
    },

    # Strategy B conditions (no biomass needed) — match fixed budget levels
    "POISSON_RW_5":  {
        "drop_interactions": False, "drop_temporal": False, "drop_obs": True,
        "gap": 0, "sparse_budget": None,
        "target_obs": 5, "poisson_strategy": "range_weighted",
        "label": "Poisson(RW) ~5/sp",
        "color": "#fbbf24",
        "desc": "Range-weighted Poisson, target ~5 obs/species (matches SPARSE_5)"
    },
    "POISSON_RW_10": {
        "drop_interactions": False, "drop_temporal": False, "drop_obs": True,
        "gap": 0, "sparse_budget": None,
        "target_obs": 10, "poisson_strategy": "range_weighted",
        "label": "Poisson(RW) ~10/sp",
        "color": "#f59e0b",
        "desc": "Range-weighted Poisson, target ~10 obs/species (matches SPARSE_10)"
    },
    "POISSON_RW_20": {
        "drop_interactions": False, "drop_temporal": False, "drop_obs": True,
        "gap": 0, "sparse_budget": None,
        "target_obs": 20, "poisson_strategy": "range_weighted",
        "label": "Poisson(RW) ~20/sp",
        "color": "#d97706",
        "desc": "Range-weighted Poisson, target ~20 obs/species (matches SPARSE_20)"
    },
}


# ==============================================================================
# INTEGRATION: how to modify build_ablation_condition
# ==============================================================================

INTEGRATION_PATCH = """
To integrate these corrections into stage2_ablation_study_v3.py:

1. Replace the existing sparsify_history_poisson function with the three new
   functions above.

2. In build_ablation_condition(), update the temporal history section:

    # ARM B POISSON — Axel's gold-standard detection model (CORRECTED)
    poisson_strategy = None
    # Read strategy from condition config (passed through cond_cfg)
    if poisson_p is not None or target_obs is not None:
        if poisson_strategy == "biomass":
            # Read biomass if available
            B_t = npz_data.get("B_t", None)
            if B_t is not None:
                B_t = np.asarray(B_t).astype(np.float32)
                if species_subset is not None:
                    B_t = B_t[:, species_subset, :, :]
                Pt = sparsify_history_poisson_biomass(Pt, B_t, poisson_p,
                                                     rng_seed=rng_seed)
            else:
                # No biomass — fall back to range-weighted
                print(f"    WARNING: No B_t in data, using range-weighted fallback")
                # Convert p to target observations using mean range (computed below)
                mean_range = Pt[-1].sum(axis=(1,2)).mean()
                target_obs = poisson_p * mean_range
                Pt = sparsify_history_poisson_range_weighted(
                    Pt, target_obs, rng_seed=rng_seed)
        elif poisson_strategy == "range_weighted":
            Pt = sparsify_history_poisson_range_weighted(
                Pt, target_obs, rng_seed=rng_seed)

3. In run_ablation_inference(), pass the strategy:

    condition = build_ablation_condition(
        ...,
        poisson_p=condition_cfg.get("poisson_p"),
        poisson_strategy=condition_cfg.get("poisson_strategy"),
        target_obs=condition_cfg.get("target_obs"),
        ...
    )

4. Replace POISSON_p001 through POISSON_p050 in ABLATION_CONDITIONS with the
   POISSON_BIO_* and POISSON_RW_* conditions above.
"""


# ==============================================================================
# DIAGNOSTIC FUNCTION — run this on your existing data FIRST
# ==============================================================================

def diagnose_observation_counts(npz_path):
    """
    Before re-running the ablation, diagnose the actual observation counts
    produced by each method on real data. This tells you whether biomass is
    available and what the true expected obs per species is.
    """
    import numpy as np
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    print(f"  Keys in npz: {keys}")

    P_t = np.asarray(npz["P_t"])
    if P_t.ndim == 3:
        T, S, N = P_t.shape
        side = int(np.sqrt(N))
        P_t = P_t.reshape(T, S, side, side)
    T, S, Y, X = P_t.shape
    P_last = (P_t[-1] > 0).astype(np.int32)

    range_sizes = P_last.sum(axis=(1, 2))
    present_species = range_sizes > 0
    mean_range = range_sizes[present_species].mean()
    median_range = np.median(range_sizes[present_species])

    print(f"\n  Data dimensions: T={T}, S={S}, Y={Y}, X={X}")
    print(f"  Present species: {present_species.sum()} of {S}")
    print(f"  Range size — mean: {mean_range:.2f}, median: {median_range:.2f}")
    print(f"  Range size — min: {range_sizes[present_species].min()}, "
          f"max: {range_sizes[present_species].max()}")

    has_biomass = "B_t" in keys or "B_last_final" in keys
    print(f"\n  Biomass available: {has_biomass}")
    if has_biomass:
        B_key = "B_t" if "B_t" in keys else "B_last_final"
        B = np.asarray(npz[B_key])
        if B.ndim == 4:
            B_last = B[-1]
        else:
            B_last = B
        mean_biomass = B_last[P_last > 0].mean() if B_last[P_last > 0].size > 0 else 0
        print(f"  Mean biomass per occupied cell: {mean_biomass:.2f}")
        print(f"  At p_obs=0.01, expected obs per occupied cell: {mean_biomass*0.01:.4f}")
        print(f"  At p_obs=0.10, expected obs per occupied cell: {mean_biomass*0.10:.4f}")

    # Diagnose what different strategies would produce
    print(f"\n  Simulated obs per species under each strategy:")
    print(f"  {'Strategy':<30} {'Mean obs/sp':<14} {'% species w/ >=1 obs'}")
    print(f"  {'-'*60}")

    rng = np.random.default_rng(42)

    # Uniform p (broken v3)
    for p in [0.01, 0.10]:
        total_obs = 0
        species_with_obs = 0
        for s in range(S):
            if not present_species[s]:
                continue
            n_occ = range_sizes[s]
            detected = rng.random(n_occ) < p
            n_det = detected.sum()
            total_obs += n_det
            if n_det >= 1:
                species_with_obs += 1
        print(f"  Uniform p={p:4.2f} (v3 broken)       "
              f"{total_obs/present_species.sum():>10.3f}    "
              f"{100*species_with_obs/present_species.sum():.1f}%")

    # Range-weighted
    for target in [5, 10, 20]:
        total_obs = 0
        species_with_obs = 0
        for s in range(S):
            if not present_species[s]:
                continue
            n_occ = range_sizes[s]
            eff_p = min(1.0, target / n_occ)
            detected = rng.random(n_occ) < eff_p
            n_det = detected.sum()
            total_obs += n_det
            if n_det >= 1:
                species_with_obs += 1
        print(f"  Range-weighted target={target:2d}          "
              f"{total_obs/present_species.sum():>10.3f}    "
              f"{100*species_with_obs/present_species.sum():.1f}%")

    # Biomass-weighted (if available)
    if has_biomass:
        B_key = "B_t" if "B_t" in keys else "B_last_final"
        B = np.asarray(npz[B_key])
        if B.ndim == 4:
            B_last = B[-1]
        else:
            B_last = B
        for p in [0.01, 0.05, 0.10]:
            expected = B_last * p
            observed = rng.poisson(expected)
            detected = observed >= 1
            total_obs_per_sp = detected.sum(axis=(1, 2))
            species_with_obs = (total_obs_per_sp >= 1).sum()
            total_obs = total_obs_per_sp[present_species].sum()
            print(f"  Biomass-weighted p={p:4.2f}          "
                  f"{total_obs/present_species.sum():>10.3f}    "
                  f"{100*species_with_obs/present_species.sum():.1f}%")

    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python poisson_corrected.py <path_to_training.npz>")
        print("       Run this first to diagnose your data before re-running ablation")
        sys.exit(1)
    diagnose_observation_counts(sys.argv[1])