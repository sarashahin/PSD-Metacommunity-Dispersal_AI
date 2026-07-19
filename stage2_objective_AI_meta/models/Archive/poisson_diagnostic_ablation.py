#!/usr/bin/env python3
"""
===============================================================================
POISSON DETECTION — CORRECTED WITH BODY_MASS CONVERSION (v5 FINAL)
===============================================================================

DIAGNOSIS FROM SOURCE CODE REVIEW:
  config.py defines:  BODY_MASS = 1e-4
  models_ibm.py uses: B = self.N * BODY_MASS
  THRESHOLD = 10 * BODY_MASS  (species is "present" if >= 10 individuals)

THEREFORE biomass and individual count are related by:
  N_individuals = B / BODY_MASS = B * 10000

This is the conversion that both v3 (uniform p per cell) and v4 (p applied
directly to biomass) missed. Axel's "1% per individual" must be applied to
N_individuals, not to B_biomass.

AXEL'S EXACT MODEL (now correctly implemented):

  For each species s and cell (y, x):
    N_individuals[s, y, x] = B_last[s, y, x] / BODY_MASS
    expected_observations[s, y, x] = N_individuals[s, y, x] * p_obs
    observed_count[s, y, x] ~ Poisson(expected_observations[s, y, x])
    detected[s, y, x] = 1 if observed_count >= 1 else 0

This is Axel's verbatim model ("for each individual there is a p observation
probability"). The cell is detected with probability 1 - exp(-N * p_obs).

===============================================================================
ECOLOGICAL REGIME ANALYSIS (based on your diagnostic + code constants):
===============================================================================

Mean biomass per occupied cell: 0.4   →  ~4,000 individuals per cell
Threshold presence: B = 0.001         →  10 individuals (minimum)
Max biomass per cell: 1.2             →  ~12,000 individuals per cell
Mean biomass per species (total):     0.84  →  ~8,400 individuals

At Axel's suggested p_obs = 0.01 (1%):
  Cells with >= 100 individuals (B >= 0.01) are detected ~63% of the time
  Cells with >= 1000 individuals (B >= 0.1) are detected ~100% of the time
  Since mean B per occupied cell ≈ 0.4 (4000 individuals), p_obs=0.01 is
  already near-saturation for typical cells.

Informative regime for YOUR data is p_obs ∈ [0.0001, 0.005].
At p_obs = 0.0001, expected count in mean cell = 0.4:
  P(detect) per cell = 1 - exp(-0.4) ≈ 33%
  Mean observations per species (mean range ~2 cells) ≈ 0.66

At p_obs = 0.001, expected count in mean cell = 4:
  P(detect) per cell = 1 - exp(-4) ≈ 98%
  Most cells detected, but rare species with low B still missed.

===============================================================================
RECOMMENDED ABLATION CONDITIONS:
===============================================================================

Five Poisson conditions spanning the informative regime for your data:

  POISSON_p00001  p_obs = 0.0001  (1 in 10,000 per individual)
                  Expected: low detection, matches extreme data scarcity
                  Mean obs/sp ≈ 0.6, ~30-40% species detected

  POISSON_p00005  p_obs = 0.0005
                  Expected: moderate detection, ~70% species detected
                  Mean obs/sp ≈ 1.6

  POISSON_p0001   p_obs = 0.001  (1 in 1,000 per individual)
                  Expected: good detection of abundant species
                  Mean obs/sp ≈ 2.0 (saturated by range limit of ~2)

  POISSON_p0005   p_obs = 0.005
                  Expected: near-full observation of occupied cells
                  Mean obs/sp ≈ 2.0 (range-limited)

  POISSON_p001    p_obs = 0.01  (AXEL'S STANDARD "1% PER INDIVIDUAL")
                  Expected: saturated for your data
                  Mean obs/sp ≈ 2.0 (range-limited)

===============================================================================
CRITICAL LIMITATION NOTE:
===============================================================================

Because mean range in your IBM data is only ~2 cells per species, ALL
observation schemes (fixed budget AND Poisson) saturate at ~2 obs/species.
You cannot distinguish between observation models at observation budgets
larger than the mean range.

The real test of Axel's model vs fixed-budget is therefore NOT about how
many observations per species, but about WHICH CELLS are observed:
  - Fixed budget: uniform random sampling across occupied cells
  - Poisson(N * p): abundance-biased — high-biomass cells detected first

This is a qualitatively different ecological scenario that IS testable
in your data, even though the mean obs/sp is similar.

The scientific question becomes:
  "Does MetaDiffusion recover community structure equally well when
   observations are biased toward high-abundance cells (Poisson)
   versus uniformly sampled (fixed budget)?"

If the model performs similarly under both → robust to observation bias.
If Poisson is worse → rare low-abundance cells contain information
lost under abundance-biased detection.

This is a publishable result regardless of which way it goes.
===============================================================================
"""

import numpy as np

# Constants from config.py (matched exactly to your simulation code)
BODY_MASS_UNIT = 1e-4  # from config.py — DO NOT CHANGE without matching config


def sparsify_history_poisson_axel(Pt, B_last, p_obs,
                                   body_mass=BODY_MASS_UNIT, rng_seed=42):
    """
    Axel's gold-standard abundance-weighted Poisson detection model.
    CORRECTLY converts biomass to individual counts using BODY_MASS.

    Model:
      N_individuals[s, y, x] = B_last[s, y, x] / body_mass
      observed_count ~ Poisson(N_individuals * p_obs)
      detected = (observed_count >= 1)

    All past temporal snapshots are zeroed.

    Parameters
    ----------
    Pt : ndarray, shape (T, S, Y, X)
        Binary presence/absence time series. Only used for shape reference.
    B_last : ndarray, shape (S, Y, X)
        Biomass at final snapshot. In units where B = N * body_mass.
    p_obs : float in (0, 1]
        Per-individual observation probability (Axel's 1% → p_obs=0.01).
    body_mass : float
        Conversion factor from biomass to individual count.
        Default matches config.py (BODY_MASS = 1e-4).
    rng_seed : int
        For reproducibility.

    Returns
    -------
    Pt_sparse : ndarray, shape (T, S, Y, X)
        Sparsified history. All snapshots zeroed except last, which contains
        1 at cells where at least one individual was observed.
    """
    rng = np.random.default_rng(rng_seed)
    Pt_sparse = np.zeros_like(Pt)

    # Convert biomass → individuals using the SAME body_mass used in simulation
    # N[s, y, x] is now in individual count units, matching Axel's description
    N_individuals = B_last.astype(np.float64) / body_mass

    # Axel's exact model: observed_count ~ Poisson(N × p_obs)
    expected_counts = N_individuals * p_obs
    observed_counts = rng.poisson(expected_counts)

    # Cell is detected if at least one individual is observed
    detected = (observed_counts >= 1).astype(np.float32)

    Pt_sparse[-1] = detected
    return Pt_sparse


# ============================================================================
# DIAGNOSTIC — run this before the ablation to verify the regime
# ============================================================================

def diagnose_axel_model(npz_path, body_mass=BODY_MASS_UNIT):
    """
    Comprehensive diagnostic of Axel's model with correct body_mass conversion.
    """
    print(f"\n{'='*70}")
    print(f"  AXEL'S EXACT MODEL — with CORRECT BODY_MASS conversion")
    print(f"  body_mass = {body_mass}  (individuals = biomass / body_mass)")
    print(f"  File: {npz_path}")
    print(f"{'='*70}\n")

    npz = np.load(npz_path, allow_pickle=True)

    # Load P_t and B_last
    P_t = np.asarray(npz["P_t"])
    if P_t.ndim == 3:
        T, S, N = P_t.shape
        side = int(np.sqrt(N))
        P_t = P_t.reshape(T, S, side, side)
    T, S, Y, X = P_t.shape
    P_last = (P_t[-1] > 0).astype(np.int32)

    B_last = np.asarray(npz["B_last"]).astype(np.float64)
    if B_last.ndim == 2 and B_last.shape[0] == S:
        side_b = int(np.sqrt(B_last.shape[1]))
        B_last = B_last.reshape(S, side_b, side_b)

    range_sizes = P_last.sum(axis=(1, 2))
    present_species = range_sizes > 0
    n_present = present_species.sum()

    # Individual counts
    N_ind = B_last / body_mass  # shape (S, Y, X)

    print(f"  Data: T={T}, S={S}, Y={Y}, X={X}")
    print(f"  Present species: {n_present}/{S}")
    print(f"  Range mean={range_sizes[present_species].mean():.2f}, "
          f"max={range_sizes[present_species].max()}")
    print()
    print(f"  BIOMASS STATS:")
    print(f"    Mean B per occupied cell: {B_last[P_last>0].mean():.4f}")
    print(f"    → Mean individuals per occupied cell: "
          f"{B_last[P_last>0].mean()/body_mass:.1f}")
    print(f"    Min B (threshold = 10 * body_mass): {10 * body_mass}")
    print(f"    Max B: {B_last.max():.4f}")
    print(f"    → Max individuals per cell: {B_last.max()/body_mass:.0f}")
    print()
    print(f"  AXEL'S MODEL — detection rates across p_obs values:")
    print(f"  {'p_obs':<12}{'% cells detected':<20}{'Mean obs/sp':<16}"
          f"{'% species detected'}")
    print(f"  {'-'*70}")

    rng = np.random.default_rng(42)
    results = []
    for p_obs in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
        expected = N_ind * p_obs
        observed = rng.poisson(expected)
        detected = (observed >= 1).astype(np.int32)

        # Only count detections at cells where species was actually present
        detected_valid = detected * P_last
        mean_obs_per_sp = detected_valid[:, :, :].sum(axis=(1, 2))
        total_cells_detected = detected_valid.sum()
        total_occupied = P_last.sum()
        pct_cells_detected = (100 * total_cells_detected / total_occupied
                              if total_occupied > 0 else 0)
        species_detected = (mean_obs_per_sp >= 1).sum()
        pct_species = 100 * species_detected / n_present if n_present > 0 else 0
        avg_obs = (mean_obs_per_sp[present_species].mean()
                   if n_present > 0 else 0)

        print(f"  {p_obs:<12.5f}{pct_cells_detected:<20.2f}"
              f"{avg_obs:<16.3f}{pct_species:.1f}%")
        results.append({
            'p_obs': p_obs, 'pct_cells': pct_cells_detected,
            'mean_obs': avg_obs, 'pct_species': pct_species
        })

    print()
    print(f"  RECOMMENDED p_obs VALUES FOR ABLATION:")
    print(f"  {'-'*60}")
    # Find p_obs giving target % species detected
    targets = [(10, 'Extreme sparsity'), (30, 'IUCN-like rare'),
               (50, 'Sparse survey'), (80, 'Moderate'), (95, 'Near full')]
    for target_pct, label in targets:
        # Linear interp
        below = [r for r in results if r['pct_species'] <= target_pct]
        above = [r for r in results if r['pct_species'] > target_pct]
        if not above:
            rec = results[-1]['p_obs']
        elif not below:
            rec = results[0]['p_obs']
        else:
            lo = below[-1]
            hi = above[0]
            frac = ((target_pct - lo['pct_species']) /
                    (hi['pct_species'] - lo['pct_species']))
            rec = lo['p_obs'] + frac * (hi['p_obs'] - lo['p_obs'])
        print(f"    {label:<22} ({target_pct}% sp detected): p_obs ≈ {rec:.5f}")

    print()
    return results


# ============================================================================
# INTEGRATION INSTRUCTIONS for stage2_ablation_study_v5.py
# ============================================================================

INTEGRATION_V5 = """
PATCH INSTRUCTIONS: Create stage2_ablation_study_v5.py from v3:

1. Add BODY_MASS constant at the top of the file:
   BODY_MASS_UNIT = 1e-4   # from config.py — matches simulation

2. Replace the broken sparsify_history_poisson function with
   sparsify_history_poisson_axel from above.

3. In build_ablation_condition, replace the POISSON branch:

      # ARM B POISSON — Axel's abundance-weighted detection with proper
      # biomass-to-individual conversion
      if poisson_p is not None:
          if "B_last" not in npz_data:
              raise ValueError("B_last required for Poisson ablation")
          B_last = np.asarray(npz_data["B_last"]).astype(np.float32)
          if species_subset is not None:
              B_last = B_last[species_subset]
          # Ensure 3D shape
          if B_last.ndim == 2:
              side = int(np.sqrt(B_last.shape[1]))
              B_last = B_last.reshape(B_last.shape[0], side, side)
          Pt = sparsify_history_poisson_axel(
              Pt, B_last, poisson_p,
              body_mass=BODY_MASS_UNIT, rng_seed=rng_seed)

4. Replace the five POISSON conditions in ABLATION_CONDITIONS with:

      ("POISSON_p00001",  {"drop_interactions": False, "drop_temporal": False,
                           "drop_obs": True, "gap": 0, "sparse_budget": None,
                           "poisson_p": 0.00001,
                           "label": "Poisson p=0.001%",  "color": "#fecaca",
                           "desc": "Axel's Poisson, 0.001% detection per individual"}),
      ("POISSON_p0001",   {"drop_interactions": False, "drop_temporal": False,
                           "drop_obs": True, "gap": 0, "sparse_budget": None,
                           "poisson_p": 0.0001,
                           "label": "Poisson p=0.01%",   "color": "#fca5a5",
                           "desc": "Axel's Poisson, 0.01% detection per individual"}),
      ("POISSON_p0005",   {"drop_interactions": False, "drop_temporal": False,
                           "drop_obs": True, "gap": 0, "sparse_budget": None,
                           "poisson_p": 0.0005,
                           "label": "Poisson p=0.05%",   "color": "#f87171",
                           "desc": "Axel's Poisson, 0.05% detection per individual"}),
      ("POISSON_p001",    {"drop_interactions": False, "drop_temporal": False,
                           "drop_obs": True, "gap": 0, "sparse_budget": None,
                           "poisson_p": 0.001,
                           "label": "Poisson p=0.1%",    "color": "#ef4444",
                           "desc": "Axel's Poisson, 0.1% detection per individual"}),
      ("POISSON_p01",     {"drop_interactions": False, "drop_temporal": False,
                           "drop_obs": True, "gap": 0, "sparse_budget": None,
                           "poisson_p": 0.01,
                           "label": "Poisson p=1%",      "color": "#dc2626",
                           "desc": "AXEL STANDARD: 1% per individual (may saturate)"}),

5. Run ONLY the Poisson conditions:
      python stage2_ablation_study_v5.py \\
          --ibm-dir results/data/ \\
          --checkpoint stage2_outputs/checkpoints/best_model.pt \\
          --stage2-dir AI_simulation/stage2/ \\
          --output-dir stage2_ablation_v5_poisson/ \\
          --n-worlds 8 --n-samples 8 \\
          --only-poisson

   Expected runtime: ~10-12 hours (6 of 17 conditions).
"""


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python poisson_axel_correct.py <path_to_training.npz>")
        print("       Run diagnostic on a representative IBM world first.")
        print()
        print(INTEGRATION_V5)
        sys.exit(0)
    diagnose_axel_model(sys.argv[1])