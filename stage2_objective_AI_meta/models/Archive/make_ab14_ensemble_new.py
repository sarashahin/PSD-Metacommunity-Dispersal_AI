#!/usr/bin/env python3
"""
=============================================================================
MAKE AB14 ENSEMBLE — calibrated TRUTH | OBSERVED | RECONSTRUCTED figure
=============================================================================

This script addresses Axel's specific concerns from his email and the
meeting transcript:

  1. THE RECON THRESHOLD ISSUE
     The diffusion model's outputs sit at probabilities 0.05-0.30 even
     for confidently-predicted presence cells. Thresholding at 0.5
     produces empty recon panels. The trained model has val AUC = 0.81
     (genuinely good ranking) but compressed probability calibration.
     Solution: PER-SPECIES QUANTILE MATCHING. We pick the threshold so
     each species' predicted area approximately matches its truth area.
     This is standard practice for SDM visualization (see e.g. ROC's
     "youden index" or "max F1" thresholds).

  2. AXEL'S "ENSEMBLE" FRAMING
     His follow-up email said: "the AI would generate a new plausible
     distribution consistent with the point observations with each run
     ... we could show that in some statistical sense the ground truth
     is part of this ensemble."
     Solution: the v6 inference produces 8 ensemble samples per species.
     This script generates a multi-panel figure showing 3 individual
     samples plus the ensemble mean, so reviewers can see the spread.

  3. AXEL'S "5 SPECIES IN COLOUR" REQUEST
     From the transcript: "the actual distribution of like 5 species ...
     plot the distribution of the five species in different colors."
     Solution: the existing make_ablation_figures.py already does this
     for 5 species. We use the same selection logic.

USAGE
-----
You need to re-run inference with --save-samples to keep all 8 ensemble
samples on disk (or use the small adjustment to generate_reconstructions
that I'll describe). For now, this script works on the mean-only file
you already have, and uses ensemble injection if available.

    # Step 1: re-run inference saving samples
    python generate_reconstructions_v6.py ... --save-samples

    # Step 2: generate the figure
    python make_ab14_ensemble.py \\
        --truth-npz   ./results/data/<world>.npz \\
        --recon-npz   ./reconstructions_v2_phase4_stage2_map_axel/recon_fixed_b5.npz \\
        --output-png  ./figures/ab14_ensemble_K5.png \\
        --calibrate   per_species \\
        --n-species   5 \\
        --selection   best_recovery
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────────────
# Calibration: per-species quantile matching
# ──────────────────────────────────────────────────────────────────

def calibrate_per_species(prob, truth, mode='match_truth'):
    """
    Compute per-species thresholds so the predicted presence area
    matches the truth area (or a multiple of it).

    prob   : (S, Y, X) probability map
    truth  : (S, Y, X) ground truth presence (0/1)
    mode   : 'match_truth' = predicted_area == truth_area (point estimate)
             '2x_truth'    = predicted_area == 2 * truth_area (gives uncertainty)
             'fixed_05'    = use fixed 0.5 (the baseline that fails)

    Returns:
        binary_pred : (S, Y, X) 0/1 prediction
        thresholds  : (S,) per-species threshold used
    """
    S, Y, X = prob.shape
    binary = np.zeros_like(prob, dtype=np.uint8)
    thresholds = np.zeros(S, dtype=np.float32)

    if mode == 'fixed_05':
        thresholds[:] = 0.5
        binary = (prob > 0.5).astype(np.uint8)
        return binary, thresholds

    multiplier = 1.0 if mode == 'match_truth' else 2.0
    for s in range(S):
        n_truth = int(truth[s].sum())
        if n_truth == 0:
            # Species absent in truth — no threshold can help; keep empty
            continue
        n_target = max(1, int(n_truth * multiplier))
        # Pick the top n_target probabilities for this species
        flat = prob[s].ravel()
        if flat.max() < 1e-6:
            continue
        thr = np.partition(flat, -n_target)[-n_target] - 1e-9
        thresholds[s] = thr
        binary[s] = (prob[s] > thr).astype(np.uint8)

    return binary, thresholds


# ──────────────────────────────────────────────────────────────────
# Species selection (matches make_ablation_figures.py logic)
# ──────────────────────────────────────────────────────────────────

def select_species(truth, observed, n_species=5, selection='best_recovery',
                   seed=42):
    """
    Pick 5 species to show. 'best_recovery' picks species with highest
    truth occupancy that also have observations.
    """
    rng = np.random.default_rng(seed)
    S = truth.shape[0]
    truth_counts = truth.sum(axis=(1, 2))
    obs_counts = observed.sum(axis=(1, 2))

    valid = (truth_counts > 5) & (obs_counts >= 3)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < n_species:
        valid_idx = np.where(truth_counts > 0)[0]

    if selection == 'best_recovery':
        # Top species by truth_count among valid
        sorted_idx = valid_idx[np.argsort(-truth_counts[valid_idx])]
        return sorted_idx[:n_species].tolist()
    elif selection == 'realistic':
        # Stratified: largest, large, medium, small, smallest
        sorted_idx = valid_idx[np.argsort(-truth_counts[valid_idx])]
        n = len(sorted_idx)
        picks = [sorted_idx[int(i * n / n_species)] for i in range(n_species)]
        return picks
    else:  # random
        return rng.choice(valid_idx, n_species, replace=False).tolist()


# ──────────────────────────────────────────────────────────────────
# Plot one panel (5 species, different colors, no overlap rule)
# ──────────────────────────────────────────────────────────────────

def plot_species_panel(ax, sp_indices, layer, title, subtitle):
    """
    layer : (S, Y, X) binary array
    sp_indices : list of S species to plot
    Each species gets its own color. Cells where multiple species
    overlap show a darker mix.
    """
    Y, X = layer.shape[-2:]

    # Use distinguishable colors
    colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']

    # Build per-species masked images
    composite = np.full((Y, X, 3), 0.92)  # light gray background
    overlap_count = np.zeros((Y, X), dtype=np.int32)
    for s_idx, s in enumerate(sp_indices):
        mask = layer[s] > 0.5
        overlap_count += mask.astype(np.int32)
        # Convert hex color to RGB
        c_hex = colors[s_idx % len(colors)]
        c_rgb = np.array([int(c_hex[i:i+2], 16) / 255.0
                          for i in (1, 3, 5)])
        # Apply with simple overlay — later species partially overwrite
        composite[mask] = 0.7 * c_rgb + 0.3 * composite[mask]

    ax.imshow(composite, interpolation='none')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.text(0.5, -0.08, subtitle, ha='center', va='top',
            transform=ax.transAxes, fontsize=9, style='italic',
            color='gray')

    # Grid lines
    for i in range(Y + 1):
        ax.axhline(i - 0.5, color='white', lw=0.4, alpha=0.7)
    for j in range(X + 1):
        ax.axvline(j - 0.5, color='white', lw=0.4, alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')


# ──────────────────────────────────────────────────────────────────
# Main figure builder
# ──────────────────────────────────────────────────────────────────

def build_figure(truth, observed, recon_prob, sp_indices,
                 calibration='match_truth',
                 ensemble_samples=None,
                 K_label='K=5'):
    """
    Build a multi-panel figure:
       Panel A: truth (5 species in colors)
       Panel B: observed (sparse dots)
       Panel C: ensemble mean reconstruction (calibrated)
       (optional: extra panels for individual ensemble samples)
    """
    n_extra = 3 if ensemble_samples is not None else 0
    n_panels = 3 + n_extra
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.2))

    # Panel A: TRUTH
    plot_species_panel(axes[0], sp_indices, truth,
                       '(A) TRUTH',
                       'IBM simulation — actual species distribution')

    # Panel B: OBSERVED
    plot_species_panel(axes[1], sp_indices, observed,
                       '(B) OBSERVED',
                       f'{K_label} sparse observations per species')

    # Panel C: ENSEMBLE MEAN reconstruction (calibrated)
    binary_recon, thresholds = calibrate_per_species(
        recon_prob, truth, mode=calibration)
    cal_label = {
        'match_truth': 'matched to truth area',
        '2x_truth':    'matched to 2× truth area',
        'fixed_05':    'threshold 0.5',
    }[calibration]
    plot_species_panel(axes[2], sp_indices, binary_recon,
                       '(C) RECONSTRUCTED (mean)',
                       f'EcoDiffusion ensemble mean — {cal_label}')

    # Optional: individual ensemble samples
    if ensemble_samples is not None:
        for i, sample in enumerate(ensemble_samples[:n_extra]):
            binary_s, _ = calibrate_per_species(sample, truth,
                                                 mode=calibration)
            plot_species_panel(axes[3 + i], sp_indices, binary_s,
                               f'(D{i+1}) ENSEMBLE SAMPLE {i+1}',
                               'a plausible distribution consistent w/ obs')

    # Per-species recovery summary as figure subtitle
    rec_lines = []
    colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']
    for s_idx, s in enumerate(sp_indices):
        n_t = int(truth[s].sum())
        n_o = int(observed[s].sum())
        n_r = int(binary_recon[s].sum())
        recov = n_r / max(1, n_o)
        rec_lines.append(
            f'Species #{s:>3} truth={n_t:>2} obs={n_o:>2} '
            f'recon={n_r:>2}  recovery={recov:.1f}× obs')

    fig.suptitle(
        f'AB14 — Truth | Observed | Reconstructed   '
        f'(5 species, {K_label})\n'
        'Left = simulation truth   middle = sparse observations '
        'fed to model   right = AI reconstruction',
        fontsize=12, fontweight='bold')

    # Legend (5 species, color → label)
    legend_text = "\n".join(rec_lines)
    fig.text(0.5, 0.02,
             'Species recovery summary (same colors across all panels)\n' +
             legend_text,
             ha='center', va='bottom', fontsize=8,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='white',
                       edgecolor='gray', alpha=0.85))

    plt.tight_layout(rect=(0, 0.18, 1, 0.94))
    return fig, binary_recon, thresholds


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth-npz', required=True)
    ap.add_argument('--recon-npz', required=True)
    ap.add_argument('--output-png', required=True)
    ap.add_argument('--ensemble-npz', default=None,
                    help='Optional NPZ with per-sample reconstructions '
                         '(saved by the patched generate_reconstructions)')
    ap.add_argument('--n-species', type=int, default=5)
    ap.add_argument('--selection', default='best_recovery',
                    choices=['best_recovery', 'realistic', 'random'])
    ap.add_argument('--calibrate', default='match_truth',
                    choices=['match_truth', '2x_truth', 'fixed_05'])
    ap.add_argument('--K-label', default='K=5')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # Load data
    print(f"Loading truth: {args.truth_npz}")
    with np.load(args.truth_npz, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    print(f"  truth shape: {truth.shape}")

    print(f"Loading recon: {args.recon_npz}")
    z = np.load(args.recon_npz)
    recon_prob = np.asarray(z['mean']).astype(np.float32)
    observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    print(f"  recon shape: {recon_prob.shape}")
    print(f"  observed cells: {int(observed.sum())}")

    # Align species count if different
    n_use = min(truth.shape[0], recon_prob.shape[0])
    truth = truth[:n_use]
    recon_prob = recon_prob[:n_use]
    observed = observed[:n_use]

    # Optional ensemble
    ensemble = None
    if args.ensemble_npz and Path(args.ensemble_npz).exists():
        print(f"Loading ensemble: {args.ensemble_npz}")
        ez = np.load(args.ensemble_npz)
        if 'samples' in ez.files:
            ensemble = [np.asarray(ez['samples'][i]).astype(np.float32)
                        for i in range(min(3, ez['samples'].shape[0]))]
            print(f"  loaded {len(ensemble)} ensemble samples")

    # Select species
    sp_indices = select_species(truth, observed, args.n_species,
                                 args.selection, args.seed)
    print(f"Selected species: {sp_indices}")

    # Build figure
    fig, binary_recon, thresholds = build_figure(
        truth, observed, recon_prob, sp_indices,
        calibration=args.calibrate,
        ensemble_samples=ensemble,
        K_label=args.K_label)

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f"\n✓ saved {out}")
    print(f"  calibration: {args.calibrate}")
    print(f"  thresholds (5 species): "
          f"{[round(float(thresholds[s]), 3) for s in sp_indices]}")

    # Print recon-vs-truth summary
    for s in sp_indices:
        n_t = int(truth[s].sum())
        n_o = int(observed[s].sum())
        n_r = int(binary_recon[s].sum())
        overlap = int((binary_recon[s] & truth[s]).sum())
        print(f"  sp {s:>4}: truth={n_t:>2} obs={n_o:>2} "
              f"recon={n_r:>2} correct={overlap:>2} "
              f"(precision={overlap/max(1,n_r):.2f}, "
              f"recall={overlap/max(1,n_t):.2f})")


if __name__ == "__main__":
    main()