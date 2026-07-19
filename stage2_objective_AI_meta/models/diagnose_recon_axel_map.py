#!/usr/bin/env python3
"""
=============================================================================
DIAGNOSE_RECON.PY  —  one-shot health check on a recon NPZ file
=============================================================================

Run this on any reconstructions_*/recon_*.npz to instantly see whether the
model is:
  (a) echoing the input  (v5 bug: high echo, zero fill-in)
  (b) ignoring the input (v6 without phase 4: low echo, low fill-in,
                          flat probability everywhere)
  (c) actually reconstructing (post phase 4: high echo, substantial fill-in,
                                bimodal probability distribution)

USAGE
-----
    python diagnose_recon.py path/to/recon.npz
    python diagnose_recon.py path/to/recon.npz --truth path/to/world.npz

If you pass --truth, you also get an AUC computation against the ground truth.
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def diagnose_one(recon_path, truth_path=None):
    print(f"\n{'='*72}\n{recon_path}\n{'='*72}")
    z = np.load(recon_path)
    if 'mean' not in z:
        print("  no 'mean' key — not a v5/v6 recon file")
        return
    mean = z['mean']
    has_noisy = 'noisy_input' in z
    print(f"  shape:         {mean.shape}")
    print(f"  sample_mode:   {z.get('sample_mode', 'unknown (probably v5)')}")
    print(f"  prob max:      {mean.max():.4f}")
    print(f"  prob mean:     {mean.mean():.5f}")
    print(f"  cells > 0.5:   {int((mean > 0.5).sum()):,}")
    print(f"  cells > 0.3:   {int((mean > 0.3).sum()):,}")
    print(f"  cells > 0.1:   {int((mean > 0.1).sum()):,}")

    if has_noisy:
        noisy = z['noisy_input']
        echo   = int(((mean > 0.5) & (noisy > 0.5)).sum())
        fillin = int(((mean > 0.5) & (noisy < 0.5)).sum())
        lost   = int(((mean < 0.5) & (noisy > 0.5)).sum())
        n_input = int(noisy.sum())
        print(f"\n  Echo / fill-in / lost decomposition (threshold 0.5):")
        print(f"    input cells:             {n_input:>8,}")
        print(f"    echo (recon=1, obs=1):   {echo:>8,}  ({100*echo/max(1,n_input):.1f}% of input)")
        print(f"    fill-in (recon=1, obs=0):{fillin:>8,}")
        print(f"    lost (recon=0, obs=1):   {lost:>8,}")

        obs_mask = noisy > 0.5
        if obs_mask.sum() > 0:
            print(f"\n  Probability AT observed cells:")
            print(f"    mean: {mean[obs_mask].mean():.4f}   max: {mean[obs_mask].max():.4f}")
            print(f"    >0.5: {int((mean[obs_mask] > 0.5).sum())}  "
                  f"({100*(mean[obs_mask] > 0.5).mean():.1f}%)")
            print(f"\n  Probability AT UNobserved cells:")
            print(f"    mean: {mean[~obs_mask].mean():.5f}   max: {mean[~obs_mask].max():.4f}")
            print(f"    >0.5: {int((mean[~obs_mask] > 0.5).sum())}")

        # Diagnosis
        print(f"\n  --- DIAGNOSIS ---")
        diff = abs(mean[obs_mask].mean() - mean[~obs_mask].mean()) if obs_mask.sum() > 0 else 0
        if echo / max(1, n_input) > 0.9 and fillin == 0:
            print("    V5 ECHO BUG: model is copying input, not reconstructing.")
            print("    Fix: use sample_v6 with mode='extrapolate'")
        elif diff < 0.02 and mean.max() < 0.85:
            print("    MODEL IGNORES INPUT: probability is the same at obs and "
                  "unobs cells, and max prob is low.")
            print("    Cause: model was never trained on sparse-history -> full-target.")
            print("    Fix: run finetune_phase4_infill_v2.py")
        elif echo / max(1, n_input) > 0.7 and fillin > n_input * 0.3:
            print("    HEALTHY RECONSTRUCTION:")
            print(f"      - model retains {100*echo/max(1,n_input):.0f}% of obs cells")
            print(f"      - model adds {fillin:,} new cells via extrapolation")
            print(f"      - this is what Axel's figure needs")
        else:
            print("    AMBIGUOUS - check the figure visually.")

    if truth_path:
        print(f"\n  --- AUC vs TRUTH ---")
        from sklearn.metrics import roc_auc_score
        with np.load(truth_path, allow_pickle=True) as td:
            P_truth = np.asarray(td['P_last_final']).astype(np.float32)
        n_use = min(P_truth.shape[0], mean.shape[0])
        truth = (P_truth[:n_use] > 0.5).astype(int)
        pred = mean[:n_use]
        flat_t = truth.ravel()
        flat_p = pred.ravel()
        if flat_t.sum() > 5 and flat_t.sum() < flat_t.size - 5:
            auc_overall = roc_auc_score(flat_t, flat_p)
            print(f"    AUC (all cells):      {auc_overall:.4f}")
        if has_noisy:
            unobs_flat = (z['noisy_input'][:n_use] < 0.5).ravel()
            if unobs_flat.sum() > 0:
                t_un = flat_t[unobs_flat]
                p_un = flat_p[unobs_flat]
                if t_un.sum() > 5 and t_un.sum() < t_un.size - 5:
                    auc_extrap = roc_auc_score(t_un, p_un)
                    print(f"    AUC at UNOBS cells:   {auc_extrap:.4f}  "
                          f"<- THIS is the metric that matters for AB14")
                    if auc_extrap < 0.55:
                        print("      essentially random; phase 4 fine-tune needed")
                    elif auc_extrap < 0.70:
                        print("      weak extrapolation; phase 4 fine-tune recommended")
                    elif auc_extrap < 0.85:
                        print("      moderate extrapolation; figure may be OK")
                    else:
                        print("      strong extrapolation; figure should look great")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recon_npz", nargs="+")
    p.add_argument("--truth", default=None)
    args = p.parse_args()
    for r in args.recon_npz:
        diagnose_one(r, truth_path=args.truth)


if __name__ == "__main__":
    main()