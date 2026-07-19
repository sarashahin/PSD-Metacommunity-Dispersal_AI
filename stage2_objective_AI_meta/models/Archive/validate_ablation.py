#!/usr/bin/env python3
"""
=============================================================================
VALIDATE_ABLATION_V2.PY  —  drop-in replacement for validate_ablation.py
=============================================================================

WHY THIS REPLACES V1
--------------------
The old validate_ablation.py used a binary "FULL_dominates_{variant}" test:
PASS if FULL beats the ablation on ≥ 2 of 3 metrics, FAIL otherwise. That
framing was unsuitable for this study because:

  - When the ablation has NO real effect (e.g. NO_NETWORK, NO_SPECIES_FEATS),
    metric noise can flip 1 of the 3 sub-tests, falsely flagging the result
    as a FAIL — but the actual scientific finding is a clean "predictor not
    contributing", which is exactly what Axel's transcript invited.

  - When an ablation is BENEFICIAL (e.g. NO_ENV beating FULL on AUC at
    unobserved cells), the v1 logic flagged that as a FAIL too, even
    though it's the most interesting finding of the study.

V2 uses a three-tier classification per ablation, plus structural
sanity tests that PASS/FAIL meaningfully:

  STRUCTURAL TESTS (PASS/FAIL is meaningful):
    - FULL_present
    - FULL_inpainting_works           (echo at obs > 70%)
    - NO_HISTORY_obs_mask_zeroed      (echo at obs near 0)
    - all_finite                      (no NaN/Inf in any variant)

  CLASSIFICATION (each ablation gets one label, never a FAIL):
    CRITICAL    : max relative drop > 50 %  (predictor is essential)
    MODERATE    : 20 % < max drop ≤ 50 %    (predictor contributes)
    NEGLIGIBLE  : -10 % ≤ max drop ≤ 20 %   (predictor doesn't matter)
    BENEFICIAL  : max drop < -10 %          (model is better without it)

The classification draws on FOUR metrics (meaningful_mean_recall,
meaningful_union_recall, meaningful_pix_cov_union, auc_unobs) instead of
v1's three. The "max drop" is the worst of the four — a predictor that
crashes ANY metric by 50 % is critical.

USAGE
-----
Identical CLI to v1, so existing commands keep working:

  python validate_ablation_v2.py \\
      --ablation-dir   ./ablation_v7_world5_stage2_inpaint \\
      --truth-npz      ./results/data/<world>.npz \\
      --K              5 \\
      --calibrate      match_truth \\
      --output-csv     ./ablation_v7_world5_stage2_inpaint/ablation_metrics_1x.csv

Multi-world mode also identical.
=============================================================================
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = ['FULL', 'NO_HISTORY', 'NO_NETWORK', 'NO_ENV', 'NO_SPECIES_FEATS']

CLASSIFICATION_METRICS = [
    'meaningful_mean_recall',
    'meaningful_union_recall',
    'meaningful_pix_cov_union',
    'auc_unobs',
]


# -----------------------------------------------------------------------
# Stratified coverage  (calibration matches Axel's "match_truth")
# -----------------------------------------------------------------------

def calibrate_per_species(prob, truth, mode='match_truth'):
    S = prob.shape[0]
    binary = np.zeros_like(prob, dtype=np.uint8)
    if mode == 'fixed_05':
        return (prob > 0.5).astype(np.uint8)
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


def stratified_metrics(truth, mean_pred, samples, K, calibrate='match_truth'):
    n_use = min(truth.shape[0], samples.shape[1])
    truth = truth[:n_use]
    samples = samples[:, :n_use]
    mean_pred = mean_pred[:n_use]
    n_ens = samples.shape[0]

    binary_samples = np.zeros_like(samples, dtype=np.uint8)
    for i in range(n_ens):
        binary_samples[i] = calibrate_per_species(
            samples[i], truth, mode=calibrate)
    binary_mean = calibrate_per_species(
        mean_pred, truth, mode=calibrate)
    ensemble_union = (binary_samples.sum(axis=0) > 0).astype(np.uint8)

    rows = []
    for s in range(n_use):
        n_t = int(truth[s].sum())
        if n_t == 0:
            continue
        rows.append({
            'truth_cells': n_t,
            'mean_correct': int((binary_mean[s] & truth[s]).sum()),
            'mean_recall':  int((binary_mean[s] & truth[s]).sum()) / n_t,
            'union_correct': int((ensemble_union[s] & truth[s]).sum()),
            'union_recall':  int((ensemble_union[s] & truth[s]).sum()) / n_t,
        })

    def agg(stratum):
        if not stratum:
            return None
        truth_total = sum(r['truth_cells'] for r in stratum)
        return {
            'n_species': len(stratum),
            'truth_cells': truth_total,
            'mean_recall':  float(np.mean([r['mean_recall']  for r in stratum])),
            'union_recall': float(np.mean([r['union_recall'] for r in stratum])),
            'pix_cov_mean':  sum(r['mean_correct']  for r in stratum) / max(1, truth_total),
            'pix_cov_union': sum(r['union_correct'] for r in stratum) / max(1, truth_total),
        }

    return {
        'meaningful':  agg([r for r in rows if r['truth_cells'] > K]),
        'real':        agg([r for r in rows if r['truth_cells'] > 2 * K]),
        'all_present': agg(rows),
    }


def compute_auc(truth, mean_pred, observed):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None, None

    n_use = min(truth.shape[0], mean_pred.shape[0])
    truth = (truth[:n_use] > 0.5).astype(int)
    pred = mean_pred[:n_use]
    obs = observed[:n_use]

    flat_t = truth.ravel()
    flat_p = pred.ravel()
    auc_overall = None
    if 5 < flat_t.sum() < flat_t.size - 5:
        auc_overall = float(roc_auc_score(flat_t, flat_p))

    auc_unobs = None
    unobs_flat = (obs < 0.5).ravel()
    if unobs_flat.sum() > 0:
        t_un = flat_t[unobs_flat]
        p_un = flat_p[unobs_flat]
        if 5 < t_un.sum() < t_un.size - 5:
            auc_unobs = float(roc_auc_score(t_un, p_un))

    return auc_overall, auc_unobs


# -----------------------------------------------------------------------
# Per-world evaluator
# -----------------------------------------------------------------------

def evaluate_one_world(truth_path, ablation_dir, K, variants, calibrate):
    truth_path = Path(truth_path)
    ablation_dir = Path(ablation_dir)
    world_name = truth_path.name

    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)

    rows = []
    for variant in variants:
        samples_path = ablation_dir / f'recon_{variant}_b{K}_samples.npz'
        if not samples_path.exists():
            print(f"  ⚠ skip {variant}: {samples_path.name} missing")
            continue
        z = np.load(samples_path)
        samples = np.asarray(z['samples']).astype(np.float32)
        mean_pred = np.asarray(z['mean']).astype(np.float32)
        observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)

        m = stratified_metrics(truth, mean_pred, samples, K, calibrate)
        auc_overall, auc_unobs = compute_auc(truth, mean_pred, observed)

        n_obs = int(observed.sum())
        bin_mean = (mean_pred > 0.5).astype(np.uint8)
        echo = int((bin_mean & observed).sum())
        fillin = int((bin_mean & (1 - observed)).sum())

        # Sanity: any non-finite?
        n_nonfinite = int((~np.isfinite(samples)).sum())

        row = {
            'world': world_name,
            'variant': variant,
            'K': K,
            'n_obs': n_obs,
            'echo_at_obs': echo,
            'fillin_cells': fillin,
            'auc_overall': auc_overall,
            'auc_unobs': auc_unobs,
            'n_nonfinite_in_samples': n_nonfinite,
        }
        for stratum_name in ('meaningful', 'real', 'all_present'):
            stats = m.get(stratum_name)
            if stats:
                for k, v in stats.items():
                    row[f'{stratum_name}_{k}'] = v
        rows.append(row)
        auc_str = f"{auc_unobs:.3f}" if auc_unobs is not None else "n/a"
        print(f"  ✓ {variant:<20} mean_rec(meaningful)="
              f"{row.get('meaningful_mean_recall', 0):.3f}  "
              f"union_rec={row.get('meaningful_union_recall', 0):.3f}  "
              f"auc_unobs={auc_str}")
    return rows


# -----------------------------------------------------------------------
# THREE-TIER CLASSIFICATION  (the headline change vs v1)
# -----------------------------------------------------------------------

def classify_ablation(full_row, ablation_row, metrics=CLASSIFICATION_METRICS):
    """
    Classify the ablation effect on FOUR metrics, return:
      - per-metric relative deltas (signed %, NEGATIVE = ablation hurt)
      - max drop across metrics (a single scalar)
      - classification label
      - one-line interpretation string
    """
    deltas = {}
    for m in metrics:
        f_v = full_row.get(m)
        a_v = ablation_row.get(m)
        if f_v is None or a_v is None:
            deltas[m] = None
            continue
        f_v = float(f_v); a_v = float(a_v)
        denom = max(0.001, abs(f_v))
        # POSITIVE delta = ablation is WORSE than FULL (the predictor mattered)
        # NEGATIVE delta = ablation is BETTER than FULL (predictor was harmful)
        deltas[m] = (f_v - a_v) / denom * 100.0

    valid = [d for d in deltas.values() if d is not None]
    if not valid:
        return {
            'deltas': deltas, 'max_drop': None, 'min_drop': None,
            'category': 'UNKNOWN',
            'interpretation': 'Insufficient data to classify.'
        }
    max_drop = max(valid)
    min_drop = min(valid)

    # Classify by worst-case impact
    if max_drop > 50:
        cat = 'CRITICAL'
        msg = ('Removing this predictor causes a >50% drop on at least '
               'one metric. The predictor is essential.')
    elif max_drop > 20:
        cat = 'MODERATE'
        msg = ('Removing this predictor causes a 20–50% drop on at least '
               'one metric. The predictor contributes.')
    elif max_drop >= -10:
        cat = 'NEGLIGIBLE'
        msg = ('Worst-case drop is within ±10–20%. The predictor is not '
               'contributing measurably; the model can be simplified by '
               'removing it.')
    else:
        cat = 'BENEFICIAL'
        msg = ('The model performs BETTER without this predictor on at '
               'least one metric. Investigate whether the predictor is '
               'mis-calibrated or noisy.')

    return {
        'deltas': deltas,
        'max_drop': max_drop, 'min_drop': min_drop,
        'category': cat,
        'interpretation': msg,
    }


# -----------------------------------------------------------------------
# Cross-variant validation tests  (PASS/FAIL is now meaningful)
# -----------------------------------------------------------------------

def cross_variant_validation(rows_for_world, log_lines):
    by_variant = {r['variant']: r for r in rows_for_world}
    tests = []

    # Structural test 1: FULL exists
    if 'FULL' not in by_variant:
        tests.append(('FULL_present', False, "FULL variant missing"))
        return tests, {}
    tests.append(('FULL_present', True, "FULL variant present"))

    full = by_variant['FULL']

    # Structural test 2: FULL inpainting actually overlaid observations
    n_obs = full['n_obs']
    echo_frac = full['echo_at_obs'] / max(1, n_obs)
    pass_t2 = echo_frac > 0.7
    tests.append(('FULL_inpainting_works',
                  pass_t2,
                  f"FULL echo={full['echo_at_obs']}/{n_obs} "
                  f"({echo_frac:.1%}) — expected >70%"))

    # Structural test 3: NO_HISTORY obs mask was correctly zeroed
    if 'NO_HISTORY' in by_variant:
        nh = by_variant['NO_HISTORY']
        nh_echo_frac = nh['echo_at_obs'] / max(1, n_obs)
        pass_t3 = nh_echo_frac < 0.20
        tests.append(('NO_HISTORY_obs_mask_zeroed',
                      pass_t3,
                      f"NO_HISTORY echo={nh['echo_at_obs']}/{n_obs} "
                      f"({nh_echo_frac:.1%}) — expected <20%"))

    # Structural test 4: all variants produce finite output
    nonfinite_total = sum(r.get('n_nonfinite_in_samples', 0)
                          for r in rows_for_world)
    tests.append(('all_variants_finite',
                  nonfinite_total == 0,
                  f"non-finite values across all variants: {nonfinite_total}"))

    # Classifications (these are NEVER FAILs — they are interpretations)
    classifications = {}
    for variant in VARIANTS:
        if variant == 'FULL' or variant not in by_variant:
            continue
        cl = classify_ablation(full, by_variant[variant])
        classifications[variant] = cl

    # Log
    log_lines.append("\n  STRUCTURAL VALIDATION:")
    for name, ok, msg in tests:
        status = "PASS" if ok else "FAIL"
        log_lines.append(f"    [{status}]  {name:<32}  {msg}")

    log_lines.append("\n  ABLATION CLASSIFICATION (3-tier, no PASS/FAIL):")
    for variant, cl in classifications.items():
        log_lines.append(f"    {variant:<20}  [{cl['category']:<10}]  "
                          f"max_drop={cl['max_drop']:+6.1f}%   "
                          f"min_drop={cl['min_drop']:+6.1f}%")
        log_lines.append(f"      {cl['interpretation']}")
        for m, d in cl['deltas'].items():
            if d is None:
                continue
            log_lines.append(f"        {m:<32}  Δ = {d:+7.1f}%")

    return tests, classifications


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Single-world mode
    ap.add_argument('--ablation-dir', default=None)
    ap.add_argument('--truth-npz', default=None)
    # Multi-world mode
    ap.add_argument('--ablation-dir-pattern', default=None)
    ap.add_argument('--wide-range-csv', default=None)
    ap.add_argument('--truth-dir', default=None)
    ap.add_argument('--top-n-worlds', type=int, default=5)
    # Common
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--variants', nargs='+', default=VARIANTS)
    ap.add_argument('--calibrate', default='match_truth',
                    choices=['match_truth', '2x_truth'])
    ap.add_argument('--output-csv', required=True)
    args = ap.parse_args()

    log_lines = ["VALIDATE_ABLATION_V2 run log",
                 f"K={args.K}, calibrate={args.calibrate}",
                 f"variants={args.variants}"]

    all_rows = []

    if args.ablation_dir and args.truth_npz:
        print(f"\nMode: single-world")
        print(f"  ablation_dir: {args.ablation_dir}")
        print(f"  truth_npz:    {args.truth_npz}")
        rows = evaluate_one_world(
            args.truth_npz, args.ablation_dir,
            args.K, args.variants, args.calibrate)
        all_rows.extend(rows)
        tests, classifications = cross_variant_validation(rows, log_lines)
        n_pass = sum(1 for _, ok, _ in tests if ok)
        n_total = len(tests)
        print(f"\n  Structural tests: {n_pass}/{n_total} passed")
        for r in rows:
            r['structural_pass'] = (n_pass == n_total)
            cl = classifications.get(r['variant'])
            if cl is not None:
                r['effect_category'] = cl['category']
                r['max_relative_drop_pct'] = (
                    f"{cl['max_drop']:.1f}" if cl['max_drop'] is not None
                    else '')
                r['interpretation'] = cl['interpretation']
            else:
                r['effect_category'] = 'BASELINE' if r['variant'] == 'FULL' else ''
                r['max_relative_drop_pct'] = ''
                r['interpretation'] = ''

    elif args.ablation_dir_pattern and args.wide_range_csv \
            and args.truth_dir:
        print(f"\nMode: multi-world")
        from collections import defaultdict as dd
        world_count = dd(int)
        with open(args.wide_range_csv) as f:
            for r in csv.DictReader(f):
                world_count[r['world']] += 1
        top_worlds = sorted(
            world_count.items(), key=lambda x: -x[1])[:args.top_n_worlds]
        print(f"  Evaluating top {len(top_worlds)} worlds")

        for world_name, n_wide in top_worlds:
            world_stem = world_name.replace('.npz', '')
            ablation_dir = Path(
                args.ablation_dir_pattern.format(world_stem=world_stem))
            truth_path = Path(args.truth_dir) / world_name
            if not ablation_dir.exists():
                print(f"  ⚠ skip {world_name}: {ablation_dir} missing")
                continue
            print(f"\n  --- {world_name[:60]}  (n_wide={n_wide}) ---")
            rows = evaluate_one_world(
                truth_path, ablation_dir,
                args.K, args.variants, args.calibrate)
            tests, classifications = cross_variant_validation(rows, log_lines)
            n_pass = sum(1 for _, ok, _ in tests if ok)
            n_total = len(tests)
            for r in rows:
                r['structural_pass'] = (n_pass == n_total)
                cl = classifications.get(r['variant'])
                if cl is not None:
                    r['effect_category'] = cl['category']
                    r['max_relative_drop_pct'] = (
                        f"{cl['max_drop']:.1f}" if cl['max_drop'] is not None
                        else '')
                    r['interpretation'] = cl['interpretation']
                else:
                    r['effect_category'] = 'BASELINE' if r['variant'] == 'FULL' else ''
                    r['max_relative_drop_pct'] = ''
                    r['interpretation'] = ''
            all_rows.extend(rows)

    else:
        print("Specify either:")
        print("  Single-world: --ablation-dir AND --truth-npz")
        print("  Multi-world : --ablation-dir-pattern AND "
              "--wide-range-csv AND --truth-dir")
        return 1

    if not all_rows:
        print("No rows produced.")
        return 1

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_rows for k in r.keys()})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n  ✓ wrote {out}")

    # Per-world classification summary print
    print(f"\n{'='*72}")
    print(f"  PREDICTOR IMPORTANCE — classification ({args.calibrate})")
    print(f"{'='*72}")
    by_world = defaultdict(dict)
    for r in all_rows:
        by_world[r['world']][r['variant']] = r
    for world, vars_dict in by_world.items():
        print(f"\n  {world[:60]}:")
        for variant in VARIANTS:
            if variant == 'FULL' or variant not in vars_dict:
                continue
            r = vars_dict[variant]
            cat = r.get('effect_category', '?')
            drop = r.get('max_relative_drop_pct', '?')
            print(f"    {variant:<20}  [{cat:<10}]  "
                  f"max_drop={drop}%")

    log_path = out.parent / (out.stem + '_validation_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\n  ✓ validation log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())