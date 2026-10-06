#!/usr/bin/env python3
"""
=============================================================================
DIAGNOSTIC: is MetaDiffusion probability-mass-capped per species?
=============================================================================

WHY THIS DIAGNOSTIC EXISTS
--------------------------
The adaptive top-N predictor (obs_logcd → N_hat) gave Spearman rho = 0.373.
Too weak to fix the wide-range bucket.A 
retrain, we need to know: does the model itself "know" the range size?

For each species s, the per-cell probability map p_s = mean(samples_s, axis=0)
is a probability distribution over the 20x20 grid. Its TOTAL MASS is
   M_s = sum_cells( p_s )
which represents the model's expected total number of occupied cells.

If M_s scales (roughly linearly) with truth range size R_s:
   --> The model has learned range-size information; it just spreads
       probability too thinly across cells for wide ranges. A mass-based
       N_hat = round(M_s) would unlock a clean post-hoc fix.

If M_s is approximately constant regardless of R_s:
   --> The model is mass-capped during training. It allocates the same
       total expected occupancy per species regardless of true range.
       No post-hoc fix can recover wide ranges. Need to retrain with a
       range-aware loss.

The third possible outcome: M_s scales for narrow ranges but saturates
for wide ranges (sigmoid-like). Mass-based N_hat would help moderate
species but not the HARD bucket.

INPUTS
------
Reads existing reconstruction NPZs from the spatial pipeline. No new
sampling. No synthetic data. No retraining.

OUTPUTS
-------
   probability_mass_vs_truth_range.png   -- scatter + per-bucket boxplots
   mass_based_topN_per_bucket.csv        -- KS values if we use M_s as N_hat
   probability_mass_diagnosis.txt        -- text summary with verdict
=============================================================================
"""

import argparse, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats


GRID_Y, GRID_X = 20, 20
CONN = ndimage.generate_binary_structure(2, 1)


def periodic_cov_det(binary_range, Y=GRID_Y, X=GRID_X):
    yy, xx = np.where(binary_range > 0.5)
    if len(yy) < 2:
        return 0.0
    ty = 2.0 * np.pi * yy / Y
    tx = 2.0 * np.pi * xx / X
    my = np.arctan2(np.sin(ty).mean(), np.cos(ty).mean())
    mx = np.arctan2(np.sin(tx).mean(), np.cos(tx).mean())
    dy = ((ty - my + np.pi) % (2*np.pi) - np.pi) * Y / (2*np.pi)
    dx = ((tx - mx + np.pi) % (2*np.pi) - np.pi) * X / (2*np.pi)
    vy = float(np.var(dy)); vx = float(np.var(dx))
    cyx = float(((dy - dy.mean()) * (dx - dx.mean())).mean())
    return max(0.0, vy * vx - cyx ** 2)


def count_components(b):
    if b.sum() == 0: return 0
    _, n = ndimage.label(b, structure=CONN)
    return int(n)


def adaptive_top_n_binary(prob_2d, n_keep):
    flat = prob_2d.ravel()
    if n_keep >= flat.size:
        return (flat > 0).astype(np.uint8).reshape(prob_2d.shape)
    kth = np.partition(flat, -n_keep)[-n_keep]
    binary = (prob_2d >= kth).astype(np.uint8)
    if int(binary.sum()) > n_keep:
        excess = int(binary.sum()) - n_keep
        ties = np.argwhere(prob_2d == kth)
        for (yy, xx) in ties[::-1]:
            if excess <= 0: break
            if binary[yy, xx] == 1:
                binary[yy, xx] = 0
                excess -= 1
    return binary


def gather_world(truth_path, samples_path, K):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)  # [E, S, Y, X]

    n_use = min(truth.shape[0], samples.shape[1])
    truth = truth[:n_use]; samples = samples[:, :n_use]
    n_ens = samples.shape[0]

    idx = [s for s in range(n_use) if int(truth[s].sum()) > K]
    if not idx: return None
    truth_m = truth[idx]
    samples_m = samples[:, idx]

    truth_range = truth_m.sum(axis=(1, 2)).astype(np.int32)

    # Per-species probability map (ensemble mean)
    p_mean = samples_m.mean(axis=0)  # [S_meaningful, Y, X]

    # The diagnostic statistic: total expected mass per species
    mass = p_mean.sum(axis=(1, 2)).astype(np.float64)  # [S_meaningful]

    # Also compute: predicted range under MASS-BASED top-N
    # (use ensemble mean map, threshold at top-mass cells per species)
    mass_range = np.zeros(len(idx), dtype=np.int32)
    mass_ncomp = np.zeros(len(idx), dtype=np.int32)
    mass_logcd = np.zeros(len(idx), dtype=np.float64)
    for s in range(len(idx)):
        n_keep = int(np.clip(round(mass[s]), 2, GRID_Y * GRID_X // 2))
        b = adaptive_top_n_binary(p_mean[s], n_keep)
        mass_range[s] = int(b.sum())
        mass_ncomp[s] = count_components(b)
        mass_logcd[s] = np.log10(periodic_cov_det(b) + 1.0)

    # Truth statistics for the same species
    truth_ncomp = np.asarray([count_components(t) for t in truth_m], dtype=np.int32)
    truth_logcd = np.asarray([np.log10(periodic_cov_det(t) + 1.0)
                                for t in truth_m], dtype=np.float64)

    # Also compute optimal-N per species (for diagnostic):
    # what N would EACH species need to match its truth range?
    optimal_n = truth_range.copy()  # by definition, optimal N = true range

    return {
        'truth_range': truth_range,
        'truth_ncomp': truth_ncomp,
        'truth_logcd': truth_logcd,
        'mass': mass,
        'mass_range': mass_range,
        'mass_ncomp': mass_ncomp,
        'mass_logcd': mass_logcd,
        'optimal_n':  optimal_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide-range-csv',    required=True)
    ap.add_argument('--recon-dir-pattern', required=True)
    ap.add_argument('--truth-dir',         required=True)
    ap.add_argument('--K',         type=int, default=5)
    ap.add_argument('--top-n-worlds', type=int, default=30)
    ap.add_argument('--output-dir',  required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_sp_count = defaultdict(int)
    with open(args.wide_range_csv) as f:
        for row in csv.DictReader(f):
            world_sp_count[row['world']] += 1
    top_worlds = sorted(world_sp_count.items(),
                         key=lambda x: -x[1])[:args.top_n_worlds]

    print(f"\n  Computing per-species probability mass across "
          f"{len(top_worlds)} worlds (K = {args.K}) ...\n")

    pooled = {k: [] for k in ['truth_range', 'truth_ncomp', 'truth_logcd',
                                'mass', 'mass_range', 'mass_ncomp', 'mass_logcd',
                                'optimal_n']}
    truth_dir = Path(args.truth_dir)
    for world_name, _ in top_worlds:
        stem = world_name.replace('.npz', '')
        truth_path = truth_dir / world_name
        samples_path = (Path(args.recon_dir_pattern.format(world_stem=stem))
                          / f'recon_fixed_b{args.K}_samples.npz')
        if not (truth_path.exists() and samples_path.exists()):
            print(f"    skip {world_name[:55]}: missing")
            continue
        r = gather_world(truth_path, samples_path, args.K)
        if r is None: continue
        for k in pooled: pooled[k].append(r[k])
        print(f"    {world_name[:55]:55s}  n={len(r['truth_range']):4d}")

    if not pooled['truth_range']:
        print("\n  No usable worlds."); return

    for k in pooled: pooled[k] = np.concatenate(pooled[k])
    print(f"\n  Pooled n = {len(pooled['truth_range']):,}\n")

    truth_range = pooled['truth_range']
    mass        = pooled['mass']

    # Correlations
    rho_mass,  p_mass  = stats.spearmanr(mass, truth_range)
    r_mass,    pr_mass = stats.pearsonr(mass, truth_range)

    # Per-bucket means
    buckets = [
        ('EASY',     6,  10,  '#1f558e'),
        ('MODERATE', 11, 20,  '#9e6420'),
        ('HARD',     21, 10**6, '#8e1d2a'),
    ]
    bucket_stats = []
    for name, lo, hi, col in buckets:
        m_bucket = (truth_range >= lo) & (truth_range <= hi)
        n = int(m_bucket.sum())
        if n == 0:
            bucket_stats.append((name, lo, hi, n, np.nan, np.nan, np.nan, col))
            continue
        bucket_stats.append((
            name, lo, hi, n,
            float(mass[m_bucket].mean()),
            float(mass[m_bucket].std()),
            float(truth_range[m_bucket].mean()),
            col,
        ))

    # ── Figure 1: probability mass vs truth range ──
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.30,
                           top=0.88, bottom=0.09, left=0.06, right=0.99)

    # (A) scatter of mass vs truth_range, colored by bucket
    ax = fig.add_subplot(gs[0, :2])
    for name, lo, hi, n, mass_mean, mass_std, tr_mean, col in bucket_stats:
        m_bucket = (truth_range >= lo) & (truth_range <= hi)
        ax.scatter(truth_range[m_bucket], mass[m_bucket], s=6, alpha=0.25,
                    color=col, edgecolor='none',
                    label=f'{name} (n={n:,}, mean mass={mass_mean:.1f})')
    # Reference line: if mass perfectly tracked truth (y = x)
    lim = max(truth_range.max(), mass.max()) + 2
    ax.plot([0, lim], [0, lim], '--', color='black', linewidth=1.6,
            label='mass = truth range (perfect)', alpha=0.7)
    ax.set_xlabel('Truth range size (cells)', fontsize=11)
    ax.set_ylabel(r'Per-species probability mass $\sum_{cells} p$', fontsize=11)
    ax.set_title(
        f'(A) Does the model probability mass scale with truth range?\n'
        f'Spearman \u03c1 = {rho_mass:.3f},  Pearson r = {r_mass:.3f}',
        fontweight='bold', fontsize=11)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.93)
    ax.grid(alpha=0.25)

    # (B) per-bucket boxplot of mass
    ax = fig.add_subplot(gs[0, 2])
    box_data = []; box_labels = []; box_colors = []
    for name, lo, hi, n, *_ , col in bucket_stats:
        if n > 0:
            m_bucket = (truth_range >= lo) & (truth_range <= hi)
            box_data.append(mass[m_bucket])
            box_labels.append(f'{name}\n(n={n:,})')
            box_colors.append(col)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                     widths=0.6, medianprops=dict(color='black', linewidth=1.5))
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.5)
    ax.set_ylabel(r'Probability mass $\sum_{cells} p$', fontsize=11)
    ax.set_title(
        f'(B) Mass by bucket\n'
        f'If HARD bucket mass = EASY mass, the model is mass-capped.',
        fontweight='bold', fontsize=10.5)
    ax.grid(alpha=0.25, axis='y')

    # (C) per-bucket mean mass vs mean truth (bar chart)
    ax = fig.add_subplot(gs[1, 0])
    names = [b[0] for b in bucket_stats if b[3] > 0]
    mass_means = [b[4] for b in bucket_stats if b[3] > 0]
    truth_means = [b[6] for b in bucket_stats if b[3] > 0]
    colors = [b[7] for b in bucket_stats if b[3] > 0]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, truth_means, w, label='Truth range',
                    color='#0c3d6b', alpha=0.85)
    bars2 = ax.bar(x + w/2, mass_means, w, label='Model probability mass',
                    color='#d97847', alpha=0.85)
    for b, v in zip(bars1, truth_means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}',
                 ha='center', fontsize=9)
    for b, v in zip(bars2, mass_means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}',
                 ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('Cells', fontsize=11)
    ax.set_title('(C) Per-bucket means: truth range vs probability mass',
                  fontweight='bold', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.25, axis='y')

    # (D) KS table: mass-based N_hat per bucket
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis('off')
    # Compute mass-based KS per bucket
    rows = [['Bucket', 'n_truth', 'KS range\n(mass-based)', 'KS conn\n(mass-based)', 'KS spread\n(mass-based)']]
    summary_rows = []
    for name, lo, hi, n, *_ , col in bucket_stats:
        if n == 0:
            rows.append([name, '0', '--', '--', '--'])
            continue
        m_bucket = (truth_range >= lo) & (truth_range <= hi)
        ksr = stats.ks_2samp(pooled['truth_range'][m_bucket],
                              pooled['mass_range'][m_bucket]).statistic
        ksc = stats.ks_2samp(pooled['truth_ncomp'][m_bucket],
                              pooled['mass_ncomp'][m_bucket]).statistic
        kss = stats.ks_2samp(pooled['truth_logcd'][m_bucket],
                              pooled['mass_logcd'][m_bucket]).statistic
        summary_rows.append({
            'bucket': name, 'lo': lo, 'hi': hi, 'n_truth': n,
            'mass_ks_range': float(ksr),
            'mass_ks_conn':  float(ksc),
            'mass_ks_spread': float(kss),
            'mass_mean': float(mass[m_bucket].mean()),
            'truth_mean': float(truth_range[m_bucket].mean()),
        })
        def _fmt(v):
            ver = ('EXCELLENT' if v <= 0.10 else 'PASS' if v <= 0.30
                   else 'MARGINAL' if v <= 0.50 else 'FAIL')
            return f'{v:.3f} \u2192 {ver}'
        rows.append([name, f'{n:,}', _fmt(ksr), _fmt(ksc), _fmt(kss)])

    table = ax.table(cellText=rows, cellLoc='center', loc='upper center',
                      colWidths=[0.12, 0.10, 0.26, 0.26, 0.26])
    table.auto_set_font_size(False); table.set_fontsize(10.5)
    table.scale(1, 1.9)
    # Color the header row
    for j in range(len(rows[0])):
        table[(0, j)].set_facecolor('#2c4a6e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    # Color verdicts
    for i, row in enumerate(rows[1:], 1):
        for j in range(2, 5):
            cell_text = row[j]
            if 'EXCELLENT' in cell_text: color = '#d4f0d4'
            elif 'PASS'     in cell_text: color = '#e4f5e4'
            elif 'MARGINAL' in cell_text: color = '#ffe5cc'
            elif 'FAIL'     in cell_text: color = '#ffd4d4'
            else:                          color = '#f0f0f0'
            table[(i, j)].set_facecolor(color)
    ax.set_title('(D) If we use MASS as the per-species top-N: KS by bucket\n'
                  '(If HARD KS values are LOW here, we have a real post-hoc fix.)',
                  fontweight='bold', fontsize=11, pad=20)

    fig.suptitle(
        'Diagnostic: is MetaDiffusion probability-mass-capped per species?',
        fontweight='bold', fontsize=14)

    # Verdict in the footer
    if r_mass < 0.30:
        verdict = ('VERDICT: r < 0.30. The model is approximately MASS-CAPPED. '
                    'It allocates similar total probability mass per species '
                    'regardless of truth range. Post-hoc fixes from the model\u2019s '
                    'own output are insufficient; we need a training-time fix '
                    '(range-aware loss)')
    elif r_mass < 0.60:
        verdict = ('VERDICT: r is in [0.30, 0.60]. Mass partially tracks range. '
                    'Mass-based N_hat helps but does not fully recover wide ranges. '
                    'Combine with K=10 observations or a discriminator.')
    else:
        verdict = ('VERDICT: r >= 0.60. Mass tracks truth range. Mass-based '
                    'N_hat is a viable post-hoc fix; no retraining required.')

    fig.text(0.5, 0.02, verdict, ha='center', fontsize=10.5, style='italic',
              color='#444', wrap=True,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbea',
                        edgecolor='#cca64a', linewidth=1.2))

    out_fig = out_dir / 'probability_mass_vs_truth_range.png'
    plt.savefig(out_fig, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  \u2713 figure  \u2192 {out_fig}")

    # CSV
    csv_path = out_dir / 'mass_based_topN_per_bucket.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket', 'lo', 'hi', 'n_truth', 'truth_mean',
                     'mass_mean', 'mass_ks_range', 'mass_ks_conn',
                     'mass_ks_spread'])
        for r in summary_rows:
            w.writerow([r['bucket'], r['lo'], r['hi'], r['n_truth'],
                         r['truth_mean'], r['mass_mean'],
                         r['mass_ks_range'], r['mass_ks_conn'],
                         r['mass_ks_spread']])
    print(f"  \u2713 csv     \u2192 {csv_path}")

    # Text summary
    txt_path = out_dir / 'probability_mass_diagnosis.txt'
    with open(txt_path, 'w') as f:
        f.write("PROBABILITY MASS DIAGNOSTIC\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"K = {args.K},  pooled n = {len(truth_range):,}\n\n")
        f.write(f"Correlation: probability mass vs truth range size\n")
        f.write(f"   Spearman rho = {rho_mass:.4f}  (p = {p_mass:.2e})\n")
        f.write(f"   Pearson  r   = {r_mass:.4f}  (p = {pr_mass:.2e})\n\n")
        f.write("Per-bucket means:\n")
        f.write(f"   {'Bucket':<10s} {'n':>6s} {'mass_mean':>10s} {'truth_mean':>11s} {'ratio':>8s}\n")
        for r in summary_rows:
            ratio = r['mass_mean'] / max(r['truth_mean'], 0.01)
            f.write(f"   {r['bucket']:<10s} {r['n_truth']:>6d} "
                    f"{r['mass_mean']:>10.2f} {r['truth_mean']:>11.2f} "
                    f"{ratio:>8.2f}\n")
        f.write(f"\nMass-based top-N: KS values by bucket\n")
        f.write(f"   {'Bucket':<10s} {'range':>8s} {'conn':>8s} {'spread':>8s}\n")
        for r in summary_rows:
            f.write(f"   {r['bucket']:<10s} {r['mass_ks_range']:>8.3f} "
                    f"{r['mass_ks_conn']:>8.3f} {r['mass_ks_spread']:>8.3f}\n")
        f.write("\n" + verdict + "\n")
    print(f"  \u2713 summary \u2192 {txt_path}\n")

    print("=" * 60)
    print("  HEADLINE: " + verdict.replace('VERDICT: ', ''))
    print("=" * 60)


if __name__ == "__main__":
    main()