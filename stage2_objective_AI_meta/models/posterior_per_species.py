#!/usr/bin/env python3
"""
=============================================================================
POSTERIOR PER-SPECIES FIGURE  —  the ensemble as an approximate posterior
=============================================================================
Two figures from one run:

  (1) MAPS  <output>.png
      per species: TRUTH | OBSERVED(K) | SAMPLE 1..n | POSTERIOR P | UNION
      Each union recall is shown next to a random-placement CHANCE floor.

  (2) CONDITIONAL POSTERIOR  <output>_conditional.png
      For one fixed set of observations (one species), draw the ensemble and
      build the distribution of a range statistic, then mark where the TRUE
      value falls and report its percentile (the share of samples below the
      truth). Across many species a well-calibrated ensemble gives uniform
      percentiles. Range SIZE is held at the true value by the top-N rule,
      so the statistics that vary are CONNECTED PATCHES and SPATIAL SPREAD
      (the same definitions used by the distribution figure).

Top-N convention (N = true range) matches the three-map figure, so the model
'union' equals its 'ens=' for the same world AND same species.
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
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)   # 4-connectivity

PALETTE = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377',
           '#66CCEE', '#EE8866']

COL_TP = (0.20, 0.60, 0.32, 1.0)        # true cell covered by >=1 sample
COL_FN = (0.80, 0.80, 0.83, 1.0)        # true cell missed by ALL samples
COL_SUPPORT = (0.72, 0.82, 0.94, 1.0)   # posterior support, not truth
COL_BG = (1.0, 1.0, 1.0, 1.0)
COL_TRUTH_LINE = '#CC3311'


# ---------------------------------------------------------------------------
# Range statistics  (identical definitions to the distribution figure)
# ---------------------------------------------------------------------------
def periodic_cov_det(binary_range, Y=GRID_Y, X=GRID_X):
    """Toroidal (PBC-corrected) determinant of the coordinate covariance."""
    yy, xx = np.where(binary_range > 0.5)
    n = len(yy)
    if n < 2:
        return 0.0
    ty = 2.0 * np.pi * yy.astype(np.float64) / Y
    tx = 2.0 * np.pi * xx.astype(np.float64) / X
    mty = np.arctan2(np.sin(ty).mean(), np.cos(ty).mean())
    mtx = np.arctan2(np.sin(tx).mean(), np.cos(tx).mean())
    dy = ((ty - mty + np.pi) % (2.0 * np.pi) - np.pi) * Y / (2.0 * np.pi)
    dx = ((tx - mtx + np.pi) % (2.0 * np.pi) - np.pi) * X / (2.0 * np.pi)
    var_y = float(np.var(dy)); var_x = float(np.var(dx))
    cov = float(((dy - dy.mean()) * (dx - dx.mean())).mean())
    return max(0.0, var_y * var_x - cov ** 2)


def count_components(binary_map):
    """Connected components, 4-connectivity, on a TORUS (periodic edges) to
    match periodic_cov_det and the toroidal IBM grid. A patch that wraps across
    the boundary counts as ONE (non-periodic labelling would over-count it)."""
    m = np.asarray(binary_map) > 0.5
    if m.sum() == 0:
        return 0
    lab, n = ndimage.label(m, structure=CONNECTIVITY_STRUCTURE)
    parent = list(range(n + 1))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    Y, X = m.shape
    for x in range(X):
        if m[0, x] and m[Y - 1, x]:
            union(lab[0, x], lab[Y - 1, x])
    for y in range(Y):
        if m[y, 0] and m[y, X - 1]:
            union(lab[y, 0], lab[y, X - 1])
    return len({find(lab[y, x]) for y in range(Y) for x in range(X) if m[y, x]})


def pit_percentile(sample_vals, truth_val):
    """Share of ensemble values below the true value (ties count half).
    This is the percentile Axel asks for; uniform across species = calibrated."""
    s = np.asarray(sample_vals, float)
    if s.size == 0:
        return float('nan')
    return float((np.sum(s < truth_val) + 0.5 * np.sum(s == truth_val)) / s.size)


# ---------------------------------------------------------------------------
# Core ensemble maths  (top-N per sample, N = true range)
# ---------------------------------------------------------------------------
def sample_binaries_topN(samples_sp, n_target, smooth_sigma=0.0):
    """Top-N binarisation per sample (N = true range size). If smooth_sigma>0 a
    TOROIDAL Gaussian blur is applied to the probability field first, so the
    top-N cells cluster — an explicit, DISCLOSED spatial-contiguity prior that
    trades a little per-cell recall for less range fragmentation."""
    n_ens = samples_sp.shape[0]
    out = np.zeros((n_ens, GRID_Y, GRID_X), dtype=bool)
    if n_target <= 0:
        return out
    for k in range(n_ens):
        field = samples_sp[k].astype(np.float64)
        if smooth_sigma and smooth_sigma > 0:
            field = ndimage.gaussian_filter(field, smooth_sigma, mode='wrap')
        flat = field.ravel()
        if flat.max() < 1e-6:
            continue
        n = min(n_target, flat.size)
        idx = np.argpartition(flat, -n)[-n:]
        b = np.zeros(flat.size, dtype=bool); b[idx] = True
        out[k] = b.reshape(GRID_Y, GRID_X)
    return out


def posterior_frequency(binaries):
    if binaries.shape[0] == 0:
        return np.zeros((GRID_Y, GRID_X))
    return binaries.mean(axis=0)


def union_recall_novel(truth_sp, binaries, obs_sp):
    t = truth_sp > 0; novel = ~(obs_sp > 0)
    n = int((t & novel).sum())
    if n == 0:
        return float('nan')
    return int((t & novel & binaries.any(axis=0)).sum()) / n


def one_sample_mean_recall_novel(truth_sp, binaries, obs_sp):
    t = truth_sp > 0; novel = ~(obs_sp > 0)
    n = int((t & novel).sum())
    if n == 0 or binaries.shape[0] == 0:
        return float('nan')
    return float(np.mean([int((t & novel & binaries[k]).sum()) / n
                          for k in range(binaries.shape[0])]))


def pairwise_jaccard_distance(binaries):
    n = binaries.shape[0]; d = []
    for i in range(n):
        for j in range(i + 1, n):
            u = int((binaries[i] | binaries[j]).sum())
            if u:
                d.append(1.0 - int((binaries[i] & binaries[j]).sum()) / u)
    return float(np.mean(d)) if d else 0.0


def random_floor_novel(truth_sp, obs_sp, n_target, n_ens, n_trials=200, seed=0):
    """Chance floor scored exactly like the model: place n_target cells at
    random; mean per-sample and mean union recall on NOVEL cells over trials."""
    t = (truth_sp > 0) & ~(obs_sp > 0)
    denom = int(t.sum())
    if denom == 0 or n_target <= 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    ncells = GRID_Y * GRID_X; n = min(int(n_target), ncells)
    flat_t = t.ravel()
    one_vals = np.empty(n_trials); uni_vals = np.empty(n_trials)
    for ti in range(n_trials):
        covered = np.zeros(ncells, dtype=bool); first = np.zeros(ncells, dtype=bool)
        for e in range(n_ens):
            idx = rng.choice(ncells, size=n, replace=False)
            if e == 0:
                first[idx] = True
            covered[idx] = True
        one_vals[ti] = int((flat_t & first).sum()) / denom
        uni_vals[ti] = int((flat_t & covered).sum()) / denom
    return float(one_vals.mean()), float(uni_vals.mean())


# ---------------------------------------------------------------------------
# Per-species metrics (single source of truth for both figures + CSV)
# ---------------------------------------------------------------------------
def species_metrics(truth, samples, observed, sp, n_ens, rand_trials, smooth_sigma=0.0):
    n_target = int(truth[sp].sum())
    binaries = sample_binaries_topN(samples[:, sp], n_target, smooth_sigma=smooth_sigma)
    post = posterior_frequency(binaries)
    ens_rec = union_recall_novel(truth[sp], binaries, observed[sp])
    one_rec = one_sample_mean_recall_novel(truth[sp], binaries, observed[sp])
    jac = pairwise_jaccard_distance(binaries)
    rand_one, rand_uni = random_floor_novel(truth[sp], observed[sp], n_target,
                                            n_ens, n_trials=rand_trials,
                                            seed=12345 + int(sp))
    ratio = (ens_rec / rand_uni) if (rand_uni and rand_uni > 0
                                     and not np.isnan(ens_rec)) else float('nan')
    # conditional shape distributions across the ensemble (size fixed = n_target)
    patches = np.array([count_components(binaries[k]) for k in range(n_ens)], float)
    spread = np.array([np.log10(periodic_cov_det(binaries[k]) + 1.0)
                       for k in range(n_ens)], float)
    patch_truth = float(count_components(truth[sp] > 0))
    spread_truth = float(np.log10(periodic_cov_det(truth[sp] > 0) + 1.0))
    return dict(n_target=n_target, binaries=binaries, post=post,
                ens_rec=ens_rec, one_rec=one_rec, jac=jac,
                rand_one=rand_one, rand_uni=rand_uni, ratio=ratio,
                patches=patches, spread=spread,
                patch_truth=patch_truth, spread_truth=spread_truth,
                patch_pct=pit_percentile(patches, patch_truth),
                spread_pct=pit_percentile(spread, spread_truth))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def binary_rgba(binary_map, rgb):
    pale = np.array(rgb) * 0.18 + 0.82
    img = np.empty((GRID_Y, GRID_X, 4), np.float32)
    for y in range(GRID_Y):
        for x in range(GRID_X):
            img[y, x] = (*rgb, 1.0) if binary_map[y, x] > 0.5 else (*pale[:3], 1.0)
    return img


def union_vs_truth_rgba(truth_sp, binaries):
    t = truth_sp > 0
    u = binaries.any(axis=0) if binaries.shape[0] else np.zeros_like(t)
    img = np.empty((GRID_Y, GRID_X, 4), np.float32)
    for y in range(GRID_Y):
        for x in range(GRID_X):
            tt, uu = t[y, x], u[y, x]
            if tt and uu:        img[y, x] = COL_TP
            elif tt and not uu:  img[y, x] = COL_FN
            elif uu and not tt:  img[y, x] = COL_SUPPORT
            else:                img[y, x] = COL_BG
    return img


def add_grid(ax):
    for k in range(GRID_X + 1):
        ax.axvline(k - 0.5, color='#dddddd', lw=0.3, zorder=0)
    for k in range(GRID_Y + 1):
        ax.axhline(k - 0.5, color='#dddddd', lw=0.3, zorder=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, GRID_X - 0.5); ax.set_ylim(GRID_Y - 0.5, -0.5)


def overlay_truth_outline(ax, truth_sp, colour='red'):
    for y, x in np.argwhere(truth_sp > 0):
        ax.add_patch(mpatches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                     edgecolor=colour, facecolor='none', lw=1.0))


def overlay_obs(ax, obs_sp):
    for y, x in np.argwhere(obs_sp > 0):
        ax.add_patch(mpatches.Circle((x, y), 0.32, facecolor='yellow',
                     edgecolor='black', lw=0.6))


def append_csv(csv_path, rows, fieldnames):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Species selection (deterministic; prefer wide-range species)
# ---------------------------------------------------------------------------
def pick_species(truth, mean_pred, K, n_species, mode='topwide', seed=0):
    """Choose species. 'topwide' (default) = widest/highest-confidence (stable
    shape statistics but biased toward wide ranges). 'random' = a representative
    random draw across the whole range-size distribution (deterministic given
    seed) — the honest sample for a calibration/PIT."""
    ranges = truth.sum(axis=(1, 2)).astype(int)
    if mode == 'random':
        eligible = [s for s in range(truth.shape[0]) if ranges[s] > K]
        rng = np.random.default_rng(seed)
        if len(eligible) <= n_species:
            return eligible
        return sorted(rng.choice(eligible, size=n_species, replace=False).tolist())

    def conf(s):
        flat = mean_pred[s].ravel()
        if ranges[s] <= 0 or flat.max() < 1e-6:
            return 0.0
        n = min(ranges[s], flat.size)
        return float(np.partition(flat, -n)[-n:].sum())

    cands = [(s, ranges[s], conf(s)) for s in range(truth.shape[0])
             if ranges[s] >= max(K + 5, 6)]
    if len(cands) < n_species:
        cands = [(s, ranges[s], conf(s)) for s in range(truth.shape[0])
                 if ranges[s] > K]
    cands.sort(key=lambda c: (-c[2], -c[1]))
    return [c[0] for c in cands[:n_species]]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def parse_params(world_name):
    out = {}
    for tok in world_name.replace('.npz', '').split('_'):
        for key in ('thr', 'env', 'dr'):
            if tok.startswith(key) and len(tok) > len(key):
                out[key] = tok[len(key):].replace('p', '.').replace('em0', 'e-0')
    return out


def load_world(truth_dir, recon_pattern, world_stem, K, recon_filename=None):
    truth_path = Path(truth_dir) / f'{world_stem}.npz'
    recon_dir = Path(recon_pattern.format(world_stem=world_stem))
    fname = recon_filename if recon_filename else f'recon_fixed_b{K}_samples.npz'
    samples_path = recon_dir / fname
    if not truth_path.exists() or not samples_path.exists():
        raise FileNotFoundError(f"need:\n  {truth_path}\n  {samples_path}")
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    mean_pred = (np.asarray(z['mean']).astype(np.float32)
                 if 'mean' in z.files else samples.mean(axis=0))
    if 'noisy_input' in z.files:
        observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    elif 'obs_mask' in z.files:
        observed = np.asarray(z['obs_mask']).astype(np.uint8)
    else:
        observed = (samples.mean(axis=0) >= 0.99).astype(np.uint8)
    n = min(truth.shape[0], samples.shape[1], observed.shape[0])
    return {'world': f'{world_stem}.npz', 'truth': truth[:n],
            'samples': samples[:, :n], 'mean_pred': mean_pred[:n],
            'observed': observed[:n]}


# ---------------------------------------------------------------------------
# Figure 1: maps
# ---------------------------------------------------------------------------
def draw_maps_figure(per_sp, chosen, world, params, K, n_ens, n_samples_shown, output_path, obs_label=None):
    obs_col = obs_label if obs_label else f'K={K}'
    obs_txt = obs_label if obs_label else f'K={K} obs/species'
    truth, observed = world['truth'], world['observed']
    n_rows = len(chosen)
    n_cols = 2 + n_samples_shown + 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.15 * n_cols, 2.35 * n_rows), squeeze=False)
    col_titles = (['TRUTH', f'OBSERVED ({obs_col})']
                  + [f'SAMPLE {i + 1}' for i in range(n_samples_shown)]
                  + [f'POSTERIOR P\n({n_ens}-sample frequency)', 'ENSEMBLE UNION\nvs truth'])
    assert n_cols == len(col_titles), (n_cols, len(col_titles))

    posterior_im = None
    for r, sp in enumerate(chosen):
        m = per_sp[sp]; rgb = mcolors.to_rgb(PALETTE[r % len(PALETTE)])
        binaries, post = m['binaries'], m['post']
        c = 0
        ax = axes[r, c]; ax.imshow(binary_rgba(truth[sp], rgb), interpolation='nearest')
        add_grid(ax); overlay_obs(ax, observed[sp]); c += 1
        ax = axes[r, c]; ax.imshow(binary_rgba(observed[sp], rgb), interpolation='nearest')
        add_grid(ax); c += 1
        for i in range(n_samples_shown):
            ax = axes[r, c]; ax.imshow(binary_rgba(binaries[i], rgb), interpolation='nearest')
            add_grid(ax); overlay_truth_outline(ax, truth[sp]); c += 1
        ax = axes[r, c]
        posterior_im = ax.imshow(post, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        add_grid(ax); overlay_truth_outline(ax, truth[sp], colour='cyan')
        overlay_obs(ax, observed[sp]); c += 1
        ax = axes[r, c]
        ax.imshow(union_vs_truth_rgba(truth[sp], binaries), interpolation='nearest')
        add_grid(ax); c += 1

        axes[r, 0].set_ylabel(f"sp #{sp}\nrange N={m['n_target']}", rotation=0,
                              ha='right', va='center', labelpad=40, fontsize=9,
                              fontweight='bold', color=rgb)
        es = f"{m['ens_rec']:.0%}" if not np.isnan(m['ens_rec']) else "n/a"
        os_ = f"{m['one_rec']:.0%}" if not np.isnan(m['one_rec']) else "n/a"
        rs = f"{m['rand_uni']:.0%}" if not np.isnan(m['rand_uni']) else "n/a"
        ros = f"{m['rand_one']:.0%}" if not np.isnan(m['rand_one']) else "n/a"
        ratio_s = f"{m['ratio']:.1f}\u00d7" if not np.isnan(m['ratio']) else "n/a"
        axes[r, n_cols - 1].text(
            0.5, -0.14,
            f"union = {es}  (chance {rs}, {ratio_s})\n"
            f"one sample \u2248 {os_} (chance {ros})   Jaccard = {m['jac']:.2f}",
            transform=axes[r, n_cols - 1].transAxes, ha='center', va='top',
            fontsize=7.5, family='monospace', color='#333')

    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=9.5, fontweight='bold', pad=6)

    cax = fig.add_axes([0.915, 0.30, 0.012, 0.40])
    cbar = fig.colorbar(posterior_im, cax=cax)
    cbar.set_label('P(cell occupied | observations)\n= fraction of samples', fontsize=8)

    fig.suptitle(
        "The ensemble as an approximate posterior: many plausible reconstructions "
        "from K sparse observations\n"
        f"World: thr={params.get('thr', '?')}, env={params.get('env', '?')}, "
        f"dr={params.get('dr', '?')}   |   {obs_txt}   |   {n_ens}-sample ensemble",
        fontweight='bold', fontsize=12.5, y=0.99)
    fig.text(
        0.5, 0.01,
        "Each SAMPLE is one reconstruction at top-N (N = true range): the model returns "
        "DIFFERENT plausible ranges from the same K points (red outline = truth).  "
        f"POSTERIOR P = fraction of the {n_ens} samples placing the species in each cell "
        "(cyan outline = truth).  ENSEMBLE UNION vs truth: green = true cell covered by "
        "\u22651 sample, grey = missed by all, pale blue = posterior support outside truth.  "
        "Under each row: ensemble union recall on novel cells with, in brackets, the CHANCE "
        "level (random placement of the same N-cell budget) and the union/chance ratio; "
        "union must beat chance to count.",
        ha='center', va='bottom', fontsize=8, style='italic', color='#555', wrap=True)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.11,
                        hspace=0.32, wspace=0.08)
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: conditional posterior of range shape (Axel's "where does truth fall")
# ---------------------------------------------------------------------------
def draw_conditional_figure(per_sp, chosen, params, K, n_ens, output_path, obs_label=None):
    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.6, 1.95 * n_rows), squeeze=False)
    for r, sp in enumerate(chosen):
        m = per_sp[sp]
        # connected patches (integer-valued)
        ax = axes[r, 0]; vals = m['patches']
        lo = int(min(vals.min(), m['patch_truth'])); hi = int(max(vals.max(), m['patch_truth']))
        ax.hist(vals, bins=np.arange(lo - 0.5, hi + 1.5, 1),
                color='#4477AA', alpha=0.6, edgecolor='white')
        ax.axvline(m['patch_truth'], color=COL_TRUTH_LINE, lw=2.2)
        ax.set_title(f"sp #{sp} \u2014 connected patches\ntruth at {m['patch_pct']*100:.0f}th pctile",
                     fontsize=8.5)
        ax.set_yticks([])
        # spatial spread (continuous)
        ax = axes[r, 1]; vals = m['spread']
        ax.hist(vals, bins=6, color='#228833', alpha=0.6, edgecolor='white')
        ax.axvline(m['spread_truth'], color=COL_TRUTH_LINE, lw=2.2)
        ax.set_title(f"sp #{sp} \u2014 spatial spread\ntruth at {m['spread_pct']*100:.0f}th pctile",
                     fontsize=8.5)
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("connected patches", fontsize=9)
    axes[-1, 1].set_xlabel(r"$\log_{10}(\det\,\Sigma_{yx}+1)$", fontsize=9)

    fig.suptitle(
        "Conditional posterior of range shape: does the truth fall within the ensemble?\n"
        f"World: thr={params.get('thr', '?')}, env={params.get('env', '?')}, "
        f"dr={params.get('dr', '?')}   |   {obs_label if obs_label else f'K={K}'}   |   {n_ens} samples per species "
        "(range size held at the true value)",
        fontweight='bold', fontsize=11, y=0.99)
    fig.text(
        0.5, 0.005,
        "Each panel: the statistic across the ensemble samples for one species "
        "(red line = true value). The percentile is the share of samples below the truth; "
        "if the ensemble is well calibrated these percentiles are uniform across species. "
        "Coarse with few samples \u2014 it sharpens as the ensemble size grows.",
        ha='center', va='bottom', fontsize=7.5, style='italic', color='#555', wrap=True)
    plt.tight_layout(rect=[0, 0.045, 1, 0.92])
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def make_posterior_figure(world, output_path, K, n_species=4,
                          n_samples_shown=3, csv_path=None, rand_trials=200, obs_label=None,
                          csv_only=False, select='topwide', select_seed=0, smooth_sigma=0.0):
    truth, samples = world['truth'], world['samples']
    mean_pred, observed = world['mean_pred'], world['observed']
    n_ens = samples.shape[0]
    n_samples_shown = min(n_samples_shown, n_ens)
    params = parse_params(world['world'])

    chosen = pick_species(truth, mean_pred, K, n_species, mode=select, seed=select_seed)
    if not chosen:
        print("  no species with truth_range > K found"); return

    per_sp = {sp: species_metrics(truth, samples, observed, sp, n_ens, rand_trials,
                                  smooth_sigma=smooth_sigma)
              for sp in chosen}

    if not csv_only:
        output_path = Path(output_path)
        cond_path = output_path.with_name(output_path.stem + '_conditional' + output_path.suffix)
        draw_maps_figure(per_sp, chosen, world, params, K, n_ens, n_samples_shown, output_path, obs_label=obs_label)
        draw_conditional_figure(per_sp, chosen, params, K, n_ens, cond_path, obs_label=obs_label)
        print(f"  \u2713 maps figure        \u2192 {output_path}")
        print(f"  \u2713 conditional figure \u2192 {cond_path}")
    else:
        print(f"  \u2713 csv-only (no figures) | species: {chosen}")
    print(f"     {n_ens}-sample ensemble | species: {chosen}")
    rows = []
    for sp in chosen:
        m = per_sp[sp]
        es = f"{m['ens_rec']:.0%}" if not np.isnan(m['ens_rec']) else "n/a"
        os_ = f"{m['one_rec']:.0%}" if not np.isnan(m['one_rec']) else "n/a"
        rs = f"{m['rand_uni']:.0%}" if not np.isnan(m['rand_uni']) else "n/a"
        ratio_s = f"{m['ratio']:.1f}x" if not np.isnan(m['ratio']) else "n/a"
        print(f"     sp #{sp:>4d}  N={m['n_target']:>3d}  union={es:>4}  chance={rs:>4}  "
              f"({ratio_s})  patches truth@{m['patch_pct']*100:>3.0f}pct  "
              f"spread truth@{m['spread_pct']*100:>3.0f}pct")
        rows.append({
            'world_stem': world['world'].replace('.npz', ''),
            'species': int(sp), 'range_N': m['n_target'],
            'n_obs': int(observed[sp].sum()), 'K': K, 'n_ens': n_ens,
            'one_sample_recall':      '' if np.isnan(m['one_rec']) else round(float(m['one_rec']), 4),
            'rand_one_sample_recall': '' if np.isnan(m['rand_one']) else round(float(m['rand_one']), 4),
            'union_recall':           '' if np.isnan(m['ens_rec']) else round(float(m['ens_rec']), 4),
            'rand_union_recall':      '' if np.isnan(m['rand_uni']) else round(float(m['rand_uni']), 4),
            'union_over_chance':      '' if np.isnan(m['ratio'])   else round(float(m['ratio']), 2),
            'jaccard_diversity': round(float(m['jac']), 4),
            'patches_truth': int(m['patch_truth']), 'patches_pctile': round(m['patch_pct'], 3),
            'spread_truth': round(float(m['spread_truth']), 3),
            'spread_pctile': round(m['spread_pct'], 3),
        })
    if csv_path:
        append_csv(csv_path, rows,
                   ['world_stem', 'species', 'range_N', 'n_obs', 'K', 'n_ens',
                    'one_sample_recall', 'rand_one_sample_recall',
                    'union_recall', 'rand_union_recall', 'union_over_chance',
                    'jaccard_diversity', 'patches_truth', 'patches_pctile',
                    'spread_truth', 'spread_pctile'])
        print(f"  \u2713 metrics appended  \u2192 {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Posterior per-species figure (ensemble = approximate posterior)")
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--recon-dir-pattern', required=True, help="pattern with {world_stem}")
    ap.add_argument('--world-stem', required=True, help="world filename without .npz")
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--n-species', type=int, default=4)
    ap.add_argument('--n-samples-shown', type=int, default=3)
    ap.add_argument('--rand-trials', type=int, default=200)
    ap.add_argument('--output-path', default=None)
    ap.add_argument('--csv-path', default=None)
    ap.add_argument('--recon-filename', default=None,
                    help="exact samples filename in the recon dir (default recon_fixed_b{K}_samples.npz)")
    ap.add_argument('--obs-label', default=None,
                    help="label for the title/column, e.g. 'proportional (p=0.10)' (default 'K=<K>')")
    ap.add_argument('--csv-only', action='store_true',
                    help="compute metrics and append CSV only; skip rendering both figures")
    ap.add_argument('--select', choices=['topwide', 'random'], default='topwide',
                    help="species selection: topwide (default) or a representative random sample")
    ap.add_argument('--select-seed', type=int, default=0, help="seed for --select random")
    ap.add_argument('--smooth', type=float, default=0.0,
                    help="toroidal Gaussian sigma applied to each sample before top-N "
                         "(0 = off; a disclosed spatial-contiguity prior)")
    args = ap.parse_args()
    if not args.csv_only and not args.output_path:
        ap.error('--output-path is required unless --csv-only is set')
    world = load_world(args.truth_dir, args.recon_dir_pattern, args.world_stem, args.K,
                       recon_filename=args.recon_filename)
    make_posterior_figure(world, args.output_path, args.K, n_species=args.n_species,
                          n_samples_shown=args.n_samples_shown, csv_path=args.csv_path,
                          rand_trials=args.rand_trials, obs_label=args.obs_label,
                          csv_only=args.csv_only, select=args.select,
                          select_seed=args.select_seed, smooth_sigma=args.smooth)


if __name__ == "__main__":
    main()