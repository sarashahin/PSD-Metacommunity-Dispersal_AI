#!/usr/bin/env python3
"""
=============================================================================
axel_ecological_distribution_figure.py  (Figure 4, distributional realism)
Elsevier / Ecological Informatics, 190 mm double column
=============================================================================
Writes a vector PDF at exactly 190 x 125 mm, a 600-dpi PNG, a values CSV,
a per-community router CSV and, when copies are found, a CSV of the
evaluation files dropped as copies.

What changed on 4 Oct 2026
1. Copies are removed before anything is computed. Evaluation files whose
   final occupancy maps are identical (they differ only in the dispersal-
   rate label, which the simulator did not apply) are kept once. In each
   group the in_dist file is kept, and --regime is applied after removal.
   --no-dedupe restores the old behaviour.
2. Leak-free threshold router (--router nested, the default). For every
   held-out community the range-size predictor, the wide-range share, the
   class cut-off tau and the two probability cut-offs come from the other
   communities only. The cut-offs are chosen from --grid by the smallest
   sum of the three KS distances on the other communities, whose own
   range-size predictions are made leave-one-out inside that set.
   --router legacy reproduces the earlier numbers: fixed cut-offs
   (--threshold-hard 0.80, --threshold-moderate 0.95) and a wide-range
   share computed over all communities, including the one evaluated.
3. Drawing: plain tick labels on log axes; fixed ticks only on log axes
   (panel c had lost its ticks); axis labels 8.6 pt so subscripts print at
   6 pt; mean labels 7 pt; constrained layout; saved without
   bbox_inches='tight', so the PDF page is exactly 190 x 125 mm.
4. isotonic_fit applies the same pool-adjacent-violators rule as before,
   written as a stack: same blocks and values, much faster.

USAGE (from PSD_Dispersal_pool)
  python AI_simulation/stage2/models/axel_ecological_distribution_figure.py \
      --wide-range-csv    ./results/data/data_eval_unseen/eval_unseen_manifest.csv \
      --recon-dir-pattern './reconstructions_unseen/{world_stem}' \
      --truth-dir         ./results/data/data_eval_unseen \
      --K 10 --regime all \
      --output ./figures_map_axel_stage2_new/unseen_eval/fig4_v2_all/Fig4_distributional_realism

  Regression check (old method on the old 30 files): add --no-dedupe
  --router legacy and use another --output. It must print 759 true and
  6,072 reconstructed ranges and D = 0.139, 0.229, 0.102.
=============================================================================
"""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
from scipy import ndimage, stats

GRID_Y, GRID_X = 20, 20
WIDE_CELLS = 21                      # 'wide-ranging' = true range of 21 cells or more
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)
MM = 1.0 / 25.4
W_DOUBLE = 190 * MM                  # Elsevier double column
H_FIGURE = 125 * MM
RANGE_TICKS = (10, 20, 30, 50, 100, 200)
DENSITY_TICKS = (1e-4, 1e-3, 1e-2, 1e-1)
SPREAD_LABEL = 'Spatial spread\n' r'$\log_{10}(\det\Sigma+1)$'
DEFAULT_GRID = ('0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,'
                '0.80,0.85,0.90,0.95,0.975,0.99')

COL_TRUTH = '#26456E'                # dark blue, filled
COL_PRED = '#D1783C'                 # warm orange, step
COL_GREY = '#666666'


def _pick_font():
    for name in ('Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans'):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            print(f"  font: '{name}'")
            return name
        except Exception:
            continue
    return 'DejaVu Sans'


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [_pick_font()],
    'font.size': 7.5, 'axes.labelsize': 8.6, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'lines.linewidth': 1.2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
})

# ─────────────────────────────────────────────────────────────────────
#  SHAPE STATISTICS AND FEATURES (unchanged)
# ─────────────────────────────────────────────────────────────────────
INFORMATIVE_FEATURES = ('obs_logcd', 'top30_sum', 'top20_sum', 'top10_sum',
                        'prob_q95', 'prob_q90', 'prob_gini', 'peak_to_mean')


def _gini(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.sum() < 1e-9:
        return 0.0
    a = np.sort(a)
    n = a.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * a).sum() - (n + 1) * a.sum())
                 / (n * a.sum() + 1e-12))


def periodic_cov_det(binary_range, Y=GRID_Y, X=GRID_X):
    yy, xx = np.where(binary_range > 0.5)
    n = len(yy)
    if n < 2:
        return 0.0
    theta_y = 2.0 * np.pi * yy.astype(np.float64) / Y
    theta_x = 2.0 * np.pi * xx.astype(np.float64) / X
    mean_theta_y = np.arctan2(np.sin(theta_y).mean(), np.cos(theta_y).mean())
    mean_theta_x = np.arctan2(np.sin(theta_x).mean(), np.cos(theta_x).mean())
    diff_y = (theta_y - mean_theta_y + np.pi) % (2.0 * np.pi) - np.pi
    diff_x = (theta_x - mean_theta_x + np.pi) % (2.0 * np.pi) - np.pi
    dy = diff_y * Y / (2.0 * np.pi)
    dx = diff_x * X / (2.0 * np.pi)
    var_y = float(np.var(dy))
    var_x = float(np.var(dx))
    cov_yx = float(((dy - dy.mean()) * (dx - dx.mean())).mean())
    return max(0.0, var_y * var_x - cov_yx ** 2)


def compute_features_one_species(prob_map_ens, obs_cells):
    mean_prob = prob_map_ens.mean(axis=0)
    flat = mean_prob.ravel()
    obs_flat = obs_cells.ravel().astype(bool)
    flat_extrap = flat[~obs_flat] if (~obs_flat).any() else flat
    sorted_extrap = np.sort(flat_extrap)[::-1]
    return {
        'obs_logcd':    float(np.log10(periodic_cov_det(obs_cells) + 1.0)),
        'top30_sum':    float(sorted_extrap[:min(30, sorted_extrap.size)].sum()),
        'top20_sum':    float(sorted_extrap[:min(20, sorted_extrap.size)].sum()),
        'top10_sum':    float(sorted_extrap[:min(10, sorted_extrap.size)].sum()),
        'prob_q95':     float(np.quantile(flat_extrap, 0.95)),
        'prob_q90':     float(np.quantile(flat_extrap, 0.90)),
        'prob_gini':    float(_gini(flat_extrap)),
        'peak_to_mean': float(flat_extrap.max() / max(1e-9, flat_extrap.mean())),
    }


def count_components(binary_map):
    if binary_map.sum() == 0:
        return 0
    _, n = ndimage.label(binary_map, structure=CONNECTIVITY_STRUCTURE)
    return int(n)


# ─────────────────────────────────────────────────────────────────────
#  RANGE-SIZE PREDICTOR (ridge + isotonic, unchanged rule)
# ─────────────────────────────────────────────────────────────────────
def fit_ridge(X, y, alpha=1.0):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def isotonic_fit(x_raw, y_target):
    """Pool adjacent violators, merging only strict violations, left to right
    with backtracking; each block keeps the x of its first point. Same blocks
    and values as the earlier while-loop version."""
    order = np.argsort(x_raw)
    xs = x_raw[order].astype(np.float64)
    ys = y_target[order].astype(np.float64)
    bx, by, bw = [], [], []
    for x, y in zip(xs.tolist(), ys.tolist()):
        bx.append(x); by.append(y); bw.append(1.0)
        while len(by) > 1 and by[-2] > by[-1]:
            w = bw[-2] + bw[-1]
            by[-2] = (by[-2] * bw[-2] + by[-1] * bw[-1]) / w
            bw[-2] = w
            bx.pop(); by.pop(); bw.pop()
    return np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)


def isotonic_apply(x_s, y_s, x_q):
    return np.interp(x_q, x_s, y_s, left=y_s[0], right=y_s[-1])


def predict_range_size(X_tr, y_tr, X_te, alpha):
    """Ridge on standardised shape features, isotonic map to cells, clipped
    to the grid. Trained only on the rows passed in."""
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0); sd[sd < 1e-9] = 1.0
    Xtr = (X_tr - mu) / sd
    Xte = (X_te - mu) / sd
    beta = fit_ridge(Xtr, y_tr - y_tr.mean(), alpha=alpha)
    score_tr = Xtr @ beta + y_tr.mean()
    score_te = Xte @ beta + y_tr.mean()
    x_s, y_s = isotonic_fit(score_tr, y_tr)
    return np.clip(isotonic_apply(x_s, y_s, score_te), 1.0, GRID_Y * GRID_X)


# ─────────────────────────────────────────────────────────────────────
#  LOADING, COPY REMOVAL
# ─────────────────────────────────────────────────────────────────────
def _load_world(truth_path, samples_path, K):
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    if 'obs_mask' in z.files:
        obs_mask = np.asarray(z['obs_mask']).astype(np.uint8)
    elif 'noisy_input' in z.files:
        obs_mask = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    else:
        obs_mask = (samples.mean(axis=0) >= 0.99).astype(np.uint8)
    n_use = min(truth.shape[0], samples.shape[1], obs_mask.shape[0])
    truth = truth[:n_use]; samples = samples[:, :n_use]; obs_mask = obs_mask[:n_use]
    keep = [s for s in range(n_use) if int(truth[s].sum()) > K]
    return truth, samples, obs_mask, keep


def truth_hash(truth_path):
    with np.load(truth_path, allow_pickle=True) as td:
        a = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def select_communities(args):
    """Manifest order; files that exist; copies removed (in_dist copy kept);
    then the regime filter; then --top-n-worlds."""
    counts, regime = defaultdict(int), {}
    with open(args.wide_range_csv) as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        key = 'world' if 'world' in cols else 'filename'
        has_regime = 'regime' in cols
        for row in rdr:
            counts[row[key]] += 1
            regime.setdefault(row[key], row['regime'] if has_regime else '')
    if args.regime != 'all' and not has_regime:
        raise SystemExit("  --regime needs a 'regime' column in --wide-range-csv")
    ordered = [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])]

    truth_dir = Path(args.truth_dir)
    usable = []
    for wn in ordered:
        stem = wn.replace('.npz', '')
        tp = truth_dir / wn
        sp = Path(args.recon_dir_pattern.format(world_stem=stem)) / \
            f'recon_fixed_b{args.K}_samples.npz'
        if tp.exists() and sp.exists():
            usable.append((tp, sp, wn))
        else:
            print(f"    skip {wn[:55]}: missing truth or samples")

    dropped = []
    if args.dedupe:
        groups = defaultdict(list)
        for item in usable:
            groups[truth_hash(item[0])].append(item)
        keepers = set()
        for members in groups.values():
            keeper = next((m for m in members if regime.get(m[2]) == 'in_dist'), members[0])
            keepers.add(keeper[2])
            for m in members:
                if m[2] != keeper[2]:
                    dropped.append((m[2], keeper[2], regime.get(m[2], ''),
                                    regime.get(keeper[2], '')))
        usable = [u for u in usable if u[2] in keepers]
    if args.regime != 'all':
        usable = [u for u in usable if regime.get(u[2]) == args.regime]
    return usable[:args.top_n_worlds], dropped, regime


def load_communities(world_paths, K):
    print(f"\n  loading and featurising {len(world_paths)} communities ...")
    comms = []
    for tp, sp, wn in world_paths:
        truth, samples, obs_mask, keep = _load_world(tp, sp, K)
        if not keep:
            print(f"    skip {wn[:55]}: no species with range > {K}")
            continue
        feats = [compute_features_one_species(samples[:, s], obs_mask[s]) for s in keep]
        truth_m = truth[keep]
        comms.append(dict(
            name=wn,
            X=np.asarray([[f[fn] for fn in INFORMATIVE_FEATURES] for f in feats],
                         dtype=np.float64),
            y=truth_m.sum(axis=(1, 2)).astype(np.float64),
            tr=truth_m.sum(axis=(1, 2)).astype(np.int32),
            tn=np.asarray([count_components(t) for t in truth_m], dtype=np.int32),
            tl=np.asarray([np.log10(periodic_cov_det(t) + 1.0) for t in truth_m],
                          dtype=np.float64),
            samples=samples[:, keep]))
        print(f"    {wn[:55]:55s}  n_species = {len(keep):4d}")
    return comms


# ─────────────────────────────────────────────────────────────────────
#  BINARISED STATISTICS, ONE TABLE PER CUT-OFF LEVEL
# ─────────────────────────────────────────────────────────────────────
def binarised_stats(samples_keep, levels):
    """Size, patch count and log spread of every sample map at every level,
    each (n_levels, n_ens, n_sp). Maps with fewer than 2 cells are not
    counted later, as before."""
    G = len(levels)
    n_ens, n_sp = samples_keep.shape[:2]
    size = np.zeros((G, n_ens, n_sp), np.int32)
    comp = np.zeros((G, n_ens, n_sp), np.int32)
    lcd = np.zeros((G, n_ens, n_sp), np.float64)
    for g, thr in enumerate(levels):
        B = samples_keep >= float(thr)              # float32 comparison, as before
        size[g] = B.sum(axis=(2, 3))
        for k in range(n_ens):
            for s in range(n_sp):
                if size[g, k, s] >= 2:
                    b = B[k, s].astype(np.uint8)
                    comp[g, k, s] = count_components(b)
                    lcd[g, k, s] = np.log10(periodic_cov_det(b) + 1.0)
    return size, comp, lcd


def assemble(table, level_idx):
    """Pooled predicted statistics for one community, one level per species."""
    size, comp, lcd = table
    sp = np.arange(size.shape[2])
    sz = size[level_idx, :, sp]                     # (n_sp, n_ens)
    ok = sz >= 2
    return sz[ok], comp[level_idx, :, sp][ok], lcd[level_idx, :, sp][ok]


def ks_stat(a, b):
    """Two-sample KS distance; same formula as scipy.stats.ks_2samp."""
    a = np.sort(a); b = np.sort(b)
    pooled = np.concatenate([a, b])
    fa = np.searchsorted(a, pooled, side='right') / len(a)
    fb = np.searchsorted(b, pooled, side='right') / len(b)
    return float(np.max(np.abs(fa - fb)))


# ─────────────────────────────────────────────────────────────────────
#  ROUTERS: which cut-off each species gets
# ─────────────────────────────────────────────────────────────────────
def _cat(arrays, idx):
    return np.concatenate([arrays[i] for i in idx])


def router_legacy(comms, alpha, thr_hard, thr_moderate):
    """Earlier rule. The wide-range share uses every community's truth,
    including the one being evaluated."""
    C = len(comms)
    Xs = [c['X'] for c in comms]; ys = [c['y'] for c in comms]
    n_hat = [predict_range_size(_cat(Xs, [v for v in range(C) if v != w]),
                                _cat(ys, [v for v in range(C) if v != w]),
                                Xs[w], alpha) for w in range(C)]
    share = float((_cat(ys, range(C)) >= WIDE_CELLS).mean())
    cutoffs, rows = [], []
    for w in range(C):
        tau = float(np.quantile(_cat(n_hat, [v for v in range(C) if v != w]), 1.0 - share))
        wide = n_hat[w] >= tau
        cutoffs.append(np.where(wide, thr_hard, thr_moderate))
        rows.append(dict(wide_share=share, tau=tau, cut_wide=thr_hard,
                         cut_other=thr_moderate, n_wide=int(wide.sum()), criterion=''))
    return n_hat, cutoffs, rows


def router_nested(comms, alpha, levels, cand, tables, metric):
    """Leak-free rule. Nothing about community w's truth is used to set w's
    cut-offs: predictor, share, tau and the two cut-offs all come from the
    other communities, whose own predictions are made leave-one-out."""
    C = len(comms)
    Xs = [c['X'] for c in comms]; ys = [c['y'] for c in comms]
    truth = [(c['tr'], c['tn'], c['tl']) for c in comms]
    n_hat_outer, cutoffs, rows = [], [], []
    for w in range(C):
        T = [v for v in range(C) if v != w]
        n_hat_w = predict_range_size(_cat(Xs, T), _cat(ys, T), Xs[w], alpha)
        n_hat_outer.append(n_hat_w)
        inner = {v: predict_range_size(_cat(Xs, [u for u in T if u != v]),
                                       _cat(ys, [u for u in T if u != v]),
                                       Xs[v], alpha) for v in T}
        share = float((_cat(ys, T) >= WIDE_CELLS).mean())
        wide_T = {}
        for v in T:
            others = np.concatenate([inner[u] for u in T if u != v])
            wide_T[v] = inner[v] >= np.quantile(others, 1.0 - share)

        t_r = np.concatenate([truth[v][0] for v in T])
        t_c = np.concatenate([truth[v][1] for v in T])
        t_l = np.concatenate([truth[v][2] for v in T])
        best = (np.inf, None, None)
        for gh in cand:
            for gm in cand:
                parts = [assemble(tables[v], np.where(wide_T[v], gh, gm)) for v in T]
                pr = np.concatenate([p[0] for p in parts])
                if pr.size == 0:
                    continue
                pc = np.concatenate([p[1] for p in parts])
                pl = np.concatenate([p[2] for p in parts])
                d = (ks_stat(t_r, pr), ks_stat(t_c, pc), ks_stat(t_l, pl))
                score = sum(d) if metric == 'sum' else max(d)
                if score < best[0]:
                    best = (score, gh, gm)
        score, gh, gm = best

        tau_w = float(np.quantile(np.concatenate([inner[v] for v in T]), 1.0 - share))
        wide_w = n_hat_w >= tau_w
        cutoffs.append(np.where(wide_w, levels[gh], levels[gm]))
        rows.append(dict(wide_share=share, tau=tau_w, cut_wide=levels[gh],
                         cut_other=levels[gm], n_wide=int(wide_w.sum()),
                         criterion=round(score, 4)))
        print(f"    held out {comms[w]['name'][:45]:45s}  cut-offs "
              f"{levels[gh]:.3g} (wide) / {levels[gm]:.3g}  "
              f"wide {int(wide_w.sum()):3d}/{len(wide_w):3d}")
    return n_hat_outer, cutoffs, rows


def pool(comms, tables, level_of, cutoffs):
    out = {k: [] for k in ('truth_range', 'truth_ncomp', 'truth_logcd',
                           'pred_range', 'pred_ncomp', 'pred_logcd')}
    for c, table, cut in zip(comms, tables, cutoffs):
        idx = np.asarray([level_of[round(float(x), 6)] for x in cut], dtype=np.int64)
        pr, pc, pl = assemble(table, idx)
        out['truth_range'].append(c['tr']); out['truth_ncomp'].append(c['tn'])
        out['truth_logcd'].append(c['tl']); out['pred_range'].append(pr)
        out['pred_ncomp'].append(pc); out['pred_logcd'].append(pl)
    return {k: np.concatenate(v) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────────────────
def plain_log(axis, ticks):
    """Plain tick labels (10, 20, 0.01 ...) on a log axis, no minor labels."""
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())


def _subplots(nrows, ncols, figsize):
    major, minor = (int(p) for p in matplotlib.__version__.split('.')[:2])
    pads = dict(w_pad=0.04, h_pad=0.06, wspace=0.08, hspace=0.12)
    if (major, minor) >= (3, 6):
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout='constrained')
        fig.get_layout_engine().set(**pads)
    else:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        fig.set_constrained_layout_pads(**pads)
    return fig, ax


def _tag(ax, letter):
    ax.text(-0.20, 1.06, f'({letter})', transform=ax.transAxes,
            fontsize=8.5, fontweight='bold', va='top', ha='left')


def _tidy(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.4, linestyle=':')


def _hist_panel(ax, truth_arr, pred_arr, xlabel, bins, log_x=False,
                log_y=False, step_centers=None):
    ax.hist(truth_arr, bins=bins, color=COL_TRUTH, alpha=0.50,
            edgecolor='none', density=True, label='True')
    h, edges = np.histogram(pred_arr, bins=bins, density=True)
    centers = step_centers if step_centers is not None else 0.5 * (edges[:-1] + edges[1:])
    ax.step(centers, h, where='mid', color=COL_PRED, linewidth=1.3,
            label='Reconstructed')
    if log_x:
        ax.set_xscale('log')
        plain_log(ax.xaxis, RANGE_TICKS)
    if log_y:
        ax.set_yscale('log')
        plain_log(ax.yaxis, DENSITY_TICKS)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    _tidy(ax)


def _cdf_panel(ax, truth_arr, pred_arr, xlabel, ks_value, log_x=False, x_lim=None):
    xs_t = np.sort(truth_arr); ys_t = np.arange(1, len(xs_t) + 1) / len(xs_t)
    xs_p = np.sort(pred_arr);  ys_p = np.arange(1, len(xs_p) + 1) / len(xs_p)
    pred_at_truth = np.searchsorted(xs_p, xs_t, side='right') / len(xs_p)
    gaps = np.abs(ys_t - pred_at_truth)
    i = int(np.argmax(gaps)) if len(gaps) else 0
    ks_x, y_t, y_p = float(xs_t[i]), float(ys_t[i]), float(pred_at_truth[i])

    ax.fill_between(xs_t, 0, ys_t, step='post', color=COL_TRUTH, alpha=0.14)
    ax.step(xs_t, ys_t, where='post', color=COL_TRUTH, linewidth=1.3, label='True')
    ax.step(xs_p, ys_p, where='post', color=COL_PRED, linewidth=1.3,
            label='Reconstructed')

    lo, hi = sorted([y_t, y_p])
    ax.annotate('', xy=(ks_x, hi), xytext=(ks_x, lo),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.0,
                                shrinkA=0, shrinkB=0))
    dx = ks_x * 1.15 if (log_x and ks_x > 0) else ks_x + 0.03 * np.ptp(ax.get_xlim())
    ax.text(dx, 0.5 * (lo + hi), f'$D$ = {ks_value:.3f}', fontsize=7,
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor=COL_GREY, linewidth=0.4, alpha=0.95))
    if log_x:
        ax.set_xscale('log')
        plain_log(ax.xaxis, RANGE_TICKS)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel(xlabel); ax.set_ylabel('Cumulative fraction')
    _tidy(ax)


def make_figure(pooled, out_base, K, meta):
    tr, tn, tl = pooled['truth_range'], pooled['truth_ncomp'], pooled['truth_logcd']
    pr, pc, pl = pooled['pred_range'], pooled['pred_ncomp'], pooled['pred_logcd']
    ks_r = float(stats.ks_2samp(tr, pr).statistic)
    ks_c = float(stats.ks_2samp(tn, pc).statistic)
    ks_s = float(stats.ks_2samp(tl, pl).statistic)

    fig, ax = _subplots(2, 3, (W_DOUBLE, H_FIGURE))

    log_bins = np.logspace(np.log10(max(K, 1) + 1),
                           np.log10(max(tr.max(), pr.max()) + 1), 20)
    _hist_panel(ax[0, 0], tr, pr, 'Range size (occupied cells)', log_bins,
                log_x=True, log_y=True)
    ax[0, 0].legend(loc='upper right', frameon=False, handlelength=1.4)

    mx = int(max(tn.max(), pc.max()))
    cbins = np.arange(0.5, mx + 1.5, 1)
    _hist_panel(ax[0, 1], tn, pc, 'Connected patches per species', cbins,
                step_centers=np.arange(1, mx + 1))
    ax[0, 1].set_xlim(0.5, min(mx + 0.5, 30))

    sbins = np.linspace(min(tl.min(), pl.min()), max(tl.max(), pl.max()), 30)
    _hist_panel(ax[0, 2], tl, pl, SPREAD_LABEL, sbins)

    _cdf_panel(ax[1, 0], tr, pr, 'Range size (occupied cells)', ks_r, log_x=True)
    _cdf_panel(ax[1, 1], tn, pc, 'Connected patches per species', ks_c,
               x_lim=(0, min(mx + 1, 30)))
    _cdf_panel(ax[1, 2], tl, pl, SPREAD_LABEL, ks_s)
    for a in (ax[0, 1], ax[1, 1]):                  # patch counts are integers
        a.xaxis.set_major_locator(MaxNLocator(integer=True))

    for a, letter in zip(ax.ravel(), 'abcdef'):
        _tag(a, letter)

    for a, (t_mu, p_mu, fmt) in zip(
            ax[0], [(tr.mean(), pr.mean(), '{:.1f}'),
                    (tn.mean(), pc.mean(), '{:.1f}'),
                    (tl.mean(), pl.mean(), '{:.2f}')]):
        a.text(0.03, 0.03, f'true {fmt.format(t_mu)}\nrecon {fmt.format(p_mu)}',
               transform=a.transAxes, fontsize=7, va='bottom', ha='left',
               color=COL_GREY, linespacing=1.25,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='none', alpha=0.8))

    out = Path(out_base)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), facecolor='white')
    fig.savefig(out.with_suffix('.png'), dpi=600, facecolor='white')
    plt.close(fig)

    print(f"\n  n true ranges        = {len(tr):,}")
    print(f"  n reconstructed      = {len(pr):,}  "
          f"({len(pr) / max(1, len(tr)):.1f} per species)")
    print(f"  D, range size        = {ks_r:.3f}   "
          f"means {tr.mean():.1f} true vs {pr.mean():.1f} reconstructed")
    print(f"  D, fragmentation     = {ks_c:.3f}   "
          f"means {tn.mean():.1f} vs {pc.mean():.1f}")
    print(f"  D, spatial spread    = {ks_s:.3f}   "
          f"means {tl.mean():.2f} vs {pl.mean():.2f}")
    print(f"\n  saved -> {out.with_suffix('.pdf')}  (vector, 190 x 125 mm)")
    print(f"  saved -> {out.with_suffix('.png')}  (600 dpi)")

    with open(out.with_name(out.name + '_values.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['statistic', 'n_true', 'n_reconstructed', 'mean_true',
                    'mean_reconstructed', 'KS_D', 'n_communities', 'router', 'dedupe'])
        for name, t, p, d in (('range_size', tr, pr, ks_r),
                              ('fragmentation', tn, pc, ks_c),
                              ('spatial_spread', tl, pl, ks_s)):
            w.writerow([name, len(t), len(p), t.mean(), p.mean(), d,
                        meta['n_communities'], meta['router'], meta['dedupe']])
    print(f"  saved -> {out.with_name(out.name + '_values.csv')}")
    return ks_r, ks_c, ks_s


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Figure 4, distributional realism')
    ap.add_argument('--wide-range-csv', required=True,
                    help="manifest with a 'world' (or 'filename') column and, "
                         "optionally, 'regime'")
    ap.add_argument('--recon-dir-pattern', required=True)
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--threshold-moderate', type=float, default=0.95,
                    help='legacy router only')
    ap.add_argument('--threshold-hard', type=float, default=0.80,
                    help='legacy router only')
    ap.add_argument('--top-n-worlds', type=int, default=30)
    ap.add_argument('--output', required=True,
                    help='base path; writes .pdf, .png and CSVs')
    ap.add_argument('--regime', default='all', choices=['all', 'in_dist', 'extrap'],
                    help='applied after copies are removed')
    ap.add_argument('--router', default='nested', choices=['nested', 'legacy'])
    ap.add_argument('--grid', default=DEFAULT_GRID,
                    help='candidate cut-offs for the nested router')
    ap.add_argument('--select-metric', default='sum', choices=['sum', 'max'],
                    help='combine the three KS distances by sum or max')
    ap.add_argument('--ridge-alpha', type=float, default=1.0)
    ap.add_argument('--no-dedupe', dest='dedupe', action='store_false',
                    help='keep files whose final maps are identical (old behaviour)')
    args = ap.parse_args()

    world_paths, dropped, regime = select_communities(args)
    print(f"\n  {len(world_paths)} communities selected (regime = {args.regime}, "
          f"copies {'removed' if args.dedupe else 'kept'})")
    for d in dropped:
        print(f"    dropped {d[0][:60]}  (identical to {d[1][:60]})")
    if not world_paths:
        print("\n  no usable communities.")
        return

    comms = load_communities(world_paths, args.K)
    if len(comms) < 3:
        print("\n  need >= 3 communities for leave-one-out ridge. aborting.")
        return
    reg_count = Counter(regime.get(c['name'], '') for c in comms)
    print("  regimes: " + ", ".join(f"{k or 'unlabelled'} {v}" for k, v in sorted(reg_count.items())))

    grid = sorted({round(float(x), 6) for x in args.grid.split(',') if x.strip()})
    levels = sorted(set(grid) | {round(args.threshold_hard, 6),
                                 round(args.threshold_moderate, 6)})
    level_of = {v: i for i, v in enumerate(levels)}
    cand = [level_of[v] for v in grid]
    print(f"\n  binarising samples at {len(levels)} cut-off levels ...")
    tables = [binarised_stats(c['samples'], levels) for c in comms]

    n_hat_l, cut_l, rows_l = router_legacy(comms, args.ridge_alpha,
                                           args.threshold_hard, args.threshold_moderate)
    if args.router == 'nested':
        print("\n  nested router: cut-offs chosen on the other communities")
        n_hat, cutoffs, rows = router_nested(comms, args.ridge_alpha, levels, cand,
                                             tables, args.select_metric)
    else:
        n_hat, cutoffs, rows = n_hat_l, cut_l, rows_l

    rho, _ = stats.spearmanr(np.concatenate(n_hat), np.concatenate([c['y'] for c in comms]))
    print(f"\n  range-size predictor: Spearman rho = {rho:+.3f}  "
          f"(pooled n_species = {sum(len(c['y']) for c in comms)})")
    if args.router == 'nested':
        pairs = Counter((r['cut_wide'], r['cut_other']) for r in rows)
        print("  chosen cut-offs (wide / other): " +
              ", ".join(f"{a:.3g}/{b:.3g} x{n}" for (a, b), n in pairs.most_common()))
        edges = {levels[cand[0]], levels[cand[-1]]}
        n_edge = sum((r['cut_wide'] in edges) or (r['cut_other'] in edges) for r in rows)
        if n_edge:
            print(f"  note: {n_edge} of {len(rows)} communities chose a cut-off at the "
                  f"edge of --grid ({min(edges):g} or {max(edges):g}); widen --grid and rerun")
        legacy = pool(comms, tables, level_of, cut_l)
        print("  legacy router on the same communities: D = "
              f"{stats.ks_2samp(legacy['truth_range'], legacy['pred_range']).statistic:.3f}, "
              f"{stats.ks_2samp(legacy['truth_ncomp'], legacy['pred_ncomp']).statistic:.3f}, "
              f"{stats.ks_2samp(legacy['truth_logcd'], legacy['pred_logcd']).statistic:.3f}")

    pooled = pool(comms, tables, level_of, cutoffs)
    meta = dict(n_communities=len(comms), router=args.router, dedupe=args.dedupe)
    make_figure(pooled, args.output, args.K, meta)

    out = Path(args.output)
    with open(out.with_name(out.name + '_router.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['world', 'regime', 'n_species', 'wide_share', 'tau', 'cut_wide',
                    'cut_other', 'n_called_wide', 'criterion_on_other_communities'])
        for c, r in zip(comms, rows):
            w.writerow([c['name'], regime.get(c['name'], ''), len(c['y']),
                        f"{r['wide_share']:.4f}", f"{r['tau']:.3f}", r['cut_wide'],
                        r['cut_other'], r['n_wide'], r['criterion']])
    print(f"  saved -> {out.with_name(out.name + '_router.csv')}")
    if dropped:
        with open(out.with_name(out.name + '_dropped.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dropped_world', 'identical_to', 'regime_dropped', 'regime_kept'])
            w.writerows(dropped)
        print(f"  saved -> {out.with_name(out.name + '_dropped.csv')}")


if __name__ == '__main__':
    main()