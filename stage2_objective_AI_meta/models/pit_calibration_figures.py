#!/usr/bin/env python3
"""
=============================================================================
CALIBRATION FIGURES  -  one script, two figures, Elsevier artwork geometry
  (Ecological Informatics / Ecological Modelling)
=============================================================================
  rank     Fig_calibration_<regime>   pooled PIT / rank histogram, one regime
  records  Fig_calibration_vs_records coverage against records per species
  all      both, from one command

Input is the CSV written by posterior_per_species.py, which records per species

    {stat}_pctile = ( #samples < truth + 0.5 * #samples == truth ) / n_ens

together with world_stem, species, range_N, n_obs, K and n_ens. If the CSV also
carries {stat}_nbelow and {stat}_nequal the RANDOMISED PIT (b + U*e)/n is used
instead of the mid-rank; it is exactly uniform under a correct ensemble even
when the statistic ties heavily, which the mid-rank is not.

WHAT THIS SCRIPT GETS RIGHT THAT THE EARLIER ONES DID NOT
---------------------------------------------------------
1. The PIT is DISCRETE: with n_ens members it lands only on k/(2*n_ens). A flat
   reference at N/bins is unreachable and some bins are structurally empty. The
   null is computed exactly from the discrete support, and --bins auto picks the
   largest divisor of n_ens+1 under 20 so the null is genuinely flat (17 bins at
   n_ens=50, 9 at n_ens=8).
2. "truth within 5-95%" is NOT 90%. The rank is uniform on {0..n_ens}, so the
   attainable coverage is 45/51 = 88.2% at n_ens=50 and 7/9 = 77.8% at n_ens=8.
   Both figures draw that line.
3. scipy.stats.kstest(vals,'uniform') assumes a continuous uniform and is
   invalid here. The headline test is a chi-square against the exact discrete
   null, with a Monte-Carlo p-value when an expected count falls below 5.
   CAUTION: on a heavily tied integer statistic such as connected patches, the
   MID-RANK chi-square rejects a perfectly exchangeable ensemble every time
   (chi2 of 140-800 on synthetic exchangeable data at 50 members). The p-value
   is therefore SUPPRESSED from the figure and the caption whenever ties exceed
   15% and the PIT is a mid-rank. Coverage and mean percentile are unaffected.
4. Species inside one world share a latent field, and the same species appears
   under several sampling schemes. Intervals are world-clustered bootstraps; a
   degenerate zero-width bootstrap falls back to Clopper-Pearson on the world
   count; duplicated (regime, world, species) keys are counted and reported.
5. Empty bins are dropped from the axis instead of leaving a dangling tick, and
   bins holding fewer than --min-n species are drawn as open markers off the
   trend line.
6. Binning on record count confounds scheme with range size: 5 records under
   K=5 is a narrow species, 5 records under p=0.10 is a range near 50. The
   records figure prints median true range per bin per scheme and draws each
   scheme separately. --common-species restricts to species present in every
   scheme for a like-for-like comparison.

Ecological Informatics / Elsevier compliance (checked 2 Oct 2026):
  - widths 90 / 140 / 190 mm (Elsevier artwork sizing);
  - lettering 7 pt for normal text, >= 6 pt only for sub/superscripts. Every
    text element here (ticks, legends, panel notes, n labels) is drawn at
    --font-pt, and --font-pt below 7 aborts. chi-squared is written with the
    Unicode characters, so there is no shrunken mathtext superscript;
  - vector PDF (accepted format); PNG at >= 1000 dpi, the guide's minimum for
    bitmapped line drawings (3543 px at 90 mm, 7480 px at 190 mm);
  - TrueType embedding (pdf.fonttype 42);
  - no title inside the figure: the guide puts the brief title in the caption,
    which is printed to stdout;
  - each figure as its own file with a logical name: --submit-as Figure_6
    writes an extra Figure_6.pdf beside the working file;
  - p-values below 0.001 are printed as "p < 0.001", never "p = 0.000".

Range size carries no percentile: the top-N rule pins every sample at the true
range size, so its rank is degenerate.
=============================================================================
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from scipy import stats

# ---------------------------------------------------------------------------
# Elsevier geometry, Paul Tol colourblind-safe palette
# ---------------------------------------------------------------------------
MM = 1.0 / 25.4
WIDTH_MM = {'single': 90.0, '1.5': 140.0, 'double': 190.0}
C_BAR, C_NULL, C_BAND = '#4477AA', '#EE6677', '#BBBBBB'
C_TXT, C_GREY = '#222222', '#777777'
TOL = ['#4477AA', '#228833', '#CCBB44', '#AA3377', '#66CCEE', '#EE8866']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
NICE = {'patches': 'connected patches', 'spread': 'spatial spread',
        'range': 'range size'}
OBS_BINS = [(0, 0, '0'), (1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'),
            (5, 9, '5-9'), (10, 10 ** 9, '10+')]
PREFERRED_FONTS = ['Arial', 'Helvetica', 'Nimbus Sans', 'Liberation Sans',
                   'DejaVu Sans']
LO, HI = 0.05, 0.95
MIN_TEXT_PT = 7.0          # Elsevier artwork sizing: normal lettering
MIN_LINEART_DPI = 1000     # Ecological Informatics guide: bitmapped line drawings

RANK_VALUE_HEADER = [
    'stat', 'bin_left', 'bin_right', 'observed_species', 'expected_null',
    'n_species', 'n_ens', 'coverage', 'coverage_ci_lo', 'coverage_ci_hi',
    'ci_method', 'coverage_null', 'mean_pctile', 'chi2', 'df', 'p_value',
    'test', 'tie_share_lower_bound', 'randomised_pit', 'n_worlds', 'label']

RECORDS_VALUE_HEADER = [
    'stat', 'regime', 'obs_bin', 'obs_lo', 'obs_hi', 'n_species',
    'n_unique_species', 'n_worlds', 'coverage', 'ci_lo', 'ci_hi', 'ci_method',
    'below_min_n', 'coverage_null', 'median_range_N', 'n_ens']


def set_style(base_pt):
    have = {f.name for f in font_manager.fontManager.ttflist}
    chain = [f for f in PREFERRED_FONTS if f in have] or ['DejaVu Sans']
    if chain[0] in ('Arial', 'Helvetica', 'Nimbus Sans'):
        print(f"  font check: '{chain[0]}'")
    else:
        print(f"  ! font check: Arial/Helvetica absent, using '{chain[0]}' "
              f"(apt install fonts-liberation gives Liberation Sans, metric-"
              f"compatible with Arial).")
    # every text element at base_pt: Elsevier asks 7 pt for normal lettering
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': chain,
        'font.size': base_pt, 'axes.labelsize': base_pt,
        'axes.titlesize': base_pt, 'xtick.labelsize': base_pt,
        'ytick.labelsize': base_pt, 'legend.fontsize': base_pt,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 2.4,
        'ytick.major.size': 2.4, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.unicode_minus': False,
    })


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_spec(spec):
    """'label=path' or 'path'; rsplit because labels such as 'K=5' and
    'p=0.10' contain '=' themselves."""
    if '=' in spec and not Path(spec).exists():
        lab, path = spec.rsplit('=', 1)
        return lab.strip(), path.strip()
    return Path(spec).stem, spec


def load_specs(specs):
    rows, missing = [], []
    for spec in specs:
        lab, pat = parse_spec(spec)
        hits = sorted(glob.glob(pat)) or [pat]
        for p in hits:
            if not Path(p).exists():
                missing.append(p)
                continue
            with open(p, newline='') as f:
                rd = csv.DictReader(f)
                if rd.fieldnames is None:
                    print(f"  ! empty CSV, skipped: {p}")
                    continue
                n0 = len(rows)
                for r in rd:
                    r['__regime__'], r['__src__'] = lab, p
                    rows.append(r)
                print(f"  loaded {len(rows) - n0:>5d} row(s)  [{lab}]  {p}")
    for m in missing:
        print(f"  ! missing CSV, skipped: {m}")
    return rows


def _f(r, k):
    s = (r.get(k) or '').strip()
    if s == '':
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _i(r, k):
    v = _f(r, k)
    return None if v is None else int(round(v))


def extract(rows, stat, want_regime=None, seed=0):
    """Pull one statistic. Every returned array has the SAME length; a row that
    fails any check is skipped from all of them together."""
    col, cb, ce = f'{stat}_pctile', f'{stat}_nbelow', f'{stat}_nequal'
    use = [r for r in rows if want_regime is None or r['__regime__'] == want_regime]
    randomised = bool(use) and all(
        any(_f(r, c) is not None for r in use) for c in (cb, ce))
    rng = np.random.default_rng(seed)
    v, n, w, nob, rgn, reg, spid, ids = [], [], [], [], [], [], [], []
    bad = 0
    for r in use:
        val = _f(r, col)
        if val is None or not (-1e-9 <= val <= 1 + 1e-9):
            bad += 1
            continue
        ne = _i(r, 'n_ens') or -1
        if randomised and ne > 0:
            b, e = _f(r, cb), _f(r, ce)
            if b is not None and e is not None:
                val = (b + rng.random() * e) / ne
        sp = str(r.get('species') or '')
        wd = str(r.get('world_stem') or r.get('__src__') or 'pooled')
        v.append(min(max(val, 0.0), 1.0))
        n.append(ne)
        w.append(wd)
        nob.append(_i(r, 'n_obs'))
        rgn.append(_i(r, 'range_N'))
        reg.append(r['__regime__'])
        spid.append(sp)                        # <- was missing; every downstream
        ids.append((r['__regime__'], wd, sp))  #    mask then broke on an empty array
    dup = len(ids) - len(set(ids))
    if dup:
        print(f"  ! '{stat}': {dup} duplicated (regime, world, species) key(s). "
              f"posterior_per_species.py APPENDS; delete the CSV before re-running.")
    if bad:
        print(f"  ! '{stat}': {bad} row(s) with a missing or out-of-range percentile")
    out = dict(v=np.asarray(v, float), n_ens=np.asarray(n, int),
               world=np.asarray(w, dtype=object),
               n_obs=np.asarray([-1 if x is None else x for x in nob], int),
               range_N=np.asarray([-1 if x is None else x for x in rgn], int),
               species=np.asarray(spid, dtype=object),
               regime=np.asarray(reg, dtype=object), randomised=randomised)
    sizes = {k: out[k].size for k in
             ('v', 'n_ens', 'world', 'n_obs', 'range_N', 'species', 'regime')}
    if len(set(sizes.values())) != 1:          # cheap guard against the bug above
        sys.exit(f"  ABORT: internal array length mismatch for '{stat}': {sizes}")
    return out


# ---------------------------------------------------------------------------
# Exact discrete null:  rank r ~ Uniform{0..n_ens},  PIT = r / n_ens
# ---------------------------------------------------------------------------
def auto_bins(n_ens, cap=20):
    """Largest B <= cap that divides n_ens+1 exactly, so the null is flat."""
    m = n_ens + 1
    div = [b for b in range(2, cap + 1) if m % b == 0]
    return max(div) if div else min(m, cap)


def null_probs(n_ens, edges):
    support = np.arange(n_ens + 1, dtype=float) / n_ens
    counts, _ = np.histogram(support, bins=edges)
    return counts / float(n_ens + 1)


def null_coverage(n_ens, lo=LO, hi=HI):
    s = np.arange(n_ens + 1, dtype=float) / n_ens
    return float(np.mean((s >= lo) & (s <= hi)))


def one_n_ens(d, tag, force_pool):
    good = d['n_ens'] > 0
    uniq = sorted(set(d['n_ens'][good].tolist()))
    if not uniq:
        return 0
    if len(uniq) > 1:
        msg = (f"  ! {tag}: mixed ensemble sizes {uniq}. Ranks from different "
               f"n_ens are not poolable (the null itself differs)")
        if not force_pool:
            sys.exit(msg + "; split the runs or pass --force-pool.")
        print(msg + "; --force-pool set, using the smallest.")
    return uniq[0]


def tie_floor(vals, n_ens, dp=3):
    """Lower bound on the tie share: an EVEN number of ties lands back on the
    integer grid and is invisible in a mid-rank."""
    if n_ens <= 0 or vals.size == 0:
        return float('nan'), float('nan')
    h = vals * 2.0 * n_ens
    tol = (10.0 ** -dp) * 2.0 * n_ens + 1e-6
    near = np.abs(h - np.round(h)) <= tol
    off = float(np.mean(~near))
    hr = np.round(h).astype(int)
    tied = float(np.mean((hr % 2 == 1) & near)) if near.any() else float('nan')
    return tied, off


def gof(vals, edges, p_null, seed=0, n_mc=20000):
    obs, _ = np.histogram(vals, bins=edges)
    N = int(obs.sum())
    exp = N * p_null
    live = p_null > 0
    if N == 0 or live.sum() < 2:
        return float('nan'), 0, float('nan'), 'none', obs, exp, 0
    chi2 = float(np.sum((obs[live] - exp[live]) ** 2 / exp[live]))
    dof = int(live.sum()) - 1
    leak = int(obs[~live].sum())
    if exp[live].min() >= 5 and leak == 0:
        p, how = float(stats.chi2.sf(chi2, dof)), 'chi2'
    else:
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(N, p_null, size=n_mc).astype(float)
        ref = np.sum((draws[:, live] - exp[live]) ** 2 / exp[live], axis=1)
        p, how = float((np.sum(ref >= chi2) + 1) / (n_mc + 1)), 'chi2-MC'
    return chi2, dof, p, how, obs, exp, leak


def cov(v):
    return float(np.mean((v >= LO) & (v <= HI))) if v.size else float('nan')


def fmt_p(p):
    """'p < 0.001' below the third decimal, 'p = 0.012' otherwise. Never
    'p = 0.000', which reads as an impossible exact zero."""
    if p is None or np.isnan(p):
        return 'p = n/a'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval, applied to the CLUSTER count, so it treats one
    world as one effective observation. Deliberately conservative."""
    if n <= 0:
        return float('nan'), float('nan')
    lo = 0.0 if k <= 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def clustered_ci(vals, clusters, n_boot=2000, seed=1, stat=cov):
    """World-clustered percentile bootstrap. Returns (lo, hi, n_clusters,
    method). A percentile bootstrap collapses to zero width whenever every
    cluster gives the same value (8 species, all covered, 5 worlds -> every
    resample returns 1.00); reporting [100%, 100%] from 8 species is not
    defensible, so that case falls back to Clopper-Pearson on the worlds."""
    keys = np.unique(clusters)
    nk = int(keys.size)
    if vals.size == 0 or nk < 2:
        return float('nan'), float('nan'), nk, 'none'
    idx = {k: np.flatnonzero(clusters == k) for k in keys}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([idx[k] for k in rng.choice(keys, nk, replace=True)])
        out[b] = stat(vals[sel])
    lo = float(np.nanpercentile(out, 2.5))
    hi = float(np.nanpercentile(out, 97.5))
    if hi - lo < 1e-9:
        c = stat(vals)
        lo, hi = clopper_pearson(int(round(c * nk)), nk)
        return lo, hi, nk, 'clopper-pearson-on-worlds'
    return lo, hi, nk, 'cluster-bootstrap'


def slug(text):
    """Filename-safe label: 'p=0.10' -> 'p_0_10'."""
    out = ''.join(ch if (ch.isalnum() or ch in '-_') else '_' for ch in text)
    while '__' in out:
        out = out.replace('__', '_')
    return out.strip('_') or 'figure'


def audit_text(fig, floor=MIN_TEXT_PT):
    """Every visible text in the figure must be at least `floor` pt. Returns the
    smallest size found; aborts before saving if any text is below the floor."""
    small, sizes = [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        fs = float(t.get_fontsize())
        sizes.append(fs)
        if fs < floor - 1e-6:
            small.append((fs, t.get_text()[:30].replace('\n', ' ')))
    if small:
        sys.exit(f"  ABORT: {len(small)} text element(s) below {floor:g} pt, e.g. "
                 f"{small[:3]}. Elsevier asks 7 pt for normal lettering.")
    return min(sizes) if sizes else float('nan')


def save(fig, output, dpi, width_key, submit_as=None):
    """String slicing only: Path.with_suffix would eat the '.10' in a stem
    such as 'Fig_calibration_p=0.10'. `submit_as` (e.g. 'Figure_6') also writes
    that name as a vector PDF in the same folder, for the submission system."""
    txt = str(output)
    for ext in ('.png', '.pdf', '.eps', '.tif', '.tiff'):
        if txt.lower().endswith(ext):
            txt = txt[:-len(ext)]
            break
    stem = Path(txt)
    stem.parent.mkdir(parents=True, exist_ok=True)
    smallest = audit_text(fig)
    pdf_p, png_p = Path(txt + '.pdf'), Path(txt + '.png')
    fig.savefig(pdf_p)
    fig.savefig(png_p, dpi=dpi)
    if submit_as:
        sub_p = stem.parent / (Path(submit_as).name.split('.pdf')[0] + '.pdf')
        fig.savefig(sub_p)
        print(f"  saved -> {sub_p}  (submission copy, vector)")
    plt.close(fig)
    w = WIDTH_MM[width_key]
    px, px_min = int(round(w * MM * dpi)), int(round(w * MM * MIN_LINEART_DPI))
    print(f"  saved -> {pdf_p}  (vector, {w:.0f} mm, pdf.fonttype=42; "
          f"smallest text {smallest:g} pt)")
    print(f"  saved -> {png_p}  ({dpi} dpi, {px} px wide; guide minimum for line "
          f"art {px_min} px at {w:.0f} mm)")
    if dpi < MIN_LINEART_DPI:
        print(f"  ! PNG below {MIN_LINEART_DPI} dpi: not acceptable as bitmapped line "
              f"art. Submit the PDF, or re-run with --dpi {MIN_LINEART_DPI}.")
    return stem


def write_values(path, header, rows):
    """Every row must have exactly as many fields as the header. A mismatch
    silently shifts every column, so it aborts instead."""
    for i, r in enumerate(rows):
        if len(r) != len(header):
            sys.exit(f"  ABORT: values row {i} has {len(r)} fields, header has "
                     f"{len(header)}. Refusing to write a misaligned file.")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  saved -> {path}  (every plotted number)")


# ===========================================================================
# FIGURE 1  -  pooled rank histogram, one regime
# ===========================================================================
def panel_rank(ax, res, letter, annotate, base_pt):
    if res is None:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=base_pt)
        ax.set_xticks([]); ax.set_yticks([])
        return
    edges, obs, exp, N = res['edges'], res['obs'], res['exp'], res['N']
    ctr, wid = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    band = stats.binom.ppf(np.array([[0.025], [0.975]]), N, res['p_null'])
    ax.bar(ctr, band[1] - band[0], bottom=band[0], width=wid, color=C_BAND,
           alpha=0.55, edgecolor='none', zorder=1)
    ax.bar(ctr, obs, width=wid * 0.92, color=C_BAR, edgecolor='white',
           linewidth=0.4, zorder=3)
    ax.step(np.append(edges, edges[-1]), np.append(np.append(exp[0], exp), 0),
            where='pre', color=C_NULL, ls='--', lw=1.0, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(obs.max(), band[1].max()) * 1.42)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel('percentile of truth within ensemble')
    ax.set_ylabel('species')
    ax.set_title(f"({letter}) {res['name']}", loc='left', fontweight='bold', pad=3.0)
    ax.spines[['top', 'right']].set_visible(False)
    if annotate == 'compact':
        # a mid-rank p-value on a heavily tied statistic is not reportable
        show_p = res['randomised'] or not (res['tied'] > 0.15)
        # Unicode chi and superscript two, at full size: a mathtext superscript
        # would print near 0.7 x base_pt, under the 6 pt floor for superscripts
        line3 = (f"χ² {fmt_p(res['p'])}" if show_p and not np.isnan(res['p'])
                 else f"mean percentile {res['mean']:.2f}")
        ax.text(0.985, 0.965,
                f"n = {N}\ncoverage {res['cov']:.0%} (null {res['cov0']:.0%})\n{line3}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=base_pt, color=C_TXT, linespacing=1.35)


def figure_rank(rows, args):
    headers = set().union(*[set(r) for r in rows]) if rows else set()
    wanted = [s for s in args.stats if f'{s}_pctile' in headers]
    for s in args.stats:
        if s not in wanted:
            print(f"  ! column '{s}_pctile' absent, panel dropped"
                  + ("  (range size is pinned by the top-N rule)" if s == 'range' else ""))
    if not wanted:
        sys.exit("  ABORT: none of the requested *_pctile columns exist.")

    results, ne_seen = [], set()
    for stat in wanted:
        d = extract(rows, stat, seed=args.seed)
        if d['v'].size == 0:
            results.append(None)
            continue
        n_ens = one_n_ens(d, f"'{stat}'", args.force_pool)
        if n_ens <= 1:
            print(f"  ! '{stat}': n_ens={n_ens}, no rank information; panel dropped")
            results.append(None)
            continue
        ne_seen.add(n_ens)
        B = auto_bins(n_ens) if args.bins == 'auto' else int(args.bins)
        edges = np.linspace(0, 1, B + 1)
        p_null = null_probs(n_ens, edges)
        chi2, dof, p, how, obs, exp, leak = gof(d['v'], edges, p_null, seed=args.seed)
        c, c0 = cov(d['v']), null_coverage(n_ens)
        lo, hi, nw, cim = clustered_ci(d['v'], d['world'], args.n_boot, args.seed + 1)
        tied, off = tie_floor(d['v'], n_ens)
        results.append(dict(name=NICE.get(stat, stat), stat=stat, edges=edges,
                            obs=obs, exp=exp, p_null=p_null, N=int(d['v'].size),
                            cov=c, cov0=c0, ci=(lo, hi), ci_method=cim, nclus=nw,
                            chi2=chi2, dof=dof, p=p, how=how, n_ens=n_ens,
                            mean=float(d['v'].mean()), tied=tied,
                            randomised=d['randomised'],
                            empty=int(np.sum(p_null == 0)), leak=leak))
        print(f"  {stat:>8}: n={d['v'].size}  n_ens={n_ens}  bins={B}  "
              f"coverage={c:.1%} [{lo:.1%}, {hi:.1%}] vs null {c0:.1%}  "
              f"mean pctile={d['v'].mean():.3f}  {how} p={p:.5f} ({fmt_p(p)}) "
              f"(chi2={chi2:.2f}, df={dof})  "
              f"{'PIT=randomised' if d['randomised'] else f'ties>={tied:.0%}'}  "
              f"worlds={nw}  empty-bins={int(np.sum(p_null == 0))}")
        if d['randomised'] or not (tied > 0.15):
            print(f"      p-value check for the text: {fmt_p(p)} -> "
                  f"{'can be reported as p < 0.001' if p < 0.001 else 'NOT below 0.001'}")
        if leak:
            print(f"      ! {leak} value(s) sit in bins the null cannot reach; use --bins auto")
        if off > 0.01:
            print(f"      ! {off:.1%} of values are off the k/(2*n_ens) grid; "
                  f"n_ens in the CSV may not match the values")
        if (not d['randomised']) and tied > 0.15:
            print(f"      ! at least {tied:.0%} of species tie the truth exactly and the "
                  f"PIT is a MID-RANK. On synthetic exchangeable data a mid-rank "
                  f"chi-square rejects at this tie rate every time, so DO NOT quote this "
                  f"p-value; it is suppressed in the figure and the caption. Coverage and "
                  f"mean percentile are unaffected. Emit {stat}_nbelow / {stat}_nequal "
                  f"from posterior_per_species.py and re-run.")

    if not any(results):
        sys.exit("  ABORT: no panel has usable rank data.")

    w_in = WIDTH_MM[args.width] * MM
    fig, axes = plt.subplots(1, len(results), figsize=(w_in, args.height_mm * MM),
                             squeeze=False, layout='constrained')
    for j, res in enumerate(results):
        panel_rank(axes[0, j], res, chr(ord('a') + j), args.annotate, args.font_pt)
    if args.draft_title:
        fig.suptitle('Ensemble calibration' + (f'  [{args.label}]' if args.label else ''),
                     fontsize=args.font_pt + 1, fontweight='bold')
    stem = save(fig, args.output, args.dpi, args.width, getattr(args, 'submit_as', None))

    out_rows = []
    for r in results:
        if r is None:
            continue
        for b in range(len(r['obs'])):
            out_rows.append([
                r['stat'], f"{r['edges'][b]:.4f}", f"{r['edges'][b + 1]:.4f}",
                int(r['obs'][b]), f"{r['exp'][b]:.3f}", r['N'], r['n_ens'],
                f"{r['cov']:.4f}", f"{r['ci'][0]:.4f}", f"{r['ci'][1]:.4f}",
                r['ci_method'], f"{r['cov0']:.4f}", f"{r['mean']:.4f}",
                f"{r['chi2']:.4f}", r['dof'], f"{r['p']:.5f}", r['how'],
                f"{r['tied']:.4f}", int(r['randomised']), r['nclus'], args.label])
    write_values(stem.parent / (stem.name + '_values.csv'), RANK_VALUE_HEADER, out_rows)

    live = [r for r in results if r]
    parts = []
    for i, r in enumerate(live):
        txt = (f"({chr(ord('a') + i)}) {r['name']}: {r['N']} species, coverage "
               f"{r['cov']:.0%} (95% CI {r['ci'][0]:.0%}-{r['ci'][1]:.0%}; "
               f"exchangeable null {r['cov0']:.0%}), mean percentile {r['mean']:.2f}")
        if r['randomised'] or not (r['tied'] > 0.15):
            txt += f", {r['how']} {fmt_p(r['p'])}"
        else:
            txt += (f" (no goodness-of-fit p-value: the statistic ties for at least "
                    f"{r['tied']:.0%} of species and the stored PIT is a mid-rank)")
        parts.append(txt)
    print("\n  CAPTION\n"
          f"  Fig. X. Rank histogram of the true range statistic within the "
          f"{'/'.join(str(x) for x in sorted(ne_seen))}-member reconstruction ensemble"
          + (f", {args.label}" if args.label else "") + ". Bars are species counts; the "
          f"dashed step is the expectation under the exchangeable null, in which the rank "
          f"of the truth is uniform on {{0,...,n}}; the grey band is the pointwise 95% "
          f"interval of that null. Flat bars mean the ensemble brackets the truth "
          f"honestly, a left lean means the truth sits below the ensemble, a U shape "
          f"means the ensemble is too narrow. " + "; ".join(parts) +
          f". Intervals are world-clustered bootstraps "
          f"({max(r['nclus'] for r in live)} worlds, {args.n_boot} resamples).\n")

    if args.also_ks:
        for stat in wanted:
            d = extract(rows, stat, seed=args.seed)
            if d['v'].size:
                k = stats.kstest(d['v'], 'uniform')
                print(f"  [indicative only] {stat}: continuous KS D={k.statistic:.4f} "
                      f"p={k.pvalue:.4f}; invalid for a discrete rank.")


# ===========================================================================
# FIGURE 2  -  coverage against records per species
# ===========================================================================
def bin_curve(d, mask_extra=None, n_boot=2000, seed=1):
    out = []
    for lo_, hi_, lab in OBS_BINS:
        m = (d['n_obs'] >= lo_) & (d['n_obs'] <= hi_)
        if mask_extra is not None:
            m &= mask_extra
        v = d['v'][m]
        cl, ch, nw, how = clustered_ci(v, d['world'][m], n_boot, seed)
        rn = d['range_N'][m]
        # a pooled bin counts the same species once per scheme; the unique count
        # is the honest measure of how much independent data it holds
        uniq = len(set(zip(d['world'][m].tolist(), d['species'][m].tolist())))
        out.append(dict(label=lab, lo=lo_, hi=hi_, n=int(m.sum()), n_uniq=uniq,
                        cov=cov(v), ci=(cl, ch), nworld=nw, ci_method=how,
                        med_range=(float(np.median(rn[rn > 0])) if (rn > 0).any()
                                   else float('nan'))))
    return out


def figure_records(rows, args):
    stats_present = [s for s in args.stats
                     if any((r.get(f'{s}_pctile') or '').strip() for r in rows)]
    if not stats_present:
        sys.exit("  ABORT: no *_pctile column found for the requested statistics.")
    if not any((r.get('n_obs') or '').strip() for r in rows):
        sys.exit("  ABORT: no n_obs column; re-run posterior_per_species.py.")

    regimes = sorted({r['__regime__'] for r in rows})
    data = {s: extract(rows, s, seed=args.seed) for s in stats_present}

    n_ens_all = set()
    for s, d in data.items():
        n = one_n_ens(d, f"'{s}'", args.force_pool)
        if n > 1:
            n_ens_all.add(n)
    if not n_ens_all:
        sys.exit("  ABORT: no usable n_ens.")
    if len(n_ens_all) > 1 and not args.force_pool:
        sys.exit(f"  ABORT: ensemble sizes {sorted(n_ens_all)} differ across statistics.")
    n_ens = min(n_ens_all)
    c0 = null_coverage(n_ens)

    # ---- optional like-for-like restriction to species shared by every regime
    if args.common_species and len(regimes) > 1:
        for s, d in data.items():
            keys = list(zip(d['world'].tolist(), d['species'].tolist()))
            sets = [{k for k, g in zip(keys, d['regime'].tolist()) if g == gg}
                    for gg in regimes]
            common = set.intersection(*sets) if sets else set()
            keep = np.array([k in common for k in keys], bool)
            print(f"  --common-species '{s}': {len(common)} of {len(set(keys))} "
                  f"(world, species) keys appear in all {len(regimes)} regimes; "
                  f"keeping {int(keep.sum())} of {keep.size} rows")
            if keep.sum() == 0:
                sys.exit("  ABORT: --common-species left no rows. The CSVs share no "
                         "(world_stem, species) key, so the schemes selected different "
                         "species. Drop the flag, or re-run posterior_per_species.py "
                         "with the same species list across schemes.")
            for k in ('v', 'n_ens', 'world', 'n_obs', 'range_N', 'regime', 'species'):
                d[k] = d[k][keep]

    # ---- what the zero-record bin is telling you
    for s, d in data.items():
        z = int(np.sum(d['n_obs'] == 0))
        if z == 0:
            print(f"  '{s}': NO species has zero records. "
                  f"generate_reconstructions_proportional.py calls "
                  f"sparsify_history_proportional(..., min_obs=1), which guarantees at "
                  f"least one observed cell per present species, so the zero-record case "
                  f"is untested by these runs. Say so in the methods.")
        else:
            print(f"  '{s}': {z} species with zero records (kept as bin '0')")
        neg = int(np.sum(d['n_obs'] < 0))
        if neg:
            print(f"  ! '{s}': {neg} row(s) with unreadable n_obs, excluded")

    pooled = {s: bin_curve(d, n_boot=args.n_boot, seed=args.seed + 1)
              for s, d in data.items()}
    per_reg = {s: {g: bin_curve(d, mask_extra=(d['regime'] == g),
                                n_boot=args.n_boot, seed=args.seed + 1)
                   for g in regimes} for s, d in data.items()}

    # ---- drop bins no species can reach; a dangling empty tick reads as a bug
    keep_bin = [any(pooled[s][i]['n'] > 0 for s in stats_present)
                for i in range(len(OBS_BINS))]
    sel = [i for i in range(len(OBS_BINS)) if keep_bin[i]]
    dropped = [OBS_BINS[i][2] for i in range(len(OBS_BINS)) if not keep_bin[i]]
    if dropped:
        print(f"  empty bins removed from the axis: {', '.join(dropped)}")
    if not sel:
        sys.exit("  ABORT: every record bin is empty, so there is nothing to plot. "
                 "Check that n_obs is populated in the CSVs.")
    pooled = {s: [pooled[s][i] for i in sel] for s in stats_present}
    per_reg = {s: {g: [per_reg[s][g][i] for i in sel] for g in regimes}
               for s in stats_present}

    thin = sorted({b['label'] for s in stats_present for b in pooled[s]
                   if 0 < b['n'] < args.min_n})
    if thin:
        print(f"  bins below --min-n={args.min_n} drawn as open markers and left out of "
              f"the trend line: {', '.join(thin)}")
    for s in stats_present:
        for b in pooled[s]:
            if b['ci_method'] == 'clopper-pearson-on-worlds':
                print(f"  ! '{s}' bin {b['label']}: every world gave the same coverage, so "
                      f"the bootstrap interval collapsed to zero width. Reporting "
                      f"Clopper-Pearson on {b['nworld']} worlds instead "
                      f"({b['ci'][0]:.0%}-{b['ci'][1]:.0%}).")
        occupied = [b['nworld'] for b in pooled[s] if b['n']]
        if occupied and min(occupied) < 10:
            print(f"  ! '{s}': some bins rest on fewer than 10 worlds; percentile "
                  f"bootstrap intervals from so few clusters are coarse.")

    xs = np.arange(len(sel))
    labels = [OBS_BINS[i][2] for i in sel]

    ncol = 1 if args.pooled_only else 2
    w_in = WIDTH_MM[args.width] * MM
    fig, axes = plt.subplots(1, ncol, figsize=(w_in, args.height_mm * MM),
                             squeeze=False, layout='constrained')

    # ---------------- panel (a): pooled ----------------
    ax = axes[0, 0]
    for i, s in enumerate(stats_present):
        y = np.array([b['cov'] for b in pooled[s]]) * 100
        lo_ = np.array([b['ci'][0] for b in pooled[s]]) * 100
        hi_ = np.array([b['ci'][1] for b in pooled[s]]) * 100
        nn = np.array([b['n'] for b in pooled[s]])
        err = [np.nan_to_num(y - lo_, nan=0.0), np.nan_to_num(hi_ - y, nan=0.0)]
        solid = (~np.isnan(y)) & (nn >= args.min_n)
        if solid.any():
            ax.errorbar(xs[solid], y[solid], yerr=[err[0][solid], err[1][solid]],
                        fmt=MARKERS[i % len(MARKERS)] + '-', color=TOL[i % len(TOL)],
                        ms=3.2, lw=1.1, elinewidth=0.7, capsize=1.8,
                        label=NICE.get(s, s))
        thin_m = (~np.isnan(y)) & (nn < args.min_n) & (nn > 0)
        if thin_m.any():                       # shown, but not joined to the trend
            ax.errorbar(xs[thin_m], y[thin_m], yerr=[err[0][thin_m], err[1][thin_m]],
                        fmt=MARKERS[i % len(MARKERS)], mfc='white',
                        color=TOL[i % len(TOL)], ms=3.2, lw=0, elinewidth=0.6,
                        capsize=1.8, label=None if solid.any() else NICE.get(s, s))
    ax.axhline(c0 * 100, ls='--', lw=1.0, color=C_NULL, zorder=1)
    # left end: the fewest-record bins sit lowest, so the label stays clear
    ax.text(-0.35, c0 * 100 + 1.6, f'exchangeable null {c0:.0%}',
            ha='left', va='bottom', fontsize=args.font_pt, color=C_NULL)
    for i, b in enumerate(pooled[stats_present[0]]):
        if b['n']:
            txt = (f"n={b['n']}" if b['n'] == b['n_uniq']
                   else f"n={b['n']}\n({b['n_uniq']} sp.)")
            ax.annotate(txt, (xs[i], 1.5), ha='center', va='bottom',
                        fontsize=args.font_pt, color=C_GREY, linespacing=1.1)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel('records per species')
    ax.set_ylabel('truth within ensemble (5\u201395%)  [%]')
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.45, len(sel) - 0.55)        # len(sel), not len(OBS_BINS)
    ax.set_title('(a) pooled over sampling schemes', loc='left',
                 fontweight='bold', pad=3.0)
    # lower right, lifted above the n labels: with 7 pt text an upper-left
    # legend sits on the one-record points and the 88% label
    ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 0.13),
              handlelength=1.5)
    ax.spines[['top', 'right']].set_visible(False)

    # ---------------- panel (b): per scheme ----------------
    if ncol == 2:
        ax = axes[0, 1]
        s0 = stats_present[0]
        for i, g in enumerate(regimes):
            b = per_reg[s0][g]
            y = np.array([q['cov'] for q in b]) * 100
            nn = np.array([q['n'] for q in b])
            solid = (~np.isnan(y)) & (nn >= args.min_n)
            if solid.any():
                ax.plot(xs[solid], y[solid], MARKERS[i % len(MARKERS)] + '-',
                        color=TOL[i % len(TOL)], ms=3.0, lw=1.0, label=g)
            thin_m = (~np.isnan(y)) & (nn < args.min_n) & (nn > 0)
            if thin_m.any():
                ax.plot(xs[thin_m], y[thin_m], MARKERS[i % len(MARKERS)],
                        mfc='white', color=TOL[i % len(TOL)], ms=3.0, lw=0,
                        label=None if solid.any() else g)
        ax.axhline(c0 * 100, ls='--', lw=1.0, color=C_NULL, zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel('records per species')
        ax.set_ylabel('truth within ensemble (5\u201395%)  [%]')
        ax.set_ylim(0, 104)
        ax.set_xlim(-0.45, len(sel) - 0.55)
        ax.set_title(f'(b) {NICE.get(s0, s0)}, by sampling scheme', loc='left',
                     fontweight='bold', pad=3.0)
        ax.legend(frameon=False, loc='lower right', handlelength=1.5,
                  fontsize=args.font_pt)
        ax.spines[['top', 'right']].set_visible(False)

    if args.draft_title:
        fig.suptitle('Calibration against records per species',
                     fontsize=args.font_pt + 1, fontweight='bold')
    stem = save(fig, args.output, args.dpi, args.width, getattr(args, 'submit_as', None))

    out_rows = []
    for s in stats_present:
        for tag, curve in [('POOLED', pooled[s])] + [(g, per_reg[s][g]) for g in regimes]:
            for b in curve:
                out_rows.append([
                    s, tag, b['label'], b['lo'],
                    '' if b['hi'] > 10 ** 8 else b['hi'], b['n'], b['n_uniq'],
                    b['nworld'], f"{b['cov']:.4f}", f"{b['ci'][0]:.4f}",
                    f"{b['ci'][1]:.4f}", b['ci_method'], int(b['n'] < args.min_n),
                    f"{c0:.4f}", f"{b['med_range']:.1f}", n_ens])
    write_values(stem.parent / (stem.name + '_values.csv'),
                 RECORDS_VALUE_HEADER, out_rows)

    n_cases = sum(d['v'].size for d in data.values()) // max(len(data), 1)
    print(f"\n  pooled {n_cases} species-cases from {len(regimes)} scheme(s): "
          f"{', '.join(regimes)}")
    print(f"  {'bin':>5} {'n':>5} {'uniq':>5} {'worlds':>7} " +
          " ".join(f"{NICE.get(s, s)[:9]:>10}" for s in stats_present) +
          "   median range_N")
    for i, b0 in enumerate(pooled[stats_present[0]]):
        cells = " ".join(('       n/a' if np.isnan(pooled[s][i]['cov'])
                          else f"{pooled[s][i]['cov'] * 100:9.0f}%")
                         for s in stats_present)
        mr = '  n/a' if np.isnan(b0['med_range']) else f"{b0['med_range']:5.0f}"
        print(f"  {b0['label']:>5} {b0['n']:>5} {b0['n_uniq']:>5} "
              f"{b0['nworld']:>7} {cells}   {mr}")

    print("\n  CONFOUND CHECK  (median true range size per bin, by scheme)")
    for g in regimes:
        line = "  ".join(
            f"{b['label']}:{'n/a' if np.isnan(b['med_range']) else int(b['med_range'])}"
            for b in per_reg[stats_present[0]][g])
        print(f"    {g:>12}: {line}")
    print("  Equal record counts from different schemes do NOT mean equal species. "
          "Read panel (b) and the median ranges before claiming one law.")

    def solid_bins(s):
        return [b for b in pooled[s] if b['n'] >= args.min_n and not np.isnan(b['cov'])]

    parts = [f"{NICE.get(s, s)} rises from {solid_bins(s)[0]['cov']:.0%} at "
             f"{solid_bins(s)[0]['label']} record(s) to {solid_bins(s)[-1]['cov']:.0%} "
             f"at {solid_bins(s)[-1]['label']}"
             for s in stats_present if len(solid_bins(s)) >= 2]
    seg = ("; ".join(parts) + ". ") if parts else ""
    print("\n  CAPTION\n"
          f"  Fig. Y. Share of species whose true range statistic falls inside the "
          f"5-95 percentile band of the {n_ens}-member ensemble, against the number of "
          f"records that species carried. (a) pooled over the {len(regimes)} sampling "
          f"schemes ({', '.join(regimes)}), with world-clustered bootstrap intervals; "
          f"(b) the same curve drawn separately per scheme. The dashed line at "
          f"{c0:.0%} is the coverage attainable under the exchangeable null, where the "
          f"rank of the truth is uniform on {{0,...,{n_ens}}}; it is below 90% because "
          f"the rank is discrete. Open markers mark bins holding fewer than "
          f"{args.min_n} species and are not joined to the trend. {seg}"
          f"Because record count and range size are linked under proportional "
          f"sampling, median true range size per bin is reported in the accompanying "
          f"values file.\n")


# ===========================================================================
def build_parser():
    ap = argparse.ArgumentParser(
        description="Calibration figures for the reconstruction ensemble, "
                    "Elsevier artwork geometry")
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--stats', nargs='+', default=['patches', 'spread'])
        p.add_argument('--width', choices=list(WIDTH_MM), default='double')
        p.add_argument('--font-pt', type=float, default=7.0)
        p.add_argument('--dpi', type=int, default=1000,
                       help='PNG resolution; the guide asks >= 1000 dpi for line art')
        p.add_argument('--draft-title', action='store_true',
                       help='in-figure title, internal review only')
        p.add_argument('--force-pool', action='store_true')
        p.add_argument('--n-boot', type=int, default=2000)
        p.add_argument('--seed', type=int, default=0)

    r = sub.add_parser('rank', help='pooled PIT / rank histogram, ONE regime')
    r.add_argument('--csv', nargs='+', required=True, help='path or label=path')
    r.add_argument('--output', required=True)
    r.add_argument('--label', default='')
    r.add_argument('--submit-as', default=None,
                   help="also write this name as PDF, e.g. 'Figure_S8'")
    r.add_argument('--bins', default='auto')
    r.add_argument('--height-mm', type=float, default=72.0)
    r.add_argument('--annotate', choices=['compact', 'none'], default='compact')
    r.add_argument('--also-ks', action='store_true')
    common(r)

    c = sub.add_parser('records', help='coverage against records per species')
    c.add_argument('--csv', nargs='+', required=True,
                   help='label=path, one per sampling scheme')
    c.add_argument('--output', required=True)
    c.add_argument('--height-mm', type=float, default=84.0)
    c.add_argument('--pooled-only', action='store_true')
    c.add_argument('--min-n', type=int, default=10,
                   help='bins with fewer species are drawn open, off the trend')
    c.add_argument('--submit-as', default=None,
                   help="also write this name as PDF, e.g. 'Figure_6'")
    c.add_argument('--common-species', action='store_true',
                   help='keep only (world, species) present in EVERY regime')
    common(c)

    a = sub.add_parser('all', help='every figure from one command')
    a.add_argument('--csv', nargs='+', required=True, help='label=path, one per scheme')
    a.add_argument('--outdir', required=True)
    a.add_argument('--bins', default='auto')
    a.add_argument('--annotate', choices=['compact', 'none'], default='compact')
    a.add_argument('--also-ks', action='store_true')
    a.add_argument('--pooled-only', action='store_true')
    a.add_argument('--min-n', type=int, default=10)
    a.add_argument('--common-species', action='store_true')
    a.add_argument('--submit-records-as', default=None,
                   help="also write the records figure under this name, e.g. 'Figure_6'")
    common(a)
    return ap


def main():
    args = build_parser().parse_args()
    if args.font_pt < MIN_TEXT_PT:
        sys.exit(f"  ABORT: --font-pt below {MIN_TEXT_PT:g} pt, the Elsevier size for "
                 f"normal lettering.")
    set_style(args.font_pt)

    if args.cmd == 'rank':
        rows = load_specs(args.csv)
        if not rows:
            sys.exit("  ABORT: no rows.")
        regs = {r['__regime__'] for r in rows}
        if len(regs) > 1:
            print(f"  ! {len(regs)} regimes in one rank histogram ({sorted(regs)}); "
                  f"a rank histogram describes ONE sampling scheme.")
        figure_rank(rows, args)

    elif args.cmd == 'records':
        rows = load_specs(args.csv)
        if not rows:
            sys.exit("  ABORT: no rows.")
        figure_records(rows, args)

    else:   # all
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        specs = [parse_spec(s) for s in args.csv]
        slugs = [slug(l) for l, _ in specs]
        if len(set(slugs)) != len(slugs):
            sys.exit(f"  ABORT: labels collide once made filename-safe: {slugs}.")
        for lab, path in specs:
            print(f"\n=== rank histogram: {lab} ===")
            a = argparse.Namespace(**vars(args))
            a.csv = [f"{lab}={path}"]
            a.output = str(outdir / f"Fig_calibration_{slug(lab)}")
            a.label = lab
            a.height_mm = 72.0
            sub_rows = load_specs(a.csv)
            if not sub_rows:
                print(f"  ! no rows for {lab}, skipped")
                continue
            figure_rank(sub_rows, a)
        print("\n=== coverage against records ===")
        a = argparse.Namespace(**vars(args))
        a.csv = args.csv
        a.output = str(outdir / 'Fig_calibration_vs_records')
        a.submit_as = args.submit_records_as
        a.height_mm = 84.0
        figure_records(load_specs(a.csv), a)


if __name__ == '__main__':
    main()