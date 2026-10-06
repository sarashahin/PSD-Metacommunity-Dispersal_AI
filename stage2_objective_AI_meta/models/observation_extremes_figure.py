#!/usr/bin/env python3
"""
=============================================================================
OBSERVATION SCHEMES: fixed-K versus proportional  -  diagnostic figure
Elsevier artwork geometry (Ecological Informatics / Ecological Modelling)
=============================================================================
Two observation extremes bracket the real recording process:

  fixed-K       every species contributes the same number of records, K.
  proportional  every occupied cell is recorded independently with
                probability p, so wide-ranging species get many records and
                narrow-ranging species few.

This script draws the diagnostic ONLY. The reconstructions come from
generate_reconstructions_proportional.py.

UPDATE 2 OCT 2026  (Supplementary Figures S2 and S3, Word numbers 9 and 10)
---------------------------------------------------------------------------
A. THE SHADED FLOOR NOW MATCHES THE COVERAGE ANALYSIS. The old floor,
   max(K+3, ceil(1.5K)) = 8 at K = 5, belonged to pick_species mode 'rare'.
   The coverage CSVs (pit_K5_random_s1, pit_PROP_random_s1, pit_PROP30) were
   drawn with mode 'random', whose rule is range > K, so the analysed pool
   starts at K+1 = 6 cells. --analysed-min overrides it; the default is K+1.
B. EXACT SELECTION REPLICA. --match-selection N --select-seed S re-draws
   pick_species 'random' line for line (eligible = range > K, one
   default_rng(S) per world, choice without replacement, sorted). With
   --verify-csv the drawn species are compared with the CSV, world by world.
   The old replica took the narrowest species, which is the 'rare' rule.
C. ARTWORK RULES (Elsevier artwork sizing; Ecological Informatics guide,
   both read 2 Oct 2026): every text element at --font-pt, which may not go
   below 7 pt; every text is audited before saving; PNG at >= 1000 dpi
   (bitmapped line art); vector PDF; no title inside the figure;
   --submit-as Figure_S2 writes a separately named PDF for upload.
D. Wording: the floor describes the coverage analysis only. Other analyses
   use their own range filters, so the log no longer says "every result".

WHY THE VERSION BEFORE THAT COULD NOT BE TRUSTED (kept for the record)
----------------------------------------------------------------------
1. NOT REPRODUCIBLE. The per-world seed was `args.seed + abs(hash(stem))`.
   Python randomises string hashing per process (PYTHONHASHSEED), so the same
   command gave a different figure on every run; measured offsets on three
   consecutive runs were 815260650, 1653910337, 1784459072.
2. NOT THE SAME MASKS AS THE GENERATOR, despite the docstring saying so.
   sparsify_history_proportional uses np.random.default_rng(rng_seed) with
   rng_seed=42, the SAME seed for every world. The default here reproduces
   that exactly; --per-world-seed opts into independent worlds with a stable
   CRC32 hash, and says so in the log.
3. THRESHOLD DIVERGENCE, silent. The generator selects cells with
   `Pt[-1][s] > 0`; the old loader binarised at `> 0.5`. If the truth array is
   continuous these disagree. The default here is the generator's `> 0` and
   the log reports how many cells fall in the disputed band.
4. plt.savefig("FigX.pdf") wrote to the working directory, so every run
   overwrote the previous PDF regardless of --fig-path.
5. The world count in the title, `len(set(stems)) - len(skipped)`, ignored
   stems whose file did not exist, so it could overcount.
6. The zero-record rate was always reported as 0 because min_obs=1 tops every
   species up. The pure-Bernoulli rate, which is the number the methods
   section needs, was never computed. Both are reported now.

--verify-csv re-simulates the mask for exactly the species named in a pit CSV
and compares index by index. A distribution can agree by luck; an index-matched
comparison cannot.
=============================================================================
"""

import argparse
import csv
import glob
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

MM = 1.0 / 25.4
WIDTH_MM = {'single': 90.0, '1.5': 140.0, 'double': 190.0}
C_FIXED, C_PROP, C_LINE, C_GREY = '#CC6677', '#4477AA', '#EE6677', '#777777'
C_ACTUAL = '#228833'
PREFERRED_FONTS = ['Arial', 'Helvetica', 'Nimbus Sans', 'Liberation Sans',
                   'DejaVu Sans']
TRUTH_KEYS = ['P_last_final', 'P_final', 'P']
MIN_TEXT_PT = 7.0          # Elsevier artwork sizing: normal lettering
MIN_LINEART_DPI = 1000     # Ecological Informatics guide: bitmapped line drawings
WRITTEN = []


def set_style(base_pt):
    have = {f.name for f in font_manager.fontManager.ttflist}
    chain = [f for f in PREFERRED_FONTS if f in have] or ['DejaVu Sans']
    if chain[0] in ('Arial', 'Helvetica', 'Nimbus Sans'):
        print(f"  font check: '{chain[0]}'")
    else:
        print(f"  ! font check: Arial/Helvetica absent, using '{chain[0]}'.")
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
def load_truth(path, threshold, forced_key=None):
    """Return (binary S,Y,X array, key used, #cells in the disputed band).

    The disputed band is 0 < value <= 0.5: cells the generator counts as
    occupied (`> 0`) but a 0.5 threshold would drop. Non-zero means the two
    scripts are looking at different truths."""
    with np.load(path, allow_pickle=True) as td:
        files = list(td.files)
        arr, used = None, None
        for k in ([forced_key] if forced_key else []) + TRUTH_KEYS:
            if k and k in files:
                a = np.asarray(td[k])
                if a.ndim == 3:
                    arr, used = a, k
                    break
        if arr is None and 'P_t' in files:
            a = np.asarray(td['P_t'])
            if a.ndim == 4:
                arr, used = a[-1], 'P_t[-1]'
    if arr is None:
        raise KeyError(f"no 3-D truth array; keys present: {files}")
    disputed = int(np.sum((arr > 0) & (arr <= 0.5)))
    return (arr > threshold).astype(np.uint8), used, disputed


def proportional_mask(truth_world, p, rng, min_obs=1):
    """Byte-for-byte the rule in sparsify_history_proportional: iterate species
    in order, draw len(occupied) uniforms from ONE generator, and when the draw
    falls short of min_obs REPLACE the whole keep vector with min_obs cells
    chosen at random (the shortfall is not topped up, it is redrawn)."""
    S = truth_world.shape[0]
    mask = np.zeros_like(truth_world)
    counts = np.zeros(S, dtype=int)
    raw = np.full(S, -1, dtype=int)          # pure-Bernoulli count, before top-up
    for s in range(S):
        cells = np.argwhere(truth_world[s] > 0)
        if len(cells) == 0:
            continue
        keep = rng.random(len(cells)) < p
        raw[s] = int(keep.sum())
        if keep.sum() < min_obs:
            idx = rng.choice(len(cells), size=min(min_obs, len(cells)), replace=False)
            keep = np.zeros(len(cells), dtype=bool)
            keep[idx] = True
        for (y, x) in cells[keep]:
            mask[s, y, x] = 1
        counts[s] = int(keep.sum())
    return mask, counts, raw


def pick_random(ranges, K, n_species, seed):
    """pick_species(mode='random') line for line: eligible = range > K, one
    default_rng(seed) per world, choice without replacement, sorted."""
    eligible = [s for s in range(len(ranges)) if ranges[s] > K]
    rng = np.random.default_rng(seed)
    if len(eligible) <= n_species:
        return eligible
    return sorted(rng.choice(eligible, size=n_species, replace=False).tolist())


def plain_log_ticks(axis, lo, hi, cands=(1, 2, 5, 10, 20, 50, 100, 200, 500,
                                          1000, 10000, 100000)):
    """Plain numbers on a log axis ('10', not 10 with a superscript 1): the
    default mathtext labels print the exponent near 0.7 x the font size, under
    the 6 pt Elsevier floor for superscripts at 7 pt text."""
    ticks = [c for c in cands if lo <= c <= hi]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())


def world_seed(stem, base, per_world):
    """CRC32 is stable across processes and Python versions; hash() is not."""
    if not per_world:
        return base
    return (base + zlib.crc32(stem.encode())) % (2 ** 32)


def audit_text(fig, floor=MIN_TEXT_PT):
    """Every visible text must be at least `floor` pt; abort before saving
    otherwise. Returns the smallest size found."""
    fig.canvas.draw()                    # tick labels exist only after a draw
    small, sizes = [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        fs = float(t.get_fontsize())
        txt = t.get_text()
        if '$' in txt and ('^' in txt or '_' in txt):
            fs *= 0.7                        # mathtext sub/superscript scale
            if fs < 6.0 - 1e-6:              # Elsevier floor for sub/superscripts
                small.append((round(fs, 2), txt[:30]))
            continue
        sizes.append(fs)
        if fs < floor - 1e-6:
            small.append((fs, txt[:30].replace('\n', ' ')))
    if small:
        sys.exit(f"  ABORT: {len(small)} text element(s) below {floor:g} pt, e.g. "
                 f"{small[:3]}. Elsevier asks 7 pt for normal lettering and 6 pt for "
                 f"sub/superscripts (a log axis prints 10^n as a superscript).")
    return min(sizes) if sizes else float('nan')


def save(fig, output, dpi, width_key, sources=(), n_points=None, submit_as=None):
    if n_points is not None and n_points <= 0:
        sys.exit(f"  ABORT: nothing plotted for '{output}'. Refusing to write a "
                 f"blank figure.")
    txt = str(output)
    for ext in ('.png', '.pdf', '.eps', '.tif', '.tiff'):
        if txt.lower().endswith(ext):
            txt = txt[:-len(ext)]
            break
    stem = Path(txt)
    stem.parent.mkdir(parents=True, exist_ok=True)
    smallest = audit_text(fig)
    pdf_p, png_p = Path(txt + '.pdf'), Path(txt + '.png')
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    prov = f"observation_extremes | {ts} | {', '.join(str(x) for x in sources)}"
    meta_pdf = {'Title': stem.name, 'Subject': prov,
                'Author': 'observation_extremes_figure.py', 'Creator': 'matplotlib'}
    fig.savefig(pdf_p, metadata=meta_pdf)
    fig.savefig(png_p, dpi=dpi, metadata={'Software': 'observation_extremes_figure.py',
                                          'Description': prov})
    if submit_as:
        sub_p = stem.parent / (Path(submit_as).name.split('.pdf')[0] + '.pdf')
        fig.savefig(sub_p, metadata=meta_pdf)
        print(f"  saved -> {sub_p}  (submission copy, vector)")
        WRITTEN.append(sub_p)
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
    WRITTEN.extend([pdf_p, png_p])
    return stem


def verify_against_csv(paths, truth_dir, prob, min_obs, seed, per_world, threshold,
                       forced_key, K, analysed_min, select_seed):
    """Re-run the observation model for exactly the species named in the pit CSVs
    and compare species by species. A distribution can agree by luck; an
    index-matched comparison cannot. Then re-draw the species selection with
    pick_species 'random' and compare it with the CSV, world by world."""
    rows = []
    for pat in paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            if not Path(p).exists():
                print(f"  ! --verify-csv missing, skipped: {p}")
                continue
            with open(p, newline='') as f:
                rows += list(csv.DictReader(f))
    need = {'world_stem', 'species', 'n_obs'}
    if not rows or not need.issubset(rows[0]):
        print(f"  ! --verify-csv needs columns {sorted(need)}; skipping verification")
        return
    by_world, seen, dup = {}, set(), 0
    for r in rows:
        try:
            key = (r['world_stem'], int(float(r['species'])))
        except (ValueError, TypeError, KeyError):
            continue
        if key in seen:                      # same species twice means two regimes
            dup += 1                         # were passed at once, which is meaningless
            continue
        seen.add(key)
        try:
            by_world.setdefault(r['world_stem'], []).append(
                (int(float(r['species'])), int(float(r['n_obs'])),
                 int(float(r.get('range_N') or -1))))
        except (ValueError, TypeError):
            continue
    if dup:
        print(f"  ! --verify-csv: {dup} duplicated (world, species) row(s) ignored. "
              f"Passing CSVs from DIFFERENT sampling regimes at once gives a "
              f"meaningless comparison; verify one regime per run.")

    print(f"\n  EXACT VERIFICATION  (re-simulating the mask for the named species)")
    print(f"  {'world':<44} {'n_obs match':>12} {'range match':>12} "
          f"{'sim mean':>9} {'actual':>8}")
    tot = ok_n = ok_r = 0
    sim_sum = act_sum = 0
    sel_ranges, elig_ranges = [], []
    sel_ok, sel_tot = 0, 0
    for stem, recs in sorted(by_world.items()):
        tp = Path(truth_dir) / f'{stem}.npz'
        if not tp.exists():
            print(f"  ! truth not found for {stem[:50]}")
            continue
        truth, _, _ = load_truth(tp, threshold, forced_key)
        _, counts, _ = proportional_mask(
            truth, prob, np.random.default_rng(world_seed(stem, seed, per_world)),
            min_obs)
        N = truth.sum(axis=(1, 2)).astype(int)
        valid = [(s, a, rn) for s, a, rn in recs if 0 <= s < counts.size]
        if len(valid) != len(recs):
            print(f"  ! {len(recs) - len(valid)} species index out of range in {stem[:40]}")
        mn = sum(counts[s] == a for s, a, _ in valid)
        mr = sum(N[s] == rn for s, _, rn in valid if rn >= 0)
        nr = sum(1 for _, _, rn in valid if rn >= 0)
        sm = float(np.mean([counts[s] for s, _, _ in valid])) if valid else float('nan')
        am = float(np.mean([a for _, a, _ in valid])) if valid else float('nan')
        sel_ranges += [N[s] for s, _, _ in valid]
        elig_ranges += list(N[N >= analysed_min])
        tot += len(valid); ok_n += mn; ok_r += mr
        sim_sum += sum(counts[s] for s, _, _ in valid)
        act_sum += sum(a for _, a, _ in valid)
        replica = set(pick_random(N, K, len(recs), select_seed))
        sel_tot += 1
        sel_ok += int(replica == {s for s, _, _ in recs})
        print(f"  {stem[:42]:<44} {f'{mn}/{len(valid)}':>12} "
              f"{(f'{mr}/{nr}' if nr else 'n/a'):>12} {sm:>9.2f} {am:>8.2f}")
    if not tot:
        print("  ! nothing verified")
        return
    print(f"  {'POOLED':<44} {f'{ok_n}/{tot}':>12} {f'{ok_r}/{tot}':>12} "
          f"{sim_sum / tot:>9.3f} {act_sum / tot:>8.3f}")
    if ok_n == tot and ok_r == tot:
        print(f"  -> EXACT MATCH on all {tot} species. The observation model in this "
              f"script is the one that produced the reconstructions.")
    else:
        print(f"  -> MISMATCH on {tot - ok_n} record count(s) and {tot - ok_r} range(s). "
              f"Check --prob, --min-obs, --rng-seed and --truth-threshold against the "
              f"generator before using any of these numbers.")

    sr, er = np.asarray(sel_ranges), np.asarray(elig_ranges)
    print(f"\n  SELECTION CHECK  (which species did the coverage analysis use?)")
    print(f"    pick_species 'random' (range > K={K}, seed {select_seed}) re-drawn: "
          f"{sel_ok}/{sel_tot} worlds give exactly the CSV's species")
    print(f"    analysed species  : n={sr.size:>5}  median range {np.median(sr):.0f}  "
          f"min {sr.min()}  max {sr.max()}")
    print(f"    eligible pool (>= {analysed_min}): n={er.size:>5}  median range "
          f"{np.median(er):.0f}")
    below = int(np.sum(sr < analysed_min))
    if below:
        print(f"    ! {below} analysed species have a range below {analysed_min}: the "
              f"shaded floor in the figure does NOT describe these CSVs.")
    else:
        print(f"    -> every analysed species has a range >= {analysed_min}, so the "
              f"shaded floor matches the CSVs.")
    if sel_ok != sel_tot:
        print(f"    ! the 'random' re-draw differs in {sel_tot - sel_ok} world(s): check "
              f"--select-seed and --K-compare against the posterior_per_species.py run.")


def read_actual_nobs(paths):
    """n_obs actually used by the reconstructions, from the pit CSVs."""
    vals = []
    for pat in paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            if not Path(p).exists():
                print(f"  ! --compare-csv missing, skipped: {p}")
                continue
            with open(p, newline='') as f:
                for r in csv.DictReader(f):
                    s = (r.get('n_obs') or '').strip()
                    if s:
                        try:
                            vals.append(int(float(s)))
                        except ValueError:
                            pass
    return np.asarray(vals, int)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="fixed-K vs proportional observation counts, Elsevier geometry")
    ap.add_argument('--truth-dir', required=True)
    ap.add_argument('--world-stems', default=None,
                    help='comma-separated stems (no .npz); omit to use --pattern')
    ap.add_argument('--pattern', default='*.npz')
    ap.add_argument('--truth-key', default=None)
    ap.add_argument('--truth-threshold', type=float, default=0.0,
                    help='cells counted as occupied are value > this. The '
                         'generator uses 0.0; pass 0.5 only if you change it there too')
    ap.add_argument('--prob', type=float, default=0.10)
    ap.add_argument('--min-obs', type=int, default=1,
                    help='must match --obs-min of generate_reconstructions_proportional.py')
    ap.add_argument('--K-compare', type=int, default=5)
    ap.add_argument('--analysed-min', type=int, default=None,
                    help='smallest range the coverage analysis can draw. Default K+1, '
                         "the rule of pick_species mode 'random' (range > K)")
    ap.add_argument('--min-range', type=int, default=0,
                    help='drop species with a true range below this before plotting')
    ap.add_argument('--log-x', action='store_true',
                    help='log range axis in panel (a). Occupancy is heavily '
                         'right-skewed, so a linear axis buries most species at the left')
    ap.add_argument('--rng-seed', type=int, default=42,
                    help='must match --rng-seed of the generator (default there is 42)')
    ap.add_argument('--per-world-seed', action='store_true',
                    help='independent CRC32-derived seed per world. This does NOT '
                         'reproduce the generator, which reuses one seed everywhere')
    ap.add_argument('--match-selection', type=int, default=0, metavar='N',
                    help="re-draw N species per world with pick_species 'random' and "
                         "compare their record counts with --compare-csv")
    ap.add_argument('--select-seed', type=int, default=1,
                    help="seed of pick_species 'random' (the coverage CSVs used 1)")
    ap.add_argument('--verify-csv', nargs='*', default=None,
                    help='pit CSV(s) with world_stem, species, n_obs and range_N. '
                         'Re-simulates the mask for EXACTLY those species and checks '
                         'n_obs, range_N and the species selection. One regime per run')
    ap.add_argument('--compare-csv', nargs='*', default=None,
                    help='pit CSV(s) with an n_obs column, for the distribution '
                         'overlay in panel (b)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--output', default=None, help='figure stem; default inside --out-dir')
    ap.add_argument('--submit-as', default=None,
                    help="also write this name as PDF, e.g. 'Figure_S2'")
    ap.add_argument('--save-masks', action='store_true')
    ap.add_argument('--width', choices=list(WIDTH_MM), default='double')
    ap.add_argument('--height-mm', type=float, default=78.0)
    ap.add_argument('--font-pt', type=float, default=7.0)
    ap.add_argument('--dpi', type=int, default=1000,
                    help='PNG resolution; the guide asks >= 1000 dpi for line art')
    ap.add_argument('--draft-title', action='store_true')
    args = ap.parse_args()

    if args.font_pt < MIN_TEXT_PT:
        sys.exit(f'  ABORT: --font-pt below {MIN_TEXT_PT:g} pt, the Elsevier size for '
                 f'normal lettering.')
    set_style(args.font_pt)
    K = args.K_compare
    floor = args.analysed_min if args.analysed_min is not None else K + 1

    truth_dir = Path(args.truth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (out_dir / f'Fig_observation_extremes_p{args.prob:.2f}')

    if args.per_world_seed:
        print(f"  seeding: CRC32 per world from base {args.rng_seed}. This does NOT "
              f"match the generator, which seeds every world with {args.rng_seed}.")
    else:
        print(f"  seeding: np.random.default_rng({args.rng_seed}) per world, "
              f"reproducing sparsify_history_proportional exactly.")
    print(f"  analysed floor: range >= {floor} "
          f"({'set by --analysed-min' if args.analysed_min is not None else 'K+1, the random-selection rule'})")

    if args.world_stems:
        stems = [s.strip() for s in args.world_stems.split(',') if s.strip()]
    else:
        stems = sorted(Path(p).stem for p in glob.glob(str(truth_dir / args.pattern)))
    if not stems:
        sys.exit(f'  ABORT: no files matching {args.pattern} in {truth_dir}')

    all_N, all_prop, all_fixed, all_raw = [], [], [], []
    sel_prop, sel_N = [], []
    keys_used, disputed_total, ok_worlds, skipped = set(), 0, 0, []
    per_world_rows = []

    print(f"\n  proportional p = {args.prob:.2f}   min_obs = {args.min_obs}   "
          f"threshold > {args.truth_threshold}")
    print(f"  {'world':<44} {'species':>7} {'mean N':>7} {'mean obs':>9} "
          f"{'0-rec (Bernoulli)':>18}")
    for stem in stems:
        tp = truth_dir / f'{stem}.npz'
        if not tp.exists():
            skipped.append((stem, 'file not found'))
            continue
        try:
            truth, key_used, disputed = load_truth(tp, args.truth_threshold,
                                                   args.truth_key)
        except (KeyError, OSError, ValueError) as e:
            skipped.append((stem, str(e)))
            continue
        if truth.ndim != 3 or truth.shape[0] == 0:
            skipped.append((stem, f'bad shape {truth.shape}'))
            continue
        keys_used.add(key_used)
        disputed_total += disputed
        rng = np.random.default_rng(world_seed(stem, args.rng_seed, args.per_world_seed))
        mask, counts, raw = proportional_mask(truth, args.prob, rng, args.min_obs)
        if args.save_masks:
            np.savez(out_dir / f'{stem}__obsmask_p{args.prob:.2f}.npz',
                     obs_mask=mask.astype(np.uint8), prob=np.float32(args.prob),
                     world_stem=stem, rng_seed=np.int64(args.rng_seed),
                     min_obs=np.int64(args.min_obs))
        N = truth.sum(axis=(1, 2)).astype(int)
        present = N > 0
        if not present.any():
            skipped.append((stem, f'no species occupies a cell above the '
                                  f'threshold > {args.truth_threshold}'))
            continue
        all_N += list(N[present])
        all_prop += list(counts[present])
        all_raw += list(raw[present])
        all_fixed += list(np.minimum(K, N[present]))
        if args.match_selection:
            order = pick_random(N, K, args.match_selection, args.select_seed)
            sel_prop += list(counts[order])
            sel_N += list(N[order])
        ok_worlds += 1
        zero_bern = float(np.mean(raw[present] == 0))
        per_world_rows.append([stem, int(present.sum()), f"{N[present].mean():.2f}",
                               f"{counts[present].mean():.2f}", f"{zero_bern:.4f}",
                               key_used, disputed])
        print(f"  {stem[:42]:<44} {int(present.sum()):>7} {N[present].mean():>7.1f} "
              f"{counts[present].mean():>9.1f} {zero_bern:>17.1%}")

    if skipped:
        print(f"\n  skipped {len(skipped)} file(s):")
        for s, why in skipped[:10]:
            print(f"    - {s[:56]}  ({why[:60]})")
    if not all_N:
        sys.exit('  ABORT: no usable truth files.')

    if len(keys_used) > 1:
        sys.exit(f"  ABORT: worlds use different truth keys {sorted(keys_used)}. "
                 f"Force one with --truth-key.")
    print(f"\n  truth key: {keys_used.pop()}   worlds used: {ok_worlds}")
    if disputed_total:
        print(f"  ! DISPUTED BAND: {disputed_total} cell(s) have 0 < value <= 0.5. "
              f"The generator counts them as occupied (> 0); a 0.5 threshold would "
              f"not. Keep --truth-threshold 0.0 to match the generator.")

    N = np.asarray(all_N); prop = np.asarray(all_prop)
    fixed = np.asarray(all_fixed); raw = np.asarray(all_raw)

    # ---- two structural facts about THESE worlds, before any filtering
    degenerate = float(np.mean(N <= K))
    above_floor = float(np.mean(N >= floor))
    print(f"\n  OCCUPANCY STRUCTURE (all {N.size} present species)")
    print(f"    mean true range {N.mean():.2f} cells, median {np.median(N):.0f}, "
          f"max {N.max()}")
    print(f"    range <= K={K}: {degenerate:.1%} of species. For these the "
          f"fixed budget observes the ENTIRE range, so there is nothing to reconstruct.")
    print(f"    range >= {floor}: {above_floor:.1%}. This is the pool the coverage "
          f"analysis draws its species from; the shaded area is everything below.")

    if args.min_range:
        keep = N >= args.min_range
        if not keep.any():
            sys.exit(f"  ABORT: --min-range {args.min_range} removes every species "
                     f"(max range is {N.max()}).")
        print(f"  --min-range {args.min_range}: keeping {int(keep.sum())} of {N.size} "
              f"species")
        N, prop, fixed, raw = N[keep], prop[keep], fixed[keep], raw[keep]

    zero_after = float(np.mean(prop == 0))
    zero_bern = float(np.mean(raw == 0))
    topped = float(np.mean(raw < args.min_obs))
    print(f"\n  species (present, pooled): {N.size}")
    print(f"  proportional records: median {np.median(prop):.0f}, "
          f"range {prop.min()}-{prop.max()}, mean {prop.mean():.2f} "
          f"(expected p*N = {args.prob * N.mean():.2f})")
    print(f"  species with ZERO records under pure Bernoulli : {zero_bern:.1%}")
    print(f"  species topped up to min_obs={args.min_obs}      : {topped:.1%}")
    print(f"  species with ZERO records as actually run       : {zero_after:.1%}")
    print(f"  -> the methods sentence: at p={args.prob:.2f} a pure Bernoulli process "
          f"would leave {zero_bern:.0%} of species with no record; min_obs="
          f"{args.min_obs} removes them, so the zero-record case is untested.")

    if args.verify_csv:
        verify_against_csv(args.verify_csv, truth_dir, args.prob, args.min_obs,
                           args.rng_seed, args.per_world_seed, args.truth_threshold,
                           args.truth_key, K, floor, args.select_seed)

    actual = read_actual_nobs(args.compare_csv) if args.compare_csv else np.array([])
    if actual.size:
        qs = [0.25, 0.5, 0.75]
        print(f"\n  CROSS-CHECK against the counts the reconstructions used "
              f"({actual.size} species-rows)")
        print(f"    all species here      : n={prop.size:>5}  median {np.median(prop):.0f}  "
              f"quartiles {np.quantile(prop, qs).round(1)}  mean {prop.mean():.2f}")
        if sel_prop:
            sp = np.asarray(sel_prop)
            print(f"    'random' re-draw here : n={sp.size:>5}  median {np.median(sp):.0f}  "
                  f"quartiles {np.quantile(sp, qs).round(1)}  mean {sp.mean():.2f}  "
                  f"(median range {np.median(sel_N):.0f})")
        print(f"    actually used         : n={actual.size:>5}  median {np.median(actual):.0f}  "
              f"quartiles {np.quantile(actual, qs).round(1)}  mean {actual.mean():.2f}")
        if sel_prop:
            sp = np.asarray(sel_prop)
            same = sp.size == actual.size and np.array_equal(np.sort(sp), np.sort(actual))
            print(f"    -> record counts of the re-draw and the CSV are "
                  f"{'IDENTICAL' if same else 'NOT identical'} as multisets")

    # ---------------- figure ----------------
    fp = args.font_pt
    w_in = WIDTH_MM[args.width] * MM
    fig, axes = plt.subplots(1, 2, figsize=(w_in, args.height_mm * MM),
                             layout='constrained')

    ax = axes[0]
    edge = floor - 0.5                        # ranges are integers: the boundary
    if not args.min_range and above_floor < 1.0:   # sits between K and K+1
        ax.axvspan(0.5 if args.log_x else 0, edge, color='#E8E8E8', zorder=0)
        ax.axvline(edge, color=C_GREY, lw=0.6, ls=':', zorder=1)
        ax.text(edge, 0.02, f'  analysed: range ≥ {floor}  ({above_floor:.1%})',
                transform=ax.get_xaxis_transform(), ha='left', va='bottom',
                fontsize=fp, color=C_GREY)
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, N.size)   # fixed-K is a
    # rasterized: ~36k vector points bloat the PDF and stall some viewers; text,
    # axes and the p*N line stay vector
    ax.scatter(N, fixed + jit, s=2.4, color=C_FIXED, alpha=0.35,  # step function;
               linewidths=0, rasterized=True,                     # jitter both
               label=f'fixed budget (K = {K}, min(K, N))')
    ax.scatter(N, prop + jit, s=2.4, color=C_PROP, alpha=0.35, linewidths=0,
               rasterized=True,
               label=f'proportional (p = {args.prob:.2f}, at least {args.min_obs})')
    xs = np.linspace(max(1, N.min()), N.max(), 200)
    ax.plot(xs, args.prob * xs, '--', color=C_LINE, lw=0.9, zorder=5)
    ax.text(N.max() * 0.98, args.prob * N.max() * 0.98, 'p·N',
            ha='right', va='top', fontsize=fp, color=C_LINE)
    if args.log_x:
        ax.set_xscale('log')
        plain_log_ticks(ax.xaxis, 1, int(N.max()))
    ax.set_xlabel('true range size (occupied cells)')
    ax.set_ylabel('records for that species')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # counts are integers
    ax.set_title('(a) records against range size', loc='left', fontweight='bold', pad=3.0)
    ax.legend(frameon=False, loc='upper left', handlelength=1.2, markerscale=2.5)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    hi = int(max(prop.max(), fixed.max()))
    bins = np.arange(-0.5, hi + 1.5, 1)
    ax.hist(fixed, bins=bins, histtype='step', lw=1.1, color=C_FIXED,
            label=f'fixed budget (K = {K})')
    ax.hist(prop, bins=bins, histtype='step', lw=1.1, color=C_PROP,
            label=f'proportional (p = {args.prob:.2f})')
    if actual.size:
        ax.hist(actual, bins=bins, histtype='step', lw=1.1, ls=':', color=C_ACTUAL,
                label='used by the reconstructions')
    ax.set_yscale('log')
    top = max(np.histogram(v, bins=bins)[0].max() for v in (fixed, prop, actual) if v.size)
    plain_log_ticks(ax.yaxis, 1, int(top),
                    cands=(1, 10, 100, 1000, 10000, 100000))
    ax.set_xlabel('records per species')
    ax.set_ylabel('species')
    ax.set_xlim(-0.5, min(hi, 30) + 0.5)
    ax.text(0.985, 0.62, f"{topped:.0%} of species were forced\nup to the "
            f"{args.min_obs}-record floor", transform=ax.transAxes, ha='right',
            va='top', fontsize=fp, color=C_GREY, linespacing=1.25, zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))  # lines pass behind
    ax.set_title('(b) distribution of record counts', loc='left',
                 fontweight='bold', pad=3.0)
    ax.legend(frameon=False, loc='upper right', handlelength=1.4)
    ax.spines[['top', 'right']].set_visible(False)

    if args.draft_title:
        fig.suptitle(f'Observation extremes, p={args.prob:.2f}',
                     fontsize=fp + 1, fontweight='bold')
    stem = save(fig, output, args.dpi, args.width,
                sources=[str(truth_dir / args.pattern)], n_points=N.size,
                submit_as=args.submit_as)

    # ---------------- traceable numbers ----------------
    vp = Path(str(stem) + '_values.csv')
    with open(vp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['world_stem', 'n_species_present', 'mean_range_N',
                    'mean_records_proportional', 'zero_record_rate_pure_bernoulli',
                    'truth_key', 'cells_in_disputed_band'])
        w.writerows(per_world_rows)
        w.writerow([])
        w.writerow(['POOLED', N.size, f"{N.mean():.3f}", f"{prop.mean():.3f}",
                    f"{zero_bern:.4f}", 'n/a', disputed_total])
        w.writerow(['STRUCTURE', f"median_range={np.median(N):.0f}",
                    f"max_range={N.max()}", f"frac_range_le_K={degenerate:.4f}",
                    f"frac_range_ge_floor({floor})={above_floor:.4f}",
                    f"frac_forced_to_min_obs={topped:.4f}", ''])
        w.writerow(['SETTINGS', f"p={args.prob}", f"min_obs={args.min_obs}",
                    f"K={K}", f"rng_seed={args.rng_seed}",
                    f"per_world_seed={int(args.per_world_seed)}",
                    f"threshold>{args.truth_threshold}; analysed_min={floor}"])
    WRITTEN.append(vp)
    print(f"  saved -> {vp}  ({len(per_world_rows)} worlds + pooled + settings)")

    print(f"\n  CAPTION\n"
          f"  Fig. Z. Two observation schemes over {ok_worlds} communities and "
          f"{N.size} species. (a) records per species against true range size: a "
          f"fixed budget gives every species min(K, N) records regardless of range "
          f"size, while proportional recording gives on average p·N (dashed "
          f"line), so narrow-ranging species receive few records and wide-ranging "
          f"species many. Points are jittered vertically by up to 0.18 because both "
          f"counts are integers. The shaded area (ranges of {floor - 1} cells or "
          f"fewer) is excluded from the coverage analysis, which draws its species "
          f"from those occupying more than {floor - 1} cells. (b) the resulting "
          f"distributions of record counts, log scale. At p = {args.prob:.2f} the "
          f"median species receives {np.median(prop):.0f} record(s) and a pure "
          f"Bernoulli process would leave {zero_bern:.0%} with none; the sampler tops "
          f"these up to {args.min_obs} so the inpainting always has a seed, so the "
          f"zero-record case is not tested.\n")

    print('  MANIFEST')
    for p in WRITTEN:
        print(f"    {p}  {p.stat().st_size:>9,d} B")


if __name__ == '__main__':
    main()