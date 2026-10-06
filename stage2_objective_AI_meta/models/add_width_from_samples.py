#!/usr/bin/env python3
"""Add ensemble-width columns (patch_width, spread_width) to the Figure 6 PIT
CSVs for EXACTLY the species already in them, so no selection settings are
needed. For every world it finds the reconstruction file whose samples
reproduce the CSV's range_N, n_obs and both percentiles, and only then
computes the 5-95% widths. Binarisation and shape statistics are copied
verbatim from posterior_per_species.py.

  python3 add_width_from_samples.py --root . --pit A.csv B.csv C.csv
writes A_width.csv, B_width.csv, C_width.csv next to the inputs.

  python3 add_width_from_samples.py --root . --pit A.csv B.csv C.csv --rules
writes nothing. For every community it reruns pick_species (copied verbatim)
under each rule and reports which rule, and which seed, returns exactly the
species in the CSV, in the CSV's order. If one file used the random rule, it
prints the commands that re-score the other files with that rule and seed, so
every scheme gets the same species.
"""
import argparse, csv, sys, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage

GRID_Y = GRID_X = 20
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(2, 1)


# ---------------- verbatim from posterior_per_species.py ----------------
def periodic_cov_det(binary_range, Y=GRID_Y, X=GRID_X):
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
    s = np.asarray(sample_vals, float)
    if s.size == 0:
        return float('nan')
    return float((np.sum(s < truth_val) + 0.5 * np.sum(s == truth_val)) / s.size)


def sample_binaries_topN(samples_sp, n_target, smooth_sigma=0.0):
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


def pick_species(truth, mean_pred, K, n_species, mode='topwide', seed=0):
    """Choose species to display.
      'rare'    = narrowest (range-restricted) species that still have hidden
                  cells to predict (range >= ~1.5K and some predicted mass);
                  the conservation-relevant case and this study's focus.
      'topwide' = widest / highest-confidence (stable shape stats, biased wide).
      'random'  = representative random draw across the range-size distribution
                  (deterministic given seed); the honest sample for calibration.
    All modes require range > K so at least one occupied cell is hidden."""
    ranges = truth.sum(axis=(1, 2)).astype(int)

    def conf(s):
        flat = mean_pred[s].ravel()
        if ranges[s] <= 0 or flat.max() < 1e-6:
            return 0.0
        n = min(ranges[s], flat.size)
        return float(np.partition(flat, -n)[-n:].sum())

    if mode == 'random':
        eligible = [s for s in range(truth.shape[0]) if ranges[s] > K]
        rng = np.random.default_rng(seed)
        if len(eligible) <= n_species:
            return eligible
        return sorted(rng.choice(eligible, size=n_species, replace=False).tolist())

    if mode == 'rare':
        floor = max(K + 3, int(np.ceil(1.5 * K)))     # enough hidden cells to matter
        elig = [(s, ranges[s], conf(s)) for s in range(truth.shape[0])
                if ranges[s] >= floor and conf(s) > 0]
        if len(elig) < n_species:                     # relax if the world is sparse
            elig = [(s, ranges[s], conf(s)) for s in range(truth.shape[0])
                    if ranges[s] > K]
        elig.sort(key=lambda c: (c[1], -c[2]))        # narrowest first, then confident
        return [c[0] for c in elig[:n_species]]

    # 'topwide'
    cands = [(s, ranges[s], conf(s)) for s in range(truth.shape[0])
             if ranges[s] >= max(K + 5, 6)]
    if len(cands) < n_species:
        cands = [(s, ranges[s], conf(s)) for s in range(truth.shape[0])
                 if ranges[s] > K]
    cands.sort(key=lambda c: (-c[2], -c[1]))
    return [c[0] for c in cands[:n_species]]
# ------------------------------------------------------------------------


def mean_pred_of(samples_path, n):
    """As posterior_per_species.load_world: the stored 'mean' if present, else the sample mean."""
    z = np.load(samples_path)
    m = (np.asarray(z['mean']).astype(np.float32) if 'mean' in z.files
         else np.asarray(z['samples']).astype(np.float32).mean(axis=0))
    return m[:n]


def rules_for(order, truth, mean_pred, K, nsp, n_seeds):
    """Rules whose pick_species output equals `order` (the CSV's species, in CSV order).
    Seeds are searched with the same two lines pick_species uses for 'random', and every
    seed found is confirmed with the verbatim pick_species."""
    hits = [m for m in ('rare', 'topwide') if pick_species(truth, mean_pred, K, nsp, m) == order]
    ranges = truth.sum(axis=(1, 2)).astype(int)
    eligible = [s for s in range(truth.shape[0]) if ranges[s] > K]
    if len(eligible) <= nsp:
        return hits, ('any' if eligible == order else set())
    seeds = {s for s in range(n_seeds)
             if sorted(np.random.default_rng(s).choice(eligible, size=nsp, replace=False).tolist()) == order}
    return hits, {s for s in seeds if pick_species(truth, mean_pred, K, nsp, 'random', s) == order}


def shape_stats(truth_sp, samples_sp, smooth):
    b = sample_binaries_topN(samples_sp, int(truth_sp.sum()), smooth_sigma=smooth)
    patches = np.array([count_components(b[k]) for k in range(b.shape[0])], float)
    spread = np.array([np.log10(periodic_cov_det(b[k]) + 1.0) for k in range(b.shape[0])], float)
    pt = float(count_components(truth_sp > 0))
    st = float(np.log10(periodic_cov_det(truth_sp > 0) + 1.0))
    return patches, spread, pit_percentile(patches, pt), pit_percentile(spread, st)


def n_members(path):
    """First dimension of 'samples' read from the npz header, without loading it."""
    try:
        with zipfile.ZipFile(path) as zf, zf.open("samples.npy") as f:
            v = np.lib.format.read_magic(f)
            rd = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(v)
            return rd(f)[0][0] if rd else None
    except (KeyError, zipfile.BadZipFile, OSError, ValueError, IndexError):
        return None


def load(truth_path, samples_path):
    """Same arrays and thresholds as posterior_per_species.load_world."""
    with np.load(truth_path, allow_pickle=True) as td:
        truth = (np.asarray(td['P_last_final']) > 0.5).astype(np.uint8)
    z = np.load(samples_path)
    samples = np.asarray(z['samples']).astype(np.float32)
    if 'noisy_input' in z.files:
        observed = (np.asarray(z['noisy_input']) > 0.5).astype(np.uint8)
    elif 'obs_mask' in z.files:
        observed = np.asarray(z['obs_mask']).astype(np.uint8)
    else:
        observed = (samples.mean(axis=0) >= 0.99).astype(np.uint8)
    n = min(truth.shape[0], samples.shape[1], observed.shape[0])
    return truth[:n], samples[:, :n], observed[:n]


def reproduces(rows, truth, samples, observed, smooth):
    """True only if range_N, n_obs and both percentiles match for EVERY row."""
    out = []
    for r in rows:
        sp = int(r['species'])
        if sp >= truth.shape[0] or int(truth[sp].sum()) != int(float(r['range_N'])) \
                or int(observed[sp].sum()) != int(float(r['n_obs'])):
            return None
        patches, spread, pp, spp = shape_stats(truth[sp], samples[:, sp], smooth)
        if abs(round(pp, 3) - float(r['patches_pctile'])) > 5e-4 or \
           abs(round(spp, 3) - float(r['spread_pctile'])) > 5e-4:
            return None
        out.append((np.quantile(patches, 0.95) - np.quantile(patches, 0.05),
                    np.quantile(spread, 0.95) - np.quantile(spread, 0.05)))
    return out


def rules_summary(report, root):
    """One rule per CSV, then the commands that put every CSV on the random rule
    and seed of the file that used it (same K and species count, so same species)."""
    print("\nSELECTION RULE PER FILE")
    rule = {}
    for pit, worlds in report.items():
        n = len(worlds)
        modes = set.intersection(*(set(v['modes']) for v in worlds.values()))
        fixed = [v['seeds'] for v in worlds.values() if v['seeds'] != 'any']
        seeds = sorted(set.intersection(*fixed)) if fixed else [0]   # 'any' everywhere: every seed works
        if not seeds:
            label = []
        elif fixed:
            label = [f"random, seed {seeds[0]}" + (f" (also {seeds[1:]})" if len(seeds) > 1 else "")]
        else:
            label = ["random, any seed (no community has more species than requested)"]
        parts = sorted(modes) + label
        rule[pit] = (sorted(modes), seeds)
        print(f"  {pit}: {' / '.join(parts) if parts else f'NO rule reproduces all {n} communities'}")
    ref = [p for p, (m, sd) in rule.items() if sd and not m]
    if not ref:
        print("\nNo file is on the random rule, so there is no seed to copy. Nothing to re-score from.")
        return
    seed = rule[ref[0]][1][0]
    todo = [p for p, (m, sd) in rule.items() if seed not in sd]
    if not todo:
        print(f"\nAll files already use the random rule with seed {seed}: same species in every scheme.")
        return
    pps = Path(sys.argv[0]).with_name('posterior_per_species.py')
    print(f"\nRE-SCORE on the random rule, seed {seed}, as in {ref[0]}")
    if not pps.exists():
        print(f"  ! {pps} not found: replace that path with the posterior_per_species.py you used for Figure 6")
    news = []
    base = lambda q: Path(q).stem[:-6] if Path(q).stem.endswith('_width') else Path(q).stem
    for pit in todo:
        new = Path(pit).with_name(base(pit) + f'_random_s{seed}.csv')
        news.append(new)
        print(f"  rm -f '{new}'")
        for w, v in sorted(report[pit].items()):
            c, i = v['samples'], v['samples'].parts.index(w)
            pattern = Path(*v['samples'].parts[:i], '{world_stem}', *v['samples'].parts[i + 1:-1])
            print(f"  python3 '{pps}' --truth-dir '{v['truth'].parent}' --recon-dir-pattern '{pattern}' "
                  f"--recon-filename '{c.name}' --world-stem '{w}' --K {v['K']} --n-species {v['nsp']} "
                  f"--select random --select-seed {seed} --csv-only --csv-path '{new}'")
    me, wbr = Path(sys.argv[0]), Path(sys.argv[0]).with_name('width_by_records.py')
    refw = Path(ref[0]).with_name(base(ref[0]) + '_width.csv')
    print(f"\nThen widths, and the check that every scheme now holds the same species:")
    print(f"  python3 '{me}' --root '{root}' --pit " + " ".join(f"'{x}'" for x in news))
    print(f"  python3 '{wbr}' --describe " + " ".join(f"'{x.with_name(x.stem + '_width.csv')}'" for x in news)
          + f" '{refw}'")
    print("  verification: 'same species in every file: yes'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='folder searched for truth and *_samples.npz files')
    ap.add_argument('--pit', nargs='+', required=True, help='the Figure 6 PIT CSVs')
    ap.add_argument('--smooth', type=float, default=0.0, help='must match the run that made the CSVs')
    ap.add_argument('--rules', action='store_true', help='report the selection rule of each CSV; write nothing')
    ap.add_argument('--seeds', type=int, default=1000, help='random seeds tried with --rules (0 .. N-1)')
    a = ap.parse_args()

    print('indexing .npz files under', a.root, '...')
    npz = list(Path(a.root).rglob('*.npz'))
    outs, ok_all, report = [], True, {}
    for pit in a.pit:
        with open(pit, newline='') as f:
            rows = list(csv.DictReader(f)); fields = list(rows[0].keys()) if rows else []
        by_world = defaultdict(list)
        for r in rows:
            by_world[r['world_stem']].append(r)
        print(f"\n{pit}: {len(rows)} rows, {len(by_world)} worlds")
        widths = {}
        for w, wr in sorted(by_world.items()):
            n_ens = int(float(wr[0]['n_ens']))
            truths = [p for p in npz if p.name == f'{w}.npz']
            cands = [p for p in npz if p.name.endswith('_samples.npz') and w in p.parts
                     and n_members(p) in (n_ens, None)]
            hit = None
            for t in truths:
                for c in cands:
                    try:
                        res = reproduces(wr, *load(t, c), a.smooth)
                    except (KeyError, OSError, ValueError):
                        continue
                    if res is not None:
                        hit = (c, res, t); break
                if hit:
                    break
            if not hit:
                ok_all = False
                print(f"  MISS  {w[:60]}  ({len(truths)} truth file(s), {len(cands)} candidate sample file(s))")
                continue
            print(f"  OK    {w[:60]}  <- {hit[0]}")
            for r, wd in zip(wr, hit[1]):
                widths[(w, r['species'])] = wd
            if a.rules:
                truth = load(hit[2], hit[0])[0]
                K = int(float(wr[0]['K']))
                nsp = max(len(v) for v in by_world.values())
                modes, seeds = rules_for([int(r['species']) for r in wr], truth,
                                         mean_pred_of(hit[0], truth.shape[0]), K, nsp, a.seeds)
                report.setdefault(pit, {})[w] = dict(modes=modes, seeds=seeds, K=K, nsp=nsp,
                                                     truth=hit[2], samples=hit[0])
                print(f"        rule: {', '.join(modes) or '-'}; random seeds: "
                      f"{seeds if seeds == 'any' else (sorted(seeds) or '-')}")
        if a.rules or len(widths) != len(rows):
            continue
        out = Path(pit).with_name(Path(pit).stem + '_width.csv')
        with open(out, 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=fields + ['patch_width', 'spread_width'])
            wtr.writeheader()
            for r in rows:
                pw, sw = widths[(r['world_stem'], r['species'])]
                wtr.writerow({**r, 'patch_width': round(float(pw), 3), 'spread_width': round(float(sw), 3)})
        outs.append(out)
        print(f"  wrote {out}")
    if not ok_all:
        sys.exit("\nVERIFY FAILED: some worlds have no reconstruction file that reproduces the CSV. "
                 "Point --root at the folder that holds the reconstructions, or pass --smooth if the CSV used it.")
    print(f"\nVERIFY OK: every row's range_N, n_obs and both percentiles were reproduced from the samples.")
    if a.rules:
        return rules_summary(report, a.root)
    print("Next, run:\n  python3 " + str(Path(sys.argv[0]).with_name('width_by_records.py')) + " --csv " +
          " ".join(f'"{o}"' for o in outs))


if __name__ == '__main__':
    main()