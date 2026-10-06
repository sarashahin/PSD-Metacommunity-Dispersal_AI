#!/usr/bin/env python3
"""Coverage AND ensemble width per record bin from ONE set of PIT CSVs, so the
width paragraph of Section 3.4 uses the same species as Figure 6. Bins and the
coverage rule are copied from calibration_vs_nobs.py. Check values are from
Section 3.4 of the manuscript dated 29 Sept 2026.

  --survey DIR   list every PIT CSV under DIR, say which ones are Figure 6's,
                 and print the --csv command to run next
  --csv FILES    check FILES reproduce Section 3.4, then print the paragraph
  --describe F   coverage and centring by record count and range size, which
                 species-selection rule made each file (from the row order
                 posterior_per_species.py writes), and whether files hold the
                 same species with identical values
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9")]
PUBLISHED = {"1": (23, 43), "2": (39, 56), "3-4": (56, 70), "5-9": (75, 80)}  # pooled, per bin
SCHEMES = {"K=5": (74, 79), "p=0.30": (47, 58), "p=0.10": (35, 57)}            # per scheme
BASE = {"n_obs", "patches_pctile", "spread_pctile"}
WIDTH = {"patch_width", "spread_width"}


def read(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def col(rows, k):
    return np.array([float(r[k]) for r in rows])


def cov(v):
    return 100 * float(np.mean((v >= 0.05) & (v <= 0.95))) if v.size else float("nan")


def bins(rows):
    nob, pp, sp = (col(rows, k) for k in ("n_obs", "patches_pctile", "spread_pctile"))
    out = {}
    for lo, hi, lab in BINS:
        m = (nob >= lo) & (nob <= hi)
        c = (cov(pp[m]), cov(sp[m]))
        out[lab] = (int(m.sum()), c, m.sum() > 0 and tuple(round(x) for x in c) == PUBLISHED[lab], m)
    return out


def survey(root):
    found = []
    for p in sorted(Path(root).rglob("*.csv")):
        try:
            with open(p, newline="") as f:
                head = set(next(csv.reader(f), []))
            if not BASE <= head:
                continue
            rows = read([p])
            c = (round(cov(col(rows, "patches_pctile"))), round(cov(col(rows, "spread_pctile"))))
            pooled = all(b[2] for b in bins(rows).values())
        except (OSError, UnicodeDecodeError, ValueError, KeyError):
            continue
        if not rows:
            continue
        scheme = next((s for s, v in SCHEMES.items() if v == c), None)
        has_w = WIDTH <= head
        found.append(dict(path=p, rows=len(rows), width=has_w, scheme=scheme, pooled=pooled))
        tag = (f"<- Figure 6, {scheme}" if scheme else
               "<- Figure 6, all schemes in one file" if pooled else "")
        print(f"  {p}\n      rows {len(rows)}   width columns {'yes' if has_w else 'NO'}   "
              f"coverage {c[0]}%/{c[1]}%   {tag}")
    if not found:
        sys.exit(f"no PIT CSV (columns {sorted(BASE)}) under {root}")
    pick = [f for f in found if f["pooled"]][:1]
    if not pick:
        by = {s: [f for f in found if f["scheme"] == s] for s in SCHEMES}
        missing = [s for s, v in by.items() if not v]
        if missing:
            sys.exit(f"\nNo set reproduces Section 3.4: nothing matches {', '.join(missing)}. "
                     f"The Figure 6 CSVs are not under {root}.")
        if any(len(v) > 1 for v in by.values()):
            print("\n  ! more than one file matches a scheme; the first of each is used below")
        pick = [by[s][0] for s in SCHEMES]
    if not all(f["width"] for f in pick):
        sys.exit("\nThe Figure 6 files have NO width columns (they predate them). Regenerate "
                 "them with the width-enabled posterior_per_species.py, same settings as the "
                 "Figure 6 runs, writing to a NEW --csv path, then run --survey again.")
    print("\nNext, run:\n  python3 " + sys.argv[0] + " --csv " +
          " ".join(f'"{f["path"]}"' for f in pick))


def describe(paths):
    """Coverage, mean percentile and share of truths below the middle, overall,
    by record count and by range-size third (thirds fixed on the FIRST file, so
    files are compared on the same species groups)."""
    runs = [(p, read([p])) for p in paths]
    keys = [{(r.get("world_stem"), r.get("species")) for r in rows} for _, rows in runs]
    if len(runs) > 1:
        same = all(k == keys[0] for k in keys)
        print(f"same species in every file: {'yes' if same else 'NO (comparison across files is not like for like)'}")
        if not same:
            print(f"species in all files: {len(set.intersection(*keys))} (files hold "
                  f"{', '.join(str(len(k)) for k in keys)})")
        if same:
            print(f"identical range_N, n_obs and both percentiles in every file: "
                  f"{'yes' if all(values(rows) == values(runs[0][1]) for _, rows in runs) else 'no'}")
    rn0 = col(runs[0][1], "range_N")
    e1, e2 = (float(x) for x in np.quantile(rn0, [1 / 3, 2 / 3]))
    print(f"range thirds (from {runs[0][0]}): narrowest <= {e1:g} cells, middle {e1:g}-{e2:g}, widest > {e2:g}")
    print(f"{'':16s} {'n':>5s} {'coverage':>13s} {'mean pctile':>13s} {'below middle':>14s}"
          f"{'   median width' if all(WIDTH <= set(rows[0]) for _, rows in runs if rows) else ''}"
          "   (fragmentation/spread)")
    for p, rows in runs:
        rn = col(rows, "range_N")
        print(f"\n{p}\n  rows {len(rows)}  worlds {len({r.get('world_stem') for r in rows})}  "
              f"n_ens {sorted({r.get('n_ens') for r in rows})}  K {sorted({r.get('K') for r in rows})}  "
              f"range_N min/median/max {rn.min():g}/{np.median(rn):g}/{rn.max():g}")
        print("  " + selection(rows))
        group_lines(rows, e1, e2)
    if len(runs) > 1:
        pooled = [r for _, rows in runs for r in rows]
        print(f"\npooled over the {len(runs)} files ({len(pooled)} species-cases; the record bins of Figure 6)")
        group_lines(pooled, e1, e2)


def group_lines(rows, e1, e2):
    """One line per group: all, record bins (Figure 6 bins plus 10+), range thirds."""
    nob, pp, sp, rn = (col(rows, k) for k in ("n_obs", "patches_pctile", "spread_pctile", "range_N"))
    groups = [("all", np.ones(len(rows), bool))]
    groups += [(f"records {lab}", (nob >= lo) & (nob <= hi)) for lo, hi, lab in BINS + [(10, 10**9, "10+")]]
    groups += [("narrowest third", rn <= e1), ("middle third", (rn > e1) & (rn <= e2)), ("widest third", rn > e2)]
    wd = (col(rows, "patch_width"), col(rows, "spread_width")) if all(WIDTH <= set(r) for r in rows) else None
    for lab, m in groups:
        if not m.any():
            continue
        print(f"  {lab:16s} {int(m.sum()):5d} {cov(pp[m]):5.0f}%/{cov(sp[m]):3.0f}%   "
              f"{pp[m].mean():5.2f}/{sp[m].mean():4.2f}   "
              f"{100 * np.mean(pp[m] < 0.5):5.0f}%/{100 * np.mean(sp[m] < 0.5):3.0f}%"
              + (f"   {np.median(wd[0][m]):6.2f}/{np.median(wd[1][m]):4.2f}" if wd else ""))


def values(rows):
    """(world, species) -> range_N, n_obs and both percentiles, rounded to 3 dp."""
    return {(r["world_stem"], r["species"]): (int(float(r["range_N"])), int(float(r["n_obs"])),
            round(float(r["patches_pctile"]), 3), round(float(r["spread_pctile"]), 3)) for r in rows}


def selection(rows):
    """Which pick_species rule wrote these rows. posterior_per_species.py appends one
    block per community in the order pick_species returns: 'rare' sorts by range
    (narrowest first), 'random' sorts by species index, 'topwide' by confidence."""
    blocks = []
    for r in rows:
        if not blocks or blocks[-1][0] != r["world_stem"]:
            blocks.append((r["world_stem"], []))
        blocks[-1][1].append(r)
    n = len(blocks)
    by_rng = sum(all(int(float(a["range_N"])) <= int(float(b["range_N"])) for a, b in zip(g, g[1:])) for _, g in blocks)
    by_sp = sum(all(int(a["species"]) < int(b["species"]) for a, b in zip(g, g[1:])) for _, g in blocks)
    if by_rng == n and by_sp < n:
        rule = "narrowest-first rule ('rare')"
    elif by_sp == n and by_rng < n:
        rule = "random rule ('random', rows sorted by species index)"
    elif by_sp == n:
        rule = "ambiguous: rows are in both orders, so the order cannot tell"
    else:
        rule = "rule NOT identified (rows re-sorted, or another rule)"
    k = int(float(rows[0]["K"]))
    floor = max(k + 3, int(np.ceil(1.5 * k)))
    low = [min(int(float(r["range_N"])) for r in g) for _, g in blocks]
    out = (f"order within each community: by range in {by_rng}/{n}, by species index in {by_sp}/{n}"
           f" -> {rule}\n  rows per community {','.join(str(len(g)) for _, g in blocks)}; "
           f"smallest range per community {','.join(map(str, low))}")
    if n != len({w for w, _ in blocks}):
        out += "\n  ! a community appears in more than one block (appended twice?)"
    if "rare" in rule:
        out += (f"\n  'rare' floor at K={k} is {floor} cells; communities whose smallest range is below it "
                f"fell back to range > {k}: {sum(x < floor for x in low)}/{n}")
    return out


def main_csv(paths):
    rows = read(paths)
    need = BASE | WIDTH
    if not rows or need - set(rows[0]):
        sys.exit(f"missing columns {sorted(need - set(rows[0] if rows else []))}: regenerate "
                 "the PIT CSVs with the width-enabled posterior_per_species.py")
    pw, sw = col(rows, "patch_width"), col(rows, "spread_width")
    res, ok = {}, True
    print(f"{len(rows)} species-cases from {len(paths)} CSV(s)")
    for lab, (n, c, match, m) in bins(rows).items():
        w = (float(np.median(pw[m])), float(np.median(sw[m]))) if n else (float("nan"),) * 2
        ok &= bool(match)
        res[lab] = (c, w)
        print(f"  {lab:>4}  n={n:4d}  coverage {c[0]:3.0f}%/{c[1]:3.0f}%  median width "
              f"frag {w[0]:.2f} spread {w[1]:.2f}   published {PUBLISHED[lab]}  "
              f"{'OK' if match else 'MISMATCH'}")
    if not ok:
        sys.exit("VERIFY FAILED: coverage does not reproduce Section 3.4, so these are not the "
                 "Figure 6 species. Use --survey to find the right files.")
    print("VERIFY OK: same species as Figure 6.")
    (c1, w1), (c3, w3), (c5, w5) = res["1"], res["3-4"], res["5-9"]
    if not (w3[0] > w1[0] and w5[0] < w3[0] and w5[1] < w3[1]):
        sys.exit("PATTERN CHANGED: width does not rise to 3-4 records and fall at 5-9 on these "
                 "species. The paragraph's claim does not hold; rewrite it, do not paste.")
    print("\n% ---- paste into Section 3.4 ----")
    print(f"Coverage on its own can be bought by making the ensemble wider, so we also measured "
          f"the median $5$--$95\\%$ span of each species' own ensemble. From one record to three or "
          f"four, coverage and width rise together, so part of the early gain comes from "
          f"diffuseness: fragmentation width goes from ${w1[0]:.1f}$ to ${w3[0]:.1f}$ as coverage "
          f"goes from ${c1[0]:.0f}\\%$ to ${c3[0]:.0f}\\%$. From three or four records to five to "
          f"nine the two separate: coverage rises to ${c5[0]:.0f}\\%$ while width falls to "
          f"${w5[0]:.1f}$ for fragmentation and from ${w3[1]:.2f}$ to ${w5[1]:.2f}$ for spread, so "
          f"in that range the ensembles are sharper as well as better centred.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", nargs="+", help="the PIT CSVs behind Figure 6")
    g.add_argument("--survey", metavar="DIR", help="folder to search for PIT CSVs")
    g.add_argument("--describe", nargs="+", metavar="CSV",
                   help="coverage and centring by record count and range size, e.g. the smoothing files")
    a = ap.parse_args()
    if a.survey:
        survey(a.survey)
    elif a.describe:
        describe(a.describe)
    else:
        main_csv(a.csv)