#!/usr/bin/env python3
"""
pick_inference_worlds.py — REV2

Selects inference worlds from BOTH sweeps:
  - batchA (sweep_worlds.py)         — main sweep, 240 worlds
  - batchB_highmix (sweep_worlds_highmix.py) — extra 24 high-mixing worlds

Targets Axel's transcript concerns:
  - Hard band (range ≥ 21): low ls, high vr, high dr      [from batchA only]
  - Connectivity (Axel 26:00): high ld                   [batchB has more]
  - Generalisation across regimes: varied ls, vr, thr    [both batches]
  - Sparse-obs robustness: thr=5                          [batchA]

REV2 fixes:
  (1) Searches case-insensitively for batchA/batchB tags
  (2) Adds 12 new batchB_highmix worlds (LDD-rich connectivity tests)
  (3) Reports per-axis coverage so you see WHICH Axel concerns are covered
  (4) Distinguishes "missing from disk" vs "not in either sweep"

USAGE
-----
    python pick_inference_worlds.py \\
        --truth-dir   ./results/data \\
        --manifest    ./figures_map_axel_stage2_new/inference_worlds_manifest_REV2.csv
"""

import argparse
import csv
import re
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
#  Filename formatting (matches run_all_rps.py output convention)
# ──────────────────────────────────────────────────────────────────────

def fmt_p(value):
    return f"{value}".replace(".", "p")


def fmt_dr(value):
    s = f"{value:.0e}"                          # '2e-08'
    s = s.replace("-0", "-").replace("e-", "em")
    m = re.match(r"(\d+)em(\d+)", s)
    if m:
        return f"{m.group(1)}em{int(m.group(2)):02d}"
    return s


def candidate_names(batch_tag, ls, vr, thr, env, dr, ld):
    """Return both upper- and lower-case candidate filenames."""
    body = (f"_{batch_tag}_ls{fmt_p(ls)}_vr{fmt_p(vr)}_"
            f"thr{fmt_p(thr)}_env{env}_grid20x20_"
            f"dr{fmt_dr(dr)}_ld{fmt_p(ld)}_training.npz")
    return [
        f"pool22510000{body.lower()}",
        f"pool22510000{body}",
    ]


def find_world_on_disk(truth_dir, candidates):
    """Return the first existing path among candidates, else None."""
    for c in candidates:
        p = truth_dir / c
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────────────────
#  Inference world plan
# ──────────────────────────────────────────────────────────────────────
#
# Each entry: (label, batch, ls, vr, thr, env, dr, ld, axel_concern)
# batch: 'A' = sweep_worlds.py        (batchA)
#        'B' = sweep_worlds_highmix.py (batchB_highmix)

WORLDS = [
    # ── 10 ORIGINAL WORLDS (batchA) ─────────────────────────────
    ("existing_01",  'A', 10.0, 0.001, 1.0, 123, 2e-08, 0.06, "baseline"),
    ("existing_02",  'A', 10.0, 0.001, 1.0, 123, 5e-08, 0.06, "baseline"),
    ("existing_03",  'A', 10.0, 0.001, 1.0, 456, 2e-08, 0.06, "baseline"),
    ("existing_04",  'A', 10.0, 0.001, 1.0, 456, 5e-08, 0.06, "baseline"),
    ("existing_05",  'A', 10.0, 0.001, 3.0, 123, 2e-08, 0.06, "baseline"),
    ("existing_06",  'A', 10.0, 0.001, 3.0, 123, 5e-08, 0.06, "baseline"),
    ("existing_07",  'A', 10.0, 0.001, 3.0, 456, 2e-08, 0.06, "baseline"),
    ("existing_08",  'A', 10.0, 0.001, 3.0, 456, 2e-08, 0.0,  "baseline"),
    ("existing_09",  'A', 10.0, 0.001, 3.0, 456, 5e-08, 0.06, "baseline"),
    ("existing_10",  'A', 10.0, 0.001, 3.0, 456, 5e-08, 0.0,  "baseline"),

    # ── ROBUSTNESS / GENERALISATION (batchA, exists) ─────────────
    ("sparse_obs_01",   'A', 10.0, 0.001, 5.0, 123, 2e-08, 0.06,
        "sparse-obs robustness (thr=5)"),
    ("mid_regime_01",   'A',  5.0, 0.002, 1.0, 123, 5e-08, 0.06,
        "generalisation (ls=5, vr=0.002)"),
    ("mid_regime_02",   'A',  5.0, 0.002, 3.0, 456, 5e-08, 0.06,
        "generalisation (ls=5, vr=0.002)"),

    # ── HARD BAND / WIDE-RANGE (batchA only — REQUIRES NEW SIM) ──
    ("hardband_01",     'A',  2.5, 0.004, 1.0, 123, 1e-07, 0.12,
        "hard band power (range >= 21)"),
    ("hardband_02",     'A',  2.5, 0.004, 1.0, 456, 1e-07, 0.12,
        "hard band power (range >= 21)"),
    ("wide_range_01",   'A',  2.5, 0.004, 3.0, 456, 1e-07, 0.20,
        "widest-range stress test"),

    # ── CONNECTIVITY high-LDD (batchA only — REQUIRES NEW SIM) ───
    ("connectivity_A1", 'A', 10.0, 0.001, 1.0, 123, 5e-08, 0.20,
        "multi-patch (Axel 26:00), thr=1"),
    ("connectivity_A2", 'A', 10.0, 0.001, 1.0, 456, 5e-08, 0.20,
        "multi-patch (Axel 26:00), thr=1"),

    # ── CONNECTIVITY high-LDD (batchB_highmix — already exists) ──
    # ls=5, vr=0.002, thr=3 with ld=0.06/0.12/0.2 × envs × drs
    ("connectivity_B1", 'B',  5.0, 0.002, 3.0, 123, 5e-08, 0.20,
        "multi-patch high-mix (batchB)"),
    ("connectivity_B2", 'B',  5.0, 0.002, 3.0, 456, 5e-08, 0.20,
        "multi-patch high-mix (batchB)"),
    ("connectivity_B3", 'B',  5.0, 0.002, 3.0, 123, 1e-07, 0.20,
        "multi-patch high-mix, faster dispersal"),
    ("connectivity_B4", 'B',  5.0, 0.002, 3.0, 456, 1e-07, 0.20,
        "multi-patch high-mix, faster dispersal"),
    ("connectivity_B5", 'B', 10.0, 0.002, 3.0, 123, 5e-08, 0.20,
        "multi-patch ls=10 mid-vr"),
    ("connectivity_B6", 'B', 10.0, 0.002, 3.0, 456, 5e-08, 0.20,
        "multi-patch ls=10 mid-vr"),
    ("connectivity_B7", 'B',  5.0, 0.002, 3.0, 123, 5e-08, 0.12,
        "multi-patch ld=0.12"),
    ("connectivity_B8", 'B',  5.0, 0.002, 3.0, 456, 5e-08, 0.12,
        "multi-patch ld=0.12"),
    ("connectivity_B9", 'B', 10.0, 0.002, 3.0, 123, 1e-07, 0.12,
        "ls=10 high-dr ld=0.12"),
    ("connectivity_B10", 'B', 10.0, 0.002, 3.0, 456, 1e-07, 0.12,
        "ls=10 high-dr ld=0.12"),
    ("connectivity_B11", 'B', 10.0, 0.002, 3.0, 123, 1e-07, 0.20,
        "ls=10 high-dr ld=0.20"),
    ("connectivity_B12", 'B', 10.0, 0.002, 3.0, 456, 1e-07, 0.20,
        "ls=10 high-dr ld=0.20"),
]


# ──────────────────────────────────────────────────────────────────────
#  Coverage report
# ──────────────────────────────────────────────────────────────────────

def axel_coverage(found):
    """How many worlds cover each Axel concern?"""
    concerns = {
        "hard band power (range >= 21)":    0,
        "multi-patch (connectivity)":       0,
        "sparse-obs robustness (thr=5)":    0,
        "generalisation (mid ls/vr)":       0,
        "baseline":                         0,
        "widest-range stress":              0,
    }
    for _, _, concern in found:
        cl = concern.lower()
        if "hard band" in cl or "range >= 21" in cl:
            concerns["hard band power (range >= 21)"] += 1
        elif "multi-patch" in cl or "connectivity" in cl or "ld=0.20" in cl or "ld=0.12" in cl:
            concerns["multi-patch (connectivity)"] += 1
        elif "sparse-obs" in cl or "thr=5" in cl:
            concerns["sparse-obs robustness (thr=5)"] += 1
        elif "generalisation" in cl or "mid" in cl:
            concerns["generalisation (mid ls/vr)"] += 1
        elif "widest" in cl or "stress" in cl:
            concerns["widest-range stress"] += 1
        elif "baseline" in cl:
            concerns["baseline"] += 1
    return concerns


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)

    print(f"\n{'='*82}")
    print(f"  INFERENCE WORLD SELECTION  (REV2 — handles batchA AND batchB_highmix)")
    print(f"{'='*82}")
    print(f"  Truth dir: {truth_dir.resolve()}\n")

    found, missing = [], []
    rows_for_csv = []

    for entry in WORLDS:
        if len(entry) == 9:
            label, batch, ls, vr, thr, env, dr, ld, concern = entry
        else:
            print(f"  ⚠ malformed entry: {entry}"); continue

        batch_tag_lower = "batcha" if batch == 'A' else "batchb_highmix"
        batch_tag_other = "batchA" if batch == 'A' else "batchB_highmix"

        candidates = (
            candidate_names(batch_tag_lower, ls, vr, thr, env, dr, ld) +
            candidate_names(batch_tag_other, ls, vr, thr, env, dr, ld)
        )
        fpath = find_world_on_disk(truth_dir, candidates)

        if fpath is not None:
            found.append((label, fpath.name, concern))
            rows_for_csv.append({
                "label": label, "batch": batch, "filename": fpath.name,
                "concern": concern, "ls": ls, "vr": vr, "thr": thr,
                "env": env, "dr": dr, "ld": ld, "found": "yes",
            })
            print(f"  ✓ {label:<20s} [{batch}]  {concern:<40s}  {fpath.name}")
        else:
            params = f"ls={ls},vr={vr},thr={thr},env={env},dr={dr:.0e},ld={ld}"
            missing.append((label, batch, params, concern))
            rows_for_csv.append({
                "label": label, "batch": batch, "filename": "",
                "concern": concern, "ls": ls, "vr": vr, "thr": thr,
                "env": env, "dr": dr, "ld": ld, "found": "no",
            })
            print(f"  ✗ {label:<20s} [{batch}]  MISSING ({params})")

    print(f"\n  Found: {len(found)} / {len(WORLDS)}\n")

    # Coverage report
    coverage = axel_coverage(found)
    print(f"  AXEL CONCERN COVERAGE (found worlds only):")
    for k, v in coverage.items():
        flag = "" if v >= 2 else "  ⚠ low coverage"
        print(f"    {k:<40s}  n = {v}{flag}")

    # Missing list
    if missing:
        print(f"\n  MISSING WORLDS — regenerate with sweep_worlds_extra.py:")
        print(f"  ----------------------------------------------------------------------")
        for label, batch, params, concern in missing:
            print(f"    {label:<20s} [{batch}]  {concern}")
            print(f"      {params}")

    if args.manifest:
        out = Path(args.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = ["label", "batch", "filename", "concern",
                "ls", "vr", "thr", "env", "dr", "ld", "found"]
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_for_csv:
                w.writerow(r)
        print(f"\n  ✓ Manifest: {out}")

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())