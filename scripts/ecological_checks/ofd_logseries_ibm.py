#!/usr/bin/env python
"""
ofd_logseries_ibm.py

Compute the occupancy frequency distribution (OFD) X_n from IBM worlds and
fit the LSPOM log-series (Box 1, Eq. (1) in Axel & Jacob's paper):

    X_n ≈ (Z / (n m)) * (m / (m + 1))**n ,

where:
  - n      = number of sites occupied
  - X_n    = number of species occupying n sites
  - Z      = total number of local populations (sum over n of n * X_n)
  - m      = mixing rate parameter to be fitted

The script:
  * reads one or many *_training.npz files (IBM worlds),
  * extracts a presence array per world (P_last_final if available),
  * computes occupancies per species,
  * aggregates OFDs across worlds (mean + SEM),
  * fits m by least-squares on log(X_n),
  * and makes a Fig.3(c)-style plot (points + error bars + dashed log-series)
    plus a log-10 inset.

Usage example
-------------

# aggregate across all pool22510000 batchA worlds with ld=0.0
python scripts/ecological_checks/ofd_logseries_ibm.py \
  "results/data/pool22510000_batcha*ld0p0*_training.npz" \
  --max-n-main 30 \
  --out results/figures_ibm_like/ofd_ibm_ld0p0_vs_logseries_allworlds.png
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.size": 8,
#     "axes.labelsize": 9,
#     "axes.titlesize": 9,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 8,
# })



# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _load_occupancies(path, p_key="P_last_final", b_key="B_last", thr=None):
    """
    Return per-species occupancies n_s for one world:

        n_s = number of sites (patches) where species s is present.

    Prefers P_last_final (already thresholded); falls back to
    B_last >= thr * BODY_MASS if needed.
    """
    with np.load(path) as data:
        if p_key is not None and p_key in data.files:
            P = data[p_key].astype(bool)          # (S,Y,X)
        else:
            # Fallback: construct presence from biomass
            if b_key not in data.files:
                raise KeyError(
                    f"{path}: neither {p_key} nor {b_key} present in file"
                )
            B = data[b_key]                      # (S,Y,X)
            if thr is None:
                # Use detection_threshold * BODY_MASS if available
                if ("detection_threshold" in data.files and
                        "BODY_MASS" in data.files):
                    thr_mass = float(data["detection_threshold"]) * float(
                        data["BODY_MASS"]
                    )
                else:
                    # very permissive fallback
                    thr_mass = 0.0
            else:
                # cli thr is interpreted directly in biomass units
                thr_mass = float(thr)
            P = (B >= thr_mass)

    if P.ndim != 3:
        raise ValueError(f"{path}: presence array has shape {P.shape}, expected (S,Y,X)")

    S = P.shape[0]
    occ = P.reshape(S, -1).sum(axis=1).astype(np.int64)  # sites per species
    # drop species that are nowhere present
    occ = occ[occ > 0]
    return occ


def build_ofd(occ_list):
    """
    From a list of occupancy arrays (one per world), build:

      - n_vals : 1..n_max
      - X_mean : mean OFD across worlds
      - X_sem  : SEM across worlds
      - X_full_per_world : (W, n_max) OFD for each world (optional diagnostics)
    """
    if not occ_list:
        raise ValueError("No occupancy data provided")

    # filter out worlds where no species are present
    occ_list = [o for o in occ_list if o.size > 0]
    if not occ_list:
        raise ValueError("All worlds had zero occupancy (no species present)")

    n_max = max(int(o.max()) for o in occ_list)
    W = len(occ_list)

    # X_stack has length n_max+1 (including occupancy=0, which we drop later)
    X_stack = np.zeros((W, n_max + 1), dtype=np.float64)
    for i, occ in enumerate(occ_list):
        x = np.bincount(occ, minlength=n_max + 1)
        X_stack[i, : x.size] = x

    # drop n=0
    X_per_world = X_stack[:, 1:]               # (W, n_max)
    n_vals = np.arange(1, n_max + 1, dtype=np.int64)

    X_mean = X_per_world.mean(axis=0)
    if W > 1:
        X_sem = X_per_world.std(axis=0, ddof=1) / np.sqrt(W)
    else:
        X_sem = np.zeros_like(X_mean)

    return n_vals, X_mean, X_sem, X_per_world


def fit_logseries(n_vals, X_mean):
    """
    Fit the log-series mixing rate m by simple 1D search in log-space:

       X_n(m) = (Z / (n m)) * (m / (m + 1))**n

    using least-squares in log(X_n) to reduce the influence of very common
    classes. Returns (m_hat, Z, X_model_full).
    """
    mask = X_mean > 0
    n = n_vals[mask]
    X = X_mean[mask]

    if n.size < 2:
        raise RuntimeError("Not enough occupied classes to fit log-series")

    Z = float(np.sum(n * X))  # total number of local populations

    def loss(m):
        if m <= 0:
            return np.inf
        r = m / (m + 1.0)
        X_pred = (Z / (n * m)) * np.power(r, n)
        return np.mean(
            (np.log(X + 1e-8) - np.log(X_pred + 1e-8)) ** 2
        )

    # coarse grid in log10 m
    log_m_grid = np.linspace(-3.0, 2.0, 200)  # 10^-3 ... 10^2
    m_grid = np.power(10.0, log_m_grid)
    losses = np.array([loss(m) for m in m_grid])
    j = int(np.argmin(losses))
    m0 = m_grid[j]

    # simple local refinement around best grid point (golden-style)
    m_min = m_grid[max(j - 1, 0)]
    m_max = m_grid[min(j + 1, len(m_grid) - 1)]
    for _ in range(40):
        a, c = m_min, m_max
        m1 = a + (c - a) / 3.0
        m2 = c - (c - a) / 3.0
        l1, l2 = loss(m1), loss(m2)
        if l1 < l2:
            m_max, m0 = m2, m1
        else:
            m_min, m0 = m1, m2
        if (m_max - m_min) / max(m0, 1e-6) < 1e-3:
            break

    m_hat = float(m0)
    r = m_hat / (m_hat + 1.0)
    X_model_full = (Z / (n_vals * m_hat)) * np.power(r, n_vals)

    return m_hat, Z, X_model_full


def make_plot(n_vals, X_mean, X_sem, X_model_full, m_hat, Z,
              max_n_main, out_path):
    """
    Create a single panel:
      - main: mean OFD ± SEM (points + errorbars) + dashed log-series
      - inset: full range with log-y (X_n + 1e-2).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_full = n_vals
    max_n_main = min(max_n_main, int(n_full.max()))

    main_mask = (n_full >= 1) & (n_full <= max_n_main)
    n_main = n_full[main_mask]
    X_main = X_mean[main_mask]
    E_main = X_sem[main_mask]
    X_model_main = X_model_full[main_mask]

    # ---- figure + main axis ------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

    # mean OFD with SEM
    ax.errorbar(
        n_main,
        X_main,
        yerr=E_main,
        fmt="o",
        markersize=3,
        elinewidth=0.8,
        capsize=2,
        label="IBM OFD (mean ± SEM)",
    )

    # dashed log-series fit
    ax.plot(
        n_main,
        X_model_main,
        "--",
        linewidth=1.2,
        label=f"Log-series fit (m = {m_hat:.2f})",
    )

    ax.set_xlim(0, max_n_main + 1)
    ax.set_xlabel("No. sites", fontsize=10)
    ax.set_ylabel("No. taxa", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    # ax.legend(frameon=False, fontsize=8, loc="lower right")

    # ---- inset with log y-scale (full range) -------------------------------
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax_in = inset_axes(
            ax,
            width="55%", height="55%",
            loc="upper right",
            borderpad=1.2,
        )

        eps = 1e-2
        # IBM OFD
        ax_in.plot(n_full, X_mean + eps, "o", markersize=2,
                   label="IBM OFD (mean ± SEM)")
        # log-series
        ax_in.plot(n_full, X_model_full + eps, "--", linewidth=0.8,
                   label=f"Log-series fit (m = {m_hat:.2f})")

        ax_in.set_yscale("log")
        ax_in.set_xlim(0, n_full.max() + 1)

        # light ticks + tiny labels so it’s readable but not dominating
        ax_in.tick_params(axis="both", labelsize=6)
        ax_in.set_xlabel("Sites", fontsize=6)
        ax_in.set_ylabel("Taxa", fontsize=6)
        ax_in.set_title("log-scale", fontsize=7)

        # tiny legend so colours are explained
        ax_in.legend(frameon=False, fontsize=6, loc="best",
                     handlelength=1.2, borderpad=0.1, labelspacing=0.2)
    except Exception:
        # if inset_axes is unavailable, just skip inset
        pass

    # manual layout to avoid tight_layout warning from inset
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.98)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # also save underlying OFD + fit as npz next to the figure
    np.savez_compressed(
        out_path.with_suffix(".npz"),
        n=n_full,
        X_mean=X_mean,
        X_sem=X_sem,
        X_model=X_model_full,
        m_hat=np.float64(m_hat),
        Z=np.float64(Z),
    )



# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Aggregate IBM OFDs across worlds and fit log-series."
    )
    p.add_argument(
        "patterns",
        nargs="+",
        help="One or more glob patterns for *_training.npz files.",
    )
    p.add_argument(
        "--p-key",
        type=str,
        default="P_last_final",
        help="Presence key to use (default: P_last_final).",
    )
    p.add_argument(
        "--b-key",
        type=str,
        default="B_last",
        help="Biomass key if presence array is missing (default: B_last).",
    )
    p.add_argument(
        "--detect-threshold",
        type=float,
        default=None,
        help="Optional biomass threshold (in absolute biomass units) "
             "if P_last_final is not available. If omitted, use "
             "detection_threshold * BODY_MASS from each world.",
    )
    p.add_argument(
        "--max-n-main",
        type=int,
        default=30,
        help="Max occupancy class to show on the main panel (default: 30).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results/figures_ibm_like/ofd_ibm_vs_logseries_allworlds.png",
        help="Output PNG path.",
    )
    args = p.parse_args(argv)

    # expand patterns → list of files
    files = []
    for pat in args.patterns:
        matched = sorted(glob.glob(pat))
        if not matched:
            print(f"[WARN] no files matched pattern: {pat}")
        files.extend(matched)

    if not files:
        raise SystemExit("ERROR: no files matched any pattern.")

    print(f"[INFO] using {len(files)} training worlds")

    # load occupancies
    occ_list = []
    for f in files:
        try:
            occ = _load_occupancies(
                f, p_key=args.p_key, b_key=args.b_key, thr=args.detect_threshold
            )
            if occ.size == 0:
                print(f"[WARN] {f}: all species absent at final time, skipped.")
                continue
            occ_list.append(occ)
        except Exception as e:
            print(f"[WARN] skipping {f}: {e}")

    if not occ_list:
        raise SystemExit("ERROR: no valid worlds after loading occupancies.")

    # build OFD and fit log-series
    n_vals, X_mean, X_sem, _ = build_ofd(occ_list)
    m_hat, Z, X_model_full = fit_logseries(n_vals, X_mean)

    print(f"[INFO] fitted mixing rate m ≈ {m_hat:.4g}, Z = {Z:.4g}")
    make_plot(
        n_vals,
        X_mean,
        X_sem,
        X_model_full,
        m_hat,
        Z,
        args.max_n_main,
        args.out,
    )
    print(f"[SAVE] figure → {args.out}")
    print(f"[SAVE] OFD data → {Path(args.out).with_suffix('.npz')}")


if __name__ == "__main__":
    main()
