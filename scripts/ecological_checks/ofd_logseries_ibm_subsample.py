#!/usr/bin/env python
"""
ofd_logseries_ibm_subsample.py

Like ofd_logseries_ibm.py, but with an extra step:

  * before computing occupancies, we randomly sample only a fraction of
    the patches (sites) on the Y×X grid for each world, to mimic empirical
    sampling of only some locations.

This lets us test Axel's idea that nearly log-series OFDs might arise
from connected ranges when only a subset of patches is sampled.

For each world:
  - load the final presence array P (S, Y, X),
  - flatten patches to a 1D list of sites,
  - sample k sites uniformly at random (k = sample_frac * Y*X),
  - compute occupancies n_s only on these sampled sites,
  - aggregate occupancies across worlds (and optional replicates),
  - build the OFD and fit the LSPOM log-series:

        X_n(m) = (Z / (n m)) * (m / (m + 1))**n,

    as in Axel & Jacob's paper.

The output is a Fig.3(c)-style plot and an .npz file summarising the OFD,
the fitted m, and the sampling settings.

Usage example
-------------

# full grid (for baseline comparison)
python scripts/ecological_checks/ofd_logseries_ibm_subsample.py \
  "results/data/pool22510000*ld0p06*_training.npz" \
  --sample-frac 1.0 \
  --max-n-main 30 \
  --out results/figures_ibm_like/ofd_ibm_ld0p06_subsample_f1p0.png

# sample only 10% of patches, 10 independent subsampling replicates
python scripts/ecological_checks/ofd_logseries_ibm_subsample.py \
  "results/data/pool22510000*ld0p06*_training.npz" \
  --sample-frac 0.10 \
  --n-reps 10 \
  --max-n-main 30 \
  --out results/figures_ibm_like/ofd_ibm_ld0p06_subsample_f0p10.png
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# helpers: loading presence and computing occupancies with sub-sampling
# ---------------------------------------------------------------------


def _load_presence(path, p_key="P_last_final", b_key="B_last", thr=None):
    """
    Return presence array P with shape (S, Y, X), dtype=bool, for one world.

    Prefers P_last_final (already thresholded); falls back to
    B_last >= thr * BODY_MASS if needed (as in ofd_logseries_ibm.py).
    """
    with np.load(path) as data:
        if p_key is not None and p_key in data.files:
            P = data[p_key].astype(bool)  # (S, Y, X)
        else:
            # Fallback: threshold biomass to get presence
            if b_key not in data.files:
                raise KeyError(f"{path}: neither {p_key} nor {b_key} present in file")

            B = data[b_key]  # (S, Y, X)
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
                # CLI thr is interpreted directly in biomass units
                thr_mass = float(thr)

            P = (B >= thr_mass)

    if P.ndim != 3:
        raise ValueError(f"{path}: presence array has shape {P.shape}, "
                         f"expected (S, Y, X)")

    return P.astype(bool)


def _load_occupancies_subsample(
    path,
    p_key="P_last_final",
    b_key="B_last",
    thr=None,
    sample_frac=1.0,
    rng=None,
):
    """
    Return per-species occupancies n_s for one world, *after* randomly
    sampling only a fraction of patches.

        n_s = number of sampled sites where species s is present.

    Parameters
    ----------
    sample_frac : float in (0, 1]
        Fraction of all Y*X patches to include in the sample. 1.0 means
        use all patches (no subsampling), which reproduces the behaviour
        of the original ofd_logseries_ibm.py.

    rng : np.random.Generator
        Random number generator used to choose sampled sites when
        sample_frac < 1.0. Must not be None in that case.
    """
    P = _load_presence(path, p_key=p_key, b_key=b_key, thr=thr)  # (S, Y, X)
    S, Y, X = P.shape

    P_flat = P.reshape(S, -1)          # (S, Y*X)
    n_sites_total = P_flat.shape[1]

    if sample_frac is None or sample_frac >= 1.0:
        # no subsampling: use all patches
        occ = P_flat.sum(axis=1).astype(np.int64)
    else:
        if sample_frac <= 0.0:
            raise ValueError("sample_frac must be > 0")
        if rng is None:
            raise ValueError("rng must be provided when sample_frac < 1.0")

        n_sample = max(1, int(round(sample_frac * n_sites_total)))
        n_sample = min(n_sample, n_sites_total)

        idx = rng.choice(n_sites_total, size=n_sample, replace=False)
        occ = P_flat[:, idx].sum(axis=1).astype(np.int64)

    # drop species that are nowhere present in the sampled sites
    occ = occ[occ > 0]
    return occ


# ---------------------------------------------------------------------
# build OFD + fit log-series (adapted from ofd_logseries_ibm.py)
# ---------------------------------------------------------------------


def build_ofd(occ_list):
    """
    From a list of occupancy arrays (one per world or per world×rep),
    build:

      - n_vals : 1..n_max
      - X_mean : mean OFD across occ_list
      - X_sem  : SEM across occ_list
      - X_full_per_world : (W, n_max) OFD for each element in occ_list
    """
    if not occ_list:
        raise ValueError("No occupancy data provided")

    # Filter out worlds where no species are present
    occ_list = [o for o in occ_list if o.size > 0]
    if not occ_list:
        raise ValueError("All samples had zero occupancy (no species present)")

    n_max = max(int(o.max()) for o in occ_list)
    W = len(occ_list)

    # X_stack has length n_max+1 (including occupancy=0, which we drop later)
    X_stack = np.zeros((W, n_max + 1), dtype=np.float64)
    for i, occ in enumerate(occ_list):
        x = np.bincount(occ, minlength=n_max + 1)
        X_stack[i, : x.size] = x

    # drop n=0
    X_per_world = X_stack[:, 1:]  # (W, n_max)
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
    classes. Returns (m_hat, Z, X_model_full, log_mse).

    log_mse is the mean squared error in log-space at the optimum m.
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
        return np.mean((np.log(X + 1e-8) - np.log(X_pred + 1e-8)) ** 2)

    # Coarse grid in log10 m
    log_m_grid = np.linspace(-3.0, 2.0, 200)  # 10^-3 ... 10^2
    m_grid = np.power(10.0, log_m_grid)
    losses = np.array([loss(m) for m in m_grid])
    j = int(np.argmin(losses))
    m0 = m_grid[j]

    # Simple local refinement around best grid point (golden-style)
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
    log_mse = float(loss(m_hat))

    return m_hat, Z, X_model_full, log_mse


def make_plot(
    n_vals,
    X_mean,
    X_sem,
    X_model_full,
    m_hat,
    Z,
    max_n_main,
    sample_frac,
    n_reps,
    out_path,
):
    """
    Create a single panel:

      - main: mean OFD ± SEM (points + errorbars) + dashed log-series
      - inset: full range with log-y (X_n + 1e-2), with legend explaining
        IBM vs log-series and showing m.

    The panel also annotates the sampling fraction and number of replicates.
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

    # Annotate sampling settings
    title = f"sample_frac = {sample_frac:.3f}, reps = {n_reps}"
    ax.set_title(title, fontsize=4)

    # ax.legend(frameon=False, fontsize=8, loc="upper right")

    # ---- inset with log y-scale (full range) -------------------------------
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax_in = inset_axes(
            ax,
            width="55%",
            height="55%",
            loc="upper right",
            borderpad=1.2,
        )

        eps = 1e-2
        # IBM OFD
        ax_in.plot(
            n_full,
            X_mean + eps,
            "o",
            markersize=2,
            label="IBM OFD (mean ± SEM)",
        )
        # log-series
        ax_in.plot(
            n_full,
            X_model_full + eps,
            "--",
            linewidth=0.8,
            label=f"Log-series fit (m = {m_hat:.2f})",
        )

        ax_in.set_yscale("log")
        ax_in.set_xlim(0, n_full.max() + 1)

        ax_in.tick_params(axis="both", labelsize=6)
        ax_in.set_xlabel("Sites", fontsize=6)
        ax_in.set_ylabel("Taxa", fontsize=6)
        ax_in.set_title("log-scale", fontsize=7)

        ax_in.legend(
            frameon=False,
            fontsize=6,
            loc="best",
            handlelength=1.2,
            borderpad=0.1,
            labelspacing=0.2,
        )
    except Exception:
        # if inset_axes is unavailable, just skip inset
        pass

    # Manual layout to avoid tight_layout warning from inset
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.96)

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
        sample_frac=np.float64(sample_frac),
        n_reps=np.int64(n_reps),
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Aggregate IBM OFDs across worlds with random patch subsampling "
            "and fit log-series."
        )
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
        help=(
            "Optional biomass threshold (in absolute biomass units) "
            "if P_last_final is not available. If omitted, use "
            "detection_threshold * BODY_MASS from each world."
        ),
    )
    p.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help=(
            "Fraction of patches to sample per world (0 < f <= 1). "
            "f = 1.0 uses all patches (no subsampling)."
        ),
    )
    p.add_argument(
        "--n-reps",
        type=int,
        default=1,
        help=(
            "Number of independent random subsampling replicates per world. "
            "All replicates are combined when building the OFD."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed for subsampling.",
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
        default="results/figures_ibm_like/ofd_ibm_subsample.png",
        help="Output PNG path.",
    )
    args = p.parse_args(argv)

    if args.sample_frac <= 0.0 or args.sample_frac > 1.0:
        raise SystemExit("ERROR: --sample-frac must be in (0, 1].")

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
    print(
        f"[INFO] sample_frac = {args.sample_frac}, "
        f"n_reps = {args.n_reps}, seed = {args.seed}"
    )

    rng = np.random.default_rng(args.seed)

    # load occupancies for all worlds × replicates
    occ_list = []
    for rep in range(args.n_reps):
        if args.n_reps > 1:
            print(f"[INFO] subsampling replicate {rep + 1}/{args.n_reps}")
        for fpath in files:
            try:
                occ = _load_occupancies_subsample(
                    fpath,
                    p_key=args.p_key,
                    b_key=args.b_key,
                    thr=args.detect_threshold,
                    sample_frac=args.sample_frac,
                    rng=rng,
                )
                if occ.size == 0:
                    print(
                        f"[WARN] {fpath} (rep {rep}): "
                        "all species absent at sampled sites, skipped."
                    )
                    continue
                occ_list.append(occ)
            except Exception as e:
                print(f"[WARN] skipping {fpath} (rep {rep}): {e}")

    if not occ_list:
        raise SystemExit("ERROR: no valid occupancies after subsampling.")

    # build OFD and fit log-series
    n_vals, X_mean, X_sem, _ = build_ofd(occ_list)
    m_hat, Z, X_model_full, log_mse = fit_logseries(n_vals, X_mean)

    print(
        "[INFO] fitted mixing rate m ≈ "
        f"{m_hat:.4g}, Z = {Z:.4g}, log-MSE = {log_mse:.3g}"
    )

    make_plot(
        n_vals,
        X_mean,
        X_sem,
        X_model_full,
        m_hat,
        Z,
        args.max_n_main,
        args.sample_frac,
        args.n_reps,
        args.out,
    )
    print(f"[SAVE] figure → {args.out}")
    print(f"[SAVE] OFD data → {Path(args.out).with_suffix('.npz')}")


if __name__ == "__main__":
    main()
