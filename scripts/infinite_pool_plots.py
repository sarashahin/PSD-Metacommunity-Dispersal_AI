# make_ibm_like_figures.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    out = {"d": d}
    # optional keys
    for k in ("IBM_B","IBM_t","ASM_round","ASM_attempts","ASM_established","ASM_gamma","ASM_alpha_bar","ASM_alpha_patches",
              "Y","X"):
        out[k] = d[k] if k in d.files else None
    return out

# ---------------- helpers (non-breaking) ----------------
def _scalar_from_npz(dd, key):
    """Best-effort fetch of a scalar (or None) from dd['d']."""
    dfile = dd.get("d")
    try:
        if (dfile is not None) and (key in dfile.files):
            arr = np.asarray(dfile[key])
            return float(arr.ravel()[0])
    except Exception:
        pass
    return None

def _presence_mask(B, dd):
    """
    Presence/absence using detection threshold if available:
    presence if B >= detection_threshold * BODY_MASS, else B > 0.
    """
    body_mass = _scalar_from_npz(dd, "BODY_MASS")
    det_thr   = _scalar_from_npz(dd, "detection_threshold")
    if (body_mass is not None) and (det_thr is not None):
        thr_mass = det_thr * body_mass
        return (B >= thr_mass)
    return (B > 0)

def _presence_mask_anyshape(B, dd):
    """
    Presence/absence using detection threshold if available.
    Works for (S,Y,X) or (T,S,Y,X) arrays; returns a boolean array of same shape.
    """
    body_mass = _scalar_from_npz(dd, "BODY_MASS")
    det_thr   = _scalar_from_npz(dd, "detection_threshold")
    if (body_mass is not None) and (det_thr is not None):
        thr_mass = det_thr * body_mass
        return (B >= thr_mass)
    return (B > 0)


# ---------------- Fig.1a (true assembly history) ----------
def plot_fig1a_from_history(d, outdir: Path):
    # ---- load required series ----
    inv_true = d.get("ASM_established")     # cumulative established (length R)
    gamma    = d.get("ASM_gamma")           # regional richness after each window
    alpha_bar= d.get("ASM_alpha_bar")       # mean local richness after each window
    alpha_p  = d.get("ASM_alpha_patches")   # (R, ≤3) fixed patches

    missing = [n for n,v in (("ASM_established",inv_true),
                             ("ASM_gamma",gamma),
                             ("ASM_alpha_bar",alpha_bar)) if v is None]
    if missing:
        raise RuntimeError(f"Missing history fields: {', '.join(missing)}")

    inv_true = np.asarray(inv_true, dtype=np.int64)
    gamma    = np.asarray(gamma,    dtype=float)
    alpha_bar= np.asarray(alpha_bar,dtype=float)
    alpha_p  = np.asarray(alpha_p) if alpha_p is not None else None

    # ---- collapse to UNIQUE invasion counts (keep the *last* record per x) ----
    order = np.argsort(inv_true, kind="mergesort")
    x_sorted = inv_true[order]
    g_sorted = gamma[order]
    a_sorted = alpha_bar[order]
    ap_sorted = alpha_p[order] if alpha_p is not None else None

    x_unique, counts = np.unique(x_sorted, return_counts=True)
    ends = np.cumsum(counts) - 1

    x = x_sorted[ends]
    g = g_sorted[ends]
    a = a_sorted[ends]
    ap = ap_sorted[ends] if ap_sorted is not None else None

    if x[0] != 0:
        x = np.r_[0, x]
        g = np.r_[g[0], g]
        a = np.r_[a[0], a]
        if ap is not None:
            ap = np.vstack([ap[0], ap])

    print(f"[fig1a] using TRUE invasions: total windows={len(inv_true)}, "
          f"unique invasion steps={len(x)} (max={x.max()})")

    # ---- step plot (paper-like) ----
    plt.figure(figsize=(6.2, 4.6))
    plt.step(x, g, where="post", color="k",  lw=2.0, label="γ")
    plt.step(x, a, where="post", color="C3", lw=2.0, label="ᾱ")
    if ap is not None and ap.ndim == 2:
        for j in range(ap.shape[1]):
            plt.step(x, ap[:, j], where="post", lw=1.0, alpha=0.85, label=f"α{j+1}")

    plt.xlabel("No. invasions (cumulative, established)")
    plt.ylabel("Species richness")
    plt.legend(frameon=False)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(outdir / f"fig1a_richness_over_invasions.{ext}", dpi=300)
    plt.close()

# --------------- Fig.5 (SAD/RSD + co-occurrence) -------------------------
def plot_fig5(B_last, outdir: Path):
    S,Ny,Nx = B_last.shape
    # SAD regional
    reg = B_last.sum((1,2)); reg = reg/reg.sum() if reg.sum()>0 else reg
    reg_vals = np.log10(reg[reg>0] + 1e-16)
    # SAD local (all patches)
    loc_vals = []
    for y in range(Ny):
        for x in range(Nx):
            p = B_last[:,y,x]; tot = p.sum()
            if tot>0:
                q = p/tot
                loc_vals.extend(np.log10(q[q>0] + 1e-16))
    loc_vals = np.asarray(loc_vals)

    # RSD
    rsd = (B_last>0).sum((1,2)).astype(float)/(Ny*Nx)

    # Co-occurrence (simple Pearson + t-test)
    from math import sqrt
    P = Ny*Nx; X = B_last.reshape(S, P)
    pos=neg=nonsig=0
    try:
        from scipy.stats import t as _t
        tcrit = _t.ppf(1-0.05/2, df=P-2) if P>=3 else np.inf
    except Exception:
        tcrit = 12.706 if P-2==1 else 4.303 if P-2==2 else 3.182 if P-2==3 else 2.776
    for i in range(S):
        xi = X[i]-X[i].mean(); sxi = X[i].std()
        if sxi==0: continue
        for j in range(i+1,S):
            xj = X[j]-X[j].mean(); sxj = X[j].std()
            if sxj==0: continue
            r = float((xi*xj).sum()/(P*sxi*sxj))
            df=P-2; sig = (df>=1) and (abs(r)*sqrt(df/max(1e-12,1-r*r))>=tcrit)
            if sig and r>0: pos+=1
            elif sig and r<0: neg+=1
            else: nonsig+=1
    tot = max(1, pos+neg+nonsig)

    # Plot (a,b)
    fig = plt.figure(figsize=(9,7))
    ax1 = fig.add_subplot(2,2,1)
    ax1.hist(loc_vals, bins=20, alpha=0.8)
    ax1.set_title("(a) Species Biomass Distributions")
    ax1.set_xlabel("log10 (proportional biomass)"); ax1.set_ylabel("Frequency")
    ax1b = fig.add_subplot(2,2,3)
    ax1b.hist(reg_vals, bins=20, color="gray", alpha=0.9)
    ax1b.set_xlabel("log10 (proportional biomass)"); ax1b.set_ylabel("Frequency")
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(rsd, bins=np.linspace(0,1,21), edgecolor="black")
    ax2.set_title("(b) Range Size Distributions")
    ax2.set_xlabel("Range size (fraction of patches)"); ax2.set_ylabel("Frequency")
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(outdir / f"fig5ab_like_SAD_RSD.{ext}", dpi=300)
    plt.close(fig)

    # Plot (c)
    fig = plt.figure(figsize=(4.6,4.2))
    ax = fig.add_subplot(1,1,1)
    vals = [100*neg/tot, 100*nonsig/tot, 100*pos/tot]
    ax.bar([0],[vals[0]],label="Negative")
    ax.bar([0],[vals[1]],bottom=[vals[0]],label="Non-sig")
    ax.bar([0],[vals[2]],bottom=[vals[0]+vals[1]],label="Positive")
    ax.set_ylim(0,100); ax.set_xticks([0]); ax.set_xticklabels(["All species"])
    ax.set_ylabel("Percent of pairs"); ax.set_title("(c) Spatial correlation of co-occurrence")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(outdir / f"fig5c_like_cooccurrence.{ext}", dpi=300)
    plt.close(fig)

# ---------------- Fig.6 (SAR) --------------------------------------------
def plot_fig6(B_last, outdir: Path):
    S,Ny,Nx = B_last.shape
    P = Ny*Nx
    pres = (B_last>0).reshape(S, P)
    areas, means, sds = [], [], []
    rng = np.random.default_rng(0)
    for A in range(1, P+1):
        combs = list(combinations(range(P), A))
        if len(combs) > 2000:
            combs = [tuple(sorted(rng.choice(P, size=A, replace=False))) for _ in range(2000)]
        Svals = []
        for idx in combs:
            sub = pres[:, idx]
            Svals.append(int(sub.any(1).sum()))
        areas.append(A); means.append(np.mean(Svals)); sds.append(np.std(Svals, ddof=1) if len(Svals)>1 else 0.0)
    areas, means, sds = np.asarray(areas,float), np.asarray(means,float), np.asarray(sds,float)
    x = np.log10(areas); y = np.log10(np.maximum(means, 1e-9))
    A = np.vstack([x, np.ones_like(x)]).T
    z, a = np.linalg.lstsq(A, y, rcond=None)[0]
    xfit = np.linspace(x.min(), x.max(), 100); yfit = a + z*xfit

    plt.figure(figsize=(6.2,5.0))
    plt.errorbar(x, y, yerr=(sds/(means*np.log(10)+1e-12)), fmt="o", capsize=2)
    plt.plot(xfit, yfit, "--")
    plt.xlabel("log10 (Area)"); plt.ylabel("log10 (Species richness)")
    plt.title(f"SAR fit: z ≈ {z:.2f}")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(outdir / f"fig6_sar_single.{ext}", dpi=300)
    plt.close()


# --------------------------------fig5b_range_size_three---Range Size Distributions----------------------
def plot_fig5b_range_size_three(dd, outdir: Path):
    """
    Fig. 5(b) – Range Size Distributions at ~20%, 50%, and 100% of γ.
    Proxy: choose species with the largest ranges to represent earlier assembly
    stages (top-20% and top-50% by range size), and compare to all species (100%).

    Saves: results/figures_ibm_like/fig5b_range_size_three.{png,pdf}
    """
    # ---- get a spatial snapshot ----
    B_last = dd.get("B_last")
    if B_last is None:
        B = dd.get("IBM_B")
        if B is None or B.ndim != 4:
            raise RuntimeError("Need a spatial snapshot: missing B_last and IBM_B is not 4-D. "
                               "Re-run with --ibm-record-mode full or keep B_last in the training NPZ.")
        B_last = B[-1]  # (S,Y,X)

    if B_last.ndim != 3:
        raise RuntimeError("Expected B_last with shape (S,Y,X).")

    S, Y, X = B_last.shape
    P = Y * X

    # ---- presence/absence with your detection threshold (if stored) ----
    pres = _presence_mask_anyshape(B_last, dd).astype(bool)  # (S,Y,X)

    # consider only species that exist in at least one patch
    occ_counts = pres.sum(axis=(1, 2))                # (S,)
    alive_mask = (occ_counts > 0)
    if not np.any(alive_mask):
        raise RuntimeError("No extant species in snapshot → cannot build RSD.")
    pres = pres[alive_mask]
    occ_counts = occ_counts[alive_mask]
    gamma = int(alive_mask.sum())

    # ---- range size per species = fraction of patches occupied ----
    rsd_all = occ_counts / float(P)                   # (γ,) in [0,1]

    # sort by range size (large → small) to emulate early→late assembly
    order = np.argsort(-rsd_all)
    rsd_sorted = rsd_all[order]

    # target sets: 20%, 50%, 100% of γ   (at least 1 species each)
    n20 = max(1, int(np.ceil(0.20 * gamma)))
    n50 = max(1, int(np.ceil(0.50 * gamma)))
    n100 = gamma

    groups = [
        ("20%",  rsd_sorted[:n20]),
        ("50%",  rsd_sorted[:n50]),
        ("100%", rsd_sorted[:n100]),
    ]

    # ---- plotting (three stacked panels, shared x-axis 0..1) ----
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5.2, 7.0), sharex=True)
    bin_edges = np.linspace(0, 1, 21)  # 20 equal bins in [0,1]

    facecols = ["#81c3c3", "#2c3459", "#97C216"]  # light → dark gray
    for ax, (label, data), fc in zip(axes, groups, facecols):
        # make sure it's a 1D array
        data = np.asarray(data, float)
        ax.hist(data, bins=bin_edges, edgecolor="black", facecolor=fc)
        ax.set_ylabel("Frequency")
        ax.text(0.98, 0.85, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=11)

        # a little padding so the final bin edge is visible
        ax.set_xlim(0.0, 1.0)
        ax.grid(False)

    axes[-1].set_xlabel("Range size (fraction of patches)")

    # a small overarching title (optional)
    fig.suptitle("Range Size Distributions", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig5b_range_size_three.{ext}", dpi=300)
    plt.close(fig)

    print(f"[fig5b] γ={gamma} species | panels built with n20={n20}, n50={n50}, n100={n100} "
          f"| saved to {outdir}")


# ---------------- NEW: Fig.4(a) – Bray–Curtis similarity -----------------
def plot_fig4a_braycurtis(dd, outdir: Path):
    B = dd.get("IBM_B")
    t = dd.get("IBM_t")
    if B is None or B.ndim != 4:
        raise RuntimeError("Fig.4(a) needs IBM_B with shape (T,S,Y,X). Re-run with --ibm-record-mode full.")
    T, S, Ny, Nx = B.shape
    if t is None or len(t) != T:
        t = np.arange(T)

    def bc_sim(x, y):
        sx = float(np.sum(x)); sy = float(np.sum(y))
        if (sx + sy) <= 0:  # both zero
            return 1.0
        return 2.0 * float(np.minimum(x, y).sum()) / (sx + sy)

    # Regional composition vectors
    R = B.sum(axis=(2, 3))            # (T, S)
    R0 = R[0]
    bc_reg = np.array([bc_sim(R0, R[k]) for k in range(T)])

    # Three fixed local patches (clip to grid)
    picks = [(0, 0), (0, min(Nx-1,1)), (min(Ny-1,1), 0)]
    picks = picks[:min(3, Ny*Nx)]

    bc_loc = []
    for (yy, xx) in picks:
        L = B[:, :, yy, xx]          # (T, S)
        L0 = L[0]
        bc_loc.append(np.array([bc_sim(L0, L[k]) for k in range(T)]))

    # Plot
    plt.figure(figsize=(6.6, 4.6))
    plt.plot(t, bc_reg, 'k-', lw=2.0, label="Regional")
    for i, arr in enumerate(bc_loc, 1):
        plt.plot(t, arr, '-', lw=1.4, alpha=0.9, label=f"Local {i}", color=f"C{i}")
    plt.ylim(0, 1.02)
    plt.xlabel("Time, T")
    plt.ylabel("Bray–Curtis similarity")
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(outdir / f"fig4a_braycurtis_similarity.{ext}", dpi=300)
    plt.close()

# -------- NEW: Fig.3(a) – Local vs Regional species richness ------------
def plot_fig3a_local_vs_regional_history(dd, outdir: Path):
    """
    Fig.3(a)-style scatter using the *assembly history*:
      x = ASM_gamma  (regional richness after each window)
      y = ASM_alpha_patches[:, j]  (local richness for up to 3 fixed patches)

    Falls back to ASM_alpha_bar if ASM_alpha_patches is unavailable.
    This gives a wide spread on the x-axis (γ changes across assembly).
    """
    gamma = dd.get("ASM_gamma")                 # (R,)
    a_p   = dd.get("ASM_alpha_patches")         # (R, ≤3)  integers per fixed patch
    a_bar = dd.get("ASM_alpha_bar")             # (R,)     mean local richness

    if gamma is None:
        raise RuntimeError("Fig.3a(history) requires ASM_gamma in the NPZ.")
    gamma = np.asarray(gamma, dtype=float)

    # Prefer per-patch richness so we get many points; else fallback to mean.
    series = []
    labels = []
    if a_p is not None:
        a_p = np.asarray(a_p)
        # Keep all windows; we want the full spread of γ
        for j in range(a_p.shape[1]):
            series.append(np.asarray(a_p[:, j], dtype=float))
            labels.append(f"Local {j+1}")
    elif a_bar is not None:
        series.append(np.asarray(a_bar, dtype=float))
        labels.append("Mean local (ᾱ)")
    else:
        raise RuntimeError("Fig.3a(history) needs ASM_alpha_patches or ASM_alpha_bar.")

    # Build the scatter arrays (stack all local series against the same γ)
    X, Y = [], []
    for y in series:
        # Ensure same length
        L = min(len(gamma), len(y))
        X.append(gamma[:L])
        Y.append(y[:L])
    X = np.concatenate(X)
    Y = np.concatenate(Y)

    # Plot in the paper’s style: open circles for 'Total'
    plt.figure(figsize=(6.6, 5.0))
    rng = np.random.default_rng(0)
    Yj = np.clip(Y + rng.normal(0, 0.15, size=Y.shape), 0, None)
    plt.scatter(X, Yj, facecolors='none', edgecolors='k',
                s=14, linewidths=0.6, alpha=0.85, label="Total")

    plt.xlim(0, max(5, X.max()*1.02))
    plt.ylim(0, max(5, Y.max()*1.02))
    plt.xlabel("Regional species richness")
    plt.ylabel("Local species richness")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(outdir / f"fig3a_local_vs_regional.{ext}", dpi=300)
    plt.close()

# ---------------- main ----------------
def main(npz_path: str):
    dd = load_npz(npz_path)
    outdir = Path("results/figures_ibm_like"); outdir.mkdir(parents=True, exist_ok=True)

    # Fig.1a (true history if present)
    plot_fig1a_from_history(dd, outdir)

    # NEW: Fig.4(a) – BC similarity
    plot_fig4a_braycurtis(dd, outdir)

   # Fig.3(a) – Local vs Regional richness *from assembly history* (wide x-spread)
    plot_fig3a_local_vs_regional_history(dd, outdir)

    # Fig.5(b) – three range-size histograms (20%, 50%, 100%)
    plot_fig5b_range_size_three(dd, outdir)


    # Fig.5 and Fig.6 from spatial snapshot
    B_last = dd.get("B_last")
    if B_last is None:
        B = dd.get("IBM_B")
        if B is None or B.ndim != 4:
            raise RuntimeError("Need a spatial snapshot: missing B_last and IBM_B is not 4-D. "
                               "Re-run with --ibm-record-mode full or keep B_last in the training NPZ.")
        B_last = B[-1]
    plot_fig5(B_last, outdir)
    plot_fig6(B_last, outdir)
    print("Saved figures to:", outdir.resolve())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()
    main(args.npz)







