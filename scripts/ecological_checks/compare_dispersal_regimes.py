# scripts/compare_dispersal_regimes.py

from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from load_training_worlds import DataCatalog


# ---------------------- small helpers ----------------------
def _get_any(w, keys):
    for k in keys:
        try:
            return w.get(k)
        except KeyError:
            pass
    return None

def _finite(x):
    return np.isfinite(x)

def _safe_quantiles(x, qs):
    x = np.asarray(x)
    x = x[_finite(x)]
    if x.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.quantile(x, q)) for q in qs}

def connected_components_4(mask2d: np.ndarray) -> int:
    Ny, Nx = mask2d.shape
    seen = np.zeros_like(mask2d, dtype=bool)
    stack = []
    n = 0
    for y in range(Ny):
        for x in range(Nx):
            if not mask2d[y, x] or seen[y, x]:
                continue
            n += 1
            stack.append((y, x))
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < Ny and 0 <= nx < Nx and mask2d[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return n


# ---------------------- world selection ----------------------
def world_meta(dc, i):
    with dc[i] as w:
        dr  = float(np.array(w.get("DISPERSAL_RATE")).squeeze())
        ldd = float(np.array(w.get("LONG_DISTANCE_PROB")).squeeze())
        name = w.path.name
    return {"world": i, "file": name, "DISPERSAL_RATE": dr, "LONG_DISTANCE_PROB": ldd}

def select_worlds(dc, low_world=None, high_world=None):
    metas = [world_meta(dc, i) for i in range(len(dc))]
    if low_world is None:
        low_world  = min(metas, key=lambda m: (m["DISPERSAL_RATE"], m["LONG_DISTANCE_PROB"]))["world"]
    if high_world is None:
        high_world = max(metas, key=lambda m: (m["DISPERSAL_RATE"], m["LONG_DISTANCE_PROB"]))["world"]
    return low_world, high_world, metas


# ---------------------- r_net construction ----------------------
def _broadcast_species_scalar(v, S, Y, X):
    """v can be scalar or (S,) → (S,Y,X)"""
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        return np.full((S, Y, X), float(v))
    if v.ndim == 1 and v.shape[0] == S:
        return v[:, None, None] * np.ones((1, Y, X), dtype=float)
    return None

def get_r_net(w, S, Y, X) -> tuple[str, np.ndarray | None]:
    """
    Prefer ENV_r_field[s,y,x], else fall back to species scalar r0 from r_base/r_mean.
    r_net[s,y,x] = r_env_or_r0 - c_ss * B_s - sum_k w[s,k] * B[idx[s,k]]
    """
    # biomass
    try:
        B = np.asarray(w.get("B_last"), dtype=float)  # (S,Y,X)
        assert B.shape == (S, Y, X)
    except Exception:
        return "absent", None

    # competition sparsity (required for comp_sum)
    try:
        Cidx = np.asarray(w.get("C_topk_idx"), dtype=int)    # (S,K)
        Cw   = np.asarray(w.get("C_topk_w"),  dtype=float)   # (S,K)
        assert Cidx.ndim == 2 and Cw.shape == Cidx.shape
    except Exception:
        return "absent", None

    # preferred: species×patch env field
    r_env = _get_any(w, ["ENV_r_field", "r_field"])
    mode = "absent"
    rS = None
    if r_env is not None:
        r_env = np.asarray(r_env, dtype=float)
        if r_env.shape == (S, Y, X):
            rS = r_env
            mode = "env_species_patch"

    # fallback: species scalar baseline (e.g. r_base or r_mean)
    if rS is None:
        r0 = _get_any(w, ["r_base", "r_mean"])
        rS = _broadcast_species_scalar(r0 if r0 is not None else 1.0, S, Y, X)
        if rS is None:
            return "absent", None
        mode = "species_scalar_fallback"

    # gather competitors' biomasses -> (S,K,Y,X)
    try:
        B_comp = B[Cidx]                                  # fancy index on species axis
        comp_sum = (B_comp * Cw[:, :, None, None]).sum(axis=1)  # (S,Y,X)
    except Exception:
        return "absent", None

    SELF_C = 1.0  # change if your c_ss != 1
    r_net = rS - SELF_C * B - comp_sum
    return f"derived_{mode}", r_net


# ---------------------- per-world stats ----------------------
def per_world_stats(dc, idx):
    with dc[idx] as w:
        Pfin = w.get_final_occupancy().astype(bool)   # (S,Y,X)
        S, Y, X = Pfin.shape
        area = Y * X

        meta = {
            "world": idx,
            "file": w.path.name,
            "DISPERSAL_RATE": float(np.array(w.get("DISPERSAL_RATE")).squeeze()),
            "LONG_DISTANCE_PROB": float(np.array(w.get("LONG_DISTANCE_PROB")).squeeze()),
        }

        # biomass (for per-species mean_B_on_occupied)
        B = _get_any(w, ["B_last", "B_lastK"])
        if B is not None:
            B = np.asarray(B, dtype=float)
            if B.shape != Pfin.shape:
                if B.ndim == 3 and B.shape == (Y, X, S):
                    B = np.moveaxis(B, -1, 0)  # (S,Y,X)
                else:
                    B = None

        # who is present?
        present = Pfin.reshape(S, -1).sum(axis=1) > 0
        spp_ids = np.where(present)[0]

        # range size + connected components
        range_size = Pfin.reshape(S, -1).sum(axis=1)[present] / area
        components = np.array([connected_components_4(Pfin[s]) for s in spp_ids], int)

        # rescue from r_net (env − self − competition)
        r_mode, r_net = get_r_net(w, S, Y, X)
        meta["rescue_index_mode"] = r_mode

        rescue_cond = np.full(len(spp_ids), np.nan)
        rescue_glob = np.full(len(spp_ids), np.nan)
        if r_net is not None:
            occ_counts = Pfin[spp_ids].reshape(len(spp_ids), -1).sum(axis=1).astype(float)
            neg_and_occ = ((r_net[spp_ids] < 0) & Pfin[spp_ids]).reshape(len(spp_ids), -1).sum(axis=1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                rescue_cond = neg_and_occ / occ_counts
                rescue_glob = neg_and_occ / area

        # mean biomass over occupied cells (optional)
        mean_B_on_occ = np.full(len(spp_ids), np.nan)
        B_stats = {}
        if B is not None:
            B_stats = {
                "finite_frac": float(np.isfinite(B).mean()),
                "min": float(np.nanmin(B)) if np.isfinite(B).any() else None,
                "p50": float(np.nanmedian(B)) if np.isfinite(B).any() else None,
                "p95": float(np.nanpercentile(B, 95)) if np.isfinite(B).any() else None,
                "max": float(np.nanmax(B)) if np.isfinite(B).any() else None,
            }
            for i, s in enumerate(spp_ids):
                if Pfin[s].any():
                    vals = B[s][Pfin[s]]
                    vals = vals[_finite(vals)]
                    if vals.size:
                        mean_B_on_occ[i] = vals.mean()

        # pack per-species table (columns fixed order)
        per_species = np.stack(
            [spp_ids, range_size, components, rescue_cond, rescue_glob, mean_B_on_occ],
            axis=1
        )

        meta["has_biomass"] = bool(B is not None)
        meta["biomass_stats"] = B_stats
        return meta, (Y, X), Pfin, B, r_net, per_species


# ---------------------- plots & exports ----------------------
def make_histograms(per_species, outdir: Path, title: str):
    outdir.mkdir(parents=True, exist_ok=True)
    sid = per_species[:, 0].astype(int)
    rs  = per_species[:, 1].astype(float)
    cc  = per_species[:, 2].astype(int)
    rci = per_species[:, 3].astype(float)

    summary = {
        "n_extant": int(len(sid)),
        "connected": int((cc == 1).sum()),
        "fragmented": int((cc > 1).sum()),
        "mean_components": float(cc.mean()) if len(cc) else 0.0,
        "range_size_mean": float(rs.mean()) if len(rs) else 0.0,
        "range_size_q": _safe_quantiles(rs, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)),
        "rescue_cond_mean": float(np.nanmean(rci)) if np.isfinite(rci).any() else None,
        "rescue_cond_q": _safe_quantiles(rci[np.isfinite(rci)], (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
                         if np.isfinite(rci).any() else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    np.savetxt(outdir / "per_species.csv",
               per_species,
               fmt=["%d", "%.8f", "%d", "%.6f", "%.6f", "%.6e"],
               delimiter=",",
               header="species_id,range_size,components,rescue_frac_cond,rescue_frac_global,mean_B_on_occupied",
               comments="")

    plt.figure(figsize=(6, 4))
    plt.hist(rs, bins=50)
    plt.xlabel("range size (fraction of patches)")
    plt.ylabel("count of species")
    plt.title(f"{title} — Range size")
    plt.tight_layout()
    plt.savefig(outdir / "hist_range_size.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    bins = np.arange(1, (cc.max() if len(cc) else 1) + 2) - 0.5
    plt.hist(cc, bins=bins)
    plt.xlabel("# connected components")
    plt.ylabel("count of species")
    plt.title(f"{title} — Connected components")
    plt.xticks(np.arange(1, bins[-1] + 0.5))
    plt.tight_layout()
    plt.savefig(outdir / "hist_components.png", dpi=180)
    plt.close()

    if np.isfinite(rci).any():
        plt.figure(figsize=(6, 4))
        plt.hist(rci[np.isfinite(rci)], bins=50, range=(0, 1))
        plt.xlabel("rescue index (fraction of occupied patches with r_net<0)")
        plt.ylabel("count of species")
        plt.title(f"{title} — Rescue index")
        plt.tight_layout()
        plt.savefig(outdir / "hist_rescue_index.png", dpi=180)
        plt.close()

def _montage_single_channel(stack_SYX, ids, outpng: Path, title: str,
                            cmap="viridis", vmin=None, vmax=None):
    import math
    ids = list(map(int, ids))
    n = len(ids)
    ncols = min(8, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6*ncols, 1.6*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, sid in zip(axes, ids):
        ax.imshow(stack_SYX[sid], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"s={sid}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    fig.savefig(outpng, dpi=180)
    plt.close(fig)

def _choose_scale_for_species(Bs, Ps):
    vals = Bs[Ps]
    vals = vals[_finite(vals)]
    pos = vals[vals > 0]
    if pos.size == 0:
        if vals.size == 0:
            return False, 0.0, 1.0
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return False, 0.0, 1.0
        return False, vmin, vmax
    p5, p95 = np.percentile(pos, [5, 95])
    vmin = max(0.0, p5)
    vmax = p95 if p95 > vmin else pos.max()
    use_log = (pos.max() / max(pos.mean(), 1e-12)) > 1e4
    if vmax <= vmin:
        vmax = vmin + (1e-12 if not use_log else 1.0)
    return use_log, vmin, vmax

def dump_montages(Pfin, B, outdir: Path, world_idx: int):
    outdir.mkdir(parents=True, exist_ok=True)
    S, Y, X = Pfin.shape
    area = Y * X
    present = Pfin.reshape(S, -1).sum(axis=1) > 0
    spp = np.where(present)[0]
    rs = Pfin.reshape(S, -1).sum(axis=1) / area
    K = min(32, len(spp))
    order = np.argsort(rs[spp])
    rare_ids = spp[order[:K]]
    common_ids = spp[order[-K:]]

    _montage_single_channel(Pfin, rare_ids, outdir / "montage_rare.png",
                            title=f"World {world_idx}: rare species (small ranges)")
    _montage_single_channel(Pfin, common_ids[::-1], outdir / "montage_common.png",
                            title=f"World {world_idx}: common species (large ranges)")

    if B is not None and B.shape == Pfin.shape:
        B_vis = np.zeros_like(B, dtype=float)
        for sid in np.concatenate([rare_ids, common_ids]):
            use_log, vmin, vmax = _choose_scale_for_species(B[sid], Pfin[sid])
            B_vis[sid] = np.log1p(np.clip(B[sid], 0, None)) if use_log else B[sid]
        _montage_single_channel(B_vis, rare_ids, outdir / "montage_rare_B.png",
                                title=f"World {world_idx}: rare species biomass",
                                cmap="viridis", vmin=None, vmax=None)
        _montage_single_channel(B_vis, common_ids[::-1], outdir / "montage_common_B.png",
                                title=f"World {world_idx}: common species biomass",
                                cmap="viridis", vmin=None, vmax=None)

def dump_rescue_overlays(Pfin, r_net, outdir: Path, tag: str, topk: int = 16):
    """Overlay occupied cells (green) and occupied sinks with r_net<0 (yellow)."""
    if r_net is None:
        return
    S, Y, X = Pfin.shape
    area = Y * X
    present = Pfin.reshape(S, -1).sum(axis=1) > 0
    spp = np.where(present)[0]
    if len(spp) == 0:
        return
    occ = Pfin[spp].reshape(len(spp), -1).sum(axis=1).astype(float)
    neg = ((r_net[spp] < 0) & Pfin[spp]).reshape(len(spp), -1).sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rescue = neg / np.maximum(occ, 1.0)

    order = np.argsort(np.nan_to_num(rescue, nan=-1.0))[::-1]
    pick = spp[order[:min(topk, len(order))]]

    import math
    n = len(pick)
    ncols = min(8, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6*ncols, 1.6*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, sid in zip(axes, pick):
        occ_mask = Pfin[sid]
        sink_mask = (r_net[sid] < 0) & occ_mask
        # 0: empty, 1: occupied non-sink, 2: occupied sink
        panel = np.zeros((Y, X), dtype=int)
        panel[occ_mask] = 1
        panel[sink_mask] = 2
        ax.imshow(panel, origin="upper", vmin=0, vmax=2)
        ax.set_title(f"s={sid}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{tag}: occupied (green) vs sinks (yellow)", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / "overlay_rescue.png", dpi=180)
    plt.close(fig)


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results/data folder")
    ap.add_argument("--out",  required=True, help="output root folder")
    ap.add_argument("--low_world", type=int, default=None)
    ap.add_argument("--high_world", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root); out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dc = DataCatalog(str(root))
    lw, hw, metas = select_worlds(dc, args.low_world, args.high_world)

    metaL, (Yl, Xl), P_L, B_L, rL, perL = per_world_stats(dc, lw)
    metaH, (Yh, Xh), P_H, B_H, rH, perH = per_world_stats(dc, hw)

    (out / "world_selection.json").write_text(json.dumps({
        "all_worlds": metas, "low_world": metaL, "high_world": metaH
    }, indent=2))

    outL = out / f"low_dispersal_world{lw}_DR{metaL['DISPERSAL_RATE']:.2e}_LDD{metaL['LONG_DISTANCE_PROB']:.3f}"
    outH = out / f"high_dispersal_world{hw}_DR{metaH['DISPERSAL_RATE']:.2e}_LDD{metaH['LONG_DISTANCE_PROB']:.3f}"

    make_histograms(perL, outL, title=f"World {lw} (low dispersal)")
    make_histograms(perH, outH, title=f"World {hw} (high dispersal)")
    dump_montages(P_L, B_L, outL, world_idx=lw)
    dump_montages(P_H, B_H, outH, world_idx=hw)
    dump_rescue_overlays(P_L, rL, outL, tag=f"World {lw}")
    dump_rescue_overlays(P_H, rH, outH, tag=f"World {hw}")

    print("[done]")
    print("Low-dispersal:", metaL)
    print("High-dispersal:", metaH)
    print("Outputs:\n ", outL, "\n ", outH)

if __name__ == "__main__":
    main()
