# scripts/check_detection_sensitivity.py

from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

from load_training_worlds import DataCatalog


def connected_components_4(mask2d: np.ndarray) -> int:
    Ny, Nx = mask2d.shape
    visited = np.zeros_like(mask2d, dtype=bool)
    stack = []
    n_comp = 0
    for y in range(Ny):
        for x in range(Nx):
            if not mask2d[y, x] or visited[y, x]:
                continue
            n_comp += 1
            stack.append((y, x))
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < Ny and 0 <= nx < Nx:
                        if mask2d[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
    return n_comp


def per_factor_stats(B_t, dt, bm, factor, r_net, outdir: Path):
    """
    For one factor * detection_threshold, compute final occupancy and basic stats.
    """
    T, S, Y, X = B_t.shape
    area = Y * X
    # presence definition
    thr_mass = factor * dt * bm
    if thr_mass > 0:
        P_t = (B_t >= thr_mass).astype(np.uint8)
    else:
        P_t = (B_t > 0).astype(np.uint8)

    Pfin = P_t[-1].astype(bool)  # (S,Y,X)
    present = Pfin.reshape(S, -1).sum(axis=1) > 0
    spp = np.where(present)[0]
    if spp.size == 0:
        return {
            "factor": factor,
            "n_extant": 0,
            "range_size_mean": 0.0,
            "mean_components": 0.0,
            "rescue_cond_mean": None,
        }

    occ_counts = Pfin.reshape(S, -1).sum(axis=1)[present].astype(float)
    range_size = occ_counts / area
    comps = np.array([connected_components_4(Pfin[s]) for s in spp], int)

    # rescue conditional (using provided r_net, independent of detection)
    rescue_cond = None
    if r_net is not None:
        rescue = np.full(len(spp), np.nan)
        for i, s in enumerate(spp):
            occ = Pfin[s]
            k = occ.sum()
            if k == 0:
                continue
            neg = (r_net[s] < 0) & occ
            rescue[i] = neg.sum() / float(k)
        if np.isfinite(rescue).any():
            rescue_cond = float(np.nanmean(rescue))

    return {
        "factor": factor,
        "n_extant": int(len(spp)),
        "range_size_mean": float(range_size.mean()),
        "mean_components": float(comps.mean()),
        "rescue_cond_mean": rescue_cond,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity of ranges and rescue to detection threshold."
    )
    ap.add_argument("--root", required=True, help="results/data folder")
    ap.add_argument("--out", required=True, help="output folder (e.g. results/ecology_detection)")
    ap.add_argument("--world", type=int, default=0, help="catalog world index")
    ap.add_argument("--factors", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0],
                    help="multipliers for detection_threshold (default: 0.5 1.0 2.0 4.0)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    dc = DataCatalog(str(root))

    out_world = out / f"world{args.world:03d}"
    out_world.mkdir(parents=True, exist_ok=True)

    with dc[args.world] as w:
        if not (hasattr(w, "has") and (w.has("IBM_B") or w.has("P_t"))):
            raise RuntimeError("World has no IBM_B or P_t → cannot do detection sensitivity.")

        if hasattr(w, "has") and w.has("IBM_B"):
            B_t = w.get("IBM_B")  # (T,S,Y,X)
        else:
            # if P_t exists but no IBM_B, we can't change detection; abort
            raise RuntimeError("Only P_t present; no IBM_B → cannot vary detection threshold.")

        dt = float(w.get("detection_threshold")) if w.has("detection_threshold") else 0.0
        bm = float(w.get("BODY_MASS")) if w.has("BODY_MASS") else 1.0

        # optional r_net from previous analysis (if stored); else set None
        try:
            r_net = w.get("r_net_last")  # only if you saved it; else None
        except Exception:
            r_net = None

        stats = []
        for f in args.factors:
            st = per_factor_stats(B_t, dt, bm, f, r_net, out_world)
            stats.append(st)

    # save JSON
    (out_world / "detection_sensitivity.json").write_text(json.dumps(stats, indent=2))

    # simple plots: n_extant and mean range vs factor
    factors = [s["factor"] for s in stats]
    n_extant = [s["n_extant"] for s in stats]
    mean_range = [s["range_size_mean"] for s in stats]

    plt.figure(figsize=(6, 4))
    plt.plot(factors, n_extant, marker="o")
    plt.xscale("log")
    plt.xlabel("detection_threshold factor")
    plt.ylabel("n_extant (final)")
    plt.title("Extant richness vs detection threshold factor")
    plt.tight_layout()
    plt.savefig(out_world / "n_extant_vs_factor.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(factors, mean_range, marker="o")
    plt.xscale("log")
    plt.xlabel("detection_threshold factor")
    plt.ylabel("mean range size (fraction)")
    plt.title("Mean range size vs detection threshold factor")
    plt.tight_layout()
    plt.savefig(out_world / "mean_range_vs_factor.png", dpi=180)
    plt.close()

    print("[done] detection sensitivity for world", args.world, "→", out_world)


if __name__ == "__main__":
    main()
