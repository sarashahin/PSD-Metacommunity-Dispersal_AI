# scripts/check_equilibrium_timeseries.py

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


def main():
    ap = argparse.ArgumentParser(
        description="Community-level time series to assess equilibrium (richness, mean range)."
    )
    ap.add_argument("--root", required=True, help="results/data folder")
    ap.add_argument("--out", required=True, help="output folder (e.g. results/ecology_equilibrium)")
    ap.add_argument("--world", type=int, default=0, help="catalog world index")
    ap.add_argument("--stride", type=int, default=10,
                    help="time stride for computing statistics (default: 10)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out_world = out / f"world{args.world:03d}"
    out_world.mkdir(parents=True, exist_ok=True)

    dc = DataCatalog(str(root))
    with dc[args.world] as w:
        # get presence time series
        if hasattr(w, "has") and w.has("P_t"):
            P_t = w.get("P_t")  # (T,S,Y,X)
            t_axis = w.get("IBM_t") if w.has("IBM_t") else np.arange(P_t.shape[0])
        elif hasattr(w, "has") and w.has("IBM_B"):
            B_t = w.get("IBM_B")  # (T,S,Y,X)
            # threshold biomass → presence using detection_threshold * BODY_MASS
            dt = float(w.get("detection_threshold")) if w.has("detection_threshold") else 0.0
            bm = float(w.get("BODY_MASS")) if w.has("BODY_MASS") else 1.0
            thr_mass = dt * bm
            if thr_mass > 0:
                P_t = (B_t >= thr_mass).astype(np.uint8)
            else:
                P_t = (B_t > 0).astype(np.uint8)
            t_axis = w.get("IBM_t") if w.has("IBM_t") else np.arange(P_t.shape[0])
        else:
            raise RuntimeError("World has no P_t or IBM_B → cannot do equilibrium check.")

        T, S, Y, X = P_t.shape
        area = Y * X

        stride = max(1, int(args.stride))
        idxs = np.arange(0, T, stride, dtype=int)

        richness = []
        mean_range = []

        for ti in idxs:
            P = P_t[ti].astype(bool)  # (S,Y,X)
            occ_counts = P.reshape(S, -1).sum(axis=1)
            extant = occ_counts > 0
            n_ext = extant.sum()
            richness.append(float(n_ext))
            if n_ext > 0:
                rs = occ_counts[extant] / area
                mean_range.append(float(rs.mean()))
            else:
                mean_range.append(0.0)

        richness = np.array(richness)
        mean_range = np.array(mean_range)
        times = np.asarray(t_axis)[idxs]

        # simple equilibrium diagnostic: compare last quarter vs previous quarter
        n_idx = len(idxs)
        if n_idx >= 4:
            q = n_idx // 4
            last_q = slice(-q, None)
            prev_q = slice(-2 * q, -q)

            rich_last = richness[last_q].mean()
            rich_prev = richness[prev_q].mean()
            range_last = mean_range[last_q].mean()
            range_prev = mean_range[prev_q].mean()

            eq_summary = {
                "world": args.world,
                "file": w.path.name,
                "richness_prev_quarter": rich_prev,
                "richness_last_quarter": rich_last,
                "richness_rel_change": (rich_last - rich_prev) / max(rich_prev, 1e-9),
                "mean_range_prev_quarter": range_prev,
                "mean_range_last_quarter": range_last,
                "mean_range_rel_change": (range_last - range_prev) / max(range_prev, 1e-9),
            }
        else:
            eq_summary = {
                "world": args.world,
                "file": w.path.name,
                "richness_prev_quarter": None,
                "richness_last_quarter": None,
                "richness_rel_change": None,
                "mean_range_prev_quarter": None,
                "mean_range_last_quarter": None,
                "mean_range_rel_change": None,
            }

        (out_world / "equilibrium_summary.json").write_text(json.dumps(eq_summary, indent=2))

        # plots
        plt.figure(figsize=(7, 4))
        plt.plot(times, richness, label="richness")
        plt.xlabel("time")
        plt.ylabel("# extant species")
        plt.title("Richness vs time")
        plt.tight_layout()
        plt.savefig(out_world / "richness_vs_time.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(times, mean_range, label="mean range size")
        plt.xlabel("time")
        plt.ylabel("mean range size (fraction)")
        plt.title("Mean range size vs time")
        plt.tight_layout()
        plt.savefig(out_world / "mean_range_vs_time.png", dpi=180)
        plt.close()

        print("[done] equilibrium check for world", args.world, "→", out_world)


if __name__ == "__main__":
    main()
