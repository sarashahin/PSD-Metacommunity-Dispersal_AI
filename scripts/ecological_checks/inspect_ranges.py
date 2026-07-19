# inspect_ranges.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from load_training_worlds import DataCatalog   # your loader


def pick_rarest_species(Pfin: np.ndarray, K: int):
    S, Y, X = Pfin.shape
    area = Y * X
    occ = Pfin.reshape(S, -1).sum(axis=1)
    extant = np.where(occ > 0)[0]
    ranges = occ[extant] / float(area)
    order = np.argsort(ranges)  # ascending => rarest first
    return extant[order[:min(K, len(extant))]]

# ---------------- connected-component counter ----------------
def connected_components_4(mask2d: np.ndarray) -> int:
    """
    Count 4-neighbour connected components in a 2D boolean mask.
    """
    Ny, Nx = mask2d.shape
    visited = np.zeros_like(mask2d, dtype=bool)
    n_comp = 0
    stack = []

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

# ---------------- final-snapshot connected vs fragmented ----------------
def analyse_connectedness(world_idx: int, root: str):
    """
    For one training world:
    - load final occupancy
    - count how many species have 1 connected component vs >1
    """
    dc = DataCatalog(root)
    with dc[world_idx] as w:
        Pfin = w.get_final_occupancy()   # (S, Y, X)
        S, Y, X = Pfin.shape

        n_conn = 0
        n_frag = 0
        comp_counts = []

        for s in range(S):
            mask = Pfin[s].astype(bool)
            if not mask.any():
                continue
            c = connected_components_4(mask)
            comp_counts.append(c)
            if c == 1:
                n_conn += 1
            else:
                n_frag += 1

        print(f"[world {world_idx}] total extant species = {n_conn + n_frag}")
        print(f"  connected ranges:  {n_conn}")
        print(f"  fragmented ranges: {n_frag}")
        if comp_counts:
            print(f"  mean #components among extant species: {np.mean(comp_counts):.2f}")

# ---------------- movies / time-series for a few species ----------------
def movie_for_species(world_idx: int, root: str, species_ids=None, K: int = 3):
    """
    For one world, plot for a few species:
    - 3 spatial snapshots (early, middle, late)
    - range size vs time (fraction of patches)
    - number of connected components vs time
    """
    dc = DataCatalog(root)
    with dc[world_idx] as w:
        # --- get presence time-series ---
        if hasattr(w, "has") and w.has("P_t"):
            P_t = w.get("P_t")                 # (T,S,Y,X)
            IBM_t = w.get("IBM_t") if w.has("IBM_t") else np.arange(P_t.shape[0])
        elif hasattr(w, "has") and w.has("IBM_B"):
            B_t = w.get("IBM_B")              # (T,S,Y,X)
            # threshold biomass → presence using detection_threshold * BODY_MASS
            try:
                det = float(w.get("detection_threshold")) if w.has("detection_threshold") else 0.0
            except Exception:
                det = 0.0
            try:
                body_mass = float(w.get("BODY_MASS")) if w.has("BODY_MASS") else 1.0
            except Exception:
                body_mass = 1.0
            thr_mass = det * body_mass
            if thr_mass > 0:
                P_t = (B_t >= thr_mass).astype(np.uint8)
            else:
                P_t = (B_t > 0).astype(np.uint8)
            IBM_t = w.get("IBM_t") if w.has("IBM_t") else np.arange(P_t.shape[0])
        else:
            raise RuntimeError(
                f"{w.path.name} has no P_t or IBM_B → no IBM time series stored. "
                "Pick another --world (one of the batchA IBM worlds)."
            )

        T, S, Y, X = P_t.shape

        # choose species: explicitly or K random extant species
        if species_ids is None:
            P_any = w.get("P_any")      # (S, Y, X)
            extant = np.where(P_any.reshape(S, -1).sum(axis=1) > 0)[0]
            rng = np.random.default_rng(0)
            species_ids = rng.choice(extant, size=min(K, len(extant)), replace=False)

        for s in species_ids:
            presence = P_t[:, s]                          # (T, Y, X), 0/1
            range_size = presence.reshape(T, -1).mean(axis=1)  # fraction of patches
            n_comp = np.array([
                connected_components_4(p.astype(bool))
                for p in presence
            ])

            fig, axes = plt.subplots(2, 3, figsize=(10, 5))
            axes = axes.ravel()

            # snapshots at a few times
            idxs = [0, T // 2, T - 1]
            for j, ti in enumerate(idxs):
                im = axes[j].imshow(presence[ti], origin="upper")
                axes[j].set_title(f"s={s}, t={IBM_t[ti]:.0f}")
                fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

            # range size vs time
            axes[3].plot(IBM_t, range_size)
            axes[3].set_xlabel("time")
            axes[3].set_ylabel("range size (fraction)")
            axes[3].set_title("Range size over time")

            # number of components vs time
            axes[4].plot(IBM_t, n_comp, drawstyle="steps-mid")
            axes[4].set_xlabel("time")
            axes[4].set_ylabel("# components")
            axes[4].set_title("Connected components over time")

            axes[5].axis("off")
            fig.suptitle(f"Species {s} – range evolution", y=0.98)
            fig.tight_layout()
            plt.show()

# ---------------- simple CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Inspect IBM range shapes and their time evolution."
    )
    ap.add_argument("--root", required=True,
                    help="Path to results/data/ with *_training.npz")
    ap.add_argument("--world", type=int, default=0,
                    help="Catalog index of world (default: 0)")
    ap.add_argument("--movie", action="store_true",
                    help="Also show movies/time-series for a few species")
    ap.add_argument("--species", type=int, nargs="*",
                    help="Optional explicit species indices for movies")
    ap.add_argument("--rarest", type=int, default=0,
    help="If >0, auto-pick K rarest extant species for --movie")
    args = ap.parse_args()

    analyse_connectedness(args.world, args.root)
    if args.movie:
        if args.rarest and not args.species:
            dc = DataCatalog(args.root)
            with dc[args.world] as w:
                Pfin = w.get_final_occupancy()
                auto_species = pick_rarest_species(Pfin, args.rarest)
            print(f"[auto] rarest species: {auto_species.tolist()}")
            species = auto_species.tolist()
        else:
            species = args.species if args.species else None
        movie_for_species(args.world, args.root, species_ids=species)
