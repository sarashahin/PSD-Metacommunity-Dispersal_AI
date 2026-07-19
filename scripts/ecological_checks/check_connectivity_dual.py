# scripts/ecological_checks/check_connectivity_dual.py

from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

from load_training_worlds import DataCatalog


def _get_any(w, keys):
    for k in keys:
        try:
            return w.get(k)
        except KeyError:
            continue
    return None


def connected_components_grid(mask2d: np.ndarray) -> int:
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


def connected_components_graph(present_vec: np.ndarray, adj: np.ndarray) -> int:
    """
    present_vec: length N boolean (patches where species is present)
    adj: NxN adjacency matrix (non-zero = dispersal link)
    """
    N = present_vec.size
    visited = np.zeros(N, dtype=bool)
    n_comp = 0
    for i in range(N):
        if not present_vec[i] or visited[i]:
            continue
        n_comp += 1
        stack = [i]
        visited[i] = True
        while stack:
            v = stack.pop()
            neighbors = np.where((adj[v] != 0) & present_vec & (~visited))[0]
            if neighbors.size:
                visited[neighbors] = True
                stack.extend(neighbors.tolist())
    return n_comp


def _build_adj_from_grid(world, N) -> np.ndarray:
    """
    Build an NxN adjacency matrix from GRID_u / GRID_v / GRID_w.

    Handles two cases:

    1) CSR-style row pointers:
       - GRID_u has length N+1, entries are offsets into GRID_v / GRID_w.

    2) Edge-list style (your worlds):
       - GRID_u, GRID_v, GRID_w all have length E (number of edges),
         and each index e stores one edge u[e] -> v[e] with weight w[e].
    """
    grid_u = np.asarray(world.get("GRID_u"))
    grid_v = np.asarray(world.get("GRID_v"))
    grid_w = np.asarray(world.get("GRID_w"))

    if grid_u.ndim != 1 or grid_v.ndim != 1 or grid_w.ndim != 1:
        raise RuntimeError(
            f"GRID_u/v/w have unexpected shapes: "
            f"u={grid_u.shape}, v={grid_v.shape}, w={grid_w.shape}"
        )

    E = grid_u.size
    A = np.zeros((N, N), float)

    # ---- case 1: classic CSR with N+1 row pointers ----
    if E == N + 1:
        for i in range(N):
            start = int(grid_u[i])
            end = int(grid_u[i + 1])
            if end <= start:
                continue
            nbrs = grid_v[start:end].astype(int)
            wts = grid_w[start:end].astype(float)
            A[i, nbrs] = wts

    # ---- case 2: edge-list style: one entry per edge ----
    elif grid_v.size == E and grid_w.size == E:
        for e in range(E):
            i = int(grid_u[e])
            j = int(grid_v[e])
            if 0 <= i < N and 0 <= j < N:
                A[i, j] = max(A[i, j], float(grid_w[e]))

    else:
        raise RuntimeError(
            "GRID_u/GRID_v/GRID_w have unsupported lengths for building adjacency:\n"
            f"  N={N}, len(u)={grid_u.size}, len(v)={grid_v.size}, len(w)={grid_w.size}"
        )

    # for connectivity we only care about undirected links
    A = np.maximum(A, A.T)
    return A


def _find_adjacency_matrix(w, N, Y, X) -> tuple[str, np.ndarray]:
    """
    Try several common key names / shapes for the patch adjacency / dispersal network.
    Returns (key_used, adj_NxN).
    Will also fall back to GRID_u/GRID_v/GRID_w representation.
    """
    candidate_keys = [
        "ADJACENCY", "A_disp", "A_topo", "Dmat", "disp_mat", "dMat",
        "A", "A_net", "A_patch",
        "patch_adj", "patch_adjacency",
        "D_disp", "D_topo", "D_patch",
        "network_adj", "NETWORK_ADJ",
    ]

    # --- 1) direct NxN matrix ---
    for key in candidate_keys:
        try:
            arr = w.get(key)
        except KeyError:
            continue
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape == (N, N):
            return key, arr.astype(float)

    # --- 2) 4-D (Y,X,Y,X) tensor – convert to NxN ---
    for key in candidate_keys:
        try:
            arr = w.get(key)
        except KeyError:
            continue
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape == (Y, X, Y, X):
            A = arr.reshape(N, N)
            return key + " (Y,X,Y,X→N,N)", A.astype(float)

    # --- 3) sparse edge list (src, dst) ---
    try:
        src = np.asarray(w.get("EDGES_SRC"))
        dst = np.asarray(w.get("EDGES_DST"))
        if src.shape == dst.shape:
            A = np.zeros((N, N), float)
            A[src, dst] = 1.0
            A[dst, src] = 1.0
            return "EDGES_SRC/EDGES_DST", A
    except Exception:
        pass

    # --- 4) GRID_u / GRID_v / GRID_w representation (your worlds) ---
    try:
        A = _build_adj_from_grid(w, N)
        return "GRID_u/GRID_v/GRID_w", A
    except KeyError:
        # keys missing → fall through to error below
        pass
    except RuntimeError as e:
        raise RuntimeError(f"Failed to build adjacency from GRID_u/v/w: {e}") from e

    # --- 5) total failure → give a helpful error with all keys ---
    try:
        all_keys = list(w.keys())
    except Exception:
        all_keys = ["<cannot list keys>"]

    raise RuntimeError(
        "No adjacency / dispersal matrix found in world.\n"
        f"Looked for keys: {candidate_keys + ['EDGES_SRC/EDGES_DST', 'GRID_u/GRID_v/GRID_w']}\n"
        f"Available keys in this world: {all_keys}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare geometric vs dispersal-network connectivity for species ranges."
    )
    ap.add_argument("--root", required=True, help="results/data folder")
    ap.add_argument(
        "--out",
        required=True,
        help="output folder (e.g. results/ecology_connectivity)",
    )
    ap.add_argument("--world", type=int, default=0, help="catalog world index")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out_world = out / f"world{args.world:03d}"
    out_world.mkdir(parents=True, exist_ok=True)

    dc = DataCatalog(str(root))
    with dc[args.world] as w:
        Pfin = w.get_final_occupancy().astype(bool)  # (S,Y,X)
        S, Y, X = Pfin.shape
        N = Y * X

        # --- find adjacency / dispersal network on patch graph ---
        key_used, adj = _find_adjacency_matrix(w, N, Y, X)
        print(f"[info] using adjacency from key: {key_used}, shape={adj.shape}")

        present = Pfin.reshape(S, -1).sum(axis=1) > 0
        spp_ids = np.where(present)[0]

        comps_grid = []
        comps_graph = []
        for s in spp_ids:
            mask2d = Pfin[s]
            cg = connected_components_grid(mask2d)
            vec = mask2d.reshape(-1)
            cG = connected_components_graph(vec, adj)
            comps_grid.append(cg)
            comps_graph.append(cG)

        comps_grid = np.array(comps_grid, int)
        comps_graph = np.array(comps_graph, int)

        diff = comps_grid - comps_graph
        summary = {
            "world": args.world,
            "file": w.path.name,
            "adjacency_key_used": key_used,
            "n_extant": int(len(spp_ids)),
            "frac_graph_less_fragmented": float((diff > 0).sum() / len(diff)),
            "frac_equal": float((diff == 0).sum() / len(diff)),
            "frac_graph_more_fragmented": float((diff < 0).sum() / len(diff)),
        }
        (out_world / "connectivity_dual_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        # scatter plot grid-components vs graph-components
        plt.figure(figsize=(5, 5))
        plt.scatter(comps_grid, comps_graph, s=5, alpha=0.5)
        plt.xlabel("# components (grid, 4-neighbour)")
        plt.ylabel("# components (dispersal network)")
        plt.title("Geometric vs network connectivity")
        plt.tight_layout()
        plt.savefig(out_world / "components_grid_vs_graph.png", dpi=180)
        plt.close()

        print("[done] connectivity dual for world", args.world, "→", out_world)


if __name__ == "__main__":
    main()
