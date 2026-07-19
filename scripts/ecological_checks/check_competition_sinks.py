# scripts/check_competition_sinks.py

from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from load_training_worlds import DataCatalog


def _get_any(w, keys):
    for k in keys:
        try:
            return w.get(k)
        except KeyError:
            continue
    return None


def _finite(x):
    return np.isfinite(x)


def compute_r_decomposition(w, Pfin):
    """
    Return r_env, r_self, r_comp, r_net with shapes (S,Y,X).
    Uses ENV_r_field or r_base/r_mean + C_topk_idx/C_topk_w + B_last.
    """
    B = np.asarray(w.get("B_last"), dtype=float)  # (S,Y,X)
    S, Y, X = B.shape

    # competition sparsity
    Cidx = np.asarray(w.get("C_topk_idx"), dtype=int)    # (S,K)
    Cw   = np.asarray(w.get("C_topk_w"),  dtype=float)   # (S,K)
    assert Cidx.ndim == 2 and Cw.shape == Cidx.shape

    # environment
    r_env = _get_any(w, ["ENV_r_field", "r_field"])
    if r_env is not None:
        r_env = np.asarray(r_env, dtype=float)
        if r_env.shape != (S, Y, X):
            raise RuntimeError(f"Unexpected ENV_r_field/r_field shape: {r_env.shape}")
    else:
        # fallback: species scalar baseline (r_base or r_mean)
        r0 = _get_any(w, ["r_base", "r_mean"])
        if r0 is None:
            raise RuntimeError("No ENV_r_field/r_field/r_base/r_mean → cannot decompose r_net")
        r0 = np.asarray(r0, dtype=float)
        if r0.ndim == 0:
            r_env = np.full((S, Y, X), float(r0))
        elif r0.ndim == 1 and r0.shape[0] == S:
            r_env = r0[:, None, None] * np.ones((1, Y, X), dtype=float)
        else:
            raise RuntimeError(f"Unexpected r0 shape: {r0.shape}")

    # self-limitation term
    SELF_C = 1.0  # keep consistent with LV self-term
    r_self = SELF_C * B  # this will be subtracted

    # competition term
    B_comp = B[Cidx]  # (S,K,Y,X)
    r_comp = (B_comp * Cw[:, :, None, None]).sum(axis=1)

    # net growth
    r_net = r_env - r_self - r_comp
    return r_env, r_self, r_comp, r_net


def classify_sinks(Pfin, r_env, r_self, r_comp, r_net):
    """
    For occupied cells, classify sink patches into:
        0: empty
        1: source (r_net >= 0)
        2: env-sink (r_env < 0, occupied)
        3: comp-sink (r_env >= 0, r_env - r_self > 0, r_net < 0)
        4: self-sink (r_env >= 0, r_env - r_self <= 0, r_net < 0)
    """
    S, Y, X = Pfin.shape
    cls = np.zeros((S, Y, X), dtype=np.int8)

    occ = Pfin.astype(bool)
    sink = (r_net < 0) & occ
    source = (r_net >= 0) & occ

    cls[source] = 1  # source cells

    # sink subtypes
    env_neg = (r_env < 0) & sink
    self_eff = r_env - r_self  # growth if only env + self
    comp_sink = (~env_neg) & (self_eff > 0) & sink
    self_sink = (~env_neg) & (self_eff <= 0) & sink

    cls[env_neg] = 2
    cls[comp_sink] = 3
    cls[self_sink] = 4

    return cls


def per_world_competition(dc, world_idx: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    with dc[world_idx] as w:
        Pfin = w.get_final_occupancy().astype(bool)  # (S,Y,X)
        S, Y, X = Pfin.shape
        r_env, r_self, r_comp, r_net = compute_r_decomposition(w, Pfin)
        cls = classify_sinks(Pfin, r_env, r_self, r_comp, r_net)

        # summary over occupied cells only
        occ = Pfin
        env_sinks = (cls == 2) & occ
        comp_sinks = (cls == 3) & occ
        self_sinks = (cls == 4) & occ
        sources    = (cls == 1) & occ

        occ_total = float(occ.sum())
        summary = {
            "world": world_idx,
            "file": w.path.name,
            "DISPERSAL_RATE": float(np.array(w.get("DISPERSAL_RATE")).squeeze()),
            "LONG_DISTANCE_PROB": float(np.array(w.get("LONG_DISTANCE_PROB")).squeeze()),
            "occupied_cells": occ_total,
            "sources": int(sources.sum()),
            "env_sinks": int(env_sinks.sum()),
            "comp_sinks": int(comp_sinks.sum()),
            "self_sinks": int(self_sinks.sum()),
        }
        for key in ["sources", "env_sinks", "comp_sinks", "self_sinks"]:
            summary[key + "_fraction"] = (
                summary[key] / occ_total if occ_total > 0 else 0.0
            )

        # choose species with most comp-sink cells for visualisation
        comp_by_species = comp_sinks.reshape(S, -1).sum(axis=1)
        if comp_by_species.max() > 0:
            sid = int(np.argmax(comp_by_species))
        else:
            # fallback: most env-sinks, or most occupied
            env_by_species = env_sinks.reshape(S, -1).sum(axis=1)
            if env_by_species.max() > 0:
                sid = int(np.argmax(env_by_species))
            else:
                occ_by_species = occ.reshape(S, -1).sum(axis=1)
                if occ_by_species.max() > 0:
                    sid = int(np.argmax(occ_by_species))
                else:
                    sid = 0

        # save summary json
        (outdir / "competition_summary.json").write_text(json.dumps(summary, indent=2))

        # ---------- nicer figure with discrete colours and labels ----------
        panel = cls[sid]  # 0 empty, 1 source, 2 env-sink, 3 comp-sink, 4 self-sink

        # discrete colormap for 0–4
        # 0: empty (dark), 1: source (green), 2: env-sink (yellow),
        # 3: comp-sink (red), 4: self-sink (blue)
        cmap = mcolors.ListedColormap(
            ["black", "limegreen", "gold", "red", "dodgerblue"]
        )
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(panel, origin="upper", cmap=cmap, norm=norm)

        ax.set_xlabel("patch x")
        ax.set_ylabel("patch y")
        ax.set_title(f"World {world_idx}, species {sid}", fontsize=10)

        # colourbar with class labels
        cbar = fig.colorbar(
            im,
            ax=ax,
            boundaries=bounds,
            ticks=[0, 1, 2, 3, 4],
            fraction=0.046,
            pad=0.04,
        )
        cbar.ax.set_yticklabels(
            ["empty (0)", "source (1)", "env-sink (2)", "comp-sink (3)", "self-sink (4)"],
            fontsize=8,
        )

        # keep everything inside figure (no clipping of title / colourbar)
        fig.tight_layout()
        out_png = outdir / f"species_{sid}_competition_map.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Decompose r_net into env/self/competition and classify sink patches."
    )
    ap.add_argument("--root", required=True, help="results/data folder")
    ap.add_argument(
        "--out",
        required=True,
        help="output folder (e.g. results/ecology_competition)",
    )
    ap.add_argument("--world", type=int, default=0, help="catalog world index")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    dc = DataCatalog(str(root))

    out_world = out / f"world{args.world:03d}"
    per_world_competition(dc, args.world, out_world)
    print("[done] competition sinks for world", args.world, "→", out_world)


if __name__ == "__main__":
    main()
