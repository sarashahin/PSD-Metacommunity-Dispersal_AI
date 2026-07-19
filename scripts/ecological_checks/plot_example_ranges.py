# scripts/ecological_checks/plot_example_ranges.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from load_training_worlds import DataCatalog

def plot_species_world(root, world_idx, species_ids, outdir):
    dc = DataCatalog(root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with dc[world_idx] as w:
        Pfin = w.get_final_occupancy().astype(bool)  # (S, Y, X)
        S, Y, X = Pfin.shape
        n_sites = Y * X

        for sid in species_ids:
            mask = Pfin[sid]
            n_patches = int(mask.sum())
            range_size = n_patches / n_sites

            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(mask, origin="upper")
            ax.set_title(
                f"world {world_idx}, species {sid}\n"
                f"{n_patches} patches (range_size={range_size:.4f})"
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(outdir / f"world{world_idx:03d}_species{sid:04d}.png", dpi=180)
            plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results/data folder")
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--species", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plot_species_world(args.root, args.world, args.species, args.out)
