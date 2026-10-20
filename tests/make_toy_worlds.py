#!/usr/bin/env python3
"""
Write tiny synthetic metacommunities ("worlds") in the same .npz format as the
Stage-1 simulation, so the Stage-2 code can be tested without the real data.

The numbers are random and have no ecological meaning. They only have the
right keys, shapes and dtypes for data_preprocessing.SimulationWorld.

Usage:
    python tests/make_toy_worlds.py --out-dir /tmp/toy_data --n-worlds 8
"""
import argparse
from pathlib import Path

import numpy as np

BUDGETS = (1, 5, 10, 20, 50, 100)


def make_world(rng, n_species, n_t, grid=20, k_links=4):
    yy, xx = np.mgrid[0:grid, 0:grid] / (grid - 1)

    # Smooth per-species suitability field: a random Gaussian bump
    cy, cx = rng.uniform(0, 1, (2, n_species, 1, 1))
    width = rng.uniform(0.1, 0.4, (n_species, 1, 1))
    env = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * width ** 2)).astype(np.float32)

    # Biomass grows toward the suitability field over time, with noise
    growth = np.linspace(0.2, 1.0, n_t)[:, None, None, None]
    noise = rng.uniform(0.6, 1.4, (n_t, n_species, grid, grid))
    biomass = env[None] * growth * noise
    biomass[biomass < 0.35] = 0.0
    biomass = biomass.astype(np.float32)

    presence = (biomass > 0).astype(np.int8)
    p_last = presence[-1]

    # Observation masks: up to K occupied cells per species
    masks = {}
    for k in BUDGETS:
        m = np.zeros_like(p_last)
        for s in range(n_species):
            cells = np.flatnonzero(p_last[s])
            if cells.size:
                pick = rng.choice(cells, size=min(k, cells.size), replace=False)
                m[s].flat[pick] = 1
        masks[f"obs_mask_{k}"] = m

    c_idx = rng.integers(0, n_species, (n_species, k_links)).astype(np.int64)
    c_w = rng.normal(0, 0.1, (n_species, k_links)).astype(np.float32)

    return dict(
        IBM_B=biomass,
        P_t=presence,
        P_last_final=p_last,
        ENV_r_field=env,
        C_topk_idx=c_idx,
        C_topk_w=c_w,
        prevalence_final=(p_last.mean(axis=(1, 2))).astype(np.float32),
        deg_in=np.bincount(c_idx.ravel(), minlength=n_species).astype(np.float32),
        deg_out=np.full(n_species, k_links, dtype=np.float32),
        r_base=rng.uniform(0.5, 1.5, n_species).astype(np.float32),
        Y=np.int64(grid),
        X=np.int64(grid),
        BODY_MASS=np.float64(1e-4),
        **masks,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-worlds", type=int, default=8)
    ap.add_argument("--n-species", type=int, default=40)
    ap.add_argument("--n-timesteps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for i in range(a.n_worlds):
        # Vary species count a little so padding to max_species is exercised
        s = a.n_species - (i % 3) * 2
        np.savez_compressed(out / f"toy_world_{i:02d}_training.npz",
                            **make_world(rng, s, a.n_timesteps))
    print(f"wrote {a.n_worlds} toy worlds to {out}")


if __name__ == "__main__":
    main()
