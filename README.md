# PSD Metacommunity — Dispersal + AI (MetaDiffusion)

![project badges](badges.svg)

A two-stage pipeline for **process-aware species-distribution modelling** with
uncertainty, built on a Lotka–Volterra / individual-based metacommunity simulation:

> **👋 New here, or not a machine-learning person?** Start with
> **[`GETTING_STARTED.md`](GETTING_STARTED.md)** — a plain-language walkthrough for
> ecologists: what this does, what you need, a 7-step quickstart, and a glossary that
> translates every AI term into one sentence.

1. **Stage 1 — Simulation.** An individual-based metacommunity model produces
   **metacommunities** — complete spatio-temporal species distributions (a grid of local
   communities linked by dispersal) with known environment, interactions, and dispersal.
   → [`infinite_pool_simulation/`](infinite_pool_simulation/README.md),
   [`scripts/`](scripts/README.md)
2. **Stage 2 — Objective AI.** A conditional **diffusion model** ("MetaDiffusion") trains
   on those simulated metacommunities and reconstructs species maps from sparse observations, with
   ensemble-based uncertainty. → [`stage2_objective_AI_meta/`](stage2_objective_AI_meta/README.md)

```
 Stage 1: sim params ─(scripts/sweep_worlds*.py → infinite_pool_simulation/run_all_rps.py)→ results/data/*_training.npz
 Stage 2: metacommunities ─(preprocess → train → reconstruct → evaluate → ablate → figures)→ figures_map_axel_stage2_new/
```

> **Terminology.** This documentation uses the ecological term **metacommunity** for
> each simulated dataset. The **code, filenames, and command-line flags** use the word
> **`world`** (`--world-stem`, `--top-n-worlds`, `{world_stem}`, `multi_world_*`,
> `data_eval_unseen`, `sweep_worlds.py`) — type those exactly as shown. One
> *metacommunity* = one `world` in the code.

## Repository layout

| Path | What | README |
|---|---|---|
| `infinite_pool_simulation/` | Stage-1 IBM simulation (⚠️ placeholder stubs in git; real code is local) | [link](infinite_pool_simulation/README.md) |
| `scripts/` | Metacommunity-generation sweeps + `ecological_checks/` validators | [link](scripts/README.md) |
| `stage2_objective_AI_meta/` | Stage-2 MetaDiffusion: preprocessing, training, inference, ablation, figures | [link](stage2_objective_AI_meta/README.md) |
| `stage2_objective_AI_meta/models/` | Model, sampling, reconstruction, ablation, evaluation, figure scripts | [link](stage2_objective_AI_meta/models/README.md) |
| `stage2_objective_AI_meta/figures_map_stage2_new/` | Rendered final figures (unseen-eval) | [link](stage2_objective_AI_meta/figures_map_stage2_new/README.md) |

## Quick start (Stage 2)

```bash
python -m venv lvmeta && source lvmeta/bin/activate
pip install torch numpy tqdm scikit-learn scipy matplotlib pyyaml

cd stage2_objective_AI_meta
python smoke_test_spatial.py --simulation-dir ../results/data          # sanity check data
python run_training.py --simulation-dir ../results/data \
    --output-dir ../stage2_outputs_SPATIAL_FIXAB --total-epochs 500 --batch-size 2
```

Then reconstruct, evaluate, ablate, and reproduce figures — full copy-paste commands are
in [`stage2_objective_AI_meta/README.md`](stage2_objective_AI_meta/README.md).

## Notes on reproducibility

- **External artifacts** (`results/data/`, checkpoints, `reconstructions_*/`) are large
  and **not stored in git**; the READMEs document exactly how each is produced.
- The scripts were authored under the path `AI_simulation/stage2/`; in this repo pass
  `--stage2-dir stage2_objective_AI_meta` (or rely on auto-detection).
- CI (`.github/workflows`) runs default flake8/Django templates that fail on the Stage-1
  placeholder stubs and the absence of a Django app — these failures are **pre-existing
  and unrelated** to the Stage-2 code.
