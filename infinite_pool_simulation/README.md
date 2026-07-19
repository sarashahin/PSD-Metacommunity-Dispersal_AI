# `infinite_pool_simulation/` — Stage 1: Metacommunity Simulation

![project badges](badges.svg)

Stage 1 of the pipeline: the **individual-based (IBM) Lotka–Volterra metacommunity
simulation** that produces the "metacommunity" datasets (`*_training.npz`) consumed by the
Stage-2 AI (`../stage2_objective_AI_meta/`).

## ⚠️ Status: placeholder stubs in this repository

Every `.py`/`.sh` in this folder is currently a **placeholder** (each file contains only
`<!-- content from <name> -->`). The real simulation code lives in the author's local
workspace and was **not committed** here. This README documents the **interface** so the
pipeline is reproducible once the real Stage-1 code is dropped back into this folder.

To restore: copy the real modules over these stubs (same filenames), keeping
`run_all_rps.py` as the entry point.

## Entry point & how it's driven

The simulation is launched via **`run_all_rps.py`**, normally through the sweep drivers
in [`../scripts/`](../scripts/README.md) (one process per metacommunity / per GPU):

```bash
python run_all_rps.py \
    --pool 22510000 --skip-psd --skip-ode \
    --tmax 10000 --record 200 --no-movie \
    --ibm-frac-multi 0.05 --ibm-window-steps 1000 --ibm-max-attempts 10000 \
    --ibm-richness-cap None --save-env-field --ibm-record-mode full \
    --obs-budgets 1,5,10,20,50,100 \
    ... (per-world: --ls --vr --thr --env --dr --ld) ...
```

The parameter axes (set by `../scripts/sweep_worlds.py`): environment length-scale
`ls`, variance `vr`, competition threshold `thr`, environment seed `env`
(train: 123/456; unseen-eval: 789/2024), dispersal rate `dr`, long-distance dispersal
`ld`. Output filename convention:
`pool22510000_batch<a/b>_ls{}_vr{}_thr{}_env{}_grid20x20_dr{}_ld{}_training.npz`.

## Output contract (what Stage 2 needs)

Each metacommunity `.npz` must contain at least (see
[`../stage2_objective_AI_meta/README.md`](../stage2_objective_AI_meta/README.md) §2):

- `IBM_B` — biomass `(T, S, Y, X)` **(required)**
- `P_t` — presence over time; `P_last_final` — final distribution **(Stage-2 target)**
- `ENV_r_field` — per-species environmental niche `(S, Y, X)`
- `C_topk_idx` / `C_topk_w` — interaction graph
- `obs_mask_{1,5,10,20,50,100}` — sparse-observation masks (used by Stage-2 infill/inference)
- species stats: `prevalence_final`, `deg_in`, `deg_out`, `r_base`, …
- scalars: `DISPERSAL_RATE`, `LONG_DISTANCE_PROB`, `ENV_length_scale`, `ENV_var_r`,
  `BODY_MASS` (≈1e-4), `gamma`
- grid: 20×20, 50 timesteps

## Module map (by filename)

`main.py` / `run_all_rps.py` (entry & orchestration) · `models_ibm.py`, `models_ode.py`,
`models_psd.py`, `models_psd2.py` (IBM / ODE / PSD dynamics) · `dispersal.py`,
`environment.py`, `euler_simple_safe.py` (dispersal kernel, env field, integrator) ·
`assembly_stepwise_ibm.py`, `assembly_stepwise_psd2.py`, `assembly_utils.py` (community
assembly) · `config.py`, `config_base.py`, `config_utils.py` (configuration) ·
`analysis.py`, `analysis_extended.py`, `utils.py`, `utils_vis.py`, `visualization.py`,
`colour_bank.py` (analysis & plotting) · `gpu_patch.py` (CUDA/CuPy) · `quick_check.py`,
`test_models.py`, `test_output.py` (tests) · `run_rps_dynamics.py`,
`regenerate_all_inference_worlds_SPATIAL.sh` (drivers).

> Because these are stubs, the repo's CI (`flake8`) flags them as syntax errors — that
> is expected and unrelated to Stage 2.
