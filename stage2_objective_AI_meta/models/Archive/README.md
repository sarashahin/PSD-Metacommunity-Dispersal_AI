# `models/Archive/` — Older pipeline versions (v2 / v5 / v6 / v7)

![project badges](badges.svg)

Predecessors of the current spatial pipeline, kept for provenance and because a few are
still invoked by the current tooling (noted below). The **current** equivalents live in
`../` (`generate_reconstructions_spatial.py`, `run_ablation_v7.py` calls into here, etc.).

## Model / sampling (non-spatial line)
- `encoders.py`, `attention.py` — earlier encoder/attention blocks.
- `ecodiffusion_sample_v6_map_axel.py` — v6 sampler (`sample_v6`, extrapolation).
- `ecodiffusion_sample_v7_inpaint.py` — v7 inpaint sampler (`sample_v7`). **Still used**:
  `../run_ablation_v7.py` copies it into `models/` and imports `sample_v7`.

## Reconstruction (older)
- `generate_reconstructions_v7_inpaint.py` — non-spatial predecessor of
  `generate_reconstructions_spatial.py`; adds `--history-sparsify {all_frames,last_only}`
  (default `all_frames`), `--eta 0.5`. Referenced by `run_multiworld.sh` and the
  regenerate scripts.
- `generate_reconstructions_v6_map_axel.py` — v6; adds `--poisson-p` and
  `--mode {echo,extrapolate,guided}`.
- `make_ab14_ensemble_new.py` — builds the Truth|Observed|Reconstructed 3-panel figure.
- `compute_ensemble_coverage.py` — non-stratified coverage (predecessor of
  `../compute_ensemble_coverage_stratified.py`).

## Ablation (v2/v5 — writes `stage2_ablation_figures/`)
This is the pipeline that produces the user's `stage2_ablation_figures/` directory.
```bash
# run the 17-condition study (env/temporal/interaction + gap/sparse/Poisson regimes)
python stage2_ablation_study.py \
    --ibm-dir ../../results/data/ \
    --checkpoint <CKPT>.pt \
    --stage2-dir ../.. \
    --output-dir ../../stage2_ablation_v5_poisson/ \
    --n-samples 8 --n-worlds 8 --only-poisson         # -> ablation_v2_results.json + figures

# make the AB1..AB14 figures into stage2_ablation_figures/
python make_ablation_figures.py ablation_v5_merged.json \
    --truth-npz ../../results/data/<world>.npz --ab14-demo --ab14-p-obs 0.001
#   --output-dir defaults to stage2_ablation_figures/
```
Helpers `poisson_corrected_ablation.py` and `poisson_diagnostic_ablation.py` are the
biomass→individuals conversion references (the v5 "diagnostic" divides biomass by
`BODY_MASS=1e-4` before Poisson thinning — this is what makes `p_obs=0.01` saturate).
Run either as `python <file>.py <world_training.npz>` for a standalone diagnostic.

- `validate_ablation.py`, `ablation_visualize_all.py` — the **current v7** validator +
  visualizer (used by `../run_multi_world_ablation.sh`); see [`../README.md`](../README.md) §4.
- `select_wide_range_worlds.py` — predecessor of `../build_wide_range_csv.py`.
- `stage2_ablation_study.py`, `stage2_axel_community_validation.py`,
  `stage2_real_model_test.py`, `stage2_validation_visualization.py`,
  `reconstruct_history_newtrain.py`, `objective2_figure_suite.py`,
  `make_cross_world_summary_figure.py`, `fix_species_features_stage2_visulise.py` —
  earlier validation/figure experiments, superseded by the `axel_*` scripts in `../`.

> Prefer the current scripts in `../` unless you are specifically reproducing an older
> result. When you do run an Archive script, pass `--stage2-dir stage2_objective_AI_meta`.
