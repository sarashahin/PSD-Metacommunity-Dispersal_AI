# `models/` — Architecture, Inference, Ablation, Evaluation & Figures

![project badges](badges.svg)

This directory holds the MetaDiffusion **model code** plus every script for
**inference (reconstruction)**, **ablation**, **multi-metacommunity evaluation**, and
**figure generation**. Read [`../README.md`](../README.md) first for the overall
pipeline and path conventions.

> **Path note:** scripts default to `--stage2-dir AI_simulation/stage2`; in this repo
> pass `--stage2-dir stage2_objective_AI_meta` (or omit — `find_stage2_dir` auto-detects
> by locating `models/ecodiffusion.py` + `configs/config.py`). Checkpoints,
> `results/data/`, and `reconstructions_*/` are external (not in git).

---

## 1. Model architecture (library modules — not run directly)

| File | Role |
|---|---|
| `env_encoder.py` | Environmental CNN (spatial autocorrelation, multi-scale + attention) |
| `interaction_encoder.py` | Species-interaction GNN over the `C_topk` graph |
| `temporal_encoder.py` | Temporal Transformer over the `P_t` history |
| `diffusion.py` | Diffusion process (cosine schedule, DDIM sampling) |
| `unet.py` | U-Net denoiser (species as channels) |
| `ecodiffusion.py` | Combined non-spatial model (`create_fixed_model`) — v6/v7 line |
| `ecodiffusion_spatial_cond.py` | **Spatial-conditioned** model (`create_spatial_cond_model`), U-Net `input_proj` = 4 in-channels (noisy + obs_mask + env-suitability + obs-decay). Used by current training/inference. |
| `ecodiffusion_sample_spatial.py` | Sampler `sample_spatial(n_samples=…)` for the spatial model |

`REVIEW_ANALYSIS.md` documents the collate/edge-index fixes and curriculum pseudocode.

---

## 2. Reconstruction (inference)

All generators load a checkpoint, read a truth metacommunity `.npz`, sparsify its last frame
to a set of "observations", then run the diffusion sampler `--n-ensemble` times and
save the ensemble mean + all members.

### `generate_reconstructions_spatial.py` (current, fixed-budget)
```bash
python generate_reconstructions_spatial.py \
    --checkpoint  <CKPT>.pt \
    --truth-npz   ../../results/data/<world>.npz \
    --output-dir  ../../reconstructions_spatial_149/<world_stem> \
    --fixed-budgets 5 10 \
    --n-ensemble  8 \
    --mode        inpaint \
    --ddim-steps  50 \
    --eta         0.15
```
Defaults: `--output-dir ./reconstructions_spatial`, `--fixed-budgets 5 10`,
`--mode inpaint` (choices `inpaint|soft_inpaint|extrapolate`),
`--repaint-iterations 2`, `--n-ensemble 8`, `--chunk-size 200`, `--ddim-steps 50`,
`--eta 0.15`, `--rng-seed 42`, `--device` auto. Writes `recon_fixed_b{K}.npz` (keys
`mean, noisy_input, sample_mode, n_ensemble`) and `recon_fixed_b{K}_samples.npz`
(adds `samples` = `(n_ensemble, S, Y, X)`). Asserts the checkpoint is a **spatial**
model (`input_proj.in_channels == 4`).

### `generate_reconstructions_proportional.py` (proportional observations)
Thin wrapper reusing the spatial sampler; instead of keeping K cells it keeps each
occupied cell with probability `p`:
```bash
python generate_reconstructions_proportional.py \
    --checkpoint <CKPT>.pt --truth-npz ../../results/data/data_eval_unseen/<stem>.npz \
    --output-dir ../../reconstructions_proportional_p30_n50/<stem> \
    --obs-prob 0.30 --n-ensemble 50 --mode inpaint --ddim-steps 50 --eta 0.15
```
Defaults: `--obs-prob 0.10`, `--obs-min 1`. Writes `recon_prop_p{p:.2f}.npz` /
`recon_prop_p{p:.2f}_samples.npz`.

### Reproducing the user's reconstruction directories
| Directory | Command | Meaning |
|---|---|---|
| `reconstructions_proportional_k5_n50/<stem>` | `generate_reconstructions_spatial.py --fixed-budgets 5 --n-ensemble 50` | fixed **K=5**, **50-member ensemble** (`n50`) — feeds calibration/recall |
| `reconstructions_proportional_n50/<stem>` | `generate_reconstructions_proportional.py --obs-prob 0.10 --n-ensemble 50` | **10 %** observed, **50-member ensemble** |
| `reconstructions_proportional_p30_n50/<stem>` | `generate_reconstructions_proportional.py --obs-prob 0.30 --n-ensemble 50` | **30 %** observed, **50-member ensemble** |
| `reconstructions_spatial_149/<stem>` | `generate_reconstructions_spatial.py` with the **epoch-149** checkpoint (`--n-ensemble 8`) | training/seen metacommunities |
| `reconstructions_unseen/<stem>` | `generate_reconstructions_spatial.py --fixed-budgets 5 10 --n-ensemble 8` on `data_eval_unseen/` | unseen metacommunities, **8-member ensemble** (feeds map/posterior figures) |

> **`n50` = ensemble of 50 members, not "50 metacommunities".** The three `*_n50`
> directories were generated with `--n-ensemble 50` and drive the calibration / PIT /
> recall figures; the map and posterior figures use the default **8**-member sets — so
**K=5 was run at both ensemble sizes** (8 in `reconstructions_unseen`, 50 in
`reconstructions_proportional_k5_n50`). This is
> confirmed by the `n_ens` column in the committed result CSVs
> (`figures_map_stage2_new/unseen_eval/calibration*/pit_*.csv` = 50;
> `posterior_*/*_metrics_*.csv` = 8).

**Ensembles:** `--n-ensemble` sets the number of stochastic samples. **Two sizes were used
in this project:** the default **8** for the per-species map / posterior figures
(`reconstructions_unseen`, `reconstructions_spatial_149`), and **50** for the calibration /
PIT / recall analyses (the `*_n50` directories — for K=5, p=0.10 and p=0.30; the `n50`
suffix literally means "ensemble of 50"). Diversity comes from fresh Gaussian noise +
stochastic DDIM (`--eta 0.15`); only `--rng-seed` (obs sparsification) is fixed, so the
diffusion noise uses ambient torch RNG.

### Coverage
```bash
python compute_ensemble_coverage_stratified.py \
    --truth-npz ../../results/data/data_eval_unseen/<stem>.npz \
    --samples-npz ../../reconstructions_unseen/<stem>/recon_fixed_b5_samples.npz \
    --K 5 --calibrate match_truth \
    --output-csv ../../figures_map_axel_stage2_new/coverage_stratified_K5.csv
```
Strata: Trivial (range ≤ K), Sparse (K<range≤2K), Real (>2K); reports mean/union
recall + pixel coverage. `--calibrate {match_truth|2x_truth|fixed_05}`.

---

## 3. Unseen metacommunities, metacommunity selection & multi-metacommunity drivers

```bash
# pick inference worlds + build a regenerate script
python pick_inference_worlds.py --truth-dir ../../results/data \
    --manifest ../../figures_map_axel_stage2_new/inference_worlds_manifest_REV2.csv
python build_regenerate_script.py <manifest.csv> <out_script.sh>   # positional args

# wide-range species (input to eval & figures)
python build_wide_range_csv.py --K 5      # -> wide_range_species.csv

# end-to-end multi-world reconstruction + eval (10 worlds, K=5)
bash run_multiworld.sh

# 1x and 2x multi-world summaries
python multi_world_v7_evaluation.py --wide-range-csv <wide_range.csv> \
    --recon-dir-pattern './reconstructions_v7_inpaint_{world_stem}_stage2' \
    --truth-dir ../../results/data --K 5 --top-n-worlds 10 \
    --output-csv ../../figures_map_axel_stage2_new/multi_world_K5_summary.csv
python multi_world_2x_evaluation.py ... --output-csv .../multi_world_K5_2x_summary.csv
```
`run_multiworld.sh` and the regenerate scripts default the checkpoint to
`stage2_outputs_new/checkpoints/best_model.pt` — edit it to your epoch-149 FIXAB
checkpoint. The committed `AI_simulation/stage2/models/regenerate_all_inference_worlds_SPATIAL.sh`
is a hand-listed 18-metacommunity variant (adapt `STAGE2_DIR` to `stage2_objective_AI_meta`).

---

## 4. Ablation study

Two pipelines exist. The **current** one removes a single input predictor:

```bash
# 1) run ablation over 3 worlds (FULL, NO_HISTORY, NO_NETWORK, NO_ENV, NO_SPECIES_FEATS)
bash run_multi_world_ablation.sh          # writes ablation_v7_<label>_stage2_inpaint/
#   (or a single world:)
python run_ablation_v7.py --checkpoint <CKPT>.pt \
    --truth-npz ../../results/data/<world>.npz \
    --output-dir ../../ablation_v7_world5 \
    --K 5 --variants FULL NO_HISTORY NO_NETWORK NO_ENV NO_SPECIES_FEATS \
    --n-ensemble 8 --verbose

# 2) metrics CSV
python Archive/validate_ablation.py --ablation-dir ../../ablation_v7_world5_stage2_inpaint \
    --truth-npz ../../results/data/<world5>.npz --K 5 --calibrate match_truth \
    --output-csv ../../ablation_v7_world5_stage2_inpaint/ablation_metrics_1x.csv

# 3) figures (fig_A1..fig_E)
python Archive/ablation_visualize_all.py \
    --metrics-csv ../../ablation_v7_world5_stage2_inpaint/ablation_metrics_1x.csv \
    --ablation-dir ../../ablation_v7_world5_stage2_inpaint \
    --truth-npz ../../results/data/<world5>.npz --K 5 \
    --output-dir ../../ablation_v7_world5_stage2_inpaint/figures_all_1x
```
`run_ablation_v7.py` defaults: `--K 5`, `--mode inpaint`, `--repaint-iterations 2`,
`--n-ensemble 8`, `--chunk-size 200`, `--ddim-steps 50`, `--eta 0.5`, `--rng-seed 42`.
Variants zero a single conditioning input (history / GNN edges / env / species-feats).

The **legacy v2/v5** study (17 conditions incl. Poisson observation regimes) and its
figures — which write to `stage2_ablation_figures/` — live in `Archive/`; see
[`Archive/README.md`](Archive/README.md).

---

## 5. Figures

Run from `stage2_objective_AI_meta/` (or `models/` where noted). Figure/calibration
scripts are CPU-only.

### Per-species ecological maps (unseen-eval) → `Fig02_three_map_K5_<stem>.png`
```bash
python models/axel_per_species_map_ecological.py \
    --truth-dir results/data/data_eval_unseen \
    --recon-dir-pattern './reconstructions_unseen/{world_stem}' \
    --world-stem <STEM> --figure-style three_map --threshold-mode match_truth \
    --K 5 --n-species 5 \
    --output-path figures_map_axel_stage2_new/unseen_eval/Fig02_three_map_K5_<STEM>.png
```
Loop over `results/data/data_eval_unseen/eval_unseen_manifest.csv` (column 1) to
regenerate the whole folder. Defaults: `--figure-style three_map` (choices
`three_map|grid|both`), `--threshold-mode match_truth` (`match_truth|v3|fixed`),
`--threshold 0.80` (fixed only), `--n-species 5`, `--K 10`.

### Cross-metacommunity summary → `Fig03_cross_world_summary_K*.{png,csv}`
```bash
python models/axel_cross_world_summary_figure.py --truth-dir results/data \
    --recon-dir-pattern './reconstructions_unseen/{world_stem}' --K 5 --n-species 5 \
    --output-path figures_map_axel_stage2_new/unseen_eval/Fig03_cross_world_summary_K5.png \
    --output-csv  figures_map_axel_stage2_new/unseen_eval/Fig03_cross_world_summary_K5.csv
# ..._metrics.py for the metrics variant
```

### Ecological distribution (KS) → `dist_K*_indist/three_distributions_v3.*`
```bash
python models/axel_ecological_distribution_figure.py \
    --wide-range-csv figures_map_axel_stage2_new/unseen_eval/wide_range_indist_K5.csv \
    --recon-dir-pattern './reconstructions_unseen/{world_stem}' --truth-dir results/data \
    --method v3 --K 5 --top-n-worlds 30 \
    --output-dir figures_map_axel_stage2_new/unseen_eval/dist_K5_indist
```

### Posterior / ensemble per-species + conditional PIT
```bash
python models/posterior_per_species.py --truth-dir results/data/data_eval_unseen \
    --recon-dir-pattern './reconstructions_unseen/{world_stem}' --world-stem <STEM> \
    --K 5 --n-species 4 --n-samples-shown 3 \
    --output-path figures_map_axel_stage2_new/unseen_eval/posterior_per_species/Fig_posterior_K5_<STEM>.png \
    --csv-path    figures_map_axel_stage2_new/unseen_eval/posterior_per_species/posterior_metrics_K5.csv
```
The `--csv-path` PIT CSVs (`patches_pctile, spread_pctile, n_obs, n_ens`) feed:

### Calibration figures
```bash
# PIT / rank histogram
python models/pit_calibration_histogram.py --csv <pit_*.csv> --stats patches spread \
    --bins 10 --label 'K=5' --output figures_map_axel_stage2_new/unseen_eval/calibration/Fig_calibration_K5.png
# calibration vs number of records
python models/calibration_vs_nobs.py --csv <pit_*.csv> \
    --output figures_map_axel_stage2_new/unseen_eval/calibration/Fig_calibration_vs_records.png
# bootstrap KS + count-calibration (paths hardcoded to unseen dirs; args = K N_BOOT)
python axel_bootstrap_and_calibration.py 5 2000     # -> unseen_eval/calibration_diag_K5.{csv,png}
python baseline_smoother_compare.py 5               # -> unseen_eval/baseline_compare_K5.csv
```

### Recall vs observations → `recall/Fig_recall_vs_observations.*`
```bash
python models/recall_vs_observations.py --truth-dir results/data/data_eval_unseen \
    --world-stems <STEM1> <STEM2> ... --labels "K=5" "p=0.10" "p=0.30" \
    --recon-dir-patterns './reconstructions_proportional_k5_n50/{world_stem}' \
                         './reconstructions_proportional_n50/{world_stem}' \
                         './reconstructions_proportional_p30_n50/{world_stem}' \
    --recon-filenames recon_fixed_b5_samples.npz recon_prop_p0.10_samples.npz recon_prop_p0.30_samples.npz \
    --x both --include-ens \
    --output figures_map_axel_stage2_new/unseen_eval/recall/Fig_recall_vs_observations \
    --csv    figures_map_axel_stage2_new/unseen_eval/recall/recall_vs_observations.csv
```

### Final unified figure suite → `final_unified_149_K5/Fig01..Fig15`
```bash
python models/make_figure1_honest_map.py \
    --multi-world-csv    figures_map_axel_stage2_new/unseen_eval/multi_world_K5_summary.csv \
    --multi-world-csv-2x figures_map_axel_stage2_new/unseen_eval/multi_world_K5_2x_summary.csv \
    --truth-dir results/data --recon-dir-pattern './reconstructions_unseen/{world_stem}' \
    --K 5 --output-dir figures_map_axel_stage2_new/unseen_eval/final_unified_149_K5
```

### Threshold / routing diagnostics (optional)
`axel_three_axis_threshold_alignment.py`, `axel_fine_threshold_sweep.py`,
`axel_adaptive_routing_v2_multifeature.py`, `axel_adaptive_routing_v3_bucketclassifier.py`,
`axel_adaptive_threshold_diagnosis.py`, `axel_probability_mass_diagnosis.py` — each
takes `--wide-range-csv --recon-dir-pattern --truth-dir --K --top-n-worlds --output-dir`
and writes threshold-sweep PNGs. `models/diagnose_recon_axel_map.py <recon.npz> [--truth <t>]`
is a stdout health check.

> **Accuracy:** `--threshold-mode match_truth` (top-N where N = true range size) is the
> honest "what the model recovered" panel. Reproduce with the **epoch-149** checkpoint
> to match the `final_unified_149_K5` figures (threshold p ≥ 0.9, KS=0.047).
