# `figures_map_stage2_new/` — Final Stage-2 Figures (unseen-eval)

![project badges](badges.svg)

This folder holds the **rendered figures and summary CSVs** for Stage 2, evaluated on
the **unseen** metacommunities (new environment seeds 789/2024 the model never trained on).
It was transferred from the local working directory `figures_map_axel_stage2_new/`;
the scripts that produce these outputs still write to a directory named
`figures_map_axel_stage2_new/` (note the `_axel_`), so when you regenerate them either
point the scripts' `--output-*` at this folder or rename accordingly.

All generator commands live in [`../models/README.md`](../models/README.md) §"Figures".
Below is the map from **each output here → the exact script that produced it**, so any
figure can be reproduced. Run from `stage2_objective_AI_meta/`.

## `unseen_eval/` contents → producing script

| Output (file / subfolder) | Produced by | Key inputs |
|---|---|---|
| `Fig02_three_map_K5_<stem>.png` | `models/axel_per_species_map_ecological.py` (`--figure-style three_map`) | `reconstructions_unseen/<stem>/recon_fixed_b5_samples.npz` + `results/data/data_eval_unseen/<stem>.npz` |
| `Fig02_three_map_metrics_K5_<stem>.png` | `models/axel_per_species_map_ecological.py` (metrics overlay) | same |
| `Fig03_cross_world_summary_K{5,10}[_sp30][_metrics].{png,csv}` | `models/axel_cross_world_summary_figure.py` and `..._metrics.py` | `--truth-dir`, `--recon-dir-pattern` |
| `dist_K{5,10}_indist/three_distributions*_v3.{png,csv}` | `models/axel_ecological_distribution_figure.py` (`--method v3`) | `wide_range_species.csv`, recon dirs |
| `calibration/`, `calibration_p30/` (`Fig_calibration*.png`, `pit_*.csv`) | `models/pit_calibration_histogram.py` + `models/calibration_vs_nobs.py`, fed by `models/posterior_per_species.py --csv-path` | PIT CSVs |
| `calibration_diag_K{5,10}.{csv,png}`, `calibration_indist*_K10.csv` | `axel_bootstrap_and_calibration.py <K> <N_BOOT>` (paths hardcoded to unseen dirs) | `reconstructions_unseen/`, `data_eval_unseen/` |
| `baseline_compare_K{5,10}.csv` | `baseline_smoother_compare.py <K>` (diffusion vs Gaussian smoother) | same |
| `posterior_per_species/`, `posterior_PROP/` (`Fig_posterior_*`, `*_conditional.png`, `posterior_metrics_*.csv`) | `models/posterior_per_species.py` | recon dirs (fixed & proportional) |
| `recall/`, `recall_effort/` (`Fig_recall_*`, `recall_vs_observations.csv`, `rve_*.csv`) | `models/recall_vs_observations.py` | fixed + proportional recon dirs |
| `multi_world_K5_summary.csv` | `models/multi_world_v7_evaluation.py` | `wide_range_species.csv`, recon dirs |
| `multi_world_K5_2x_summary.csv` | `models/multi_world_2x_evaluation.py` (`--calibrate 2x_truth`) | same |
| `final_unified_149_K5/Fig01..Fig15*.png` + `*.csv` | `models/make_figure1_honest_map.py` | the two `multi_world_*` CSVs + recon dirs |
| `wide_range_indist_K{5,10}.csv` | `models/build_wide_range_csv.py --K {5,10}` | truth metacommunities |
| `range_estimator_diagnostic_K10.csv` | `range_estimator_diagnostic.py` | recon dirs |
| `training_range_audit_K5.csv` | `audit_training_range_distribution.py` | training metacommunities |
| `proportional_masks/obs_count_extremes_p0.10.png` | `models/proportional_observations.py` | truth metacommunities |
| `showcase/`, `showcase_p30/` | curated `posterior_per_species.py` / `axel_per_species_map_ecological.py` runs on low-dispersal metacommunities | recon dirs |

### Ensemble sizes behind these figures
The **calibration / PIT** (`calibration*/`, `calibration_p30/`) and **recall**
(`recall/`, `recall_effort/`) outputs come from the **50-member** ensemble reconstructions
(the `reconstructions_*_n50` folders — `n50` = "ensemble of 50", for K=5, p=0.10, p=0.30).
The **per-species map** (`Fig02_*`) and **posterior** (`posterior_*`) outputs come from the
default **8-member** ensembles (`reconstructions_unseen`). This is verifiable in the CSVs
here: the `n_ens` column is **50** in `calibration*/pit_*.csv` and **8** in
`posterior_*/*_metrics_*.csv`.

### The `_149_K5` naming
`final_unified_149_K5/` = figures built from the **epoch-149** FIXAB checkpoint at
fixed budget **K=5**. To reproduce byte-for-byte you must use the epoch-149 checkpoint
(the cov-det optimum threshold at that epoch is p ≥ 0.9, KS=0.047). See the training
section of [`../README.md`](../README.md) for what "149" means.

### Reproduce everything here
1. Train → epoch-149 FIXAB checkpoint (`../README.md` §3).
2. Generate `reconstructions_unseen/`, `reconstructions_proportional_*` (`../README.md` §5).
3. Run the figure scripts in [`../models/README.md`](../models/README.md) §"Figures",
   pointing `--output-*` into this folder.
