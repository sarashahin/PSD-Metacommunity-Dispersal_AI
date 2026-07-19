# Stage 2 — Objective AI (MetaDiffusion) — Reproducible Pipeline

![project badges](badges.svg)

This directory contains **Stage 2** of the PSD Metacommunity project: a conditional
**diffusion model** ("MetaDiffusion") that learns ecological species-distribution
patterns from the Lotka–Volterra / individual-based metacommunity simulation, then
reconstructs species maps from sparse observations at inference time.

Everything below is written so that **anyone can reproduce the full pipeline from
scratch**, end to end, with copy-paste commands. Read this file first; each
sub-directory has its own `README.md` with the details for the scripts it holds.

> **👋 Not a machine-learning person?** Start with
> [`../GETTING_STARTED.md`](../GETTING_STARTED.md) — a plain-language walkthrough for
> ecologists (what this does, what you need, a 7-step quickstart, and a glossary that
> translates every AI term used here). This file is the technical reference; that one is
> the gentle on-ramp. A short glossary is also at the [end of this file](#10-glossary-plain-language).

> **These READMEs were reconstructed by reading the actual scripts** (argparse
> blocks, shell drivers, docstrings). Every command below reflects real argument
> names and defaults found in the code.

> **Terminology — "metacommunity" = "world" in the code.** In this documentation each
> simulated dataset is called a **metacommunity**: a 20×20 grid of local communities
> connected by dispersal, evolved over 50 timesteps (matching the project name,
> *PSD-Metacommunity-Dispersal*). The **code, filenames, and CLI flags** still use the
> word **`world`** (e.g. `--world-stem`, `--top-n-worlds`, `{world_stem}`,
> `data_eval_unseen`, `multi_world_*`, `sweep_worlds.py`) — those are literal and must
> be typed exactly as shown. So: prose says *metacommunity*; commands say *world*.

---

## 0. Important path & environment notes (READ THIS)

**Historical path vs. this repo.** The scripts were written when this folder lived
at `AI_simulation/stage2/`, so their hard-coded USAGE examples say
`--stage2-dir AI_simulation/stage2`. In this repository the same code lives at
`stage2_objective_AI_meta/`. Two consequences:

- Pass **`--stage2-dir stage2_objective_AI_meta`** (or omit it — most scripts
  auto-locate the stage-2 dir by searching for `models/ecodiffusion.py` +
  `configs/config.py` via `find_stage2_dir`).
- One shell script was committed under a doubly-nested legacy path:
  `models/AI_simulation/stage2/models/regenerate_all_inference_worlds_SPATIAL.sh`.
  Treat its internal paths as illustrative and adapt `STAGE2_DIR` to
  `stage2_objective_AI_meta`.

**External artifacts not stored in git** (they are large / machine-local):

| Artifact | What it is | Produced by |
|---|---|---|
| `results/data/*_training.npz` | Simulation "metacommunities" (training set) | Stage 1 simulation (see `../infinite_pool_simulation/`) |
| `results/data/data_eval_unseen/` | Unseen evaluation metacommunities + manifest | `design_unseen_eval_worlds.py` → `scripts/sweep_worlds_eval30.py` |
| `stage2_outputs_*/checkpoints/*.pt` | Trained model checkpoints | `run_training.py` |
| `reconstructions_*/` | Per-metacommunity inference outputs (`.npz`) | `models/generate_reconstructions_*.py` |
| `figures_map_axel_stage2_new/` | Final figures & summary CSVs | the `axel_*` / figure scripts |

Create a working root and run everything from it (paths below assume this layout):

```
<PROJECT_ROOT>/
├── stage2_objective_AI_meta/     # this code
├── results/data/                 # simulation metacommunities (*_training.npz)
│   └── data_eval_unseen/         # unseen eval metacommunities + manifest
├── stage2_outputs_.../checkpoints/
├── reconstructions_.../
└── figures_map_axel_stage2_new/
```

**Python environment** (no `requirements.txt` is shipped; these are the imports the
code actually uses):

```bash
python -m venv lvmeta && source lvmeta/bin/activate      # or conda
pip install torch numpy tqdm scikit-learn scipy matplotlib pyyaml
# torch.amp.autocast/GradScaler API implies torch >= ~2.3
```

`torch_geometric` is only needed by the **legacy** `data/preprocessing.py`, which is
**not** used by the active pipeline (see `data/README.md`).

**Device.** Every torch script takes `--device` (or auto-selects `cuda` if
available, else `cpu`). Training and reconstruction need a GPU for realistic runtime;
figure/calibration scripts are CPU-only.

---

## 1. The pipeline at a glance

```
 STAGE 1  (../infinite_pool_simulation, ../scripts)
   run_all_rps.py / sweep_worlds*.py  ──►  results/data/*_training.npz   (240 metacommunities, 20×20, 50 t-steps)
                                                     │
 STAGE 2  (this directory)                           ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ (a) PREPROCESS + SPLIT     data_preprocessing.py / author_clean_split │
   │ (b) TRAIN (4-phase curric) run_training.py  ──► stage2_outputs_*/ckpt │
   │ (c) UNSEEN METACOMMUNITIES design_unseen_eval_worlds.py ──► data_eval_unseen/
   │ (d) RECONSTRUCT (inference) models/generate_reconstructions_*.py ──► reconstructions_*/
   │ (e) EVALUATE / COVERAGE    models/multi_world_*_evaluation.py, compute_ensemble_coverage*
   │ (f) ABLATION               models/run_ablation_v7.py (+ Archive v2/v5) ──► stage2_ablation_figures/
   │ (g) FIGURES                models/axel_*.py ──► figures_map_axel_stage2_new/
   └──────────────────────────────────────────────────────────────────────┘
```

---

## 2. (a) Preprocess & freeze the train/val/test split

Training loads `*_training.npz` metacommunities directly; the split is otherwise computed
**on the fly** (seed 42) and **not saved** — a known train/eval leakage risk. Freeze
it explicitly first:

```bash
cd stage2_objective_AI_meta

# Option 1 — reproduce & audit the exact on-the-fly split (writes split_manifest.csv + audit)
python freeze_split_and_audit.py --simulation-dir ../results/data

# Option 2 — build a deliberate held-out split (you name the eval worlds)
python author_clean_split.py \
    --simulation-dir ../results/data \
    --seed 42 --val-ratio 0.083
```

Sanity-check the data before a long run:

```bash
python smoke_test_spatial.py --simulation-dir ../results/data
python audit_training_range_distribution.py --simulation-dir ../results/data   # KS check of range dist
```

Expected `.npz` keys per metacommunity (fallback names in the loader): `IBM_B` **(required)**,
`P_t`, `ENV_r_field`, `C_topk_idx`/`C_topk_w`, `P_last_final` **(training target)**,
`obs_mask_{1,5,10,20,50,100}`, plus species stats (`prevalence_final`, `deg_in`,
`deg_out`, `r_base`, …). Only 20×20 metacommunities are kept. See `data/README.md`.

---

## 3. (b) Train the diffusion model

Entry point is **`run_training.py`** (`training.py` is a library — running it directly
exits with a pointer). On startup it auto-installs the v2 patches and the **FIXAB**
patches (`fix_a_b_extrapolation.install_all`) and builds the **spatial-conditioned**
model (`models/ecodiffusion_spatial_cond.py`).

```bash
python run_training.py \
    --simulation-dir  ../results/data \
    --output-dir      ../stage2_outputs_SPATIAL_FIXAB \
    --total-epochs    500 \
    --batch-size      2 \
    --lr              1e-4 \
    --history-length  10 \
    --hist-sparsify-K 5 10 \
    --seed            42
```

**Key argument defaults** (from `run_training.py`): `--output-dir ./stage2_outputs_v2`,
`--batch-size 2`, `--lr 1e-4`, `--history-length 10`, `--hist-sparsify-K 5 10`,
`--target-mode last`, `--seed 42`, phase boundaries `--phase1-epochs 50
--phase2-epochs 100 --phase3-epochs 200 --phase4-epochs 500` (these CLI defaults
override `configs/config.py`).

**4-phase curriculum** (by epoch):

| Phase | Epochs | Mode | Adds to conditioning |
|---|---|---|---|
| 1 | 0–49 | `equilibrium` | `env`, coords |
| 2 | 50–99 | `interaction` | + GNN graph (`edge_index/weight`, `species_features`) |
| 3 | 100–199 | `temporal` | + `history_P/history_B` (last-10 frames) |
| 4 | 200–500 | `infill` | + `obs_mask`, `observed` (sparse obs) + FIXAB extrapolation loss |

**Checkpoints** land in `<output-dir>/checkpoints/`:
- `best_model.pt` — **best epoch selected by validation AUC on rare species**
  (`best_metric = val_auc_rare`); a new best requires a strictly higher value.
- `last_checkpoint.pt` — rolling, overwritten every validation epoch (crash-safe resume).
- `checkpoint_epoch_{N}.pt` — only if `save_best_only=False` (off by default).
- `training_history.json`, and `<output-dir>/preprocessor.pkl`.
- Cadence: validate every 5 epochs, save every 25, early-stopping patience 30 (per phase).

**Resume:**

```bash
python run_training.py --simulation-dir ../results/data \
    --output-dir ../stage2_outputs_SPATIAL_FIXAB \
    --resume     ../stage2_outputs_SPATIAL_FIXAB/checkpoints/last_checkpoint.pt
```

### What "SPATIAL_FIXAB_149" and the checkpoints mean

- **SPATIAL** = the spatial-conditioned model (`ecodiffusion_spatial_cond`, U-Net
  `input_proj` has **4** in-channels: 1 noisy + 3 spatial: obs_mask, env-suitability,
  obs-decay distance field).
- **FIXAB** = the two patches in `fix_a_b_extrapolation.py`, installed automatically
  by `run_training.py`:
  - **Fix A** — `ExtrapolationLoss`: weighted BCE at **unobserved** cells only
    (true-presence up-weighted ×20), added in phase ≥ 4. Verify it fires with
    `python verify_fix_a_fires.py --simulation-dir ../results/data`.
  - **Fix B** — multiplies the `obs_mask` channel by 0.5 so the model can't cheat by
    copying observations (requires training from scratch).
- **149** = an **epoch number**, not a magic directory. The final reconstructions/
  figures used the **epoch-149** FIXAB checkpoint (KS=0.047 at p≥0.9, an improvement
  over the epoch-109 checkpoint `FIXAB_epoch109_BACKUP.pt`). A directory named
  `stage2_outputs_SPATIAL_FIXAB_149/checkpoints/` is simply the run whose selected
  checkpoint is epoch 149. Point `--checkpoint` at that `.pt` for all downstream steps.

---

## 4. (c) Generate UNSEEN evaluation metacommunities

Unseen metacommunities use **new environment seeds (789, 2024)** the model never trained on.

```bash
# 1) Design the 30-world table + emit the sweep script + manifest
python design_unseen_eval_worlds.py --sim-dir ../results/data
#   -> ../results/data/data_eval_unseen/eval_unseen_manifest.csv
#   -> ../results/data/data_eval_unseen/pretraining_pool_combos.csv
#   -> ../scripts/sweep_worlds_eval30.py

# 2) Actually simulate the worlds (Stage-1 simulator; needs the real sim code)
python ../scripts/sweep_worlds_eval30.py --limit 1   # smoke test ONE world
python ../scripts/sweep_worlds_eval30.py             # full run (moves worlds into data_eval_unseen/)

# 3) Verify novelty + required keys (obs_mask_5, P_last_final, ENV_r_field, P_t)
python verify_unseen_eval_worlds.py \
    --sim-dir  ../results/data/data_eval_unseen \
    --manifest ../results/data/data_eval_unseen/eval_unseen_manifest.csv \
    --snapshot ../results/data/data_eval_unseen/pretraining_pool_combos.csv \
    --need-budget 5
```

Add `--in-dist-only` to `design_unseen_eval_worlds.py` to keep only the 24
in-distribution metacommunities (drop the 6 extrapolation metacommunities).

---

## 5. (d) Reconstruct species maps at inference

See `models/README.md` for full detail. The three observation regimes and their
output directories:

```bash
cd stage2_objective_AI_meta/models
CKPT=../../stage2_outputs_SPATIAL_FIXAB_149/checkpoints/best_model.pt   # epoch-149 FIXAB

# Fixed budget K=5  (dir suffix k5)   — uses generate_reconstructions_spatial.py
python generate_reconstructions_spatial.py \
    --checkpoint "$CKPT" \
    --truth-npz  ../../results/data/data_eval_unseen/<STEM>.npz \
    --output-dir ../../reconstructions_proportional_k5_n50/<STEM> \
    --fixed-budgets 5 --n-ensemble 50 --mode inpaint --ddim-steps 50 --eta 0.15

# Proportional p=0.10 (dir suffix n50)  — 10% of each range observed
python generate_reconstructions_proportional.py \
    --checkpoint "$CKPT" \
    --truth-npz  ../../results/data/data_eval_unseen/<STEM>.npz \
    --output-dir ../../reconstructions_proportional_n50/<STEM> \
    --obs-prob 0.10 --n-ensemble 50 --mode inpaint --ddim-steps 50 --eta 0.15

# Proportional p=0.30 (dir suffix p30_n50)
python generate_reconstructions_proportional.py \
    --checkpoint "$CKPT" \
    --truth-npz  ../../results/data/data_eval_unseen/<STEM>.npz \
    --output-dir ../../reconstructions_proportional_p30_n50/<STEM> \
    --obs-prob 0.30 --n-ensemble 50 --mode inpaint --ddim-steps 50 --eta 0.15

# Unseen-eval fixed budgets, ensemble 8 (feeds the per-species map + posterior figures)
python generate_reconstructions_spatial.py \
    --checkpoint "$CKPT" \
    --truth-npz  ../../results/data/data_eval_unseen/<STEM>.npz \
    --output-dir ../../reconstructions_unseen/<STEM> \
    --fixed-budgets 5 10 --n-ensemble 8 --mode inpaint --ddim-steps 50 --eta 0.15
```

**Directory-name decoder:** `p30` = per-cell observation probability 0.30 · `k5` =
fixed budget K=5 · **`n50` = an ensemble of 50 members** (`--n-ensemble 50`) · `_149` =
the epoch-149 checkpoint. **Ensembles** = `--n-ensemble` stochastic samples; diversity
comes from fresh Gaussian noise + stochastic DDIM (`eta`). **Two ensemble sizes were used
in this project:** the default **8** for the per-species map / posterior figures
(`reconstructions_unseen`, `reconstructions_spatial_149`), and **50** for the calibration /
PIT / recall analyses — the `*_n50` directories, for **K=5, p=0.10 and p=0.30 alike**
(so **K=5 was generated at both ensemble sizes — 8** in `reconstructions_unseen` **and 50**
in `reconstructions_proportional_k5_n50`). Each
metacommunity writes `recon_fixed_b{K}.npz` / `recon_prop_p{p}.npz` plus a `*_samples.npz`
holding all ensemble members. (Verified from the `n_ens` column in the committed result
CSVs: `figures_map_stage2_new/unseen_eval/calibration*/pit_*.csv` = **50**,
`posterior_*/*_metrics_*.csv` = **8**.)

---

## 6. (e) Evaluate coverage & multi-metacommunity summaries

```bash
cd stage2_objective_AI_meta/models

# Ensemble coverage, stratified by range size
python compute_ensemble_coverage_stratified.py \
    --truth-npz   ../../results/data/data_eval_unseen/<STEM>.npz \
    --samples-npz ../../reconstructions_unseen/<STEM>/recon_fixed_b5_samples.npz \
    --K 5 --output-csv ../../figures_map_axel_stage2_new/coverage_stratified_K5.csv

# Wide-range species list (input to multi-world eval & figures)
python build_wide_range_csv.py --K 5      # -> wide_range_species.csv

# 1× and 2× multi-world summaries
python multi_world_v7_evaluation.py \
    --wide-range-csv ../../figures_map_axel_stage2_new/wide_range_species.csv \
    --recon-dir-pattern './reconstructions_v7_inpaint_{world_stem}_stage2' \
    --truth-dir ../../results/data --K 5 --top-n-worlds 10 \
    --output-csv ../../figures_map_axel_stage2_new/multi_world_K5_summary.csv
python multi_world_2x_evaluation.py ... --output-csv .../multi_world_K5_2x_summary.csv
```

---

## 7. (f) Ablation study

Two pipelines exist (details in `models/README.md` and `models/Archive/README.md`):

```bash
cd stage2_objective_AI_meta/models

# CURRENT — remove-one-predictor ablation (v7) over 3 worlds
bash run_multi_world_ablation.sh
# then metrics + figures:
python Archive/validate_ablation.py --ablation-dir ../../ablation_v7_world5_stage2_inpaint \
    --truth-npz ../../results/data/<world5>.npz --K 5 \
    --output-csv ../../ablation_v7_world5_stage2_inpaint/ablation_metrics_1x.csv
python Archive/ablation_visualize_all.py \
    --metrics-csv ../../ablation_v7_world5_stage2_inpaint/ablation_metrics_1x.csv \
    --ablation-dir ../../ablation_v7_world5_stage2_inpaint \
    --truth-npz ../../results/data/<world5>.npz --K 5 \
    --output-dir ../../ablation_v7_world5_stage2_inpaint/figures_all_1x

# LEGACY v2/v5 study — this is what writes the stage2_ablation_figures/ directory
python Archive/make_ablation_figures.py ablation_v5_merged.json \
    --truth-npz ../../results/data/<world>.npz --ab14-demo --ab14-p-obs 0.001
#   --output-dir defaults to  stage2_ablation_figures/
```

---

## 8. (g) Reproduce the final figures (high accuracy)

Full per-figure commands are in `models/README.md`. The headline reproductions:

```bash
cd stage2_objective_AI_meta/models

# Per-species ecological maps for the UNSEEN-eval worlds
#  -> figures_map_axel_stage2_new/unseen_eval/<STEM>_three_map_K5.png
tail -n +2 ../../results/data/data_eval_unseen/eval_unseen_manifest.csv | cut -d, -f1 | while read fn; do
  stem="${fn%.npz}"
  python axel_per_species_map_ecological.py \
      --truth-dir ../../results/data/data_eval_unseen \
      --recon-dir-pattern './reconstructions_unseen/{world_stem}' \
      --world-stem "$stem" --K 5 --n-species 5 --threshold-mode match_truth \
      --output-path ../../figures_map_axel_stage2_new/unseen_eval/${stem}_three_map_K5.png
done

# Coverage / calibration figures in the same folder (paths hardcoded to the unseen dirs)
cd ..                                   # run these from stage2_objective_AI_meta/
python axel_bootstrap_and_calibration.py 5 2000    # -> unseen_eval/calibration_diag_K5.{csv,png}
python baseline_smoother_compare.py 5              # -> unseen_eval/baseline_compare_K5.csv

# Final unified figure
python models/make_figure1_honest_map.py \
    --multi-world-csv    figures_map_axel_stage2_new/multi_world_K5_summary.csv \
    --multi-world-csv-2x figures_map_axel_stage2_new/multi_world_K5_2x_summary.csv \
    --truth-dir results/data --recon-dir-pattern './reconstructions_v7_inpaint_{world_stem}_stage2' \
    --output-dir figures_map_axel_stage2_new/final_unified
```

> **Threshold accuracy note.** `--threshold-mode match_truth` selects the top-N cells
> where N = the species' true range size — this is the honest "what did the model
> recover" panel. The epoch-149 checkpoint shifted the cov-det optimum threshold to
> **p ≥ 0.9** (KS=0.047); use the epoch-149 checkpoint to match the published figures.

---

## 9. Directory map

| Path | Contents | README |
|---|---|---|
| `configs/` | `config.py` dataclasses (all hyperparameters) | `configs/README.md` |
| `data/` | Legacy `preprocessing.py` (NOT used by training) | `data/README.md` |
| `models/` | Model architecture, sampling, reconstruction, ablation, evaluation, figures | `models/README.md` |
| `models/Archive/` | Older (v2/v5/v6/v7) study & figure scripts | `models/Archive/README.md` |
| `figures_map_stage2_new/` | Transferred figure outputs (`unseen_eval/`) | `figures_map_stage2_new/README.md` |
| (top level `.py`) | Preprocessing, training, split, unseen-metacommunity design, FIXAB, calibration | this file |

See `STAGE2_COMPLETE_SUMMARY.md` for the design rationale (why diffusion, why
simulation-only training, loss-function derivations).

---

## 10. Glossary (plain language)

Quick translations of the terms used above; the **full** glossary (every term, with
context) is in [`../GETTING_STARTED.md`](../GETTING_STARTED.md#glossary).

- **metacommunity** — one simulated landscape (a `world` in the code): a 20×20 grid of
  local communities, thousands of species, 50 time steps; one `.npz` file.
- **epoch** — one full pass of training over the data (here up to 500).
- **checkpoint** (`.pt`) — a saved snapshot of the trained model. `best_model.pt` = the
  best-scoring one; "epoch 149" = the snapshot from the 149th pass.
- **curriculum (4 phases)** — the model is taught in stages: environment → + interactions
  → + history → + filling maps from sparse observations.
- **observation budget `K` / proportion `p`** — how many sightings we pretend to have
  (`K=5` = 5 cells/species; `p=0.30` = 30 % of the range). Lower = harder.
- **reconstruction / inpaint** — predicting the full species map from those few sightings.
- **ensemble** — several plausible maps per species (**8** for the map/posterior figures,
  **50** for the calibration & recall analyses — the `*_n50` reconstruction sets);
  agreement = confidence, disagreement = uncertainty.
- **calibration / PIT** — is the model's confidence *honest*? A flat PIT histogram = yes.
- **AUC (rare-species)** — 0.5–1.0 score for separating presence from absence (higher
  better; the model is chosen by its score on rare species).
- **KS** — 0–1 distance between two distributions (smaller = closer match; `0.047` = very
  close).
- **ablation** — turning off one input to measure how much it mattered.
- **FIXAB** — two training fixes that force genuine prediction instead of copying the
  observations back.
