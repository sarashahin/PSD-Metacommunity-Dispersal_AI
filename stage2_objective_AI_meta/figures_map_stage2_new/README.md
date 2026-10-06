# `figures_map_stage2_new/` — the paper's figures

![project badges](badges.svg)

This folder holds the **final figures used in the Objective‑2 paper**, plus the small
CSV tables of numbers behind each one. Every figure here was produced on the
**unseen test worlds** — metacommunities grown from environment seeds the model never
saw during training (seeds 789 / 2024) — so they show how the model does on genuinely
new data, not on data it memorised.

All figures live under `unseen_eval/` in **9 groups**. Each group below says, in plain
words: **what you are looking at**, **which paper figure it is**, and **the one script
that makes it** (all scripts are in [`../models/`](../models/README.md)).

---

## A 60‑second glossary (plain English)

| Term | What it means here |
|---|---|
| **Records / observations** | The handful of grid cells where a species was actually "seen". The model only gets these, and has to guess the rest of the range. |
| **K5** | Fixed budget: exactly **5 records** per species. |
| **p10 / p30** | Proportional budget: **10% / 30%** of the species' true range is recorded. |
| **Ensemble (8 or 50)** | The model doesn't give one map — it draws **many plausible maps**. `8ens` = 8 draws (used for pictures), `50ens` = 50 draws (used for statistics). Where the draws agree = confident; where they disagree = uncertain. |
| **Posterior map** | The per‑cell probability a species is present, taken as the fraction of ensemble draws that put it there. |
| **Conditional** | Same picture, but each draw is forced to have the true number of occupied cells, so only the *shape* of the range varies. |
| **Recall (near / far)** | Of the hidden occupied cells, how many the model finds — split into **near** a record (≤2 cells) vs **far**. |
| **Coverage / calibration** | Whether the true range actually falls inside the spread of the ensemble — i.e. whether the model's confidence is **honest**. |

---

## The 9 figure groups

### 1. `paper_posterior_k5/` — per‑species range reconstructions → **Paper Figure 2**
What it shows: for a few example species, the **true range** next to the model's
**reconstructed probability map** from only a few records. Variants cover the fixed
budget (`main_K5`, `rare_K5`), proportional budgets (`rare_p10`, `rare_p30`,
`wide_p30`), ensemble sizes (`8ens`, `50ens`), and `_conditional` (fixed range size).
This is the headline result: plausible, record‑anchored maps for rare species.
- Tables: `baseline_compare_*.csv` (model vs a simple Gaussian smoother), `rare_audit_*.csv`, `smoother_*.log`.
- Regenerate: `python models/posterior_per_species.py --truth-dir <…> --recon-dir-pattern <…> --world-stem <…> --K 5 --output-path <…>`

### 2. `recall/` — how many hidden cells are recovered → **Paper Figure 5**
What it shows: recall plotted against the number of records (and against the fraction
of range observed), split into **near** vs **far** cells, with the random‑chance floor.
The message: the model beats chance strongly **near** records, and sits at chance far away.
- Files: `Fig_recall_vs_observations.*`, `Fig_recall_by_regime.*`, `recall_vs_observations.csv`.
- Regenerate: `python models/recall_vs_observations.py --truth-dir <…> --world-stems <…> --labels <…> --recon-dir-patterns <…> --recon-filenames <…>`

### 3. `fig4_v2_all/` and 4. `fig4_v2_indist/` — are the *shapes* realistic? → **Paper Figure 4**
What it shows: across many species, the model's reconstructed ranges have the same
**fragmentation** (how broken‑up) and **spatial spread** as the true ranges.
`fig4_v2_all` pools all test worlds; `fig4_v2_indist` uses only the in‑distribution ones.
- Files: `Fig4_distributional_realism.*` (+ `_values.csv`, `_router.csv`, `_dropped.csv`).
- Regenerate: `python models/axel_ecological_distribution_figure.py --wide-range-csv <…> --truth-dir <…> --recon-dir-pattern <…> --K 5 --output <…>`

### 5. `fig4_regression/` — the shape relationship as a fit → **Paper Figure 4 (variant)**
Same realism test shown as a regression (`Fig4.*`, `Fig4_router.csv`, `Fig4_values.csv`),
produced by the same script with the regression grid option.

### 6. `calibration_figs/` — is the confidence honest? → **Paper Figure 6 + calibration**
What it shows: **coverage/calibration** at the fixed budget (`K_5`) and proportional
budgets (`p_0_10`, `p_0_30`), and how calibration improves with more records
(`vs_records`). Each has a `_values.csv` with the exact numbers.
- Regenerate: `python models/pit_calibration_figures.py --csv <pit_values.csv> --output <…>`

### 7. `obs_schemes/` — what the two recording rules look like → **Supplementary**
What it shows: side‑by‑side of the fixed‑budget vs proportional recording schemes at
10% and 30% (`Fig_observation_extremes_p0.10`, `p0.30`), so readers see the two ways
records accumulate in nature.
- Regenerate: `python models/observation_extremes_figure.py --truth-dir <…> --world-stems <…> --prob 0.10 --output <…>`

### 8. `figs_EI/` — the final assembled panels → **Figure 6, Figure S2, Figure S3**
The paper‑ready, labelled versions: `Figure_6.pdf` (calibration), `Figure_S2.pdf`
(observation schemes) and `Figure_S3.pdf`, bundled with the calibration and
observation‑extreme panels they are built from. Made by `pit_calibration_figures.py`
and `observation_extremes_figure.py`.

### 9. `ablation/` — which inputs actually matter → **Paper Figure 7**
What it shows: a **leave‑one‑out** test — remove the records, the interaction network,
the environment, or the species features, and see what happens to recall and shape.
The message: removing the **records** destroys performance; removing the process inputs
barely changes it. `Fig_ablation*` are the figures; `ablation_per_species*.csv` the data.
- Regenerate (two steps): run the arms with `python models/run_ablation_spatial.py --stage2-dir <…> --checkpoint <…> --truth-npz <…> --output-dir <…> --variants <…>`, then make the figure with `python models/ablation_analyse.py --csv ablation_per_species.csv --out ablation/Fig_ablation`.

---

## Notes for exact reproduction
- **Ensemble sizes:** map pictures use the **8‑member** ensembles; calibration/recall
  statistics use the **50‑member** ensembles (`*_50ens`). This is recorded in the CSVs.
- **File types:** each figure ships as both `.pdf` (vector, for the manuscript) and
  `.png` (preview), with a `_values.csv` holding the plotted numbers.
- **Script output folder:** the generator scripts were written to output into a local
  folder named `figures_map_axel_stage2_new/` (note `_axel_`). When you regenerate,
  point each script's `--output`/`--out-dir` at this folder, or rename accordingly.
- **Full flag list:** every script accepts `--help` (`python models/<script>.py --help`)
  for the complete, exact set of options and paths.
- Model weights, reconstruction `.npz` files and raw simulation data are **not** stored
  here (too large); see the repository root `README.md` for where they live and how the
  reconstructions are produced before these figures are drawn.
