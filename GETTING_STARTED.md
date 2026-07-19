# Getting Started — a plain-language guide (for ecologists)

![project badges](badges_getting_started.svg)

This guide is for someone who is **comfortable running commands in a terminal but is
not a machine-learning person**. It explains, in ecology terms, what this project does
and walks you through running it from start to finish. The detailed
[`README.md`](README.md) files stay technical; this page is the gentle on-ramp.

> New to the jargon? Jump to the **[Glossary](#glossary)** at the bottom — every
> AI/ML term used in the project is translated into one plain sentence.

---

## 1. What this project does (in one paragraph)

We simulate many virtual landscapes ("**metacommunities**" — grids of local communities
linked by dispersal, each with thousands of species evolving over time). We then train
an AI (a **diffusion model** we call **MetaDiffusion**) to look at a *few* scattered
sightings of a species and **fill in the rest of its map** — like an ecologist inferring
a range from patchy survey records, but done for thousands of species at once, **with an
honest estimate of uncertainty**. The goal (project Objective 2) is *high-confidence
distribution maps, especially for rare species*.

```
   A few observed cells                    A full predicted map + uncertainty
   (sparse survey)          MetaDiffusion        (many possible maps = an "ensemble")
   ┌───────────┐              ═══════►        ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ ·   x     │                              │ ▓▓░       │  │ ▓░░       │  │ ▓▓▓       │
   │     x   · │                              │ ▓▓▓░      │  │ ▓▓░░      │  │ ▓▓▓░      │
   │ ·       x │                              │  ░▓▓      │  │  ▓▓       │  │  ░▓▓      │
   └───────────┘                              └───────────┘  └───────────┘  └───────────┘
```

---

## 2. Before you start (prerequisites & expectations)

**Skills:** you need to be able to open a terminal, `cd` into folders, and paste
commands. You do **not** need to understand the AI internals.

**Hardware:**
- An **NVIDIA GPU is strongly recommended.** Training on a CPU is not practical.
  Reconstruction (making maps) also wants a GPU. Only the figure/plot scripts are fine
  on a laptop CPU.
- **Disk:** the datasets and outputs are **large — budget tens of GB** (each simulated
  metacommunity is hundreds of MB; there are ~240 of them, plus reconstruction outputs).

**Time (rough):**
- Training the model: **many hours to a few days** on a single modern GPU.
- Reconstructing one metacommunity: minutes.
- Making figures: seconds to minutes.

**Software:** Python 3.10+ and the packages in step 1 below.

**Inputs you must already have (or generate):**
- The **simulation datasets** `results/data/*_training.npz` (from Stage 1). ⚠️ The Stage-1
  simulation code is **not** included in this repository (only placeholder files) — if you
  don't have the real simulator, you cannot regenerate the data from scratch, but you can
  still run everything downstream if someone gives you the `.npz` files.
- A **trained model file** (a `.pt` "checkpoint") before you can reconstruct maps — you
  either train it yourself (step 4) or obtain one.

---

## 3. The pipeline in plain words

| Step | Plain meaning | Main script |
|---|---|---|
| **1. Simulate** | Create virtual landscapes with known "true" species maps | `infinite_pool_simulation/` (Stage 1) |
| **2. Prepare** | Load those datasets, split into learn / check / final-test sets | `data_preprocessing.py`, `author_clean_split.py` |
| **3. Train** | Teach the AI to fill maps from sparse sightings | `run_training.py` |
| **4. New test landscapes** | Make brand-new landscapes the AI never saw (fair test) | `design_unseen_eval_worlds.py` |
| **5. Reconstruct** | Give the AI a few sightings; it predicts full maps | `models/generate_reconstructions_*.py` |
| **6. Score** | Measure how good and how honest the predictions are | `models/multi_world_*_evaluation.py` |
| **7. Figures** | Draw the maps and accuracy/uncertainty plots | `models/axel_*.py` |

---

## 4. Step-by-step quickstart

All commands are run from the project's top folder unless it says otherwise. These are the
**minimal** commands; see the linked READMEs for every option.

### Step 1 — Set up Python
```bash
python -m venv lvmeta && source lvmeta/bin/activate     # make an isolated environment
pip install torch numpy tqdm scikit-learn scipy matplotlib pyyaml
python -c "import torch; print('GPU available:', torch.cuda.is_available())"   # should print True
```

### Step 2 — Get the simulation datasets
Put the `*_training.npz` files in `results/data/`. If you have the real Stage-1 code, you
generate them with the parameter sweeps (see
[`scripts/README.md`](scripts/README.md)); otherwise obtain them from the project author.
Quick check they look right:
```bash
cd stage2_objective_AI_meta
python smoke_test_spatial.py --simulation-dir ../results/data
```

### Step 3 — Train the model  (needs a GPU; takes hours–days)
```bash
python run_training.py \
    --simulation-dir ../results/data \
    --output-dir     ../stage2_outputs_SPATIAL_FIXAB \
    --total-epochs   500 --batch-size 2
```
When it finishes (or as it runs) the **best model** is saved as
`../stage2_outputs_SPATIAL_FIXAB/checkpoints/best_model.pt`. You can stop and resume with
`--resume <path-to>/last_checkpoint.pt`. (Details:
[`stage2_objective_AI_meta/README.md`](stage2_objective_AI_meta/README.md) §3.)

### Step 4 — Make fresh, unseen test landscapes
```bash
python design_unseen_eval_worlds.py --sim-dir ../results/data     # designs them + a run script
python ../scripts/sweep_worlds_eval30.py                          # actually simulates them (needs Stage-1 code)
python verify_unseen_eval_worlds.py \
    --sim-dir ../results/data/data_eval_unseen \
    --manifest ../results/data/data_eval_unseen/eval_unseen_manifest.csv \
    --snapshot ../results/data/data_eval_unseen/pretraining_pool_combos.csv
```

### Step 5 — Reconstruct maps from sparse sightings
Pick a metacommunity file name (without `.npz`) as `<STEM>`:
```bash
cd models
CKPT=../../stage2_outputs_SPATIAL_FIXAB/checkpoints/best_model.pt
python generate_reconstructions_spatial.py \
    --checkpoint "$CKPT" \
    --truth-npz  ../../results/data/data_eval_unseen/<STEM>.npz \
    --output-dir ../../reconstructions_unseen/<STEM> \
    --fixed-budgets 5 10 --n-ensemble 8
```
`--fixed-budgets 5 10` = "assume 5, then 10, observed cells per species"; `--n-ensemble 8`
= "produce 8 possible maps so we can express uncertainty". (More regimes — 10 % vs 30 %
of the range observed — in [`models/README.md`](stage2_objective_AI_meta/models/README.md) §2.)

### Step 6 — Score the predictions
```bash
python compute_ensemble_coverage_stratified.py \
    --truth-npz   ../../results/data/data_eval_unseen/<STEM>.npz \
    --samples-npz ../../reconstructions_unseen/<STEM>/recon_fixed_b5_samples.npz --K 5
```

### Step 7 — Draw the figures
```bash
python axel_per_species_map_ecological.py \
    --truth-dir ../../results/data/data_eval_unseen \
    --recon-dir-pattern './reconstructions_unseen/{world_stem}' \
    --world-stem <STEM> --K 5 --n-species 5 \
    --output-path ../../figures_map_axel_stage2_new/unseen_eval/<STEM>_map.png
```
Every figure and the exact command that makes it are listed in
[`figures_map_stage2_new/README.md`](stage2_objective_AI_meta/figures_map_stage2_new/README.md).

---

## 5. "I just want to answer this question" — cheat sheet

| Your question | Where to look |
|---|---|
| How were the virtual landscapes made? | [`infinite_pool_simulation/README.md`](infinite_pool_simulation/README.md), [`scripts/README.md`](scripts/README.md) |
| How do I train the model? | [`stage2_objective_AI_meta/README.md`](stage2_objective_AI_meta/README.md) §3 |
| What does "best epoch" / checkpoint mean? | Glossary below + Stage-2 README §3 |
| How do I make maps from few observations? | [`models/README.md`](stage2_objective_AI_meta/models/README.md) §2 |
| How good / honest are the maps? | Stage-2 README §6; figures README |
| Which script made figure X? | [`figures_map_stage2_new/README.md`](stage2_objective_AI_meta/figures_map_stage2_new/README.md) |

---

## Glossary

Plain-language meaning of every AI/ML term used in the project (code names in `mono`).

**Metacommunity** (`world` in the code) — one simulated landscape: a 20×20 grid of local
communities linked by dispersal, thousands of species, 50 time steps. One `.npz` file.

**`.npz` file** — a NumPy data file bundling many arrays (the biomass maps, environment,
interaction network, etc.) for one metacommunity.

**Diffusion model / MetaDiffusion** — the type of AI used. It learns to turn "pure noise"
into a realistic species map, guided by the environment, the interaction network, and your
observations. Think of it as a very well-trained "map completer".

**Training** — the AI adjusts itself repeatedly to get better at reproducing the known true
maps. **Validation** — a held-back set used to check progress during training. **Test /
unseen** — landscapes kept completely separate for a fair final grade.

**Epoch** — one full pass of the AI through the training data. Training runs many epochs
(here up to 500).

**Checkpoint** — a saved snapshot of the trained AI (a `.pt` file). `best_model.pt` is the
snapshot that scored best on rare species; `last_checkpoint.pt` is the most recent (for
resuming). "**Epoch 149**" just means "the checkpoint saved at the 149th pass".

**Curriculum (4 phases)** — the AI is taught in stages of increasing difficulty:
environment only → add species interactions → add time/history → finally learn to fill maps
from sparse observations. Like teaching easy concepts before hard ones.

**Observation budget `K` / proportion `p`** — how many sightings we pretend to have.
`K=5` = five observed cells per species; `p=0.30` = 30 % of the species' true range
observed. Lower = harder.

**Reconstruction / inpainting** — giving the AI the sparse observations and asking it to
predict the full map (the missing cells).

**Ensemble** — instead of one prediction, the AI makes several slightly different plausible
maps. Where they agree = high confidence; where they disagree = uncertainty. This project
used **8** members for the map/posterior figures and a larger **50** members for the
calibration and recall analyses (the reconstruction folders whose names end in `_n50` —
`n50` literally means "ensemble of 50", for the K=5, 10 %, and 30 % observation cases).

**Uncertainty / calibration** — whether the model's confidence is *honest*. A **PIT
histogram** and the **calibration** plots check this: a flat/uniform histogram means
well-calibrated (the truth falls where the model said it would, at the right rate).

**AUC (rare-species AUC)** — "Area Under the ROC Curve", a 0.5–1.0 score for how well the
predicted map separates presence from absence; **rare-species AUC** measures this for
uncommon species specifically. Higher is better; 0.5 = no better than chance.

**Recall / coverage** — recall = fraction of the true occupied cells the model recovered;
coverage = whether the true map falls inside the ensemble's spread.

**KS statistic** — Kolmogorov–Smirnov: a 0–1 number measuring how different two
distributions are (smaller = more similar). We use it to check predicted range-size
distributions match reality; `KS=0.047` is a very close match.

**Ablation** — deliberately switching off one input (the environment, the interaction
network, the history, or the observations) to measure how much it mattered. If accuracy
drops a lot, that input was important.

**DDIM / `--ddim-steps` / `--eta`** — knobs controlling how the AI generates a map
(how many refinement steps, and how much randomness). **Defaults are fine**; you rarely
change them.

**GPU / CUDA** — the graphics-card hardware that makes training/inference fast. `--device
cuda` uses it; `--device cpu` falls back to the (much slower) processor.

**FIXAB (`Fix A` + `Fix B`)** — two training fixes specific to this project. *Fix A* rewards
the model for getting **unobserved** cells right (not just copying the sightings back).
*Fix B* deliberately weakens the "here are the sightings" input so the model can't cheat by
copying. Together they force it to genuinely *predict*.
