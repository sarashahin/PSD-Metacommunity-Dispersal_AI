# Developer notes

Notes for anyone working on the code in this repository. Read this first.

## About the project

PhD research code by Sara Shahin, supervised by Axel G. Rossberg (the `axel_*`
scripts are named after him). The paper is "Objective 2" of the PhD. Two stages:

1. Stage 1 (`infinite_pool_simulation/`, `scripts/`): an individual-based
   Lotka-Volterra metacommunity simulation. It writes `results/data/*_training.npz`,
   one file per metacommunity: a 20x20 grid, thousands of species, 50 timesteps.
2. Stage 2 (`stage2_objective_AI_meta/`): "MetaDiffusion", a conditional diffusion
   model trained on those simulations. It reconstructs species range maps from a few
   records, with an ensemble of samples giving the uncertainty.

Prose says "metacommunity"; code, file names and CLI flags say "world". Keep that
convention.

## Things that are not in git

- All files in `infinite_pool_simulation/` are empty placeholders. The real Stage-1
  code is on Sara's own machine. Do not edit or "fix" these stubs.
- No data (`*.npz`), checkpoints (`*.pt`) or reconstruction folders. `.gitignore`
  excludes them. Never commit them.
- `models/Archive/` is mentioned in the READMEs but is not in the repository.
- `axel_per_species_map_ecological_metrics.py`, imported by
  `models/axel_cross_world_summary_figure_metrics.py`, is not in the repository.

## Environment

```bash
python -m venv lvmeta && source lvmeta/bin/activate
pip install -r requirements.txt
```

Tested in a cloud session on Python 3.13 and torch 2.14 (CPU). Training and full
reconstructions need a GPU; the cloud sessions have none, so only run small tests there.

## Checking that the code still works

There is no real data in the repository, so tests use small synthetic worlds:

```bash
bash tests/run_checks.sh
```

This writes toy worlds with `tests/make_toy_worlds.py`, then runs
`smoke_test_spatial.py` (6 checks, includes a 2-epoch training run),
`verify_fix_a_fires.py` and one reconstruction. It takes a few minutes on CPU.
Run it after any change to the Stage-2 code. The toy numbers are random, so only
"runs without error and writes the expected files" is meaningful, not the scores.

## How the code is laid out

- Entry point for training: `stage2_objective_AI_meta/run_training.py`. `training.py`
  is a library. `run_training.py` installs the `*_v2_patch.py` modules and the FIXAB
  patches (`fix_a_b_extrapolation.py`) at runtime by monkey-patching, so the code that
  runs is the base file plus the patch. Read both before changing either.
- Model: `models/ecodiffusion_spatial_cond.py` (U-Net `input_proj` has 4 channels).
- Reconstruction: `models/generate_reconstructions_spatial.py`;
  `generate_reconstructions_proportional.py` and `run_ablation_spatial.py` import from
  it. The published results use this plain version (the "deployed configuration"
  in the paper, Supplementary Table S3).
- Published figures and their CSVs: `stage2_objective_AI_meta/figures_map_stage2_new/`.
  Its README maps each figure to the script that makes it. Do not regenerate or
  overwrite these without being asked.
- Many scripts run work at import time and use paths relative to the repo root
  (for example `results/data/data_eval_unseen/eval_unseen_manifest.csv`). Scripts in
  `models/` import their neighbours by bare name, so run them from inside `models/`.
- Old hard-coded paths say `AI_simulation/stage2/`; in this repo that is
  `stage2_objective_AI_meta/`.

## Rules for changes

- Results in the paper depend on this code. Do not change numerical behaviour
  (losses, sampling, thresholds, feature construction, seeds, splits) unless Sara asks.
  Cleanups must give identical outputs.
- Keep the existing style: long docstring headers that explain why, argparse CLIs.
- Work on a branch and open a pull request; do not push to `main`.
- Commits and pull requests are authored by Sara Shahin only. No co-author or
  tool-attribution lines in commit messages, pull requests or files.
- Writing for Sara: plain, simple English, no jargon without a short explanation.
