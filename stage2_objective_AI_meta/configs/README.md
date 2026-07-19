# `configs/` — Stage-2 Configuration

![project badges](badges.svg)

`config.py` defines the dataclasses holding **all hyperparameters** (data, model,
diffusion, training, paths). It is imported everywhere via
`from configs.config import get_default_config` (and `get_device`).

You normally do **not** run this file. Override values with CLI flags on
`run_training.py` (which win over these defaults) or with a YAML via
`load_config_from_yaml`.

## Key defaults (as of `config.py`)

| Group | Field | Default |
|---|---|---|
| data | `grid_size` | `(20, 20)` |
| data | `n_timesteps` | `50` |
| data | `n_species_max` | `4000` |
| data | `npz_pattern` | `"*_training.npz"` |
| data | `train/val/test_ratio` | `0.833 / 0.083 / 0.084` |
| diffusion | `diffusion_steps` | `1000` |
| diffusion | `beta_schedule` | `"cosine"` |
| model | `unet_base_channels` | `64` |
| training | `batch_size` | `1` (CLI default is `2`) |
| training | `learning_rate` | `1e-4` |
| training | `total_epochs` | `500` |
| training | `phase{1..4}_epochs` | `50 / 100 / 150 / 200` (CLI overrides to `50/100/200/500`) |
| training | `seed` | `42` |
| training | `best_metric` | `"val_auc_rare"` |
| training | `save_best_only` | `True` |
| training | `val_every_epochs` | `5` |
| training | `save_every_epochs` | `25` |
| training | `early_stopping_patience` | `30` |
| training | `use_amp` | `True` |
| training | `grad_clip_norm` | `1.0` |
| optim | AdamW | `weight_decay=0.01`, `betas=(0.9,0.999)`, `warmup_epochs=10`, `min_lr=1e-6` |
| paths | `checkpoint_dir` | `<output-dir>/checkpoints` |

`get_device()` returns `cuda` → `mps` → `cpu`. Note the **CLI defaults in
`run_training.py` intentionally override** the phase-epoch and batch-size values here.
