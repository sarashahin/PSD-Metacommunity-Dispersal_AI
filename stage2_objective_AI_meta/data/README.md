# `data/` — Legacy preprocessing (NOT used by training)

![project badges](badges.svg)

⚠️ **`data/preprocessing.py` is legacy and is NOT wired into the active pipeline.**
`run_training.py` imports `from data_preprocessing import create_dataloaders` — the
**top-level** `../data_preprocessing.py`, not this module.

This `data/preprocessing.py` is an older/alternative implementation that depends on
`torch_geometric` and references config fields (`config.data.data_root`, `n_train`,
`n_species`) that **do not exist** in `configs/config.py`. It will not run against the
current config. It is kept for reference only.

**Use instead:**
- `../data_preprocessing.py` — the real loader (`SimulationWorld`, `create_dataloaders`,
  `compute_global_stats`, `eco_collate_fn`). See [`../README.md`](../README.md) §2 for
  the expected `.npz` keys and the train/val/test split.
- `../data_preprocessing_v2_patch.py` — auto-installed by `run_training.py`
  (last-N history frames + history in infill mode).

If you want to remove the `torch_geometric` dependency from your environment, you can
ignore this folder entirely.
