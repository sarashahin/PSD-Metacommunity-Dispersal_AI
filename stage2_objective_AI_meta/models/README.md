# `models/` — the MetaDiffusion model and all analysis scripts

![project badges](badges.svg)

This folder contains **two things**:

1. **The model itself** — *MetaDiffusion*, the AI that reconstructs a species' full
   range map from a handful of records.
2. **Every script** that runs the model on the test worlds and draws the paper's
   figures (so all results are reproducible).

Nothing here needs a GPU to *read*; running the generation steps does.

---

## The model in plain words

Give the model a few places a species was seen, and it predicts where else it lives —
**with honest uncertainty**. It works like this:

- **Three "readers" (encoders)** each summarise one kind of information per species:
  - the **environment** it experiences (a small image‑reading network, CNN),
  - its **competitors** (a network‑reading model, GNN),
  - its recent **history** of presence (a sequence model, Transformer).
- A **denoiser (U‑Net)** then starts from random noise and, step by step, turns it into
  a plausible range map that (a) matches the few records and (b) looks like a real range.
- Because each run starts from different noise, running it many times gives an
  **ensemble** of plausible maps. Agreement = confidence; disagreement = uncertainty.

Training details (the 4‑stage curriculum and the "FIXAB" trick that forces the model to
*predict* rather than *copy* the records) are in the repository root
[`README.md`](../README.md).

---

## What each file is

### Core model (the network)
| File | Plain description |
|---|---|
| `ecodiffusion.py` | The main MetaDiffusion model — ties the encoders to the denoiser. |
| `ecodiffusion_spatial_cond.py` | The deployed variant with **spatial conditioning** (records + environment + distance channels). |
| `ecodiffusion_sample_spatial.py` | The **sampler** — runs the reverse diffusion (DDIM) and "repaint" infilling. |
| `diffusion.py` | The diffusion maths: how noise is added and removed over steps. |
| `unet.py` | The U‑Net denoiser that predicts the clean map at each step. |
| `env_encoder.py` | Reads each species' **environment** field (CNN). |
| `interaction_encoder.py` | Reads the **competition network** (GNN). |
| `temporal_encoder.py` | Reads the **occupancy history** (Transformer). |

### Make reconstructions (run the model)
| File | Plain description |
|---|---|
| `generate_reconstructions_spatial.py` | Reconstruct ranges at a **fixed budget** (e.g. K=5 records). |
| `generate_reconstructions_proportional.py` | Reconstruct at a **proportional budget** (e.g. 10% / 30% of the range). |
| `generate_reconstructions_spatial_FEATFIX.py` | Variant with the species‑feature fix. |
| `pick_inference_worlds.py` | Choose which test worlds to reconstruct. |

### Leave‑one‑out ablation (which inputs matter) → Figure 7
| File | Plain description |
|---|---|
| `run_ablation_spatial.py` | Runs the study: remove each input in turn and re‑reconstruct. |
| `gen_ENVONLY.py`, `gen_FEATONLY.py`, `gen_FIXB.py`, `gen_NOLEAK.py`, `gen_OLD_NOLEAK.py` | The individual ablation **arms** (environment‑only, features‑only, Fix‑B, records‑removed). |
| `ablation_analyse.py`, `ablation_analyse_v2.py` | Turn the ablation table into `Fig_ablation`. |

### Figures and analysis (make the paper figures)
| File | Makes | Paper figure |
|---|---|---|
| `posterior_per_species.py` | per‑species posterior maps | **Fig 2** |
| `axel_ecological_distribution_figure.py` | shape‑realism (fragmentation/spread) | **Fig 4** |
| `recall_vs_observations.py` | recall vs records / effort | **Fig 5** |
| `pit_calibration_figures.py` | calibration / coverage | **Fig 6** |
| `observation_extremes_figure.py` | recording‑scheme panels | **Fig S2/S3** |
| `axel_per_species_map_ecological.py` | truth‑vs‑prediction "three‑map" panels | support |
| `axel_cross_world_summary_figure.py`, `..._metrics.py` | across‑world summaries | support |
| `axel_distribution_tests_inpaint.py` | per‑species distribution tests | support |
| `make_figure1_honest_map.py` | assembled multi‑panel figure | support |
| `multi_world_v7_evaluation.py`, `multi_world_2x_evaluation.py` | pooled metrics across worlds (1× and 2× calibration) | tables |
| `compute_ensemble_coverage_stratified.py` | coverage split by range size | tables |
| `width_by_records.py`, `add_width_from_samples.py` | range‑width vs record count | support |

### Threshold / routing diagnostics
These decide **how a probability map becomes a yes/no presence map** (picking the cut‑off),
and check it is not an artefact:
`axel_adaptive_routing_v2_multifeature.py`, `axel_adaptive_routing_v3_bucketclassifier.py`,
`axel_adaptive_threshold_diagnosis.py`, `axel_three_axis_threshold_alignment.py`,
`axel_fine_threshold_sweep.py`, `axel_probability_mass_diagnosis.py`, `diagnose_recon_axel_map.py`.

### Helpers
| File | Plain description |
|---|---|
| `build_wide_range_csv.py` | List the wide‑range species used in shape tests. |
| `build_regenerate_script.py` | Builds a batch script to regenerate all reconstructions. |
| `run_multiworld.sh`, `run_multi_world_ablation.sh` | Batch runners. |
| `REVIEW_ANALYSIS.md` | Internal analysis notes. |
| `__init__.py` | Makes the folder importable as a Python package. |

---

## Typical order to run things

1. **Train** the model → produces the epoch‑149 checkpoint (see root `README.md`).
2. **Reconstruct** the unseen worlds:
   `python models/generate_reconstructions_spatial.py …` (and the proportional one).
3. **Make figures** with the figure scripts above, pointing their `--output` at
   [`../figures_map_stage2_new/`](../figures_map_stage2_new/README.md).

Every script prints its full options with `--help`:
```bash
python models/<script>.py --help
```

> Large files (model weights `*.pt`, reconstruction `*.npz`, raw simulation data) are
> **not** stored in git — see the repository root `README.md` for where to get them.
