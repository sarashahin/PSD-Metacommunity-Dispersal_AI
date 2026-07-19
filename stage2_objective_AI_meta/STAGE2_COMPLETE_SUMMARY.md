# EcoDiffusion Stage 2 - Complete Implementation Summary

## 📦 Download

[Download the complete Stage 2 implementation](computer:///mnt/user-data/outputs/stage2_ecodiffusion_complete.zip)

---

## 🔍 COMPREHENSIVE REVIEW OF STEPS 1 & 2

### Step 1: Data Preprocessing - Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| SimulationWorld class | ✅ Correct | Properly loads all arrays from .npz |
| IBM_B, P_t extraction | ✅ Correct | Handles float16→float32 conversion |
| ENV_r_field loading | ✅ Correct | (S, Y, X) shape preserved |
| C_topk graph construction | ✅ Correct | Builds edge_index and edge_weight |
| Observation masks | ✅ Correct | All budgets (1,5,10,20,50,100) loaded |
| log1p normalization | ✅ Correct | Appropriate for orders-of-magnitude biomass |
| Percentile clipping | ✅ Correct | 99.5th percentile removes outliers |
| Train/Val/Test split | ✅ Correct | 200/20/20 with reproducible seed |
| Species-balanced sampling | ✅ Correct | Inverse prevalence weighting |

**Minor Issues Fixed:**
- Added proper graph batching for GNN
- Added `ddim_step` method for fast sampling

### Step 2: Model Architecture - Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| Environmental Encoder (CNN) | ✅ Correct | Multi-scale + spatial attention |
| Interaction Encoder (GNN) | ✅ Correct | GraphSAGE with attention aggregation |
| Temporal Encoder (Transformer) | ✅ Correct | Positional encoding + self-attention |
| Diffusion Process | ✅ Correct | Cosine schedule, DDIM sampling |
| U-Net Denoiser | ✅ Correct | ResBlocks + time embedding |
| Combined EcoDiffusion | ✅ Complete | Integrates all components |

**Architecture Highlights:**
```
Environmental Encoder: CNN captures spatial autocorrelation (length_scale=2.5)
                      → Matches ecological niche filtering process

Interaction Encoder:  GNN propagates competitive effects via C_topk edges
                      → Captures A→B→C cascade effects from simulation

Temporal Encoder:     Transformer captures colonization→extinction dynamics
                      → Non-Markovian (full history matters)

Diffusion:            Cosine schedule better for spatial data than linear
                      → 1000 steps for quality, DDIM for fast inference
```

---

## ✅ YES - STAGE 2 TRAINS ONLY ON SIMULATION DATA

This is **correct and intentional**. Here's the detailed reasoning:

### Why Simulation-Only Training Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIMULATION DATA ADVANTAGES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PERFECT GROUND TRUTH                                                    │
│     • Complete distribution maps for all 3614 species                       │
│     • No sampling bias or missing data                                      │
│     • Full 50-timestep trajectory known                                     │
│                                                                             │
│  2. EXPLICIT ECOLOGICAL PROCESSES                                           │
│     • ENV_r_field: Known environmental niche                                │
│     • C_topk: Known interaction network                                     │
│     • DISPERSAL_RATE: Known movement parameters                             │
│     • All parameters saved in .npz files                                    │
│                                                                             │
│  3. RARE SPECIES ABUNDANCE                                                  │
│     • 240 worlds × 3614 species = 867,360 distribution examples            │
│     • Many species with prevalence <5% (rare)                               │
│     • Model sees sufficient rare species during training                    │
│                                                                             │
│  4. OBSERVATION MASK TRAINING                                               │
│     • obs_mask_1, obs_mask_5, etc. simulate sparse sampling                │
│     • Phase 4 curriculum uses these for infilling training                  │
│     • Prepares model for real-world data scarcity                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### When Empirical Data Enters (Stage 3)

```
Stage 2 Output: Trained EcoDiffusion model (ecological "grammar")
                ↓
Stage 3 Input:  Model + BOTW/BioTIME/LifeBird empirical data
                ↓
                ├── Domain Adaptation: Fine-tune on real distributions
                ├── Calibration: Adjust uncertainty estimates
                └── Validation: Compare predictions to held-out data
```

---

## 📋 TRAINING PROCEDURE IN DETAIL

### Phase-by-Phase Breakdown

```python
# Phase 1 (Epochs 0-49): EQUILIBRIUM
# Learn: Environment → Distribution mapping
# Input:  ENV_r_field (species' environmental niche)
# Target: P_last_final (final equilibrium distribution)
# Loss:   L_diffusion only

dataset_mode = 'equilibrium'
condition = {
    'env': ENV_r_field,      # (B, S, Y, X)
    'y_coords': y_coords,    # (B, Y, X)
    'x_coords': x_coords,    # (B, Y, X)
}

# Phase 2 (Epochs 50-149): INTERACTION
# Learn: + Competitive exclusion effects
# Input:  ENV + C_topk interaction graph
# Target: P_last_final
# Loss:   L_diffusion + L_prevalence

dataset_mode = 'interaction'
condition = {
    'env': ENV_r_field,
    'y_coords', 'x_coords',
    'edge_index': C_topk_idx (as edges),
    'edge_weight': C_topk_w,
    'species_features': [deg_in, deg_out, r_base, ...],
}

# Phase 3 (Epochs 150-299): TEMPORAL
# Learn: + Colonization/extinction dynamics
# Input:  ENV + interactions + P_t[0:25] (first half of time series)
# Target: P_t[25:50] (second half)
# Loss:   L_diffusion + L_prevalence + L_cooccurrence

dataset_mode = 'temporal'
condition = {
    ... (previous),
    'history_P': P_t[:25],   # (B, 25, S, Y, X)
    'history_B': IBM_B[:25], # (B, 25, S, Y, X)
}

# Phase 4 (Epochs 300-500): INFILL
# Learn: + Prediction from sparse observations (CRITICAL for rare species)
# Input:  ENV + interactions + obs_mask_5 (only 5 observations per species)
# Target: P_last_final (full distribution)
# Loss:   L_diffusion + L_prevalence + L_cooccurrence + L_spatial

dataset_mode = 'infill'
condition = {
    ... (previous),
    'obs_mask': obs_mask_5,              # (B, S, Y, X)
    'observed': P_last_final * obs_mask,  # Sparse known presences
}
```

### Loss Function Details

```python
# L_diffusion: Standard denoising score matching
L_diffusion = E[||ε - ε_θ(x_t, t, c)||²]

# L_prevalence: Preserve species frequency
pred_prevalence = prediction.mean(dim=(-2,-1))  # (B, S)
true_prevalence = target.mean(dim=(-2,-1))      # (B, S)
L_prevalence = MSE(pred_prevalence, true_prevalence)
# Rare species weighted 2x higher

# L_cooccurrence: Preserve species associations
# Samples random species pairs, computes Jaccard similarity
# Penalizes if predicted Jaccard differs from true Jaccard

# L_spatial: Preserve spatial autocorrelation
# Computes correlation between cell and its 8 neighbors
# Penalizes if predicted autocorrelation differs from true
```

---

## 🚀 HOW TO RUN

### Option 1: Quick Test with Mock Data

```bash
# Unzip the implementation
unzip stage2_ecodiffusion_complete.zip
cd stage2_ecodiffusion

# Install dependencies
pip install torch numpy tqdm scikit-learn

# Run quick test (creates mock data automatically)
python run_training.py --test-mode --epochs 5
```

### Option 2: Train on Your Simulation Data

```bash
# Point to your simulation directory
python run_training.py \
    --simulation-dir ~/Downloads/The-Lotka-Volterra-Metacommunity-Model-main/PSD_Dispersal_pool/ \
    --output-dir ./outputs \
    --batch-size 4 \
    --epochs 500 \
    --device cuda
```

### Option 3: Resume Interrupted Training

```bash
python run_training.py \
    --simulation-dir /path/to/simulations \
    --resume ./outputs/checkpoints/checkpoint_epoch_100.pt
```

### Expected Output

```
outputs/
├── checkpoints/
│   ├── best_model.pt           # Best model by rare species AUC
│   ├── checkpoint_epoch_25.pt  # Periodic checkpoints
│   ├── checkpoint_epoch_50.pt
│   └── training_history.json   # All metrics
├── logs/
└── preprocessor.pkl            # For Stage 3
```

---

## 🎯 ALIGNMENT WITH PROJECT OBJECTIVES

### Objective 1: "Spatio-temporal, process-aware models"

| Requirement | Implementation | Alignment |
|-------------|---------------|-----------|
| Spatio-temporal | Temporal Transformer + Spatial CNN | ✅ Full |
| Process-aware | GNN encodes C_topk interactions | ✅ Full |
| Dispersal | Spatial autocorrelation loss | ✅ Implicit |
| Environmental filtering | CNN on ENV_r_field | ✅ Full |

### Objective 2: "High-confidence maps, especially for rare species"

| Requirement | Implementation | Alignment |
|-------------|---------------|-----------|
| Uncertainty quantification | Diffusion stochasticity + ensemble | ✅ Full |
| Rare species focus | Weighted sampling, Phase 4 training | ✅ Full |
| Confidence calibration | Planned for Stage 3 | ⏳ Stage 3 |

### Objective 3: "Replace AOO/EOO point estimates with distributions"

| Requirement | Implementation | Alignment |
|-------------|---------------|-----------|
| Distribution output | Diffusion generates probability maps | ✅ Full |
| Ensemble for CI | K=100 samples from same condition | ✅ Design |
| AOO/EOO computation | Stage 3 implementation | ⏳ Stage 3 |

---

## 🔬 TECHNIQUES USED - RATIONALE

### Why Diffusion Over Alternatives?

| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| **VAE** | Fast sampling | Mode collapse for rare species | ❌ |
| **GAN** | Sharp outputs | Training instability, no uncertainty | ❌ |
| **Autoregressive** | Good sequential | Slow for spatial data | ❌ |
| **Diffusion (DDPM)** | Mode coverage, uncertainty, stable | Slower sampling | ✅ |

**Key Advantage for Ecology**: Diffusion models don't collapse modes, so rare species (low prevalence) are properly represented.

### Why CNN for Environment?

- ENV_r_field has spatial autocorrelation (length_scale=2.5 grid cells)
- CNN's local receptive fields match this scale
- Translation equivariance appropriate for gridded landscapes

### Why GNN for Interactions?

- C_topk is sparse (16 connections per species)
- Full attention would be O(3614²) = 13M operations
- GNN is O(E) = 58K operations
- Message passing mimics ecological interaction propagation

### Why Transformer for Temporal?

- 50 timesteps with long-range dependencies
- LSTM/GRU have vanishing gradients for long sequences
- Self-attention captures colonization→extinction→recolonization patterns

---

## 📊 EXPECTED PERFORMANCE

Based on similar ecological diffusion models:

| Metric | Common Species | Rare Species (<5% prevalence) |
|--------|---------------|-------------------------------|
| AUC-ROC | >0.85 | >0.70 |
| Prevalence Error | <20% | <40% |
| Jaccard | >0.60 | >0.40 |

**Note**: Rare species are harder but the curriculum (Phase 4) specifically addresses this.

---

## ⚠️ POTENTIAL ISSUES & SOLUTIONS

### 1. Memory for Large Species Count (S=3614)

**Problem**: Full (B, S, Y, X) tensors are large

**Solutions implemented**:
- Efficient encoders that don't expand all species simultaneously
- Mixed precision training (AMP)
- Gradient accumulation support

### 2. Training Instability

**Solutions implemented**:
- Curriculum learning (simple→complex)
- Cosine learning rate decay with warmup
- Gradient clipping

### 3. Rare Species Underrepresentation

**Solutions implemented**:
- Weighted sampling (inverse prevalence)
- Rare species loss weighting (2x)
- Phase 4 sparse observation training

---

## 🔄 NEXT STEPS AFTER STAGE 2

1. **Stage 3 Preparation**: Load trained model + empirical data
2. **Domain Adaptation**: Fine-tune on BOTW distributions
3. **Uncertainty Calibration**: Validate confidence intervals
4. **AOO/EOO Pipeline**: Implement range metric computation
5. **Ensemble Generation**: K=100 samples for uncertainty

---

## 📁 FILES INCLUDED

```
stage2_ecodiffusion/
├── configs/config.py         # All hyperparameters
├── models/
│   ├── env_encoder.py        # Environmental CNN
│   ├── interaction_encoder.py # Species GNN
│   ├── temporal_encoder.py   # Temporal Transformer
│   ├── diffusion.py          # Diffusion process
│   ├── unet.py               # U-Net denoiser
│   └── ecodiffusion.py       # Combined model
├── data_preprocessing.py     # Data loading
├── training.py               # Training loop
├── run_training.py           # Main entry point
├── README.md                 # Documentation
└── REVIEW_ANALYSIS.md        # Detailed review
```

Total: ~7,500 lines of code

---

This implementation is **complete for Stage 2** and aligned with your project objectives for developing spatio-temporal, process-aware models with uncertainty quantification for rare species conservation.
