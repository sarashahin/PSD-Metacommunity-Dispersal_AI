# Stage 2 Implementation Review & Analysis

## Executive Summary

This document provides a comprehensive review of the Stage 2 implementation for the EcoDiffusion model, identifying issues, corrections needed, and alignment with project objectives.

---

## 1. STEP 1: DATA PREPROCESSING REVIEW

### ✅ What's Implemented Correctly:

1. **SimulationWorld Class**
   - Correctly loads all required arrays: IBM_B, P_t, ENV_r_field, C_topk_*, obs_masks
   - Handles missing data with sensible defaults
   - Extracts simulation parameters (DISPERSAL_RATE, BODY_MASS, etc.)
   - Properly derives P_t from IBM_B if missing

2. **Data Normalization**
   - log1p transform for biomass ✓ (handles orders of magnitude)
   - Percentile clipping (99.5) ✓ (removes outliers)
   - Environment normalization to [0,1] ✓

3. **Species Features for GNN**
   - 8 features capturing ecological attributes ✓
   - Includes network degrees, prevalence, interaction strength ✓

4. **Train/Val/Test Split**
   - 200/20/20 split ratio ✓
   - Reproducible with seed ✓
   - Uses training data only for statistics ✓

5. **Weighted Sampling**
   - Inverse prevalence weighting for rare species ✓
   - WeightedRandomSampler implementation ✓

### ⚠️ Issues Found & Corrections Needed:

#### Issue 1.1: Variable Species Count Handling
**Problem**: Your simulations have FIXED species count (3614 per world), but the code handles variable counts. This adds unnecessary complexity.

**Correction**: For your specific case, we can simplify by assuming fixed species count, but keep the flexible code for future use.

#### Issue 1.2: Graph Batching
**Problem**: The collate_fn keeps edge_index as a list, which doesn't work well for batched GNN processing.

**Correction**: Need to offset edge indices for batch processing (standard in PyTorch Geometric).

```python
# Current (problematic):
collated['condition']['edge_index'].append(val)

# Corrected:
# For batched GNN, need to offset indices
offset = batch_idx * n_species
edge_index_offset = edge_index + offset
```

#### Issue 1.3: Memory Efficiency for Large Species Count
**Problem**: Loading all 240 worlds × 3614 species × 50 timesteps into memory may exceed RAM.

**Correction**: Implement lazy loading or memory-mapped arrays for large-scale training.

#### Issue 1.4: Missing Data Validation
**Problem**: No validation that all worlds have consistent dimensions.

**Correction**: Add dimension checking in preprocessing.

---

## 2. STEP 2: MODEL ARCHITECTURE REVIEW

### ✅ What's Implemented Correctly:

1. **Environmental Encoder (CNN)**
   - Multi-scale feature extraction ✓
   - Spatial attention mechanism ✓
   - Both global and spatial features output ✓
   - Kernel size matches ENV_r_field length_scale ✓

2. **Interaction Encoder (GNN)**
   - GraphAttention layers with edge weights ✓
   - Handles sparse C_topk structure ✓
   - Multi-hop aggregation ✓
   - Skip connections ✓

3. **Temporal Encoder (Transformer)**
   - Positional encoding (learned) ✓
   - Multi-head attention ✓
   - Handles 50 timesteps ✓
   - Spatial pooling before temporal encoding ✓

4. **Diffusion Process**
   - Cosine beta schedule ✓
   - Forward/reverse process ✓
   - Training loss computation ✓
   - DDIM sampling for fast inference ✓

5. **U-Net Denoiser**
   - ResNet blocks with time embedding ✓
   - Skip connections ✓
   - Multi-resolution processing ✓

### ⚠️ Issues Found & Corrections Needed:

#### Issue 2.1: Memory Explosion in Environmental Encoder
**Problem**: Reshaping (B, S, Y, X) to (B*S, 3, Y, X) for 3614 species is very memory intensive.

Current: `x = x.view(B * S, self.in_channels, Y, X)`
For B=4, S=3614: Creates 14,456 × 3 × 20 × 20 tensor = 17.4M elements per forward pass

**Correction**: Use the EfficientEnvironmentEncoder that processes shared spatial features + per-species modulation.

#### Issue 2.2: GNN Batch Processing
**Problem**: InteractionEncoder processes each batch item separately in a loop, which is slow.

```python
# Current (slow):
for b in range(B):
    x = self.input_proj(species_features[b])
    ...
```

**Correction**: Use PyG's batch handling or implement vectorized batch processing.

#### Issue 2.3: Missing Conditioning Fusion Module
**Problem**: No module to combine environmental, interaction, and temporal encodings into unified conditioning.

**Correction**: Need ConditioningModule that fuses all encodings appropriately for U-Net injection.

#### Issue 2.4: U-Net Species Dimension Handling
**Problem**: Standard image U-Net treats channels as features, but we have (S, Y, X) where S=3614 species.

**Correction**: Need species-aware U-Net that processes species as a batch dimension or uses species-specific processing.

---

## 3. TRAINING PROCESS CLARIFICATION

### Yes, Stage 2 trains ONLY on simulation data

This is CORRECT because:

1. **Simulation provides ecological "grammar"**
   - 240 worlds with known processes
   - Explicit interactions, dispersal, environmental filtering
   - Perfect ground truth (no sampling bias)

2. **Empirical data comes later (Stage 3)**
   - Domain adaptation from sim → real
   - Calibration on BOTW, BioTIME
   - Fine-tuning with joint loss

### Training Procedure for Stage 2:

```
Phase 1 (Epochs 1-50): EQUILIBRIUM MODE
├── Input: ENV_r_field + coords
├── Target: P_last_final (final presence map)
├── Model learns: Environment → Distribution mapping
└── Loss: L_diffusion only

Phase 2 (Epochs 51-150): INTERACTION MODE  
├── Input: ENV + interaction graph
├── Target: P_last_final
├── Model learns: + Competitive exclusion
└── Loss: L_diffusion + L_prevalence

Phase 3 (Epochs 151-300): TEMPORAL MODE
├── Input: ENV + interactions + P_t[0:25]
├── Target: P_t[25:50]
├── Model learns: + Colonization/extinction dynamics
└── Loss: L_diffusion + L_prevalence + L_cooccurrence

Phase 4 (Epochs 301-500): INFILL MODE
├── Input: ENV + interactions + obs_mask_5
├── Target: P_last_final (full distribution)
├── Model learns: + Sparse data infilling
└── Loss: All losses + rare species weighting
```

---

## 4. ALIGNMENT WITH PROJECT OBJECTIVES

### Objective 1: "Spatio-temporal, process-aware models"
| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Spatio-temporal | Temporal Transformer + Spatial CNN | ✅ |
| Process-aware | GNN encodes interactions, ENV encodes niche | ✅ |
| Dispersal-aware | Position encoding, spatial autocorrelation | ⚠️ Implicit only |

**Recommendation**: Add explicit dispersal kernel encoding.

### Objective 2: "High-confidence maps for rare species"
| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Uncertainty quantification | Ensemble + diffusion stochasticity | ✅ |
| Rare species focus | Weighted sampling, Phase 4 sparse infilling | ✅ |
| Confidence calibration | Planned for Stage 3 | ⏳ |

### Objective 3: "Replace AOO/EOO point estimates with distributions"
| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Distribution output | Diffusion model outputs probability map | ✅ |
| Ensemble for CI | K=100 samples planned | ✅ Design |
| AOO/EOO computation | Not yet implemented | ❌ |

---

## 5. MISSING COMPONENTS

1. **`__init__.py` files** - For proper package imports
2. **Combined EcoDiffusion model** - Integrates all encoders + U-Net
3. **Training loop** - With curriculum learning
4. **Validation metrics** - AUC-ROC, Moran's I, Jaccard
5. **Auxiliary losses** - L_prevalence, L_cooccurrence, L_spatial
6. **Main training script** - Entry point with argument parsing
7. **Checkpoint management** - Save/load best models
8. **Logging utilities** - Training progress, metrics

---

## 6. PERFORMANCE CONSIDERATIONS

### Memory Optimization Strategies:
1. Use `torch.cuda.amp` for mixed precision ✅ (configured)
2. Gradient checkpointing for U-Net (not implemented)
3. Lazy data loading (not implemented)
4. Species chunking for very large S

### Computation Efficiency:
1. DDIM for fast sampling ✅
2. Efficient GNN with sparse ops ✅
3. Shared temporal encoding ✅

---

## 7. RECOMMENDED FIXES (Priority Order)

1. **HIGH**: Create combined EcoDiffusion model
2. **HIGH**: Implement training loop with curriculum
3. **HIGH**: Fix graph batching in collate_fn
4. **MEDIUM**: Implement auxiliary losses
5. **MEDIUM**: Add validation metrics
6. **LOW**: Memory optimization for large-scale training
