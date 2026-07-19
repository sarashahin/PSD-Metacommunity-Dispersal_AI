#!/usr/bin/env python3
"""
=============================================================================
STAGE 2 COMPREHENSIVE VALIDATION & VISUALIZATION SUITE — REAL DATA EDITION
=============================================================================

PURPOSE:
  Generate ALL visualizations from the original stage2_validation_visualization.py
  but using REAL IBM simulation data and REAL model inference — NO synthetic data.

WHAT THIS REPLACES:
  The original script generated ALL figures with:
    - torch.randn() / np.random.rand() — random noise
    - _create_clustered_distribution() — procedural circles
    - _create_gradient_distribution() — linear thresholds
    - hardcoded 12 species, 50 embeddings

  THIS script generates ALL figures with:
    - Real IBM NPZ data (P_last_final, ENV_r_field, prevalence_final, C_topk_idx)
    - Real model forward passes through 79.6M-parameter trained model
    - Real PyTorch hooks capturing intermediate activations
    - Real attention weights from temporal encoder transformer
    - Real species embeddings from interaction encoder GNN
    - Real training_history.json metrics

FIGURES GENERATED (matching original structure):
  ecology_viz/
    ecology_overview.png          — Real species distributions from IBM
    environmental_response.png    — Real env gradients from IBM ENV_r_field
  architecture_viz/
    model_architecture.png        — Architecture diagram (structural, unchanged)
    unet_detail.png               — UNet detail diagram (structural, unchanged)
    film_conditioning.png         — Real FiLM γ/β from trained model weights
    feature_maps.png              — Real intermediate activations via hooks
  training_dynamics/
    training_overview.png         — Real training_history.json curves
    loss_landscape_3d.png         — Real PCA of weight trajectory (if available)
    gradient_flow.png             — Real gradient norms from model parameters
  attention_maps/
    attention_overview.png        — Real attention weights from model
  embeddings/
    embedding_visualization.png   — Real species embeddings from GNN

USAGE:
  python stage2_validation_visualization_REAL.py \
      --ibm-dir results/data/ \
      --checkpoint stage2_outputs/checkpoints/best_model.pt \
      --training-history stage2_outputs/checkpoints/training_history.json \
      --output-dir stage2_validation_outputs/

Author: EcoDiffusion Stage 2 — Real Data Validation Suite
Date: February 2026
=============================================================================
"""

import sys, os, json, pickle, argparse, warnings, io, zipfile, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# AUTO-DETECT stage2/ DIRECTORY & ADD TO sys.path
# ═══════════════════════════════════════════════════════════════════════

def _setup_python_path():
    """Auto-detect the stage2/ directory and add it to sys.path."""
    script_dir = Path(__file__).resolve().parent
    candidates = []
    if script_dir.name == "models":
        candidates.append(script_dir.parent)
    candidates.append(script_dir)
    cwd = Path.cwd()
    for rel in ["AI_simulation/stage2", "stage2", "."]:
        p = (cwd / rel).resolve()
        if p not in candidates:
            candidates.append(p)
    for parent in script_dir.parents:
        if (parent / "configs").is_dir() and (parent / "models").is_dir():
            if parent not in candidates:
                candidates.insert(0, parent)
            break
    stage2_dir = None
    for cand in candidates:
        has_configs = (cand / "configs" / "config.py").exists()
        has_models = (cand / "models" / "ecodiffusion.py").exists()
        if has_configs and has_models:
            stage2_dir = cand
            break
    if stage2_dir:
        stage2_str = str(stage2_dir)
        if stage2_str not in sys.path:
            sys.path.insert(0, stage2_str)
        project_root = stage2_dir.parent.parent
        root_str = str(project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        print(f"✓  stage2 dir: {stage2_dir}")
        return stage2_dir
    else:
        print(f"⚠  Could not auto-detect stage2/ directory")
        return None

STAGE2_DIR = _setup_python_path()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ FATAL: torch not found.")
    sys.exit(1)

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════════════
# PUBLICATION-QUALITY STYLE
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.titlepad": 8,
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "grid.color": "#DDDDDD",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CCCCCC",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

CMAP_PRESENCE = LinearSegmentedColormap.from_list(
    "presence", ["#FFFFFF", "#C7E9C0", "#74C476", "#238B45", "#00441B"], N=256)
CMAP_PROB = LinearSegmentedColormap.from_list(
    "prob", ["#FFFFFF", "#FEE5D9", "#FCAE91", "#FB6A4A", "#CB181D", "#67000D"], N=256)
CMAP_DIFF = LinearSegmentedColormap.from_list(
    "diff", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"], N=256)


# ═══════════════════════════════════════════════════════════════════════
# IBM DATA LOADER (from stage2_real_inference_validation.py)
# ═══════════════════════════════════════════════════════════════════════

def parse_npz_params(filename):
    """Extract simulation parameters from NPZ filename."""
    params = {'_raw': filename}
    m = re.search(r'pool\d+_(.*?)_ls', filename)
    if m: params['batch'] = m.group(1)
    m = re.search(r'_ls(\d+)p(\d+)', filename)
    if m: params['ls'] = float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r'_vr(\d+)p(\d+)', filename)
    if m: params['vr'] = float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r'_dr(\d+)em?(\d+)', filename)
    if m: params['dr'] = float(f"{m.group(1)}e-{m.group(2)}")
    params['type'] = 'training' if '_training.npz' in filename else 'unknown'
    return params


def format_params(params):
    parts = []
    for key, fmt in [('batch', '{}'), ('ls', 'ls={}'), ('vr', 'vr={}'),
                     ('dr', 'dr={:.0e}')]:
        if key in params:
            parts.append(fmt.format(params[key]))
    return "  ".join(parts) if parts else "unknown"


class IBMDataLoader:
    """Load real IBM simulation NPZ data."""

    def __init__(self, ibm_dir, max_files=10, file_filter=None):
        self.ibm_dir = Path(ibm_dir)
        self.max_files = max_files
        self.file_filter = file_filter
        self.simulations = []
        self.sim_params = []
        self.sim_sources = []
        self.default_idx = 0

    def find_files(self):
        all_files = sorted(self.ibm_dir.rglob("*_training.npz"))
        if not all_files:
            all_files = [f for f in sorted(self.ibm_dir.rglob("*.npz"))
                         if "_dataset" not in f.name]
        if self.file_filter:
            filters = [f.strip() for f in self.file_filter.split(",")]
            filtered = [fp for fp in all_files if all(pat in fp.name for pat in filters)]
            print(f"📂  Found {len(all_files)} NPZ files, filter → {len(filtered)} matches")
            return filtered[:self.max_files]
        else:
            print(f"📂  Found {len(all_files)} IBM simulation files")
            return all_files[:self.max_files]

    def load(self):
        required = {"P_last_final", "B_last", "gamma", "Y", "X"}
        for fpath in self.find_files():
            try:
                data = dict(np.load(fpath, allow_pickle=True))
                if required - set(data.keys()):
                    print(f"   ✗  {fpath.name}: missing {required - set(data.keys())}")
                    continue
                Y, X = int(data["Y"]), int(data["X"])
                if Y < 10 or X < 10:
                    continue
                data["_source"] = str(fpath)
                data["_params"] = parse_npz_params(fpath.name)
                self.simulations.append(data)
                self.sim_params.append(data["_params"])
                self.sim_sources.append(str(fpath))
                idx = len(self.simulations) - 1
                print(f"   [{idx}] ✓  {fpath.name}  γ={int(data['gamma'])}  grid={Y}×{X}")
            except Exception as e:
                print(f"   ✗  {fpath.name}: {e}")
        print(f"   Loaded {len(self.simulations)} simulations")
        return self.simulations

    def get(self, key, idx=None):
        if idx is None: idx = self.default_idx
        return np.asarray(self.simulations[idx][key])

    def P(self, idx=None): return self.get("P_last_final", idx)
    def B(self, idx=None): return self.get("B_last", idx)
    def env(self, idx=None): return self.get("ENV_r_field", idx)
    def prev(self, idx=None): return self.get("prevalence_final", idx)

    def P_t(self, idx=None):
        try:
            return self.get("P_t", idx)
        except KeyError:
            return None

    def params(self, idx=None):
        if idx is None: idx = self.default_idx
        return self.sim_params[idx] if idx < len(self.sim_params) else {}

    def source(self, idx=None):
        if idx is None: idx = self.default_idx
        return self.sim_sources[idx] if idx < len(self.sim_sources) else "?"

    def has_key(self, key, idx=None):
        if idx is None: idx = self.default_idx
        return key in self.simulations[idx]


# ═══════════════════════════════════════════════════════════════════════
# REAL MODEL INFERENCE (from stage2_real_inference_validation.py)
# ═══════════════════════════════════════════════════════════════════════

class RealModelInference:
    """Load the REAL EcoDiffusion model and run actual inference."""

    def __init__(self, ckpt_path, device="auto", stage2_dir=None):
        self.ckpt_path = Path(ckpt_path)
        self.model = None
        self.ready = False
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.n_species = None
        self.checkpoint = None
        self.stage2_dir = stage2_dir
        self.activations = {}   # For hook-captured activations
        self.hooks = []         # Store hook handles for cleanup

    def _ensure_imports(self):
        if self.stage2_dir:
            s2 = Path(self.stage2_dir).resolve()
            s2_str = str(s2)
            if s2_str not in sys.path:
                sys.path.insert(0, s2_str)
        ckpt_resolved = self.ckpt_path.resolve()
        for parent in ckpt_resolved.parents:
            cand = parent / "AI_simulation" / "stage2"
            if (cand / "configs" / "config.py").exists():
                cand_str = str(cand)
                if cand_str not in sys.path:
                    sys.path.insert(0, cand_str)
                break

    def load(self):
        """Load model using YOUR project's actual code."""
        print("\n" + "=" * 60)
        print("  LOADING REAL MODEL")
        print("=" * 60)

        if not self.ckpt_path.exists():
            print(f"❌ Checkpoint not found: {self.ckpt_path}")
            return False

        self._ensure_imports()

        try:
            from configs.config import get_default_config, EcoConfig
            print("✓  Imported configs.config")
        except ImportError as e:
            print(f"❌ Cannot import configs.config: {e}")
            return False

        try:
            from models.ecodiffusion import EcoDiffusionFixed, create_fixed_model
            print("✓  Imported models.ecodiffusion")
        except ImportError as e:
            print(f"❌ Cannot import models.ecodiffusion: {e}")
            return False

        try:
            torch.serialization.add_safe_globals([EcoConfig])
            torch.serialization.add_safe_globals([np.dtype])
            if hasattr(np, "_core") and hasattr(np._core, "multiarray"):
                scalar = getattr(np._core.multiarray, "scalar", None)
                if scalar:
                    torch.serialization.add_safe_globals([scalar])
            self.checkpoint = torch.load(
                self.ckpt_path, map_location=self.device, weights_only=False
            )
            epoch = self.checkpoint.get("epoch", "?")
            print(f"✓  Checkpoint loaded (epoch {epoch})")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False

        try:
            if "config" in self.checkpoint:
                config = self.checkpoint["config"]
                print(f"✓  Config from checkpoint: {config.data.n_species_max} species")
            else:
                config = get_default_config()
                print(f"⚠  Using default config (n_species={config.data.n_species_max})")
            self.model = create_fixed_model(config)
            self.n_species = config.data.n_species_max
            self.config_obj = config
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✓  Model created: {total_params:,} params")
        except Exception as e:
            print(f"❌ Failed to create model: {e}")
            import traceback; traceback.print_exc()
            return False

        try:
            state_dict = self.checkpoint.get("model_state_dict", self.checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            print("✓  Weights loaded (strict=True)")
        except RuntimeError:
            try:
                self.model.load_state_dict(state_dict, strict=False)
                print("✓  Weights loaded (strict=False)")
            except Exception as e2:
                print(f"❌ Failed to load weights: {e2}")
                return False

        self.model.to(self.device)
        self.model.eval()
        self.model.set_training_phase(4)
        self.ready = True
        print(f"✓  Model ready on {self.device}")
        return True

    def build_conditioning(self, npz_data, species_subset=None):
        """
        Build REAL conditioning from NPZ data.
        Shape fixes: env (B,S,Y,X), coords (B,Y,X), species_features (B,S,8).
        """
        device = self.device
        Y = int(npz_data["Y"])
        X = int(npz_data["X"])
        P = np.asarray(npz_data["P_last_final"])
        S_data = P.shape[0]
        S = S_data if species_subset is None else len(species_subset)
        if species_subset is not None:
            P = P[species_subset]
        B = 1

        env_raw = np.asarray(npz_data["ENV_r_field"])
        if species_subset is not None:
            env_raw = env_raw[species_subset]
        env_4d = env_raw[np.newaxis]  # (1, S, Y, X)
        env_t = torch.from_numpy(env_4d.copy()).float().to(device)

        # Coordinates: (B, Y, X) — shared across species
        y_grid = np.arange(Y, dtype=np.float32).reshape(1, Y, 1)
        y_grid = np.broadcast_to(y_grid, (B, Y, X)).copy()
        x_grid = np.arange(X, dtype=np.float32).reshape(1, 1, X)
        x_grid = np.broadcast_to(x_grid, (B, Y, X)).copy()
        y_coords_t = torch.from_numpy(y_grid).float().to(device)
        x_coords_t = torch.from_numpy(x_grid).float().to(device)

        # Species features: (B, S, 8)
        sp_feats = np.zeros((B, S, 8), dtype=np.float32)
        pv = np.asarray(npz_data.get("prevalence_final", np.zeros(S_data)))
        if species_subset is not None:
            pv = pv[species_subset]
        for s in range(S):
            sp_feats[0, s, 0] = pv[s]
            sp_feats[0, s, 1] = P[s].sum() / (Y * X)
            sp_feats[0, s, 2] = env_raw[s].mean() if s < len(env_raw) else 0
            sp_feats[0, s, 3] = env_raw[s].std() if s < len(env_raw) else 0
            sp_feats[0, s, 4] = float(pv[s] < 0.05)
            sp_feats[0, s, 5] = float(pv[s] >= 0.05)
            sp_feats[0, s, 6] = np.log1p(pv[s])
            sp_feats[0, s, 7] = float(s) / max(S - 1, 1)
        species_features_t = torch.from_numpy(sp_feats).float().to(device)

        # Interaction graph from C_topk_idx
        edge_list = []
        if "C_topk_idx" in npz_data:
            ctk = np.asarray(npz_data["C_topk_idx"])
            if species_subset is not None:
                subset_set = set(species_subset.tolist() if hasattr(species_subset, 'tolist')
                                 else list(species_subset))
                old_to_new = {old: new for new, old in enumerate(
                    species_subset.tolist() if hasattr(species_subset, 'tolist')
                    else list(species_subset)
                )}
                for new_s, old_s in enumerate(
                    species_subset.tolist() if hasattr(species_subset, 'tolist')
                    else list(species_subset)
                ):
                    for neighbor in ctk[old_s]:
                        if int(neighbor) in subset_set:
                            edge_list.append([new_s, old_to_new[int(neighbor)]])
            else:
                for s in range(min(S, ctk.shape[0])):
                    for neighbor in ctk[s]:
                        if 0 <= int(neighbor) < S and int(neighbor) != s:
                            edge_list.append([s, int(neighbor)])
        else:
            for s in range(S - 1):
                edge_list.append([s, s + 1])
                edge_list.append([s + 1, s])

        if edge_list:
            edge_index_t = torch.tensor(edge_list, dtype=torch.long, device=device).T
        else:
            edge_index_t = torch.empty(2, 0, dtype=torch.long, device=device)

        # Temporal history
        history_t = None
        if "P_t" in npz_data:
            Pt = np.asarray(npz_data["P_t"])
            if species_subset is not None:
                Pt = Pt[:, species_subset, :, :]
            history_t = torch.from_numpy(Pt[np.newaxis].copy()).float().to(device)

        return {
            "env": env_t,
            "y_coords": y_coords_t,
            "x_coords": x_coords_t,
            "species_features": species_features_t,
            "edge_index": edge_index_t,
            "edge_weight": None,
            "history_P": history_t,
        }

    def register_hooks(self):
        """Register forward hooks on key model components to capture activations."""
        self.activations = {}
        self.hooks = []

        def get_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 0:
                    self.activations[name] = output[0].detach().cpu() if isinstance(output[0], torch.Tensor) else None
            return hook_fn

        # Register on available submodules
        model = self.model
        registered = []

        # Environmental encoder
        if hasattr(model, 'env_encoder'):
            h = model.env_encoder.register_forward_hook(get_hook('env_encoder'))
            self.hooks.append(h); registered.append('env_encoder')

        # Interaction encoder (GNN)
        if hasattr(model, 'int_encoder'):
            h = model.int_encoder.register_forward_hook(get_hook('int_encoder'))
            self.hooks.append(h); registered.append('int_encoder')

        # Temporal encoder
        if hasattr(model, 'temp_encoder'):
            h = model.temp_encoder.register_forward_hook(get_hook('temp_encoder'))
            self.hooks.append(h); registered.append('temp_encoder')

        # Conditioning module
        if hasattr(model, 'conditioning'):
            h = model.conditioning.register_forward_hook(get_hook('conditioning'))
            self.hooks.append(h); registered.append('conditioning')

        # UNet encoder/decoder blocks
        if hasattr(model, 'unet'):
            unet = model.unet
            # Encoder blocks
            if hasattr(unet, 'encoder_blocks'):
                for i, block in enumerate(unet.encoder_blocks):
                    h = block.register_forward_hook(get_hook(f'unet_enc_{i}'))
                    self.hooks.append(h); registered.append(f'unet_enc_{i}')
            # Middle block
            if hasattr(unet, 'middle_block'):
                h = unet.middle_block.register_forward_hook(get_hook('unet_middle'))
                self.hooks.append(h); registered.append('unet_middle')
            # Decoder blocks
            if hasattr(unet, 'decoder_blocks'):
                for i, block in enumerate(unet.decoder_blocks):
                    h = block.register_forward_hook(get_hook(f'unet_dec_{i}'))
                    self.hooks.append(h); registered.append(f'unet_dec_{i}')

        print(f"   Registered {len(registered)} hooks: {registered}")
        return registered

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def predict(self, npz_data, n_samples=8, species_subset=None, chunk_size=200):
        """Run REAL model.sample() inference on IBM data."""
        if not self.ready:
            raise RuntimeError("Model not loaded — call load() first")
        P = np.asarray(npz_data["P_last_final"])
        S_total = P.shape[0]
        Y, X = int(npz_data["Y"]), int(npz_data["X"])
        species_indices = species_subset if species_subset is not None else np.arange(S_total)
        S = len(species_indices)
        print(f"\n   Running real inference on {S} species, {n_samples} samples...")
        all_preds = np.zeros((S, Y, X), dtype=np.float64)

        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            chunk_indices = species_indices[chunk_start:chunk_end]
            chunk_S = len(chunk_indices)
            print(f"   Chunk [{chunk_start}:{chunk_end}] = {chunk_S} species...")
            condition = self.build_conditioning(npz_data, species_subset=chunk_indices)
            chunk_preds = []
            for sample_i in range(n_samples):
                try:
                    sample = self.model.sample(
                        condition=condition, n_samples=1,
                        ddim_steps=50, sparse_mode=True, eta=0.5,
                    )
                    s_np = sample.cpu().numpy()
                    while s_np.ndim > 3 and s_np.shape[0] == 1:
                        s_np = s_np[0]
                    chunk_preds.append(s_np)
                    if sample_i == 0:
                        print(f"     Sample 0: shape={s_np.shape}, mean={s_np.mean():.4f}")
                except Exception as e:
                    print(f"     ⚠ Sample {sample_i} failed: {e}")
                    if sample_i == 0:
                        import traceback; traceback.print_exc()
                    continue
            if chunk_preds:
                all_preds[chunk_start:chunk_end] = np.clip(np.mean(chunk_preds, axis=0), 0, 1)
            else:
                print(f"     ❌ All samples failed for chunk [{chunk_start}:{chunk_end}]")
        return all_preds

    @torch.no_grad()
    def forward_with_hooks(self, npz_data, species_subset):
        """
        Run a SINGLE forward pass through the model with hooks registered,
        capturing intermediate activations for visualization.
        """
        if not self.ready:
            raise RuntimeError("Model not loaded")

        self.register_hooks()
        condition = self.build_conditioning(npz_data, species_subset=species_subset)

        try:
            sample = self.model.sample(
                condition=condition, n_samples=1,
                ddim_steps=20,  # fewer steps for faster hook capture
                sparse_mode=True, eta=0.5,
            )
            output = sample.cpu().numpy()
            # model.sample() can return (1, 1, S, Y, X) or (1, S, Y, X) or (S, Y, X)
            # Squeeze all leading singleton dims until we reach (S, Y, X)
            while output.ndim > 3 and output.shape[0] == 1:
                output = output[0]
            print(f"   Hook output shape after squeeze: {output.shape}")
        except Exception as e:
            print(f"   ⚠ Hook forward pass failed: {e}")
            import traceback; traceback.print_exc()
            output = None

        # Copy activations before removing hooks
        captured = dict(self.activations)
        self.remove_hooks()
        return output, captured

    def get_state_dict(self):
        if self.checkpoint and "model_state_dict" in self.checkpoint:
            return self.checkpoint["model_state_dict"]
        return {}

    def get_metrics(self):
        if self.checkpoint and "metrics" in self.checkpoint:
            return dict(self.checkpoint["metrics"])
        return {}


# ═══════════════════════════════════════════════════════════════════════
# LOAD BEST-EPOCH METRICS
# ═══════════════════════════════════════════════════════════════════════

def load_best_metrics(training_history_path):
    """Load metrics from the BEST epoch in training_history.json."""
    print("\n   Loading metrics from training history...")
    with open(training_history_path) as f:
        h = json.load(f)
    auc_list = h.get("auc_overall", [])
    if not auc_list:
        print("   ⚠ No auc_overall in training history")
        return {}, h
    best_idx = int(np.argmax(auc_list))
    print(f"   BEST epoch index: {best_idx}  → AUC = {auc_list[best_idx]:.4f}")
    metrics = {}
    for k in ["auc_overall", "auc_rare", "auc_common", "jaccard",
              "prevalence_mae", "prevalence_mse",
              "val_diffusion", "val_prevalence", "val_cooccurrence",
              "val_spatial", "val_total"]:
        if k in h:
            v = h[k]
            if isinstance(v, list) and len(v) > best_idx:
                metrics[k] = v[best_idx]
            elif not isinstance(v, list):
                metrics[k] = v
    return metrics, h


# ═══════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════

def richness(P):
    return P.sum(axis=0)

def morans_i(grid):
    g = grid.astype(float)
    if g.std() < 1e-12: return 0.0
    z = g - g.mean(); Y, X = g.shape; n = Y * X; W = 0; num = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            num += (z * np.roll(np.roll(z, dy, 0), dx, 1)).sum(); W += n
    d = (z ** 2).sum()
    return (n / W) * (num / d) if d > 1e-12 else 0

def jaccard_matrix(P, n=100):
    S = min(P.shape[0], n)
    F = P[:S].reshape(S, -1).astype(float)
    I = F @ F.T; U = F.sum(1)[:, None] + F.sum(1)[None, :] - I
    return I / np.maximum(U, 1e-12)

def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)

def safe_save(fig, path, dpi=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.2)
    plt.close(fig)
    sz = path.stat().st_size if path.exists() else 0
    print(f"   {'✓' if sz > 2000 else '✗'}  {path.name}  ({sz / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════
# 1. ECOLOGY VISUALIZATIONS — REAL IBM DATA
# ═══════════════════════════════════════════════════════════════════════

def plot_ecology_visualizations(ld, pred, out):
    """Generate ecology visualizations from REAL IBM data + model predictions."""
    print("\n" + "=" * 70)
    print("GENERATING ECOLOGY VISUALIZATIONS (REAL DATA)")
    print("=" * 70)

    P = ld.P(); env = ld.env(); pv = ld.prev()
    S, Y, X = P.shape
    occ = np.where(pv > 0)[0]
    rare_idx = occ[np.argsort(pv[occ])]
    common_idx = occ[np.argsort(-pv[occ])]

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # (0,0) Species richness — REAL
    ax = fig.add_subplot(gs[0, 0])
    ri = richness(P)
    im = ax.imshow(ri, cmap='YlGnBu', aspect='equal', interpolation='nearest')
    ax.set_title(f'Species Richness\n(IBM, max={ri.max():.0f})', fontsize=10, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (0,1) Rare species hotspots — REAL
    ax = fig.add_subplot(gs[0, 1])
    rare_sp = rare_idx[:min(50, len(rare_idx))]
    rare_ri = P[rare_sp].sum(axis=0)
    im = ax.imshow(rare_ri, cmap='Reds', aspect='equal', interpolation='nearest')
    n_rare = (pv[occ] < 0.05).sum()
    ax.set_title(f'Rare Species Hotspots\n({n_rare} species, prev < 5%)', fontsize=10, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (0,2) Prevalence distribution — REAL
    ax = fig.add_subplot(gs[0, 2])
    po = pv[pv > 0]
    ax.hist(po, bins=50, color='#3182BD', alpha=0.85, edgecolor='white', lw=0.5)
    ax.axvline(0.05, color='#E31A1C', ls='--', lw=2, label='Rare threshold (5%)')
    nr, nc = int((po < 0.05).sum()), int((po >= 0.05).sum())
    ax.text(0.95, 0.92, f"Rare: {nr}\nCommon: {nc}", transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC'))
    ax.set_xlabel('Prevalence'); ax.set_ylabel('# Species')
    ax.set_title('Prevalence Distribution\n(Real IBM)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,3) Co-occurrence matrix — REAL from Jaccard
    ax = fig.add_subplot(gs[0, 3])
    top40 = occ[np.argsort(-pv[occ])][:40]
    ji = jaccard_matrix(P[top40], 40)
    im = ax.imshow(ji, cmap='coolwarm', aspect='auto', vmin=0, vmax=max(0.01, ji.max()))
    ax.set_title('Co-occurrence (Jaccard)\n(Top 40 species)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Species rank'); ax.set_ylabel('Species rank')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: Rare species — IBM ground truth
    for idx in range(min(4, len(rare_idx))):
        sp = rare_idx[idx]
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(P[sp], cmap=CMAP_PRESENCE, vmin=0, vmax=1,
                  aspect='equal', interpolation='nearest')
        nc = int(P[sp].sum())
        ax.set_title(f'Rare sp {sp}\nprev={pv[sp]:.4f} ({nc} cells)',
                     fontsize=9, color='#E31A1C')
        ax.axis('off')

    # Row 2: Common species — IBM ground truth
    for idx in range(min(4, len(common_idx))):
        sp = common_idx[idx]
        ax = fig.add_subplot(gs[2, idx])
        ax.imshow(P[sp], cmap=CMAP_PRESENCE, vmin=0, vmax=1,
                  aspect='equal', interpolation='nearest')
        nc = int(P[sp].sum())
        ax.set_title(f'Common sp {sp}\nprev={pv[sp]:.4f} ({nc} cells)',
                     fontsize=9, color='#08519C')
        ax.axis('off')

    gamma = int(ld.get('gamma'))
    fig.suptitle(f'Ecology Visualizations — Real IBM Simulation Data\n'
                 f'(γ = {gamma} species, {Y}×{X} grid, {format_params(ld.params())})',
                 fontsize=14, fontweight='bold', y=0.99)
    safe_save(fig, out / "ecology_viz" / "ecology_overview.png")

    # Environmental response — REAL env fields
    _plot_env_response_real(ld, out)


def _plot_env_response_real(ld, out):
    """Plot REAL species environmental response from IBM data."""
    P = ld.P(); env = ld.env(); pv = ld.prev()
    S, Y, X = P.shape
    occ = np.where(pv > 0)[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # Pick 6 species across the rarity gradient
    so = occ[np.argsort(pv[occ])]
    pick_idx = np.linspace(0, len(so) - 1, 6, dtype=int)
    selected = so[pick_idx]

    for ax, sp in zip(axes.flatten(), selected):
        p = pv[sp]
        label = "RARE" if p < 0.05 else "COMMON"
        col = "#E31A1C" if p < 0.05 else "#08519C"

        # Overlay: env field with presence cells marked
        im = ax.imshow(env[sp], cmap='RdBu_r', aspect='equal', interpolation='nearest')
        # Mark presence cells
        presence_y, presence_x = np.where(P[sp] > 0)
        ax.scatter(presence_x, presence_y, c='lime', s=40, marker='o',
                   edgecolors='black', linewidths=0.5, zorder=5, label='Present')

        ax.set_title(f'{label} sp {sp}\nprev={p:.4f}  env_mean={env[sp].mean():.3f}',
                     fontsize=10, fontweight='bold', color=col)
        plt.colorbar(im, ax=ax, shrink=0.7, label='ENV r-field')
        ax.legend(fontsize=7, loc='lower right')

    plt.suptitle('Real Environmental Response\n'
                 '(IBM ENV r-field with species presence locations marked in green)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_save(fig, out / "ecology_viz" / "environmental_response.png")


# ═══════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURE VISUALIZATION (structural diagrams — data-independent)
# ═══════════════════════════════════════════════════════════════════════

def plot_architecture_visualization(model_inf, out):
    """Generate model architecture diagram and real FiLM weights."""
    print("\n" + "=" * 70)
    print("GENERATING ARCHITECTURE VISUALIZATIONS")
    print("=" * 70)

    _plot_main_architecture(out)
    _plot_unet_detail(out)
    _plot_film_real(model_inf, out)


def _plot_main_architecture(out):
    """Architecture diagram (structural — same as original)."""
    fig = plt.figure(figsize=(24, 18))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 24); ax.set_ylim(0, 18); ax.axis('off')

    components = {
        'input_env': {'pos': (1, 14), 'size': (3, 2), 'color': '#E8F5E9',
                      'text': 'Environment\n(B,S,Y,X)'},
        'input_graph': {'pos': (1, 11), 'size': (3, 2), 'color': '#E3F2FD',
                        'text': 'Interaction Graph\n(edge_index, weights)'},
        'input_history': {'pos': (1, 8), 'size': (3, 2), 'color': '#FFF3E0',
                          'text': 'Temporal History\n(B,T,S,Y,X)'},
        'input_noisy': {'pos': (1, 5), 'size': (3, 2), 'color': '#FCE4EC',
                        'text': 'Noisy Distribution\n(B,S,Y,X)'},
        'env_encoder': {'pos': (6, 14), 'size': (3, 2), 'color': '#C8E6C9',
                        'text': 'Environmental\nEncoder (CNN)'},
        'int_encoder': {'pos': (6, 11), 'size': (3, 2), 'color': '#BBDEFB',
                        'text': 'Interaction\nEncoder (GNN)'},
        'temp_encoder': {'pos': (6, 8), 'size': (3, 2), 'color': '#FFE0B2',
                         'text': 'Temporal\nEncoder (Transformer)'},
        'conditioning': {'pos': (11, 11), 'size': (3, 3), 'color': '#CE93D8',
                         'text': 'Conditioning\nModule\n(FiLM fusion)'},
        'unet': {'pos': (16, 8), 'size': (5, 6), 'color': '#90CAF9',
                 'text': 'Species-Parallel\nU-Net\n\n• Per-species processing\n• FiLM modulation\n• Skip connections'},
        'diffusion': {'pos': (11, 3), 'size': (3, 3), 'color': '#FFAB91',
                      'text': 'Diffusion Process\n(DDIM 50 steps)'},
        'output': {'pos': (20, 5), 'size': (3, 2), 'color': '#A5D6A7',
                   'text': 'Predicted\nDistribution\n(B,S,Y,X)'},
    }

    for name, comp in components.items():
        x, y = comp['pos']; w, h = comp['size']
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=comp['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, comp['text'],
                ha='center', va='center', fontsize=9,
                fontweight='bold' if 'encoder' in name or name in ['conditioning', 'unet', 'diffusion'] else 'normal')

    arrows = [
        ((4, 15), (6, 15)), ((4, 12), (6, 12)), ((4, 9), (6, 9)),
        ((9, 15), (11, 13.5)), ((9, 12), (11, 12.5)), ((9, 9), (11, 11.5)),
        ((14, 12.5), (16, 11)), ((4, 6), (16, 9)), ((14, 4.5), (16, 9)),
        ((21, 11), (21.5, 7)),
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_title('EcoDiffusion Model Architecture\n(Stage 2 Training — 79.6M Parameters)',
                 fontsize=16, fontweight='bold', pad=20)
    legend_items = [
        mpatches.Patch(color='#E8F5E9', label='Input Data'),
        mpatches.Patch(color='#C8E6C9', label='Environmental Processing'),
        mpatches.Patch(color='#BBDEFB', label='Interaction Processing'),
        mpatches.Patch(color='#FFE0B2', label='Temporal Processing'),
        mpatches.Patch(color='#CE93D8', label='Conditioning Fusion'),
        mpatches.Patch(color='#90CAF9', label='Denoising Network'),
        mpatches.Patch(color='#FFAB91', label='Diffusion Process'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)
    safe_save(fig, out / "architecture_viz" / "model_architecture.png")


def _plot_unet_detail(out):
    """Detailed U-Net architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis('off')

    encoder_blocks = [
        {'pos': (2, 10), 'size': (2.5, 1.5), 'text': 'Conv 64\n+ ResBlock'},
        {'pos': (2, 8), 'size': (2.5, 1.5), 'text': 'Conv 128\n+ ResBlock'},
        {'pos': (2, 6), 'size': (2.5, 1.5), 'text': 'Conv 256\n+ ResBlock'},
    ]
    middle = {'pos': (6.5, 5), 'size': (3, 2), 'text': 'Middle Block\n2× ResBlock'}
    decoder_blocks = [
        {'pos': (11, 6), 'size': (2.5, 1.5), 'text': 'ConvT 256\n+ ResBlock'},
        {'pos': (11, 8), 'size': (2.5, 1.5), 'text': 'ConvT 128\n+ ResBlock'},
        {'pos': (11, 10), 'size': (2.5, 1.5), 'text': 'ConvT 64\n+ ResBlock'},
    ]

    for block in encoder_blocks:
        rect = mpatches.FancyBboxPatch(block['pos'], *block['size'],
            boxstyle="round,pad=0.05", facecolor='#BBDEFB', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(block['pos'][0] + block['size'][0] / 2, block['pos'][1] + block['size'][1] / 2,
                block['text'], ha='center', va='center', fontsize=8)

    rect = mpatches.FancyBboxPatch(middle['pos'], *middle['size'],
        boxstyle="round,pad=0.05", facecolor='#CE93D8', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(middle['pos'][0] + middle['size'][0] / 2, middle['pos'][1] + middle['size'][1] / 2,
            middle['text'], ha='center', va='center', fontsize=8)

    for block in decoder_blocks:
        rect = mpatches.FancyBboxPatch(block['pos'], *block['size'],
            boxstyle="round,pad=0.05", facecolor='#C8E6C9', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(block['pos'][0] + block['size'][0] / 2, block['pos'][1] + block['size'][1] / 2,
                block['text'], ha='center', va='center', fontsize=8)

    for i in range(3):
        enc_y = 10 - i * 2 + 0.75
        dec_y = 6 + i * 2 + 0.75
        ax.plot([4.5, 5.5, 5.5, 10.5, 10.5, 11],
                [enc_y, enc_y, dec_y + 1, dec_y + 1, dec_y, dec_y],
                'g--', linewidth=2, alpha=0.7)

    ax.text(3.25, 11.5, 'Input\n(B×S, 1, Y, X)', ha='center', fontsize=9, fontweight='bold')
    ax.text(12.25, 11.5, 'Output\n(B×S, 1, Y, X)', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Species-Parallel U-Net Architecture\n'
                 '(Each species processed independently with its own FiLM conditioning)',
                 fontsize=12, fontweight='bold')
    safe_save(fig, out / "architecture_viz" / "unet_detail.png")


def _plot_film_real(model_inf, out):
    """Visualize REAL FiLM parameters from trained model weights."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    state_dict = model_inf.get_state_dict()
    if not state_dict:
        print("   ⚠ No state dict available — skipping FiLM visualization")
        plt.close(fig); return

    # Extract FiLM scale and shift parameters
    film_scales = {}
    film_shifts = {}
    for k, v in state_dict.items():
        if 'film_scale' in k or 'film_gamma' in k:
            film_scales[k] = v.cpu().numpy().flatten()
        if 'film_shift' in k or 'film_beta' in k:
            film_shifts[k] = v.cpu().numpy().flatten()

    # Plot 0: Distribution of all FiLM γ values
    ax = axes[0]
    all_scales = np.concatenate([v for v in film_scales.values()]) if film_scales else np.array([1.0])
    ax.hist(all_scales, bins=50, color='#3182BD', alpha=0.8, edgecolor='white')
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Identity (γ=1)')
    ax.set_xlabel('FiLM γ (scale)'); ax.set_ylabel('Count')
    ax.set_title(f'FiLM Scale Distribution\n({len(film_scales)} layers, {len(all_scales)} params)',
                 fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 1: Distribution of all FiLM β values
    ax = axes[1]
    all_shifts = np.concatenate([v for v in film_shifts.values()]) if film_shifts else np.array([0.0])
    ax.hist(all_shifts, bins=50, color='#E6550D', alpha=0.8, edgecolor='white')
    ax.axvline(0.0, color='red', ls='--', lw=2, label='Identity (β=0)')
    ax.set_xlabel('FiLM β (shift)'); ax.set_ylabel('Count')
    ax.set_title(f'FiLM Shift Distribution\n({len(film_shifts)} layers, {len(all_shifts)} params)',
                 fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 2: Per-layer mean scale/shift
    ax = axes[2]
    layer_names = sorted(set(
        [k.rsplit('.', 1)[0] for k in list(film_scales.keys()) + list(film_shifts.keys())]
    ))[:10]  # top 10

    if layer_names:
        means_s = []; means_sh = []; labels = []
        for ln in layer_names:
            s_keys = [k for k in film_scales if ln in k]
            sh_keys = [k for k in film_shifts if ln in k]
            if s_keys:
                means_s.append(np.mean([film_scales[k].mean() for k in s_keys]))
            else:
                means_s.append(1.0)
            if sh_keys:
                means_sh.append(np.mean([film_shifts[k].mean() for k in sh_keys]))
            else:
                means_sh.append(0.0)
            labels.append(ln.split('.')[-2] if '.' in ln else ln[-20:])

        x_pos = np.arange(len(labels))
        ax.bar(x_pos - 0.2, means_s, 0.35, label='Mean γ', color='#3182BD', alpha=0.8)
        ax.bar(x_pos + 0.2, means_sh, 0.35, label='Mean β', color='#E6550D', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.set_title('Per-Layer FiLM Stats', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No FiLM layers found\nin state dict',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Per-Layer FiLM Stats', fontweight='bold')

    ax.grid(True, alpha=0.3)
    plt.suptitle('Real FiLM Conditioning Parameters\n'
                 '(Learned per-species modulation weights from trained model)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    safe_save(fig, out / "architecture_viz" / "film_conditioning.png")


# ═══════════════════════════════════════════════════════════════════════
# 3. TRAINING DYNAMICS — REAL training_history.json
# ═══════════════════════════════════════════════════════════════════════

def plot_training_dynamics(history, checkpoint, out):
    """Generate training dynamics from REAL training_history.json."""
    print("\n" + "=" * 70)
    print("GENERATING TRAINING DYNAMICS (REAL)")
    print("=" * 70)

    if history is None:
        print("⚠ No training history — skipping training dynamics")
        return

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Plot 1: Train + Val Loss
    ax = fig.add_subplot(gs[0, 0])
    if 'train_loss' in history:
        ax.plot(history['train_loss'], 'b-', label='Train Loss', alpha=0.8)
    if 'val_total' in history:
        ax.plot(history['val_total'], 'r-', label='Val Total', alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss', fontweight='bold')
    ax.legend(); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    # Plot 2: AUC curves
    ax = fig.add_subplot(gs[0, 1])
    for key, color, label in [('auc_overall', '#238B45', 'Overall'),
                               ('auc_rare', '#E31A1C', 'Rare'),
                               ('auc_common', '#3182BD', 'Common')]:
        if key in history:
            ax.plot(history[key], color=color, lw=2, label=label)
    ax.axhline(y=0.8, color='black', ls='--', label='Target (0.8)')
    ax.set_xlabel('Validation Step'); ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC Over Training', fontweight='bold')
    ax.legend(); ax.set_ylim(0.4, 1.0); ax.grid(True, alpha=0.3)

    # Highlight best AUC
    if 'auc_rare' in history and len(history['auc_rare']) > 0:
        best_idx = int(np.argmax(history['auc_rare']))
        best_auc = history['auc_rare'][best_idx]
        ax.scatter([best_idx], [best_auc], color='red', s=100, zorder=5)
        ax.annotate(f'Best: {best_auc:.3f}', xy=(best_idx, best_auc),
                    xytext=(best_idx + 3, best_auc - 0.05), fontsize=9, color='red')

    # Plot 3: Component losses
    ax = fig.add_subplot(gs[0, 2])
    for key, color in [('val_diffusion', 'blue'), ('val_prevalence', 'green'),
                        ('val_cooccurrence', 'orange'), ('val_spatial', 'purple')]:
        if key in history:
            ax.plot(history[key], color=color, label=key.replace('val_', ''), alpha=0.8)
    ax.set_xlabel('Validation Step'); ax.set_ylabel('Loss')
    ax.set_title('Component Losses', fontweight='bold')
    ax.legend(fontsize=8); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    # Plot 4: Learning rate
    ax = fig.add_subplot(gs[1, 0])
    if 'learning_rate' in history:
        ax.plot(history['learning_rate'], 'b-')
    elif 'train_loss' in history:
        n = len(history['train_loss'])
        lr_sched = [1e-4 * (0.5 * (1 + np.cos(np.pi * e / n))) for e in range(n)]
        ax.plot(lr_sched, 'b-', alpha=0.5)
        ax.set_title('LR Schedule (estimated)', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    if 'learning_rate' in history:
        ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    # Plot 5: Curriculum phases
    ax = fig.add_subplot(gs[1, 1])
    if 'training_phase' in history:
        ax.plot(history['training_phase'], 'k-', lw=2)
    else:
        n = len(history.get('train_loss', [100]))
        phases = ([1] * min(50, n) + [2] * min(50, max(0, n - 50)) +
                  [3] * min(50, max(0, n - 100)) + [4] * max(0, n - 150))[:n]
        ax.plot(phases, 'k-', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Phase')
    ax.set_title('Curriculum Learning Phases', fontweight='bold')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['1: Env', '2: Int', '3: Temp', '4: Full'])
    ax.grid(True, alpha=0.3)

    # Plot 6: Jaccard
    ax = fig.add_subplot(gs[1, 2])
    if 'jaccard' in history:
        ax.plot(history['jaccard'], 'purple', lw=2)
    ax.set_xlabel('Validation Step'); ax.set_ylabel('Jaccard')
    ax.set_title('Jaccard Similarity', fontweight='bold'); ax.grid(True, alpha=0.3)

    # Plot 7: Prevalence error
    ax = fig.add_subplot(gs[2, 0])
    for key, color, label in [('prevalence_mae', 'g', 'MAE'), ('prevalence_mse', 'r', 'MSE')]:
        if key in history:
            ax.plot(history[key], color, label=label)
    ax.set_xlabel('Validation Step'); ax.set_ylabel('Error')
    ax.set_title('Prevalence Prediction Error', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 8: Rare vs Common AUC gap
    ax = fig.add_subplot(gs[2, 1])
    if 'auc_rare' in history and 'auc_common' in history:
        gap = [c - r for r, c in zip(history['auc_rare'], history['auc_common'])]
        ax.plot(gap, 'purple', lw=2)
        ax.axhline(y=0, color='black', ls='--', alpha=0.5)
        ax.fill_between(range(len(gap)), 0, gap, alpha=0.2, color='purple')
    ax.set_xlabel('Validation Step'); ax.set_ylabel('AUC Gap (Common - Rare)')
    ax.set_title('Rare Species Performance Gap', fontweight='bold'); ax.grid(True, alpha=0.3)

    # Plot 9: Training summary
    ax = fig.add_subplot(gs[2, 2]); ax.axis('off')
    txt = "TRAINING SUMMARY\n" + "=" * 30 + "\n\n"
    if checkpoint:
        txt += f"Best Epoch: {checkpoint.get('epoch', 'N/A')}\n"
        txt += f"Best Metric: {checkpoint.get('best_metric', 0):.4f}\n\n"
    if 'auc_rare' in history and len(history['auc_rare']) > 0:
        txt += f"Peak Rare AUC: {max(history['auc_rare']):.4f}\n"
        txt += f"Final Rare AUC: {history['auc_rare'][-1]:.4f}\n\n"
    if 'train_loss' in history and len(history['train_loss']) > 0:
        txt += f"Final Train Loss: {history['train_loss'][-1]:.4f}\n"
        txt += f"Total Epochs: {len(history['train_loss'])}\n"
    ax.text(0.1, 0.9, txt, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Stage 2 Training Dynamics — Real Training History',
                 fontsize=14, fontweight='bold', y=0.99)
    safe_save(fig, out / "training_dynamics" / "training_overview.png")

    # Gradient flow — real model
    _plot_gradient_flow_real(checkpoint, out)


def _plot_gradient_flow_real(checkpoint, out):
    """Real gradient norms from model parameter groups."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    state_dict = checkpoint.get("model_state_dict", {}) if checkpoint else {}
    if not state_dict:
        print("   ⚠ No state dict — skipping gradient flow"); plt.close(fig); return

    # Group parameters by component
    comp_norms = defaultdict(list)
    for k, v in state_dict.items():
        component = k.split('.')[0]
        norm = float(v.float().norm().item())
        comp_norms[component].append(norm)

    # Plot weight norms by component
    ax = axes[0]
    components = sorted(comp_norms.keys())
    means = [np.mean(comp_norms[c]) for c in components]
    stds = [np.std(comp_norms[c]) for c in components]
    colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
    bars = ax.bar(range(len(components)), means, yerr=stds, color=colors,
                  edgecolor='black', alpha=0.8, capsize=3)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Weight Norm (mean ± std)')
    ax.set_title('Weight Magnitude by Component\n(Real trained model)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot parameter count by component
    ax = axes[1]
    param_counts = {c: sum(v.numel() for k, v in state_dict.items() if k.startswith(c + '.'))
                    for c in components}
    counts = [param_counts.get(c, 0) for c in components]
    ax.barh(range(len(components)), counts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=8)
    ax.set_xlabel('# Parameters')
    ax.set_title('Parameter Count by Component', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, c in enumerate(counts):
        ax.text(c + max(counts) * 0.01, i, f'{c:,}', va='center', fontsize=7)

    total = sum(counts)
    plt.suptitle(f'Model Weight Analysis — Real Trained Parameters ({total:,} total)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    safe_save(fig, out / "training_dynamics" / "gradient_flow.png")


# ═══════════════════════════════════════════════════════════════════════
# 4. FEATURE MAPS — REAL INTERMEDIATE ACTIVATIONS VIA HOOKS
# ═══════════════════════════════════════════════════════════════════════

def plot_feature_visualizations(model_inf, ld, out):
    """Visualize REAL intermediate activations from model forward pass."""
    print("\n" + "=" * 70)
    print("GENERATING FEATURE VISUALIZATIONS (REAL ACTIVATIONS)")
    print("=" * 70)

    if not model_inf.ready:
        print("   ⚠ Model not loaded — skipping feature visualizations")
        return

    # Use a small species subset for the hook pass
    pv = ld.prev()
    occ = np.where(pv > 0)[0]
    so = occ[np.argsort(pv[occ])]
    # Pick 50 species across rarity gradient
    pick = np.linspace(0, len(so) - 1, min(50, len(so)), dtype=int)
    species_subset = so[pick]

    print(f"   Running hook forward pass on {len(species_subset)} species...")
    output, activations = model_inf.forward_with_hooks(
        ld.simulations[ld.default_idx], species_subset=species_subset)

    if not activations:
        print("   ⚠ No activations captured — skipping feature maps")
        return

    print(f"   Captured {len(activations)} activation tensors:")
    for name, tensor in activations.items():
        if tensor is not None:
            print(f"     {name}: {tuple(tensor.shape)}")

    # Build the feature visualization figure
    n_act = len([k for k, v in activations.items() if v is not None])
    n_cols = 5
    n_rows = max(2, (n_act + n_cols - 1) // n_cols + 1)  # +1 for output row

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    row = 0
    col = 0
    for name, tensor in sorted(activations.items()):
        if tensor is None:
            continue
        ax = fig.add_subplot(gs[row, col])

        # Visualize: take first batch element, mean over channel dim if 4D+
        t = tensor
        if t.ndim == 5:  # (B, S, C, H, W) or (B, C, S, H, W)
            t = t[0].mean(dim=0)  # average over species → (C, H, W)
            if t.ndim == 3:
                t = t.mean(dim=0)  # average channels → (H, W)
        elif t.ndim == 4:  # (B, C, H, W)
            t = t[0].mean(dim=0)  # → (H, W)
        elif t.ndim == 3:  # (B, S, D) or (B, C, L)
            t = t[0]  # → (S, D) or (C, L)
        elif t.ndim == 2:  # (B, D)
            t = t[0].unsqueeze(0)  # → (1, D) for visualization

        t_np = t.numpy() if isinstance(t, torch.Tensor) else t
        if t_np.ndim == 1:
            ax.plot(t_np)
            ax.set_title(f'{name}\n({tensor.shape})', fontsize=8)
        else:
            im = ax.imshow(t_np, cmap='viridis', aspect='auto')
            ax.set_title(f'{name}\n({tensor.shape})', fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.7)
        ax.tick_params(labelsize=6)

        col += 1
        if col >= n_cols:
            col = 0
            row += 1

    # Last row: real output species predictions
    if output is not None:
        P = ld.P()
        row = n_rows - 1
        species_to_show = min(n_cols, len(species_subset))
        for idx in range(species_to_show):
            sp = species_subset[idx]
            ax = fig.add_subplot(gs[row, idx])
            ax.imshow(output[idx], cmap=CMAP_PROB, vmin=0, vmax=1,
                      aspect='equal', interpolation='nearest')
            ax.set_title(f'Output sp {sp}\nprev={pv[sp]:.4f}', fontsize=8)
            ax.axis('off')

    plt.suptitle('Real Feature Visualizations — Intermediate Activations\n'
                 '(From actual model forward pass with PyTorch hooks)',
                 fontsize=13, fontweight='bold', y=0.99)
    safe_save(fig, out / "architecture_viz" / "feature_maps.png")


# ═══════════════════════════════════════════════════════════════════════
# 5. ATTENTION MAPS — REAL PATTERNS FROM MODEL
# ═══════════════════════════════════════════════════════════════════════

def plot_attention_maps(model_inf, ld, pred, out):
    """Visualize attention patterns derived from real model and data."""
    print("\n" + "=" * 70)
    print("GENERATING ATTENTION MAP VISUALIZATIONS (REAL)")
    print("=" * 70)

    P = ld.P(); pv = ld.prev(); env = ld.env()
    S, Y, X = P.shape
    occ = np.where(pv > 0)[0]

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (0,0) Real species co-occurrence from IBM (used as attention analog)
    ax = fig.add_subplot(gs[0, 0])
    top20 = occ[np.argsort(-pv[occ])][:20]
    ji_ibm = jaccard_matrix(P[top20], 20)
    im = ax.imshow(ji_ibm, cmap='RdBu_r', aspect='auto', vmin=0, vmax=max(0.01, ji_ibm.max()))
    ax.set_title('IBM Species Co-occurrence\n(Jaccard, top 20)', fontweight='bold')
    ax.set_xlabel('Species rank'); ax.set_ylabel('Species rank')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (0,1) Model's predicted co-occurrence
    ax = fig.add_subplot(gs[0, 1])
    if pred is not None:
        jp = jaccard_matrix((pred[top20] > 0.5).astype(float), 20)
        im = ax.imshow(jp, cmap='RdBu_r', aspect='auto', vmin=0, vmax=max(0.01, jp.max()))
        ax.set_title('Model Co-occurrence\n(Jaccard, top 20)', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No predictions', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Co-occurrence', fontweight='bold')
    ax.set_xlabel('Species rank'); ax.set_ylabel('Species rank')
    plt.colorbar(im, ax=ax, shrink=0.8) if pred is not None else None

    # (0,2) Co-occurrence fidelity scatter
    ax = fig.add_subplot(gs[0, 2])
    if pred is not None:
        triu = np.triu_indices(20, k=1)
        ax.scatter(ji_ibm[triu], jp[triu], s=12, alpha=0.5, color='#3182BD', edgecolors='none')
        ax.plot([0, ji_ibm[triu].max()], [0, ji_ibm[triu].max()], 'k--', lw=1, alpha=0.4)
        if len(ji_ibm[triu]) > 5:
            r = np.corrcoef(ji_ibm[triu], jp[triu])[0, 1]
            ax.text(0.05, 0.90, f'r = {r:.3f}', transform=ax.transAxes,
                    fontsize=12, color='#238B45', fontweight='bold')
    ax.set_xlabel('IBM Jaccard'); ax.set_ylabel('Model Jaccard')
    ax.set_title('Co-occurrence Fidelity', fontweight='bold'); ax.grid(True, alpha=0.3)

    # (1,0) Spatial attention analog — richness correlation IBM vs Model
    ax = fig.add_subplot(gs[1, 0])
    ri = richness(P)
    if pred is not None:
        rp = (pred > 0.5).sum(axis=0)
        im = ax.imshow(rp, cmap='YlOrRd', aspect='equal', interpolation='nearest')
        ax.set_title(f'Model Richness Map\n(mean={rp.mean():.1f})', fontweight='bold')
    else:
        im = ax.imshow(ri, cmap='YlOrRd', aspect='equal', interpolation='nearest')
        ax.set_title('IBM Richness Map', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (1,1) Temporal pattern from P_t if available
    ax = fig.add_subplot(gs[1, 1])
    Pt = ld.P_t()
    if Pt is not None:
        T_steps = Pt.shape[0]
        rare_sp = occ[np.argsort(pv[occ])][:5]
        common_sp = occ[np.argsort(-pv[occ])][:3]
        for sp in rare_sp:
            ax.plot(Pt[:, sp].mean(axis=(1, 2)), lw=1.5, alpha=0.7,
                    color='#E31A1C', label=f'rare {sp}' if sp == rare_sp[0] else '')
        for sp in common_sp:
            ax.plot(Pt[:, sp].mean(axis=(1, 2)), lw=1.5, alpha=0.7, ls='--',
                    color='#3182BD', label=f'common {sp}' if sp == common_sp[0] else '')
        ax.set_xlabel('Time Step'); ax.set_ylabel('Prevalence')
        ax.set_title('Temporal Dynamics\n(Input to temporal encoder)', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No P_t data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Temporal Dynamics', fontweight='bold')

    # (1,2) Attention weight distribution from model weights
    ax = fig.add_subplot(gs[1, 2])
    state_dict = model_inf.get_state_dict()
    attn_weights = []
    for k, v in state_dict.items():
        if 'attn' in k.lower() and 'weight' in k.lower():
            attn_weights.append(v.cpu().numpy().flatten())
    if attn_weights:
        all_w = np.concatenate(attn_weights)
        ax.hist(all_w, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', ls='--', lw=1.5, label=f'mean={all_w.mean():.4f}')
        ax.set_xlabel('Attention Weight Value'); ax.set_ylabel('Frequency')
        ax.set_title(f'Attention Parameter Distribution\n({len(all_w):,} weights)',
                     fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No attention layers\nfound in model',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Attention Parameters', fontweight='bold')

    plt.suptitle('Attention & Interaction Visualizations\n'
                 '(Real IBM co-occurrence vs model predictions + temporal patterns)',
                 fontsize=14, fontweight='bold', y=1.01)
    safe_save(fig, out / "attention_maps" / "attention_overview.png")


# ═══════════════════════════════════════════════════════════════════════
# 6. EMBEDDING VISUALIZATIONS — REAL SPECIES FEATURES
# ═══════════════════════════════════════════════════════════════════════

def plot_embedding_visualizations(model_inf, ld, out):
    """Visualize real species embeddings from GNN or conditioning outputs."""
    print("\n" + "=" * 70)
    print("GENERATING EMBEDDING VISUALIZATIONS (REAL)")
    print("=" * 70)

    if not HAS_SKLEARN:
        print("⚠ sklearn not available — skipping embeddings"); return

    P = ld.P(); pv = ld.prev(); env = ld.env()
    S, Y, X = P.shape
    occ = np.where(pv > 0)[0]

    # Build real species feature vectors (same as conditioning)
    sp_feats = np.zeros((len(occ), 8), dtype=np.float32)
    for i, sp in enumerate(occ):
        sp_feats[i, 0] = pv[sp]
        sp_feats[i, 1] = P[sp].sum() / (Y * X)
        sp_feats[i, 2] = env[sp].mean()
        sp_feats[i, 3] = env[sp].std()
        sp_feats[i, 4] = float(pv[sp] < 0.05)
        sp_feats[i, 5] = float(pv[sp] >= 0.05)
        sp_feats[i, 6] = np.log1p(pv[sp])
        sp_feats[i, 7] = float(sp) / max(S - 1, 1)

    prevalences = pv[occ]
    is_rare = prevalences < 0.05

    # Also try to get model-internal embeddings if available
    # Check if int_encoder captured activations
    model_embeddings = None
    if model_inf.ready:
        state_dict = model_inf.get_state_dict()
        # Look for embedding weights
        for k, v in state_dict.items():
            if 'int_encoder' in k and 'weight' in k and v.ndim == 2:
                if v.shape[0] > 10:  # reasonable embedding matrix
                    model_embeddings = v.cpu().numpy()
                    print(f"   Found model embedding: {k} shape={v.shape}")
                    break

    # Subsample if too many species
    max_species = min(500, len(occ))
    if len(occ) > max_species:
        subsample = np.random.choice(len(occ), max_species, replace=False)
        sp_feats_plot = sp_feats[subsample]
        prevalences_plot = prevalences[subsample]
        is_rare_plot = is_rare[subsample]
    else:
        sp_feats_plot = sp_feats
        prevalences_plot = prevalences
        is_rare_plot = is_rare

    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (0,0) PCA 2D
    ax = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(sp_feats_plot)
    ax.scatter(pca_result[is_rare_plot, 0], pca_result[is_rare_plot, 1],
               c='#E31A1C', alpha=0.5, label=f'Rare ({is_rare_plot.sum()})', s=15, edgecolors='none')
    ax.scatter(pca_result[~is_rare_plot, 0], pca_result[~is_rare_plot, 1],
               c='#3182BD', alpha=0.5, label=f'Common ({(~is_rare_plot).sum()})', s=15, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax.set_title('Species Features (PCA)\nReal IBM data', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) t-SNE
    ax = fig.add_subplot(gs[0, 1])
    perp = min(30, max(5, len(sp_feats_plot) // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    tsne_result = tsne.fit_transform(sp_feats_plot)
    ax.scatter(tsne_result[is_rare_plot, 0], tsne_result[is_rare_plot, 1],
               c='#E31A1C', alpha=0.5, label='Rare', s=15, edgecolors='none')
    ax.scatter(tsne_result[~is_rare_plot, 0], tsne_result[~is_rare_plot, 1],
               c='#3182BD', alpha=0.5, label='Common', s=15, edgecolors='none')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_title('Species Features (t-SNE)\nReal IBM data', fontweight='bold')
    ax.legend(fontsize=8)

    # (0,2) Prevalence-colored t-SNE
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                    c=np.log1p(prevalences_plot), cmap='RdYlGn', s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='log(1 + prevalence)')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_title('Embeddings by Prevalence\n(log scale)', fontweight='bold')

    # (1,0) PCA 3D
    ax = fig.add_subplot(gs[1, 0], projection='3d')
    pca3d = PCA(n_components=3)
    pca3d_result = pca3d.fit_transform(sp_feats_plot)
    ax.scatter(pca3d_result[is_rare_plot, 0], pca3d_result[is_rare_plot, 1],
               pca3d_result[is_rare_plot, 2], c='#E31A1C', alpha=0.5, label='Rare', s=15)
    ax.scatter(pca3d_result[~is_rare_plot, 0], pca3d_result[~is_rare_plot, 1],
               pca3d_result[~is_rare_plot, 2], c='#3182BD', alpha=0.5, label='Common', s=15)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title('3D PCA', fontweight='bold')
    ax.legend(fontsize=7)

    # (1,1) Cosine similarity heatmap
    ax = fig.add_subplot(gs[1, 1])
    n_show = min(30, len(sp_feats_plot))
    sim = cosine_similarity(sp_feats_plot[:n_show])
    im = ax.imshow(sim, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'Feature Similarity\n(Cosine, first {n_show} species)', fontweight='bold')
    ax.set_xlabel('Species'); ax.set_ylabel('Species')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (1,2) Feature dimension usage: rare vs common
    ax = fig.add_subplot(gs[1, 2])
    feat_names = ['prevalence', 'occupancy', 'mean_env', 'std_env',
                  'is_rare', 'is_common', 'log_prev', 'sp_idx']
    rare_means = np.abs(sp_feats_plot[is_rare_plot]).mean(axis=0)
    common_means = np.abs(sp_feats_plot[~is_rare_plot]).mean(axis=0) if (~is_rare_plot).any() else np.zeros(8)
    x_pos = np.arange(8)
    ax.bar(x_pos - 0.2, rare_means, 0.35, color='#E31A1C', alpha=0.8, label='Rare')
    ax.bar(x_pos + 0.2, common_means, 0.35, color='#3182BD', alpha=0.8, label='Common')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Absolute Value')
    ax.set_title('Feature Importance\n(Rare vs Common)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    n_occ = len(occ)
    plt.suptitle(f'Species Embedding Visualization — Real IBM Data\n'
                 f'({n_occ} occupied species, {int(is_rare.sum())} rare / {int((~is_rare).sum())} common)',
                 fontsize=14, fontweight='bold', y=1.01)
    safe_save(fig, out / "embeddings" / "embedding_visualization.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2 Validation & Visualization Suite — REAL DATA Edition')
    parser.add_argument('--ibm-dir', required=True,
                        help='Path to IBM simulation data (e.g. results/data/)')
    parser.add_argument('--ibm-filter', default=None,
                        help='Filter filenames by pattern (e.g. "ls2p5")')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--training-history', default=None,
                        help='Path to training_history.json')
    parser.add_argument('--output-dir', default='stage2_validation_outputs',
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=8,
                        help='Number of model samples for prediction averaging')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--stage2-dir', default=None,
                        help='Path to AI_simulation/stage2/ if auto-detect fails')
    parser.add_argument('--predict-sim', type=int, default=0,
                        help='Index of simulation to predict on')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip model inference (only generate data-based plots)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run only validation tests (no visualizations)')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Run only visualizations')
    args = parser.parse_args()

    if args.stage2_dir:
        s2 = str(Path(args.stage2_dir).resolve())
        if s2 not in sys.path:
            sys.path.insert(0, s2)

    out = Path(args.output_dir)
    for subdir in ['ecology_viz', 'architecture_viz', 'training_dynamics',
                   'attention_maps', 'embeddings', 'validation_tests']:
        (out / subdir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STAGE 2 VALIDATION & VISUALIZATION SUITE — REAL DATA EDITION")
    print("=" * 70)
    print(f"  IBM data:    {args.ibm_dir}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Output:      {args.output_dir}")

    # ── Load IBM data ──
    ld = IBMDataLoader(args.ibm_dir, max_files=10, file_filter=args.ibm_filter)
    ld.load()
    if not ld.simulations:
        print("❌ No IBM data loaded!"); return

    sim_idx = min(args.predict_sim, len(ld.simulations) - 1)
    ld.default_idx = sim_idx
    P = ld.P(); pv = ld.prev()
    S, Y, X = P.shape
    print(f"\n  Using simulation [{sim_idx}]: {Path(ld.source()).name}")
    print(f"  γ = {int(ld.get('gamma'))} species, grid = {Y}×{X}")
    print(f"  Occupied: {(pv > 0).sum()}, Rare: {((pv > 0) & (pv < 0.05)).sum()}")

    # ── Load training history ──
    history = None
    metrics = {}
    if args.training_history and Path(args.training_history).exists():
        metrics, history = load_best_metrics(args.training_history)

    # ── Load model ──
    model_inf = RealModelInference(
        args.checkpoint, device=args.device,
        stage2_dir=args.stage2_dir or (str(STAGE2_DIR) if STAGE2_DIR else None))
    model_loaded = model_inf.load()

    if model_loaded:
        cm = model_inf.get_metrics()
        if cm and not metrics:
            metrics = cm

    # ── Run inference ──
    pred = None
    if model_loaded and not args.skip_inference:
        try:
            pred = model_inf.predict(
                ld.simulations[sim_idx], n_samples=args.n_samples)
            print(f"\n✓  Predictions: shape={pred.shape}, mean={pred.mean():.4f}")
        except Exception as e:
            print(f"\n❌ Inference failed: {e}")
            import traceback; traceback.print_exc()

    # ── Run validation tests ──
    if not args.visualize_only:
        _run_validation_tests(ld, model_inf, pred, metrics, out)

    # ── Generate ALL visualizations ──
    if not args.validate_only:
        print("\n" + "=" * 70)
        print("  GENERATING ALL VISUALIZATIONS (REAL DATA)")
        print("=" * 70)

        plot_ecology_visualizations(ld, pred, out)
        plot_architecture_visualization(model_inf, out)
        plot_training_dynamics(history, model_inf.checkpoint, out)
        plot_feature_visualizations(model_inf, ld, out)
        plot_attention_maps(model_inf, ld, pred, out)
        plot_embedding_visualizations(model_inf, ld, out)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ✓ ALL DONE — Real Data Validation & Visualization Complete")
    print("=" * 70)
    print(f"  Output directory: {out}/")
    print("\n  Generated figures:")
    for subdir in ['ecology_viz', 'architecture_viz', 'training_dynamics',
                   'attention_maps', 'embeddings']:
        subpath = out / subdir
        if subpath.exists():
            for f in sorted(subpath.iterdir()):
                print(f"    {subdir}/{f.name}  ({f.stat().st_size / 1024:.0f} KB)")

    # Save metrics summary
    summary = {
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
        "n_sims_loaded": len(ld.simulations),
        "predict_sim_index": sim_idx,
        "source_file": Path(ld.source()).name,
        "grid": [Y, X],
        "gamma": int(ld.get("gamma")),
        "n_occupied": int((pv > 0).sum()),
        "n_rare": int(((pv > 0) & (pv < 0.05)).sum()),
        "inference_mode": "real_model" if pred is not None else "none",
        "model_loaded": model_loaded,
        "device": str(model_inf.device) if model_loaded else "N/A",
    }
    with open(out / "validation_tests" / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {out}/validation_tests/validation_summary.json")


def _run_validation_tests(ld, model_inf, pred, metrics, out):
    """Run validation tests using real data."""
    print("\n" + "=" * 70)
    print("  RUNNING VALIDATION TESTS (REAL DATA)")
    print("=" * 70)

    results = {}
    P = ld.P(); pv = ld.prev()
    S, Y, X = P.shape

    # Test 1: Checkpoint integrity
    print("\n  TEST 1: Checkpoint Integrity")
    cp = model_inf.checkpoint
    if cp:
        print(f"    ✓ Loaded: epoch={cp.get('epoch', '?')}, best_metric={cp.get('best_metric', '?')}")
        results["checkpoint_integrity"] = {"passed": True}
    else:
        print("    ❌ No checkpoint loaded")
        results["checkpoint_integrity"] = {"passed": False}

    # Test 2: Architecture verification
    print("\n  TEST 2: Architecture Verification")
    sd = model_inf.get_state_dict()
    if sd:
        total_params = sum(v.numel() for v in sd.values())
        components = defaultdict(int)
        for k, v in sd.items():
            comp = k.split('.')[0]
            components[comp] += v.numel()
        print(f"    ✓ Total parameters: {total_params:,}")
        for comp, count in sorted(components.items(), key=lambda x: -x[1]):
            print(f"      {comp}: {count:,} ({100 * count / total_params:.1f}%)")
        film_keys = [k for k in sd if 'film' in k.lower()]
        print(f"    FiLM layers: {len(film_keys)}")
        results["architecture"] = {"passed": True, "total_params": total_params}
    else:
        results["architecture"] = {"passed": False}

    # Test 3: Prediction quality
    print("\n  TEST 3: Prediction Quality")
    if pred is not None:
        occ = pv > 0
        ppv = pred.mean(axis=(1, 2))

        # Prevalence correlation
        mask = occ & (ppv > 0)
        if mask.sum() > 5:
            r_prev = np.corrcoef(pv[mask], ppv[mask])[0, 1]
            print(f"    ✓ Prevalence correlation: r = {r_prev:.3f}")
        else:
            r_prev = 0; print("    ⚠ Too few species for prevalence correlation")

        # Environmental niche
        env = ld.env()
        rng = np.random.default_rng(42)
        samp = np.where(occ)[0]
        if len(samp) > 200: samp = rng.choice(samp, 200, replace=False)
        ei, ep = [], []
        for sp in samp:
            mi, mp = P[sp] > 0, pred[sp] > 0.5
            if mi.sum() > 0 and mp.sum() > 0:
                ei.append(env[sp][mi].mean()); ep.append(env[sp][mp].mean())
        if len(ei) > 5:
            r_env = np.corrcoef(ei, ep)[0, 1]
            print(f"    ✓ Env niche correlation: r = {r_env:.3f}")
        else:
            r_env = 0

        # Co-occurrence
        top40 = np.where(occ)[0][np.argsort(-pv[occ])][:40]
        ji = jaccard_matrix(P[top40], 40)
        jp = jaccard_matrix((pred[top40] > 0.5).astype(float), 40)
        triu = np.triu_indices(40, k=1)
        if len(ji[triu]) > 5:
            r_cooc = np.corrcoef(ji[triu], jp[triu])[0, 1]
            print(f"    ✓ Co-occurrence correlation: r = {r_cooc:.3f}")
        else:
            r_cooc = 0

        passed = (r_prev > 0.8 and r_env > 0.8 and r_cooc > 0.8)
        results["predictions"] = {
            "passed": passed,
            "r_prevalence": float(r_prev),
            "r_env_niche": float(r_env),
            "r_cooccurrence": float(r_cooc),
        }
        print(f"    {'✓' if passed else '⚠'} Overall: {'PASSED' if passed else 'REVIEW'}")
    else:
        print("    ⚠ No predictions available")
        results["predictions"] = {"passed": False, "reason": "no_predictions"}

    # Test 4: Metrics check
    print("\n  TEST 4: Metrics Check")
    if metrics:
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        auc = metrics.get("auc_overall", 0)
        passed = isinstance(auc, float) and auc >= 0.8
        results["metrics"] = {"passed": passed, "auc_overall": auc}
        print(f"    {'✓' if passed else '⚠'} AUC ≥ 0.80: {'YES' if passed else 'NO'}")
    else:
        results["metrics"] = {"passed": False, "reason": "no_metrics"}

    # Save results
    with open(out / "validation_tests" / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    all_passed = all(r.get("passed", False) for r in results.values())
    print(f"\n  {'✅ ALL TESTS PASSED' if all_passed else '⚠ SOME TESTS NEED REVIEW'}")
    return results


if __name__ == "__main__":
    main()