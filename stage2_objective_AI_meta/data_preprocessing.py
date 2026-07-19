"""
=============================================================================
STEP 1: DATA PREPROCESSING MODULE
=============================================================================
This module handles loading, validating, and preprocessing the IBM simulation
outputs for training the EcoDiffusion model.

KEY PROCESSING STEPS:
1.1 Load and validate all 240 worlds
1.2 Extract: IBM_B, P_t, ENV_r_field, C_topk_*, obs_mask_*
1.3 Normalize biomass: log1p transform, scale to [0,1]
1.4 Build interaction graphs from C_topk_idx, C_topk_w
1.5 Split: 200 train / 20 val / 20 test
1.6 Create data loaders with species-balanced sampling

REASONING FOR PREPROCESSING CHOICES:
- log1p transform: Biomass spans orders of magnitude; log stabilizes variance
- Percentile clipping: Removes extreme outliers that destabilize training
- Species-balanced sampling: Ensures rare species are adequately represented
- Graph construction: Explicit interaction structure for GNN encoder
=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import pickle
import gc
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationWorld:
    """Represents a single simulation world."""
    
    def __init__(self, npz_path: str):
        self.path = Path(npz_path)
        self.world_id = self.path.stem
        self._load_data()
    
    def _load_data(self):
        with np.load(self.path, allow_pickle=True) as data:
            self.available_keys = list(data.keys())
            
            def get_array(possible_keys, required=True, default=None):
                for key in possible_keys:
                    if key in data:
                        return data[key]
                if required:
                    raise ValueError(f"Missing one of {possible_keys} in {self.path}")
                return default
            
            # -----------------------------------------------------------------
            # Core biomass: DEFINES canonical grid and species dimensions
            # -----------------------------------------------------------------
            IBM_B_raw = get_array(['IBM_B', 'B', 'biomass'], required=True)
            self.IBM_B = IBM_B_raw.astype(np.float32)
            
            if self.IBM_B.ndim == 4:
                # (T, S, Y, X)
                self.n_timesteps, self.n_species, self.grid_y, self.grid_x = self.IBM_B.shape
            elif self.IBM_B.ndim == 3:
                # (S, Y, X) -> add time dimension
                self.n_species, self.grid_y, self.grid_x = self.IBM_B.shape
                self.IBM_B = self.IBM_B[np.newaxis, ...]
                self.n_timesteps = 1
            else:
                raise ValueError(f"{self.path.name}: Unexpected IBM_B shape {self.IBM_B.shape}")
            
            self.grid_shape = (self.grid_y, self.grid_x)
            
            # -----------------------------------------------------------------
            # Presence time series P_t: enforce same grid as IBM_B
            # -----------------------------------------------------------------
            P_t_raw = get_array(['P_t', 'P', 'presence'], required=False)
            if P_t_raw is not None:
                P_t = P_t_raw.astype(np.int8)
                if P_t.ndim == 3:
                    # (S, Y, X) -> add time dim
                    P_t = P_t[np.newaxis, ...]
                # >>> FIX 1: force P_t to match IBM_B grid and species dims
                if (
                    P_t.shape[-2:] != (self.grid_y, self.grid_x)
                    or P_t.shape[1] != self.n_species
                ):
                    logger.warning(
                        f"{self.path.name}: P_t shape {P_t.shape} inconsistent with "
                        f"IBM_B {self.IBM_B.shape}; recomputing P_t from IBM_B > 0"
                    )
                    P_t = (self.IBM_B > 0).astype(np.int8)
                self.P_t = P_t
            else:
                # No P_t provided: derive from biomass
                self.P_t = (self.IBM_B > 0).astype(np.int8)
            
            # -----------------------------------------------------------------
            # Environmental field: enforce full (S, Y, X) with correct grid
            # -----------------------------------------------------------------
            ENV_raw = get_array(['ENV_r_field', 'ENV_r', 'env_field'], required=False)
            if ENV_raw is not None:
                env = ENV_raw.astype(np.float32)
                if env.ndim == 4:
                    # Often stored as (T, S, Y, X) or (1, S, Y, X) – take first slice
                    env = env[0]
                elif env.ndim == 3:
                    # (S, Y, X) – OK
                    pass
                elif env.ndim == 2:
                    # (Y, X) – broadcast across species
                    env = np.tile(env[np.newaxis, ...], (self.n_species, 1, 1))
                else:
                    logger.warning(
                        f"{self.path.name}: ENV_r_field shape {env.shape} unexpected; "
                        "using ones field"
                    )
                    env = np.ones((self.n_species, self.grid_y, self.grid_x), dtype=np.float32)
                
                # >>> FIX 2: enforce exact (S, Y, X) shape
                if env.shape != (self.n_species, self.grid_y, self.grid_x):
                    logger.warning(
                        f"{self.path.name}: ENV_r_field shape {env.shape} inconsistent with "
                        f"IBM_B {self.IBM_B.shape}; using ones field"
                    )
                    env = np.ones((self.n_species, self.grid_y, self.grid_x), dtype=np.float32)
                
                self.ENV_r_field = env
            else:
                self.ENV_r_field = np.ones(
                    (self.n_species, self.grid_y, self.grid_x), dtype=np.float32
                )
            
            # -----------------------------------------------------------------
            # Interaction network (unchanged)
            # -----------------------------------------------------------------
            C_idx = get_array(['C_topk_idx', 'C_idx'], required=False)
            C_w = get_array(['C_topk_w', 'C_w'], required=False)
            
            if C_idx is not None:
                self.C_topk_idx = C_idx.astype(np.int64)
                self.C_topk_w = C_w.astype(np.float32) if C_w is not None else np.ones_like(
                    self.C_topk_idx, dtype=np.float32
                )
                self.n_interactions = (
                    self.C_topk_idx.shape[1] if self.C_topk_idx.ndim > 1 else 1
                )
            else:
                self.C_topk_idx = np.zeros((self.n_species, 1), dtype=np.int64)
                self.C_topk_w = np.zeros((self.n_species, 1), dtype=np.float32)
                self.n_interactions = 0
            
            # -----------------------------------------------------------------
            # Final biomass B_last: force same grid & species as IBM_B
            # -----------------------------------------------------------------
            B_last_raw = get_array(['B_last', 'B_final'], required=False)
            if B_last_raw is not None:
                B_last = B_last_raw.astype(np.float32)
                if B_last.ndim == 4:
                    B_last = B_last[-1]  # last time slice
                # >>> FIX 3: enforce shape consistency for B_last
                if (
                    B_last.shape[-2:] != (self.grid_y, self.grid_x)
                    or B_last.shape[0] != self.n_species
                ):
                    logger.warning(
                        f"{self.path.name}: B_last shape {B_last.shape} inconsistent with "
                        f"IBM_B {self.IBM_B.shape}; using IBM_B[-1] instead"
                    )
                    B_last = self.IBM_B[-1]
                self.B_last = B_last
            else:
                self.B_last = self.IBM_B[-1]
            
            # -----------------------------------------------------------------
            # Final presence P_last_final: force same grid & species as IBM_B
            # -----------------------------------------------------------------
            P_last_raw = get_array(['P_last_final', 'P_last'], required=False)
            if P_last_raw is not None:
                P_last = P_last_raw.astype(np.int8)
                if P_last.ndim == 4:
                    P_last = P_last[-1]  # last time slice
                # >>> FIX 4: enforce shape consistency for P_last_final
                if (
                    P_last.shape[-2:] != (self.grid_y, self.grid_x)
                    or P_last.shape[0] != self.n_species
                ):
                    logger.warning(
                        f"{self.path.name}: P_last shape {P_last.shape} inconsistent with "
                        f"IBM_B {self.IBM_B.shape}; using P_t[-1] instead"
                    )
                    P_last = self.P_t[-1]
                self.P_last_final = P_last
            else:
                self.P_last_final = self.P_t[-1]
            
            # -----------------------------------------------------------------
            # Observation masks (same; masks may have weird shapes, but we
            # re-pad later in EcoDataset with pad_to_max_species)
            # -----------------------------------------------------------------
            self.obs_masks = {}
            for key in data.keys():
                if 'obs_mask' in key.lower():
                    try:
                        budget = int(''.join(filter(str.isdigit, key.split('_')[-1])))
                        self.obs_masks[budget] = data[key].astype(np.int8)
                    except Exception:
                        pass
            
            # -----------------------------------------------------------------
            # Species-level statistics (use consistent P_last_final & grid)
            # -----------------------------------------------------------------
            prev_final = get_array(['prevalence_final', 'prevalence'], required=False)
            if prev_final is not None:
                self.prevalence_final = prev_final.astype(np.float32)
                # >>> FIX 5: ensure length matches n_species
                if self.prevalence_final.shape[0] != self.n_species:
                    logger.warning(
                        f"{self.path.name}: prevalence_final length {self.prevalence_final.shape[0]} "
                        f"!= n_species {self.n_species}; recomputing from P_last_final"
                    )
                    self.prevalence_final = (
                        self.P_last_final.sum(axis=(1, 2)) / (self.grid_y * self.grid_x)
                    ).astype(np.float32)
            else:
                self.prevalence_final = (
                    self.P_last_final.sum(axis=(1, 2)) / (self.grid_y * self.grid_x)
                ).astype(np.float32)
            
            prev_any = get_array(['prevalence_any'], required=False)
            if prev_any is not None:
                self.prevalence_any = prev_any.astype(np.float32)
                if self.prevalence_any.shape[0] != self.n_species:
                    logger.warning(
                        f"{self.path.name}: prevalence_any length {self.prevalence_any.shape[0]} "
                        f"!= n_species {self.n_species}; recomputing from P_t"
                    )
                    self.prevalence_any = (
                        (self.P_t.any(axis=0)).sum(axis=(1, 2)) / (self.grid_y * self.grid_x)
                    ).astype(np.float32)
            else:
                self.prevalence_any = (
                    (self.P_t.any(axis=0)).sum(axis=(1, 2)) / (self.grid_y * self.grid_x)
                ).astype(np.float32)
            
            # Remaining metadata (unchanged)
            deg_in = get_array(['deg_in'], required=False)
            self.deg_in = deg_in.astype(np.float32) if deg_in is not None else np.zeros(
                self.n_species, dtype=np.float32
            )
            
            deg_out = get_array(['deg_out'], required=False)
            self.deg_out = deg_out.astype(np.float32) if deg_out is not None else np.zeros(
                self.n_species, dtype=np.float32
            )
            
            r_base = get_array(['r_base', 'r'], required=False)
            self.r_base = r_base.astype(np.float32) if r_base is not None else np.ones(
                self.n_species, dtype=np.float32
            )
            
            alpha = get_array(['alpha_abs_mean'], required=False)
            self.alpha_abs_mean = alpha.astype(np.float32) if alpha is not None else np.zeros(
                self.n_species, dtype=np.float32
            )
            
            def get_scalar(possible_keys, default):
                for key in possible_keys:
                    if key in data:
                        val = data[key]
                        return float(val) if np.isscalar(val) or val.ndim == 0 else float(val.item())
                return default
            
            self.gamma = int(get_scalar(['gamma'], self.n_species))
            self.body_mass = get_scalar(['BODY_MASS'], 1e-4)
            self.dispersal_rate = get_scalar(['DISPERSAL_RATE'], 5e-8)
            self.ldd_prob = get_scalar(['LONG_DISTANCE_PROB'], 0.0)
            self.env_length_scale = get_scalar(['ENV_length_scale'], 2.5)
            self.env_var_r = get_scalar(['ENV_var_r'], 0.001)

    
    def get_species_features(self) -> np.ndarray:
        features = np.zeros((self.n_species, 8), dtype=np.float32)
        features[:, 0] = np.log1p(self.body_mass)
        features[:, 1] = self.r_base
        features[:, 2] = np.log1p(self.deg_in)
        features[:, 3] = np.log1p(self.deg_out)
        features[:, 4] = self.prevalence_final
        features[:, 5] = self.prevalence_any
        features[:, 6] = self.alpha_abs_mean
        features[:, 7] = np.log1p(self.ENV_r_field.var(axis=(1, 2)) * 1000)
        return features
    
    def get_interaction_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        source = np.repeat(np.arange(self.n_species), self.n_interactions)
        target = self.C_topk_idx.flatten()
        edge_index = np.stack([source, target], axis=0)
        edge_weight = self.C_topk_w.flatten()
        valid_mask = (source != target) & (target >= 0) & (target < self.n_species)
        return edge_index[:, valid_mask].astype(np.int64), edge_weight[valid_mask].astype(np.float32)


def get_world_metadata_fast(npz_path: str) -> Dict:
    """Extract lightweight metadata without loading full arrays.

    >>> FIX 6: always derive n_species & grid_shape from full-resolution arrays
    to avoid mixing 3x3 and 20x20 worlds.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        # 1) Determine base array for shape (prefer IBM_B, then P_t)
        base_arr = None
        if 'IBM_B' in data:
            base_arr = data['IBM_B']
        elif 'P_t' in data:
            base_arr = data['P_t']
        elif 'P_last_final' in data:
            base_arr = data['P_last_final']
        elif 'B_last' in data:
            base_arr = data['B_last']
        else:
            raise ValueError(
                f"Could not find IBM_B / P_t / P_last_final / B_last in {npz_path}"
            )
        
        # Normalize base_arr shape and read n_species, grid_shape
        if base_arr.ndim == 4:
            # (T, S, Y, X)
            n_species = base_arr.shape[1]
            grid_shape = base_arr.shape[-2:]
        elif base_arr.ndim == 3:
            # (S, Y, X)
            n_species = base_arr.shape[0]
            grid_shape = base_arr.shape[-2:]
        else:
            raise ValueError(
                f"{npz_path}: unexpected base array shape {base_arr.shape}"
            )
        
        # 2) Determine prevalence (length must match n_species)
        if 'prevalence_final' in data:
            prevalence = np.asarray(data['prevalence_final'], dtype=np.float32)
        elif 'P_last_final' in data and data['P_last_final'].ndim >= 3:
            P_last = data['P_last_final']
            if P_last.ndim == 4:
                P_last = P_last[-1]
            prevalence = (
                P_last.sum(axis=(-2, -1)) / (P_last.shape[-2] * P_last.shape[-1])
            ).astype(np.float32)
        elif 'P_t' in data and data['P_t'].ndim == 4:
            P = data['P_t']
            P_last = P[-1]
            prevalence = (
                P_last.sum(axis=(1, 2)) / (P_last.shape[1] * P_last.shape[2])
            ).astype(np.float32)
        else:
            prevalence = np.full(n_species, 0.5, dtype=np.float32)
        
        # >>> FIX 7: align prevalence length with n_species
        if prevalence.shape[0] != n_species:
            if prevalence.shape[0] > n_species:
                prevalence = prevalence[:n_species]
            else:
                tmp = np.zeros(n_species, dtype=np.float32)
                tmp[:prevalence.shape[0]] = prevalence
                prevalence = tmp
        
        return {
            'prevalence': prevalence,
            'n_species': n_species,
            'grid_shape': tuple(grid_shape),
        }



class DataPreprocessor:
    """Preprocesses simulation data for model training."""
    
    def __init__(self, config):
        self.config = config
        self.stats = {}
        self.max_species = 0
    
    def compute_global_stats(self, world_paths: List[str], max_worlds: int = 30) -> Dict[str, Any]:
        logger.info(f"Computing normalization statistics from {min(len(world_paths), max_worlds)} worlds...")
        
        all_biomass = []
        all_env = []
        all_prevalences = []
        
        sample_paths = world_paths[:max_worlds]
        
        for path in tqdm(sample_paths, desc="Sampling statistics"):
            try:
                world = SimulationWorld(path)
                
                nonzero = world.IBM_B[world.IBM_B > 0]
                if len(nonzero) > 0:
                    sample_size = min(5000, len(nonzero))
                    all_biomass.extend(np.random.choice(nonzero, sample_size, replace=False))
                
                all_env.extend(world.ENV_r_field.flatten()[:5000])
                all_prevalences.extend(world.prevalence_final)
                
                del world
                gc.collect()
            except Exception as e:
                logger.warning(f"Failed to sample stats from {path}: {e}")
        
        all_biomass = np.array(all_biomass)
        all_env = np.array(all_env)
        all_prevalences = np.array(all_prevalences)
        
        self.stats = {
            'biomass_log_mean': np.mean(np.log1p(all_biomass)),
            'biomass_log_std': np.std(np.log1p(all_biomass)),
            'biomass_log_max': np.percentile(np.log1p(all_biomass), self.config.data.biomass_clip_percentile),
            'biomass_max': np.percentile(all_biomass, self.config.data.biomass_clip_percentile),
            'env_mean': np.mean(all_env),
            'env_std': np.std(all_env),
            'env_min': np.min(all_env),
            'env_max': np.max(all_env),
            'prevalence_mean': np.mean(all_prevalences),
            'prevalence_std': np.std(all_prevalences),
            'rare_species_count': np.sum(all_prevalences < self.config.data.rare_species_threshold),
            'common_species_count': np.sum(all_prevalences >= self.config.data.rare_species_threshold),
        }
        
        logger.info(f"Biomass log max: {self.stats['biomass_log_max']:.4f}")
        logger.info(f"Rare species: {self.stats['rare_species_count']}, Common: {self.stats['common_species_count']}")
        
        return self.stats
    
    def normalize_biomass(self, biomass: np.ndarray) -> np.ndarray:
        log_biomass = np.log1p(biomass)
        log_biomass = np.clip(log_biomass, 0, self.stats['biomass_log_max'])
        return (log_biomass / self.stats['biomass_log_max']).astype(np.float32)
    
    def normalize_environment(self, env: np.ndarray) -> np.ndarray:
        normalized = (env - self.stats['env_min']) / (self.stats['env_max'] - self.stats['env_min'] + 1e-8)
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    def add_spatial_coordinates(self, Y: int, X: int) -> Tuple[np.ndarray, np.ndarray]:
        y_coords = np.linspace(0, 1, Y).reshape(Y, 1).repeat(X, axis=1)
        x_coords = np.linspace(0, 1, X).reshape(1, X).repeat(Y, axis=0)
        return y_coords.astype(np.float32), x_coords.astype(np.float32)
    
    def preprocess_world(self, world: SimulationWorld) -> Dict[str, torch.Tensor]:
        processed = {}
        
        processed['IBM_B'] = torch.from_numpy(self.normalize_biomass(world.IBM_B))
        processed['P_t'] = torch.from_numpy(world.P_t.astype(np.float32))
        processed['B_last'] = torch.from_numpy(self.normalize_biomass(world.B_last))
        processed['P_last'] = torch.from_numpy(world.P_last_final.astype(np.float32))
        processed['ENV_r_field'] = torch.from_numpy(self.normalize_environment(world.ENV_r_field))
        
        y_coords, x_coords = self.add_spatial_coordinates(world.grid_y, world.grid_x)
        processed['y_coords'] = torch.from_numpy(y_coords)
        processed['x_coords'] = torch.from_numpy(x_coords)
        
        edge_index, edge_weight = world.get_interaction_graph()
        processed['edge_index'] = torch.from_numpy(edge_index)
        processed['edge_weight'] = torch.from_numpy(edge_weight)
        processed['species_features'] = torch.from_numpy(world.get_species_features())
        
        for budget, mask in world.obs_masks.items():
            processed[f'obs_mask_{budget}'] = torch.from_numpy(mask.astype(np.float32))
        
        processed['n_species'] = world.n_species
        processed['prevalence'] = torch.from_numpy(world.prevalence_final)
        processed['world_id'] = world.world_id
        
        return processed


def pad_to_max_species(tensor: torch.Tensor, max_species: int, species_dim: int = 0) -> torch.Tensor:
    """Pad tensor along species dimension to max_species."""
    current_size = tensor.shape[species_dim]
    if current_size >= max_species:
        return tensor
    
    pad_size = max_species - current_size
    
    # Build padding tuple (PyTorch pads from last dim backwards)
    # For dim=0 on (S, Y, X): need (0, 0, 0, 0, 0, pad_size)
    # For dim=1 on (T, S, Y, X): need (0, 0, 0, 0, 0, pad_size, 0, 0)
    ndim = tensor.dim()
    padding = [0] * (2 * ndim)
    # species_dim from the end: ndim - 1 - species_dim
    pad_idx = 2 * (ndim - 1 - species_dim) + 1
    padding[pad_idx] = pad_size
    
    return F.pad(tensor, padding)


class EcoDataset(Dataset):
    """PyTorch Dataset with LAZY LOADING and FIXED MAX SPECIES."""
    
    def __init__(
        self,
        world_paths: List[str],
        preprocessor: DataPreprocessor,
        max_species: int,
        mode: str = 'equilibrium',
        target_timestep: int = -1,
        obs_budget: int = 100,
        return_all_timesteps: bool = False,
        cache_size: int = 4,
    ):
        self.world_paths = world_paths
        self.preprocessor = preprocessor
        self.max_species = max_species
        self.mode = mode
        self.target_timestep = target_timestep
        self.obs_budget = obs_budget
        self.return_all_timesteps = return_all_timesteps
        
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []
        
        logger.info(f"Initializing dataset with {len(world_paths)} worlds (max_species={max_species})")
        self._compute_sampling_weights()
    
    def _compute_sampling_weights(self):
        world_prevalences = []
        for path in tqdm(self.world_paths, desc="Extracting metadata"):
            try:
                meta = get_world_metadata_fast(path)
                world_prevalences.append(np.mean(meta['prevalence']))
            except:
                world_prevalences.append(0.5)
        
        world_prevalences = np.array(world_prevalences)
        weights = 1.0 / (world_prevalences + 0.001)
        weights = np.clip(weights, 1.0, 100.0)
        self.world_weights = weights / weights.sum()
    
    def _load_and_preprocess(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache:
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return self._cache[idx]
        
        world = SimulationWorld(self.world_paths[idx])
        processed = self.preprocessor.preprocess_world(world)
        del world
        gc.collect()
        
        if len(self._cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
            gc.collect()
        
        self._cache[idx] = processed
        self._cache_order.append(idx)
        return processed
    
    def __len__(self) -> int:
        return len(self.world_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        proc = self._load_and_preprocess(idx)
        sample = {}
        n_sp = proc['n_species']
        
        # Target - pad to max_species
        if self.target_timestep == -1:
            target = proc['P_last']
            target_biomass = proc['B_last']
        else:
            target = proc['P_t'][self.target_timestep]
            target_biomass = proc['IBM_B'][self.target_timestep]
        
        sample['target'] = pad_to_max_species(target, self.max_species, species_dim=0)
        sample['target_biomass'] = pad_to_max_species(target_biomass, self.max_species, species_dim=0)
        
        # Conditioning
        if self.mode == 'equilibrium':
            sample['condition'] = {
                'env': pad_to_max_species(proc['ENV_r_field'], self.max_species, species_dim=0),
                'y_coords': proc['y_coords'],
                'x_coords': proc['x_coords'],
            }
        elif self.mode == 'interaction':
            sample['condition'] = {
                'env': pad_to_max_species(proc['ENV_r_field'], self.max_species, species_dim=0),
                'y_coords': proc['y_coords'],
                'x_coords': proc['x_coords'],
                'edge_index': proc['edge_index'],
                'edge_weight': proc['edge_weight'],
                'species_features': pad_to_max_species(proc['species_features'], self.max_species, species_dim=0),
            }
        elif self.mode == 'temporal':
            n_history = proc['P_t'].shape[0] // 2
            sample['condition'] = {
                'env': pad_to_max_species(proc['ENV_r_field'], self.max_species, species_dim=0),
                'y_coords': proc['y_coords'],
                'x_coords': proc['x_coords'],
                'edge_index': proc['edge_index'],
                'edge_weight': proc['edge_weight'],
                'species_features': pad_to_max_species(proc['species_features'], self.max_species, species_dim=0),
                'history_P': pad_to_max_species(proc['P_t'][:n_history], self.max_species, species_dim=1),
                'history_B': pad_to_max_species(proc['IBM_B'][:n_history], self.max_species, species_dim=1),
            }
        elif self.mode == 'infill':
            obs_mask_key = f'obs_mask_{self.obs_budget}'
            if obs_mask_key in proc:
                obs_mask = proc[obs_mask_key]
            else:
                available = [int(k.split('_')[-1]) for k in proc.keys() if k.startswith('obs_mask_')]
                obs_mask = proc[f'obs_mask_{max(available)}'] if available else torch.ones_like(proc['P_last'])
            
            sample['condition'] = {
                'env': pad_to_max_species(proc['ENV_r_field'], self.max_species, species_dim=0),
                'y_coords': proc['y_coords'],
                'x_coords': proc['x_coords'],
                'edge_index': proc['edge_index'],
                'edge_weight': proc['edge_weight'],
                'species_features': pad_to_max_species(proc['species_features'], self.max_species, species_dim=0),
                'obs_mask': pad_to_max_species(obs_mask, self.max_species, species_dim=0),
                'observed': pad_to_max_species(proc['P_last'] * obs_mask, self.max_species, species_dim=0),
            }
        elif self.mode == 'full':
            sample['condition'] = {
                'env': pad_to_max_species(proc['ENV_r_field'], self.max_species, species_dim=0),
                'y_coords': proc['y_coords'],
                'x_coords': proc['x_coords'],
                'edge_index': proc['edge_index'],
                'edge_weight': proc['edge_weight'],
                'species_features': pad_to_max_species(proc['species_features'], self.max_species, species_dim=0),
                'history_P': pad_to_max_species(proc['P_t'], self.max_species, species_dim=1),
                'history_B': pad_to_max_species(proc['IBM_B'], self.max_species, species_dim=1),
            }
        
        # Metadata
        species_mask = torch.zeros(self.max_species, dtype=torch.bool)
        species_mask[:n_sp] = True
        
        padded_prevalence = torch.zeros(self.max_species)
        padded_prevalence[:n_sp] = proc['prevalence']
        
        sample['metadata'] = {
            'world_id': proc['world_id'],
            'n_species': n_sp,
            'species_mask': species_mask,
            'prevalence': padded_prevalence,
        }
        
        return sample


def eco_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function - samples are already padded to same size."""
    collated = {
        'target': [],
        'target_biomass': [],
        'condition': defaultdict(list),
        'metadata': {
            'world_ids': [],
            'n_species': [],
            'species_mask': [],
            'prevalence': [],
        }
    }
    
    for sample in batch:
        collated['target'].append(sample['target'])
        collated['target_biomass'].append(sample['target_biomass'])
        collated['metadata']['species_mask'].append(sample['metadata']['species_mask'])
        collated['metadata']['prevalence'].append(sample['metadata']['prevalence'])
        collated['metadata']['world_ids'].append(sample['metadata']['world_id'])
        collated['metadata']['n_species'].append(sample['metadata']['n_species'])
        
        for key, val in sample['condition'].items():
            collated['condition'][key].append(val)
    
    collated['target'] = torch.stack(collated['target'])
    collated['target_biomass'] = torch.stack(collated['target_biomass'])
    collated['metadata']['species_mask'] = torch.stack(collated['metadata']['species_mask'])
    collated['metadata']['prevalence'] = torch.stack(collated['metadata']['prevalence'])
    
    for key in list(collated['condition'].keys()):
        vals = collated['condition'][key]
        if key not in ['edge_index', 'edge_weight'] and isinstance(vals[0], torch.Tensor):
            try:
                collated['condition'][key] = torch.stack(vals)
            except:
                pass
    
    return collated


def create_dataloaders(
    simulation_dir: str,
    config,
    mode: str = 'equilibrium',
) -> Tuple[DataLoader, DataLoader, DataLoader, DataPreprocessor]:
    """
    Create dataloaders. Returns max_species via preprocessor.max_species.
    """
    logger.info(f"Loading simulation worlds from: {simulation_dir}")
    
    sim_path = Path(simulation_dir)
    npz_files = sorted(sim_path.glob(config.paths.npz_pattern))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found matching {config.paths.npz_pattern}")
    
    logger.info(f"Found {len(npz_files)} simulation files")
    
    # Validate files AND find max species
    valid_paths = []
    max_species = 0
    
    for npz_file in tqdm(npz_files, desc="Validating files"):
        try:
            meta = get_world_metadata_fast(str(npz_file))
            if meta['grid_shape'] == (20, 20):
                valid_paths.append(str(npz_file))
                max_species = max(max_species, meta['n_species'])
        except Exception as e:
            logger.warning(f"Skipping {npz_file.name}: {e}")
    
    logger.info(f"Found {len(valid_paths)} valid 20x20 worlds")
    logger.info(f"Maximum species count: {max_species}")
    
    if len(valid_paths) == 0:
        raise ValueError("No valid worlds found!")
    
    # Split
    n_total = len(valid_paths)
    n_train = int(n_total * config.data.train_ratio)
    n_val = int(n_total * config.data.val_ratio)
    
    if n_total >= 3 and n_val == 0:
        n_val = 1
        n_train = n_train - 1 if n_train > 1 else n_train
    
    np.random.seed(config.training.seed)
    indices = np.random.permutation(n_total)
    
    train_paths = [valid_paths[i] for i in indices[:n_train]]
    val_paths = [valid_paths[i] for i in indices[n_train:n_train + n_val]]
    test_paths = [valid_paths[i] for i in indices[n_train + n_val:]]
    
    logger.info(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    # Preprocessor
    preprocessor = DataPreprocessor(config)
    preprocessor.max_species = max_species
    preprocessor.compute_global_stats(train_paths, max_worlds=30)
    
    # Datasets with fixed max_species
    train_dataset = EcoDataset(train_paths, preprocessor, max_species, mode=mode, cache_size=4)
    val_dataset = EcoDataset(val_paths, preprocessor, max_species, mode=mode, cache_size=2)
    test_dataset = EcoDataset(test_paths, preprocessor, max_species, mode=mode, cache_size=2)

        # >>> SMOKE CHECK 1: dataset-level target shape consistency
    def _check_dataset_shapes(dataset: EcoDataset, name: str, n_samples: int = 5):
        base_shape = None
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            shape = sample['target'].shape
            if base_shape is None:
                base_shape = shape
            elif shape != base_shape:
                raise ValueError(
                    f"{name} dataset has inconsistent 'target' shapes: "
                    f"{base_shape} vs {shape} (index {i}). "
                    "Check grid sizes in your .npz files."
                )
    
    _check_dataset_shapes(train_dataset, "train")
    _check_dataset_shapes(val_dataset, "val")
    _check_dataset_shapes(test_dataset, "test")

    
    train_sampler = WeightedRandomSampler(
        weights=train_dataset.world_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    loader_kwargs = {
        'batch_size': config.training.batch_size,
        'num_workers': 0,
        'pin_memory': False,
        'collate_fn': eco_collate_fn,
    }
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader, preprocessor


def save_preprocessor(preprocessor: DataPreprocessor, path: str):
    state = {'stats': preprocessor.stats, 'max_species': preprocessor.max_species}
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    logger.info(f"Saved preprocessor to {path}")


def load_preprocessor(path: str, config) -> DataPreprocessor:
    with open(path, 'rb') as f:
        state = pickle.load(f)
    preprocessor = DataPreprocessor(config)
    preprocessor.stats = state['stats']
    preprocessor.max_species = state.get('max_species', 4000)
    return preprocessor
