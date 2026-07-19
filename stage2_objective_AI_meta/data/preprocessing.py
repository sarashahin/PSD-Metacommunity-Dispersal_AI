"""
Data Preprocessing Module
=========================

Handles loading, preprocessing, and batching of IBM simulation data.

Key responsibilities:
1. Load 240 world .npz files from parameter sweep
2. Extract and validate: IBM_B, P_t, ENV_r_field, C_topk_*, obs_mask_*
3. Normalize data appropriately for neural network training
4. Build species interaction graphs for GraphSAGE
5. Create data loaders with species-balanced sampling

Design Decisions:
- Lazy loading: Don't load all 240 worlds into memory at once
- Memory mapping: Use numpy memmap for large arrays when possible
- Chunked processing: Handle 3614 species in manageable chunks
- Balanced sampling: Oversample rare species for better representation
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data as GraphData
import warnings


@dataclass
class WorldData:
    """
    Container for a single world's data.
    
    Attributes match IBM output structure:
    - biomass: (T, S, H, W) - biomass time series
    - presence: (T, S, H, W) - presence/absence time series
    - env_field: (S, H, W) - environmental growth rate field
    - interaction_idx: (S, K) - top-K competitor indices
    - interaction_weights: (S, K) - interaction strengths
    - obs_masks: Dict[int, (S, H, W)] - observation masks at different budgets
    - metadata: Dict - world parameters and statistics
    """
    world_id: str
    biomass: np.ndarray  # (T, S, H, W)
    presence: np.ndarray  # (T, S, H, W)
    env_field: np.ndarray  # (S, H, W)
    interaction_idx: np.ndarray  # (S, K)
    interaction_weights: np.ndarray  # (S, K)
    obs_masks: Dict[int, np.ndarray]  # budget -> (S, H, W)
    metadata: Dict


class IBMDataLoader:
    """
    Loader for IBM simulation output files.
    
    Handles the parameter sweep output structure:
    - 240 worlds from 3×3×3×2×3×4 parameter combinations
    - Each world stored as .npz file
    
    Key arrays extracted:
    - IBM_B: Biomass (float16 for memory efficiency)
    - P_t: Presence/absence (bool)
    - ENV_r_field: Environmental field (float32)
    - C_topk_idx: Interaction indices (int32)
    - C_topk_w: Interaction weights (float32)
    - obs_mask_*: Observation masks (bool)
    """
    
    def __init__(self, data_root: str, verbose: bool = True):
        """
        Initialize the data loader.
        
        Args:
            data_root: Path to PSD_Dispersal_pool directory
            verbose: Print loading progress
        """
        self.data_root = Path(os.path.expanduser(data_root))
        self.verbose = verbose
        
        # Discover all world files
        self.world_files = self._discover_worlds()
        self.n_worlds = len(self.world_files)
        
        if verbose:
            print(f"[IBMDataLoader] Found {self.n_worlds} world files in {self.data_root}")
    
    def _discover_worlds(self) -> List[Path]:
        """Find all .npz world files in the data directory."""
        # Look for files matching the naming pattern
        patterns = [
            "world_*.npz",
            "sim_*.npz",
            "*.npz"
        ]
        
        world_files = []
        for pattern in patterns:
            found = list(self.data_root.glob(f"**/{pattern}"))
            if found:
                world_files = found
                break
        
        # Sort for reproducibility
        world_files = sorted(world_files, key=lambda x: x.stem)
        
        return world_files
    
    def load_world(self, world_path: Union[str, Path]) -> WorldData:
        """
        Load a single world's data from .npz file.
        
        Args:
            world_path: Path to the .npz file
            
        Returns:
            WorldData object containing all arrays
        """
        world_path = Path(world_path)
        
        if self.verbose:
            print(f"[IBMDataLoader] Loading {world_path.name}...")
        
        # Load with allow_pickle for metadata dict
        data = np.load(world_path, allow_pickle=True)
        
        # Extract required arrays with validation
        biomass = self._extract_array(data, ['IBM_B', 'biomass', 'B'])
        presence = self._extract_array(data, ['P_t', 'presence', 'P'])
        env_field = self._extract_array(data, ['ENV_r_field', 'env_field', 'r_field'])
        
        # Interaction network
        interaction_idx = self._extract_array(data, ['C_topk_idx', 'interaction_idx', 'comp_idx'])
        interaction_weights = self._extract_array(data, ['C_topk_w', 'interaction_weights', 'comp_w'])
        
        # Observation masks at different budgets
        obs_masks = {}
        for budget in [1, 5, 10, 20, 50, 100]:
            key_variants = [f'obs_mask_{budget}', f'obs_{budget}', f'mask_{budget}']
            try:
                mask = self._extract_array(data, key_variants)
                obs_masks[budget] = mask.astype(bool)
            except KeyError:
                pass  # Not all budgets may be present
        
        # Extract metadata if available
        metadata = {}
        for meta_key in ['params', 'metadata', 'config']:
            if meta_key in data:
                metadata = data[meta_key].item() if data[meta_key].ndim == 0 else dict(data[meta_key])
                break
        
        # Add derived statistics
        metadata['world_id'] = world_path.stem
        metadata['n_species'] = presence.shape[1]
        metadata['grid_size'] = presence.shape[2:4]
        metadata['n_timesteps'] = presence.shape[0]
        metadata['final_prevalence'] = presence[-1].mean(axis=(1, 2))  # Per-species prevalence
        
        return WorldData(
            world_id=world_path.stem,
            biomass=biomass,
            presence=presence,
            env_field=env_field,
            interaction_idx=interaction_idx,
            interaction_weights=interaction_weights,
            obs_masks=obs_masks,
            metadata=metadata
        )
    
    def _extract_array(self, data: np.lib.npyio.NpzFile, 
                       key_variants: List[str]) -> np.ndarray:
        """Extract array trying multiple possible key names."""
        for key in key_variants:
            if key in data:
                return data[key]
        raise KeyError(f"None of {key_variants} found in .npz file. Available: {list(data.keys())}")
    
    def load_all_worlds(self, max_worlds: Optional[int] = None) -> List[WorldData]:
        """
        Load all (or first N) worlds.
        
        Warning: This loads all data into memory. Use with caution.
        """
        n_to_load = min(max_worlds or self.n_worlds, self.n_worlds)
        worlds = []
        
        for i, world_file in enumerate(self.world_files[:n_to_load]):
            if self.verbose and i % 10 == 0:
                print(f"[IBMDataLoader] Loading world {i+1}/{n_to_load}")
            worlds.append(self.load_world(world_file))
        
        return worlds


class DataPreprocessor:
    """
    Preprocesses IBM data for neural network training.
    
    Key preprocessing steps:
    1. Biomass normalization: log1p transform + scale to [0,1]
    2. Spatial coordinate encoding: Add normalized x,y coordinates
    3. Interaction graph construction: Build PyG Data object
    4. Species trait computation: body_mass, r_base, connectivity
    
    Design rationale:
    - log1p for biomass: Handles extreme values, preserves zeros
    - Coordinate encoding: Helps CNN learn spatial patterns
    - Graph construction: Enables GraphSAGE message passing
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: EcoConfig object with data and model settings
        """
        self.config = config
        self.data_config = config.data
        self.model_config = config.model
        
        # Statistics computed during fit
        self.biomass_max = None
        self.env_stats = None
        self._fitted = False
    
    def fit(self, worlds: List[WorldData]):
        """
        Compute normalization statistics from training worlds.
        
        Args:
            worlds: List of training WorldData objects
        """
        print("[Preprocessor] Computing normalization statistics...")
        
        # Compute global biomass maximum for scaling
        all_biomass_max = []
        all_env_means = []
        all_env_stds = []
        
        for world in worlds:
            # log1p transform before finding max
            log_biomass = np.log1p(world.biomass)
            all_biomass_max.append(log_biomass.max())
            
            # Environment field statistics
            all_env_means.append(world.env_field.mean())
            all_env_stds.append(world.env_field.std())
        
        self.biomass_max = max(all_biomass_max)
        self.env_stats = {
            'mean': np.mean(all_env_means),
            'std': np.mean(all_env_stds)
        }
        
        self._fitted = True
        print(f"[Preprocessor] Fitted: biomass_max={self.biomass_max:.3f}, "
              f"env_mean={self.env_stats['mean']:.6f}")
    
    def transform_biomass(self, biomass: np.ndarray) -> np.ndarray:
        """
        Transform biomass to [0,1] range using log1p and scaling.
        
        B_normalized = log1p(B) / max_log_biomass
        """
        assert self._fitted, "Preprocessor must be fitted before transform"
        log_biomass = np.log1p(biomass.astype(np.float32))
        return log_biomass / (self.biomass_max + 1e-8)
    
    def transform_env_field(self, env_field: np.ndarray) -> np.ndarray:
        """
        Standardize environmental field.
        """
        assert self._fitted, "Preprocessor must be fitted before transform"
        return (env_field - self.env_stats['mean']) / (self.env_stats['std'] + 1e-8)
    
    def create_spatial_coords(self, height: int, width: int) -> np.ndarray:
        """
        Create normalized spatial coordinate channels.
        
        Returns: (2, H, W) array with x and y coordinates in [0,1]
        """
        y_coords = np.linspace(0, 1, height)[:, None].repeat(width, axis=1)
        x_coords = np.linspace(0, 1, width)[None, :].repeat(height, axis=0)
        return np.stack([x_coords, y_coords], axis=0).astype(np.float32)
    
    def build_interaction_graph(self, 
                                interaction_idx: np.ndarray,
                                interaction_weights: np.ndarray,
                                prevalence: np.ndarray) -> GraphData:
        """
        Build PyTorch Geometric graph from interaction network.
        
        Node features:
        - body_mass: Fixed at 1e-4 (from simulation params)
        - r_base: Base growth rate (could be extracted from ENV_r_field mean)
        - degree_in: Number of species that compete with this one
        - degree_out: Number of species this one competes with (always K=16)
        
        Edges:
        - From C_topk_idx: species i has edge to its top-K competitors
        - Edge weights from C_topk_w
        
        Args:
            interaction_idx: (S, K) indices of top competitors
            interaction_weights: (S, K) interaction strengths
            prevalence: (S,) final prevalence of each species
            
        Returns:
            PyTorch Geometric Data object
        """
        n_species = interaction_idx.shape[0]
        k_interactions = interaction_idx.shape[1]
        
        # Build edge index (COO format)
        # Edge from i to interaction_idx[i, k] means i competes with that species
        source_nodes = np.repeat(np.arange(n_species), k_interactions)
        target_nodes = interaction_idx.flatten()
        
        # Remove self-loops and invalid indices
        valid_mask = (source_nodes != target_nodes) & (target_nodes >= 0) & (target_nodes < n_species)
        source_nodes = source_nodes[valid_mask]
        target_nodes = target_nodes[valid_mask]
        
        edge_index = np.stack([source_nodes, target_nodes], axis=0)
        edge_weights = interaction_weights.flatten()[valid_mask]
        
        # Compute node degrees
        degree_out = np.bincount(source_nodes, minlength=n_species)
        degree_in = np.bincount(target_nodes, minlength=n_species)
        
        # Node features: [body_mass, prevalence, degree_in_norm, degree_out_norm]
        body_mass = np.full(n_species, 1e-4, dtype=np.float32)
        
        # Normalize degrees
        max_degree = max(degree_in.max(), degree_out.max(), 1)
        degree_in_norm = degree_in.astype(np.float32) / max_degree
        degree_out_norm = degree_out.astype(np.float32) / max_degree
        
        # Normalize prevalence
        prevalence_norm = prevalence.astype(np.float32)
        
        node_features = np.stack([
            body_mass,
            prevalence_norm,
            degree_in_norm,
            degree_out_norm
        ], axis=1)
        
        # Create PyG Data object
        graph = GraphData(
            x=torch.from_numpy(node_features).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_weights).float().unsqueeze(-1)
        )
        
        return graph
    
    def process_world(self, world: WorldData) -> Dict:
        """
        Process a single world into model-ready format.
        
        Returns dictionary with all tensors needed for training.
        """
        assert self._fitted, "Preprocessor must be fitted before processing"
        
        # Get dimensions
        n_timesteps, n_species, height, width = world.presence.shape
        
        # Process biomass
        biomass_norm = self.transform_biomass(world.biomass)
        
        # Process presence (already boolean, convert to float)
        presence = world.presence.astype(np.float32)
        
        # Process environment field
        env_norm = self.transform_env_field(world.env_field)
        
        # Create spatial coordinates
        spatial_coords = self.create_spatial_coords(height, width)
        
        # Build interaction graph
        final_prevalence = presence[-1].mean(axis=(1, 2))  # (S,)
        interaction_graph = self.build_interaction_graph(
            world.interaction_idx,
            world.interaction_weights,
            final_prevalence
        )
        
        # Process observation masks
        obs_masks = {
            budget: mask.astype(np.float32) 
            for budget, mask in world.obs_masks.items()
        }
        
        return {
            'world_id': world.world_id,
            'biomass': torch.from_numpy(biomass_norm).float(),  # (T, S, H, W)
            'presence': torch.from_numpy(presence).float(),  # (T, S, H, W)
            'env_field': torch.from_numpy(env_norm).float(),  # (S, H, W)
            'spatial_coords': torch.from_numpy(spatial_coords).float(),  # (2, H, W)
            'interaction_graph': interaction_graph,
            'obs_masks': {k: torch.from_numpy(v).float() for k, v in obs_masks.items()},
            'final_prevalence': torch.from_numpy(final_prevalence).float(),  # (S,)
            'metadata': world.metadata
        }


class EcoDataset(Dataset):
    """
    PyTorch Dataset for EcoDiffusion training.
    
    Supports different curriculum phases:
    - Phase 1: env_field → final_presence
    - Phase 2: env_field + interaction_graph → final_presence  
    - Phase 3: env_field + interaction_graph + temporal_history → future_presence
    - Phase 4: sparse_obs + env_field + interaction_graph → full_presence
    
    Memory optimization:
    - Lazy loading: Only load worlds when accessed
    - Caching: Keep recently accessed worlds in memory
    - Chunking: Can yield species chunks instead of full arrays
    """
    
    def __init__(self, 
                 world_files: List[Path],
                 preprocessor: DataPreprocessor,
                 loader: IBMDataLoader,
                 phase: int = 1,
                 cache_size: int = 10,
                 species_chunk_size: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            world_files: List of paths to world .npz files
            preprocessor: Fitted DataPreprocessor instance
            loader: IBMDataLoader instance
            phase: Curriculum phase (1-4)
            cache_size: Number of worlds to keep in memory
            species_chunk_size: If set, yield species chunks of this size
        """
        self.world_files = world_files
        self.preprocessor = preprocessor
        self.loader = loader
        self.phase = phase
        self.species_chunk_size = species_chunk_size
        
        # LRU cache for loaded worlds
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        
        # Precompute dataset length
        self._length = self._compute_length()
    
    def _compute_length(self) -> int:
        """Compute dataset length based on phase and chunking."""
        base_length = len(self.world_files)
        
        if self.species_chunk_size is not None:
            # One sample per species chunk per world
            n_species = self.preprocessor.config.data.n_species
            n_chunks = (n_species + self.species_chunk_size - 1) // self.species_chunk_size
            return base_length * n_chunks
        
        return base_length
    
    def __len__(self) -> int:
        return self._length
    
    def _load_world(self, idx: int) -> Dict:
        """Load world with caching."""
        world_idx = idx % len(self.world_files)
        
        if world_idx in self.cache:
            return self.cache[world_idx]
        
        # Load and process world
        world = self.loader.load_world(self.world_files[world_idx])
        processed = self.preprocessor.process_world(world)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[world_idx] = processed
        self.cache_order.append(world_idx)
        
        return processed
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.
        
        Returns different data based on curriculum phase:
        - Phase 1: Environment → Distribution
        - Phase 2: + Interaction context
        - Phase 3: + Temporal dynamics
        - Phase 4: + Sparse observation infilling
        """
        world_data = self._load_world(idx)
        
        # Handle species chunking
        if self.species_chunk_size is not None:
            chunk_idx = idx // len(self.world_files)
            start_species = chunk_idx * self.species_chunk_size
            end_species = min(start_species + self.species_chunk_size, 
                            world_data['presence'].shape[1])
            species_slice = slice(start_species, end_species)
        else:
            species_slice = slice(None)
        
        # Build sample based on phase
        sample = self._build_phase_sample(world_data, species_slice)
        
        return sample
    
    def _build_phase_sample(self, world_data: Dict, species_slice: slice) -> Dict:
        """Build training sample based on curriculum phase."""
        
        # Target: final presence/absence distribution
        target = world_data['presence'][-1, species_slice]  # (S, H, W)
        
        # Always include spatial coordinates
        spatial_coords = world_data['spatial_coords']  # (2, H, W)
        
        sample = {
            'target': target,
            'spatial_coords': spatial_coords,
            'world_id': world_data['world_id'],
        }
        
        # Phase 1+: Environment field
        env_field = world_data['env_field'][species_slice]  # (S, H, W)
        sample['env_field'] = env_field
        
        # Phase 2+: Interaction graph
        if self.phase >= 2:
            # For species chunking, we need to subset the graph
            if species_slice != slice(None):
                sample['interaction_graph'] = self._subset_graph(
                    world_data['interaction_graph'], species_slice)
            else:
                sample['interaction_graph'] = world_data['interaction_graph']
        
        # Phase 3+: Temporal history
        if self.phase >= 3:
            # Use first half of time series as input, predict second half
            mid_t = world_data['presence'].shape[0] // 2
            temporal_history = world_data['presence'][:mid_t, species_slice]  # (T/2, S, H, W)
            sample['temporal_history'] = temporal_history
            
            # Target becomes second half's final state
            sample['target'] = world_data['presence'][-1, species_slice]
        
        # Phase 4: Sparse observations
        if self.phase >= 4:
            # Randomly select observation budget for training
            budgets = list(world_data['obs_masks'].keys())
            # Bias towards sparser observations (more challenging)
            weights = [1.0 / (b + 1) for b in budgets]
            weights = np.array(weights) / sum(weights)
            budget = np.random.choice(budgets, p=weights)
            
            obs_mask = world_data['obs_masks'][budget][species_slice]  # (S, H, W)
            observed_presence = world_data['presence'][-1, species_slice] * obs_mask
            
            sample['obs_mask'] = obs_mask
            sample['observed_presence'] = observed_presence
            sample['obs_budget'] = budget
        
        return sample
    
    def _subset_graph(self, graph: GraphData, species_slice: slice) -> GraphData:
        """Subset interaction graph for species chunk."""
        start = species_slice.start or 0
        stop = species_slice.stop
        
        # Get nodes in range
        mask = (graph.edge_index[0] >= start) & (graph.edge_index[0] < stop) & \
               (graph.edge_index[1] >= start) & (graph.edge_index[1] < stop)
        
        new_edge_index = graph.edge_index[:, mask] - start
        new_edge_attr = graph.edge_attr[mask] if graph.edge_attr is not None else None
        new_x = graph.x[start:stop]
        
        return GraphData(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    
    def set_phase(self, phase: int):
        """Update curriculum phase."""
        assert 1 <= phase <= 4, f"Phase must be 1-4, got {phase}"
        self.phase = phase


def create_data_loaders(config, verbose: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    This is the main entry point for data preparation.
    
    Args:
        config: EcoConfig with data settings
        verbose: Print progress information
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config.data
    
    # Initialize loader
    loader = IBMDataLoader(data_config.data_root, verbose=verbose)
    
    if loader.n_worlds == 0:
        raise ValueError(f"No world files found in {data_config.data_root}")
    
    # Split world files
    all_files = loader.world_files
    
    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(config.training.seed)
    indices = rng.permutation(len(all_files))
    
    train_end = data_config.n_train
    val_end = train_end + data_config.n_val
    
    train_files = [all_files[i] for i in indices[:train_end]]
    val_files = [all_files[i] for i in indices[train_end:val_end]]
    test_files = [all_files[i] for i in indices[val_end:]]
    
    if verbose:
        print(f"[DataLoader] Split: {len(train_files)} train, "
              f"{len(val_files)} val, {len(test_files)} test")
    
    # Initialize and fit preprocessor on training data
    preprocessor = DataPreprocessor(config)
    
    if verbose:
        print("[DataLoader] Fitting preprocessor on training worlds...")
    
    # Load a subset for fitting statistics (memory efficient)
    fit_worlds = [loader.load_world(f) for f in train_files[:min(20, len(train_files))]]
    preprocessor.fit(fit_worlds)
    del fit_worlds  # Free memory
    
    # Create datasets
    train_dataset = EcoDataset(
        train_files, preprocessor, loader, 
        phase=1,  # Start with phase 1
        cache_size=config.data.num_workers * 2,
        species_chunk_size=config.model.species_chunk_size if config.device != "cpu" else None
    )
    
    val_dataset = EcoDataset(
        val_files, preprocessor, loader,
        phase=1,
        cache_size=5,
        species_chunk_size=config.model.species_chunk_size if config.device != "cpu" else None
    )
    
    test_dataset = EcoDataset(
        test_files, preprocessor, loader,
        phase=1,
        cache_size=5,
        species_chunk_size=config.model.species_chunk_size if config.device != "cpu" else None
    )
    
    # Determine batch size
    batch_size = config.training.batch_size
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.num_workers if config.device != "cpu" else 0,
        pin_memory=data_config.pin_memory and config.device == "cuda",
        prefetch_factor=data_config.prefetch_factor if config.device != "cpu" else None,
        drop_last=True,
        collate_fn=eco_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, data_config.num_workers) if config.device != "cpu" else 0,
        pin_memory=data_config.pin_memory and config.device == "cuda",
        collate_fn=eco_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, data_config.num_workers) if config.device != "cpu" else 0,
        pin_memory=data_config.pin_memory and config.device == "cuda",
        collate_fn=eco_collate_fn
    )
    
    if verbose:
        print(f"[DataLoader] Created loaders with batch_size={batch_size}")
    
    return train_loader, val_loader, test_loader, preprocessor


def eco_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for EcoDiffusion batches.
    
    Handles:
    - Standard tensor stacking
    - Graph batching for PyTorch Geometric
    - Variable-presence keys across phases
    """
    from torch_geometric.data import Batch as GraphBatch
    
    collated = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [b[key] for b in batch]
        
        if key == 'interaction_graph':
            # Batch graphs using PyG
            collated[key] = GraphBatch.from_data_list(values)
        elif key in ['world_id', 'obs_budget']:
            # Keep as list
            collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            collated[key] = torch.stack(values, dim=0)
        else:
            # Keep as list for other types
            collated[key] = values
    
    return collated


# ============================================================================
# Synthetic Data Generator (for testing without real data)
# ============================================================================

class SyntheticWorldGenerator:
    """
    Generates synthetic world data for testing the pipeline.
    
    Useful when:
    - Real IBM data is not available
    - Testing model architecture
    - Debugging training pipeline
    """
    
    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState(config.training.seed)
    
    def generate_world(self, world_id: str = "synthetic_0") -> WorldData:
        """Generate a synthetic world with realistic structure."""
        n_species = self.config.data.n_species
        n_timesteps = self.config.data.n_timesteps
        height, width = self.config.data.grid_size
        n_interactions = self.config.data.n_interactions
        
        # Generate environmental field (Gaussian Random Field approximation)
        env_field = self._generate_grf(n_species, height, width, length_scale=2.5)
        
        # Generate presence/absence from environment (sigmoid of env_field)
        presence_probs = 1 / (1 + np.exp(-env_field * 10))
        presence = self.rng.binomial(1, presence_probs).astype(bool)
        
        # Expand to time series with small random changes
        presence_t = np.zeros((n_timesteps, n_species, height, width), dtype=bool)
        presence_t[0] = presence
        for t in range(1, n_timesteps):
            # Small random colonization/extinction
            change_prob = 0.02
            changes = self.rng.binomial(1, change_prob, presence.shape).astype(bool)
            presence_t[t] = presence_t[t-1] ^ changes
        
        # Generate biomass from presence
        biomass = presence_t.astype(np.float32) * self.rng.exponential(10, presence_t.shape)
        
        # Generate interaction network
        interaction_idx = np.zeros((n_species, n_interactions), dtype=np.int32)
        interaction_weights = np.zeros((n_species, n_interactions), dtype=np.float32)
        
        for s in range(n_species):
            # Random competitors (excluding self)
            others = np.concatenate([np.arange(s), np.arange(s+1, n_species)])
            if len(others) >= n_interactions:
                competitors = self.rng.choice(others, n_interactions, replace=False)
            else:
                competitors = np.pad(others, (0, n_interactions - len(others)), constant_values=-1)
            interaction_idx[s] = competitors
            interaction_weights[s] = self.rng.uniform(0.1, 0.5, n_interactions)
        
        # Generate observation masks
        obs_masks = {}
        for budget in [1, 5, 10, 20, 50, 100]:
            mask = np.zeros((n_species, height, width), dtype=bool)
            # Random cells observed
            for s in range(n_species):
                n_obs = min(budget, height * width)
                obs_idx = self.rng.choice(height * width, n_obs, replace=False)
                obs_y, obs_x = np.unravel_index(obs_idx, (height, width))
                mask[s, obs_y, obs_x] = True
            obs_masks[budget] = mask
        
        return WorldData(
            world_id=world_id,
            biomass=biomass.astype(np.float16),
            presence=presence_t,
            env_field=env_field.astype(np.float32),
            interaction_idx=interaction_idx,
            interaction_weights=interaction_weights,
            obs_masks=obs_masks,
            metadata={'synthetic': True}
        )
    
    def _generate_grf(self, n_species: int, height: int, width: int, 
                      length_scale: float = 2.5) -> np.ndarray:
        """Generate approximate Gaussian Random Field."""
        # Simple approximation using smoothed random noise
        from scipy.ndimage import gaussian_filter
        
        raw = self.rng.randn(n_species, height, width)
        smoothed = np.array([
            gaussian_filter(raw[s], sigma=length_scale) 
            for s in range(n_species)
        ])
        return smoothed
    
    def generate_worlds(self, n_worlds: int, save_dir: str = None) -> List[WorldData]:
        """Generate multiple synthetic worlds."""
        worlds = []
        for i in range(n_worlds):
            world = self.generate_world(f"synthetic_{i}")
            worlds.append(world)
            
            if save_dir:
                self._save_world(world, save_dir)
        
        return worlds
    
    def _save_world(self, world: WorldData, save_dir: str):
        """Save synthetic world to .npz file."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{world.world_id}.npz"
        
        np.savez_compressed(
            save_path,
            IBM_B=world.biomass,
            P_t=world.presence,
            ENV_r_field=world.env_field,
            C_topk_idx=world.interaction_idx,
            C_topk_w=world.interaction_weights,
            **{f'obs_mask_{k}': v for k, v in world.obs_masks.items()},
            metadata=world.metadata
        )


if __name__ == "__main__":
    """Test data loading and preprocessing."""
    from config import get_debug_config
    
    print("=" * 60)
    print("Testing Data Preprocessing Pipeline")
    print("=" * 60)
    
    config = get_debug_config()
    
    # Test synthetic data generation
    print("\n[Test 1] Generating synthetic data...")
    generator = SyntheticWorldGenerator(config)
    synthetic_dir = "/tmp/synthetic_worlds"
    generator.generate_worlds(5, save_dir=synthetic_dir)
    print(f"  Generated 5 synthetic worlds in {synthetic_dir}")
    
    # Test data loading
    print("\n[Test 2] Testing data loader...")
    config.data.data_root = synthetic_dir
    config.data.n_train = 3
    config.data.n_val = 1
    config.data.n_test = 1
    config.data.n_worlds = 5
    
    train_loader, val_loader, test_loader, preprocessor = create_data_loaders(config)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test batch retrieval
    print("\n[Test 3] Testing batch retrieval...")
    batch = next(iter(train_loader))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Target shape: {batch['target'].shape}")
    print(f"  Env field shape: {batch['env_field'].shape}")
    
    print("\n[SUCCESS] Data preprocessing tests passed!")
