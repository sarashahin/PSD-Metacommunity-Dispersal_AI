"""
=============================================================================
STAGE 2 CONFIGURATION: EcoDiffusion Training Configuration
=============================================================================
This configuration file contains all hyperparameters, paths, and settings
for training the EcoDiffusion model on IBM simulation outputs.

REASONING FOR KEY CHOICES:
- Diffusion steps (1000): Standard for high-quality generation, balances quality/speed
- Cosine beta schedule: Smoother noise schedule, better for spatial data
- Curriculum phases: Gradually introduce complexity to stabilize training
- Learning rate (1e-4): Conservative for diffusion models, prevents mode collapse
- Batch size: Adjusted based on GPU memory, species count
=============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch


import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))


@dataclass
class PathConfig:
    """File paths configuration."""
    # Input data paths
    simulation_dir: str = "./simulation_outputs"  # Directory containing .npz files
    output_dir: str = "./stage2_outputs"
    checkpoint_dir: str = "./stage2_outputs/checkpoints"
    log_dir: str = "./stage2_outputs/logs"
    
    # Specific file patterns
    npz_pattern: str =  "*_training.npz"  # Pattern to match simulation files
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class DataConfig:
    """Data preprocessing configuration."""
    # Expected data dimensions from simulation
    n_species_max: int = 4000  # Maximum species in any world
    grid_size: Tuple[int, int] = (20, 20)  # Spatial grid dimensions
    n_timesteps: int = 50  # Number of time snapshots
    
    # Data split ratios
    train_ratio: float = 0.833  # 200/240 worlds for training
    val_ratio: float = 0.083   # 20/240 worlds for validation
    test_ratio: float = 0.084  # 20/240 worlds for testing
    
    # Normalization settings
    biomass_log_transform: bool = True  # Apply log1p to biomass
    biomass_clip_percentile: float = 99.5  # Clip extreme values
    
    # Observation masks to use
    obs_mask_budgets: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    
    # Species filtering
    min_prevalence: float = 0.001  # Minimum species prevalence to include
    
    # Rare species definition
    rare_species_threshold: float = 0.05  # Species with prevalence < 5% are "rare"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # ========== Environmental Encoder (CNN) ==========
    # REASONING: CNN is ideal for ENV_r_field because:
    # 1. Spatial autocorrelation in environment (length_scale=2.5)
    # 2. Local receptive fields match ecological niche filtering
    # 3. Translation equivariance appropriate for gridded landscapes
    env_encoder_in_channels: int = 2  # ENV_r_field + x_coord + y_coord
    env_encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    env_encoder_kernel_size: int = 3
    env_encoder_output_dim: int = 256
    
    # ========== Interaction Encoder (Graph Neural Network) ==========
    # REASONING: GNN captures species interactions because:
    # 1. Interaction matrix is sparse (16 connections per species via C_topk)
    # 2. Message passing mimics ecological interaction propagation
    # 3. Captures indirect effects (A→B→C competition cascades)
    gnn_input_dim: int = 8  # Species features: body_mass, r_base, deg_in, deg_out, etc.
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 256
    gnn_num_layers: int = 3
    gnn_heads: int = 4  # Attention heads
    gnn_dropout: float = 0.1
    
    # ========== Temporal Encoder (Transformer) ==========
    # REASONING: Transformer for temporal dynamics because:
    # 1. Long-range temporal dependencies (colonization → persistence → extinction)
    # 2. Self-attention captures non-Markovian dynamics
    # 3. Handles variable time gaps between snapshots
    temporal_input_dim: int = 400  # 20x20 flattened spatial grid
    temporal_hidden_dim: int = 256
    temporal_num_heads: int = 8
    temporal_num_layers: int = 4
    temporal_dropout: float = 0.1
    temporal_max_seq_len: int = 50  # Maximum timesteps
    
    # ========== Denoising U-Net ==========
    # REASONING: U-Net with attention for diffusion because:
    # 1. Skip connections preserve fine-grained spatial details
    # 2. Multi-scale processing captures both local and global patterns
    # 3. Dual attention (spatial + species) captures community structure
    unet_base_channels: int = 64
    unet_channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: List[int] = field(default_factory=lambda: [10, 5])
    unet_dropout: float = 0.1
    unet_use_spatial_attention: bool = True
    unet_use_species_attention: bool = True
    
    # ========== Diffusion Process ==========
    # REASONING for diffusion settings:
    # 1. 1000 steps: Standard for high-quality generation
    # 2. Cosine schedule: Smoother, better for spatial data than linear
    # 3. Variance type 'learned': More flexibility for ecological data
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"  # Options: "linear", "cosine", "sqrt"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    variance_type: str = "fixed_small"  # Options: "fixed_small", "fixed_large", "learned"
    
    # ========== Conditioning ==========
    conditioning_dim: int = 512  # Combined conditioning embedding dimension
    condition_dropout: float = 0.1  # Dropout for classifier-free guidance


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # ========== General Training ==========
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 0
    pin_memory: bool = False  # Disable for lazy loading
    
    # ========== Batch Sizes ==========
    # REASONING: Smaller batch due to large species dimension (3614 species)
    # Adjusted based on GPU memory availability
    batch_size: int = 1  # Worlds per batch
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    
    # ========== Optimizer ==========
    # REASONING: AdamW with weight decay for regularization
    # Conservative learning rate for stable diffusion training
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # ========== Learning Rate Schedule ==========
    # REASONING: Cosine decay with warmup for smooth convergence
    lr_scheduler: str = "cosine"  # Options: "cosine", "linear", "constant"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # ========== Curriculum Learning Phases ==========
    # REASONING: Gradually introduce complexity
    # Phase 1: Learn environmental filtering (simplest)
    # Phase 2: Add interaction effects
    # Phase 3: Add temporal dynamics
    # Phase 4: Add sparse observation infilling (hardest)
    phase1_epochs: int = 50   # Environment → Distribution
    phase2_epochs: int = 100  # + Interaction context
    phase3_epochs: int = 150  # + Temporal dynamics
    phase4_epochs: int = 200  # + Sparse observation infilling
    total_epochs: int = 500
    
    # ========== Loss Weights ==========
    # REASONING: Diffusion loss is primary, auxiliary losses for ecological constraints
    loss_diffusion_weight: float = 1.0
    loss_prevalence_weight: float = 0.1  # Preserve species frequency
    loss_cooccurrence_weight: float = 0.1  # Preserve species associations
    loss_spatial_weight: float = 0.05  # Spatial autocorrelation
    
    # ========== Validation & Checkpointing ==========
    val_every_epochs: int = 5
    save_every_epochs: int = 25
    early_stopping_patience: int = 30
    save_best_only: bool = True
    best_metric: str = "val_auc_rare"  # Optimize for rare species AUC
    
    # ========== Mixed Precision ==========
    use_amp: bool = True  # Automatic Mixed Precision for GPU efficiency
    grad_clip_norm: float = 1.0
    
    # ========== Logging ==========
    log_every_steps: int = 50
    use_wandb: bool = False  # Set to True if using Weights & Biases
    project_name: str = "ecodiffusion_stage2"
    run_name: Optional[str] = None


@dataclass
class EcoConfig:
    """
    Master configuration combining all sub-configurations.
    """
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def get_device(self) -> torch.device:
        """Determine the device to use for training."""
        if self.training.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.training.device)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure ratios sum to 1
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        assert abs(total_ratio - 1.0) < 0.01, f"Data split ratios must sum to 1, got {total_ratio}"
        
        # Ensure curriculum phases are cumulative
        assert self.training.phase1_epochs < self.training.phase2_epochs, \
            "Phase 2 must start after Phase 1"
        assert self.training.phase2_epochs < self.training.phase3_epochs, \
            "Phase 3 must start after Phase 2"
        assert self.training.phase3_epochs < self.training.phase4_epochs, \
            "Phase 4 must start after Phase 3"


def get_default_config() -> EcoConfig:
    """Return the default configuration."""
    return EcoConfig()


def load_config_from_yaml(yaml_path: str) -> EcoConfig:
    """Load configuration from a YAML file (optional)."""
    import yaml
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = EcoConfig()
    
    # Update paths
    if 'paths' in config_dict:
        for key, value in config_dict['paths'].items():
            if hasattr(config.paths, key):
                setattr(config.paths, key, value)
    
    # Update data config
    if 'data' in config_dict:
        for key, value in config_dict['data'].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
    
    # Update model config
    if 'model' in config_dict:
        for key, value in config_dict['model'].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
    
    # Update training config
    if 'training' in config_dict:
        for key, value in config_dict['training'].items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
    
    return config


if __name__ == "__main__":
    # Print default configuration for verification
    config = get_default_config()
    print("=" * 60)
    print("ECODIFFUSION STAGE 2 - DEFAULT CONFIGURATION")
    print("=" * 60)
    print(f"\nDevice: {config.get_device()}")
    print(f"\nPaths:")
    print(f"  Simulation dir: {config.paths.simulation_dir}")
    print(f"  Output dir: {config.paths.output_dir}")
    print(f"\nData:")
    print(f"  Grid size: {config.data.grid_size}")
    print(f"  N timesteps: {config.data.n_timesteps}")
    print(f"  Train/Val/Test: {config.data.train_ratio}/{config.data.val_ratio}/{config.data.test_ratio}")
    print(f"\nModel:")
    print(f"  Diffusion steps: {config.model.diffusion_steps}")
    print(f"  U-Net channels: {config.model.unet_base_channels}")
    print(f"\nTraining:")
    print(f"  Total epochs: {config.training.total_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print("=" * 60)
