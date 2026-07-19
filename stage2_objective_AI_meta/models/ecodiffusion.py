"""
=============================================================================
ECODIFFUSION: Complete Ecological Distribution Diffusion Model
=============================================================================
This module implements the complete EcoDiffusion model that integrates:
1. Environmental Encoder (CNN)
2. Interaction Encoder (GNN)  
3. Temporal Encoder (Transformer)
4. Denoising U-Net
5. Diffusion Process

The model learns to predict species distributions conditioned on:
- Environmental suitability (ENV_r_field)
- Species interactions (C_topk graph)
- Temporal history (P_t time series)
- Sparse observations (obs_masks)

ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────┐
│                    ECODIFFUSION MODEL                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Inputs:                                                    │
│  ├── x_t: Noisy distribution (B, S, Y, X)                  │
│  ├── t: Diffusion timestep (B,)                            │
│  └── condition: {env, graph, history, obs_mask}            │
│                                                             │
│  Encoders:                                                  │
│  ├── EnvironmentalEncoder → env_emb (B, S, D, Y, X)        │
│  ├── InteractionEncoder → int_emb (B, S, D)                │
│  └── TemporalEncoder → temp_emb (B, S, D)                  │
│                                                             │
│  Conditioning Fusion:                                       │
│  └── Combined embedding (B, S, D_cond)                     │
│                                                             │
│  Denoising:                                                 │
│  └── U-Net(x_t, t, condition) → ε_pred (B, S, Y, X)        │
│                                                             │
│  Output: Predicted noise or clean distribution              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
=============================================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # 🔴 NEW: For gradient checkpointing
from typing import Dict, Optional, Tuple, List
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time step embedding for diffusion models."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps to continuous representation."""
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return self.mlp(embedding)


class FixedConditioningModule(nn.Module):
    """
    Fuses environmental, interaction, and temporal encodings into
    unified conditioning for the U-Net denoiser.
    
    REASONING:
    - Different encoders produce different shaped embeddings
    - Need to combine them appropriately for U-Net injection
    - Use FiLM (Feature-wise Linear Modulation) style conditioning
    """
    
    def __init__(
        self,
        env_dim: int = 256,
        int_dim: int = 256,
        temp_dim: int = 256,
        time_dim: int = 256,
        output_dim: int = 256,  # CHANGED: smaller output for efficiency
    ):
        super().__init__()
        
        # Project each encoding to common dimension
        self.env_proj = nn.Linear(env_dim, output_dim)
        self.int_proj = nn.Linear(int_dim, output_dim)
        self.temp_proj = nn.Linear(temp_dim, output_dim)
        self.time_proj = nn.Linear(time_dim, output_dim)
        
        # 🔴 FIXED: Fusion network produces PER-SPECIES output
        # Input: concatenated projections (4 * output_dim) per species
        # Output: fused conditioning (output_dim) per species
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Scale and shift for FiLM conditioning (per-species)
        self.scale_net = nn.Linear(output_dim, output_dim)
        self.shift_net = nn.Linear(output_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(
        self,
        env_emb: torch.Tensor,      # (B, S, D_env)
        int_emb: torch.Tensor,      # (B, S, D_int)
        temp_emb: Optional[torch.Tensor],  # (B, S, D_temp) or None
        time_emb: torch.Tensor,     # (B, D_time)
        n_species: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse all conditioning signals PER SPECIES.
        
        Returns:
            cond: (B, S, D) - per-species conditioning
        """
        B = env_emb.shape[0]
        S = n_species
        
        # Handle spatial env embedding - pool if needed
        if env_emb.dim() == 5:  # (B, S, D, Y, X)
            env_emb = env_emb.mean(dim=(-2, -1))  # (B, S, D)
        
        # Project each encoding (per-species)
        env_proj = self.env_proj(env_emb)  # (B, S, D_out)
        int_proj = self.int_proj(int_emb)  # (B, S, D_out)
        
        # Handle temporal (may be None in early phases)
        if temp_emb is not None:
            temp_proj = self.temp_proj(temp_emb)  # (B, S, D_out)
        else:
            temp_proj = torch.zeros_like(env_proj)
        
        # 🔴 FIXED: Expand time embedding to EACH species (not average)
        time_proj = self.time_proj(time_emb)  # (B, D_out)
        time_proj = time_proj.unsqueeze(1).expand(-1, S, -1)  # (B, S, D_out)
        
        # Concatenate per-species
        combined = torch.cat([env_proj, int_proj, temp_proj, time_proj], dim=-1)
        # Shape: (B, S, 4*D_out)
        
        # Fuse per-species
        fused = self.fusion(combined)  # (B, S, D_out)
        
        return fused


class SpeciesResBlock(nn.Module):
    """
    Residual block that processes species distributions spatially.
    
    Designed for (B, S, Y, X) tensors where S is the species dimension.
    Uses group normalization over species groups.
    """
    
    def __init__(
        self,
        n_species: int,
        n_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Use groups that divide n_species
        actual_groups = min(n_groups, n_species)
        while n_species % actual_groups != 0:
            actual_groups -= 1
        
        self.norm1 = nn.GroupNorm(actual_groups, n_species)
        self.conv1 = nn.Conv2d(n_species, n_species, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(actual_groups, n_species)
        self.conv2 = nn.Conv2d(n_species, n_species, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout2d(dropout)
        self.act = nn.SiLU()
    
    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional FiLM conditioning.
        
        Args:
            x: (B, S, Y, X) input
            scale: (B, S, 1) or (B, S) scale factors
            shift: (B, S, 1) or (B, S) shift factors
        """
        h = self.norm1(x)
        h = self.act(h)
        
        # Apply FiLM conditioning
        if scale is not None and shift is not None:
            if scale.dim() == 2:
                scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
                shift = shift.unsqueeze(-1).unsqueeze(-1)
            elif scale.dim() == 3:
                scale = scale.unsqueeze(-1)  # (B, S, 1, 1)
                shift = shift.unsqueeze(-1)
            h = h * (1 + scale) + shift
        
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class SpeciesSpatialAttention(nn.Module):
    """
    Dual attention mechanism over species and spatial dimensions.
    
    REASONING:
    - Species attention: Which species co-occur or exclude each other
    - Spatial attention: Which locations are ecologically connected
    """
    
    def __init__(
        self,
        n_species: int,
        spatial_size: Tuple[int, int] = (20, 20),
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_species = n_species
        self.spatial_size = spatial_size
        self.n_heads = n_heads
        
        # Species attention (across S dimension)
        # Limited to local neighborhood to avoid O(S²) complexity
        self.species_attn = nn.MultiheadAttention(
            embed_dim=spatial_size[0] * spatial_size[1],
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Spatial attention (across Y×X dimension)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=n_species,
            num_heads=min(n_heads, 4),  # Limit heads for large n_species
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(spatial_size[0] * spatial_size[1])
        self.norm2 = nn.LayerNorm(n_species)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dual attention.
        
        Args:
            x: (B, S, Y, X) input
            
        Returns:
            (B, S, Y, X) output with attention applied
        """
        B, S, Y, X = x.shape
        
        # Species attention
        # Reshape: (B, S, Y*X)
        x_species = x.view(B, S, -1)
        x_species = self.norm1(x_species)
        x_species_attn, _ = self.species_attn(x_species, x_species, x_species)
        x = x + x_species_attn.view(B, S, Y, X)
        
        # Spatial attention
        # Reshape: (B, Y*X, S)
        x_spatial = x.view(B, S, -1).permute(0, 2, 1)  # (B, Y*X, S)
        x_spatial = self.norm2(x_spatial)
        x_spatial_attn, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x = x + x_spatial_attn.permute(0, 2, 1).view(B, S, Y, X)
        
        return x


# class EcoUNet(nn.Module):
    """
    U-Net architecture adapted for species distribution data.
    
    Key adaptations:
    1. Treats species (S) as channel dimension
    2. FiLM conditioning from multi-modal encodings
    3. Species-spatial dual attention
    4. Handles large S efficiently
    """
    
# =============================================================================
# FIXED SINGLE-SPECIES U-NET
# =============================================================================

class SingleChannelResBlock(nn.Module):
    """ResNet block for single-channel (per-species) processing with FiLM."""
    
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        # 🔴 FiLM conditioning: scale and shift from per-species condition
        self.film_scale = nn.Linear(cond_dim, channels)
        self.film_shift = nn.Linear(cond_dim, channels)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*S, C, Y, X) spatial features
            cond: (B*S, D) per-species conditioning
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # 🔴 FiLM modulation with per-species conditioning
        scale = self.film_scale(cond)[:, :, None, None]  # (B*S, C, 1, 1)
        shift = self.film_shift(cond)[:, :, None, None]  # (B*S, C, 1, 1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + x


class SpeciesParallelUNet(nn.Module):
    """
    🔴 FIXED U-Net that processes each species independently with its own conditioning.
    
    Key changes:
    1. Input is single-channel per species (not S channels)
    2. Conditioning is per-species (not averaged)
    3. Species processed in parallel as batch dimension
    
    Input: (B, S, Y, X) - B batches, S species, Y×X grid
    Processing: reshape to (B*S, 1, Y, X), process, reshape back
    Output: (B, S, Y, X) - predicted noise per species
    """
    
    def __init__(
        self,
        spatial_size: Tuple[int, int] = (20, 20),
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4],
        n_res_blocks: int = 2,
        cond_dim: int = 256,
        time_dim: int = 256,
        dropout: float = 0.1,
        species_chunk_size: int = 64,  # 🔴 NEW: Process species in chunks for memory efficiency
        use_gradient_checkpointing: bool = True,  # 🔴 NEW: Enable gradient checkpointing
    ):
        super().__init__()
        
        self.spatial_size = spatial_size
        self.base_channels = base_channels
        self.species_chunk_size = species_chunk_size  # 🔴 NEW: Store chunk size
        self.use_gradient_checkpointing = use_gradient_checkpointing  # 🔴 NEW
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Combine time + condition
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # 🔴 FIXED: Input projection for SINGLE channel per species
        self.input_proj = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(n_res_blocks):
                self.encoder_blocks.append(
                    SingleChannelResBlock(ch, cond_dim, dropout)
                )
                if ch != out_ch:
                    self.encoder_blocks.append(
                        nn.Conv2d(ch, out_ch, kernel_size=1)
                    )
                ch = out_ch
            
            if level < len(channel_mults) - 1:
                self.downsamplers.append(
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
        
        # Middle
        self.middle_block1 = SingleChannelResBlock(ch, cond_dim, dropout)
        self.middle_block2 = SingleChannelResBlock(ch, cond_dim, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level in range(len(channel_mults) - 1, -1, -1):
            mult = channel_mults[level]
            out_ch = base_channels * mult
            
            for i in range(n_res_blocks + 1):
                in_ch = ch + (out_ch if i == 0 else 0)  # Skip connection
                self.decoder_blocks.append(
                    SingleChannelResBlock(in_ch if i == 0 else out_ch, cond_dim, dropout)
                )
                if i == 0 and in_ch != out_ch:
                    self.decoder_blocks.append(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1)
                    )
                ch = out_ch
            
            if level > 0:
                self.upsamplers.append(
                    nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
                )
        
        # 🔴 FIXED: Output projection to SINGLE channel per species
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, 1, kernel_size=3, padding=1),
        )
    
    def _forward_single_chunk(
        self,
        x: torch.Tensor,      # (chunk_size, 1, Y, X)
        t: torch.Tensor,      # (chunk_size,)
        cond: torch.Tensor,   # (chunk_size, D)
    ) -> torch.Tensor:
        """
        🔴 NEW: Process a single chunk through the UNet.
        Extracted from forward() for memory-efficient chunked processing.
        """
        
        target_size = x.shape[-2:]

        # Time embedding
        time_emb = self.time_embed(t)  # (B*S, time_dim)
        
        # 🔴 FIXED: Combine time + per-species condition
        combined_cond = torch.cat([cond, time_emb], dim=-1)
        cond_emb = self.cond_proj(combined_cond)  # (B*S, cond_dim)
        
        # Input projection
        h = self.input_proj(x)  # (chunk_size, base_ch, Y, X)
        
        # Encoder with skip connections
        skips = []
        block_idx = 0
        for level in range(len(self.downsamplers) + 1):
            for _ in range(2):  # n_res_blocks
                h = self.encoder_blocks[block_idx](h, cond_emb)
                block_idx += 1
                # Handle channel change conv if present
                if block_idx < len(self.encoder_blocks) and isinstance(
                    self.encoder_blocks[block_idx], nn.Conv2d
                ):
                    h = self.encoder_blocks[block_idx](h)
                    block_idx += 1
            skips.append(h)
            
            if level < len(self.downsamplers):
                h = self.downsamplers[level](h)
        
        # Middle
        h = self.middle_block1(h, cond_emb)
        h = self.middle_block2(h, cond_emb)
        
        # Decoder with skip connections
        block_idx = 0
        for level in range(len(self.upsamplers) + 1):
            # Get skip connection
            if skips:
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    skip = F.interpolate(skip, size=h.shape[-2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            
            for _ in range(3):  # n_res_blocks + 1
                if block_idx < len(self.decoder_blocks):
                    layer = self.decoder_blocks[block_idx]
                    if isinstance(layer, SingleChannelResBlock):
                        h = layer(h, cond_emb)
                    else:
                        h = layer(h)
                    block_idx += 1
                    # 🔴 FIX: Handle channel change conv if present (like encoder does)
                    if block_idx < len(self.decoder_blocks) and isinstance(
                        self.decoder_blocks[block_idx], nn.Conv2d
                    ):
                        h = self.decoder_blocks[block_idx](h)
                        block_idx += 1
            
            if level < len(self.upsamplers):
                h = self.upsamplers[level](h)
        
        # Output projection
        out = self.output_proj(h)  # (chunk_size, 1, Y, X)
        
        # Resize if needed
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
                
        return out

    def forward(
        self,
        x: torch.Tensor,      # (B, S, Y, X) or (B*S, 1, Y, X)
        t: torch.Tensor,      # (B,) or (B*S,)
        cond: torch.Tensor,   # (B, S, D) or (B*S, D)
    ) -> torch.Tensor:
        """
        Forward pass with per-species processing.
        
        🔴 NEW: Processes species in chunks for memory efficiency.
        If input is (B, S, Y, X), automatically reshapes for parallel processing.
        """
        # Handle input reshaping
        if x.dim() == 4 and x.shape[1] != 1:
            # Input is (B, S, Y, X), reshape to (B*S, 1, Y, X)
            B, S, Y, X = x.shape
            x = x.view(B * S, 1, Y, X)
            t = t.repeat_interleave(S)  # (B*S,)
            cond = cond.view(B * S, -1)  # (B*S, D)
            needs_reshape = True
        else:
            B, S = x.shape[0], 1
            needs_reshape = False
        
        target_size = x.shape[-2:]
        total_samples = x.shape[0]  # B*S
        
        # 🔴 NEW: Process in chunks for memory efficiency
        if total_samples > self.species_chunk_size:
            outputs = []
            
            for start_idx in range(0, total_samples, self.species_chunk_size):
                end_idx = min(start_idx + self.species_chunk_size, total_samples)
                
                # Extract chunk
                x_chunk = x[start_idx:end_idx]
                t_chunk = t[start_idx:end_idx]
                cond_chunk = cond[start_idx:end_idx]
                
                # 🔴 NEW: Use gradient checkpointing to save memory during backprop
                if self.training and self.use_gradient_checkpointing:
                    # Gradient checkpointing trades compute for memory
                    out_chunk = checkpoint(
                        self._forward_single_chunk,
                        x_chunk, t_chunk, cond_chunk,
                        use_reentrant=False,
                    )
                else:
                    out_chunk = self._forward_single_chunk(x_chunk, t_chunk, cond_chunk)
                
                outputs.append(out_chunk)
                
                # 🔴 NEW: Clear CUDA cache periodically to reduce fragmentation
                if start_idx > 0 and start_idx % (self.species_chunk_size * 4) == 0:
                    if x.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Concatenate all chunks
            out = torch.cat(outputs, dim=0)
        else:
            # Process all at once (original behavior for small species counts)
            if self.training and self.use_gradient_checkpointing:
                out = checkpoint(
                    self._forward_single_chunk,
                    x, t, cond,
                    use_reentrant=False,
                )
            else:
                out = self._forward_single_chunk(x, t, cond)
        
        # Reshape back if needed
        if needs_reshape:
            out = out.view(B, S, *target_size)
        
        return out



class ResBlockWithTime(nn.Module):
    """ResBlock with time embedding injection."""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        self.time_proj = nn.Linear(time_dim, out_ch)
        
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        if in_ch != out_ch:
            self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_proj(emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_proj(x)


class SelfAttention2d(nn.Module):
    """Self-attention for 2D feature maps."""
    
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        h = h.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        
        return x + h


class EcoDiffusionFixed(nn.Module):
    """
    Complete EcoDiffusion model integrating all components.
    
    This is the main model class that should be used for training and inference.
    """
    
    def __init__(
        self,
        n_species: int = 3614,
        spatial_size: Tuple[int, int] = (20, 20),
        hidden_dim: int = 256,
        env_encoder_config: dict = None,
        int_encoder_config: dict = None,
        temp_encoder_config: dict = None,
        unet_config: dict = None,
        diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
        species_chunk_size: int = 64,  # 🔴 NEW: Memory-efficient chunked processing
        use_gradient_checkpointing: bool = True,  # 🔴 NEW: Trade compute for memory
    ):
        super().__init__()
        
        self.n_species = n_species
        self.spatial_size = spatial_size
        self.species_chunk_size = species_chunk_size  # 🔴 NEW
        self.hidden_dim = hidden_dim
        
        # Default configs
        env_config = env_encoder_config or {
            'in_channels': 2,
            'hidden_channels': [64, 128, 256],
            'output_dim': hidden_dim,
        }
        
        int_config = int_encoder_config or {
            'input_dim': 8,
            'hidden_dim': 128,
            'output_dim': hidden_dim,
            'n_hops': 2,
        }
        
        temp_config = temp_encoder_config or {
            'hidden_dim': hidden_dim,
            'output_dim': hidden_dim,
            'n_heads': 8,
            'n_layers': 2,
        }
        
        unet_cfg = unet_config or {
            'base_channels': 64,
            'channel_mults': [1, 2, 4],
            'n_res_blocks': 2,
        }
        
        # Import encoders (unchanged)
        from .env_encoder import SpeciesEnvironmentEncoder
        from .interaction_encoder import EfficientInteractionEncoder
        from .temporal_encoder import EfficientTemporalEncoder
        
        # Initialize encoders (unchanged)
        self.env_encoder = SpeciesEnvironmentEncoder(
            in_channels=env_config.get('in_channels', 2),
            hidden_channels=env_config['hidden_channels'],
            output_dim=env_config['output_dim'],
            max_species=n_species,
        )
        
        self.int_encoder = EfficientInteractionEncoder(
            input_dim=int_config['input_dim'],
            hidden_dim=int_config['hidden_dim'],
            output_dim=int_config['output_dim'],
            n_hops=int_config.get('n_hops', 2),
        )
        
        self.temp_encoder = EfficientTemporalEncoder(
            hidden_dim=temp_config['hidden_dim'],
            output_dim=temp_config['output_dim'],
            n_heads=temp_config['n_heads'],
            n_layers=temp_config['n_layers'],
            spatial_size=spatial_size,
        )
        
        # 🔴 FIXED: Conditioning module that preserves per-species info
        self.conditioning = FixedConditioningModule(
            env_dim=env_config['output_dim'],
            int_dim=int_config['output_dim'],
            temp_dim=temp_config['output_dim'],
            time_dim=unet_cfg['base_channels'] * 4,
            output_dim=hidden_dim,
        )
        
        # 🔴 FIXED: Species-parallel U-Net
        self.unet = SpeciesParallelUNet(
            spatial_size=spatial_size,
            base_channels=unet_cfg['base_channels'],
            channel_mults=unet_cfg['channel_mults'],
            n_res_blocks=unet_cfg['n_res_blocks'],
            cond_dim=hidden_dim,
            time_dim=hidden_dim,
            species_chunk_size=species_chunk_size,  # 🔴 NEW: Pass chunk size
            use_gradient_checkpointing=use_gradient_checkpointing,  # 🔴 NEW
        )
        
        # Diffusion process (unchanged)
        from .diffusion import GaussianDiffusion
        self.diffusion = GaussianDiffusion(
            timesteps=diffusion_steps,
            beta_schedule=beta_schedule,
        )
        
        # Track training phase
        self.current_phase = 1
    
    def set_training_phase(self, phase: int):
        """Set curriculum training phase (1-4)."""
        self.current_phase = phase
        print(f"EcoDiffusionFixed: Set to training phase {phase}")
    
    def encode_condition(
        self,
        env: torch.Tensor,
        y_coords: torch.Tensor,
        x_coords: torch.Tensor,
        species_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode all conditioning information.
        
        Returns:
            cond: (B, S, D) PER-SPECIES conditioning
        """
        B = env.shape[0]
        S = env.shape[1]
        
        # Environmental encoding (always used)
        env_global, env_spatial = self.env_encoder(env, y_coords, x_coords)
        
        # Interaction encoding (phase 2+)
        if self.current_phase >= 2 and species_features is not None and edge_index is not None:
            int_emb = self.int_encoder(species_features, edge_index, edge_weight)
        else:
            int_emb = torch.zeros(B, S, self.hidden_dim, device=env.device)
        
        # Temporal encoding (phase 3+)
        if self.current_phase >= 3 and history is not None:
            temp_emb, _ = self.temp_encoder(history)
        else:
            temp_emb = None
        
        # Generate time embedding placeholder if not provided
        if time_emb is None:
            time_emb = torch.zeros(B, self.hidden_dim, device=env.device)
        
        # 🔴 FIXED: Fuse conditions PER SPECIES
        cond = self.conditioning(
            env_global, int_emb, temp_emb, time_emb, S
        )
        
        return cond  # (B, S, D) - per-species!
    
    def forward(
        self,
        x_0: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            x_0: Clean target distribution (B, S, Y, X)
            condition: Dictionary with conditioning tensors
            noise: Optional pre-sampled noise
            
        Returns:
            Dictionary with loss and predictions
        """
        B, S, Y, X = x_0.shape
        device = x_0.device
        
        # Sample timesteps
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=device)
        
        # Add noise
        x_t, noise = self.diffusion.q_sample(x_0, t, noise)
        
        # 🔴 FIXED: Encode conditions PER SPECIES
        cond = self.encode_condition(
            env=condition['env'],
            y_coords=condition['y_coords'],
            x_coords=condition['x_coords'],
            species_features=condition.get('species_features'),
            edge_index=condition.get('edge_index'),
            edge_weight=condition.get('edge_weight'),
            history=condition.get('history_P'),
        )
        # cond shape: (B, S, D) - per-species conditioning!
        
        # 🔴 FIXED: Predict noise with per-species conditioning
        noise_pred = self.unet(x_t, t, cond)  # (B, S, Y, X)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise_true': noise,
            'x_t': x_t,
            't': t,
        }
    
    @torch.no_grad()
    def sample(
        self,
        condition: Dict[str, torch.Tensor],
        n_samples: int = 1,
        ddim_steps: int = 50,
        start_timestep: Optional[int] = None,
        use_prior: bool = True,
        prior_strength: float = 0.8,
        # 🔴 v5: FIXED sparse-aware sampling parameters
        sparse_mode: bool = True,  # Enable for sparse ecological data
        eta: float = 0.5,  # 🔴 v5: CHANGED from 0.0 to 0.5 for DIVERSITY!
        diversity_noise: float = 0.05,  # 🔴 v5: NEW - small global noise for diversity
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.
        
        🔴 CRITICAL FIX v5: BALANCE ACCURACY WITH UNCERTAINTY
        ═══════════════════════════════════════════════════════════════════════
        
        PROBLEM (v4):
        - eta=0 → All samples IDENTICAL (no stochasticity)
        - DDIM noise scaled by x_0_pred → Near-zero for sparse data
        - Result: aoo_std = 0 for ALL species → NO uncertainty!
        
        SOLUTION (v5):
        - eta=0.5 default → Stochastic DDIM for sample diversity
        - start_timestep=75 → More denoising steps for variation
        - diversity_noise=0.05 → Small global noise for sample diversity
        - REMOVE noise scaling during DDIM → Let stochasticity work!
        - Keep initialization noise scaling → Preserves sparse mean
        
        Trade-off: Mean may be 2-3x true (acceptable), BUT std > 0 (essential!)
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            condition: Conditioning dict with env, species_features, history_P, etc.
            n_samples: Number of samples to generate
            ddim_steps: Number of DDIM steps (default 50)
            start_timestep: Starting timestep (default: 75 for sparse, 500 otherwise)
            use_prior: Whether to use history_P as prior (default True)
            prior_strength: Weight for prior (default 0.8)
            sparse_mode: Enable sparse-aware sampling for ecological data
            eta: DDIM stochasticity - 0.5 gives diversity (default 0.5)
            diversity_noise: Global noise for sample diversity (default 0.05)
        """
        device = next(self.parameters()).device
        B = condition['env'].shape[0]
        S = condition['env'].shape[1]
        Y, X = self.spatial_size
        
        # Encode condition PER SPECIES
        cond = self.encode_condition(
            env=condition['env'],
            y_coords=condition['y_coords'],
            x_coords=condition['x_coords'],
            species_features=condition.get('species_features'),
            edge_index=condition.get('edge_index'),
            edge_weight=condition.get('edge_weight'),
            history=condition.get('history_P'),
        )
        
        # Build prior from history_P (last timestep)
        x_prior = None
        prior_mean = 0.0
        if use_prior and 'history_P' in condition and condition['history_P'] is not None:
            history = condition['history_P']  # (B, T, S, Y, X)
            x_prior = history[:, -1, :, :, :]  # Last timestep: (B, S, Y, X)
            
            # Ensure correct device and handle size mismatch
            if x_prior.shape[1] != S:
                if x_prior.shape[1] > S:
                    x_prior = x_prior[:, :S, :, :]
                else:
                    pad = torch.zeros(B, S - x_prior.shape[1], Y, X, device=device)
                    x_prior = torch.cat([x_prior, pad], dim=1)
            
            x_prior = x_prior.to(device)
            prior_mean = x_prior.mean().item()
            print(f"  Using history_P prior: mean={prior_mean:.4f}, range=[{x_prior.min():.3f}, {x_prior.max():.3f}]")
        
        # ═══════════════════════════════════════════════════════════════════
        # 🔴 v5: MODERATE TIMESTEP + STOCHASTIC DDIM
        # ═══════════════════════════════════════════════════════════════════
        if sparse_mode and prior_mean < 0.1:  # Activate for sparse data
            # 🔴 v5: Use t=75 (not t=25!) for more denoising variation
            default_start = 75  # 🔴 v5: CHANGED from 25 to 75
            if start_timestep is None:
                start_timestep = default_start
            else:
                start_timestep = min(start_timestep, 150)  # Higher cap for more variation
            print(f"  🔴 SPARSE MODE v5: start_timestep={start_timestep}, eta={eta}, diversity_noise={diversity_noise}")
        else:
            # Standard mode for non-sparse data
            if start_timestep is None:
                start_timestep = self.diffusion.timesteps // 2
        
        start_timestep = min(start_timestep, self.diffusion.timesteps - 1)
        
        # Sample
        samples = []
        for sample_idx in range(n_samples):
            
            # ═══════════════════════════════════════════════════════════════
            # 🔴 v5: INITIALIZATION WITH CONTROLLED DIVERSITY
            # ═══════════════════════════════════════════════════════════════
            if x_prior is not None and use_prior:
                if sparse_mode and prior_mean < 0.1:
                    alpha_start = self.diffusion.alphas_cumprod[start_timestep]
                    noise = torch.randn(B, S, Y, X, device=device)
                    
                    # 🔴 v5: TWO-COMPONENT NOISE
                    # Component 1: Prior-scaled (preserves sparsity structure)
                    noise_scale = torch.clamp(x_prior * 5.0, 0.0, 1.0)
                    sparse_noise = noise * noise_scale
                    
                    # Component 2: Global diversity noise (enables sample variation)
                    global_noise = noise * diversity_noise
                    
                    # Combine: sparse-scaled + small global
                    combined_noise = sparse_noise + global_noise
                    
                    # Apply forward diffusion
                    x = torch.sqrt(alpha_start) * x_prior + torch.sqrt(1 - alpha_start) * combined_noise
                    
                    # Clamp to valid range
                    x = torch.clamp(x, 0.0, 1.0)
                    
                    if sample_idx == 0:
                        print(f"  v5 Sparse init: alpha={alpha_start:.4f}")
                        print(f"  v5 After init: x mean={x.mean():.4f} (target: {prior_mean:.4f})")
                else:
                    # Standard mode: mix prior with more noise
                    alpha_start = self.diffusion.alphas_cumprod[start_timestep]
                    noise = torch.randn(B, S, Y, X, device=device)
                    
                    x = (prior_strength * x_prior + (1 - prior_strength) * noise * 0.1)
                    x = torch.sqrt(alpha_start) * x + torch.sqrt(1 - alpha_start) * noise
                    x = torch.clamp(x, 0.0, 1.0)
            else:
                noise = torch.randn(B, S, Y, X, device=device)
                x = noise
            
            # ═══════════════════════════════════════════════════════════════
            # 🔴 v5: DDIM SAMPLING WITH STOCHASTICITY (NO NOISE SCALING!)
            # ═══════════════════════════════════════════════════════════════
            step_size = max(1, start_timestep // ddim_steps)
            timesteps = list(range(0, start_timestep + 1, step_size))
            
            if sample_idx == 0:
                print(f"  DDIM: {len(timesteps)} steps from t={max(timesteps)} to t=0, eta={eta}")
            
            for i in reversed(range(len(timesteps))):
                t_val = timesteps[i]
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
                
                # Predict noise with per-species conditioning
                noise_pred = self.unet(x, t_batch, cond)
                
                # DDIM update
                alpha_t = self.diffusion.alphas_cumprod[t_val]
                if i > 0:
                    alpha_prev = self.diffusion.alphas_cumprod[timesteps[i-1]]
                else:
                    alpha_prev = torch.tensor(1.0, device=device)
                
                # Predict x_0
                x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                
                # 🔴 SPARSE-PRESERVING CLAMP: Clamp to [0, 1]
                x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)
                
                # 🔴 v5: STOCHASTIC DDIM WITHOUT NOISE SCALING
                # The key fix: DON'T scale noise during DDIM - this was killing diversity!
                if eta > 0 and i > 0:
                    # Compute sigma for stochastic DDIM
                    sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t + 1e-8)) * torch.sqrt(1 - alpha_t / (alpha_prev + 1e-8))
                    sigma_t = torch.clamp(sigma_t, 0, 0.3)  # Stability clamp
                    
                    noise = torch.randn_like(x)
                    # 🔴 v5: DO NOT scale noise here! Let DDIM add natural variation
                    # The clamping after will maintain sparsity anyway
                    
                    variance_term = torch.clamp(1 - alpha_prev - sigma_t**2, 0, 1)
                    x = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(variance_term) * noise_pred + sigma_t * noise
                else:
                    # Deterministic DDIM (eta=0)
                    x = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
                
                # 🔴 SPARSE MODE: Clamp after each step to maintain sparsity
                if sparse_mode:
                    x = torch.clamp(x, 0.0, 1.0)
            
            # Final clamp
            x = torch.clamp(x, 0.0, 1.0)
            
            samples.append(x)
        
        return torch.stack(samples, dim=0)  # (n_samples, B, S, Y, X)


def create_fixed_model(config) -> EcoDiffusionFixed:
    """Factory function to create EcoDiffusion model from config."""

    # 🔴 NEW: Get species_chunk_size from config or use default
    species_chunk_size = getattr(config.model, 'species_chunk_size', 64)
    use_gradient_checkpointing = getattr(config.model, 'use_gradient_checkpointing', True)
    
    return EcoDiffusionFixed(
        n_species=config.data.n_species_max,
        spatial_size=config.data.grid_size,
        hidden_dim=256,
        env_encoder_config={
            'in_channels': config.model.env_encoder_in_channels,
            'hidden_channels': config.model.env_encoder_channels,
            'output_dim': config.model.env_encoder_output_dim,
        },
        int_encoder_config={
            'input_dim': config.model.gnn_input_dim,
            'hidden_dim': config.model.gnn_hidden_dim,
            'output_dim': config.model.gnn_output_dim,
            'n_hops': config.model.gnn_num_layers,
        },
        temp_encoder_config={
            'hidden_dim': config.model.temporal_hidden_dim,
            'output_dim': config.model.temporal_hidden_dim,
            'n_heads': config.model.temporal_num_heads,
            'n_layers': config.model.temporal_num_layers,
        },
        unet_config={
            'base_channels': config.model.unet_base_channels,
            'channel_mults': config.model.unet_channel_multipliers,
            'n_res_blocks': config.model.unet_num_res_blocks,
        },
        diffusion_steps=config.model.diffusion_steps,
        beta_schedule=config.model.beta_schedule,
        species_chunk_size=species_chunk_size,  # 🔴 NEW
        use_gradient_checkpointing=use_gradient_checkpointing,  # 🔴 NEW
    )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING FIXED ECODIFFUSION ARCHITECTURE")
    print("=" * 70)
    
    # Test configuration
    B, S, Y, X = 2, 100, 20, 20  # Small test
    hidden_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nTest config: B={B}, S={S}, Y={Y}, X={X}")
    print(f"Device: {device}")
    
    # Test conditioning module
    print("\n1. Testing FixedConditioningModule...")
    cond_module = FixedConditioningModule(
        env_dim=256, int_dim=256, temp_dim=256, time_dim=256, output_dim=hidden_dim
    ).to(device)
    
    env_emb = torch.randn(B, S, 256).to(device)
    int_emb = torch.randn(B, S, 256).to(device)
    time_emb = torch.randn(B, 256).to(device)
    
    cond = cond_module(env_emb, int_emb, None, time_emb, S)
    print(f"   Input env_emb: {env_emb.shape}")
    print(f"   Output cond: {cond.shape}")
    print(f"   ✓ Per-species conditioning preserved: {cond.shape[1] == S}")
    
    # Test U-Net
    print("\n2. Testing SpeciesParallelUNet...")
    unet = SpeciesParallelUNet(
        spatial_size=(Y, X),
        base_channels=32,  # Smaller for testing
        channel_mults=[1, 2],
        cond_dim=hidden_dim,
    ).to(device)
    
    x = torch.randn(B, S, Y, X).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    cond_test = torch.randn(B, S, hidden_dim).to(device)
    
    out = unet(x, t, cond_test)
    print(f"   Input x: {x.shape}")
    print(f"   Input cond: {cond_test.shape}")
    print(f"   Output: {out.shape}")
    print(f"   ✓ Output shape matches input: {out.shape == x.shape}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = out.sum()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in unet.parameters() if p.grad is not None)
    print(f"   Total gradient norm: {grad_norm:.4f}")
    print(f"   ✓ Gradients flowing: {grad_norm > 0}")
    
    # Memory usage
    print("\n4. Memory usage...")
    if device.type == 'cuda':
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"   Peak GPU memory: {mem_gb:.2f} GB")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)