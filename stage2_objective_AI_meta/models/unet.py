"""
=============================================================================
DENOISING U-NET: Species-Spatial Dual Attention Architecture
=============================================================================
This module implements the core denoising network for the diffusion model,
featuring a U-Net architecture with dual attention mechanisms.

REASONING FOR U-NET ARCHITECTURE:

1. Skip Connections Preserve Spatial Detail
   - Species distributions have fine-grained spatial structure
   - U-Net skip connections maintain high-frequency patterns
   - Critical for preserving range boundary precision

2. Multi-Scale Processing
   - Local patterns: Individual patch occupancy
   - Regional patterns: Dispersal corridors, barriers
   - Global patterns: Range-wide distribution shapes
   - U-Net naturally captures all scales

3. Dual Attention Mechanism
   - Spatial attention: Learn which locations matter for each species
   - Species attention: Learn which species co-occur/exclude
   - Both are essential for ecological prediction

4. Conditioning Integration
   - Time embedding: Which diffusion step we're at
   - Environment: Local habitat suitability
   - Interactions: Competition/facilitation context
   - History: Temporal dynamics context

ARCHITECTURE OVERVIEW:
Input: (B, S, Y, X) noisy species distributions
Conditioning: time_emb, env_emb, interaction_emb, temporal_emb
Output: (B, S, Y, X) predicted noise (or clean distribution)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time step embedding as used in DDPM.
    
    REASONING:
    - Encodes diffusion timestep as continuous embedding
    - Sinusoidal basis allows extrapolation
    - Standard approach in diffusion models
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP to process embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.
        
        Args:
            t: (B,) integer timesteps
            
        Returns:
            (B, dim) time embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return self.mlp(embedding)


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding injection.
    
    Architecture:
    - GroupNorm → SiLU → Conv → GroupNorm → SiLU → Dropout → Conv
    - Time embedding added after first GroupNorm
    - Skip connection (with projection if channels change)
    
    REASONING:
    - GroupNorm: Works better than BatchNorm for small batches
    - SiLU: Smooth activation, standard in modern diffusion models
    - Time injection: Allows time-dependent processing
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) input features
            time_emb: (B, time_dim) time embedding
            
        Returns:
            (B, C_out, H, W) output features
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class SpatialAttentionBlock(nn.Module):
    """
    Self-attention over spatial dimensions.
    
    Allows each spatial location to attend to all others,
    capturing long-range spatial dependencies.
    
    REASONING:
    - Convolutions have limited receptive field
    - Attention captures global spatial patterns
    - Important for dispersal and range continuity
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            (B, C, H, W) attended features
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, heads, head_dim, HW)
        
        # Transpose for attention: (B, heads, HW, head_dim)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        return x + self.proj(out)


class SpeciesAttentionBlock(nn.Module):
    """
    Self-attention over species dimension.
    
    Allows each species to attend to all others,
    capturing co-occurrence and competition patterns.
    
    REASONING:
    - Species distributions are not independent
    - Competition creates mutual exclusion patterns
    - Facilitation creates co-occurrence patterns
    - Attention learns these relationships from data
    """
    
    def __init__(
        self,
        spatial_dim: int,  # H * W
        num_heads: int = 4,
        max_species: int = 4000,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.spatial_dim = spatial_dim
        self.head_dim = spatial_dim // num_heads
        
        # Project spatial features for attention
        self.q_proj = nn.Linear(spatial_dim, spatial_dim)
        self.k_proj = nn.Linear(spatial_dim, spatial_dim)
        self.v_proj = nn.Linear(spatial_dim, spatial_dim)
        self.out_proj = nn.Linear(spatial_dim, spatial_dim)
        
        self.norm = nn.LayerNorm(spatial_dim)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        species_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, S, H, W) species distributions
            species_mask: (B, S) valid species mask
            
        Returns:
            (B, S, H, W) attended distributions
        """
        B, S, H, W = x.shape
        
        # Flatten spatial: (B, S, H*W)
        x_flat = x.view(B, S, -1)
        
        # Normalize
        x_norm = self.norm(x_flat)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(B, S, self.num_heads, self.head_dim)
        
        # Transpose: (B, heads, S, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Attention: (B, heads, S, S)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Mask invalid species
        if species_mask is not None:
            mask = species_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # (B, heads, S, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, S, -1)  # (B, S, H*W)
        
        out = self.out_proj(out)
        
        # Residual + reshape
        out = x_flat + out
        return out.view(B, S, H, W)


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    
    Contains:
    - Multiple residual blocks
    - Optional spatial attention
    - Downsampling (strided conv)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout=dropout,
            ))
            
            if use_attention:
                self.attn_blocks.append(SpatialAttentionBlock(out_channels, num_heads))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features
            time_emb: Time embedding
            
        Returns:
            (downsampled_features, skip_connection)
        """
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, time_emb)
            x = attn_block(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder.
    
    Contains:
    - Upsampling (nearest neighbor + conv)
    - Skip connection concatenation
    - Multiple residual blocks
    - Optional spatial attention
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(
                (in_channels + skip_channels) if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout=dropout,
            ))
            
            if use_attention:
                self.attn_blocks.append(SpatialAttentionBlock(out_channels, num_heads))
            else:
                self.attn_blocks.append(nn.Identity())
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features from lower level
            skip: Skip connection from encoder
            time_emb: Time embedding
            
        Returns:
            Upsampled features
        """
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        
        x = torch.cat([x, skip], dim=1)
        
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, time_emb)
            x = attn_block(x)
        
        return x


class MiddleBlock(nn.Module):
    """
    Middle block of U-Net (bottleneck).
    
    Contains residual blocks with attention at the lowest resolution.
    """
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.res1 = ResidualBlock(channels, channels, time_emb_dim, dropout=dropout)
        self.attn = SpatialAttentionBlock(channels, num_heads)
        self.res2 = ResidualBlock(channels, channels, time_emb_dim, dropout=dropout)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, time_emb)
        x = self.attn(x)
        x = self.res2(x, time_emb)
        return x


class ConditioningProjection(nn.Module):
    """
    Projects conditioning embeddings to U-Net channel dimension.
    
    Takes environment, interaction, and temporal embeddings and
    combines them into a single conditioning vector.
    """
    
    def __init__(
        self,
        env_dim: int = 256,
        interaction_dim: int = 256,
        temporal_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Individual projections
        self.env_proj = nn.Linear(env_dim, output_dim)
        self.interaction_proj = nn.Linear(interaction_dim, output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, output_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        env_emb: Optional[torch.Tensor] = None,
        interaction_emb: Optional[torch.Tensor] = None,
        temporal_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combine conditioning embeddings.
        
        Args:
            env_emb: (B, S, env_dim) environment embedding
            interaction_emb: (B, S, interaction_dim) interaction embedding
            temporal_emb: (B, S, temporal_dim) temporal embedding
            
        Returns:
            (B, S, output_dim) combined conditioning
        """
        embeddings = []
        
        if env_emb is not None:
            embeddings.append(self.env_proj(env_emb))
        else:
            # Use zero embedding if not provided
            B, S = interaction_emb.shape[:2] if interaction_emb is not None else temporal_emb.shape[:2]
            embeddings.append(torch.zeros(B, S, self.env_proj.out_features, device=env_emb.device if env_emb is not None else 'cpu'))
        
        if interaction_emb is not None:
            embeddings.append(self.interaction_proj(interaction_emb))
        else:
            B, S = embeddings[0].shape[:2]
            device = embeddings[0].device
            embeddings.append(torch.zeros(B, S, self.interaction_proj.out_features, device=device))
        
        if temporal_emb is not None:
            embeddings.append(self.temporal_proj(temporal_emb))
        else:
            B, S = embeddings[0].shape[:2]
            device = embeddings[0].device
            embeddings.append(torch.zeros(B, S, self.temporal_proj.out_features, device=device))
        
        combined = torch.cat(embeddings, dim=-1)
        fused = self.fusion(combined)
        
        return self.norm(fused)


class EcoUNet(nn.Module):
    """
    U-Net for species distribution denoising.
    
    This is the core denoising network that predicts noise (or clean data)
    given noisy species distributions and conditioning information.
    
    ARCHITECTURE:
    1. Input projection: (B, S, Y, X) → (B, C, Y, X) via 1x1 conv
    2. Encoder: Downsampling path with residual blocks + attention
    3. Middle: Bottleneck with attention
    4. Decoder: Upsampling path with skip connections
    5. Output projection: (B, C, Y, X) → (B, S, Y, X)
    
    Conditioning is injected via:
    - Time embedding: Added to residual blocks
    - Condition embedding: Modulates features via FiLM
    
    Input:
        x: (B, S, Y, X) noisy species distributions
        t: (B,) diffusion timesteps
        condition: Dict with env_emb, interaction_emb, temporal_emb
        
    Output:
        (B, S, Y, X) predicted noise
    """
    
    def __init__(
        self,
        in_channels: int = 3614,  # Number of species
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [10, 5],  # Apply attention at these resolutions
        time_emb_dim: int = 256,
        condition_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_species_attention: bool = True,
        spatial_size: Tuple[int, int] = (20, 20),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.use_species_attention = use_species_attention
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        
        # Condition projection
        self.condition_proj = ConditioningProjection(
            env_dim=condition_dim,
            interaction_dim=condition_dim,
            temporal_dim=condition_dim,
            output_dim=time_emb_dim,
            dropout=dropout,
        )
        
        # Input projection: Species → Channels
        # Use grouped convolution for efficiency
        self.in_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Build channel sizes
        channels = [base_channels * m for m in channel_multipliers]
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        current_res = spatial_size[0]
        
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            use_attn = current_res in attention_resolutions
            self.encoder.append(DownBlock(
                in_ch, out_ch, time_emb_dim,
                num_res_blocks=num_res_blocks,
                use_attention=use_attn,
                num_heads=num_heads,
                dropout=dropout,
            ))
            in_ch = out_ch
            current_res = current_res // 2
        
        # Middle (bottleneck)
        self.middle = MiddleBlock(
            channels[-1], time_emb_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        
        for i, out_ch in enumerate(reversed(channels[:-1])):
            in_ch = channels[-(i + 1)]
            skip_ch = channels[-(i + 2)]
            current_res = current_res * 2
            use_attn = current_res in attention_resolutions
            
            self.decoder.append(UpBlock(
                in_ch, out_ch, skip_ch, time_emb_dim,
                num_res_blocks=num_res_blocks,
                use_attention=use_attn,
                num_heads=num_heads,
                dropout=dropout,
            ))
        
        # Final upsampling to original resolution
        self.final_up = UpBlock(
            channels[0], base_channels, base_channels, time_emb_dim,
            num_res_blocks=num_res_blocks,
            use_attention=spatial_size[0] in attention_resolutions,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Output projection: Channels → Species
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_proj = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        
        # Optional species attention
        if use_species_attention:
            self.species_attn = SpeciesAttentionBlock(
                spatial_dim=spatial_size[0] * spatial_size[1],
                num_heads=num_heads,
            )
        
        # Zero-initialize output for stable training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        env_emb: Optional[torch.Tensor] = None,
        interaction_emb: Optional[torch.Tensor] = None,
        temporal_emb: Optional[torch.Tensor] = None,
        species_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, S, Y, X) noisy species distributions
            t: (B,) diffusion timesteps
            env_emb: (B, S, D) environment embeddings
            interaction_emb: (B, S, D) interaction embeddings
            temporal_emb: (B, S, D) temporal embeddings
            species_mask: (B, S) valid species mask
            
        Returns:
            (B, S, Y, X) predicted noise
        """
        B, S, Y, X = x.shape
        
        # Time embedding
        time_emb = self.time_embed(t)  # (B, time_dim)
        
        # Condition embedding (per-species)
        if any(e is not None for e in [env_emb, interaction_emb, temporal_emb]):
            cond_emb = self.condition_proj(env_emb, interaction_emb, temporal_emb)  # (B, S, time_dim)
            # Pool across species for global conditioning
            if species_mask is not None:
                cond_emb = (cond_emb * species_mask.unsqueeze(-1).float()).sum(dim=1) / (species_mask.sum(dim=1, keepdim=True).float() + 1e-10)
            else:
                cond_emb = cond_emb.mean(dim=1)  # (B, time_dim)
            
            # Add to time embedding
            time_emb = time_emb + cond_emb
        
        # Input projection
        h = self.in_proj(x)  # (B, base_ch, Y, X)
        
        # Encoder
        skips = [h]
        for block in self.encoder:
            h, skip = block(h, time_emb)
            skips.append(skip)
        
        # Middle
        h = self.middle(h, time_emb)
        
        # Decoder
        for block in self.decoder:
            skip = skips.pop()
            h = block(h, skip, time_emb)
        
        # Final upsampling
        skip = skips.pop()
        h = self.final_up(h, skip, time_emb)
        
        # Output projection
        h = self.out_norm(h)
        h = F.silu(h)
        out = self.out_proj(h)  # (B, S, Y, X)
        
        # Optional species attention
        if self.use_species_attention:
            out = self.species_attn(out, species_mask)
        
        return out


class EfficientEcoUNet(nn.Module):
    """
    Memory-efficient U-Net for very large species counts.
    
    Instead of processing all species as channels (which requires
    huge memory for S=3614), processes species in chunks and uses
    a shared backbone.
    
    REASONING:
    - Full (B, 3614, 20, 20) tensor is ~1.4GB in float32
    - With channels multiplied by 8 in U-Net, this explodes
    - Chunked processing keeps memory manageable
    - Shared backbone captures common spatial patterns
    """
    
    def __init__(
        self,
        max_species: int = 4000,
        chunk_size: int = 256,  # Process species in chunks
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        time_emb_dim: int = 256,
        condition_dim: int = 256,
        spatial_size: Tuple[int, int] = (20, 20),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.max_species = max_species
        self.chunk_size = chunk_size
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        
        # Species embedding (learned embedding per species)
        self.species_embed = nn.Embedding(max_species, condition_dim)
        
        # Shared spatial backbone
        # Input: (B, chunk_size + condition_channels, Y, X)
        self.in_proj = nn.Conv2d(chunk_size + condition_dim, base_channels, kernel_size=3, padding=1)
        
        # Simple encoder-decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        channels = [base_channels * m for m in channel_multipliers]
        
        in_ch = base_channels
        for out_ch in channels:
            self.encoder.append(nn.Sequential(
                ResidualBlock(in_ch, out_ch, time_emb_dim, dropout=dropout),
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
            ))
            in_ch = out_ch
        
        self.middle = ResidualBlock(channels[-1], channels[-1], time_emb_dim, dropout=dropout)
        
        for i, out_ch in enumerate(reversed(channels[:-1])):
            in_ch = channels[-(i + 1)]
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ResidualBlock(in_ch, out_ch, time_emb_dim, dropout=dropout),
            ))
        
        # Final layers
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(channels[0], base_channels, time_emb_dim, dropout=dropout),
        )
        
        self.out_proj = nn.Conv2d(base_channels, chunk_size, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        env_emb: Optional[torch.Tensor] = None,
        interaction_emb: Optional[torch.Tensor] = None,
        temporal_emb: Optional[torch.Tensor] = None,
        species_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with chunked species processing.
        
        Args:
            x: (B, S, Y, X) noisy species distributions
            t: (B,) diffusion timesteps
            
        Returns:
            (B, S, Y, X) predicted noise
        """
        B, S, Y, X = x.shape
        device = x.device
        
        # Time embedding
        time_emb = self.time_embed(t)  # (B, time_dim)
        
        # Process species in chunks
        outputs = []
        
        for start_idx in range(0, S, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, S)
            chunk_size = end_idx - start_idx
            
            # Get chunk
            x_chunk = x[:, start_idx:end_idx]  # (B, chunk, Y, X)
            
            # Pad if necessary
            if chunk_size < self.chunk_size:
                pad_size = self.chunk_size - chunk_size
                x_chunk = F.pad(x_chunk, (0, 0, 0, 0, 0, pad_size))
            
            # Get species conditioning
            species_ids = torch.arange(start_idx, start_idx + self.chunk_size, device=device)
            species_ids = species_ids.clamp(max=self.max_species - 1)
            species_cond = self.species_embed(species_ids)  # (chunk, cond_dim)
            
            # Broadcast to spatial dimensions
            species_cond = species_cond.view(1, -1, 1, 1).expand(B, -1, Y, X)
            
            # Concatenate
            h = torch.cat([x_chunk, species_cond], dim=1)
            
            # Forward through network
            h = self.in_proj(h)
            
            skips = []
            for enc in self.encoder:
                h = enc[0](h, time_emb)  # ResBlock
                skips.append(h)
                h = enc[1](h)  # Downsample
            
            h = self.middle(h, time_emb)
            
            for dec, skip in zip(self.decoder, reversed(skips[:-1])):
                h = dec[0](h)  # Upsample
                # Size matching
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
                h = h + skip
                h = dec[1](h, time_emb)  # ResBlock
            
            h = self.final[0](h)  # Upsample
            h = self.final[1](h, time_emb)  # ResBlock
            
            out_chunk = self.out_proj(h)  # (B, chunk_size, Y, X)
            
            # Remove padding
            out_chunk = out_chunk[:, :chunk_size]
            outputs.append(out_chunk)
        
        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    print("=" * 60)
    print("U-NET DENOISING MODEL TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test parameters
    B = 2
    S = 100  # Smaller for testing
    Y, X = 20, 20
    
    # Test standard U-Net
    print("\n1. Testing EcoUNet...")
    model = EcoUNet(
        in_channels=S,
        base_channels=32,
        channel_multipliers=[1, 2, 4],
        time_emb_dim=128,
        condition_dim=128,
        spatial_size=(Y, X),
        use_species_attention=True,
    ).to(device)
    
    x = torch.randn(B, S, Y, X).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    env_emb = torch.randn(B, S, 128).to(device)
    interaction_emb = torch.randn(B, S, 128).to(device)
    
    out = model(x, t, env_emb=env_emb, interaction_emb=interaction_emb)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print("   ✓ Gradients computed successfully")
    
    # Test efficient U-Net
    print("\n2. Testing EfficientEcoUNet...")
    model_eff = EfficientEcoUNet(
        max_species=4000,
        chunk_size=64,
        base_channels=32,
        channel_multipliers=[1, 2, 4],
        time_emb_dim=128,
        spatial_size=(Y, X),
    ).to(device)
    
    out_eff = model_eff(x, t)
    print(f"   Output shape: {out_eff.shape}")
    
    n_params_eff = sum(p.numel() for p in model_eff.parameters())
    print(f"   Parameters: {n_params_eff:,}")
    
    print("\n" + "=" * 60)
    print("✓ U-Net denoising model tests passed!")
    print("=" * 60)
