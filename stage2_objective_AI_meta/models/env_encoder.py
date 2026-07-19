"""
=============================================================================
ENVIRONMENTAL ENCODER: CNN-based Environment Feature Extractor
=============================================================================
This module implements a Convolutional Neural Network to encode the 
environmental growth rate field (ENV_r_field) and spatial coordinates.

REASONING FOR CNN ARCHITECTURE:
1. ENV_r_field has spatial autocorrelation (length_scale=2.5 grid cells)
   - Local receptive fields match this spatial scale
   - Deeper layers capture longer-range spatial patterns

2. Environmental filtering is a local process in ecology
   - Species respond to local environmental conditions
   - CNN's locality assumption matches ecological reality

3. Translation equivariance is appropriate
   - Same environmental pattern should produce same response regardless
     of location (the coordinates provide position information separately)

4. Multi-scale feature extraction
   - Different levels capture different spatial scales
   - Species may respond to both local and regional environment
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """
    Basic convolutional block with BatchNorm and activation.
    
    Architecture: Conv2d → BatchNorm → Activation → (Optional Dropout)
    
    REASONING:
    - BatchNorm: Stabilizes training, allows higher learning rates
    - SiLU activation: Smooth, performs well in diffusion models (used in DDPM++)
    - Residual connection: Helps with gradient flow in deep networks
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # SiLU (Swish) activation
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Residual projection if dimensions don't match
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for environment features.
    
    Learns to weight different spatial locations based on their importance
    for predicting species distributions.
    
    REASONING:
    - Some locations may be more informative (e.g., habitat edges)
    - Attention allows dynamic weighting based on input
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, H, W = x.shape
        
        # Compute attention
        q = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # (B, HW, C/8)
        k = self.key(x).view(batch, -1, H * W)  # (B, C/8, HW)
        
        attention = torch.bmm(q, k)  # (B, HW, HW)
        attention = F.softmax(attention / (C // 8) ** 0.5, dim=-1)
        
        v = self.value(x).view(batch, -1, H * W)  # (B, C, HW)
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch, C, H, W)
        
        return self.gamma * out + x


class EnvironmentalEncoder(nn.Module):
    """
    CNN-based encoder for environmental data.
    
    Takes ENV_r_field (S, Y, X) and spatial coordinates, produces feature
    embeddings that capture environmental patterns relevant for species
    distribution prediction.
    
    Architecture:
    1. Per-species encoding: Process each species' environment independently
    2. Multi-scale feature extraction: Progressively larger receptive fields
    3. Spatial attention: Learn to weight important locations
    4. Global + local features: Combine spatial-specific and aggregated features
    
    Input:
        env: Environmental field (B, S, Y, X)
        coords: Spatial coordinates (B, 2, Y, X) for y and x
    
    Output:
        features: (B, S, output_dim) per-species environmental embeddings
        spatial_features: (B, S, output_dim, Y, X) spatial feature maps
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # ENV_r_field + y_coord + x_coord
        hidden_channels: list = [64, 128, 256],
        output_dim: int = 256,
        kernel_size: int = 3,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Build encoder layers
        layers = []
        channels = [in_channels] + hidden_channels
        
        for i in range(len(channels) - 1):
            layers.append(ConvBlock(
                channels[i], channels[i + 1],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                dropout=dropout if i > 0 else 0,
            ))
            
            # Add attention at intermediate layer
            if use_attention and i == len(channels) - 2:
                layers.append(SpatialAttention(channels[i + 1]))
        
        self.encoder = nn.Sequential(*layers)
        
        # Output projection for spatial features
        self.spatial_proj = nn.Conv2d(hidden_channels[-1], output_dim, kernel_size=1)
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_channels[-1], output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(
        self,
        env: torch.Tensor,
        y_coords: torch.Tensor,
        x_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            env: (B, S, Y, X) environmental field
            y_coords: (B, Y, X) normalized y coordinates
            x_coords: (B, X, Y) normalized x coordinates
            
        Returns:
            global_features: (B, S, output_dim) per-species embeddings
            spatial_features: (B, S, output_dim, Y, X) spatial feature maps
        """
        B, S, Y, X = env.shape
        
        # Expand coordinates to match species dimension
        # y_coords, x_coords: (B, Y, X) → (B, S, Y, X)
        y_coords_expanded = y_coords.unsqueeze(1).expand(-1, S, -1, -1)
        x_coords_expanded = x_coords.unsqueeze(1).expand(-1, S, -1, -1)
        
        # Concatenate environment with coordinates
        # Input: (B, S, 3, Y, X) where 3 = env + y + x
        x = torch.stack([env, y_coords_expanded, x_coords_expanded], dim=2)
        
        # Reshape to process all species together: (B*S, 3, Y, X)
        x = x.view(B * S, self.in_channels, Y, X)
        
        # Encode
        features = self.encoder(x)  # (B*S, C_hidden, Y, X)
        
        # Spatial features
        spatial_features = self.spatial_proj(features)  # (B*S, output_dim, Y, X)
        spatial_features = spatial_features.view(B, S, self.output_dim, Y, X)
        
        # Global features
        global_pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # (B*S, C_hidden)
        global_features = self.global_proj(global_pooled)  # (B*S, output_dim)
        global_features = global_features.view(B, S, self.output_dim)
        
        return global_features, spatial_features


class SpeciesEnvironmentEncoder(nn.Module):
    """
    Alternative encoder that processes environment for all species jointly.
    
    This version is more memory efficient for large species counts by using
    shared convolutions followed by species-specific projections.
    
    REASONING:
    - Original approach: (B*S, C, Y, X) can be very large (4*3614*256*20*20)
    - This approach: Process (B, C, Y, X) then broadcast to species
    - Trade-off: Less species-specific environmental learning, but faster
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # y_coord + x_coord (env handled per-species)
        hidden_channels: list = [64, 128, 256],
        output_dim: int = 256,
        max_species: int = 4000,
    ):
        super().__init__()
        
        # Shared spatial encoder (for coordinates)
        layers = []
        channels = [in_channels] + hidden_channels
        for i in range(len(channels) - 1):
            layers.append(ConvBlock(channels[i], channels[i + 1]))
        self.spatial_encoder = nn.Sequential(*layers)
        
        # Per-species environment integration
        # Instead of full (S, hidden) projection, use low-rank factorization
        self.env_proj = nn.Sequential(
            nn.Linear(1, 64),  # From ENV_r_field value
            nn.SiLU(),
            nn.Linear(64, output_dim),
        )
        
        # Combine spatial and environment features
        self.combiner = nn.Sequential(
            nn.Linear(hidden_channels[-1] + output_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        
        self.output_dim = output_dim
    
    def forward(
        self,
        env: torch.Tensor,
        y_coords: torch.Tensor,
        x_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            env: (B, S, Y, X) environmental field
            y_coords: (B, Y, X) normalized y coordinates
            x_coords: (B, Y, X) normalized x coordinates
            
        Returns:
            global_features: (B, S, output_dim)
            spatial_features: (B, S, output_dim, Y, X)
        """
        B, S, Y, X = env.shape
        
        # Encode spatial structure (shared across species)
        coords = torch.stack([y_coords, x_coords], dim=1)  # (B, 2, Y, X)
        spatial_feats = self.spatial_encoder(coords)  # (B, C_hidden, Y, X)
        
        # Project environment per species
        # env: (B, S, Y, X) → (B, S, Y, X, 1) → (B, S, Y, X, output_dim)
        env_feats = self.env_proj(env.unsqueeze(-1))  # (B, S, Y, X, output_dim)
        env_feats = env_feats.permute(0, 1, 4, 2, 3)  # (B, S, output_dim, Y, X)
        
        # Combine: broadcast spatial features to all species
        # spatial_feats: (B, C_hidden, Y, X) → (B, 1, C_hidden, Y, X)
        spatial_feats_expanded = spatial_feats.unsqueeze(1)  # (B, 1, C_hidden, Y, X)
        
        # Concatenate and combine
        # Reshape for linear layer
        combined = torch.cat([
            spatial_feats_expanded.expand(-1, S, -1, -1, -1),  # (B, S, C_hidden, Y, X)
            env_feats,  # (B, S, output_dim, Y, X)
        ], dim=2)  # (B, S, C_hidden + output_dim, Y, X)
        
        combined = combined.permute(0, 1, 3, 4, 2)  # (B, S, Y, X, C_hidden + output_dim)
        spatial_features = self.combiner(combined)  # (B, S, Y, X, output_dim)
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3)  # (B, S, output_dim, Y, X)
        
        # Global features via pooling
        global_features = spatial_features.mean(dim=(-2, -1))  # (B, S, output_dim)
        
        return global_features, spatial_features


if __name__ == "__main__":
    # Test the environmental encoder
    print("=" * 60)
    print("ENVIRONMENTAL ENCODER TEST")
    print("=" * 60)
    
    # Test parameters
    B, S, Y, X = 2, 100, 20, 20
    output_dim = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create mock inputs
    env = torch.randn(B, S, Y, X).to(device)
    y_coords = torch.linspace(0, 1, Y).view(1, Y, 1).expand(B, -1, X).to(device)
    x_coords = torch.linspace(0, 1, X).view(1, 1, X).expand(B, Y, -1).to(device)
    
    # Test standard encoder
    print("\n1. Testing EnvironmentalEncoder...")
    encoder = EnvironmentalEncoder(
        in_channels=2,
        hidden_channels=[64, 128, 256],
        output_dim=output_dim,
    ).to(device)
    
    global_feats, spatial_feats = encoder(env, y_coords, x_coords)
    print(f"   Input env shape: {env.shape}")
    print(f"   Global features shape: {global_feats.shape}")
    print(f"   Spatial features shape: {spatial_feats.shape}")
    
    # Parameter count
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Test memory-efficient encoder
    print("\n2. Testing SpeciesEnvironmentEncoder...")
    encoder_efficient = SpeciesEnvironmentEncoder(
        in_channels=2,
        hidden_channels=[64, 128, 256],
        output_dim=output_dim,
    ).to(device)
    
    global_feats2, spatial_feats2 = encoder_efficient(env, y_coords, x_coords)
    print(f"   Global features shape: {global_feats2.shape}")
    print(f"   Spatial features shape: {spatial_feats2.shape}")
    
    n_params2 = sum(p.numel() for p in encoder_efficient.parameters())
    print(f"   Parameters: {n_params2:,}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = global_feats.sum() + spatial_feats.sum()
    loss.backward()
    print("   ✓ Gradients computed successfully")
    
    print("\n" + "=" * 60)
    print("✓ Environmental encoder tests passed!")
    print("=" * 60)
