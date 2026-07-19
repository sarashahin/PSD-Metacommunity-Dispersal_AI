"""
Encoder Modules for EcoDiffusion
=================================

Three specialized encoders that process different aspects of ecological data:

1. EnvironmentalEncoder (CNN):
   - Processes ENV_r_field spatial patterns
   - Captures niche filtering effects
   - Uses CNN because spatial autocorrelation (length_scale=2.5)

2. InteractionEncoder (GraphSAGE with Attention):
   - Processes species interaction network
   - Captures competitive exclusion effects
   - Uses GNN because sparse interaction matrix (16 edges/node)

3. TemporalEncoder (Transformer):
   - Processes P_t time series
   - Captures colonization-extinction dynamics
   - Uses Transformer for long-range dependencies

Design Philosophy:
- Each encoder outputs fixed-size embeddings (256-dim by default)
- Embeddings are concatenated to condition the diffusion model
- Memory-efficient implementations for 3614 species
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch_geometric.nn import SAGEConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data as GraphData, Batch as GraphBatch


# ============================================================================
# Environmental Encoder (CNN)
# ============================================================================

class EnvironmentalEncoder(nn.Module):
    """
    CNN encoder for environmental field (ENV_r_field).
    
    Architecture:
    - Input: ENV_r_field stacked with spatial coordinates → (3, H, W) per species
    - Output: (S, embed_dim) environmental embeddings per species
    
    Design Rationale:
    - CNN matches the spatial autocorrelation in ENV_r_field (length_scale=2.5)
    - Progressive downsampling captures multi-scale environmental patterns
    - Species-wise processing allows parallelization
    
    The environmental field represents the local growth rate variation
    caused by Gaussian Random Field perturbations. High ENV values indicate
    favorable habitat, low values indicate unsuitable conditions.
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # ENV_r_field + x_coord + y_coord
        hidden_channels: List[int] = [32, 64, 128, 256],
        output_dim: int = 256,
        kernel_size: int = 3,
        grid_size: Tuple[int, int] = (20, 20)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        # Build convolutional layers
        layers = []
        curr_channels = in_channels
        
        for i, out_ch in enumerate(hidden_channels):
            layers.extend([
                nn.Conv2d(curr_channels, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),  # GELU works better than ReLU for transformers/diffusion
            ])
            # Downsample every other layer (if grid allows)
            if i < len(hidden_channels) - 1 and i % 2 == 1:
                layers.append(nn.MaxPool2d(2))
            curr_channels = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Compute output spatial size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *grid_size)
            conv_out = self.conv_layers(dummy)
            self.conv_output_size = conv_out.shape[1] * conv_out.shape[2] * conv_out.shape[3]
        
        # Final projection to embedding dimension
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(
        self, 
        env_field: torch.Tensor,  # (B, S, H, W) or (B, S, 1, H, W) 
        spatial_coords: torch.Tensor  # (B, 2, H, W)
    ) -> torch.Tensor:
        """
        Encode environmental field for all species.
        
        Args:
            env_field: Environmental growth rate field, shape (B, S, H, W)
            spatial_coords: Normalized x,y coordinates, shape (B, 2, H, W)
            
        Returns:
            Environmental embeddings, shape (B, S, embed_dim)
        """
        B, S, H, W = env_field.shape
        
        # Expand spatial coords for all species: (B, 2, H, W) → (B, S, 2, H, W)
        coords_expanded = spatial_coords.unsqueeze(1).expand(B, S, 2, H, W)
        
        # Stack env_field with coordinates: (B, S, 3, H, W)
        env_with_coords = torch.cat([
            env_field.unsqueeze(2),  # (B, S, 1, H, W)
            coords_expanded
        ], dim=2)
        
        # Reshape for batch processing: (B*S, 3, H, W)
        x = env_with_coords.view(B * S, self.in_channels, H, W)
        
        # Apply convolutions
        x = self.conv_layers(x)  # (B*S, C, H', W')
        
        # Flatten and project
        x = x.view(B * S, -1)  # (B*S, C*H'*W')
        x = self.fc(x)  # (B*S, embed_dim)
        
        # Reshape back: (B, S, embed_dim)
        embeddings = x.view(B, S, self.output_dim)
        
        return embeddings


class EnvironmentalEncoderPooled(nn.Module):
    """
    Memory-efficient variant that produces a single pooled embedding per world.
    
    Use this when:
    - Memory is constrained
    - Only need world-level environmental context
    
    Output: (B, embed_dim) instead of (B, S, embed_dim)
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # Just ENV_r_field mean across species
        hidden_channels: List[int] = [32, 64, 128, 256],
        output_dim: int = 256,
        kernel_size: int = 3,
        grid_size: Tuple[int, int] = (20, 20)
    ):
        super().__init__()
        
        self.in_channels = in_channels + 2  # Add spatial coords
        self.output_dim = output_dim
        
        # Build convolutional layers
        layers = []
        curr_channels = self.in_channels
        
        for i, out_ch in enumerate(hidden_channels):
            layers.extend([
                nn.Conv2d(curr_channels, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ])
            if i < len(hidden_channels) - 1 and i % 2 == 1:
                layers.append(nn.MaxPool2d(2))
            curr_channels = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling + projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self, 
        env_field: torch.Tensor,  # (B, S, H, W)
        spatial_coords: torch.Tensor  # (B, 2, H, W)
    ) -> torch.Tensor:
        """Returns pooled embedding (B, embed_dim)."""
        # Mean across species
        env_mean = env_field.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate with spatial coords
        x = torch.cat([env_mean, spatial_coords], dim=1)  # (B, 3, H, W)
        
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        
        return x


# ============================================================================
# Interaction Encoder (GraphSAGE with Attention)
# ============================================================================

class InteractionEncoder(nn.Module):
    """
    Graph Neural Network encoder for species interaction network.
    
    Architecture:
    - Input: Species node features + interaction edges from C_topk
    - Output: (S, embed_dim) interaction embeddings per species
    
    Design Rationale:
    - GraphSAGE handles inductive learning (new species in test)
    - Attention aggregation weights important competitors more
    - Message passing mimics ecological interaction propagation
    - Multi-hop captures indirect effects (A→B→C competitive cascades)
    
    The interaction network represents competitive relationships where
    C_topk_idx[i] contains the top-16 competitors of species i, and
    C_topk_w[i] contains the corresponding interaction strengths (0.4 typical).
    """
    
    def __init__(
        self,
        node_features: int = 4,  # body_mass, prevalence, degree_in, degree_out
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_weights: bool = True
    ):
        super().__init__()
        
        self.node_features = node_features
        self.output_dim = output_dim
        self.use_edge_weights = use_edge_weights
        
        # Initial node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # GraphSAGE layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            # Use GATv2Conv for attention-based aggregation
            self.conv_layers.append(
                GATv2Conv(
                    in_dim, 
                    out_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=1 if use_edge_weights else None,
                    concat=True
                )
            )
            self.layer_norms.append(nn.LayerNorm(out_dim))
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        graph: GraphBatch  # PyG Batch containing node features and edges
    ) -> torch.Tensor:
        """
        Encode species interaction network.
        
        Args:
            graph: PyTorch Geometric Batch with:
                   - x: Node features (N_total, node_features)
                   - edge_index: Edge connectivity (2, E_total)
                   - edge_attr: Edge weights (E_total, 1) [optional]
                   - batch: Batch assignment (N_total,)
                   
        Returns:
            Interaction embeddings (B, S, embed_dim) or (N_total, embed_dim)
        """
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr if self.use_edge_weights else None
        
        # Initial embedding
        x = self.node_embed(x)
        
        # Message passing layers
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            # Residual connection where dimensions match
            residual = x if x.shape[-1] == conv.out_channels * conv.heads else None
            
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
            
            if residual is not None:
                x = x + residual
        
        # Final projection
        embeddings = self.output_proj(x)
        
        return embeddings, graph.batch
    
    def get_batched_embeddings(
        self, 
        embeddings: torch.Tensor, 
        batch: torch.Tensor,
        n_species: int
    ) -> torch.Tensor:
        """
        Reshape flat embeddings to (B, S, embed_dim).
        
        Args:
            embeddings: (N_total, embed_dim)
            batch: (N_total,) batch indices
            n_species: Number of species per world
            
        Returns:
            (B, S, embed_dim) batched embeddings
        """
        batch_size = batch.max().item() + 1
        
        # Initialize output tensor
        out = torch.zeros(
            batch_size, n_species, self.output_dim,
            device=embeddings.device, dtype=embeddings.dtype
        )
        
        # Scatter embeddings to correct positions
        for b in range(batch_size):
            mask = batch == b
            out[b, :mask.sum()] = embeddings[mask]
        
        return out


class InteractionEncoderSimple(nn.Module):
    """
    Simplified interaction encoder using standard SAGEConv.
    
    Use when:
    - Training on CPU (faster than attention)
    - Memory is very constrained
    - Edge weights are not important
    """
    
    def __init__(
        self,
        node_features: int = 4,
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Node embedding
        self.node_embed = nn.Linear(node_features, hidden_dims[0])
        
        # SAGEConv layers (mean aggregation)
        self.conv_layers = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.conv_layers.append(
                SAGEConv(dims[i], dims[i + 1], aggr='mean')
            )
        
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graph: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_embed(graph.x)
        
        for conv in self.conv_layers:
            x = conv(x, graph.edge_index)
            x = F.gelu(x)
            x = self.dropout(x)
        
        return self.output_proj(x), graph.batch


# ============================================================================
# Temporal Encoder (Transformer)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SpatialPooler(nn.Module):
    """Pool spatial dimensions (H, W) to single vector per species per timestep."""
    
    def __init__(self, grid_size: Tuple[int, int], embed_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial dimensions.
        
        Args:
            x: (B, T, S, H, W) presence time series
            
        Returns:
            (B, T, S, embed_dim) temporal features
        """
        B, T, S, H, W = x.shape
        
        # Reshape for pooling: (B*T*S, 1, H, W)
        x = x.view(B * T * S, 1, H, W)
        
        # Pool: (B*T*S, 1, 1, 1) → (B*T*S, 1)
        x = self.pool(x).view(B * T * S, 1)
        
        # Project to embedding dim: (B*T*S, embed_dim)
        x = self.proj(x)
        
        # Reshape: (B, T, S, embed_dim)
        return x.view(B, T, S, -1)


class TemporalEncoder(nn.Module):
    """
    Transformer encoder for temporal presence/absence dynamics.
    
    Architecture:
    - Input: P_t time series (T, S, H, W) per batch
    - Spatial pooling: (T, S, H, W) → (T, S, embed)
    - Temporal self-attention across T
    - Output: (S, embed_dim) temporal context per species
    
    Design Rationale:
    - Self-attention captures non-Markovian dynamics in colonization-extinction
    - Long-range dependencies (early colonization affects late distribution)
    - Per-species processing maintains species identity
    
    The temporal dynamics reflect assembly processes:
    - Early timesteps: Initial colonization attempts
    - Middle timesteps: Competitive sorting
    - Late timesteps: Equilibrium dynamics
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 50,
        grid_size: Tuple[int, int] = (20, 20)
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        # Spatial pooling + initial embedding
        self.spatial_pooler = SpatialPooler(grid_size, embed_dim)
        
        # Positional encoding for temporal dimension
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Temporal aggregation (attention pooling over time)
        self.time_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.time_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.output_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        temporal_history: torch.Tensor,  # (B, T, S, H, W)
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Encode temporal presence dynamics.
        
        Args:
            temporal_history: Presence time series, shape (B, T, S, H, W)
            return_sequence: If True, return (B, T, S, embed), else (B, S, embed)
            
        Returns:
            Temporal embeddings, shape (B, S, embed_dim) or (B, T, S, embed_dim)
        """
        B, T, S, H, W = temporal_history.shape
        
        # Spatial pooling: (B, T, S, H, W) → (B, T, S, embed)
        x = self.spatial_pooler(temporal_history)
        
        # Process each species independently with temporal transformer
        # Reshape: (B, T, S, embed) → (B*S, T, embed)
        x = x.permute(0, 2, 1, 3).reshape(B * S, T, self.embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding: (B*S, T, embed)
        x = self.transformer(x)
        
        if return_sequence:
            # Return full sequence: (B*S, T, embed) → (B, S, T, embed) → (B, T, S, embed)
            x = x.view(B, S, T, self.embed_dim).permute(0, 2, 1, 3)
            return self.output_norm(x)
        
        # Attention pooling over time to get single vector per species
        # Query: (1, 1, embed) → (B*S, 1, embed)
        query = self.time_query.expand(B * S, 1, self.embed_dim)
        
        # Attend to temporal sequence: (B*S, 1, embed)
        pooled, _ = self.time_attention(query, x, x)
        pooled = pooled.squeeze(1)  # (B*S, embed)
        
        # Reshape: (B*S, embed) → (B, S, embed)
        output = pooled.view(B, S, self.embed_dim)
        
        return self.output_norm(output)


class TemporalEncoderSimple(nn.Module):
    """
    Simplified temporal encoder using LSTM.
    
    Use when:
    - Training on CPU (faster than Transformer)
    - Sequence length is short (<20)
    - Memory is constrained
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        grid_size: Tuple[int, int] = (20, 20)
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.spatial_pooler = SpatialPooler(grid_size, embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, temporal_history: torch.Tensor) -> torch.Tensor:
        """Returns (B, S, embed_dim) temporal embeddings."""
        B, T, S, H, W = temporal_history.shape
        
        # Spatial pooling
        x = self.spatial_pooler(temporal_history)  # (B, T, S, embed)
        
        # Process per species: (B*S, T, embed)
        x = x.permute(0, 2, 1, 3).reshape(B * S, T, self.embed_dim)
        
        # LSTM encoding
        _, (h_n, _) = self.lstm(x)
        
        # Combine forward and backward: (num_layers*2, B*S, hidden) → (B*S, hidden*2)
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        # Project and reshape
        output = self.output_proj(h_combined)
        output = output.view(B, S, self.embed_dim)
        
        return self.output_norm(output)


# ============================================================================
# Factory Functions
# ============================================================================

def create_encoders(config) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Create all three encoders based on configuration.
    
    Returns:
        (environmental_encoder, interaction_encoder, temporal_encoder)
    """
    model_config = config.model
    device = config.device
    
    # Choose encoder variants based on device/memory
    use_simple = device == "cpu"
    
    # Environmental encoder
    env_encoder = EnvironmentalEncoder(
        in_channels=model_config.env_in_channels,
        hidden_channels=model_config.env_hidden_channels,
        output_dim=model_config.env_output_dim,
        kernel_size=model_config.env_kernel_size,
        grid_size=config.data.grid_size
    )
    
    # Interaction encoder
    if use_simple:
        int_encoder = InteractionEncoderSimple(
            node_features=model_config.graph_node_features,
            hidden_dims=model_config.graph_hidden_dims,
            output_dim=model_config.graph_output_dim,
            dropout=model_config.graph_dropout
        )
    else:
        int_encoder = InteractionEncoder(
            node_features=model_config.graph_node_features,
            hidden_dims=model_config.graph_hidden_dims,
            output_dim=model_config.graph_output_dim,
            num_heads=model_config.graph_num_heads,
            dropout=model_config.graph_dropout
        )
    
    # Temporal encoder
    if use_simple:
        temp_encoder = TemporalEncoderSimple(
            embed_dim=model_config.temporal_embed_dim,
            hidden_dim=model_config.temporal_embed_dim,
            num_layers=model_config.temporal_num_layers // 2,
            dropout=model_config.temporal_dropout,
            grid_size=config.data.grid_size
        )
    else:
        temp_encoder = TemporalEncoder(
            embed_dim=model_config.temporal_embed_dim,
            num_heads=model_config.temporal_num_heads,
            num_layers=model_config.temporal_num_layers,
            dropout=model_config.temporal_dropout,
            max_len=model_config.temporal_max_len,
            grid_size=config.data.grid_size
        )
    
    return env_encoder, int_encoder, temp_encoder


if __name__ == "__main__":
    """Test encoder modules."""
    import sys
    sys.path.append('..')
    from config import get_debug_config
    
    print("=" * 60)
    print("Testing Encoder Modules")
    print("=" * 60)
    
    config = get_debug_config()
    device = torch.device("cpu")
    
    # Test dimensions
    B, S, T, H, W = 2, 100, 25, 20, 20  # Reduced species for testing
    
    # Test Environmental Encoder
    print("\n[Test 1] Environmental Encoder")
    env_encoder = EnvironmentalEncoder(
        in_channels=3,
        hidden_channels=[16, 32, 64, 128],
        output_dim=128,
        grid_size=(H, W)
    ).to(device)
    
    env_field = torch.randn(B, S, H, W, device=device)
    spatial_coords = torch.randn(B, 2, H, W, device=device)
    
    env_embed = env_encoder(env_field, spatial_coords)
    print(f"  Input: env_field {env_field.shape}, coords {spatial_coords.shape}")
    print(f"  Output: {env_embed.shape}")
    print(f"  Parameters: {sum(p.numel() for p in env_encoder.parameters()):,}")
    
    # Test Interaction Encoder
    print("\n[Test 2] Interaction Encoder (Simple)")
    int_encoder = InteractionEncoderSimple(
        node_features=4,
        hidden_dims=[32, 64, 128],
        output_dim=128
    ).to(device)
    
    # Create dummy graph
    from torch_geometric.data import Data, Batch
    graphs = []
    for _ in range(B):
        n_nodes = S
        n_edges = S * 10
        x = torch.randn(n_nodes, 4)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        graphs.append(Data(x=x, edge_index=edge_index))
    
    graph_batch = Batch.from_data_list(graphs)
    graph_batch = graph_batch.to(device)
    
    int_embed, batch_idx = int_encoder(graph_batch)
    print(f"  Input: {graph_batch.num_nodes} nodes, {graph_batch.num_edges} edges")
    print(f"  Output: {int_embed.shape}")
    print(f"  Parameters: {sum(p.numel() for p in int_encoder.parameters()):,}")
    
    # Test Temporal Encoder
    print("\n[Test 3] Temporal Encoder (Simple)")
    temp_encoder = TemporalEncoderSimple(
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        grid_size=(H, W)
    ).to(device)
    
    temporal_history = torch.randn(B, T, S, H, W, device=device)
    
    temp_embed = temp_encoder(temporal_history)
    print(f"  Input: {temporal_history.shape}")
    print(f"  Output: {temp_embed.shape}")
    print(f"  Parameters: {sum(p.numel() for p in temp_encoder.parameters()):,}")
    
    print("\n[SUCCESS] All encoder tests passed!")
