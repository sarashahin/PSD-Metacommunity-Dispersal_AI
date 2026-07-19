"""
=============================================================================
TEMPORAL ENCODER: Transformer for Ecological Time Series
=============================================================================
This module implements a Transformer encoder for temporal dynamics in
species presence/biomass time series.

REASONING FOR TRANSFORMER ARCHITECTURE:

1. Long-range temporal dependencies
   - Colonization events can affect extinction decades later
   - LSTM/GRU have vanishing gradient issues for long sequences
   - Transformer attention captures long-range dependencies directly

2. Non-Markovian dynamics
   - Species persistence depends on full history, not just previous state
   - Self-attention sees entire sequence simultaneously

3. Parallel processing
   - Unlike RNNs, Transformer processes all timesteps in parallel
   - Much faster training on GPU

4. Positional encoding handles irregular time sampling
   - Our data has 50 snapshots over 10,000 steps
   - Time embedding can encode actual time, not just position

ARCHITECTURE:
- Input: Presence/biomass sequence (T, S, Y, X) → flattened to (T, S*Y*X)
- Positional encoding: Sinusoidal or learned
- Transformer encoder: Multi-head self-attention
- Output: Temporal context (S, D) or (T, S, D)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".
    
    REASONING:
    - Sinusoidal encoding allows extrapolation to longer sequences
    - Encodes absolute position information
    - Deterministic (no learned parameters)
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (B, T, D) input tensor
            
        Returns:
            (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    
    REASONING:
    - More flexible than sinusoidal for fixed-length sequences
    - Can learn task-specific position patterns
    - Our sequences are fixed at 50 timesteps
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (B, T, D) input tensor
            
        Returns:
            (B, T, D) with positional encoding added
        """
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        pos_embed = self.pos_embedding(positions)  # (T, D)
        x = x + pos_embed.unsqueeze(0)
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """
    Time embedding for actual simulation time values.
    
    REASONING:
    - Our 50 snapshots are at irregular intervals (IBM_t: [200, 400, ...])
    - Embedding actual time values preserves temporal scale information
    - Important for learning colonization/extinction rates
    """
    
    def __init__(self, d_model: int, max_time: float = 10000.0):
        super().__init__()
        
        self.d_model = d_model
        self.max_time = max_time
        
        # MLP to embed time
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time values.
        
        Args:
            t: (B, T) or (T,) time values
            
        Returns:
            (B, T, D) or (T, D) time embeddings
        """
        t_norm = t / self.max_time  # Normalize to [0, 1]
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(-1)  # (T, 1)
        else:
            t_norm = t_norm.unsqueeze(-1)  # (B, T, 1)
        
        return self.time_mlp(t_norm)


class SpatialPooler(nn.Module):
    """
    Pool spatial dimensions for temporal encoding.
    
    Options:
    - 'mean': Simple average pooling
    - 'attention': Learned attention pooling
    - 'flatten': Keep all spatial information (expensive)
    """
    
    def __init__(self, method: str = 'attention', spatial_dim: int = 400, hidden_dim: int = 256):
        super().__init__()
        
        self.method = method
        self.spatial_dim = spatial_dim  # Y * X
        
        if method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif method == 'flatten':
            self.proj = nn.Linear(spatial_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial dimensions.
        
        Args:
            x: (B, T, S, Y, X) or (T, S, Y, X)
            
        Returns:
            (B, T, S, pooled_dim) or (T, S, pooled_dim)
        """
        has_batch = x.dim() == 5
        if not has_batch:
            x = x.unsqueeze(0)
        
        B, T, S, Y, X = x.shape
        
        if self.method == 'mean':
            out = x.mean(dim=(-2, -1))  # (B, T, S)
            out = out.unsqueeze(-1)  # (B, T, S, 1)
            
        elif self.method == 'attention':
            x_flat = x.view(B, T, S, -1)  # (B, T, S, Y*X)
            attn = self.attention(x_flat.unsqueeze(-1))  # (B, T, S, Y*X, 1)
            attn = F.softmax(attn.squeeze(-1), dim=-1)  # (B, T, S, Y*X)
            out = (x_flat * attn).sum(dim=-1, keepdim=True)  # (B, T, S, 1)
            
        elif self.method == 'flatten':
            x_flat = x.view(B, T, S, -1)  # (B, T, S, Y*X)
            out = self.proj(x_flat)  # (B, T, S, hidden_dim)
        
        if not has_batch:
            out = out.squeeze(0)
        
        return out


class TemporalTransformerBlock(nn.Module):
    """
    Single Transformer block for temporal encoding.
    
    Architecture:
    1. Multi-head self-attention (temporal dimension)
    2. Feedforward network
    3. Layer normalization (pre-norm style)
    4. Residual connections
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, T, D) input tensor
            mask: Optional attention mask
            
        Returns:
            (B, T, D) output tensor
        """
        # Self-attention with pre-norm
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out
        
        # Feedforward with pre-norm
        x = x + self.ff(self.ff_norm(x))
        
        return x


class TemporalEncoder(nn.Module):
    """
    Transformer encoder for ecological time series.
    
    Takes presence/biomass history and produces temporal context embeddings
    that capture colonization-extinction dynamics.
    
    ARCHITECTURE:
    1. Spatial pooling: Compress (Y, X) to manageable dimension
    2. Input projection: Map to transformer dimension
    3. Positional encoding: Add temporal position information
    4. Transformer blocks: Self-attention over time
    5. Output: Either full sequence or aggregated context
    
    Input:
        history: (B, T, S, Y, X) presence or biomass history
        time_values: (B, T) or (T,) actual time values (optional)
    
    Output:
        context: (B, S, D) species temporal context
        sequence: (B, T, S, D) full temporal sequence (optional)
    """
    
    def __init__(
        self,
        input_dim: int = 1,  # After spatial pooling
        hidden_dim: int = 256,
        output_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        spatial_pool: str = 'attention',
        spatial_size: Tuple[int, int] = (20, 20),
        use_time_embedding: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_time_embedding = use_time_embedding
        
        # Spatial pooling
        self.spatial_pooler = SpatialPooler(
            method=spatial_pool,
            spatial_dim=spatial_size[0] * spatial_size[1],
            hidden_dim=input_dim,
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = LearnedPositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Optional time embedding
        if use_time_embedding:
            self.time_embed = TimeEmbedding(hidden_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                hidden_dim, n_heads,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Temporal aggregation (for final context)
        self.temporal_agg = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),  # Attention over time
        )
    
    def forward(
        self,
        history: torch.Tensor,
        time_values: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            history: (B, T, S, Y, X) presence/biomass history
            time_values: (B, T) or (T,) actual time values
            return_sequence: Whether to return full temporal sequence
            
        Returns:
            context: (B, S, output_dim) aggregated temporal context
            sequence: (B, T, S, output_dim) full sequence (if return_sequence=True)
        """
        B, T, S, Y, X = history.shape
        
        # Pool spatial dimensions: (B, T, S, Y, X) → (B, T, S, 1)
        pooled = self.spatial_pooler(history)
        
        # Reshape for per-species temporal encoding
        # Process each species independently but in parallel
        # (B, T, S, D) → (B*S, T, D)
        pooled = pooled.permute(0, 2, 1, 3)  # (B, S, T, D)
        pooled = pooled.reshape(B * S, T, -1)  # (B*S, T, D)
        
        # Input projection
        x = self.input_proj(pooled)  # (B*S, T, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add time embedding if provided
        if self.use_time_embedding and time_values is not None:
            if time_values.dim() == 1:
                time_values = time_values.unsqueeze(0).expand(B * S, -1)
            else:
                time_values = time_values.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
            time_embed = self.time_embed(time_values)
            x = x + time_embed
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output projection
        sequence = self.output_proj(x)  # (B*S, T, output_dim)
        
        # Reshape back: (B*S, T, D) → (B, S, T, D)
        sequence = sequence.reshape(B, S, T, -1)
        
        # Aggregate temporal dimension for context
        # Use attention-weighted mean
        attn_weights = self.temporal_agg(x[:, :, :self.hidden_dim])  # (B*S, T, 1)
        context = (x * attn_weights).sum(dim=1)  # (B*S, hidden_dim)
        context = self.output_proj[1](self.output_proj[0](context))  # (B*S, output_dim)
        context = context.reshape(B, S, -1)  # (B, S, output_dim)
        
        if return_sequence:
            return context, sequence.permute(0, 2, 1, 3)  # (B, S, D), (B, T, S, D)
        
        return context, None


class EfficientTemporalEncoder(nn.Module):
    """
    Memory-efficient temporal encoder for large species counts.
    
    Instead of processing all species independently, uses:
    1. Shared temporal encoding across species
    2. Per-species modulation
    
    REASONING:
    - Full (B*S, T, D) tensor is very large for S=3614
    - Many temporal patterns are shared across species
    - Species-specific modulation captures differences
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        spatial_size: Tuple[int, int] = (20, 20),
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        spatial_dim = spatial_size[0] * spatial_size[1]
        
        # Shared temporal encoder (processes species-aggregated signal)
        self.shared_proj = nn.Linear(spatial_dim, hidden_dim)
        self.pos_encoder = LearnedPositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        self.shared_transformer = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Per-species modulation
        self.species_modulator = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Output combination
        self.output_combiner = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(
        self,
        history: torch.Tensor,
        time_values: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            history: (B, T, S, Y, X) presence/biomass history
            
        Returns:
            context: (B, S, output_dim) temporal context per species
        """
        B, T, S, Y, X = history.shape
        
        # Aggregate across species for shared encoding
        shared_signal = history.mean(dim=2)  # (B, T, Y, X)
        shared_signal = shared_signal.view(B, T, -1)  # (B, T, Y*X)
        
        # Shared temporal encoding
        shared = self.shared_proj(shared_signal)  # (B, T, hidden_dim)
        shared = self.pos_encoder(shared)
        
        for block in self.shared_transformer:
            shared = block(shared)
        
        shared_context = shared.mean(dim=1)  # (B, hidden_dim)
        
        # Per-species modulation
        # Compute species-specific features from their temporal patterns
        history_flat = history.permute(0, 2, 1, 3, 4)  # (B, S, T, Y, X)
        history_flat = history_flat.reshape(B, S, T, -1)  # (B, S, T, Y*X)
        
        # Use final timestep for species modulation (can extend)
        species_signal = history_flat[:, :, -1, :]  # (B, S, Y*X)
        species_mod = self.species_modulator(species_signal)  # (B, S, output_dim)
        
        # Combine shared context with species modulation
        shared_expanded = shared_context.unsqueeze(1).expand(-1, S, -1)  # (B, S, hidden_dim)
        combined = torch.cat([shared_expanded, species_mod], dim=-1)  # (B, S, hidden+output)
        
        context = self.output_combiner(combined)  # (B, S, output_dim)
        
        return context, None


if __name__ == "__main__":
    # Test the temporal encoder
    print("=" * 60)
    print("TEMPORAL ENCODER TEST")
    print("=" * 60)
    
    # Test parameters
    B = 2
    T = 25  # Half history
    S = 100  # Smaller for testing
    Y, X = 20, 20
    output_dim = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create mock history
    history = torch.randn(B, T, S, Y, X).to(device)
    time_values = torch.linspace(0, 5000, T).to(device)
    
    print(f"\nInput shape: {history.shape}")
    
    # Test standard encoder
    print("\n1. Testing TemporalEncoder...")
    encoder = TemporalEncoder(
        input_dim=1,
        hidden_dim=256,
        output_dim=output_dim,
        n_heads=8,
        n_layers=2,
        max_seq_len=T,
        spatial_size=(Y, X),
    ).to(device)
    
    context, sequence = encoder(history, time_values, return_sequence=True)
    print(f"   Context shape: {context.shape}")
    print(f"   Sequence shape: {sequence.shape if sequence is not None else 'None'}")
    
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Test efficient encoder
    print("\n2. Testing EfficientTemporalEncoder...")
    encoder_eff = EfficientTemporalEncoder(
        hidden_dim=256,
        output_dim=output_dim,
        n_heads=8,
        n_layers=2,
        spatial_size=(Y, X),
    ).to(device)
    
    context_eff, _ = encoder_eff(history)
    print(f"   Context shape: {context_eff.shape}")
    
    n_params_eff = sum(p.numel() for p in encoder_eff.parameters())
    print(f"   Parameters: {n_params_eff:,}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = context.sum()
    loss.backward()
    print("   ✓ Gradients computed successfully")
    
    print("\n" + "=" * 60)
    print("✓ Temporal encoder tests passed!")
    print("=" * 60)
