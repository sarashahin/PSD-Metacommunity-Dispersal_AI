"""
=============================================================================
INTERACTION ENCODER: Graph Neural Network for Species Interactions
=============================================================================
This module implements a Graph Neural Network to encode species interaction
networks from the C_topk_idx and C_topk_w data.

REASONING FOR GNN ARCHITECTURE:

Interaction matrix is sparse (16 connections per species via C_topk)
Full attention over 3614 species would be O(S²) = 13M operations
GNN with sparse adjacency is O(E) = 58K operations (16 × 3614)

Message passing mimics ecological interaction propagation
Competitive effects propagate: if A competes with B, and B with C, A indirectly affects C
Multi-layer GNN captures these multi-hop effects

Attention mechanism weights interaction importance
Not all interactions are equally important for distribution
Learned attention discovers ecologically meaningful weights

GraphSAGE aggregation is inductive
Can generalize to new species not seen during training
Important for real-world application with unseen species

ARCHITECTURE:
- Input: Species features (S, F) + interaction graph (E edges)
- Layers: GraphSAGE with attention aggregation
- Output: Species embeddings (S, D) capturing interaction context
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT-style) with edge weight integration.

    Computes attention weights between connected nodes, incorporating
    the predefined interaction weights from C_topk_w.

    REASONING:
    - Attention allows learning which interactions matter most
    - Edge weights from simulation encode interaction strength
    - Combining learned + predefined weights preserves ecological info
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        use_edge_weights: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.use_edge_weights = use_edge_weights

        # Per-head dimension
        self.head_dim = out_features // n_heads if concat else out_features

        # Linear projections for query, key, value
        self.W_q = nn.Linear(in_features, self.head_dim * n_heads, bias=False)
        self.W_k = nn.Linear(in_features, self.head_dim * n_heads, bias=False)
        self.W_v = nn.Linear(in_features, self.head_dim * n_heads, bias=False)

        # Edge weight transformation
        if use_edge_weights:
            self.edge_proj = nn.Sequential(
                nn.Linear(1, n_heads),
                nn.Sigmoid(),  # Bound edge contribution
            )

        # Output projection
        if concat:
            self.out_proj = nn.Linear(self.head_dim * n_heads, out_features)
        else:
            self.out_proj = nn.Linear(self.head_dim, out_features)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (S, in_features)
            edge_index: Edge indices (2, E) where [0] is source, [1] is target
            edge_weight: Edge weights (E,) from C_topk_w

        Returns:
            out: Updated node features (S, out_features)
        """
        S = x.shape[0]
        H = self.n_heads
        D = self.head_dim

        # Project to query, key, value
        q = self.W_q(x).view(S, H, D)  # (S, H, D)
        k = self.W_k(x).view(S, H, D)  # (S, H, D)
        v = self.W_v(x).view(S, H, D)  # (S, H, D)

        # Get source and target indices
        src, tgt = edge_index[0], edge_index[1]  # Each (E,)

        # Compute attention scores for edges
        q_src = q[src]  # (E, H, D)
        k_tgt = k[tgt]  # (E, H, D)

        # Scaled dot-product attention
        attn_scores = (q_src * k_tgt).sum(dim=-1) / math.sqrt(D)  # (E, H)

        # Incorporate edge weights
        if self.use_edge_weights and edge_weight is not None:
            edge_attn = self.edge_proj(edge_weight.unsqueeze(-1))  # (E, H)
            attn_scores = attn_scores + edge_attn

        # Softmax over incoming edges for each target node
        # We need to normalize per target node
        attn_weights = self._sparse_softmax(attn_scores, tgt, S)  # (E, H)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values
        v_src = v[src]  # (E, H, D)
        weighted_v = attn_weights.unsqueeze(-1) * v_src  # (E, H, D)

        # Scatter-add to target nodes
        out = torch.zeros(S, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, H, D), weighted_v)

        # Combine heads
        if self.concat:
            out = out.view(S, H * D)  # (S, H*D)
        else:
            out = out.mean(dim=1)  # (S, D)

        out = self.out_proj(out)
        out = self.layer_norm(out + x if self.in_features == self.out_features else out)

        return out

    def _sparse_softmax(
        self,
        scores: torch.Tensor,
        index: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Compute softmax over sparse edges grouped by target node.

        Args:
            scores: Attention scores (E, H)
            index: Target node indices (E,)
            n_nodes: Total number of nodes

        Returns:
            weights: Normalized attention weights (E, H)
        """
        # Compute max per target for numerical stability
        scores_max = torch.zeros(n_nodes, scores.shape[1], device=scores.device)
        scores_max.scatter_reduce_(0, index.view(-1, 1).expand(-1, scores.shape[1]), 
                                    scores, reduce='amax', include_self=False)

        # Subtract max and compute exp
        scores = scores - scores_max[index]
        exp_scores = torch.exp(scores)

        # Sum per target
        exp_sum = torch.zeros(n_nodes, scores.shape[1], device=scores.device)
        exp_sum.scatter_add_(0, index.view(-1, 1).expand(-1, scores.shape[1]), exp_scores)

        # Normalize
        weights = exp_scores / (exp_sum[index] + 1e-10)

        return weights

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer with mean/attention aggregation.

    REASONING for GraphSAGE over GCN:
    - GCN normalizes by degree, which can dilute signals from key interactions
    - GraphSAGE's sampling + aggregation is more flexible
    - Mean aggregation is permutation invariant (order of neighbors doesn't matter)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = 'mean',  # 'mean', 'attention', 'max'
        dropout: float = 0.1,
    ):
        super().__init__()

        self.aggregator = aggregator

        # Transform self features
        self.self_linear = nn.Linear(in_features, out_features)

        # Transform neighbor features
        self.neigh_linear = nn.Linear(in_features, out_features)

        # Optional attention aggregation
        if aggregator == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(in_features * 2, out_features),
                nn.LeakyReLU(0.2),
                nn.Linear(out_features, 1),
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (S, in_features)
            edge_index: Edge indices (2, E)
            edge_weight: Edge weights (E,)

        Returns:
            out: Updated node features (S, out_features)
        """
        S = x.shape[0]
        src, tgt = edge_index[0], edge_index[1]

        # Self transformation
        self_feat = self.self_linear(x)  # (S, out_features)

        # Neighbor aggregation
        neigh_feat = x[src]  # (E, in_features)

        if self.aggregator == 'mean':
            # Weighted mean aggregation
            if edge_weight is not None:
                neigh_feat = neigh_feat * edge_weight.unsqueeze(-1)

            # Aggregate to target nodes
            agg = torch.zeros(S, neigh_feat.shape[1], device=x.device)
            count = torch.zeros(S, 1, device=x.device)
            agg.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, neigh_feat.shape[1]), neigh_feat)
            count.scatter_add_(0, tgt.unsqueeze(-1), torch.ones_like(tgt, dtype=torch.float).unsqueeze(-1))
            agg = agg / (count + 1e-10)

        elif self.aggregator == 'max':
            # Max aggregation
            agg = torch.zeros(S, neigh_feat.shape[1], device=x.device) - float('inf')
            agg.scatter_reduce_(0, tgt.unsqueeze(-1).expand(-1, neigh_feat.shape[1]), 
                               neigh_feat, reduce='amax', include_self=False)
            agg = torch.where(agg == -float('inf'), torch.zeros_like(agg), agg)

        elif self.aggregator == 'attention':
            # Attention-weighted aggregation
            tgt_feat = x[tgt]  # (E, in_features)
            attn_input = torch.cat([neigh_feat, tgt_feat], dim=-1)  # (E, 2*in_features)
            attn_scores = self.attention(attn_input).squeeze(-1)  # (E,)

            # Softmax per target
            attn_weights = self._sparse_softmax(attn_scores, tgt, S)
            if edge_weight is not None:
                attn_weights = attn_weights * edge_weight

            neigh_feat = neigh_feat * attn_weights.unsqueeze(-1)
            agg = torch.zeros(S, neigh_feat.shape[1], device=x.device)
            agg.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, neigh_feat.shape[1]), neigh_feat)

        # Transform aggregated neighbors
        neigh_transformed = self.neigh_linear(agg)  # (S, out_features)

        # Combine self and neighbor features
        out = self.act(self_feat + neigh_transformed)
        out = self.dropout(out)
        out = self.layer_norm(out)

        return out

    def _sparse_softmax(self, scores: torch.Tensor, index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Compute softmax over sparse edges grouped by target node."""
        scores_max = torch.zeros(n_nodes, device=scores.device)
        scores_max.scatter_reduce_(0, index, scores, reduce='amax', include_self=False)

        scores = scores - scores_max[index]
        exp_scores = torch.exp(scores)

        exp_sum = torch.zeros(n_nodes, device=scores.device)
        exp_sum.scatter_add_(0, index, exp_scores)

        return exp_scores / (exp_sum[index] + 1e-10)

class InteractionEncoder(nn.Module):
    """
    Graph Neural Network encoder for species interaction networks.

    Takes species features and interaction graph, produces embeddings that
    capture each species' interaction context (who it competes with, how
    strongly, and how those competitors are connected).

    ARCHITECTURE:
    1. Input projection: Map species features to hidden dimension
    2. GNN layers: Propagate information along interaction edges
    3. Output projection: Final species embeddings

    The output embeddings can be used to:
    - Predict competitive exclusion effects
    - Identify species clusters/guilds
    - Condition distribution predictions on interaction context

    Input:
        species_features: (B, S, F) per-species attributes
        edge_index: (2, E) interaction edges (same for all batch items)
        edge_weight: (E,) interaction weights

    Output:
        embeddings: (B, S, output_dim) interaction-aware species embeddings
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        output_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if use_attention:
                self.layers.append(GraphAttentionLayer(
                    hidden_dim, hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=True,
                ))
            else:
                self.layers.append(GraphSAGELayer(
                    hidden_dim, hidden_dim,
                    aggregator='mean',
                    dropout=dropout,
                ))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Skip connection
        self.skip_proj = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        species_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            species_features: (B, S, input_dim) or (S, input_dim) species attributes
            edge_index: (2, E) interaction edges
            edge_weight: (E,) interaction weights

        Returns:
            embeddings: (B, S, output_dim) or (S, output_dim) species embeddings
        """
        # Handle batch dimension
        if species_features.dim() == 2:
            # No batch dimension: (S, F)
            x = self.input_proj(species_features)  # (S, hidden_dim)

            for layer in self.layers:
                x = layer(x, edge_index, edge_weight)

            out = self.output_proj(x)  # (S, output_dim)
            skip = self.skip_proj(species_features)  # (S, output_dim)

            return out + skip

        else:
            # Batch dimension: (B, S, F)
            B, S, F = species_features.shape
            outputs = []

            for b in range(B):
                x = self.input_proj(species_features[b])  # (S, hidden_dim)

                for layer in self.layers:
                    x = layer(x, edge_index, edge_weight)

                out = self.output_proj(x)  # (S, output_dim)
                skip = self.skip_proj(species_features[b])  # (S, output_dim)
                outputs.append(out + skip)

            return torch.stack(outputs, dim=0)  # (B, S, output_dim)

class EfficientInteractionEncoder(nn.Module):
    """
    Memory-efficient interaction encoder for very large species counts.

    Instead of full GNN message passing, uses:
    1. Species feature embedding
    2. Interaction aggregation via sparse matrix multiplication
    3. MLP fusion

    REASONING:
    - Full GNN on 3614 species is expensive
    - Most interaction information is captured by immediate neighbors
    - Sparse ops are highly optimized in PyTorch
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 256,
        output_dim: int = 256,
        n_hops: int = 2,  # How many neighborhood aggregations
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hops = n_hops

        # Feature embedding
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-hop fusion
        self.hop_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
            ) for _ in range(n_hops)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        species_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using sparse matrix multiplication.

        Args:
            species_features: (B, S, input_dim) or (S, input_dim)
            edge_index: (2, E) interaction edges
            edge_weight: (E,) interaction weights

        Returns:
            embeddings: (B, S, output_dim) or (S, output_dim)
        """
        has_batch = species_features.dim() == 3
        if not has_batch:
            species_features = species_features.unsqueeze(0)

        B, S, _ = species_features.shape
        device = species_features.device

        # Build sparse adjacency matrix (normalized)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=device)

        # Fix: Convert tensor to Python int for sparse tensor creation
        if isinstance(S, torch.Tensor):
            S = int(S.item())
        # Create sparse matrix
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            (S, S),
            device=device,
        ).coalesce()

        # Normalize by row sum (out-degree)
        row_sum = torch.sparse.sum(adj, dim=1).to_dense() + 1e-10
        norm_weight = edge_weight / row_sum[edge_index[0]]
        adj_norm = torch.sparse_coo_tensor(
            edge_index,
            norm_weight,
            (S, S),
            device=device,
        )

        # Embed features
        x = self.embed(species_features)  # (B, S, hidden_dim)

        # Multi-hop aggregation
        for hop in range(self.n_hops):
            # Aggregate neighbors: (B, S, H) @ (S, S) -> (B, S, H)
            # Need to handle batch dimension
            x_flat = x.view(B * S, -1).t()  # (H, B*S) for sparse mm

            # This is inefficient for batched sparse mm, so we loop
            agg_list = []
            for b in range(B):
                x_b = x[b].t()  # (H, S)
                agg_b = torch.sparse.mm(adj_norm.t(), x_b.t()).t()  # (H, S)
                agg_list.append(agg_b.t())  # (S, H)
            agg = torch.stack(agg_list, dim=0)  # (B, S, H)

            # Fuse self and neighbor
            x = self.hop_fusions[hop](torch.cat([x, agg], dim=-1))

        out = self.output_proj(x)

        if not has_batch:
            out = out.squeeze(0)

        return out

if __name__ == "__main__":
    # Test the interaction encoder
    print("=" * 60)
    print("INTERACTION ENCODER TEST")
    print("=" * 60)

    # Test parameters
    B = 2
    S = 500  # Smaller for testing
    F = 8
    K = 16  # Interactions per species
    output_dim = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # Create mock data
    species_features = torch.randn(B, S, F).to(device)

    # Create interaction graph (K neighbors per node)
    src = torch.arange(S).repeat_interleave(K)
    tgt = torch.randint(0, S, (S * K,))
    edge_index = torch.stack([src, tgt]).to(device)
    edge_weight = torch.full((S * K,), 0.4).to(device)

    # Remove self-loops
    mask = src != tgt
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]

    print(f"\nGraph: {S} nodes, {edge_index.shape[1]} edges")

    # Test standard encoder
    print("\n1. Testing InteractionEncoder...")
    encoder = InteractionEncoder(
        input_dim=F,
        hidden_dim=128,
        output_dim=output_dim,
        n_layers=2,
        n_heads=4,
        use_attention=True,
    ).to(device)

    embeddings = encoder(species_features, edge_index, edge_weight)
    print(f"   Input shape: {species_features.shape}")
    print(f"   Output shape: {embeddings.shape}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"   Parameters: {n_params:,}")

    # Test efficient encoder
    print("\n2. Testing EfficientInteractionEncoder...")
    encoder_eff = EfficientInteractionEncoder(
        input_dim=F,
        hidden_dim=256,
        output_dim=output_dim,
        n_hops=2,
    ).to(device)

    embeddings_eff = encoder_eff(species_features, edge_index, edge_weight)
    print(f"   Output shape: {embeddings_eff.shape}")

    n_params_eff = sum(p.numel() for p in encoder_eff.parameters())
    print(f"   Parameters: {n_params_eff:,}")

    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = embeddings.sum()
    loss.backward()
    print("   ✓ Gradients computed successfully")

    print("\n" + "=" * 60)
    print("✓ Interaction encoder tests passed!")
    print("=" * 60)