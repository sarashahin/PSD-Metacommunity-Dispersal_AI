"""
=============================================================================
ecodiffusion_spatial_cond.py  —  SPATIAL-CONDITIONING FIX for EcoDiffusion
=============================================================================

ROOT CAUSE THIS FILE FIXES
--------------------------
In the original model (ecodiffusion.py) the conditioning pathway is
SPATIALLY COLLAPSED. Every encoder reduces its output to a per-species
vector with NO (Y, X) dimension before it reaches the denoiser:

  encode_condition():
     env_global, env_spatial = self.env_encoder(...)   # env_spatial DISCARDED
     ...
     cond = self.conditioning(env_global, int_emb, temp_emb, ...)  # (B, S, D)

  FixedConditioningModule.forward():
     if env_emb.dim() == 5: env_emb = env_emb.mean(dim=(-2,-1))    # space pooled

  EfficientTemporalEncoder.forward():
     shared_signal = history.mean(dim=2)                          # space kept but
     species_signal = history_flat[:, :, -1, :]                   # LAST FRAME only
     -> projected straight to a vector

  SpeciesParallelUNet:
     FiLM:  h = h * (1 + scale) + shift   with scale/shift of shape (B*S, C, 1, 1)
            -> the SAME modulation at every grid cell.

So the U-Net's ONLY spatially-resolved input is the noisy x_t itself. It
cannot see WHERE the environment is suitable, or WHERE the K observations
are. It can only learn the marginal "how big / what texture is a species
range" and emit an unstructured blob (~35 cells when truth ~9). That is
exactly why:
  - individual samples over-predict (~35 cells)
  - the 8 samples are spatially uncorrelated -> their mean collapses to ~1%
  - far-cell recall ~ 0.03 (no spatial targeting)
  - the model loses to Gaussian smoothing, which DOES use obs location
  - ensemble-union precision is ~0.013 (union of blobs ~ whole grid)

THE FIX (standard Palette / SR3 / RePaint conditioning-by-concatenation)
------------------------------------------------------------------------
Concatenate spatial conditioning channels directly onto x_t BEFORE the
U-Net input projection. Per species, three extra channels:

  ch 0  obs_mask   : 1.0 at the K observed cells (the sparse evidence)
  ch 1  env        : per-species environmental suitability field
                     (Axel lists "local suitability of the environment"
                      as a key predictor, transcript 7:02)
  ch 2  obs_decay  : exp(-d^2 / sigma^2), soft distance-to-nearest-observation
                     (the spatial-autocorrelation / dispersal prior; this is
                      the signal the Gaussian-smoothing baseline exploits, so
                      giving it to the model as an INPUT means the model can
                      only do better than smoothing, never worse).

These channels are KNOWN and FIXED: they are never noised, and are passed
clean at every denoising step. The per-species FiLM vector is KEPT — it
still carries species identity, interactions and the temporal summary.

REQUIRES RETRAINING
-------------------
input_proj gains 3 input channels the old checkpoint never saw, so the old
best_model.pt cannot be reused for those weights. Train from scratch with
run_training.py pointed at create_spatial_cond_model (see bottom of file).

USAGE
-----
1. Copy this file to AI_simulation/stage2/models/
2. In run_training.py, replace:
       from models.ecodiffusion import create_fixed_model
       model = create_fixed_model(config)
   with:
       from models.ecodiffusion_spatial_cond import create_spatial_cond_model
       model = create_spatial_cond_model(config)
3. In generate_reconstructions_*, load with create_spatial_cond_model and
   use the matching sampler ecodiffusion_sample_spatial.sample_spatial.
4. training.py / training_v2_patch.py need NO change: EcoDiffusionSpatial
   builds the spatial channels internally inside forward(), reading the
   (already-sparsified) history that the v2 patch produces.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .ecodiffusion import (
    SpeciesParallelUNet,
    EcoDiffusionFixed,
)


# =============================================================================
#  SPATIAL-CONDITIONING U-NET
# =============================================================================

class SpeciesParallelUNetSpatial(SpeciesParallelUNet):
    """
    SpeciesParallelUNet that additionally accepts per-species spatial
    conditioning channels concatenated onto x_t at the input projection.

    The ONLY architectural change vs the parent is the input projection:
        parent : Conv2d(1,                  base_channels, 3, padding=1)
        this   : Conv2d(1 + n_cond_channels, base_channels, 3, padding=1)
    Everything downstream (encoder / middle / decoder / output, all FiLM
    conditioning) is unchanged and operates on `base_channels`.
    """

    def __init__(self, *args, n_cond_channels: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_cond_channels = n_cond_channels
        # Replace the input projection so it accepts x_t + conditioning channels.
        self.input_proj = nn.Conv2d(
            1 + n_cond_channels, self.base_channels,
            kernel_size=3, padding=1,
        )

    # ----- single-chunk forward (mirrors parent, only the input differs) -----
    def _forward_single_chunk(
        self,
        x: torch.Tensor,            # (chunk, 1, Y, X)        noisy map
        t: torch.Tensor,            # (chunk,)
        cond: torch.Tensor,         # (chunk, D)              per-species vector
        cond_spatial: torch.Tensor,  # (chunk, n_cond_ch, Y, X) clean spatial cond
    ) -> torch.Tensor:
        target_size = x.shape[-2:]

        # Time embedding + per-species FiLM vector (identical to parent)
        time_emb = self.time_embed(t)
        combined_cond = torch.cat([cond, time_emb], dim=-1)
        cond_emb = self.cond_proj(combined_cond)

        # ── THE FIX: concat clean spatial conditioning channels onto x_t ──
        h_in = torch.cat([x, cond_spatial], dim=1)   # (chunk, 1 + n_cond_ch, Y, X)
        h = self.input_proj(h_in)                    # (chunk, base_channels, Y, X)

        # ----- everything below is identical to the parent U-Net -----
        skips = []
        block_idx = 0
        for level in range(len(self.downsamplers) + 1):
            for _ in range(2):  # n_res_blocks
                h = self.encoder_blocks[block_idx](h, cond_emb)
                block_idx += 1
                if block_idx < len(self.encoder_blocks) and isinstance(
                    self.encoder_blocks[block_idx], nn.Conv2d
                ):
                    h = self.encoder_blocks[block_idx](h)
                    block_idx += 1
            skips.append(h)
            if level < len(self.downsamplers):
                h = self.downsamplers[level](h)

        h = self.middle_block1(h, cond_emb)
        h = self.middle_block2(h, cond_emb)

        block_idx = 0
        for level in range(len(self.upsamplers) + 1):
            if skips:
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    skip = F.interpolate(skip, size=h.shape[-2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            for _ in range(3):  # n_res_blocks + 1
                if block_idx < len(self.decoder_blocks):
                    layer = self.decoder_blocks[block_idx]
                    if isinstance(layer, type(self.middle_block1)):
                        h = layer(h, cond_emb)
                    else:
                        h = layer(h)
                    block_idx += 1
                    if block_idx < len(self.decoder_blocks) and isinstance(
                        self.decoder_blocks[block_idx], nn.Conv2d
                    ):
                        h = self.decoder_blocks[block_idx](h)
                        block_idx += 1
            if level < len(self.upsamplers):
                h = self.upsamplers[level](h)

        out = self.output_proj(h)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear',
                                align_corners=False)
        return out

    # ----- full forward: thread cond_spatial through the chunking -----
    def forward(
        self,
        x: torch.Tensor,            # (B, S, Y, X)
        t: torch.Tensor,            # (B,)
        cond: torch.Tensor,         # (B, S, D)
        cond_spatial: torch.Tensor,  # (B, S, n_cond_ch, Y, X)
    ) -> torch.Tensor:
        # Reshape (B, S, Y, X) -> (B*S, 1, Y, X), mirroring the parent.
        if x.dim() == 4 and x.shape[1] != 1:
            B, S, Y, X = x.shape
            x = x.view(B * S, 1, Y, X)
            t = t.repeat_interleave(S)
            cond = cond.reshape(B * S, -1)
            cond_spatial = cond_spatial.reshape(B * S, self.n_cond_channels, Y, X)
            needs_reshape = True
        else:
            B, S = x.shape[0], 1
            needs_reshape = False
            if cond_spatial.dim() == 5:
                cond_spatial = cond_spatial.reshape(
                    -1, self.n_cond_channels, *cond_spatial.shape[-2:])

        target_size = x.shape[-2:]
        total = x.shape[0]  # B*S

        if total > self.species_chunk_size:
            outputs = []
            for start in range(0, total, self.species_chunk_size):
                end = min(start + self.species_chunk_size, total)
                x_c = x[start:end]
                t_c = t[start:end]
                cond_c = cond[start:end]
                cs_c = cond_spatial[start:end]
                if self.training and self.use_gradient_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    out_c = checkpoint(
                        self._forward_single_chunk,
                        x_c, t_c, cond_c, cs_c, use_reentrant=False,
                    )
                else:
                    out_c = self._forward_single_chunk(x_c, t_c, cond_c, cs_c)
                outputs.append(out_c)
                if start > 0 and start % (self.species_chunk_size * 4) == 0:
                    if x.device.type == 'cuda':
                        torch.cuda.empty_cache()
            out = torch.cat(outputs, dim=0)
        else:
            if self.training and self.use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                out = checkpoint(
                    self._forward_single_chunk,
                    x, t, cond, cond_spatial, use_reentrant=False,
                )
            else:
                out = self._forward_single_chunk(x, t, cond, cond_spatial)

        if needs_reshape:
            out = out.view(B, S, *target_size)
        return out


# =============================================================================
#  ECODIFFUSION WITH SPATIAL CONDITIONING
# =============================================================================

class EcoDiffusionSpatial(EcoDiffusionFixed):
    """
    EcoDiffusionFixed with the spatial-conditioning U-Net swapped in and a
    helper that builds the (obs_mask, env, obs_decay) spatial channel stack
    from the condition dict.

    All encoders (env / interaction / temporal) and encode_condition() are
    INHERITED UNCHANGED — they still produce the per-species FiLM vector.
    We only add a parallel, spatially-resolved conditioning path.
    """

    def __init__(self, *args, n_cond_channels: int = 3,
                 obs_decay_sigma: float = 2.0,
                 dense_obs_threshold: int = 30,
                 K_obs_cap: int = 10, **kwargs):
        # dense_obs_threshold : if any species' last-frame obs_mask has more
        #   than this many cells, build_spatial_cond treats the frame as dense
        #   and applies the safety-net sparsification. 30 is well above the
        #   K=5/10/20 budgets the pipeline uses, so genuinely-sparse frames
        #   (the normal case) pass through untouched.
        # K_obs_cap : the per-species cap the safety net falls back to. Only
        #   ever used if a dense frame slips through; the normal pipeline
        #   sparsification (K=5/10/20) is preserved when the frame is sparse.
        # Capture unet_config so we can rebuild the U-Net with the *exact*
        # same hyper-parameters (channel_mults, n_res_blocks, ...) rather than
        # trying to reverse-engineer them from the constructed module.
        unet_config = kwargs.get('unet_config', None)
        if unet_config is None and len(args) >= 8:
            # positional: matches EcoDiffusionFixed.__init__ signature order
            unet_config = args[7]
        unet_cfg = unet_config or {
            'base_channels': 64, 'channel_mults': [1, 2, 4, 8], 'n_res_blocks': 2,
        }

        super().__init__(*args, **kwargs)
        self.n_cond_channels = n_cond_channels
        self.obs_decay_sigma = float(obs_decay_sigma)
        self.dense_obs_threshold = int(dense_obs_threshold)
        self.K_obs_cap = int(K_obs_cap)

        # Swap the U-Net for the spatial-conditioning version, rebuilt with
        # the exact same hyper-parameters EcoDiffusionFixed used internally.
        old = self.unet
        self.unet = SpeciesParallelUNetSpatial(
            spatial_size=self.spatial_size,
            base_channels=unet_cfg.get('base_channels', 64),
            channel_mults=unet_cfg.get('channel_mults', [1, 2, 4, 8]),
            n_res_blocks=unet_cfg.get('n_res_blocks', 2),
            cond_dim=self.hidden_dim,
            time_dim=self.hidden_dim,
            species_chunk_size=old.species_chunk_size,
            use_gradient_checkpointing=old.use_gradient_checkpointing,
            n_cond_channels=n_cond_channels,
        )
        del old

        # Pre-compute the (Y*X, Y*X) Gaussian decay kernel for obs_decay.
        Y, X = self.spatial_size
        yy, xx = torch.meshgrid(torch.arange(Y), torch.arange(X), indexing='ij')
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)  # (YX,YX)
        decay_kernel = torch.exp(-d2 / (self.obs_decay_sigma ** 2))
        self.register_buffer('obs_decay_kernel', decay_kernel)

    # ----- build the spatial conditioning channel stack -----
    def _obs_decay(self, obs_mask: torch.Tensor) -> torch.Tensor:
        """
        Soft distance-to-nearest-observation field.
        obs_mask : (B, S, Y, X)  binary
        returns  : (B, S, Y, X)  in [0,1]; decay[...,j] = max_i obs[...,i]*Kexp[i,j]
        """
        B, S, Y, X = obs_mask.shape
        flat = obs_mask.reshape(B * S, Y * X)
        out = torch.empty_like(flat)
        K = self.obs_decay_kernel  # (YX, YX)
        chunk = 64
        for i in range(0, B * S, chunk):
            seg = flat[i:i + chunk]                                # (c, YX)
            # (c, YX_i, 1) * (1, YX_i, YX_j) -> max over YX_i -> (c, YX_j)
            d = (seg.unsqueeze(-1) * K.unsqueeze(0)).amax(dim=1)
            out[i:i + chunk] = d
        return out.reshape(B, S, Y, X)

    def build_spatial_cond(self, condition: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build the (B, S, n_cond_channels, Y, X) clean conditioning stack.

        ch 0 : obs_mask  = last history frame, binarised, GUARANTEED SPARSE
        ch 1 : env       = condition['env'] (per-species suitability, in [0,1])
        ch 2 : obs_decay = soft distance-to-observation field

        DEFENSIVE SPARSIFICATION
        ------------------------
        The obs_mask channel MUST be the sparse K-point evidence, never the
        dense final frame. In the normal pipeline this is guaranteed:
          - training : training_v2_patch.patched_forward sparsifies
                       condition['history_P'] BEFORE this runs (phases >= 3)
          - inference: generate_reconstructions_*.py builds a sparse
                       history via sparsify_history_fixed_budget()

        But that correctness depends on three files agreeing. If for ANY
        reason a dense last frame reaches this method (e.g. the training
        patch did not fire, a phase-2 loader leaked history_P, a future
        refactor), training on a dense obs_mask would silently reintroduce
        the trivial-copy shortcut Axel warned about. So we DETECT density
        and sparsify here as a hard safety net. A frame is considered
        "already sparse" if no species has more than `dense_obs_threshold`
        observed cells; otherwise every species is capped to `K_obs_cap`
        cells drawn at random from its occupied cells.
        """
        env = condition['env']                       # (B, S, Y, X)
        B, S, Y, X = env.shape
        device = env.device

        hist = condition.get('history_P')
        if hist is not None and hist.numel() > 0:
            obs_mask = (hist[:, -1] > 0.5).float()   # (B, S, Y, X)
            obs_mask = self._ensure_sparse(obs_mask)
        else:
            obs_mask = torch.zeros(B, S, Y, X, device=device)

        decay = self._obs_decay(obs_mask)            # (B, S, Y, X)

        cond_spatial = torch.stack([obs_mask, env, decay], dim=2)
        return cond_spatial                         # (B, S, 3, Y, X)
    
    # inside class EcoDiffusionSpatial, anywhere in the class body
    def set_training_phase(self, phase):
        super().set_training_phase(phase)
        # parent printed "EcoDiffusionFixed: ..." — that's misleading for this class.
        # The parent's print already fired; just print a follow-up clarifier.
        print(f"  (^^ class is actually EcoDiffusionSpatial, phase={phase})")

    def _ensure_sparse(self, obs_mask: torch.Tensor) -> torch.Tensor:
        """
        Safety net: if any species' obs_mask has more than
        `self.dense_obs_threshold` cells, cap EVERY species to
        `self.K_obs_cap` randomly-chosen occupied cells. If the frame is
        already sparse (the normal case), it is returned unchanged so the
        upstream sparsification (which may use K=5/10/20) is preserved.
        """
        B, S, Y, X = obs_mask.shape
        per_species = obs_mask.reshape(B, S, Y * X).sum(-1)        # (B, S)
        if per_species.max().item() <= self.dense_obs_threshold:
            return obs_mask                                       # already sparse

        flat = obs_mask.reshape(B, S, Y * X)
        scores = torch.rand(B, S, Y * X, device=obs_mask.device)
        scores = torch.where(flat > 0.5, scores,
                             torch.full_like(scores, float('-inf')))
        K = min(self.K_obs_cap, Y * X)
        _, top_idx = scores.topk(k=K, dim=-1)                     # (B, S, K)
        sparse = torch.zeros_like(flat)
        sparse.scatter_(-1, top_idx, 1.0)
        sparse = sparse * (flat > 0.5).float()   # don't invent cells for absent species
        return sparse.reshape(B, S, Y, X)

    # ----- training forward -----
    def forward(
        self,
        x_0: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, S, Y, X = x_0.shape
        device = x_0.device

        t = torch.randint(0, self.diffusion.timesteps, (B,), device=device)
        x_t, noise = self.diffusion.q_sample(x_0, t, noise)

        # Per-species FiLM vector (inherited, unchanged).
        cond = self.encode_condition(
            env=condition['env'],
            y_coords=condition['y_coords'],
            x_coords=condition['x_coords'],
            species_features=condition.get('species_features'),
            edge_index=condition.get('edge_index'),
            edge_weight=condition.get('edge_weight'),
            history=condition.get('history_P'),
        )

        # Spatial conditioning stack (THE FIX).
        cond_spatial = self.build_spatial_cond(condition)

        noise_pred = self.unet(x_t, t, cond, cond_spatial)  # (B, S, Y, X)
        loss = F.mse_loss(noise_pred, noise)

        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise_true': noise,
            'x_t': x_t,
            't': t,
        }


# =============================================================================
#  FACTORY
# =============================================================================

def create_spatial_cond_model(config, n_cond_channels: int = 3,
                               obs_decay_sigma: float = 2.0) -> EcoDiffusionSpatial:
    """
    Drop-in replacement for create_fixed_model() that builds the
    spatial-conditioning model.
    """
    species_chunk_size = getattr(config.model, 'species_chunk_size', 64)
    use_gradient_checkpointing = getattr(
        config.model, 'use_gradient_checkpointing', True)

    return EcoDiffusionSpatial(
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
        species_chunk_size=species_chunk_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        n_cond_channels=n_cond_channels,
        obs_decay_sigma=obs_decay_sigma,
    )