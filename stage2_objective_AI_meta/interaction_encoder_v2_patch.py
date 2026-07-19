"""
=============================================================================
INTERACTION_ENCODER_V2_PATCH.PY  —  fix Phase 2+ batched edge_index crash
=============================================================================

ROOT CAUSE
----------
In your data_preprocessing.eco_collate_fn (line 689), edge_index and
edge_weight are LEFT AS PYTHON LISTS — one tensor per batch element —
because per-world graphs have different sizes:

    if key not in ['edge_index', 'edge_weight'] and isinstance(vals[0], torch.Tensor):
        try:
            collated['condition'][key] = torch.stack(vals)
        except:
            pass

interaction_encoder.forward (line 519) then does:

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], ...)   # <-- shape on a list

A list does not have .shape, and torch.sparse_coo_tensor cannot accept
a list of tensors. This is the crash:

    ValueError: only one element tensors can be converted to Python scalars

It was never triggered before because previous training was stuck in
Phase 1 (which doesn't use edge_index). Phase 2 onwards uses it; this
is the first time the code path has actually run with batch_size > 1.

THE FIX
-------
We monkey-patch InteractionEncoder.forward so that when it receives a
list of B per-sample edge tensors (the current dataloader output), it
internally builds a block-diagonal batched graph of B*S nodes, runs
ONE sparse matmul on the (B*S, B*S) adjacency, then reshapes back to
(B, S, H). This is mathematically equivalent to the per-batch loop the
encoder already does, just consolidated and free of the crash.

We deliberately do NOT patch eco_collate_fn — by the time we'd want to
patch it, the DataLoader has already captured the function reference,
so the patch wouldn't take effect. The encoder-side fix is the right
place because it works regardless of how the DataLoader was created.

WHY THIS IS SAFE
----------------
The block-diagonal pattern is mathematically equivalent to running the
GNN B times with a (S,S) adjacency per batch element. The encoder's
existing loop already does exactly that. The patch just consolidates
the loop into a single sparse matmul for efficiency and to avoid the
list-of-tensors bug.
=============================================================================
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helper: build block-diagonal batched edge_index
# ─────────────────────────────────────────────────────────────────────

def batch_edge_index(edge_index_list, edge_weight_list, S, device):
    """
    Combine B per-sample edge tensors into one block-diagonal graph.

    edge_index_list  : list of B tensors, each (2, E_i)
    edge_weight_list : list of B tensors, each (E_i,)  (or None)
    S                : species padding size
    device           : target device

    Returns:
        edge_index_b  : (2, sum(E_i))   indices in [0, B*S)
        edge_weight_b : (sum(E_i),)
        n_nodes       : B*S
    """
    B = len(edge_index_list)
    out_idx = []
    out_w = []
    for b, ei in enumerate(edge_index_list):
        if not isinstance(ei, torch.Tensor):
            continue
        if ei.numel() == 0:
            continue
        ei = ei.to(device).long()
        # Offset so block-b nodes live at [b*S, (b+1)*S)
        offset_ei = ei + b * S
        out_idx.append(offset_ei)
        if edge_weight_list and b < len(edge_weight_list) \
                and isinstance(edge_weight_list[b], torch.Tensor) \
                and edge_weight_list[b].numel() > 0:
            out_w.append(edge_weight_list[b].to(device).float())
        else:
            out_w.append(torch.ones(ei.shape[1], device=device))

    if not out_idx:
        return (torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, device=device), B * S)

    return (torch.cat(out_idx, dim=1),
            torch.cat(out_w, dim=0),
            B * S)


# ─────────────────────────────────────────────────────────────────────
# Patch interaction_encoder.forward
# ─────────────────────────────────────────────────────────────────────

def _resolve_encoder_classes():
    """
    Find the EfficientInteractionEncoder class that ecodiffusion.py
    actually uses. The import path in ecodiffusion.py is relative
    ('from .interaction_encoder import EfficientInteractionEncoder')
    so the class lives under 'models.interaction_encoder'. But this
    patch may be imported via top-level 'interaction_encoder' too.
    Both paths can resolve to the same class object OR to two
    different class objects depending on sys.path order.

    We patch every class object we can find, to be safe.
    """
    classes = []
    seen_ids = set()

    # Try multiple module paths
    for module_path in [
        'interaction_encoder',
        'models.interaction_encoder',
        'AI_simulation.stage2.models.interaction_encoder',
    ]:
        try:
            mod = __import__(module_path, fromlist=['*'])
        except (ImportError, ModuleNotFoundError):
            continue
        for class_name in ['EfficientInteractionEncoder', 'InteractionEncoder']:
            cls = getattr(mod, class_name, None)
            if cls is not None and id(cls) not in seen_ids:
                classes.append(cls)
                seen_ids.add(id(cls))

    # Last resort: scan sys.modules for any module whose name ends with
    # 'interaction_encoder' (catches odd import setups)
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not mod_name.endswith('interaction_encoder'):
            continue
        for class_name in ['EfficientInteractionEncoder', 'InteractionEncoder']:
            cls = getattr(mod, class_name, None)
            if cls is not None and id(cls) not in seen_ids:
                classes.append(cls)
                seen_ids.add(id(cls))

    return classes


def install_encoder_patch():
    """
    Patch EfficientInteractionEncoder (the class actually used by
    ecodiffusion.py — see line 747: 'from .interaction_encoder import
    EfficientInteractionEncoder'). This is the class whose .forward
    crashed at line 525 in your traceback.

    Also patches InteractionEncoder (the larger sibling class) for
    completeness in case it's used elsewhere. We patch every class
    object we can locate via every possible module path, because
    Python can hold the same class under multiple names.
    """
    classes_to_patch = _resolve_encoder_classes()
    if not classes_to_patch:
        logger.warning("   ⚠ no interaction_encoder classes found to patch")
        return

    for cls in classes_to_patch:
        if getattr(cls, '_v2_patched', False):
            continue

        original_forward = cls.forward

        def make_patched_forward(_original_forward):
            def patched_forward(self, species_features, edge_index, edge_weight=None):
                has_batch = species_features.dim() == 3
                if not has_batch:
                    species_features = species_features.unsqueeze(0)

                B, S, _ = species_features.shape
                device = species_features.device
                if isinstance(S, torch.Tensor):
                    S = int(S.item())

                # ── KEY FIX: handle list-of-tensors from collate ──
                if isinstance(edge_index, list):
                    ei_list = edge_index
                    ew_list = edge_weight if isinstance(edge_weight, list) \
                        else [None] * len(ei_list)
                    edge_index, edge_weight, _ = batch_edge_index(
                        ei_list, ew_list, S, device)

                # Empty-graph guard
                if edge_index is None or edge_index.numel() == 0 \
                        or edge_index.shape[1] == 0:
                    x = self.embed(species_features)
                    for hop in range(self.n_hops):
                        agg = torch.zeros_like(x)
                        x = self.hop_fusions[hop](torch.cat([x, agg], dim=-1))
                    out = self.output_proj(x)
                    return out.squeeze(0) if not has_batch else out

                edge_index = edge_index.to(device).long()
                if edge_weight is None:
                    edge_weight = torch.ones(edge_index.shape[1], device=device)
                else:
                    edge_weight = edge_weight.to(device).float()

                # Detect: single graph (max_idx < S) vs batched (max_idx >= S)
                max_idx = int(edge_index.max().item())
                if max_idx >= S:
                    # Block-diagonal batched: B*S nodes
                    N = B * S

                    # ──────────────────────────────────────────────
                    # AMP FIX (definitive): torch.sparse.mm has NO fp16
                    # CUDA kernel. The error
                    #    NotImplementedError: addmm_sparse_cuda not
                    #    implemented for 'Half'
                    # arises because PyTorch's autocast treats
                    # torch.sparse.mm as autocastable and downcasts its
                    # inputs to fp16 on the way in — even if we already
                    # cast them to fp32 with .to(torch.float32).
                    #
                    # The only reliable cure is to explicitly DISABLE
                    # autocast around the entire sparse region with a
                    # context manager. Inside this context, all ops run
                    # in fp32 regardless of the surrounding autocast
                    # state. We then cast outputs back to whatever dtype
                    # the outer autocast context expects so the rest of
                    # the model stays in fp16.
                    # ──────────────────────────────────────────────

                    # Detect outer autocast state so we can restore it
                    # after the sparse region. autocast nesting is OK.
                    outer_dtype = species_features.dtype
                    if outer_dtype not in (torch.float16, torch.bfloat16):
                        outer_dtype = torch.float32  # not actually autocasting

                    # Detect device type so autocast(disabled=False) targets
                    # the right autocast context (CUDA vs CPU)
                    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

                    with torch.amp.autocast(device_type=autocast_device, enabled=False):
                        # All inputs to sparse ops MUST be fp32
                        sf32 = species_features.float()
                        x_flat = self.embed(sf32).reshape(N, -1)  # (B*S, H) fp32

                        ew_f = edge_weight.float()
                        adj = torch.sparse_coo_tensor(
                            edge_index, ew_f, (N, N), device=device,
                        ).coalesce()
                        row_sum = torch.sparse.sum(adj, dim=1).to_dense() + 1e-10
                        norm_w = adj.values() / row_sum[adj.indices()[0]]
                        adj_norm = torch.sparse_coo_tensor(
                            adj.indices(), norm_w, (N, N), device=device,
                        )

                        x = x_flat
                        for hop in range(self.n_hops):
                            agg = torch.sparse.mm(adj_norm, x)        # (B*S, H) fp32
                            fused = self.hop_fusions[hop](
                                torch.cat([x, agg], dim=-1).reshape(B, S, -1)
                            )
                            x = fused.reshape(N, -1)
                        out_f32 = self.output_proj(x.reshape(B, S, -1))

                    # Cast back to AMP dtype for the rest of the model
                    out = out_f32.to(outer_dtype)
                    return out.squeeze(0) if not has_batch else out
                else:
                    # Single-graph path — wrap in disabled autocast too,
                    # because the original implementation also uses
                    # torch.sparse.mm internally (line 555).
                    sf = species_features if has_batch \
                        else species_features.squeeze(0)
                    outer_dtype = sf.dtype
                    if outer_dtype not in (torch.float16, torch.bfloat16):
                        outer_dtype = torch.float32
                    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
                    with torch.amp.autocast(device_type=autocast_device, enabled=False):
                        out_f32 = _original_forward(
                            self, sf.float(), edge_index, edge_weight.float(),
                        )
                    return out_f32.to(outer_dtype)
            return patched_forward

        cls.forward = make_patched_forward(original_forward)
        cls._v2_patched = True
        logger.info(f"   ✓ {cls.__name__} (id={id(cls)}) .forward patched")


def auto_install():
    install_encoder_patch()


auto_install()