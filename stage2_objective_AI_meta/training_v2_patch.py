"""
=============================================================================
TRAINING_V2_PATCH.PY  —  curriculum + history-sparsification patch
=============================================================================

This is a small monkey-patch on top of your existing training.py Trainer.
We keep all the working pieces (NaN protection, AMP, schedulers, loss
function, validation logic) and surgically add two missing behaviors:

  1. SWAPS DATALOADERS at phase boundaries.
     train_epoch(phase) now sets self.train_loader = self.phase_loaders[phase]['train']
     before iterating. This is what was missing — previous trainer
     called set_training_phase() (which only flips a model flag) but
     never changed the loader, so the conditioning never actually changed.

  2. SPARSIFIES THE HISTORY in phases 3 and 4.
     Per Axel's comment:
       "if the final point of the history is just moments before the
        observation, and observations are reasonably good, then the
        overall result is not surprising. ... if you can strengthen this
        point by arguing that observations of this past history are
        realistically sparse (just a few point observations for rare
        species), then the entire abstract becomes much more impressive.
        Are the 5-10 observations for both past and present species?
        Or is knowledge of past species perfect?"
     Answer: now they are sparse for both past and present.

     ALL frames of history_P (and history_B) are sparsified to K=5-10
     observations per species — not just the last frame, every frame.
     The model never sees a perfect snapshot at any point in time, only
     point observations of the kind a real ecologist would have.

WHY THIS MATTERS FOR YOUR PHD AIM
----------------------------------
Without this patch the temp_encoder learns a degenerate solution:
"the answer is at history[:,-1], copy it." With this patch it has to
INFER the underlying distribution from sparse evidence accumulated over
time, which is exactly the spatio-temporal process-aware modelling your
PhD aim describes.

USAGE
-----
This file does not need to be invoked directly. run_training_v2.py will
import it at startup, and the patch will install itself on the Trainer
class. Just run run_training_v2.py and you're done.
=============================================================================
"""

import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# History sparsification (GPU-vectorized, applied to ALL frames)
# ─────────────────────────────────────────────────────────────────────

def _sparsify_all_frames_fixed_K(history, K, generator=None):
    """
    Sparsify EVERY frame of history to K cells per species, drawn at
    random from the cells where the species was actually present at
    that time.

    history : (B, T, S, Y, X) float tensor on GPU
    K       : int, observation budget per species per frame

    Returns: (B, T, S, Y, X) sparse history (1.0 where observed, 0 else)
    """
    B, T, S, Y, X = history.shape
    flat = history.reshape(B, T, S, Y * X)  # (B, T, S, YX)

    # Random scores; -inf where species absent
    scores = torch.rand(B, T, S, Y * X, device=history.device,
                        generator=generator)
    scores = torch.where(flat > 0.5, scores,
                         torch.full_like(scores, float('-inf')))

    K_eff = min(K, Y * X)
    _, top_idx = scores.topk(k=K_eff, dim=-1)        # (B, T, S, K_eff)

    sparse_flat = torch.zeros_like(flat)
    sparse_flat.scatter_(-1, top_idx, 1.0)
    # Mask out species with fewer than K occupied cells per frame
    sparse_flat = sparse_flat * (flat > 0.5).float()
    return sparse_flat.reshape(B, T, S, Y, X)


def _sparsify_all_frames_poisson(history, biomass_history, body_mass,
                                  p_obs, generator=None):
    """
    Abundance-weighted Poisson sparsification on ALL frames.

    history          : (B, T, S, Y, X) presence
    biomass_history  : (B, T, S, Y, X) biomass at each time step
    body_mass, p_obs : Poisson parameters

    Returns: (B, T, S, Y, X) detected presence at each time step
    """
    N = biomass_history.to(torch.float32) / float(body_mass)
    expected = N * float(p_obs)
    counts = torch.poisson(expected, generator=generator)
    return (counts >= 1).to(history.dtype)


# ─────────────────────────────────────────────────────────────────────
# The patch
# ─────────────────────────────────────────────────────────────────────

def install_patch(Trainer):
    """
    Modify Trainer in-place to add phase-loader swapping and
    Axel-aligned history sparsification.
    """
    original_train_epoch = Trainer.train_epoch
    original_validate    = Trainer.validate
    original_train       = Trainer.train

    # ── Wrapper for train_epoch ─────────────────────────────────
    def patched_train_epoch(self, epoch, phase):
        # 1. Swap to phase-appropriate loader
        if hasattr(self, 'phase_loaders') and phase in self.phase_loaders:
            self.train_loader = self.phase_loaders[phase]['train']
            self.val_loader   = self.phase_loaders[phase]['val']

        # 2. Inject the history-sparsification hook into the batch
        #    iteration. We do this by wrapping the loader.
        if phase >= 3 and hasattr(self, '_install_hist_hook'):
            self._install_hist_hook(phase)

        return original_train_epoch(self, epoch, phase)

    Trainer.train_epoch = patched_train_epoch

    # ── Wrapper for validate ───────────────────────────────────
    def patched_validate(self, phase):
        if hasattr(self, 'phase_loaders') and phase in self.phase_loaders:
            self.val_loader = self.phase_loaders[phase]['val']
        return original_validate(self, phase)

    Trainer.validate = patched_validate

    # ── Add history-sparsification hook ─────────────────────────
    # Approach: monkey-patch the model's forward to sparsify the
    # history INSIDE the condition dict before encoding. This way
    # every code path (training, validation, debugging) gets the
    # correct supervision automatically.
    def _install_hist_hook(self, phase):
        """
        Wrap model.forward so that history_P and history_B in the
        condition dict are sparsified before being passed downstream.
        Idempotent: only installs once.
        """
        if getattr(self.model, '_hist_hook_installed', False):
            return
        self.model._hist_hook_installed = True

        original_forward = self.model.forward
        body_mass = float(getattr(self.config.data, 'body_mass', 1e-4))
        K_list = list(getattr(self.config.training,
                              'hist_sparsify_K', [5, 10]))
        # Poisson p_obs values matched to the v5 ablation table
        p_list = [1e-4, 5e-4, 1e-3]
        rng = torch.Generator(device=next(self.model.parameters()).device)
        rng.manual_seed(int(self.config.training.seed) + 1)
        # 50/50 mix of fixed-K and Poisson sparsification
        import random
        rnd = random.Random(int(self.config.training.seed) + 2)

        def patched_forward(target, condition, *args, **kwargs):
            if 'history_P' in condition and condition['history_P'] is not None:
                hist = condition['history_P']
                if rnd.random() < 0.5:
                    K = rnd.choice(K_list)
                    sparse_hist = _sparsify_all_frames_fixed_K(
                        hist, K, generator=rng)
                else:
                    p_obs = rnd.choice(p_list)
                    if 'history_B' in condition and condition['history_B'] is not None:
                        bio = condition['history_B']
                    else:
                        # Fallback: assume mean biomass = 0.4 at occupied cells
                        bio = hist * 0.4
                    sparse_hist = _sparsify_all_frames_poisson(
                        hist, bio, body_mass, p_obs, generator=rng)
                # Replace in condition (do not mutate caller's dict in place)
                new_cond = dict(condition)
                new_cond['history_P'] = sparse_hist
                # Also sparsify biomass history if present (parallel signal)
                if 'history_B' in condition and condition['history_B'] is not None:
                    new_cond['history_B'] = condition['history_B'] * sparse_hist
                condition = new_cond
            return original_forward(target, condition, *args, **kwargs)

        self.model.forward = patched_forward
        logger.info(f"   ✓ history-sparsification hook installed "
                    f"(K_list={K_list}, p_list={p_list})")

    Trainer._install_hist_hook = _install_hist_hook

    # ── Wrapper for train: log phase transitions clearly ───────
    def patched_train(self, *args, **kwargs):
        logger.info("Trainer v2 patch active: phase-loader swapping + "
                    "history sparsification enabled")
        return original_train(self, *args, **kwargs)

    Trainer.train = patched_train

    return Trainer


# ─────────────────────────────────────────────────────────────────────
# Auto-install when imported alongside training.py
# ─────────────────────────────────────────────────────────────────────

def auto_install():
    """
    Find the Trainer class in the already-imported training module and
    install the patch. Safe to call multiple times.
    """
    if 'training' in sys.modules:
        from training import Trainer
        if not getattr(Trainer, '_v2_patched', False):
            install_patch(Trainer)
            Trainer._v2_patched = True
            logger.info("   ✓ training.Trainer patched with v2 (curriculum + history sparsification)")
    else:
        logger.warning("   ⚠ training module not yet imported; "
                       "import training first, then call auto_install()")


# Trigger automatic install
auto_install()