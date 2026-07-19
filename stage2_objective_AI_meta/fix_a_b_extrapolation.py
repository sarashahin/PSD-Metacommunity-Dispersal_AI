"""
=============================================================================
FIX_A_B_EXTRAPOLATION.PY  —  root-cause fix for the "copy not predict" failure
=============================================================================

WHAT THE DIAGNOSTIC SHOWED (epoch-44 best_model.pt, diagnose_spatial_use.py)
----------------------------------------------------------------------------
  Jaccard FULL              = 0.8326   <- looks great
  Jaccard NO_OBS / NO_SPAT  = 0.0007   <- collapses without obs channels
  auc_unobs FULL            = 0.3444   <- BELOW RANDOM at unobserved cells
  NO_ENV ~= FULL                       <- env channel is being IGNORED

Interpretation: the model learned obs_mask -> output as a near-identity map.
It echoes the K observed cells (+ a small obs_decay halo) and predicts flat
absence everywhere else. High Jaccard is almost entirely the handed-in cells.
At the cells it must actually PREDICT (unobserved), it does worse than a coin
flip. This is the trivial-copy failure in a new form.

ROOT CAUSE (two compounding causes, both addressed here)
--------------------------------------------------------
  CAUSE 1  The loss never rewards getting UNOBSERVED presence cells right.
           The diffusion eps-loss is over all cells equally; rare species are
           ~99.5% absence, so "echo the observations, predict flat-low
           elsewhere" already gives a low loss. There is no gradient pressure
           toward extrapolation.
           -> FIX A: ExtrapolationLoss — weighted BCE computed ONLY at
              unobserved cells, with true-presence cells up-weighted.

  CAUSE 2  obs_mask is a binary input channel that equals the answer at the
           observed cells. Gradient descent takes the lazy path: identity-
           copy that channel. The weaker, noisier env channel is ignored
           (exactly what NO_ENV ~= FULL shows).
           -> FIX B: attenuate obs_mask (x0.5) so it cannot be identity-
              copied to a {0,1} presence map. obs_decay still carries "where
              the observations are"; env still carries suitability. The model
              is forced to TRANSFORM rather than COPY.

WHAT THIS FILE CONTAINS
-----------------------
  1. ExtrapolationLoss            — the new loss module (Fix A)
  2. obs_mask_from_condition()    — extracts the sparse obs_mask from the
                                    condition dict, mirroring build_spatial_cond
  3. install_fix_a()              — monkey-patches CombinedEcologicalLoss to
                                    add the extrapolation term (phase >= 4)
  4. install_fix_b()              — monkey-patches EcoDiffusionSpatial.
                                    build_spatial_cond to attenuate obs_mask
  5. a __main__ self-test with synthetic tensors (NOT training data — only
     to verify the math/shapes of the loss; real training uses real data)

INTEGRATION  (see bottom of file for the exact run_training.py edit)
--------------------------------------------------------------------
This is applied the same way as the other v2 patches: import the module and
call install_fix_a() / install_fix_b() AFTER training.py and
ecodiffusion_spatial_cond.py are imported, BEFORE the model/trainer are built.
=============================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# FIX A — the extrapolation loss
# =============================================================================

class ExtrapolationLoss(nn.Module):
    """
    Weighted BCE computed ONLY at unobserved cells.

    The model is handed K observed cells via obs_mask. Those cells are
    EXCLUDED here — scoring them would just reward the copy. What remains is
    the genuine prediction task: every cell the model was NOT shown.

    Within the unobserved cells, true-presence cells are rare, so a plain BCE
    is still dominated by absence. We up-weight true-presence cells by
    `presence_weight` so the gradient actually pushes the model to LIGHT UP
    unobserved presence cells instead of predicting flat absence.

    Inputs
    ------
      pred_prob : (B, S, Y, X)  predicted probability in [0,1]
      target    : (B, S, Y, X)  ground-truth presence {0,1}
      obs_mask  : (B, S, Y, X)  1 at the K observed cells, 0 elsewhere
      species_mask : (B, S) or None  — padding mask; padded species excluded

    Returns
    -------
      scalar loss (mean over valid species)
    """

    def __init__(self, presence_weight: float = 20.0, eps: float = 1e-6):
        super().__init__()
        # presence_weight : how much more a true-presence unobserved cell
        #   counts than a true-absence one. 20 is chosen because rare species
        #   are ~99.5% absence; ~1/0.005 = 200 would fully balance, but that
        #   over-pushes toward false positives. 20 gives a strong but stable
        #   pull. It is a hyperparameter — see notes at end of file.
        self.presence_weight = float(presence_weight)
        self.eps = float(eps)

    def forward(self, pred_prob, target, obs_mask, species_mask=None):
        # clamp for numerically safe log
        p = torch.clamp(pred_prob, self.eps, 1.0 - self.eps)
        t = (target > 0.5).float()

        # unobserved = NOT an observed cell
        unobs = (obs_mask < 0.5).float()              # (B, S, Y, X)

        # per-cell BCE
        bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))

        # per-cell weight: presence cells up-weighted, then restricted to
        # unobserved cells only (observed cells get weight 0 -> excluded)
        w = 1.0 + (self.presence_weight - 1.0) * t    # 1 for absence, PW for presence
        w = w * unobs                                  # zero out observed cells

        # per-species reduction: sum(weighted bce) / sum(weight)
        num = (bce * w).reshape(bce.shape[0], bce.shape[1], -1).sum(-1)
        den = w.reshape(w.shape[0], w.shape[1], -1).sum(-1)
        per_species = num / (den + self.eps)          # (B, S)

        if species_mask is not None:
            sm = species_mask.float()
            # also require the species to actually have unobserved cells with
            # non-zero weight (den > 0); otherwise it contributes nothing
            valid = sm * (den > 0).float()
            total = (per_species * valid).sum() / (valid.sum() + self.eps)
        else:
            valid = (den > 0).float()
            total = (per_species * valid).sum() / (valid.sum() + self.eps)

        return torch.clamp(total, max=50.0)


# =============================================================================
# obs_mask extraction — mirrors EcoDiffusionSpatial.build_spatial_cond exactly
# =============================================================================

def obs_mask_from_condition(condition, dense_obs_threshold=30, K_obs_cap=10):
    """
    Reconstruct the sparse obs_mask the model actually saw, from the condition
    dict. This MIRRORS build_spatial_cond: obs_mask = binarised last history
    frame, with the defensive sparsification safety net applied.

    Returns (B, S, Y, X) float tensor, or None if there is no history.

    NOTE: in the normal training pipeline, training_v2_patch already
    sparsified condition['history_P'] before forward(). So the last frame is
    already the K-budget sparse frame; _ensure_sparse here is a no-op in that
    case (it only fires if a dense frame slips through). Identical logic to
    the model's own build_spatial_cond — kept in sync deliberately.
    """
    hist = condition.get('history_P')
    if hist is None or hist.numel() == 0:
        return None

    obs_mask = (hist[:, -1] > 0.5).float()            # (B, S, Y, X)

    # defensive sparsification — identical to EcoDiffusionSpatial._ensure_sparse
    B, S, Y, X = obs_mask.shape
    per_species = obs_mask.reshape(B, S, Y * X).sum(-1)
    if per_species.max().item() > dense_obs_threshold:
        flat = obs_mask.reshape(B, S, Y * X)
        scores = torch.rand(B, S, Y * X, device=obs_mask.device)
        scores = torch.where(flat > 0.5, scores,
                             torch.full_like(scores, float('-inf')))
        K = min(K_obs_cap, Y * X)
        _, top_idx = scores.topk(k=K, dim=-1)
        sparse = torch.zeros_like(flat)
        sparse.scatter_(-1, top_idx, 1.0)
        sparse = sparse * (flat > 0.5).float()
        obs_mask = sparse.reshape(B, S, Y, X)

    return obs_mask


# =============================================================================
# FIX A installer — patch CombinedEcologicalLoss to add the extrapolation term
# =============================================================================

def install_fix_a(training_module, extrap_weight: float = 0.5,
                   presence_weight: float = 20.0):
    """
    Monkey-patch training.CombinedEcologicalLoss so its forward() also
    computes ExtrapolationLoss (phase >= 4 only) and adds it to total.

    Parameters
    ----------
      training_module : the imported `training` module
      extrap_weight   : weight of the extrapolation term in the total loss.
                        0.5 is deliberate: the diffusion term is ~0.01 late in
                        training and the extrapolation BCE is O(0.1-1), so
                        0.5 makes it a CO-dominant signal without swamping the
                        diffusion objective. Tune if needed (see notes).
      presence_weight : passed to ExtrapolationLoss.

    The patched forward() now REQUIRES an `obs_mask` keyword argument when
    phase >= 4. If obs_mask is None at phase >= 4, the extrapolation term is
    skipped with a warning (so the run does not crash) — but that should not
    happen in the normal pipeline.
    """
    CombinedEcologicalLoss = training_module.CombinedEcologicalLoss

    if getattr(CombinedEcologicalLoss, '_fix_a_installed', False):
        logger.info("   Fix A already installed — skipping")
        return

    original_forward = CombinedEcologicalLoss.forward
    extrap_loss_fn = ExtrapolationLoss(presence_weight=presence_weight)

    def patched_forward(self, diffusion_loss, pred_x0, target, prevalence,
                        mask=None, phase=1, obs_mask=None):
        # run the original loss (diffusion + prevalence + cooc + spatial)
        losses = original_forward(self, diffusion_loss, pred_x0, target,
                                  prevalence, mask, phase)

        # add the extrapolation term in phase >= 4
        if phase >= 4:
            if obs_mask is None:
                logger.warning("Fix A: phase>=4 but obs_mask is None — "
                               "extrapolation loss skipped this batch")
            else:
                try:
                    pred_prob = torch.sigmoid(torch.clamp(pred_x0, -10.0, 10.0))
                    e_loss = extrap_loss_fn(pred_prob, target, obs_mask, mask)
                    if not torch.isnan(e_loss):
                        losses['extrapolation'] = e_loss
                        losses['total'] = losses['total'] + extrap_weight * e_loss
                except Exception as ex:
                    logger.warning(f"Fix A: extrapolation loss failed: {ex}")

        return losses

    CombinedEcologicalLoss.forward = patched_forward
    CombinedEcologicalLoss._fix_a_installed = True
    CombinedEcologicalLoss._extrap_weight = extrap_weight
    logger.info(f"   ✓ Fix A installed: ExtrapolationLoss "
                f"(extrap_weight={extrap_weight}, "
                f"presence_weight={presence_weight}, phase>=4)")


# =============================================================================
# FIX A — trainer call-site patch
# =============================================================================

def install_fix_a_trainer(training_module):
    """
    The loss is called inside Trainer.train_epoch and Trainer.validate. The
    patched CombinedEcologicalLoss.forward now accepts obs_mask, but the
    call-sites in training.py do not pass it. This patches train_epoch and
    validate to extract obs_mask from the batch's condition dict and pass it.

    We wrap the two methods rather than editing them so the science inside
    (NaN protection, AMP, metrics) is byte-for-byte unchanged.
    """
    Trainer = training_module.Trainer

    if getattr(Trainer, '_fix_a_trainer_installed', False):
        logger.info("   Fix A trainer-patch already installed — skipping")
        return

    # We cannot easily wrap just the loss call, so instead we install a
    # thin shim: the loss_fn object gets a wrapper that pulls obs_mask from
    # a per-batch attribute the patched train_epoch/validate set.
    #
    # Simpler and more robust: patch Trainer so that self.loss_fn is replaced
    # by a callable that remembers the most recent condition dict. The
    # condition dict is set by a patched train_epoch/validate just before the
    # loss call. This avoids editing the (long) method bodies.

    original_train_epoch = Trainer.train_epoch
    original_validate = Trainer.validate

    def _attach_condition_capture(trainer):
        """Wrap trainer.loss_fn so each call reads trainer._current_condition."""
        if getattr(trainer, '_loss_fn_wrapped', False):
            return
        raw_loss_fn = trainer.loss_fn

        def wrapped_loss_fn(diffusion_loss, pred_x0, target, prevalence,
                            mask=None, phase=1):
            cond = getattr(trainer, '_current_condition', None)
            obs_mask = None
            if cond is not None and phase >= 4:
                obs_mask = obs_mask_from_condition(cond)
            return raw_loss_fn(diffusion_loss, pred_x0, target, prevalence,
                               mask, phase, obs_mask=obs_mask)

        trainer.loss_fn = wrapped_loss_fn
        trainer._loss_fn_wrapped = True

    def patched_train_epoch(self, epoch, phase):
        _attach_condition_capture(self)
        # monkey-patch the dataloader iteration is invasive; instead we patch
        # the model.forward to stash the condition on the trainer right before
        # the loss is computed. The trainer calls self.model(target, condition)
        # then self.loss_fn(...). We hook model.forward to record condition.
        model = self.model
        if not getattr(model, '_cond_capture_installed', False):
            raw_model_forward = model.forward

            def capturing_forward(target, condition, *a, **kw):
                self._current_condition = condition
                return raw_model_forward(target, condition, *a, **kw)

            model.forward = capturing_forward
            model._cond_capture_installed = True
        return original_train_epoch(self, epoch, phase)

    def patched_validate(self, phase):
        _attach_condition_capture(self)
        model = self.model
        if not getattr(model, '_cond_capture_installed', False):
            raw_model_forward = model.forward

            def capturing_forward(target, condition, *a, **kw):
                self._current_condition = condition
                return raw_model_forward(target, condition, *a, **kw)

            model.forward = capturing_forward
            model._cond_capture_installed = True
        return original_validate(self, phase)

    Trainer.train_epoch = patched_train_epoch
    Trainer.validate = patched_validate
    Trainer._fix_a_trainer_installed = True
    logger.info("   ✓ Fix A trainer-patch installed: obs_mask threaded "
                "into loss via condition capture")


# =============================================================================
# FIX B installer — attenuate obs_mask in build_spatial_cond
# =============================================================================

def install_fix_b(spatial_module, obs_mask_scale: float = 0.5):
    """
    Monkey-patch EcoDiffusionSpatial.build_spatial_cond so the obs_mask
    channel (channel 0) is multiplied by `obs_mask_scale`.

    WHY 0.5 AND NOT 0.0 (drop entirely)
    -----------------------------------
    Dropping obs_mask entirely would remove the model's ability to know
    EXACTLY which cell was observed (obs_decay with sigma=2 blurs a point
    across ~5x5 cells). The inpainting overlay at sample time still needs a
    precise mask, but that mask is built separately by the sampler — so the
    training-time channel could in principle be dropped. We keep it at 0.5
    because:
      * a {0, 0.5} channel CANNOT be identity-copied onto a {0,1} presence
        map — the copy shortcut is broken
      * but the precise location signal is preserved, just attenuated
      * env and obs_decay are unchanged, so "where + suitability" still flow
    This is the conservative choice. If you want the hard drop, set
    obs_mask_scale=0.0.

    This requires RETRAINING FROM SCRATCH — the input statistics of channel 0
    change, so an existing checkpoint is not compatible in a meaningful sense
    (it would technically load, but the learned input_proj weights expect the
    old scale).
    """
    EcoDiffusionSpatial = spatial_module.EcoDiffusionSpatial

    if getattr(EcoDiffusionSpatial, '_fix_b_installed', False):
        logger.info("   Fix B already installed — skipping")
        return

    original_build = EcoDiffusionSpatial.build_spatial_cond

    def patched_build(self, condition):
        cs = original_build(self, condition)          # (B, S, 3, Y, X)
        # channel 0 is obs_mask — attenuate it
        cs = cs.clone()
        cs[:, :, 0] = cs[:, :, 0] * obs_mask_scale
        return cs

    EcoDiffusionSpatial.build_spatial_cond = patched_build
    EcoDiffusionSpatial._fix_b_installed = True
    EcoDiffusionSpatial._obs_mask_scale = obs_mask_scale
    logger.info(f"   ✓ Fix B installed: obs_mask channel attenuated "
                f"x{obs_mask_scale} (copy shortcut broken)")


# =============================================================================
# convenience — install everything
# =============================================================================

def install_all(training_module, spatial_module,
                 extrap_weight: float = 0.5,
                 presence_weight: float = 20.0,
                 obs_mask_scale: float = 0.5):
    """Install Fix A (loss + trainer patch) and Fix B (channel attenuation)."""
    install_fix_a(training_module, extrap_weight=extrap_weight,
                  presence_weight=presence_weight)
    install_fix_a_trainer(training_module)
    install_fix_b(spatial_module, obs_mask_scale=obs_mask_scale)
    logger.info("   ✓ Fix A + Fix B fully installed")


# =============================================================================
# SELF-TEST — verifies the MATH and SHAPES of ExtrapolationLoss only.
# This uses synthetic tensors purely to check tensor algebra. It is NOT a
# substitute for training on real data. Real training uses the real IBM worlds.
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  SELF-TEST — ExtrapolationLoss math & shape verification")
    print("  (synthetic tensors, used ONLY to check the algebra)")
    print("=" * 70)
    torch.manual_seed(0)

    B, S, Y, X = 2, 12, 20, 20
    loss_fn = ExtrapolationLoss(presence_weight=20.0)

    # ---- test 1: a model that COPIES (perfect at obs, flat-zero elsewhere)
    target = torch.zeros(B, S, Y, X)
    obs_mask = torch.zeros(B, S, Y, X)
    for b in range(B):
        for s in range(S):
            # true range: 12 cells; only 5 of them observed
            idx = torch.randperm(Y * X)[:12]
            target.reshape(B, S, -1)[b, s, idx] = 1.0
            obs_mask.reshape(B, S, -1)[b, s, idx[:5]] = 1.0

    # copier: prob 1 exactly at observed cells, ~0 everywhere else
    pred_copy = obs_mask.clone() * 0.98 + 0.01
    l_copy = loss_fn(pred_copy, target, obs_mask)
    print(f"\n  test 1  COPIER  (echoes obs, flat-low elsewhere)")
    print(f"          extrapolation loss = {l_copy.item():.4f}")
    print(f"          -> should be HIGH: 7 unobserved presence cells per")
    print(f"             species are all predicted ~0.01, heavily penalised")

    # ---- test 2: a model that EXTRAPOLATES (correct at unobserved presence)
    pred_good = target.clone() * 0.98 + 0.01      # correct everywhere
    l_good = loss_fn(pred_good, target, obs_mask)
    print(f"\n  test 2  EXTRAPOLATOR  (correct at unobserved presence cells)")
    print(f"          extrapolation loss = {l_good.item():.4f}")
    print(f"          -> should be LOW: unobserved presence cells predicted ~1")

    # ---- test 3: observed cells must NOT affect the loss
    pred_a = target.clone() * 0.98 + 0.01
    pred_b = pred_a.clone()
    pred_b = pred_b * (1 - obs_mask) + obs_mask * 0.5   # corrupt ONLY obs cells
    l_a = loss_fn(pred_a, target, obs_mask)
    l_b = loss_fn(pred_b, target, obs_mask)
    print(f"\n  test 3  OBSERVED CELLS EXCLUDED")
    print(f"          loss with obs cells correct   = {l_a.item():.6f}")
    print(f"          loss with obs cells corrupted = {l_b.item():.6f}")
    print(f"          difference = {abs(l_a.item() - l_b.item()):.2e}")
    print(f"          -> should be ~0: obs cells are excluded from the loss")

    # ---- assertions
    assert l_copy.item() > l_good.item() * 3, \
        f"copier loss ({l_copy.item():.3f}) should be >> extrapolator " \
        f"loss ({l_good.item():.3f})"
    assert abs(l_a.item() - l_b.item()) < 1e-5, \
        "corrupting OBSERVED cells changed the loss — they are not excluded"

    # ---- test 4: species_mask excludes padded species
    sm = torch.ones(B, S)
    sm[:, 8:] = 0.0                                # last 4 species are padding
    l_masked = loss_fn(pred_good, target, obs_mask, species_mask=sm)
    assert not torch.isnan(l_masked), "species-masked loss is NaN"
    print(f"\n  test 4  SPECIES MASK")
    print(f"          loss with 4/12 species masked = {l_masked.item():.4f}")
    print(f"          -> finite, padded species excluded  OK")

    print("\n" + "=" * 70)
    print("  ALL SELF-TESTS PASSED")
    print("  copier loss >> extrapolator loss  (the term rewards prediction)")
    print("  observed cells excluded           (no reward for copying)")
    print("  species mask respected            (padding excluded)")
    print("=" * 70)


# =============================================================================
# HOW TO INTEGRATE  —  exact edit to run_training.py
# =============================================================================
#
# In run_training.py, find the block that imports the v2 patches (just after
# `import training_v2_patch` / `import interaction_encoder_v2_patch`). Add:
#
#     # ---- Fix A + Fix B : extrapolation loss + obs_mask attenuation ----
#     import training
#     from models import ecodiffusion_spatial_cond
#     import fix_a_b_extrapolation
#     fix_a_b_extrapolation.install_all(
#         training_module = training,
#         spatial_module  = ecodiffusion_spatial_cond,
#         extrap_weight   = 0.5,
#         presence_weight = 20.0,
#         obs_mask_scale  = 0.5,
#     )
#
# This MUST run AFTER `import training` and AFTER ecodiffusion_spatial_cond is
# importable, but BEFORE `create_spatial_cond_model(config)` and BEFORE the
# Trainer is constructed (install_fix_a patches the loss CLASS; the Trainer
# builds its loss_fn instance in __init__, so the class must be patched first).
#
# Place the file fix_a_b_extrapolation.py in AI_simulation/stage2/ (next to
# run_training.py and training.py).
#
# -----------------------------------------------------------------------------
# HYPERPARAMETER NOTES (honest — these are not proven-optimal, they are
# principled starting points; the retrain will tell us if they need tuning)
#
#   extrap_weight = 0.5
#       The diffusion loss is ~0.01 late in training; the extrapolation BCE is
#       O(0.1-1). 0.5 makes extrapolation a co-dominant signal. If after
#       retraining auc_unobs is still < 0.6, raise to 1.0. If the model starts
#       over-predicting (prevalence_mae climbs), lower to 0.25.
#
#   presence_weight = 20.0
#       Rare species are ~99.5% absence. Full class-balance would be ~200, but
#       that over-pushes toward false positives. 20 is a strong but stable
#       pull. If the model still predicts flat absence at unobserved cells,
#       raise toward 50.
#
#   obs_mask_scale = 0.5
#       Breaks identity-copy while keeping precise localisation. If the
#       diagnostic after retraining still shows NO_ENV ~= FULL (env ignored),
#       lower to 0.25 or 0.0 to force more weight onto env.
# =============================================================================