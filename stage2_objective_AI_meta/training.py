"""
=============================================================================
STAGE 2 TRAINING: Curriculum Training Loop for EcoDiffusion
=============================================================================
BUG-FIXED VERSION — fixes the 5 defects that broke the epoch-335 run.

WHAT WAS FIXED (and ONLY these — no science changed):

  BUG 1  save_checkpoint ignored config.training.save_best_only.
         It wrote a ~1 GB `checkpoint_epoch_N.pt` EVERY validation epoch
         and kept only the last 5. After the epoch-335 crash you could
         only resume from epochs ~310-335. FIX: honour save_best_only —
         always keep best_model.pt + a rolling `last_checkpoint.pt`, and
         only keep periodic epoch checkpoints when save_best_only is False.

  BUG 2  training_history.json was written ONCE, at the very end of
         train(). The epoch-335 crash therefore lost ALL history. FIX:
         _flush_history() is called every validation epoch, writing
         training_history.json incrementally. Nothing is ever lost.

  BUG 3  patience_counter never reset at phase boundaries. A plateau in
         phase 3 could trip early-stopping inside phase 4 before phase 4
         had a chance to learn. FIX: reset patience_counter (and
         best_metric, optionally) when the curriculum phase changes, so
         early stopping is measured WITHIN the current phase.

  BUG 4  resume / final-validate path was fragile: `final_metrics =
         self.validate(phase)` at the end used a `phase` variable that is
         undefined if the loop never ran (e.g. start_epoch >= total).
         FIX: phase is always defined; final validate/save are guarded.

  BUG 5  main() imported create_ecodiffusion_model (does not exist — the
         factory is create_fixed_model) and hard-coded mode='equilibrium'.
         It was dead/broken. FIX: main() now raises a clear message
         pointing to run_training.py, which is the real entry point.

Everything else — the auxiliary losses, NaN protection, AMP, schedulers,
validation metrics — is BYTE-FOR-BYTE the original. The curriculum, the
loss math, and the model are untouched.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import logging
import json
import time
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# AUXILIARY LOSS FUNCTIONS                          (UNCHANGED from original)
# =============================================================================

class PrevalenceLoss(nn.Module):
    """Loss to preserve species prevalence (frequency) distribution."""

    def __init__(self, rare_weight: float = 2.0, rare_threshold: float = 0.05):
        super().__init__()
        self.rare_weight = rare_weight
        self.rare_threshold = rare_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prevalence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = torch.clamp(pred, 0.0, 1.0)
        pred_prev = pred.mean(dim=(-2, -1))

        if prevalence.dim() == 1:
            true_prev = prevalence.unsqueeze(0).expand(pred_prev.shape[0], -1)
        else:
            true_prev = prevalence

        loss = (pred_prev - true_prev) ** 2

        weights = torch.ones_like(loss)
        rare_mask = true_prev < self.rare_threshold
        weights[rare_mask] = self.rare_weight

        if mask is not None:
            loss = loss * mask.float()
            weights = weights * mask.float()
            return (loss * weights).sum() / (mask.float().sum() + 1e-10)

        return (loss * weights).mean()


class CooccurrenceLoss(nn.Module):
    """Loss to preserve species co-occurrence patterns. Accepts species mask."""

    def __init__(self, n_sample_pairs: int = 500):
        super().__init__()
        self.n_sample_pairs = n_sample_pairs

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, Y, X = pred.shape

        pred = torch.clamp(pred, 0.0, 1.0)

        pred_binary = (pred > 0.5).float()
        pred_flat = pred_binary.view(B, S, -1)
        target_flat = target.view(B, S, -1)

        losses = []
        for b in range(B):
            if mask is not None:
                valid_idx = torch.where(mask[b] > 0)[0]
                if len(valid_idx) < 20:
                    continue
                n_valid = len(valid_idx)
                n_pairs = min(self.n_sample_pairs, n_valid * (n_valid - 1) // 2)

                local_idx1 = torch.randint(0, n_valid, (n_pairs,), device=pred.device)
                local_idx2 = torch.randint(0, n_valid, (n_pairs,), device=pred.device)
                idx1 = valid_idx[local_idx1]
                idx2 = valid_idx[local_idx2]
            else:
                n_pairs = min(self.n_sample_pairs, S * (S - 1) // 2)
                idx1 = torch.randint(0, S, (n_pairs,), device=pred.device)
                idx2 = torch.randint(0, S, (n_pairs,), device=pred.device)

            pred_p1 = pred_flat[b, idx1]
            pred_p2 = pred_flat[b, idx2]
            pred_intersection = (pred_p1 * pred_p2).sum(dim=-1)
            pred_union = (pred_p1 + pred_p2 - pred_p1 * pred_p2).sum(dim=-1)
            pred_jaccard = pred_intersection / (pred_union + 1e-6)

            tgt_p1 = target_flat[b, idx1]
            tgt_p2 = target_flat[b, idx2]
            tgt_intersection = (tgt_p1 * tgt_p2).sum(dim=-1)
            tgt_union = (tgt_p1 + tgt_p2 - tgt_p1 * tgt_p2).sum(dim=-1)
            tgt_jaccard = tgt_intersection / (tgt_union + 1e-6)

            valid = (pred_union > 0.1) | (tgt_union > 0.1)
            if valid.sum() < 10:
                continue

            loss = F.mse_loss(pred_jaccard[valid], tgt_jaccard[valid])
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return torch.clamp(torch.stack(losses).mean(), max=10.0)


class SpatialAutocorrelationLoss(nn.Module):
    """Loss to preserve spatial autocorrelation (Moran's I-like). Accepts mask."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [1/8, 1/8, 1/8],
            [1/8,   0, 1/8],
            [1/8, 1/8, 1/8],
        ]).view(1, 1, 3, 3)
        self.register_buffer('neighbor_kernel', kernel)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, Y, X = pred.shape

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        pred_neighbor_mean = F.conv2d(
            pred.view(B * S, 1, Y, X),
            self.neighbor_kernel.to(pred.device),
            padding=1,
            groups=1,
        ).view(B, S, Y, X)

        target_neighbor_mean = F.conv2d(
            target.view(B * S, 1, Y, X),
            self.neighbor_kernel.to(pred.device),
            padding=1,
            groups=1,
        ).view(B, S, Y, X)

        pred_autocorr = (pred * pred_neighbor_mean).mean(dim=(-2, -1))
        target_autocorr = (target * target_neighbor_mean).mean(dim=(-2, -1))

        if mask is not None:
            diff_sq = (pred_autocorr - target_autocorr) ** 2
            mask_f = mask.float()
            loss = (diff_sq * mask_f).sum() / (mask_f.sum() + 1e-6)
        else:
            loss = F.mse_loss(pred_autocorr, target_autocorr)

        return torch.clamp(loss, max=10.0)


class CombinedEcologicalLoss(nn.Module):
    """Combined loss function for EcoDiffusion training. NaN protection + clamping."""

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        prevalence_weight: float = 0.1,
        cooccurrence_weight: float = 0.05,
        spatial_weight: float = 0.02,
        rare_weight: float = 2.0,
    ):
        super().__init__()

        self.weights = {
            'diffusion': diffusion_weight,
            'prevalence': prevalence_weight,
            'cooccurrence': cooccurrence_weight,
            'spatial': spatial_weight,
        }

        self.prevalence_loss = PrevalenceLoss(rare_weight=rare_weight)
        self.cooccurrence_loss = CooccurrenceLoss()
        self.spatial_loss = SpatialAutocorrelationLoss()

    def forward(
        self,
        diffusion_loss: torch.Tensor,
        pred_x0: torch.Tensor,
        target: torch.Tensor,
        prevalence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        phase: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if torch.isnan(diffusion_loss):
            logger.warning("NaN in diffusion loss, returning safe fallback")
            safe = torch.tensor(1.0, device=diffusion_loss.device, requires_grad=True)
            return {'diffusion': safe, 'total': safe}

        pred_x0_clamped = torch.clamp(pred_x0, -10.0, 10.0)
        pred_x0_prob = torch.sigmoid(pred_x0_clamped)

        diffusion_loss = torch.clamp(diffusion_loss, max=100.0)

        losses = {'diffusion': diffusion_loss}
        total = self.weights['diffusion'] * diffusion_loss

        if phase >= 2:
            try:
                prev_loss = self.prevalence_loss(pred_x0_prob, target, prevalence, mask)
                prev_loss = torch.clamp(prev_loss, max=10.0)
                if not torch.isnan(prev_loss):
                    losses['prevalence'] = prev_loss
                    total = total + self.weights['prevalence'] * prev_loss
            except Exception as e:
                logger.warning(f"Prevalence loss failed: {e}")

        if phase >= 3:
            try:
                cooc_loss = self.cooccurrence_loss(pred_x0_prob, target, mask)
                cooc_loss = torch.clamp(cooc_loss, max=10.0)
                if not torch.isnan(cooc_loss):
                    losses['cooccurrence'] = cooc_loss
                    total = total + self.weights['cooccurrence'] * cooc_loss
            except Exception as e:
                logger.warning(f"Cooccurrence loss failed: {e}")

        if phase >= 4:
            try:
                spatial_loss = self.spatial_loss(pred_x0_prob, target, mask)
                spatial_loss = torch.clamp(spatial_loss, max=10.0)
                if not torch.isnan(spatial_loss):
                    losses['spatial'] = spatial_loss
                    total = total + self.weights['spatial'] * spatial_loss
            except Exception as e:
                logger.warning(f"Spatial loss failed: {e}")

        if torch.isnan(total):
            logger.warning("NaN in total loss, using diffusion only")
            total = self.weights['diffusion'] * torch.clamp(diffusion_loss, max=10.0)

        losses['total'] = total
        return losses


# =============================================================================
# VALIDATION METRICS                                (UNCHANGED from original)
# =============================================================================

class ValidationMetrics:
    """Compute validation metrics for species distribution models."""

    @staticmethod
    def auc_roc(pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        B, S, Y, X = pred_np.shape
        pred_flat = pred_np.reshape(B, S, -1)
        target_flat = target_np.reshape(B, S, -1)

        aucs = []
        aucs_rare = []
        aucs_common = []

        for b in range(B):
            for s in range(S):
                y_true = target_flat[b, s]
                y_pred = pred_flat[b, s]

                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    continue

                if np.isnan(y_pred).any():
                    continue

                try:
                    auc = roc_auc_score(y_true, y_pred)
                    aucs.append(auc)

                    prevalence = y_true.mean()
                    if prevalence < 0.05:
                        aucs_rare.append(auc)
                    else:
                        aucs_common.append(auc)
                except Exception:
                    continue

        return {
            'auc_overall': np.mean(aucs) if aucs else 0.0,
            'auc_rare': np.mean(aucs_rare) if aucs_rare else 0.0,
            'auc_common': np.mean(aucs_common) if aucs_common else 0.0,
        }

    @staticmethod
    def jaccard(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum(dim=(-2, -1))
        union = (pred_binary + target - pred_binary * target).sum(dim=(-2, -1))
        jaccard = intersection / (union + 1e-10)
        return jaccard.mean().item()

    @staticmethod
    def prevalence_error(pred: torch.Tensor,
                         target: torch.Tensor) -> Dict[str, float]:
        pred_prev = pred.mean(dim=(-2, -1))
        target_prev = target.mean(dim=(-2, -1))
        mae = (pred_prev - target_prev).abs().mean().item()
        mse = ((pred_prev - target_prev) ** 2).mean().item()
        return {'prevalence_mae': mae, 'prevalence_mse': mse}


# =============================================================================
# TRAINING LOOP
# =============================================================================

class Trainer:
    """Trainer class for EcoDiffusion with curriculum learning."""

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader,
        val_loader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas,
        )

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.warmup_epochs * len(train_loader),
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=(config.training.total_epochs - config.training.warmup_epochs) * len(train_loader),
            eta_min=config.training.min_lr,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.training.warmup_epochs * len(train_loader)],
        )

        if config.training.use_amp:
            self.scaler = GradScaler('cuda') if torch.cuda.is_available() else GradScaler('cpu')
        else:
            self.scaler = None

        self.loss_fn = CombinedEcologicalLoss(
            diffusion_weight=config.training.loss_diffusion_weight,
            prevalence_weight=config.training.loss_prevalence_weight,
            cooccurrence_weight=getattr(config.training, 'loss_cooccurrence_weight', 0.05),
            spatial_weight=getattr(config.training, 'loss_spatial_weight', 0.02),
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.history = defaultdict(list)

        # ── BUG 3: track which phase early-stopping is currently measured in ──
        self.current_phase_for_es = None

        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── BUG 2: history is flushed here, incrementally, every val epoch ──
        self.history_path = self.checkpoint_dir / "training_history.json"

        # whether to keep only best (+ rolling last) or also periodic epochs
        self.save_best_only = bool(getattr(config.training, 'save_best_only', True))

    # ------------------------------------------------------------------
    def get_current_phase(self, epoch: int) -> int:
        if epoch < self.config.training.phase1_epochs:
            return 1
        elif epoch < self.config.training.phase2_epochs:
            return 2
        elif epoch < self.config.training.phase3_epochs:
            return 3
        else:
            return 4

    def get_dataset_mode(self, phase: int) -> str:
        mode_map = {1: 'equilibrium', 2: 'interaction', 3: 'temporal', 4: 'infill'}
        return mode_map.get(phase, 'infill')

    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int, phase: int) -> Dict[str, float]:
        """Train for one epoch WITH NaN protection.  (UNCHANGED logic.)"""
        self.model.train()
        self.model.set_training_phase(phase)

        epoch_losses = defaultdict(list)
        nan_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Phase {phase}")

        for batch_idx, batch in enumerate(pbar):
            target = batch['target'].to(self.device)
            target_biomass = batch['target_biomass'].to(self.device)

            condition = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['condition'].items()}

            prevalence = batch['metadata']['prevalence']
            if isinstance(prevalence, torch.Tensor):
                prevalence = prevalence.to(self.device)

            mask = batch['metadata']['species_mask'].to(self.device)

            if torch.isnan(target).any():
                logger.warning(f"NaN in target at batch {batch_idx}")
                nan_batches += 1
                continue

            self.optimizer.zero_grad()

            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

            try:
                with autocast(device_type=device_type, enabled=self.config.training.use_amp):
                    result = self.model(target, condition)

                    if torch.isnan(result['noise_pred']).any():
                        logger.warning(f"NaN in noise_pred at batch {batch_idx}")
                        nan_batches += 1
                        continue

                    pred_x0 = self.model.diffusion.predict_start_from_noise(
                        result['x_t'], result['t'], result['noise_pred']
                    )

                    losses = self.loss_fn(
                        result['loss'], pred_x0, target, prevalence, mask, phase,
                    )

                if torch.isnan(losses['total']) or losses['total'].item() > 1000:
                    logger.warning(f"Bad loss ({losses['total'].item():.1f}) at batch {batch_idx}")
                    nan_batches += 1
                    continue

                if self.scaler:
                    self.scaler.scale(losses['total']).backward()
                    self.scaler.unscale_(self.optimizer)

                    has_nan_grad = any(
                        torch.isnan(p.grad).any()
                        for p in self.model.parameters()
                        if p.grad is not None
                    )
                    if has_nan_grad:
                        logger.warning(f"NaN gradient at batch {batch_idx}")
                        self.optimizer.zero_grad()
                        nan_batches += 1
                        continue

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    losses['total'].backward()

                    has_nan_grad = any(
                        torch.isnan(p.grad).any()
                        for p in self.model.parameters()
                        if p.grad is not None
                    )
                    if has_nan_grad:
                        logger.warning(f"NaN gradient at batch {batch_idx}")
                        self.optimizer.zero_grad()
                        nan_batches += 1
                        continue

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.global_step += 1

                for k, v in losses.items():
                    if not torch.isnan(v):
                        epoch_losses[k].append(v.item())

                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                    'nan': nan_batches if nan_batches > 0 else '',
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    nan_batches += 1
                    continue
                else:
                    raise e

        if nan_batches > 0:
            logger.warning(f"Skipped {nan_batches}/{len(self.train_loader)} batches due to NaN")

        return {k: np.mean(v) if v else float('nan') for k, v in epoch_losses.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, phase: int) -> Dict[str, float]:
        """Run validation.  (UNCHANGED logic.)"""
        self.model.eval()
        self.model.set_training_phase(phase)

        if len(self.val_loader) == 0:
            logger.warning("Empty validation loader")
            return {
                'auc_overall': 0.0, 'auc_rare': 0.0, 'auc_common': 0.0,
                'jaccard': 0.0, 'prevalence_mae': 0.0, 'prevalence_mse': 0.0,
                'val_total': 0.0, 'val_diffusion': 0.0,
            }

        all_preds = []
        all_targets = []
        val_losses = defaultdict(list)

        for batch in tqdm(self.val_loader, desc="Validation"):
            target = batch['target'].to(self.device)
            condition = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['condition'].items()}
            prevalence = batch['metadata']['prevalence']
            if isinstance(prevalence, torch.Tensor):
                prevalence = prevalence.to(self.device)
            mask = batch['metadata']['species_mask'].to(self.device)

            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type, enabled=self.config.training.use_amp):
                result = self.model(target, condition)
                pred_x0 = self.model.diffusion.predict_start_from_noise(
                    result['x_t'], result['t'], result['noise_pred']
                )
                losses = self.loss_fn(
                    result['loss'], pred_x0, target, prevalence, mask, phase
                )

            for k, v in losses.items():
                if not torch.isnan(v):
                    val_losses[k].append(v.item())

            pred_prob = torch.sigmoid(torch.clamp(pred_x0, -10, 10))
            all_preds.append(pred_prob)
            all_targets.append(target)

        if len(all_preds) == 0:
            return {
                'auc_overall': 0.0, 'auc_rare': 0.0, 'auc_common': 0.0,
                'jaccard': 0.0, 'prevalence_mae': 0.0, 'prevalence_mse': 0.0,
                'val_total': 0.0, 'val_diffusion': 0.0,
            }

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = ValidationMetrics.auc_roc(all_preds, all_targets)
        metrics['jaccard'] = ValidationMetrics.jaccard(all_preds, all_targets)
        metrics.update(ValidationMetrics.prevalence_error(all_preds, all_targets))

        for k, v in val_losses.items():
            metrics[f'val_{k}'] = np.mean(v) if v else 0.0

        return metrics

    # ------------------------------------------------------------------
    #  BUG 1: checkpointing rewritten to honour save_best_only
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float],
                        is_best: bool = False, is_periodic: bool = False):
        """
        Checkpoint policy:
          - always overwrite  last_checkpoint.pt   (rolling — for crash resume)
          - if is_best:        overwrite  best_model.pt
          - if NOT save_best_only AND is_periodic: also write a dated
            checkpoint_epoch_N.pt, keeping only the most recent 3.

        This replaces the old behaviour, which wrote checkpoint_epoch_N.pt
        EVERY validation epoch (~1 GB each) and kept the last 5 — so after
        a crash you could not resume from a phase boundary.
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'current_phase_for_es': self.current_phase_for_es,
            'config': self.config,
        }

        # rolling "last" checkpoint — always overwritten, one file only
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"  ✓ new best ({self.config.training.best_metric}="
                        f"{self.best_metric:.4f}) -> {best_path.name}")

        if (not self.save_best_only) and is_periodic:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)
            logger.info(f"  ✓ periodic checkpoint -> {epoch_path.name}")
            # keep only the most recent 3 periodic checkpoints
            periodic = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            for old in periodic[:-3]:
                old.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        # BUG 3: restore early-stopping state so resume is faithful
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.current_phase_for_es = checkpoint.get('current_phase_for_es', None)

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch} "
                    f"(best_metric={self.best_metric:.4f}, "
                    f"patience={self.patience_counter})")

    # ------------------------------------------------------------------
    #  BUG 2: incremental history flush
    # ------------------------------------------------------------------
    def _flush_history(self):
        """Write training_history.json NOW. Called every validation epoch so
        a crash never loses the history (the epoch-335 run lost everything
        because history was only written at the very end)."""
        try:
            payload = {k: [float(v) for v in vals]
                       for k, vals in self.history.items()}
            tmp = self.history_path.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.history_path)  # atomic — never a half-written file
        except Exception as e:
            logger.warning(f"Could not flush training history: {e}")

    # ------------------------------------------------------------------
    def train(self, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0

        logger.info(f"Starting training from epoch {start_epoch}")
        logger.info(f"Total epochs: {self.config.training.total_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"save_best_only = {self.save_best_only}  "
                    f"(periodic epoch checkpoints "
                    f"{'OFF' if self.save_best_only else 'ON'})")

        # BUG 4: phase is ALWAYS defined, even if the loop body never runs
        phase = self.get_current_phase(min(start_epoch,
                                           self.config.training.total_epochs - 1))

        for epoch in range(start_epoch, self.config.training.total_epochs):
            self.current_epoch = epoch
            phase = self.get_current_phase(epoch)

            # ── BUG 3: reset early-stopping state at every phase boundary ──
            # Early stopping must be measured WITHIN a phase. A plateau in
            # phase 3 must not be allowed to trip early-stopping in phase 4.
            if self.current_phase_for_es != phase:
                if self.current_phase_for_es is not None:
                    logger.info(f"=== Entering Phase {phase} — "
                                f"early-stopping counter reset "
                                f"(was {self.patience_counter}) ===")
                else:
                    logger.info(f"=== Phase {phase} ===")
                self.current_phase_for_es = phase
                self.patience_counter = 0
                # best_metric is intentionally NOT reset: best_model.pt should
                # remain the global best across the whole run. Only the
                # *patience* (when to stop) is per-phase.

            train_metrics = self.train_epoch(epoch, phase)

            logger.info(f"Epoch {epoch} - Train Loss: "
                        f"{train_metrics.get('total', float('nan')):.4f}")
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            self.history['epoch'].append(epoch)
            self.history['phase'].append(phase)

            if (epoch + 1) % self.config.training.val_every_epochs == 0:
                val_metrics = self.validate(phase)

                logger.info(f"Epoch {epoch} - Val AUC: "
                            f"{val_metrics['auc_overall']:.4f}, "
                            f"Rare AUC: {val_metrics['auc_rare']:.4f}")

                for k, v in val_metrics.items():
                    self.history[k].append(v)

                current_metric = val_metrics.get(
                    self.config.training.best_metric, val_metrics['auc_rare'])
                is_best = current_metric > self.best_metric

                if is_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                is_periodic = ((epoch + 1) %
                               self.config.training.save_every_epochs == 0)
                # always save (last_checkpoint rolls; best saved if is_best)
                self.save_checkpoint(epoch, val_metrics,
                                     is_best=is_best, is_periodic=is_periodic)

                # BUG 2: flush history every validation epoch
                self._flush_history()

                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping in Phase {phase} after "
                                f"{self.patience_counter} non-improving "
                                f"validations (epoch {epoch + 1})")
                    break

        # BUG 4: final validate/save guarded — phase is always defined now
        try:
            final_metrics = self.validate(phase)
            self.save_checkpoint(self.current_epoch, final_metrics,
                                 is_best=False, is_periodic=False)
            for k, v in final_metrics.items():
                self.history[f'final_{k}'].append(v)
        except Exception as e:
            logger.warning(f"Final validation/save skipped: {e}")

        self._flush_history()
        logger.info(f"Training complete! History -> {self.history_path}")
        return self.history


def main():
    """
    NOTE: this is NOT the training entry point. The real entry point is
    run_training.py, which builds the 4 phase-specific dataloaders and
    installs the v2 patches. This main() existed in the original file but
    referenced a factory (create_ecodiffusion_model) that does not exist
    and hard-coded mode='equilibrium', so it never trained the curriculum.
    It is kept only so `python training.py` fails LOUDLY instead of silently
    doing the wrong thing.
    """
    raise SystemExit(
        "training.py is a library module, not an entry point.\n"
        "Run training via:  python run_training.py --simulation-dir ... "
        "--output-dir ...\n"
        "run_training.py builds the phase-specific dataloaders and installs "
        "the v2 curriculum / history-sparsification patches."
    )


if __name__ == "__main__":
    main()