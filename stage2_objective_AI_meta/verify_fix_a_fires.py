#!/usr/bin/env python3
"""Standalone proof that Fix A's extrapolation loss actually FIRES in phase 4.
   Runs independently — does NOT touch the running training job.
   Usage: python AI_simulation/stage2/verify_fix_a_fires.py --simulation-dir results/data"""
import argparse, sys, traceback
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation-dir", required=True)
    ap.add_argument("--stage2-dir", default=None)
    a = ap.parse_args()
    s2 = Path(a.stage2_dir).resolve() if a.stage2_dir else Path(__file__).resolve().parent
    for p in (s2, s2/"models", s2/"configs"): sys.path.insert(0, str(p))

    from configs.config import get_default_config
    from models.ecodiffusion_spatial_cond import create_spatial_cond_model
    from data_preprocessing import create_dataloaders
    from training import Trainer
    import interaction_encoder, data_preprocessing_v2_patch
    import training_v2_patch, interaction_encoder_v2_patch
    data_preprocessing_v2_patch.auto_install(history_length=10)

    # install Fix A+B exactly as run_training.py does
    import training
    from models import ecodiffusion_spatial_cond
    import fix_a_b_extrapolation
    fix_a_b_extrapolation.install_all(training, ecodiffusion_spatial_cond,
                                      extrap_weight=0.5, presence_weight=20.0,
                                      obs_mask_scale=0.5)

    assert training.CombinedEcologicalLoss._fix_a_installed, "Fix A class patch missing"
    assert training.Trainer._fix_a_trainer_installed, "Fix A trainer patch missing"
    assert ecodiffusion_spatial_cond.EcoDiffusionSpatial._fix_b_installed, "Fix B missing"
    print("[1] OK  Fix A + Fix B installed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_default_config()
    cfg.paths.simulation_dir = a.simulation_dir
    cfg.training.batch_size = 1
    cfg.training.hist_sparsify_K = [5, 10, 20]

    tl, vl, _, _ = create_dataloaders(a.simulation_dir, cfg, mode='infill')
    batch = next(iter(tl))
    cfg.data.n_species_max = batch['target'].shape[1]
    model = create_spatial_cond_model(cfg).to(device)
    print("[2] OK  phase-4 (infill) batch + model built")

    # build a real Trainer so the Fix A trainer-patch (condition capture) is live
    tr = Trainer(model, cfg, tl, vl, device)
    tr.phase_loaders = {4: {'train': tl, 'val': vl}}

    # run ONE phase-4 training epoch on a 2-world subset (fast)
    import itertools
    class _Sub:
        def __init__(self, loader, n): self.loader, self.n = loader, n
        def __iter__(self): return itertools.islice(iter(self.loader), self.n)
        def __len__(self): return self.n
    tr.train_loader = _Sub(tl, 2)
    tr.val_loader = _Sub(vl, 1)

    print("[3] running ONE phase-4 epoch (2 worlds)...")
    metrics = tr.train_epoch(epoch=0, phase=4)
    print(f"    train_epoch returned keys: {list(metrics.keys())}")

    # THE DECISIVE CHECK
    if 'extrapolation' in metrics:
        print(f"\n[4] OK  >>> Fix A FIRES <<<  extrapolation loss = "
              f"{metrics['extrapolation']:.4f}")
        print("    The extrapolation term is active in phase 4. The running")
        print("    training job WILL apply Fix A at epoch 70. SAFE TO CONTINUE.")
    else:
        print(f"\n[4] FAIL  >>> Fix A did NOT fire <<<")
        print("    No 'extrapolation' key in phase-4 loss. The condition-capture")
        print("    is broken — obs_mask arrived as None and the term was skipped.")
        print("    KILL the running training job — it will waste 70 epochs.")
        sys.exit(1)

    print("\n" + "="*60)
    print("  VERIFIED: Fix A fires in phase 4. Running job is correct.")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("\n  VERIFICATION FAILED — investigate before trusting the run")
        sys.exit(1)