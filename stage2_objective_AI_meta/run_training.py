#!/usr/bin/env python3
"""
=============================================================================
RUN_TRAINING.PY  —  rebuilt for true 4-phase curriculum (Axel-aligned)
=============================================================================

WHAT CHANGED FROM YOUR PREVIOUS run_training.py
------------------------------------------------
The single most important fix: this script now creates dataloaders WITH the
right mode. Previously you had:

    train_loader, val_loader, test_loader, preprocessor = create_dataloaders(
        config.paths.simulation_dir, config, mode='equilibrium')

That call hard-coded mode='equilibrium' so condition only contained env +
coords. The model trained for 180+ epochs without ever seeing
species_features, edge_index, or history_P. The temp_encoder and
interaction_encoder never received gradients.

This rebuild creates FOUR dataloaders, one per curriculum phase, and the
trainer swaps to the appropriate loader at phase boundaries:

    Phase 1 (mode='equilibrium'):  env + coords
    Phase 2 (mode='interaction'):  env + coords + species_features + edges
    Phase 3 (mode='temporal'):     adds history_P / history_B
    Phase 4 (mode='infill'):       sparsifies last frame to K=5 (and Poisson)

WHAT ABOUT AXEL'S "PAST OBSERVATIONS REALISTICALLY SPARSE" POINT
-----------------------------------------------------------------
Your previous setup gave the model the FULL P_t history, with the last
frame equal to the target. As Axel pointed out:
    "if the final point of the history is just moments before the
     observation, and observations are reasonably good, then the
     overall result is not surprising"

This rebuild fixes that ecologically: in Phase 3 and Phase 4, EVERY frame
of the history is sparsified to K=5-10 observations per species, not just
the last frame. The model sees the same kind of evidence a real ecologist
would have: a few point records per species across time, never a full
distribution map. This is what makes the result publishable.

THE 4-PHASE CURRICULUM IN ECOLOGICAL TERMS
-------------------------------------------
Phase 1: "Where can each species live?" — pure habitat suitability
         (env + coords). Trains env_encoder.
Phase 2: "Who lives near whom?" — adds competitive interactions and
         species traits (prevalence, body mass). Trains interaction_encoder
         + species_embed.
Phase 3: "What did we see when?" — adds sparse temporal record sequences.
         Trains temp_encoder. ALL history frames are sparse — never the
         truth. This is the key Axel-aligned change.
Phase 4: "Reconstruct the full range from K=5 dots" — the IUCN AOO/EOO
         task. Sparsification budgets vary across batches (5, 10, 20, and
         abundance-weighted Poisson at p ∈ {1e-4, 5e-4, 1e-3}).

USAGE
-----
    python run_training.py \\
        --simulation-dir   results/data \\
        --output-dir       ./stage2_outputs_v2 \\
        --total-epochs     500 \\
        --batch-size       2 \\
        --lr               1e-4 \\
        --history-length   10 \\
        --hist-sparsify-K  5 10
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-dir", default=None,
                   help="Path containing models/, configs/. "
                        "Defaults to script's parent directory.")
    p.add_argument("--simulation-dir", required=True)
    p.add_argument("--output-dir", default="./stage2_outputs_v2")
    p.add_argument("--total-epochs", type=int, default=500)
    p.add_argument("--phase1-epochs", type=int, default=50)
    p.add_argument("--phase2-epochs", type=int, default=100)
    p.add_argument("--phase3-epochs", type=int, default=200)
    p.add_argument("--phase4-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--history-length", type=int, default=10,
                   help="Number of past snapshots fed to temp_encoder. "
                        "Default 10 (out of 50 available). Last-10 matches "
                        "Axel's 'truth | past observations | recon' framing.")
    p.add_argument("--hist-sparsify-K", type=int, nargs="+",
                   default=[5, 10],
                   help="Observation budgets per species per history frame "
                        "(used for ALL frames in phases 3-4, not just last)")
    p.add_argument("--target-mode", choices=['last', 'random'], default='last',
                   help="'last' = always P_last_final (Axel-aligned); "
                        "'random' = random snapshot (more diverse training)")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Make stage2 source tree importable ────────────────────────
    if args.stage2_dir:
        stage2_dir = Path(args.stage2_dir).resolve()
    else:
        stage2_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(stage2_dir))
    sys.path.insert(0, str(stage2_dir / "models"))
    sys.path.insert(0, str(stage2_dir / "configs"))

    from configs.config import get_default_config
    from models.ecodiffusion_spatial_cond import create_spatial_cond_model
    from data_preprocessing import create_dataloaders, save_preprocessor
    from training import Trainer
    # Force interaction_encoder to be imported as a top-level module
    # (ecodiffusion imports it as models.interaction_encoder, but our
    # patch needs the bare-name module to monkey-patch InteractionEncoder)
    import interaction_encoder  # noqa: F401
    # Install v2 patches AFTER training and data_preprocessing are imported.
    import data_preprocessing_v2_patch
    data_preprocessing_v2_patch.auto_install(
        history_length=args.history_length)
    import training_v2_patch  # auto-installs on import
    import interaction_encoder_v2_patch  # auto-installs on import

    # ---- Fix A + Fix B : extrapolation loss + obs_mask attenuation ----
    import training
    from models import ecodiffusion_spatial_cond
    import fix_a_b_extrapolation
    fix_a_b_extrapolation.install_all(
        training_module = training,
        spatial_module  = ecodiffusion_spatial_cond,
        extrap_weight   = 0.5,
        presence_weight = 20.0,
        obs_mask_scale  = 0.5,
    )

    # ── Set up paths and seeds ────────────────────────────────────
    output_path = Path(args.output_dir)
    (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_path / "logs").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Build config ──────────────────────────────────────────────
    config = get_default_config()
    config.paths.simulation_dir   = str(args.simulation_dir)
    config.paths.output_dir       = str(output_path)
    config.paths.checkpoint_dir   = str(output_path / "checkpoints")
    config.paths.log_dir          = str(output_path / "logs")
    config.training.batch_size    = args.batch_size
    config.training.learning_rate = args.lr
    config.training.total_epochs  = args.total_epochs
    config.training.phase1_epochs = args.phase1_epochs
    config.training.phase2_epochs = args.phase2_epochs
    config.training.phase3_epochs = args.phase3_epochs
    config.training.phase4_epochs = args.phase4_epochs

    # Pass our new training-protocol settings via attributes the trainer
    # will read (the trainer in training.py is rebuilt to honor these)
    config.training.history_length     = args.history_length
    config.training.hist_sparsify_K    = list(args.hist_sparsify_K)
    config.training.target_mode        = args.target_mode

    print("\n" + "=" * 72)
    print("  STAGE 2 TRAINING v2 — true 4-phase curriculum (Axel-aligned)")
    print("=" * 72)
    print(f"  device:           {device}")
    print(f"  simulation dir:   {args.simulation_dir}")
    print(f"  output dir:       {output_path.resolve()}")
    print(f"  total epochs:     {args.total_epochs}  "
          f"(phase boundaries {args.phase1_epochs}/{args.phase2_epochs}"
          f"/{args.phase3_epochs}/{args.phase4_epochs})")
    print(f"  batch size:       {args.batch_size}")
    print(f"  learning rate:    {args.lr}")
    print(f"  history length:   {args.history_length} frames "
          f"(last-{args.history_length} of 50)")
    print(f"  hist sparsify K:  {args.hist_sparsify_K}  "
          f"(applied to ALL history frames)")
    print(f"  target mode:      {args.target_mode}")

    # ── Build PHASE-SPECIFIC dataloaders ──────────────────────────
    # KEY FIX: previous code created one loader with mode='equilibrium'.
    # We now create four. The trainer swaps loaders at phase boundaries.
    print("\n  Building phase-specific dataloaders…")
    loaders = {}
    preprocessor = None
    for phase, mode in [(1, 'equilibrium'), (2, 'interaction'),
                        (3, 'temporal'), (4, 'infill')]:
        print(f"    phase {phase}: mode='{mode}'")
        train_l, val_l, test_l, prep = create_dataloaders(
            config.paths.simulation_dir, config, mode=mode)
        loaders[phase] = {'train': train_l, 'val': val_l, 'test': test_l}
        if preprocessor is None:
            preprocessor = prep
    save_preprocessor(preprocessor, str(output_path / 'preprocessor.pkl'))

    # ── Detect species count ──────────────────────────────────────
    sample = next(iter(loaders[1]['train']))
    n_species = sample['target'].shape[1]
    config.data.n_species_max = n_species
    print(f"  detected n_species = {n_species}")

    # Diagnostic: print conditioning keys at each phase
    print("\n  Conditioning keys per phase:")
    for phase in [1, 2, 3, 4]:
        s = next(iter(loaders[phase]['train']))
        keys = list(s.get('condition', {}).keys())
        print(f"    phase {phase}: {keys}")

    # ── Build model ───────────────────────────────────────────────
    print("\n  Building model…")
    model = create_spatial_cond_model(config).to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  resumed from {args.resume}")
        if missing:    print(f"    missing keys:    {len(missing)}")
        if unexpected: print(f"    unexpected keys: {len(unexpected)}")

    # ── Create trainer that knows about phase-specific loaders ──
    # We hand it the phase-1 loader as initial; the trainer's
    # train()/train_epoch() methods will swap loaders at phase boundaries
    # via the new self.phase_loaders attribute (see training.py rebuild).
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders[1]['train'],
        val_loader=loaders[1]['val'],
        device=device,
    )
    trainer.phase_loaders = loaders  # ← consumed by the rebuilt trainer

    # ── Train ─────────────────────────────────────────────────────
    print("\n  Starting training…")
    t0 = time.time()
    history = trainer.train(resume_from=args.resume)
    print(f"\n  Done in {(time.time()-t0)/3600:.1f} hours.")
    print(f"  Best checkpoint: {output_path}/checkpoints/best_model.pt")


if __name__ == "__main__":
    main()