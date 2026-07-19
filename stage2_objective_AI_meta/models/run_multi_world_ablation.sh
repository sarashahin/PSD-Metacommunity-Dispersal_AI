#!/usr/bin/env bash
# =============================================================================
# RUN_MULTI_WORLD_ABLATION.SH
# =============================================================================
#
# Runs the full multi-world ablation pipeline on three IBM-simulation worlds
# that span the parameter space (thr=1.0 vs 3.0, env=123 vs 456). Each world
# runs all 5 ablation variants sequentially. Total wall-time ≈ 3 hours of GPU
# (assuming world 5 is already done; ≈ 4.5 hours if starting from scratch).
#
# WORLDS SELECTED
# ---------------
#   WORLD 1   thr=1.0  env=123  dr=2e-08    (different threshold from world 5)
#   WORLD 5   thr=3.0  env=123  dr=2e-08    (already done; reused for parity)
#   WORLD 7   thr=3.0  env=456  dr=2e-08    (different env file)
#
# RESUME-SAFE
# -----------
# Skips a world if all 5 variant samples NPZs already exist. That means you
# can interrupt with Ctrl-C and re-run; only missing worlds will be redone.
#
# IF YOU HAVE MULTIPLE GPUs
# -------------------------
# Comment out the sequential loop and run each world in a separate terminal
# with `CUDA_VISIBLE_DEVICES=0`, `=1`, `=2`.
#
# USAGE
# -----
#   bash run_multi_world_ablation.sh
# =============================================================================

set -e
set -u

# ── PATHS ────────────────────────────────────────────────────────────────
REPO_ROOT="$(pwd)"
STAGE2_DIR="$REPO_ROOT/AI_simulation/stage2"
CHECKPOINT="$REPO_ROOT/stage2_outputs_new/checkpoints/best_model.pt"
DATA_DIR="$REPO_ROOT/results/data"
SCRIPT_DIR="$STAGE2_DIR/models"

# ── PARAMETERS ───────────────────────────────────────────────────────────
K=5
N_ENSEMBLE=8
DDIM_STEPS=50
ETA=0.5
REPAINT=2
SEED=42
VARIANTS="FULL NO_HISTORY NO_NETWORK NO_ENV NO_SPECIES_FEATS"

# ── WORLDS ───────────────────────────────────────────────────────────────
declare -a WORLDS=(
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr2em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env123_grid20x20_dr2em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr2em08_ld0p06_training.npz"
)

declare -a WORLD_LABELS=("world1" "world5" "world7")

# ── SANITY CHECKS ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  multi-world ablation pipeline"
echo "============================================================"
echo "  K=$K  ensemble=$N_ENSEMBLE  variants=$VARIANTS"
echo "  worlds:"
for i in "${!WORLDS[@]}"; do
    if [ ! -f "$DATA_DIR/${WORLDS[$i]}" ]; then
        echo "    ✗ ${WORLD_LABELS[$i]}: MISSING — $DATA_DIR/${WORLDS[$i]}"
        echo "    Cannot continue. Verify the data file exists."
        exit 1
    fi
    echo "    ✓ ${WORLD_LABELS[$i]}: ${WORLDS[$i]}"
done
echo ""

# ── STEP 1: ABLATION INFERENCE PER WORLD ────────────────────────────────
for i in "${!WORLDS[@]}"; do
    WORLD="${WORLDS[$i]}"
    LABEL="${WORLD_LABELS[$i]}"
    OUT_DIR="$REPO_ROOT/ablation_v7_${LABEL}_stage2_inpaint"

    # Resume-safe check: do we already have all 5 _samples NPZs?
    ALL_PRESENT=1
    for V in $VARIANTS; do
        FNAME="$OUT_DIR/recon_${V}_b${K}_samples.npz"
        if [ ! -f "$FNAME" ]; then
            ALL_PRESENT=0
            break
        fi
    done
    if [ "$ALL_PRESENT" -eq 1 ]; then
        echo ""
        echo "  ── ${LABEL}: SKIP (all 5 variants already present in $OUT_DIR)"
        continue
    fi

    echo ""
    echo "  ── ${LABEL}: running ablation (≈ 90 minutes)"
    python "$SCRIPT_DIR/run_ablation_v7.py" \
        --stage2-dir         "$STAGE2_DIR" \
        --checkpoint         "$CHECKPOINT" \
        --truth-npz          "$DATA_DIR/$WORLD" \
        --output-dir         "$OUT_DIR" \
        --K                  $K \
        --variants           $VARIANTS \
        --n-ensemble         $N_ENSEMBLE \
        --ddim-steps         $DDIM_STEPS \
        --eta                $ETA \
        --repaint-iterations $REPAINT \
        --rng-seed           $SEED \
        --verbose
done

echo ""
echo "============================================================"
echo "  multi-world inference DONE"
echo "============================================================"
echo "  Next step: run validate_ablation_v2.py and the visualizer"
echo "============================================================"