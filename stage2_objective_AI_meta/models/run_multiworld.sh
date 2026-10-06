#!/bin/bash
# =============================================================================
# RUN MULTI-WORLD V7 INPAINTING + AGGREGATION
# =============================================================================
#
# This script runs v7 inpainting inference on the top 10 worlds (by
# wide-range species count) from your wide_range_species.csv, then runs
# the multi-world aggregation script.
#
# Total compute time: ~5 hours on your GPU (30 min per world × 10 worlds)
# You can leave it running overnight.
#
# To use:
#   1. Save this file as run_multiworld.sh in your project root:
#         /home/sara/Downloads/The-Lotka-Volterra-Metacommunity-Model-main\ \(1\)/
#         The-Lotka-Volterra-Metacommunity-Model-main/PSD_Dispersal_pool/
#   2. Make it executable:
#         chmod +x run_multiworld.sh
#   3. Run it:
#         ./run_multiworld.sh
#
# If a world's inference is interrupted you can re-run this script —
# it skips worlds whose output already exists.
# =============================================================================

set -e   # stop if any command fails

# ── configuration ──
CHECKPOINT="./stage2_outputs_new/checkpoints/best_model.pt"
DATA_DIR="./results/data"
FIGURES_DIR="./figures_map_axel_stage2_new"

# The 10 worlds with the most wide-range species
# (these come from your wide_range_species.csv top-10)
WORLDS=(
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr2em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr5em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env456_grid20x20_dr2em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env456_grid20x20_dr5em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env123_grid20x20_dr2em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env123_grid20x20_dr5em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr2em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr2em08_ld0p0_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr5em08_ld0p06_training"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr5em08_ld0p0_training"
)

mkdir -p "$FIGURES_DIR"

echo "================================================================"
echo "  MULTI-WORLD V7 INPAINTING (10 worlds, ~5h on GPU)"
echo "================================================================"
echo ""

# ── PART 1: run inference on each world ──
COUNTER=0
for world_stem in "${WORLDS[@]}"; do
    COUNTER=$((COUNTER+1))
    OUT_DIR="./reconstructions_v7_inpaint_${world_stem}_stage2"
    SAMPLES_FILE="${OUT_DIR}/recon_fixed_b5_samples.npz"

    echo "----------------------------------------------------------------"
    echo "  WORLD ${COUNTER} / 10: ${world_stem}"
    echo "  output: ${OUT_DIR}"
    echo "----------------------------------------------------------------"

    # Skip if output already exists
    if [ -f "$SAMPLES_FILE" ]; then
        echo "  ✓ already done — skipping"
        echo ""
        continue
    fi

    # Check the truth NPZ exists
    TRUTH_NPZ="${DATA_DIR}/${world_stem}.npz"
    if [ ! -f "$TRUTH_NPZ" ]; then
        echo "  ✗ truth file missing: $TRUTH_NPZ"
        echo "  skipping this world"
        echo ""
        continue
    fi

    # Run inference
    python AI_simulation/stage2/models/generate_reconstructions_v7_inpaint.py \
        --stage2-dir         AI_simulation/stage2 \
        --checkpoint         "$CHECKPOINT" \
        --truth-npz          "$TRUTH_NPZ" \
        --output-dir         "$OUT_DIR" \
        --fixed-budgets      5 \
        --mode               inpaint \
        --repaint-iterations 2 \
        --n-ensemble         8

    echo ""
done

# ── PART 2: aggregate across worlds ──
echo "================================================================"
echo "  AGGREGATING ACROSS ALL WORLDS"
echo "================================================================"
echo ""

python AI_simulation/stage2/models/multi_world_v7_evaluation.py \
    --wide-range-csv    "${FIGURES_DIR}/wide_range_species.csv" \
    --recon-dir-pattern './reconstructions_v7_inpaint_{world_stem}_stage2' \
    --truth-dir         "$DATA_DIR" \
    --K                 5 \
    --top-n-worlds      10 \
    --output-csv        "${FIGURES_DIR}/multi_world_K5_summary.csv" \
    --calibrate         match_truth

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "================================================================"
echo ""
echo "  Summary CSV: ${FIGURES_DIR}/multi_world_K5_summary.csv"
echo "  Per-world reconstructions: ./reconstructions_v7_inpaint_*_stage2/"
echo ""