#!/usr/bin/env bash
# =============================================================================
# regenerate_all_inference_worlds.sh
# =============================================================================
# Runs v7 inpainting inference on ALL 18 inference worlds with:
#   - train-test mismatch FIX (--history-sparsify all_frames)
#   - K=5 and K=10 budgets
#   - 8 ensemble samples per species
#
# Prerequisites:
#   1. generate_reconstructions_v7_inpaint.py has the REV2 patch applied
#      (the --history-sparsify all_frames flag must exist)
#   2. pick_inference_worlds.py has been run and confirmed all 18 worlds exist
#   3. best_model.pt checkpoint is at the expected location
#
# Estimated runtime per world: ~15-25 minutes on a single GPU (K=5 + K=10).
# 18 worlds × 20 min = ~6 hours total. Run overnight.

set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(pwd)"
STAGE2_DIR="${PROJECT_ROOT}/AI_simulation/stage2"
CHECKPOINT="${PROJECT_ROOT}/stage2_outputs_new/checkpoints/best_model.pt"
TRUTH_DIR="${PROJECT_ROOT}/results/data"
OUT_BASE="${PROJECT_ROOT}/reconstructions_v7_inpaint"
RECON_SCRIPT="${STAGE2_DIR}/models/generate_reconstructions_v7_inpaint.py"

# ─── SAFETY CHECKS ─────────────────────────────────────────────────────
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "✗ Checkpoint missing: $CHECKPOINT"; exit 1
fi
if [[ ! -f "$RECON_SCRIPT" ]]; then
    echo "✗ generate_reconstructions_v7_inpaint.py missing: $RECON_SCRIPT"; exit 1
fi
if ! grep -q "history-sparsify" "$RECON_SCRIPT"; then
    echo "✗ generate_reconstructions_v7_inpaint.py does NOT have REV2 patch."
    echo "  It must contain the --history-sparsify flag."
    echo "  Apply REV2 first before running this script."
    exit 1
fi

# ─── WORLD LIST (must match pick_inference_worlds.py) ──────────────────
declare -a WORLDS=(
    # existing (10)
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr2em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr5em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env456_grid20x20_dr2em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env456_grid20x20_dr5em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env123_grid20x20_dr2em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env123_grid20x20_dr5em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr2em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr2em08_ld0p0_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr5em08_ld0p06_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr3p0_env456_grid20x20_dr5em08_ld0p0_training.npz"
    # new — hard band
    "pool22510000_batcha_ls2p5_vr0p004_thr1p0_env123_grid20x20_dr1em07_ld0p12_training.npz"
    "pool22510000_batcha_ls2p5_vr0p004_thr1p0_env456_grid20x20_dr1em07_ld0p12_training.npz"
    # new — connectivity
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env123_grid20x20_dr5em08_ld0p2_training.npz"
    "pool22510000_batcha_ls10p0_vr0p001_thr1p0_env456_grid20x20_dr5em08_ld0p2_training.npz"
    # new — sparse obs
    "pool22510000_batcha_ls10p0_vr0p001_thr5p0_env123_grid20x20_dr2em08_ld0p06_training.npz"
    # new — generalisation mid
    "pool22510000_batcha_ls5p0_vr0p002_thr1p0_env123_grid20x20_dr5em08_ld0p06_training.npz"
    "pool22510000_batcha_ls5p0_vr0p002_thr3p0_env456_grid20x20_dr5em08_ld0p06_training.npz"
    # new — wide-range stress
    "pool22510000_batcha_ls2p5_vr0p004_thr3p0_env456_grid20x20_dr1em07_ld0p2_training.npz"
)

echo ""
echo "==========================================================================="
echo "  REGENERATING INFERENCE — 18 worlds, K=5,10, all_frames sparsification"
echo "==========================================================================="
echo "  Checkpoint:    $CHECKPOINT"
echo "  Truth dir:     $TRUTH_DIR"
echo "  Output base:   $OUT_BASE"
echo "  Number worlds: ${#WORLDS[@]}"
echo "  Sparsify mode: all_frames (FIX for train/test mismatch)"
echo ""

# ─── RUN LOOP ──────────────────────────────────────────────────────────
SUCCESS=0
FAIL=0
SKIP=0
SECONDS=0

for i in "${!WORLDS[@]}"; do
    world="${WORLDS[$i]}"
    stem="${world%.npz}"
    truth_path="${TRUTH_DIR}/${world}"
    out_dir="${OUT_BASE}_${stem}_stage2"

    idx=$((i + 1))
    total=${#WORLDS[@]}
    echo ""
    echo "[${idx}/${total}] World: $world"
    echo "                  Out: $out_dir"

    if [[ ! -f "$truth_path" ]]; then
        echo "    ✗ Truth NPZ missing — skipping"
        SKIP=$((SKIP+1))
        continue
    fi

    # Skip if already done AND has both K=5 and K=10 outputs
    if [[ -f "${out_dir}/recon_fixed_b5_samples.npz" ]] && \
       [[ -f "${out_dir}/recon_fixed_b10_samples.npz" ]]; then
        # Verify it's the REV2 output by checking metadata
        if python -c "
import numpy as np, sys
d = np.load('${out_dir}/recon_fixed_b5_samples.npz', allow_pickle=True)
mode = str(d.get('history_sparsify', 'unknown'))
sys.exit(0 if mode == 'all_frames' else 1)
" 2>/dev/null; then
            echo "    ✓ Already done with all_frames — skipping"
            SUCCESS=$((SUCCESS+1))
            continue
        else
            echo "    ⚠ Old run found (last_only) — regenerating"
        fi
    fi

    mkdir -p "$out_dir"

    if python "$RECON_SCRIPT" \
        --stage2-dir         "$STAGE2_DIR" \
        --checkpoint         "$CHECKPOINT" \
        --truth-npz          "$truth_path" \
        --output-dir         "$out_dir" \
        --fixed-budgets      5 10 \
        --mode               inpaint \
        --repaint-iterations 2 \
        --n-ensemble         8 \
        --history-sparsify   all_frames \
        --verbose; then
        echo "    ✓ Success [${idx}/${total}]"
        SUCCESS=$((SUCCESS+1))
    else
        echo "    ✗ FAILED [${idx}/${total}]"
        FAIL=$((FAIL+1))
    fi
done

# ─── SUMMARY ───────────────────────────────────────────────────────────
ELAPSED=$SECONDS
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "==========================================================================="
echo "  COMPLETE"
echo "==========================================================================="
echo "  Success: $SUCCESS"
echo "  Failed:  $FAIL"
echo "  Skipped (missing truth): $SKIP"
echo "  Elapsed: ${HOURS}h ${MINS}m"
echo ""
echo "  Next steps:"
echo "    1. Update multi_world_K5_summary.csv and multi_world_K5_2x_summary.csv"
echo "       to include the new worlds:"
echo "         python multi_world_v7_evaluation.py [...args...] \\"
echo "             --top-n-worlds 18 [...]"
echo "         python multi_world_2x_evaluation.py [...args...] \\"
echo "             --top-n-worlds 18 [...]"
echo ""
echo "    2. Rerun the figures pipeline with the expanded world set:"
echo "         python make_figure1_honest_map.py [...args as before...]"
echo ""
echo "    3. Compare REV5 numbers (10 worlds, last_only) vs"
echo "       NEW numbers (18 worlds, all_frames) — the train/test fix should"
echo "       improve F1 substantially. The added worlds should provide"
echo "       hard-band power (n_hard ≥ 50)."