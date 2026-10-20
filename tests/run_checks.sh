#!/usr/bin/env bash
# Quick end-to-end check of the Stage-2 code on tiny synthetic worlds.
# No real data or GPU needed. Takes a few minutes on CPU.
#
# Usage (from anywhere):  bash tests/run_checks.sh [work_dir]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
S2="$REPO/stage2_objective_AI_meta"
WORK="${1:-$(mktemp -d)}"
PY="${PYTHON:-python}"
export MPLBACKEND=Agg

mkdir -p "$WORK"
cd "$WORK"
echo "work dir: $WORK"

echo "== 1/4 toy worlds"
"$PY" "$REPO/tests/make_toy_worlds.py" --out-dir "$WORK/toy_data"

echo "== 2/4 smoke_test_spatial.py"
"$PY" "$S2/smoke_test_spatial.py" --simulation-dir "$WORK/toy_data" 2>&1 \
    | tee "$WORK/smoke.log" | grep -E "^\s*(OK|\[[0-9]\])|PASSED|FAIL|Error"
grep -q "ALL SMOKE CHECKS PASSED" "$WORK/smoke.log"

echo "== 3/4 verify_fix_a_fires.py"
"$PY" "$S2/verify_fix_a_fires.py" --simulation-dir "$WORK/toy_data" 2>&1 \
    | tee "$WORK/fixa.log" | grep -E "VERIFIED|FAIL|Error" || true
grep -q "VERIFIED" "$WORK/fixa.log"

echo "== 4/4 reconstruction (K=5, 2 samples, 4 DDIM steps)"
cd "$S2/models"
"$PY" generate_reconstructions_spatial.py \
    --checkpoint "$WORK/smoke_out/checkpoints/best_model.pt" \
    --truth-npz "$WORK/toy_data/toy_world_07_training.npz" \
    --output-dir "$WORK/recon" \
    --fixed-budgets 5 --n-ensemble 2 --mode inpaint --ddim-steps 4 --eta 0.15 \
    > "$WORK/recon.log" 2>&1
test -f "$WORK/recon/recon_fixed_b5_samples.npz"

echo "ALL CHECKS PASSED"
