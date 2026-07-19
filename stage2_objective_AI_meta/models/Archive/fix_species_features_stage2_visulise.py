#!/usr/bin/env python3
"""
FIX 6 PATCH: Fix species_features shape from (B*S, 8) to (B, S, 8)
Run this from the same directory as stage2_real_inference_validation.py:

    python fix_species_features.py

Or specify the path:
    python fix_species_features.py /path/to/stage2_real_inference_validation.py
"""
import sys
from pathlib import Path

# Find the file to patch
if len(sys.argv) > 1:
    target = Path(sys.argv[1])
else:
    target = Path("AI_simulation/stage2/models/stage2_real_inference_validation.py")
    if not target.exists():
        target = Path("stage2_real_inference_validation.py")

if not target.exists():
    print(f"❌ File not found: {target}")
    print("   Usage: python fix_species_features.py /path/to/stage2_real_inference_validation.py")
    sys.exit(1)

print(f"📄 Patching: {target}")
code = target.read_text()

# ══════════════════════════════════════════════════════════════
# FIX 6: species_features must be (B, S, 8) NOT (B*S, 8)
#
# WHY: The GNN (InteractionEncoder) checks:
#     has_batch = species_features.dim() == 3
# With 2D (S, 8) input → returns (S, D) = 2D
# With 3D (B, S, 8) input → returns (B, S, D) = 3D
#
# The conditioning forward needs ALL tensors to be 3D (B, S, D)
# for torch.cat at line 161 of ecodiffusion.py.
# ══════════════════════════════════════════════════════════════

OLD_ALLOC = "sp_feats = np.zeros((B * S, 8), dtype=np.float32)"
NEW_ALLOC = "sp_feats = np.zeros((B, S, 8), dtype=np.float32)  # FIX 6: must be 3D (B,S,8)"

OLD_COMMENT = "# ── SPECIES FEATURES: (B*S, 8) for GNN ──"
NEW_COMMENT = "# ── SPECIES FEATURES: (B, S, 8) for GNN ──  [FIX 6]"

# Fix allocation: (B*S, 8) → (B, S, 8)
if OLD_ALLOC in code:
    code = code.replace(OLD_ALLOC, NEW_ALLOC, 1)
    print("   ✓ Fixed: sp_feats allocation (B*S,8) → (B,S,8)")
else:
    if "sp_feats = np.zeros((B, S, 8)" in code:
        print("   ℹ Already fixed: sp_feats allocation")
    else:
        print("   ⚠ Could not find sp_feats allocation line")

# Fix comment
if OLD_COMMENT in code:
    code = code.replace(OLD_COMMENT, NEW_COMMENT, 1)

# Fix indexing: sp_feats[s, ...] → sp_feats[0, s, ...]
# These are the 8 feature assignment lines
replacements = [
    ("sp_feats[s, 0] = pv[s]",                    "sp_feats[0, s, 0] = pv[s]"),
    ("sp_feats[s, 1] = P[s].sum() / (Y * X)",     "sp_feats[0, s, 1] = P[s].sum() / (Y * X)"),
    ("sp_feats[s, 2] = env_raw[s].mean()",         "sp_feats[0, s, 2] = env_raw[s].mean()"),
    ("sp_feats[s, 3] = env_raw[s].std()",          "sp_feats[0, s, 3] = env_raw[s].std()"),
    ("sp_feats[s, 4] = float(pv[s] < 0.05)",      "sp_feats[0, s, 4] = float(pv[s] < 0.05)"),
    ("sp_feats[s, 5] = float(pv[s] >= 0.05)",     "sp_feats[0, s, 5] = float(pv[s] >= 0.05)"),
    ("sp_feats[s, 6] = np.log1p(pv[s])",          "sp_feats[0, s, 6] = np.log1p(pv[s])"),
    ("sp_feats[s, 7] = float(s) / max(S - 1, 1)", "sp_feats[0, s, 7] = float(s) / max(S - 1, 1)"),
]

fixed_count = 0
for old, new in replacements:
    if old in code:
        code = code.replace(old, new, 1)
        fixed_count += 1

if fixed_count == 8:
    print(f"   ✓ Fixed: all {fixed_count} sp_feats indexing lines [s,i] → [0,s,i]")
elif fixed_count > 0:
    print(f"   ⚠ Fixed {fixed_count}/8 indexing lines (some may already be patched)")
else:
    if "sp_feats[0, s, 0]" in code:
        print("   ℹ Already fixed: sp_feats indexing")
    else:
        print("   ⚠ Could not find sp_feats indexing lines")

# Fix condition dict comment
OLD_COND = '"species_features": species_features_t,  # (B*S, 8)'
NEW_COND = '"species_features": species_features_t,  # (B, S, 8) — FIX 6: must be 3D'
if OLD_COND in code:
    code = code.replace(OLD_COND, NEW_COND, 1)
    print("   ✓ Fixed: condition dict comment")

# Write back
target.write_text(code)
print(f"\n✅ Patch applied to {target}")
print(f"   species_features will now be (1, S, 8) instead of (S, 8)")
print(f"   This ensures GNN returns 3D tensors → conditioning cat works")
print(f"\n   Re-run your command now!")