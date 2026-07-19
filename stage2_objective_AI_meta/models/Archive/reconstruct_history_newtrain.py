#!/usr/bin/env python3
"""
=============================================================================
BUILD TRAINING HISTORY  (v2 — robust loading)
=============================================================================

Builds training_history.json from checkpoint metrics + optional terminal log.

WHAT'S NEW IN v2
----------------
1. Auto-injects stage2 dir into sys.path so unpickling can find
   'configs' and 'models' modules. The previous version crashed with
   'No module named configs' because torch.load tries to deserialize
   the config object stored in each checkpoint.

2. Falls back to a "metrics-only" load when full unpickling fails.
   We only need 'epoch', 'global_step', 'best_metric', and 'metrics' —
   we don't need the model state, the optimizer state, or the config.
   So we use pickle's persistent_load=None and skip the config field
   if it can't be deserialized.

3. Doesn't crash when no checkpoints could be loaded — it just prints
   a clear error and exits with code 1.

USAGE
-----
    python build_training_history_v2.py \\
        --checkpoint-dir stage2_outputs_new/checkpoints \\
        --output         stage2_outputs_new/logs/training_history.json

    # Optionally also include terminal log for train-loss curve:
    python build_training_history_v2.py \\
        --checkpoint-dir stage2_outputs_new/checkpoints \\
        --terminal-log   terminal_log.txt \\
        --output         stage2_outputs_new/logs/training_history.json

The --stage2-dir flag is auto-detected if you put this script in
AI_simulation/stage2/models/ but you can override it.
=============================================================================
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _setup_imports(stage2_dir):
    """
    Add stage2 source tree to sys.path so checkpoints' pickled
    config objects can be deserialized.
    """
    stage2 = Path(stage2_dir).resolve()
    for p in [str(stage2), str(stage2 / 'configs'), str(stage2 / 'models')]:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_checkpoint_metrics_robust(ckpt_dir):
    """
    Extract val metrics from each .pt file. Tries three strategies in order:

      1. torch.load with weights_only=False (full object, needs imports)
      2. torch.load with weights_only=True  (PyTorch 2.4+ safe mode)
      3. raw pickle scan via custom unpickler that returns None for
         unknown classes — recovers metrics dict even when config can't
         be imported

    Returns list of dicts, one per loadable checkpoint.
    """
    import torch
    import pickle
    import io

    records = []
    for p in sorted(Path(ckpt_dir).glob('*.pt')):
        c = None

        # Strategy 1: full load
        try:
            c = torch.load(p, map_location='cpu', weights_only=False)
        except Exception as e1:
            # Strategy 2: weights-only (PyTorch 2.4+)
            try:
                c = torch.load(p, map_location='cpu', weights_only=True)
            except Exception:
                # Strategy 3: tolerant unpickler
                c = _tolerant_load(p)
                if c is None:
                    print(f"  ⚠ {p.name}: all 3 load strategies failed "
                          f"(last error: {e1})", file=sys.stderr)
                    continue

        if not isinstance(c, dict):
            print(f"  ⚠ {p.name}: not a dict, skipping", file=sys.stderr)
            continue

        raw = c.get('metrics', {}) or {}
        metrics = {}
        for k, v in raw.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                metrics[k] = None

        ep = c.get('epoch')
        gs = c.get('global_step')
        bm = c.get('best_metric')

        records.append({
            'file':        p.name,
            'epoch':       int(ep) if ep is not None else -1,
            'global_step': int(gs) if gs is not None else None,
            'best_metric': float(bm) if bm is not None else None,
            'metrics':     metrics,
        })
        print(f"  ✓ {p.name}: epoch={ep}, "
              f"auc_overall={metrics.get('auc_overall')}", file=sys.stderr)

    return records


def _tolerant_load(path):
    """
    Last-resort loader: read the pickle stream and replace any class
    that can't be imported with None. Recovers metrics from checkpoints
    that pickled a config object whose module isn't on sys.path.
    """
    import pickle
    import torch
    import io

    class TolerantUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError, ModuleNotFoundError):
                # Return a stand-in so unpickling continues
                return type(name, (object,), {'__init__': lambda self, *a, **kw: None})

    # Need to handle PyTorch's storage system separately
    try:
        with open(path, 'rb') as f:
            return torch.load(f, map_location='cpu', weights_only=False,
                              pickle_module=type('_pm', (), {
                                  'Unpickler': TolerantUnpickler,
                                  'dump': pickle.dump,
                                  'load': pickle.load,
                              }))
    except Exception:
        return None


def parse_terminal_log(log_path):
    train_loss_re = re.compile(
        r'Epoch\s+(\d+)\s*-\s*Train Loss:\s*([\d.]+)')
    val_auc_re = re.compile(
        r'Epoch\s+(\d+)\s*-\s*Val AUC:\s*([\d.]+),\s*Rare AUC:\s*([\d.]+)')
    phase_re = re.compile(r'===\s*Entering Phase\s+(\d+)\s*===')

    train_losses = []
    val_aucs    = []
    phases      = []
    seen = set()
    last_epoch = -1

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = train_loss_re.search(line)
            if m:
                ep, loss = int(m.group(1)), float(m.group(2))
                if ep not in seen:
                    train_losses.append([ep, loss])
                    seen.add(ep)
                    last_epoch = max(last_epoch, ep)
                continue
            m = val_auc_re.search(line)
            if m:
                val_aucs.append([int(m.group(1)),
                                 float(m.group(2)), float(m.group(3))])
                continue
            m = phase_re.search(line)
            if m:
                phases.append([last_epoch, int(m.group(1))])
                continue

    return train_losses, val_aucs, phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--terminal-log', default=None)
    ap.add_argument('--output', required=True)
    ap.add_argument('--stage2-dir', default=None,
                    help='Path to AI_simulation/stage2 (auto-detected if '
                         'this script lives in stage2/models/)')
    args = ap.parse_args()

    # ── Auto-detect stage2 directory ──
    if args.stage2_dir:
        stage2 = Path(args.stage2_dir)
    else:
        # If this script is in AI_simulation/stage2/models/foo.py,
        # then stage2 = parent.parent
        here = Path(__file__).resolve().parent
        stage2 = here.parent if (here.parent / 'configs').exists() else here
        # Common-case fallback: look in cwd
        if not (stage2 / 'configs').exists():
            for cand in [Path.cwd(), Path.cwd() / 'AI_simulation' / 'stage2']:
                if (cand / 'configs').exists():
                    stage2 = cand
                    break

    if (stage2 / 'configs').exists():
        _setup_imports(stage2)
        print(f"Using stage2 source tree: {stage2}", file=sys.stderr)
    else:
        print(f"⚠ stage2 source tree not found "
              f"(checked {stage2}); will rely on tolerant loader",
              file=sys.stderr)

    # ── Load checkpoints ──
    print(f"\nLoading checkpoint metrics from {args.checkpoint_dir} …",
          file=sys.stderr)
    ckpt_records = load_checkpoint_metrics_robust(args.checkpoint_dir)
    print(f"\n  loaded {len(ckpt_records)} checkpoints", file=sys.stderr)

    if not ckpt_records:
        print("\n✗ No checkpoints could be loaded. Try running with "
              "--stage2-dir AI_simulation/stage2", file=sys.stderr)
        sys.exit(1)

    # ── Optional: parse terminal log ──
    train_losses, log_val_aucs, phases = [], [], []
    if args.terminal_log:
        log_path = Path(args.terminal_log)
        if log_path.exists() and log_path.stat().st_size > 0:
            print(f"\nParsing terminal log {log_path} …", file=sys.stderr)
            train_losses, log_val_aucs, phases = parse_terminal_log(log_path)
            print(f"  found {len(train_losses)} train_loss entries, "
                  f"{len(log_val_aucs)} val entries, "
                  f"{len(phases)} phase transitions", file=sys.stderr)
        else:
            print(f"  ⚠ terminal log {log_path} empty/missing — skipping",
                  file=sys.stderr)

    # ── Build val curve from checkpoints ──
    val_curve = []
    for r in ckpt_records:
        m = r['metrics'] or {}
        if r['epoch'] >= 0 and m.get('auc_overall') is not None:
            val_curve.append({
                'epoch':          r['epoch'],
                'auc_overall':    m['auc_overall'],
                'auc_rare':       m.get('auc_rare'),
                'auc_common':     m.get('auc_common'),
                'jaccard':        m.get('jaccard'),
                'prevalence_mae': m.get('prevalence_mae'),
                'val_total':      m.get('val_total'),
                'source':         'checkpoint:' + r['file'],
            })

    best = max(val_curve, key=lambda x: x['auc_overall'] or 0, default=None)

    history = {
        'summary': {
            'n_checkpoints':          len(ckpt_records),
            'n_train_loss_entries':   len(train_losses),
            'n_val_entries_from_log': len(log_val_aucs),
            'best_val_auc':           best['auc_overall'] if best else None,
            'best_val_auc_epoch':     best['epoch'] if best else None,
            'best_val_auc_source':    best['source'] if best else None,
        },
        'train_loss':            train_losses,
        'val_curve_checkpoints': val_curve,
        'val_curve_log':         [{'epoch': e, 'auc_overall': a, 'auc_rare': r}
                                  for e, a, r in log_val_aucs],
        'phase_transitions':     [{'after_epoch': e, 'new_phase': p}
                                  for e, p in phases],
        'checkpoint_records':    ckpt_records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ wrote {out_path}", file=sys.stderr)
    if best is not None:
        print(f"  best val AUC: {best['auc_overall']:.4f} "
              f"at epoch {best['epoch']}", file=sys.stderr)
    else:
        print(f"  ⚠ no val AUC values found in checkpoints", file=sys.stderr)


if __name__ == "__main__":
    main()