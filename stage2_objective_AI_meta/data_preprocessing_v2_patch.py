"""
=============================================================================
DATA_PREPROCESSING_V2_PATCH.PY  —  history slicing + infill history fix
=============================================================================

Two surgical fixes to your data_preprocessing.EcoDataset.__getitem__:

  1. TEMPORAL MODE used proc['P_t'][:n_history] (early frames). This is
     ecologically wrong — we want the LAST `history_length` frames so the
     time sequence ends just before the target. Aligned with Axel's
     framing: "this is the truth, this is the noisy thing, that's what
     my AI recovered" — past observations should be near-past, not
     ancient history.

  2. INFILL MODE never included history_P at all (only the obs_mask).
     This is fine for testing AOO/EOO inference but during TRAINING we
     want both the K-budget last-frame sparsity (the obs_mask) AND the
     past-time history (history_P[-history_length:]). Otherwise Phase 4
     loses the temporal signal Phase 3 just learned.

Both fixes are implemented as a wrapper around EcoDataset.__getitem__
to avoid touching the 816-line data_preprocessing.py.

USAGE
-----
Imported automatically by run_training_v2.py via training_v2_patch.py.
=============================================================================
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)


def install_patch(EcoDataset, history_length=10):
    """
    Wrap EcoDataset.__getitem__ to:
      • use last-N (not first-N) frames in temporal mode
      • include history_P in infill mode too
    """
    if getattr(EcoDataset, '_v2_patched', False):
        return EcoDataset

    original_getitem = EcoDataset.__getitem__

    def patched_getitem(self, idx):
        sample = original_getitem(self, idx)
        cond = sample.get('condition', {})

        # Look up the underlying preprocessed data
        proc = self._load_and_preprocess(idx)
        n_total = int(proc['P_t'].shape[0])
        H = min(history_length, n_total)

        from data_preprocessing import pad_to_max_species

        if self.mode == 'temporal':
            # Replace history with last-H frames
            cond['history_P'] = pad_to_max_species(
                proc['P_t'][-H:], self.max_species, species_dim=1)
            cond['history_B'] = pad_to_max_species(
                proc['IBM_B'][-H:], self.max_species, species_dim=1)

        elif self.mode == 'infill':
            # Add last-H history (was missing). The infill obs_mask
            # already supplies the K-budget last-frame supervision.
            cond['history_P'] = pad_to_max_species(
                proc['P_t'][-H:], self.max_species, species_dim=1)
            cond['history_B'] = pad_to_max_species(
                proc['IBM_B'][-H:], self.max_species, species_dim=1)

        elif self.mode == 'full':
            # Truncate to last-H to match what inference will receive
            cond['history_P'] = pad_to_max_species(
                proc['P_t'][-H:], self.max_species, species_dim=1)
            cond['history_B'] = pad_to_max_species(
                proc['IBM_B'][-H:], self.max_species, species_dim=1)

        sample['condition'] = cond
        return sample

    EcoDataset.__getitem__ = patched_getitem
    EcoDataset._v2_patched = True
    logger.info(f"   ✓ EcoDataset patched: history_length={history_length}, "
                f"using last-N frames")
    return EcoDataset


def auto_install(history_length=10):
    if 'data_preprocessing' in sys.modules:
        from data_preprocessing import EcoDataset
        install_patch(EcoDataset, history_length=history_length)
    else:
        logger.warning("   ⚠ data_preprocessing not yet imported")