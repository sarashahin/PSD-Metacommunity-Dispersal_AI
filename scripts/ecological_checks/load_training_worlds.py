#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficient, robust loader for IBM training worlds (*.npz)
- Scans a root folder and indexes ONLY '*_training.npz'
- Lazy per-key access (won't pull huge arrays unless requested)
- Optional GPU (CuPy) compute for heavy ops; CPU fallback is automatic
- Safe quick-look visualizations (matplotlib, no seaborn)

Usage examples
--------------
# List worlds & light summary (no big arrays loaded)
python load_training_worlds.py --root "/home/sara/.../results/data" --summarize

# Show a few quick-look plots for the first world (safe keys only)
python load_training_worlds.py --root "/home/sara/.../results/data" --quicklook --max-worlds 1

# Load one world and inspect shapes without loading 'IBM_B'
python -c "from load_training_worlds import DataCatalog; dc=DataCatalog('.../results/data'); w=dc[0]; print(w.safe_shapes())"
"""
from __future__ import annotations
import os, io, sys, re, math, argparse, textwrap, zipfile, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

# --------------------------- optional GPU backend ---------------------------
def get_backend(prefer_gpu: bool = True):
    """
    Returns (xp, on_gpu) where xp is numpy or cupy.
    Use xp.asarray(...) to move to GPU, and to_cpu() below to bring back.
    """
    import numpy as _np
    if not prefer_gpu:
        return _np, False
    try:
        import cupy as _cp
        # respect USE_GPU=0 if set
        if os.environ.get("USE_GPU", "1") in {"0", "false", "False"}:
            return _np, False
        # make sure at least one device exists
        _ = _cp.cuda.runtime.getDeviceCount()
        return _cp, True
    except Exception:
        return _np, False

def to_cpu(arr):
    """Return a CPU ndarray for numpy/cupy input."""
    try:
        import cupy as _cp
        if isinstance(arr, _cp.ndarray):
            return _cp.asnumpy(arr)
    except Exception:
        pass
    return arr

# ------------------------ tiny helper: humanize bytes -----------------------
def _fmt_bytes(n: int) -> str:
    if n is None:
        return "?"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024.0 and i < len(units)-1:
        f /= 1024.0; i += 1
    return f"{f:.1f} {units[i]}"

# --------------- read just the header of an .npy member inside .npz ---------
def _peek_npz_member_shape_dtype(npz_path: Path, member_name: str) -> Optional[Tuple[Tuple[int, ...], str]]:
    """
    Returns (shape, dtype_str) for a member inside the npz WITHOUT loading the full array.
    This reads only the .npy header from the zip stream.
    """
    import numpy.lib.format as fmt
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            if member_name not in zf.namelist():
                return None
            with zf.open(member_name, "r") as zfh:
                # read enough bytes to cover header; 64 KB is plenty in practice
                head = io.BytesIO(zfh.read(65536))
                major, minor = fmt.read_magic(head)
                if (major, minor) == (1, 0):
                    shape, fortran, dtype = fmt.read_array_header_1_0(head)
                else:
                    # fallback for newer headers (2.0+)
                    try:
                        shape, fortran, dtype = fmt.read_array_header_2_0(head)
                    except Exception:
                        # last resort: private API (version-agnostic)
                        shape, fortran, dtype = fmt._read_array_header(head, (major, minor))  # type: ignore
                return shape, str(dtype)
    except Exception:
        return None

# ---------------------------- main world wrapper ----------------------------
class TrainingWorld:
    """
    Lazy wrapper around a single '*_training.npz' file.

    - Access small arrays directly (e.g., P_last_final, P_any)
    - Avoid loading huge arrays unless requested (e.g., IBM_B)
    - Provides quick shape/dtype peeking via the zip header for any key
    """
    SAFE_DEFAULT_KEYS = (
        "P_last_final", "P_last", "P_any", "prevalence_final", "prevalence_any",
        "w_invprev_final", "w_invprev_any",
        "gamma", "Y", "X",
        "deg_in", "deg_out",
        "C_topk_idx", "C_topk_w",
        "ENV_r_field", "r_mean", "r_std",
        "RSD", "RSD_t", "gamma_t",
        "T_first_occ", "persist_steps",
        "SP_T_first_any", "SP_T_last_any", "SP_n_recolonizations", "SP_frac_time_occupied",
    )

    # arrays that can be very large; only load on demand
    POTENTIALLY_BIG = ("IBM_B", "P_t", "B_lastK")

    def __init__(self, path: Path):
        self.path = Path(path)
        self._npz = None   # type: Optional["numpy.lib.npyio.NpzFile"]
        self._opened = False

    def __enter__(self): self._open(); return self
    def __exit__(self, exc_type, exc, tb): self.close()

    def _open(self):
        if not self._opened:
            import numpy as np
            self._npz = np.load(self.path, allow_pickle=False)
            self._opened = True

    def close(self):
        if self._opened and self._npz is not None:
            try: self._npz.close()
            except Exception: pass
        self._npz, self._opened = None, False

    # ---------------------------- info / shapes ----------------------------
    def keys(self) -> List[str]:
        self._open()
        return list(self._npz.files)  # type: ignore

    def has(self, key: str) -> bool:
        self._open()
        return key in self._npz.files  # type: ignore

    def peek_shape_dtype(self, key: str) -> Optional[Tuple[Tuple[int, ...], str]]:
        """
        Try to read shape/dtype without loading (zip header). Falls back to loading header if needed.
        """
        guess = _peek_npz_member_shape_dtype(self.path, f"{key}.npy")
        if guess is not None:
            return guess
        # fallback: this will load the array; avoid calling for large keys
        try:
            arr = self.get(key, device=None)  # CPU
            return tuple(arr.shape), str(arr.dtype)
        except Exception:
            return None

    def safe_shapes(self) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        info = {}
        for k in self.keys():
            if k in self.POTENTIALLY_BIG:
                continue
            shp_dt = self.peek_shape_dtype(k)
            if shp_dt is not None:
                info[k] = shp_dt
        return info

    # ---------------------------- get / load -------------------------------
    def get(self, key: str, device=None, dtype=None):
        """
        Load one array by key. device: an object with .asarray (e.g., numpy or cupy).
        dtype: optional dtype cast after load.
        """
        self._open()
        import numpy as _np
        if key not in self._npz.files:  # type: ignore
            raise KeyError(f"Key '{key}' not found in {self.path.name}")
        arr = self._npz[key]  # numpy ndarray (loads from zip member)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if device is not None:
            # cupy/numpy both expose asarray
            arr = device.asarray(arr)
        return arr

    # ----------------------- small derived conveniences --------------------
    def world_tag(self) -> str:
        """Extract a short tag from the filename."""
        return self.path.stem.replace("_training", "")

    def grid_shape(self) -> Tuple[int, int]:
        y = int(self.get("Y"))
        x = int(self.get("X"))
        return y, x

    def richness_map(self):
        """Compute alpha richness at the final step from the unified final occupancy."""
        import numpy as np
        P = self.get_final_occupancy()  # (S,Y,X), uint8
        return P.sum(axis=0, dtype=np.int32)  # (Y,X)

    

    def get_final_occupancy(self):
        """
        Return a binary occupancy array (S, Y, X) for the *final* state, using:
        1) P_last_final, or
        2) P_last (older schema), or
        3) B_last thresholded by detection_threshold (fallback).
        """
        import numpy as np

        if self.has("P_last_final"):
            P = self.get("P_last_final")
            if P.dtype != np.uint8:
                P = (P > 0).astype(np.uint8)
            return P

        if self.has("P_last"):
            P = self.get("P_last")
            if P.dtype != np.uint8:
                P = (P > 0).astype(np.uint8)
            return P

        if self.has("B_last"):
            B = self.get("B_last")
            thr = float(self.get("detection_threshold")) if self.has("detection_threshold") else 0.0
            return (B >= thr).astype(np.uint8)

        raise KeyError(
            f"No final occupancy key found in {self.path.name} "
            "(tried P_last_final, P_last, then thresholded B_last)."
        )

    def get_prevalence_final(self, occupancy: Optional["np.ndarray"] = None):
        """
        Return species prevalence vector at final state (S,), in [0,1].
        Prefer stored 'prevalence_final' if present; otherwise compute from occupancy.
        """
        import numpy as np

        if self.has("prevalence_final"):
            prev = self.get("prevalence_final")
            return prev.astype(np.float32, copy=False)

        if occupancy is None:
            occupancy = self.get_final_occupancy()  # (S,Y,X)

        S = occupancy.shape[0]
        return occupancy.reshape(S, -1).mean(axis=1).astype(np.float32, copy=False)


# -------------------------- catalog of many worlds --------------------------
class DataCatalog:
    """
    Finds and serves only '*_training.npz' files in a directory tree.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Root path not found: {self.root}")
        self._paths = sorted(
            p for p in self.root.rglob("*.npz")
            if p.name.endswith("_training.npz")
        )
        if not self._paths:
            warnings.warn(f"No '*_training.npz' files found under: {self.root}", RuntimeWarning)

    def __len__(self) -> int: return len(self._paths)
    def __getitem__(self, i: int) -> TrainingWorld: return TrainingWorld(self._paths[i])
    def paths(self) -> List[Path]: return list(self._paths)

    def list_brief(self, max_rows: int = 20) -> List[str]:
        rows = []
        for i, p in enumerate(self._paths[:max_rows]):
            rows.append(f"[{i:03d}] {p.name}")
        if len(self) > max_rows:
            rows.append(f"... (+{len(self)-max_rows} more)")
        return rows

# ---------------------------- quick-look plotting ---------------------------
def quicklook(world: TrainingWorld, outdir: Optional[Path] = None, species: Optional[int] = None):
    """
    Cheap, informative plots:
    - final alpha richness map
    - prevalence histogram over species
    - gamma_t time series if present; else ASM_gamma vs ASM_round if available
    - a per-species final occupancy map (auto-pick a rare species)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    Y, X = world.grid_shape()
    tag = world.world_tag()

    # 1) final richness map (safe via unified occupancy)
    try:
        Pfin = world.get_final_occupancy()         # (S,Y,X)
    except KeyError as e:
        print(f"[skip] {tag}: {e}")
        return
    alpha = Pfin.sum(axis=0, dtype=np.int32)       # (Y,X)

    # 2) prevalence stats (stored or computed)
    prev = world.get_prevalence_final(Pfin)        # (S,)

    # 3) time series: prefer gamma_t; fallback to ASM_gamma vs ASM_round
    gamma_t = world.get("gamma_t") if world.has("gamma_t") else None
    asm_gamma = world.get("ASM_gamma") if world.has("ASM_gamma") else None
    asm_round = world.get("ASM_round") if world.has("ASM_round") else None

    # 4) pick a species (prefer a rare one) for a map
    S = Pfin.shape[0]
    if species is not None and 0 <= species < S:
        sp_idx = species
    else:
        present = prev > 0
        if present.any():
            candidates = np.where(present)[0]
            sp_idx = candidates[np.argmin(prev[candidates])]
        else:
            sp_idx = 0
    sp_map = Pfin[sp_idx]

    # ---- plot ----
    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    axes = axes.ravel()

    im0 = axes[0].imshow(alpha, origin="upper")
    axes[0].set_title("Final alpha richness")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(prev, bins=30)
    axes[1].set_title("Species prevalence (final)")
    axes[1].set_xlabel("fraction of patches occupied")
    axes[1].set_ylabel("species count")

    # gamma-like panel
    if gamma_t is not None and gamma_t.size > 0:
        axes[2].plot(np.arange(gamma_t.shape[0]), gamma_t)
        axes[2].set_title("Regional richness γ(t)")
        axes[2].set_xlabel("time index")
        axes[2].set_ylabel("γ")
    elif asm_gamma is not None and asm_round is not None and asm_gamma.size == asm_round.size and asm_gamma.size > 0:
        axes[2].plot(asm_round, asm_gamma)
        axes[2].set_title("Assembly γ vs. round")
        axes[2].set_xlabel("assembly round")
        axes[2].set_ylabel("γ")
    else:
        axes[2].axis("off")
        axes[2].set_title("No γ-series available")

    im3 = axes[3].imshow(sp_map, origin="upper")
    axes[3].set_title(f"Final occupancy (species #{sp_idx})")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Quicklook: {tag}", y=0.98)
    fig.tight_layout()

    if outdir is not None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{tag}_quicklook.png"
        fig.savefig(out, dpi=160)
        print(f"[save] {out}")
    else:
        plt.show()


# ----------------------------- CLI / main -----------------------------------
def _main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Efficient loader for IBM '*_training.npz' worlds (CPU/GPU optional)."
    )
    ap.add_argument("--root", type=str, required=True,
                    help="Path to results/data/ that contains the '*_training.npz' files.")
    ap.add_argument("--summarize", action="store_true",
                    help="Print brief list of worlds and safe shapes for the first one.")
    ap.add_argument("--quicklook", action="store_true",
                    help="Render a small set of safe visualizations for the first world(s).")
    ap.add_argument("--max-worlds", type=int, default=1,
                    help="How many worlds to quicklook (default: 1).")
    ap.add_argument("--species", type=int, default=None,
                    help="Species index for the per-species map (default: auto-pick a rare one).")
    ap.add_argument("--prefer-gpu", action="store_true",
                    help="If set, compute-heavy ops can use CuPy when available.")
    ap.add_argument("--outdir", type=str, default=None,
                    help="If set, save quicklook figures to this folder instead of showing interactively.")
    
    ap.add_argument("--start-index", type=int, default=0,
                help="Start from this catalog index (default: 0).")

    args = ap.parse_args()

    dc = DataCatalog(args.root)
    print(f"[info] found {len(dc)} training worlds under: {dc.root}")
    for row in dc.list_brief(max_rows=10):
        print(" ", row)

    # backend (not used heavily in quicklook, but exposed for your own compute)
    xp, on_gpu = get_backend(prefer_gpu=args.prefer_gpu)
    print(f"[backend] xp = {xp.__name__} | on_gpu={on_gpu}")

    if args.summarize and len(dc) > 0:
        with dc[0] as w:
            print(f"\n[summary] {w.path.name}")
            safe = w.safe_shapes()
            # Show a few keys in a stable order
            for k in sorted(safe.keys()):
                shp, dt = safe[k]
                print(f" - {k:24s} shape={shp!s:>16} dtype={dt}")

            # Also show BIG keys by reading header only (no full load)
            for big in TrainingWorld.POTENTIALLY_BIG:
                if w.has(big):
                    info = w.peek_shape_dtype(big)
                    if info is not None:
                        print(f" - {big:24s} shape={info[0]!s:>16} dtype={info[1]}  (peeked)")

    if args.quicklook and len(dc) > 0:
        start = max(0, args.start_index)
        end = min(start + args.max_worlds, len(dc))
        for i in range(start, end):
            with dc[i] as w:
                try:
                    quicklook(w, outdir=Path(args.outdir) if args.outdir else None, species=args.species)
                except Exception as e:
                    print(f"[warn] quicklook failed for {w.path.name}: {e}")
                    continue



if __name__ == "__main__":
    _main()
