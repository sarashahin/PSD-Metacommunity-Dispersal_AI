#!/usr/bin/env python
"""
inspect_lifebird_tif.py

Quick inspection of a species range / habitat GeoTIFF:
- prints metadata (size, CRS, resolution, nodata, tags)
- prints basic statistics and unique values
- optionally shows a quick map

Usage:
  python inspect_lifebird_tif.py "/path/to/FRC_22678073_Resident.tif" --show
"""

from pathlib import Path
import argparse

import numpy as np
import rasterio
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tif_path", help="Path to the .tif file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a quick map of band 1",
    )
    args = parser.parse_args()

    tif_path = Path(args.tif_path)
    if not tif_path.exists():
        raise SystemExit(f"File not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        print("=== BASIC METADATA ===")
        print("path        :", tif_path)
        print("driver      :", src.driver)
        print("width x height:", src.width, "x", src.height)
        print("bands       :", src.count)
        print("dtype       :", src.dtypes[0])
        print("CRS         :", src.crs)
        print("transform   :", src.transform)
        print("nodata      :", src.nodata)

        # Pixel size (approx)
        res_x, res_y = src.res
        print("pixel size  :", res_x, "×", res_y)

        print("\n=== GLOBAL TAGS ===")
        print(src.tags())          # may contain species name, source, etc.

        print("\n=== BAND-1 TAGS ===")
        print(src.tags(1))

        # Read band 1 as a masked array (nodata handled automatically)
        band1 = src.read(1, masked=True)

    # ---- basic stats (ignoring nodata) ----
    print("\n=== STATISTICS (band 1, ignoring nodata) ===")
    print("min   :", float(band1.min()))
    print("max   :", float(band1.max()))
    print("mean  :", float(band1.mean()))
    print("std   :", float(band1.std()))

    # For small integer rasters, list unique values
    vals = band1.compressed()  # drop nodata
    if np.issubdtype(vals.dtype, np.integer) and vals.size > 0:
        # limit to avoid insane memory if file is huge
        if vals.size <= 1_000_000:
            uniques, counts = np.unique(vals, return_counts=True)
            print("\n=== UNIQUE VALUES (band 1) ===")
            total = vals.size
            for v, c in zip(uniques, counts):
                frac = c / total
                print(f"value {int(v):4d}: count={int(c):10d}, frac={frac:6.3%}")
        else:
            print("\n[info] Too many pixels for full unique-value listing; "
                  "use hist instead if needed.")

    # ---- optional map ----
    if args.show:
        if np.issubdtype(vals.dtype, np.integer) and vals.max() <= 20:
            cmap = "tab20"
        else:
            cmap = "viridis"

        plt.figure(figsize=(6, 4))
        img = plt.imshow(band1, cmap=cmap)
        plt.colorbar(img, label="value")
        plt.title(tif_path.name)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
