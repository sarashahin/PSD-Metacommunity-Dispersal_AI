# scripts/list_world_params.py
from scripts.ecological_checks.load_training_worlds import DataCatalog
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
args = ap.parse_args()

dc = DataCatalog(args.root)
print("idx\tD\tLDD\tExtant")
for i in range(len(dc)):
    try:
        with dc[i] as w:
            D   = w.get("DISPERSAL_RATE")
            LDD = w.get("LONG_DISTANCE_PROB")
            P   = w.get_final_occupancy()
            ext = (P.reshape(P.shape[0], -1).sum(axis=1) > 0).sum()
            # numpy scalars -> python
            Dv = np.asscalar(D) if hasattr(D, 'dtype') else D
            Lv = np.asscalar(LDD) if hasattr(LDD, 'dtype') else LDD
            print(f"{i}\t{Dv}\t{Lv}\t{ext}")
    except Exception as e:
        print(f"{i}\tNA\tNA\tNA")
