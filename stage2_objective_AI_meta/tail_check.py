import glob, os, csv, numpy as np
def ranges(p):
    try:
        with np.load(p, allow_pickle=True) as d:
            if "P_last_final" not in d.files:          # the anomalous file: skip, don't crash
                return None
            P = np.asarray(d["P_last_final"])
    except Exception:
        return None
    if P.ndim != 3 or tuple(P.shape[1:]) != (20, 20):
        return None
    return (P > 0.5).reshape(P.shape[0], -1).sum(1)

tr_list = [r for f in sorted(glob.glob("./results/data/*_training.npz")) if (r := ranges(f)) is not None]
tr = np.concatenate(tr_list)
ev_list = []
for row in csv.DictReader(open("./results/data/data_eval_unseen/eval_manifest_indist.csv")):
    r = ranges("./results/data/data_eval_unseen/" + row["filename"])
    if r is not None: ev_list.append(r)
ev = np.concatenate(ev_list)

trH, evH = tr[tr >= 21], ev[ev >= 21]
print("HARD subset (range>=21) percentiles  p50 / p90 / p95 / max")
print(f"  train: {np.percentile(trH,[50,90,95]).round(0)}  max {int(trH.max())}   (n={len(trH)})")
print(f"  eval : {np.percentile(evH,[50,90,95]).round(0)}  max {int(evH.max())}   (n={len(evH)})")
print("\nwide-tail density (absolute counts):")
for lo in (21, 30, 40, 50):
    print(f"  training species range>={lo}: {int((tr>=lo).sum()):5d}    eval: {int((ev>=lo).sum()):4d}")