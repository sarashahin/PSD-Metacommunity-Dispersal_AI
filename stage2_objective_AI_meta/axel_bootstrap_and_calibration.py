
# axel_bootstrap_and_calibration.py
# Two rigour add-ons for the Objective-2 distributional result, on the
# regenerated unseen IBM worlds. Reuses the EXACT loader, PBC statistics and
# truth-free symmetric count calibrator from baseline_smoother_compare.py.
#   (1) Cluster bootstrap over report worlds -> 95% CIs on the three KS stats.
#   (2) Calibration diagnostic: predicted count vs true range (CSV + PNG).
# No retraining; reads only existing recon NPZs. Diffusion model only.
import numpy as np, csv, sys
from pathlib import Path
from scipy.stats import ks_2samp
from scipy import ndimage
from sklearn.isotonic import IsotonicRegression

K        = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_BOOT   = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MANIFEST = Path("results/data/data_eval_unseen/eval_unseen_manifest.csv")
RECON    = Path("reconstructions_unseen")
TRUTH    = Path("results/data/data_eval_unseen")
OUT      = Path("figures_map_axel_stage2_new/unseen_eval"); OUT.mkdir(parents=True, exist_ok=True)
GRID_Y = GRID_X = 20
CONN = ndimage.generate_binary_structure(2, 1)
RNG  = np.random.default_rng(0)

def count_components_pbc(b):
    if b.sum() == 0: return 0
    lab, n = ndimage.label(b, structure=CONN)
    if n <= 1: return int(n)
    for x in range(b.shape[1]):
        a, c = lab[0, x], lab[-1, x]
        if a and c and a != c: lab[lab == c] = a
    for y in range(b.shape[0]):
        a, c = lab[y, 0], lab[y, -1]
        if a and c and a != c: lab[lab == c] = a
    return int(len(np.unique(lab[lab > 0])))

def periodic_cov_det(b, Y=GRID_Y, X=GRID_X):
    yy, xx = np.where(b > 0.5)
    if len(yy) < 2: return 0.0
    ty, tx = 2*np.pi*yy/Y, 2*np.pi*xx/X
    my = np.arctan2(np.sin(ty).mean(), np.cos(ty).mean())
    mx = np.arctan2(np.sin(tx).mean(), np.cos(tx).mean())
    dy = ((ty - my + np.pi) % (2*np.pi) - np.pi) * Y / (2*np.pi)
    dx = ((tx - mx + np.pi) % (2*np.pi) - np.pi) * X / (2*np.pi)
    return max(0.0, np.var(dy)*np.var(dx) - (((dy-dy.mean())*(dx-dx.mean())).mean())**2)

def stem_of(r): return r["filename"][:-4] if r["filename"].endswith(".npz") else r["filename"]

def load_world(stem, fn):
    tp = TRUTH / fn; rp = RECON / stem / f"recon_fixed_b{K}_samples.npz"
    if not (tp.exists() and rp.exists()): return None
    truth = (np.load(tp, allow_pickle=True)["P_last_final"] > 0.5).astype(np.uint8)
    z = np.load(rp); samples = z["samples"].astype(np.float32)
    mean = (z["mean"] if "mean" in z.files else samples.mean(0)).astype(np.float32)
    obs  = (z["noisy_input"] > 0.5).astype(np.uint8) if "noisy_input" in z.files \
           else (samples.var(0) < 1e-8).astype(np.uint8)
    n = min(truth.shape[0], samples.shape[1], obs.shape[0])
    return truth[:n], mean[:n], obs[:n]

rows = sorted(csv.DictReader(open(MANIFEST)), key=lambda r: r["filename"])
calib  = [w for w in (load_world(stem_of(r), r["filename"]) for r in rows[0::2]) if w]
report = [w for w in (load_world(stem_of(r), r["filename"]) for r in rows[1::2]) if w]
if not calib or not report: sys.exit("No worlds loaded — check paths.")

# --- truth-free symmetric count calibrator, fit on calib (same as baseline) ---
raw, cnt = [], []
for truth, mean, obs in calib:
    for s in range(truth.shape[0]):
        if int(truth[s].sum()) <= K: continue
        raw.append(float(mean[s].sum())); cnt.append(int(truth[s].sum()))
iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(np.array(raw), np.array(cnt))

def pred_N(p): return max(1, min(int(round(float(iso.predict([float(p.sum())])[0]))), p.size))
def topN(p):
    N = pred_N(p); flat = p.ravel(); b = np.zeros(flat.size, np.uint8)
    if flat.max() > 1e-9: b[np.argpartition(flat, -N)[-N:]] = 1
    return b.reshape(p.shape)

# --- per-world statistic vectors on REPORT (so bootstrap can resample worlds) ---
per_world = []          # each entry: dict of arrays for one world
cal_true, cal_pred = [], []   # for calibration diagnostic (true range, predicted N)
for truth, mean, obs in report:
    TR, PR, TC, PC, TS, PS = [], [], [], [], [], []
    for s in range(truth.shape[0]):
        if int(truth[s].sum()) <= K: continue
        b = topN(mean[s])
        TR.append(int(truth[s].sum())); PR.append(int(b.sum()))
        TC.append(count_components_pbc(truth[s])); PC.append(count_components_pbc(b))
        TS.append(np.log10(periodic_cov_det(truth[s]) + 1))
        PS.append(np.log10(periodic_cov_det(b) + 1))
        cal_true.append(int(truth[s].sum())); cal_pred.append(int(b.sum()))
    per_world.append({k: np.array(v) for k, v in
                      dict(TR=TR, PR=PR, TC=TC, PC=PC, TS=TS, PS=PS).items()})

def ks_of(pool, a, b): return float(ks_2samp(pool[a], pool[b]).statistic)
def pool_idx(idx):
    keys = ("TR","PR","TC","PC","TS","PS")
    return {k: np.concatenate([per_world[i][k] for i in idx]) for k in keys}

# point estimates (all report worlds)
full = pool_idx(range(len(per_world)))
pt = {"range": ks_of(full,"TR","PR"), "conn": ks_of(full,"TC","PC"), "spread": ks_of(full,"TS","PS")}

# cluster bootstrap over worlds
W = len(per_world); boot = {"range": [], "conn": [], "spread": []}
for _ in range(N_BOOT):
    idx = RNG.integers(0, W, size=W)          # resample worlds with replacement
    p = pool_idx(idx)
    boot["range"].append(ks_of(p,"TR","PR"))
    boot["conn"].append(ks_of(p,"TC","PC"))
    boot["spread"].append(ks_of(p,"TS","PS"))

print(f"\n  K={K}  report worlds={W}  species={full['TR'].size}  bootstraps={N_BOOT}")
print("  " + "-"*64)
print(f"  {'statistic':<16}{'KS':>8}{'2.5%':>10}{'97.5%':>10}   bar<=0.30")
print("  " + "-"*64)
for name, key in [("range size","range"),("connectance","conn"),("spatial spread","spread")]:
    lo, hi = np.percentile(boot[key], [2.5, 97.5])
    flag = "PASS" if hi <= 0.30 else ("PASS(pt)" if pt[key] <= 0.30 else "FAIL")
    print(f"  {name:<16}{pt[key]:>8.3f}{lo:>10.3f}{hi:>10.3f}   {flag}")
print("  " + "-"*64)
print("  PASS = whole 95% CI under 0.30; PASS(pt) = point passes but CI crosses 0.30")

# --- calibration diagnostic ---
cal_true = np.array(cal_true); cal_pred = np.array(cal_pred)
ratio = cal_pred.sum() / max(cal_true.sum(), 1)
print(f"\n  calibration: sum(pred N)/sum(true range) = {ratio:.3f}  (1.0 = unbiased overall)")
with open(OUT / f"calibration_diag_K{K}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["true_range","pred_count"])
    for t, p in zip(cal_true, cal_pred): w.writerow([t, p])
print(f"  calibration CSV -> {OUT}/calibration_diag_K{K}.csv")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    edges = np.arange(cal_true.min(), cal_true.max()+2)
    binmean = [cal_pred[(cal_true>=lo)&(cal_true<lo+1)].mean()
               if ((cal_true>=lo)&(cal_true<lo+1)).any() else np.nan for lo in edges[:-1]]
    plt.figure(figsize=(5.2,5))
    plt.scatter(cal_true, cal_pred, s=8, alpha=0.25, color="#3a6ea5", label="species")
    plt.plot(edges[:-1]+0.5, binmean, color="#c0392b", lw=2, label="binned mean pred")
    lim = max(cal_true.max(), cal_pred.max())+1
    plt.plot([0,lim],[0,lim], "k--", lw=1, label="identity")
    plt.axvline(K, color="grey", ls=":", lw=1); plt.xlabel("true range (cells)")
    plt.ylabel("predicted count N (truth-free)"); plt.xlim(0,lim); plt.ylim(0,lim)
    plt.title(f"Count calibration, K={K}\noverall ratio={ratio:.2f}"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / f"calibration_diag_K{K}.png", dpi=160); plt.close()
    print(f"  calibration PNG -> {OUT}/calibration_diag_K{K}.png")
except Exception as e:
    print(f"  (plot skipped: {e})")
