# v1 was wrong: it assumed random-cell sampling with presence-count k as the
# range proxy. The data reveals exactly K occupied cells per species (range>K),
# all presences -> k is constant and uninformative (hence v1's nan). The real
# range signal is the SPATIAL SPREAD of the K revealed presence cells. This
# measures the ceiling on per-species range recovery from that geometry, and
# compares it to v3's cross-validated rho = 0.394.
import numpy as np, csv, sys
from pathlib import Path
from scipy import stats

K        = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MANIFEST = Path("results/data/data_eval_unseen/eval_unseen_manifest.csv")
RECON    = Path("reconstructions_unseen")
TRUTH    = Path("results/data/data_eval_unseen")
GRID_Y = GRID_X = 20

def periodic_cov_det(yy, xx, Y=GRID_Y, X=GRID_X):
    if len(yy) < 2: return 0.0
    ty, tx = 2*np.pi*np.asarray(yy)/Y, 2*np.pi*np.asarray(xx)/X
    my = np.arctan2(np.sin(ty).mean(), np.cos(ty).mean())
    mx = np.arctan2(np.sin(tx).mean(), np.cos(tx).mean())
    dy = ((ty - my + np.pi) % (2*np.pi) - np.pi) * Y / (2*np.pi)
    dx = ((tx - mx + np.pi) % (2*np.pi) - np.pi) * X / (2*np.pi)
    return max(0.0, np.var(dy)*np.var(dx) - (((dy-dy.mean())*(dx-dx.mean())).mean())**2)

def pairwise_tor(yy, xx, Y=GRID_Y, X=GRID_X):
    pts = np.array(list(zip(yy, xx)), float)
    if len(pts) < 2: return 0.0, 0.0
    dsum = dmax = 0.0; npair = 0
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            dy = abs(pts[i,0]-pts[j,0]); dy = min(dy, Y-dy)
            dx = abs(pts[i,1]-pts[j,1]); dx = min(dx, X-dx)
            d = np.hypot(dy, dx); dsum += d; npair += 1
            dmax = max(dmax, d)
    return dsum/npair, dmax

def stem_of(r): return r["filename"][:-4] if r["filename"].endswith(".npz") else r["filename"]

R_all, F = [], []
rows = sorted(csv.DictReader(open(MANIFEST)), key=lambda r: r["filename"])
for r in rows:
    tp = TRUTH / r["filename"]; rp = RECON / stem_of(r) / f"recon_fixed_b{K}_samples.npz"
    if not (tp.exists() and rp.exists()): continue
    truth = (np.load(tp, allow_pickle=True)["P_last_final"] > 0.5).astype(np.uint8)
    z = np.load(rp)
    obs = (z["noisy_input"] > 0.5).astype(np.uint8) if "noisy_input" in z.files \
          else (z["samples"].astype(np.float32).var(0) < 1e-8).astype(np.uint8)
    n = min(truth.shape[0], obs.shape[0])
    for s in range(n):
        R = int(truth[s].sum())
        if R <= K: continue
        yy, xx = np.where(obs[s] > 0.5)
        if len(yy) < 2: continue
        meand, maxd = pairwise_tor(yy, xx)
        F.append([np.log10(periodic_cov_det(yy, xx) + 1.0), meand, maxd]); R_all.append(R)
R_all = np.asarray(R_all, float); F = np.asarray(F, float)

print(f"\n  K={K}  species (range>{K}, >=2 obs presences) = {R_all.size}")
print("  " + "-"*62)
names = ["obs spatial spread (logcd)", "obs mean pairwise dist", "obs max pairwise dist"]
for j, nm in enumerate(names):
    rho, _ = stats.spearmanr(F[:, j], R_all)
    print(f"  univariate  rho({nm:<26}, true range) = {rho:+.3f}")
Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
A = np.c_[np.ones(len(Fz)), Fz]
beta, *_ = np.linalg.lstsq(A, R_all, rcond=None)
rho_oracle, _ = stats.spearmanr(A @ beta, R_all)
print("  " + "-"*62)
print(f"  ORACLE ceiling (in-sample, obs-geometry only) rho = {rho_oracle:+.3f}")
print(f"  v3 model (cross-validated, uses prob-map shape) rho = +0.394")
print("  " + "-"*62)
print("  Reading: if oracle ~= 0.4, range recovery is limited by the observed")
print("  spatial-spread proxy (information limit, motivates Obj 3). If oracle is")
print("  much higher, observed geometry carries range info the model misses.")
print("  " + "-"*62)
q = np.quantile(F[:, 0], [0, .25, .5, .75, 1.0])
for i in range(4):
    m = (F[:, 0] >= q[i]) & (F[:, 0] <= q[i+1])
    if m.sum():
        print(f"  obs-spread Q{i+1}: n={int(m.sum()):4d}  true range mean={R_all[m].mean():5.1f}"
              f"  sd={R_all[m].std():4.1f}  ({int(R_all[m].min())}-{int(R_all[m].max())})")
