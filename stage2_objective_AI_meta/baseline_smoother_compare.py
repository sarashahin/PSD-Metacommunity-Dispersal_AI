
# baseline_smoother_compare.py  (v2: adds ensemble + AUC pillars; honest sparse-presence note)
# Fair baseline for Objective 2: periodic Gaussian kernel smoother over the observed
# PRESENCE cells (= the model's noisy_input). Diffusion vs smoother through IDENTICAL
# machinery (same worlds, same conditioning, same disjoint calib/report split, same
# truth-free count-calibration, same PBC statistics, same NEAR/FAR), so the ONLY
# difference is the probability map. NEW: ENS-union recall + ensemble diversity
# (Axel's "truth is in the ensemble" criterion) and calibration-free AUC.
import numpy as np, csv, sys
from pathlib import Path
from scipy.stats import ks_2samp, rankdata
from scipy import ndimage
from sklearn.isotonic import IsotonicRegression

K        = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MANIFEST = Path("results/data/data_eval_unseen/eval_unseen_manifest.csv")
RECON    = Path("reconstructions_unseen")
TRUTH    = Path("results/data/data_eval_unseen")
GRID_Y = GRID_X = 20
CONN = ndimage.generate_binary_structure(2, 1)
SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

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

def gaussian_kernel_matrix(sigma, Y=GRID_Y, X=GRID_X):
    coords = np.array([(y, x) for y in range(Y) for x in range(X)])
    dy = np.abs(coords[:, None, 0] - coords[None, :, 0]); dy = np.minimum(dy, Y - dy)
    dx = np.abs(coords[:, None, 1] - coords[None, :, 1]); dx = np.minimum(dx, X - dx)
    d2 = dy.astype(np.float64)**2 + dx.astype(np.float64)**2
    return np.exp(-d2 / (2.0 * sigma * sigma))

def smoother_prob(obs_2d, Kmat):
    obs_flat = obs_2d.ravel().astype(bool)
    if obs_flat.sum() == 0:
        return np.zeros((GRID_Y, GRID_X), dtype=np.float32)
    p = Kmat[:, obs_flat].sum(axis=1); p = p / max(p.max(), 1e-9)
    return p.reshape(GRID_Y, GRID_X).astype(np.float32)

def stem_of(r): return r["filename"][:-4] if r["filename"].endswith(".npz") else r["filename"]

def load_world(stem, fn, keep_samples):
    tp = TRUTH / fn
    rp = RECON / stem / f"recon_fixed_b{K}_samples.npz"
    if not (tp.exists() and rp.exists()): return None
    truth = (np.load(tp, allow_pickle=True)["P_last_final"] > 0.5).astype(np.uint8)
    z = np.load(rp)
    samples = z["samples"].astype(np.float32)
    mean = (z["mean"] if "mean" in z.files else samples.mean(0)).astype(np.float32)
    # observed = exactly what the model was conditioned on (presence cells in noisy_input)
    if "noisy_input" in z.files:
        obs = (z["noisy_input"] > 0.5).astype(np.uint8)
    elif "obs_mask" in z.files:
        obs = z["obs_mask"].astype(np.uint8)
    else:
        obs = (samples.var(axis=0) < 1e-8).astype(np.uint8)  # RePaint clamp fallback
    n = min(truth.shape[0], samples.shape[1], obs.shape[0])
    return (truth[:n], mean[:n], obs[:n],
            (samples[:, :n] if keep_samples else None))

def load_all(rows_subset, keep_samples):
    out = []
    for r in rows_subset:
        w = load_world(stem_of(r), r["filename"], keep_samples)
        if w is not None: out.append(w)
    return out

rows = sorted(csv.DictReader(open(MANIFEST)), key=lambda r: r["filename"])
calib_cache  = load_all(rows[0::2], keep_samples=False)   # calib: mean+obs+truth only
report_cache = load_all(rows[1::2], keep_samples=True)     # report: keep 8 samples for ENS
if not calib_cache or not report_cache:
    sys.exit("No worlds loaded — check MANIFEST / RECON / TRUTH paths at top of script.")

obs_counts = [int(obs[s].sum()) for (_, _, obs, _) in report_cache
              for s in range(obs.shape[0]) if obs[s].sum() > 0]
med = int(np.median(obs_counts))
print(f"K={K}  calib worlds={len(calib_cache)}  report worlds={len(report_cache)}")
print(f"presence cells among K={K} obs/species (report, nonzero median) = {med}")
print(f"  NOTE: most of the K observations are ABSENCES; only ~{med} are presences.")
print(f"  This is the real sparse-presence regime, not a loading bug — both methods read the")
print(f"  SAME noisy_input the diffusion model was conditioned on.\n")

def novel_recall_topN(prob, truth, obs):
    N = int(truth.sum()); flat = prob.ravel(); b = np.zeros(flat.size, np.uint8)
    if N > 0 and flat.max() > 1e-9: b[np.argpartition(flat, -N)[-N:]] = 1
    b = b.reshape(truth.shape)
    tn = (truth > 0) & (~(obs > 0)); pn = (b > 0) & (~(obs > 0)); d = int(tn.sum())
    return (int((tn & pn).sum()) / d) if d > 0 else np.nan

best_sigma, best_score = SIGMAS[0], -1.0
for sigma in SIGMAS:
    Kmat = gaussian_kernel_matrix(sigma); rec = []
    for truth, mean, obs, _ in calib_cache:
        for s in range(truth.shape[0]):
            if int(truth[s].sum()) <= K: continue
            v = novel_recall_topN(smoother_prob(obs[s], Kmat), truth[s], obs[s])
            if not np.isnan(v): rec.append(v)
    score = float(np.mean(rec)) if rec else -1.0
    if score > best_score: best_score, best_sigma = score, sigma
print(f"chosen smoother sigma = {best_sigma}  (calib novel recall = {best_score:.0%})\n")
Kmat = gaussian_kernel_matrix(best_sigma)

def fit_count_iso(which):
    raw, cnt = [], []
    for truth, mean, obs, _ in calib_cache:
        for s in range(truth.shape[0]):
            if int(truth[s].sum()) <= K: continue
            p = mean[s] if which == "diff" else smoother_prob(obs[s], Kmat)
            raw.append(float(p.sum())); cnt.append(int(truth[s].sum()))
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(np.array(raw), np.array(cnt)); return iso

iso = {"diff": fit_count_iso("diff"), "base": fit_count_iso("base")}

def pred_N(p, iso_m):
    return max(1, min(int(round(float(iso_m.predict([float(p.sum())])[0]))), p.size))

def topN_from_iso(p, iso_m):
    N = pred_N(p, iso_m); flat = p.ravel(); b = np.zeros(flat.size, np.uint8)
    if flat.max() > 1e-9: b[np.argpartition(flat, -N)[-N:]] = 1
    return b.reshape(p.shape)

def near_far_recall(truth, b, obs, radius=2):
    obs_y, obs_x = np.where(obs > 0)
    near = np.zeros_like(obs, bool); yy, xx = np.indices(obs.shape)
    for oy, ox in zip(obs_y, obs_x):
        dy = np.minimum(np.abs(yy - oy), GRID_Y - np.abs(yy - oy))
        dx = np.minimum(np.abs(xx - ox), GRID_X - np.abs(xx - ox))
        near |= (np.maximum(dy, dx) <= radius)
    nov = ~(obs > 0)
    tn_near = (truth > 0) & nov & near;  tn_far = (truth > 0) & nov & ~near
    pn_near = (b > 0) & nov & near;      pn_far = (b > 0) & nov & ~near
    rn = int((tn_near & pn_near).sum()) / max(1, int(tn_near.sum())) if tn_near.sum() > 0 else np.nan
    rf = int((tn_far  & pn_far ).sum()) / max(1, int(tn_far.sum()))  if tn_far.sum()  > 0 else np.nan
    return rn, rf

def auc_novel(prob, truth, obs):
    nov = ~(obs > 0)
    y = (truth[nov] > 0).astype(np.uint8); s = prob[nov].ravel()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def ensemble_metrics(truth_sp, samples_sp, mean_sp, obs_sp, iso_m):
    """Diffusion ensemble (TRUTH-FREE): binarise each of the 8 samples at the SAME
    count-calibrated N (N from iso on the mean), union, novel recall. Diversity =
    mean pairwise Jaccard distance among the per-sample top-N maps."""
    N = pred_N(mean_sp, iso_m); bins = []
    for k in range(samples_sp.shape[0]):
        flat = samples_sp[k].ravel(); b = np.zeros(flat.size, np.uint8)
        if flat.max() > 1e-9: b[np.argpartition(flat, -N)[-N:]] = 1
        bins.append(b.reshape(mean_sp.shape).astype(bool))
    bins = np.stack(bins)
    nov = ~(obs_sp > 0); tn = (truth_sp > 0) & nov; d = int(tn.sum())
    ens = (int((tn & bins.any(0)).sum()) / d) if d > 0 else np.nan
    dists = []; n = bins.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            u = int((bins[i] | bins[j]).sum())
            if u > 0: dists.append(1.0 - int((bins[i] & bins[j]).sum())/u)
    return ens, (float(np.mean(dists)) if dists else 0.0)

res = {m: {k: [] for k in ("TR","PR","TC","PC","TS","PS","nov","near","far","auc","ens","div")}
       for m in ("diff", "base")}
per_species_rows = []
for truth, mean, obs, samples in report_cache:
    for s in range(truth.shape[0]):
        if int(truth[s].sum()) <= K: continue
        row = {"truth_range": int(truth[s].sum())}
        base_p = smoother_prob(obs[s], Kmat)
        for m, p in [("diff", mean[s]), ("base", base_p)]:
            b = topN_from_iso(p, iso[m]); R = res[m]
            R["TR"].append(int(truth[s].sum())); R["PR"].append(int(b.sum()))
            R["TC"].append(count_components_pbc(truth[s])); R["PC"].append(count_components_pbc(b))
            R["TS"].append(np.log10(periodic_cov_det(truth[s]) + 1))
            R["PS"].append(np.log10(periodic_cov_det(b) + 1))
            nt = (truth[s] > 0) & (~(obs[s] > 0)); npd = (b > 0) & (~(obs[s] > 0)); d = int(nt.sum())
            R["nov"].append(int((nt & npd).sum()) / d if d > 0 else np.nan)
            rn, rf = near_far_recall(truth[s], b, obs[s]); R["near"].append(rn); R["far"].append(rf)
            R["auc"].append(auc_novel(p, truth[s], obs[s]))
            row[f"{m}_pred_range"] = int(b.sum()); row[f"{m}_pred_patches"] = count_components_pbc(b)
        ens_d, div_d = ensemble_metrics(truth[s], samples[:, s], mean[s], obs[s], iso["diff"])
        res["diff"]["ens"].append(ens_d); res["diff"]["div"].append(div_d)
        res["base"]["ens"].append(res["base"]["nov"][-1]); res["base"]["div"].append(0.0)  # deterministic
        row["diff_ens"] = ens_d; row["base_ens"] = res["base"]["nov"][-1]
        per_species_rows.append(row)

ks = lambda a, b: float(ks_2samp(a, b).statistic)
mn = lambda x: (float(np.mean([v for v in x if not np.isnan(v)]))
                if any(not np.isnan(v) for v in x) else float('nan'))
N = len(res["diff"]["TR"])

def line(label, dv, bv, pct=False):
    f = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.3f}")
    print(f"  {label:<26}{f(dv):>14}{f(bv):>18}")

print(f"  K={K}   report species (truth>{K}) = {N}   smoother sigma = {best_sigma}")
print("  " + "-"*58)
print(f"  {'':<26}{'DIFFUSION':>14}{'GAUSS-SMOOTHER':>18}")
print("  " + "-"*58)
print("  DISTRIBUTIONAL (lower KS = better; Axel bar <= 0.30)  [diffusion must be UNCHANGED]")
line("range-size KS",    ks(res['diff']['TR'],res['diff']['PR']), ks(res['base']['TR'],res['base']['PR']))
line("connectance KS",   ks(res['diff']['TC'],res['diff']['PC']), ks(res['base']['TC'],res['base']['PC']))
line("spatial-spread KS",ks(res['diff']['TS'],res['diff']['PS']), ks(res['base']['TS'],res['base']['PS']))
print("  " + "-"*58)
print("  POINTWISE (higher = better hit-rate; smoother EXPECTED to win)")
line("recall novel",  mn(res['diff']['nov']),  mn(res['base']['nov']),  pct=True)
line("recall NEAR",   mn(res['diff']['near']), mn(res['base']['near']), pct=True)
line("recall FAR",    mn(res['diff']['far']),  mn(res['base']['far']),  pct=True)
line("AUC novel cells",mn(res['diff']['auc']), mn(res['base']['auc']))
print("  " + "-"*58)
print("  ENSEMBLE (Axel: 'truth is part of this ensemble'; diversity = different-each-run)")
line("ENS-union recall",   mn(res['diff']['ens']), mn(res['base']['ens']), pct=True)
line("ensemble diversity", mn(res['diff']['div']), mn(res['base']['div']))
print("    smoother is deterministic -> ENS = its single-map recall, diversity = 0 by construction")
print("  " + "-"*58)
sp_d = ks(res['diff']['TS'],res['diff']['PS']); sp_b = ks(res['base']['TS'],res['base']['PS'])
en_d = mn(res['diff']['ens']); en_b = mn(res['base']['ens'])
print(f"  spatial-spread: smoother {sp_b:.3f} vs diffusion {sp_d:.3f} -> "
      f"{'SMOOTHER WORSE (expected)' if sp_b > sp_d else 'CHECK: not worse'}")
print(f"  ensemble cover: diffusion {en_d:.0%} vs smoother {en_b:.0%} -> "
      f"{'DIFFUSION COVERS MORE' if en_d > en_b else 'CHECK: ensemble not ahead'}")

OUT = Path("figures_map_axel_stage2_new/unseen_eval"); OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / f"baseline_compare_K{K}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["truth_range","diff_pred_range","base_pred_range",
                                      "diff_pred_patches","base_pred_patches","diff_ens","base_ens"])
    w.writeheader()
    for r in per_species_rows: w.writerow(r)
print(f"\n  per-species CSV -> {OUT}/baseline_compare_K{K}.csv")
