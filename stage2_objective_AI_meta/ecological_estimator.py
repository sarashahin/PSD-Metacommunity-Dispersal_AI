# ecological_estimator.py — ONE truth-free occupancy estimator for all Axel figures.
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy import ndimage

CONN = ndimage.generate_binary_structure(2, 1)

def fit_isotonic(calib_pairs, nbins=200):
    """calib_pairs: iterable of (mean_prob (S,Y,X), truth_bin (S,Y,X)) from a
    DISJOINT calibration set. Uses only aggregate freq, never per-species truth."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    n_acc = np.zeros(nbins); pos = np.zeros(nbins)
    for mp, tb in calib_pairs:
        n = min(mp.shape[0], tb.shape[0]); mp, tb = mp[:n], tb[:n]
        b = np.clip(np.digitize(mp.ravel(), edges) - 1, 0, nbins - 1)
        np.add.at(n_acc, b, 1); np.add.at(pos, b, tb.ravel())
    c = 0.5 * (edges[:-1] + edges[1:]); m = n_acc > 0
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip', increasing=True)
    iso.fit(c[m], pos[m] / n_acc[m], sample_weight=n_acc[m])
    return iso

def expected_count_binary(prob_2d, iso):
    """Truth-FREE binary: N = round(sum of calibrated p); take the top-N cells."""
    cp = (iso.predict(prob_2d.ravel()).reshape(prob_2d.shape).astype(np.float32)
          if iso is not None else prob_2d.astype(np.float32))
    N = int(round(float(cp.sum()))); N = max(0, min(N, cp.size))
    out = np.zeros(cp.size, dtype=np.uint8)
    if N > 0:
        out[np.argpartition(cp.ravel(), -N)[-N:]] = 1
    return out.reshape(prob_2d.shape), N

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

def periodic_cov_det(b, Y=20, X=20):
    yy, xx = np.where(b > 0.5)
    if len(yy) < 2: return 0.0
    ty, tx = 2*np.pi*yy/Y, 2*np.pi*xx/X
    my = np.arctan2(np.sin(ty).mean(), np.cos(ty).mean())
    mx = np.arctan2(np.sin(tx).mean(), np.cos(tx).mean())
    dy = ((ty - my + np.pi) % (2*np.pi) - np.pi) * Y / (2*np.pi)
    dx = ((tx - mx + np.pi) % (2*np.pi) - np.pi) * X / (2*np.pi)
    return max(0.0, np.var(dy)*np.var(dx) - (((dy-dy.mean())*(dx-dx.mean())).mean())**2)
