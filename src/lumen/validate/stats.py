"""The checks' statistics: errors from replicas, the pass rules, finite differences and the two-sample time test.

Rules (SPEC section 15, D45, D65, D68):

- a ratio passes when |z| <= ``Z_MAX`` (3);
- a per-sensor Pearson chi2 / ndf passes when it is at most 1 + max(3 sqrt(2 / ndf), ``CHI2_FLOOR``): three standard
  deviations of chi2 / ndf, but never tighter than a 5 % excess, which a per-sensor model difference far below the
  ratio's error already gives on SK's 11,096 sensors;
- K replicas give a t statistic with K - 1 degrees of freedom: E[z^2] = (K - 1) / (K - 3), the null of a groups'
  mean z^2, which passes within null +- 3 sqrt(2 / ndf).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstwobign

Z_MAX = 3.0
CHI2_FLOOR = 0.05


def chi2_passes(chi2_ndf, ndf, null=1.0) -> bool:
    """Return whether a chi2 / ndf is within the rule's band above its null."""
    return bool(ndf == 0 or chi2_ndf <= null + max(3.0 * np.sqrt(2.0 / ndf), CHI2_FLOOR * null))


def t_null(k: int) -> float:
    """Return E[t^2] of a t statistic over k replicas (k - 1 degrees of freedom)."""
    return (k - 1) / (k - 3) if k > 3 else float("nan")


def sem(v, axis=0):
    """Return the standard error of the mean over ``axis`` (the replicas)."""
    v = np.asarray(v, np.float64)
    return v.std(axis, ddof=1) / np.sqrt(v.shape[axis])


def richardson(f, h, one_sided=False):
    """Return the derivative of f at 0: O(h^4) central (4 sides), or O(h^2) one-sided inward (f at 0, h/2, h)."""
    if one_sided:
        return (-3.0 * f(0.0) + 4.0 * f(h / 2) - f(h)) / h
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(h / 2) - f(-h / 2)) / h
    return (4.0 * d2 - d1) / 3.0


def node_split(t, dt, nt):
    """Return (nodes (nt,), late, early): times split onto the nodes k dt linearly, as the transport splits them."""
    x = np.asarray(t, np.float64) / dt
    j = np.floor(x).astype(np.int64)
    g = x - j
    h, late, early = np.zeros(nt), 0.0, 0.0
    for jj, w in ((j, 1.0 - g), (j + 1, g)):
        ok = (jj >= 0) & (jj < nt)
        np.add.at(h, jj[ok], w[ok])
        late += float(w[jj >= nt].sum())
        early += float(w[jj < 0].sum())
    return h, late, early


def times(h_lumen, h_ref, dt, n_lumen):
    """Return the time comparison of two node histograms: mean and width each, the mean's difference and error, KS.

    The KS test uses the reference's hit count and lumen's detected weight ``n_lumen`` as the samples' sizes.
    """
    tn = dt * np.arange(len(h_ref))
    sl, sr = h_lumen.sum(), h_ref.sum()
    if sl <= 0 or sr <= 0:
        return dict(t_lumen=np.nan, t_ref=np.nan, dt=np.nan, dt_err=np.nan, ks_d=np.nan, ks_p=np.nan)
    ml, mr = (h_lumen * tn).sum() / sl, (h_ref * tn).sum() / sr
    wl = np.sqrt((h_lumen * (tn - ml) ** 2).sum() / sl)
    wr = np.sqrt((h_ref * (tn - mr) ** 2).sum() / sr)
    d = float(np.max(np.abs(np.cumsum(h_lumen) / sl - np.cumsum(h_ref) / sr)))
    en = np.sqrt(sr * n_lumen / (sr + n_lumen))
    return dict(t_lumen=ml, t_ref=mr, w_lumen=wl, w_ref=wr, dt=ml - mr, dt_err=wr / np.sqrt(sr), ks_d=d,
                ks_p=float(kstwobign.sf(d * en)))  # fmt: skip
