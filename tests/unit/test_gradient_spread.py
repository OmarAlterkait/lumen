"""The gradient check's spread band, calibrated by the FD replicas' kurtosis (lumen.validate.gradients SPREAD).

The groups' mean z^2 is tested against null + 3 sqrt((kappa - 1) / ndf), kappa the per-replica FD values' pooled
kurtosis (at least 3). Gaussian replicas keep the Pearson band; Student-t(5) replicas (kurtosis 9) get a wider one
and a false-positive rate near the 3-sigma 0.13 % (checked loosely over a few hundred seeds). K = 32, 50 groups.
"""

import numpy as np

from lumen.validate import gradients as g
from lumen.validate import stats

K, G = 32, 50


def _compared(x):
    fd = -np.concatenate([x, x.sum(1, keepdims=True)], 1)  # AD 0, FD noise about 0: the null
    return g.compare(np.zeros_like(fd), fd, np.zeros_like(fd), 1.0 / 64, 1)


def test_gaussian_replicas_keep_the_pearson_band():
    c = _compared(np.random.default_rng(0).standard_normal((K, G)))
    assert c["kappa"] == 3.0  # the sample kurtosis of Gaussian replicas falls below 3: clipped
    old = c["null"] + max(3.0 * np.sqrt(2.0 / c["ndf"]), stats.CHI2_FLOOR * c["null"])
    new = c["null"] + max(3.0 * np.sqrt((c["kappa"] - 1.0) / c["ndf"]), stats.CHI2_FLOOR * c["null"])
    assert new == old
    assert g.spread_passes(c) == stats.chi2_passes(c["chi2"], c["ndf"], c["null"])


def test_student_t5_replicas_are_calibrated():
    seeds = 300
    cs = [_compared(np.random.default_rng(s).standard_t(5, (K, G))) for s in range(seeds)]
    assert 3.3 < np.mean([c["kappa"] for c in cs]) < 6.0  # heavy tails seen (t(5): 9; K = 32 samples see less)
    fp = sum(not g.spread_passes(c) for c in cs) / seeds
    assert fp <= 0.01, fp  # near 0.13 %: at most 3 of 300
