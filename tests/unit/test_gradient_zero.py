"""The gradient check's structurally zero family (lumen.validate.gradients ZERO).

A family whose AD is exactly 0 while its outcome is drawn is judged on its totals only (FD consistent with 0, FDX 0);
an exactly-0 AD whose outcome was never proposed (the edge rows missing at an edge) is not zero and fails as any
family whose FD sees the derivative. Synthetic replicas (K = 32, 50 groups and the total) stand in for the runs.
"""

import numpy as np

from lumen.validate import gradients as g

K, G, N, H = 32, 50, 16384, 1.0 / 64


def _reps(mean, sd, seed):
    r = np.random.default_rng(seed).normal(mean, sd, size=(K, G))
    return np.concatenate([r, r.sum(-1, keepdims=True)], -1)


def _judged(ad, fd, fdx, proposed, no_draw):
    v0 = np.full((K, G + 1), 100.0)
    cq, ct = (g.compare(ad, fd, v0, H, N) for _ in range(2))
    xq, xt = (g._with_rounding(g.compare(ad, fdx, v0, H / 16, N), g.compare(ad, fdx, v0, H / 64, N)) for _ in range(2))
    if g.is_zero((ad, ad), proposed, no_draw):
        return "zero", g.judge_zero(cq, ct, xq, xt, proposed, no_draw)
    return "family", g.judge(cq, ct, xq, xt, no_draw)


def test_a_structurally_zero_family_passes_on_its_totals():
    # AD exactly 0, the edge rows proposed the outcome; FD only heavy-tailed noise about 0, FDX exactly 0
    ad, fdx = np.zeros((K, G + 1)), np.zeros((K, G + 1))
    noise = np.random.default_rng(3).standard_t(2, size=(K, G)) * 0.05
    fd = np.concatenate([noise, noise.sum(-1, keepdims=True)], -1)
    kind, (passed, note) = _judged(ad, fd, fdx, proposed=481, no_draw=False)
    assert kind == "zero" and passed, note
    assert "proposed its outcome 481 times" in note


def test_an_exactly_zero_ad_without_the_outcome_proposed_still_fails():
    # the edge rows disabled at an edge: the outcome is never drawn, AD is 0, the FD sees the derivative (1 / photon)
    ad = np.zeros((K, G + 1))
    fd = _reps(N / G, 5.0, 1)
    kind, (passed, note) = _judged(ad, fd, fd, proposed=0, no_draw=False)
    assert kind == "family" and passed is False, note


def test_a_zero_family_whose_fd_total_is_off_zero_fails():
    ad, fdx = np.zeros((K, G + 1)), np.zeros((K, G + 1))
    fd = _reps(2.0, 1.0, 2)  # a missed term: the FD total is resolved away from 0
    kind, (passed, note) = _judged(ad, fd, fdx, proposed=100, no_draw=False)
    assert kind == "zero" and not passed and "FD total off 0" in note
