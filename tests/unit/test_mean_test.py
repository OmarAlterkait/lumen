"""The comparisons' one test (``stats.mean_test``) and the exactness ladder (``stats.fd_order``) on synthetic data
with a known null and known alternatives: size at most the level, power where a difference exists.

Scenarios (K = 64 seeds, 20 groups, Q and Tw = Q x time per group, and the totals): Gaussian; heavy tails (sums of
Pareto(2.5) photons); rare events (one sample carries events of 0.1 at 0.016 per group and seed, the other events of
0.01 at ten times that rate: WCTE Poron's two roulettes, where the groups' t band rejected 97 % of null runs); Q and Tw
nearly the same light (WCTE Reflector); independent samples of 16 seeds (the value test).
"""

import math

import numpy as np
import pytest

from lumen.tools import stats

K, G, LEVEL, REPS = 64, 20, 0.05, 200


def _layout(q, t):
    tw = q * t
    return np.concatenate([q, tw, q.sum(1, keepdims=True), tw.sum(1, keepdims=True)], 1)


def _gauss(rng, shift):
    common, t = rng.normal(10, 1, (K, G)), 20 + rng.normal(0, 1, (K, G))
    a, b = common + rng.normal(0, 0.3, (K, G)), common + rng.normal(0, 0.3, (K, G))
    a[:, 0] += shift * 0.3
    return _layout(a, t), _layout(b, t), True


def _heavy(rng, shift):
    t = 20 + rng.normal(0, 1, (K, G))
    a, b = ((rng.pareto(2.5, (K, G, 300)) - 1 / 1.5).sum(-1) for _ in range(2))
    a[:, 0] += shift * math.sqrt(300 * 2.5 / (1.5**2 * 0.5))
    return _layout(a, t), _layout(b, t), True


def _rare(rng, shift):
    a = 0.1 * rng.poisson(0.016, (K, G)) * (1 + 0.5 * rng.random((K, G)))
    b = 0.01 * rng.poisson(0.16, (K, G)) * (1 + 0.5 * rng.random((K, G))) + 0.02 * shift
    t = 20 + rng.normal(0, 3, (K, G))
    return _layout(a, t), _layout(b, t), True


def _correlated(rng, shift):
    common, t = rng.normal(0.6, 0.25, (K, G)), 20 + rng.normal(0, 0.3, (K, G))
    a, b = common + rng.standard_t(6, (K, G)) * 0.0035, common.copy()
    a[:, 0] += shift * 0.0035 * math.sqrt(1.5)
    return _layout(a, t), _layout(b, t), True


def _independent(rng, shift):
    a, b, t = rng.normal(10, 1, (16, G)), rng.normal(10, 1, (16, G)), 20 + rng.normal(0, 1, (16, G))
    a[:, 0] += shift
    return _layout(a, t), _layout(b, t), False


def _rate(gen, shift):
    rng, hits = np.random.default_rng(1), 0
    for _ in range(REPS):
        a, b, paired = gen(rng, shift)
        hits += stats.mean_test(a, b, paired=paired, level=LEVEL)["passed"] is False
    return hits / REPS


# the binomial bound on REPS null runs at LEVEL: mean + 3 sd
SIZE_MAX = LEVEL + 3 * math.sqrt(LEVEL * (1 - LEVEL) / REPS)


@pytest.mark.parametrize("gen", [_gauss, _heavy, _rare, _correlated, _independent])
def test_the_size_is_at_most_the_level(gen):
    assert _rate(gen, 0.0) <= SIZE_MAX


@pytest.mark.parametrize(("gen", "shift", "power"), [(_gauss, 0.7, 0.6), (_heavy, 0.7, 0.6), (_rare, 1.0, 0.8),
                                                    (_correlated, 0.7, 0.9), (_independent, 1.5, 0.5)])  # fmt: skip
def test_a_difference_is_found(gen, shift, power):
    assert _rate(gen, shift) >= power


def test_the_groups_t_band_fails_rare_events_and_the_test_does_not():
    rng = np.random.default_rng(3)
    a, b, _ = _rare(rng, 0.0)
    d = a - b
    z = d.mean(0) / np.where(stats.sem(d) > 0, stats.sem(d), np.inf)
    assert np.mean(z[: 2 * G] ** 2) > 3.0  # the old band's statistic, far beyond its null of 1.03
    r = stats.mean_test(a, b, paired=True)
    assert r["passed"] and not r["power"][:G].all()  # groups with about one event in 64 seeds have no power


def test_power_is_the_levels_floor():
    level = stats.ALPHA / 10
    floor = stats.power_floor(level)
    assert math.exp(-floor) == pytest.approx(level)
    x = np.zeros((K, 3))
    x[:2, 0] = 1.0  # carried by 2 seeds
    x[:, 1] = 1.0 + 0.01 * np.random.default_rng(0).normal(size=K)  # by every seed
    x[:, 2] = np.where(np.arange(K) < math.ceil(floor) + 1, 1.0, 0.0)  # by just more seeds than the floor
    r = stats.mean_test(x, x * 1.0, paired=True, level=level)
    assert list(r["power"]) == [False, False, False]  # no spread: nothing to test
    r = stats.mean_test(x, x[::-1], paired=False, level=level)
    assert list(r["power"]) == [False, True, True]


def test_a_row_with_no_output_with_power_is_not_gated():
    x = np.zeros((K, 4))
    x[0] = 1.0
    assert stats.mean_test(x, np.zeros((K, 4)), paired=True)["passed"] is None


def test_correlated_outputs_count_once():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(K, 1))
    a = np.concatenate([z] * 10, 1) + 1e-6 * rng.normal(size=(K, 10))
    one = stats.mean_test(a[:, :1] + 0.4, np.zeros((K, 1)), paired=True, level=LEVEL)
    ten = stats.mean_test(a + 0.4, np.zeros((K, 10)), paired=True, level=LEVEL)
    assert ten["p"] == pytest.approx(one["p"], rel=0.25, abs=2e-3)  # ten copies are one output, by the data


def _ladder(f, h0, one_sided):
    hs = [h0 / 2**k for k in range(4)]
    return {x: f(x) for x in (hs if one_sided else hs + [-h for h in hs])}


@pytest.mark.parametrize("one_sided", [False, True])
def test_the_ladder_tests_a_smooth_realisation_and_finds_a_wrong_derivative(one_sided):
    f0 = np.full((4, 3), 100.0)
    f = lambda x: f0 + 2.0 * x + 30.0 * x**2 + 50.0 * x**3  # noqa: E731
    h0 = 1.0 / 1024 if one_sided else 1.0 / 64  # the check's H_EDGE and H
    lad = _ladder(f, h0, one_sided)
    ok = stats.fd_order(lad, f0, np.full((4, 3), 2.0), h0, one_sided, 1e-9, 1e-4)
    assert ok["applies"] and ok["tested"] == 1.0 and ok["ratio"] <= 1.0
    bad = stats.fd_order(lad, f0, np.full((4, 3), 2.0 * (1 + 1e-2)), h0, one_sided, 1e-9, 1e-4)
    assert bad["applies"] and bad["ratio"] > 1.0


def test_a_square_root_realisation_is_not_tested_and_its_order_is_measured():
    # F(x) = 2 x + sum_i c_i sqrt(delta_i + x)_+ with delta_i spread down to ~0: smooth at x = 0, never at a fixed step
    rng = np.random.default_rng(2)
    delta = rng.random(4000) ** 2 * 1e-2
    f0 = np.full((1, 1), 0.0)
    f = lambda x: np.full((1, 1), 2.0 * x + np.sum(np.sqrt(np.clip(delta + x, 0, None)) - np.sqrt(delta)) * 1e-3)  # noqa: E731
    ad = np.full((1, 1), 2.0 + np.sum(0.5 / np.sqrt(delta)) * 1e-3)
    r = stats.fd_order(_ladder(f, 1.0 / 64, False), f0, ad, 1.0 / 64, False, 1e-12, 1e-4)
    assert not r["applies"] and r["order"] < 1.0  # far below the stencil's 2


def test_a_difference_with_no_spread_fails():
    a = np.ones((K, 3))
    b = a.copy()
    b[:, 1] += 1e-3
    assert stats.mean_test(a, b, paired=True)["passed"] is False
    assert stats.mean_test(a, a.copy(), paired=False)["passed"] is None  # identical, no spread: nothing to test


def test_a_known_floor_adds_in_quadrature():
    """A paired difference with no spread over the seeds is certain; a known resolution (``floor``, e.g. a finite
    difference's float32 sides) larger than it makes it agree, and one smaller still rejects it."""
    a = np.random.default_rng(3).normal(size=(16, 3)) + 10.0
    assert stats.mean_test(a, a + 1e-3, paired=True)["passed"] is False
    assert stats.mean_test(a, a + 1e-3, paired=True, floor=np.full(3, 1e-2))["passed"] is True
    assert stats.mean_test(a, a + 1e-3, paired=True, floor=np.full(3, 1e-5))["passed"] is False


def test_a_kinks_growth_is_resolved_at_half_an_ulp():
    """A cell whose differences grow as h^-1/2 (a square-root kink closer than the ladder's last step, SBND polish:
    -0.66, -0.95, -1.35, -1.92 on one photon against AD -29) is not taken as converged once the differences are judged
    at the sides' half-ulp resolution, though a looser allowance hides the growth."""
    h0, f0 = 1.0 / 64, np.full((1, 1), 2000.0)
    kink = lambda x: np.sign(x) * 0.03 * np.sqrt(abs(x))  # noqa: E731 -- odd: its central difference ~ h^-1/2
    lad = _ladder(lambda x: f0 + 25.0 * x + kink(x), h0, False)
    ad = np.full((1, 1), 25.0 + 30.0)
    loose = stats.fd_order(lad, f0, ad, h0, False, 1e-6, 1e-4)
    tight = stats.fd_order(lad, f0, ad, h0, False, 1e-6, 1e-4, resolve=2.0**-24)
    assert loose["applies"] and loose["ratio"] > 1.0  # the allowance hides the growth: a false failure
    assert not tight["applies"]  # resolved: not converging, reported
