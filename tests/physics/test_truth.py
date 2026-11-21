"""Truth rows (physics tier) in the absorbing sensor sphere of ``toys.sphere``: a closed form, chunks and the frame.

- the absorbed rows per emitted photon against the closed form (1/2) int exp(-r(mu) / L) dmu (every photon that
  reaches the sphere is absorbed on it), at roulette 1e-4, 0.3 and 1;
- the rows are the same photons whatever the chunks (each photon's uniforms come from its id);
- the hit point is on the sphere (radius A in the sensor's frame), its angle the point's polar angle; the arrival
  time is the path length over the group velocity, at least the straight distance's;
- the event axis: per-label arrivals add up to the run's;
- speed="fast" (the diffusion jump) in a Rayleigh liquid: the rows are those of the exact walk in law (count, time
  profile, per-label counts), and the run records the fast level.
"""

import numpy as np
from scipy.stats import chi2 as chi2_dist
from test_sources import L_ABS, Z0, on_axis, pmt_sphere  # noqa: F401 -- pmt_sphere is a fixture
from toys import A

import lumen
from lumen.readout import Truth
from lumen.tools import stats
from lumen.transport.step import Controls

N = 1 << 16


def point(z, **kw):
    """A Point of N photons at z on the axis: one photon per row, as truth rows need."""
    return lumen.sources.Point((0.0, 0.0, z), n_photons=float(N), lam=400.0, **kw)


def test_truth_rows_against_the_closed_form(pmt_sphere):  # noqa: F811
    p = on_axis(Z0, L_ABS)[0]
    for r in (1e-4, 0.3, 1.0):
        h = lumen.simulate(pmt_sphere, point(Z0), samples=N, seed=11, truth=Truth(keep="absorbed"),
                           controls=Controls(roulette=r)).truth  # fmt: skip
        n = len(h)
        assert abs(n - N * p) < stats.z_limit(3) * np.sqrt(N * p * (1 - p)), (r, n, N * p)  # binomial, 3 runs
        assert np.all(h.detected)  # QE 1 over the band, efficiency 1


def test_truth_rows_do_not_depend_on_the_chunks_and_lie_on_the_sensor(pmt_sphere):  # noqa: F811
    spec = Truth(keep="absorbed", angle=True, position=True)
    a = lumen.simulate(pmt_sphere, point(Z0), samples=N, seed=5, truth=spec).truth
    b = lumen.simulate(pmt_sphere, point(Z0), samples=N, seed=5, truth=spec, chunk=5000).truth
    for k in ("photon", "sensor", "time", "angle", "position"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    np.testing.assert_allclose(np.linalg.norm(a.position, axis=1), A, atol=1e-4)
    rho = np.hypot(a.position[:, 0], a.position[:, 1])
    np.testing.assert_allclose(a.angle, np.arctan2(rho, a.position[:, 2]), atol=1e-5)
    straight = np.linalg.norm(a.position - np.array([0.0, 0.0, Z0]), axis=1)  # the sphere's frame is the world's
    assert np.all(a.time >= straight / 0.3 * 1.0 - 1e-3)  # never faster than light in vacuum


def test_the_event_axis_adds_up_to_the_run(pmt_sphere):  # noqa: F811
    src = point(Z0, label=np.arange(N) % 4)
    base = lumen.simulate(pmt_sphere, src, samples=N, seed=3)
    light = lumen.simulate(pmt_sphere, src, samples=N, seed=3, events=4)
    np.testing.assert_allclose(np.asarray(light.arrivals).sum(0), np.asarray(base.arrivals), rtol=1e-6)
    (g,) = base.groups
    assert light.event(2).group(g).charge.shape == base.group(g).charge.shape
    np.testing.assert_allclose(np.asarray(light.group(g).charge).sum(0), np.asarray(base.group(g).charge),
                               rtol=1e-5, atol=1e-9)  # fmt: skip


def test_truth_rows_with_the_jump_are_the_exact_walk_s_in_law(toy):
    """A Rayleigh liquid ball (0.1 m scattering length, 10 m absorption, every wall photon absorbed): the analog rows of
    speed="fast" and "exact" (independent photons) agree in total, per label and in their time profile, and both
    agree with the expected arrivals; the run records the level it ran."""
    det = toy("sphere", r=0.0, absorption=10.0, rayleigh=0.1).det
    n = 1 << 15
    src = lumen.sources.Point((0.0, 0.0, 0.2), n_photons=1.0, lam=400.0, label=np.arange(n) % 3)  # a photon per row
    rows = {}
    for sp, seed in (("exact", 21), ("fast", 22)):
        light = lumen.simulate(det, src, samples=n, seed=seed, speed=sp, truth=Truth(keep="absorbed"))
        assert dict(light.provenance)["speed"] == sp
        rows[sp] = light.truth
    want = lumen.simulate(det, src, samples=n, seed=23, speed="fast")
    a, e = float(np.sum(want.arrivals)), float(np.sqrt(np.sum(np.square(want.arrivals_error))))
    ex, fa = rows["exact"], rows["fast"]
    for h in (ex, fa):
        assert abs(len(h) - a) < stats.z_limit(2) * np.sqrt(a + e * e), (len(h), a)  # Poisson and 8 replicas
    for lab in range(3):
        m, k = np.sum(ex.label == lab), np.sum(fa.label == lab)
        assert abs(m - k) < stats.z_limit(3) * np.sqrt(m + k), (lab, m, k)  # Poisson, 3 labels
    edges = np.quantile(np.r_[ex.time, fa.time], np.linspace(0.0, 1.0, 11))
    edges[-1] += 1.0
    m, k = np.histogram(ex.time, edges)[0], np.histogram(fa.time, edges)[0]
    chi2 = float(np.sum((m - k) ** 2 / np.maximum(m + k, 1)))
    assert chi2 < chi2_dist.isf(stats.ALPHA / stats.SUITE_TESTS, 9), (chi2, m, k)  # 9 dof at the suites' level
