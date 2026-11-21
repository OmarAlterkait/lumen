"""Truth rows, the event axis, source labels and hit axes, on the small toy detector (plain-JAX backend)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_api import _toy, _two_groups
from test_importer import write

import lumen
from lumen import readout as rd
from lumen.readout import Truth
from lumen.readout import grid as tg
from lumen.sensors import value_sets
from lumen.sources import CherenkovSteps, FromArrays, Photons, Point, source
from lumen.transport import state
from lumen.transport import step as steps

N = 4096
INSIDE = (0.0, 0.0, 0.3)  # in the toy's water, 0.3 m from its first PMT


@pytest.fixture(scope="module")
def det(tmp_path_factory):
    from test_api import RECIPE

    tmp = tmp_path_factory.mktemp("truth")
    return lumen.load(lumen.convert(write(tmp, _toy()), RECIPE, tmp / "toy.lumen"))


def run(det, src, **kw):
    return lumen.simulate(det, src, samples=kw.pop("samples", N), seed=kw.pop("seed", 1), backend="xla", **kw)


@pytest.mark.slow  # minutes on 4 CPUs (measured 2-6 min on roma, 4 workers)
def test_truth_rows_are_one_absorption_per_photon_on_the_sensors(det):
    """Absorbed rows: at most one per photon, in photon order, on a sensor, with the hit point on its surface (the
    PMT is a tube of radius 0.125 m and half-length 0.05 m in its own frame) and the arrival time after emission."""
    light = run(det, Point(INSIDE, n_photons=float(N), lam=400.0), truth=Truth(keep="absorbed", angle=True,
                                                                               position=True))  # fmt: skip
    h = light.truth
    assert len(h) > 50 and h.n_photons == N and np.all(h.detected)  # a counting group: every absorption counts
    assert np.all(np.diff(h.photon) > 0) and np.all((h.sensor >= 0) & (h.sensor < 2)) and np.all(h.time > 0.0)
    rho, z = np.hypot(h.position[:, 0], h.position[:, 1]), np.abs(h.position[:, 2])
    assert np.all(np.isclose(rho, 0.125, atol=1e-4) | np.isclose(z, 0.05, atol=1e-4))
    np.testing.assert_allclose(h.angle, np.arctan2(rho, h.position[:, 2]), atol=1e-5)
    assert np.all(h.label == -1)  # the source has no labels
    by = h.sum_by()
    np.testing.assert_array_equal(by["n"], np.bincount(h.sensor, minlength=2)[by["sensor"]])


@pytest.mark.slow  # minutes on 4 CPUs (measured 2-6 min on roma, 4 workers)
def test_truth_totals_are_the_expected_arrivals(det):
    """Per sensor, the analog rows count the expected arrivals of independent photons, at roulette 1e-4 and 1."""
    src = Point(INSIDE, n_photons=float(4 * N), lam=400.0)
    want = run(det, src, samples=4 * N, seed=7)
    for r in (1e-4, 1.0):
        h = run(det, src, samples=4 * N, seed=3, truth=Truth(), controls=steps.Controls(roulette=r)).truth
        n = np.bincount(h.sensor, minlength=2)
        a, e = np.asarray(want.arrivals), np.asarray(want.arrivals_error)
        assert np.all(np.abs(n - a) < 4.0 * np.sqrt(a + e * e)), (r, n, a)


@pytest.mark.slow  # minutes on 4 CPUs (measured 2-6 min on roma, 4 workers)
def test_detected_rows_follow_each_group_model(tmp_path):
    """Two modelled groups (lumen:sensor): every absorbed row is kept with its flag, and the detected rows per sensor
    are the expected photoelectrons (efficiency QE CE of the group's model) within Monte Carlo error."""
    det = lumen.load(_two_groups(tmp_path))
    src = Point(INSIDE, n_photons=float(8 * N), lam=400.0)
    light = run(det, src, samples=8 * N, seed=2, gate=(0.0, 100.0), truth=Truth(keep="absorbed"))
    h, p = light.truth, det.params()
    assert 0 < h.detected.sum() < len(h)
    assert h.groups == ("near", "far") and sorted(h.group_of.tolist()) == [0, 1]
    for g in h.groups:
        s = int(np.flatnonzero(h.group_of == h.groups.index(g))[0])
        n = int(np.sum(h.detected & (h.sensor == s)))
        pe = float(rd.pe(light.group(g), p)[0])
        assert abs(n - pe) < 4.0 * np.sqrt(pe) + 1.0, (g, n, pe)
    only = run(det, src, samples=8 * N, seed=2, gate=(0.0, 100.0), truth=Truth()).truth
    np.testing.assert_array_equal(only.photon, h.photon[h.detected])  # keep="detected": the same rows, thinned


def test_truth_refuses_what_it_cannot_record_exactly(det):
    src = Point(INSIDE, n_photons=float(2 * N), lam=400.0)  # two photons per row
    with pytest.raises(ValueError, match="stands for 2 photons"):
        run(det, src, truth=Truth())
    with pytest.raises(ValueError, match="not differentiable"):
        jax.jvp(lambda n: run(det, Point(INSIDE, n_photons=n, lam=400.0), truth=Truth()).arrivals, (float(N),),
                (1.0,))  # fmt: skip
    with pytest.raises(ValueError, match="roulette"):
        run(det, src, truth=Truth(), controls=steps.Controls(roulette=2.0))
    with pytest.raises(ValueError, match="keep"):
        run(det, src, truth=Truth(keep="all"))


def test_truth_rows_are_decided_at_the_run_s_own_values(det):
    """A reference equal to the run's values gives the rows of a run without one; any other reference is refused (the
    photons' fates are read at the live values)."""
    src = Point(INSIDE, n_photons=float(N), lam=400.0)
    p = det.params()
    a, b = run(det, src, truth=Truth()).truth, run(det, src, truth=Truth(), reference=p).truth
    for k in ("photon", "sensor", "time", "detected"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    key = next(k for k in p if k.endswith("/RINDEX"))
    with pytest.raises(ValueError, match="reference= other than params is refused"):
        run(det, src, truth=Truth(), reference=p.scale(key, 1.01))


def test_the_analog_roulette_keeps_an_open_fate_and_the_survival_law():
    """An open analog photon always survives; a decided one with (p - w k) / (1 - w k), so the total is p."""
    an = state.analog_start(jnp.float32(0.1), jnp.float32(0.5))
    assert float(an.x) == pytest.approx(0.2)
    closed = an._replace(x=jnp.float32(jnp.inf))
    w, p = 0.05, 0.05 / 0.3
    u = (jnp.arange(100_000, dtype=jnp.float32) + 0.5) / 100_000
    keep = jax.vmap(lambda r: steps._analog_roulette(closed, w, p, r))(u)
    np.testing.assert_allclose(float(keep.mean()), (p - w * 0.5) / (1 - w * 0.5), atol=1e-4)
    assert bool(steps._analog_roulette(an, w, p, jnp.float32(0.999)))
    assert not bool(state.analog_start(jnp.float32(0.6), jnp.float32(0.5)).x < jnp.inf)  # u >= k: never emitted


def test_sum_by_groups_by_any_integer_columns():
    rng = np.random.default_rng(0)
    n = 1000
    lab, sen, t = rng.integers(-1, 4, n), rng.integers(0, 7, n), rng.random(n)
    det_ = rng.random(n) < 0.7
    h = rd.TruthHits(
        photon=np.arange(n),
        label=lab,
        sensor=sen,
        time=t,
        detected=det_,
        angle=None,
        position=None,
        n_photons=n,
        groups=("g",),
        group_of=np.zeros(7, np.int32),
    )
    out = h.sum_by(("label", "sensor"))
    for a, b, c, f in zip(out["label"], out["sensor"], out["n"], out["t_first"], strict=True):
        m = det_ & (lab == a) & (sen == b)
        assert c == m.sum() and f == t[m].min()
    assert out["n"].sum() == det_.sum() and h.sum_by(("sensor",), detected=False)["n"].sum() == n


@pytest.mark.slow  # minutes on 4 CPUs (measured 2-6 min on roma, 4 workers)
def test_events_split_every_output_by_label(det):
    """events=E: each photon's output goes to its label's event; the events sum to the run without them (same
    photons, same random numbers); Light.event(e) is one event; labels outside [0, E) or none are refused."""
    lab = np.arange(N) % 3
    src = Point(INSIDE, n_photons=1.0, lam=400.0, label=lab)  # a label per row: one photon per row
    base, base_t = run(det, src), run(det, src, truth=Truth())
    plain, light = run(det, src, events=3), run(det, src, events=3, truth=Truth())
    assert light.arrivals.shape == (3, 2) and light.mean_arrival_time.shape == (3, 2) and light.events == 3
    np.testing.assert_allclose(plain.arrivals.sum(0), base.arrivals, rtol=1e-4)
    np.testing.assert_allclose(light.arrivals.sum(0), base_t.arrivals, rtol=1e-4)
    np.testing.assert_array_equal(light.truth.photon, base_t.truth.photon)  # the same rows, now labelled
    one = light.event(1)
    assert one.arrivals.shape == (2,) and np.all(one.truth.label == 1) and dict(one.provenance)["event"] == 1
    assert np.all(light.truth.label == light.truth.photon % 3)
    with pytest.raises(ValueError, match="outside the events"):
        run(det, src, events=2)
    with pytest.raises(ValueError, match="give the source a label"):
        run(det, Point(INSIDE, n_photons=float(N), lam=400.0), events=2)


def test_sources_give_their_rows_labels():
    key = jax.random.PRNGKey(0)
    e = Point((0.0, 0.0, 0.0), lam=400.0, label=7).emit(None, key, 8, 3)
    np.testing.assert_array_equal(e.label, 7)
    e = Point((0.0, 0.0, 0.0), lam=400.0, label=np.arange(20)).emit(None, key, 8, 3)
    np.testing.assert_array_equal(e.label, np.arange(3, 11))
    z3 = np.zeros((5, 3))
    e = FromArrays(z3, np.tile([0.0, 0.0, 1.0], (5, 1)), np.zeros(5), np.full(5, 400.0), np.ones(5),
                   label=np.arange(5) + 10).emit(None, key, 4, 2)  # fmt: skip
    np.testing.assert_array_equal(e.label, [12, 13, 14, 14])  # the rows padded to 8 as every per-row field (inactive)
    a = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    steps_ = CherenkovSteps("Water", start=a, end=a + [1.0, 0, 0], t_start=np.zeros(3), t_end=np.zeros(3),
                            beta_start=np.full(3, 0.99), beta_end=np.full(3, 0.99), charge=np.ones(3),
                            label=np.array([4, 5, 6]))  # fmt: skip
    p = {"materials/Water/RINDEX": lumen.format.calibrations.Table(np.array([2.0, 4.0]), np.array([1.33, 1.34]))}
    e = steps_.emit(p, key, 64)
    np.testing.assert_array_equal(e.label, 4 + np.floor(np.asarray(e.x)[:, 0] / 5.0).astype(int))  # its step's

    @source()
    def two(v, key, n):
        d = jax.random.normal(key, (n, 3))
        d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
        z = jnp.zeros(n)
        return Photons(jnp.zeros((n, 3)), d, z, z + 400.0, z + 1.0 / n, label=jnp.arange(n, dtype=jnp.int32) % 2)

    two.check()
    np.testing.assert_array_equal(two.emit(None, key, 4, 0, 4).label, [0, 1, 0, 1])
    with pytest.raises(TypeError, match="integer"):
        Point((0.0, 0.0, 0.0), lam=400.0, label=0.5).emit(None, key, 2)


def test_axes_put_the_hit_point_on_the_charge_histogram():
    """A value set with extra axes (``axes``: features no table reads): each deposit's weights sum to its lambda x
    theta weight, the hat functions on the hit coordinate place it, and pe (summed over the extra axes) is the grid's
    without them."""
    import readout_toys as toys

    m = toys.model()
    m2 = toys.model(axes={"rho_m": [0.0, 0.05, 0.1, 0.15], "z_m": [0.0, 0.05]})
    ph = toys.photons(500, 3, seed=1)
    rng = np.random.default_rng(1)
    ph["pos"] = np.c_[rng.uniform(-0.1, 0.1, (500, 2)), np.full(500, 0.02)]
    h1, h2 = tg.from_photons(m, (0.0, 120.0), 3, ph, group="pmt"), tg.from_photons(m2, (0.0, 120.0), 3, ph, group="pmt")
    assert h2.charge.shape == h1.charge.shape + (4, 2) and h2.grid.n_cells == 8 * h1.grid.n_cells
    np.testing.assert_allclose(h2.charge.sum((-2, -1)), h1.charge, rtol=1e-5, atol=1e-6)
    p = toys.params(m)
    np.testing.assert_allclose(rd.pe(h2, p), rd.pe(h1, p), rtol=1e-5)
    rho = np.hypot(ph["pos"][:, 0], ph["pos"][:, 1])
    mean_rho = (h2.charge.sum((0, 1, 2, 4)) * np.array([0.0, 0.05, 0.1, 0.15])).sum() / h2.charge.sum()
    np.testing.assert_allclose(mean_rho, np.average(rho, weights=ph["weight"]), rtol=1e-4)  # hats keep the mean
    with pytest.raises(value_sets.SensorValuesError, match="names no axis"):
        toys.model(axes={"phi_deg": [0.0, 1.0]})
