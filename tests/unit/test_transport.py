"""The transport's host-free pieces: the estimator (proposal, choice, edge rows and floor), the photon's parts, the
default item's cells and the trace's integer halves. The step itself is checked by the physics tier (tests/physics)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen import diagnostics
from lumen import run as runner
from lumen._core.safe import safe_div
from lumen._core.static import Static
from lumen.build import lookup
from lumen.compare import photons
from lumen.media import bulk, laws, phase
from lumen.readout import fill
from lumen.readout.grid import Grid
from lumen.record import Photons
from lumen.surfaces import types
from lumen.surfaces.outcomes import DIFFUSE, N_OUTCOMES, SPECULAR
from lumen.transport import estimator, state

F32 = jnp.float32


def baseline_defensive(p_ref, mix):
    """A defensive mixture proposal, written out literally: the estimator is its mix 0."""
    p_ref = jax.lax.stop_gradient(p_ref)
    sup = p_ref > 0.0
    tot = jnp.sum(p_ref)
    n = jnp.maximum(jnp.sum(sup), 1)
    q = (1.0 - mix) * safe_div(p_ref, tot) + mix * sup / n
    return jnp.where(tot > 0.0, q, 0.0)


def test_the_proposal_is_the_baselines_defensive_mix_at_zero_bit_for_bit():
    r = np.random.default_rng(0)
    p = r.uniform(size=(500, 10)).astype(np.float32) * (r.uniform(size=(500, 10)) > 0.4)
    p[0] = 0.0
    a = jax.jit(jax.vmap(estimator.proposal))(jnp.asarray(p))
    b = jax.jit(jax.vmap(lambda x: baseline_defensive(x, 0.0)))(jnp.asarray(p))
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    u = jnp.asarray(r.uniform(size=500), F32)
    k = jax.vmap(estimator.choose)(a, u)
    assert int(k[0]) == -1 and bool(jnp.all(jnp.asarray(p)[jnp.arange(1, 500), k[1:]] > 0))


P = types.PAINTED.name  # the frontpainted type without TRANSMITTANCE (PAINTED_T with T fixed at 0)


def _tables():
    """Two rows of one painted type and a water with Mie: the estimator reads the rows' types and the media."""
    water = bulk.Medium("w", laws.constant("index"), None, (bulk.mie(laws.constant("length")),))
    rows = (lookup.Row(types.PAINTED, "a"), lookup.Row(types.PAINTED, "b"))
    support = np.zeros((2, N_OUTCOMES), bool)
    support[:, list(types.PAINTED.outcomes)] = True
    return SimpleNamespace(rows=rows, media=(water,), table=SimpleNamespace(support=support)), None


def test_edge_rows_come_from_the_parameters_and_their_defensive_roles():
    """Every row of a parameter with a defensive role is an edge row, whatever its value: no value is read."""
    tables, _ = _tables()
    E = estimator.EdgeRows
    assert estimator.edge_rows(tables, [("surfaces", 0, "R"), ("surfaces", 1, "R")]) == E(surfaces=(0, 1))
    assert estimator.edge_rows(tables, [("surfaces", 1, "R")]) == E(surfaces=(1,))
    assert estimator.edge_rows(tables, [("surfaces", 0, "f_spec")]) == E(surfaces=(0,))
    assert not estimator.edge_rows(tables, [("surfaces", 0, "T")])  # PAINTED's T is fixed: no parameter
    mie = ("media", "w", "mie", "lobe", "forward_ratio")
    assert estimator.edge_rows(tables, [mie]) == E(lobes=(("w", "mie", "forward_ratio"),))
    assert not estimator.edge_rows(tables, [("media", "w", "mie", "lobe", "nope")])  # no role: no edge
    assert estimator.edge_rows(tables, []) == E() and not E()


def test_the_edge_floor_proposes_the_zero_outcomes_at_their_share():
    tables, _ = _tables()
    pr = jnp.zeros(N_OUTCOMES, F32)  # R = 0: every continuing outcome has probability 0 (the rest absorbed)
    q = estimator.edge_floor(pr, jnp.int32(0), tables, estimator.EdgeRows(surfaces=(0,)))
    fl = (q > 0) & (pr <= 0)
    support = list(types.PAINTED.outcomes)
    assert sorted(np.flatnonzero(np.asarray(fl)).tolist()) == sorted(support)
    e = estimator.DEFENSIVE_SHARE  # the share e of the whole distribution, absorption included:
    # at S = 0 the draw's total e / (1 - e) is the reference weight's factor, so a photon in a cavity of absorbing
    # walls meets the roulette (a share of S would leave it unchanged at 1: never ended)
    assert float(jnp.sum(jnp.where(fl, q, 0.0))) == pytest.approx(e / (1 - e), rel=1e-6)
    pr = jnp.zeros(N_OUTCOMES, F32).at[SPECULAR].set(0.5)  # half absorbed
    q = estimator.edge_floor(pr, jnp.int32(0), tables, estimator.EdgeRows(surfaces=(0,)))
    fl = (q > 0) & (pr <= 0)
    assert float(jnp.sum(jnp.where(fl, q, 0.0))) == pytest.approx(e / (1 - e), rel=1e-6)  # whatever S
    q2 = estimator.edge_floor(pr, jnp.int32(1), tables, estimator.EdgeRows(surfaces=(0,)))  # not an edge row
    assert np.array_equal(np.asarray(q2), np.asarray(pr))
    e = estimator.DEFENSIVE_SHARE  # Mie's lobe choice: the same rule over (r, 1 - r)
    for r, want in ((1.0, 1.0 - e), (0.0, e), (0.5, 0.5), (0.995, 0.995)):
        lobe = {"forward_ratio": jnp.asarray(r, F32), "g_forward": 0.6}
        assert float(phase.MIE_HG.cover(lobe, e)["forward_ratio"]) == pytest.approx(want, rel=1e-6)


def test_a_covered_spectrum_draws_its_massless_intervals_at_the_share():
    """A spectrum interval with both knots 0 (a live table can put light there) gets the share e of the draws,
    uniform in energy; a table without one is drawn as is, bitwise."""
    from lumen.media import spectra

    e = estimator.DEFENSIVE_SHARE
    en, s = jnp.asarray([2.0, 2.5, 3.0, 3.5], F32), jnp.asarray([1.0, 1.0, 0.0, 0.0], F32)
    e0, h, d, c = spectra.covered(en, s, e)
    np.testing.assert_allclose(np.asarray(d * h), [(1 - e) * 2 / 3, (1 - e) / 3, e], rtol=1e-6)
    at = jnp.asarray([2.2, 2.7, 3.2], F32)
    np.testing.assert_allclose(np.asarray(spectra.covered_density(en, s, at, e)), np.asarray(d), rtol=1e-6)
    full = jnp.asarray([1.0, 2.0, 1.0, 0.5], F32)
    for a, b in zip(spectra.covered(en, full, e), spectra.g4_arrays(en, full), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def _row(n=3):
    z = jnp.zeros(n, F32)
    return Photons(jnp.zeros((n, 3), F32), jnp.tile(jnp.asarray([0.0, 0.0, 1.0]), (n, 1)), z, z + 400.0, z + 1.0,
                   jnp.zeros((n, 2, 3), F32))  # fmt: skip


def test_the_photon_has_only_the_parts_asked_for():
    row = jax.tree_util.tree_map(lambda a: a[0], _row())
    ph = state.start(row, 5, 2)
    assert ph.analog is None and ph.gen is None
    assert int(ph.vol) == 5 and int(ph.cls) == 2 and bool(ph.alive)
    assert float(ph.wr) == float(ph.w) == 1.0  # the reference weight starts at the weight
    assert int(state.start(row, -1, 0, gen=True).gen) == 0


def test_the_checks_loop_is_the_readout_by_sensor_and_band():
    """The checks' loop (``lumen.compare.photons.loop``) is the production loop on a grid of wavelength bands plus
    the time-weighted charge "Tw" of the charge cells."""
    spec = photons.loop(fill.HistogramLayout(3, Grid.bands((300.0, 400.0, 500.0))))
    out = spec.outputs()
    assert out["charge:0"] == out["charge2:0"] == (6, True) and out["Tw"] == (6, True) and "stuck" not in out

    def dep(sensor, lam):  # a deposit record as the readout reads it
        return runner.Record(jnp.int32(sensor), F32(0.5), F32(4.0), F32(lam))

    adds = [(k, int(c), float(v)) for k, c, v in spec.emits(dep(2, 450.0))]
    assert adds[:3] == [("charge:0", 2 * 2 + 1, 0.5), ("arrival", 2, 0.5), ("arrival_t", 2, 2.0)]
    assert adds[-1] == ("Tw", 5, 2.0)
    assert [(k, int(c), float(v)) for k, c, v in spec.moments(dep(2, 450.0))] == [
        ("charge2:0", 5, 0.5), ("arrival2", 2, 0.5), ("pe2", 2, 0.0)]  # no acceptance weights: no pe  # fmt: skip
    outside = {k: int(c) for k, c, _ in spec.emits(dep(2, 650.0))}
    assert outside["charge:0"] == outside["Tw"] == -1 and outside["arrival"] == 2  # outside the bands: arrives only
    assert {k: c for k, c, _ in spec.emits(dep(-1, 450.0))}["charge:0"] == -1  # no sensor


def test_the_traces_integers_are_exact_in_two_halves():
    s = np.array([-1, 0, 65535, 65536, 123456789, 2**31 - 2], np.int32)
    hi, lo = diagnostics.halves(jnp.asarray(s))
    assert np.array_equal(
        (np.asarray(hi, np.int64) * diagnostics.HALF + np.asarray(lo, np.int64) - 1).astype(np.int32), s
    )
    assert int(diagnostics.last_event(jnp.int32(3), jnp.int32(-2))) == 1
    assert int(diagnostics.last_event(jnp.int32(3), jnp.int32(0))) == 3  # STRAIGHT keeps it
    assert int(diagnostics.last_event(jnp.int32(3), jnp.int32(DIFFUSE))) == 2 + DIFFUSE
    assert phase.MIE_HG.n_rand == 3  # the bulk layout's lobe block holds Mie's three uniforms


# a model stand-in (hashable: a static argument of the loop)
NO_ANALOG = type("Model", (), {"analog": False, "local": ()})
NO_ANALOG.static = Static(NO_ANALOG)  # the jits' static argument (lumen.transport.step.Model.static)


def test_the_runner_passes_tallymap_its_schedule(monkeypatch):
    """lumen asks for step scheduling and its register cap explicitly, so a change of tallymap's defaults cannot slow
    it silently."""
    from lumen import run

    seen = {}

    def vmap(fn, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(run.tm, "vmap", vmap)
    spec = photons.loop(fill.HistogramLayout(1, Grid.bands((300.0, 500.0))))
    with pytest.raises(RuntimeError, match="stop"):
        run.run(spec, NO_ANALOG, None, _row(2), jnp.zeros(2, jnp.uint32), backend="xla")
    assert seen["schedule"] == run.SCHEDULE == "step" and seen["backend"] == "xla"
    assert seen["min_blocks_per_sm"] == run.MIN_BLOCKS


def test_auto_is_the_kernel_backend_without_a_gpu(monkeypatch):
    """backend="auto" and "cpu": tallymap's kernel backend wherever JAX runs (its C++/OpenMP target when JAX has no
    GPU); "xla" is its plain-JAX backend."""
    from lumen import run

    seen = {}

    def vmap(fn, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(run.tm, "vmap", vmap)
    monkeypatch.setattr(run.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(run._env, "gpu_expected", lambda: None)  # no GPU expected, whatever this job was given
    spec = photons.loop(fill.HistogramLayout(1, Grid.bands((300.0, 500.0))))
    with pytest.raises(RuntimeError, match="stop"):
        run.run(spec, NO_ANALOG, None, _row(2), jnp.zeros(2, jnp.uint32), backend="auto")
    assert seen["backend"] == run.KERNEL == "auto"


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def test_the_hit_angle_reads_only_the_sensors_own_pose():
    """docs/dev/decisions.md, 10: the CE coordinate of a crossing point, frame by frame.

    (a) inside a glass mother and (b) straight in the water: one world pose, one angle (the mother never enters);
    (c) turned 180 degrees about x, its cathode with it: the same angle;
    (d) the solid's origin moved along its axis, (e) its axis off z: the angle changes, the documented 0.2 limit (a
    declared axis and reference point).
    """
    from lumen.sensors.features import polar_angle

    def sensor_angle(R, t, x):  # the deposit's point in the placement's frame (step._local), then the feature
        return polar_angle(R.T @ (x - t))

    rng = np.random.default_rng(0)
    u = rng.normal(size=(64, 3)).astype(np.float32)
    local = 0.25 * u / np.linalg.norm(u, axis=1, keepdims=True)
    local[:, 2] = np.abs(local[:, 2])  # the cathode: the +z half of a sphere about the solid's origin
    want = np.arccos(np.abs(local[:, 2]) / 0.25)
    R, t = _rot_x(0.3), np.array([1.0, -2.0, 0.5], np.float32)
    world = local @ R.T + t
    ang = jax.vmap(lambda x: sensor_angle(jnp.asarray(R), jnp.asarray(t), x))
    a = np.asarray(ang(jnp.asarray(world)))
    np.testing.assert_allclose(a, want, atol=2e-6)  # (a) = (b): the sensor's world pose only
    Rf = R @ _rot_x(np.pi)  # (c) the PMT turned over, its cathode points with it
    np.testing.assert_allclose(np.asarray(jax.vmap(lambda x: sensor_angle(jnp.asarray(Rf), jnp.asarray(t), x))(
        jnp.asarray(local @ Rf.T + t))), want, atol=2e-6)  # fmt: skip
    behind = np.asarray(ang(jnp.asarray(local * np.array([1, 1, -1], np.float32) @ R.T + t)))
    np.testing.assert_allclose(behind, np.pi - want, atol=2e-6)  # signed: a point behind the plane is past 90 degrees
    t_d = t + R @ np.array([0.0, 0.0, 0.05], np.float32)  # (d) the origin 5 cm up the axis
    a_d = np.asarray(jax.vmap(lambda x: sensor_angle(jnp.asarray(R), jnp.asarray(t_d), x))(jnp.asarray(world)))
    assert np.max(np.abs(a_d - want)) > 1e-2
    R_e = R @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)  # (e) the axis along local y
    a_e = np.asarray(jax.vmap(lambda x: sensor_angle(jnp.asarray(R_e), jnp.asarray(t), x))(jnp.asarray(world)))
    assert np.max(np.abs(a_e - want)) > 1e-1


def test_a_trace_writes_only_the_traced_photons():
    """Trace's cells are indexed by the photon: one beyond n_traced writes nothing (it was left to tallymap's drops)."""
    import jax.numpy as jnp

    from lumen.diagnostics import Trace

    tr = Trace(steps=2, n_traced=3)
    ph = SimpleNamespace(alive=True, vertex=jnp.int32(0), x=jnp.zeros(3), d=jnp.zeros(3), vol=0, cls=0, srf=0)
    dep = SimpleNamespace(event=jnp.int32(-1), lost=0.0, abandoned=0.0, pair=0)
    for pid, written in ((2, True), (3, False)):
        _, adds = tr.update(tr.init(ph), ph, ph, dep, jnp.int32(pid))
        assert all((int(cell) >= 0) == written for _, cell, _ in adds)


def test_the_time_histogram_holds_the_detected_light_at_the_values_given():
    """A deposit fills a weighted time histogram (Grid.weighted) by the QE x CE of its charge cells read from the
    run's weights (an operand), so light those values do not detect does not shape the time; the charge histogram is
    unweighted; a weighted grid without weights is refused."""
    g = Grid.bands((-np.inf, 200.0, np.inf), t0=0.0, dt=1.0, nt=8, weighted=True)
    loop = fill.readout((fill.HistogramLayout(1, g),))

    def dep(lam):  # a deposit record as the readout reads it
        return runner.Record(jnp.int32(0), F32(2.0), F32(3.0), F32(lam), label=jnp.int32(0))

    for tw in ((0.0, 0.25), (0.5, 1.0)):
        for lam, cell in ((128.0, 0), (450.0, 1), (1500.0, 1)):  # the outer bands hold beyond the edges
            adds = loop.emits(dep(lam), (jnp.asarray(tw, F32),))
            got = sum(float(v) for k, c, v in adds if k == "time:0" and int(c) >= 0)
            assert got == pytest.approx(2.0 * tw[cell]), (tw, lam)
            assert sum(float(v) for k, c, v in adds if k == "charge:0" and int(c) >= 0) == pytest.approx(2.0)
            pe = [float(v) for k, c, v in loop.moments(dep(lam), (jnp.asarray(tw, F32),)) if k == "pe2"]
            assert pe == [pytest.approx(2.0 * tw[cell])]  # the pe moment's add: q times its cells' acceptance
    with pytest.raises(ValueError, match="weighted time histogram"):
        loop.emits(dep(450.0))


def test_the_readout_per_block_of_sensors_is_the_readout_in_one_launch():
    """The readout built from deposit records block by block (``fill.plan``, ``fill.place``) is the one-launch
    readout bit for bit: each sensor's cells are the same adds into the same exact sums, whatever its block."""
    g = Grid.bands((300.0, 400.0, 500.0), t0=0.0, dt=1.0, nt=16, weighted=True)
    lays = (fill.HistogramLayout(5, g, index=(0, 1, 2, 3, 4, -1, -1, -1)),
            fill.HistogramLayout(3, Grid.bands((300.0, 500.0)), index=(-1,) * 5 + (0, 1, 2)))  # fmt: skip
    n, r = 300, np.random.default_rng(3)
    rec = dict(sensor=r.integers(0, 8, n), q=r.uniform(0, 1, n), w0=r.uniform(0.5, 2, n), time=r.uniform(-3, 20, n),
               lam=r.uniform(250, 550, n), photon=np.sort(r.integers(0, 100, n)))  # records per photon  # fmt: skip
    rec = {k: jnp.asarray(rec[k], d) for k, d in runner.record_fields((), False)}
    weights = (jnp.asarray([0.3, 0.7], F32), jnp.ones(1, F32))
    one, _ = runner.rebuild(fill.readout(lays), rec, n, weights)
    blocks = fill.plan(lays, runner.readout_bytes(int(2.5 * fill.sensor_cells(lays[0])), False, "auto"), False, "auto")
    assert len(blocks) > 2
    got = None
    for b in blocks:
        got = fill.place(lays, b, runner.rebuild(fill.readout(lays, b), rec, n, weights)[0], got)
    assert set(got) == set(one)
    for k in one:
        assert np.array_equal(np.asarray(got[k]), np.asarray(one[k])), k
    assert float(np.asarray(one["arrival"]).sum()) == pytest.approx(float(np.sum(rec["q"] * rec["w0"])), rel=1e-6)


def test_a_moment_squares_each_photons_sum_over_its_records():
    """A moment cell (``Readout.moments``, ``lumen.run.rebuild``) is sum_i x_i^2, x_i photon i's adds to the cell summed
    over its records before the square (not each record's add squared), whatever the photon's records in between."""
    lay = fill.HistogramLayout(4, Grid.bands((300.0, 400.0, 500.0)))
    n, r = 400, np.random.default_rng(5)
    photon = np.sort(r.integers(0, 60, n))  # records in (photon, step) order, up to ~15 per photon
    rec = dict(sensor=r.integers(0, 4, n), q=r.uniform(0, 1, n), w0=r.uniform(0.5, 2, n), time=r.uniform(0, 9, n),
               lam=r.uniform(250, 550, n), photon=photon)  # fmt: skip
    rec = {k: jnp.asarray(rec[k], d) for k, d in runner.record_fields((), False)}
    out, _ = runner.rebuild(fill.readout((lay,)), rec, n)
    x = {k: np.asarray(v, np.float64) for k, v in rec.items()}
    add = x["q"] * x["w0"]
    band = np.where((x["lam"] >= 300) & (x["lam"] < 400), 0, np.where((x["lam"] >= 400) & (x["lam"] < 500), 1, -1))
    arr2, ch2 = np.zeros(4), np.zeros(8)
    for p in np.unique(photon):
        m = photon == p
        arr2 += np.bincount(x["sensor"][m].astype(int), add[m], 4) ** 2
        b = m & (band >= 0)
        ch2 += np.bincount((x["sensor"][b] * 2 + band[b]).astype(int), add[b], 8) ** 2
    np.testing.assert_allclose(np.asarray(out["arrival2"]), arr2, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out["charge2:0"]), ch2, rtol=1e-5)
    per_record = np.bincount(x["sensor"].astype(int), add * add, 4)
    assert np.all(np.asarray(out["arrival2"]) > per_record * (1 + 1e-3))  # the photons' sums, not the records'


def test_the_health_s_own_launch_adds_the_photons_fates_exactly():
    """The photon loop returns each photon's fates and ``run._health_sums`` adds them in a launch of its own: exact
    sums, the same whatever the order the rows come in, where a float32 running sum loses the small ones."""
    from lumen import run

    n = 1 << 12
    rows = np.full((n, len(run.HEALTH_SUMS)), 2.0**-4, np.float32)
    rows[0] = 2.0**20
    a = run._health_sums(jnp.asarray(rows), "xla")
    b = run._health_sums(jnp.asarray(rows[::-1].copy()), "xla")
    for k in run.HEALTH_SUMS:
        assert np.asarray(a[k]).tolist() == np.asarray(b[k]).tolist(), k
    exact = np.float32(rows[:, 0].astype(np.float64).sum())  # the exact sum, in the output's float32
    assert np.asarray(a[run.HEALTH_SUMS[0]]).sum() == exact
    running = np.float32(0.0)
    for v in rows[:, 0]:
        running = np.float32(running + v)
    assert running != exact  # a float32 running sum does not reach it
