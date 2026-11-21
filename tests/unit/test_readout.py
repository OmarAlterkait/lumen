"""The readout after tracking against closed forms: charge = N eta, the cells, the time folds, the compound Poisson
charge, the expected waveform, sampled against expected, and the memory refusal."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import readout_toys as toys
from jax.experimental import enable_x64
from scipy import stats

from lumen import readout as rd
from lumen import run as runner
from lumen.readout import grid as tg
from lumen.readout.detection import values
from lumen.rules.choices import TIME_SPREAD_REACH
from lumen.sensors import components as comp
from lumen.sensors import value_sets


def _eta(m, v, ph):
    qe = np.interp(ph["lam"], m.axis("wavelength").knots, v["sensors/pmt/qe"])
    ce = np.interp(ph["hit"], m.axis("theta").knots, v["sensors/pmt/ce"])
    return ph["weight"] * v["sensors/pmt/efficiency"][ph["sensor"]] * qe * ce


def test_charge_is_n_eta():
    """pe = sum over photons of weight x efficiency x QE(lambda) x CE(theta), Geant4's linear interpolation; its error
    (at the value set's own values) is the photons' moment: sqrt(n / (n - 1) (sum eta^2 - (sum eta)^2 / n))."""
    m = toys.model()
    ph = toys.photons(2000, 3, seed=3)
    h = tg.from_photons(m, (0.0, 120.0), 3, ph, group="pmt")
    p = toys.params(m)
    eta0 = _eta(m, p, ph)
    n, s = len(ph["sensor"]), np.bincount(ph["sensor"], eta0, 3)
    want_err = np.sqrt(n / (n - 1) * (np.bincount(ph["sensor"], eta0**2, 3) - s * s / n))
    np.testing.assert_allclose(np.asarray(rd.pe_error(h)), want_err, rtol=1e-4)
    assert h.charge2.shape == h.charge.shape and np.all(np.asarray(h.charge2) >= 0)
    p["sensors/pmt/efficiency"] = np.asarray([0.9, 0.5, 0.7], np.float32)
    want = np.bincount(ph["sensor"], _eta(m, p, ph), 3)
    np.testing.assert_allclose(rd.pe(h, p), want, rtol=2e-6)


def test_light_outside_the_time_grid_is_in_the_overflow_and_every_total():
    """No light is dropped: the same photons before, inside or after the time grid give the same charge, and the time
    histogram's total (nodes plus the early / late cells) is the same."""
    m = toys.model()
    gate = (0.0, 120.0)
    g = m.grid(gate)
    ph = toys.photons(400, 3, seed=5)
    inside = tg.from_photons(m, gate, 3, ph, group="pmt")

    def total(h):
        return float(np.asarray(h.time).sum() + np.asarray(h.late)[..., 0].sum() + np.asarray(h.early)[..., 0].sum())

    for t in (g.t0 - 50.0, g.t0 + g.dt * g.nt + 50.0):
        out = tg.from_photons(m, gate, 3, ph | dict(time=np.full(len(ph["time"]), t)), group="pmt")
        np.testing.assert_array_equal(np.asarray(out.charge), np.asarray(inside.charge))
        assert total(out) == pytest.approx(total(inside), rel=1e-5)


def test_a_time_past_the_grid_is_late_and_coarse_float32_times_are_refused():
    """A time far past the grid is late, never wrapped to an early index by the int32 cast (review7 F2 #6); times
    whose float32 steps are coarser than the time nodes (absolute times of 1e9 ns: 64 ns steps) are refused."""
    g = rd.Grid(t0=0.0, dt=0.5, nt=40)
    t = jnp.asarray([25.0, 3e9, -3e9, 1e12])
    dep = runner.Record(jnp.zeros(4, jnp.int32), jnp.ones(4), t, jnp.zeros(4))
    idx, w, _, ow = jax.vmap(lambda d: tg.time_cells(d, g))(dep)
    np.testing.assert_allclose(np.asarray(ow)[1:], [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], atol=1e-6)  # late, early
    assert np.all(np.asarray(idx)[1:] == -1)
    m = toys.model()
    ph = toys.photons(10, 3, seed=1)
    with pytest.raises(tg.GateError, match="relative to the event"):
        tg.from_photons(m, (1e9, 1e9 + 120.0), 3, ph | dict(time=ph["time"] + 1e9), group="pmt")


def test_time_cells_keep_moments():
    """The cubic time split keeps a deposit's mass, mean, second and third moments."""
    g = rd.Grid(t0=0.0, dt=0.5, nt=40)
    t = jnp.linspace(1.0, 18.0, 37)
    dep = runner.Record(jnp.zeros(37, jnp.int32), jnp.ones(37), t, jnp.zeros(37))
    idx, w, _, ow = jax.vmap(lambda d: tg.time_cells(d, g))(dep)
    tn = g.t0 + g.dt * (idx % g.nt)
    for k in range(4):
        np.testing.assert_allclose(jnp.sum(w * tn**k, -1), t**k, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(ow, 0.0, atol=0.0)


def test_density_closed_form_tts():
    """A delta arrival: the density is the bin mass of the Gaussian TTS about the arrival plus the offset."""
    m = toys.model()
    gate = (0.0, 100.0)
    h = tg.from_photons(m, gate, 1, dict(sensor=np.zeros(1, int), lam=np.full(1, 400.0), pos=toys.AXIS,
                                        time=np.full(1, 21.3)), group="pmt")  # fmt: skip
    p = toys.params(m, 1)
    p["sensors/pmt/dark_rate"] = np.zeros(1, np.float32)
    d = rd.pe_density(h, p)
    e = gate[0] + d.dt * np.arange(d.rate.shape[1] + 1)
    s, t0 = 1.2, 21.3 + 5.0
    cdf = stats.norm.cdf(e, loc=t0, scale=s)
    want = float(rd.pe(h, p)[0]) * np.diff(cdf)
    np.testing.assert_allclose(d.rate[0], want, atol=2e-4 * want.max())


def test_emg_tts_closed_form():
    """The emg time spread's CDF is scipy's exponentially modified Gaussian."""
    x = jnp.linspace(-5.0, 20.0, 101)
    p = dict(sigma=jnp.float32(0.7), tau=jnp.float32(2.5), shift=jnp.float32(-0.2))
    got = comp.EMGTTS().cdf(p, x)
    np.testing.assert_allclose(got, stats.exponnorm.cdf(x, 2.5 / 0.7, loc=-0.2, scale=0.7), atol=2e-6)


def test_a_windows_charge_is_one_photoelectron_plus_a_compound_poisson_sum():
    """Q of a window of 1 + Poisson(m) photoelectrons (it opens at one): mass 1, mean (1 + m) m1, variance
    (m2 - m1^2) + m m2; a Gaussian SPE in closed form."""
    m = toys.model(threshold=None)
    p = toys.params(m)
    v = values(rd.Histograms(charge=None, model=m, group="pmt"), p)
    mu = jnp.asarray([0.0, 0.3, 2.0, 7.5])
    qpdf = rd.hits.window_charge(m, v, mu)
    q = rd.hits.charge_grid(m)[2]
    m1, m2 = (float(x) for x in m.charge.moments(v["charge"]))
    np.testing.assert_allclose(qpdf.sum(-1), 1.0, rtol=1e-5)
    mean = (qpdf * q).sum(-1)
    np.testing.assert_allclose(mean, (1 + mu) * m1, rtol=2e-4)
    var = (m2 - m1 * m1) + mu * m2
    np.testing.assert_allclose((qpdf * q**2).sum(-1), var + ((1 + mu) * m1) ** 2, rtol=2e-3)
    g = toys.model(threshold=None, charge=dict(form="gauss_exp", sigma_pe=0.35, fraction=0.0, tau_pe=0.5))
    vg = values(rd.Histograms(charge=None, model=g, group="pmt"), toys.params(g))
    qg = rd.hits.window_charge(g, vg, mu)
    n = np.arange(1, 60)
    pois = stats.poisson.pmf(n[None, :] - 1, np.asarray(mu)[:, None])
    dq = g.charge_step
    want = (
        pois[:, :, None]
        * (
            stats.norm.cdf(q + dq / 2, n[:, None], 0.35 * np.sqrt(n[:, None]))
            - stats.norm.cdf(q - dq / 2, n[:, None], 0.35 * np.sqrt(n[:, None]))
        )[None]
    ).sum(1)
    np.testing.assert_allclose(qg, want, atol=5e-6)
    rec = np.asarray(rd.hits.recorded(m, v))  # no threshold, no noise: a window is recorded when its charge is > 0
    np.testing.assert_allclose(float(rd.hits.record_probability(jnp.asarray(rec), 0.0)), rec[0], rtol=1e-6)


def test_expected_waveform_delta_arrival():
    """One sensor, light at a single time: mean = baseline + gain mu m1 <h>(t - t_arrival - T) for a 0.8 ns TTS."""
    m = toys.model(tts=dict(form="gaussian", sigma_ns=0.8))
    ph = dict(sensor=np.zeros(1, int), lam=np.full(1, 400.0), pos=toys.AXIS, time=np.full(1, 30.0),
              weight=np.full(1, 50.0))  # fmt: skip
    h = tg.from_photons(m, (0.0, 100.0), 1, ph, group="pmt")
    p = toys.params(m, 1)
    p["sensors/pmt/dark_rate"] = np.zeros(1, np.float32)
    w = rd.expected_waveform(h, p)
    mu = float(rd.pe(h, p)[0])
    t = w.t0 + w.dt * np.arange(w.mean.shape[1])
    # the pulse of a photoelectron at T ~ N(35, 0.8): average the bigauss over that spread numerically
    u = np.linspace(-6, 6, 4001) * 0.8
    gw = stats.norm.pdf(u, 0, 0.8) * (u[1] - u[0])
    d = t[:, None] - 35.0 - u[None, :] - 6.0
    shape = np.where(d < 0, np.exp(-0.5 * (d / 1.5) ** 2), np.exp(-0.5 * (d / 4.0) ** 2))
    want = 100.0 - 20.0 * mu * (shape * gw).sum(-1)
    np.testing.assert_allclose(w.mean[0], want, atol=2e-3 * 20 * mu)
    assert np.all(w.variance >= 1.5**2 - 1e-3)  # FFT round-off of the shot-noise term


def test_sampled_against_expected():
    """The mean and variance of sampled waveforms and the hit probability agree with the expectation."""
    m = toys.model(tts=dict(form="gaussian", sigma_ns=1.0))
    h = toys.hist(m, n=300, n_sensors=3, t0=30.0, spread=3.0)
    p = toys.params(m)
    w = rd.expected_waveform(h, p)
    eh = rd.expected_hits(h, p)
    evs = [rd.sample(h, p, seed=s) for s in range(500)]
    wf = np.stack([e.waveforms for e in evs])
    z = (wf.mean(0) - np.asarray(w.mean)) / np.sqrt(np.asarray(w.variance) / len(evs))
    assert np.abs(z).max() < 5.0 and abs(z.mean()) < 0.2
    np.testing.assert_allclose(wf.var(0).mean(), np.asarray(w.variance).mean(), rtol=0.08)
    hit = np.stack([np.bincount(e.hits["sensor"], minlength=3) > 0 for e in evs]).mean(0)
    np.testing.assert_allclose(hit, eh.p_hit, atol=4.0 * np.sqrt(0.25 / len(evs)))


def test_hits_nll_runs_and_prefers_truth():
    """The hit likelihood is finite and lower at the true transit offset than 3 ns away."""
    m = toys.model()
    h = toys.hist(m, n=600, n_sensors=3, t0=30.0, spread=1.0)
    p = toys.params(m)
    ev = rd.sample(h, p, seed=4)
    data = rd.Hits.from_list(ev.hits["sensor"], ev.hits["t"], ev.hits["q"], 3)

    def nll(off):
        q = dict(p)
        q["sensors/pmt/t_offset"] = jnp.full(3, off, jnp.float32)
        return rd.hits_nll(rd.expected_hits(h, q), data)

    assert np.isfinite(nll(5.0)) and nll(5.0) < nll(8.0)


def test_the_readout_has_no_photon_to_photoelectron_draw():
    """``photoelectrons`` (arrivals to photoelectrons) had no caller and broke on sensors past the highest id given."""
    assert not hasattr(rd, "photoelectrons") and "photoelectrons" not in rd.__all__


def test_chunks():
    assert tg.photon_chunks(10, 4) == [(0, 4), (4, 4), (8, 2)]


def test_sensor_cells_are_the_cells_the_readout_fills():
    """The memory plan counts exactly the readout's histogram outputs: the charge cells and their moment, one time
    histogram and its four overflow cells per sensor"""
    from lumen.readout import fill

    m = toys.model()
    g = m.grid((0.0, 120.0))
    lay = fill.HistogramLayout(5, g)
    out = fill.readout((lay,)).outputs()
    cells = sum(out[k][0] for k in ("charge:0", "charge2:0", "time:0", "over_q:0", "over_qt:0"))
    assert out["time:0"][0] == 5 * g.nt
    assert 5 * fill.sensor_cells(lay) == cells


def test_a_value_outside_its_declared_range_is_refused_at_load():
    """The time grid is built for the ranges: a value set's own value outside its declared range is refused, naming
    the key; a width's range is derived ([r, 1 / r] x its value, rules.choices.TIME_SPREAD_REACH), never declared"""
    with pytest.raises(value_sets.SensorValuesError, match="t_offset"):
        toys.model(t_offset_ns=-1.0)
    with pytest.raises(value_sets.SensorValuesError, match="derived, not declared"):
        toys.model(ranges={**toys.toy_set()["ranges"], "tts/sigma": [0.5, 4.0]})
    m = toys.model(tts=dict(form="gaussian", sigma_ns=0.5))
    assert m.ranges["tts/sigma"] == (0.5 * TIME_SPREAD_REACH, 0.5 / TIME_SPREAD_REACH)


def _fill_time(h, grid, ph):
    """Return h with its time histogram refilled from photons ``ph`` on ``grid`` (as tracking fills it)."""
    dep = runner.Record(jnp.asarray(ph["sensor"], jnp.int32), jnp.ones(len(ph["lam"])), jnp.asarray(ph["time"]),
                        jnp.asarray(ph["lam"]))  # fmt: skip
    ti, tw, oi, ow = (np.asarray(a) for a in jax.vmap(lambda d: tg.time_cells(d, grid))(dep))
    s = h.charge.shape[0]
    w = np.asarray(ph["weight"])[:, None]
    time, over = np.zeros(s * grid.nt), np.zeros((2 * s, 2))
    np.add.at(time, ti[ti >= 0], (w * tw)[ti >= 0])
    for c, val in ((0, w * ow), (1, w * ow * np.asarray(ph["time"])[:, None])):
        np.add.at(over[:, c], oi[oi >= 0], val[oi >= 0])
    over = over.reshape(s, 2, 2)
    return dataclasses.replace(h, time=jnp.asarray(time.reshape(s, grid.nt)), late=jnp.asarray(over[:, 0]),
                               early=jnp.asarray(over[:, 1]), grid=grid)  # fmt: skip


def test_time_grid_accuracy_against_a_fine_reference():
    """At the declared minimum time spread the grid's density is within the cubic split's bound of a grid 8 times
    finer, for light at single instants (the worst case): chi^2 per photoelectron <= (9/16)^2 (1/4)^8 / 24 = 2.0e-7
    (``lumen.readout.grid.node_spacing``). The hit likelihood's error is not this bound alone: its no-record integral
    Lambda_r is a midpoint rule in the node masses (docs/explanation/known-limits.md), so an event's NLL is not checked
    against it."""
    m = toys.model(tts=dict(form="gaussian", sigma_ns=0.8))
    gate = (0.0, 100.0)
    ph = toys.photons(400, 20, t0=0.0, seed=5)
    ph["time"] = np.random.default_rng(6).uniform(20.0, 70.0, 400)  # deltas at every phase between nodes
    ph["weight"] = ph["weight"] * 5.0
    h = tg.from_photons(m, gate, 20, ph, group="pmt")
    g = h.grid
    fine = dataclasses.replace(g, dt=g.dt / 8, nt=8 * (g.nt - 1) + 1)
    p = toys.params(m, 20)
    p["sensors/pmt/dark_rate"] = np.zeros(20, np.float32)
    with enable_x64():
        h64, hf = (jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), x)
                   for x in (_fill_time(h, g, ph), _fill_time(h, fine, ph)))  # fmt: skip
        p64 = {k: jnp.asarray(v, jnp.float64) for k, v in p.items()}
        rc = np.asarray(rd.pe_density(h64, p64).rate)
        rf = np.asarray(rd.pe_density(hf, p64).rate).reshape(20, rc.shape[1], 8).sum(-1)
        on = rf > 1e-9 * rf.max()
        d = np.sum((rc - rf)[on] ** 2 / rf[on]) / rf.sum()
        assert 1e-12 < d < 2.0e-7, d


def test_the_recorded_share_of_an_interval_without_node_light_is_the_local_one():
    """With W = 0 every window holds one photoelectron, so r is one constant and Lambda_r = r Lambda over any interval,
    also one holding no node light (light reaching it only through the time spread). Differenced cumulative sums gave
    their rounding's ratio there (measured: 49 of 9,417 gaps of 40 events off by > 1e-4, up to r = 1.85)."""
    m = toys.model(tts=dict(form="gaussian", sigma_ns=0.8))
    ph = toys.photons(400, 20, t0=0.0, seed=5)
    ph["time"] = np.random.default_rng(6).uniform(20.0, 70.0, 400)
    h = tg.from_photons(m, (0.0, 100.0), 20, ph, group="pmt")
    p = toys.params(m, 20)
    p["sensors/pmt/frontend/window"] = np.zeros_like(p["sensors/pmt/frontend/window"])
    e = rd.expected_hits(h, p)
    share = rd.hits._record_share(e)
    r = np.asarray(share[-1])
    ab = np.sort(np.random.default_rng(0).uniform(0.0, 100.0, (2, 20 * 500)), 0).astype(np.float32)
    rows = jnp.repeat(jnp.arange(20), 500)
    np.testing.assert_allclose(np.asarray(rd.hits._mean_r(share, rows, *jnp.asarray(ab))), r[0, 0], rtol=1e-5)


def test_the_grid_does_not_depend_on_values():
    """The time grid comes from the value set's ranges and the gate, never the values a run is given: calibrating a
    value keeps it. The width's range is the set's own width times [r, 1 / r] (TIME_SPREAD_REACH)"""
    g = toys.model().grid((0.0, 100.0))
    assert toys.model(t_offset_ns=9.0).grid((0.0, 100.0)) == g
    assert toys.model(tts=dict(form="gaussian", sigma_ns=3.0)).grid((0.0, 100.0)) != g  # another set's width
    nodes = int(np.ceil(2.0 / (1.2 * TIME_SPREAD_REACH / 4) - 1e-9))
    assert g.dt == 2.0 / nodes  # the narrowest spread / 4, a whole number of nodes per 2 ns sample


def test_density_sums_to_the_pe_at_the_declared_minimum_spread():
    """At the declared minimum time spread (the grid's finest) the density over the gate sums to the photoelectrons"""
    m = toys.model(tts=dict(form="gaussian", sigma_ns=0.8))
    h = tg.from_photons(m, (0.0, 100.0), 1, dict(sensor=np.zeros(1, int), lam=np.full(1, 400.0), pos=toys.AXIS,
                                                 time=np.full(1, 41.3)), group="pmt")  # fmt: skip
    p = toys.params(m, 1)
    p["sensors/pmt/dark_rate"] = np.zeros(1, np.float32)
    np.testing.assert_allclose(float(rd.pe_density(h, p).rate.sum()), float(rd.pe(h, p)[0]), rtol=1e-4)


def test_a_delay_table_bin_is_integrated_exactly_against_the_cubic_weights():
    """No sub-points: each uniform bin's taps sum to 1 and carry its mean and variance exactly (the cubic split
    reproduces moments up to the third), at any node spacing against the bins."""
    from lumen.sensors.components import DelayTable

    tab = DelayTable((0.0, 1.0, 2.5, 4.0, 7.3))
    for dt in (0.3, 0.25, 1.0, 1.7):
        off0, ntap, rows, cols, wts = tab.taps(dt)
        m = np.zeros((ntap, len(tab.edges) - 1))
        np.add.at(m, (rows, cols), wts)
        t = (off0 + np.arange(m.shape[0])) * dt
        e = np.asarray(tab.edges)
        np.testing.assert_allclose(m.sum(0), 1.0, atol=1e-12)
        np.testing.assert_allclose(t @ m, 0.5 * (e[1:] + e[:-1]), atol=1e-12)
        np.testing.assert_allclose((t * t) @ m, (0.5 * (e[1:] + e[:-1])) ** 2 + np.diff(e) ** 2 / 12, atol=1e-11)


@pytest.mark.parametrize("gate", [(100.0, 100.0), (500.0, 0.0), (0.0, float("inf"))])
def test_a_timed_grid_refuses_a_gate_that_reads_no_light(gate):
    with pytest.raises(ValueError, match="readout gate"):
        toys.model().grid(gate)


def test_the_time_weight_is_one_path_for_values_and_tracers():
    """The time histogram's weight per charge cell is the same, bitwise, at concrete values and traced (a run at a
    value and the primal of its derivative weigh the time histogram alike): QE x CE, two tables."""
    m = toys.model()
    v = m.split(toys.params(m), "pmt")
    assert len(m.tables) >= 2
    concrete = np.asarray(m.acceptance(v))
    traced = np.asarray(jax.jit(lambda x: m.acceptance(x))(v))
    assert concrete.dtype == np.float32 and np.array_equal(concrete, traced)


def test_sensor_ids_outside_the_group_are_refused():
    """A sensor id outside [0, S) is refused where it enters: a photon list, a hit list, given photoelectrons."""
    m = toys.model()
    ph = toys.photons(50, 3, seed=1)
    with pytest.raises(ValueError, match=r"sensor ids are integers in \[0, 2\)"):
        tg.from_photons(m, (0.0, 120.0), 2, ph, group="pmt")
    with pytest.raises(ValueError, match="must be finite"):
        tg.from_photons(
            m, (0.0, 120.0), 3, dict(ph, time=np.where(np.arange(50) == 3, np.nan, ph["time"])), group="pmt"
        )
    with pytest.raises(ValueError, match="sensor ids"):
        rd.Hits.from_list([0, -1], [1.0, 2.0], [1.0, 1.0], 3)
    v = values(tg.from_photons(m, (0.0, 120.0), 3, ph, group="pmt"), toys.params(m))
    with pytest.raises(ValueError, match="sensor ids"):
        rd.electronics(m, v, [0, 3], [1.0, 2.0], np.random.default_rng(0), 3)


def test_hits_of_another_group_size_are_refused():
    """hits_nll refuses a hit list made for another number of sensors (JAX would clamp its rows silently)."""
    m = toys.model()
    h = tg.from_photons(m, (0.0, 120.0), 3, toys.photons(200, 3, seed=2), group="pmt")
    eh = rd.expected_hits(h, toys.params(m))
    with pytest.raises(ValueError, match="hits_nll: the hits are of 4 sensors"):
        rd.hits_nll(eh, rd.Hits.from_list([0], [10.0], [1.0], 4))


def test_pe_of_a_light_is_every_sensor_s_and_the_public_names():
    """``pe`` of Histograms is its group's; of a Light, every sensor of the detector in the arrivals' order: each
    group's at its sensors (``Histograms.sensors``), 0 at a sensor of no modelled group. ``lumen`` names neither
    ``surfaces`` nor ``Speed``, and ``dir(lumen)`` lists the lazy names."""
    import lumen
    from lumen.api import Light

    m = toys.model()
    p = toys.params(m)
    h = tg.from_photons(m, (0.0, 120.0), 3, toys.photons(500, 3, seed=4), group="pmt")
    a, b = (dataclasses.replace(h, sensors=s) for s in ((4, 0, 2), (1, 3, 5)))  # two groups of one value set
    light = Light(arrivals=np.zeros(7), arrivals_error=None, mean_arrival_time=None, health=None,
                  groups={"near": a, "far": b})  # fmt: skip
    q = np.asarray(rd.pe(h, p))
    want = np.zeros(7)
    want[[4, 0, 2]] += q
    want[[1, 3, 5]] += q
    np.testing.assert_allclose(np.asarray(rd.pe(light, p)), want, rtol=1e-6)
    assert "surfaces" not in lumen.__all__ and "Speed" not in lumen.__all__ and not hasattr(lumen, "Speed")
    assert {"Controls", "compare", "diagnostics", "sensors", "tools"} <= set(dir(lumen))
