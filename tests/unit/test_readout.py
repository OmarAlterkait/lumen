"""The readout after tracking against closed forms: charge = N eta, the cells, the time folds, the compound Poisson
charge, the expected waveform, sampled against expected, and the memory refusal."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import readout_toys as toys
from scipy import stats

from lumen import readout as rd
from lumen.readout.detection import values
from lumen.sensors import components as comp


def _eta(m, v, ph):
    qe = np.interp(ph["lam"], m.qe_knots, v["sensors/pmt/qe"])
    ce = np.interp(ph["hit"], m.ce_knots, v["sensors/pmt/ce"])
    return ph["weight"] * v["sensors/pmt/efficiency"][ph["sensor"]] * qe * ce


def test_charge_is_n_eta():
    """pe = sum over photons of weight x efficiency x QE(lambda) x CE(theta), Geant4's linear interpolation."""
    m = toys.model()
    ph = toys.photons(2000, 3, seed=3)
    h = rd.from_photons(m, (0.0, 120.0), 3, ph, replicas=4)
    p = toys.params(m)
    p["sensors/pmt/efficiency"] = np.asarray([0.9, 0.5, 0.7], np.float32)
    want = np.bincount(ph["sensor"], _eta(m, p, ph), 3)
    np.testing.assert_allclose(rd.pe(h, p), want, rtol=2e-6)
    assert rd.pe_error(h, p).shape == (3,)


@pytest.mark.parametrize("split", ["cubic", "linear"])
def test_time_cells_keep_moments(split):
    """The time split keeps a deposit's mass and mean (and second and third moments for the cubic split)."""
    g = rd.Grid(t0=0.0, dt=0.5, nt=40, split=split)
    t = jnp.linspace(1.0, 18.0, 37)
    dep = rd.grid._Photons(jnp.zeros(37, jnp.int32), jnp.zeros(37), jnp.zeros(37), t)
    idx, w, _, ow = jax.vmap(lambda d: rd.time_cells(d, g))(dep)
    tn = g.t0 + g.dt * (idx % g.nt)
    for k in range(4 if split == "cubic" else 2):
        np.testing.assert_allclose(jnp.sum(w * tn**k, -1), t**k, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(ow, 0.0, atol=0.0)


def test_density_closed_form_tts_and_emission():
    """A delta arrival: the density is the bin mass of (exponential emission convolved with the Gaussian TTS)."""
    m = toys.model()
    gate = (0.0, 100.0)
    h = rd.from_photons(m, gate, 1, dict(sensor=np.zeros(1, int), lam=np.full(1, 400.0), hit=np.zeros(1),
                                        time=np.full(1, 21.3)))  # fmt: skip
    prof = ((1.0,), (7.0,), (0.0,))
    h = rd.Histograms(h.charge, h.time, h.late, h.early, prof, h.grid, h.model, h.group, h.gate)
    p = toys.params(m, 1)
    p["sensors/pmt/dark_rate"] = np.zeros(1, np.float32)
    d = rd.pe_density(h, p)
    e = gate[0] + d.dt * np.arange(d.rate.shape[1] + 1)
    s, tau, t0 = 1.2, 7.0, 21.3 + 5.0
    cdf = stats.exponnorm.cdf(e, tau / s, loc=t0, scale=s)  # Gaussian convolved with an exponential
    want = float(rd.pe(h, p)[0]) * np.diff(cdf)
    np.testing.assert_allclose(d.rate[0], want, atol=2e-4 * want.max())


def test_emg_tts_closed_form():
    """The emg time spread's CDF is scipy's exponentially modified Gaussian."""
    x = jnp.linspace(-5.0, 20.0, 101)
    p = dict(sigma=jnp.float32(0.7), tau=jnp.float32(2.5), shift=jnp.float32(-0.2))
    got = comp.EMGTTS().cdf(p, x)
    np.testing.assert_allclose(got, stats.exponnorm.cdf(x, 2.5 / 0.7, loc=-0.2, scale=0.7), atol=2e-6)


def test_compound_poisson_moments_and_gaussian_spe():
    """Q of Poisson(mu) photoelectrons: mass 1 - e^-mu, mean mu m1, variance mu m2; a Gaussian SPE in closed form."""
    m = toys.model(threshold=None)
    p = toys.params(m)
    v = values(rd.Histograms(None, model=m), p)
    mu = jnp.asarray([0.3, 2.0, 7.5])
    qpdf, _ = rd.hits.compound_charge(m, v, mu)
    q = rd.hits.charge_grid(m)[2]
    m1, m2 = (float(x) for x in m.charge.moments(v["charge"]))
    np.testing.assert_allclose(qpdf.sum(-1), 1.0 - np.exp(-mu), rtol=1e-5)
    mean = (qpdf * q).sum(-1)
    np.testing.assert_allclose(mean, mu * m1, rtol=2e-4)
    np.testing.assert_allclose((qpdf * q**2).sum(-1), mu * m2 + (mu * m1) ** 2, rtol=2e-3)
    g = toys.model(threshold=None, charge=dict(form="gauss_exp", sigma_pe=0.35, fraction=0.0, tau_pe=0.5))
    vg = values(rd.Histograms(None, model=g), toys.params(g))
    qg, _ = rd.hits.compound_charge(g, vg, mu)
    n = np.arange(1, 60)
    pois = stats.poisson.pmf(n[None, :], np.asarray(mu)[:, None])
    dq = g.charge_step
    want = (
        pois[:, :, None]
        * (
            stats.norm.cdf(q + dq / 2, n[:, None], 0.35 * np.sqrt(n[:, None]))
            - stats.norm.cdf(q - dq / 2, n[:, None], 0.35 * np.sqrt(n[:, None]))
        )[None]
    ).sum(1)
    np.testing.assert_allclose(qg, want, atol=5e-6)


def test_expected_waveform_delta_arrival():
    """One sensor, light at a single time: mean = baseline + gain mu m1 <h>(t - t_arrival - T) for a 0-width TTS."""
    m = toys.model(tts=dict(form="gaussian", sigma_ns=0.8))
    h = rd.from_photons(m, (0.0, 100.0), 1, dict(sensor=np.zeros(1, int), lam=np.full(1, 400.0), hit=np.zeros(1),
                                                time=np.full(1, 30.0), weight=np.full(1, 50.0)))  # fmt: skip
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
    h = toys.hist(m, n=300, n_sensors=3, t0=30.0, spread=3.0, replicas=1)
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


def test_memory_refusal_and_chunks():
    g = rd.Grid(lam_knots=(300.0, 600.0), t0=0.0, dt=0.5, nt=3000)
    assert rd.check_memory(g, 100, 16, 0, 1e9) > 0
    with pytest.raises(MemoryError, match="Shorten the readout gate"):
        rd.check_memory(g, 11096, 16, 7, 1e9)
    assert rd.photon_chunks(10, 4) == [(0, 4), (4, 4), (8, 2)]
