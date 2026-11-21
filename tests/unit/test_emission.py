"""The emission record (lumen.media.emission): its time profile's components against Geant4's samplers, its draw
against G4OpWLS's at the reference, and its spectrum's derivative against finite differences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen.media import emission, spectra
from lumen.tools import stats
from lumen.units import HC

F32 = jnp.float32
N = 200_000
ENERGY = np.array([2.4, 2.6, 2.8, 3.0, 3.3, 3.5])
VALUE = np.array([0.0, 0.4, 1.0, 0.7, 0.3, 0.0])


def geant4_sample_time(tau_rise, tau_decay, rg, n):
    """G4Scintillation::sample_time (Geant4 11.2) literally, vectorised: an exponential envelope of the bi-exponential,
    accepted with probability bi_exp / (d single_exp)."""
    out = np.empty(0)
    d = (tau_rise + tau_decay) / tau_decay
    while out.size < n:
        u1, u2 = rg.random(n), rg.random(n)
        t = -tau_decay * np.log(1.0 - u1)
        gg = d * np.exp(-t / tau_decay) / tau_decay
        bi = np.exp(-t / tau_decay) * (1 - np.exp(-t / tau_rise)) / tau_decay / tau_decay * (tau_rise + tau_decay)
        out = np.concatenate([out, t[u2 <= bi / gg]])
    return out[:n]


def uniforms(n, k, seed=0):
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, k), F32, 1e-7, 1.0)


@pytest.mark.parametrize("rise", [0.5, 6.0])
def test_a_rise_component_is_geant4s_finite_rise_time(rise):
    """("decay", "rise") draws Geant4's finite-rise-time profile: its quantiles are G4Scintillation::sample_time's
    within sampling error (two samples of 2e5), and rise 0 adds nothing (the plain decay)."""
    tau = 10.0
    prof = emission.TimeProfile(("decay", "rise"))
    got = np.asarray(jax.vmap(lambda u: prof.sample({"tau": F32(tau), "rise": F32(rise)}, u))(uniforms(N, 2)))
    want = geant4_sample_time(rise, tau, np.random.default_rng(1), N)
    q = np.linspace(0.05, 0.95, 19)
    a, b = np.quantile(got, q), np.quantile(want, q)
    # each quantile's error sqrt(q (1 - q) / N) / density (the density taken as the decay's, (1 - q) / (tau + rise)),
    # in each of two samples
    tol = stats.z_limit(len(q)) * np.sqrt(2 * q * (1 - q) / N) * (tau + rise) / (1 - q)
    assert np.all(np.abs(a - b) < tol), (a, b)
    u = uniforms(8, 2)
    zero = jax.vmap(lambda u: prof.sample({"tau": F32(tau), "rise": F32(0.0)}, u))(u)
    plain = jax.vmap(lambda u: emission.TimeProfile(("decay",)).sample({"tau": F32(tau)}, u))(u)
    np.testing.assert_array_equal(np.asarray(zero), np.asarray(plain))


def test_delta_decay_and_table_components():
    """ "delta" is tau, "decay" Exp(tau) and "table" a tabulated profile drawn as Geant4 draws a table (its mean is
    the table's); an unknown component is refused."""
    u = uniforms(N, 1)
    d = jax.vmap(lambda u: emission.TimeProfile(("delta",)).sample({"tau": F32(2.5)}, u))(u)
    assert np.all(np.asarray(d) == np.float32(2.5))
    e = np.asarray(jax.vmap(lambda u: emission.TimeProfile(("decay",)).sample({"tau": F32(4.0)}, u))(u))
    assert abs(e.mean() - 4.0) < 5 * 4.0 / np.sqrt(N)
    tab = {"time": jnp.asarray([0.0, 1.0, 3.0], F32), "value": jnp.asarray([0.0, 2.0, 0.0], F32)}
    t = np.asarray(jax.vmap(lambda u: emission.TimeProfile(("table",)).sample({"table": tab}, u))(u))
    # the sampler's density is each interval's mean, uniform within it: masses 1 and 2 at means 0.5 and 2
    assert abs(t.mean() - (1 * 0.5 + 2 * 2.0) / 3) < 5 * t.std() / np.sqrt(N) and t.min() >= 0 and t.max() <= 3
    with pytest.raises(ValueError, match="components"):
        emission.TimeProfile(("gauss",))


def test_the_record_draws_g4opwls_at_the_reference():
    """At the reference the record's draw is G4OpWLS's (the spectrum truncated at the absorbed energy, by its inverse
    CDF) and its weight the yield times Geant4's chance of such an energy (``spectra.kept``), to float32 rounding."""
    rec = emission.EmissionRecord("w", emission.wls_profile("exponential"))
    v = {"yield": F32(1.3), "tau": F32(2.0), "spectrum": {"energy": jnp.asarray(ENERGY, F32),
                                                          "value": jnp.asarray(VALUE, F32)}}  # fmt: skip
    u = uniforms(4096, 2)
    for lam_in in (380.0, 450.0, 480.0):
        lam, dt, w, _ = jax.vmap(lambda r: rec.sample(v, v, r, F32(lam_in)))(u)
        arrays = spectra.g4_arrays(ENERGY, VALUE)
        f = spectra.cumulative(arrays, F32(lam_in))
        want, kept = jax.vmap(lambda r: spectra.sample(arrays, r[0] * f))(u), spectra.kept(f)
        np.testing.assert_allclose(np.asarray(lam), np.asarray(want), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(w), 1.3 * np.asarray(kept), rtol=2e-6)
        assert np.all(np.asarray(lam) >= lam_in * (1 - 1e-6))
    lam, _, w, _ = rec.sample(v, v, u[0], F32(1000.0))  # absorbed below the spectrum: no emission
    assert float(w) == 0.0 and np.isfinite(float(lam))


def test_the_spectrum_has_a_derivative():
    """The expected re-emitted light below a wavelength cut, E[w 1(lam > cut)] over draws at the reference, has the
    derivative in a spectrum knot that finite differences of its closed form give (the live / reference density
    weight: the sources' rule)."""
    rec = emission.EmissionRecord("w", emission.wls_profile("delta"))
    lam_in, cut, k = F32(390.0), 440.0, 2
    u = uniforms(N, 2, seed=3)

    def spec(x):
        return {"energy": jnp.asarray(ENERGY, F32), "value": jnp.asarray(VALUE, F32).at[k].set(x)}

    def mc(x):
        ref = {"yield": F32(1.0), "tau": F32(0.0), "spectrum": jax.tree_util.tree_map(jax.lax.stop_gradient, spec(x))}
        live = dict(ref, spectrum=spec(x))
        lam, _, w, _ = jax.vmap(lambda r: rec.sample(live, ref, r, lam_in))(u)
        return jnp.mean(w * (lam > cut))

    def exact(x):  # kept(F) * mass(E < HC / cut) / F, F the mass below the absorbed energy (float64)
        v = np.asarray(VALUE, np.float64).copy()
        v[k] = x
        a = spectra.g4_arrays(ENERGY, v)
        e0, h, d, c = (np.asarray(t, np.float64) for t in a)

        def cdf(e):
            i = np.clip(np.searchsorted(e0, e, side="right") - 1, 0, len(e0) - 1)
            return float(np.clip(c[i] + d[i] * np.clip(e - e0[i], 0, h[i]), 0, 1)) if e >= e0[0] else 0.0

        f = cdf(HC / 390.0)
        return (1 - (1 - f) ** 100) * cdf(HC / cut) / f

    x0 = float(VALUE[k])
    g = float(jax.grad(mc)(F32(x0)))
    h = 1e-3
    fd = (exact(x0 + h) - exact(x0 - h)) / (2 * h)
    assert abs(float(mc(F32(x0))) - exact(x0)) < 0.01
    assert g == pytest.approx(fd, rel=0.05, abs=0.01), (g, fd)
