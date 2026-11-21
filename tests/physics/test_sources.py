"""``lumen.simulate`` with ``lumen.sources`` (physics tier): a Point in the absorbing sensor sphere of ``toys.sphere``.

- the arrivals per emitted photon against the closed form (1/2) int exp(-r(mu) / L) dmu of a source at z0 on the
  axis (every photon detected with weight exp(-r / L));
- the run is the same in any chunks; a source-position derivative is refused (not in this version);
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toys import GPU, A

import lumen
from lumen import sources
from lumen.build import build
from lumen.tools.stats import z_limit

F32 = jnp.float32
L_ABS, Z0 = 2.0, 0.3


def on_axis(z0, length, n=400):
    """(Q/N, d(Q/N)/dz) of a source at z0 on the axis of a sphere of radius A, absorption length L (quadrature)."""
    mu, w = np.polynomial.legendre.leggauss(n)
    root = np.sqrt(A**2 - z0**2 * (1 - mu**2))
    r = -z0 * mu + root
    dr = -mu - z0 * (1 - mu**2) / root
    return float(0.5 * np.sum(w * np.exp(-r / length))), float(0.5 * np.sum(w * np.exp(-r / length) * (-dr / length)))


@pytest.fixture(scope="module")
def pmt_sphere(toy, tmp_path_factory):
    """The sphere toy with its sensor read by lumen:sensor at identity (QE 1 at every wavelength, efficiency 1)."""
    det0 = toy("sphere", r=0.0, absorption=L_ABS).det
    vs = tmp_path_factory.mktemp("values") / "identity.json"
    vs.write_text(json.dumps(dict(
        schema="lumen-sensor-values/1", model="lumen:sensor", name="identity", readout="pe",
        provenance=dict(source="tests/physics/test_sources.py", status="measured"),
        qe=dict(band_edges_nm=[], values=[1.0]), ce=None, efficiency=1.0, t_offset_ns=0.0, tts=None,
        charge=None, gain=1.0, threshold=None, dark_rate_hz=0.0,
        frontend=dict(sampling_ns=1.0, unit="pe", window_ns=0.0, charge_step_pe=0.25, charge_max_pe=4.0))))  # fmt: skip
    g0 = next(g for g in det0.sensor_set.groups if g.rule == "border_into")
    g = dataclasses.replace(g0, values=str(vs), settings={})  # the model's own efficiency
    sset = dataclasses.replace(det0.sensor_set, groups=tuple(g if x is g0 else x for x in det0.sensor_set.groups))
    parts = build.build(det0.file, sset)
    params = build.parameters(det0.file, sset, parts)
    return lumen.Detector(det0.name, det0.path, det0.file, sset, parts, det0._columns, det0.calibration, params)


def point(z):
    return sources.Point(jnp.stack([F32(0.0), F32(0.0), jnp.asarray(z, F32)]), n_photons=1.0, lam=400.0)


def test_a_point_through_simulate_closed_form_and_chunks(pmt_sphere):
    """The closed form, and the same run in chunks. Each chunk's exact sum comes back as float32, and the chunks add in
    float32: chunking changes only that rounding (the order of photons within a chunk changes no bit)."""
    n = 1 << 16
    light = lumen.simulate(pmt_sphere, point(Z0), samples=n, seed=3)
    q, err = float(np.asarray(light.arrivals, np.float64).sum()), float(np.asarray(light.arrivals_error).sum())
    assert abs(q - on_axis(Z0, L_ABS)[0]) < z_limit() * err, (q, err)  # the error from the photons' moments
    chunked = lumen.simulate(pmt_sphere, point(Z0), samples=n, seed=3, chunk=1 << 13)
    rtol = 1e-6 if GPU else 1e-5  # GPU: exact sums per chunk, float32 between chunks; CPU: float32 sums throughout
    np.testing.assert_allclose(np.asarray(chunked.arrivals), np.asarray(light.arrivals), rtol=rtol)
    # a chunk that does not divide the samples: a photon's records are in one launch, so its moment is the same and
    # the error bars do not depend on the chunk size (beyond the launches' float32 sum)
    ragged = lumen.simulate(pmt_sphere, point(Z0), samples=n, seed=3, chunk=5000)
    np.testing.assert_allclose(np.asarray(ragged.arrivals), np.asarray(light.arrivals), rtol=rtol)
    np.testing.assert_allclose(np.asarray(ragged.arrivals_error), np.asarray(light.arrivals_error), rtol=1e-4)


def test_a_source_position_derivative_is_refused_through_simulate(pmt_sphere):
    """A tangent on a position (a field or its ``source/position`` key; isotropic or a beam): one refusal."""

    def arrivals(z, direction=None):
        src = sources.Point(jnp.stack([F32(0.0), F32(0.0), z]), lam=400.0, direction=direction)
        return lumen.simulate(pmt_sphere, src, samples=1024, seed=1).arrivals.sum()

    def by_key(z):
        src = sources.Point((0.0, 0.0, Z0), lam=400.0)
        p = src.parameters(pmt_sphere.params()).set("source/position", jnp.stack([F32(0.0), F32(0.0), z]))
        return lumen.simulate(pmt_sphere, src, p, samples=1024, seed=1).arrivals.sum()

    for f in (arrivals, lambda z: arrivals(z, (0.0, 0.0, 1.0)), by_key):
        with pytest.raises(sources.GradientNotAvailable, match="moves where the light starts"):
            jax.jvp(f, (F32(Z0),), (F32(1.0),))


def test_simulate_reports_the_exact_sums_nonfinite_bins(pmt_sphere, monkeypatch):
    """A bin of the exact sums is exact or NaN; tallymap's count of the NaN ones is ``health.nonfinite``."""
    from lumen import run as runner

    light = lumen.simulate(pmt_sphere, point(Z0), samples=1 << 12, seed=3)
    assert int(light.health.nonfinite) == 0
    real = runner.trace
    monkeypatch.setattr(runner, "trace", lambda *a, **k: real(*a, **k)._replace(overflow=jnp.int32(7)))
    assert int(lumen.simulate(pmt_sphere, point(Z0), samples=1 << 12, seed=3).health.nonfinite) == 7


def test_a_detector_without_a_sensor_model_simulates_arrivals_only(toy):
    """No group with a sensor model (counting sensors only, as SBND's and MicroBooNE's shipped recipes): simulate
    gives the arrivals, their error, mean time and health, and no histograms."""
    det = toy("sphere", r=0.0, absorption=L_ABS).det
    light = lumen.simulate(det, point(Z0), samples=1 << 14, seed=3)
    q = float(np.asarray(light.arrivals, np.float64).sum())
    assert light.groups is None and q > 0.0 and np.isfinite(np.asarray(light.mean_arrival_time)).all()
    assert abs(q - on_axis(Z0, L_ABS)[0]) < z_limit() * float(np.asarray(light.arrivals_error).sum())


def test_photons_given_as_arrays_refuse_a_derivative_that_moves_them(pmt_sphere):
    """Photons given as arrays (not drawn from a density) with a traced position: simulate refuses the derivative (it
    was silently ~140x too small, the solid-angle term missing); a bare function is no source."""
    base = point(Z0).emit(None, jax.random.PRNGKey(0), 4096, 0, 4096)

    def run(z):
        x = base.x.at[:, 2].set(z)
        src = sources.FromArrays(x, base.d, base.t, base.lam, base.w)
        return lumen.simulate(pmt_sphere, src, samples=4096, seed=1).arrivals.sum()

    assert float(run(F32(Z0))) > 0.0
    with pytest.raises(sources.GradientNotAvailable, match="moves where the light starts"):  # a tangent on its position
        jax.jvp(run, (F32(Z0),), (F32(1.0),))
    with pytest.raises(TypeError, match="not a lumen.sources.Source"):
        lumen.simulate(pmt_sphere, lambda first, count: base, samples=4096, seed=1)


def test_weights_of_a_source_normalised_to_one_photon_are_unbiased(pmt_sphere):
    """n_photons = 1 (w = 1 / samples) and n_photons = samples (w = 1) on the same seed give the same light scaled by
    samples, a power of two (exact in float32), and the health scales with them: the photon loop's roulette is
    relative to a photon's start weight, so every decision is the same, and the sums are exact."""
    n = 1 << 16
    one = lumen.simulate(pmt_sphere, sources.Point((0.0, 0.0, Z0), n_photons=1.0, lam=400.0), samples=n, seed=3)
    full = lumen.simulate(pmt_sphere, sources.Point((0.0, 0.0, Z0), n_photons=float(n), lam=400.0), samples=n, seed=3)
    np.testing.assert_allclose(np.asarray(one.arrivals) * n, np.asarray(full.arrivals), rtol=1e-6)
    assert int(one.health.nonfinite) == 0 and int(full.health.nonfinite) == 0
    assert float(np.asarray(one.health.lost).sum()) * n == pytest.approx(
        float(np.asarray(full.health.lost).sum()), abs=1e-3
    )
