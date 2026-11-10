"""``lumen.simulate`` with ``lumen.sources`` (tier b): a Point in the absorbing sensor sphere of ``toys.sphere``.

- the arrivals per emitted photon against the closed form (1/2) int exp(-r(mu) / L) dmu of a source at z0 on the
  axis (every photon detected with weight exp(-r / L));
- the source-position derivative through ``wrt=("source/position",)`` (the moving-source method) against its quadrature,
  per replica of the charge histogram;
- the run is the same in any chunks; a beam's position is refused;
- the Point's rows moved by the source's own position parameter are the moving-source test's rows moved by dz.
"""

import dataclasses
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toys import GPU, A

import lumen
from lumen import sources
from lumen.build import build

F32 = jnp.float32
Z_MAX = 4.5
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
    """The sphere toy with its sensor read by lumen:pmt at identity (QE 1 over the band, efficiency 1)."""
    det0 = toy("sphere", r=0.0, absorption=L_ABS).det
    vs = tmp_path_factory.mktemp("values") / "identity.json"
    vs.write_text(json.dumps(dict(
        schema="lumen-pmt-values/1", model="lumen:pmt", model_version=1, name="identity", version=1, readout="pe",
        provenance=dict(source="tests/physics/test_sources.py", status="measured"),
        qe=dict(band_edges_nm=[300.0, 600.0], values=[1.0]), ce=None, efficiency=1.0, t_offset_ns=0.0, tts=None,
        charge=None, gain=1.0, threshold=None, dark_rate_hz=0.0,
        frontend=dict(sampling_ns=1.0, unit="pe", window_ns=0.0, charge_step_pe=0.25, charge_max_pe=4.0))))  # fmt: skip
    g0 = next(g for g in det0.sensor_set.groups if g.rule == "border_into")
    g = dataclasses.replace(g0, model="lumen:pmt", defaults=str(vs))
    sset = dataclasses.replace(det0.sensor_set, groups=tuple(g if x is g0 else x for x in det0.sensor_set.groups))
    parts = build.build(det0.file, sset)
    params = build.parameters(det0.file, sset, parts)
    return lumen.Detector(det0.name, det0.path, det0.file, sset, parts, det0.sensors, det0.calibration, params)


def point(z):
    return sources.Point(jnp.stack([F32(0.0), F32(0.0), jnp.asarray(z, F32)]), n_photons=1.0, lam=400.0)


def test_a_point_through_simulate_closed_form_and_chunks(pmt_sphere):
    n = 1 << 16
    light = lumen.simulate(pmt_sphere, point(Z0), samples=n, seed=3, replicas=16)
    q, err = float(np.asarray(light.arrivals, np.float64).sum()), float(np.asarray(light.arrivals_error).sum())
    assert abs(q - on_axis(Z0, L_ABS)[0]) < Z_MAX * err, (q, err)
    chunked = lumen.simulate(pmt_sphere, point(Z0), samples=n, seed=3, replicas=16, chunk=1 << 13)
    if GPU:  # exact sums: the same run in any chunks
        assert np.asarray(chunked.arrivals).tobytes() == np.asarray(light.arrivals).tobytes()
    else:
        np.testing.assert_allclose(np.asarray(chunked.arrivals), np.asarray(light.arrivals), rtol=1e-5)


def test_the_source_position_derivative_through_simulate(pmt_sphere):
    k = 16

    def charge(z):  # per replica, per emitted photon (each replica holds 1/k of the photons)
        light = lumen.simulate(pmt_sphere, point(z), samples=k * 8192, seed=5, replicas=k, wrt=("source/position",))
        return light.histograms.charge.sum((1, 2, 3)) * k

    q, dq = jax.jvp(charge, (F32(Z0),), (F32(1.0),))
    dq = np.asarray(dq, np.float64)
    want = on_axis(Z0, L_ABS)[1]
    assert abs((dq.mean() - want) / (dq.std(ddof=1) / math.sqrt(k))) < Z_MAX, (dq.mean(), want)
    beam = sources.Point((0.0, 0.0, Z0), lam=400.0, direction=(0.0, 0.0, 1.0))
    with pytest.raises(sources.GradientNotAvailable):
        lumen.simulate(pmt_sphere, beam, samples=1024, seed=1, replicas=2, wrt=("source/position",))


def test_the_points_rows_move_with_its_position_parameter():
    n = 512
    src = sources.Point((0.0, 0.0, Z0), n_photons=float(n), lam=400.0)
    key = jax.random.PRNGKey(0)

    def moved(dz):  # the moving-source test's rows: the photons' x + dz along z
        r = src.emit(None, key, n)
        return r._replace(x=r.x + jnp.asarray([0.0, 0.0, 1.0], F32) * dz)

    def through(dz):  # the same move as the source's own parameter
        return src.emit(
            {"source/position": jnp.asarray([0.0, 0.0, Z0], F32) + jnp.asarray([0.0, 0.0, 1.0], F32) * dz}, key, n
        )

    a, b = (jax.jvp(f, (F32(0.0),), (F32(1.0),)) for f in (moved, through))
    for u, v in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert np.asarray(u).tobytes() == np.asarray(v).tobytes()


def test_simulate_reports_and_refuses_dropped_exact_emits(pmt_sphere, monkeypatch):
    from lumen import api
    from lumen import run as runner

    light = lumen.simulate(pmt_sphere, point(Z0), samples=1 << 12, seed=3)
    assert int(light.health["overflow"]) == 0
    real = runner.run
    monkeypatch.setattr(runner, "run", lambda *a, **k: real(*a, **k)._replace(overflow=jnp.int32(7)))
    with pytest.raises(api.SumOverflow, match="7 emit"):
        lumen.simulate(pmt_sphere, point(Z0), samples=1 << 12, seed=3)
