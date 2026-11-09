"""Closed-form physics through the navigator and the step (tier b), on the toys of ``toys.py``.

- a point source at the centre of a specular sensor sphere: Q/N = (1 - R) e^(-a/L) / (1 - R e^(-2a/L)), each photon
  deterministic, so value and derivatives (R, L) are exact to float32;
- the same sphere Lambertian: Q/N = (1 - R) e^(-a/L) / (1 - R E), E = 2 (1 - (1 + b) e^(-b)) / b^2, b = 2a/L;
- an enclosing sensor with Rayleigh scattering and no absorption: Q/N = 1 for any R and scattering length, so both
  derivatives are 0 (in expectation: the detection is expected over the free path, so each photon's charge varies);
- an opaque cube before the sensor, as a box and as a tessellated mesh: the same charge, the shadow's solid angle;
- border directions: a detecting border A -> B counts light from A into B, none from B into A;
- g4ref's overlap toys (coincident and partly overlapping siblings) against Geant4 10.7.4's detected fraction and
  mean arrival time;
- the moving-source method: d(Q/N)/dz of an off-centre source in an absorbing sphere against the float64
  quadrature, and with a refracting glass shell before the sensor against same-random-number differences.
Statistical comparisons use |z| < ``Z_MAX`` over independent replicas.
"""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toys import DATA, A, simulate

from lumen.geometry import clearance as ray_clear
from lumen.transport import item, state
from lumen.transport import step as steps
from lumen.validate import gradients, photons, stats

F32 = jnp.float32
Z_MAX = 4.5
LAM = 400.0


def rows_at(n, x=(0.0, 0.0, 0.0), seed=0):
    return photons.isotropic(jnp.tile(jnp.asarray(x, F32), (n, 1)), jnp.full(n, LAM, F32), jax.random.PRNGKey(seed))


def families(s, *names):
    """The packed parameters and the gradient check's families whose names contain ``names`` (in that order)."""
    p0 = photons.packed(s.det)
    fams = gradients.families(s.det.parts.optics, p0)
    return p0, [next(f for f in fams if nm in f[0]) for nm in names]


def charge(s, p0, fams, rows, replicas=1, **kw):
    """Return th -> per-replica charge per photon (the families' th as in the gradient check)."""
    n = rows.x.shape[0] / replicas

    def q(th):
        p, pr = gradients.apply(p0, fams, th), gradients.apply(p0, fams, jax.lax.stop_gradient(th))
        return simulate(s, p, rows, reference=pr, replicas=replicas, **kw)["Q"].reshape(replicas, -1).sum(1) / n

    return q


def specular(r, length):
    return (1 - r) * jnp.exp(-A / length) / (1 - r * jnp.exp(-2 * A / length))


def lambert(r, length):
    b = 2 * A / length
    return (1 - r) * jnp.exp(-A / length) / (1 - r * 2 * (1 - (1 + b) * jnp.exp(-b)) / b**2)


def test_specular_sphere_closed_form_and_exact_derivatives(toy):
    s = toy("sphere", r=0.3, absorption=2.0)
    p0, fams = families(s, ".R[", "Liquid:absorption")
    q = charge(s, p0, fams, rows_at(1024))
    th = jnp.zeros(2, F32)
    v = float(q(th)[0])
    d = [float(jax.jvp(lambda t: q(t)[0], (th,), (th.at[j].set(1.0),))[1]) for j in range(2)]
    dr, dl = (float(x) for x in jax.grad(specular, argnums=(0, 1))(0.3, 2.0))
    assert v == pytest.approx(float(specular(0.3, 2.0)), rel=3e-5)
    assert d[0] == pytest.approx(0.3 * dr, rel=1e-4)  # the family adds min(R, 1 - R) th to R
    assert d[1] == pytest.approx(2.0 * dl, rel=1e-4)  # ... and multiplies L by 1 + th


def test_lambertian_sphere_closed_form_and_derivatives(toy):
    s = toy("sphere", r=0.5, finish="ground", sigma_alpha=0.1, absorption=2.0)
    p0, fams = families(s, ".R[", "Liquid:absorption")
    k = 32
    q = charge(s, p0, fams, rows_at(k * 4096), replicas=k)
    th = jnp.zeros(2, F32)
    v = np.asarray(q(th), np.float64)
    dr, dl = (float(x) for x in jax.grad(lambert, argnums=(0, 1))(0.5, 2.0))
    assert abs((v.mean() - float(lambert(0.5, 2.0))) / stats.sem(v)) < Z_MAX
    for j, want in ((0, 0.5 * dr), (1, 2.0 * dl)):
        dv = np.asarray(jax.jvp(q, (th,), (th.at[j].set(1.0),))[1], np.float64)
        assert abs((dv.mean() - want) / stats.sem(dv)) < Z_MAX, (j, dv.mean(), want)


def test_an_enclosing_sensor_detects_every_photon_whatever_the_scattering(toy):
    s = toy("sphere", r=0.5, absorption=1e9, rayleigh=0.5)
    p0, fams = families(s, ".R[", "Liquid:rayleigh")
    k = 16
    q = charge(s, p0, fams, rows_at(k * 4096), replicas=k)
    th = jnp.zeros(2, F32)
    v = np.asarray(q(th), np.float64)  # expected detection over the free path: unbiased, not one per photon
    assert abs((v.mean() - 1.0) / stats.sem(v)) < Z_MAX
    for j in range(2):
        dv = np.asarray(jax.jvp(q, (th,), (th.at[j].set(1.0),))[1], np.float64)
        assert abs(dv.mean() / stats.sem(dv)) < Z_MAX, (j, dv.mean())


def test_an_opaque_mesh_casts_the_shadow_of_the_same_box(toy):
    half, z = 0.1, 0.5
    omega = 4 * math.asin((2 * half) ** 2 / ((2 * half) ** 2 + 4 * (z - half) ** 2))
    want = math.exp(-A / 1e3) * (1 - omega / (4 * math.pi))
    k, out = 16, {}
    for kind in ("box", "mesh"):
        s = toy("sphere", r=0.0, absorption=1e3, occluder=(kind, half, z))
        out[kind] = np.asarray(charge(s, *families(s, "Liquid:absorption"), rows_at(k * 16384), replicas=k)(
            jnp.zeros(1, F32)), np.float64)  # fmt: skip
    assert np.max(np.abs(out["box"] - out["mesh"])) < 1e-4  # the same photons meet the same face
    assert abs((out["box"].mean() - want) / stats.sem(out["box"])) < Z_MAX


def test_a_border_surface_counts_in_its_own_direction_only(toy):
    s = toy("border_pair")
    p0 = photons.packed(s.det)
    into_b = photons.pencil((-0.25, 0.0, 0.0), (1.0, 0.0, 0.0), LAM, 256)
    into_a = photons.pencil((0.25, 0.0, 0.0), (-1.0, 0.0, 0.0), LAM, 256)
    assert float(simulate(s, p0, into_b)["Q"].sum()) / 256 == pytest.approx(math.exp(-0.25 / 1e3), rel=1e-5)
    assert float(simulate(s, p0, into_a)["Q"].sum()) == 0.0


@pytest.mark.parametrize("name", ["coinT_AB", "partT_BA", "vx3_ABC"])
def test_an_overlap_toy_through_the_step_matches_geant4(toy, name):
    g4 = json.loads((DATA / "geant4.json").read_text())[name]
    s = toy("overlap", name)
    k, n = 16, 1 << 18
    rows = photons.pencil((-0.8, 0.0, 0.0), (1.0, 0.0, 0.0), g4["wavelength_nm"], n)
    out = simulate(s, photons.packed(s.det), rows, replicas=k, controls=steps.Controls(t_max=10.0, roulette=0.0))
    q = np.asarray(out["Q"], np.float64).reshape(k, -1).sum(1)
    tw = np.asarray(out["Tw"], np.float64).reshape(k, -1).sum(1)
    f = q / (n / k)
    assert abs((f.mean() - g4["fraction"]) / np.hypot(stats.sem(f), g4["fraction_error"])) < Z_MAX
    t = tw / q
    assert abs((tw.sum() / q.sum() - g4["mean_time_ns"]) / np.hypot(stats.sem(t), g4["mean_time_error_ns"])) < Z_MAX


def absorbed_sphere_dq_dz(z0, length, n=400):
    """d(Q/N)/dz of a source at z0 on the axis of a sphere of radius A, absorption length L (float64 quadrature)."""
    mu, w = np.polynomial.legendre.leggauss(n)
    root = np.sqrt(A**2 - z0**2 * (1 - mu**2))
    r = -z0 * mu + root
    dr = -mu - z0 * (1 - mu**2) / root
    return float(0.5 * np.sum(w * np.exp(-r / length) * (-dr / length)))


def moving_charge(s, rows, replicas):
    """Return dz -> per-replica charge per photon with the source moved by dz along z (the moving-source method)."""
    nav = s.model.geometry.nav
    cls = jnp.asarray(np.asarray(nav.host["cls"])[np.asarray(jax.jit(jax.vmap(nav.locate))(rows.x))], jnp.int32)
    spec = item.photons(s.tally)._replace(init=lambda model, ops, ph: ph)
    p0 = photons.packed(s.det)
    n = rows.x.shape[0] / replicas

    def q(dz):
        r = rows._replace(x=rows.x + jnp.asarray([0.0, 0.0, 1.0], F32) * dz)
        ph = jax.vmap(lambda e, c: state.start(e, -1, c, stacked=s.model.stacked, moving=True))(r, cls)
        return simulate(s, p0, ph, replicas=replicas, spec=spec)["Q"].reshape(replicas, -1).sum(1) / n

    return q


@pytest.mark.parametrize("glass", [False, True], ids=["curved sensor", "behind a refracting shell"])
def test_the_moving_source_derivative(toy, glass):
    z0, k = 0.3, 16
    base = toy("sphere", r=0.0, absorption=2.0, glass=glass)
    point = np.array([[0.0, 0.0, z0]], np.float32)
    field = ray_clear.Clearance(point, ray_clear.ray_clearance(base.model.geometry.nav, point))
    s = toy("sphere", r=0.0, absorption=2.0, glass=glass, step=steps.StepConfig(moving_source=True), clearance=field)
    q = moving_charge(s, rows_at(k * 8192, (0.0, 0.0, z0)), k)
    d = np.asarray(jax.jvp(q, (F32(0.0),), (F32(1.0),))[1], np.float64)
    if not glass:
        want = absorbed_sphere_dq_dz(z0, 2.0)
        assert abs((d.mean() - want) / stats.sem(d)) < Z_MAX, (d.mean(), want)
        return
    h = 1e-3  # same random numbers on every side: the derivative of the estimator's expectation
    fd = stats.richardson(lambda x: np.asarray(q(F32(x)), np.float64), h)
    assert abs((d - fd).mean() / stats.sem(d - fd)) < Z_MAX, (d.mean(), fd.mean())
