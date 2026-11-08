"""Polarisation: the Jones algebra and Fresnel amplitudes against closed forms, the polarised Rayleigh density and
sampler, and the Cherenkov state."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen.media import phase
from lumen.surfaces import fresnel, polar

F32 = jnp.float32


def rand_unit(key, n):
    v = jax.random.normal(key, (n, 3), F32)
    return v / jnp.linalg.norm(v, axis=1, keepdims=True)


def rand_state(key, d):
    """A random complex Jones vector perpendicular to d (n, 3), |E| = 1."""
    k1, k2 = jax.random.split(key)
    a, b = jax.random.normal(k1, d.shape, F32), jax.random.normal(k2, d.shape, F32)
    a = a - jnp.sum(a * d, -1, keepdims=True) * d
    b = b - jnp.sum(b * d, -1, keepdims=True) * d
    return jax.vmap(polar.normalise)(a, b)


def np_reflectances(c, n1, n2):
    s2 = (n1 / n2) ** 2 * (1 - c * c)
    if s2 >= 1:
        return 1.0, 1.0
    ct = math.sqrt(1 - s2)
    rs = (n1 * c - n2 * ct) / (n1 * c + n2 * ct)
    rp = (n2 * c - n1 * ct) / (n2 * c + n1 * ct)
    return rs * rs, rp * rp


@pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.33, 1.49), (1.6, 1.33)])
def test_amplitudes_against_float64_and_energy(n1, n2):
    for c in np.linspace(0.01, 1.0, 37):
        rs, rp, ts, tp = (complex(x) for x in fresnel.amplitudes(F32(c), F32(n1), F32(n2)))
        Rs, Rp = np_reflectances(c, n1, n2)
        assert abs(abs(rs) ** 2 - Rs) < 2e-5 and abs(abs(rp) ** 2 - Rp) < 2e-5, (c, rs, rp, Rs, Rp)
        assert abs(0.5 * (Rs + Rp) - float(fresnel.unpolarised_reflectance(F32(c), F32(n1), F32(n2)))) < 2e-5
        s2 = (n1 / n2) ** 2 * (1 - c * c)
        if s2 < 1:  # T = (n2 cos_t / n1 cos_i) |t|^2 = 1 - R per component
            f = n2 * math.sqrt(1 - s2) / (n1 * c)
            assert abs(f * abs(ts) ** 2 + Rs - 1) < 3e-5 and abs(f * abs(tp) ** 2 + Rp - 1) < 3e-5
        else:
            assert abs(abs(rs) - 1) < 1e-5 and abs(abs(rp) - 1) < 1e-5
    rs, rp, _, _ = (complex(x) for x in fresnel.amplitudes(F32(1.0), F32(n1), F32(n2)))
    assert abs(rp + rs) < 1e-6  # normal incidence: r_p p_r = r_s p_i, every state alike
    cb = math.cos(math.atan(n2 / n1))  # Brewster: no p reflection
    assert abs(complex(fresnel.amplitudes(F32(cb), F32(n1), F32(n2))[1])) < 1e-3


def test_total_internal_reflection_phase_is_the_rhomb_formula():
    """Glass 1.51 to air: tan(delta / 2) = cos t sqrt(sin^2 t - m^2) / sin^2 t; 54.6 deg gives the Fresnel rhomb's
    45 deg; one sign for every incidence, so successive reflections add."""
    n1, n2 = 1.51, 1.0
    m = n2 / n1
    ds = []
    for th in np.radians([42.0, 48.0, 54.6, 60.0, 75.0]):
        rs, rp, _, _ = (complex(x) for x in fresnel.amplitudes(F32(math.cos(th)), F32(n1), F32(n2)))
        d = (np.angle(rs) - np.angle(rp)) % (2 * math.pi)
        d = min(d, 2 * math.pi - d)
        want = 2 * math.atan(math.cos(th) * math.sqrt(math.sin(th) ** 2 - m * m) / math.sin(th) ** 2)
        assert abs(d - want) < 2e-4, (math.degrees(th), d, want)
        ds.append(np.angle(rs * np.conj(rp)))
    assert all(np.sign(ds) == np.sign(ds[0]))
    assert abs(math.degrees(abs(ds[2])) - 45.0) < 0.1


def test_perfect_mirror_on_the_sp_basis():
    key = jax.random.PRNGKey(3)
    n = 512
    wi = rand_unit(key, n)
    nrm = rand_unit(jax.random.fold_in(key, 1), n)
    nrm = jnp.where(jnp.sum(nrm * wi, -1, keepdims=True) > 0, -nrm, nrm)  # facing: wi . n < 0
    er, ei = rand_state(jax.random.fold_in(key, 2), wi)
    wo = wi - 2 * jnp.sum(wi * nrm, -1, keepdims=True) * nrm
    got = jax.vmap(lambda a, b, w, n_, k: polar.jones(a, b, w, n_, k, polar.MIRROR))(er, ei, wi, nrm, wo)
    want = jax.vmap(polar.mirror)(er, ei, nrm)
    overlap = jnp.abs(jnp.sum((got[0] - 1j * got[1]) * (want[0] + 1j * want[1]), -1))  # equal up to a phase
    assert float(jnp.min(overlap)) > 1 - 1e-5


def test_rayleigh_sampler_moments_density_and_unpolarised_marginal():
    key = jax.random.PRNGKey(11)
    n = 400_000
    wi = jnp.tile(jnp.asarray([0.0, 0.0, 1.0], F32), (n, 1))
    states = [
        (jnp.asarray([1.0, 0, 0]), jnp.zeros(3)),
        (jnp.asarray([1.0, 0, 0]) / math.sqrt(2), jnp.asarray([0, 1.0, 0]) / math.sqrt(2)),
        polar.normalise(jnp.asarray([0.8, 0.3, 0]), jnp.asarray([-0.2, 0.5, 0])),
    ]
    for i, (er, ei) in enumerate(states):
        er, ei = jnp.asarray(er, F32), jnp.asarray(ei, F32)
        u = jax.random.uniform(jax.random.fold_in(key, i), (n, 3), F32, 1e-7, 1 - 1e-7)
        k = jax.vmap(lambda uu: phase.rayleigh_sample(er, ei, uu))(u)
        M = np.outer(er, er) + np.outer(ei, ei)
        S = np.asarray(k).T @ np.asarray(k) / n
        assert np.max(np.abs(S - (0.4 * np.eye(3) - 0.2 * M))) < 4e-3, (S, M)
        v = rand_unit(jax.random.fold_in(key, 100 + i), n)
        integral = 4 * math.pi * float(jnp.mean(jnp.exp(jax.vmap(lambda w: phase.rayleigh_logpdf(er, ei, w))(v))))
        assert abs(integral - 1) < 5e-3
    er = jnp.asarray([1.0, 0, 0], F32)  # linear along x: nothing scattered along x
    k = jax.vmap(lambda uu: phase.rayleigh_sample(er, jnp.zeros(3, F32), uu))(
        jax.random.uniform(key, (n, 3), F32, 1e-7, 1 - 1e-7)
    )
    assert float(jnp.mean(jnp.abs(k[:, 0]) > 0.999)) < 2e-4
    ks = jax.random.split(key, 2)  # unpolarised: mu has density 3/8 (1 + mu^2), E[mu^2] = 2/5, E[mu^4] = 9/35
    e = jax.vmap(lambda d, uu: polar.random_linear(d, uu))(wi, jax.random.uniform(ks[0], (n,)))
    k = jax.vmap(lambda a, uu: phase.rayleigh_sample(a, jnp.zeros(3, F32), uu))(
        e, jax.random.uniform(ks[1], (n, 3), F32, 1e-7, 1 - 1e-7)
    )
    mu = np.asarray(k[:, 2], np.float64)
    assert abs((mu**2).mean() - 0.4) < 3e-3 and abs((mu**4).mean() - 9 / 35) < 3e-3


def test_rayleigh_after_a_90_degree_scatter_is_linear():
    wi = jnp.asarray([0.0, 0.0, 1.0], F32)
    k1 = jnp.asarray([1.0, 0.0, 0.0], F32)  # scattered along x: the plane x-z
    for a in np.linspace(0.0, 1.0, 9):
        er, ei = polar.project(polar.random_linear(wi, F32(a)), jnp.zeros(3, F32), k1)
        assert abs(abs(float(er[1])) - 1.0) < 1e-6 or float(jnp.sum(er * er)) == 0.0


def test_cherenkov_state_in_the_track_plane():
    key = jax.random.PRNGKey(5)
    axis = rand_unit(key, 256)
    d = rand_unit(jax.random.fold_in(key, 1), 256)
    e = jax.vmap(polar.cherenkov)(d, axis)
    assert float(jnp.max(jnp.abs(jnp.sum(e * d, -1)))) < 1e-5
    assert float(jnp.max(jnp.abs(jnp.sum(e * jnp.cross(axis, d), -1)))) < 1e-5
    assert float(jnp.max(jnp.abs(jnp.linalg.norm(e, axis=-1) - 1))) < 1e-5
