"""Frames, safe math, random streams and units."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from lumen import units
from lumen._core import frames, rng, safe

F32 = jnp.float32


def _dirs(n, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    v = np.concatenate([v, [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1e-7, -1], [1e-7, 0, 1], [0.6, 0.8, 0.0]]])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_basis_is_orthonormal_and_right_handed():
    n = _dirs(20_000)
    t1, t2 = (np.asarray(x, np.float64) for x in jax.vmap(frames.basis)(jnp.asarray(n, F32)))
    for a, b in ((t1, t1), (t2, t2)):
        np.testing.assert_allclose(np.sum(a * b, 1), 1.0, atol=3e-6)
    for a, b in ((t1, t2), (t1, n), (t2, n)):
        np.testing.assert_allclose(np.sum(a * b, 1), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.cross(t1, t2), n, atol=5e-6)
    v = np.random.default_rng(1).normal(size=(len(n), 3))
    back = jax.vmap(lambda x, m: frames.to_world(frames.to_local(x, m), m))(jnp.asarray(v, F32), jnp.asarray(n, F32))
    np.testing.assert_allclose(np.asarray(back), v, atol=2e-5)


def test_safe_ops_have_finite_values_and_gradients_at_their_singular_points():
    z3 = jnp.zeros(3, F32)
    for f, x in (
        (safe.safe_norm, z3),
        (lambda v: jnp.sum(safe.unit(v)), z3),
        (safe.safe_sqrt, F32(0.0)),
        (safe.safe_sqrt, F32(-1.0)),
        (safe.safe_log, F32(0.0)),
        (lambda b: safe.safe_div(F32(1.0), b), F32(0.0)),
    ):
        v, g = f(x), jax.grad(f)(x)
        assert np.all(np.isfinite(np.asarray(v))) and np.all(np.isfinite(np.asarray(g))), f
    x = jnp.asarray([3.0, 4.0, 0.0], F32)
    assert abs(float(safe.safe_norm(x)) - 5.0) < 1e-6
    np.testing.assert_allclose(np.asarray(safe.unit(x)), [0.6, 0.8, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(jax.grad(safe.safe_norm)(x)), [0.6, 0.8, 0.0], atol=1e-7)


def test_clip01_derivative_is_the_linear_one_on_the_closed_interval():
    for x, want in ((-0.1, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.1, 0.0)):
        assert float(jax.grad(safe.clip01)(F32(x))) == want, x


def test_uniforms_are_open_unit_interval_deterministic_and_block_stable():
    keys = jax.vmap(lambda i: rng.vertex_key(rng.photon_key(jax.random.PRNGKey(0), i), 3))(jnp.arange(200_000))
    u = np.asarray(jax.vmap(lambda k: rng.uniforms(k, 64))(keys), np.float64)
    assert u.min() > 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 5 * math.sqrt(1 / 12 / u.size) and abs(u.var() - 1 / 12) < 1e-4
    u2 = np.asarray(jax.vmap(lambda k: rng.uniforms(k, 64))(keys), np.float64)
    assert np.array_equal(u, u2)
    k0 = rng.photon_key(jax.random.PRNGKey(0), 0)
    assert not np.array_equal(
        np.asarray(rng.uniforms(rng.vertex_key(k0, 0), 8)), np.asarray(rng.uniforms(rng.vertex_key(k0, 1), 8))
    )
    # a block appended to a layout leaves the earlier blocks' draws unchanged
    a = rng.draw(keys[0], rng.Layout(path=1, lobe=2))
    b = rng.draw(keys[0], rng.Layout(path=1, lobe=2, extra=5))
    for k in ("path", "lobe"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))
    assert rng.Layout(path=1, lobe=2) == rng.Layout(path=1, lobe=2)
    assert hash(rng.Layout(a=1)) == hash(rng.Layout(a=1))


def test_units():
    assert abs(units.energy_of(1239.841984) - 1.0) < 1e-12
    assert abs(units.wavelength_of(units.energy_of(400.0)) - 400.0) < 1e-9
    assert units.MM * 1000 == 1.0 and units.US == 1000.0
