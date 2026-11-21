"""The reference: every draw, and the roulette, at ``simulate(reference=)``; the live values only in the weights.

A sphere of scattering, absorbing water with a glossy wall (``toys.sphere``). At ``reference=params`` the run is the
default run, bitwise. With the reference held, the light at a seed is a smooth function of the live values: a central
finite difference converges to the forward derivative as h^2 (the roulette reads the reference weight, so no survival
test switches as a value moves). A derivative that moves the photons is refused in a medium with a bulk channel.

The physics tier (one GPU).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lumen
from lumen.format.calibrations import Table
from lumen.sources import GradientNotAvailable

N = 1 << 16
SRC = lumen.sources.Point(position=(0.05, 0.0, 0.0), lam=400.0, n_photons=1.0)


@pytest.fixture(scope="module")
def sphere(toy):
    return toy("sphere", rayleigh=1.0, absorption=3.0, finish="ground", sigma_alpha=0.2).det


def _run(det, p, seed, **kw):
    return lumen.simulate(det, SRC, p, samples=N, seed=seed, **kw).arrivals


def _scaled(p, key, th):
    v = p[key]
    return p.set(key, Table(jnp.asarray(v.energy), jnp.asarray(v.value) * (1.0 + th)))


def test_the_reference_at_the_run_values_is_the_default_run(sphere):
    p = sphere.params()
    a, b = np.asarray(_run(sphere, p, 3)), np.asarray(_run(sphere, p, 3, reference=p))
    assert a.tobytes() == b.tobytes() and a.sum() > 0


@pytest.mark.parametrize("name", ["RAYLEIGH", "ABSLENGTH"])
def test_a_fixed_reference_finite_difference_converges_to_the_derivative(sphere, name):
    p = sphere.params()
    key = next(k for k in p if k.endswith("/" + name))
    for seed in (1, 2):
        ad = float(jnp.sum(jax.jvp(lambda t, s=seed: _run(sphere, _scaled(p, key, t), s),
                                   (jnp.float32(0.0),), (jnp.float32(1.0),))[1]))  # fmt: skip
        err = []
        for h in (1 / 16, 1 / 64):
            up, dn = (float(jnp.sum(_run(sphere, _scaled(p, key, x), seed, reference=p))) for x in (h, -h))
            err.append(abs((up - dn) / (2 * h) - ad) / abs(ad))
        assert err[1] < 2e-3 and err[1] < err[0] / 4, (name, seed, ad, err)  # O(h^2): 16x smaller at h / 4


def test_a_source_position_derivative_is_refused(sphere):
    def f(x):
        src = lumen.sources.Point(position=jnp.stack([x, 0.0, 0.0]), lam=400.0, n_photons=1.0)
        return lumen.simulate(sphere, src, samples=1024, seed=1).arrivals

    with pytest.raises(GradientNotAvailable, match="moves where the light starts"):
        jax.jvp(f, (jnp.float32(0.05),), (jnp.float32(1.0),))
