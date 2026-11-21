"""GLISUR with r a parameter (``facets.covered("glisur")``): facets drawn with the shell as its cover, so the
r-derivative of a path's weight is bounded and its expectation is the derivative of Geant4's own draw.

Drawn from the GLISUR ball alone the score has tail index 3/2 (infinite variance; gradfam_p3.md §3).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen.rules.choices import DEFENSIVE_SHARE
from lumen.surfaces import facets
from lumen.tools import stats

R0, N = 0.5, 1 << 16
UP = jnp.array([0.0, 0.0, 1.0], jnp.float32)


def _uniforms(key, n):
    u = jax.random.uniform(jax.random.PRNGKey(key), (n, 3), jnp.float32, 1e-7, 1 - 1e-7)
    return ((u * 2.0**23).astype(jnp.int32).astype(jnp.float32) + 0.5) * 2.0**-23  # seed_key's exact grid


def _paths(theta):
    wi = jnp.array([math.sin(theta), 0.0, -math.cos(theta)], jnp.float32)
    lobe = facets.FACET_LOBES["glisur+cover"]

    def one(u):
        def w(r):
            def visit(c, w0, w1):  # the path's live density at r and the density drawn from at R0, facet by facet
                lp, lq = c
                lp = lp + lobe.facet_logpdf({"sigma_alpha": r}, w0, UP, w1)
                return lp, lq + lobe.facet_proposal_logpdf({"sigma_alpha": jnp.float32(R0)}, w0, UP, w1)

            z = jnp.float32(0.0)
            (lp, lq), _, wo, k = lobe.path({"sigma_alpha": jnp.float32(R0)}, wi, UP, u, visit, (z, z))
            wo = jax.lax.stop_gradient(wo)
            on = lq > 0.5 * facets.LOG_TINY
            return jnp.where(on, jnp.exp(lp - jnp.where(on, lq, 0.0)), 0.0) * (wo[2] >= 0.0), wo, k

        (v, d, k), (t, _, _) = jax.jvp(w, (jnp.float32(R0),), (jnp.float32(1.0),))
        return v, t, d, k

    return jax.jit(jax.vmap(one))(_uniforms(3, N))


@pytest.mark.parametrize("theta", [0.0, 1.3])  # h = cos(theta) above r, and below it (the V . M cut bites)
def test_the_weight_is_unbiased_and_its_derivative_bounded(theta):
    v, t, _, k = (np.asarray(x, np.float64) for x in _paths(theta))
    # E_q[p / q] = 1; with no spread (one facet, uncut) p / q is 1 to its float32 rounding (8 ulps: a few operations)
    assert abs(v.mean() - 1.0) < stats.z_limit() * v.std() / math.sqrt(N) + 8 * np.finfo(np.float32).eps
    if theta == 0.0:  # one facet, uncut ball: |d p / dr| / q <= (A / V) / e = (3 / r) / e
        assert np.all(k == 1) and np.abs(t).max() <= 3.0 / R0 / DEFENSIVE_SHARE * (1 + 1e-3)


def test_the_derivative_is_geant4s():
    """d/dr E[d_z] from the weights against Geant4's draw at r0 +- h with common uniforms (grazing incidence)."""
    _, t, d, _ = _paths(1.3)
    ad = np.asarray(t, np.float64) * np.asarray(d[:, 2], np.float64)
    wi = jnp.array([math.sin(1.3), 0.0, -math.cos(1.3)], jnp.float32)

    def analog(r):
        def one(u):
            wo = facets.GLISUR.path({"sigma_alpha": r}, wi, UP, u, lambda c, w0, w1: c, ())[2]
            return wo[2] * (wo[2] >= 0.0)

        return np.asarray(jax.jit(jax.vmap(one))(_uniforms(4, 4 * N)), np.float64)

    fd = [(analog(jnp.float32(R0 + h)) - analog(jnp.float32(R0 - h))) / (2 * h) for h in (0.02, 0.01)]
    rich = (4 * fd[1] - fd[0]) / 3
    err = math.hypot(ad.std() / math.sqrt(N), rich.std() / math.sqrt(len(rich)))
    assert abs(ad.mean() - rich.mean()) < stats.z_limit() * err, (ad.mean(), rich.mean(), err)
