"""The unified facet chain's path density is differentiated exactly in sigma_alpha (the ANNIE steel case).

Chains drawn by the lobe's own sampler at sigma 0.3; d log p(path) / d th at sigma (1 + th): float32 AD (as the
step computes it) against float64 central differences on the same held paths. A float32 finite difference of the same
function is biased by rounding (about 2.5e-4 per chain at h = 1/1024): the comparison must be made in float64.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64

from lumen.surfaces import facets

SIGMA, N = 0.3, 512


def chains():
    rng = np.random.default_rng(0)
    wi = rng.normal(size=(N, 3))
    wi[:, 2] = -np.abs(wi[:, 2])
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    n = np.tile([0.0, 0.0, 1.0], (N, 1))
    rand = (rng.integers(0, 2**23, size=(N, 3)) + 0.5) * 2.0**-23
    path = jax.vmap(lambda w_, n_, r_: facets.UNIFIED.path({"sigma_alpha": jnp.float32(SIGMA)}, w_, n_, r_))
    W, k = path(jnp.asarray(wi, jnp.float32), jnp.asarray(n, jnp.float32), jnp.asarray(rand, jnp.float32))
    return wi, n, np.asarray(W), np.asarray(k)


def logp(dtype):
    def f(th, wi, n, W, k):
        s = jnp.asarray(SIGMA, dtype) * (1 + th)
        return facets.UNIFIED.path_logpdf({"sigma_alpha": s}, wi.astype(dtype), n.astype(dtype), W.astype(dtype), k)

    return f


def test_the_chain_density_derivative_in_sigma_is_exact():
    wi, n, W, k = chains()
    args = tuple(jnp.asarray(a) for a in (wi, n, W, k))
    ad32 = np.asarray(jax.vmap(jax.grad(logp(jnp.float32)), in_axes=(None, 0, 0, 0, 0))(jnp.float32(0.0), *args))
    with enable_x64():
        f = jax.vmap(logp(jnp.float64), in_axes=(None, 0, 0, 0, 0))
        a64 = tuple(jnp.asarray(a) for a in (wi, n, W, k))
        h = 1e-4
        fd64 = (np.asarray(f(jnp.float64(h), *a64)) - np.asarray(f(jnp.float64(-h), *a64))) / (2 * h)
    d = ad32 - fd64
    assert np.sqrt(np.mean(d * d)) < 5e-5 and abs(d.mean()) < 5e-6, (np.sqrt(np.mean(d * d)), d.mean())
