"""Likelihoods for expected and observed counts (charge per sensor, or time histograms: more Poisson bins).

mu = lumen.readout.pe(light, p)                   # (S,): or the time nodes, (S, nt)
lumen.tools.poisson_nll(mu, n)                    # sum mu - n + n log(n / mu): the deviance / 2
"""

from __future__ import annotations

import jax.numpy as jnp


def poisson_nll(mu, n):
    """Return the Poisson negative log-likelihood of counts ``n`` given ``mu``, less its value at mu = n (traceable)."""
    mu, n = jnp.asarray(mu), jnp.asarray(n, jnp.asarray(mu).dtype)
    term = jnp.where(n > 0, n * jnp.log(jnp.where(n > 0, n, 1.0) / mu), 0.0)
    return jnp.sum(mu - n + term)
