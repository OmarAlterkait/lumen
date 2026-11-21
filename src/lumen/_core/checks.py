"""Checks of plug-in code, shared by layers that must not import each other.

No host callbacks in model code, a bulk lobe's sampler against its density, and the chi^2 pass rule
(``lumen.tools.stats``, ``lumen.tools.extend``).

A leaf: it imports only jax, numpy, ``lumen._core`` and the constants (``lumen.units``, ``lumen.rules.choices``).
"""

from __future__ import annotations

import numpy as np

from lumen._core import rng
from lumen._core.errors import ModelError
from lumen.rules.choices import (
    LOBE_AZIMUTHS,
    LOBE_BINS,
    LOBE_MIN_EXPECTED,
    LOBE_NODES,
    LOBE_NORM_TOL,
    LOBE_SAMPLES,
)
from lumen.units import TWO_PI

CALLBACKS = ("pure_callback", "io_callback", "callback")  # host callbacks (jax.debug.print is allowed)


def chi2_passes(chi2_ndf, ndf, null=1.0, var=2.0) -> bool:
    """Return whether a chi2 / ndf is within three standard deviations above its null: ``null`` the mean of one term
    and ``var`` its variance under the null (1 and 2: Pearson terms of known variance)."""  # noqa: D205, D209
    return bool(ndf == 0 or chi2_ndf <= null + 3.0 * np.sqrt(var / ndf))


def callbacks(jaxpr):
    """Yield the host-callback primitives of a jaxpr, nested ones included."""
    for eq in jaxpr.eqns:
        if eq.primitive.name in CALLBACKS:
            yield eq.primitive.name
        for p in eq.params.values():
            for sub in p if isinstance(p, (tuple, list)) else (p,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from callbacks(inner)


def no_callbacks(what, fn, *args):
    """Refuse (ModelError) a function whose trace at ``args`` calls the host."""
    import jax

    bad = sorted(callbacks(jax.make_jaxpr(fn)(*args).jaxpr))
    if bad:
        raise ModelError(f"{what}: host callbacks {bad} (model code is plain JAX, per photon)")


def check_lobe(lobe, p, what="lobe"):
    """Check a bulk lobe: at most 3 uniforms, plain JAX, its sampler against its density (chi^2 in the cosine about
    the incoming direction, ``chi2_passes``) and the density's normalisation."""  # noqa: D205, D209
    import jax
    import jax.numpy as jnp

    if lobe.n_rand > rng.LOBE_UNIFORMS:
        raise ModelError(f"{what}: {lobe.n_rand} uniforms; a lobe draws at most {rng.LOBE_UNIFORMS}")
    if lobe.logpdf is None or lobe.surface:
        raise ModelError(f"{what}: a bulk lobe has a density (logpdf) and no surface normal")
    pj = {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}
    wi = jnp.asarray([0.0, 0.0, 1.0])
    rand = rng.uniforms(jax.random.PRNGKey(0), LOBE_SAMPLES * lobe.n_rand).reshape(LOBE_SAMPLES, lobe.n_rand)
    no_callbacks(what, lambda r: lobe.sample(pj, wi, wi, r), rand[0])
    wo = jax.vmap(lambda r: lobe.sample(pj, wi, wi, r))(rand)
    mu = np.clip(np.asarray(wo[:, 2], np.float64), -1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, LOBE_BINS + 1)
    count = np.histogram(mu, edges)[0]
    # each bin's probability: Gauss-Legendre in mu, the azimuth averaged on a ring
    x, wq = np.polynomial.legendre.leggauss(LOBE_NODES)
    lo, hi = edges[:-1, None], edges[1:, None]
    m_q = (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel()
    phi = np.linspace(0.0, TWO_PI, LOBE_AZIMUTHS, endpoint=False)
    sq = np.sqrt(np.clip(1 - m_q**2, 0, 1))
    dirs = np.stack([np.outer(sq, np.cos(phi)), np.outer(sq, np.sin(phi)), np.outer(m_q, np.ones_like(phi))], -1)
    lp = jax.vmap(jax.vmap(lambda d: lobe.logpdf(pj, wi, wi, d)))(jnp.asarray(dirs, jnp.float32))
    dens = TWO_PI * np.exp(np.asarray(lp, np.float64)).mean(1)  # per unit mu
    prob = (dens.reshape(LOBE_BINS, -1) * wq).sum(1) * 0.5 * (edges[1:] - edges[:-1])
    if abs(prob.sum() - 1.0) > LOBE_NORM_TOL:
        raise ModelError(f"{what}: its density integrates to {prob.sum():.4f}, not 1")
    expect = prob / prob.sum() * LOBE_SAMPLES
    ok = expect > LOBE_MIN_EXPECTED
    chi2 = float(np.sum((count[ok] - expect[ok]) ** 2 / expect[ok]) / max(ok.sum() - 1, 1))
    if not chi2_passes(chi2, max(int(ok.sum()) - 1, 1)):
        raise ModelError(f"{what}: its sampler does not draw its density (chi2/ndf {chi2:.2f})")


__all__ = ["ModelError", "callbacks", "check_lobe", "chi2_passes", "no_callbacks"]
