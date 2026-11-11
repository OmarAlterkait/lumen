"""Checks for code that extends lumen: a source function's draws, a medium lobe, any derivative.

    lumen.extend.check_source(led, logpdf=lambda d: 4.0 * jnp.log(jnp.clip(-d[2], 1e-12)))  # draws vs the density
    lumen.extend.check_lobe(lumen.media.henyey_greenstein("g"), {"g": 0.9})                  # sampler vs density
    lumen.validate.check_gradient(f, theta0)                                                  # AD against FD

``check_source`` draws ``n`` photons of a source function at its defaults (or ``params``), checks them as ``simulate``
does at first use, and compares the weighted directions with ``exp(logpdf)`` (normalised over the sphere) in cells of
equal solid angle, each cell's density by quadrature: a Pearson chi^2 against ``lumen.validate.stats.chi2_passes``.
With a ``dlogp``, it also compares it with ``grad_logpdf(logpdf)`` in the tangent plane. The medium and surface models
are checked at their first build (``lumen.models``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen.models import ModelError, check_lobe
from lumen.validate import check_gradient, stats

N_MU, N_PHI = 16, 16  # equal solid-angle bins: uniform in cos(theta) and phi about +z
QUAD_MU, QUAD_PHI = 16, 16  # quadrature points per cell, in cos(theta) and phi
GRAD_TOL = 1e-3  # relative tangent-plane difference of dlogp from grad_logpdf


@dataclass(frozen=True)
class SourceReport:
    """``check_source``'s result: chi^2 / ndf of the directions, the dlogp difference, and whether both pass."""

    chi2: float
    ndf: int
    dlogp_error: float
    passes: bool

    def __str__(self):
        """Summarise."""
        return (f"check_source: chi2/ndf {self.chi2:.2f} over {self.ndf}, dlogp error {self.dlogp_error:.2g}: "
                f"{'passes' if self.passes else 'FAILS'}")  # fmt: skip


def _bins(d):
    mu = np.clip(d[:, 2], -1.0, 1.0 - 1e-12)
    phi = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi)
    return np.minimum(((mu + 1.0) / 2.0 * N_MU).astype(int), N_MU - 1) * N_PHI + np.minimum(
        (phi / (2 * np.pi) * N_PHI).astype(int), N_PHI - 1
    )


def _cell_mass(logpdf):
    """Return each cell's share of exp(logpdf): Gauss-Legendre in cos(theta) and midpoints in phi within each cell."""
    x, wq = np.polynomial.legendre.leggauss(QUAD_MU)
    mu_e, phi_e = np.linspace(-1.0, 1.0, N_MU + 1), np.linspace(0.0, 2 * np.pi, N_PHI + 1)
    mu = 0.5 * (mu_e[1:, None] - mu_e[:-1, None]) * x + 0.5 * (mu_e[1:, None] + mu_e[:-1, None])  # (N_MU, q)
    k = (np.arange(QUAD_PHI) + 0.5) / QUAD_PHI
    phi = phi_e[:-1, None] + (phi_e[1:, None] - phi_e[:-1, None]) * k  # (N_PHI, q)
    m, f = mu[:, None, :, None], phi[None, :, None, :]
    s = np.sqrt(np.clip(1.0 - m * m, 0.0, 1.0))
    d = np.stack(np.broadcast_arrays(s * np.cos(f), s * np.sin(f), m), -1).reshape(-1, 3)
    p = np.exp(np.asarray(jax.vmap(logpdf)(jnp.asarray(d, jnp.float32)), np.float64))
    p = p.reshape(N_MU, N_PHI, QUAD_MU, QUAD_PHI) * wq[None, None, :, None]
    mass = p.sum((2, 3)).reshape(-1)
    return mass / mass.sum()


def check_source(src, logpdf=None, *, n: int = 1 << 16, params=None, seed: int = 0) -> SourceReport:
    """Return the SourceReport of a source function (see the module docstring); ``logpdf`` of one direction (3,)."""
    from lumen.sources.function import grad_logpdf

    src.check(params)
    ph = src(src.values(params), jax.random.PRNGKey(seed), int(n))
    d, w = np.asarray(ph.d, np.float64), np.asarray(ph.w, np.float64)
    if logpdf is None:
        return SourceReport(0.0, 0, 0.0, True)
    b = _bins(d)
    obs = np.bincount(b, w, N_MU * N_PHI)
    var = np.bincount(b, w * w, N_MU * N_PHI)
    expect = _cell_mass(logpdf) * w.sum()
    ok = var > 0.0
    ndf = max(int(ok.sum()) - 1, 1)
    chi2 = float(np.sum((obs[ok] - expect[ok]) ** 2 / var[ok]) / ndf)
    err = 0.0
    if ph.dlogp is not None:
        want = np.asarray(grad_logpdf(logpdf, ph.d), np.float64)
        tang = lambda v: v - np.sum(v * d, 1, keepdims=True) * d  # noqa: E731 (the chain reads the tangent part)
        a, b_ = tang(np.asarray(ph.dlogp, np.float64)), tang(want)
        err = float(np.max(np.linalg.norm(a - b_, axis=1) / np.maximum(np.linalg.norm(b_, axis=1), 1.0)))
    return SourceReport(chi2, ndf, err, bool(stats.chi2_passes(chi2, ndf) and err <= GRAD_TOL))


__all__ = ["ModelError", "SourceReport", "check_gradient", "check_lobe", "check_source"]
