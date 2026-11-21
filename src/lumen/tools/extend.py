"""Checks for code that extends lumen: a source function's draws, a medium lobe, any derivative.

    lumen.tools.extend.check_source(led, logpdf=lambda d: 4.0 * jnp.log(jnp.clip(-d[2], 1e-12)))  # draws vs the density
    lumen.tools.extend.check_lobe(lumen.media.henyey_greenstein("g"), {"g": 0.9})                  # sampler vs density
    lumen.tools.check_gradient(f, theta0)                                                  # AD against FD

``check_source`` draws ``n`` photons of a source function at its defaults (or ``params``), checks them as ``simulate``
does at first use, and compares the weighted directions with ``exp(logpdf)`` (normalised over the sphere) in cells of
equal solid angle, each cell's density by quadrature: a Pearson chi^2 against ``lumen.tools.stats.chi2_passes``. A
medium model is checked at its first build (``lumen.models``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core.checks import check_lobe
from lumen._core.errors import ModelError
from lumen.tools import stats
from lumen.tools.check import check_gradient
from lumen.units import TWO_PI

N_MU, N_PHI = 16, 16  # equal solid-angle bins: uniform in cos(theta) and phi about +z
QUAD_MU, QUAD_PHI = 16, 16  # quadrature points per cell, in cos(theta) and phi


@dataclass(frozen=True)
class SourceReport:
    """``check_source``'s result: chi^2 / ndf of the directions and whether it passes."""

    chi2: float
    ndf: int
    passes: bool

    def __str__(self):
        """Summarise."""
        return f"check_source: chi2/ndf {self.chi2:.2f} over {self.ndf}: {'passes' if self.passes else 'FAILS'}"


def _bins(d):
    mu = np.clip(d[:, 2], -1.0, 1.0 - 1e-12)
    phi = np.mod(np.arctan2(d[:, 1], d[:, 0]), TWO_PI)
    return np.minimum(((mu + 1.0) / 2.0 * N_MU).astype(int), N_MU - 1) * N_PHI + np.minimum(
        (phi / TWO_PI * N_PHI).astype(int), N_PHI - 1
    )


def _cell_mass(logpdf):
    """Return each cell's share of exp(logpdf): Gauss-Legendre in cos(theta) and midpoints in phi within each cell."""
    x, wq = np.polynomial.legendre.leggauss(QUAD_MU)
    mu_e, phi_e = np.linspace(-1.0, 1.0, N_MU + 1), np.linspace(0.0, TWO_PI, N_PHI + 1)
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


def check_source(src, logpdf, *, n: int = 1 << 16, params=None, seed: int = 0) -> SourceReport:
    """Return the SourceReport of a source function (see the module docstring); ``logpdf`` of one direction (3,), the
    density the draws are compared with (required: with nothing to compare, nothing passes)."""  # noqa: D205, D209
    src.check(params)
    ph = src(src.values(params), jax.random.PRNGKey(seed), int(n))
    d, w = np.asarray(ph.d, np.float64), np.asarray(ph.w, np.float64)
    b = _bins(d)
    obs = np.bincount(b, w, N_MU * N_PHI)
    var = np.bincount(b, w * w, N_MU * N_PHI)
    expect = _cell_mass(logpdf) * w.sum()
    ok = var > 0.0
    ndf = max(int(ok.sum()) - 1, 1)
    chi2 = float(np.sum((obs[ok] - expect[ok]) ** 2 / var[ok]) / ndf)
    return SourceReport(chi2, ndf, bool(stats.chi2_passes(chi2, ndf)))


__all__ = ["ModelError", "SourceReport", "check_gradient", "check_lobe", "check_source"]
