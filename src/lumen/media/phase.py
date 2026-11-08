"""Phase functions: bulk scattering lobes about the incoming direction.

Isotropic and Rayleigh scattering, with densities per solid angle; mu = wo . wi is the cosine of the scattering
angle. Rayleigh scattering also has its polarised form (G4OpRayleigh): the scattered intensity is proportional to
1 - |k' . E|^2 for the photon's Jones vector E, and the new state is the projection E - (k' . E) k'. With
E = er + i ei and |er|^2 + |ei|^2 = 1 the density is the mixture |er|^2 dipole(er) + |ei|^2 dipole(ei), which
``rayleigh_sample`` draws exactly.
"""

import math

import jax.numpy as jnp

from lumen._core import frames
from lumen._core.safe import safe_sqrt, sanitize, unit
from lumen.surfaces.lobes import Lobe

TWO_PI = 2.0 * math.pi
FOUR_PI = 4.0 * math.pi
EIGHT_PI_3 = 8.0 * math.pi / 3.0
LOG_TINY = 1e-30  # the density's floor in the logarithm, where no light goes


def _about(axis, mu, phi):
    s = safe_sqrt(1.0 - mu * mu)
    return frames.to_world(jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), mu], -1), axis)


def _ones(p, wi, n, wo):
    return jnp.ones(wi.shape[:-1])


def _cbrt(x):
    return jnp.sign(x) * jnp.abs(x) ** (1.0 / 3.0)


def rayleigh_mu(rand):
    """Return the inverse CDF of p(mu) = 3/8 (1 + mu^2): the real root of mu^3 + 3 mu + 4 - 8 rand = 0 (Cardano)."""
    q = 4.0 - 8.0 * rand
    r = jnp.sqrt(0.25 * q * q + 1.0)
    return jnp.clip(_cbrt(-0.5 * q + r) + _cbrt(-0.5 * q - r), -1.0, 1.0)


def dipole_cos(rand):
    """Return the inverse CDF of p(c) = 3/4 (1 - c^2) on [-1, 1]: the root of c^3 - 3c + 4 rand - 2 = 0 there."""
    return jnp.clip(2.0 * jnp.cos((jnp.arccos(jnp.clip(1.0 - 2.0 * rand, -1.0, 1.0)) - TWO_PI) / 3.0), -1.0, 1.0)


def rayleigh_sample(er, ei, rand):
    """Return k' drawn from 3 / (8 pi) (1 - |k' . E|^2), using 3 uniforms.

    The dipole axis is er with probability |er|^2, else ei; then the dipole distribution about that axis.
    """
    a2 = jnp.sum(er * er, -1)
    norm2 = jnp.sum(er * er, -1) + jnp.sum(ei * ei, -1)
    axis = jnp.where((rand[..., 2] * norm2 < a2)[..., None], er, ei)
    axis = unit(axis)
    c = dipole_cos(rand[..., 0])
    s = safe_sqrt(1.0 - c * c)
    phi = TWO_PI * rand[..., 1]
    return frames.to_world(jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), c], -1), axis)


def rayleigh_logpdf(er, ei, k):
    """Return the log of 3 / (8 pi) (1 - |k . E|^2) for a state with |E| = 1."""
    v = 1.0 - jnp.sum(k * er, -1) ** 2 - jnp.sum(k * ei, -1) ** 2
    ok = v > LOG_TINY
    return jnp.where(ok, jnp.log(sanitize(ok, v, 1.0)), math.log(LOG_TINY)) - math.log(EIGHT_PI_3)


ISOTROPIC = Lobe(
    "isotropic",
    delta=False,
    surface=False,
    n_rand=2,
    params=(),
    sample=lambda p, wi, n, rand: _about(wi, 1.0 - 2.0 * rand[..., 0], TWO_PI * rand[..., 1]),
    logpdf=lambda p, wi, n, wo: jnp.full(wi.shape[:-1], -math.log(FOUR_PI)),
    weight=_ones,
    axis=lambda p, wi, n: wi,
)
RAYLEIGH = Lobe(
    "rayleigh",
    delta=False,
    surface=False,
    n_rand=2,
    params=(),
    sample=lambda p, wi, n, rand: _about(wi, rayleigh_mu(rand[..., 0]), TWO_PI * rand[..., 1]),
    logpdf=lambda p, wi, n, wo: jnp.log(3.0 / (16.0 * math.pi) * (1.0 + jnp.sum(wo * wi, -1) ** 2)),
    weight=_ones,
    axis=lambda p, wi, n: wi,
    pol_sample=lambda p, wi, er, ei, rand: rayleigh_sample(er, ei, rand),
    pol_logpdf=lambda p, wi, er, ei, wo: rayleigh_logpdf(er, ei, wo),
)
