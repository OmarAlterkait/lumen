"""Phase functions: bulk scattering lobes about the incoming direction.

Isotropic and Rayleigh scattering, with densities per solid angle; mu = wo . wi is the cosine of the scattering
angle. Rayleigh scattering also has its polarised form (G4OpRayleigh): the scattered intensity is proportional to
1 - |k' . E|^2 for the photon's Jones vector E, and the new state is the projection E - (k' . E) k'. With
E = er + i ei and |er|^2 + |ei|^2 = 1 the density is the mixture |er|^2 dipole(er) + |ei|^2 dipole(ei), which
``rayleigh_sample`` draws exactly.

Mie scattering is Geant4's G4OpMieHG (identical in 10.7 and 11.2): a forward Henyey-Greenstein lobe HG(g_f) with
probability r, else the backward lobe HG(-g_b),

    p(mu) = r HG(mu; g_f) + (1 - r) HG(mu; -g_b),    HG(mu; g) = (1 - g^2) / (4 pi (1 + g^2 - 2 g mu)^(3/2)),

with g_f = MIEHG_FORWARD, g_b = MIEHG_BACKWARD (positive for a backward-peaked lobe) and r = MIEHG_FORWARD_RATIO.
``MIE_HG`` draws it with Geant4's own inverse CDF in a form that never cancels in float32 (``mie_hg_cos``), and its
density is the mixture over both lobes (``mie_hg_logpdf``): the exact density of the sampler, smooth in the direction
and in the three parameters. The direction density does not depend on the polarisation (the new state is the
projection, as for Rayleigh scattering).
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
MIE_G_MAX = 0.999  # the largest |g| of a Mie lobe: float32 resolves directions to ~3.5e-4 rad, the lobe is ~1 - g wide
MIE_PARAMS = ("g_forward", "g_backward", "forward_ratio")  # MIEHG_FORWARD, MIEHG_BACKWARD, MIEHG_FORWARD_RATIO


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


def mie_hg_cos(g, rand):
    """Return (1 - mu, 1 + mu) of Geant4's Henyey-Greenstein draw for one uniform ``rand`` and asymmetry ``g``.

    Notes
    -----
    G4OpMieHG::PostStepDoIt: mu = 2 r (1 + g)^2 (1 - g + g r) / (1 - g + 2 g r)^2 - 1, the exact inverse CDF of
    HG(g). With D = 1 - g + 2 g r, the identity D^2 - r (1 + g)^2 (1 - g + g r) = (1 - g)^2 (1 - r) (1 + g r) gives

        1 - mu = 2 (1 - g)^2 (1 - r) (1 + g r) / D^2,    1 + mu = 2 r (1 + g)^2 (1 - g + g r) / D^2.

    Every factor is a sum of positive terms, so nothing cancels in float32 for either sign of g: for g < 0,
    D = (1 - g)(1 - r) + (1 + g) r and 1 + g r = (1 - r) + (1 + g) r. At g = 0 it is exactly 1 - mu = 2 (1 - r).
    """
    up = g >= 0.0
    d = jnp.where(up, (1.0 - g) + 2.0 * g * rand, (1.0 - g) * (1.0 - rand) + (1.0 + g) * rand)
    one_gr = jnp.where(up, 1.0 + g * rand, (1.0 - rand) + (1.0 + g) * rand)
    d2 = d * d
    c_minus = 2.0 * (1.0 - g) ** 2 * (1.0 - rand) * one_gr / d2
    c_plus = 2.0 * rand * (1.0 + g) ** 2 * ((1.0 - g) + g * rand) / d2
    return c_minus, c_plus


def hg_density(g, c_minus, c_plus):
    """Return HG(mu; g) per steradian from c_minus = 1 - mu and c_plus = 1 + mu, without cancellation.

    1 + g^2 - 2 g mu is (1 - g)^2 + 2 g (1 - mu) for g >= 0 and (1 + g)^2 - 2 g (1 + mu) for g < 0: positive terms
    either way (the two forms are equal, so the density is smooth across g = 0).
    """
    q = jnp.where(g >= 0.0, (1.0 - g) ** 2 + 2.0 * g * c_minus, (1.0 + g) ** 2 - 2.0 * g * c_plus)
    return (1.0 - g) * (1.0 + g) / (FOUR_PI * q**1.5)


def mie_hg_sample(p, wi, rand):
    """Return a direction drawn by G4OpMieHG about ``wi`` with 3 uniforms: lobe, cosine, azimuth.

    ``rand[0] <= forward_ratio`` takes the forward lobe (Geant4's ``G4UniformRand() <= forwardRatio``), else the
    backward one, theta -> pi - theta; the azimuth is uniform about ``wi``.
    """
    forward = rand[..., 0] <= p["forward_ratio"]
    g = jnp.where(forward, p["g_forward"], p["g_backward"])
    c_minus, c_plus = mie_hg_cos(g, rand[..., 1])
    c_minus, c_plus = jnp.where(forward, c_minus, c_plus), jnp.where(forward, c_plus, c_minus)
    mu = jnp.where(c_minus < c_plus, 1.0 - c_minus, c_plus - 1.0)
    s = safe_sqrt(c_minus * c_plus)
    phi = TWO_PI * rand[..., 2]
    return frames.to_world(jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), mu], -1), wi)


def mie_hg_logpdf(p, wi, wo):
    """Return the log density per steradian of ``mie_hg_sample``: both lobes, the lobe choice integrated out.

    Notes
    -----
    1 - mu = |wo - wi|^2 / 2 and 1 + mu = |wo + wi|^2 / 2 for unit vectors (neither cancels); the forward lobe is
    HG(g_f) and the backward one HG(-g_b). For |g| <= MIE_G_MAX each lobe lies in [2e-6, 1.6e5] per sr, so the
    mixture is summed directly.
    """
    r = p["forward_ratio"]
    c_minus = 0.5 * jnp.sum((wo - wi) ** 2, -1)
    c_plus = 0.5 * jnp.sum((wo + wi) ** 2, -1)
    forward = hg_density(p["g_forward"], c_minus, c_plus)
    backward = hg_density(-p["g_backward"], c_minus, c_plus)
    return jnp.log(r * forward + (1.0 - r) * backward)


MIE_HG = Lobe(
    "mie_hg",
    delta=False,
    surface=False,
    n_rand=3,
    params=MIE_PARAMS,
    sample=lambda p, wi, n, rand: mie_hg_sample(p, wi, rand),
    logpdf=lambda p, wi, n, wo: mie_hg_logpdf(p, wi, wo),
    weight=_ones,
    axis=lambda p, wi, n: wi,
)
