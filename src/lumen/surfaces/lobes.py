"""Directional distributions ("lobes"), one record for all, and the surface lobes: specular and Lambertian.

A ``Lobe`` describes the distribution of the outgoing direction ``wo`` given the incoming propagation direction
``wi``, a facing normal ``n`` (wi . n < 0; unused by bulk phase functions) and parameters ``p`` (a dict of arrays):

- ``sample(p, wi, n, rand) -> wo`` with ``rand`` the lobe's (n_rand,) uniforms in (0, 1);
- ``logpdf(p, wi, n, wo)``: the solid-angle density of ``sample`` (None for a deterministic lobe);
- ``weight(p, wi, n, wo)``: the throughput f cos / pdf of a sampled direction, in [0, 1];
- ``axis(p, wi, n)``: the lobe's centre.

Directions are drawn at the reference values and weighted by exp(logpdf(live) - logpdf(reference)) times the live
weight: that is the only way parameters enter a drawn direction. A deterministic lobe's direction is a
differentiable map.

A chain lobe (Geant4's facet reflections, repeated until the photon leaves the surface) has no closed-form marginal
density (``logpdf`` None). It declares its path instead: ``path(p, wi, n, rand) -> (W (k_max, 3), k)`` with
wo = W[k - 1], and ``path_logpdf(p, wi, n, W, k)``. A polarised phase function also declares its density given the
photon's Jones vector: ``pol_sample(p, wi, er, ei, rand)`` and ``pol_logpdf(p, wi, er, ei, wo)``.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp

from lumen._core import frames
from lumen._core.safe import safe_log

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, eq=False)
class Lobe:
    """A distribution of outgoing directions; static, hashed by identity. See the module docstring."""

    name: str
    delta: bool
    surface: bool  # True: n is the facing surface normal; False: a phase function (n unused)
    n_rand: int
    params: tuple[str, ...]
    sample: Callable
    logpdf: Callable | None
    weight: Callable
    axis: Callable
    path: Callable | None = None
    path_logpdf: Callable | None = None
    pol_sample: Callable | None = None
    pol_logpdf: Callable | None = None
    domains: dict = field(default_factory=dict)  # parameter -> (lo, hi): an edge makes part of the lobe impossible

    def __repr__(self):
        """Show the lobe's name."""
        return f"Lobe({self.name})"


def _ones(p, wi, n, wo):
    return jnp.ones(wi.shape[:-1])


def _mirror(p, wi, n, rand):
    return frames.reflect(wi, n)


SPECULAR = Lobe(
    "specular",
    delta=True,
    surface=True,
    n_rand=0,
    params=(),
    sample=_mirror,
    logpdf=None,
    weight=_ones,
    axis=lambda p, wi, n: frames.reflect(wi, n),
)


def _cosine(rand, axis):
    r = jnp.sqrt(rand[..., 0])
    phi = TWO_PI * rand[..., 1]
    local = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), jnp.sqrt(1.0 - rand[..., 0])], -1)
    return frames.to_world(local, axis)


def _cosine_logpdf(wo, axis):
    return safe_log(jnp.sum(wo * axis, -1)) - math.log(math.pi)


LAMBERT = Lobe(
    "lambert",
    delta=False,
    surface=True,
    n_rand=2,
    params=(),
    sample=lambda p, wi, n, rand: _cosine(rand, n),
    logpdf=lambda p, wi, n, wo: _cosine_logpdf(wo, n),
    weight=_ones,
    axis=lambda p, wi, n: n,
)
LAMBERT_T = Lobe(
    "lambert_t",
    delta=False,
    surface=True,
    n_rand=2,
    params=(),
    sample=lambda p, wi, n, rand: _cosine(rand, -n),
    logpdf=lambda p, wi, n, wo: _cosine_logpdf(wo, -n),
    weight=_ones,
    axis=lambda p, wi, n: -n,
)
