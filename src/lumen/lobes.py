"""Directional distributions ("lobes"), one record for all, and the Lambertian surface lobes.

A ``Lobe`` describes the distribution of the outgoing direction ``wo`` given the incoming propagation direction
``wi``, a facing normal ``n`` (wi . n < 0; unused by bulk phase functions) and parameters ``p`` (a dict of arrays):

- ``sample(p, wi, n, rand) -> wo`` with ``rand`` the lobe's (n_rand,) uniforms in (0, 1);
- ``logpdf(p, wi, n, wo)``: the solid-angle density of ``sample`` (None for a facet-path lobe, below).

Directions are drawn at the reference values and weighted by exp(logpdf(live) - logpdf(reference)): that is the only
way parameters enter a drawn direction.

A facet-path lobe (Geant4's facet reflections, repeated until the photon leaves the surface, uncapped) has no
closed-form marginal density (``logpdf`` None). It declares its path instead: ``path(p, wi, n, rand, visit, carry) ->
(carry, w_prev, wo, k)`` draws the path and folds ``visit(carry, w0, w1)`` over its k facets (w_prev: the last facet's
incoming direction), and ``facet_logpdf(p, wi, n, wo)`` is one facet's density: the path density is its sum. A
polarised phase function also declares its density given the photon's Jones vector: ``pol_sample(p, wi, er, ei,
rand)`` and ``pol_logpdf(p, wi, er, ei, wo)``.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp

from lumen._core import frames
from lumen._core.safe import safe_log
from lumen.units import TWO_PI


@dataclass(frozen=True, eq=False)
class Lobe:
    """A distribution of outgoing directions; static, hashed by identity. See the module docstring."""

    name: str
    surface: bool  # True: n is the facing surface normal; False: a phase function (n unused)
    n_rand: int
    params: tuple[str, ...]
    sample: Callable
    logpdf: Callable | None
    path: Callable | None = None
    facet_logpdf: Callable | None = None
    facet_proposal_logpdf: Callable | None = None  # one facet's density ``path`` draws from, when not ``facet_logpdf``
    pol_sample: Callable | None = None
    pol_logpdf: Callable | None = None
    # the one defensive mixture (lumen.transport.estimator): {parameter: role} the parameters that can zero a part of
    # the lobe, and ``cover(p, e)``: the parameters the lobe is drawn at, its reference ``p`` mixed with e of its cover
    defensive: dict = field(default_factory=dict)
    cover: Callable | None = None
    declared: tuple = ()  # ((name, Param), ...): a phase function of your own's parameters (``media.phase_function``)

    def __repr__(self):
        """Show the lobe's name."""
        return f"Lobe({self.name})"


def _cosine(rand, axis):
    r = jnp.sqrt(rand[..., 0])
    phi = TWO_PI * rand[..., 1]
    local = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), jnp.sqrt(1.0 - rand[..., 0])], -1)
    return frames.to_world(local, axis)


def _cosine_logpdf(wo, axis):
    return safe_log(jnp.sum(wo * axis, -1)) - math.log(math.pi)


LAMBERT = Lobe(
    "lambert",
    surface=True,
    n_rand=2,
    params=(),
    sample=lambda p, wi, n, rand: _cosine(rand, n),
    logpdf=lambda p, wi, n, wo: _cosine_logpdf(wo, n),
)
