"""Reweighting: the only place where random choices meet parameters.

Every random choice is drawn from probabilities at the reference values (stop-gradient) and weighted by live /
reference. In value the weight is 1 when live equals reference; its derivative is the score of the choice, so the photon
weight carries every upstream score and a deposit's derivative is the sum of pathwise and score terms. Drawn directions
and distances are kept fixed in the world frame; their densities are re-evaluated live. A surface outcome is drawn from
the reference probabilities of the continuing outcomes alone, normalised.

**The one defensive mixture.** A draw at the reference covers the live density only where the reference density is
positive. A sampled part declares the parameters that can make it fail to (a share in [0, 1] that can zero a
probability: role "zero"; a parameter that moves the edge of a support: role "support") and a covering density c,
positive wherever such a live density can be positive and the reference is not: a surface outcome
(``lumen.surfaces.types.Interaction.defensive``; c uniform over its outcomes of reference probability 0), GLISUR's
facet (``lumen.surfaces.facets.covered``; c the shell), a bulk lobe's choice (``lumen.lobes.Lobe.defensive``: Mie's two
lobes), a scintillation component and a spectrum interval (``lumen.sources``, ``lumen.media.spectra.covered``). When
such a parameter is keyed (decided by structure, never by a value or a tangent: ``edge_rows``; a source's and a
spectrum's shares are always parameters) the part is drawn from (1 - e) p_ref + e c over the event's whole
distribution (a surface's absorption included), e = ``DEFENSIVE_SHARE``, and from p_ref where nothing is covered at the
reference (e is not paid). The live weight multiplies p_live / q; the reference weight carries the total of the weights
drawn from (a surface's continuing share, the covering share e / (1 - e) included): the roulette keeps a photon with
probability wr / eps and divides its derivative by it, so a reference weight carrying p_ref / q (0 at a covered draw)
would end the photons that carry the derivative (measured: x51 the variance, gradfam_p3.md section 7). The covering
share is of the whole distribution, not of its continuing part: at a surface that absorbs every photon at the
reference (S = 0) a share of S would leave the reference weight unchanged, and a photon of weight 0 in a cavity of such
surfaces (a PMT's inside, every wall absorbing) would never meet the roulette (millions of steps).

**No photon is ended for carrying weight 0.** After a covered draw of live probability 0 the photon goes on with
weight 0 in value and its derivative's weight; a second such draw on the same path carries the second derivative (a
path through two reflections at R = 0 goes as R^2), so ending it would drop that term (measured: 44 % of d2Q/dR2 at
a reflectivity of 0 on the YAML toy, tests/physics/test_ghost_hessian.py). Such a photon ends as any other: absorbed,
by the roulette (its reference weight falls by e / (1 - e) at each covered draw of a surface absorbing everything at
the reference: unbiased, the survivors carrying the rest), or at the step guard (``health.truncated``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core.safe import safe_div
from lumen.rules.choices import DEFENSIVE_SHARE  # stated there (no expectation depends on it)
from lumen.surfaces.outcomes import N_OUTCOMES

sg = jax.lax.stop_gradient


def reweight_log(lp_live, lp_ref):
    """Return exp(lp_live - sg(lp_ref)) for log densities (``lumen._core.safe.score`` where both are one finite lp)."""
    return jnp.exp(lp_live - sg(lp_ref))


def proposal(p_ref):
    """Return the outcome proposal: the reference probabilities normalised (0 where they are all 0)."""
    p_ref = sg(p_ref)
    tot = jnp.sum(p_ref)
    return jnp.where(tot > 0.0, safe_div(p_ref, tot), 0.0)


def choose(q, rand):
    """Return the index drawn from the probability vector q with one uniform; -1 if q is all zero."""
    c = jnp.cumsum(q)
    k = jnp.argmax(c > rand * c[-1])
    return jnp.where(c[-1] > 0.0, k, -1)


def role(tables, row, name):
    """Return the defensive role of the surface row ``row``'s parameter ``name`` (its type's ``role``), or None."""
    return tables.rows[row].surface_type.role(name)


def lobe_role(tables, medium, channel, name):
    """Return the defensive role of a bulk lobe's parameter (the lobe's ``defensive``), or None."""
    for m in tables.media:
        for ch in m.channels if m.name == medium else ():
            if ch.name == channel:
                return ch.lobe.defensive.get(name)
    return None


def defensive(p, cover, e=DEFENSIVE_SHARE):
    """Return the weights a categorical part is drawn from: the one defensive mixture.

    ``p``: the reference probabilities of the part's outcomes, a share of the event's whole distribution (which sums to
    1 with the outcomes not drawn here: a surface's absorption); ``cover`` (bool): the outcomes the part may propose.
    Every covered outcome of reference probability 0 gets e / (1 - e) / n (n of them): the covering density takes the
    share e of the whole distribution, whatever share of it ``p`` holds; ``p`` itself where none is covered. Where
    ``p`` is all 0 (a surface that absorbs every photon at the reference) the draw's total is e / (1 - e), the factor
    the reference weight takes (``lumen.transport.step``): such a photon falls below the roulette's ε as any photon
    whose reference weight falls, and the roulette ends it or keeps it, unbiased.
    """
    p = sg(p)
    fl = cover & (p <= 0.0)
    n = jnp.maximum(jnp.sum(fl), 1)
    eps = jnp.asarray(e / (1.0 - e), p.dtype) / n
    return jnp.where(fl, eps, p)


def defensive_q(p, cover, e=DEFENSIVE_SHARE):
    """Return the normalised probabilities of ``defensive`` (``p`` itself, normalised as given, where none is
    covered)."""  # noqa: D205, D209
    q = defensive(p, cover, e)
    return jnp.where(jnp.any(cover & (sg(p) <= 0.0)), q / jnp.sum(q), sg(p))


class EdgeRows(NamedTuple):
    """The defensive rows (``edge_rows``): ``surfaces`` the surface table's rows, ``lobes`` the bulk lobes' parameters
    (medium, channel, name). Hashable (a static part of ``StepConfig``); false when there are none."""  # noqa: D205, D209

    surfaces: tuple[int, ...] = ()
    lobes: tuple[tuple[str, str, str], ...] = ()

    def __bool__(self):
        """Return whether there is any edge row."""
        return bool(self.surfaces or self.lobes)


def edge_rows(tables, paths) -> EdgeRows:
    """Return the defensive rows of the packed-parameter ``paths`` (the detector's parameters): every row of a
    parameter with a defensive role, whatever its value (decided by structure, so the estimator never depends on a
    value).

    ``paths``: ("surfaces", row, name) for a surface parameter (its row type's ``role``), giving the surface row; and
    ("media", medium, channel, "lobe", name) for a bulk lobe's parameter (the lobe's ``defensive``), giving (medium,
    channel, name). Anything else has none. Where nothing is covered at the reference the rows draw from it as is.
    """  # noqa: D205
    surfaces, lobes = set(), set()
    for path in paths:
        if len(path) >= 5 and path[0] == "media" and path[3] == "lobe":
            if lobe_role(tables, path[1], path[2], path[4]) is not None:
                lobes.add((path[1], path[2], path[4]))
        elif len(path) == 3 and path[0] == "surfaces" and role(tables, path[1], path[2]) is not None:
            surfaces.add(int(path[1]))
    return EdgeRows(surfaces=tuple(sorted(surfaces)), lobes=tuple(sorted(lobes)))


def _row_table(tables, rows):
    """Return the static (n_rows,) bool table of the surface edge rows."""
    a = np.zeros(max(len(tables.rows), 1), bool)
    a[list(rows.surfaces)] = True
    return a


def edge_floor(pr_r, row, tables, rows):
    """Return the reference probabilities a surface outcome is drawn from: at an edge row the one defensive mixture
    (``defensive``) over its continuing outcomes, its support's outcomes covered; elsewhere the reference."""  # noqa: D205, D209
    E = jnp.asarray(_row_table(tables, rows))
    sup = np.zeros((max(len(tables.rows), 1), N_OUTCOMES), bool)
    sup[: len(tables.rows)] = tables.table.support
    k = jnp.maximum(row, 0)
    return defensive(pr_r, ((row >= 0) & E[k]) & jnp.asarray(sup)[k])
