"""Reweighting: the only place where random choices meet parameters.

Every random choice is drawn from probabilities at the reference values (stop-gradient) and weighted by live /
reference. In value the weight is 1 when live equals reference; its derivative is the score of the choice (the "magic
box"), so the photon weight carries every upstream score and a deposit's derivative is the sum of pathwise and score
terms. Drawn directions and distances are held fixed in the world frame; their densities are re-evaluated live. A
surface outcome is drawn from the reference probabilities of the continuing outcomes alone, normalised.

**The edge floor** (the edge rule, OUT or WITHIN): a ``wrt`` parameter whose value sits at an edge of its domain (a
fraction at 0 or 1) turns its rows on (``edge_rows``: derived from ``wrt`` and the domains, never an option). At those
rows every continuing outcome of the surface type whose reference probability is 0 is proposed at eps: drawn, the
photon carries eps / q and ``Photon.z`` carries p / eps, exactly 0 in value and dp / eps in derivative. eps gives the
floored outcomes the share e = EDGE_BUDGET (1 - S) of the draws, S the continuing reference mass. A Mie forward ratio
at 0 or 1 in ``wrt`` has its reference moved inside by the same share (``mie_reference``); its live weight is the
direction density's ratio.

Notes
-----
Ported from the baseline's lumen.estimator (magic, magic_log, choose, edge_rows_for, edge_rows, edge_floor). The
defensive mix is gone: the port is its mix 0 (the proposal is the normalised reference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core.safe import safe_div
from lumen.surfaces.outcomes import N_OUTCOMES, TERMINAL

sg = jax.lax.stop_gradient
EDGE_BUDGET = 0.01  # the share of a vertex's draws the floored outcomes take, times (1 - S) (baseline's edge_budget)
EDGE_SHARE_MIN, EDGE_SHARE_MAX = 1e-6, 0.5  # that share's clip
# the surface parameters that are fractions (domain [0, 1]); a layered type's backing parameter b:<name> takes its own
FRACTIONS = frozenset({"R", "T", "A", "a", "eff", "qe", "det", "eff_fwd", "f_fwd", "f_spec", "f_glossy", "p_spike",
                       "p_lobe", "p_back"})  # fmt: skip
MIE_RATIO = "forward_ratio"


def magic(live, ref):
    """Return live / sg(ref): value live / ref (1 when equal), derivative d live / ref; 0 where ref = 0."""
    return safe_div(live, sg(ref))


def magic_log(lp_live, lp_ref):
    """Return exp(lp_live - sg(lp_ref)) for log densities."""
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


def is_fraction(name):
    """Return whether a surface parameter is a fraction (domain [0, 1])."""
    return name.split(":")[-1] in FRACTIONS


def _at_edge(a):
    return np.any((a <= 0.0) | (a >= 1.0))


def edge_rows(tables, params, wrt):
    """Return the edge rows, ((type name, row), ...): where a ``wrt`` parameter sits at an edge of its domain.

    ``wrt`` holds packed-parameter paths: ("kernel", type, name) (every row) or ("kernel", type, name, row) for
    surface fractions, and ("media", medium, "mie", "lobe", "forward_ratio") for a Mie forward ratio (the row is then
    ("mie:" + medium, 0)). Anything else has no edge here.
    """
    out = set()
    names = {k.name for k in tables.kinds}
    for path in wrt:
        if len(path) >= 5 and path[0] == "media" and path[2:5] == ("mie", "lobe", MIE_RATIO):
            if _at_edge(np.asarray(params["media"][path[1]]["mie"]["lobe"][MIE_RATIO], np.float64)):
                out.add((f"mie:{path[1]}", 0))
            continue
        if len(path) < 3 or path[0] != "kernel" or path[1] not in names or not is_fraction(path[2]):
            continue
        v = params["kernel"][path[1]][path[2]]
        a = np.asarray(v["value"] if isinstance(v, dict) else v, np.float64)
        a = a.reshape(a.shape[0], -1) if a.ndim else a.reshape(1, 1)
        rows = range(a.shape[0]) if len(path) < 4 or path[3] is None else (int(path[3]),)
        out |= {(path[1], r) for r in rows if _at_edge(a[r])}
    return tuple(sorted(out))


def _row_table(tables, rows):
    """Return the static (n_types, n_rows) bool table of the surface edge rows."""
    names = [k.name for k in tables.kinds]
    rows = [(k, r) for k, r in rows if not k.startswith("mie:")]
    a = np.zeros((max(len(names), 1), max([r for _, r in rows] + [0]) + 1), bool)
    for kn, r in rows:
        if kn not in names:
            raise ValueError(f"edge rows: no surface type {kn!r} (types {names})")
        a[names.index(kn), int(r)] = True
    return a


def edge_floor(pr_r, kidx, row, tables, rows):
    """Return (the reference probabilities the outcome is drawn from, which outcomes were floored (K,) bool).

    At an edge row every continuing outcome of the type's support with reference probability 0 is proposed at eps
    (module docstring); elsewhere the reference is unchanged.
    """
    E = jnp.asarray(_row_table(tables, rows))
    sup = np.zeros((max(len(tables.kinds), 1), N_OUTCOMES), bool)
    for i, kd in enumerate(tables.kinds):
        for o in kd.outcomes:
            sup[i, o] = not TERMINAL[o]
    k = jnp.maximum(kidx, 0)
    at = (kidx >= 0) & (row >= 0) & (row < E.shape[1]) & E[k, jnp.clip(row, 0, E.shape[1] - 1)]
    p = sg(pr_r)
    cont = ~jnp.asarray(TERMINAL)
    fl = at & jnp.asarray(sup)[k] & (p <= 0.0)
    S = jnp.sum(jnp.where(cont, p, 0.0))
    n = jnp.maximum(jnp.sum(fl), 1)
    e = jnp.clip(EDGE_BUDGET * (1.0 - S), EDGE_SHARE_MIN, EDGE_SHARE_MAX)
    eps = jnp.where(S > 0.0, S * e / (1.0 - e), 1.0) / n
    return jnp.where(fl, eps.astype(pr_r.dtype), p), fl


def mie_reference(lobe_params, medium, rows):
    """Return a Mie lobe's reference parameters with an edge forward ratio moved inside by EDGE_BUDGET."""
    if (f"mie:{medium}", 0) not in rows:
        return lobe_params
    r = lobe_params[MIE_RATIO]
    return dict(lobe_params, **{MIE_RATIO: jnp.clip(r, EDGE_BUDGET, 1.0 - EDGE_BUDGET)})
