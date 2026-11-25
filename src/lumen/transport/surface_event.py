"""A surface event: the surface row's outcome probabilities at a crossing, and the continuing outcome drawn there.

The crossed pair's row of the surface table (``lumen.build.lookup.SurfaceTable``: its columns gathered by the row and
read by the one kernel, ``lumen.surfaces.types.kernel``) gives the continuing shares and the detected share. The detect
deposit is made in expectation by the step; one continuing outcome is drawn from the reference probabilities and
weighted live / reference. Its direction: a deterministic map (straight, refraction, mirror, backscatter) with the Jones
map of the row's complex amplitudes, Geant4's facet path for GLOSSY (``lumen.surfaces.facets``, weighted as a whole path
by its live / reference path density; a dielectric one may end beyond the surface: ``Outcome.far``) and a Lambertian
lobe for DIFFUSE. What each type means: G4OpBoundaryProcess
(PostStepDoIt, DielectricMetal, DoReflection).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lumen import lobes
from lumen._core.safe import literal, ratio, safe_div, unit
from lumen.surfaces import facets, outcomes, polar, types
from lumen.surfaces.outcomes import DIFFUSE, GLOSSY, N_OUTCOMES
from lumen.transport.estimator import reweight_log

sg = jax.lax.stop_gradient
F32 = jnp.float32


def _per_row(a, r):
    """Return row r's entry of a per-row array of the table's structure: a constant when every row has the same."""
    a = np.asarray(a)
    if a.shape[0] == 0 or np.all(a == a[:1]):
        return jnp.asarray(a[0]) if a.shape[0] else jnp.asarray(0, a.dtype)
    return jnp.asarray(a)[r]


def _reader(tab, pk, r, lam):
    """Return get(column): row r's value of a table column at the wavelengths ``lam`` (``types.kernel``)."""

    def get(c):
        if c not in tab.params:  # no row reads it as a parameter and the rows' constant is one: a literal
            v = tab.consts[c][0]
            return literal(v) if np.ndim(v) else F32(v)
        v = pk[c]
        if c not in tab.laws:
            return v[r]
        row = {q: x[r] for q, x in v.items() if q != "const"}
        forms = tab.laws[c]
        if len(forms) == 1:
            x = forms[0](row, lam)
        else:
            x = jax.lax.switch(_per_row(tab.law_of[c], r), [lambda _, f=f: f(row, lam) for f in forms], None)
        reads = np.asarray([c in rd for rd in tab.reads])
        return x if reads.all() else jnp.where(_per_row(reads, r), x, v["const"][r])

    return get


def type_probs(tables, pk, row, ctx):
    """Return the TypeEval of the surface table's row ``row`` at ctx (``pk``: the packed columns); row < 0 (never
    crossed): zeros."""  # noqa: D205, D209
    tab = tables.table
    r = jnp.maximum(row, 0)
    f = types.Flags(*(_per_row(a, r) for a in tab.flags))
    ev = types.kernel(_reader(tab, pk, r, ctx["lam"]), f, ctx, tab.models, tab.any_per_normal)
    none = types.TypeEval(probs=jnp.zeros(N_OUTCOMES, F32), detect=F32(0.0),
                          sigma_alpha=F32(types.DEFAULT_WIDTH), amps8=literal(types.MIRROR_AMPS8),
                          inv_an=F32(1.0))  # fmt: skip
    return jax.tree_util.tree_map(lambda a, b: jnp.where(row >= 0, a, b), ev, none)


def _amplitudes(a8):
    return tuple(a8[2 * i] + 1j * a8[2 * i + 1] for i in range(4))


# ---------------------------------------------------------------------------------------- the outcome's direction
class Lobes(NamedTuple):
    """The drawn outcome's inputs: incoming direction, normal and state at the crossing, the widths, uniforms, and
    the crossing's live indices (n1, n2) (a dielectric facet path's)."""  # noqa: D205, D209

    wi: jax.Array
    n: jax.Array
    er: jax.Array
    ei: jax.Array
    width: dict
    width_ref: dict
    rand: jax.Array
    shared: bool
    idx: tuple


class Outcome(NamedTuple):
    """The drawn outcome's direction, lobe weight, drawn lobe weight (``outcome_direction``) and Jones vector; ``far``:
    its lobe left the photon beyond the surface (a dielectric facet path's last direction points there)."""  # noqa: D205, D209

    d: jax.Array
    w: jax.Array
    w_drawn: jax.Array
    er: jax.Array
    ei: jax.Array
    far: jax.Array


def _facet_path_branch(lobe, lb):
    def f(_):
        proposal = lobe.facet_proposal_logpdf  # a covered lobe: drawn from its mixture at the reference width

        def visit(c, w0, w1):
            """Add one facet's live and reference (or drawn-from) log density; mirror the polarisation."""
            lp, lr, er, ei, _, _ = c
            lp = lp + lobe.facet_logpdf(lb.width, w0, lb.n, w1)
            if proposal is not None:
                lr = lr + sg(proposal(lb.width_ref, sg(w0), lb.n, w1))
            elif not lb.shared:
                lr = lr + lobe.facet_logpdf(lb.width_ref, w0, lb.n, w1)
            m_r, m_i = polar.mirror(er, ei, unit(w1 - w0))
            return lp, lr, m_r, m_i, er, ei

        z = jnp.zeros((), lb.wi.dtype)
        c, w0, d_, _ = lobe.path(lb.width_ref, lb.wi, lb.n, lb.rand, visit, (z, z, lb.er, lb.ei, lb.er, lb.ei))
        lp, lr, er, ei, _, _ = c
        d_ = sg(d_)
        if proposal is None:
            w_ = reweight_log(lp, sg(lp) if lb.shared else lr)
            wd_ = jnp.ones_like(w_)  # drawn from the reference density: over itself, 1
        else:  # a draw that rounds off its support has weight 0
            on = lr > 0.5 * facets.LOG_TINY
            w_ = jnp.where(on, reweight_log(lp, jnp.where(on, lr, 0.0)), 0.0)
            wd_ = on.astype(w_.dtype)  # the density drawn from over itself: what the reference weight carries
        return Outcome(d=d_, w=w_, w_drawn=sg(wd_), er=er, ei=ei, far=jnp.bool_(False))

    return f


def _dielectric_path_branch(lobe, lb):
    def f(_):
        """Return the GLISUR dielectric facet path's Outcome: drawn at the reference width and at the live indices and
        state (held), weighted as a whole path, live over the density drawn from; its Jones vector the live one.

        The indices drawn at are the live ones, as a polished interface refracts (``outcome_direction``): a refracted
        direction is a map of the facet whose size is the index gap, so a direction drawn at other indices implies
        another facet, outside the reference ball once the index moves by a share of the gap (at matched indices at
        any move), and no weight corrects it.
        """
        proposal = lobe.facet_proposal_logpdf
        idx = (sg(lb.idx[0]), sg(lb.idx[1]))

        def visit(c, ev):
            lp, lr, er, ei, er_r, ei_r = c
            lp_k, _, er1, ei1 = lobe.facet_logpdf({**lb.width, "idx": lb.idx}, ev, (er, ei), lb.n)
            lp = lp + lp_k
            if proposal is not None or not lb.shared:
                q = proposal if proposal is not None else lobe.facet_logpdf
                ev_r = ev._replace(w0=sg(ev.w0))
                lr_k, _, er_r, ei_r = q({**lb.width_ref, "idx": idx}, ev_r, (er_r, ei_r), lb.n)
                lr = lr + sg(lr_k)
            return lp, lr, er1, ei1, er_r, ei_r

        z = jnp.zeros((), lb.wi.dtype)
        p_ref = {**lb.width_ref, "idx": idx, "er": lb.er, "ei": lb.ei}
        start = (z, z, lb.er, lb.ei, sg(lb.er), sg(lb.ei))
        c, d_, far, _ = lobe.path(p_ref, lb.wi, lb.n, lb.rand, visit, start)
        lp, lr, er, ei, _, _ = c
        d_ = sg(d_)
        if proposal is None:
            w_ = reweight_log(lp, sg(lp) if lb.shared else lr)
            wd_ = jnp.ones_like(w_)
        else:
            on = lr > 0.5 * facets.LOG_TINY
            w_ = jnp.where(on, reweight_log(lp, jnp.where(on, lr, 0.0)), 0.0)
            wd_ = on.astype(w_.dtype)
        return Outcome(d=d_, w=w_, w_drawn=sg(wd_), er=er, ei=ei, far=far)

    return f


def _lambert_branch(lobe, state, lb):
    def f(_):
        d_ = sg(lobe.sample({}, lb.wi, lb.n, lb.rand))
        w_ = reweight_log(lobe.logpdf({}, lb.wi, lb.n, d_), lobe.logpdf({}, sg(lb.wi), sg(lb.n), d_))
        er, ei = state(d_)
        return Outcome(d=d_, w=w_, w_drawn=jnp.ones_like(w_), er=er, ei=ei, far=jnp.bool_(False))

    return f


def _by_row(index, branches):
    """Return the branch of a row's lobe (``index``: the row's entry in the distinct lobes ``branches``)."""
    if len(branches) == 1:
        return branches[0]
    return lambda _: jax.lax.switch(index, branches, None)


def outcome_direction(tables, row, o, cross, lb):
    """Return the Outcome after outcome o (a ``lax.switch``).

    The drawn lobe weight is the weight at the reference values: the density drawn from over itself (what the reference
    weight carries): 1 for every lobe drawn from its reference density, the proposal's support indicator for a lobe
    drawn from its own proposal.

    ``cross`` = (wi, nrm, eta = n1 / n2, (er, ei, amps8)) at the crossing for the deterministic maps;
    ``lb`` (Lobes) for the drawn ones.
    """
    wi, nrm, eta, (er0, ei0, a8) = cross
    s_ax = polar.s_axis(wi, nrm)
    comps = polar.components(er0, ei0, wi, s_ax)
    amps = _amplitudes(a8)
    one = jnp.ones((), wi.dtype)

    def delta(k_):
        def f(_):
            d_ = outcomes.direction_rules(wi, nrm, eta)[k_]()
            e_ = outcomes.jones_rules(comps, amps, s_ax, (er0, ei0), unit(d_))[k_]()
            return Outcome(d=d_, w=one, w_drawn=one, er=e_[0], ei=e_[1], far=jnp.bool_(False))

        return f

    def none_e(d_):
        return jnp.zeros(3, F32), jnp.zeros(3, F32)

    def facet_e(d_):  # Geant4's DoReflection: the implied facet
        return polar.mirror(lb.er, lb.ei, unit(d_ - lb.wi))

    tab, r = tables.table, jnp.maximum(row, 0)
    br = [delta(k_) for k_ in range(N_OUTCOMES)]
    paths = [facets.FACET_LOBES[c] for c in tab.lobes]
    branch = [(_dielectric_path_branch if L.refracts else _facet_path_branch)(L, lb) for L in paths]
    br[GLOSSY] = _by_row(_per_row(tab.lobe_of, r), branch)
    br[DIFFUSE] = _lambert_branch(lobes.LAMBERT, facet_e, lb)
    return jax.lax.switch(o, br, None)


def outcome_weight(w, sc):
    """Return the weight ``w`` after the drawn outcome: reached (live / reference), attenuated, times p_live / q."""
    os_ = sc.os_
    w_surf = w * ratio(sc.reach_live, sc.reach_ref) * sc.att_if
    r_o = safe_div(sc.live_eval.probs[os_], sg(sc.p_proposal[os_]))  # a drawn outcome has q > 0 (estimator.choose)
    return w_surf * jnp.where(sc.drawn, r_o, 0.0)


__all__ = ["type_probs"]
