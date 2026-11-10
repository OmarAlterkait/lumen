"""A surface event: the surface type's outcome probabilities at a crossing, and the continuing outcome drawn there.

The surface type of the crossed pair (the legacy surface types, ``lumen.surfaces.types``; one ``lax.switch``
branch each, its parameters gathered by the pair's row) gives the ten outcomes' probabilities. The DETECT share is
deposited in expectation by the step; one continuing outcome is drawn from the reference probabilities and weighted
live / reference. Its direction: a deterministic map (straight, refraction, mirror, backscatter) with the Jones map of
the type's complex amplitudes, Geant4's facet path for GLOSSY (``lumen.surfaces.facets``, weighted as a whole path
by its live / reference path density), a Lambertian lobe for DIFFUSE and DIFFUSE_T, and the type's re-emission lobe
for REEMIT. A layered surface is crossed one layer per vertex at the same point. What each type means:
G4OpBoundaryProcess (PostStepDoIt, DielectricMetal, DoReflection).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen._core.safe import literal, unit
from lumen.surfaces import facets, lobes, outcomes, polar
from lumen.surfaces.outcomes import DELTA, DETECT, DIFFUSE, DIFFUSE_T, FAR_SIDE, GLOSSY, N_OUTCOMES, REEMIT, STRAIGHT
from lumen.transport import moving_source as ms
from lumen.transport.estimator import reweight, reweight_log
from lumen.transport.medium import reemit_draw

sg = jax.lax.stop_gradient
F32 = jnp.float32
DEFAULT_WIDTH = 0.1  # the GLOSSY lobe's width a type without one reports (never drawn)
NO_AMPLITUDES = (-1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)  # a perfect mirror's (r_s, r_p, t_s, t_p), (re, im) pairs


class TypeEval(NamedTuple):
    """A surface type evaluated at a crossing.

    Outcome probabilities (10,), the GLOSSY width, the REEMIT delay, the layered-surface flags (this vertex crosses
    the film; the photon's face is coated) and the amplitudes (8,).
    """

    probs: jax.Array
    sigma_alpha: jax.Array
    tau: jax.Array
    film: jax.Array
    coated: jax.Array
    amps8: jax.Array


def _amps8(stype, p, ctx):
    a = stype.amps(p, ctx) if stype.amps is not None else (*polar.MIRROR, 1.0 + 0.0j, 1.0 + 0.0j)
    a = [jnp.asarray(z, jnp.complex64) for z in a]
    return jnp.stack([f(z) for z in a for f in (jnp.real, jnp.imag)]).astype(F32)


def type_probs(tables, pk, k, row, ctx):
    """Return the TypeEval of surface type k (row ``row`` of its parameters) at ctx; k < 0 (never crossed): zeros."""

    def one(stype):
        def f(_):
            p = {}
            for name in stype.params:
                law = (tables.type_laws or {}).get((stype.name, name))
                v = pk[stype.name][name]
                r = jnp.maximum(row, 0)
                p[name] = v[r] if law is None else law({q: x[r] for q, x in v.items()}, ctx["lam"])
            lp = stype.lobe_params(p, ctx)
            return TypeEval(stype.probs(p, ctx).astype(F32), jnp.asarray(lp.get("sigma_alpha", DEFAULT_WIDTH), F32),
                            jnp.asarray(lp.get("tau", 0.0), F32), jnp.asarray(lp.get("film", False), jnp.bool_),
                            jnp.asarray(lp.get("coated", False), jnp.bool_), _amps8(stype, p, ctx))  # fmt: skip

        return f

    def none(_):
        return TypeEval(jnp.zeros(N_OUTCOMES, F32), F32(DEFAULT_WIDTH), F32(0.0), jnp.bool_(False), jnp.bool_(False),
                        literal(NO_AMPLITUDES))  # fmt: skip

    return jax.lax.switch(jnp.where(k >= 0, k + 1, 0), [none] + [one(kd) for kd in tables.surface_types], None)


def facet_path_state(er, ei, wi, path, k):
    """Return the polarisation after a GLOSSY facet path: a perfect mirror about each facet unit(W_j - W_j-1)."""
    prev = jnp.concatenate([wi[None], path[:-1]], 0)

    def body(c, x):
        e_r, e_i = c
        w1, w0, j = x
        m_r, m_i = polar.mirror(e_r, e_i, unit(w1 - w0))
        use = j < k
        return (jnp.where(use, m_r, e_r), jnp.where(use, m_i, e_i)), None

    (er, ei), _ = jax.lax.scan(body, (er, ei), (path, prev, jnp.arange(path.shape[0])))
    return er, ei


def _amplitudes(a8):
    return tuple(a8[2 * i] + 1j * a8[2 * i + 1] for i in range(4))


def propagate(ch, jets, o, rule_ctx):
    """Return the chain through a deterministic outcome o at the segment's end (``moving_source``).

    The ray after it from the jets of the segment function (G: y, t, n, d), the log weight's u-gradient (attenuation
    and the outcome's probability), the time's and the polarisation's (the outcome's Jones map), by forward mode.
    ``rule_ctx``: (tables, live surface params, type, row, ctx, (mu_total, v_g, zero-length)).
    """
    tables, pk, stype, row, ctx, (mu_tot, v_g, zero) = rule_ctx
    G, G1, G2 = jets
    A, B = ms.col(G1, "y"), ms.col(G1, "d")
    CA, CB = (sg(ms.col(G2, k)) for k in ("y", "d"))
    t_u = jnp.where(zero, 0.0, sg(ms.col(G1, "t")))  # a zero-length segment in an opaque medium: none (step)

    def rule(wi_, n_, er_, ei_):
        out = type_probs(tables, pk, stype, row, dict(ctx, wi=wi_, n=n_, er=er_, ei=ei_))
        lpo = jnp.log(jnp.maximum(out.probs[o], 1e-30))
        d_o = outcomes.direction(o, wi_, n_, ctx["n1"] / ctx["n2"])
        s_ax = polar.s_axis(wi_, n_, er_)
        comps = polar.components(er_, ei_, wi_, s_ax)
        maps = outcomes.jones_rules(comps, _amplitudes(out.amps8), s_ax, (er_, ei_), unit(d_o))
        e_r, e_i = jax.lax.switch(o, [lambda _, f=f: f() for f in maps], None)
        return lpo, e_r, e_i

    wi0, n0 = sg(ctx["wi"]), sg(G["n"])
    er0, ei0 = sg(ctx["er"]), sg(ctx["ei"])
    tans = [jax.jvp(rule, (wi0, n0, er0, ei0), (sg(ch.B[:, i]), sg(G1[i]["n"]), sg(ch.Er[:, i]), sg(ch.Ei[:, i])))[1]
            for i in range(2)]  # fmt: skip
    lp_u = jnp.stack([t[0] for t in tans])
    Er, Ei = (sg(jnp.stack([t[c] for t in tans], 1)) for c in (1, 2))
    h = sg(ch.h) - sg(mu_tot) * t_u + lp_u
    return ms.Chain(A, B, CA, CB, h, sg(ch.ht) + t_u / sg(v_g), Er, Ei, ch.xo)


# ---------------------------------------------------------------------------------------- the outcome's direction
class _Lobes(NamedTuple):
    """The drawn outcome's inputs: incoming direction, normal and state at the pinned end, the widths, uniforms."""

    wi: jax.Array
    n: jax.Array
    er: jax.Array
    ei: jax.Array
    width: dict
    width_ref: dict
    rand: jax.Array
    shared: bool
    moving: bool


def _facet_path_branch(lobe, lb):
    def f(_):
        W, k = lobe.path(lb.width_ref, lb.wi, lb.n, lb.rand)
        W = sg(W)
        d_ = W[k - 1]
        lp = lobe.path_logpdf(lb.width, lb.wi, lb.n, W, k)
        w_ = reweight_log(lp, sg(lp) if lb.shared else lobe.path_logpdf(lb.width_ref, lb.wi, lb.n, W, k))
        e_ = facet_path_state(lb.er, lb.ei, lb.wi, W, k)
        sens = ()
        if lb.moving:  # the facet path's last direction is the one the next end pins
            g = sg(
                jax.jacfwd(lambda wl: lobe.path_logpdf(sg(lb.width), sg(lb.wi), sg(lb.n), W.at[k - 1].set(wl), k))(d_)
            )
            E = ms.sensitivity(lambda wl: facet_path_state(sg(lb.er), sg(lb.ei), sg(lb.wi), W.at[k - 1].set(wl), k), d_)
            sens = (g, *E)
        return (d_, w_ * lobe.weight(lb.width, lb.wi, lb.n, d_), *e_, *sens)

    return f


def _lambert_branch(lobe, state, lb):
    def f(_):
        d_ = sg(lobe.sample({}, lb.wi, lb.n, lb.rand))
        sens = ()
        if lb.moving:
            g = sg(jax.jacfwd(lambda wo: lobe.logpdf({}, sg(lb.wi), sg(lb.n), wo))(d_))
            sens = (g, *ms.sensitivity(state, d_))
        w_ = reweight_log(lobe.logpdf({}, lb.wi, lb.n, d_), lobe.logpdf({}, sg(lb.wi), sg(lb.n), d_))
        return (d_, w_, *state(d_), *sens)

    return f


def _by_type(stype, branches):
    if len({id(b) for b, _ in branches}) == 1:
        return branches[0][1]
    return lambda _: jax.lax.switch(jnp.maximum(stype, 0), [f for _, f in branches], None)


def outcome_direction(tables, stype, o, cross, lb):
    """Return (direction, lobe weight, er, ei, *sensitivities) after outcome o (a ``lax.switch`` over the ten).

    ``cross`` = (wi, nrm, eta = n1 / n2, (er, ei, amps8)) at the crossing for the deterministic maps;
    ``lb`` (_Lobes) at the pinned end for the drawn ones.
    """
    wi, nrm, eta, (er0, ei0, a8) = cross
    s_ax = polar.s_axis(wi, nrm, er0)
    comps = polar.components(er0, ei0, wi, s_ax)
    amps = _amplitudes(a8)
    one = jnp.ones((), wi.dtype)

    def delta(k_):
        def f(_):
            d_ = outcomes.direction_rules(wi, nrm, eta)[k_]()
            e_ = outcomes.jones_rules(comps, amps, s_ax, (er0, ei0), unit(d_))[k_]()
            return (d_, one, *e_, *(ms.no_sensitivities() if lb.moving else ()))

        return f

    def none_e(d_):
        return jnp.zeros(3, F32), jnp.zeros(3, F32)

    def facet_e(d_):  # Geant4's DoReflection: the implied facet
        return polar.mirror(lb.er, lb.ei, unit(d_ - lb.wi))

    br = [delta(k_) for k_ in range(N_OUTCOMES)]
    paths = [facets.FACET_LOBES[kd.glossy] for kd in tables.surface_types] or [facets.UNIFIED]
    br[GLOSSY] = _by_type(stype, [(c, _facet_path_branch(c, lb)) for c in paths])
    br[DIFFUSE] = _lambert_branch(lobes.LAMBERT, facet_e, lb)
    br[DIFFUSE_T] = _lambert_branch(lobes.LAMBERT_T, none_e, lb)  # a diffuser: unpolarised
    rlobes = [kd.reemit_lobe or lobes.LAMBERT for kd in tables.surface_types] or [lobes.LAMBERT]
    br[REEMIT] = _by_type(stype, [(r, _lambert_branch(r, none_e, lb)) for r in rlobes])
    return jax.lax.switch(o, br, None)


def reemission(tables, stype, o, lam_tau, rand, arrays_of):
    """Return (wavelength, delay) after outcome o: REEMIT draws them from the type's spectrum and Exp(tau).

    ``arrays_of(spectrum)``: its tables as operands.
    """
    lam, tau = lam_tau
    if not any(kd.reemit is not None for kd in tables.surface_types):
        return lam, F32(0.0)

    def remit(kd):
        return lambda _: reemit_draw(kd.reemit, tau, rand, arrays_of)

    def keep(_):
        return lam, F32(0.0)

    lam_r, dt_r = jax.lax.switch(jnp.maximum(stype, 0), [remit(kd) if kd.reemit is not None else keep
                                                        for kd in tables.surface_types], None)  # fmt: skip
    return jnp.where(o == REEMIT, lam_r, lam), jnp.where(o == REEMIT, dt_r, 0.0)


def is_stack(tables, stype):
    """Return whether surface type stype is a layered one (a film over a backing); -1 (none): False."""
    return jnp.asarray([kd.stack for kd in tables.surface_types] or [False])[jnp.maximum(stype, 0)] & (stype >= 0)


def layer_after(tables, stype, ok, film_coated, dirs, far):
    """Return (layer, far side) after a layered surface's outcome; other types keep layer 0 and their far side.

    Into the stack: the next layer (film, then backing); out of it: the film (from the backing, on the coated face) or
    out (from the film to the medium; the backing's transmission to the far side).
    """
    film, coated = film_coated
    d_new, nrm = dirs
    into = jnp.sum(d_new * nrm) < 0.0
    layer = jnp.where(film, jnp.where(into, 2, 0), jnp.where(~into & coated, 1, 0)).astype(jnp.int32)
    stk = is_stack(tables, stype)
    layer = jnp.where(stk & ok, layer, 0)
    return layer, jnp.where(stk, (layer == 0) & into, far)


def outcome_weight(w, sc, delta_o, edge):
    """Return the weight after the drawn outcome: reached (live / reference), attenuated, times p_live / q.

    ``w`` the photon's weight; at an edge row a floored outcome takes the floor's ratio instead.
    """
    os_, E = sc.os_, sc.end
    w_surf = jnp.where(delta_o, w * reweight(sc.reach_live, sc.reach_ref) * sc.att_if,
                       w * reweight(E.reach, sc.reach_ref) * E.att * E.f)  # fmt: skip
    q_o = jnp.maximum(sg(sc.p_proposal[os_]), 1e-30)
    r_o = jnp.where(delta_o, sc.live_eval.probs[os_], E.pr[os_]) / q_o
    if edge:  # drawn at eps where p = 0: the photon as at p = eps, p / eps in z
        r_o = jnp.where(sc.floored[os_], sc.p_draw[os_] / q_o, r_o)
    return w_surf * jnp.where(sc.drawn, r_o, 0.0)


__all__ = ["DELTA", "DETECT", "FAR_SIDE", "STRAIGHT", "TypeEval", "type_probs"]
