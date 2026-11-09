"""A surface event: the surface type's outcome probabilities at a crossing, and the continuing outcome drawn there.

The surface type of the crossed pair (the legacy surface types, ``lumen.surfaces._legacy``; one ``lax.switch``
branch each, its parameters gathered by the pair's row) gives the ten outcomes' probabilities. The DETECT share is
deposited in expectation by the step; one continuing outcome is drawn from the reference probabilities and weighted
live / reference. Its direction: a deterministic map (straight, refraction, mirror, backscatter) with the Jones map of
the type's complex amplitudes, Geant4's facet chain for GLOSSY (``lumen.surfaces.facets``, held as a whole path and
weighted by its live / reference path density), a Lambertian lobe for DIFFUSE and DIFFUSE_T, and the type's
re-emission lobe for REEMIT. A layered surface is crossed one layer per vertex at the same point.

Notes
-----
Ported from the baseline's lumen.step (_kernel_probs, interface_event, _propagate, _chain_state); G4OpBoundaryProcess
(PostStepDoIt, DielectricMetal, DoReflection) for what each type means.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen._core.safe import literal, unit
from lumen.surfaces import facets, lobes, outcomes, polar
from lumen.surfaces.outcomes import DELTA, DETECT, DIFFUSE, DIFFUSE_T, FAR_SIDE, GLOSSY, N_OUTCOMES, REEMIT, STRAIGHT
from lumen.transport import moving_source as ms
from lumen.transport.estimator import magic, magic_log

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


def _amps8(kind, p, ctx):
    a = kind.amps(p, ctx) if kind.amps is not None else (*polar.MIRROR, 1.0 + 0.0j, 1.0 + 0.0j)
    a = [jnp.asarray(z, jnp.complex64) for z in a]
    return jnp.stack([f(z) for z in a for f in (jnp.real, jnp.imag)]).astype(F32)


def type_probs(tables, pk, k, row, ctx):
    """Return the TypeEval of surface type k (row ``row`` of its parameters) at ctx; k < 0 (never crossed): zeros."""

    def one(kind):
        def f(_):
            p = {}
            for name in kind.params:
                law = (tables.kernel_laws or {}).get((kind.name, name))
                v = pk[kind.name][name]
                r = jnp.maximum(row, 0)
                p[name] = v[r] if law is None else law({q: x[r] for q, x in v.items()}, ctx["lam"])
            lp = kind.lobe_params(p, ctx)
            return TypeEval(kind.probs(p, ctx).astype(F32), jnp.asarray(lp.get("sigma_alpha", DEFAULT_WIDTH), F32),
                            jnp.asarray(lp.get("tau", 0.0), F32), jnp.asarray(lp.get("film", False), jnp.bool_),
                            jnp.asarray(lp.get("coated", False), jnp.bool_), _amps8(kind, p, ctx))  # fmt: skip

        return f

    def none(_):
        return TypeEval(jnp.zeros(N_OUTCOMES, F32), F32(DEFAULT_WIDTH), F32(0.0), jnp.bool_(False), jnp.bool_(False),
                        literal(NO_AMPLITUDES))  # fmt: skip

    return jax.lax.switch(jnp.where(k >= 0, k + 1, 0), [none] + [one(kd) for kd in tables.kinds], None)


def chain_state(er, ei, wi, path, k):
    """Return the polarisation after a GLOSSY chain: a perfect mirror about each facet unit(W_j - W_j-1)."""
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
    tables, pk, kidx, row, ctx, (mu_tot, v_g, zero) = rule_ctx
    G, G1, G2 = jets

    def col(fs, k):
        return jnp.stack([f[k] for f in fs], -1)

    A, B = col(G1, "y"), col(G1, "d")
    CA, CB = (sg(col(G2, k)) for k in ("y", "d"))
    t_u = jnp.where(zero, 0.0, sg(col(G1, "t")))  # a zero-length segment in an opaque medium: none (step)

    def rule(wi_, n_, er_, ei_):
        out = type_probs(tables, pk, kidx, row, dict(ctx, wi=wi_, n=n_, er=er_, ei=ei_))
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
    """The drawn outcome's inputs: incoming direction, normal and state at the held end, the widths, uniforms."""

    wi: jax.Array
    n: jax.Array
    er: jax.Array
    ei: jax.Array
    width: dict
    width_ref: dict
    rand: jax.Array
    shared: bool
    moving: bool


def _no_sens():
    z = jnp.zeros((3, 2), F32)
    return (jnp.zeros(3, F32), z, z)


def _chain_branch(lobe, lb):
    def f(_):
        W, k = lobe.path(lb.width_ref, lb.wi, lb.n, lb.rand)
        W = sg(W)
        d_ = W[k - 1]
        lp = lobe.path_logpdf(lb.width, lb.wi, lb.n, W, k)
        w_ = magic_log(lp, sg(lp) if lb.shared else lobe.path_logpdf(lb.width_ref, lb.wi, lb.n, W, k))
        e_ = chain_state(lb.er, lb.ei, lb.wi, W, k)
        sens = ()
        if lb.moving:  # the chain's last direction is the one the next end holds
            g = sg(
                jax.jacfwd(lambda wl: lobe.path_logpdf(sg(lb.width), sg(lb.wi), sg(lb.n), W.at[k - 1].set(wl), k))(d_)
            )
            E = ms.sensitivity(lambda wl: chain_state(sg(lb.er), sg(lb.ei), sg(lb.wi), W.at[k - 1].set(wl), k), d_)
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
        w_ = magic_log(lobe.logpdf({}, lb.wi, lb.n, d_), lobe.logpdf({}, sg(lb.wi), sg(lb.n), d_))
        return (d_, w_, *state(d_), *sens)

    return f


def _by_type(kidx, branches):
    if len({id(b) for b, _ in branches}) == 1:
        return branches[0][1]
    return lambda _: jax.lax.switch(jnp.maximum(kidx, 0), [f for _, f in branches], None)


def outcome_direction(tables, kidx, o, cross, lb):
    """Return (direction, lobe weight, er, ei, *sensitivities) after outcome o (a ``lax.switch`` over the ten).

    ``cross`` = (wi, nrm, eta = n1 / n2, (er, ei, amps8)) at the crossing for the deterministic maps;
    ``lb`` (_Lobes) at the held end for the drawn ones.
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
            return (d_, one, *e_, *(_no_sens() if lb.moving else ()))

        return f

    def none_e(d_):
        return jnp.zeros(3, F32), jnp.zeros(3, F32)

    def facet_e(d_):  # Geant4's DoReflection: the implied facet
        return polar.mirror(lb.er, lb.ei, unit(d_ - lb.wi))

    br = [delta(k_) for k_ in range(N_OUTCOMES)]
    chains = [facets.CHAINS[kd.glossy] for kd in tables.kinds] or [facets.UNIFIED]
    br[GLOSSY] = _by_type(kidx, [(c, _chain_branch(c, lb)) for c in chains])
    br[DIFFUSE] = _lambert_branch(lobes.LAMBERT, facet_e, lb)
    br[DIFFUSE_T] = _lambert_branch(lobes.LAMBERT_T, none_e, lb)  # a diffuser: unpolarised
    rlobes = [kd.reemit_lobe or lobes.LAMBERT for kd in tables.kinds] or [lobes.LAMBERT]
    br[REEMIT] = _by_type(kidx, [(r, _lambert_branch(r, none_e, lb)) for r in rlobes])
    return jax.lax.switch(o, br, None)


def reemission(tables, kidx, o, lam_tau, rand, arrays_of):
    """Return (wavelength, delay) after outcome o: REEMIT draws them from the type's spectrum and Exp(tau).

    ``arrays_of(spectrum)``: its tables as operands (None: its own).
    """
    lam, tau = lam_tau
    if not any(kd.reemit is not None for kd in tables.kinds):
        return lam, F32(0.0)

    def remit(kd):
        return lambda _: (sg(kd.reemit.sample(rand[0], arrays_of(kd.reemit))), -tau * jnp.log(rand[1]))

    def keep(_):
        return lam, F32(0.0)

    lam_r, dt_r = jax.lax.switch(jnp.maximum(kidx, 0), [remit(kd) if kd.reemit is not None else keep
                                                        for kd in tables.kinds], None)  # fmt: skip
    return jnp.where(o == REEMIT, lam_r, lam), jnp.where(o == REEMIT, dt_r, 0.0)


def layer_after(tables, kidx, ok, film_coated, dirs, far):
    """Return (layer, far side) after a layered surface's outcome; other types keep layer 0 and their far side.

    Into the stack: the next layer (film, then backing); out of it: the film (from the backing, on the coated face) or
    out (from the film to the medium; the backing's transmission to the far side).
    """
    film, coated = film_coated
    d_new, nrm = dirs
    into = jnp.sum(d_new * nrm) < 0.0
    layer = jnp.where(film, jnp.where(into, 2, 0), jnp.where(~into & coated, 1, 0)).astype(jnp.int32)
    stk = jnp.asarray([kd.stack for kd in tables.kinds] or [False])[jnp.maximum(kidx, 0)] & (kidx >= 0)
    layer = jnp.where(stk & ok, layer, 0)
    return layer, jnp.where(stk, (layer == 0) & into, far)


def outcome_weight(vx, o, delta_o, edge):
    """Return the weight after outcome o: reached (live / reference), attenuated, times p_live / q (or the floor's)."""
    os_ = jnp.maximum(o, 0)
    w_surf = jnp.where(delta_o, vx.w * magic(vx.reach_live, vx.reach_ref) * vx.att_if,
                       vx.endw(vx.w * magic(vx.end.reach, vx.reach_ref) * vx.end.att))  # fmt: skip
    pr_o = jnp.where(delta_o, vx.pr[os_], vx.end.pr[os_])
    r_o = pr_o / jnp.maximum(sg(vx.qo[os_]), 1e-30)
    if edge:  # drawn at eps where p = 0: the photon as at p = eps, p / eps in z
        r_o = jnp.where(vx.fl[os_], vx.pr_q[os_] / jnp.maximum(sg(vx.qo[os_]), 1e-30), r_o)
    return w_surf * jnp.where(o >= 0, r_o, 0.0)


__all__ = ["DELTA", "DETECT", "FAR_SIDE", "STRAIGHT", "TypeEval", "type_probs"]
