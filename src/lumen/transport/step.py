"""One step of a photon: Geant4's optical physics between two vertices, with reweighting.

    ph2, dep = step(model, ops, ph, key)

The order of operations and the health counters: docs/dev/design.md section 8. Detection is expected, not sampled:
every crossing deposits the DETECT share times the probability of reaching it, and the photon goes on with a
continuing outcome or a bulk event. Distances are in m, times in ns; time advances with the medium's group velocity.
Geant4: G4OpBoundaryProcess, G4OpAbsorption, G4OpRayleigh, G4OpMieHG, G4OpWLS, G4Navigator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from lumen._core import rng
from lumen._core.safe import unit
from lumen.rules import navigation
from lumen.surfaces import facets, polar
from lumen.surfaces.outcomes import DELTA, DETECT, FAR_SIDE, STRAIGHT, TERMINAL
from lumen.transport import estimator, medium, state
from lumen.transport import moving_source as ms
from lumen.transport import surface_event as se
from lumen.units import C_VACUUM

sg = jax.lax.stop_gradient
F32, I32 = jnp.float32, jnp.int32
LAYOUT = rng.Layout(path=1, channel=1, outcome=1, lobe=3, roulette=1, reemit=2)
POLARISATION_STREAM = 0x9017  # the vertex key's fold that draws an unpolarised photon's linear state
W_MIN = 1e-9  # a photon below this weight retires (its expected contribution is below it)
ROULETTE_FLOOR = 1e-30  # the roulette weight's floor in its survival ratio (0: off, never divided by)
ZERO_STEP_BANDS = 0.125  # a crossing within this many bands of the start is a zero-length step (below 1 ulp)


@dataclass(frozen=True)
class StepConfig:
    """What the step is compiled for, derived from ``wrt``: the moving-source part, the edge rows."""

    moving_source: bool = False
    edge_rows: tuple = ()


class Controls(NamedTuple):
    """The step's values: ``t_max`` (m) a query's reach, ``t_end`` (ns) the time limit, ``roulette`` its weight."""

    t_max: Any = 1e3
    t_end: Any = float("inf")
    roulette: Any = 1e-4


@dataclass(frozen=True, eq=False)
class Model:
    """The compiled model: geometry (with its classes), surface and medium tables, step config, clearance field.

    The clearance field (``lumen.geometry.clearance.Clearance``) serves the moving-source method; ``hit_angle``: the
    deposit carries the hit angle (a sensor model reads CE).
    """

    geometry: Any
    tables: Any
    step: StepConfig = StepConfig()
    clearance: Any = None
    hit_angle: bool = False

    @property
    def stacked(self):
        """Whether a layered surface exists (the photon then carries ``Stack``)."""
        return any(kd.stack for kd in self.tables.kinds)

    @property
    def spectra(self):
        """The re-emission spectra of the media and surface types, in a fixed order (their tables are operands)."""
        found = [c.reemit for m in self.tables.media for c in m.channels] + [k.reemit for k in self.tables.kinds]
        return tuple({id(s): s for s in found if s is not None}.values())


class Operands(NamedTuple):
    """The traced inputs of the photon loop (docs/dev/design.md section 5); every field is an operand.

    ``nav``: the navigator's arrays (``Navigator.operands``); ``tables``: the facet, spectrum and clearance tables
    (``operands``).
    """

    params: Any
    reference: Any = None
    controls: Controls = Controls()
    nav: Any = None
    tables: Any = None
    grid: Any = None


def operands(model, params, reference=None, controls=None, grid=None):
    """Return the Operands of a model: every array the loop reads, as values (no recompile when they change)."""
    tables = dict(unified=facets.table_operands(), spectra=tuple(s.arrays() for s in model.spectra),
                  clearance=None if model.clearance is None else model.clearance.operands())  # fmt: skip
    ctl = Controls(*(jnp.asarray(v, F32) for v in (controls or Controls())))
    return Operands(params, reference, ctl, model.geometry.nav.operands(), tables, grid)


def geometry(model, ops):
    """Return the model's geometry reading the operands' navigator arrays."""
    return model.geometry.bind(ops.nav)


class Deposit(NamedTuple):
    """What one step deposits: the expected detection (sensor, weight, time ns, wavelength nm) and the health.

    ``pushed`` / ``abandon`` flag the stuck rule; ``event`` (-2 a bulk event, else the surface outcome, -1 none) and
    ``pair`` (class left x C + class met) are for diagnostics. ``hit`` (rad): the hit angle on the sensor
    (``hit_angle``; 0 when the model has none).
    """

    sensor: jax.Array
    q: jax.Array
    time: jax.Array
    lam: jax.Array
    lost: jax.Array
    abandoned: jax.Array
    escaped: jax.Array
    late: jax.Array
    pushed: jax.Array
    abandon: jax.Array
    event: jax.Array
    pair: jax.Array
    hit: jax.Array = 0.0


class _End(NamedTuple):
    """The segment's end as the outcome sees it, held by the moving-source method or as drawn.

    ``f`` weight factor; ``t`` distance (m); ``wi``, ``er``, ``ei`` direction and Jones vector; ``dt`` time shift (ns);
    ``pr`` outcome probabilities; ``reach`` exp(-mu t); ``att`` exp(-mu_abs t); ``n`` normal; ``jets`` (moving source).
    """

    f: Any
    t: Any
    wi: Any
    er: Any
    ei: Any
    dt: Any
    pr: Any
    reach: Any
    att: Any
    n: Any
    jets: Any = None


class _Free(NamedTuple):
    """The free path's end: weight factor, length s (m), direction, Jones vector, time shift dt (ns), position x (m)."""

    f: Any
    s: Any
    wi: Any
    er: Any
    ei: Any
    dt: Any
    x: Any


class Vertex(NamedTuple):
    """What one step knows before its event.

    Medium: ``mu_abs``, ``mu`` (scattering), ``mu_c`` (per channel), ``*_r`` at the reference (1/m), ``v_g`` (m/ns),
    ``mi`` the medium. ``s`` the free path (m). Crossing: ``q`` (the navigator's dict), ``t_if`` its distance (m),
    ``reach_live`` / ``reach_ref`` = exp(-mu t_if), ``att_if`` = exp(-mu_abs t_if). Surface: ``kidx`` the surface
    type, ``row``, ``cont`` (a continue event), ``te`` (live TypeEval), ``sigma_r``, ``pr_q`` (drawn from), ``fl``
    (floored), ``qo`` (the proposal), ``o`` the drawn outcome (-1 none), ``os_`` it clamped, ``drawn`` (o >= 0);
    ``ctx`` the crossing as the surface type reads it (lam, wi, n, n1, n2, er, ei: the state, filled for an
    unpolarised photon). ``u``: the step's uniforms. ``end``, ``free``, ``chain``: the moving-source parts.
    """

    u: dict
    mu_abs: Any
    mu_c: Any
    mu_c_r: Any
    mu: Any
    mu_r: Any
    v_g: Any
    mi: Any
    s: Any
    q: dict
    t_if: Any
    t_att: Any
    zero_ev: Any
    reach_live: Any
    reach_ref: Any
    att_if: Any
    kidx: Any
    row: Any
    cont: Any
    te: Any
    sigma_r: Any
    pr_q: Any
    fl: Any
    qo: Any
    o: Any
    os_: Any
    drawn: Any
    ctx: dict
    chain: Any
    scatter: Any
    end: _End | None = None
    free: _Free | None = None


def _recover(ph):
    """Return (photon, pushed, the band at x): G4Navigator's stuck-track recovery.

    After ZERO_STEPS_BEFORE_PUSH zero-length crossings the photon is pushed past the band and relocated.
    """
    b0 = navigation.step_band(sg(ph.x))
    stuck = ph.alive & (ph.nz >= navigation.ZERO_STEPS_BEFORE_PUSH)
    x = jnp.where(stuck, ph.x + navigation.PUSH_DISTANCE_TOLERANCES * b0 * sg(ph.d), ph.x)
    return ph._replace(x=x, vol=jnp.where(stuck, -1, ph.vol), srf=jnp.where(stuck, -1, ph.srf)), stuck, b0


def _query(model, ops, ph, cls0):
    """Return the next visible crossing (the navigator's dict); inside a layered surface, the remembered one."""
    geom, moving = geometry(model, ops), model.step.moving_source
    t_q = ops.controls.t_max

    def looked(_):
        r = geom.query(None, ph.x, ph.d, ph.cls, t_q, vol=ph.vol, last=ph.srf, sn=ph.sn)
        keys = ("t", "hit", "normal", "cls_after", "tag", "cls_before", "cls_start", "t_pass0", "t_pass1", "vol_after",
                "vol_before", "vol_start", "vol_reflect", "item", "cont") + (("win",) if moving else ())  # fmt: skip
        return {k: r[k] for k in keys}

    if not model.stacked:
        return looked(None)
    st, inf = ph.stack, F32(jnp.inf)

    def held(_):
        return dict(t=jnp.zeros_like(ph.t), hit=jnp.bool_(True), normal=ph.sn, cls_after=st.scls, tag=st.stag,
                    cls_before=cls0, cls_start=cls0, t_pass0=inf, t_pass1=inf, vol_after=st.svol, vol_before=ph.vol,
                    vol_start=ph.vol, vol_reflect=ph.vol, item=ph.srf, cont=jnp.bool_(False),
                    **(dict(win=jnp.asarray([-1, -1, 0, 1], I32)) if moving else {}))  # fmt: skip

    return jax.lax.cond(st.layer > 0, held, looked, None)


def _straight_if(cont, probs):
    """Return the outcome probabilities, a straight pass at a continue event (any type's row stands in)."""
    return jnp.where(cont, jax.nn.one_hot(STRAIGHT, probs.shape[-1], dtype=probs.dtype), probs)


def _surface(model, ops, q, ctx, shared):
    """Return (type, row, continue event, live TypeEval, reference probabilities, reference width) at the crossing."""
    tables = model.tables
    cb, c_k = q["cls_before"], q["cls_after"]
    kidx = jnp.asarray(tables.kind_of)[cb, c_k]
    row = jnp.asarray(tables.row_of)[cb, c_k]
    # a continue event (the query ran out of rounds on invisible crossings, or StepTooSmall): a straight pass, no
    # surface type, no deposit, never lost; any type's row stands in, overridden
    cont = q["cont"] & q["hit"]
    kidx, row = jnp.where(cont, jnp.maximum(kidx, 0), kidx), jnp.where(cont, jnp.maximum(row, 0), row)
    te = se.type_probs(tables, ops.params["kernel"], kidx, row, ctx)
    te = te._replace(probs=_straight_if(cont, te.probs), film=te.film & ~cont, coated=te.coated & ~cont)
    if shared:
        pr_r, sigma_r = sg(te.probs), sg(te.sigma_alpha)
    else:
        ctx_r = dict(ctx, n1=sg(ctx["n1"]), n2=sg(ctx["n2"]), er=sg(ctx["er"]), ei=sg(ctx["ei"]))
        kr = se.type_probs(tables, ops.reference["kernel"], kidx, row, ctx_r)
        pr_r, sigma_r = sg(kr.probs), sg(kr.sigma_alpha)
    return kidx, row, cont, te, _straight_if(cont, pr_r), sigma_r


def _held_end(model, ops, ph, vx):
    """Return the _End of the segment: held by the moving-source method (its jets), or as drawn."""
    q, o, ctx = vx.q, vx.os_, vx.ctx
    if not model.step.moving_source:
        return _End(1.0, vx.t_if, ph.d, ctx["er"], ctx["ei"], 0.0, vx.te.probs, vx.reach_live, vx.att_if, q["normal"])
    geom, chain, n1, n2 = geometry(model, ops), vx.chain, ctx["n1"], ctx["n2"]

    def seg(x_, d_):
        def lk(_):
            t_, n_ = geom.surface(None, q, x_, d_)
            return t_.astype(F32), n_.astype(F32)

        t_, n_ = jax.lax.cond(ph.stack.layer > 0, lambda _: (F32(0.0), ph.sn), lk, None) if model.stacked else lk(0)
        return dict(y=x_ + t_ * d_, t=t_, n=n_, d=unit(se.outcomes.direction(o, d_, n_, n1 / n2)))

    G, G1, G2 = ms.jet(seg, ph.x, ph.d, chain)
    held = ph.moving.free & q["hit"] & ph.alive & ((ph.stack.layer == 0) if model.stacked else True)
    du, f = ms.surface_end(chain, G["y"], q["normal"], [g["y"] for g in G1], [g["y"] for g in G2], held)
    t_c = ms.moved(vx.t_att, jnp.where(vx.zero_ev, 0.0, ms.col(G1, "t")), du)
    wi_c = ms.moved(ph.d, chain.B, du)
    er_c, ei_c = ms.moved(ctx["er"], chain.Er, du), ms.moved(ctx["ei"], chain.Ei, du)
    n_c = ms.moved(G["n"], ms.col(G1, "n"), du)  # the surface's normal at the HELD end
    pr_c = se.type_probs(model.tables, ops.params["kernel"], vx.kidx, vx.row,
                         dict(ctx, wi=wi_c, n=n_c, er=er_c, ei=ei_c)).probs  # fmt: skip
    m, ma = vx.mu, vx.mu_abs
    return _End(f, t_c, wi_c, er_c, ei_c, jnp.sum(sg(chain.ht) * du), _straight_if(vx.cont, pr_c), jnp.exp(-m * t_c),
                jnp.exp(-ma * t_c), n_c, (G, G1, G2))  # fmt: skip


def _free_end(model, ops, ph, vx):
    """Return the _Free of the free path: held in the volume by the moving-source method, or as drawn."""
    er0, ei0 = vx.ctx["er"], vx.ctx["ei"]
    if not model.step.moving_source:
        return _Free(1.0, vx.s, ph.d, er0, ei0, 0.0, ph.x + vx.s * ph.d)
    field = None if model.clearance is None else (lambda x: model.clearance(ops.tables["clearance"], x))
    return _Free(*ms.free_path_end(vx.chain, (ph.x, ph.d), vx.s, field, (er0, ei0), ph.moving.free & ph.alive))


def _vertex(model, ops, ph, key):
    """Return (photon after the stuck rule, pushed, the band at x, the Vertex) of one step, before its event."""
    tables, cfg = model.tables, model.step
    moving, shared = cfg.moving_source, ops.reference is None
    ph, pushed, b0 = _recover(ph)
    vkey = rng.vertex_key(key, ph.vertex)
    u = rng.draw(vkey, LAYOUT)
    er0, ei0 = polar.fill(ph.d, ph.er, ph.ei, jax.random.uniform(jax.random.fold_in(vkey, POLARISATION_STREAM)))
    cls0 = jnp.where(ph.cls < 0, -1 - ph.cls, ph.cls)  # < 0: not known past invisible crossings (located)
    mi = jnp.asarray(tables.medium_of)[cls0]
    pm, pmr = ops.params["media"], (ops.params if shared else ops.reference)["media"]
    n1, ng, mu_abs, mu_c = medium.medium_at(tables, pm, ph.lam, mi)
    mu_c_r = sg(mu_c) if shared else sg(medium.medium_at(tables, pmr, ph.lam, mi)[3])
    mu, mu_r = jnp.sum(mu_c), jnp.sum(mu_c_r)
    s = sg(-jnp.log(u["path"][0]) / jnp.maximum(mu_r, 1e-30))  # the free path at the reference event rate
    q = _query(model, ops, ph, cls0)
    t_if = q["t"]
    chain = ms.fresh_state(ph.moving.chain, ph.d, er0, ei0, polar.unpolarised(ph.er, ph.ei)) if moving else None
    # a zero-length event (the band snapped it to t = 0) in a medium opaque on the band's scale (mu b0 > 1): the value
    # took the sliver as length 0, so its attenuation and time carry no derivative
    zero_ev = ((q["win"][3] > 0) & q["hit"] & ((mu + mu_abs) * b0 > 1.0)) if moving else jnp.bool_(False)
    t_att = jnp.where(zero_ev, sg(t_if), t_if) if moving else t_if
    reach_live, reach_ref, att_if = jnp.exp(-mu * t_att), jnp.exp(-mu_r * sg(t_if)), jnp.exp(-mu_abs * t_att)
    n2 = medium.index_at(tables, pm, ph.lam, jnp.asarray(tables.medium_of)[q["cls_after"]])
    ctx = dict(lam=ph.lam, wi=ph.d, n=q["normal"], n1=n1, n2=n2, er=er0, ei=ei0)
    if model.stacked:
        ctx["layer"] = ph.stack.layer
    kidx, row, cont, te, pr_r, sigma_r = _surface(model, ops, q, ctx, shared)
    pr_q, fl = estimator.edge_floor(pr_r, kidx, row, tables, cfg.edge_rows) if cfg.edge_rows else (pr_r, None)
    qo = estimator.proposal(jnp.where(jnp.asarray(TERMINAL), 0.0, pr_q))
    o = estimator.choose(qo, u["outcome"][0])
    scatter = (mu_r > 0.0) & (s < sg(t_if)) & ph.alive
    vx = Vertex(u=u, mu_abs=mu_abs, mu_c=mu_c, mu_c_r=mu_c_r, mu=mu, mu_r=mu_r, v_g=C_VACUUM / ng, mi=mi, s=s, q=q,
                t_if=t_if, t_att=t_att, zero_ev=zero_ev, reach_live=reach_live, reach_ref=reach_ref, att_if=att_if,
                kidx=kidx, row=row, cont=cont, te=te, sigma_r=sigma_r, pr_q=pr_q, fl=fl, qo=qo, o=o,
                os_=jnp.maximum(o, 0), drawn=o >= 0, ctx=ctx, chain=chain, scatter=scatter)  # fmt: skip
    vx = vx._replace(end=_held_end(model, ops, ph, vx))
    return ph, pushed, b0, vx._replace(free=_free_end(model, ops, ph, vx))


def _surface_event(model, ops, ph, vx):
    """Return the Next of the continuing outcome drawn at the crossing (one ``lax.switch`` over the ten)."""
    tables, cfg, q, E = model.tables, model.step, vx.q, vx.end
    o, os_, moving, ctx = vx.o, vx.os_, cfg.moving_source, vx.ctx
    delta_o = jnp.asarray(DELTA)[os_]
    w_if = se.outcome_weight(ph.w, vx, delta_o, bool(cfg.edge_rows))
    nrm = q["normal"]
    width = {"tables": ops.tables["unified"]}
    lb = se._Lobes(E.wi, E.n, E.er, E.ei, {"sigma_alpha": vx.te.sigma_alpha, **width},
                   {"sigma_alpha": vx.sigma_r, **width}, vx.u["lobe"], ops.reference is None, moving)  # fmt: skip
    cross = (ph.d, nrm, ctx["n1"] / ctx["n2"], (ctx["er"], ctx["ei"], vx.te.amps8))
    d_new, w_lobe, er_n, ei_n, *sens = se.outcome_direction(tables, vx.kidx, os_, cross, lb)
    arrays_of = medium.spectrum_arrays(model, ops)
    lam2, dt = se.reemission(tables, vx.kidx, os_, (ph.lam, vx.te.tau), vx.u["reemit"], arrays_of)
    d_new = unit(d_new)  # no norm drift over many bounces
    far = jnp.asarray(FAR_SIDE)[os_] & vx.drawn
    ok = q["hit"] & vx.drawn & (vx.kidx >= 0)
    x_hit = ph.x + vx.t_if * ph.d
    layer = jnp.int32(0)
    if model.stacked:
        layer, far = se.layer_after(tables, vx.kidx, ok, (vx.te.film, vx.te.coated), (d_new, nrm), far)
    sn, chain, free, dtime = q["normal"], None, None, vx.t_if / vx.v_g + dt
    if moving:
        x_hit = jnp.where(delta_o, x_hit, sg(x_hit))  # a drawn outcome starts at the held end
        x_hit, d_new, w_lobe, er_n, ei_n, dtime, chain, free = _moved_outcome(
            model, ops, ph, vx, (x_hit, d_new, w_lobe, er_n, ei_n, dt, ok, layer, delta_o), sens
        )
        sn = jnp.where(delta_o, nrm, E.n)
    # a reflection at a relocation stays in the volume relocated into (vol_reflect), and so in its class
    stay = far | (q["vol_reflect"] != q["vol_before"])
    return state.Next(x_hit, d_new, w_if * w_lobe, jnp.where(stay, q["cls_after"], q["cls_before"]),
                      jnp.where(stay, q["tag"], ph.tag), ok, lam2, dtime, layer, sn.astype(F32), o.astype(I32), er_n,
                      ei_n, chain, free)  # fmt: skip


def _moved_outcome(model, ops, ph, vx, out, sens):
    """Return the outcome's state with the moving-source chain.

    Propagated through a deterministic outcome (and re-anchored), or started afresh at a drawn one.
    """
    x_hit, d_new, w_lobe, er_n, ei_n, dt, ok, layer, delta_o = out
    E, (G, G1, G2) = vx.end, vx.end.jets
    rule_ctx = (model.tables, ops.params["kernel"], vx.kidx, vx.row, vx.ctx, (vx.mu + vx.mu_abs, vx.v_g, vx.zero_ev))
    ch_d = se.propagate(vx.chain, (G, G1, G2), vx.os_, rule_ctx)
    ch_f = ms.free_start(x_hit, d_new, sens[0], E=(sens[1], sens[2]))
    free = jnp.where(delta_o, ph.moving.free, True)
    dtime = jnp.where(delta_o, vx.t_att / vx.v_g, E.dt + E.t / vx.v_g) + dt
    # the gauge of u re-anchored where the chain continues (value 0): the vertex just met held, every chain-carried
    # field moved with it, so the photon's own derivative stays O(1)
    dua, _ = ms.anchor(ch_d, x_hit, G["n"], delta_o & ph.moving.free & ok & (layer == 0))
    x_hit, d_new = ms.moved(x_hit, ch_d.A, dua), ms.moved(d_new, ch_d.B, dua)
    w_lobe = w_lobe * jnp.exp(jnp.sum(sg(ch_d.h) * dua))
    er_n, ei_n = ms.moved(er_n, ch_d.Er, dua), ms.moved(ei_n, ch_d.Ei, dua)
    dtime = dtime + jnp.sum(sg(ch_d.ht) * dua)
    chain = jax.tree_util.tree_map(lambda a, b: jnp.where(delta_o, a, b), ms.rebase(ch_d, dua), ch_f)
    return x_hit, d_new, w_lobe, er_n, ei_n, dtime, chain, free


def _bulk(model, ops, ph, vx):
    """Return the Next of a bulk event at the free-path end: the free path ends before the next surface.

    The channel is drawn from the reference rates (in Geant4's fixed order: Rayleigh, Mie, WLS) and weighted by the
    live / reference density of the free path in that channel (mu_c exp(-mu s)), times the absorption over the path
    (exp(-mu_abs s), a weight, never an event: G4OpAbsorption in expectation). The direction comes from the channel's
    lobe (``medium.channel_draw``). The photon's class there: before the first invisible crossing the start's, after
    the last the one before the visible crossing, between them unknown (-1 - class).
    """
    q, cb, s, F = vx.q, vx.q["cls_before"], vx.s, vx.free
    next_cls = jnp.where(s < q["t_pass0"], q["cls_start"], jnp.where(s > q["t_pass1"], cb, -1 - cb))
    cs = jnp.maximum(estimator.choose(vx.mu_c_r, vx.u["channel"][0]), 0)
    w = estimator.magic(vx.mu_c[cs] * jnp.exp(-vx.mu * F.s), vx.mu_c_r[cs] * jnp.exp(-vx.mu_r * s))
    w = w * jnp.exp(-vx.mu_abs * F.s) * F.f
    b = medium.channel_draw(model, ops, vx.mi, cs, F.wi, (F.er, F.ei), ph.lam, vx.u)
    chain, free = None, None
    if model.step.moving_source:
        chain, free = ms.free_start(F.x, b.wo, b.sens[0], E=b.sens[1:]), jnp.bool_(True)
    return state.Next(x=F.x, d=b.wo, w=ph.w * w * estimator.magic_log(b.lp, b.lpr) * b.yield_w, cls=next_cls,
                      tag=ph.tag, ok=jnp.bool_(True), lam=b.lam, dtime=F.dt + F.s / vx.v_g + b.delay,
                      layer=jnp.int32(0), sn=q["normal"].astype(F32), outcome=jnp.int32(-1), er=b.er, ei=b.ei,
                      chain=chain, free=free)  # fmt: skip


def _after(model, ops, ph, vx, nx, b0):
    """Return the state after the event's roulette, w_min, edge factor, stuck rule and time limit.

    (alive, weight, z, nz, abandon, abandoned weight, late weight).
    """
    q, scatter, ctl, os_ = vx.q, vx.scatter, ops.controls, vx.os_
    w_new = nx.w
    low = sg(w_new) < ctl.roulette  # Russian roulette: survive with probability w / roulette, unbiased
    p_s = sg(w_new) / jnp.maximum(ctl.roulette, ROULETTE_FLOOR)  # roulette 0 (off): no 0 / 0 in reverse mode
    w_new = jnp.where(low, jnp.where(vx.u["roulette"][0] < p_s, w_new / jnp.maximum(p_s, 1e-30), 0.0), w_new)
    alive = ph.alive & nx.ok & (sg(w_new) > W_MIN)
    z_new = 1.0
    if model.step.edge_rows:
        e_draw = (q["hit"] & ph.alive & (vx.kidx >= 0)) & ~scatter & vx.drawn & vx.fl[os_]
        pr_o = jnp.where(jnp.asarray(DELTA)[os_], vx.te.probs[os_], vx.end.pr[os_])
        z_new = ph.z * jnp.where(e_draw, pr_o / vx.pr_q[os_], 1.0)
        alive = alive & ~(e_draw & (sg(ph.z) == 0.0))  # a second such draw: 0 in value and derivative, retired
    layer0 = (ph.stack.layer == 0) if model.stacked else True
    nz_new = jnp.where(q["hit"] & ~scatter & layer0 & (sg(vx.t_if) <= ZERO_STEP_BANDS * b0), ph.nz + 1, 0).astype(I32)
    abandon = alive & (nz_new >= navigation.ZERO_STEPS_BEFORE_KILL)
    abandoned = jnp.where(abandon, w_new * z_new, 0.0)
    alive = alive & ~abandon
    in_time = sg(ph.t + nx.dtime) <= ctl.t_end  # past the time limit: it can never count
    late = jnp.where(alive & ~in_time, w_new * z_new, 0.0)
    return alive & in_time, w_new, z_new, nz_new, abandon, abandoned, late


def _advance(model, vx, nx):
    """Return (vol, srf, svol) after the event (the navigator's state for the next query)."""
    q = vx.q
    # a bulk event: the volume the free path ends in (before the first invisible crossing the start's, after the
    # last the one before the visible crossing, located between); a surface outcome that continues beyond takes the
    # volume after (a layered surface's far side: the remembered one), a reflection the one it was located in
    vol_b = jnp.where(vx.s < q["t_pass0"], q["vol_start"], jnp.where(vx.s > q["t_pass1"], q["vol_before"], -1))
    after = jnp.asarray(FAR_SIDE)[vx.os_] & vx.drawn
    if model.stacked:
        stk = se.is_stack(model.tables, vx.kidx)
        after = jnp.where(stk, (nx.layer == 0) & (jnp.sum(nx.d * q["normal"]) < 0.0), after)
    vol = jnp.where(vx.scatter, vol_b, jnp.where(after, q["vol_after"], q["vol_reflect"])).astype(I32)
    srf = jnp.where(vx.scatter | ~q["hit"], -1, q["item"]).astype(I32)
    return vol, srf, q["vol_after"].astype(I32)


def step(model, ops, ph, key):
    """Return (the photon after one step, its Deposit); ``key`` is the photon's key (``rng.photon_key``)."""
    cfg, tables = model.step, model.tables
    ph, pushed, b0, vx = _vertex(model, ops, ph, key)
    q, E, scatter = vx.q, vx.end, vx.scatter
    z = ph.z if cfg.edge_rows else 1.0  # deposits carry w z (the edge rows' factor; 1 without them)
    at_if = q["hit"] & ph.alive & (vx.kidx >= 0)
    dep_q = jnp.where(at_if, ph.w * E.reach * E.att * E.pr[DETECT] * E.f, 0.0) * z
    lost = jnp.where(q["hit"] & ph.alive & (vx.kidx < 0) & ~scatter,
                     ph.w * estimator.magic(E.reach, vx.reach_ref) * E.att * E.f, 0.0) * z  # fmt: skip
    nx = jax.lax.cond(scatter, lambda _: _bulk(model, ops, ph, vx), lambda _: _surface_event(model, ops, ph, vx), None)
    alive, w_new, z_new, nz_new, abandon, abandoned, late = _after(model, ops, ph, vx, nx, b0)
    vol, srf, svol = _advance(model, vx, nx)
    stack = state.Stack(nx.layer, svol, q["cls_after"].astype(I32), q["tag"].astype(I32)) if model.stacked else None
    mv = state.MovingSource(jnp.asarray(nx.free), jax.tree_util.tree_map(lambda a: a.astype(F32), nx.chain)) \
        if cfg.moving_source else None  # fmt: skip
    ph2 = state.Photon(nx.x, nx.d, ph.t + nx.dtime, nx.lam, jnp.where(alive, w_new, 0.0), alive, ph.vertex + 1,
                       nx.er.astype(F32), nx.ei.astype(F32), vol, srf, nx.sn, nz_new, nx.cls, nx.tag, stack, mv,
                       z_new if cfg.edge_rows else None)  # fmt: skip
    escaped = jnp.where(ph.alive & ~q["hit"] & ~scatter, ph.w * z, 0.0)  # met no surface within t_max, no bulk event
    cb, c_k = q["cls_before"], q["cls_after"]
    dpre = tables.deposit_pre  # static: a border sensor credits the region left
    tag_dep = q["tag"] if dpre is None else jnp.where(jnp.asarray(dpre)[cb, c_k], ph.tag, q["tag"])
    time = ph.t + E.dt + E.t / vx.v_g  # without the moving source dt is 0 and t the crossing's distance
    event, pair = jnp.where(scatter, -2, nx.outcome).astype(I32), cb * jnp.asarray(tables.kind_of).shape[0] + c_k
    out_side = False if dpre is None else jnp.asarray(dpre)[cb, c_k]
    hit = hit_angle(model, ops, ph, vx, out_side) if model.hit_angle else F32(0.0)
    dep = Deposit(jnp.where(at_if, tag_dep, -1), dep_q, time, ph.lam, lost, abandoned, escaped, late, pushed, abandon,
                  event, pair, hit)  # fmt: skip
    return ph2, dep


def hit_angle(model, ops, ph, vx, out_side):
    """Return the hit angle (rad) of a crossing into a sensor: acos(|z| / r) in the frame of the volume outside it.

    WCSim's definition (WCSimWCSD.cc ProcessHits, ProcessHits_boundary): into the sensor, the step's start point in
    the frame of the volume it starts in (the glass); out of it (``out_side``, credited to the volume left), the
    crossing point in the frame of the volume after (the glass). Evaluated as atan2(rho, |z|): finite derivatives on
    the axis.
    """
    q, D = vx.q, geometry(model, ops).nav.dev()
    v = jnp.where(out_side, q["vol_after"], q["vol_before"])
    x = jnp.where(out_side, ph.x + vx.t_if * ph.d, ph.x)
    r = D["R"][v].T @ (x - D["t"][v])
    return jnp.arctan2(jnp.sqrt(r[0] * r[0] + r[1] * r[1] + 1e-30), jnp.abs(r[2]))
