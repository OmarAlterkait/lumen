"""One step of a photon: Geant4's optical physics between two vertices, with the reweighting estimator.

    ph2, dep = step(model, ops, ph, key)

In order: draw the step's random numbers (``LAYOUT``, from the photon's key and vertex count) -> recover a stuck
photon (``lumen.rules.navigation``'s stuck rule) -> evaluate the medium at the live and reference values -> draw the
free path at the reference event rate -> the next visible crossing (the navigator) -> the moving-source holds (or the
photon as drawn) -> a bulk event if the free path ends first, else a surface event -> Russian roulette and the time
limit -> advance (``vol``, ``srf``, ``sn``, ``nz``) -> the deposit.

**Expected detection.** Detection is never sampled: at every crossing the DETECT share times the probability of
reaching the surface is deposited (Rao-Blackwellised over the free path), and the photon goes on with a continuing
outcome or a bulk event. Time advances with the medium's group index; the wavelength changes only at re-emission.

**Health.** ``lost``: a crossing no surface type joins (0 for a complete description); ``escaped``: no crossing within
``t_max`` and no bulk event; ``abandoned``: the stuck rule gave up (Geant4's fAbandonThreshold_NoZeroSteps);
``late``: retired past ``t_end``.

Notes
-----
Ported from the baseline's lumen.step at its mix 0, on the navigator, polarised: its defensive mix, absorb_sample,
roulette_vertex, polarised=False, the push contract, query_mfp and chart_r0 as an option are gone (SPEC section 8).
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
from lumen.transport.bulk_event import bulk_event
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
    """The static model: the geometry, the optics tables, the step config and the clearance field.

    The geometry carries its optical classes (``lumen.geometry._legacy_tracked``); the clearance field
    (``lumen.geometry.clearance.Clearance``) serves the moving-source method; ``hit_angle``: the deposit carries the
    hit angle (a sensor model reads CE).
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
    """The traced inputs (SPEC section 5): navigator arrays, float tables, parameters, reference, controls, grid.

    ``nav`` (``Navigator.operands``) and ``tables`` (``operands``) None: the model's own arrays, closed over.
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
    """Return the model's geometry reading the operands' navigator arrays (its own when there are none)."""
    return model.geometry if ops.nav is None else model.geometry.bind(ops.nav)


def spectrum_arrays(model, ops):
    """Return spectrum -> its tables (the operands', or None: the spectrum's own)."""
    if ops.tables is None:
        return lambda spec: None
    return lambda spec: ops.tables["spectra"][[id(s) for s in model.spectra].index(id(spec))]


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
    """The segment's end as the outcome sees it: held by the moving-source method, or as drawn."""

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
    """The free path's end: weight factor, length, direction, state, time shift, position."""

    f: Any
    s: Any
    wi: Any
    er: Any
    ei: Any
    dt: Any
    x: Any


class Vertex(NamedTuple):
    """Everything one step computed before its event."""

    u: dict
    w: Any
    er0: Any
    ei0: Any
    n1: Any
    n2: Any
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
    pr: Any
    te: Any
    sigma_r: Any
    pr_q: Any
    fl: Any
    qo: Any
    o: Any
    ctx: dict
    end: _End
    free: _Free
    chain: Any
    endw: Any


def _recover(ph):
    """Return (photon, pushed, the band at x): G4Navigator's stuck-track recovery.

    After ZERO_STEPS_BEFORE_PUSH zero-length crossings the photon is pushed past the band and relocated.
    """
    b0 = navigation.step_band(sg(ph.x))
    stuck = ph.alive & (ph.nz >= navigation.ZERO_STEPS_BEFORE_PUSH)
    x = jnp.where(stuck, ph.x + navigation.PUSH_DISTANCE_TOLERANCES * b0 * sg(ph.d), ph.x)
    return ph._replace(x=x, vol=jnp.where(stuck, -1, ph.vol), srf=jnp.where(stuck, -1, ph.srf)), stuck, b0


def _query(model, ops, ph, cls0, moving):
    """Return the next visible crossing (the baseline's query dict); inside a layered surface, the remembered one."""
    geom = geometry(model, ops)
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


def _surface(model, ops, ph, q, ctx, shared):
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
    one_straight = jax.nn.one_hot(STRAIGHT, te.probs.shape[-1], dtype=te.probs.dtype)
    te = te._replace(probs=jnp.where(cont, one_straight, te.probs), film=te.film & ~cont, coated=te.coated & ~cont)
    if shared:
        pr_r, sigma_r = sg(te.probs), sg(te.sigma_alpha)
    else:
        ctx_r = dict(ctx, n1=sg(ctx["n1"]), n2=sg(ctx["n2"]), er=sg(ctx["er"]), ei=sg(ctx["ei"]))
        kr = se.type_probs(tables, ops.reference["kernel"], kidx, row, ctx_r)
        pr_r, sigma_r = sg(kr.probs), sg(kr.sigma_alpha)
    return kidx, row, cont, te, jnp.where(cont, one_straight, pr_r), sigma_r


def _held_end(model, ops, ph, v0, chain):
    """Return the _End of the segment: held by the moving-source method (its jets), or as drawn."""
    q, o = v0["q"], jnp.maximum(v0["o"], 0)
    if not model.step.moving_source:
        return _End(1.0, v0["t_if"], ph.d, v0["er0"], v0["ei0"], 0.0, v0["te"].probs, v0["reach_live"], v0["att_if"],
                    q["normal"])  # fmt: skip
    geom, n1, n2 = geometry(model, ops), v0["ctx"]["n1"], v0["ctx"]["n2"]

    def seg(x_, d_):
        def lk(_):
            t_, n_ = geom.surface(None, q, x_, d_)
            return t_.astype(F32), n_.astype(F32)

        t_, n_ = jax.lax.cond(ph.stack.layer > 0, lambda _: (F32(0.0), ph.sn), lk, None) if model.stacked else lk(0)
        return dict(y=x_ + t_ * d_, t=t_, n=n_, d=unit(se.outcomes.direction(o, d_, n_, n1 / n2)))

    G, G1, G2 = ms.jet(seg, ph.x, ph.d, chain)

    def col(fs, k):
        return jnp.stack([f[k] for f in fs], -1)

    held = ph.moving.free & q["hit"] & ph.alive & ((ph.stack.layer == 0) if model.stacked else True)
    du, f = ms.surface_end(chain, G["y"], q["normal"], [g["y"] for g in G1], [g["y"] for g in G2], held)
    t_c = ms.moved(v0["t_att"], jnp.where(v0["zero_ev"], 0.0, col(G1, "t")), du)
    wi_c = ms.moved(ph.d, chain.B, du)
    er_c, ei_c = ms.moved(v0["er0"], chain.Er, du), ms.moved(v0["ei0"], chain.Ei, du)
    n_c = ms.moved(G["n"], col(G1, "n"), du)  # the surface's normal at the HELD end
    pr_c = se.type_probs(model.tables, ops.params["kernel"], v0["kidx"], v0["row"],
                         dict(v0["ctx"], wi=wi_c, n=n_c, er=er_c, ei=ei_c)).probs  # fmt: skip
    pr_c = jnp.where(v0["cont"], jax.nn.one_hot(STRAIGHT, pr_c.shape[-1], dtype=pr_c.dtype), pr_c)
    m, ma = v0["mu"], v0["mu_abs"]
    return _End(f, t_c, wi_c, er_c, ei_c, jnp.sum(sg(chain.ht) * du), pr_c, jnp.exp(-m * t_c), jnp.exp(-ma * t_c),
                n_c, (G, G1, G2))  # fmt: skip


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
    q = _query(model, ops, ph, cls0, moving)
    t_if = q["t"]
    chain = ph.moving.chain if moving else None
    if moving:
        chain = ms.fresh_state(chain, ph.d, er0, ei0, polar.unpolarised(ph.er, ph.ei))
    # a zero-length event (the band snapped it to t = 0) in a medium opaque on the band's scale (mu b0 > 1): the value
    # took the sliver as length 0, so its attenuation and time carry no derivative
    zero_ev = ((q["win"][3] > 0) & q["hit"] & ((mu + mu_abs) * b0 > 1.0)) if moving else jnp.bool_(False)
    t_att = jnp.where(zero_ev, sg(t_if), t_if) if moving else t_if
    reach_live, reach_ref, att_if = jnp.exp(-mu * t_att), jnp.exp(-mu_r * sg(t_if)), jnp.exp(-mu_abs * t_att)
    n2 = medium.index_at(tables, pm, ph.lam, jnp.asarray(tables.medium_of)[q["cls_after"]])
    ctx = dict(lam=ph.lam, wi=ph.d, n=q["normal"], n1=n1, n2=n2, er=er0, ei=ei0)
    if model.stacked:
        ctx["layer"] = ph.stack.layer
    kidx, row, cont, te, pr_r, sigma_r = _surface(model, ops, ph, q, ctx, shared)
    pr_q, fl = estimator.edge_floor(pr_r, kidx, row, tables, cfg.edge_rows) if cfg.edge_rows else (pr_r, None)
    qo = estimator.proposal(jnp.where(jnp.asarray(TERMINAL), 0.0, pr_q))
    o = estimator.choose(qo, u["outcome"][0])
    v0 = dict(q=q, o=o, t_if=t_if, er0=er0, ei0=ei0, te=te, reach_live=reach_live, att_if=att_if, ctx=ctx, kidx=kidx,
              row=row, cont=cont, t_att=t_att, zero_ev=zero_ev, mu=mu, mu_abs=mu_abs)  # fmt: skip
    end = _held_end(model, ops, ph, v0, chain)
    if moving:
        clear = ops.tables["clearance"] if ops.tables is not None else None
        field = None if model.clearance is None else (lambda x: model.clearance(clear, x))
        fp = ms.free_path_end(chain, (ph.x, ph.d), s, field, (er0, ei0), ph.moving.free & ph.alive)
        free = _Free(*fp)
    else:
        free = _Free(1.0, s, ph.d, er0, ei0, 0.0, ph.x + s * ph.d)
    endw = (lambda v: v * end.f) if moving else (lambda v: v)
    vx = Vertex(u, ph.w, er0, ei0, n1, n2, mu_abs, mu_c, mu_c_r, mu, mu_r, C_VACUUM / ng, mi, s, q, t_if, t_att,
                zero_ev, reach_live, reach_ref, att_if, kidx, row, cont, te.probs, te, sigma_r, pr_q, fl, qo, o, ctx,
                end, free,
               chain, endw)  # fmt: skip
    return ph, pushed, b0, vx


def _surface_event(model, ops, ph, vx):
    """Return the Next of the continuing outcome drawn at the crossing (one ``lax.switch`` over the ten)."""
    tables, cfg, q, E = model.tables, model.step, vx.q, vx.end
    o, moving = vx.o, cfg.moving_source
    os_ = jnp.maximum(o, 0)
    delta_o = jnp.asarray(DELTA)[os_]
    w_if = se.outcome_weight(vx, o, delta_o, bool(cfg.edge_rows))
    nrm = q["normal"]
    width = {} if ops.tables is None else {"tables": ops.tables["unified"]}
    lb = se._Lobes(E.wi, E.n, E.er, E.ei, {"sigma_alpha": vx.te.sigma_alpha, **width},
                   {"sigma_alpha": vx.sigma_r, **width}, vx.u["lobe"], ops.reference is None, moving)  # fmt: skip
    cross = (ph.d, nrm, vx.n1 / vx.n2, (vx.er0, vx.ei0, vx.te.amps8))
    d_new, w_lobe, er_n, ei_n, *sens = se.outcome_direction(tables, vx.kidx, os_, cross, lb)
    lam2, dt = se.reemission(tables, vx.kidx, os_, (ph.lam, vx.te.tau), vx.u["reemit"], spectrum_arrays(model, ops))
    d_new = unit(d_new)  # no norm drift over many bounces
    far = jnp.asarray(FAR_SIDE)[os_] & (o >= 0)
    ok = q["hit"] & (o >= 0) & (vx.kidx >= 0)
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
    ch_d = se.propagate(vx.chain, (G, G1, G2), jnp.maximum(vx.o, 0), rule_ctx)
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
    q, cb = vx.q, vx.q["cls_before"]
    s = vx.s
    next_cls = jnp.where(s < q["t_pass0"], q["cls_start"], jnp.where(s > q["t_pass1"], cb, -1 - cb))
    shared = ops.reference is None
    pmr = (ops.params if shared else ops.reference)["media"]
    ctx = (model.tables, ops.params["media"], pmr, vx.u, shared, model.step.moving_source, model.step.edge_rows,
           spectrum_arrays(model, ops))  # fmt: skip
    return bulk_event(ph, vx, next_cls, ctx)


def _after(model, ops, ph, vx, nx, b0):
    """Return the state after the event's roulette, w_min, edge factor, stuck rule and time limit.

    (alive, weight, z, nz, abandon, abandoned weight, late weight, scatter).
    """
    q, scatter, ctl = vx.q, (vx.mu_r > 0.0) & (vx.s < sg(vx.t_if)) & ph.alive, ops.controls
    w_new = nx.w
    low = sg(w_new) < ctl.roulette  # Russian roulette: survive with probability w / roulette, unbiased
    p_s = sg(w_new) / jnp.maximum(ctl.roulette, ROULETTE_FLOOR)  # roulette 0 (off): no 0 / 0 in reverse mode
    w_new = jnp.where(low, jnp.where(vx.u["roulette"][0] < p_s, w_new / jnp.maximum(p_s, 1e-30), 0.0), w_new)
    alive = ph.alive & nx.ok & (sg(w_new) > W_MIN)
    z_new = ph.z
    if model.step.edge_rows:
        os_ = jnp.maximum(vx.o, 0)
        e_draw = (q["hit"] & ph.alive & (vx.kidx >= 0)) & ~scatter & (vx.o >= 0) & vx.fl[os_]
        z_new = ph.z * jnp.where(e_draw, jnp.where(jnp.asarray(DELTA)[os_], vx.pr[os_], vx.end.pr[os_]) / vx.pr_q[os_],
                                 1.0)  # fmt: skip
        alive = alive & ~(e_draw & (sg(ph.z) == 0.0))  # a second such draw: 0 in value and derivative, retired
    wz = (lambda v: v * z_new) if model.step.edge_rows else (lambda v: v)
    layer0 = (ph.stack.layer == 0) if model.stacked else True
    nz_new = jnp.where(q["hit"] & ~scatter & layer0 & (sg(vx.t_if) <= ZERO_STEP_BANDS * b0), ph.nz + 1, 0).astype(I32)
    abandon = alive & (nz_new >= navigation.ZERO_STEPS_BEFORE_KILL)
    abandoned = jnp.where(abandon, wz(w_new), 0.0)
    alive = alive & ~abandon
    in_time = sg(ph.t + nx.dtime) <= ctl.t_end  # past the time limit: it can never count
    late = jnp.where(alive & ~in_time, wz(w_new), 0.0)
    return alive & in_time, w_new, z_new, nz_new, abandon, abandoned, late, scatter


def _advance(model, ph, vx, nx, scatter):
    """Return (vol, srf, svol) after the event (the navigator's state for the next query)."""
    q = vx.q
    # a bulk event: the volume the free path ends in (before the first invisible crossing the start's, after the
    # last the one before the visible crossing, located between); a surface outcome that continues beyond takes the
    # volume after (a layered surface's far side: the remembered one), a reflection the one it was located in
    vol_b = jnp.where(vx.s < q["t_pass0"], q["vol_start"], jnp.where(vx.s > q["t_pass1"], q["vol_before"], -1))
    os_ = jnp.maximum(vx.o, 0)
    after = jnp.asarray(FAR_SIDE)[os_] & (vx.o >= 0)
    if model.stacked:
        stk = jnp.asarray([kd.stack for kd in model.tables.kinds] or [False])[jnp.maximum(vx.kidx, 0)] & (vx.kidx >= 0)
        after = jnp.where(stk, (nx.layer == 0) & (jnp.sum(nx.d * q["normal"]) < 0.0), after)
    vol = jnp.where(scatter, vol_b, jnp.where(after, q["vol_after"], q["vol_reflect"])).astype(I32)
    srf = jnp.where(scatter | ~q["hit"], -1, q["item"]).astype(I32)
    return vol, srf, q["vol_after"].astype(I32)


def step(model, ops, ph, key):
    """Return (the photon after one step, its Deposit); ``key`` is the photon's key (``rng.photon_key``)."""
    cfg, tables = model.step, model.tables
    ph, pushed, b0, vx = _vertex(model, ops, ph, key)
    q = vx.q
    scatter = (vx.mu_r > 0.0) & (vx.s < sg(vx.t_if)) & ph.alive
    at_if = q["hit"] & ph.alive & (vx.kidx >= 0)
    dep_q = jnp.where(at_if & q["hit"], vx.endw(ph.w * vx.end.reach * vx.end.att * vx.end.pr[DETECT]), 0.0)
    lost = jnp.where(q["hit"] & ph.alive & (vx.kidx < 0) & ~scatter,
                     vx.endw(ph.w * estimator.magic(vx.end.reach, vx.reach_ref) * vx.end.att), 0.0)  # fmt: skip
    if cfg.edge_rows:  # deposits carry w z
        dep_q, lost = dep_q * ph.z, lost * ph.z
    nx = jax.lax.cond(scatter, lambda _: _bulk(model, ops, ph, vx), lambda _: _surface_event(model, ops, ph, vx), None)
    alive, w_new, z_new, nz_new, abandon, abandoned, late, scatter = _after(model, ops, ph, vx, nx, b0)
    vol, srf, svol = _advance(model, ph, vx, nx, scatter)
    stack = state.Stack(nx.layer, svol, q["cls_after"].astype(I32), q["tag"].astype(I32)) if model.stacked else None
    mv = state.MovingSource(jnp.asarray(nx.free), jax.tree_util.tree_map(lambda a: a.astype(F32), nx.chain)) \
        if cfg.moving_source else None  # fmt: skip
    ph2 = state.Photon(nx.x, nx.d, ph.t + nx.dtime, nx.lam, jnp.where(alive, w_new, 0.0), alive, ph.vertex + 1,
                       nx.er.astype(F32), nx.ei.astype(F32), vol, srf, nx.sn, nz_new, nx.cls, nx.tag, stack, mv,
                       z_new if cfg.edge_rows else None)  # fmt: skip
    wz = ph.w * ph.z if cfg.edge_rows else ph.w
    escaped = jnp.where(ph.alive & ~q["hit"] & ~scatter, wz, 0.0)  # met no surface within t_max and no bulk event
    cb, c_k = q["cls_before"], q["cls_after"]
    dpre = tables.deposit_pre  # static: a border sensor credits the region left
    tag_dep = q["tag"] if dpre is None else jnp.where(jnp.asarray(dpre)[cb, c_k], ph.tag, q["tag"])
    time = ph.t + vx.end.dt + vx.end.t / vx.v_g if cfg.moving_source else ph.t + vx.t_if / vx.v_g
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
