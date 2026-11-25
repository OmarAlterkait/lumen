"""One step of a photon: Geant4's optical physics between two vertices, with reweighting.

    ph2, dep = step(model, ops, ph, key)

The order of operations and the health counters: docs/dev/architecture.md, "Photon state, step and estimator". Detection
is expected, not sampled: every crossing deposits its detect deposit times the probability of reaching it, and the
photon goes on with a continuing outcome or a bulk event. Distances are in m, times in ns; time advances with the
medium's group velocity. Geant4: G4OpBoundaryProcess, G4OpAbsorption, G4OpRayleigh, G4OpMieHG, G4OpWLS, G4Navigator.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lumen import speed
from lumen._core import rng
from lumen._core.rng import JUMP_STREAM, POLARISATION_STREAM
from lumen._core.safe import ratio, safe_div, unit
from lumen._core.static import Static
from lumen.record import VECTOR_FIELDS
from lumen.rules import navigation
from lumen.rules.choices import FLOOR32, ROULETTE
from lumen.surfaces import facets, polar
from lumen.surfaces.outcomes import FAR_SIDE, STRAIGHT
from lumen.transport import estimator, fastpath, medium, state
from lumen.transport import surface_event as se
from lumen.units import C_VACUUM

sg = jax.lax.stop_gradient
F32, I32 = jnp.float32, jnp.int32
LAYOUT = rng.Layout(path=1, channel=1, outcome=1, lobe=rng.LOBE_UNIFORMS, roulette=1, reemit=rng.REEMIT_UNIFORMS)


@dataclass(frozen=True)
class StepConfig:
    """What the step is compiled for: the edge rows (the detector's), the speed levers on (a frozenset of names,
    ``lumen.speed``; the default, ``speed.EXACT``: none, the reference code).

    ``moving_source``: the switch of a source-position derivative's method (a separately compiled step carrying its
    per-photon state in ``Photon.moving``). No method is in this version: it stays False, and True is refused
    (``operands``); the switch and the slot are kept so the method can return as a module of its own.
    """  # noqa: D205

    edge_rows: estimator.EdgeRows = estimator.EdgeRows()
    levers: Any = speed.EXACT
    moving_source: bool = False


class Controls(NamedTuple):
    """The step's values: ``max_distance`` (m) a query's reach, ``roulette``.

    ``max_distance`` None (the default): the geometry's own reach, the diagonal of the box holding every volume
    (``Model.reach``), past which no surface lies; a number overrides it. There is no time cut: every photon is
    followed until it ends (the roulette, absorption, escape, the stuck rule or the vertex guard).

    ``roulette`` (ε > 0, default ``ROULETTE``): the photon's weight is relative to its emission row (1 at the start).
    While its reference weight (``Photon.wr``, the weight at the reference values) is at least ε it carries that
    weight as it falls; below ε it plays Geant4's survive-or-absorb lottery: it survives with probability wr / ε and
    then carries the reference weight ε (both weights divided by that probability), else it is absorbed (weight 0).
    ε = 1 is analog transport, as Geant4 does it, and ε -> 0 fully weighted.
    Unbiased at any ε, for values and derivatives (one ε for both): it trades run time against noise, never the
    expectation. The decision reads only the reference, so with the reference held a run is smooth in the live values.
    A photon ends only so (or at a terminal outcome, the stuck rule). A run knob, not a detector
    property (docs/explanation/known-limits.md, "The roulette's ε").
    """

    max_distance: Any = None
    roulette: Any = ROULETTE


@dataclass(frozen=True, eq=False)
class Model:
    """The compiled model: geometry (with its classes), surface and medium tables, step config.

    ``local``: the deposit fields beyond sensor, weight, time and wavelength it carries (a subset of ``pos``, ``dir``,
    ``normal``, ``gen``, ``inv_an``: what the run's sensor features and readout stages read,
    ``lumen.sensors.features``; ``gen`` also makes the photon count its re-emissions); ``analog``: the photon carries
    its analog record (``Photon.analog``, truth rows) and the deposit its reached share.
    """

    geometry: Any
    tables: Any
    step: StepConfig = StepConfig()
    local: tuple = ()
    analog: bool = False
    jump: Any = None  # the diffusion jump's data (lumen.transport.fastpath.Jump), with the speed lever "jump" only

    def __post_init__(self):
        """Fix the model's ``static``: the jitted functions' static argument (``lumen._core.static.Static``), equal for
        two models of one structure (two loads of a detector share their compiled programs) and holding the model
        weakly (jax keeps its static arguments for good: a model's arrays go when the model goes)."""  # noqa: D205, D209
        object.__setattr__(self, "static", Static(self))

    @functools.cached_property
    def reach(self) -> float:
        """The farthest a straight path goes inside the geometry (m): the diagonal of the box holding every volume."""
        nav = self.geometry.nav
        lo, hi = np.min(np.asarray(nav.lo_w, np.float64), 0), np.max(np.asarray(nav.hi_w, np.float64), 0)
        return float(np.linalg.norm(hi - lo))

    @functools.cached_property
    def device_tables(self) -> dict:
        """The model's own tables (the facets') as device arrays: moved once (``operands``)."""
        with jax.ensure_compile_time_eval():  # device arrays, also when first asked for under a trace
            out = dict(unified=facets.table_operands())
            if "jump" in self.step.levers:  # the jump's clearance grids (lumen.speed)
                out["diff"] = fastpath.operands(self.jump)
            return out


class Operands(NamedTuple):
    """The traced inputs of the photon loop (docs/dev/architecture.md, "Data flow and signatures"); every operand.

    ``nav``: the navigator's arrays (``Navigator.operands``); ``tables``: the facet tables
    (``operands``).
    """

    params: Any
    reference: Any = None
    controls: Controls = Controls()
    nav: Any = None
    tables: Any = None


def operands(model, params, reference, controls):
    """Return the Operands of a model: every array the loop reads, as values (no recompile when they change).

    ``reference``: the packed values every draw is made at (None: ``params``, the run's own).
    ``controls``: a Controls whose roulette is a number ε > 0 (a max_distance of None is the model's reach; a traced
    roulette is counted by ``simulate``'s ``health.invalid`` where it is not > 0). The model's own tables are fixed
    with it (``Model.device_tables``).
    """
    if model.step.moving_source:
        raise NotImplementedError("StepConfig.moving_source: a source-position derivative is not available in this "
                                  "version (docs/explanation/known-limits.md)")  # fmt: skip
    traced = isinstance(controls.roulette, jax.core.Tracer)
    if controls.roulette is None or not (traced or float(controls.roulette) > 0.0):
        raise ValueError(f"operands: the roulette is a number ε > 0 (a photon ends only at weight 0), got "
                         f"{controls.roulette!r}")  # fmt: skip
    if reference is not None:
        reference = jax.tree_util.tree_map(sg, reference)
    c = controls if controls.max_distance is not None else controls._replace(max_distance=model.reach)
    ctl = Controls(*(jnp.asarray(v, F32) for v in c))
    return Operands(params, reference, ctl, model.geometry.nav.operands(), model.device_tables)


def geometry(model, ops):
    """Return the model's geometry reading the operands' navigator arrays."""
    return model.geometry.bind(ops.nav)


class Deposit(NamedTuple):
    """What one step deposits: the expected detection (sensor, weight, time ns, wavelength nm) and the health.

    ``event`` (-2 a bulk event, else the surface outcome, -1
    none) and ``pair`` (class left x C + class met) are for diagnostics. In the frame of the sensor's placement
    (``sensor_frame``): ``pos`` (3,) m the crossing point, ``dir`` (3,) the photon's direction, ``normal`` (3,) the
    surface normal there; ``gen`` the photon's re-emission count (``Photon.gen``): each carried when ``Model.local``
    names it, else None (the facts transport knows at the sensor; a sensor's features are readout, never computed
    here). ``inv_an``: 1 / A_n, the crossed surface's absorptance at normal incidence at the photon's wavelength (a
    surface type read per normal incidence, ``lumen.surfaces.types.per_normal``; 1 elsewhere), carried when a
    readout stage reads it (``Model.local``): a sensor model whose QE is the detection at normal incidence divides
    the deposit by A_n (``lumen.readout.grid.Grid.normalise``). ``label``, the photon's emission label
    (``Photons.label``, set by the loop when its rows carry one) and ``share``, the reached share (``Model.analog``:
    the deposit's weight given that the tracked photon reaches the crossing, w att p_detect f, 0 when it scatters
    first), are None when off, so the loop's other outputs do not change (docs/dev/feature_interfaces.md). ``q`` is the
    physical detected share (the surface's detect deposit: what arrivals count); ``share`` the reached share of the
    analog law. The weight accounting (``lumen.api.Health``): ``distance_cut``, ``absorbed_bulk``,
    ``absorbed_surface`` (``state.Next.fates``), ``rouletted`` (the roulette's net) and ``reweighted`` (the event's
    expected continuing weight minus the weight it goes on with: the drawn direction's and outcome's live / proposal
    factors, 0 in expectation).
    """

    sensor: jax.Array
    q: jax.Array
    time: jax.Array
    lam: jax.Array
    lost: jax.Array
    abandoned: jax.Array
    escaped: jax.Array
    distance_cut: jax.Array
    absorbed_bulk: jax.Array
    absorbed_surface: jax.Array
    rouletted: jax.Array
    reweighted: jax.Array
    event: jax.Array
    pair: jax.Array
    label: jax.Array | None = None
    share: jax.Array | None = None
    pos: jax.Array | None = None
    dir: jax.Array | None = None
    normal: jax.Array | None = None
    gen: jax.Array | None = None
    inv_an: jax.Array | None = None


class StepContext(NamedTuple):
    """What one step knows before its event.

    Medium: ``mu_abs``, ``mu`` (scattering), ``mu_c`` (per channel), ``*_r`` at the reference (1/m, stop-gradient),
    ``v_g`` the group velocity (m/ns), ``medium_idx`` the medium. ``s`` the free path
    (m). Crossing: ``q`` (the navigator's dict), ``dist`` its distance (m), ``reach_live`` / ``reach_ref`` =
    exp(-mu dist), ``att_if`` / ``att_r`` = exp(-mu_abs dist) live / at the reference. Surface: ``row`` the surface
    table's row (-1 none), ``cont`` (a continue event), ``live_eval`` (live TypeEval), ``sigma_r``, ``p_draw`` (drawn
    from), ``p_proposal`` (the proposal), ``o`` the drawn outcome (-1 none), ``os_`` it clamped, ``drawn`` (o >= 0);
    ``ctx`` the crossing as the surface kernel reads it (lam, wi, n, n1, n2, er, ei: the state, filled for an
    unpolarised photon). ``rand``: the step's uniforms; ``scatter``: the free path ends before the crossing.
    """

    rand: dict
    mu_abs: Any
    mu_abs_r: Any
    mu_c: Any
    mu_c_r: Any
    mu: Any
    mu_r: Any
    v_g: Any
    medium_idx: Any
    s: Any
    q: dict
    dist: Any
    reach_live: Any
    reach_ref: Any
    att_if: Any
    att_r: Any
    row: Any
    cont: Any
    live_eval: Any
    sigma_r: Any
    p_draw: Any
    p_proposal: Any
    o: Any
    os_: Any
    drawn: Any
    ctx: dict
    scatter: Any


def _recover(model, ops, ph):
    """Return (photon, the band at x): G4Navigator's stuck-track recovery.

    After ZERO_STEPS_BEFORE_PUSH zero-length crossings the photon is pushed past the band and located there: its volume
    and its class (the medium the step reads) are those of the point it lands on.
    """
    b0 = navigation.step_band(sg(ph.x))
    stuck = ph.alive & (ph.nz >= navigation.ZERO_STEPS_BEFORE_PUSH)
    x = jnp.where(stuck, ph.x + navigation.PUSH_DISTANCE_TOLERANCES * b0 * sg(ph.d), ph.x)
    nav = geometry(model, ops).nav
    vol = nav.locate(sg(x), start=jnp.where(stuck, -1, ph.vol)).astype(I32)  # the descent only where pushed
    cls = jnp.where(stuck, nav.dev()["cls"][vol], ph.cls).astype(I32)
    return ph._replace(x=x, vol=vol, cls=cls, srf=jnp.where(stuck, -1, ph.srf)), b0


def _query(model, ops, ph, s):
    """Return the next visible crossing (the navigator's dict).

    ``s``: the free path; ``vol_at`` / ``cls_at`` are the volume and class the ray is in there.
    """
    geom = geometry(model, ops)
    # a geometry that accepts the free path may stop the query just past it where the crossing beyond cannot deposit
    # (lumen.geometry.reach)
    sf = dict(s_free=s) if getattr(geom, "accepts_s_free", False) else {}
    r = geom.query(None, ph.x, ph.d, ph.cls, ops.controls.max_distance, vol=ph.vol, last=ph.srf, sn=ph.sn, s_at=s, **sf)
    keys = ("t", "hit", "normal", "cls_after", "sensor", "cls_before", "cls_start", "t_pass0", "t_pass1",
            "vol_after", "vol_before", "vol_start", "item", "cont", "vol_at", "cls_at")  # fmt: skip
    return {k: r[k] for k in keys}


def _straight_if(cont, probs):
    """Return the outcome probabilities, a straight pass at a continue event (the first row stands in)."""
    return jnp.where(cont, jax.nn.one_hot(STRAIGHT, probs.shape[-1], dtype=probs.dtype), probs)


def _surface(model, q, zero):
    """Return (row, continue event) at the crossing; ``zero``: a zero-length step (``ZERO_STEP_BANDS``).

    A continue event (zero-length invisible crossings in a row, the stuck rule's case, or StepTooSmall): a straight
    pass, no surface type, no deposit, never lost; the first row stands in, overridden. Geant4's StepTooSmall is a
    step <= kCarTolerance (1e-12 m: G4OpBoundaryProcess::PostStepDoIt returns before it reads a material or surface).
    lumen's zero-length step is one below ZERO_STEP_BANDS bands, microns at a detector's scale: there Geant4 takes a
    real step and applies the surface, so only a zero-length step out of a class with no surface type (a photon a
    reflection left in a medium without RINDEX, at the face it reflected off) is StepTooSmall here. Applied to every
    zero-length step it skipped surfaces Geant4 applies (measured).
    """
    tables = model.tables
    cb, c_k = q["cls_before"], q["cls_after"]
    row = jnp.asarray(tables.row_of)[cb, c_k]
    cont = (q["cont"] | (zero & (row < 0))) & q["hit"]
    return jnp.where(cont, jnp.maximum(row, 0), row), cont


def _type_eval(model, pk, surf, ctx):
    """Return the TypeEval of the crossing's surface row ``surf`` = (row, cont) at the packed values ``pk``."""
    row, cont = surf
    ev = se.type_probs(model.tables, pk["surfaces"], row, ctx)
    return ev._replace(probs=_straight_if(cont, ev.probs), detect=jnp.where(cont, 0.0, ev.detect))


def _crossing(model, pk, ph, q, surf, n1, pol):  # noqa: PLR0913 -- the crossing's inputs
    """Return (the crossing as the surface kernel reads it, its TypeEval) at the packed values ``pk``.

    ``n1``: the index of the medium the photon is in at those values; ``pol`` = (er, ei) its filled state. The one
    function serves the live values and the reference, so the reference's crossing has every value-dependent entry at
    the reference by construction.
    """
    mi_after = jnp.asarray(model.tables.medium_of)[q["cls_after"]]
    n2 = medium.index_at(model.tables, pk["media"], ph.lam, mi_after)
    ctx = dict(lam=ph.lam, wi=ph.d, n=q["normal"], n1=n1, n2=n2, er=pol[0], ei=pol[1])
    return ctx, _type_eval(model, pk, surf, ctx)


def _step_context(model, ops, ph, key):
    """Return (photon after the stuck rule, the band at x, the StepContext) of one step, before its event.

    The medium and the crossing's surface are evaluated at the live values and, when the run has a reference of its
    own, at the reference by the same functions (stop-gradient): the draws are made at the reference, and the
    reference weight and time are closed forms in those values (``_event``).
    """
    tables, cfg = model.tables, model.step
    shared = ops.reference is None
    ph, b0 = _recover(model, ops, ph)
    vkey = rng.vertex_key(key, ph.vertex)
    rand = rng.draw(vkey, LAYOUT)
    if "jump" in cfg.levers:  # the walk's own key (lumen.transport.fastpath): a fold of the vertex key, no draw shifted
        rand = dict(rand, jkey=jax.random.fold_in(vkey, JUMP_STREAM))
    er0, ei0 = polar.fill(ph.d, ph.er, ph.ei, rng.uniforms(jax.random.fold_in(vkey, POLARISATION_STREAM), 1)[0])
    medium_idx = jnp.asarray(tables.medium_of)[ph.cls]
    n1, ng, mu_abs, mu_c = medium.medium_at(tables, ops.params["media"], ph.lam, medium_idx)
    ref_m = (n1, ng, mu_abs, mu_c) if shared else medium.medium_at(tables, ops.reference["media"], ph.lam, medium_idx)
    n1_r, _, mu_abs_r, mu_c_r = (sg(v) for v in ref_m)
    mu_r = jnp.sum(mu_c_r)
    s = sg(-jnp.log(rand["path"][0]) / jnp.maximum(mu_r, FLOOR32))  # the free path at the reference event rate
    q = _query(model, ops, ph, s)
    dist = q["t"]
    zero = sg(dist) <= navigation.ZERO_STEP_BANDS * b0  # a zero-length step: below one float32 ulp of the coordinates
    surf = _surface(model, q, zero)
    row, cont = surf
    ctx, live_eval = _crossing(model, ops.params, ph, q, surf, n1, (er0, ei0))
    ref_eval = live_eval if shared else _crossing(model, ops.reference, ph, q, surf, n1_r, (er0, ei0))[1]
    ref_eval = jax.tree_util.tree_map(sg, ref_eval)
    pr_r = ref_eval.probs
    p_draw = estimator.edge_floor(pr_r, row, tables, cfg.edge_rows) if cfg.edge_rows else pr_r
    p_proposal = estimator.proposal(p_draw)
    o = estimator.choose(p_proposal, rand["outcome"][0])
    mu = jnp.sum(mu_c)
    sc = StepContext(rand=rand, mu_abs=mu_abs, mu_abs_r=mu_abs_r, mu_c=mu_c, mu_c_r=mu_c_r, mu=mu, mu_r=mu_r,
                     v_g=C_VACUUM / ng, medium_idx=medium_idx, s=s, q=q, dist=dist,
                     reach_live=jnp.exp(-mu * dist), reach_ref=jnp.exp(-mu_r * sg(dist)),
                     att_if=jnp.exp(-mu_abs * dist), att_r=jnp.exp(-mu_abs_r * sg(dist)), row=row, cont=cont,
                     live_eval=live_eval, sigma_r=ref_eval.sigma_alpha, p_draw=p_draw, p_proposal=p_proposal, o=o,
                     os_=jnp.maximum(o, 0), drawn=o >= 0, ctx=ctx,
                     scatter=(mu_r > 0.0) & (s < sg(dist)) & ph.alive)  # fmt: skip
    return ph, b0, sc


def _surface_event(model, ops, ph, sc):
    """Return (the Next of the continuing outcome drawn at the crossing (one ``lax.switch``), the reference weight after
    it) (``_event``)."""  # noqa: D205, D209
    tables, q = model.tables, sc.q
    o, os_, ctx = sc.o, sc.os_, sc.ctx
    w_if = se.outcome_weight(ph.w, sc)  # p_live / q
    nrm = q["normal"]
    width = {"tables": ops.tables["unified"]}
    lb = se.Lobes(wi=ph.d, n=nrm, er=ctx["er"], ei=ctx["ei"], width={"sigma_alpha": sc.live_eval.sigma_alpha, **width},
                  width_ref={"sigma_alpha": sc.sigma_r, **width}, rand=sc.rand["lobe"],
                  shared=ops.reference is None)  # fmt: skip
    cross = (ph.d, nrm, ctx["n1"] / ctx["n2"], (ctx["er"], ctx["ei"], sc.live_eval.amps8))
    out = se.outcome_direction(tables, sc.row, os_, cross, lb)
    d_new = unit(out.d)  # no norm drift over many bounces
    far = jnp.asarray(FAR_SIDE)[os_] & sc.drawn
    ok = q["hit"] & sc.drawn & (sc.row >= 0)
    x_hit = ph.x + sc.dist * ph.d
    # at the reference every live / reference ratio is 1: the reached share (where it is defined), the attenuation at
    # the reference, the outcome at the probabilities it is drawn from, the lobe's density drawn from over itself
    r_o = jnp.where(sc.drawn, safe_div(sc.p_draw[os_], sc.p_proposal[os_]), 0.0)
    wr = sg(sg(ph.wr) * jnp.where(sc.reach_ref != 0.0, 1.0, 0.0) * sc.att_r * r_o * out.w_drawn)
    w_at = ph.w * ratio(sc.reach_live, sc.reach_ref)  # the weight that reaches the crossing (live / reference)
    s_cont = jnp.sum(sc.live_eval.probs)  # the continuing shares
    fates = (
        jnp.where(ok, w_at * sc.att_if * s_cont, 0.0),
        jnp.where(q["hit"], w_at * (1.0 - sc.att_if), 0.0),
        jnp.where(q["hit"] & (sc.row >= 0), w_at * sc.att_if * (1.0 - s_cont), 0.0),
    )  # state.Next.fates
    nx = state.Next(x=x_hit, d=d_new, w=w_if * out.w, cls=jnp.where(far, q["cls_after"], q["cls_before"]),
                    ok=ok, lam=ph.lam, dtime=sc.dist / sc.v_g,
                    sn=q["normal"].astype(F32), outcome=o.astype(I32), er=out.er, ei=out.ei,
                    reemitted=jnp.bool_(False), fates=fates)  # fmt: skip
    return nx, wr


def _bulk(model, ops, ph, sc):
    """Return (the Next of a bulk event at the free-path end, the reference weight after it, the time it took at the
    reference): the free path ends before the next surface.

    The channel is drawn from the reference rates (in Geant4's fixed order: Rayleigh, Mie, WLS) and weighted by the
    live / reference density of the free path in that channel (mu_c exp(-mu s)), times the absorption over the path
    (exp(-mu_abs s), a weight, never an event: G4OpAbsorption in expectation). The direction comes from the channel's
    lobe (``medium.channel_draw``). The photon's class there: before the first invisible crossing the start's, after
    the last the one before the visible crossing, between them the one the walk found at s. At the reference the
    channel's and the lobe's ratios are 1: the reference weight takes the absorption, the emission's weight and the
    jump's factor at the reference values.
    """  # noqa: D205
    q, cb, s = sc.q, sc.q["cls_before"], sc.s
    next_cls = jnp.where(s < q["t_pass0"], q["cls_start"], jnp.where(s > q["t_pass1"], cb, q["cls_at"]))
    cs = jnp.maximum(estimator.choose(sc.mu_c_r, sc.rand["channel"][0]), 0)
    path_r = sc.mu_c_r[cs] * jnp.exp(-sc.mu_r * s)
    w_path, att = ratio(sc.mu_c[cs] * jnp.exp(-sc.mu * s), path_r), jnp.exp(-sc.mu_abs * s)
    w = w_path * att
    w_r = ratio(path_r, path_r) * jnp.exp(-sc.mu_abs_r * s)
    x_s = ph.x + s * ph.d
    b = medium.channel_draw(model, ops, sc.medium_idx, cs, ph.d, (sc.ctx["er"], sc.ctx["ei"]), ph.lam, sc.rand)
    w_at = ph.w * w_path  # the weight that reaches the event (live / reference)
    fates = (w_at * att * b.yield_w, w_at * (1.0 - att * b.yield_w), jnp.zeros_like(w_at))  # state.Next.fates
    rl = estimator.reweight_log(b.lp, b.lpr)
    if "jump" in model.step.levers:  # lumen.speed: the diffusion jump
        # the walk's factor enters both weights: w its likelihood ratio from mu_ref to the live values, wr its value at
        # the reference values, exp(-mu_abs_r L) (the walk's length L held)
        x_n, d_n, er_n, ei_n, w_j, dt_j, L = fastpath.diffuse(model, ops, ph, sc, x_s, b)
        # the walk's factor (its absorption) is the walk's path: absorbed
        fates = (fates[0] * w_j, fates[1] + fates[0] * (1.0 - w_j), fates[2])
        return state.Next(x=x_n, d=d_n, w=ph.w * w * rl * b.yield_w * w_j, cls=next_cls,
                          ok=jnp.bool_(True), lam=b.lam, dtime=s / sc.v_g + b.delay + dt_j,
                          sn=q["normal"].astype(F32), outcome=jnp.int32(-1), er=er_n, ei=ei_n,
                          reemitted=b.reemitted, fates=fates), \
            sg(sg(ph.wr) * w_r * b.yield_r * jnp.exp(-sc.mu_abs_r * L))  # fmt: skip
    return state.Next(x=x_s, d=b.wo, w=ph.w * w * rl * b.yield_w, cls=next_cls,
                      ok=jnp.bool_(True), lam=b.lam, dtime=s / sc.v_g + b.delay,
                      sn=q["normal"].astype(F32), outcome=jnp.int32(-1), er=b.er, ei=b.ei,
                      reemitted=b.reemitted, fates=fates), sg(sg(ph.wr) * w_r * b.yield_r)  # fmt: skip


def _event(model, ops, ph, sc):
    """Return (the Next of the step's event, the reference weight after it).

    A bulk event or the surface's drawn outcome (one ``lax.cond``). The reference weight is the weight the photon would
    carry had the live values been the reference: the event's factor at the reference values on the photon's
    reference weight, a closed form (every live / reference ratio of a choice drawn at the reference is 1 there; a
    drawn outcome's probability is the one it is drawn from, an edge row's floor included: the live probability is 0
    there). When the run's values are the reference it is the weight itself in value.
    """
    return jax.lax.cond(sc.scatter, lambda _: _bulk(model, ops, ph, sc), lambda _: _surface_event(model, ops, ph, sc),
                        None)  # fmt: skip


class After(NamedTuple):
    """The state after the event's roulette and stuck rule (``_after``)."""

    alive: Any
    w: Any  # the weight
    wr: Any  # the reference weight
    nz: Any
    abandoned: Any  # the weight abandoned by the stuck rule
    rouletted: Any  # the weight the roulette ended minus the weight it added (0 in expectation)
    analog: Any  # the Analog (None without)


def _after(ops, ph, sc, nx, wr, b0, an=None):  # noqa: PLR0913
    """Return the After of the event's roulette and stuck rule.

    ``wr``: the weight after the event at the reference values (``_event``). A photon whose reference weight wr is
    below ε survives with
    probability p_s = wr / ε, both weights divided by p_s, else both are 0 and it ends: the only way a weight ends it.
    The decision reads the reference alone, so with the reference held it does not move with the live values; the
    live weight (value and derivative) is what the photon deposits. ``an``: the photon's Analog after the event
    (``Model.analog``), whose open fate the roulette keeps (``_analog_roulette``) and rescales with the weight.
    """
    q, scatter, ctl = sc.q, sc.scatter, ops.controls
    w_new, wr = nx.w, sg(wr)
    low, p_s = wr < ctl.roulette, wr / ctl.roulette  # Russian roulette, on the reference weight
    keep = sc.rand["roulette"][0] < p_s if an is None else _analog_roulette(an, wr, p_s, sc.rand["roulette"][0])
    kept, lost = low & keep, low & ~keep
    p_kept = jnp.where(kept, p_s, 1.0)  # p_s > 0 where it divides
    w_in = w_new
    w_new = jnp.where(kept, w_new / p_kept, jnp.where(lost, 0.0, w_new))
    # killed: w_in leaves; kept (probability p_s): w_in (1 / p_s - 1) is added; each event's expectation is 0
    rouletted = jnp.where(ph.alive & nx.ok, w_in - w_new, 0.0)
    wr_new = jnp.where(kept, wr / p_kept, jnp.where(lost, 0.0, wr))
    if an is not None:  # an open fate the roulette kept is rescaled with the weight (uniform on [0, w) again)
        an = an._replace(x=jnp.where(kept & jnp.isfinite(an.x), an.x / p_kept, an.x))
    alive = ph.alive & nx.ok & ~lost
    nz_new = jnp.where(q["hit"] & ~scatter & (sg(sc.dist) <= navigation.ZERO_STEP_BANDS * b0), ph.nz + 1, 0).astype(I32)
    abandon = alive & (nz_new >= navigation.ZERO_STEPS_BEFORE_KILL)
    abandoned = jnp.where(abandon, w_new, 0.0)
    alive = alive & ~abandon
    return After(alive=alive, w=w_new, wr=wr_new, nz=nz_new, abandoned=abandoned, rouletted=rouletted, analog=an)


def _advance(sc, ph, b0):
    """Return (vol, srf, held) after the event: the navigator's state for the next query.

    ``held``: the photon keeps the surface it sat on (a bulk event that did not move it off that face).
    """
    q = sc.q
    # a bulk event: the volume the free path ends in (before the first invisible crossing the start's, after the
    # last the one before the visible crossing, between them the one the walk found there: the ray's, as Geant4
    # steps through each crossing); a surface outcome that continues beyond takes the volume after, a reflection the
    # one it was located in
    vol_b = jnp.where(sc.s < q["t_pass0"], q["vol_start"], jnp.where(sc.s > q["t_pass1"], q["vol_before"], q["vol_at"]))
    after = jnp.asarray(FAR_SIDE)[sc.os_] & sc.drawn
    vol = jnp.where(sc.scatter, vol_b, jnp.where(after, q["vol_after"], q["vol_before"])).astype(I32)
    # a bulk event within a zero-length step of the start leaves the photon on the surface it sat on: its point did
    # not move off that face, and the band rules read the face from the photon's state, never from a float32 probe of
    # the point (a re-emission at a face it sits within one ulp of: the side is still the direction's)
    held = sc.scatter & (sg(sc.s) <= navigation.ZERO_STEP_BANDS * b0) & (ph.srf >= 0)
    srf = jnp.where(held, ph.srf, jnp.where(sc.scatter | ~q["hit"], -1, q["item"])).astype(I32)
    return vol, srf, held


def step(model, ops, ph, key):
    """Return (the photon after one step, its Deposit); ``key`` is the photon's key (``rng.photon_key``)."""
    tables = model.tables
    ph, b0, sc = _step_context(model, ops, ph, key)
    q, scatter = sc.q, sc.scatter
    at_if = q["hit"] & ph.alive & (sc.row >= 0)
    inv_an = sc.live_eval.inv_an  # 1 / A_n of a type read per normal incidence, else 1 (types.per_normal)
    p_det = sc.live_eval.detect
    dep_q = jnp.where(at_if, ph.w * sc.reach_live * sc.att_if * p_det, 0.0)  # the physical detected share
    lost = jnp.where(q["hit"] & ph.alive & (sc.row < 0) & ~scatter,
                     ph.w * ratio(sc.reach_live, sc.reach_ref) * sc.att_if, 0.0)  # fmt: skip
    share = jnp.where(at_if & ~scatter, ph.w * sc.att_if * p_det, 0.0) if model.analog else None
    nx, wr = _event(model, ops, ph, sc)
    expected, bulk, surface = nx.fates
    a_surf = jnp.where(ph.alive, surface, 0.0)  # absorbed at the crossing (0 unless the photon reached it)
    # met no surface within max_distance and no bulk event: the photon ends at max_distance with the weight that
    # reaches it (live / reference, attenuated; the absorption on the way is absorbed_bulk), as every fate's weight is
    # the weight where it ends; past a max_distance shorter than the geometry's reach (Model.reach) it was cut there
    far = ph.alive & ~q["hit"] & ~scatter
    d_end = jnp.minimum(sg(sc.dist), ops.controls.max_distance)
    w_far = ph.w * ratio(jnp.exp(-sc.mu * d_end), jnp.exp(-sc.mu_r * d_end))
    att_far = jnp.exp(-sc.mu_abs * d_end)
    cut = ops.controls.max_distance < F32(model.reach)  # the default max_distance is the reach: nothing is cut
    escaped = jnp.where(far & ~cut, w_far * att_far, 0.0)
    distance_cut = jnp.where(far & cut, w_far * att_far, 0.0)
    a_bulk = jnp.where(ph.alive, bulk, 0.0) + jnp.where(far, w_far * (1.0 - att_far), 0.0)
    reweighted = jnp.where(ph.alive & nx.ok, expected - nx.w, 0.0)  # the drawn factors' net (0 in expectation)
    cb, c_k = q["cls_before"], q["cls_after"]
    dpre = tables.deposit_pre  # static: a border sensor credits the region left
    out_side = False if dpre is None else jnp.asarray(dpre)[cb, c_k]
    local = _local(model, ops, sc, out_side, ph) | ({"inv_an": inv_an} if "inv_an" in model.local else {})
    sensor_dep = _sensor_dep(model, ops, dpre, q)
    time = ph.t + sc.dist / sc.v_g
    an = None
    if model.analog:
        at = (sensor_dep, time, ph.lam, inv_an, _analog_fields(local))
        an = _analog_crossing(ph.analog, sg(share), sg(nx.w), at)
    af = _after(ops, ph, sc, nx, wr, b0, an)
    alive = af.alive
    vol, srf, held = _advance(sc, ph, b0)
    gen = None if ph.gen is None else ph.gen + nx.reemitted.astype(I32)
    ph2 = state.Photon(x=nx.x, d=nx.d, t=ph.t + nx.dtime, lam=nx.lam, w=jnp.where(alive, af.w, 0.0),
                       wr=jnp.where(alive, af.wr, 0.0), alive=alive, vertex=ph.vertex + 1, er=nx.er.astype(F32),
                       ei=nx.ei.astype(F32), vol=vol, srf=srf, sn=jnp.where(held, ph.sn, nx.sn), nz=af.nz,
                       cls=nx.cls,
                       analog=af.analog, gen=gen)  # fmt: skip
    event, pair = jnp.where(scatter, -2, nx.outcome).astype(I32), cb * jnp.asarray(tables.row_of).shape[0] + c_k
    dep = Deposit(sensor=jnp.where(at_if, sensor_dep, -1), q=dep_q, time=time, lam=ph.lam, lost=lost,
                  abandoned=af.abandoned, escaped=escaped, distance_cut=distance_cut, absorbed_bulk=a_bulk,
                  absorbed_surface=a_surf, rouletted=af.rouletted, reweighted=reweighted, event=event, pair=pair,
                  share=share, **local)  # fmt: skip
    if tables.born_pair is not None:
        ph2, dep = _born(model, ops, ph, ph2, dep)
    return ph2, dep


def _born(model, ops, ph, ph2, dep):
    """Return (photon, Deposit) of a step, a photon emitted inside a sensor volume counted at its emission.

    A sensor group counting by volume (Geant4's sensitive detector: every step in the volume, the first included;
    ``lumen.rules.sensors.counts_born``) deposits at its first step what its entry would detect (the entry's detect
    deposit, ``Optics.born_pair``), at the emission's time, and the photon ends there (the entry's other share is
    absorbed), as it ends on entry. Its analog photon is absorbed with that share. The deposit carries the live weight
    and the photon ends with both weights 0, as every deposit and loss does under the reference weight.
    """
    tables = model.tables
    bp = jnp.asarray(tables.born_pair)[jnp.maximum(ph.cls, 0)]
    sensor = jnp.asarray(geometry(model, ops).sensor_of_vol)[jnp.maximum(ph.vol, 0)]
    born = (ph.vertex == 0) & ph.alive & (ph.vol >= 0) & (bp >= 0) & (sensor >= 0)
    row = jnp.asarray(tables.row_of)[bp, ph.cls]
    one = F32(1.0)
    ctx = dict(lam=ph.lam, wi=ph.d, n=-ph.d, n1=one, n2=one)
    pk = ops.params["surfaces"]
    ev = jax.lax.cond(born, lambda k: se.type_probs(tables, pk, k, ctx), lambda k: se.type_probs(tables, pk, -1, ctx),
                      row)  # fmt: skip
    d = ev.detect
    zero = jnp.zeros_like(dep.q)
    pick = lambda new, old: jnp.where(born, new, old)  # noqa: E731
    an = ph2.analog
    local = _born_local(model, ops, ph) | ({"inv_an": ev.inv_an} if "inv_an" in model.local else {})
    if model.analog:
        a_b = _analog_crossing(ph.analog, sg(ph.w * d), F32(0.0),
                               (sensor, ph.t, ph.lam, ev.inv_an, _analog_fields(local)))  # fmt: skip
        an = jax.tree_util.tree_map(pick, a_b, ph2.analog)
    ph2 = ph2._replace(
        alive=ph2.alive & ~born, w=pick(0.0, ph2.w).astype(F32), wr=pick(0.0, ph2.wr).astype(F32), analog=an
    )
    dep = dep._replace(sensor=pick(sensor, dep.sensor), q=pick(ph.w * d, dep.q), time=pick(ph.t, dep.time),
                       lost=pick(zero, dep.lost), abandoned=pick(zero, dep.abandoned), escaped=pick(zero, dep.escaped),
                       distance_cut=pick(zero, dep.distance_cut), absorbed_bulk=pick(zero, dep.absorbed_bulk),
                       absorbed_surface=pick(ph.w, dep.absorbed_surface), rouletted=pick(zero, dep.rouletted),
                       reweighted=pick(zero, dep.reweighted),
                       **{k: pick(v, getattr(dep, k)) for k, v in local.items()},
                       share=None if dep.share is None else pick(ph.w * d, dep.share))  # fmt: skip
    return ph2, dep


def _sensor_dep(model, ops, dpre, q):
    """Return the sensor a crossing's deposit credits: the one entered, or for a border sensor the region the ray left
    (the placed volume before the crossing, ``q["vol_before"]``: also for a photon born inside it)."""  # noqa: D205, D209
    if dpre is None:
        return q["sensor"]
    vb = q["vol_before"]
    left = jnp.where(vb >= 0, jnp.asarray(geometry(model, ops).sensor_of_vol)[jnp.maximum(vb, 0)], -1)
    return jnp.where(jnp.asarray(dpre)[q["cls_before"], q["cls_after"]], left, q["sensor"])


def _analog_crossing(an, share, w_after, at):
    """Return the Analog after a step: absorbed at the crossing (x < share), else x - share; decided (inf) once x is
    beyond the weight left after the event (the photon's analog fate was another loss).

    Given an open fate, x is uniform on [0, w): it falls in the crossing's reached share with probability share / w,
    the analog law (Geant4's: one cathode absorption per photon). ``at``: (sensor, time, lam, scale, local) of the
    crossing, recorded at the absorption: the sensor type's per-normal ``scale`` and the deposit's local fields
    ``local`` ({field: value}, ``Model.local``).
    """  # noqa: D205
    sensor, time, lam, scale, local = at
    absorbed = an.x < share
    x = jnp.where(absorbed, jnp.inf, an.x - share)
    x = jnp.where(x < w_after, x, jnp.inf)
    pick = lambda new, old: jnp.where(absorbed, new, old)  # noqa: E731
    return state.Analog(x=x, k=an.k, sensor=pick(sensor, an.sensor).astype(I32), time=pick(sg(time), an.time),
                        lam=pick(sg(lam), an.lam), scale=pick(sg(scale), an.scale),
                        **{k: pick(sg(v), getattr(an, k)) for k, v in local.items()})  # fmt: skip


def _analog_fields(local: dict) -> dict:
    """Return the local fields the analog record keeps (``state.analog_fields``)."""
    return {k: local[k] for k in state.analog_fields(local)}


def _analog_roulette(an, w, p_s, rand):
    """Return the roulette's keep of a photon carrying an Analog: always while its analog fate is open, else with
    the probability that makes the keep's total p_s (w / roulette), whatever the fate.

    P(open | the path) = w k (the analog weight times the photon's relative weight), so the photon survives with
    probability w k + (1 - w k) (p_s - w k) / (1 - w k) = p_s: the forward's roulette is unchanged in law, and an open
    analog photon is never ended by it (exact for a roulette <= 1 and k <= 1). At roulette 1 and k = 1 the tracked
    photon ends exactly when its analog fate is decided.
    """  # noqa: D205
    pi = w * an.k  # < 1 below the roulette's ε <= 1 (k <= 1), where the keep is read
    extra = (p_s - pi) / jnp.where(pi < 1.0, 1.0 - pi, 1.0)
    return jnp.isfinite(an.x) | (rand < extra)


def sensor_frame(model, ops, sc, out_side):
    """Return (R, t) of the sensor's placement at a crossing: the volume entered (for ``out_side`` the volume left)."""
    q, D = sc.q, geometry(model, ops).nav.dev()
    v = jnp.where(out_side, q["vol_before"], q["vol_after"])
    return D["R"][v], D["t"][v]


def _frame_fields(model, frame, x, d, n, gen) -> dict:
    """Return the deposit's local fields (``Model.local``): the point x, direction d and normal n in the frame
    ``frame`` = (R, t) of a sensor's placement (a thunk: read only when a geometric field is carried), and the
    re-emission count ``gen``."""  # noqa: D205, D209
    out = {}
    if any(k in model.local for k in VECTOR_FIELDS):
        R, t = frame()
        geo = {"pos": lambda: R.T @ (x - t), "dir": lambda: R.T @ d, "normal": lambda: R.T @ n}
        out = {k: geo[k]().astype(F32) for k in VECTOR_FIELDS if k in model.local}
    if "gen" in model.local:
        out["gen"] = gen
    return out


def _local(model, ops, sc, out_side, ph) -> dict:
    """Return the deposit's local fields at a crossing, in the sensor placement's frame (``_frame_fields``)."""
    x = ph.x + sc.dist * ph.d
    return _frame_fields(model, lambda: sensor_frame(model, ops, sc, out_side), x, ph.d, sc.q["normal"], ph.gen)


def _born_local(model, ops, ph) -> dict:
    """Return the local fields of a photon counted at its emission inside a sensor volume: the emission point and
    direction in the volume's frame, the normal against the direction (the entry's context, ``_born``)."""  # noqa: D205, D209

    def frame():
        D, v = geometry(model, ops).nav.dev(), jnp.maximum(ph.vol, 0)
        return D["R"][v], D["t"][v]

    return _frame_fields(model, frame, ph.x, ph.d, -ph.d, ph.gen)
