"""The diffusion jump (``StepConfig.jump``): its data, built from the detector, and its step code.

At a bulk Rayleigh event in a medium whose only channel is a Rayleigh lobe without re-emission, with a linear
polarisation state after the event (ei = 0 exactly), the photon walks its own scatterings inside the ball B(x, c) of
the proven clearance c at x, with no geometry query: free paths Exp(mu_ref), lumen's own polarised Rayleigh draw
(``phase.rayleigh_sample``) and projection (``polar.project``), until a flight leaves the ball (the photon is put
at the crossing) or the walk has made ceil((c mu_ref)^2) scatterings (the ball's diffusion time; any stopping time
is exact, this one only bounds a walk's length). No surface lies inside the ball, so this is the plain walk with
its geometry queries skipped. Weight exp(-mu_abs L) (mu / mu_ref)^n exp(-(mu - mu_ref) L) (n scatterings, path L:
the likelihood ratio of the walk, as the plain walk's per-flight factors), time L / v_g. The walk's uniforms are the
photon's own (the vertex key's fold ``JUMP_STREAM``), so photons and chunks stay independent. On with the analog
record: at the reference values the factor is exp(-mu_abs L), and the analog photon's open fate (uniform on [0, w))
survives it with that probability, as along the plain walk's flights.

Its data are built from the detector, never given (``prepare``, once per Model; the grids cached by content): per
eligible medium m, a clearance grid: a proven lower bound of the distance to every surface
(``lumen.geometry.clearance``) over the boxes of m's volumes in which a walk is worth it, keyed by the navigator's
build key (``lumen.geometry.cache.key(scene, options)``) and the grid. Its spacing changes only how tight the bound
is (speed); it is h_m = 2 R_0 lambda_m / sqrt(3) with lambda_m the medium's shortest nominal Rayleigh length: the
lookup at the nearest node loses at most h sqrt(3) / 2 = R_0 lambda_m. A grid above GRID_BYTES is built at the
coarser spacing that fits (any spacing is a proven bound). A volume whose box half-width is below R_0 lambda_m is
left out (its balls hold few scatterings); the largest half-width over m's volumes caps the bound (exactly: no point
of m is farther from a surface). R_0 and GRID_BYTES are speed constants: any ball inside the proven clearance is
exact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen import _cache
from lumen._core import rng
from lumen.geometry import clearance
from lumen.geometry.cache import code_digest
from lumen.media import phase
from lumen.rules.choices import FLOOR32, GRID_COARSEN, JUMP_FLOOR, JUMP_STEPS
from lumen.surfaces import polar

F32, I32 = jnp.float32, jnp.int32
sg = jax.lax.stop_gradient
GRID_BYTES = 1 << 26  # the device memory one medium's clearance grid may take (float32 nodes): a resource limit,
#                       speed only (a coarser grid is the same proven bound, looser)
SQRT3 = float(np.sqrt(3.0))
R_0 = 4.0  # scattering lengths: the smallest ball a grid is built for (speed only: any ball inside the clearance is
# exact)


@dataclass(frozen=True, eq=False)
class Jump:
    """The jump's host data: one clearance grid per medium it acts in."""

    media: tuple  # medium indices with a grid
    grids: tuple  # per medium: dict(lo, h, n, clear)
    info: tuple  # per eligible medium: what was decided (recorded)


def eligible(medium) -> bool:
    """Whether a medium's only channel is a Rayleigh lobe without re-emission (the walk inside the ball is its walk)."""
    return (
        len(medium.channels) == 1 and medium.channels[0].emission is None and medium.channels[0].lobe is phase.RAYLEIGH
    )


def _shortest_rayleigh(optics, library, m):
    """Return the shortest Rayleigh length (m) of medium ``m`` as built (the library's, float64): its channel's table
    or constant, a GDML's RAYLEIGH or the one derived from RINDEX (G4OpRayleigh's), alike."""  # noqa: D205, D209
    p = library.media[optics.media_names[m]].params[optics.media[m].channels[0].name]["length"]
    return float(np.min(np.asarray(p["value"], np.float64)))


def _grid_key(nav, m, lo, h, n, cap):
    hh = hashlib.sha256(str(nav.build_key).encode())
    hh.update(code_digest(("lumen.geometry.clearance", "lumen.transport.fastpath")).encode())
    hh.update(repr((int(m), np.asarray(lo, np.float64).tobytes().hex(), float(h), tuple(int(x) for x in n),
                    float(cap))).encode())  # fmt: skip
    return hh.hexdigest()


def _bound(nav, lo, h, n, cap):
    """Return ``clearance.bound`` evaluated now, also under ``jax.jit`` (it builds the detector's host table).

    Its device arithmetic reads no traced value, so it runs at trace time rather than being staged into the trace (a
    first fast run under ``jax.jit`` failed with TracerArrayConversionError).
    """
    with jax.ensure_compile_time_eval():
        return clearance.bound(nav, lo, h, n, cap)


def _clearance(nav, m, lo, h, n, cap):
    if nav.build_key is None:
        return _bound(nav, lo, h, n, cap)
    path = _cache.entry("build", f"clear-{_grid_key(nav, m, lo, h, n, cap)}")
    f = path / "clear.npz"
    if f.is_file():
        with np.load(f) as z:
            return z["clear"]
    c = _bound(nav, lo, h, n, cap)
    with _cache.directory(path) as tmp:
        np.savez(tmp / "clear.npz", clear=c)
    return c


def _fit(nav, vols, h):
    """Return (grid, h) at spacing h, or at the coarser spacing whose grid fits GRID_BYTES (float32 nodes)."""
    g = clearance.grid(nav, vols, h)
    while 4 * int(np.prod(g["n"])) > GRID_BYTES:
        h *= max((4 * float(np.prod(g["n"])) / GRID_BYTES) ** (1.0 / 3.0), GRID_COARSEN)
        g = clearance.grid(nav, vols, h)
    return g, h


def prepare(geometry, optics, library):
    """Return the jump's data for a detector (a ``Jump``; no media: it can act nowhere)."""
    nav = geometry.nav
    cls = np.asarray(nav.host["cls"])[: nav.n]
    med_of = np.asarray(optics.medium_of)[cls]
    lo_w, hi_w = np.asarray(nav.lo_w, np.float64)[: nav.n], np.asarray(nav.hi_w, np.float64)[: nav.n]
    half = 0.5 * np.min(hi_w - lo_w, 1)
    media, grids, info = [], [], []
    for m, med in enumerate(optics.media):
        if not eligible(med):
            continue
        lam = _shortest_rayleigh(optics, library, m)
        vols = np.flatnonzero(med_of == m)
        if lam <= 0 or not len(vols):
            info.append(dict(medium=optics.media_names[m], built=False, why="no volume"))
            continue
        fire = vols[half[vols] >= R_0 * lam]
        h0 = 2.0 * R_0 * lam / SQRT3
        rec = dict(medium=optics.media_names[m], shortest_rayleigh=lam, h_derived=h0, volumes=len(vols),
                   can_jump=len(fire))  # fmt: skip
        if not len(fire):
            info.append(dict(rec, built=False, why="no volume holds the smallest jump"))
            continue
        g, h = _fit(nav, fire, h0)
        cap = float(half[vols].max())
        c = _clearance(nav, m, g["lo"], h, g["n"], cap)
        rec = dict(rec, h=h, nodes=int(np.prod(g["n"])), cap=cap, clear_max=float(c.max()))
        if float(c.max()) < R_0 * lam:  # no node holds the smallest ball: the grid is not worth it
            info.append(dict(rec, built=False, why="the largest clearance holds no jump"))
            continue
        media.append(m)
        grids.append(dict(lo=g["lo"], h=h, n=g["n"], clear=c))
        info.append(dict(rec, built=True))
    return Jump(media=tuple(media), grids=tuple(grids), info=tuple(info))  # no media: the lever is off


def operands(jump):
    """Return the jump's device arrays (once per Model)."""
    with jax.ensure_compile_time_eval():
        return dict(grids=tuple(dict(lo=jnp.asarray(g["lo"], F32), h=jnp.asarray(g["h"], F32),
                                     n=jnp.asarray(g["n"], I32), clear=jnp.asarray(g["clear"], F32))
                                for g in jump.grids))  # fmt: skip


def _clear_at(G, xs):  # noqa: N803 -- G the clearance grids
    """Return (inside the grid, the clearance bound at xs) of one grid (the nearest node's bound minus |xs - node|)."""
    lo, h, n, clr = G["lo"], G["h"], G["n"], G["clear"]
    fi = (xs - lo) / h
    inside = jnp.all((fi > -0.5) & (fi < n.astype(F32) - 0.5))
    i = jnp.clip(jnp.floor(fi + 0.5).astype(I32), 0, n - 1)
    return inside, clr[i[0], i[1], i[2]] - jnp.linalg.norm(xs - (lo + h * i.astype(F32)))


def diffuse(model, ops, ph, sc, x, b):
    """Return (x, d, er, ei, weight factor, time shift, the walk's length) of the walk inside the clearance ball
    (identity, length 0, where it does not apply). Everything the walk decides or draws is held (stop_gradient): its
    tangents are the weight's, through the live mu, as the plain walk's, and the start state's: the exit state carries
    it through the walk's projections and the weight the scores of the walk's Rayleigh densities in the state (exact's
    lp with the live state)."""  # noqa: D205, D209
    D = ops.tables["diff"]
    xs = sg(x).astype(F32)  # where the walk may act (the grids are float32): a decision, no tangent
    inside, c = jnp.bool_(False), F32(0.0)
    for m, G in zip(model.jump.media, D["grids"], strict=True):
        here = sc.medium_idx == m
        ins_m, c_m = _clear_at(G, xs)
        inside = jnp.where(here, ins_m, inside)
        c = jnp.where(here, c_m, c)
    mu_r = sg(sc.mu_r)
    lin = jnp.all(b.ei == 0.0)
    ok = inside & (c > 0.0) & lin & ph.alive & (mu_r > 0.0)
    c = jnp.where(ok, c, 0.0)
    budget = jnp.where(ok, jnp.ceil(jnp.minimum(jnp.square(c * mu_r), JUMP_STEPS)), 0.0).astype(I32)
    key = sc.rand["jkey"]
    zero = jnp.zeros(3, F32)
    I3 = jnp.eye(3, dtype=F32)  # traced here (a module constant would be a closed-over array)

    def body(cy):
        y, d, e, L, n, j, _, Mr, Mi, g = cy
        u = rng.uniforms(jax.random.fold_in(key, j), 4)
        s = -jnp.log(u[0]) / mu_r
        by = jnp.sum(y * d)
        t = -by + jnp.sqrt(jnp.maximum(by * by - (jnp.sum(y * y) - c * c), 0.0))  # the flight's exit of the ball
        out = s >= t
        k = phase.rayleigh_sample(e, zero, u[1:4])
        e2, _ = polar.project(e, zero, k)
        # the state's sensitivity to the start state (values): e2 = P e / |P e| with P = 1 - k k^T, and at ei = 0 the
        # er and ei parts decouple (|E|^2's ei term is second order); the score of the drawn k: d log p(k | e) / de
        kd = jnp.sum(k * e)
        P = I3 - jnp.outer(k, k)
        inv = 1.0 / jnp.maximum(jnp.sqrt(jnp.maximum(1.0 - kd * kd, 0.0)), JUMP_FLOOR)
        Jr = (I3 - jnp.outer(e2, e2)) @ P * inv
        gs = -2.0 * kd * k / jnp.maximum(1.0 - kd * kd, JUMP_FLOOR)
        keep = lambda new, old: jnp.where(out, old, new)  # noqa: E731
        return (y + jnp.where(out, t, s) * d, jnp.where(out, d, k), jnp.where(out, e, e2), L + jnp.where(out, t, s),
                n + jnp.where(out, 0, 1), j + 1, out, keep(Jr @ Mr, Mr), keep(P @ Mi * inv, Mi),
                keep(g + gs @ Mr, g))  # fmt: skip

    init = (zero, sg(b.wo), sg(b.er), F32(0.0), jnp.int32(0), jnp.int32(0), ~ok, I3, I3, zero)  # n: an integer count
    y, d, e, L, n, _, _, Mr, Mi, g = jax.lax.while_loop(lambda cy: (cy[5] < budget) & ~cy[6], body, init)
    lr = jnp.log(jnp.maximum(sc.mu, FLOOR32) / jnp.maximum(mu_r, FLOOR32))
    wj = jnp.exp(-sc.mu_abs * L) * jnp.exp(n.astype(F32) * lr - (sc.mu - mu_r) * L)
    # the start state's tangent (exact's live state): through the walk's states and its densities' scores, applied
    # outside the loop (values Mr, Mi, g; zero in value, so the walk's values are unchanged)
    der, dei = b.er - sg(b.er), b.ei - sg(b.ei)
    wj = wj * jnp.exp(jnp.sum(g * der))
    er_n, ei_n = e + Mr @ der, Mi @ dei
    return (jnp.where(ok, x + sg(y), x), jnp.where(ok, d, b.wo), jnp.where(ok, er_n, b.er), jnp.where(ok, ei_n, b.ei),
            jnp.where(ok, wj, 1.0), jnp.where(ok, L / sc.v_g, 0.0), sg(jnp.where(ok, L, 0.0)))  # fmt: skip
