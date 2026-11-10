"""The navigator (device): where a photon is, and the next visible crossing along its ray, by Geant4's volume tracking.

The photon carries its placed volume ``vol`` (one solid per volume, placements overlap-free and daughters inside their
mothers: ``lumen.validate.overlaps`` checks it) and the surface it sits on (``srf``, ``lumen.geometry.surface_id``).
A query is Geant4's step: the nearer of the exit of vol's own solid and the first entry into one of vol's daughters
(a BVH over each volume's daughters); the next volume from the tree (the daughter entered, or the mother, then
relocated by ``lumen.rules.navigation``); a crossing into the same class (or a pass pair) continues the scan.

Tolerance by surface identity, not by a push: the photon starts at its hit point and passes ``srf``; an event of that
item within the band of the start is that same crossing; any other item's event counts, however close. A start state
that disagrees with the ray (vol's solid does not hold the start, or a daughter does) is resolved as Geant4 resolves
it: a zero-length exit to the mother, or a zero-length entry into that daughter.

    nav = Navigator.from_scene(scene)                      # host: build (or read the build cache)
    vol = locate(nav, x)                                    # jnp, one thread (or a jax.vmap batch)
    cx = next_crossing(nav, o, d, vol, srf=-1, t_max=1e3)  # Crossing
    t, n = surface_at(nav, cx, o, d)                        # the winner's (t, normal), live in (o, d)

A query starts at its origin. ``t`` is the distance along the ray (m) throughout this module.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from lumen.geometry import bvh, navigator_build, surface_id
from lumen.geometry.csg import Csg, pick
from lumen.rules import navigation as rules

F32, I32 = jnp.float32, jnp.int32
INF = surface_id.INF
STACK = navigator_build.STACK
MAX_LOCATE_DEPTH = 64  # locate descends at most this many levels
MAX_SETTLE_MOVES = 8  # a relocation moves at most this many volumes
MAX_ROUNDS = 16  # invisible crossings followed per query before a straight continue event
MAX_ANCESTORS = 8  # a relocation's ancestor test looks this many levels up
TINY_DIR = 1e-30  # a direction component below this is replaced by +-this (the slab test's inverse)
NORM_FLOOR = surface_id.NORM_FLOOR
sg = lax.stop_gradient

__all__ = ["Crossing", "Navigator", "locate", "next_crossing", "surface_at"]


class Crossing(NamedTuple):
    """The next visible crossing of a query (one thread).

    ``t`` (live through custom_jvp), ``hit``, ``normal`` (faces the photon); the volumes: ``vol_start`` (located),
    ``vol_before`` / ``vol_after`` (of the visible crossing), ``vol_reflect`` (where a reflection stays); ``srf`` the
    surface met; ``step_too_small``: a straight continue event (Geant4's StepTooSmall, or the rounds ran out on
    invisible crossings); ``winner`` (item, leaf, face, zero): the handle on the winning surface (``surface_at``). The
    legacy optical classes (``cls_*``), ``sensor``, ``zero`` and the invisible crossings passed (``t_pass0``,
    ``t_pass1``, ``n_pass``) serve the step.
    """

    t: jax.Array
    hit: jax.Array
    normal: jax.Array
    vol_start: jax.Array
    vol_before: jax.Array
    vol_after: jax.Array
    vol_reflect: jax.Array
    srf: jax.Array
    step_too_small: jax.Array
    winner: jax.Array
    cls_start: jax.Array
    cls_before: jax.Array
    cls_after: jax.Array
    sensor: jax.Array
    zero: jax.Array
    t_pass0: jax.Array
    t_pass1: jax.Array
    n_pass: jax.Array


class _Enter(NamedTuple):
    """The fixed inputs of one volume visit (``Navigator.enter``)."""

    od: object
    dd: object
    t_lo: object
    t_bound: object
    last: object
    band: object
    sd: tuple
    shear: tuple | None


class _Query(NamedTuple):
    """The fixed inputs of one query."""

    od: object
    dd: object
    tmax: object
    xa: object
    snv: object
    pass_tab: object
    max_rounds: int


def _pk(v_, k_):
    return jnp.where(k_ == 0, v_[0], jnp.where(k_ == 1, v_[1], v_[2]))


def _shear(dd):
    """Return the watertight ray-triangle test's shear of direction dd: (kx, ky, kz, Sx, Sy, Sz)."""
    ad = jnp.abs(dd)
    kz = jnp.where((ad[0] >= ad[1]) & (ad[0] >= ad[2]), 0, jnp.where(ad[1] >= ad[2], 1, 2))
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    sw = jnp.where(kz == 0, dd[0], jnp.where(kz == 1, dd[1], dd[2])) < 0
    kx, ky = jnp.where(sw, ky, kx), jnp.where(sw, kx, ky)
    dz = _pk(dd, kz)
    return kx, ky, kz, _pk(dd, kx) / dz, _pk(dd, ky) / dz, 1.0 / dz


def _choose(better, new, old):
    return tuple(jnp.where(better, a_, b_) for a_, b_ in zip(new, old, strict=False))


class Navigator:
    """The navigator on its arrays (``navigator_build.NavArrays``); device methods trace into the caller's kernel."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.E = Csg(arrays.csg)
        self.host = arrays.host
        self.n, self.n_mesh, self.chunk = arrays.n, arrays.n_mesh, arrays.chunk
        self.CH0, self.SRF, self.bg = arrays.ch0, arrays.srf, arrays.bg
        self.names, self.defs, self.stats = list(arrays.names), list(arrays.defs), dict(arrays.stats)
        self.coincident, self.kids = list(arrays.coincident), arrays.kids
        self.lo_w, self.hi_w = arrays.lo_w, arrays.hi_w
        self._D = None
        self._live = None

    @classmethod
    def from_scene(cls, scene, cache=False, **options):
        """Return the navigator of a lowered Scene; ``cache``: read or write the build cache (``lumen.build.cache``)."""
        if cache:
            from lumen.build import cache as build_cache

            return cls(build_cache.nav_arrays(scene, **options))
        return cls(navigator_build.build(scene, **options))

    def with_classes(self, vol_cls, mesh_cls=(), outside_cls=None):
        """Return a navigator reporting another class per volume, mesh and the outside."""
        return Navigator(self.arrays.with_classes(vol_cls, mesh_cls, outside_cls))

    def dev(self):
        """Return the host arrays as device arrays (made once, at trace time; or the operands ``bind`` gave)."""
        if self._D is None:
            with jax.ensure_compile_time_eval():
                self._D = {k: jnp.asarray(v) for k, v in self.host.items()}
        return self._D

    def operands(self):
        """Return the navigator's device arrays, to enter a loop as operands (``bind``): no array is closed over."""
        return dict(nav=self.dev(), csg=self.E.dev())

    def bind(self, ops):
        """Return this navigator reading its device arrays from ``ops`` (``operands``, traced)."""
        nav, nav.E = copy.copy(self), copy.copy(self.E)
        nav._D, nav.E._dev = ops["nav"], ops["csg"]
        return nav

    # ---------------------------------------------------------------------------------------- per-volume solid calls
    def frame(self, i, od, dd):
        """Return (kind, origin, direction) of a ray in volume i's frame."""
        D = self.dev()
        R, t = D["R"][i], D["t"][i]
        return D["kind"][i], R.T @ (od - t), R.T @ dd

    def first_change_in(self, i, od, dd, t0, t1):
        """Return volume i's first membership change in (t0, t1]: (hit, t, inside after, leaf, face, inside0)."""
        k, ol, dl = self.frame(i, od, dd)
        r = self.E.first_change(k, ol, dl, t0, t1)
        return r["hit"], r["t"], r["inside"], r["leaf"], r["face"], r["inside0"]

    def ray_holds(self, i, od, dd, t):
        """Return whether volume i's solid holds the ray just after t."""
        k, ol, dl = self.frame(i, od, dd)
        return self.E.inside_ray(k, ol, dl, t)

    def owns_ray(self, i, od, dd, t, pad):
        """Return whether volume i holds od + t dd as Geant4 places it: its solid does, no hole filler does."""
        D = self.dev()
        n = self.n

        def visit(j, g):
            jc = jnp.minimum(j, n - 1)
            return g | ((j < n) & (D["parent"][jc] != i) & self.ray_holds(jc, od, dd, t))

        filled = pick(D["fill"][i], lambda: self.scan(i, od + t * dd, visit, jnp.bool_(False), lambda g: g, pad),
                      lambda: jnp.bool_(False))  # fmt: skip
        return self.ray_holds(i, od, dd, t) & ~filled

    def holds_point(self, i, x):
        """Return whether volume i's solid holds the point x (closed, no tolerance)."""
        D = self.dev()
        return self.E.inside(D["kind"][i], D["R"][i].T @ (x - D["t"][i]))

    # ---------------------------------------------------------------------------------------- BVH scans (points)
    def scan(self, v, x, visit, init, done, pad=0.0):
        """Visit v's daughters whose box holds x: visit(item, acc) on the BVH leaves; stop when done(acc).

        Each frame group's BVH is scanned in its frame; not the mesh group (no chunk holds a point).
        """
        D = self.dev()

        def grp(c):
            g, acc = c
            xg = D["gR"][g].T @ x

            def body(c2):
                node, acc = c2
                inside = jnp.all((D["lo"][node] - pad <= xg) & (xg <= D["hi"][node] + pad))
                leaf = D["count"][node] > 0
                acc = lax.fori_loop(0, jnp.where(inside & leaf, D["count"][node], 0),
                                    lambda k, a: visit(D["items"][D["start"][node] + k], a), acc)  # fmt: skip
                nxt = jnp.where(inside & ~leaf, node + 1, D["skip"][node])
                return jnp.where(done(acc), D["g1"][g], nxt), acc

            _, acc = lax.while_loop(lambda c2: c2[0] < D["g1"][g], body, (D["g0"][g], acc))
            return g + 1, acc

        return lax.while_loop(lambda c: (c[0] < D["vg1u"][v]) & ~done(c[1]), grp, (D["vg0"][v], init))[1]

    def _find(self, v, x):
        """Return the daughter of v whose solid holds the point x, or -1."""
        n = self.n

        def visit(i, g):
            ok = (g < 0) & (i < n)
            return jnp.where(ok & self.holds_point(jnp.minimum(i, n - 1), x), i, g)

        return self.scan(v, x, visit, I32(-1), lambda g: g >= 0)

    def locate(self, x, start=None):
        """Return the deepest volume whose solid holds x (``n``: outside every volume).

        ``start``: a volume already known (>= 0) is returned as is (the descent's loop starts done, so a caller that
        locates only unknown starts pays nothing for the known ones).
        """
        x = jnp.asarray(x, F32)
        known = jnp.bool_(False) if start is None else jnp.asarray(start, I32) >= 0

        def body(c):
            v, _, k = c
            f = self._find(v, x)
            return jnp.where(f >= 0, f, v), f < 0, k + 1

        v0 = I32(self.n) if start is None else jnp.where(known, jnp.asarray(start, I32), I32(self.n))
        return lax.while_loop(lambda c: ~c[1] & (c[2] < MAX_LOCATE_DEPTH), body, (v0, known, I32(0)))[0]

    def _find_ray(self, v, od, dd, t, excl, pad):
        """Return the daughter of v (other than excl) holding od + t dd (just after t, along the ray), or -1."""
        n = self.n
        x = od + t * dd

        def visit(i, g):  # Geant4's LevelLocate: the highest index first
            ok = rules.level_locate_wins(i, g, excl, n)
            return jnp.where(ok & self.ray_holds(jnp.minimum(i, n - 1), od, dd, t), i, g)

        return self.scan(v, x, visit, I32(-1), lambda g: jnp.bool_(False), pad)

    def _settle(self, v, ray, crossed, pad):
        """Return (the volume holding the ray just after t, from v's neighbourhood; whether it moved).

        Down into a daughter holding it, up while v does not (a zero-length volume is passed through). The item just
        crossed is decided by the crossing itself, never re-entered or left here. ``ray`` = (od, dd, t).
        """
        od, dd, t = ray
        D = self.dev()
        n = self.n

        def body(c):
            v_, _, k = c
            f = self._find_ray(v_, od, dd, t, crossed, pad)
            vi = jnp.minimum(v_, n - 1)
            ins = (v_ >= n) | (v_ == crossed) | self.ray_holds(vi, od, dd, t)
            nv = jnp.where(f >= 0, f, jnp.where(ins, v_, D["parent"][v_]))
            return nv, (f < 0) & ins, k + 1

        v2, _, _ = lax.while_loop(lambda c: ~c[1] & (c[2] < MAX_SETTLE_MOVES), body, (v, jnp.bool_(False), I32(0)))
        return v2, v2 != v

    # ---------------------------------------------------------------------------------------- the daughter entry
    def _triangle(self, ctx, i, k0, kn, b):
        """Return the best event after testing one triangle of a mesh chunk (the facet rule); ``b`` = (q, best)."""
        q, bb, hi_t = b
        D = self.dev()
        od, dd, t_lo, band = ctx.od, ctx.dd, ctx.t_lo, ctx.band
        kx, ky, kz, Sx, Sy, Sz = ctx.shear
        k = k0 + jnp.minimum(q, kn - 1)
        A, B, C = D["V0"][k] - od, D["V1"][k] - od, D["V2"][k] - od

        def sh(P):  # noqa: N803 -- P a triangle's vertex
            return (_pk(P, kx) - Sx * _pk(P, kz), _pk(P, ky) - Sy * _pk(P, kz), Sz * _pk(P, kz))

        (Ax, Ay, _), (Bx, By, _), (Cx, Cy, _) = sh(A), sh(B), sh(C)
        U = Cx * By - Cy * Bx
        V_ = Ax * Cy - Ay * Cx
        W_ = Bx * Ay - By * Ax
        inside = ((U >= 0) & (V_ >= 0) & (W_ >= 0)) | ((U <= 0) & (V_ <= 0) & (W_ <= 0))
        det = U + V_ + W_
        ok = inside & (det != 0) & (q < kn)
        # the sheared test decides inside / outside only (watertight); t comes from the plane: the start's height s0
        # over it (outward +) over the slope (the test's own t interpolates the vertices' depths metres away)
        Nk = D["N"][k]
        s0 = jnp.dot(Nk, od + t_lo * dd - D["V0"][k])
        nd = jnp.dot(Nk, dd)
        t_k = t_lo - s0 / jnp.where(nd != 0, nd, F32(-1.0))
        # the facet rule: met only entering it (n . d < 0), a start behind its plane within the tolerance is on it;
        # a back facet met first from clearly behind is the inside of the opaque solid: the event face = -2
        tol = F32(rules.FACET_BEHIND_BANDS) * band
        front = ok & (nd < 0) & (s0 > -tol) & (t_lo - t_k <= band)
        back = ok & (nd > 0) & (s0 < -tol) & (t_k > t_lo)
        t_c = jnp.maximum(t_k, t_lo)
        better = (front | back) & (t_c < hi_t) & ((t_c < bb[0]) | ((t_c == bb[0]) & bb[5]))
        return _choose(better, (t_c, i, k, jnp.where(back, I32(-2), I32(-1)), t_k <= t_lo, jnp.bool_(False)), bb)

    def _mesh_visit(self, ctx, i, b):
        """Return the best event after a mesh chunk (item i >= CH0)."""
        D = self.dev()
        sd = ctx.sd
        c = jnp.clip(i - self.CH0, 0, D["cstart"].shape[0] - 1)
        k0, kn = D["cstart"][c], D["cn"][c]
        hi_t = jnp.minimum(b[0], ctx.t_bound)
        # the triangle the photon sits on (srf): heading into it, a zero-length re-hit (its boundary process again);
        # else it is no crossing -- a plane is never met twice
        on_k = sd[0] & (i == ctx.last)
        re = on_k & (sd[3] < 0) & ((ctx.t_lo < b[0]) | ((ctx.t_lo == b[0]) & b[5]))
        b = _choose(re, (ctx.t_lo, i, sd[1], I32(-1), jnp.bool_(True), jnp.bool_(False)), b)
        return lax.fori_loop(0, self.chunk, lambda q, bb: self._triangle(ctx, i, k0, kn, (q, bb, hi_t)), b)

    def _on_face(self, ctx, i, fr, kn_i):
        """Return whether the start is on the first change's face (the on-face rule: K_ON bands, the normal decides)."""
        dt = fr["t"] - ctx.t_lo
        lim = rules.K_ON * ctx.band

        def grazing():  # between lim and lim / sin(0.25 deg): the normal decides
            g = self.hit_grad(i, jnp.maximum(fr["leaf"], 0), fr["face"], ctx.od + fr["t"] * ctx.dd)
            cs = jnp.abs(jnp.dot(g, ctx.dd)) / jnp.maximum(jnp.linalg.norm(g), NORM_FLOOR)
            return dt * cs < lim

        far = pick(~kn_i & fr["hit"] & (dt >= lim) & (dt < rules.GRAZE * lim), grazing, lambda: jnp.bool_(False))
        return ~kn_i & fr["hit"] & ((dt < lim) | far)

    def _unit_change(self, ctx, i, b, is_own):
        """Return (the first_change result, whether the item is the one sat on) of volume i in this visit."""
        sd = ctx.sd
        bnd = jnp.minimum(b[0], ctx.t_bound) + ctx.band
        k, ol, dl = self.frame(i, ctx.od, ctx.dd)
        kn_i = sd[0] & (i == ctx.last)
        # the face sat on: its root of the start's kind -- a plane or a quadric has no other; a torus (two of each
        # kind), only within the band along the ray
        ED = self.E.dev()
        tor = ED["T"]["typ"][ED["K"]["l0"][k] + jnp.maximum(sd[1], 0)] == 2
        t_on = ctx.t_lo + rules.K_ON * ctx.band / jnp.maximum(jnp.abs(sd[3]), 1.0 / rules.GRAZE)
        t_hi = jnp.where(tor, t_on, F32(INF))
        # while the ray is within that face's band its leaf is on the side the photon crossed to, whatever a float32
        # probe there says (Geant4: within kCarTolerance the direction decides)
        lf_on = jnp.where(kn_i, sd[1], I32(-2))
        r = self.E.first_change(k, ol, dl, ctx.t_lo, bnd, want=~is_own, init=(kn_i, is_own),
                                skip=(lf_on, sd[2], sd[4], t_hi), on_face=(lf_on, sd[4], t_on))  # fmt: skip
        return r, kn_i

    def _unit_visit(self, ctx, i, b, is_own):
        """Return the best event after volume i: own (item 0) its first exit, a daughter its first entry.

        A zero-length exit if the start is outside own, a zero-length entry if it is inside a daughter (never the item
        just crossed: its identity). The item sat on (side known): in it iff own; its direction decides.
        """
        sd = ctx.sd
        r, kn_i = self._unit_change(ctx, i, b, is_own)
        h, t, lf, f, in0 = r["hit"], r["t"], r["leaf"], r["face"], r["inside0"]
        fr = r["first"]
        on = self._on_face(ctx, i, fr, kn_i)
        # the item sat on: in own unless heading out (s > 0) -- or its float32 ray has no exit from it at all; in a
        # daughter only heading in
        by_s = kn_i & jnp.where(is_own, sd[3] > 0, sd[3] < 0)
        ins = jnp.where(kn_i, jnp.where(is_own, (sd[3] <= 0) & h, sd[3] < 0), jnp.where(on, fr["inside"], in0))
        h = h & (is_own | ~ins)  # a daughter the start is in: its zero-length entry or none
        zero = jnp.where(is_own, ~ins, ins & ((i != ctx.last) | kn_i))
        # a zero-length event's face: the face sat on when s decided it, the face at the start when the on-face rule
        # did; else none
        lf = jnp.where(zero, jnp.where(by_s, sd[1], jnp.where(on, fr["leaf"], I32(-1))), lf)
        f = jnp.where(zero, jnp.where(by_s, sd[2], jnp.where(on, fr["face"], I32(-1))), f)
        te = jnp.where(zero, ctx.t_lo, jnp.where(h, t, F32(INF)))
        # own is item 0: it sets the best (its exit, INF if none); a daughter beats it on a tie (the entry-tie rule)
        better = is_own | ((zero | h) & rules.entry_wins(te, i, b))
        return _choose(better, (te, i, lf, f, zero, is_own), b)

    def _visit(self, ctx, i, b, is_own):
        if self.n_mesh:
            return pick((i >= self.CH0) & ~is_own, lambda: self._mesh_visit(ctx, i, b),
                        lambda: self._unit_visit(ctx, jnp.minimum(i, self.n - 1), b, is_own))  # fmt: skip
        return self._unit_visit(ctx, i, b, is_own)

    def _node(self, ctx, vi_, c):
        """One step of a group's stackless BVH traversal (node -2: the volume's own solid, item 0)."""
        D = self.dev()
        node, stack, sp, best, nb, ne, odg, invg, gend = c
        ps = node == -2
        nd = jnp.maximum(node, 0)

        def box(n_):
            return bvh.slab(D["lo"][n_], D["hi"][n_], odg, invg)

        tin, tout = box(nd)
        lim = jnp.minimum(best[0], ctx.t_bound) + ctx.band
        hit = ps | ((tin <= tout) & (tout >= ctx.t_lo) & (tin <= lim))
        leaf = ps | (D["count"][nd] > 0)
        n_ev = jnp.where(ps, 1, jnp.where(hit & leaf, D["count"][nd], 0))
        best = lax.fori_loop(0, n_ev, lambda k, b: self._visit(ctx, jnp.where(ps, vi_, D["items"][D["start"][nd] + k]),
                                                               b, ps), best)  # fmt: skip
        last_n = jnp.maximum(gend - 1, 0)
        lc = nd + 1
        rc = D["skip"][jnp.minimum(lc, last_n)]
        tl, _ = box(jnp.minimum(lc, last_n))
        tr, _ = box(jnp.minimum(rc, last_n))
        near, far = jnp.where(tl <= tr, lc, rc), jnp.where(tl <= tr, rc, lc)
        inner = hit & ~leaf
        stack = jnp.where(inner, stack.at[jnp.minimum(sp, STACK - 1)].set(far), stack)
        sp = jnp.where(inner, jnp.minimum(sp + 1, STACK), sp)
        sp2 = jnp.where(inner, sp, sp - 1)
        nxt = jnp.where(inner, near, jnp.where(sp2 >= 0, stack[jnp.maximum(sp2, 0)], I32(-1)))
        return jnp.where(ps, I32(-1), nxt), stack, sp2, best, nb + 1, ne + n_ev, odg, invg, gend

    def _group(self, ctx, v, vi_, g, c):
        """Traverse one frame group of v (a group index below vg0[v]: the own pseudo-group, item 0)."""
        D = self.dev()
        best, nb, ne = c
        pg = g < D["vg0"][v]
        gg = jnp.maximum(g, 0)
        Rg = D["gR"][gg]
        odg, ddg = Rg.T @ ctx.od, Rg.T @ ctx.dd
        invg = 1.0 / jnp.where(jnp.abs(ddg) < TINY_DIR, jnp.where(ddg < 0, -TINY_DIR, TINY_DIR), ddg)
        r0 = D["g0"][gg]
        start = jnp.where(pg, I32(-2), jnp.where(r0 < D["g1"][gg], r0, I32(-1)))
        c0 = (start, jnp.zeros(STACK, I32), I32(0), best, nb, ne, odg, invg, D["g1"][gg])
        _, _, _, best, nb, ne, *_ = lax.while_loop(lambda c_: (c_[0] >= 0) | (c_[0] == -2),
                                                   lambda c_: self._node(ctx, vi_, c_), c0)  # fmt: skip
        return best, nb, ne

    def enter(self, v, ctx, own=False, vi=None):
        """Return the photon's next event in volume v: ((t, item, leaf, face, zero, is_own), nodes, evals).

        Its own exit (``own``: item 0, before the daughters' BVHs) and the first entry into a daughter after t_lo
        (before t_bound); zero: a zero-length event at t_lo. ``ctx.sd`` = (known, leaf, face, s, entry): the photon
        sits on item ``last``'s face (leaf, face), s = d . n_out there, entry: d . grad < 0.
        """
        D = self.dev()
        ctx = ctx._replace(shear=_shear(ctx.dd) if self.n_mesh else None)
        none = (F32(INF), I32(-1), I32(-1), I32(-1), jnp.bool_(False), jnp.bool_(False))
        vi_ = I32(0) if vi is None else jnp.asarray(vi, I32)
        g_lo = D["vg0"][v] - jnp.asarray(own, I32)  # one pseudo-group first: the own volume (item 0)
        return lax.fori_loop(g_lo, D["vg1"][v], lambda g, c: self._group(ctx, v, vi_, g, c), (none, I32(0), I32(0)))

    # ---------------------------------------------------------------------------------------- surface identity
    def srf_decode(self, s):
        """Return (item, leaf, face, o, known, e) of a surface code (``surface_id.srf_decode``)."""
        return surface_id.srf_decode(self, s)

    def srf_pack(self, item, leaf, face, o, e):
        """Return the surface code of (item, leaf, face, o, e) (``surface_id.srf_pack``)."""
        return surface_id.srf_pack(self, item, leaf, face, o, e)

    def hit_grad(self, item, leaf, face, x):
        """Return the winning surface's gradient at x (``surface_id.hit_grad``)."""
        return surface_id.hit_grad(self, item, leaf, face, x)

    def live_t(self, item, leaf, face, o, d, bt):
        """Return the live crossing distance (``surface_id.live_t``)."""
        return surface_id.live_t(self, item, leaf, face, o, d, bt)

    # ---------------------------------------------------------------------------------------- the query
    def _side(self, q, last_, t_lo):
        """Return the side state (known, leaf, face, s, entry) of the srf a round starts on (s = d . n_out)."""
        l_it, l_lf, l_fc, l_o, l_kn, l_e = self.srf_decode(last_)
        od, dd = q.od, q.dd
        if q.snv is None:

            def s_of():  # s = d . n_out at the start: one leaf gradient of the item
                g = self.hit_grad(l_it, jnp.maximum(l_lf, 0), l_fc, od + t_lo * dd)
                gd = jnp.dot(g, dd)
                return gd / jnp.maximum(jnp.linalg.norm(g), NORM_FLOOR) * jnp.where(l_o, F32(1.0), F32(-1.0)), gd < 0

            s_, en_ = pick(l_kn, s_of, lambda: (F32(0.0), jnp.bool_(False)))
        else:
            # the event's own normal (against the way in): n_out = +normal for an entry, -normal for an exit
            s_ = jnp.where(l_kn, jnp.dot(q.snv, dd) * jnp.where(l_e, F32(1.0), F32(-1.0)), F32(0.0))
            en_ = l_kn & (s_ * jnp.where(l_o, F32(1.0), F32(-1.0)) < 0)
        return l_it, (l_kn, l_lf, l_fc, s_, en_)

    def _next_volume(self, q, ev, v, vi, step):
        """Return (the volume after the round's event, whether it was relocated, StepTooSmall)."""
        D = self.dev()
        OUT = I32(self.n)
        t_c, u_in, is_own, z_in, t_lo, band, l_it, it, hit = ev
        has_own = v < OUT
        go_in = ~is_own
        # StepTooSmall: the photon was just located in a coincident partner (its srf the member it left, v that one's
        # partner) and its first event is v's own zero-length exit: a straight continue event, not an interface
        l_c = jnp.clip(l_it, 0, OUT - 1)
        stt = ((it == 0) & has_own & is_own & (z_in | (t_c - t_lo <= rules.K_ON * band)) & (l_it >= 0) & (l_it < OUT)
               & (D["coin_hi"][l_c] == v) & hit)  # fmt: skip
        is_ch = u_in >= self.CH0
        v_in = jnp.where(is_ch, OUT + 1 + D["cmesh"][jnp.clip(u_in - self.CH0, 0, D["cmesh"].shape[0] - 1)], u_in)
        nxt = jnp.where(go_in, v_in, D["parent"][jnp.minimum(v, D["parent"].shape[0] - 1)])
        unit_x = jnp.where(go_in, jnp.where(is_ch, I32(-1), u_in), vi)
        # leaving a volume with a coincident partner: located in the partner (zero-length), as Geant4 does
        co = jnp.where(~go_in & has_own & ~stt, D["coin_hi"][vi], I32(-1))
        nxt, moved = pick(hit & ~(go_in & is_ch) & (co < 0),
                          lambda: self._settle(nxt, (q.od, q.dd, t_c + step), unit_x, step + band),
                          lambda: (nxt, jnp.bool_(False)))  # fmt: skip
        nxt = jnp.where(hit & (co >= 0), co, nxt)
        return nxt, moved, stt

    def _reflect_volume(self, q, ev, v, vi, nxt):
        """Return the volume a reflection here stays in (Geant4 keeps the photon where it was located).

        After a crossing of a face of nxt that is a zero-length step back into v (StepTooSmall); a photon relocated
        into a volume already holding the point (an overlapping or coincident sibling, a daughter of one) stays in it
        -- not one whose solid holds the point only because the import lowered its holes.
        """
        D = self.dev()
        OUT = I32(self.n)
        t_c, is_own, hit, step, band = ev
        anc = lax.fori_loop(0, MAX_ANCESTORS, lambda k_, c_: (jnp.where(c_[0] < OUT, D["parent"][jnp.minimum(
            c_[0], OUT - 1)], c_[0]), c_[1] | (c_[0] == nxt)), (vi, jnp.bool_(False)))[1]  # fmt: skip
        n_c = jnp.minimum(nxt, OUT - 1)
        reloc = hit & is_own & (v < OUT) & (nxt < OUT) & (nxt != vi) & ~anc
        # the point just before the crossing: two bands back at the crossing's own scale
        t_b = t_c - 2.0 * rules.band(q.xa, t_c)
        owns = pick(reloc, lambda: self.owns_ray(n_c, q.od, q.dd, t_b, step + band), lambda: jnp.bool_(False))
        return jnp.where(owns, nxt, v)

    def _round(self, q, c):
        """One round of a query: the next event in the volume, the volume after it, and whether it is visible."""
        D = self.dev()
        OUT = I32(self.n)
        t_lo, v, last_, _, it, rec, nb, ne, cur, pz = c
        band = rules.band(q.xa, t_lo)
        step = band
        vi = jnp.minimum(v, OUT - 1)
        has_own = v < OUT
        # last_: the photon's srf (round 0: the item it sits on, with its face and outward sign when known), then the
        # item each crossing of this query made (the band rules)
        l_it, side = self._side(q, last_, t_lo)
        ctx = _Enter(q.od, q.dd, t_lo, q.tmax, l_it, band, side, None)
        (t_c, u_in, l_in, f_in, z_in, is_own), nb1, ne1 = self.enter(v, ctx, own=has_own, vi=vi)
        hit = t_c < q.tmax
        nxt, moved, stt = self._next_volume(q, (t_c, u_in, is_own, z_in, t_lo, band, l_it, it, hit), v, vi, step)
        v_refl = self._reflect_volume(q, (t_c, is_own, hit, step, band), v, vi, nxt)
        item = jnp.where(~is_own, u_in, vi)
        after = D["cls"][jnp.where(hit, nxt, v)]
        vis = hit & (after != cur)
        if q.pass_tab is not None:  # a pass pair (one medium, the PASS kernel) is no vertex either
            P = q.pass_tab
            npl = P.shape[0]
            vis = vis & ~((cur < npl) & (after < npl) & P[jnp.clip(cur, 0, npl - 1), jnp.clip(after, 0, npl - 1)])
        # every invisible crossing (the same class, or a pass pair) is recorded alike -- the first and the last, their
        # count, and the volume the run ends in -- so the step places a bulk event past any of them
        inv = hit & ~vis
        pz = (jnp.where(inv & (pz[0] == F32(INF)), t_c, pz[0]), jnp.where(inv, t_c, pz[1]), pz[2] + inv.astype(I32),
              jnp.where(inv, nxt, pz[3]), hit)  # fmt: skip
        vis = vis | stt
        done = (~hit) | vis | (it + 1 >= q.max_rounds)
        rec = jnp.where(hit, jnp.stack([lax.bitcast_convert_type(t_c, I32), item, l_in, f_in, z_in.astype(I32), v,
                                        vis.astype(I32), stt.astype(I32), v_refl]), rec)  # fmt: skip
        return (jnp.where(moved, t_c + step, t_c), jnp.where(hit, nxt, v), item, done, it + 1, rec, nb + nb1,
                ne + ne1, jnp.where(hit, after, cur), pz)  # fmt: skip

    def _finish(self, q, o, d, end):
        """Return the query's outputs from the last round's record: the live t, the normal, the surface met."""
        D = self.dev()
        vol, cls0, v_after, rounds, rec, cls_after, pz = end
        bt, bi, bl, bf, zero, vb = lax.bitcast_convert_type(rec[0], F32), rec[1], rec[2], rec[3], rec[4] > 0, rec[5]
        # hit = a visible crossing, or (cont) the rounds ran out on invisible ones with one in the last round: a
        # straight event at the last crossing; a scan that ends on invisible crossings and then nothing is no hit
        vis_end = rec[6] == 1
        cont = (~vis_end & pz[4] & (rounds >= q.max_rounds)) | (rec[7] == 1)
        hit = (vis_end | cont) & (bt < q.tmax)
        live = hit & ~zero & (bl >= 0)
        if self.n_mesh:
            bt = surface_id.mesh_plane_t(self, (live, bi, bl, bt), q.od, q.dd, q.xa)
        # the live t of every event on a known face, a zero-length one too
        lt = hit & (bl >= 0)
        bt_live = jnp.where(lt, self.live_t(bi, bl, bf, o, d, jnp.where(lt, bt, F32(INF))), bt)
        g_hit = self.hit_grad(bi, jnp.maximum(bl, 0), bf, o + jnp.where(bt_live < INF, bt_live, 0.0) * d)
        nrm = g_hit / jnp.maximum(jnp.linalg.norm(g_hit), NORM_FLOOR)
        nrm = jnp.where(hit & (bl >= 0), nrm, -d)  # a face known (a zero-length event on the face sat on too)
        # srf: the item met with its face and outward sign -- an entry heads against n_out, an exit along it
        gd = jnp.dot(g_hit, d)
        srf = jnp.where(hit, self.srf_pack(bi, bl, bf, jnp.where(bi != vb, gd < 0, gd > 0), bi != vb), I32(-1))
        nrm = jnp.where(jnp.sum(nrm * d) > 0, -nrm, nrm)
        # vol_before: the volume just before the visible crossing; without a hit, where the invisible run ends
        vb = jnp.where(hit, vb, pz[3])
        # a mesh's back facet met first: the photon is inside the opaque solid -- the crossing leaves the solid's class
        inside_m = hit & (bi >= self.CH0) & (bf == -2)
        cls_b = jnp.where(inside_m, cls_after, D["cls"][vb])
        cls_a = jnp.where(inside_m, D["cls"][vb], jnp.where(hit, cls_after, D["cls"][vb]))
        return dict(t=jnp.where(hit, bt_live, q.tmax), hit=hit, cont=cont,
                    vol_after=jnp.where(hit & ~inside_m, v_after, jnp.where(inside_m, vb, vol)), vol_before=vb,
                    vol_reflect=jnp.where(hit, rec[8], vb), vol_start=vol, cls_before=cls_b, cls_start=cls0,
                    cls_after=cls_a, normal=nrm,
                    sensor=jnp.where(hit, D["sen"][jnp.where(hit, v_after, vol)], I32(-1)),
                    item=jnp.where(hit, bi, -1), leaf=bl, face=bf, zero=hit & zero, t_pass0=pz[0], t_pass1=pz[1],
                    n_pass=pz[2], srf=srf)  # fmt: skip

    def query(self, o, d, vol, last=-1, t_max=1e3, pass_tab=None, counts=False, max_rounds=MAX_ROUNDS, sn=None):  # noqa: PLR0913 -- the step's query
        """Return the first visible crossing along (o, d) for a photon in volume vol, as a dict.

        Keys: t, hit, cont, vol_after, vol_before, vol_reflect, vol_start, cls_before, cls_start, cls_after, normal,
        sensor, item, leaf, face, zero, t_pass0, t_pass1, n_pass, srf (+ nodes, evals, rounds with ``counts``).
        ``last``: the srf a query returned (the item with its face and outward sign: the side is the state, the
        direction decides), or a plain item, or -1. ``sn``: the normal that query gave (s = d . n_out from it).
        ``pass_tab``: (C, C) bool, the class pairs crossed without a vertex.
        """
        D = self.dev()
        o, d = jnp.asarray(o, F32), jnp.asarray(d, F32)
        od, dd = sg(o), sg(d)
        vol = jnp.asarray(vol, I32)
        cls0 = D["cls"][vol]
        snv = None if sn is None else sg(jnp.asarray(sn, F32))
        P = None if pass_tab is None else jnp.asarray(pass_tab)
        q = _Query(od, dd, F32(t_max), jnp.max(jnp.abs(od)), snv, P, max_rounds)
        rec0 = jnp.stack([lax.bitcast_convert_type(F32(INF), I32), I32(-1), I32(-1), I32(-1), I32(0), vol, I32(0),
                          I32(0), vol])  # fmt: skip
        c0 = (F32(0.0), vol, jnp.asarray(last, I32), jnp.bool_(False), I32(0), rec0, I32(0), I32(0), cls0,
              (F32(INF), F32(INF), I32(0), vol, jnp.bool_(False)))  # fmt: skip
        _, v_after, _, _, rounds, rec, nb, ne, cls_after, pz = lax.while_loop(
            lambda c: ~c[3], lambda c: self._round(q, c), c0
        )
        out = self._finish(q, o, d, (vol, cls0, v_after, rounds, rec, cls_after, pz))
        if counts:
            out.update(nodes=nb, evals=ne, rounds=rounds)
        return out


# ------------------------------------------------------------------------------------------------- the functions
def locate(nav, x, start=None):
    """Return the deepest volume holding x (``nav.n``: outside every volume); a known ``start`` (>= 0) as is."""
    return nav.locate(x, start)


def next_crossing(nav, o, d, vol, srf=-1, sn=None, t_max=1e3, pass_tab=None):  # noqa: PLR0913 -- the photon's state
    """Return the next visible crossing (a Crossing) of a photon at o, heading d, in volume vol, on surface srf."""
    r = nav.query(o, d, vol, last=srf, t_max=t_max, pass_tab=pass_tab, sn=sn)
    win = jnp.stack([r["item"], r["leaf"], r["face"], r["zero"].astype(I32)])
    return Crossing(t=r["t"], hit=r["hit"], normal=r["normal"], vol_start=r["vol_start"], vol_before=r["vol_before"],
                    vol_after=r["vol_after"], vol_reflect=r["vol_reflect"], srf=r["srf"], step_too_small=r["cont"],
                    winner=win, cls_start=r["cls_start"], cls_before=r["cls_before"], cls_after=r["cls_after"],
                    sensor=r["sensor"], zero=r["zero"], t_pass0=r["t_pass0"], t_pass1=r["t_pass1"],
                    n_pass=r["n_pass"])  # fmt: skip


def surface_at(nav, cx, x, d):
    """Return (t, normal) of the surface crossing cx met, as functions of the ray (x, d), the winner held.

    The values and derivatives the query itself gives (its traversal is detached; only the winner's live t and normal
    carry derivatives), without walking again. ``cx`` is a Crossing, or a mapping with ``t``, ``hit`` and ``win``.
    """
    if isinstance(cx, Crossing):
        t_q, hit, win = cx.t, cx.hit, cx.winner
    else:
        t_q, hit, win = cx["t"], cx["hit"], cx["win"]
    it, lf, fc = win[0], win[1], win[2]
    x, d = jnp.asarray(x, F32), jnp.asarray(d, F32)
    live = hit & (lf >= 0)  # a zero-length event on a known face too (the query's)
    t0 = sg(t_q)
    t = jnp.where(live, nav.live_t(it, lf, fc, x, d, jnp.where(live, t0, INF)), t0)
    g = nav.hit_grad(it, jnp.maximum(lf, 0), fc, x + jnp.where(t < INF, t, 0.0) * d)
    n = g / jnp.maximum(jnp.linalg.norm(g), NORM_FLOOR)
    n = jnp.where(hit & (lf >= 0), n, -d)
    return t, jnp.where(jnp.sum(n * d) > 0, -n, n)
