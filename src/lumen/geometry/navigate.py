"""The navigator (device): where a photon is, and the next visible crossing along its ray, by Geant4's volume tracking.

The photon carries its placed volume ``vol`` (one solid per volume, daughters inside their mothers:
``lumen.build.placements`` checks it; a region overlapping siblings share is its owner's:
``lumen.geometry.ownership``) and the surface it sits on (``srf``, ``lumen.geometry.surface_id``). A query is Geant4's
step: the nearer of the exit of vol's own solid and the first entry into one of vol's daughters or owners (a BVH over
each volume's candidates); the next volume from the tree (the candidate entered, or the mother, then relocated by
``lumen.rules.navigation``); a crossing that is no vertex (``vertex_tab``: the same class, or a pass pair; but into
another volume of a class whose own pair has a surface, not another piece of one split union) continues the scan.

Tolerance by surface identity, not by a push: the photon starts at its hit point and passes ``srf``; an event of that
item within the band of the start is that same crossing; any other item's event counts, however close. A start state
that disagrees with the ray (vol's solid does not hold the start, or a daughter does) is resolved as Geant4 resolves
it: a zero-length exit to the mother, or a zero-length entry into that daughter.

    nav = Navigator.from_scene(scene)                      # host: build (or read the build cache)
    vol = nav.locate(x)                                     # jnp, one thread (or a jax.vmap batch)
    r = nav.query(o, d, vol, t_max=t_max, last=-1)          # the first visible crossing (t_max: m), a dict
    t, n = surface_at(nav, cx, o, d)  # cx: r's t, hit and win (item, leaf, face, zero): the winner's (t, normal), live

A query starts at its origin. ``t`` is the distance along the ray (m) throughout this module.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from lumen._core.safe import safe_norm
from lumen.geometry import bvh, navigator_build, surface_id
from lumen.geometry.csg import Csg, pick
from lumen.rules import navigation as rules
from lumen.rules.tolerances import TINY_DIR
from lumen.units import HUGE

F32, I32 = jnp.float32, jnp.int32
STACK = navigator_build.STACK
sg = lax.stop_gradient

__all__ = ["Navigator", "surface_at"]


class _Enter(NamedTuple):
    """The fixed inputs of one volume visit (``Navigator.enter``); ``v`` the volume visited."""

    v: object
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
    vtx: object
    s_at: object


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
        self.host = dict(arrays.host)
        self.n, self.n_mesh, self.chunk = arrays.n, arrays.n_mesh, arrays.chunk
        self.CH0, self.SRF = arrays.ch0, arrays.srf
        self.names, self.defs, self.stats = list(arrays.names), list(arrays.defs), dict(arrays.stats)
        self.origins, self.pieces = list(arrays.origins), list(arrays.pieces)
        self.owned, self.kids = list(arrays.owned), arrays.kids
        self.clipped, self.thin = list(arrays.clipped), list(arrays.thin)
        # what the geometry needs compiled: ancestor exits where a volume reaches out
        self.clipping = bool(self.clipped)
        self.lo_w, self.hi_w = arrays.lo_w, arrays.hi_w
        self._D = None
        self._live = None
        self.build_key = None  # the build cache's key of these arrays (lumen.geometry.cache.key), when built through it

    @classmethod
    def from_scene(cls, scene, cache=False, **options):
        """Return the navigator of a lowered Scene; ``cache``: read or write the cache (``lumen.geometry.cache``)."""
        if cache:
            from lumen.geometry import cache as build_cache

            arrays, k = build_cache.nav_arrays(scene, **options)
            nav = cls(arrays)
            nav.build_key = k
            return nav
        return cls(navigator_build.build(scene, **options))

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

    def is_ancestor(self, i, v):
        """Return whether volume i is an ancestor of volume v (a candidate v reaches out of: its face is v's exit)."""
        if not self.clipping:
            return jnp.bool_(False)
        D = self.dev()
        n = self.n
        ic = jnp.clip(i, 0, n - 1)
        return (i >= 0) & (i < v) & (v < n) & (v < i + D["size"][ic])

    def rank(self, i):
        """Return the rank of candidate item i: minus its place in the rule's order (``prio``; a mesh chunk its mesh's),
        so the volume a point belongs to ranks highest among the candidates holding it."""  # noqa: D205, D209
        D = self.dev()
        j = jnp.where(i >= self.CH0, self.n + 1 + D["cmesh"][jnp.clip(i - self.CH0, 0, D["cmesh"].shape[0] - 1)], i)
        return jnp.where(i >= 0, -D["prio"][jnp.clip(j, 0, D["prio"].shape[0] - 1)], I32(0))

    def _find(self, v, x):
        """Return the daughter (or owner) of v whose solid holds the point x, or -1 (Geant4's LevelLocate: the first in
        the rule's order)."""  # noqa: D205, D209
        n = self.n

        def visit(i, g):
            ok = rules.level_locate_wins(i, g, I32(-1), n, self.rank(i), self.rank(g)) & ~self.is_ancestor(i, v)
            return jnp.where(ok & self.holds_point(jnp.minimum(i, n - 1), x), i, g)

        return self.scan(v, x, visit, I32(-1), lambda g: jnp.bool_(False))

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
        return lax.while_loop(lambda c: ~c[1], body, (v0, known, I32(0)))[0]  # each move goes down, or to an owner

    def _find_ray(self, v, od, dd, t, excl, pad):
        """Return the daughter of v (other than excl) holding od + t dd (just after t, along the ray), or -1."""
        n = self.n
        x = od + t * dd

        def visit(i, g):  # Geant4's LevelLocate: the first in the rule's order
            ok = rules.level_locate_wins(i, g, excl, n, self.rank(i), self.rank(g)) & ~self.is_ancestor(i, v)
            return jnp.where(ok & self.ray_holds(jnp.minimum(i, n - 1), od, dd, t), i, g)

        return self.scan(v, x, visit, I32(-1), lambda g: jnp.bool_(False), pad)

    def _settle(self, v, ray, crossed, pad):
        """Return (the volume holding the ray just after t, from v's neighbourhood; whether it moved).

        Up while v's solid does not hold it, then down into a daughter or an owner holding it (an owner first: the
        candidates' rank), as Geant4's LocateGlobalPointAndSetup (a daughter's part outside its mother is not found
        from outside the mother; a zero-length volume is passed through). The item just crossed is decided by the
        crossing itself, never re-entered or left here. ``ray`` = (od, dd, t).
        """
        od, dd, t = ray
        D = self.dev()
        n = self.n

        def body(c):  # held: v_ is the daughter _find_ray found holding the ray (its own test, not repeated)
            v_, _, k, held = c
            vi = jnp.minimum(v_, n - 1)
            test = ~held & (v_ < n) & (v_ != crossed)
            ins = ~test | pick(test, lambda: self.ray_holds(vi, od, dd, t), lambda: jnp.bool_(True))
            f = pick(ins, lambda: self._find_ray(v_, od, dd, t, crossed, pad), lambda: I32(-1))
            nv = jnp.where(ins, jnp.where(f >= 0, f, v_), D["parent"][jnp.minimum(v_, D["parent"].shape[0] - 1)])
            return nv, (f < 0) & ins, k + 1, ins & (f >= 0)

        v2, *_ = lax.while_loop(lambda c: ~c[1], body, (v, jnp.bool_(False), I32(0), jnp.bool_(False)))  # up, then down
        return v2, v2 != v

    # ---------------------------------------------------------------------------------------- the daughter entry
    def _crossing(self, ctx, k):
        """Return (whether the ray crosses triangle k, its t, n . d, the start's height s0 over its plane)."""
        D = self.dev()
        od, dd, t_lo = ctx.od, ctx.dd, ctx.t_lo
        kx, ky, kz, Sx, Sy, Sz = ctx.shear
        A, B, C = D["V0"][k] - od, D["V1"][k] - od, D["V2"][k] - od

        def sh(P):  # noqa: N803 -- P a triangle's vertex
            return (_pk(P, kx) - Sx * _pk(P, kz), _pk(P, ky) - Sy * _pk(P, kz), Sz * _pk(P, kz))

        (Ax, Ay, _), (Bx, By, _), (Cx, Cy, _) = sh(A), sh(B), sh(C)
        U = Cx * By - Cy * Bx
        V_ = Ax * Cy - Ay * Cx
        W_ = Bx * Ay - By * Ax
        inside = ((U >= 0) & (V_ >= 0) & (W_ >= 0)) | ((U <= 0) & (V_ <= 0) & (W_ <= 0))
        det = U + V_ + W_
        # the sheared test decides inside / outside only (watertight); t comes from the plane: the start's height s0
        # over it (outward +) over the slope (the test's own t interpolates the vertices' depths metres away)
        Nk = D["N"][k]
        s0 = jnp.dot(Nk, od + t_lo * dd - D["V0"][k])
        nd = jnp.dot(Nk, dd)
        t_k = t_lo - s0 / jnp.where(nd != 0, nd, F32(-1.0))
        return inside & (det != 0), t_k, nd, s0

    def _triangle(self, ctx, i, k0, kn, b):
        """Return the best event after testing one triangle of a mesh chunk (the facet rule); ``b`` = (q, best)."""
        q, bb, hi_t = b
        t_lo, band = ctx.t_lo, ctx.band
        k = k0 + jnp.minimum(q, kn - 1)
        crosses, t_k, nd, s0 = self._crossing(ctx, k)
        ok = crosses & (q < kn)
        # the facet rule: met entering it (n . d < 0), a start behind its plane within the tolerance on it; a facet
        # met from clearly behind (a start inside the closed surface, where no volume holds a point as the mesh's) is
        # an entry into the opaque part alike
        tol = F32(rules.FACET_BEHIND_BANDS) * band
        front = ok & (nd < 0) & (s0 > -tol) & (t_lo - t_k <= band)
        back = ok & (nd > 0) & (s0 < -tol) & (t_k > t_lo)
        t_c = jnp.maximum(t_k, t_lo)
        better = (front | back) & (t_c < hi_t) & ((t_c < bb[0]) | ((t_c == bb[0]) & bb[5]))
        return _choose(better, (t_c, i, k, I32(-1), t_k <= t_lo, jnp.bool_(False)), bb)

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
        """Return whether the start is on the first change's face (the on-face rule: within K_ON bands of it).

        A change within K_ON bands along the ray is on it; one up to GRAZE times farther (a grazing ray) is on it when
        the start's own distance from the face is within K_ON bands (``surface_id.face_dist``), unless the ray crosses
        the face the other way at the start, steeper than GRAZE resolves (a ray heading into a curved solid from its
        face, whose chord ends on that face again: that end is a later crossing).
        """
        dt = fr["t"] - ctx.t_lo
        lim = rules.K_ON * ctx.band

        def grazing():  # between lim and GRAZE lim along the ray: the start's own distance from the face decides
            x0 = ctx.od + ctx.t_lo * ctx.dd
            lf = jnp.maximum(fr["leaf"], 0)
            g0 = surface_id.hit_grad(self, i, lf, fr["face"], x0)
            gc = surface_id.hit_grad(self, i, lf, fr["face"], ctx.od + fr["t"] * ctx.dd)
            s0 = jnp.dot(g0, ctx.dd) / safe_norm(g0)
            other_way = (s0 * jnp.dot(gc, ctx.dd) < 0) & (jnp.abs(s0) > 1.0 / rules.GRAZE)
            return (surface_id.face_dist(self, i, lf, fr["face"], x0) < lim) & ~other_way

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
        t_hi = jnp.where(tor, t_on, F32(HUGE))
        # while the ray is within that face's band its leaf is on the side the photon crossed to, whatever a float32
        # probe there says (Geant4: within kCarTolerance the direction decides)
        lf_on = jnp.where(kn_i, sd[1], I32(-2))
        r = self.E.first_change(k, ol, dl, ctx.t_lo, bnd, want=~is_own, init=(kn_i, is_own),
                                skip=(lf_on, sd[2], sd[4], t_hi), on_face=(lf_on, sd[4], t_on))  # fmt: skip
        return r, kn_i

    def _unit_visit(self, ctx, i, b, is_own):
        """Return the best event after volume i: own (item 0) its first exit, a daughter or an owner its first entry.

        An ancestor the volume is not proved inside: its first exit from a start inside it, when earlier than every
        other event (a start outside it lies on its face where the volume touches it: no event; the volume's own exit
        wins a tie).

        A zero-length exit if the start is outside own, a zero-length entry if it is inside a daughter
        (never the item just crossed: its identity). The item sat on (side known): in it iff own; its direction
        decides.
        """
        sd = ctx.sd
        clip = ~is_own & self.is_ancestor(i, ctx.v)
        own = is_own
        is_own = is_own | clip  # an exit: the first change of a solid the photon is in
        r, kn_i = self._unit_change(ctx, i, b, is_own)
        h, t, lf, f, in0 = r["hit"], r["t"], r["leaf"], r["face"], r["inside0"]
        fr = r["first"]
        on = self._on_face(ctx, i, fr, kn_i)
        # the item sat on: in own unless heading out (s > 0) -- or its float32 ray has no exit from it at all; in a
        # daughter only heading in
        by_s = kn_i & jnp.where(is_own, sd[3] > 0, sd[3] < 0)
        ins = jnp.where(kn_i, jnp.where(is_own, (sd[3] <= 0) & h, sd[3] < 0), jnp.where(on, fr["inside"], in0))
        h = h & (is_own | ~ins) & (~clip | ins)  # a daughter the start is in: its zero-length entry or none
        zero = jnp.where(is_own, ~ins & ~clip, ins & ((i != ctx.last) | kn_i))
        # a zero-length event's face: the face sat on when s decided it, the face at the start when the on-face rule
        # did; else none
        lf = jnp.where(zero, jnp.where(by_s, sd[1], jnp.where(on, fr["leaf"], I32(-1))), lf)
        f = jnp.where(zero, jnp.where(by_s, sd[2], jnp.where(on, fr["face"], I32(-1))), f)
        te = jnp.where(zero, ctx.t_lo, jnp.where(h, t, F32(HUGE)))
        # own is item 0: it sets the best (its exit, HUGE if none); a daughter beats it on a tie (the entry-tie rule)
        wins = rules.entry_wins(te, self.rank(i), b, self.rank(b[1]))
        better = own | (clip & h & (te < b[0])) | (~is_own & (zero | h) & wins)
        return _choose(better, (te, i, lf, f, zero, own), b)

    def _visit(self, ctx, i, b, is_own):
        if self.n_mesh:
            return pick((i >= self.CH0) & ~is_own, lambda: self._mesh_visit(ctx, i, b),
                        lambda: self._unit_visit(ctx, jnp.minimum(i, self.n - 1), b, is_own))  # fmt: skip
        return self._unit_visit(ctx, i, b, is_own)

    def _meets(self, ctx, n_, odg, invg, best):
        """Return (whether the ray meets node n_'s box after t_lo and within the best event so far, its entry t)."""
        D = self.dev()
        tin, tout = bvh.slab(D["lo"][n_], D["hi"][n_], odg, invg)
        return (tin <= tout) & (tout >= ctx.t_lo) & (tin <= jnp.minimum(best[0], ctx.t_bound) + ctx.band), tin

    def _node(self, ctx, vi_, c):
        """One step of a group's BVH traversal from a node whose box the ray meets (node -2: the volume's own solid,
        item 0): a leaf visits its items; an inner node goes on to the children whose boxes the ray meets, the nearer
        first, the farther onto the stack with its entry t; then the stack's top, its entries past the best event
        dropped. The items are visited in the order, and against the best events, of a walk that tests each node's box
        when it reaches it."""  # noqa: D205, D209
        D = self.dev()
        node, stack, tins, sp, best, odg, invg, gend = c
        ps = node == -2
        nd = jnp.maximum(node, 0)
        leaf = ps | (D["count"][nd] > 0)
        n_ev = jnp.where(ps, 1, jnp.where(leaf, D["count"][nd], 0))
        best = lax.fori_loop(0, n_ev, lambda k, b: self._visit(ctx, jnp.where(ps, vi_, D["items"][D["start"][nd] + k]),
                                                                b, ps), best)  # fmt: skip
        last_n = jnp.maximum(gend - 1, 0)
        lc = nd + 1
        rc = D["skip"][jnp.minimum(lc, last_n)]
        hl, tl = self._meets(ctx, jnp.minimum(lc, last_n), odg, invg, best)
        hr, tr = self._meets(ctx, jnp.minimum(rc, last_n), odg, invg, best)
        first = tl <= tr
        near, far = jnp.where(first, lc, rc), jnp.where(first, rc, lc)
        h_near, h_far = ~leaf & jnp.where(first, hl, hr), ~leaf & jnp.where(first, hr, hl)
        push = h_near & h_far
        top = jnp.minimum(sp, STACK - 1)
        stack = jnp.where(push, stack.at[top].set(far), stack)
        tins = jnp.where(push, tins.at[top].set(jnp.where(first, tr, tl)), tins)
        sp = jnp.where(push, jnp.minimum(sp + 1, STACK), sp)
        lim = jnp.minimum(best[0], ctx.t_bound) + ctx.band
        sp_c = lax.while_loop(lambda s: (s > 0) & (tins[jnp.maximum(s - 1, 0)] > lim), lambda s: s - 1, sp)
        down = h_near | h_far
        popped = jnp.where(sp_c > 0, stack[jnp.maximum(sp_c - 1, 0)], I32(-1))
        nxt = jnp.where(down, jnp.where(h_near, near, far), popped)
        sp = jnp.where(down, sp, jnp.maximum(sp_c - 1, 0))
        return jnp.where(ps, I32(-1), nxt), stack, tins, sp, best, odg, invg, gend

    def _group(self, ctx, v, vi_, g, c):
        """Traverse one frame group of v (a group index below vg0[v]: the own pseudo-group, item 0)."""
        D = self.dev()
        best = c
        pg = g < D["vg0"][v]
        gg = jnp.maximum(g, 0)
        Rg = D["gR"][gg]
        odg, ddg = Rg.T @ ctx.od, Rg.T @ ctx.dd
        invg = 1.0 / jnp.where(jnp.abs(ddg) < TINY_DIR, jnp.where(ddg < 0, -TINY_DIR, TINY_DIR), ddg)
        r0 = D["g0"][gg]
        root = (r0 < D["g1"][gg]) & self._meets(ctx, jnp.minimum(r0, D["lo"].shape[0] - 1), odg, invg, best)[0]
        start = jnp.where(pg, I32(-2), jnp.where(root, r0, I32(-1)))
        c0 = (start, jnp.zeros(STACK, I32), jnp.zeros(STACK, F32), I32(0), best, odg, invg, D["g1"][gg])
        _, _, _, _, best, *_ = lax.while_loop(lambda c_: (c_[0] >= 0) | (c_[0] == -2),
                                              lambda c_: self._node(ctx, vi_, c_), c0)  # fmt: skip
        return best

    def enter(self, v, ctx, own=False, vi=None):
        """Return the photon's next event in volume v: (t, item, leaf, face, zero, is_own).

        Its own exit (``own``: item 0, before the daughters' BVHs) and the first entry into a daughter after t_lo
        (before t_bound); zero: a zero-length event at t_lo. ``ctx.sd`` = (known, leaf, face, s, entry): the photon
        sits on item ``last``'s face (leaf, face), s = d . n_out there, entry: d . grad < 0.
        """
        D = self.dev()
        ctx = ctx._replace(shear=_shear(ctx.dd) if self.n_mesh else None)
        none = (F32(HUGE), I32(-1), I32(-1), I32(-1), jnp.bool_(False), jnp.bool_(False))
        vi_ = I32(0) if vi is None else jnp.asarray(vi, I32)
        g_lo = D["vg0"][v] - jnp.asarray(own, I32)  # one pseudo-group first: the own volume (item 0)
        return lax.fori_loop(g_lo, D["vg1"][v], lambda g, c: self._group(ctx, v, vi_, g, c), none)

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
                return gd / safe_norm(g) * jnp.where(l_o, F32(1.0), F32(-1.0)), gd < 0

            s_, en_ = pick(l_kn, s_of, lambda: (F32(0.0), jnp.bool_(False)))
        else:
            # the event's own normal (against the way in): n_out = +normal for an entry, -normal for an exit
            s_ = jnp.where(l_kn, jnp.dot(q.snv, dd) * jnp.where(l_e, F32(1.0), F32(-1.0)), F32(0.0))
            en_ = l_kn & (s_ * jnp.where(l_o, F32(1.0), F32(-1.0)) < 0)
        return l_it, (l_kn, l_lf, l_fc, s_, en_)

    def _next_volume(self, q, ev, v, vi, step):
        """Return (the volume after the round's event, whether it was relocated).

        A daughter's or an owner's face: entered, then down; v's own face, or an ancestor's v reaches out of: located
        from beyond that face, up while a solid does not hold the point, then down (Geant4's
        LocateGlobalPointAndSetup).
        """
        D = self.dev()
        OUT = I32(self.n)
        t_c, u_in, is_own, band, hit = ev
        is_ch = u_in >= self.CH0
        uc = jnp.clip(u_in, 0, OUT - 1)
        clip = ~is_own & ~is_ch & self.is_ancestor(u_in, v)
        go_in = ~is_own & ~clip
        v_in = jnp.where(is_ch, OUT + 1 + D["cmesh"][jnp.clip(u_in - self.CH0, 0, D["cmesh"].shape[0] - 1)], u_in)
        up = jnp.where(clip, D["parent"][uc], D["parent"][jnp.minimum(v, D["parent"].shape[0] - 1)])
        nxt = jnp.where(go_in, v_in, up)
        unit_x = jnp.where(go_in, jnp.where(is_ch, I32(-1), u_in), vi)
        ray = (q.od, q.dd, t_c + step)
        return pick(hit & ~(go_in & is_ch), lambda: self._settle(nxt, ray, unit_x, step + band),
                    lambda: (nxt, jnp.bool_(False)))  # fmt: skip

    def _round(self, q, c):
        """One round of a query: the next event in the volume, the volume after it, and whether it is visible."""
        D = self.dev()
        OUT = I32(self.n)
        t_lo, v, last_, _, it, rec, cur, pz = c
        band = rules.band(q.xa + jnp.abs(t_lo))
        vi = jnp.minimum(v, OUT - 1)
        has_own = v < OUT
        # last_: the photon's srf (round 0: the item it sits on, with its face and outward sign when known), then the
        # item each crossing of this query made (the band rules)
        l_it, side = self._side(q, last_, t_lo)
        ctx = _Enter(v, q.od, q.dd, t_lo, q.tmax, l_it, band, side, None)
        t_c, u_in, l_in, f_in, z_in, is_own = self.enter(v, ctx, own=has_own, vi=vi)
        hit = t_c < q.tmax
        # the relocation's step past the crossing: the band at the crossing (not at the round's start: the same
        # crossing steps alike however many rounds led to it)
        step = rules.band(q.xa + jnp.abs(jnp.where(hit, t_c, t_lo)))
        nxt, moved = self._next_volume(q, (t_c, u_in, is_own, step, hit), v, vi, step)
        item = jnp.where(~is_own, u_in, vi)
        after = D["cls"][jnp.where(hit, nxt, v)]
        # a volume the ray only touches is never entered: entered at t_in (the round before), it holds no point of the
        # ray up to this event -- a zero-length stay (a zero-length event at the round's start, which is t_in or the
        # relocation's step past it; or this event within a zero-length step of t_in), or its midpoint outside it --
        # so the crossing is from the volume before it (v_in: itself the one before a touched volume)
        t_in, v_in = pz[8], pz[7]
        zl = hit & (it > 0) & (z_in | (t_c - t_in <= rules.ZERO_STEP_BANDS * band))
        touched = zl | pick(
            hit & ~zl & (it > 0) & (v < OUT) & (after != cur),
            lambda: ~self.holds_point(vi, q.od + (0.5 * (t_in + t_c)) * q.dd),
            lambda: jnp.bool_(False),
        )
        cur = jnp.where(touched, D["cls"][v_in], cur)
        v_b = jnp.where(touched, v_in, v)
        # a vertex: a crossing into another volume (or class) whose class pair the vertex table marks (default: the
        # class changes); the pieces of one split union are one volume
        vis = hit & (after != cur)
        if q.vtx is not None:
            V = q.vtx
            nv = V.shape[0]
            mark = V[jnp.clip(cur, 0, nv - 1), jnp.clip(after, 0, nv - 1)]
            pc = D["piece"]
            moved_vol = (nxt != v) & ~((pc[nxt] >= 0) & (pc[nxt] == pc[v]))
            vis = hit & ((after != cur) | moved_vol) & jnp.where((cur < nv) & (after < nv), mark, after != cur)
        # every invisible crossing (the same class, or a pass pair) is recorded alike -- the first and the last, their
        # count, the volume the run ends in, and the volume the ray is in at s_at (a bulk event's distance: the ray's
        # own volume there, as Geant4 steps through each crossing) -- so the step places a bulk event past any of them
        inv = hit & ~vis
        # zero-length invisible crossings in a row (Geant4's zero steps): at ZERO_STEPS_BEFORE_PUSH the step's stuck
        # rule takes over (a straight continue event); every other invisible crossing is followed, however many
        zc = jnp.where(inv & (t_c - t_lo <= rules.ZERO_STEP_BANDS * band), pz[6] + 1, I32(0))
        pz = (jnp.where(inv & (pz[0] == F32(HUGE)), t_c, pz[0]), jnp.where(inv, t_c, pz[1]), pz[2] + inv.astype(I32),
              jnp.where(inv, nxt, pz[3]), hit, jnp.where(inv & (t_c <= q.s_at), nxt, pz[5]), zc,
              jnp.where(hit, v_b, v_in), jnp.where(hit, t_c, t_in))  # fmt: skip
        done = (~hit) | vis | (zc >= rules.ZERO_STEPS_BEFORE_PUSH)
        rec = jnp.where(hit, jnp.stack([lax.bitcast_convert_type(t_c, I32), item, l_in, f_in, z_in.astype(I32), v_b,
                                        vis.astype(I32)]), rec)  # fmt: skip
        return (jnp.where(moved, t_c + step, t_c), jnp.where(hit, nxt, v), item, done, it + 1, rec,
                jnp.where(hit, after, cur), pz)  # fmt: skip

    def _finish(self, q, o, d, end):
        """Return the query's outputs from the last round's record: the live t, the normal, the surface met."""
        D = self.dev()
        vol, cls0, v_after, rounds, rec, cls_after, pz = end
        bt, bi, bl, bf, zero, vb = lax.bitcast_convert_type(rec[0], F32), rec[1], rec[2], rec[3], rec[4] > 0, rec[5]
        # hit = a visible crossing, or (cont) zero-length invisible crossings in a row: a straight event at the last
        # crossing; a scan that ends on invisible crossings and then nothing is no hit
        vis_end = rec[6] == 1
        cont = ~vis_end & pz[4] & (pz[6] >= rules.ZERO_STEPS_BEFORE_PUSH)
        hit = (vis_end | cont) & (bt < q.tmax)
        live = hit & ~zero & (bl >= 0)
        if self.n_mesh:
            bt = surface_id.mesh_plane_t(self, (live, bi, bl, bt), q.od, q.dd, q.xa)
        # the live t of every event on a known face, a zero-length one too
        lt = hit & (bl >= 0)
        bt_live = jnp.where(lt, self.live_t(bi, bl, bf, o, d, jnp.where(lt, bt, F32(HUGE))), bt)
        g_hit = self.hit_grad(bi, jnp.maximum(bl, 0), bf, o + jnp.where(bt_live < HUGE, bt_live, 0.0) * d)
        nrm = g_hit / safe_norm(g_hit)
        nrm = jnp.where(hit & (bl >= 0), nrm, -d)  # a face known (a zero-length event on the face sat on too)
        # srf: the item met with its face and outward sign -- an entry heads against n_out, an exit along it
        gd = jnp.dot(g_hit, d)
        entry = (bi != vb) & ~self.is_ancestor(bi, vb)  # an ancestor's face, reached out of, is an exit
        srf = jnp.where(hit, self.srf_pack(bi, bl, bf, jnp.where(entry, gd < 0, gd > 0), entry), I32(-1))
        nrm = jnp.where(jnp.sum(nrm * d) > 0, -nrm, nrm)
        # vol_before: the volume just before the visible crossing; without a hit, where the invisible run ends
        vb = jnp.where(hit, vb, pz[3])
        cls_b = D["cls"][vb]
        cls_a = jnp.where(hit, cls_after, D["cls"][vb])
        return dict(t=jnp.where(hit, bt_live, q.tmax), hit=hit, cont=cont,
                    vol_after=jnp.where(hit, v_after, vol), vol_before=vb,
                    vol_start=vol, cls_before=cls_b, cls_start=cls0,
                    cls_after=cls_a, normal=nrm,
                    item=jnp.where(hit, bi, -1), leaf=bl, face=bf, zero=hit & zero, t_pass0=pz[0], t_pass1=pz[1],
                    n_pass=pz[2], vol_at=pz[5], srf=srf)  # fmt: skip

    def query(  # noqa: PLR0913 -- the step's query
        self, o, d, vol, *, t_max, last=-1, vertex_tab=None, sn=None, s_at=None
    ):
        """Return the first visible crossing along (o, d) before ``t_max`` (m) for a photon in volume vol, as a dict.

        Keys: t, hit, cont, vol_after, vol_before, vol_start, cls_before, cls_start, cls_after, normal,
        item, leaf, face, zero, t_pass0, t_pass1, n_pass, vol_at, srf; and rounds (the volumes visited).
        ``last``: the srf a query returned (the item with its face and outward sign: the side is the state, the
        direction decides), or a plain item, or -1. ``sn``: the normal that query gave (s = d . n_out from it).
        ``vertex_tab``: (C, C) bool, the class pairs a crossing into another volume is a vertex at
        (``lumen.build.tracked.vertices``; None: where the class changes). ``s_at``: a distance along the ray;
        ``vol_at`` is the volume the ray is in there (past the invisible crossings before it), as the walk found it.
        """
        D = self.dev()
        o, d = jnp.asarray(o, F32), jnp.asarray(d, F32)
        od, dd = sg(o), sg(d)
        vol = jnp.asarray(vol, I32)
        cls0 = D["cls"][vol]
        snv = None if sn is None else sg(jnp.asarray(sn, F32))
        V = None if vertex_tab is None else jnp.asarray(vertex_tab)
        s_q = F32(HUGE) if s_at is None else sg(jnp.asarray(s_at, F32))
        q = _Query(od, dd, F32(t_max), jnp.max(jnp.abs(od)), snv, V, s_q)
        rec0 = jnp.stack([lax.bitcast_convert_type(F32(HUGE), I32), I32(-1), I32(-1), I32(-1), I32(0), vol, I32(0)])
        c0 = (F32(0.0), vol, jnp.asarray(last, I32), jnp.bool_(False), I32(0), rec0, cls0,
              (F32(HUGE), F32(HUGE), I32(0), vol, jnp.bool_(False), vol, I32(0), vol, F32(0.0)))  # fmt: skip
        _, v_after, _, _, rounds, rec, cls_after, pz = lax.while_loop(lambda c: ~c[3], lambda c: self._round(q, c), c0)
        out = self._finish(q, o, d, (vol, cls0, v_after, rounds, rec, cls_after, pz))
        return dict(out, rounds=rounds)


def surface_at(nav, cx, x, d):
    """Return (t, normal) of the surface crossing cx met, as functions of the ray (x, d), the winner held.

    The values and derivatives the query itself gives (its traversal is detached; only the winner's live t and normal
    carry derivatives), without walking again. ``cx`` is a mapping with the query's ``t`` and ``hit`` and ``win``, the
    winner (item, leaf, face, zero).
    """
    t_q, hit, win = cx["t"], cx["hit"], cx["win"]
    it, lf, fc = win[0], win[1], win[2]
    x, d = jnp.asarray(x, F32), jnp.asarray(d, F32)
    live = hit & (lf >= 0)  # a zero-length event on a known face too (the query's)
    t0 = sg(t_q)
    t = jnp.where(live, nav.live_t(it, lf, fc, x, d, jnp.where(live, t0, HUGE)), t0)
    g = nav.hit_grad(it, jnp.maximum(lf, 0), fc, x + jnp.where(t < HUGE, t, 0.0) * d)
    n = g / safe_norm(g)
    n = jnp.where(hit & (lf >= 0), n, -d)
    return t, jnp.where(jnp.sum(n * d) > 0, -n, n)
