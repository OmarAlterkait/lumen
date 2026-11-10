"""The device CSG evaluator: one data-driven evaluator for every solid of a scene (``csg_build.CsgTables``).

A navigator asks a solid only whether it holds a point and where a ray next enters or leaves it. Every kind runs the
same instructions on different data, so a warp whose threads hold different solids does not run every kind's code in
turn: leaves are gathered from the tables one scalar at a time, and a kind's boolean is its postfix program.

Semantics. Membership along a ray is half-open, [lo, hi) per leaf interval: "inside at t" means inside just after t.
``first_change(kind, o, d, t0, t1)`` is the first t in (t0, t1] where the solid's membership changes: the nearest
leaf endpoint e after t0 whose boolean, evaluated from the leaves' intervals at e + band, differs from the boolean
before it. Endpoints within the band of e are the same crossing (coincident faces of one solid, a union seam); a chord
longer than the band is a crossing of its own however thin. No point tests along the ray: a point test misses grazing
exits in float32; the intervals do not.

    E = Csg(tables)
    E.inside(k, x); E.inside_ray(k, o, d, t)              # jnp scalars, one thread
    r = E.first_change(k, o, d, t0, t1)                   # dict(hit, t, inside (after), leaf, face, inside0, first)
    g = E.leaf_grad(k, leaf, face, x)                     # the winning surface's gradient (live t, normal)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from lumen.geometry import leaf_kernels
from lumen.geometry.csg_build import INF
from lumen.rules import navigation as rules

F32, I32 = jnp.float32, jnp.int32
BIG = leaf_kernels.BIG
PARALLEL = 1e-12  # a plane with |n . d| below this is parallel to the ray
SMALL_DIR = 1e-20  # a direction component below this: the ray is parallel to that slab of a leaf box
TORUS_PAD_REL = 1e-4  # the torus pretests' pad, relative to R + r
MAX_SCAN_PER_LEAF = 4  # first_change scans at most this many endpoints per leaf (plus MAX_SCAN_EXTRA)
MAX_SCAN_EXTRA = 5


def pick(p, a, b):
    """Return a() if p else b(), over pytrees (thunks): a lax.cond (under jax.vmap, both sides and a select)."""
    return lax.cond(p, lambda _: a(), lambda _: b(), None)


def tau_at(o, t):
    """Return the float32 band of a crossing at t on a ray from o: endpoints closer than this are one crossing."""
    return rules.band(jnp.max(jnp.abs(o)), t)


class _Scan(NamedTuple):
    """The fixed inputs of one first_change call."""

    kind: object
    o: list
    d: list
    t1: object
    nl: object
    band: object
    want: object
    init: object
    skip: object
    hold: object


class Csg:
    """The device evaluator over a scene's CSG tables (jnp scalars, one thread)."""

    def __init__(self, tables):
        self.tables = tables
        self.has_torus = tables.has_torus
        self.empty_leaf = tables.empty_leaf
        self._dev = None

    def dev(self):
        """Return the tables as device arrays (made once, at trace time)."""
        if self._dev is None:
            with jax.ensure_compile_time_eval():
                host = dict(T=self.tables.T, K=self.tables.K, G=self.tables.G)
                self._dev = jax.tree.map(jnp.asarray, host)
        return self._dev

    @property
    def T(self):  # noqa: N802 -- the leaf tables' name in the module docstring
        """Return the leaf tables (device)."""
        return self.dev()["T"]

    # ------------------------------------------------------------------ leaf primitives (one leaf id, scalars)
    def _o(self, g):
        return [self.T["o0"][g], self.T["o1"][g], self.T["o2"][g]]

    def _qo(self, g):
        return [self.T["q0"][g], self.T["q1"][g], self.T["q2"][g]]

    def _qv(self, g, xl):
        Ax = self._Av(g, xl)
        b = [self.T["b0"][g], self.T["b1"][g], self.T["b2"][g]]
        return (xl[0] * Ax[0] + xl[1] * Ax[1] + xl[2] * Ax[2] + 2.0 * (b[0] * xl[0] + b[1] * xl[1] + b[2] * xl[2])
                + self.T["c"][g])  # fmt: skip

    def _Av(self, g, v):  # noqa: N802 -- A v, the quadric's matrix
        return [self.T[f"A{a}0"][g] * v[0] + self.T[f"A{a}1"][g] * v[1] + self.T[f"A{a}2"][g] * v[2] for a in range(3)]

    def _Mv(self, g, v, tr=False):  # noqa: N802 -- M v, the torus frame's inverse
        def key(a, b):
            return f"m{b}{a}" if tr else f"m{a}{b}"

        return [self.T[key(a, 0)][g] * v[0] + self.T[key(a, 1)][g] * v[1] + self.T[key(a, 2)][g] * v[2]
                for a in range(3)]  # fmt: skip

    def _planes(self, g, o, d):
        """Return (t0p, t1p, f0p, f1p, empty_par) of leaf g's plane slab along o + t d (o relative to its origin)."""
        inf = F32(INF)
        p0 = self.T["p0"][g]

        def body(k, c):
            t0p, t1p, f0p, f1p, ep = c
            q = p0 + k
            n = [self.T["n0"][q], self.T["n1"][q], self.T["n2"][q]]
            nd = n[0] * d[0] + n[1] * d[1] + n[2] * d[2]
            rhs = self.T["h"][q] - (n[0] * o[0] + n[1] * o[1] + n[2] * o[2])
            par = jnp.abs(nd) < PARALLEL
            tp = rhs / jnp.where(par, 1.0, nd)
            ent = jnp.where(nd < -PARALLEL, tp, -inf)
            ext = jnp.where(nd > PARALLEL, tp, inf)
            u0 = ent > t0p
            u1 = ext < t1p
            return (jnp.where(u0, ent, t0p), jnp.where(u1, ext, t1p), jnp.where(u0, k + 1, f0p),
                    jnp.where(u1, k + 1, f1p), ep | (par & (rhs < 0.0)))  # fmt: skip

        return lax.fori_loop(0, self.T["np_"][g], body, (-inf, inf, I32(-1), I32(-1), jnp.bool_(False)))

    def _q_iv(self, g, o, d):
        """Return a quadric leaf's two intervals (lo0, hi0, flo0, fhi0, lo1, hi1, flo1, fhi1) along o + t d."""
        inf, ninf, m1 = F32(INF), F32(-INF), I32(-1)
        og, oq = self._o(g), self._qo(g)
        y = [o[a] - oq[a] for a in range(3)]
        o = [o[a] - og[a] for a in range(3)]
        t0p, t1p, f0p, f1p, empty_par = self._planes(g, o, d)
        has_q = self.T["typ"][g] == 1
        b = [self.T["b0"][g], self.T["b1"][g], self.T["b2"][g]]

        def bd(v):
            return b[0] * v[0] + b[1] * v[1] + b[2] * v[2]

        L0, R0, L0q, R0q, L1, R1 = leaf_kernels.quad_line(lambda v: self._Av(g, v), bd, self.T["c"][g], y, d,
                                                          self.T["sa"][g], rules.FLOAT32_EPS, inf)  # fmt: skip
        L0 = jnp.where(has_q, L0, ninf)
        R0 = jnp.where(has_q, R0, inf)
        L0q = has_q & L0q
        R0q = has_q & R0q
        L1 = jnp.where(has_q, L1, inf)
        R1 = jnp.where(has_q, R1, ninf)

        def clip(L_, R_, Lq, Rq):  # noqa: N803 -- a piece's ends and their quadric flags
            lo = jnp.maximum(L_, t0p)
            hi = jnp.minimum(R_, t1p)
            f_lo = jnp.where(L_ > t0p, jnp.where(Lq, 0, f0p), f0p).astype(I32)
            f_hi = jnp.where(R_ < t1p, jnp.where(Rq, 0, f1p), f1p).astype(I32)
            ok = (lo < hi) & (~empty_par)
            return (jnp.where(ok, lo, inf), jnp.where(ok, hi, inf), jnp.where(ok, f_lo, m1), jnp.where(ok, f_hi, m1))

        a0 = clip(L0, R0, L0q, R0q)
        a1 = clip(L1, R1, jnp.bool_(True), jnp.bool_(False))
        sw = (a0[0] >= inf) & (a1[0] < inf)
        pairs = list(zip(a0, a1, strict=True))
        a0, a1 = [jnp.where(sw, y_, x_) for x_, y_ in pairs], [jnp.where(sw, x_, y_) for x_, y_ in pairs]
        return (a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3])

    def _torus_window(self, g, ol, du, kk, w):
        """Return whether the window w = (w0, w1) of the ray meets the torus (exact ball / slab / annulus pretests)."""
        R, r = self.T["R"][g], self.T["r"][g]
        Rb = (R + r) * (1.0 + TORUS_PAD_REL)
        bb = ol[0] * du[0] + ol[1] * du[1] + ol[2] * du[2]
        disc = Rb * Rb - (ol[0] * ol[0] + ol[1] * ol[1] + ol[2] * ol[2] - bb * bb)
        sq = jnp.sqrt(jnp.maximum(disc, 0.0))
        s_lo = jnp.maximum(-bb - sq, w[0] * kk)
        s_hi = jnp.minimum(-bb + sq, w[1] * kk)
        meet = (disc > 0) & (s_lo <= s_hi)
        pad = TORUS_PAD_REL * (R + r)
        rz = r + pad
        small = jnp.abs(du[2]) < PARALLEL
        iz = 1.0 / jnp.where(small, 1.0, du[2])
        za, zb = (-rz - ol[2]) * iz, (rz - ol[2]) * iz
        s_lo = jnp.where(small, s_lo, jnp.maximum(s_lo, jnp.minimum(za, zb)))
        s_hi = jnp.where(small, s_hi, jnp.minimum(s_hi, jnp.maximum(za, zb)))
        meet = meet & (s_lo <= s_hi) & (~small | (jnp.abs(ol[2]) <= rz))
        qa = du[0] * du[0] + du[1] * du[1]
        qb = ol[0] * du[0] + ol[1] * du[1]
        qc = ol[0] * ol[0] + ol[1] * ol[1]

        def rho2(s_):
            return qa * s_ * s_ + 2.0 * qb * s_ + qc

        s_m = jnp.clip(-qb / jnp.where(qa > PARALLEL, qa, 1.0), s_lo, s_hi)
        r2_min = jnp.where(qa > PARALLEL, rho2(s_m), qc)
        r2_max = jnp.maximum(rho2(s_lo), rho2(s_hi))
        r_out = R + r + pad
        r_in = jnp.maximum(R - r - pad, 0.0)
        return meet & (r2_min <= r_out * r_out) & (r2_max >= r_in * r_in) & (self.T["typ"][g] == 2)

    def _t_iv(self, g, o, d, w0, w1):
        """Return a torus leaf's intervals under the exact pretests of the window [w0, w1]."""
        og = self._o(g)
        R, r = self.T["R"][g], self.T["r"][g]
        ol = self._Mv(g, [o[a] - og[a] for a in range(3)])
        dl = self._Mv(g, d)
        kk = jnp.sqrt(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2])
        du = [dl[a] / kk for a in range(3)]
        meet = self._torus_window(g, ol, du, kk, (w0, w1))
        inf, m1 = F32(INF), I32(-1)

        def solve(a):
            s = leaf_kernels.torus_set(a[2], a[3], jnp.stack(a[0]), jnp.stack(a[1]))
            return s["lo"][0], s["hi"][0], s["lo"][1], s["hi"][1]

        lo0, hi0, lo1, hi1 = lax.cond(meet, solve, lambda a: (inf, inf, inf, inf), (ol, du, R, r))

        def sc(v):
            return jnp.where(jnp.abs(v) < BIG, v / kk, v)

        f0 = jnp.where(lo0 < BIG, I32(0), m1)
        f1 = jnp.where(lo1 < BIG, I32(0), m1)
        return (sc(lo0), sc(hi0), f0, f0, sc(lo1), sc(hi1), f1, f1)

    def _iv(self, g, o, d, w0, w1):
        if not self.has_torus:
            return self._q_iv(g, o, d)
        return pick(self.T["typ"][g] == 2, lambda: self._t_iv(g, o, d, w0, w1), lambda: self._q_iv(g, o, d))

    def _meets(self, g, o, d, w0, w1):
        """Return whether o + t d, t in [w0, w1], meets leaf g's box (slabs; an empty box never)."""
        lo_t, hi_t, ne = w0, w1, jnp.bool_(True)
        for a in range(3):
            lo, hi = self.T[f"bl{a}"][g], self.T[f"bh{a}"][g]
            small = jnp.abs(d[a]) < SMALL_DIR
            inv = 1.0 / jnp.where(small, 1.0, d[a])
            ta, tb = (lo - o[a]) * inv, (hi - o[a]) * inv
            lo_t = jnp.where(small, lo_t, jnp.maximum(lo_t, jnp.minimum(ta, tb)))
            hi_t = jnp.where(small, hi_t, jnp.minimum(hi_t, jnp.maximum(ta, tb)))
            ne = ne & (lo <= hi) & (~small | ((lo <= o[a]) & (o[a] <= hi)))
        return ne & (lo_t <= hi_t)

    def _iv_in(self, g, o, d, w0, w1):
        """Return _iv where the window [w0, w1] meets leaf g's box; else no intervals (the leaf is false there)."""
        inf, m1 = F32(INF), I32(-1)
        return pick(self._meets(g, o, d, w0, w1), lambda: self._iv(g, o, d, w0, w1),
                    lambda: (inf, inf, m1, m1, inf, inf, m1, m1))  # fmt: skip

    def _inside_leaf(self, g, x):
        og, oq = self._o(g), self._qo(g)
        xl = [x[a] - og[a] for a in range(3)]
        qv = self._qv(g, [x[a] - oq[a] for a in range(3)])
        p0 = self.T["p0"][g]

        def body(k, ok):
            q = p0 + k
            return ok & (self.T["n0"][q] * xl[0] + self.T["n1"][q] * xl[1] + self.T["n2"][q] * xl[2] <= self.T["h"][q])

        pl = lax.fori_loop(0, self.T["np_"][g], body, jnp.bool_(True))
        u = self._Mv(g, xl)
        rho = jnp.sqrt(u[0] * u[0] + u[1] * u[1])
        R, r = self.T["R"][g], self.T["r"][g]
        tin = (rho - R) ** 2 + u[2] * u[2] <= r * r
        typ = self.T["typ"][g]
        return jnp.where(typ == 2, tin, pl & ((typ == 0) | (qv <= 0.0)))

    # ------------------------------------------------------------------ the boolean
    def _program(self, kind, member_of):
        """Return the kind's postfix program, PUSH = member_of(kind-local leaf) (int32 0/1), stack in one int32."""
        D = self.dev()
        p0, pl = D["K"]["p0"][kind], D["K"]["pl"][kind]

        def step(p, s):
            op, arg = D["G"]["op"][p0 + p], D["G"]["arg"][p0 + p]
            bb = s & 1
            aa = (s >> 1) & 1
            r = jnp.where(op == 1, aa | bb, jnp.where(op == 2, aa & bb, aa & (1 - bb)))
            push_ = op == 0
            lv = pick(push_, lambda: member_of(arg), lambda: I32(0))
            return jnp.where(push_, (s << 1) | (lv & 1), ((s >> 2) << 1) | r)

        return (lax.fori_loop(0, pl, step, I32(0)) & 1) == 1

    def _leaf(self, kind, l):  # noqa: E741 -- l: a kind-local leaf
        """Return the global leaf id of kind-local leaf l (the empty leaf past the end)."""
        K = self.dev()["K"]
        return jnp.where(l < K["nl"][kind], K["l0"][kind] + l, self.empty_leaf)

    def inside(self, kind, x):
        """Return whether the solid holds the point x (its frame; closed sets, no tolerance)."""
        x = [x[a] for a in range(3)]
        return self._program(kind, lambda lf: self._inside_leaf(self._leaf(kind, lf), x).astype(I32))

    def _mem_at(self, kind, o, d, t, hold=None):
        """Return the solid's membership just after o + t d (the leaves' intervals).

        ``hold``: (leaf, inside, t_end) -- that leaf's membership is ``inside`` before t_end (the face the photon sits
        on, within its tolerance: first_change).
        """

        def member_of(l):  # noqa: E741 -- a kind-local leaf
            iv = self._iv_in(self._leaf(kind, l), o, d, t, t)
            m = ((iv[0] <= t) & (t < iv[1])) | ((iv[4] <= t) & (t < iv[5]))
            if hold is not None:
                m = jnp.where((l == hold[0]) & (t < hold[2]), hold[1], m)
            return m.astype(I32)

        return self._program(kind, member_of)

    def inside_ray(self, kind, o, d, t):
        """Return whether the solid holds o + t d just after t (the leaves' intervals along the ray: lo <= t < hi)."""
        o = [o[a] for a in range(3)]
        d = [d[a] for a in range(3)]
        return self._mem_at(kind, o, d, jnp.asarray(t, F32))

    # ------------------------------------------------------------------ the first change along a ray
    def _nearest(self, s, t_cur):
        """Return the nearest leaf endpoint after t_cur (<= t1): (e, leaf, face, entry)."""
        inf = F32(INF)

        def leaf(l, c):  # noqa: E741 -- a kind-local leaf
            e1, lb, fc, ek = c
            iv = self._iv_in(self._leaf(s.kind, l), s.o, s.d, t_cur, s.t1)
            for ep, fp, en in ((iv[0], iv[2], True), (iv[1], iv[3], False), (iv[4], iv[6], True),
                               (iv[5], iv[7], False)):  # fmt: skip
                ok = (jnp.abs(ep) < BIG) & (ep <= s.t1) & (ep > t_cur) & (ep < e1)
                if s.skip is not None:
                    ok = ok & ~((l == s.skip[0]) & (fp == s.skip[1]) & (s.skip[2] == en) & (ep < s.skip[3]))
                e1 = jnp.where(ok, ep, e1)
                lb = jnp.where(ok, l, lb)
                fc = jnp.where(ok, fp, fc)
                ek = jnp.where(ok, en, ek)
            return e1, lb, fc, ek

        return lax.fori_loop(0, s.nl, leaf, (inf, I32(-1), I32(-1), jnp.bool_(False)))

    def _membership(self, s, first, has, tp, now, hd):
        if s.init is None:
            return pick(has, lambda: self._mem_at(s.kind, s.o, s.d, tp, hd), lambda: now)
        return pick(has & ~(first & s.init[0]), lambda: self._mem_at(s.kind, s.o, s.d, tp, hd),
                    lambda: jnp.where(first & s.init[0], s.init[1], now))  # fmt: skip

    def _change(self, s, t0, c):
        """One step of first_change's scan: the next endpoint, the membership past it, whether it is the change."""
        t_cur, done, now, lb_h, fc_h, now0, it, fs, hend = c
        first = it == 0
        e, lb, fc, ek = pick(first, lambda: (t0, I32(-1), I32(-1), jnp.bool_(False)), lambda: self._nearest(s, t_cur))
        has = first | (e < INF)
        tp = jnp.where(first, t0, e + s.band(e))
        hd = None
        if s.hold is not None:  # the held leaf's other kind of endpoint ends the hold
            hend = jnp.where(~first & has & (lb == s.hold[0]) & (ek != s.hold[1]), jnp.minimum(hend, e), hend)
            hd = (s.hold[0], s.hold[1], hend)
        aft = self._membership(s, first, has, tp, now, hd)
        now0 = jnp.where(first, aft, now0)
        chg = ~first & has & (aft != now)
        vis = chg if s.want is None else chg & (aft == s.want)
        new = chg & ~fs[0]  # the first change of any kind (the on-face rule reads it)
        fs = (fs[0] | chg, jnp.where(new, e, fs[1]), jnp.where(new, aft, fs[2]), jnp.where(new, lb, fs[3]),
              jnp.where(new, fc, fs[4]))  # fmt: skip
        now = jnp.where(has, aft, now)  # the state past every change: a passed kind changes it too
        t_nx = jnp.where(first, t0, jnp.where(vis, e, jnp.where(has, tp, F32(INF))))
        return (t_nx, vis, now, jnp.where(vis, lb, lb_h), jnp.where(vis, fc, fc_h), now0, it + 1, fs, hend)

    def first_change(self, kind, o, d, t0, t1, tau=None, want=None, init=None, skip=None, hold=None):  # noqa: PLR0913 -- the scan's options
        """Return the first change of the solid's membership in (t0, t1].

        Returns dict(hit, t, inside (after it), leaf (kind-local), face, inside0 (just after t0), first (the first
        change of any kind: hit, t, inside, leaf, face)). ``want``: None any change; True the first entry, False the
        first exit (the other kind is passed). ``init``: (known, inside) -- where known, the membership at t0 is given,
        not probed (the side a photon is on). ``skip``: (leaf, face, entry, t_hi) -- that face's endpoints of that kind
        before t_hi are no crossing (the face the photon sits on). ``hold``: (leaf, inside, t_end) -- that leaf's
        membership is ``inside`` for every probe before t_end and before its next endpoint of the other kind (Geant4's
        surface rule: within the tolerance the side is the state, never a float32 probe of it). Endpoints within the
        band of the nearest one are one crossing.
        """
        o = [o[a] for a in range(3)]
        d = [d[a] for a in range(3)]
        t0 = jnp.asarray(t0, F32)
        t1 = jnp.asarray(t1, F32)
        nl = self.dev()["K"]["nl"][kind]
        xa = jnp.maximum(jnp.maximum(jnp.abs(o[0]), jnp.abs(o[1])), jnp.abs(o[2]))

        def band(t):
            return rules.band(xa, t) if tau is None else tau

        s = _Scan(kind, o, d, t1, nl, band, want, init, skip, hold)

        def cond(c):
            return (~c[1]) & (c[0] < INF) & (c[6] < MAX_SCAN_PER_LEAF * nl + MAX_SCAN_EXTRA)

        fs0 = (jnp.bool_(False), F32(INF), jnp.bool_(False), I32(-1), I32(-1))
        h0 = F32(-INF) if hold is None else jnp.asarray(hold[2], F32)
        c0 = (t0, jnp.bool_(False), jnp.bool_(False), I32(-1), I32(-1), jnp.bool_(False), I32(0), fs0, h0)
        t_end, hit, aft, lb, fc, now0, _, fs, _ = lax.while_loop(cond, lambda c: self._change(s, t0, c), c0)
        return dict(hit=hit, t=jnp.where(hit, t_end, t1), inside=jnp.where(hit, aft, now0), leaf=lb, face=fc,
                    inside0=now0, first=dict(hit=fs[0], t=fs[1], inside=fs[2], leaf=fs[3], face=fs[4]))  # fmt: skip

    # ------------------------------------------------------------------ the winning leaf's gradient (normal, AD)
    def leaf_grad(self, kind, leaf, face, x):
        """Return the gradient of kind-local leaf ``leaf`` (face 0 its quadric, >= 1 its planes) at x (kind frame)."""
        K = self.dev()["K"]
        g = K["l0"][kind] + jnp.maximum(leaf, 0)
        og, oq = self._o(g), self._qo(g)
        xl = [x[a] - og[a] for a in range(3)]
        Ax = self._Av(g, [x[a] - oq[a] for a in range(3)])
        b = [self.T["b0"][g], self.T["b1"][g], self.T["b2"][g]]
        gq = [2.0 * (Ax[a] + b[a]) for a in range(3)]
        q = self.T["p0"][g] + jnp.maximum(face - 1, 0)
        gp = [self.T["n0"][q], self.T["n1"][q], self.T["n2"][q]]
        u = self._Mv(g, xl)
        rho = jnp.sqrt(jnp.maximum(u[0] * u[0] + u[1] * u[1], 1e-30))
        kk = (rho - self.T["R"][g]) / rho
        gt = self._Mv(g, [kk * u[0], kk * u[1], u[2]], tr=True)
        typ = self.T["typ"][g]
        return jnp.stack([jnp.where(typ == 2, gt[a], jnp.where(face >= 1, gp[a], gq[a])) for a in range(3)])
