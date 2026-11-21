"""Sensor reachability: a query bounded by the free path where the next surface cannot deposit (always on).

On a step whose free path s ends before the next visible crossing (a bulk event), the step reads of that crossing only
the expected deposit at it (``step``: ``at_if`` ... ``E.dep``, credited to ``_sensor_dep``), and that deposit reaches a
cell only when the crossing enters a sensor volume or leaves one (a border sensor, the photon's own ``sensor``: it is
inside that sensor's volume, a sensor class is never a pass pair, ``tracked.vertices``). Everything else a bulk step
reads -- the volume and class at s (``t_pass0`` / ``t_pass1`` / ``vol_at``), lost / escaped (gated by ``~scatter``) --
lies before s. So where the crossing beyond s can touch no sensor and no mesh (whose crossing distance is moved off the
round's own, ``surface_id.mesh_plane_t``), the query may stop past s: ``t_max`` -> s + L, L = (GRAZE K_ON + 2) bands at
the ray's scale, lumen's own look-ahead (a round reads the first change of a solid up to GRAZE K_ON bands past its start
for the on-face rule, in a window ending at min(best, t_max) + band: a smaller t_max cuts that look-ahead and can turn a
start into a zero-length event, as it did with t_max = the next float above s). Then every round that starts before s is
the unbounded query's own (the same arithmetic; ``t_max`` only prunes candidates beyond it), a crossing found at t <= s
is the unbounded one, and none found by s means the unbounded one lies beyond s: a bulk event in both, with the same
invisible crossings before s.

PROOF that the crossing beyond s touches no sensor. The photon is in its volume v's COMPONENT: the volumes joined to v
by crossings without a vertex (``tracked.vertices``: a parent and a daughter, or a volume and a candidate it lists,
lumen's only moves between volumes). Until its first visible crossing the photon is inside members of the component,
so in the bounding box B of their padded world boxes; the crossing is a point of a member's boundary, in B, and a ray
leaves the convex B once: the crossing lies on the segment [0, T], T = B's exit along the ray. If that segment meets
no sensor volume's box and no mesh's box (each padded by lumen's contact tolerance at the scene's scale, which also
covers the float32 slab test's rounding), the crossing touches no sensor and no mesh. A segment that meets one, a
start outside B or in no known volume, a photon sitting on a surface (its srf known: lumen decides whether it is still
in its own volume from whether that volume has an exit inside the round's window, which a bound would cut), or s >= T:
the query as given (no bound).

Cost: the box test runs only on a step whose free path ends inside B (``s < T``, one slab test decides), so a detector
whose photons mostly reach a surface before scattering pays one slab test per step, and one whose sensors line every
wall (water Cherenkov detectors) finds a sensor box early in the walk (any-hit, the walk stops at the first).

    G2 = reach_of(G)      # G: lumen's TrackedGeometry (or the RegionGeometry); a drop-in, query(..., s_free=s)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from lumen.geometry import bvh
from lumen.rules import navigation as rules
from lumen.rules.tolerances import TINY_DIR

F32, I32 = jnp.float32, jnp.int32


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build(nav, sensor_of_vol, vertex_tab):
    """Return (host arrays, report): per volume its component's padded box, and a tree of the sensor and mesh boxes."""
    H, n = nav.host, nav.n
    cls = np.asarray(H["cls"], np.int64)
    parent = np.asarray(H["parent"][:n], np.int64)
    V = None if vertex_tab is None else np.asarray(vertex_tab, bool)

    def invisible(a, b):  # a pair crossed both ways without a vertex (tracked.vertices; None: where the class changes)
        ca, cb = cls[a], cls[b]
        if V is None:
            return ca == cb
        return ca < V.shape[0] and cb < V.shape[0] and not V[ca, cb] and not V[cb, ca]

    uf = _UF(n)
    for v in range(n):
        if parent[v] < n and invisible(v, parent[v]):
            uf.union(v, int(parent[v]))
        for k in nav.kids[v]:
            if k < n and invisible(v, k):
                uf.union(v, int(k))
    lo_w, hi_w = np.asarray(nav.lo_w, np.float64)[:n], np.asarray(nav.hi_w, np.float64)[:n]
    scale = float(np.max(np.abs(np.concatenate([lo_w, hi_w]))))
    pad = float(rules.contact(scale))  # lumen's contact tolerance at the scene's scale (K_ON bands)
    root = np.array([uf.find(v) for v in range(n)], np.int64)
    roots, comp = np.unique(root, return_inverse=True)
    c_lo = np.full((len(roots), 3), np.inf)
    c_hi = np.full((len(roots), 3), -np.inf)
    np.minimum.at(c_lo, comp, lo_w)
    np.maximum.at(c_hi, comp, hi_w)
    # the boxes a bounded segment must not meet: every sensor volume's, every mesh's (from its triangles)
    sov = np.asarray(sensor_of_vol)
    flag_lo = [lo_w[v] for v in range(n) if sov[v] >= 0]
    flag_hi = [hi_w[v] for v in range(n) if sov[v] >= 0]
    n_sensor_boxes = len(flag_lo)
    if nav.n_mesh:
        cst, cn, cm = (np.asarray(H[k]) for k in ("cstart", "cn", "cmesh"))
        tri = np.stack([np.asarray(H[k], np.float64) for k in ("V0", "V1", "V2")], 1)  # (triangles, 3, 3)
        for m in range(nav.n_mesh):
            ks = np.concatenate(
                [np.arange(cst[c], cst[c] + cn[c]) for c in np.nonzero(cm == m)[0]] or [np.zeros(0, int)]
            )
            if len(ks):
                flag_lo.append(tri[ks].min((0, 1)))
                flag_hi.append(tri[ks].max((0, 1)))
    host = dict(
        comp=np.append(comp, -1).astype(np.int32),  # the outside (n): no component, never bounded
        c_lo=(c_lo - pad).astype(np.float32),
        c_hi=(c_hi + pad).astype(np.float32),
    )
    if flag_lo:
        B = bvh.build_many(np.array(flag_lo) - pad, np.array(flag_hi) + pad, np.array([0, len(flag_lo)]))
        host.update(f_lo=B["lo"], f_hi=B["hi"], f_skip=B["skip"], f_count=B["count"])
    else:
        host.update(
            f_lo=np.zeros((1, 3), np.float32),
            f_hi=np.zeros((1, 3), np.float32),
            f_skip=np.ones(1, np.int32),
            f_count=np.zeros(1, np.int32),
        )
    report = dict(components=len(roots), sensor_boxes=n_sensor_boxes, mesh_boxes=len(flag_lo) - n_sensor_boxes, pad=pad,
                  tree_nodes=len(host["f_skip"]) if flag_lo else 0)  # fmt: skip
    return host, report


def _inv(d):
    return 1.0 / jnp.where(jnp.abs(d) < TINY_DIR, jnp.where(d < 0, -TINY_DIR, TINY_DIR), d)


def _meets(R, n_nodes, x, inv, t1):  # noqa: N803 -- R the reach arrays
    """Whether the segment [0, t1] of the ray meets a flagged box (skip-pointer walk; stops at the first)."""
    if not n_nodes:
        return jnp.bool_(False)

    def step(c):
        node, _ = c
        tin, tout = bvh.slab(R["f_lo"][node], R["f_hi"][node], x, inv)
        hit = (tin <= tout) & (tout >= 0.0) & (tin <= t1)
        leaf = R["f_count"][node] > 0
        return jnp.where(hit & ~leaf, node + 1, R["f_skip"][node]), hit & leaf

    return lax.while_loop(lambda c: (c[0] < n_nodes) & ~c[1], step, (I32(0), jnp.bool_(False)))[1]


class ReachNav:
    """The wrapped geometry's navigator plus the reach arrays in its operands (the step binds both); attributes are the
    navigator's."""  # noqa: D205, D209

    def __init__(self, base, rh, rd=None):
        self.__dict__.update(base=base, _rh=rh, _rd=rd)

    def __getattr__(self, k):
        """Return the navigator's attribute."""
        return getattr(self.__dict__["base"], k)

    def rdev(self):
        """Return the reach arrays as device arrays (made once, at trace time; or the operands ``bind`` gave)."""
        if self.__dict__["_rd"] is None:
            with jax.ensure_compile_time_eval():
                self.__dict__["_rd"] = {k: jnp.asarray(v) for k, v in self.__dict__["_rh"].items()}
        return self.__dict__["_rd"]

    def operands(self):
        """Return the navigator's operands with the reach arrays (``reach``)."""
        o = dict(self.__dict__["base"].operands())
        o["reach"] = self.rdev()
        return o


class ReachGeometry:
    """A geometry whose query, given the free path (``s_free``), stops just past it where the crossing beyond cannot
    deposit (module docstring); every other attribute is the wrapped geometry's."""  # noqa: D205, D209

    accepts_s_free = True

    def __init__(self, inner, host=None, report=None, rd=None):
        """``inner``: lumen's TrackedGeometry or a drop-in (the RegionGeometry)."""
        if host is None:
            host, report = build(inner.nav, inner.sensor_of_vol, inner.vertex_tab)
        self.__dict__.update(inner=inner, host=host, report=report, nav=ReachNav(inner.nav, host, rd))

    def __getattr__(self, k):
        """Return the wrapped geometry's attribute."""
        return getattr(self.__dict__["inner"], k)

    def __hash__(self):
        """Hash by identity (a static argument of the step)."""
        return id(self)

    def bind(self, ops):
        """Return this geometry reading its arrays from the operands ``ops``."""
        return ReachGeometry(self.inner.bind(ops), self.host, self.report, ops["reach"])

    def query(self, gp, x, d, cls, t_max, vol=-1, last=-1, sn=None, s_at=None, s_free=None):  # noqa: PLR0913
        """Return the wrapped query; with ``s_free`` (the step's free path) bounded just past it where proven."""
        if s_free is None:
            return self.inner.query(gp, x, d, cls, t_max, vol=vol, last=last, sn=sn, s_at=s_at)
        R = self.nav.rdev()
        vol = jnp.asarray(vol, I32)
        c = R["comp"][jnp.clip(vol, 0, R["comp"].shape[0] - 1)]
        c = jnp.where(vol >= 0, c, -1)
        cc = jnp.maximum(c, 0)
        inv = _inv(d)
        tin, tout = bvh.slab(R["c_lo"][cc], R["c_hi"][cc], x, inv)
        s = jnp.asarray(s_free, F32)
        # a photon sitting on a surface (last >= 0) is never bounded: lumen decides whether it is in its own volume from
        # whether that volume has an exit within the round's window (Navigator._unit_visit: ins = s <= 0 & h), and a
        # t_max short of that exit would turn the start into a zero-length exit (as after a reflection)
        cand = (c >= 0) & (jnp.asarray(last) < 0) & (tin <= 0.0) & (s < tout) & (s < t_max)
        far = lax.cond(cand, lambda _: _meets(R, self.report["tree_nodes"], x, inv, tout), lambda _: jnp.bool_(True),
                       None)  # fmt: skip
        # past s by the navigator's own look-ahead: a round starting before s reads its first change up to GRAZE K_ON
        # bands ahead (the on-face rule, Navigator._on_face) in a window min(best, t_max) + band; t_max must not cut it
        look = (rules.GRAZE * rules.K_ON + 2) * rules.band(jnp.max(jnp.abs(x)) + s)
        tm = lax.stop_gradient(jnp.where(cand & ~far, jnp.minimum(jnp.asarray(t_max, F32), s + look), t_max))
        return self.inner.query(gp, x, d, cls, tm, vol=vol, last=last, sn=sn, s_at=s_at)


def reach_of(geom):
    """Return ``geom`` with sensor reachability (module docstring; always on: bit for bit on every observable)."""
    return ReachGeometry(geom)
