"""The float64 referee of the geometry query: point labels by Geant4's locate rule, and the first visible change.

Independent of any query structure, it reads a ``lumen.build.scene.Scene`` directly: no clipping decided by
sampling, no CSG interval algebra, no acceleration structure beyond a uniform grid of boxes. A point's label (medium,
sensor) is the Scene's own rule evaluated by point membership (``solids.Compiled.inside_np``): the last member
instance in depth-first pre-order (Geant4's locate rank) that contains the point and whose clip chain contains it (a
clipped daughter's members count only inside the union of its mother's effective members). Along a ray the label can
change only where the ray crosses a leaf surface, so the candidate points are all leaf roots (quadric, planes, the
torus quartic by companion eigenvalues, Newton-polished) and the label is evaluated between them.

    orc = Oracle(scene)
    med, sen = orc.label(x)                                  # (n, 3) -> (n,), (n,)
    inst = orc.locate(x)                                     # the member instance holding each point (-1: outside)
    orc.inst_volume[inst], orc.inst_placement[inst]          # its logical volume, the placement (physvol) name
    rec = orc.query(o, d, t_min, t_max, push)                # the first visible label change

``query`` emulates the transport's contract: t* is the first leaf-root point where the label changes; the label
after is taken at t* + push; a change whose medium after equals the medium before (a layer thinner than the push, or
a null crossing) is not visible and the scan resumes at t* + push, as the transport would. A ``Mesh`` member is an
opaque part: it holds no point (never a label); its first triangle hit (Moller-Trumbore, float64) is a change to its
medium. Host numpy, one ray at a time: a referee, not a transport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lumen.build import scene as sc
from lumen.build import solids as so

T_MERGE = 1e-10  # m: candidate points closer than this are one point (coincident faces)
TINY = 1e-300  # a coefficient below this is zero
ROOT_IMAG = 1e-6  # a torus root is real when its imaginary part is below this times the torus scale
NEWTON_STEPS = 3  # Newton polish of the torus roots
BOX_PAD = 1e-9  # m: instance boxes are padded by this
SEGMENT_PAD = 1e-12  # m: a box meets a segment when the slab test overlaps by -this or more
GLOBAL_FRAC = 0.125  # an instance covering more than this fraction of the grid's cells is tested everywhere
MAX_CELLS = 2_000_000  # the grid has at most about this many cells
CHUNK_CELLS = 4.0  # the ray is scanned in chunks of this many cells
MAX_SKIPS = 64  # invisible changes passed before a ray gives up
T_FAR = 1e3  # m: oracle_first_visible's ray length
AMBIGUOUS_BAND = 1e-6  # m: a second change this close after a visible one marks the hit ambiguous
F32_REL = 8 * 2.0**-23  # start_ambiguous: a ball of this many float32 ulps of |x|_inf
TORUS_SCALE_FLOOR = 1e-12  # m: the smallest torus scale the real-root test uses
DN_FLOOR = 1e-3  # mismatch: |d.n| below this is taken as this (grazing crossings)


# ===================================================================================================== leaf roots
def _quadric_roots(leaf, ol, d):
    """Return the roots (1-D array of t) of one QLeaf's quadric and planes along ol + t d (leaf frame, f64).

    Notation: A, b, c the quadric, N, H the planes (``leaf`` = (A, b, c, has_q, N, H)).
    """
    A, b, c, has_q, N, H = leaf
    out = []
    if has_q:
        a2 = float(d @ (A @ d))
        a1 = float(d @ (A @ ol) + b @ d)
        a0 = float(ol @ (A @ ol) + 2.0 * b @ ol + c)
        if abs(a2) > TINY:
            disc = a1 * a1 - a2 * a0
            if disc >= 0.0:
                s = -(a1 + np.copysign(np.sqrt(disc), a1))
                out += [s / a2, a0 / s] if s != 0.0 else [0.0]
        elif abs(a1) > TINY:
            out.append(-a0 / (2.0 * a1))
    if len(N):
        nd = N @ d
        ok = np.abs(nd) > TINY
        out += list((H[ok] - N[ok] @ ol) / nd[ok])
    return np.asarray(out, float)


def _torus_roots(big, small, minv, ol, d):
    """Return the real roots of a TorusLeaf ``{(rho - R)^2 + z^2 <= r^2}`` (u = minv x) along ol + t d.

    Notation: R = big, r = small.
    """
    R, r = big, small
    ou, du = minv @ ol, minv @ d
    a2 = float(du @ du)
    t0 = -float(ou @ du) / a2  # re-origin at the closest approach to the centre
    ou = ou + t0 * du
    a1 = 2.0 * float(ou @ du)
    k = float(ou @ ou) + R * R - r * r
    b2 = du[0] ** 2 + du[1] ** 2
    b1 = 2.0 * (ou[0] * du[0] + ou[1] * du[1])
    b0 = ou[0] ** 2 + ou[1] ** 2
    co = np.array([a2 * a2, 2 * a2 * a1, a1 * a1 + 2 * a2 * k - 4 * R * R * b2, 2 * a1 * k - 4 * R * R * b1,
                   k * k - 4 * R * R * b0])  # fmt: skip
    z = np.roots(co)
    scale = max(R + r, TORUS_SCALE_FLOOR) / np.sqrt(a2)
    s = z.real[np.abs(z.imag) <= ROOT_IMAG * scale]
    dco = np.polyder(co)
    for _ in range(NEWTON_STEPS):
        f, g = np.polyval(co, s), np.polyval(dco, s)
        s = np.where(np.abs(g) > 0, s - f / np.where(g == 0, 1.0, g), s)
    return s + t0


class _Solid:
    """One unique member solid: its Compiled (point membership) and its leaves as root tables."""

    def __init__(self, node):
        self.C = so.Compiled(node)  # float64 numpy membership
        self.leaves = []
        for lf in so.flatten(node)[1]:  # the raw leaves (no common-origin rewrite)
            if lf.kind == "q":
                self.leaves.append(("q", (lf.A, lf.b, lf.c, lf.has_q, lf.N, lf.H), lf.origin))
            else:
                self.leaves.append(("t", (lf.R, lf.r, np.linalg.inv(lf.lin)), lf.origin))
        bb = so.node_bbox(node)
        self.bbox = None if bb is None else (np.asarray(bb[0], float), np.asarray(bb[1], float))

    def roots(self, ol, d):
        """Return every leaf root along ol + t d (solid frame)."""
        out = []
        for kind, data, org in self.leaves:
            if kind == "q":
                out.append(_quadric_roots(data, ol - org, d))
            else:
                out.append(_torus_roots(*data, ol - org, d))
        return np.concatenate(out) if out else np.zeros(0)


@dataclass
class _Instances:
    """The member instances in rank order, the opaque meshes' world triangles, the definition instances' clippers."""

    solids: list
    sol: list
    rot: list
    t: list
    med: list
    sen: list
    di: list
    vol: list
    pv: list
    di_clipper: list
    mtri: list
    mmed: list
    msen: list
    path_id: int = 0


def _flatten(scene):
    """Walk the Scene in Geant4's locate order: the member instances, meshes and clip chains."""
    acc = _Instances(*([] for _ in range(13)))
    sid = {}

    def visit(dname, rw, tw, clipper, msen, pv):
        d = scene.definitions[dname]
        me = len(acc.di_clipper)
        acc.di_clipper.append(clipper)
        for m in sorted(d.members, key=lambda m_: m_.rank):
            sen = msen if m.label.sensor == sc.PLACEMENT else m.label.sensor
            if isinstance(m.solid, sc.Mesh):  # an opaque free-form part: its triangles, never a label
                acc.mtri.append(m.solid.tris @ rw.T + tw)
                acc.mmed.append(np.full(len(m.solid.tris), int(m.label.medium)))
                acc.msen.append(np.full(len(m.solid.tris), int(sen)))
                continue
            if id(m.solid) not in sid:
                sid[id(m.solid)] = len(acc.solids)
                acc.solids.append(_Solid(m.solid))
            acc.sol.append(sid[id(m.solid)])
            acc.rot.append(rw)
            acc.t.append(tw)
            acc.di.append(me)
            acc.vol.append(m.label.volume)
            acc.pv.append(pv)
            acc.med.append(int(m.label.medium))
            if m.label.sensor == sc.PATH:
                acc.sen.append(acc.path_id)
                acc.path_id += 1
            else:
                acc.sen.append(int(sen))
        own = me if d.members else clipper
        for pl in d.placements:
            rot, t = np.asarray(pl.R, float), np.asarray(pl.t, float)
            visit(pl.definition, rw @ rot, rw @ t + tw, own if pl.clip else -1, pl.sensor if pl.sensor >= 0 else msen,
                  pl.name or pl.definition)  # fmt: skip

    visit(scene.root, np.eye(3), np.zeros(3), -1, -1, scene.root)
    return acc


def _cells(glo, cell, dims, blo, bhi):
    """Return the linear indices of the grid cells the box [blo, bhi] meets."""
    c0 = np.clip(np.floor((blo - glo) / cell), 0, dims - 1).astype(np.int64)
    c1 = np.clip(np.floor((bhi - glo) / cell), 0, dims - 1).astype(np.int64)
    ix, iy, iz = (np.arange(c0[k], c1[k] + 1) for k in range(3))
    zz, yy, xx = np.meshgrid(iz, iy, ix, indexing="ij")
    return ((zz * dims[1] + yy) * dims[0] + xx).ravel()


def _slabs(lo, hi, o, d):
    """Entry and exit t of the ray o + t d into the boxes [lo, hi] (n, 3)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    return np.nanmax(np.minimum(t1, t2), 1), np.nanmin(np.maximum(t1, t2), 1)


# ======================================================================================================== the oracle
class Oracle:
    """The Scene flattened into member instances in Geant4's locate order, with a uniform grid of their world boxes.

    Notation: R the instances' world rotations (n, 3, 3).
    """

    def __init__(self, scene, cell=None, global_frac=GLOBAL_FRAC, max_cells=MAX_CELLS):
        self.scene = scene
        self.outside = (int(scene.outside.medium), int(scene.outside.sensor))
        acc = _flatten(scene)
        self.mtri = np.concatenate(acc.mtri) if acc.mtri else np.zeros((0, 3, 3))
        self.mmed = np.concatenate(acc.mmed) if acc.mmed else np.zeros(0, np.int64)
        self.msen = np.concatenate(acc.msen) if acc.msen else np.zeros(0, np.int64)
        self.solids = acc.solids
        self.sol = np.asarray(acc.sol, np.int64)
        self.R = np.asarray(acc.rot, float).reshape(-1, 3, 3)
        self.t = np.asarray(acc.t, float).reshape(-1, 3)
        self.med = np.asarray(acc.med, np.int64)
        self.sen = np.asarray(acc.sen, np.int64)
        self.di = np.asarray(acc.di, np.int64)
        self.di_clipper = np.asarray(acc.di_clipper, np.int64)
        self.inst_volume = np.asarray(acc.vol + [scene.outside.volume], object)  # [-1]: the outside
        self.inst_placement = np.asarray(acc.pv + [""], object)
        self._boxes()
        self._grid(cell, global_frac, max_cells)

    def _boxes(self):
        """World boxes of the instances (unbounded solids: always candidates)."""
        n = len(self.sol)
        lo, hi = np.full((n, 3), -np.inf), np.full((n, 3), np.inf)
        for i in range(n):
            bb = self.solids[self.sol[i]].bbox
            if bb is None:
                continue
            cs = so.corners(*bb) @ self.R[i].T + self.t[i]
            lo[i], hi[i] = cs.min(0) - BOX_PAD, cs.max(0) + BOX_PAD
        self.lo, self.hi = lo, hi

    def _grid(self, cell, global_frac, max_cells):
        lo, hi = self.lo, self.hi
        fin = np.all(np.isfinite(lo), 1)
        glo, ghi = lo[fin].min(0), hi[fin].max(0)
        if cell is None:
            cell = float(np.median((hi[fin] - lo[fin]).max(1)))
            cell = max(cell, float((np.prod(ghi - glo) / max_cells) ** (1 / 3)))
        dims = np.maximum(1, np.ceil((ghi - glo) / cell)).astype(np.int64)
        self.glo, self.cell, self.dims = glo, cell, dims
        c0 = np.clip(np.floor((lo - glo) / cell), 0, dims - 1).astype(np.int64)
        c1 = np.clip(np.floor((hi - glo) / cell), 0, dims - 1).astype(np.int64)
        ncov = np.where(fin, np.prod(c1 - c0 + 1, 1), np.inf)
        glob = ncov > global_frac * np.prod(dims)
        self.glob = np.flatnonzero(glob)
        cells, objs = [], []
        for i in np.flatnonzero(~glob):
            lin = _cells(glo, cell, dims, lo[i], hi[i])
            cells.append(lin)
            objs.append(np.full(lin.size, i, np.int64))
        cells = np.concatenate(cells) if cells else np.zeros(0, np.int64)
        objs = np.concatenate(objs) if objs else np.zeros(0, np.int64)
        order = np.lexsort((objs, cells))
        self.ids = objs[order]
        self.start = np.searchsorted(cells[order], np.arange(np.prod(dims) + 1))

    # ------------------------------------------------------------------------------------------------ candidates
    def _cells_of_box(self, blo, bhi):
        return _cells(self.glo, self.cell, self.dims, blo, bhi)

    def _gather(self, lins):
        parts = [self.ids[self.start[c] : self.start[c + 1]] for c in lins]
        return np.unique(np.concatenate(parts + [self.glob]))

    def candidates_segment(self, o, d, ta, tb):
        """Return the member instances whose world box meets the segment o + t d, t in [ta, tb] (rank order)."""
        p, q = o + ta * d, o + tb * d
        ids = self._gather(self._cells_of_box(np.minimum(p, q), np.maximum(p, q)))
        tn, tf = _slabs(self.lo[ids], self.hi[ids], o, d)
        return ids[(np.maximum(tn, ta) <= np.minimum(tf, tb) + SEGMENT_PAD)]

    def candidates_points(self, x):
        """Return the member instances whose world box holds one of the points x (rank order)."""
        lins = self._cells_of_box(x.min(0), x.max(0)) if len(x) else np.zeros(0, np.int64)
        ids = self._gather(lins)
        ok = np.any(np.all((self.lo[ids][None] <= x[:, None]) & (x[:, None] <= self.hi[ids][None]), -1), 0)
        return ids[ok]

    # ---------------------------------------------------------------------------------------------------- labels
    def label_among(self, x, ids, with_instance=False):
        """Return the labels (medium, sensor) of points x given a candidate set holding every instance containing them.

        ``with_instance``: also the holding instance (-1 outside).
        """
        x = np.asarray(x, float).reshape(-1, 3)
        n = len(x)
        med = np.full(n, self.outside[0], np.int64)
        sen = np.full(n, self.outside[1], np.int64)
        if not len(ids):
            return (med, sen, np.full(n, -1, np.int64)) if with_instance else (med, sen)
        ins = np.zeros((n, len(ids)), bool)
        for s in np.unique(self.sol[ids]):
            for j in np.flatnonzero(self.sol[ids] == s):
                i = ids[j]
                ins[:, j] = self.solids[s].C.inside_np((x - self.t[i]) @ self.R[i])
        eff_di = {}
        best = np.full(n, -1, np.int64)
        for j, i in enumerate(ids):  # rank order: a clipper's members precede its daughters
            cl = self.di_clipper[self.di[i]]
            e = ins[:, j] & (eff_di.get(cl, np.zeros(n, bool)) if cl >= 0 else True)
            eff_di[self.di[i]] = eff_di.get(self.di[i], np.zeros(n, bool)) | e
            best = np.where(e, j, best)
        hit = best >= 0
        med[hit] = self.med[ids[best[hit]]]
        sen[hit] = self.sen[ids[best[hit]]]
        if with_instance:
            return med, sen, np.where(hit, ids[np.maximum(best, 0)], -1)
        return med, sen

    def label(self, x):
        """Return (medium, sensor) of a few points (one candidate set for all: ``label_many`` for many points)."""
        x = np.asarray(x, float).reshape(-1, 3)
        return self.label_among(x, self.candidates_points(x))

    def _points_in_box(self, i, x, order, slin, allp):
        lo, hi = self.lo[i], self.hi[i]
        if self._is_glob[i]:
            cand = allp
        else:
            cells = self._cells_of_box(lo, hi)
            a_, b_ = np.searchsorted(slin, cells), np.searchsorted(slin, cells, side="right")
            if not np.any(b_ > a_):
                return allp[:0]
            cand = np.concatenate([order[p:q] for p, q in zip(a_, b_, strict=True) if q > p])
        return cand[np.all((lo <= x[cand]) & (x[cand] <= hi), 1)]

    def label_many(self, x):
        """Return (medium, sensor, instance) of many scattered points.

        One pass over the member instances in rank order, each testing only the points in its box (found through
        the grid's cells; a global instance tests all of them), the clip chain kept as the point set of each
        definition instance's effective members.
        """
        x = np.asarray(x, float).reshape(-1, 3)
        n = len(x)
        c = np.floor((x - self.glo) / self.cell).astype(np.int64)
        ing = np.all((c >= 0) & (c < self.dims), 1)
        lin = np.where(ing, (c[:, 2] * self.dims[1] + c[:, 1]) * self.dims[0] + c[:, 0], -1)
        order = np.argsort(lin, kind="stable")
        slin = lin[order]
        best = np.full(n, -1, np.int64)
        eff = {}  # definition instance -> point indices in its members
        self._is_glob = np.zeros(len(self.sol), bool)
        self._is_glob[self.glob] = True
        allp = np.arange(n)
        for i in range(len(self.sol)):
            pts = self._points_in_box(i, x, order, slin, allp)
            if not len(pts):
                continue
            ins = self.solids[self.sol[i]].C.inside_np((x[pts] - self.t[i]) @ self.R[i])
            cl = self.di_clipper[self.di[i]]
            if cl >= 0:
                ins &= np.isin(pts, eff.get(cl, np.zeros(0, np.int64)), assume_unique=True)
            got = pts[ins]
            if len(got):
                d_ = self.di[i]
                eff[d_] = np.union1d(eff[d_], got) if d_ in eff else np.sort(got)
                best[got] = i
        hit = best >= 0
        med = np.where(hit, self.med[np.maximum(best, 0)], self.outside[0])
        sen = np.where(hit, self.sen[np.maximum(best, 0)], self.outside[1])
        return med, sen, best

    def locate(self, x):
        """Return the member instance holding each point (-1: outside): an index into inst_volume / inst_placement."""
        return self.label_many(x)[2]

    # ------------------------------------------------------------------------------------------------ the query
    def _changes(self, o, d, span, lab0, ids=None):
        """Return the first label change in (ta, tb] = span after the label lab0 just after ta: (t, label) or None.

        ``ids``: the candidate instances when the caller knows them (default: the grid's, for the segment).
        """
        ta, tb = span
        ids = self.candidates_segment(o, d, ta, tb) if ids is None else ids
        ts = [np.array([ta, tb])]
        for i in ids:
            ol, dl = self.R[i].T @ (o - self.t[i]), self.R[i].T @ d
            ts.append(self.solids[self.sol[i]].roots(ol, dl))
        ts = np.concatenate(ts)
        ts = np.sort(ts[np.isfinite(ts) & (ts > ta + T_MERGE) & (ts <= tb)])  # a root within T_MERGE of ta is ta
        if len(ts) > 1:
            ts = ts[np.concatenate([[True], np.diff(ts) > T_MERGE])]
        pts = np.concatenate([[ta], ts])
        mids = 0.5 * (pts[:-1] + pts[1:]) if len(ts) else np.zeros(0)
        if not len(mids):
            return None
        m, s = self.label_among(o[None] + mids[:, None] * d[None], ids)
        prev = lab0
        for k in range(len(mids)):
            if (m[k], s[k]) != prev:
                return pts[k], (int(m[k]), int(s[k]))
            prev = (m[k], s[k])
        return None

    def mesh_first(self, o, d, ta, tb):
        """Return the first hit on an opaque mesh triangle in (ta, tb]: (t, triangle) or None (Moller-Trumbore, f64).

        Notation: V0, V1, V2 the triangles' vertices.
        """
        if not len(self.mtri):
            return None
        V0, V1, V2 = self.mtri[:, 0], self.mtri[:, 1], self.mtri[:, 2]
        e1, e2 = V1 - V0, V2 - V0
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > TINY
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s_ = o - V0
        u = np.einsum("ij,ij->i", s_, p) * inv
        q = np.cross(s_, e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > ta + T_MERGE) & (t <= tb)
        if not hit.any():
            return None
        k = int(np.flatnonzero(hit)[np.argmin(t[hit])])
        return float(t[k]), k

    def start_ambiguous(self, x, rel=F32_REL):
        """Return whether a surface lies within float32 resolution of x (labels differ on a ball of rel * |x|_inf)."""
        dl = rel * max(1.0, float(np.max(np.abs(x))))
        pts = x[None] + dl * np.concatenate([np.zeros((1, 3)), np.eye(3), -np.eye(3)])
        m, s_ = self.label(pts)
        return bool(np.any(m != m[0]) | np.any(s_ != s_[0]))

    def _scan_end(self, o, d, t_min, t_max):
        """Return the end of the scan (beyond every finite box the label is constant) and the local box segments."""
        fin = np.all(np.isfinite(self.lo), 1)
        _, t_out = _slabs(self.lo[fin].min(0)[None], self.hi[fin].max(0)[None], o, d)
        t_end = min(float(t_max), float(t_out[0]))
        # the ray's segments inside local instance boxes (one slab test over all of them): a chunk that meets none
        # extends to the next box entry (only globals live there), so empty stretches cost one step
        loc = np.ones(len(self.lo), bool)
        loc[self.glob] = False
        loc &= np.all(np.isfinite(self.lo), 1)
        b_in, b_out = _slabs(self.lo[loc], self.hi[loc], o, d)
        on = (b_in <= b_out) & (b_out >= t_min) & (b_in <= t_end)
        if len(self.mtri):  # the meshes' extent bounds the scan too
            mlo, mhi = self.mtri.reshape(-1, 3).min(0), self.mtri.reshape(-1, 3).max(0)
            _, m_out = _slabs(mlo[None], mhi[None], o, d)
            t_end = max(t_end, min(float(t_max), float(m_out[0])))
        return t_end, b_in[on], b_out[on]

    def query(self, o, d, t_min, t_max, push=0.0, **kw):
        """Return the first visible label change along o + t d, t in (t_min, t_max].

        Returns a dict(hit, t, before (medium, sensor), after (medium, sensor) at t + push, skipped, ambiguous,
        start). ``kw``: ``medium``, the transport's current medium (the query's input; default the label at
        o + t_min d) -- a start label that differs from it is reported (start = 'off', 'off-within-f32' when a
        surface is within float32 resolution of o) and the scan proceeds from the given medium, as the transport
        does; ``chunk`` (m, default 4 cells) and ``max_skips``.
        """
        o = np.asarray(o, float)
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        medium = kw.get("medium")
        lab_o = tuple(int(v[0]) for v in self.label(o[None] + t_min * d[None]))
        start = "ok"
        if medium is not None and int(medium) != lab_o[0]:
            start = "off-within-f32" if self.start_ambiguous(o) else "off"
        lab_b = lab_o if medium is None else (int(medium), lab_o[1] if int(medium) == lab_o[0] else -1)
        ray = _Ray(self, o, d, float(push), lab_b, kw.get("chunk") or CHUNK_CELLS * self.cell,
                   kw.get("max_skips", MAX_SKIPS))  # fmt: skip
        out = ray.scan(float(t_min), float(t_max), lab_o)
        out["start"] = start
        return out


class _Ray:
    """The scan of one query along a ray: chunk by chunk, opaque mesh hits and invisible changes passed."""

    def __init__(self, orc, o, d, push, lab_b, chunk, max_skips):  # noqa: PLR0913 -- the query's inputs
        self.orc, self.o, self.d, self.push, self.lab_b = orc, o, d, push, lab_b
        self.chunk, self.max_skips, self.skipped = chunk, max_skips, 0

    def _record(self, t, after, ambiguous=False):
        return dict(hit=True, t=t, before=self.lab_b, after=after, skipped=self.skipped, ambiguous=ambiguous)

    def _mesh_label(self, mh):
        return int(self.orc.mmed[mh[1]]), int(self.orc.msen[mh[1]])

    def _chunk_end(self, ta, t_end, mh, boxes):
        """Return the end of the next chunk and its candidates (None: the grid's; beyond local boxes the globals)."""
        b_in, b_out = boxes
        tb = min(ta + self.chunk, t_end, mh[0] if mh is not None else np.inf)
        only_glob = None
        if not np.any((b_in <= tb) & (b_out >= ta)):
            nxt = b_in[b_in > tb]
            tb = min(float(nxt.min()) if len(nxt) else t_end, t_end)
            only_glob = self.orc.glob
        if mh is not None:
            tb = min(tb, mh[0])
        return tb, only_glob

    def scan(self, t_min, t_max, cur):
        """Scan from t_min; return the query's record (without ``start``)."""
        orc, o, d = self.orc, self.o, self.d
        t_end, b_in, b_out = orc._scan_end(o, d, t_min, t_max)
        mh = orc.mesh_first(o, d, t_min, t_end)
        ta = t_min
        while ta < t_end:
            if mh is not None and ta >= mh[0]:
                # the opaque mesh is the next change: visible unless its medium is the medium before
                if self._mesh_label(mh)[0] != self.lab_b[0]:
                    return self._record(mh[0], self._mesh_label(mh))
                ta = mh[0] + self.push
                mh = orc.mesh_first(o, d, ta, t_end)
                continue
            tb, only_glob = self._chunk_end(ta, t_end, mh, (b_in, b_out))
            ch = orc._changes(o, d, (ta, tb), cur, ids=only_glob)
            if ch is None:
                ta = tb
                continue
            t, lab_geo = ch
            after = tuple(int(v[0]) for v in orc.label(o[None] + (t + self.push) * d[None])) if self.push else lab_geo
            if after[0] != self.lab_b[0]:  # visible: the medium changes
                return self._record(t, after, orc._changes(o, d, (t, t + AMBIGUOUS_BAND), lab_geo) is not None)
            self.skipped += 1
            if self.skipped > self.max_skips:
                break
            ta, cur = t + self.push, after
        if mh is not None and self._mesh_label(mh)[0] != self.lab_b[0]:  # the scan ended on the mesh hit itself
            return self._record(mh[0], self._mesh_label(mh))
        return dict(hit=False, t=float(t_max), before=self.lab_b, after=self.lab_b, skipped=self.skipped,
                    ambiguous=False)  # fmt: skip


# ========================================================================================================= checks
def oracle_first_visible(orc, origins, dirs, push, t_min=1e-7, medium=None):
    """Return the oracle's first visible change of each ray, as arrays (hit, t, after, sensor, before, amb, start)."""
    recs = [orc.query(origins[i], dirs[i], t_min, T_FAR, push=push, medium=None if medium is None else int(medium[i]))
            for i in range(len(origins))]  # fmt: skip
    return dict(hit=np.array([r["hit"] for r in recs]), t=np.array([r["t"] if r["hit"] else np.inf for r in recs]),
                after=np.array([r["after"][0] for r in recs]), sensor=np.array([r["after"][1] for r in recs]),
                before=np.array([r["before"][0] for r in recs]), amb=np.array([r["ambiguous"] for r in recs]),
                start=np.array([r["start"] for r in recs]))  # fmt: skip


def mismatch(e, o, t_tol=2e-5, push=0.0, early=None, ulps=8):
    """Return, per ray, whether an engine's first visible crossing disagrees with the oracle's.

    A crossing reported earlier than the oracle's by at most the push, with the same medium and sensor after, is the
    push contract consuming a layer of the starting medium (the engine stops at an object surface that is not a label
    change, the real change lies within the push after it): not a mismatch; ``early`` (a dict) receives its mask. The
    t tolerance is the larger of t_tol * max(1, t) and ``ulps`` float32 ulps of the hit's coordinates along the normal
    (|dt| |d.n|): a grazing crossing's t is only resolved to ulp / |d.n|.
    """
    bad = (e["hit"] != o["hit"]) | (e["before"] != o["before"])
    both = e["hit"] & o["hit"]
    tol = t_tol * np.maximum(1.0, np.where(np.isfinite(o["t"]), o["t"], 1.0))
    if "dn" in e:
        ulp = np.spacing(np.maximum(e["xabs"], 1.0).astype(np.float32)).astype(float)
        tol = np.maximum(tol, ulps * ulp / np.maximum(e["dn"], DN_FLOOR))
    dt = np.where(both, e["t"] - o["t"], 0.0)
    same = (e["after"] == o["after"]) & (e["sensor"] == o["sensor"])
    ok_t = np.abs(dt) <= tol
    early_ok = (dt < -tol) & (dt >= -(push + tol))
    bad |= both & ~(same & (ok_t | early_ok))
    if early is not None:
        early["mask"] = both & same & early_ok
    return bad
