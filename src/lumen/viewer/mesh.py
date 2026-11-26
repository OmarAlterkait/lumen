"""Triangle meshes of a detector file's solids, for display: closed surfaces in the solid's own frame, m.

    meshes = Tessellator(df)
    m = meshes.solid("PMTGlass")          # SolidMesh(vertices (n, 3) float32, triangles (k, 3) uint32, exact, note)

Curved faces are cut into flat ones (``SEGMENTS`` per turn); booleans are evaluated on the meshes by ``manifold3d``
(the ``view`` extra). No mesh is taken as right by its construction: each one is compared with the solid lumen traces
(``lumen.build.lowering.SolidMapper``; ``matches``), and one that differs, or whose solid lumen does not trace, has
``exact`` False and the reason in ``note``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lumen.build.lowering import SolidMapper, boolean_pose
from lumen.geometry import scene as sc
from lumen.geometry import solids as so
from lumen.rules.tolerances import CUT_NORMAL_FLOOR
from lumen.units import MM, TWO_PI

try:
    import manifold3d as _mf
except ImportError:  # the view extra is not installed: ``require``
    _mf = None

SEGMENTS = 48  # flat faces per full turn of a curved face, for a solid drawn a few times
MIN_SEGMENTS = 6  # per arc, however short
LOD = ((50, 48), (500, 32), (5000, 20), (None, 12))  # (solid drawn fewer than n times, faces per turn)
PHI_FULL = 1e-9  # rad: a phi range this close to 2 pi is the full turn
CHECK_SEGMENTS = 192  # faces per turn of the mesh compared with lumen's solid
CHECK_POINTS = 128  # points in the solid's box, and on the mesh, of that comparison
CHECK_SAGITTAS = 3.0  # its tolerance: this many times the sagitta of those faces (a face curved two ways has two)
CHECK_FLAT = 1e-6  # its tolerance for a solid of flat faces, as a share of its size (float32 coordinates)


@dataclass(frozen=True)
class SolidMesh:
    """A solid's display mesh: ``vertices`` (n, 3) m in its frame, ``triangles`` (k, 3), outward-facing.

    ``exact``: it matches the solid lumen traces, to its flat faces (``matches``); ``note``: why not; ``refused``: why
    lumen does not trace the solid ("": it does).
    """

    vertices: np.ndarray
    triangles: np.ndarray
    exact: bool
    note: str
    refused: str


def _orient(loop):
    """Return a closed (r, z) or (x, y) loop counter-clockwise (positive area)."""
    p = np.asarray(loop, float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if area >= 0 else p[::-1]


def _triangulate(loops):
    """Return triangles (k, 3) indexing the concatenated loops (outer CCW, holes CW)."""
    return np.asarray(_mf.triangulate([np.asarray(lp, float) for lp in loops]), np.int64).reshape(-1, 3)


def _arc_segments(dphi, fixed=None, seg=SEGMENTS):
    return fixed if fixed else max(MIN_SEGMENTS, math.ceil(seg * dphi / TWO_PI))


def require():
    """Raise ImportError when the view extra is not installed."""
    if _mf is None:
        raise ImportError("lumen.viewer needs manifold3d: install lumen's view extra (pip install -e '.[view]')")


def segments_for(uses):
    """Return the faces per turn of a solid drawn ``uses`` times (``LOD``)."""
    return next(n for limit, n in LOD if limit is None or uses < limit)


def lathe(loops, startphi=0.0, deltaphi=TWO_PI, sides=None, radial=1.0, seg=SEGMENTS):
    """Return the surface of revolution about z of (r, z) loops (outer CCW, holes CW) as (vertices, triangles).

    ``sides`` fixes the number of flat faces over ``deltaphi`` (a polyhedra), whose corners then sit at
    ``startphi + i deltaphi / sides`` with the radii multiplied by ``radial`` (apothem to corner).
    """
    full = deltaphi <= 0 or deltaphi >= TWO_PI - PHI_FULL
    dphi = TWO_PI if full else deltaphi
    n = _arc_segments(dphi, sides, seg)
    phis = startphi + dphi * np.arange(n + (0 if full else 1)) / n
    rings = len(phis)
    verts, tris = [], []
    base = 0
    for loop in loops:
        p = np.asarray(loop, float)
        k = len(p)
        r, z = p[:, 0] * radial, p[:, 1]
        ring = np.stack([np.outer(np.cos(phis), r), np.outer(np.sin(phis), r), np.broadcast_to(z, (rings, k))], -1)
        verts.append(ring.reshape(-1, 3))
        idx = base + np.arange(rings * k).reshape(rings, k)
        a, d = idx[:n], idx[(np.arange(n) + 1) % rings]
        b, c = np.roll(a, -1, axis=1), np.roll(d, -1, axis=1)
        tris += [np.stack(q, -1).reshape(-1, 3) for q in ((a, b, c), (a, c, d))]
        base += rings * k
    v = np.concatenate(verts)
    if not full:  # the two phi caps: the loops triangulated in (r, z)
        cap = _triangulate(loops)
        offs = np.cumsum([0] + [len(lp) * rings for lp in loops[:-1]])
        lens = [len(lp) for lp in loops]

        def ring_index(t, j):
            out = np.empty_like(t)
            starts = np.cumsum([0] + lens[:-1])
            for q, (s0, ln) in enumerate(zip(starts, lens, strict=True)):
                m = (t >= s0) & (t < s0 + ln)
                out[m] = offs[q] + j * ln + (t[m] - s0)
            return out

        first, last = ring_index(cap, 0), ring_index(cap, rings - 1)
        tris += [first[:, ::-1], last]
    return _closed(v, np.concatenate(tris))


def _closed(v, t):
    """Return (vertices, triangles) with coincident vertices merged, degenerate triangles dropped, faces outward."""
    key = np.round(v / max(np.finfo(float).tiny, float(np.abs(v).max(initial=0.0))) * 2**36).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    v, t = v[first], inv.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    if _signed_volume(v, t) < 0:
        t = t[:, ::-1]
    return v, t


def _signed_volume(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c)))) / 6.0


def _hull(points):
    return _mf.Manifold.hull_points(np.asarray(points, float))


def _arc(r, t0, t1, n):
    th = np.linspace(t0, t1, n + 1)
    return [(r * math.sin(a), r * math.cos(a)) for a in th]


def _sphere_loop(rmin, rmax, t0, dt, seg=SEGMENTS):
    n = max(MIN_SEGMENTS, math.ceil(seg * dt / TWO_PI))
    outer = _arc(rmax, t0, t0 + dt, n)
    inner = _arc(rmin, t0 + dt, t0, n) if rmin > 0 else [(0.0, 0.0)]
    return [_orient(outer + inner)]


def _profile_curve(fn, hz, seg, rmin_fn=None):
    z = np.linspace(-hz, hz, seg // 2 + 1)
    outer = [(fn(q), q) for q in z]
    inner = [(rmin_fn(q), q) for q in z[::-1]] if rmin_fn else [(0.0, hz), (0.0, -hz)]
    return [_orient(outer + inner)]


def _planes_loop(planes, generic):
    if generic:  # (r, z) points
        return [_orient([(r, z) for r, z in planes])]
    outer = [(rmax, z) for rmin, rmax, z in planes]
    inner = [(rmin, z) for rmin, rmax, z in reversed(planes)]
    return [_orient(outer + inner)]


def _scale_xy(mesh, sx, sy, sz=1.0):
    v, t = mesh
    return v * np.array([sx, sy, sz]), t


def _ring(r_in, r_out, hz):
    """Return the (r, z) rectangle of a tube between r_in and r_out, over -hz..hz."""
    return [_orient([(r_in, -hz), (r_out, -hz), (r_out, hz), (r_in, hz)])]


def _tube(s, g, seg):
    return lathe(_ring(g("rmin"), g("rmax"), g("z") / 2), g("startphi"), g("deltaphi", TWO_PI), seg=seg)


def _cut_tube(s, g, seg):
    """Return G4CutTubs: a tube long enough for both cuts, cut by its planes (outward normals ``low``, ``high``)."""
    hz, rmax = g("z") / 2, g("rmax")
    cuts = [(np.array([g(f"{end}{k}") for k in "XYZ"]), z) for end, z in (("low", -hz), ("high", hz))]
    cuts = [(n / np.linalg.norm(n), z) for n, z in cuts]
    reach = hz + max(rmax * math.hypot(n[0], n[1]) / max(abs(n[2]), CUT_NORMAL_FLOOR) for n, _ in cuts)
    m = _manifold(lathe(_ring(g("rmin"), rmax, reach), g("startphi"), g("deltaphi", TWO_PI), seg=seg))
    for n, z in cuts:
        m = m.trim_by_plane(tuple(-n), float(-n[2] * z))  # keep n . (x - (0, 0, z)) <= 0
    return m


def _cone(s, g, seg):
    hz = g("z") / 2
    loop = [_orient([(g("rmin1"), -hz), (g("rmax1"), -hz), (g("rmax2"), hz), (g("rmin2"), hz)])]
    return lathe(loop, g("startphi"), g("deltaphi", TWO_PI), seg=seg)


def _polycone(s, g, seg):
    return lathe(_planes_loop(s.planes, s.type == "genericPolycone"), g("startphi"), g("deltaphi", TWO_PI), seg=seg)


def _polyhedra(s, g, seg):
    n, dphi = int(g("numsides")), g("deltaphi", TWO_PI)
    full = dphi <= 0 or dphi >= TWO_PI - PHI_FULL
    generic = s.type == "genericPolyhedra"  # its (r, z) points are the corners; a polyhedra's radii are apothems
    radial = 1.0 if generic else 1.0 / math.cos(0.5 * (TWO_PI if full else dphi) / n)
    return lathe(_planes_loop(s.planes, generic), g("startphi"), dphi, sides=n, radial=radial, seg=seg)


def _sphere(s, g, seg):
    loop = _sphere_loop(g("rmin"), g("rmax"), g("starttheta"), g("deltatheta", math.pi), seg)
    return lathe(loop, g("startphi"), g("deltaphi", TWO_PI), seg=seg)


def _torus(s, g, seg):
    circle = lambda r: [(g("rtor") + r * math.cos(a), r * math.sin(a)) for a in TWO_PI * np.arange(seg) / seg]  # noqa: E731
    loops = [_orient(circle(g("rmax")))] + ([_orient(circle(g("rmin")))[::-1]] if g("rmin") > 0 else [])
    return lathe(loops, g("startphi"), g("deltaphi", TWO_PI), seg=seg)


def _elcone(s, g, seg):
    zm = g("zmax")
    zc = min(g("zcut"), zm)  # G4EllipticalCone: |z| <= min(zcut, zmax)
    return _scale_xy(lathe([_orient([(0, -zc), (zm + zc, -zc), (zm - zc, zc), (0, zc)])], seg=seg), g("dx"), g("dy"))


def _paraboloid(s, g, seg):
    hz, rl, rh = g("dz"), g("rlo"), g("rhi")
    k1, k2 = (rh * rh - rl * rl) / (2 * hz), (rh * rh + rl * rl) / 2
    return lathe(_profile_curve(lambda z: math.sqrt(max(0.0, k1 * z + k2)), hz, seg), seg=seg)


def _hype(s, g, seg):
    hz, to, ti = g("z") / 2, math.tan(g("outst")), math.tan(g("inst"))
    rin = (lambda z: math.sqrt(g("rmin") ** 2 + (ti * z) ** 2)) if (g("rmin") > 0 or ti > 0) else None
    return lathe(_profile_curve(lambda z: math.sqrt(g("rmax") ** 2 + (to * z) ** 2), hz, seg, rin), seg=seg)


def _ellipsoid(s, g, seg):
    """Return G4Ellipsoid: both cuts 0 is no cut, otherwise they are clamped to [-cz, cz]."""
    cz, lo, hi = g("cz"), g("zcut1"), g("zcut2")
    if lo == 0.0 and hi == 0.0:
        lo, hi = -cz, cz
    t0, t1 = (math.acos(max(-1.0, min(1.0, z / cz))) for z in (hi, lo))  # on the unit sphere
    loop = [_orient([(0.0, math.cos(t0)), *_arc(1.0, t0, t1, max(MIN_SEGMENTS, seg // 2)), (0.0, math.cos(t1))])]
    return _scale_xy(lathe(loop, seg=seg), g("ax"), g("by"), cz)


PRIMITIVES = {
    "box": lambda s, g, seg: _hull([(i * g("x") / 2, j * g("y") / 2, k * g("z") / 2)
                                    for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]),
    "tube": _tube, "cutTube": _cut_tube, "cone": _cone, "polycone": _polycone, "genericPolycone": _polycone,
    "polyhedra": _polyhedra, "genericPolyhedra": _polyhedra, "sphere": _sphere,
    "orb": lambda s, g, seg: lathe(_sphere_loop(0.0, g("r"), 0.0, math.pi, seg), seg=seg),
    "torus": _torus,
    "eltube": lambda s, g, seg: _scale_xy(lathe(_ring(0.0, 1.0, g("dz")), seg=seg), g("dx"), g("dy")),
    "elcone": _elcone,
    "ellipsoid": _ellipsoid,
    "paraboloid": _paraboloid, "hype": _hype,
    **dict.fromkeys(("trd", "para", "trap", "arb8", "tet"), lambda s, g, seg: _hull(_polytope_vertices(s))),
    "xtru": lambda s, g, seg: _xtru(s.vertices, s.sections),
}  # fmt: skip


def primitive(s, df, seg=SEGMENTS):
    """Return a primitive solid as (vertices, triangles), or a manifold3d solid, in m; ``seg`` faces per turn."""
    if s.type == "tessellated":
        tri = np.asarray(df.meshes[s.mesh], float).reshape(-1, 3)
        return _closed(tri, np.arange(len(tri)).reshape(-1, 3))
    make = PRIMITIVES.get(s.type)
    if make is None:
        raise NotImplementedError(f"no display mesh for a {s.type} solid")
    return make(s, lambda k, d=0.0: float(s.params.get(k, d)), seg)


def _polytope_vertices(s):
    """Return the corners of a convex polytope solid (trd, para, trap, arb8, tet), m."""
    g = lambda k: float(s.params.get(k, 0.0))  # noqa: E731
    if s.type == "tet":
        return s.vertices
    if s.type == "arb8":
        dz = g("dz")
        return [(g(f"v{i}x"), g(f"v{i}y"), -dz if i <= 4 else dz) for i in range(1, 9)]
    if s.type == "trd":
        hz = g("z") / 2
        return [(sx * g(f"x{k}") / 2, sy * g(f"y{k}") / 2, (-hz if k == 1 else hz))
                for k in (1, 2) for sx in (-1, 1) for sy in (-1, 1)]  # fmt: skip
    if s.type == "para":
        dx, dy, dz = g("x") / 2, g("y") / 2, g("z") / 2
        ta, tt, ph = math.tan(g("alpha")), math.tan(g("theta")), g("phi")
        return [(sx * dx + sy * dy * ta + sz * dz * tt * math.cos(ph), sy * dy + sz * dz * tt * math.sin(ph), sz * dz)
                for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]  # fmt: skip
    dz = g("z") / 2  # trap
    tth, cph, sph = math.tan(g("theta")), math.cos(g("phi")), math.sin(g("phi"))
    out = []
    faces = ((-1, g("y1"), g("x1"), g("x2"), g("alpha1")), (1, g("y2"), g("x3"), g("x4"), g("alpha2")))
    for sz, y, xa, xb, al in faces:
        cx, cy, ta = sz * dz * tth * cph, sz * dz * tth * sph, math.tan(al)
        for sy, hx in ((-1, xa / 2), (1, xb / 2)):
            for sx in (-1, 1):
                out.append((cx + sy * y / 2 * ta + sx * hx, cy + sy * y / 2, sz * dz))
    return out


def _xtru(verts, sections):
    poly = _orient(verts)
    k = len(poly)
    rings = [np.column_stack([xo + sc * poly[:, 0], yo + sc * poly[:, 1], np.full(k, z)]) for z, xo, yo, sc in sections]
    v = np.concatenate(rings)
    tris = []
    for j in range(len(rings) - 1):
        for i in range(k):
            i1 = (i + 1) % k
            a, b, c, d = j * k + i, j * k + i1, (j + 1) * k + i1, (j + 1) * k + i
            tris += [(a, b, c), (a, c, d)]
    cap = _triangulate([poly])
    tris += [tuple(t) for t in cap[:, ::-1]] + [tuple(t + (len(rings) - 1) * k) for t in cap]
    return _closed(v, np.asarray(tris, np.int64))


def _manifold(mesh):
    """Return a manifold3d.Manifold of (vertices, triangles), or None when they do not bound a solid."""
    if isinstance(mesh, _mf.Manifold):
        return mesh
    v, t = mesh
    m = _mf.Manifold(_mf.Mesh64(np.ascontiguousarray(v, np.float64), np.ascontiguousarray(t, np.uint64)))
    return m if m.status() == _mf.Error.NoError and not m.is_empty() else None


def _arrays(mesh):
    if isinstance(mesh, _mf.Manifold):
        out = mesh.to_mesh64()
        return np.asarray(out.vert_properties, np.float64)[:, :3], np.asarray(out.tri_verts, np.int64)
    return mesh


def _transform(mesh, lin, t):
    lin, t = np.asarray(lin, float), np.asarray(t, float)
    if isinstance(mesh, _mf.Manifold):
        return mesh.transform(np.ascontiguousarray(np.column_stack([lin, t]), np.float64))
    v, tri = mesh
    v = v @ lin.T + t
    return v, (tri if np.linalg.det(lin) > 0 else tri[:, ::-1])


def _concat(meshes):
    vs, ts, n = [], [], 0
    for m in meshes:
        v, t = _arrays(m)
        vs.append(v)
        ts.append(t + n)
        n += len(v)
    return np.concatenate(vs), np.concatenate(ts)


def _inside(v, t, x):
    """Return whether each point of ``x`` (n, 3) lies in a closed mesh: the parity of the faces above it."""
    p = v[t]
    lo, hi = p[..., :2].min(1), p[..., :2].max(1)
    q, k = np.nonzero(np.all((lo <= x[:, None, :2]) & (x[:, None, :2] <= hi), axis=2))
    a, e, f = p[k, 0], p[k, 1] - p[k, 0], p[k, 2] - p[k, 0]
    d = x[q] - a
    det = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
    det = np.where(det == 0.0, np.inf, det)  # a vertical face is never crossed
    s = (d[:, 0] * f[:, 1] - d[:, 1] * f[:, 0]) / det
    r = (e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]) / det
    above = (s >= 0) & (r >= 0) & (s + r <= 1) & np.isfinite(det) & (s * e[:, 2] + r * f[:, 2] > d[:, 2])
    return np.bincount(q[above], minlength=len(x)) % 2 == 1


def _on_mesh(v, t, n, rng):
    """Return ``n`` points uniform over a mesh's area."""
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    k = rng.choice(len(t), n, p=area / area.sum())
    s, r = np.sqrt(rng.random((n, 1))), rng.random((n, 1))
    return (1 - s) * a[k] + s * (1 - r) * b[k] + s * r * c[k]


def _ball(tol):
    """Return the offsets that probe a point's neighbourhood: 0, and the axes and diagonals at tol / 2 and tol."""
    u = np.concatenate([np.eye(3), -np.eye(3), so.corners(-np.ones(3), np.ones(3))])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return np.concatenate([np.zeros((1, 3)), 0.5 * tol * u, tol * u])


def matches(v, t, solid, box) -> bool:
    """Return whether a closed mesh of ``CHECK_SEGMENTS`` faces per turn bounds the solid lumen traces (in ``box``).

    Every point in their common box is in both or in neither, and lumen's surface passes by every point of the mesh,
    within the sagitta of the flat faces (``CHECK_SAGITTAS``); ``CHECK_POINTS`` points of each.
    """
    inside, rng = solid.inside_np, np.random.default_rng(0)
    lo, hi = np.minimum(box[0], v.min(0)), np.maximum(box[1], v.max(0))
    curved = any(lf.kind != "q" or lf.has_q for lf in solid.leaves)
    rel = CHECK_SAGITTAS * (1.0 - math.cos(math.pi / CHECK_SEGMENTS)) if curved else CHECK_FLAT
    ball = _ball(rel * np.maximum(abs(lo), abs(hi)).max())

    def near(x):
        q = inside((x[:, None] + ball).reshape(-1, x.shape[1])).reshape(len(x), len(ball))
        return q.any(1) & ~q.all(1)

    x = rng.uniform(lo, hi, (CHECK_POINTS, len(lo)))
    x = x[_inside(v, t, x) != inside(x)]
    return bool(near(x).all() and near(_on_mesh(v, t, CHECK_POINTS, rng)).all())


class Tessellator:
    """The display meshes of a detector file's solids, each solid tessellated and checked once."""

    def __init__(self, df, uses=None):
        self.df, self.mapper = df, SolidMapper(df)
        self.uses = _uses(df, uses or {})
        self._built: dict = {}

    def solid(self, name) -> SolidMesh:
        """Return the display mesh of a solid (``SolidMesh``), compared with the solid lumen traces."""
        mesh, note = self._build(name, None)
        v, t = _arrays(mesh)
        traced, refused = self._traced(name)
        note = note or refused or self._difference(name, traced)
        return SolidMesh(np.asarray(v, np.float32), np.asarray(t, np.uint32), not note, note, refused)

    def _traced(self, name):
        """Return (the solid lumen traces: its ``Compiled``, None for a mesh; why lumen does not trace it, "")."""
        try:
            node = self.mapper.solid(name)
            return (None if isinstance(node, sc.Mesh) else (so.Compiled(node), so.node_bbox(node))), ""
        except Exception as err:  # whatever stops the lowering stops the conversion
            return None, f"lumen does not trace this solid: {err}"

    def _difference(self, name, traced):
        fine, note = self._build(name, CHECK_SEGMENTS)
        if note or traced is None or matches(*_arrays(fine), *traced):  # a mesh: the triangles drawn are lumen's
            return note
        return "not matched to the solid lumen traces"

    def _build(self, name, seg):
        """Return (mesh, note) of a solid at ``seg`` faces per turn (None: by its number of uses)."""
        key = (name, seg)
        if key not in self._built:
            try:
                self._built[key] = self._make(self.df.solids[name], seg)
            except Exception as err:  # drawn as a marker, with the reason
                box = _hull([(i, j, k) for i in (-MM, MM) for j in (-MM, MM) for k in (-MM, MM)])
                self._built[key] = (box, f"not drawn ({type(err).__name__}: {err}); a 2 mm marker")
        return self._built[key]

    def _make(self, s, seg):
        if s.type in ("union", "subtraction", "intersection", "multiUnion"):
            return self._boolean(s, seg)
        if s.type == "scaledSolid":
            mesh, note = self._build(s.refs[0], seg)
            return _transform(mesh, np.diag(np.asarray(s.scale, float)), np.zeros(3)), note
        return primitive(s, self.df, seg or segments_for(self.uses.get(s.name, 1))), ""

    def _boolean(self, s, seg):
        parts = [(self._build(r, seg), boolean_pose(tr)) for r, tr in zip(s.refs, s.transforms, strict=True)]
        meshes = [_transform(m, *pose) for (m, _), pose in parts]
        note = "; ".join(dict.fromkeys(n for (_, n), _ in parts if n))
        solids = [_manifold(m) for m in meshes]
        union = s.type in ("union", "multiUnion")
        if any(m is None for m in solids):
            what = "its operands" if union else "its first operand"
            return _concat(meshes) if union else meshes[0], f"drawn as {what} (one is not a closed surface)"
        op = _mf.OpType.Add if union else (_mf.OpType.Subtract if s.type == "subtraction" else _mf.OpType.Intersect)
        return _mf.Manifold.batch_boolean(solids, op), note


def _uses(df, lv_uses):
    """Return {solid: times drawn}: a volume's solid by its placed copies, a boolean's operands by the boolean's."""
    out: dict = {}

    def add(name, n):
        out[name] = out.get(name, 0) + n
        s = df.solids[name]
        for r in s.refs:
            add(r, n)

    for lv, n in lv_uses.items():
        solid = df.volumes[lv].solid
        if solid and n:
            add(solid, n)
    return out
