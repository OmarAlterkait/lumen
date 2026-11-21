"""Triangle meshes of a detector file's solids, for display: closed surfaces in the solid's own frame, m.

    meshes = Tessellator(df)
    m = meshes.solid("PMTGlass")          # SolidMesh(vertices (n, 3) float32, triangles (k, 3) uint32, exact, note)

Each solid follows GDML's parameters (full lengths, polyhedra radii as apothems). Curved faces are cut into flat ones
(``SEGMENTS`` per turn). Booleans are evaluated exactly on those meshes with ``manifold3d`` (the ``view`` extra);
without it, or where an operand is not a closed surface (an open tessellated solid), a union shows both operands and a
subtraction or intersection its first one, with ``exact`` False and the reason in ``note``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lumen.build.lowering import boolean_pose
from lumen.units import MM

try:
    import manifold3d as _mf
except ImportError:  # pragma: no cover - exercised only without the extra
    _mf = None

SEGMENTS = 48  # flat faces per full turn of a curved face, for a solid drawn a few times
MIN_SEGMENTS = 6  # per arc, however short
LOD = ((50, 48), (500, 32), (5000, 20), (None, 12))  # (solid drawn fewer than n times, faces per turn)
CURVE_POINTS = 16  # points along a hyperbolic or parabolic profile
TWOPI = 2 * math.pi
PHI_FULL = 1e-9  # rad: a phi range this close to 2 pi is the full turn
TWIST_REL = 1e-9  # an arb8 whose edges' cross products pass this share of its size squared is twisted


@dataclass(frozen=True)
class SolidMesh:
    """A solid's display mesh: ``vertices`` (n, 3) m in its frame, ``triangles`` (k, 3), outward-facing."""

    vertices: np.ndarray
    triangles: np.ndarray
    exact: bool = True
    note: str = ""


def _orient(loop):
    """Return a closed (r, z) or (x, y) loop counter-clockwise (positive area)."""
    p = np.asarray(loop, float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if area >= 0 else p[::-1]


def _triangulate(loops):
    """Return triangles (k, 3) indexing the concatenated loops (outer CCW, holes CW)."""
    if _mf is not None:
        return np.asarray(_mf.triangulate([np.asarray(lp, float) for lp in loops]), np.int64).reshape(-1, 3)
    p = np.asarray(loops[0], float)  # a fan: correct for a convex outer loop without holes
    return np.array([(0, i, i + 1) for i in range(1, len(p) - 1)], np.int64)


def _arc_segments(dphi, fixed=None, seg=SEGMENTS):
    return fixed if fixed else max(MIN_SEGMENTS, math.ceil(seg * dphi / TWOPI))


def segments_for(uses):
    """Return the faces per turn of a solid drawn ``uses`` times (``LOD``)."""
    return next(n for limit, n in LOD if limit is None or uses < limit)


def lathe(loops, startphi=0.0, deltaphi=TWOPI, sides=None, radial=1.0, seg=SEGMENTS):
    """Return the surface of revolution about z of (r, z) loops (outer CCW, holes CW) as (vertices, triangles).

    ``sides`` fixes the number of flat faces over ``deltaphi`` (a polyhedra), whose corners then sit at
    ``startphi + i deltaphi / sides`` with the radii multiplied by ``radial`` (apothem to corner).
    """
    full = deltaphi <= 0 or deltaphi >= TWOPI - PHI_FULL
    dphi = TWOPI if full else deltaphi
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
        for j in range(n):
            j1 = (j + 1) % rings
            for i in range(k):
                i1 = (i + 1) % k
                a, b, c, d = idx[j, i], idx[j, i1], idx[j1, i1], idx[j1, i]
                tris += [(a, b, c), (a, c, d)]
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
        tris += [tuple(t) for t in first[:, ::-1]] + [tuple(t) for t in last]
    return _closed(v, np.asarray(tris, np.int64))


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
    p = np.asarray(points, float)
    if _mf is not None:
        return _mf.Manifold.hull_points(p)
    from scipy.spatial import ConvexHull

    h = ConvexHull(p)
    return _closed(p, np.asarray(h.simplices, np.int64))


def _arc(r, t0, t1, n):
    th = np.linspace(t0, t1, n + 1)
    return [(r * math.sin(a), r * math.cos(a)) for a in th]


def _sphere_loop(rmin, rmax, t0, dt, seg=SEGMENTS):
    n = max(MIN_SEGMENTS, math.ceil(seg * dt / TWOPI))
    outer = _arc(rmax, t0, t0 + dt, n)
    inner = _arc(rmin, t0 + dt, t0, n) if rmin > 0 else [(0.0, 0.0)]
    return [_orient(outer + inner)]


def _profile_curve(fn, hz, rmin_fn=None):
    z = np.linspace(-hz, hz, CURVE_POINTS)
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
    hz = g("z") / 2
    m = lathe(_ring(g("rmin"), g("rmax"), hz), g("startphi"), g("deltaphi", TWOPI), seg=seg)
    return _cut_tube(m, s, hz) if s.type == "cutTube" else m


def _cone(s, g, seg):
    hz = g("z") / 2
    loop = [_orient([(g("rmin1"), -hz), (g("rmax1"), -hz), (g("rmax2"), hz), (g("rmin2"), hz)])]
    return lathe(loop, g("startphi"), g("deltaphi", TWOPI), seg=seg)


def _polycone(s, g, seg):
    return lathe(_planes_loop(s.planes, s.type == "genericPolycone"), g("startphi"), g("deltaphi", TWOPI), seg=seg)


def _polyhedra(s, g, seg):
    n, dphi = int(g("numsides")), g("deltaphi", TWOPI)
    full = dphi <= 0 or dphi >= TWOPI - PHI_FULL
    radial = 1.0 / math.cos(0.5 * (TWOPI if full else dphi) / n)  # GDML's radii are apothems: to the corners
    loops = _planes_loop(s.planes, s.type == "genericPolyhedra")
    return lathe(loops, g("startphi"), dphi, sides=n, radial=radial, seg=seg)


def _sphere(s, g, seg):
    loop = _sphere_loop(g("rmin"), g("rmax"), g("starttheta"), g("deltatheta", math.pi), seg)
    return lathe(loop, g("startphi"), g("deltaphi", TWOPI), seg=seg)


def _torus(s, g, seg):
    circle = lambda r: [(g("rtor") + r * math.cos(a), r * math.sin(a)) for a in TWOPI * np.arange(seg) / seg]  # noqa: E731
    loops = [_orient(circle(g("rmax")))] + ([_orient(circle(g("rmin")))[::-1]] if g("rmin") > 0 else [])
    return lathe(loops, g("startphi"), g("deltaphi", TWOPI), seg=seg)


def _elcone(s, g, seg):
    zm, zc = g("zmax"), g("zcut")
    return _scale_xy(lathe([_orient([(0, -zc), (zm + zc, -zc), (zm - zc, zc), (0, zc)])], seg=seg), g("dx"), g("dy"))


def _paraboloid(s, g, seg):
    hz, rl, rh = g("dz"), g("rlo"), g("rhi")
    k1, k2 = (rh * rh - rl * rl) / (2 * hz), (rh * rh + rl * rl) / 2
    return lathe(_profile_curve(lambda z: math.sqrt(max(0.0, k1 * z + k2)), hz), seg=seg)


def _hype(s, g, seg):
    hz, to, ti = g("z") / 2, math.tan(g("outst")), math.tan(g("inst"))
    rin = (lambda z: math.sqrt(g("rmin") ** 2 + (ti * z) ** 2)) if (g("rmin") > 0 or ti > 0) else None
    return lathe(_profile_curve(lambda z: math.sqrt(g("rmax") ** 2 + (to * z) ** 2), hz, rin), seg=seg)


PRIMITIVES = {
    "box": lambda s, g, seg: _hull([(i * g("x") / 2, j * g("y") / 2, k * g("z") / 2)
                                    for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]),
    "tube": _tube, "cutTube": _tube, "cone": _cone, "polycone": _polycone, "genericPolycone": _polycone,
    "polyhedra": _polyhedra, "genericPolyhedra": _polyhedra, "sphere": _sphere,
    "orb": lambda s, g, seg: lathe(_sphere_loop(0.0, g("r"), 0.0, math.pi, seg), seg=seg),
    "torus": _torus,
    "eltube": lambda s, g, seg: _scale_xy(lathe(_ring(0.0, 1.0, g("dz")), seg=seg), g("dx"), g("dy")),
    "elcone": _elcone,
    "ellipsoid": lambda s, g, seg: _ellipsoid((g("ax"), g("by"), g("cz")), g("zcut1"), g("zcut2"), seg),
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


def _ellipsoid(axes, zcut1, zcut2, seg=SEGMENTS):
    ax, by, cz = axes
    mesh = _scale_xy(lathe(_sphere_loop(0.0, 1.0, 0.0, math.pi, seg), seg=seg), ax, by, cz)
    if zcut1 == 0.0 and zcut2 == 0.0:
        return mesh
    lo, hi = max(-cz, zcut1 if zcut1 != 0.0 else -cz), min(cz, zcut2 if zcut2 != 0.0 else cz)
    if _mf is None:
        return mesh
    m = _manifold(mesh)
    if m is None:
        return mesh
    return m.trim_by_plane((0, 0, 1), lo).trim_by_plane((0, 0, -1), -hi)


def _cut_tube(mesh, s, hz):
    """Return a tube cut by G4CutTubs's planes (outward normals ``low``, ``high`` through z = -hz, +hz)."""
    m = _manifold(mesh) if _mf is not None else None
    if m is None:
        return mesh
    lo = np.array([s.params.get(k, 0.0) for k in ("lowX", "lowY", "lowZ")], float)
    hi = np.array([s.params.get(k, 0.0) for k in ("highX", "highY", "highZ")], float)
    for n, z in ((lo, -hz), (hi, hz)):
        if np.linalg.norm(n) > 0:
            n = n / np.linalg.norm(n)
            m = m.trim_by_plane(tuple(-n), float(-n[2] * z))  # keep n . (x - (0, 0, z)) <= 0
    return m


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
    if _mf is None:
        return None
    if isinstance(mesh, _mf.Manifold):
        return mesh
    v, t = mesh
    m = _mf.Manifold(_mf.Mesh(np.ascontiguousarray(v, np.float32), np.ascontiguousarray(t, np.uint32)))
    return m if m.status() == _mf.Error.NoError and not m.is_empty() else None


def _arrays(mesh):
    if _mf is not None and isinstance(mesh, _mf.Manifold):
        out = mesh.to_mesh()
        return np.asarray(out.vert_properties, np.float64)[:, :3], np.asarray(out.tri_verts, np.int64)
    return mesh


def _transform(mesh, lin, t):
    lin, t = np.asarray(lin, float), np.asarray(t, float)
    if _mf is not None and isinstance(mesh, _mf.Manifold):
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


class Tessellator:
    """The display meshes of a detector file's solids, each solid tessellated once."""

    def __init__(self, df, uses=None):
        self.df = df
        self.uses = _uses(df, uses or {})
        self._built: dict = {}

    def solid(self, name) -> SolidMesh:
        """Return the display mesh of a solid (``SolidMesh``)."""
        mesh, exact, note = self._build(name)
        v, t = _arrays(mesh)
        return SolidMesh(np.asarray(v, np.float32), np.asarray(t, np.uint32), exact, note)

    def _build(self, name):
        if name not in self._built:
            try:
                self._built[name] = self._make(self.df.solids[name])
            except Exception as err:
                box = _hull([(i, j, k) for i in (-MM, MM) for j in (-MM, MM) for k in (-MM, MM)])
                self._built[name] = (box, False, f"not drawn ({type(err).__name__}: {err}); a 2 mm marker")
        return self._built[name]

    def _make(self, s):
        if s.type in ("union", "subtraction", "intersection"):
            return self._boolean(s)
        if s.type == "multiUnion":
            parts = [self._part(r, tr) for r, tr in zip(s.refs, s.transforms, strict=True)]
            exact = all(p[1] for p in parts)
            ms = [_manifold(p[0]) for p in parts]
            if _mf is not None and all(m is not None for m in ms):
                return _mf.Manifold.batch_boolean(ms, _mf.OpType.Add), exact, _join(p[2] for p in parts)
            return _concat([p[0] for p in parts]), False, "multi-union drawn as its overlapping parts"
        if s.type == "scaledSolid":
            mesh, exact, note = self._build(s.refs[0])
            return _transform(mesh, np.diag(np.asarray(s.scale, float)), np.zeros(3)), exact, note
        exact = not (s.type == "arb8" and _twisted(s))
        seg = segments_for(self.uses.get(s.name, 1))
        return primitive(s, self.df, seg), exact, "" if exact else "twisted arb8 drawn as its convex hull"

    def _part(self, ref, tr):
        mesh, exact, note = self._build(ref)
        return _transform(mesh, *boolean_pose(tr)), exact, note

    def _boolean(self, s):
        (a, ea, na), (b, eb, nb) = (self._part(r, tr) for r, tr in zip(s.refs, s.transforms, strict=True))
        ma, mb = _manifold(a), _manifold(b)
        note = _join((na, nb))
        if ma is not None and mb is not None:
            out = ma + mb if s.type == "union" else (ma - mb if s.type == "subtraction" else ma ^ mb)
            return out, ea and eb, note
        why = "manifold3d not installed" if _mf is None else "an operand is not a closed surface"
        if s.type == "union":
            return _concat([a, b]), False, _join((note, f"union drawn as both operands ({why})"))
        return a, False, _join((note, f"{s.type} drawn as its first operand ({why})"))


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


def _join(notes):
    return "; ".join(dict.fromkeys(n for n in notes if n))


def _twisted(s):
    """Return whether an arb8's side faces are twisted (its top and bottom quadrilaterals are not parallel)."""
    v = np.array([[s.params.get(f"v{i}x", 0.0), s.params.get(f"v{i}y", 0.0)] for i in range(1, 9)], float)
    e1, e2 = v[1:4] - v[0:3], v[5:8] - v[4:7]
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return bool(np.any(np.abs(cross) > TWIST_REL * max(np.finfo(float).tiny, float(np.abs(v).max()) ** 2)))
