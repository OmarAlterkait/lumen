"""Solids on the host: the CSG model every geometry query is built from, and Geant4's solids in it.

A solid is a static CSG tree of plain Python objects whose leaves are

- ``QLeaf``: ``{x : q(x) <= 0} & {n_k . x <= h_k}`` with ``q(x) = x'A x + 2 b.x + c`` (A symmetric 3x3), or planes only
  (``has_q`` False). Along a line ``{q <= 0}`` is at most two intervals, so a leaf meets a ray in at most two: exact
  for cones without a nappe selector, one-sheet hyperboloids and every convex quadric or polytope.
- ``TorusLeaf``: ``{(rho - R)^2 + z^2 <= r^2}`` in its own frame, placed by any invertible linear map.

and whose inner nodes are ``Op`` (union, inter, sub) and ``Xf`` (a child placed by ``x_parent = L x_child + t``).
``flatten`` bakes every transform into the leaves; ``Compiled`` holds a flattened solid with float64 point membership
(``inside_np``), what the float64 oracle evaluates. Every leaf carries its own origin and the quadric its own frame
(``quad_frame``), so coefficients stay of the size of the leaf, not of its distance from the world origin.

The ``g4_*`` constructors build Geant4's solids from GDML's parameters, in metres and radians as the detector file
holds them (full lengths where GDML gives full lengths). Constructors attach an exact axis-aligned box
(``node.bbox``); ``node_bbox`` combines boxes by the CSG rules.

The device kernels (the interval sets along a ray, the CSG interval algebra, the torus root finder) live with the
navigator (``lumen.geometry``). Lengths are in metres.
"""

from __future__ import annotations

import math

import numpy as np

OPS = ("union", "inter", "sub")
TWOPI = 2 * math.pi
FULL_PHI_TOL = 1e-12  # rad: a phi range this close to 2 pi is the full circle
ANGLE_TOL = 1e-15  # rad: theta limits this close to 0, pi / 2 or pi are those angles
CYLINDER_REL = 1e-15  # a frustum whose radii agree to this (relative) is a cylinder
Z_TOL = 1e-15  # m: polycone / polyhedra sections thinner than this are dropped
MERGE_REL = 1e-12  # relative: consecutive polycone sections affine to this are one section
SHARE_TOL = 1e-12  # plane normals / quadric coefficients this close are the same surface (relative)
CONVEX_TOL = 1e-15  # an eigenvalue of A above -this is non-negative (a convex quadric)
DEGENERATE_FACE = 1e-14  # a face whose normal is this small (relative to the size squared) has collapsed
VERTEX_TOL = 1e-9  # relative: a polytope vertex beyond a face plane by more than this is not convex
TWIST_TOL = 1e-9  # m: an arb8 side face twisted by more than this is not planar
NORMAL_FLOOR = 1e-12  # a cut-tube plane normal with |n_z| below this is treated as this
TINY = 1e-300  # the smallest scale divided by
HALF_TURN_MARGIN = 1e-9  # rad: a polyhedra side spans at most pi - this in its bounding radius
# G4Cons replaces a zero inner radius at one end (the other > 0) by 1e3 * kRadTolerance = 1e-6 mm
# (G4Cons.cc, since geom-solids-csg-V02-00-02)
G4CONS_ZERO_RMIN = 1e-9  # m


# ===================================================================================================== the CSG tree
class QLeaf:
    """A quadric (optionally) intersected with half-spaces, in the solid's orientation, at its own origin.

    ``A, b, c`` define ``q(x) = x'A x + 2 b.x + c`` relative to ``origin``; ``N``, ``H`` the planes ``n . x <= h``
    (normalised); ``qo`` the quadric's own frame relative to ``origin`` (``quad_frame``). Notation: A the quadric's
    matrix, N the plane normals (k, 3), H their offsets (k,).
    """

    kind = "q"

    def __init__(self, a, b, c, has_q, planes, origin=(0.0, 0.0, 0.0), qo=(0.0, 0.0, 0.0)):  # noqa: PLR0913 -- a leaf's defining data
        self.A = np.asarray(a, float).reshape(3, 3)
        self.b = np.asarray(b, float).reshape(3)
        self.c = float(c)
        self.has_q = bool(has_q)
        self.N = np.asarray([p[0] for p in planes], float).reshape(-1, 3)
        self.H = np.asarray([p[1] for p in planes], float).reshape(-1)
        nn = np.linalg.norm(self.N, axis=1) if len(self.N) else np.zeros(0)
        if len(self.N):
            self.N = self.N / nn[:, None]
            self.H = self.H / nn
        self.origin = np.asarray(origin, float)
        self.qo = np.asarray(qo, float)  # the quadric's own frame, relative to origin (quad_frame)

    @property
    def nfaces(self):
        """Return the number of faces: the quadric and one per plane."""
        return 1 + self.N.shape[0]

    def affine(self, lin, t):
        """Return the leaf of the image set ``{lin x + t : x in leaf}`` (lin invertible).

        Notation: Li the inverse of lin; A, N as in the class.
        """
        Li = np.linalg.inv(lin)
        A = Li.T @ self.A @ Li
        b = Li.T @ self.b
        N = self.N @ Li
        return QLeaf(A, b, self.c, self.has_q, list(zip(N, self.H, strict=True)), origin=lin @ self.origin + t,
                     qo=lin @ getattr(self, "qo", np.zeros(3)))  # fmt: skip


class TorusLeaf:
    """A solid torus ``{(rho - R)^2 + z^2 <= r^2}`` in canonical coordinates u, placed by ``x = lin u + origin``.

    ``lin`` is any invertible linear map, so a G4ScaledSolid of a torus (a quartic, not a torus) is exact. Notation:
    R the swept radius, r the tube radius.
    """

    kind = "torus"
    nfaces = 1

    def __init__(self, swept, tube, lin=None, origin=(0.0, 0.0, 0.0)):
        self.R, self.r = float(swept), float(tube)
        self.lin = np.asarray(np.eye(3) if lin is None else lin, float)
        self.origin = np.asarray(origin, float)

    def affine(self, lin, t):
        """Return the torus of the image set ``{lin x + t : x in torus}``."""
        return TorusLeaf(self.R, self.r, lin=lin @ self.lin, origin=lin @ self.origin + t)


class Op:
    """A boolean of two nodes: ``op`` is "union", "inter" or "sub" (a minus b)."""

    def __init__(self, op, a, b):
        if op not in OPS:
            raise ValueError(f"Op: unknown operation {op!r} (one of {OPS})")
        self.op, self.a, self.b = op, a, b


class Xf:
    """A child node placed by ``x_parent = L x_child + t``. Notation: L the linear map."""

    def __init__(self, lin, t, child):
        self.L = np.asarray(lin, float)
        self.t = np.asarray(t, float)
        self.child = child


def flatten(node, lin=None, t=None, leaves=None, polarity=None, sign=1):
    """Bake every transform into the leaves; return ``(expr, leaves)`` with ``expr`` over leaf indices.

    ``polarity`` (an optional list) receives, per leaf, +1 or -1: the sign relating the leaf's own outward gradient to
    the solid's outward normal on that leaf's faces (-1 once per subtraction right-hand side on the path).
    """
    lin = np.eye(3) if lin is None else lin
    t = np.zeros(3) if t is None else t
    if leaves is None:
        leaves = []
    if isinstance(node, Xf):
        return flatten(node.child, lin @ node.L, lin @ node.t + t, leaves, polarity, sign)
    if isinstance(node, Op):
        ea, _ = flatten(node.a, lin, t, leaves, polarity, sign)
        eb, _ = flatten(node.b, lin, t, leaves, polarity, -sign if node.op == "sub" else sign)
        return (node.op, ea, eb), leaves
    leaves.append(node.affine(lin, t))
    if polarity is not None:
        polarity.append(sign)
    return len(leaves) - 1, leaves


def quad_frame(lf):
    """Return ``(qo, bq, cq)``, float64: the quadric of a QLeaf in its own frame.

    ``x = lf.origin + qo + y`` and ``q = y'A y + 2 bq.y + cq``. ``qo`` is where the leaf's constructor centred it (a
    tube on its axis, a sphere at its centre, a frustum at its mid-section); a shared origin moves the leaf origin but
    not this frame, so a thin quadric far from the shared origin keeps coefficients of its own size (a 76 um wire 1 m
    off: c = -r^2, not 1 m^2 - r^2, which float32 rounds to 1 m^2 and the wire vanishes).
    """
    qo = np.asarray(getattr(lf, "qo", np.zeros(3)), float)
    return qo, lf.b + lf.A @ qo, float(qo @ lf.A @ qo + 2.0 * lf.b @ qo + lf.c)


def _move_origin(lf, new_origin):
    dlt = new_origin - lf.origin
    lf.c = float(dlt @ lf.A @ dlt + 2.0 * lf.b @ dlt + lf.c)
    lf.b = lf.b + lf.A @ dlt
    if lf.N.shape[0]:
        lf.H = lf.H - lf.N @ dlt
    lf.origin = new_origin.copy()
    lf.qo = getattr(lf, "qo", np.zeros(3)) - dlt  # the quadric's own frame stays where it was (quad_frame)


def _share_quadric(lf, ident):
    """One representative per (A, b, c) up to a positive scale (the sense is kept)."""
    s_ = max(np.max(np.abs(lf.A)), np.max(np.abs(lf.b)), abs(lf.c), TINY)
    for aq, bq, cq, qq in ident.quads:
        sq = max(np.max(np.abs(aq)), np.max(np.abs(bq)), abs(cq), TINY)
        if (np.max(np.abs(lf.A / s_ - aq / sq)) < SHARE_TOL and np.max(np.abs(lf.b / s_ - bq / sq)) < SHARE_TOL
                and abs(lf.c / s_ - cq / sq) < SHARE_TOL):  # fmt: skip
            lf.A, lf.b, lf.c, lf.qo = aq.copy(), bq.copy(), float(cq), qq.copy()
            return
    ident.quads.append((lf.A.copy(), lf.b.copy(), float(lf.c), lf.qo.copy()))


def _share_planes(lf, canon):
    for k in range(lf.N.shape[0]):
        n, h = lf.N[k], lf.H[k]
        tol_h = SHARE_TOL * (1.0 + abs(h))
        for cn, ch in canon:
            if np.max(np.abs(n - cn)) < SHARE_TOL and abs(h - ch) < tol_h:
                lf.N[k], lf.H[k] = cn, ch
                break
            if np.max(np.abs(n + cn)) < SHARE_TOL and abs(h + ch) < tol_h:
                lf.N[k], lf.H[k] = -cn, -ch
                break
        else:
            canon.append((n.copy(), float(h)))


def common_origin_and_shared_planes(leaves, ident=None):
    """Give every QLeaf of one solid the same origin and make coplanar planes bit-identical; return the leaves.

    A plane shared by two leaves (a union seam, the coplanar caps of a subtraction, the internal faces of an xtru
    triangulation, polycone section planes) then yields the identical crossing in both leaves (opposite orientation:
    n and h negated exactly, so the same quotient), and the boolean cancels it with no tolerance. Without this the
    copies round differently and only a merge tolerance proportional to the coordinates hides the seam, which also
    erases genuinely thin layers seen from far away (a 50 um slab at 20 m vanished for 44% of rays from 40 m in f32).

    """
    q = [lf for lf in leaves if lf.kind == "q"]
    if not q:
        return leaves
    common = np.mean([lf.origin for lf in q], axis=0) if ident is None else ident.origin
    for lf in q:
        _move_origin(lf, common)
    canon = [] if ident is None else ident.planes  # (n, h) representatives
    for lf in q:
        if ident is not None and lf.has_q:
            _share_quadric(lf, ident)
        _share_planes(lf, canon)
    return leaves


class Compiled:
    """A flattened solid: its leaves (float64 numpy) and its static CSG expression, with float64 point membership.

    What the float64 oracle evaluates (the device arrays and the ray query live with the navigator).
    """

    def __init__(self, node, ident=None):
        self.polarity = []
        self.expr, self.leaves = flatten(node, polarity=self.polarity)
        self.leaves = common_origin_and_shared_planes(self.leaves, ident)
        self.face_base = np.cumsum([0] + [lf.nfaces for lf in self.leaves])[:-1].astype(int)

    def inside_np(self, x):
        """Return whether each point of ``x`` (n, 3) lies in the closed solid (float64, no tolerance)."""
        x = np.asarray(x, float)
        return _inside(self.expr, self.leaves, x)


def _leaf_inside(lf, x):
    xl = x - lf.origin
    if lf.kind == "q":
        ok = np.all(xl @ lf.N.T <= lf.H, axis=1) if lf.N.shape[0] else np.ones(len(x), bool)
        if lf.has_q:
            ok &= np.einsum("ni,ij,nj->n", xl, lf.A, xl) + 2 * xl @ lf.b + lf.c <= 0
        return ok
    u = xl @ np.linalg.inv(lf.lin).T
    rho = np.hypot(u[:, 0], u[:, 1])
    return (rho - lf.R) ** 2 + u[:, 2] ** 2 <= lf.r**2


def _inside(expr, leaves, x):
    if isinstance(expr, int):
        return _leaf_inside(leaves[expr], x)
    op, a, b = expr
    ia, ib = _inside(a, leaves, x), _inside(b, leaves, x)
    return ia | ib if op == "union" else (ia & ib if op == "inter" else ia & ~ib)


# ================================================================================================ leaf constructors
def q_sphere(radius, origin=(0, 0, 0)):
    """Return the ball of the given radius about ``origin``."""
    return QLeaf(np.eye(3), np.zeros(3), -radius * radius, True, [], origin)


def q_cyl(radius, z1=None, z2=None, origin=(0, 0, 0)):
    """Return ``rho <= radius``, optionally ``z1 <= z <= z2`` (solid-frame z, the origin at the section centre)."""
    o = np.asarray(origin, float)
    pl = []
    if z2 is not None:
        pl.append(((0, 0, 1), z2 - o[2]))
    if z1 is not None:
        pl.append(((0, 0, -1), -(z1 - o[2])))
    return QLeaf(np.diag([1.0, 1.0, 0.0]), np.zeros(3), -radius * radius, True, pl, o)


def q_frustum(z1, r1, z2, r2, caps=True, selector=True):
    """Return ``rho <= r(z)``, r linear from r1 at z1 to r2 at z2 (z1 < z2), with its origin at (0, 0, zm).

    Notation: A the quadric's matrix.
    """
    zm = 0.5 * (z1 + z2)
    o = np.array([0.0, 0.0, zm])
    pl = [((0, 0, 1), z2 - zm), ((0, 0, -1), -(z1 - zm))] if caps else []
    if abs(r2 - r1) < CYLINDER_REL * max(1.0, abs(r1)):
        return QLeaf(np.diag([1.0, 1.0, 0.0]), np.zeros(3), -r1 * r1, True, pl, o)
    k = (r2 - r1) / (z2 - z1)
    r0 = 0.5 * (r1 + r2)  # radius at local z = 0
    A = np.diag([1.0, 1.0, -k * k])
    b = np.array([0.0, 0.0, -r0 * k])
    if selector:
        pl.append(((0, 0, -k), r0))  # r0 + k z >= 0: one nappe
    return QLeaf(A, b, -r0 * r0, True, pl, o)


def q_halfspace(n, h):
    """Return the half-space ``n . x <= h``."""
    return QLeaf(np.zeros((3, 3)), np.zeros(3), 0.0, False, [(n, h)])


def q_planes(planes, origin=(0, 0, 0)):
    """Return the intersection of the half-spaces ``[(n, h)]`` (world offsets), at its own origin."""
    o = np.asarray(origin, float)
    pl = [(np.asarray(n, float), h - np.dot(n, o)) for n, h in planes]
    return QLeaf(np.zeros((3, 3)), np.zeros(3), 0.0, False, pl, o)


def q_cone_theta(theta, up=True):
    """Return the solid cone of half-angle theta < pi/2 about +z (up) or -z, apex at the origin."""
    t2 = math.tan(theta) ** 2
    return QLeaf(np.diag([1.0, 1.0, -t2]), np.zeros(3), 0.0, True, [((0, 0, -1.0 if up else 1.0), 0.0)])


def phi_wedge(sphi, dphi):
    """Return ``{phi in [sphi, sphi + dphi]}`` (dphi < 2 pi): one leaf if dphi <= pi, else a union of two."""
    ephi = sphi + dphi
    n1 = (math.sin(sphi), -math.cos(sphi), 0.0)
    n2 = (-math.sin(ephi), math.cos(ephi), 0.0)
    if dphi <= math.pi + ANGLE_TOL:
        return QLeaf(np.zeros((3, 3)), np.zeros(3), 0.0, False, [(n1, 0.0), (n2, 0.0)])
    return Op("union", q_halfspace(n1, 0.0), q_halfspace(n2, 0.0))


# ===================================================================================================== polytopes
def polytope_from_faces(verts, faces):
    """Return the convex polytope of the vertices and faces (vertex index lists, any orientation).

    Raises ValueError when the vertices are not convex in position (a concave arb8 or trap base, a twisted face).
    Notation: V the vertices (n, 3), P one face's vertices.
    """
    V = np.asarray(verts, float)
    cen = V.mean(axis=0)
    planes = []
    for fc in faces:
        P = V[list(fc)]
        nrm = np.cross(P[2] - P[0], P[3] - P[1]) if len(P) >= 4 else np.cross(P[1] - P[0], P[2] - P[0])
        length = np.linalg.norm(nrm)
        if length < DEGENERATE_FACE * max(1.0, np.abs(V).max() ** 2):
            continue  # degenerate face (collapsed vertices)
        nrm = nrm / length
        h = float(np.dot(nrm, P.mean(axis=0)))
        if np.dot(nrm, cen) > h:
            nrm, h = -nrm, -h
        planes.append((nrm, h))
    tol = VERTEX_TOL * max(1.0, float(np.abs(V).max()))
    for nrm, h in planes:
        if np.max(V @ nrm - h) > tol:
            raise ValueError("polytope_from_faces: the vertices are not convex in position "
                             "(a concave arb8/trap base or a twisted face)")  # fmt: skip
    return _bb(q_planes(planes, origin=cen), V.min(0), V.max(0))


def hex_faces():
    """Return the six faces of a hexahedron: vertices 0..3 at -z, 4..7 at +z (G4Trap: -x-y, +x-y, -x+y, +x+y)."""
    return [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def ring_faces(n):
    """Return the faces of a prism over an n-gon: bottom 0..n-1, top n..2n-1."""
    fs = [tuple(range(n)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        fs.append((i, j, j + n, i + n))
    return fs


def _cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _in_tri(p, a, b, c):
    return _cross2(a, b, p) >= 0 and _cross2(b, c, p) >= 0 and _cross2(c, a, p) >= 0


def _ear(pts, idx):
    """Return the position in idx of the first ear (a convex corner holding no other vertex), or None."""
    m = len(idx)
    for k in range(m):
        i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
        a, b, c = pts[i0], pts[i1], pts[i2]
        if _cross2(a, b, c) <= 0:
            continue
        if any(_in_tri(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
            continue
        return k
    return None


EAR_GUARD = 10  # ear clipping gives up after this many times n^2 passes


def ear_clip(poly):
    """Return the index triples triangulating a simple polygon (a list of (x, y))."""
    pts = [tuple(p) for p in poly]
    n = len(pts)
    area = 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
    idx = list(range(n)) if area > 0 else list(range(n))[::-1]
    tris = []
    guard = 0
    while len(idx) > 3 and guard < EAR_GUARD * n * n:
        guard += 1
        k = _ear(pts, idx)
        if k is None:
            continue
        m = len(idx)
        tris.append((idx[(k - 1) % m], idx[k], idx[(k + 1) % m]))
        idx.pop(k)
    if len(idx) != 3:
        raise ValueError("xtru polygon: ear clipping failed (self-touching or duplicate vertices)")
    tris.append(tuple(idx))
    return tris


def union_all(nodes):
    """Return the left-folded union of the nodes (None entries skipped)."""
    nodes = [n for n in nodes if n is not None]
    out = nodes[0]
    for n in nodes[1:]:
        out = Op("union", out, n)
    return out


# ================================================================================================= bounding boxes
def _bb(node, lo, hi):
    node.bbox = (np.asarray(lo, float), np.asarray(hi, float))
    return node


def corners(lo, hi):
    """Return the eight corners (8, 3) of the box [lo, hi]."""
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def node_bbox(node):
    """Return the axis-aligned box ``(lo, hi)`` of a CSG node in its own frame; None when unbounded."""
    bb = getattr(node, "bbox", None)
    if bb is not None:
        return bb
    if isinstance(node, Xf):
        c = node_bbox(node.child)
        if c is None:
            return None
        w = corners(*c) @ node.L.T + node.t
        return w.min(0), w.max(0)
    if isinstance(node, Op):
        return _op_bbox(node.op, node_bbox(node.a), node_bbox(node.b))
    return None


def _op_bbox(op, a, b):
    if op == "sub":
        return a
    if op == "inter":
        if a is None:
            return b
        if b is None:
            return a
        return np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    if a is None or b is None:
        return None
    return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])


# ====================================================================== Geant4 solids (GDML's parameters, m and rad)
def _with_phi(node, sphi, dphi):
    if dphi >= TWOPI - FULL_PHI_TOL or dphi <= 0.0:  # G4Polycone / G4Polyhedra: dphi <= 0 means 2 pi
        return node
    return Op("inter", node, phi_wedge(sphi, dphi))


def g4_box(x, y, z):
    """Return G4Box of full lengths x, y, z."""
    hx, hy, hz = 0.5 * x, 0.5 * y, 0.5 * z
    return _bb(q_planes([((1, 0, 0), hx), ((-1, 0, 0), hx), ((0, 1, 0), hy), ((0, -1, 0), hy),
                         ((0, 0, 1), hz), ((0, 0, -1), hz)]), (-hx, -hy, -hz), (hx, hy, hz))  # fmt: skip


def g4_tube(rmin, rmax, z, startphi=0.0, deltaphi=TWOPI):
    """Return G4Tubs of full length z."""
    hz = 0.5 * z
    n = q_cyl(rmax, -hz, hz)
    if rmin > 0:
        n = Op("sub", n, q_cyl(rmin))
    return _bb(_with_phi(n, startphi, deltaphi), (-rmax, -rmax, -hz), (rmax, rmax, hz))


def g4_cuttube(rmin, rmax, z, startphi, deltaphi, low, high):  # noqa: PLR0913 -- G4CutTubs's parameters
    """Return G4CutTubs of full length z; ``low`` and ``high`` are the cut planes' outward normals (x, y, z)."""
    hz = 0.5 * z
    nl = np.array(low, float)
    nl /= np.linalg.norm(nl)
    nh = np.array(high, float)
    nh /= np.linalg.norm(nh)
    # n_low . (x - (0,0,-hz)) <= 0,  n_high . (x - (0,0,hz)) <= 0
    n = QLeaf(np.diag([1.0, 1.0, 0.0]), np.zeros(3), -(rmax**2), True, [(nl, -nl[2] * hz), (nh, nh[2] * hz)])
    if rmin > 0:
        n = Op("sub", n, q_cyl(rmin))
    dzl = rmax * np.sqrt(max(0.0, 1 - nl[2] ** 2)) / max(abs(nl[2]), NORMAL_FLOOR)
    dzh = rmax * np.sqrt(max(0.0, 1 - nh[2] ** 2)) / max(abs(nh[2]), NORMAL_FLOOR)
    return _bb(_with_phi(n, startphi, deltaphi), (-rmax, -rmax, -hz - dzl), (rmax, rmax, hz + dzh))


def g4_cone(rmin1, rmax1, rmin2, rmax2, z, phi=(0.0, TWOPI)):
    """Return G4Cons of full length z; ``phi`` is (startphi, deltaphi).

    G4Cons replaces a zero inner radius at one end (the other > 0) by 1e3 * kRadTolerance: mirrored.
    """
    hz = 0.5 * z
    if rmin1 == 0.0 and rmin2 > 0.0:
        rmin1 = G4CONS_ZERO_RMIN
    if rmin2 == 0.0 and rmin1 > 0.0:
        rmin2 = G4CONS_ZERO_RMIN
    n = q_frustum(-hz, rmax1, hz, rmax2)
    if rmin1 > 0 or rmin2 > 0:
        n = Op("sub", n, q_frustum(-hz, rmin1, hz, rmin2, caps=False))
    big = max(rmax1, rmax2)
    return _bb(_with_phi(n, *phi), (-big, -big, -hz), (big, big, hz))


def merge_affine(zs, rel=MERGE_REL):
    """Return ``[(z, rmin, rmax)]`` with the redundant interior z-planes dropped.

    A plane is dropped where z keeps its direction and rmin, rmax stay affine in z across it: consecutive sections of
    one straight profile are one section (exact: their union, the shared cap removed). Geant4 does the same
    (G4Polyhedra / G4Polycone: RemoveRedundantVertices).
    """
    out = list(zs[:1])
    for k in range(1, len(zs)):
        if k + 1 < len(zs) and len(out) >= 1:
            (za, ia, oa), (zb, ib, ob), (zc, ic, oc) = out[-1], zs[k], zs[k + 1]
            if (zb - za) * (zc - zb) > 0:
                f = (zb - za) / (zc - za)
                tol = rel * max(1.0, abs(za), abs(zc), abs(oa), abs(oc))
                if abs(ia + f * (ic - ia) - ib) <= tol and abs(oa + f * (oc - oa) - ob) <= tol:
                    continue
        out.append(zs[k])
    return out


def _sections(zs):
    """Yield consecutive (za, ia, oa, zb, ib, ob) with za < zb, sections thinner than Z_TOL dropped."""
    for (za, ia, oa), (zb, ib, ob) in zip(zs[:-1], zs[1:], strict=True):
        if abs(zb - za) < Z_TOL:
            continue
        if zb < za:
            za, ia, oa, zb, ib, ob = zb, ib, ob, za, ia, oa
        yield za, ia, oa, zb, ib, ob


def g4_polycone(startphi, deltaphi, zplanes):
    """Return G4Polycone of zplanes ``[(rmin, rmax, z)]`` (GDML order)."""
    zs = merge_affine([(z, rmin, rmax) for rmin, rmax, z in zplanes])
    secs = []
    for za, ia, oa, zb, ib, ob in _sections(zs):
        s = q_frustum(za, oa, zb, ob)
        if ia > 0 or ib > 0:
            s = Op("sub", s, q_frustum(za, ia, zb, ib, caps=False))
        secs.append(s)
    big = max(z[2] for z in zs)
    return _bb(_with_phi(union_all(secs), startphi, deltaphi), (-big, -big, min(z[0] for z in zs)),
               (big, big, max(z[0] for z in zs)))  # fmt: skip


def _polyhedra_section(za, aa, zb, ab, phi, caps):
    """Return one polyhedra section between za (apothem aa) and zb (ab); ``phi`` is (sphi, dphi, nside)."""
    sphi, dphi, nside = phi
    zm = 0.5 * (za + zb)
    k = (ab - aa) / (zb - za)
    planes = []
    if dphi <= 0.0:
        dphi = TWOPI
    for i in range(nside):
        ph = sphi + (i + 0.5) * dphi / nside
        n = np.array([math.cos(ph), math.sin(ph), -k])
        planes.append((n, aa - k * za))
    if caps:
        planes += [((0, 0, 1), zb), ((0, 0, -1), -za)]
    return q_planes(planes, origin=(0.0, 0.0, zm))


def g4_polyhedra(startphi, deltaphi, numsides, zplanes):
    """Return G4Polyhedra: GDML's rmin and rmax are the distances to the sides (apothems)."""
    zs = merge_affine([(z, rmin, rmax) for rmin, rmax, z in zplanes])
    full = deltaphi >= TWOPI - FULL_PHI_TOL or deltaphi <= 0.0
    phi = (startphi, deltaphi, int(numsides))
    secs = []
    for za, ia, oa, zb, ib, ob in _sections(zs):
        s = _polyhedra_section(za, oa, zb, ob, phi, True)
        if ia > 0 or ib > 0:
            s = Op("sub", s, _polyhedra_section(za, ia, zb, ib, phi, False))
        secs.append(s)
    n = union_all(secs)
    n = n if full else Op("inter", n, phi_wedge(startphi, deltaphi))
    dph = (TWOPI if deltaphi <= 0 else deltaphi) / int(numsides)
    big = max(z[2] for z in zs) / math.cos(0.5 * min(dph, math.pi - HALF_TURN_MARGIN))
    return _bb(n, (-big, -big, min(z[0] for z in zs)), (big, big, max(z[0] for z in zs)))


def _theta_cut(n, ts, te):
    """Keep theta in [ts, te] of node n (G4Sphere's theta range)."""
    h = math.pi / 2
    if ts > ANGLE_TOL:  # remove theta < ts
        if abs(ts - h) < ANGLE_TOL:
            n = Op("sub", n, q_halfspace((0, 0, -1), 0.0))  # z >= 0 region
        elif ts < h:
            n = Op("sub", n, q_cone_theta(ts, up=True))
        else:
            n = Op("inter", n, q_cone_theta(math.pi - ts, up=False))
    if te < math.pi - ANGLE_TOL:  # keep theta <= te
        if abs(te - h) < ANGLE_TOL:
            n = Op("inter", n, q_halfspace((0, 0, -1), 0.0))
        elif te < h:
            n = Op("inter", n, q_cone_theta(te, up=True))
        else:
            n = Op("sub", n, q_cone_theta(math.pi - te, up=False))
    return n


def g4_sphere(rmin, rmax, phi=(0.0, TWOPI), theta=(0.0, math.pi)):
    """Return G4Sphere; ``phi`` is (startphi, deltaphi), ``theta`` (starttheta, deltatheta)."""
    n = q_sphere(rmax)
    if rmin > 0:
        n = Op("sub", n, q_sphere(rmin))
    n = _with_phi(n, *phi)
    n = _theta_cut(n, theta[0], theta[0] + theta[1])
    return _bb(n, (-rmax, -rmax, -rmax), (rmax, rmax, rmax))


def g4_orb(r):
    """Return G4Orb of radius r."""
    return _bb(q_sphere(r), (-r,) * 3, (r,) * 3)


def g4_ellipsoid(ax, by, cz, zcut1=0.0, zcut2=0.0):
    """Return G4Ellipsoid (Geant4 10.7): both cuts 0 means no cut; otherwise they are clamped to [-cz, cz]."""
    a, b, c = ax, by, cz
    lo, hi = zcut1, zcut2
    if lo == 0.0 and hi == 0.0:
        lo, hi = -c, c
    lo, hi = max(lo, -c), min(hi, c)
    pl = []
    if hi < c:
        pl.append(((0, 0, 1), hi))
    if lo > -c:
        pl.append(((0, 0, -1), -lo))
    return _bb(QLeaf(np.diag([1 / a**2, 1 / b**2, 1 / c**2]), np.zeros(3), -1.0, True, pl), (-a, -b, lo), (a, b, hi))


def g4_eltube(dx, dy, dz):
    """Return G4EllipticalTube of semi-axes dx, dy and half length dz."""
    a, b, hz = dx, dy, dz
    return _bb(QLeaf(np.diag([1 / a**2, 1 / b**2, 0.0]), np.zeros(3), -1.0, True,
                     [((0, 0, 1), hz), ((0, 0, -1), hz)]), (-a, -b, -hz), (a, b, hz))  # fmt: skip


def g4_elcone(dx, dy, zmax, zcut):
    """Return G4EllipticalCone: ``(x/dx)^2 + (y/dy)^2 <= (zmax - z)^2``, ``|z| <= zcut`` (dx, dy unitless).

    Notation: A the quadric's matrix.
    """
    zm, zc = zmax, zcut
    A = np.diag([1 / dx**2, 1 / dy**2, -1.0])
    b = np.array([0.0, 0.0, zm])
    zt = min(zc, zm)  # G4: |z| <= zTopCut = min(zcut, zmax)
    rx, ry = dx * (zm + zt), dy * (zm + zt)
    return _bb(QLeaf(A, b, -zm * zm, True, [((0, 0, 1), zt), ((0, 0, -1), zt), ((0, 0, 1), zm)]),
               (-rx, -ry, -zt), (rx, ry, zt))  # fmt: skip


def g4_paraboloid(rlo, rhi, dz):
    """Return G4Paraboloid of radii rlo at -dz and rhi at +dz."""
    rl, rh, hz = rlo, rhi, dz
    k1 = (rh * rh - rl * rl) / (2 * hz)
    k2 = (rh * rh + rl * rl) / 2
    return _bb(QLeaf(np.diag([1.0, 1.0, 0.0]), np.array([0.0, 0.0, -k1 / 2]), -k2, True,
                     [((0, 0, 1), hz), ((0, 0, -1), hz)]), (-rh, -rh, -hz), (rh, rh, hz))  # fmt: skip


def g4_hype(rmin, rmax, inst, outst, z):
    """Return G4Hype of full length z, stereo angles inst and outst."""
    hz = 0.5 * z
    to2, ti2 = math.tan(outst) ** 2, math.tan(inst) ** 2
    n = QLeaf(np.diag([1.0, 1.0, -to2]), np.zeros(3), -(rmax**2), True, [((0, 0, 1), hz), ((0, 0, -1), hz)])
    if rmin > 0 or inst > 0:
        n = Op("sub", n, QLeaf(np.diag([1.0, 1.0, -ti2]), np.zeros(3), -(rmin**2), True, []))
    big = math.sqrt(rmax**2 + to2 * hz * hz)
    return _bb(n, (-big, -big, -hz), (big, big, hz))


def g4_torus(rmin, rmax, rtor, startphi=0.0, deltaphi=TWOPI):
    """Return G4Torus of swept radius rtor and tube radii rmin, rmax."""
    n = TorusLeaf(rtor, rmax)
    if rmin > 0:
        n = Op("sub", n, TorusLeaf(rtor, rmin))
    big = rtor + rmax
    return _bb(_with_phi(n, startphi, deltaphi), (-big, -big, -rmax), (big, big, rmax))


def g4_trd(x1, x2, y1, y2, z):
    """Return G4Trd of full lengths x1, y1 at -z/2, x2, y2 at +z/2."""
    hx1, hx2, hy1, hy2, hz = (0.5 * v for v in (x1, x2, y1, y2, z))
    verts = [(-hx1, -hy1, -hz), (hx1, -hy1, -hz), (-hx1, hy1, -hz), (hx1, hy1, -hz),
             (-hx2, -hy2, hz), (hx2, -hy2, hz), (-hx2, hy2, hz), (hx2, hy2, hz)]  # fmt: skip
    return polytope_from_faces(verts, hex_faces())


def g4_para(x, y, z, alpha, theta, phi):
    """Return G4Para of full lengths x, y, z and angles alpha, theta, phi.

    Notation: Y, Z the corner's y and z.
    """
    dx, dy, dz = 0.5 * x, 0.5 * y, 0.5 * z
    ta, tt = math.tan(alpha), math.tan(theta)
    verts = []
    for sz in (-1, 1):
        for sy in (-1, 1):
            for sx in (-1, 1):
                Z, Y = sz * dz, sy * dy
                verts.append((sx * dx + Y * ta + Z * tt * math.cos(phi), Y + Z * tt * math.sin(phi), Z))
    return polytope_from_faces(verts, hex_faces())


def g4_trap(z, theta, phi, bottom, top):
    """Return G4Trap of full length z; ``bottom`` is (y1, x1, x2, alpha1) at -z/2 and ``top`` (y2, x3, x4, alpha2)."""
    y1, x1, x2, alpha1 = bottom
    y2, x3, x4, alpha2 = top
    dz, dy1, dx1, dx2, dy2, dx3, dx4 = (0.5 * v for v in (z, y1, x1, x2, y2, x3, x4))
    tth, cph, sph = math.tan(theta), math.cos(phi), math.sin(phi)
    t1, t2 = math.tan(alpha1), math.tan(alpha2)
    verts = [(-dz * tth * cph - dy1 * t1 - dx1, -dz * tth * sph - dy1, -dz),
             (-dz * tth * cph - dy1 * t1 + dx1, -dz * tth * sph - dy1, -dz),
             (-dz * tth * cph + dy1 * t1 - dx2, -dz * tth * sph + dy1, -dz),
             (-dz * tth * cph + dy1 * t1 + dx2, -dz * tth * sph + dy1, -dz),
             (dz * tth * cph - dy2 * t2 - dx3, dz * tth * sph - dy2, dz),
             (dz * tth * cph - dy2 * t2 + dx3, dz * tth * sph - dy2, dz),
             (dz * tth * cph + dy2 * t2 - dx4, dz * tth * sph + dy2, dz),
             (dz * tth * cph + dy2 * t2 + dx4, dz * tth * sph + dy2, dz)]  # fmt: skip
    return polytope_from_faces(verts, hex_faces())


def g4_arb8(verts_xy, dz):
    """Return G4GenericTrap with planar faces (a twisted face raises): 8 (x, y) vertices, 0..3 at -dz.

    Notation: V the vertices (8, 3), P one side face's vertices.
    """
    hz = dz
    V = [(x, y, -hz) for x, y in verts_xy[:4]] + [(x, y, hz) for x, y in verts_xy[4:]]
    V = np.asarray(V)
    for i in range(4):
        j = (i + 1) % 4
        P = V[[i, j, j + 4, i + 4]]
        nrm = np.cross(P[2] - P[0], P[3] - P[1])
        if np.linalg.norm(nrm) > 0:
            dev = abs(np.dot(nrm / np.linalg.norm(nrm), P[1] - P[0]))
            if dev > TWIST_TOL:
                raise NotImplementedError("arb8 with a twisted (non-planar) face")
    fs = [(0, 1, 2, 3), (4, 5, 6, 7)] + [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4) for i in range(4)]
    return polytope_from_faces(V, fs)


def g4_tet(v1, v2, v3, v4):
    """Return G4Tet of four vertices (x, y, z)."""
    verts = [np.asarray(v, float) for v in (v1, v2, v3, v4)]
    return polytope_from_faces(verts, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])


def g4_xtru(verts, sections):
    """Return G4ExtrudedSolid: ``verts`` [(x, y)], ``sections`` [(z, x offset, y offset, scale)].

    Notation: V2 the polygon's vertices (n, 2), P one prism's vertices.
    """
    tris = ear_clip(verts)
    V2 = np.asarray(verts, float)
    pieces = []
    for (za, xa, ya, sa), (zb, xb, yb, sb) in zip(sections[:-1], sections[1:], strict=True):
        for tri in tris:
            P = [(xa + sa * V2[i][0], ya + sa * V2[i][1], za) for i in tri] + \
                [(xb + sb * V2[i][0], yb + sb * V2[i][1], zb) for i in tri]  # fmt: skip
            pieces.append(polytope_from_faces(P, ring_faces(3)))
    n = union_all(pieces)
    los = np.min([node_bbox(p)[0] for p in pieces], axis=0)
    his = np.max([node_bbox(p)[1] for p in pieces], axis=0)
    return _bb(n, los, his)
