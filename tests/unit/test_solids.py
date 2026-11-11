"""The host CSG model (lumen.build.solids): Geant4's solids against their defining inequalities (float64 point
membership), exact boxes, the box proofs, flattening, the shared-plane rewrite and the polygon helpers.

The float32 ray kernels are tested with the navigator; these test the host model those kernels are built from."""

import math

import numpy as np
import pytest

from lumen.build import solids

RNG = np.random.default_rng(7)


def _inside(node, x):
    return solids.Compiled(node).inside_np(x)


def _cloud(lo, hi, n=20000):
    return RNG.uniform(lo, hi, (n, 3))


def _check(node, want, lo, hi, n=20000):
    """membership equals the defining inequality on a cloud over the padded box, and the box holds the solid"""
    x = _cloud(np.asarray(lo) * 1.2 - 0.01, np.asarray(hi) * 1.2 + 0.01, n)
    got = _inside(node, x)
    np.testing.assert_array_equal(got, want(x))
    blo, bhi = solids.node_bbox(node)
    assert np.all(x[got] >= blo - 1e-12) and np.all(x[got] <= bhi + 1e-12)
    assert got.any() and not got.all()


def _rho(x):
    return np.hypot(x[:, 0], x[:, 1])


def _phi_in(x, s, d):
    return np.mod(np.arctan2(x[:, 1], x[:, 0]) - s, 2 * np.pi) <= d


def test_box_tube_and_cut_tube():
    _check(
        solids.g4_box(0.2, 0.4, 0.6),
        lambda x: np.all(np.abs(x) <= [0.1, 0.2, 0.3], 1),
        (-0.1, -0.2, -0.3),
        (0.1, 0.2, 0.3),
    )
    s, d = 0.3, 4.0  # a phi range above pi: the wedge is a union
    _check(
        solids.g4_tube(0.05, 0.2, 0.4, s, d),
        lambda x: (_rho(x) <= 0.2) & (_rho(x) >= 0.05) & (np.abs(x[:, 2]) <= 0.2) & _phi_in(x, s, d),
        (-0.2, -0.2, -0.2),
        (0.2, 0.2, 0.2),
    )
    lo, hi = np.array([0.0, -0.3, -1.0]), np.array([0.0, 0.0, 1.0])
    nl, nh = lo / np.linalg.norm(lo), hi / np.linalg.norm(hi)
    _check(
        solids.g4_cuttube(0.0, 0.1, 0.4, 0.0, solids.TWOPI, lo, hi),
        lambda x: (_rho(x) <= 0.1) & ((x - [0, 0, -0.2]) @ nl <= 0) & ((x - [0, 0, 0.2]) @ nh <= 0),
        *solids.node_bbox(solids.g4_cuttube(0.0, 0.1, 0.4, 0.0, solids.TWOPI, lo, hi)),
    )


def test_cone_and_g4cons_zero_inner_radius():
    def want(x):
        f = (x[:, 2] + 0.25) / 0.5
        return (np.abs(x[:, 2]) <= 0.25) & (_rho(x) <= 0.1 + 0.1 * f) & (_rho(x) >= 0.05 * f)

    node = solids.g4_cone(0.0, 0.1, 0.05, 0.2, 0.5)
    _check(node, want, (-0.2, -0.2, -0.25), (0.2, 0.2, 0.25))
    # G4Cons: the zero inner radius at -dz becomes 1e3 * kRadTolerance (1e-6 mm): a hole of 1 nm there
    assert node.op == "sub" and solids.G4CONS_ZERO_RMIN == 1e-9
    assert _inside(node, np.array([[5e-10, 0.0, -0.25], [2e-9, 0.0, -0.25]])).tolist() == [False, True]


def test_sphere_theta_and_phi_ranges():
    for ts, dt in ((0.0, math.pi), (0.4, 0.8), (math.pi / 2, math.pi / 2), (0.2, math.pi / 2 - 0.2), (1.9, 0.9)):

        def want(x, ts=ts, dt=dt):
            r = np.linalg.norm(x, axis=1)
            th = np.arccos(np.clip(x[:, 2] / np.maximum(r, 1e-300), -1, 1))
            return (r <= 0.3) & (r >= 0.1) & (th >= ts) & (th <= ts + dt) & _phi_in(x, 0.5, 2.0)

        _check(solids.g4_sphere(0.1, 0.3, (0.5, 2.0), (ts, dt)), want, (-0.3,) * 3, (0.3,) * 3)


def test_polycone_and_polyhedra():
    planes = [(0.0, 0.1, -0.2), (0.0, 0.1, 0.0), (0.02, 0.2, 0.1), (0.04, 0.3, 0.2)]  # (rmin, rmax, z)

    def want_cone(x):
        z, r = x[:, 2], _rho(x)
        rmax = np.interp(z, [-0.2, 0.0, 0.1, 0.2], [0.1, 0.1, 0.2, 0.3])
        rmin = np.interp(z, [-0.2, 0.0, 0.1, 0.2], [0.0, 0.0, 0.02, 0.04])
        return (z >= -0.2) & (z <= 0.2) & (r <= rmax) & (r >= rmin)

    _check(solids.g4_polycone(0.0, solids.TWOPI, planes), want_cone, (-0.3, -0.3, -0.2), (0.3, 0.3, 0.2))
    n = 6

    def want_hedra(x):  # apothems: the distance to the nearest side
        ph = np.arctan2(x[:, 1], x[:, 0])
        k = np.floor(ph / (2 * np.pi / n))
        a = _rho(x) * np.cos(ph - (k + 0.5) * 2 * np.pi / n)
        return (np.abs(x[:, 2]) <= 0.1) & (a <= 0.2) & (a >= 0.05)

    _check(
        solids.g4_polyhedra(0.0, solids.TWOPI, n, [(0.05, 0.2, -0.1), (0.05, 0.2, 0.1)]),
        want_hedra,
        (-0.25, -0.25, -0.1),
        (0.25, 0.25, 0.1),
    )
    # a straight profile over three planes is one section (Geant4 removes redundant vertices)
    assert solids.merge_affine([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 3.0)]) == [
        (0.0, 0.0, 1.0),
        (2.0, 0.0, 3.0),
    ]


def test_quadric_solids():
    _check(
        solids.g4_ellipsoid(0.1, 0.2, 0.3, -0.1, 0.2),
        lambda x: (np.sum((x / [0.1, 0.2, 0.3]) ** 2, 1) <= 1) & (x[:, 2] >= -0.1) & (x[:, 2] <= 0.2),
        (-0.1, -0.2, -0.1),
        (0.1, 0.2, 0.2),
    )
    _check(
        solids.g4_eltube(0.1, 0.2, 0.3),
        lambda x: (np.sum((x[:, :2] / [0.1, 0.2]) ** 2, 1) <= 1) & (np.abs(x[:, 2]) <= 0.3),
        (-0.1, -0.2, -0.3),
        (0.1, 0.2, 0.3),
    )
    _check(
        solids.g4_elcone(0.5, 0.25, 0.4, 0.2),
        lambda x: ((x[:, 0] / 0.5) ** 2 + (x[:, 1] / 0.25) ** 2 <= (0.4 - x[:, 2]) ** 2) & (np.abs(x[:, 2]) <= 0.2),
        *solids.node_bbox(solids.g4_elcone(0.5, 0.25, 0.4, 0.2)),
    )
    _check(
        solids.g4_paraboloid(0.05, 0.2, 0.3),
        lambda x: (_rho(x) ** 2 <= (0.2**2 - 0.05**2) / 0.6 * x[:, 2] + (0.2**2 + 0.05**2) / 2)
        & (np.abs(x[:, 2]) <= 0.3),
        (-0.2, -0.2, -0.3),
        (0.2, 0.2, 0.3),
    )
    _check(
        solids.g4_hype(0.05, 0.1, 0.3, 0.5, 0.4),
        lambda x: (_rho(x) ** 2 <= 0.1**2 + math.tan(0.5) ** 2 * x[:, 2] ** 2)
        & (_rho(x) ** 2 >= 0.05**2 + math.tan(0.3) ** 2 * x[:, 2] ** 2)
        & (np.abs(x[:, 2]) <= 0.2),
        *solids.node_bbox(solids.g4_hype(0.05, 0.1, 0.3, 0.5, 0.4)),
    )


def test_torus():
    _check(
        solids.g4_torus(0.02, 0.05, 0.2, 0.0, 3.0),
        lambda x: ((_rho(x) - 0.2) ** 2 + x[:, 2] ** 2 <= 0.05**2)
        & ((_rho(x) - 0.2) ** 2 + x[:, 2] ** 2 >= 0.02**2)
        & _phi_in(x, 0.0, 3.0),
        (-0.25, -0.25, -0.05),
        (0.25, 0.25, 0.05),
    )


def test_polytopes():
    trd = solids.g4_trd(0.2, 0.4, 0.2, 0.2, 0.2)
    _check(
        trd,
        lambda x: (np.abs(x[:, 2]) <= 0.1) & (np.abs(x[:, 0]) <= 0.15 + 0.5 * x[:, 2]) & (np.abs(x[:, 1]) <= 0.1),
        (-0.2, -0.1, -0.1),
        (0.2, 0.1, 0.1),
    )
    tet = solids.g4_tet((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1))
    _check(tet, lambda x: np.all(x >= 0, 1) & (x.sum(1) <= 0.1), (0, 0, 0), (0.1, 0.1, 0.1))
    para = solids.g4_para(0.2, 0.2, 0.2, 0.0, 0.0, 0.0)  # no skew: a box
    _check(para, lambda x: np.all(np.abs(x) <= 0.1, 1), (-0.1,) * 3, (0.1,) * 3)
    trap = solids.g4_trap(0.2, 0.0, 0.0, (0.2, 0.2, 0.2, 0.0), (0.2, 0.2, 0.2, 0.0))  # a box too
    _check(trap, lambda x: np.all(np.abs(x) <= 0.1, 1), (-0.1,) * 3, (0.1,) * 3)
    sq = [(-0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)]
    _check(solids.g4_arb8(sq + sq, 0.1), lambda x: np.all(np.abs(x) <= 0.1, 1), (-0.1,) * 3, (0.1,) * 3)
    with pytest.raises(NotImplementedError, match="twisted"):
        solids.g4_arb8(sq + [(-0.1, -0.1), (-0.1, 0.1), (0.1, 0.2), (0.1, -0.1)], 0.1)
    with pytest.raises(ValueError, match="convex"):
        solids.polytope_from_faces(
            [(0, 0, 0), (1, 0, 0), (0.2, 0.2, 0), (0, 1, 0)] + [(0, 0, 1), (1, 0, 1), (0.2, 0.2, 1), (0, 1, 1)],
            solids.ring_faces(4),
        )


def test_xtru_of_a_concave_polygon():
    poly = [(0, 0), (0.2, 0), (0.2, 0.2), (0.1, 0.05), (0, 0.2)]  # concave: ear clipping, the pieces united
    node = solids.g4_xtru(poly, [(-0.1, 0, 0, 1.0), (0.1, 0, 0, 1.0)])
    assert len(solids.ear_clip(poly)) == 3

    x = _cloud((-0.05, -0.05, -0.15), (0.25, 0.25, 0.15))
    got = _inside(node, x)
    xy = x[:, :2]
    tri_in = np.zeros(len(x), bool)
    pts = np.asarray(poly)
    for a, b, c in solids.ear_clip(poly):
        tri_in |= _in_triangle(xy, pts[a], pts[b], pts[c])
    np.testing.assert_array_equal(got, tri_in & (np.abs(x[:, 2]) <= 0.1))


def _in_triangle(p, a, b, c):
    def cross(o, u, v):
        return (u[0] - o[..., 0]) * (v[1] - o[..., 1]) - (u[1] - o[..., 1]) * (v[0] - o[..., 0])

    d1, d2, d3 = cross(p, a, b), cross(p, b, c), cross(p, c, a)
    return ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))


def test_flatten_transforms_and_polarity():
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    node = solids.Op(
        "sub",
        solids.Xf(rz, [0.5, 0, 0], solids.g4_box(0.4, 0.2, 0.2)),
        solids.Xf(np.eye(3), [0.5, 0, 0], solids.g4_orb(0.05)),
    )
    polarity = []
    expr, leaves = solids.flatten(node, polarity=polarity)
    assert expr == ("sub", 0, 1) and polarity == [1, -1]
    x = np.array([[0.5, 0.15, 0.0], [0.65, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert _inside(node, x).tolist() == [True, False, False]  # rotated: the box's long side along y; the hole
    with pytest.raises(ValueError, match="unknown operation"):
        solids.Op("xor", node, node)


def test_shared_planes_are_bit_identical():
    """two boxes sharing a face: after the rewrite the face is one plane, negated exactly"""
    a = solids.Xf(np.eye(3), [-0.1, 0, 0], solids.g4_box(0.2, 0.2, 0.2))
    b = solids.Xf(np.eye(3), [0.1, 0, 0], solids.g4_box(0.2, 0.2, 0.2))
    c = solids.Compiled(solids.Op("union", a, b))
    la, lb = c.leaves
    assert np.array_equal(la.origin, lb.origin)
    ka = int(np.flatnonzero(np.isclose(la.N[:, 0], 1.0))[0])
    kb = int(np.flatnonzero(np.isclose(lb.N[:, 0], -1.0))[0])
    assert np.array_equal(la.N[ka], -lb.N[kb]) and la.H[ka] == -lb.H[kb]
    assert c.inside_np(np.array([[0.0, 0.0, 0.0]])).tolist() == [True]  # the seam is inside
    assert not c.is_single_convex() and solids.Compiled(solids.g4_orb(0.1)).is_single_convex()


def test_box_proofs():
    big = solids.Op("sub", solids.g4_box(1.0, 1.0, 1.0), solids.g4_orb(0.1))
    inner = solids.corners(np.array([0.2, 0.2, 0.2]), np.array([0.4, 0.4, 0.4]))
    assert solids.box_inside(big, inner)
    assert not solids.box_inside(big, inner - 0.25)  # the box meets the hole
    assert solids.box_outside(big, inner + 1.0) and not solids.box_outside(big, inner)
    tor = solids.g4_torus(0.0, 0.05, 0.3)
    assert solids.box_outside(tor, inner + 1.0) and not solids.box_inside(tor, inner)
