"""The navigator (volume tracking) against the float64 oracle on small scenes: point location, the first visible change
from random points, walks continued from each hit by surface identity (no push), the live t's derivatives against
float64 finite differences, frame groups (rotated wires), split unions, meshes and the zero-length rules (the placement
check's tests are in test_placement_check.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nav_toys import (
    F32,
    I32,
    T_ORC,
    defn,
    g4,
    locate,
    make_scene,
    mesh_def,
    nav_of,
    one,
    per_ray,
    query,
    rays,
    scene_mesh,
    scene_split,
    tet_tris,
    walk_check,
)

from lumen.build import lowering, tracked
from lumen.compare import oracle
from lumen.geometry import navigate, solids
from lumen.rules import navigation


@pytest.fixture(autouse=True)
def _highest_matmul():
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.parametrize("name", ["nest", "bool", "wires", "mesh"])
def test_locate_matches_oracle(name, toys):
    sc, nav = toys(name)
    X = np.random.default_rng(3).uniform(-0.95, 0.95, (4000, 3))
    if name == "wires":  # near the wires: points in the plane volume
        X = np.random.default_rng(3).uniform([-0.01, -0.35, -0.35], [0.01, 0.35, 0.35], (4000, 3))
    v = locate(nav, X)
    med = nav.host["cls"][v]
    ref = oracle.Oracle(sc).label(X)[0]
    bad = np.flatnonzero(med != ref)
    assert len(bad) <= 2, (name, len(bad), X[bad[:4]].tolist(), med[bad[:4]].tolist(), ref[bad[:4]].tolist())


@pytest.mark.parametrize("name", ["nest", "bool", "mesh"])
def test_walks_match_oracle(name, toys):
    sc, nav = toys(name)
    agree, total, bad = walk_check(sc, nav=nav, n=800, steps=4)
    assert total > 600 and len(bad) <= 0.002 * total, (name, len(bad), total, bad[:3])


@pytest.mark.parametrize("name", ["coincident", "near"])
def test_the_volume_before_a_crossing_is_the_one_the_ray_leaves(name, toys):
    """vol_before is the volume holding the ray just before the crossing (locate(x - eps d)), across coincident faces
    and faces closer than the on-face tolerance (a zero-length stay in a volume between them is no stay: the crossing
    is from the volume before it), media shared or not, after reflections; vol_after likewise where the faces
    coincide. The "near" faces (0.3 um apart) are a float32 crossing, not the oracle's float64 one: no oracle there."""
    sc, nav = toys(name)
    near = name == "near"
    agree, total, bad = walk_check(sc, nav=nav, n=800, steps=4, sides=("vol_before",) if near else
                                   ("vol_before", "vol_after"), use_oracle=not near)  # fmt: skip
    assert total > 800 and not [b for b in bad if b[0] in ("vol_before", "vol_after")], bad[:4]
    assert len(bad) <= 0.002 * total, (len(bad), total, bad[:3])


def test_wires_frame_group_and_walks(toys):
    sc, nav = toys("wires")
    assert nav.stats["framed"] >= 1, nav.stats
    agree, total, bad = walk_check(sc, nav=nav, n=600, steps=3)
    assert len(bad) <= 0.002 * total, (len(bad), total, bad[:3])
    # rays aimed at the wire plane hit wires as the oracle says
    rng = np.random.default_rng(5)
    O_ = np.c_[np.full(500, -0.5), rng.uniform(-0.25, 0.25, (500, 2))]
    Dr = np.tile([1.0, 0, 0], (500, 1)) + rng.normal(0, 0.05, (500, 3))
    Dr /= np.linalg.norm(Dr, axis=1, keepdims=True)
    r = query(nav, O_, Dr, locate(nav, O_), t_max=10.0)
    orc = oracle.Oracle(sc)
    ref = [orc.query(O_[i], Dr[i], T_ORC, 10.0, push=0.0) for i in range(len(O_))]
    ra = np.asarray(r["cls_after"])
    assert sum(ref[i]["after"][0] != ra[i] for i in range(len(O_))) <= 1
    assert sum(x["after"][0] == 2 for x in ref) > 10  # some rays do hit a wire


def test_split_unions_exact():
    sc = scene_split()
    sp = lowering.split_unions(sc, min_leaves=8)
    assert sp.definitions["frame"].member is None and len(sp.definitions["frame"].placements) == 20
    X = np.random.default_rng(4).uniform([-0.35, -0.05, -0.05], [0.35, 0.05, 0.05], (5000, 3))
    np.testing.assert_array_equal(oracle.Oracle(sp).label(X)[0], oracle.Oracle(sc).label(X)[0])
    agree, total, bad = walk_check(sp, n=600, steps=3, orc_scene=sc)
    assert len(bad) <= 0.002 * total, (len(bad), total, bad[:3])


def _jets_case(nav, n=300, seed=10):
    O_, Dr = rays(n, 9, half=0.6)
    vol = locate(nav, O_)
    tg = np.random.default_rng(seed).normal(size=(n, 6))
    return nav, O_, Dr, vol, tg


def t_jets(nav, o, d, v, last, u, t_max):
    """(t, y) of one query and their first and second derivatives along the tangent u = (do, dd), y = o + t d the
    hit point (the custom_jvp's live t); one program for the three jet tests"""

    def f(oo, dd):
        t = nav.query(oo, dd, v, last=last, t_max=t_max)["t"]
        return jnp.concatenate([t[None], oo + t * dd])

    def g(oo, dd):
        return jax.jvp(f, (oo, dd), (u[:3], u[3:]))

    (ty, d1), (_, d2) = jax.jvp(g, (o, d), (u[:3], u[3:]))
    return ty, d1, d2


def test_live_t_derivative_matches_float64_fd(toys):
    """d t / d (o, d) by the custom_jvp against central differences of the oracle's float64 t"""
    sc, nav = toys("nest")
    nav, O_, Dr, vol, tg = _jets_case(nav)
    orc = oracle.Oracle(sc)
    ty, d1, _ = per_ray(t_jets, nav, O_, Dr, vol, np.full(len(vol), -1), tg, shared=(10.0,))
    t, dt = ty[:, 0], d1[:, 0]
    h = 1e-6
    errs = []
    for i in range(len(O_)):  # every ray: the oracle hits where the navigator does, and no step crosses a surface
        r0 = orc.query(O_[i], Dr[i], T_ORC, 10.0, push=0.0)
        assert r0["hit"] and abs(r0["t"] - t[i]) <= 1e-5, (i, r0, float(t[i]))
        dp, dm = Dr[i] + h * tg[i, 3:], Dr[i] - h * tg[i, 3:]
        # the oracle normalises d: differentiate t * |d| (the navigator's d is used as given)
        rp = orc.query(O_[i] + h * tg[i, :3], dp, T_ORC, 10.0, push=0.0)
        rm = orc.query(O_[i] - h * tg[i, :3], dm, T_ORC, 10.0, push=0.0)
        assert rp["hit"] and rm["hit"] and rp["after"] == r0["after"] == rm["after"], (i, "the step crosses a surface")
        fd = (rp["t"] / np.linalg.norm(dp) - rm["t"] / np.linalg.norm(dm)) / (2 * h)
        errs.append(abs(fd - dt[i]) / (1 + abs(fd)))
    # a float32 derivative against float64 differences (h^2 and rounding below 1e-10), every ray tested, none an
    # outlier (measured on the CPU: median 2.6e-8, largest 2.2e-7 over the 300 rays)
    assert len(errs) == len(O_) and np.median(errs) < 1e-4 and np.max(errs) < 1e-2, (np.median(errs), np.max(errs))


def test_live_t_second_derivative_matches_float64_fd(toys):
    """d2 t and d2 y (the hit point o + t d) along a tangent of (o, d) against the oracle's float64 second differences.

    The rule's t must be live both inside it and as the primal it returns (the moving-source method's second-order
    jets: a source-position gradient).
    """
    sc, nav = toys("nest")
    nav, O_, Dr, vol, tg = _jets_case(nav)
    orc = oracle.Oracle(sc)
    ty, _, d2 = per_ray(t_jets, nav, O_, Dr, vol, np.full(len(vol), -1), tg, shared=(10.0,))
    h = 1e-4
    errs = []
    for i in range(len(O_)):  # every ray: the oracle hits where the navigator does, and no step crosses a surface
        r0 = orc.query(O_[i], Dr[i], T_ORC, 10.0, push=0.0)
        assert r0["hit"] and abs(r0["t"] - ty[i, 0]) <= 1e-5, (i, r0, float(ty[i, 0]))
        vals = []
        for s in (-1, 0, 1):
            oo, dd = O_[i] + s * h * tg[i, :3], Dr[i] + s * h * tg[i, 3:]
            r = orc.query(oo, dd, T_ORC, 10.0, push=0.0)
            assert r["hit"] and r["after"] == r0["after"], (i, "the step crosses a surface")
            t = r["t"] / np.linalg.norm(dd)  # the oracle normalises d; the navigator uses it as given
            vals.append(np.concatenate([[t], oo + t * dd]))
        fd2 = (vals[2] - 2 * vals[1] + vals[0]) / h**2
        errs.append(np.max(np.abs(fd2 - d2[i]) / (1 + np.abs(fd2))))
    # float32 jets against float64 second differences (h^2 and rounding / h^2 about 1e-8), every ray tested
    # (measured on the CPU: median 9.0e-8, largest 1.2e-4 over the 300 rays)
    assert len(errs) == len(O_) and np.median(errs) < 1e-5 and np.max(errs) < 1e-2, (np.median(errs), np.max(errs))


def held_and_query(nav, o, d, v, u):
    """[(t, normal), first and second derivatives along u] of the query (fq) and of the held surface (fs)"""
    G = tracked.TrackedGeometry(nav, nav.n + 1, np.full(nav.n + 1, -1, np.int32))
    q = G.query(None, o, d, 0, 10.0, vol=v)

    def fq(oo, dd):
        r = G.query(None, oo, dd, 0, 10.0, vol=v)
        return r["t"], r["normal"]

    def fs(oo, dd):
        return G.surface(None, q, oo, dd)

    out = []
    for f in (fq, fs):

        def g1(oo, dd, f=f):
            return jax.jvp(f, (oo, dd), (u[:3], u[3:]))[1]

        out.append((f(o, d), g1(o, d), jax.jvp(g1, (o, d), (u[:3], u[3:]))[1]))
    return out


def test_held_surface_is_the_query_in_value_and_jets(toys):
    """surface_at (the winner held, no second walk) gives the query's own t and normal and their first derivatives
    along a tangent bit for bit, the second to float32 rounding"""
    sc, nav = toys("nest")
    nav = navigate.Navigator(nav.arrays.with_classes(list(range(nav.n)), [], nav.n))
    O_, Dr = rays(300, 9, half=0.6)
    vol = locate(nav, O_)
    tg = np.random.default_rng(11).normal(size=(300, 6)).astype(np.float32)
    a, b = per_ray(held_and_query, nav, O_, Dr, vol, tg)
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    for x, y in zip(la[:4], lb[:4], strict=False):  # t, normal and their first derivatives: bit for bit
        assert np.array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(la[4:], lb[4:], strict=False):  # second: to float32 rounding (fusion order)
        x, y = np.asarray(x), np.asarray(y)
        assert np.allclose(x, y, rtol=1e-4, atol=1e-6 * np.max(np.abs(x))), np.max(np.abs(x - y))


def query_and_surface(nav, o, d, v):
    """the query's dict, and surface_at on its winner"""
    r = nav.query(o, d, v, t_max=10.0)
    win = jnp.stack([r["item"], r["leaf"], r["face"], r["zero"].astype(I32)])
    return r, navigate.surface_at(nav, dict(t=r["t"], hit=r["hit"], win=win), o, d)


def test_surface_at_gives_the_query_s_crossing(toys):
    """surface_at on the query's winner gives the query's t and normal, without walking again"""
    nav = toys("nest")[1]
    o, d = np.array([[0.1, 0.0, -0.8]]), np.array([[0.0, 0.0, 1.0]])
    v = locate(nav, o)
    _, (t, n) = one(per_ray(query_and_surface, nav, o, d, v))
    r = one(query(nav, o, d, v))
    assert float(t) == float(r["t"]) and np.array_equal(np.asarray(n), np.asarray(r["normal"]))


def _q(nav):
    """one query (o, d, vol, last) -> its dict"""
    return lambda o, d, v, l_: one(query(nav, o, d, v, l_))


def test_reversal_at_the_face_just_crossed_exits(toys):
    """a photon that has just entered a volume and heads straight back out makes a zero-length exit to the mother"""
    nav = toys("nest")[1]
    q = _q(nav)
    o = np.array([0.1, 0.0, -0.8], np.float32)
    up, down = np.array([0.0, 0.0, 1.0], np.float32), np.array([0.0, 0.0, -1.0], np.float32)
    v_world = int(locate(nav, o)[0])
    r1 = q(o, up, I32(v_world), I32(-1))  # into the tank's bottom face
    assert r1["hit"] and abs(float(r1["t"]) - 0.3) < 1e-5 and int(r1["vol_after"]) != v_world
    # the hit point 0.3 um inside the face, within the identity band (8 eps |x| ~ 0.8 um)
    x = o + (np.float32(r1["t"]) + np.float32(3e-7)) * up
    r2 = q(x, down, I32(r1["vol_after"]), I32(r1["item"]))
    assert r2["hit"] and float(r2["t"]) < 1e-5 and int(r2["vol_after"]) == v_world, r2


def test_turning_back_at_the_face_just_left_enters_it(toys):
    """Geant4: a photon on X's face heading into X enters X (a zero-length step) -- also when X is the volume just
    left. The side is the photon's state (srf), only the direction decides; starts 0-3 ulps either side, 1-80 deg"""
    nav = toys("nest")[1]

    def q(X, D, v, l_):
        return query(nav, X, D, v, l_)

    o = np.array([0.1, 0.0, -0.4], np.float32)
    v_in = int(locate(nav, o)[0])
    r1 = one(query(nav, o, np.array([0.0, 0.0, -1.0]), v_in, -1))
    v_out, x0 = int(r1["vol_after"]), o + np.float32(r1["t"]) * np.array([0, 0, -1], np.float32)
    assert r1["hit"] and v_out != v_in and int(r1["item"]) == v_in, r1
    srf = int(r1["srf"])  # the tank, its bottom face, the outward sign: the state
    assert int(nav.srf_decode(srf)[0]) == v_in and bool(nav.srf_decode(srf)[4]), srf
    rng = np.random.default_rng(1)
    N = 300
    X = np.repeat(x0[None], N, 0).astype(np.float32)
    X[:, 2] += rng.integers(-3, 4, N).astype(np.float32) * np.float32(6e-8)
    ang, phi = np.radians(rng.uniform(1.0, 80.0, N)), rng.uniform(0, 2 * np.pi, N)
    Dr = np.stack([np.sin(ang) * np.cos(phi), np.sin(ang) * np.sin(phi), np.cos(ang)], 1).astype(np.float32)
    r = q(X, Dr, v_out, srf)
    ok = r["hit"] & (r["t"] < 1e-5) & (r["vol_after"] == v_in)
    assert ok.all(), (int((~ok).sum()), r["t"][~ok][:5], r["vol_after"][~ok][:5])
    # the direction from the event's own normal (as the step passes it) answers alike
    rs = query(nav, X, Dr, v_out, srf, sn=r1["normal"])
    for k_ in ("hit", "t", "vol_after", "zero", "srf"):
        np.testing.assert_array_equal(rs[k_], r[k_], err_msg=k_)
    Dr[:, 2] *= -1  # heading on away from the tank: no zero-length entry
    r = q(X, Dr, v_out, srf)
    bad = r["hit"] & (r["t"] < 1e-5)
    assert bad.sum() == 0, (int(bad.sum()), r["vol_after"][bad][:5])


def test_grazing_reflection_inside_a_far_sphere_meets_its_far_side():
    """reflected off the inside of a sphere 20 m out at 0.25-3 deg: the next crossing is the same sphere's far side, a
    chord 2 R sin(a) away (the start's own root is skipped, never the far one); below 0.25 deg never no hit"""
    sc = make_scene({"world": defn("world", solids.g4_box(100.0, 100.0, 100.0), 1, [("orb", np.eye(3), (20.0, 0, 0))]),
                     "orb": defn("orb", solids.g4_orb(0.25), 2)})  # fmt: skip
    nav = nav_of(sc)
    vo = nav.defs.index("orb")
    o = np.array([20.0, 0.0, 0.0], np.float32)
    r1 = one(query(nav, o, np.array([0.0, 1.0, 0.0]), vo, -1))
    assert r1["hit"] and abs(float(r1["t"]) - 0.25) < 1e-4, r1
    x0 = o + np.float32(r1["t"]) * np.array([0, 1, 0], np.float32)
    rng = np.random.default_rng(2)
    N = 200
    a = np.radians(rng.uniform(0.25, 3.0, N))
    phi = rng.uniform(0, 2 * np.pi, N)
    Dr = np.stack([np.cos(a) * np.cos(phi), -np.sin(a), np.cos(a) * np.sin(phi)], 1).astype(np.float32)
    X = np.repeat(x0[None], N, 0).astype(np.float32)

    def q(X, D):
        return query(nav, X, D, vo, r1["srf"])

    r = q(X, Dr)
    chord = 2 * 0.25 * np.sin(a)
    ok = r["hit"] & (np.abs(r["t"] - chord) < 1e-3 * chord + 1e-4) & (r["vol_after"] != vo)
    assert ok.all(), (int((~ok).sum()), np.degrees(a[~ok][:5]), r["t"][~ok][:5], chord[~ok][:5])
    a = np.radians(rng.uniform(0.01, 0.25, N))
    Dr = np.stack([np.cos(a) * np.cos(phi), -np.sin(a), np.cos(a) * np.sin(phi)], 1).astype(np.float32)
    r = q(X, Dr)
    ok = r["hit"] & (r["vol_after"] != vo) & (r["t"] < 2 * 0.25 * np.sin(a) + 1e-3)
    assert ok.all(), (int((~ok).sum()), np.degrees(a[~ok][:5]), r["t"][~ok][:5])


def _reentry(nav):
    q = _q(nav)
    o = np.array([0.1, 0.0, -0.8], np.float32)
    up = np.array([0.0, 0.0, 1.0], np.float32)
    v_world = int(locate(nav, o)[0])
    r1 = q(o, up, I32(v_world), I32(-1))
    x = o + (np.float32(r1["t"]) + np.float32(3e-7)) * up
    d = np.array([0.5, 0.0, -1.0], np.float32) / np.float32(np.sqrt(1.25))
    return q, x, d, I32(r1["vol_after"]), I32(r1["item"]), v_world, up


def test_zero_length_events_carry_their_face_normal(toys):
    """a zero-length exit the on-face rule decides is at the face it sits on: the normal is that face's, not -d"""
    nav = toys("nest")[1]
    q, x, d, v1, l1, v_world, _ = _reentry(nav)
    r2 = q(x, d, v1, l1)
    assert r2["hit"] and float(r2["t"]) < 1e-5 and int(r2["vol_after"]) == v_world, r2
    assert abs(float(r2["normal"][2])) > 0.999 and float(np.dot(r2["normal"], np.asarray(d))) < 0, r2["normal"]


def test_a_zero_length_event_carries_its_live_t(toys):
    """a zero-length event on a known face is the implicit function of that face too: d t / d o = -n_z / (n . d)"""
    nav = toys("nest")[1]
    q, x, d, v1, l1, _, up = _reentry(nav)
    ty, d1, _ = one(per_ray(t_jets, nav, x[None], d[None], [v1], [l1], np.r_[up, 0, 0, 0][None], shared=(1e3,)))
    t, dt = ty[0], d1[0]
    n = np.asarray(q(x, d, v1, l1)["normal"])
    assert float(t) < 1e-5
    want = -n[2] / float(np.dot(n, np.asarray(d)))
    assert abs(float(dt) - want) < 1e-4 * abs(want), (float(dt), want)


def test_a_pass_pair_leaves_the_visible_crossing_its_true_class_before():
    """glass -> body is a pass pair: the query steps over it, and the visible crossing body -> inner reports BODY as
    the class and volume before (the step's surface is the border's (body, inner), not a plain (glass, inner))"""
    sc = make_scene({"world": defn("world", solids.g4_box(4.0, 4.0, 4.0), 1, [("glass", np.eye(3), (0.0, 0.0, 0.0))]),
                     "glass": defn("glass", solids.g4_orb(0.3), 2, [("body", np.eye(3), (0.0, 0.0, 0.0))]),
                     "body": defn("body", solids.g4_orb(0.25), 3, [("inner", np.eye(3), (0.0, 0.0, 0.0))]),
                     "inner": defn("inner", solids.g4_orb(0.2), 4)})  # fmt: skip
    nav = nav_of(sc)
    V = ~np.eye(5, dtype=bool)
    V[2, 3] = V[3, 2] = False  # a pass pair: no vertex
    vg, vb, vi = (nav.defs.index(k) for k in ("glass", "body", "inner"))
    rng = np.random.default_rng(3)
    N = 200
    u = rng.normal(size=(N, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X = (u * rng.uniform(0.26, 0.295, N)[:, None]).astype(np.float32)
    Dr = -u + 0.05 * rng.normal(size=(N, 3))
    Dr = (Dr / np.linalg.norm(Dr, axis=1, keepdims=True)).astype(np.float32)
    r = query(nav, X, Dr, vg, -1, vertex_tab=V)
    ok = (r["hit"] & (r["cls_before"] == 3) & (r["vol_before"] == vb) & (r["cls_after"] == 4) & (r["vol_after"] == vi)
          & (r["n_pass"] >= 1))  # fmt: skip
    assert ok.all(), (int((~ok).sum()), r["cls_before"][~ok][:5], r["vol_before"][~ok][:5])
    np.testing.assert_allclose(np.linalg.norm(X + r["t"][:, None] * Dr, axis=1), 0.2, atol=2e-5)


def test_a_crossing_into_another_volume_of_the_same_class_is_a_vertex_where_the_diagonal_has_a_surface():
    """two touching boxes of one class (A | B: x in [-0.5, 0] and [0, 0.5]): the query steps over A -> B as one class,
    and stops there when that class's own pair has a surface (the vertex table's diagonal): the class on
    both sides, B the volume after. Leaving B into the world is a crossing either way."""
    sc = make_scene({"world": defn("world", solids.g4_box(4.0, 4.0, 4.0), 1,
                                   [("box", np.eye(3), (-0.25, 0.0, 0.0)), ("box", np.eye(3), (0.25, 0.0, 0.0))]),
                     "box": defn("box", solids.g4_box(0.5, 0.5, 0.5), 2)})  # fmt: skip
    nav = nav_of(sc)
    va, vb, vw = (int(v) for v in locate(nav, [[-0.25, 0.1, 0.0], [0.25, 0.1, 0.0], [1.5, 0.0, 0.0]]))
    assert va != vb and int(nav.host["cls"][va]) == int(nav.host["cls"][vb]) == 2
    X, D = [[-0.25, 0.1, 0.05]], [[1.0, 0.0, 0.0]]
    over = one(query(nav, X, D, va))
    assert over["hit"] and int(over["vol_after"]) == vw and int(over["n_pass"]) == 1
    assert abs(float(over["t"]) - 0.75) < 1e-5
    V = ~np.eye(3, dtype=bool)
    V[2, 2] = True
    at = one(query(nav, X, D, va, vertex_tab=V))
    assert at["hit"] and int(at["vol_before"]) == va and int(at["vol_after"]) == vb and int(at["n_pass"]) == 0
    assert int(at["cls_before"]) == int(at["cls_after"]) == 2 and abs(float(at["t"]) - 0.25) < 1e-5
    assert abs(float(at["normal"][0]) + 1.0) < 1e-6
    V[2, 2], V[1, 1] = False, True  # another class's diagonal: A -> B is still stepped over
    assert int(one(query(nav, X, D, va, vertex_tab=V))["vol_after"]) == vw


def test_a_mesh_takes_part_in_ownership_like_any_solid():
    """A tessellated solid overlapping a volume: the first placed owns the region (nothing is proved inside a mesh).
    The tetrahedron placed first owns its overlap with a box, whose candidates then hold the mesh's chunks; the box
    placed first owns it instead and never sees the mesh. A photon leaving that box at a point inside the
    tetrahedron is in the world there (a mesh holds no point), and meets the mesh at the next facet, from behind."""
    for box_first in (False, True):
        sc = scene_mesh()
        pls = sc.definitions["world"].placements
        pls[1].t = np.array([0.25, 0.1, 0.0])  # the box onto the tetrahedron
        if box_first:
            pls.reverse()
        sc.definitions["box"].member.solid = solids.g4_box(0.1, 0.1, 0.1)
        nav = nav_of(sc)
        box = nav.defs.index("box")
        assert any(k >= nav.CH0 for k in nav.kids[nav.defs.index("world")])  # its mother's items, always
        assert any(k >= nav.CH0 for k in nav.kids[box]) == (not box_first)
        assert [(a, b) for a, b, _ in nav.owned] == ([(nav.n + 1, box)] if box_first else [(box, nav.n + 1)])
    # leaving the box at x = 0.3 inside the tetrahedron: into the world, then the mesh at its facet ahead
    o, d = jnp.asarray([0.27, 0.12, 0.0], jnp.float32), jnp.asarray([1.0, 0.0, 0.0], jnp.float32)
    r = jax.jit(lambda: nav.query(o, d, box, t_max=10.0))()
    assert bool(r["hit"]) and int(r["vol_after"]) == nav.defs.index("world") and abs(float(r["t"]) - 0.03) < 1e-6, r
    r2 = jax.jit(lambda: nav.query(o + r["t"] * d, d, r["vol_after"], last=r["srf"], t_max=10.0))()
    assert bool(r2["hit"]) and int(r2["vol_after"]) == nav.n + 1 and int(r2["cls_after"]) == 6, r2


def _box_tris(lo, hi):
    """The 12 outward triangles of the box [lo, hi]."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[(hi if (k >> a) & 1 else lo)[a] for a in range(3)] for k in range(8)])
    quads = ((0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5))
    T = np.array([c[[q[0], q[i], q[i + 1]]] for q in quads for i in (1, 2)])
    out = np.einsum("ij,ij->i", np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), T.mean(1) - c.mean(0)) > 0
    return np.where(out[:, None, None], T, T[:, ::-1])


def test_a_mesh_of_interpenetrating_bodies_is_met_at_its_first_facet():
    """One mesh of two closed bodies, a small box poking through a large box's face (a CAD export's intersecting
    parts). A box placed first owns its overlap with the mesh; leaving it at a point inside the small body only, the
    photon is in the world (a mesh holds no point) and meets the mesh at the first facet ahead -- the large body's front
    facet, inside the small body -- as at any facet."""
    big, small = _box_tris((0.0, -0.2, -0.2), (0.4, 0.2, 0.2)), _box_tris((-0.1, -0.05, -0.05), (0.05, 0.05, 0.05))
    tris = np.concatenate([big, small])
    sc = make_scene({"world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("box", np.eye(3), (-0.08, 0.0, 0.0)),
                                                                           ("cad", np.eye(3), (0.0, 0.0, 0.0))]),
                     "cad": mesh_def("cad", tris, 6),
                     "box": defn("box", solids.g4_box(0.06, 0.06, 0.06), 2)})  # fmt: skip
    nav = nav_of(sc)
    box, world = nav.defs.index("box"), nav.defs.index("world")
    o, d = jnp.asarray([-0.07, 0.01, 0.005], jnp.float32), jnp.asarray([1.0, 0.0, 0.0], jnp.float32)
    r = jax.jit(lambda: nav.query(o, d, box, t_max=10.0))()
    assert bool(r["hit"]) and int(r["vol_after"]) == world and abs(float(r["t"]) - 0.02) < 1e-6, r
    r2 = jax.jit(lambda: nav.query(o + r["t"] * d, d, world, last=r["srf"], t_max=10.0))()
    assert bool(r2["hit"]) and int(r2["vol_after"]) == nav.n + 1 and abs(float(r2["t"]) - 0.05) < 1e-6, r2
    assert int(r2["cls_before"]) == 1 and int(r2["cls_after"]) == 6, r2
    # leaving it heading -x, outside both bodies: into the world
    r = jax.jit(lambda: nav.query(o, -d, box, t_max=10.0))()
    assert bool(r["hit"]) and int(r["vol_after"]) == world and int(r["cls_after"]) == 1, r


def test_each_mesh_has_its_own_mother_by_navigator_id():
    """Two meshes in two mothers (a tetrahedron in the world, a box mesh inside volume B): host["parent"] is indexed by
    navigator id (the outside at n, mesh k at n + 1 + k), so each mesh's mother is its own; leaving a small box that
    owns part of the second mesh, the photon is in B, and meets that mesh at its facet ahead (an item of B)."""
    cad = _box_tris((0.0, -0.2, -0.2), (0.3, 0.2, 0.2))
    sc = make_scene({"world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("tet", np.eye(3), (-0.6, 0.1, 0.0)),
                                                                           ("B", np.eye(3), (0.5, 0.0, 0.0))]),
                     "tet": mesh_def("tet", tet_tris(), 6),
                     "B": defn("B", solids.g4_box(0.8, 0.8, 0.8), 3, [("s", np.eye(3), (-0.03, 0.0, 0.0)),
                                                                     ("cad", np.eye(3), (0.0, 0.0, 0.0))]),
                     "cad": mesh_def("cad", cad, 7),
                     "s": defn("s", solids.g4_box(0.1, 0.1, 0.1), 2)})  # fmt: skip
    nav = nav_of(sc)
    n, par = nav.n, np.asarray(nav.host["parent"])
    w, b, s_ = (nav.defs.index(k) for k in ("world", "B", "s"))
    assert nav.n_mesh == 2 and len(par) == n + 1 + nav.n_mesh and par[n] == n
    assert par[n + 1] == w and par[n + 2] == b, par[n:]
    o, d = jnp.asarray([0.45, 0.01, 0.005], jnp.float32), jnp.asarray([1.0, 0.0, 0.0], jnp.float32)
    r = jax.jit(lambda: nav.query(o, d, s_, t_max=10.0))()
    assert bool(r["hit"]) and int(r["vol_after"]) == b and abs(float(r["t"]) - 0.07) < 1e-6, r
    r2 = jax.jit(lambda: nav.query(o + r["t"] * d, d, b, last=r["srf"], t_max=10.0))()
    assert bool(r2["hit"]) and int(r2["vol_after"]) == n + 2 and abs(float(r2["t"]) - 0.28) < 1e-5, r2
    assert int(r2["cls_after"]) == 7, r2


def test_a_query_follows_every_invisible_crossing():
    """40 water-in-water envelopes along a ray, then an absorber: one query crosses all 80 faces without a vertex and
    meets the absorber at the oracle's distance (no round limit, no continue event)"""
    pls = [("env", np.eye(3), (-0.8 + 0.02 * k, 0.0, 0.0)) for k in range(40)] + [("abs", np.eye(3), (0.4, 0.0, 0.0))]
    sc = make_scene({"world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, pls),
                     "env": defn("env", solids.g4_box(0.019, 0.1, 0.1), 1),
                     "abs": defn("abs", solids.g4_box(0.1, 0.1, 0.1), 2)})  # fmt: skip
    nav = nav_of(sc)
    o, d = np.array([-0.95, 0.001, 0.002]), np.array([1.0, 0.0, 0.0])
    r = one(query(nav, o.astype(np.float32), d, int(locate(nav, o)[0]), -1, t_max=10.0))
    assert r["hit"] and not r["cont"] and int(r["cls_after"]) == 2 and int(r["n_pass"]) == 80, r
    ro = oracle.Oracle(sc).query(o, d, T_ORC, 10.0, push=0.0)
    assert abs(float(r["t"]) - ro["t"]) < 1e-5, (r["t"], ro)


def test_no_zero_length_entry_from_a_face_heading_out():
    """a float32 start 0-3 ulps inside a daughter's face, leaving it at 0.25-10 deg: no zero-length entry (with or
    without the daughter as last); heading in, the entry is at t ~ 0"""
    sc = make_scene({"world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("d", np.eye(3), (0.5, 0.3, 0.1))]),
                     "d": defn("d", solids.g4_box(0.2, 0.2, 0.2), 2)})  # fmt: skip
    nav = nav_of(sc)
    dv = nav.defs.index("d")
    rng = np.random.default_rng(0)
    N = 400
    y, z = 0.3 + rng.uniform(-0.09, 0.09, N), 0.1 + rng.uniform(-0.09, 0.09, N)
    x = np.float32(0.6) - rng.integers(0, 4, N).astype(np.float32) * np.float32(6e-8)
    ang, phi = np.radians(rng.uniform(0.25, 10.0, N)), rng.uniform(0, 2 * np.pi, N)
    O_ = np.stack([x, y, z], 1).astype(np.float32)
    Dr = np.stack([np.sin(ang), np.cos(ang) * np.cos(phi), np.cos(ang) * np.sin(phi)], 1).astype(np.float32)
    for last in (-1, dv):
        r = query(nav, O_, Dr, 0, last)
        out = r["hit"] & (r["t"] < 1e-5)
        assert out.sum() == 0, (last, int(out.sum()))
    Dr[:, 0] *= -1  # heading in: the zero-length entry stays
    r = query(nav, O_, Dr, 0, -1)
    ins = r["hit"] & (r["t"] < 1e-5) & (r["vol_after"] == dv)
    assert ins.mean() == 1.0, ins.mean()


def _mesh_scene(tris, half=6.0):
    return make_scene({"world": defn("world", solids.g4_box(half, half, half), 1, [("f", np.eye(3), (0.0, 0.0, 0.0))]),
                       "f": mesh_def("f", tris, 6)})  # fmt: skip


def test_a_mesh_face_within_the_band_ahead_is_met():
    """a ray on triangle A, 22 um from its shared edge with B, heading into B 1.7 um ahead (within the band at 1.9 m):
    B is met (the mesh scan once started past the band and missed it)"""
    A = np.array([[0.5926313996315002, 1.4234877824783325, 1.9261622428894043],
                  [0.5951090455055237, 1.4237315654754639, 1.927188515663147],
                  [0.5951090455055237, 1.4237315654754639, 1.4344185590744019]])  # fmt: skip
    Bt = np.array([[0.6334213614463806, 1.4237315654754639, 1.4469280242919922],
                   [0.6331993937492371, 1.4237315654754639, 1.9147378206253052],
                   [0.5824094414710999, 1.4237315654754639, 1.8936717510223389]])  # fmt: skip
    nav = nav_of(_mesh_scene(np.stack([A, Bt])))
    V0 = np.asarray(nav.dev()["V0"], np.float64)
    ka = int(np.argmin(np.abs(V0 - A[0]).sum(1)))
    kb = int(np.argmin(np.abs(V0 - Bt[0]).sum(1)))
    o = np.asarray([0.5950872302055359, 1.4237298965454102, 1.8702424764633179], np.float32)
    d = np.asarray([-0.053876541554927826, 0.9535114169120789, -0.2965017557144165], np.float32)
    srf = nav.srf_pack(I32(nav.CH0), I32(ka), I32(-1), jnp.bool_(True), jnp.bool_(True))
    q = one(query(nav, o, d, 0, srf, t_max=10.0))
    assert bool(q["hit"]) and int(q["leaf"]) == kb and float(q["t"]) < 1e-5, (float(q["t"]), int(q["leaf"]), ka, kb)


def cube_tris(c, h):
    """a closed cube (centre c, half-size h), wound outward (Geant4's tessellated convention)"""
    P = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h + np.asarray(c, float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    T = []
    for a, b, c_, d in quads:
        T += [P[[a, b, c_]], P[[a, c_, d]]]
    T = np.asarray(T)
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    out = np.einsum("ij,ij->i", N, T.mean(1) - np.asarray(c, float)) > 0
    T[~out] = T[~out][:, ::-1]
    return T


def test_a_mesh_facet_is_met_only_by_a_ray_entering_it():
    """the facet rule: from outside a facet is met only by a ray entering it; a start behind its plane within the
    tolerance is on it (a zero-length hit); a start inside the closed mesh meets a facet from behind first: an entry
    into the opaque part alike (a mesh holds no point)"""
    nav = nav_of(_mesh_scene(cube_tris((1.9, 0.0, 0.0), 0.05), 8.0))
    band = 8 * 2.0**-23 * 1.95

    def q(o_, d_):
        return one(query(nav, o_, d_, 0, t_max=10.0))

    d = np.array([-0.6, 0.3, 0.2])
    d /= np.linalg.norm(d)
    r = q(jnp.asarray([1.95 - 0.1 * band, 0.01, -0.01], F32), jnp.asarray(d, F32))
    assert bool(r["hit"]) and int(r["item"]) >= nav.CH0 and float(r["t"]) <= 1e-7, (float(r["t"]), int(r["item"]))
    assert int(r["cls_after"]) == 6 and int(r["cls_before"]) == 1
    r = q(jnp.asarray([1.95 + 0.5 * band, 0.01, -0.01], F32), jnp.asarray(-d, F32))  # heading away: the wall next
    assert int(r["item"]) < nav.CH0 and float(r["t"]) > 1.0, (float(r["t"]), int(r["item"]))
    r = q(jnp.asarray([1.9, 0.0, 0.0], F32), jnp.asarray(d, F32))  # inside the cube: the back facet
    assert bool(r["hit"]) and int(r["cls_before"]) == 1 and int(r["cls_after"]) == 6
    assert abs(float(r["t"]) - 0.05 / 0.6 * np.linalg.norm([0.6, 0.3, 0.2])) < 1e-5


LONG_TRIANGLES = [
    ([-0.9224997758865356, -0.9815972447395325, 0.6962684988975525],
     [0.4934617578983307, 0.6272781491279602, 0.6025094389915466],
     [0.7844154834747314, 5.107905849399685e-07, -0.6202357411384583], True,
     [[-0.9212701320648193, -2.0992486476898193, 0.6978231072425842],
      [-0.9244762063026428, -2.0992486476898193, 0.6937683820724487],
      [-0.9244776964187622, 1.6787055730819702, 0.693769633769989]],
     [[-0.9212701320648193, -2.0992486476898193, 0.6978231072425842],
      [-0.9241046905517578, 1.6789369583129883, 0.6941911578178406],
      [-0.9237983226776123, 1.6791892051696777, 0.6945481300354004]], 0.0008876199019245243),
    ([1.2922121286392212, -1.3369330167770386, 1.8782020807266235],
     [-0.37385106086730957, 0.9108082056045532, 0.17511096596717834],
     [0.25882405042648315, -6.844286279399333e-17, 0.9659245014190674], False,
     [[1.2908289432525635, 1.3673194646835327, 1.8785731792449951],
      [1.2924301624298096, 1.3673194646835327, 1.8781441450119019],
      [1.2924301624298096, -1.7664053440093994, 1.8781441450119019]],
     [[1.2924301624298096, -1.7664053440093994, 1.8781441450119019],
      [1.2908289432525635, -1.7664053440093994, 1.8785731792449951],
      [1.2908289432525635, 1.3673194646835327, 1.8785731792449951]], 5.770443651204539e-06),
    ([-0.6171281933784485, 1.416987419128418, 0.5977534055709839],
     [-0.643323540687561, 0.448253333568573, -0.6206479072570801],
     [0.1230025365948677, 0.9469363689422607, 0.2969543933868408], False,
     [[-0.9126203060150146, 1.4168975353240967, 0.7204380035400391],
      [-0.03341631591320038, 1.4171652793884277, 0.3554057478904724],
      [-0.0328582227230072, 1.4168975353240967, 0.35602858662605286]],
     [[-0.9126203060150146, 1.4168975353240967, 0.7204380035400391],
      [-0.9126661419868469, 1.4171652793884277, 0.7196029424667358],
      [-0.03341631591320038, 1.4171652793884277, 0.3554057478904724]], 1.7027113942294043e-06)]  # fmt: skip


@pytest.mark.parametrize("case", range(len(LONG_TRIANGLES)))
def test_a_long_mesh_triangle_is_met_at_its_plane(case):
    """rays on triangle A heading into B, a 1-4 m sliver 0.5-2 mm wide, at oblique incidence: t comes from B's plane
    (the sheared test's own t interpolated the vertices' depths metres away and skipped B), where float64 meets it"""
    o, d, sn, o_flag, A, Bt, t64 = LONG_TRIANGLES[case]
    A, Bt = np.asarray(A), np.asarray(Bt)
    nav = nav_of(_mesh_scene(np.stack([A, Bt]), 8.0))
    V0 = np.asarray(nav.dev()["V0"], np.float64)
    V1 = np.asarray(nav.dev()["V1"], np.float64)
    ka = int(np.argmin(np.abs(V0 - A[0]).sum(1) + np.abs(V1 - A[1]).sum(1)))
    kb = int(np.argmin(np.abs(V0 - Bt[0]).sum(1) + np.abs(V1 - Bt[1]).sum(1)))
    srf = nav.srf_pack(I32(nav.CH0), I32(ka), I32(-1), jnp.bool_(o_flag), jnp.bool_(True))
    q = one(query(nav, o, d, 0, srf, sn=sn, t_max=10.0))
    nb = np.cross(Bt[1] - Bt[0], Bt[2] - Bt[0])
    short = 3 * 2.0**-23 * 2.0 / abs(float(np.dot(nb / np.linalg.norm(nb), d)))
    assert bool(q["hit"]) and int(q["leaf"]) == kb and t64 - short - 1e-7 < float(q["t"]) < t64 + 1e-6, (
        float(q["t"]), t64, int(q["leaf"]), ka, kb)  # fmt: skip


def test_a_grazing_entry_at_a_rim_is_in():
    """a sphere cap (a sphere minus a box) far from the origin: a photon on the sphere 0.1-0.9 bands below the cut
    plane, grazing up into the cap (0.05-0.3 deg), enters it at zero length and is then in it: its next query meets
    the far side, not a zero-length exit (the stuck rule's rim case)"""
    c = np.array([1.2, -1.1, -0.9])
    cap = g4("sub", solids.g4_orb(0.051), solids.g4_box(0.126, 0.126, 0.069194))  # the cut plane at z = 34.597 mm
    sc = make_scene({"world": defn("world", solids.g4_box(6.0, 6.0, 6.0), 1, [("cap", np.eye(3), c)]),
                     "cap": defn("cap", cap, 2)})  # fmt: skip
    nav = nav_of(sc)
    cv = nav.defs.index("cap")
    rng = np.random.default_rng(3)
    N = 2000
    band = 8.0 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(c)) + 0.05)
    R, zc = 0.051, 0.034597
    z = zc - rng.uniform(0.1, 0.9, N) * band
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    rho = np.sqrt(R * R - z * z)
    O_ = (c + np.stack([rho * np.cos(phi), rho * np.sin(phi), z], 1)).astype(np.float32)  # on the sphere
    a = np.radians(rng.uniform(0.05, 0.3, N))
    Dr = np.stack([-np.cos(phi) * np.cos(a), -np.sin(phi) * np.cos(a), np.sin(a)], 1).astype(np.float32)
    r1 = query(nav, O_, Dr, 0, -1, sn=np.zeros(3))
    assert (r1["hit"] & (r1["vol_after"] == cv)).mean() == 1.0
    X = (O_ + r1["t"][:, None] * Dr).astype(np.float32)
    r2 = query(nav, X, Dr, cv, r1["srf"], sn=r1["normal"])
    back = r2["hit"] & (r2["t"] < 1e-3)  # the chord to the far side is ~ 7 cm
    assert back.sum() == 0, (int(back.sum()), r2["t"][back][:5], r2["vol_after"][back][:5])


def _two_crossings(nv, o, d):
    v = nv.locate(o)
    r = nv.query(o, d, v, t_max=1e3)
    r2 = nv.query(o + r["t"] * d, d, r["vol_after"], last=r["srf"], sn=r["normal"], t_max=1e3)
    return r["t"], r["t"] + r2["t"], r2["normal"], r2["vol_after"]


def test_grazing_thin_tube_far_from_the_origin_is_met_where_it_is():
    """a 75 um tube in a 0.3 mm plane box at 3.6 m (ICARUS-like wires), rays near tangency: the on-face rule measures
    the start's own distance from the wire, so the wire's far tangent plane is not mistaken for the box face the photon
    sits on (that gave a zero-length entry with a skewed normal); a lone tube: no zero-length crossing"""
    r, c, x0 = 75e-6, np.array([-3.5873, -0.236, 0.0]), -3.586
    b = np.linspace(0.0745e-3, 0.07499e-3, 10)
    exact = x0 - (c[0] + np.sqrt(r * r - b * b))
    O_ = np.c_[np.full(len(b), x0), c[1] + b, np.full(len(b), 0.2494)]
    d = np.tile([-1.0, 0.0, 0.0], (len(b), 1))
    sc = make_scene({"world": defn("world", solids.g4_box(16.0, 16.0, 16.0), 1, [("plane", np.eye(3), c)]),
                     "plane": defn("plane", solids.g4_box(0.3e-3, 1.0, 2.0), 2, [("wire", np.eye(3), (0, 0, 0))]),
                     "wire": defn("wire", solids.g4_tube(0.0, r, 1.0), 3)})  # fmt: skip
    nav = nav_of(sc)
    _, t2, n2, v2 = per_ray(_two_crossings, nav, O_, d)
    assert np.all(np.abs(t2 - exact) < 1e-6), t2 - exact
    assert np.all(n2[:, 1] > 0.99) and np.all(v2 == nav.names.index("plane_0/wire_0"))
    lone = make_scene({"world": defn("world", solids.g4_box(16.0, 16.0, 16.0), 1, [("wire", np.eye(3), c)]),
                       "wire": defn("wire", solids.g4_tube(0.0, r, 1.0), 2)})  # fmt: skip
    t1, *_ = per_ray(_two_crossings, nav_of(lone), O_, d)
    assert np.all(np.abs(t1 - exact) < 1e-6), t1 - exact


def test_a_start_inside_a_curved_solid_heading_along_it_exits_where_it_does():
    """a start inside a tube or a sphere (1 mm, at 3.6 m: the on-face band is ~14 um) at a depth h beyond the band,
    heading tangent to its own iso-surface: the exit is at sqrt(R^2 - (R - h)^2), not a zero-length exit at the start
    (the face's normal at the start is perpendicular to the ray there, whatever h); within the band it is on the face"""
    c, R = np.array([-3.5873, -0.236, 0.0]), 1e-3
    for sol in (solids.g4_tube(0.0, R, 1.0), solids.g4_orb(R)):
        nav = nav_of(make_scene({"world": defn("world", solids.g4_box(16.0, 16.0, 16.0), 1, [("s", np.eye(3), c)]),
                                 "s": defn("s", sol, 2)}))  # fmt: skip
        h = np.array([30e-6, 50e-6, 120e-6, 400e-6])
        O_ = np.c_[np.full(len(h), c[0]), c[1] + (R - h), np.full(len(h), c[2])]
        t1, *_ = per_ray(_two_crossings, nav, O_, np.tile([-1.0, 0.0, 0.0], (len(h), 1)))
        np.testing.assert_allclose(t1, np.sqrt(R * R - (R - h) ** 2), rtol=1e-3)
        t_on, *_ = per_ray(_two_crossings, nav, [[c[0], c[1] + R - 2e-6, c[2]]], [[-1.0, 0.0, 0.0]])
        assert t_on[0] == 0.0


def test_a_tree_deeper_than_any_fixed_walk_is_located_and_left_at_every_level():
    """The navigator's walks end by the tree's structure, not a fixed cap truncating them silently: 80 nested boxes
    (deeper than the 64 levels locate used to stop at) locate their innermost point."""
    n = 80
    defs = {f"b{k}": defn(f"b{k}", solids.g4_box(*(2.0 - k * 0.02,) * 3), k % 5 + 1,
                          [(f"b{k + 1}", np.eye(3), (0.0, 0.0, 0.0))] if k + 1 < n else [])
            for k in range(n)}  # fmt: skip
    defs["world"] = defn("world", solids.g4_box(3.0, 3.0, 3.0), 9, [("b0", np.eye(3), (0.0, 0.0, 0.0))])
    nav = nav_of(make_scene(defs))
    v = int(np.asarray(locate(nav, [[0.0, 0.0, 0.0]]))[0])
    assert v == nav.n - 1 and nav.names[v].count("/") >= n - 1  # the innermost box (preorder: the last volume)


def test_a_ray_into_a_curved_solid_from_its_face_is_inside_it():
    """A start on an orb's face (no surface identity: another solid's coincident face, or none) heading into it at a
    shallow angle is inside: its chord's far end, on the same face but crossed the other way, is a later crossing, not
    the face the photon sits on (the grazing rule: not a ray crossing its face the other way). At 20 m from the
    origin (a band of ~20 um: the grazing window, GRAZE K_ON bands, is 2 cm, longer than the chords)."""
    world = defn("world", solids.g4_box(60.0, 60.0, 60.0), 1, [("ball", np.eye(3), (20.0, 0.0, 0.0))])
    sc = make_scene({"world": world, "ball": defn("ball", solids.g4_orb(0.27), 2)})
    nav = nav_of(sc)
    ball = nav.defs.index("ball")
    for deg in (0.3, 1.0, 3.0):
        a = np.radians(deg)
        for k in range(6):  # around the orb: the float32 start falls inside, on or outside the face
            phi = 0.3 * k
            nrm, tan = np.array([np.cos(phi), np.sin(phi), 0.0]), np.array([-np.sin(phi), np.cos(phi), 0.0])
            o = (np.array([20.0, 0.0, 0.0]) + 0.27 * nrm).astype(np.float32)
            d = (np.cos(a) * tan - np.sin(a) * nrm).astype(np.float32)  # into the ball at deg from its tangent plane
            r = one(query(nav, o, d, ball, t_max=10.0))
            assert bool(r["hit"]) and not bool(r["zero"]) and int(r["vol_after"]) == nav.defs.index("world"), (deg, k)
            np.testing.assert_allclose(float(r["t"]), 2 * 0.27 * np.sin(a), atol=2e-4, err_msg=str((deg, k)))


def _crossing(nv, o, d, v):
    r = nv.query(o, d, v, last=-1, t_max=1e3)
    return r["t"], r["vol_after"], r["srf"], r["normal"]


def _crossing_from(nv, o, d, v, last, sn):
    r = nv.query(o, d, v, last=last, sn=sn, t_max=1e3)
    return r["t"], r["vol_after"]


def test_a_slab_of_1p7_contacts_at_4m_is_crossed_face_by_face():
    """SBND's CPA foil split by its recipe (lumen-tpc): a 25.4 um TPB slab on a 25.4 um vm2000 slab in a 50.823 um
    LAr box (the film's old 23 nm slot left as LAr), at a 4 m coordinate scale, where the contact tolerance (K_ON
    bands) is 15 um: the slab is 1.7 contacts thick. Entering from the LAr, every incidence meets the TPB face and then
    the vm2000 face 25.4 um / cos further; a photon re-emitted at the TPB face (no surface remembered: a bulk event)
    reaches the vm2000 face going in and the LAr at once going out; one reflected at the vm2000 face reaches the LAr
    25.4 um / cos back. Each distance within one band (3.8 um at 4 m) of the exact one."""
    half, c, f0 = 25.4e-6, np.array([1.0005, 1.5525, 3.9]), -1.15e-8  # the foil's centre in its box (SBND's GDML)
    sc = make_scene({"world": defn("world", solids.g4_box(16.0, 16.0, 16.0), 1, [("cpa", np.eye(3), c)]),
                     "cpa": defn("cpa", solids.g4_box(50.823e-6, 0.932, 1.172), 1,
                                 [("tpb", np.eye(3), (f0 + half / 2, 0, 0)),
                                  ("foil", np.eye(3), (f0 - half / 2, 0, 0))]),
                     "tpb": defn("tpb", solids.g4_box(half, 0.932, 1.172), 2),
                     "foil": defn("foil", solids.g4_box(half, 0.932, 1.172), 3)})  # fmt: skip
    nav = nav_of(sc)
    tpb, foil = nav.names.index("cpa_0/tpb_0"), nav.names.index("cpa_0/foil_1")
    lar = {nav.names.index(""), nav.names.index("cpa_0")}  # the world (unnamed) and the CPA box: LAr
    front = c[0] + f0 + half  # the TPB's face to the LAr
    tol = navigation.band(4.0)
    cos = np.cos(np.radians([0.0, 20.0, 45.0, 60.0, 75.0]))
    sin = np.sqrt(1 - cos * cos)
    d_in = np.c_[-cos, sin * 0.6, sin * 0.8]  # into the slab from its front
    O_ = np.array([front, c[1], c[2]]) - 0.05 * d_in  # each meets the TPB's face at (front, c_y, c_z), 5 cm on
    t1, t2, _, v2 = per_ray(_two_crossings, nav, O_, d_in)
    assert np.all(np.abs(t2 - t1 - half / cos) < tol), (t2 - t1, half / cos)
    assert np.all(v2 == foil)
    face = (O_ + t1[:, None] * d_in).astype(np.float32)  # the entry points, on the TPB's face
    assert np.all(np.abs(face[:, 0] - front) < tol)
    # re-emitted there (a bulk event remembers no surface): into the slab, and out
    for d, want_v, want_t in ((d_in, foil, half / cos), (d_in * [-1, 1, 1], None, 0.0 * cos)):
        t, v, _, _ = per_ray(_crossing, nav, face, d.astype(np.float32), np.full(len(cos), tpb, np.int32))
        assert np.all(np.abs(t - want_t) < tol), (t, want_t)
        assert np.all(v == want_v) if want_v is not None else all(int(x) in lar for x in v), v
    # reflected at the vm2000 face (that surface remembered), back out through the TPB
    back = (face + (half / cos)[:, None] * d_in).astype(np.float32)
    _, _, srf, nrm = per_ray(_crossing, nav, face, d_in.astype(np.float32), np.full(len(cos), tpb, np.int32))
    d_out = (d_in * [-1, 1, 1]).astype(np.float32)
    t, v = per_ray(_crossing_from, nav, back, d_out, np.full(len(cos), tpb, np.int32), srf, nrm)
    assert np.all(np.abs(t - half / cos) < tol), (t, half / cos)
    assert all(int(x) in lar for x in v), v
