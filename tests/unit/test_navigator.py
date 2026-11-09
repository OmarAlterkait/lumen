"""The navigator (volume tracking) against the float64 oracle on small scenes: point location, the first visible change
from random points, walks continued from each hit by surface identity (no push), the live t's derivatives against
float64 finite differences, frame groups (rotated wires), split unions, meshes and the zero-length rules. Ported from
the baseline's tests/test_navigator.py (the overlap check's tests are in test_overlap_check.py)."""

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
    walk_check,
)

from lumen.build import lowering, solids
from lumen.geometry import _legacy_tracked as tracked
from lumen.geometry import navigate
from lumen.validate import oracle


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
    assert len(sp.definitions["frame"].members) == 0 and len(sp.definitions["frame"].placements) == 20
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
    for i in range(len(O_)):
        r0 = orc.query(O_[i], Dr[i], T_ORC, 10.0, push=0.0)
        if not r0["hit"] or abs(r0["t"] - t[i]) > 1e-5:
            continue
        dp, dm = Dr[i] + h * tg[i, 3:], Dr[i] - h * tg[i, 3:]
        # the oracle normalises d: differentiate t * |d| (the navigator's d is used as given)
        rp = orc.query(O_[i] + h * tg[i, :3], dp, T_ORC, 10.0, push=0.0)
        rm = orc.query(O_[i] - h * tg[i, :3], dm, T_ORC, 10.0, push=0.0)
        if not (rp["hit"] and rm["hit"]) or rp["after"] != r0["after"] or rm["after"] != r0["after"]:
            continue
        fd = (rp["t"] / np.linalg.norm(dp) - rm["t"] / np.linalg.norm(dm)) / (2 * h)
        errs.append(abs(fd - dt[i]) / (1 + abs(fd)))
    errs = np.asarray(errs)
    assert len(errs) > 150 and np.median(errs) < 1e-4 and np.mean(errs > 1e-2) < 0.01, (len(errs), np.median(errs))


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
    for i in range(len(O_)):
        r0 = orc.query(O_[i], Dr[i], T_ORC, 10.0, push=0.0)
        if not r0["hit"] or abs(r0["t"] - ty[i, 0]) > 1e-5:
            continue
        vals = []
        for s in (-1, 0, 1):
            oo, dd = O_[i] + s * h * tg[i, :3], Dr[i] + s * h * tg[i, 3:]
            r = orc.query(oo, dd, T_ORC, 10.0, push=0.0)
            if not r["hit"] or r["after"] != r0["after"]:
                break
            t = r["t"] / np.linalg.norm(dd)  # the oracle normalises d; the navigator uses it as given
            vals.append(np.concatenate([[t], oo + t * dd]))
        if len(vals) < 3:
            continue
        fd2 = (vals[2] - 2 * vals[1] + vals[0]) / h**2
        errs.append(np.max(np.abs(fd2 - d2[i]) / (1 + np.abs(fd2))))
    errs = np.asarray(errs)
    assert len(errs) > 150 and np.median(errs) < 1e-5 and np.max(errs) < 1e-2, (len(errs), np.median(errs))


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
    nav = nav.with_classes(list(range(nav.n)), [], nav.n)
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


def crossing_and_surface(nav, o, d, v):
    """next_crossing's Crossing, and surface_at on it"""
    cx = navigate.next_crossing(nav, o, d, v)
    return cx, navigate.surface_at(nav, cx, o, d)


def test_next_crossing_is_the_query(toys):
    """the Crossing record carries the query's values; surface_at on it gives the query's t"""
    nav = toys("nest")[1]
    o, d = np.array([[0.1, 0.0, -0.8]]), np.array([[0.0, 0.0, 1.0]])
    v = locate(nav, o)
    cx, (t, n) = one(per_ray(crossing_and_surface, nav, o, d, v))
    r = one(query(nav, o, d, v))
    assert float(cx.t) == float(r["t"]) and int(cx.vol_after) == int(r["vol_after"]) and int(cx.srf) == int(r["srf"])
    assert float(t) == float(r["t"]) and np.array_equal(np.asarray(n), np.asarray(r["normal"]))


def test_multi_member_fails_closed():
    from lumen.build import scene

    defs = {"world": scene.Definition("world", [scene.Member(solids.g4_orb(0.5), scene.Label(1), 0),
                                                scene.Member(solids.g4_orb(0.1), scene.Label(2), 1)], [])}  # fmt: skip
    with pytest.raises(NotImplementedError, match="members"):
        nav_of(make_scene(defs))


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
    P = np.zeros((5, 5), bool)
    P[2, 3] = P[3, 2] = True
    vg, vb, vi = (nav.defs.index(k) for k in ("glass", "body", "inner"))
    rng = np.random.default_rng(3)
    N = 200
    u = rng.normal(size=(N, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X = (u * rng.uniform(0.26, 0.295, N)[:, None]).astype(np.float32)
    Dr = -u + 0.05 * rng.normal(size=(N, 3))
    Dr = (Dr / np.linalg.norm(Dr, axis=1, keepdims=True)).astype(np.float32)
    r = query(nav, X, Dr, vg, -1, pass_tab=P)
    ok = (r["hit"] & (r["cls_before"] == 3) & (r["vol_before"] == vb) & (r["cls_after"] == 4) & (r["vol_after"] == vi)
          & (r["n_pass"] >= 1))  # fmt: skip
    assert ok.all(), (int((~ok).sum()), r["cls_before"][~ok][:5], r["vol_before"][~ok][:5])
    np.testing.assert_allclose(np.linalg.norm(X + r["t"][:, None] * Dr, axis=1), 0.2, atol=2e-5)


def test_mesh_chunks_are_items_of_their_mother_only():
    """Geant4 does not see a mesh from inside a volume it passes through: its chunks are daughters of its mother only"""
    sc = scene_mesh()
    sc.definitions["world"].placements[1].t = np.array([0.25, 0.1, 0.0])  # the box onto the tetrahedron
    nav = nav_of(sc)
    box = nav.defs.index("box")
    assert not any(k >= nav.CH0 for k in nav.kids[box])
    assert any(k >= nav.CH0 for k in nav.kids[nav.defs.index("world")])  # its mother's list


def test_rounds_run_out_on_a_row_of_same_class_envelopes():
    """40 water-in-water envelopes along a ray, then an absorber: each query stops after the rounds with a straight
    continue event; following them reaches the absorber at the oracle's distance, in the same class all along"""
    pls = [("env", np.eye(3), (-0.8 + 0.02 * k, 0.0, 0.0)) for k in range(40)] + [("abs", np.eye(3), (0.4, 0.0, 0.0))]
    sc = make_scene({"world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, pls),
                     "env": defn("env", solids.g4_box(0.019, 0.1, 0.1), 1),
                     "abs": defn("abs", solids.g4_box(0.1, 0.1, 0.1), 2)})  # fmt: skip
    nav = nav_of(sc)
    o, d = np.array([-0.95, 0.001, 0.002]), np.array([1.0, 0.0, 0.0])
    vol, last, x, n_cont = int(locate(nav, o)[0]), -1, o.astype(np.float32), 0
    for _ in range(12):
        r = one(query(nav, x, d, vol, last, t_max=10.0))
        assert r["hit"]
        x = (x + np.float32(r["t"]) * d.astype(np.float32)).astype(np.float32)
        vol, last = int(r["vol_after"]), int(r["item"])
        if not r["cont"]:
            break
        assert int(r["cls_before"]) == 1 and int(r["cls_after"]) == 1
        n_cont += 1
    assert n_cont >= 4 and int(r["cls_after"]) == 2
    ro = oracle.Oracle(sc).query(o, d, T_ORC, 10.0, push=0.0)
    assert abs((x[0] - o[0]) - ro["t"]) < 1e-5, (x, ro)


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
    tolerance is on it (a zero-length hit); a start inside the closed mesh meets a back facet first, leaving the
    mesh's opaque class"""
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
    assert bool(r["hit"]) and int(r["cls_before"]) == 6 and int(r["cls_after"]) == 1
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
