"""The navigator's host build: BVH trees cover their boxes in stackless order, frame groups, coincident siblings,
another class table without mutation, the surface codes round trip, and the build cache returns the same arrays."""

import jax
import jax.numpy as jnp
import nav_toys
import numpy as np
import pytest
from nav_toys import I32, defn, make_scene, nav_of, scene_nest

from lumen.geometry import bvh, navigate, navigator_build, solids
from lumen.geometry import cache as build_cache


def _leaves_in_order(t):
    """the items of a stackless traversal that visits every node (skip only past leaves)"""
    out, node = [], 0
    while node < len(t["skip"]):
        if t["count"][node] > 0:
            out += t["items"][t["start"][node] : t["start"][node] + t["count"][node]].tolist()
            node = t["skip"][node]
        else:
            node += 1
    return out


def test_bvh_covers_every_box_once_and_bounds_its_subtree():
    rng = np.random.default_rng(0)
    c = rng.uniform(-1, 1, (37, 3))
    lo, hi = c - 0.05, c + 0.05
    t = bvh.build_many(lo, hi, [0, len(lo)])
    assert sorted(_leaves_in_order(t)) == list(range(37))
    for node in range(len(t["skip"])):  # every node's box holds its subtree's boxes
        items = []
        for k in range(node, t["skip"][node]):
            if t["count"][k] > 0:
                items += t["items"][t["start"][k] : t["start"][k] + t["count"][k]].tolist()
        assert np.all(t["lo"][node] <= lo[items].astype(np.float32)) and np.all(
            hi[items].astype(np.float32) <= t["hi"][node]
        )
    assert np.all(t["count"][t["count"] > 0] <= bvh.LEAF_SIZE)


def test_build_many_is_each_tree_alone():
    rng = np.random.default_rng(1)
    c = rng.uniform(-1, 1, (50, 3))
    lo, hi = c - 0.02, c + 0.03
    many = bvh.build_many(lo, hi, [0, 20, 50])
    ends = list(many["tree0"][1:]) + [len(many["skip"])]
    for (a, b), t0, t1 in zip(((0, 20), (20, 50)), many["tree0"], ends, strict=True):
        one = bvh.build_many(lo[a:b], hi[a:b], [0, b - a])
        np.testing.assert_array_equal(many["lo"][t0:t1], one["lo"])
        np.testing.assert_array_equal(many["skip"][t0:t1] - t0, one["skip"])
        np.testing.assert_array_equal(many["items"][a:b] - a, one["items"])


def test_rotated_daughters_get_a_frame_group(toys):
    nav = toys("wires")[1]
    assert nav.stats["framed"] == 1 and nav.stats["groups"] == 3, nav.stats
    plane = nav.defs.index("plane")
    g0, g1 = int(nav.host["vg0"][plane]), int(nav.host["vg1"][plane])
    assert g1 - g0 == 1 and not np.allclose(nav.host["gR"][g0], np.eye(3))


def test_overlapping_siblings_get_an_owner_and_contacts_none():
    """a, b coincident (each inside the other: a, placed first, owns); c inside both (proved: c owns); f overlapping a,
    b and c, inside none (a, b and c, placed first, own); d sharing a face with a (a contact, no link); e
    apart. The owner is a daughter candidate of its non-owner."""
    box = solids.g4_box
    sc = make_scene({"world": defn("world", box(2.0, 2.0, 2.0), 1, [("a", np.eye(3), (0, 0, 0)),
                                                                   ("b", np.eye(3), (0, 0, 0)),
                                                                   ("c", np.eye(3), (0.04, 0, 0)),
                                                                   ("d", np.eye(3), (-0.2, 0, 0)),
                                                                   ("e", np.eye(3), (0.6, 0, 0)),
                                                                   ("f", np.eye(3), (0.125, 0, 0))]),
                     "a": defn("a", box(0.2, 0.2, 0.2), 2), "b": defn("b", box(0.2, 0.2, 0.2), 3),
                     "c": defn("c", box(0.1, 0.1, 0.1), 4), "d": defn("d", box(0.2, 0.2, 0.2), 5),
                     "e": defn("e", box(0.2, 0.2, 0.2), 6), "f": defn("f", box(0.1, 0.1, 0.1), 7)})  # fmt: skip
    nav = nav_of(sc)
    a, b, c, d, e, f = (nav.defs.index(k) for k in "abcdef")
    want = [(b, a, False), (a, c, False), (b, c, False), (f, a, False), (f, b, False), (f, c, False)]
    assert sorted(nav.owned) == sorted(want), nav.owned
    assert nav.stats["links"] == 6 and nav.stats["hole_links"] == 0
    assert a in nav.kids[b] and c in nav.kids[a] and c in nav.kids[b] and not set(nav.kids[d]) | set(nav.kids[e])
    prio = nav.host["prio"]  # the rule's order: the owner first
    assert prio[c] < prio[a] < prio[b] < prio[f]


def test_with_classes_returns_new_arrays(toys):
    nav = toys("mesh")[1]
    cls0 = nav.host["cls"].copy()
    nav2 = nav.arrays.with_classes(np.arange(nav.n) + 10, [7], 9)
    np.testing.assert_array_equal(nav.host["cls"], cls0)  # the first is unchanged
    assert nav2.host["cls"].tolist() == [10 + i for i in range(nav.n)] + [9, 7]
    assert nav2.host["R"] is nav.host["R"]  # every other array shared


def test_surface_codes_round_trip(toys):
    nav = toys("mesh")[1]
    su, sm, FM = nav.SRF
    cases = [(0, 0, 1, True, False), (1, 0, 3, False, True), (nav.CH0, 2, -1, True, True), (2, -1, -1, False, False)]
    for item, leaf, face, o, e in cases:
        s = nav.srf_pack(I32(item), I32(leaf), I32(face), jnp.bool_(o), jnp.bool_(e))
        it, lf, fc, oo, kn, ee = (np.asarray(x) for x in nav.srf_decode(s))
        if leaf < 0:
            assert int(s) == item and not kn
            continue
        assert int(it) == item and int(lf) == leaf and bool(oo) == o and bool(ee) == e and kn
        assert int(fc) == (face if item < nav.CH0 else -1)


def test_surface_codes_span_each_volumes_own_leaves(toys):
    """every (volume, leaf, face, o, e) of a scene with booleans round-trips, and the code space is the scene's
    leaves, not the volumes times the largest solid's (a detector of many tubes and one many-leaf solid fits int32)"""
    nav = toys("bool")[1]
    su, sm, FM = nav.SRF
    nl = np.asarray(nav.E.tables.K["nl"])[np.asarray(nav.host["kind"])]
    assert sm - su == 4 * FM * int(np.maximum(nl, 1).sum())
    cases = [
        (i, lf, fc, o, e)
        for i in range(nav.n)
        for lf in range(max(int(nl[i]), 1))
        for fc in range(FM)
        for o in (False, True)
        for e in (False, True)
    ]
    it, lf, fc, o, e = (np.asarray(c) for c in zip(*cases, strict=True))
    s = jax.vmap(nav.srf_pack)(I32(it), I32(lf), I32(fc), jnp.asarray(o), jnp.asarray(e))
    assert len(np.unique(np.asarray(s))) == len(cases) and int(np.max(s)) < sm
    d = jax.vmap(nav.srf_decode)(s)
    for got, want in zip(d[:3], (it, lf, fc), strict=True):
        np.testing.assert_array_equal(np.asarray(got), want)
    np.testing.assert_array_equal(np.asarray(d[3]), o)
    np.testing.assert_array_equal(np.asarray(d[5]), e)


def test_build_cache_returns_the_same_arrays(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path))
    sc = scene_nest()
    a = navigator_build.build(sc)
    b, k = build_cache.nav_arrays(sc)  # built and stored
    assert (tmp_path / "build" / f"nav-{k}" / "meta.json").is_file()
    c, _ = build_cache.nav_arrays(sc)  # read
    assert build_cache.scene_digest(c) == build_cache.scene_digest(a)  # every field of NavArrays: types and values
    for x in (b, c):
        assert set(x.host) == set(a.host) and all(np.array_equal(x.host[k], a.host[k]) for k in a.host)
        assert all(x.host[k].dtype == a.host[k].dtype for k in a.host)
        assert x.names == a.names and x.defs == a.defs and x.stats == a.stats and x.kids == a.kids
        assert x.srf == a.srf and x.ch0 == a.ch0 and x.owned == a.owned
        for part in ("T", "K", "G"):
            pa, px = getattr(a.csg, part), getattr(x.csg, part)
            assert set(pa) == set(px) and all(np.array_equal(pa[k], px[k]) for k in pa)
    o = jnp.asarray([0.1, 0.0, -0.8], jnp.float32)
    d = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
    ra = jax.jit(lambda o_, d_: navigate.Navigator(a).query(o_, d_, 0, t_max=10.0))(o, d)
    rc = jax.jit(lambda o_, d_: navigate.Navigator(c).query(o_, d_, 0, t_max=10.0))(o, d)
    assert float(ra["t"]) == float(rc["t"]) and int(ra["srf"]) == int(rc["srf"])
    # another scene, another entry
    sc.definitions["core"].member.label.medium = 9
    assert build_cache.key(sc, {}) != k


def test_the_scene_s_ownership_is_resolved_once_for_every_set_of_classes(tmp_path, monkeypatch):
    """The class-free stage's ownership (the costly part) has its own entry: a build for other classes or settings
    reads it, and gives the arrays an uncached build gives, bit for bit (dissolution included)."""
    from lumen.geometry import ownership

    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path))
    sc = scene_nest()
    n = navigator_build.build(sc).n
    cl = navigator_build.Classes(tuple([0] * n), (), 0, (), (0,))  # one class: every volume that may dissolve does
    want = navigator_build.build(sc, classes=cl)
    assert want.stats["transparent"] > 0
    calls = []
    real = ownership.resolve
    monkeypatch.setattr(ownership, "resolve", lambda u: calls.append(1) or real(u))
    build_cache.nav_arrays(sc)  # the placement check's build (no classes): resolves and keeps the ownership
    got, _ = build_cache.nav_arrays(sc, classes=cl)  # the first load's: another entry, the same ownership
    assert len(calls) == 1
    assert build_cache.scene_digest(got) == build_cache.scene_digest(want)
    st = navigator_build.prepare(sc)
    assert build_cache.scene_digest(navigator_build.finish(st, cl)) == build_cache.scene_digest(want)


def test_scene_without_volumes_fails_closed():
    with pytest.raises(ValueError, match="without volumes"):
        navigator_build.build(make_scene({"world": defn("world", None, 0)}))


@pytest.mark.parametrize("name", ["scene_nest", "scene_bool", "scene_wires", "scene_split", "scene_overlap",
                                  "scene_coincident", "scene_mesh"])  # fmt: skip
@pytest.mark.parametrize("one_class", [True, False])
def test_the_navigable_volumes_are_exactly_the_ones_not_dissolved(name, one_class):
    """``navigable`` (the volumes reachable from the outside's candidates through the candidates of the volumes
    reached: what the jump's clearance counts) is exactly the complement of the per-classes stage's transparent mask,
    nested dissolutions included (one class: every volume that may dissolve does), so the two cannot drift apart."""
    sc = getattr(nav_toys, name)()
    st = navigator_build.prepare(sc)
    plain = navigator_build.finish(st)
    n = plain.n
    vol_cls = [0] * n if one_class else np.asarray(plain.host["cls"])[:n].tolist()
    mesh_cls = [0] * plain.n_mesh if one_class else np.asarray(plain.host["cls"])[n + 1 :].tolist()
    out_cls = 0 if one_class else int(plain.host["cls"][n])
    k = 1 + max([out_cls, *vol_cls, *mesh_cls])
    cl = navigator_build.Classes(tuple(vol_cls), tuple(mesh_cls), out_cls, (), tuple(range(k)))
    clear = navigator_build._transparent(n, st.parent, st.own, cl)
    arr = navigator_build.finish(st, cl)
    assert np.array_equal(navigator_build.navigable(arr), np.flatnonzero(~clear))
    assert np.array_equal(navigator_build.navigable(plain), np.arange(n))
    if one_class and name == "scene_nest":
        assert clear.sum() > 1  # nested: a transparent volume inside a transparent one
