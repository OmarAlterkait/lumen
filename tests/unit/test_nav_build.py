"""The navigator's host build: BVH trees cover their boxes in stackless order, frame groups, coincident siblings,
another class table without mutation, the surface codes round trip, and the build cache returns the same arrays."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nav_toys import I32, defn, make_scene, nav_of, scene_nest

from lumen.build import cache as build_cache
from lumen.build import solids
from lumen.geometry import bvh, navigate, navigator_build


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
    t = bvh.build_bvh(lo, hi)
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
        one = bvh.build_bvh(lo[a:b], hi[a:b])
        np.testing.assert_array_equal(many["lo"][t0:t1], one["lo"])
        np.testing.assert_array_equal(many["skip"][t0:t1] - t0, one["skip"])
        np.testing.assert_array_equal(many["items"][a:b] - a, one["items"])


def test_rotated_daughters_get_a_frame_group(toys):
    nav = toys("wires")[1]
    assert nav.stats["framed"] == 1 and nav.stats["groups"] == 3, nav.stats
    plane = nav.defs.index("plane")
    g0, g1 = int(nav.host["vg0"][plane]), int(nav.host["vg1"][plane])
    assert g1 - g0 == 1 and not np.allclose(nav.host["gR"][g0], np.eye(3))


def test_coincident_siblings_are_paired_both_ways():
    sc = make_scene({"world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("a", np.eye(3), (0, 0, 0)),
                                                                           ("b", np.eye(3), (0, 0, 0)),
                                                                           ("c", np.eye(3), (0.5, 0, 0))]),
                     "a": defn("a", solids.g4_box(0.2, 0.2, 0.2), 2), "b": defn("b", solids.g4_box(0.2, 0.2, 0.2), 3),
                     "c": defn("c", solids.g4_box(0.2, 0.2, 0.2), 4)})  # fmt: skip
    nav = nav_of(sc)
    a, b, c = (nav.defs.index(k) for k in "abc")
    assert nav.coincident == [(a, b)]
    assert nav.host["coin_hi"][a] == b and nav.host["coin_hi"][b] == a and nav.host["coin_hi"][c] == -1


def test_with_classes_returns_a_new_navigator(toys):
    nav = toys("mesh")[1]
    cls0 = nav.host["cls"].copy()
    nav2 = nav.with_classes(np.arange(nav.n) + 10, [7], 9)
    np.testing.assert_array_equal(nav.host["cls"], cls0)  # the first is unchanged
    assert nav2.host["cls"].tolist() == [10 + i for i in range(nav.n)] + [9, 7]
    assert nav2.host["R"] is nav.host["R"]  # every other array shared


def test_surface_codes_round_trip(toys):
    nav = toys("mesh")[1]
    su, sm, LM, FM = nav.SRF
    cases = [(0, 0, 1, True, False), (1, 0, 3, False, True), (nav.CH0, 2, -1, True, True), (2, -1, -1, False, False)]
    for item, leaf, face, o, e in cases:
        s = nav.srf_pack(I32(item), I32(leaf), I32(face), jnp.bool_(o), jnp.bool_(e))
        it, lf, fc, oo, kn, ee = (np.asarray(x) for x in nav.srf_decode(s))
        if leaf < 0:
            assert int(s) == item and not kn
            continue
        assert int(it) == item and int(lf) == leaf and bool(oo) == o and bool(ee) == e and kn
        assert int(fc) == (face if item < nav.CH0 else -1)


def test_build_cache_returns_the_same_arrays(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path))
    sc = scene_nest()
    a = navigator_build.build(sc)
    b = build_cache.nav_arrays(sc)  # built and stored
    path = build_cache.entry_path(sc, {})
    assert (path / "meta.json").is_file()
    c = build_cache.nav_arrays(sc)  # read
    for x in (b, c):
        assert set(x.host) == set(a.host) and all(np.array_equal(x.host[k], a.host[k]) for k in a.host)
        assert all(x.host[k].dtype == a.host[k].dtype for k in a.host)
        assert x.names == a.names and x.defs == a.defs and x.stats == a.stats and x.kids == a.kids
        assert x.srf == a.srf and x.ch0 == a.ch0 and x.coincident == a.coincident
        for part in ("T", "K", "G"):
            pa, px = getattr(a.csg, part), getattr(x.csg, part)
            assert set(pa) == set(px) and all(np.array_equal(pa[k], px[k]) for k in pa)
    o = jnp.asarray([0.1, 0.0, -0.8], jnp.float32)
    d = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
    ra = jax.jit(lambda o_, d_: navigate.Navigator(a).query(o_, d_, 0))(o, d)
    rc = jax.jit(lambda o_, d_: navigate.Navigator(c).query(o_, d_, 0))(o, d)
    assert float(ra["t"]) == float(rc["t"]) and int(ra["srf"]) == int(rc["srf"])
    # another scene, another entry
    sc.definitions["core"].members[0].label.medium = 9
    assert build_cache.entry_path(sc, {}) != path


def test_scene_without_volumes_fails_closed():
    with pytest.raises(ValueError, match="without volumes"):
        navigator_build.build(make_scene({"world": defn("world", None, 0)}))
