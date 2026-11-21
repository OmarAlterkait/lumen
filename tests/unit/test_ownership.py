"""Overlapping siblings (lumen.geometry.ownership): a solid proved inside its sibling owns the region they share, else
the first placed; a contact is no link; a link proved apart is none; a non-owner's daughter that reaches into its
owner is cut by it; the pieces of a split union hold the union; the float64 oracle labels points by the same rule; a
transparent volume is dissolved into its listers. (The overlap-free GDML that says it in Geant4's terms:
test_ownership_export.)"""

import jax
import numpy as np
from nav_toys import defn, make_scene, nav_of

from lumen.build import lowering
from lumen.compare import oracle
from lumen.geometry import navigate, navigator_build, ownership, solids

box = solids.g4_box
I3 = np.eye(3)


def _scene(*placed, outside=0):
    """world (medium 1) holding (name, solid, medium, x) boxes, in that order"""
    defs = {"world": defn("world", box(2.0, 2.0, 2.0), 1, [(n, I3, (x, 0.0, 0.0)) for n, _, _, x in placed])}
    defs.update({n: defn(n, s, m) for n, s, m, _ in placed})
    return make_scene(defs, outside=outside)


def _owned(nav):
    return sorted((nav.defs[a], nav.defs[b]) for a, b, f in nav.owned if not f)


def test_a_solid_proved_inside_owns_else_the_first_placed():
    big, inner = ("a", box(0.4, 0.4, 0.4), 2, 0.0), ("b", box(0.2, 0.2, 0.2), 3, 0.05)  # b inside a
    assert _owned(nav_of(_scene(big, inner))) == [("a", "b")]
    assert _owned(nav_of(_scene(inner, big))) == [("a", "b")]
    part = ("b", box(0.2, 0.2, 0.2), 3, 0.25)  # half of b inside a: the first placed owns
    assert _owned(nav_of(_scene(big, part))) == [("b", "a")]
    assert _owned(nav_of(_scene(part, big))) == [("a", "b")]
    twin = ("c", box(0.4, 0.4, 0.4), 4, 0.1)  # the same solid as a, 0.3 m of it shared
    assert _owned(nav_of(_scene(big, twin))) == [("c", "a")]
    assert _owned(nav_of(_scene(twin, big))) == [("a", "c")]
    # an orb off a's edge, its box 1 cm into a's: no common point, proved apart -- no link
    defs = {"world": defn("world", box(2.0, 2.0, 2.0), 1, [("a", I3, (0.0, 0.0, 0.0)), ("d", I3, (0.24, 0.24, 0.0))]),
            "a": defn("a", box(0.4, 0.4, 0.4), 2), "d": defn("d", solids.g4_orb(0.05), 5)}  # fmt: skip
    assert _owned(nav_of(make_scene(defs))) == []


def test_a_contact_is_no_overlap():
    """a face shared exactly, or within the contact tolerance (K_ON bands), is a contact; deeper is an overlap; a
    sliver within the tolerance is listed as a thin pair (it may overlap by less than the tolerance)"""
    a = ("a", box(0.4, 0.4, 0.4), 2, 0.0)
    for gap, over, thin in ((0.0, False, False), (-2e-7, False, True), (-1e-4, True, False), (1e-6, False, False)):
        nav = nav_of(_scene(a, ("b", box(0.2, 0.2, 0.2), 3, 0.3 + gap)))
        assert bool(_owned(nav)) == over, (gap, nav.owned)
        assert bool(nav.thin) == thin, (gap, nav.thin)  # within the tolerance: reported (placements.check)


def _media(nav, sc, x):
    """the medium at points x: the navigator's locate and the oracle's"""
    med_nav = np.asarray(nav.host["cls"])[np.asarray(jax.jit(jax.vmap(nav.locate))(x.astype(np.float32)))]
    return med_nav, oracle.Oracle(sc).label_many(x)[0]


def test_a_thin_overlap_along_a_face_belongs_to_the_owner():
    """a 0.4 m cube pressed 0.05 mm into the 2 cm wall of a hollow box around it (its box inside the hollow box's: a
    link either way): the 0.05 mm, or 1 um, is the wall's (placed first, neither inside the other), the cavity the
    cube's, for the navigator and the oracle alike"""
    shell = solids.Op("sub", box(0.44, 0.44, 0.44), box(0.4, 0.4, 0.4))  # a 2 cm wall, the cavity [-0.2, 0.2]
    for dx in (5e-5, 1e-6):
        sc = _scene(("s", shell, 2, 0.0), ("c", box(0.4, 0.4, 0.4), 3, dx))
        nav = nav_of(sc)
        assert _owned(nav) == [("c", "s")], nav.owned
        x = np.array([[0.2 + dx / 2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.21, 0.0, 0.0], [0.2 - 1e-4, 0.1, 0.1]])
        med_nav, med_orc = _media(nav, sc, x)
        want = [2, 3, 2, 3]
        assert med_nav.tolist() == med_orc.tolist() == want, (dx, med_nav, med_orc)


def test_the_pieces_of_a_split_union_hold_the_union():
    """((b1 u b2) - s) u b3, split into b1 - s, b2 - s and b3 (overlapping, the union's volume each: the first placed
    owns): every point of the union is the union's medium, b3's part inside b1 and s included"""
    b1, b2, b3 = box(0.4, 0.2, 0.2), box(0.2, 0.4, 0.2), box(0.2, 0.2, 0.4)
    s_ = solids.Xf(I3, np.array([0.1, 0.0, 0.0]), box(0.1, 0.1, 0.1))
    u12 = solids.Op("union", b1, b2)
    sub = solids.Op("sub", u12, s_)
    u = solids.Op("union", sub, b3)
    for op in (u12, sub, u):
        op.g4bool = True
    sc = lowering.split_unions(_scene(("u", u, 2, 0.0)), min_leaves=2)
    assert len(sc.definitions["u"].placements) == 3
    x = np.array([[0.08, 0.0, 0.0], [0.08, 0.0, 0.15], [0.12, 0.0, 0.0], [0.0, 0.15, 0.0], [0.15, 0.15, 0.15]])
    med_nav, med_orc = _media(nav_of(sc), sc, x)
    assert med_nav.tolist() == med_orc.tolist() == [2, 2, 1, 2, 1], (med_nav, med_orc)


def _walk(nav, o, d, n=6):
    """the classes after each visible crossing of a straight ray (no physics): (x along d, class)"""
    q = jax.jit(lambda o_, v, l_: nav.query(o_, d, v, last=l_, t_max=10.0))
    v, last, out, s = int(jax.jit(nav.locate)(o)), -1, [], 0.0
    cls = np.asarray(nav.host["cls"])
    for _ in range(n):
        r = q(o, v, last)
        if not bool(r["hit"]):
            break
        s += float(r["t"])
        out.append((round(s, 4), int(cls[int(r["vol_after"])])))
        o, v, last = o + r["t"] * d, r["vol_after"], r["srf"]
    return out


def test_a_non_owners_daughter_is_cut_by_the_owner_and_the_owner_shows_its_daughters():
    """A (0.4 m, medium 2) holds D (medium 5) reaching x = 0.15; B (0.2 m at x = 0.2, medium 3, placed first) with its
    daughter W (medium 4) owns [0.1, 0.2]: along +x from inside D the photon leaves D into B at 0.1, then W"""
    defs = {
        "world": defn("world", box(2.0, 2.0, 2.0), 1, [("B", I3, (0.2, 0, 0)), ("A", I3, (0.0, 0, 0))]),
        "A": defn("A", box(0.4, 0.4, 0.4), 2, [("D", I3, (0.0, 0, 0))]),
        "D": defn("D", box(0.3, 0.1, 0.1), 5),
        "B": defn("B", box(0.2, 0.2, 0.2), 3, [("W", I3, (0.05, 0, 0))]),
        "W": defn("W", box(0.02, 0.02, 0.02), 4),
    }
    nav = nav_of(make_scene(defs))
    a, b, d = (nav.defs.index(k) for k in "ABD")
    assert (a, b, False) in nav.owned and b in nav.kids[d]
    w = _walk(nav, np.float32([-0.05, 0, 0]), np.float32([1.0, 0, 0]))
    assert w[:3] == [(0.15, 3), (0.29, 4), (0.31, 3)], w  # distances from x = -0.05
    orc = oracle.Oracle(make_scene(defs))
    pts = np.array([[0.12, 0, 0], [0.0, 0, 0], [0.25, 0, 0], [0.29, 0, 0], [-0.15, 0.15, 0]])
    assert orc.label(pts)[0].tolist() == [3, 5, 4, 3, 2]


def test_a_daughter_holds_only_what_lies_inside_its_mother():
    """M (0.4 m box, medium 2) holds D (0.2 m, medium 3) reaching 0.05 m out of it, and D holds E (medium 4) lying
    wholly outside M: along +x from M's centre the photon enters D at 0.05 and leaves at M's face (0.2) into the world;
    E is never met; a point in D beyond M is the world's, for the navigator and the oracle alike"""
    defs = {
        "world": defn("world", box(2.0, 2.0, 2.0), 1, [("M", I3, (0.0, 0, 0))]),
        "M": defn("M", box(0.4, 0.4, 0.4), 2, [("D", I3, (0.15, 0, 0))]),
        "D": defn("D", box(0.2, 0.2, 0.2), 3, [("E", I3, (0.08, 0, 0))]),
        "E": defn("E", box(0.04, 0.04, 0.04), 4),
    }
    nav = nav_of(make_scene(defs))
    m, d, e = (nav.defs.index(k) for k in "MDE")
    assert (d, m) in nav.clipped and m in nav.kids[d] and m in nav.kids[e]
    assert _walk(nav, np.float32([0.0, 0, 0]), np.float32([1.0, 0, 0]))[:2] == [(0.05, 3), (0.2, 1)]
    assert int(jax.jit(nav.locate)(np.float32([0.22, 0, 0]))) == nav.defs.index("world")
    pts = np.array([[0.1, 0, 0], [0.22, 0, 0], [0.19, 0.05, 0], [-0.1, 0, 0]])
    assert oracle.Oracle(make_scene(defs)).label(pts)[0].tolist() == [3, 1, 3, 2]


def test_the_oracle_and_the_navigator_locate_overlapping_siblings_alike():
    sc = _scene(
        ("a", box(0.4, 0.4, 0.4), 2, 0.0), ("b", box(0.2, 0.3, 0.2), 3, 0.25), ("c", solids.g4_orb(0.15), 4, -0.1)
    )
    nav = nav_of(sc)
    x = np.random.default_rng(3).uniform(-0.45, 0.45, size=(4000, 3)).astype(np.float32)
    med_nav = np.asarray(nav.host["cls"])[np.asarray(jax.jit(jax.vmap(nav.locate))(x))]
    med_orc = oracle.Oracle(sc).label_many(x.astype(np.float64))[0]
    on_face = np.zeros(len(x), bool)
    for c in (-0.2, 0.2, 0.15, 0.35, 0.05):  # points on a face are either side's
        on_face |= np.abs(x[:, 0] - c) < 1e-6
    assert np.array_equal(med_nav[~on_face], med_orc[~on_face])
    assert {2, 3, 4} <= set(med_orc.tolist())


def test_a_volume_of_its_listers_class_is_dissolved():
    """C (medium 1, the tank's) holds W (medium 4); P (medium 1) holds Q (medium 5) and is a sensor: given the classes,
    C is no candidate and W is the tank's, an exit from W goes to the tank, and the walk meets the same classes in
    fewer rounds; P stays"""
    defs = {
        "world": defn("world", box(2.0, 2.0, 2.0), 1, [("C", I3, (0.0, 0, 0)), ("P", I3, (0.6, 0, 0))]),
        "C": defn("C", box(0.4, 0.4, 0.4), 1, [("W", I3, (0.1, 0, 0))]),
        "W": defn("W", box(0.1, 0.1, 0.1), 4),
        "P": defn("P", box(0.2, 0.2, 0.2), 1, [("Q", I3, (0.0, 0, 0))]),
        "Q": defn("Q", box(0.1, 0.1, 0.1), 5),
    }
    sc = make_scene(defs)
    plain = nav_of(sc)
    world, c, w, p = (plain.defs.index(k) for k in ("world", "C", "W", "P"))
    cls = np.asarray(plain.host["cls"])
    arr = navigator_build.build(
        sc,
        classes=navigator_build.Classes(tuple(cls[: plain.n].tolist()), (), int(cls[plain.n]), (p,), tuple(range(8))),
    )
    clear = navigate.Navigator(arr)
    assert arr.stats["transparent"] == 1 and c not in clear.kids[world] and w in clear.kids[world]
    assert int(clear.host["parent"][w]) == world and p in clear.kids[world]
    o, d = np.float32([-0.5, 0.0, 0.0]), np.float32([1.0, 0.0, 0.0])
    assert _walk(clear, o, d) == _walk(plain, o, d)
    q0 = plain.query(o, d, world, t_max=10.0)
    q1 = clear.query(o, d, world, t_max=10.0)
    assert int(q1["rounds"]) < int(q0["rounds"]) and int(q1["vol_after"]) == int(q0["vol_after"]) == w


def test_a_holed_solid_ranks_after_its_fillers():
    """h (holed, filled by f, placed before it), f and x, the three boxes overlapping, none inside another: f ranks
    before h, h (placed first) before x"""
    u = ownership.Units(parent=np.full(3, -1), kind=np.arange(3), nodes=[box(0.2, 0.2, 0.2)] * 3,
                        box=((np.zeros(3), np.full(3, 0.1)),) * 3, R=np.broadcast_to(I3, (3, 3, 3)),
                        t=np.array([[0.0, 0, 0], [0.05, 0, 0], [0.1, 0, 0]]), place=np.arange(3),
                        forced=np.array([[0, 1]]))  # fmt: skip
    own = ownership.resolve(u)
    assert own.prio[1] < own.prio[0] < own.prio[2], own.prio
    assert sorted(map(tuple, own.pairs.tolist())) == [(0, 1), (2, 0), (2, 1)], own.pairs


def test_the_rule_order_is_the_placement_order_with_proved_insides_ahead():
    """a cycle of pairwise choices is no order: c inside a (c ahead), a placed before b, b before c -- the order is
    one list all the same, each proved inside ahead of its container"""
    pos = ownership.sibling_order(np.arange(3), np.array([[2, 0]]), np.zeros((0, 2), np.int64))
    assert pos[2] < pos[0] and sorted(pos.tolist()) == [0, 1, 2]
    none = np.zeros((0, 2), np.int64)
    assert ownership.sibling_order(np.array([2, 0, 1]), none, none).tolist() == [2, 0, 1]
