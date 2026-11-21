"""Overlapping siblings navigated by ownership (lumen.geometry.ownership), no configuration:

- the region two siblings share belongs to a solid proved inside the other, else to the one placed first: a photon in
  the non-owner enters the owner at the owner's face, and leaves the owner into the non-owner;
- coincident siblings (one solid, one pose): the first placed fills the region, the other holds nothing;
- an entry tie (coincident faces) goes to the first in the rule's order (the owner).

The toys are GDML strings read by the importer (the coinc_sib layout: a 4 m world, two 0.4 m cubes)."""

import jax
import numpy as np
from nav_toys import defn, locate, make_scene, one, query

from lumen.build import lowering
from lumen.gdml import importer
from lumen.geometry import navigate, solids

WORLD, A_, B_ = 0, 1, 2
CUBES = """<?xml version="1.0"?>
<gdml>
 <materials>
  <element name="H" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="MW"><D value="1"/><fraction n="1" ref="H"/></material>
  <material name="MA"><D value="1"/><fraction n="1" ref="H"/></material>
  <material name="MB"><D value="1"/><fraction n="1" ref="H"/></material>
 </materials>
 <solids>
  <box name="wbox" x="4000" y="4000" z="4000" lunit="mm"/>
  <box name="cube" x="400" y="400" z="400" lunit="mm"/>
 </solids>
 <structure>
  <volume name="volA"><materialref ref="MA"/><solidref ref="cube"/></volume>
  <volume name="volB"><materialref ref="MB"/><solidref ref="cube"/></volume>
  <volume name="volWorld"><materialref ref="MW"/><solidref ref="wbox"/>
   {physvols}
  </volume>
 </structure>
 <setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""


def _physvol(k, off_b):
    pos = 0.0 if k == "A" else off_b * 1000.0
    return f'<physvol name="p{k}"><volumeref ref="vol{k}"/><position name="x{k}" x="{pos}" y="0" z="0" unit="mm"/>' \
           "</physvol>"  # fmt: skip


def cubes(tmp_path, order, off_b=0.0):
    """world (MW) holding A (MA, a 0.4 m cube at the origin) and B (MB, the same cube at x = off_b), placed in
    ``order``; returns (scene, the classes (media ids) of world, A and B)"""
    path = tmp_path / f"cubes_{order}_{off_b}.gdml"
    path.write_text(CUBES.format(physvols="".join(_physvol(k, off_b) for k in order)))
    loaded = lowering.lower(importer.read_gdml(path), detector="volWorld")
    return loaded.scene, tuple(loaded.media[m] for m in ("MW", "MA", "MB"))


def _q(nav):
    """one query (o, d, vol, last) at t_max 10 m; one program serves every navigator of the cubes' layout"""
    return lambda o, d, v, l_: one(query(nav, o, d, v, l_, 10.0, by_layout=True))


def walk(toy, n=4):
    """the pencil from x = -0.8 m along +x: (x, class after as 0 world, 1 A, 2 B, cont) of its first n crossings"""
    sc, ids = toy
    nav = navigate.Navigator.from_scene(sc)
    q = _q(nav)
    d = np.asarray([1.0, 0.0, 0.0], np.float32)
    o = np.asarray([-0.8, 0.0, 0.0], np.float32)
    v, last, out, x = locate(nav, o, by_layout=True)[0], -1, [], -0.8
    cls = np.asarray(nav.host["cls"])
    for _ in range(n):
        r = q(o, d, v, last)
        if not bool(r["hit"]):
            break
        x += float(r["t"])
        c = int(cls[int(r["vol_after"])])
        out.append((round(x, 4), ids.index(c) if c in ids else -1, bool(r["cont"])))  # -1: the outside
        o, v, last = o + r["t"] * d, r["vol_after"], r["srf"]
    return nav, out


def test_coincident_siblings_fill_with_the_first_placed(tmp_path):
    """equal volumes, one pose: the first placed owns all of it; the photon crosses it once, the other is never seen"""
    nav, w = walk(cubes(tmp_path, "AB"))
    assert [(2, 1, False)] == [tuple(x) for x in nav.owned]  # (non-owner, owner; the world is volume 0)
    assert w[:2] == [(-0.2, A_, False), (0.2, WORLD, False)], w
    _, w2 = walk(cubes(tmp_path, "BA"))
    assert w2[:2] == [(-0.2, B_, False), (0.2, WORLD, False)], w2


def test_partial_overlap_of_equal_volumes_goes_to_the_first_placed(tmp_path):
    """B over [-0.1, 0.3]: A placed first owns [-0.1, 0.2] (A, then B's rest); B first owns it (A's rest, then B)"""
    _, w = walk(cubes(tmp_path, "AB", 0.1))
    assert [(x, c) for x, c, _ in w[:3]] == [(-0.2, A_), (0.2, B_), (0.3, WORLD)], w
    _, w = walk(cubes(tmp_path, "BA", 0.1))
    assert [(x, c) for x, c, _ in w[:3]] == [(-0.2, A_), (-0.1, B_), (0.3, WORLD)], w
    _, w = walk(cubes(tmp_path, "AB", -0.1))  # B over [-0.3, 0.1]
    assert [(x, c) for x, c, _ in w[:3]] == [(-0.3, B_), (-0.2, A_), (0.2, WORLD)], w
    _, w = walk(cubes(tmp_path, "BA", -0.1))
    assert [(x, c) for x, c, _ in w[:3]] == [(-0.3, B_), (0.1, A_), (0.2, WORLD)], w


def test_a_solid_inside_owns_whatever_the_order_else_the_first_placed(tmp_path):
    """a 0.2 m cube S over [0.1, 0.3], half inside the 0.4 m cube A: the first placed owns the shared part; S over
    [0.0, 0.2], inside A: S owns it in either order"""
    for x_s, order, want in ((200.0, "AS", [(-0.2, A_), (0.2, B_), (0.3, WORLD)]),
                             (200.0, "SA", [(-0.2, A_), (0.1, B_), (0.3, WORLD)]),
                             (100.0, "AS", [(-0.2, A_), (0.0, B_), (0.2, WORLD)]),
                             (100.0, "SA", [(-0.2, A_), (0.0, B_), (0.2, WORLD)])):  # fmt: skip
        x = {"A": 0.0, "S": x_s}
        body = "".join(f'<physvol name="p{k}"><volumeref ref="vol{k}"/><position name="x{k}" x="{x[k]}" y="0" z="0" '
                       'unit="mm"/></physvol>' for k in order)  # fmt: skip
        gd = CUBES.replace('<box name="cube" x="400" y="400" z="400" lunit="mm"/>',
                           '<box name="cube" x="400" y="400" z="400" lunit="mm"/><box name="small" x="200" y="200" '
                           'z="200" lunit="mm"/>').replace('<volume name="volB"><materialref ref="MB"/><solidref '
                           'ref="cube"/></volume>', '<volume name="volS"><materialref ref="MB"/><solidref ref="small"/>'
                           '</volume>').format(physvols=body)  # fmt: skip
        path = tmp_path / f"small_{order}_{x_s}.gdml"
        path.write_text(gd)
        loaded = lowering.lower(importer.read_gdml(path), detector="volWorld")
        ids = tuple(loaded.media[m] for m in ("MW", "MA", "MB"))
        _, w = walk((loaded.scene, ids))
        assert [(x, c) for x, c, _ in w[:3]] == want, (x_s, order, w)


def test_a_reflection_at_an_owner_face_stays_where_the_photon_was(tmp_path):
    """Geant4 reflects with the photon located where it was: at the face between an owner and its non-owner (A owns
    [-0.1, 0.2] of B, the first placed of equal volumes) a reflection stays in A, a crossing goes into B"""
    nav = navigate.Navigator.from_scene(cubes(tmp_path, "AB", 0.1)[0])
    q = _q(nav)
    d = np.asarray([1.0, 0.0, 0.0], np.float32)
    o = np.asarray([-0.8, 0.0, 0.0], np.float32)
    r1 = q(o, d, locate(nav, o, by_layout=True)[0], -1)  # the entry into A: a reflection stays in the world
    assert int(r1["vol_before"]) == 0
    o2 = o + r1["t"] * d
    r2 = q(o2, d, r1["vol_after"], r1["srf"])  # A's exit at 0.2, into B
    assert abs(float(o2[0] + r2["t"]) - 0.2) < 1e-5
    assert int(r2["vol_after"]) == 2 and int(r2["vol_before"]) == 1  # a reflection stays in vol_before


def test_a_reflection_at_a_lowered_hole_steps_back_into_the_filler():
    """the import keeps a holed solid without its holes and the siblings filling them own the holes
    (Scene.hole_fillers): a filler's face inside the holed solid's slab is the hole's wall, a touching face in Geant4
    -- a reflection there steps back into the filler; the same pair as a plain overlap (no hole, F placed first)
    gives the shared region to F, so the same"""

    def at0(name, solid, med, pls=()):
        return defn(name, solid, med, [(p, np.eye(3), np.zeros(3)) for p in pls])

    for holed in (True, False):
        S = "S"  # S: a 0.2 m slab across x = 0 (holed: Definition.holed, filled by F); F: a 0.4 m cube on it
        defs = {
            "world": at0("world", solids.g4_box(4.0, 4.0, 4.0), WORLD, [S, "F"] if holed else ["F", S]),
            S: at0(S, solids.g4_box(0.2, 3.0, 3.0), A_),
            "F": at0("F", solids.g4_box(0.4, 0.4, 0.4), B_),
        }
        defs[S].holed = holed
        nav = navigate.Navigator.from_scene(make_scene(defs, outside=3, hole_fillers=["F"] if holed else []))
        q = _q(nav)
        o = np.asarray([0.0, 0.0, 0.0], np.float32)
        d = np.asarray([0.0, 1.0, 0.0], np.float32)
        f_, s_ = nav.names.index("F_1" if holed else "F_0"), nav.names.index("S_0" if holed else "S_1")
        r = q(o, d, f_, -1)  # the photon in F; F's face y = 0.2 lies inside S's slab
        assert abs(float(r["t"]) - 0.2) < 1e-5 and int(r["vol_after"]) == s_ and int(r["vol_before"]) == f_, holed


def test_locate_picks_the_owner(toys, tmp_path):
    """a point in a ∩ b is a's, placed first, neither inside the other (the same volume a ray leaving b there is
    located in); two coincident placements: the first placed"""
    _, nav = toys("overlap")
    a, b = nav.names.index("a_0"), nav.names.index("b_1")
    assert list(locate(nav, [[0.025, 0.0, 0.0], [0.045, 0.1, -0.1], [0.1, 0.0, 0.0]])) == [a, a, b]
    o = np.asarray([0.15, 0.0, 0.0], np.float32)
    r = one(query(nav, o, [-1.0, 0.0, 0.0], b, t_max=10.0))  # a's face at x = 0.05, inside b: a entered there
    assert int(r["vol_after"]) == a and abs(float(r["t"]) - 0.1) < 1e-6
    for order in ("AB", "BA"):
        nav2 = navigate.Navigator.from_scene(cubes(tmp_path, order)[0])
        assert int(locate(nav2, [[0.05, 0.0, 0.0]], by_layout=True)[0]) == 1, order


def test_the_volume_at_a_bulk_event_is_the_rays_and_a_fresh_locates():
    """invisible crossings (one class) through overlapping siblings: the walk is in A over the overlap (A, placed
    first, owns it), as Geant4 stepping through each crossing, and a fresh point locate agrees. The query's vol_at is
    the ray's volume at s_at, before, inside and past the overlap"""
    sc = make_scene(
        {
            "world": defn(
                "world", solids.g4_box(4.0, 4.0, 4.0), 1, [("A", np.eye(3), (0.0, 0, 0)), ("B", np.eye(3), (0.2, 0, 0))]
            ),
            "A": defn("A", solids.g4_box(0.4, 0.4, 0.4), 1),
            "B": defn("B", solids.g4_box(0.4, 0.4, 0.4), 1),
        }
    )
    nav = navigate.Navigator.from_scene(sc)
    a, b = nav.names.index("A_0"), nav.names.index("B_1")
    o, d = np.asarray([-0.8, 0.0, 0.0], np.float32), np.asarray([1.0, 0.0, 0.0], np.float32)
    v0 = int(locate(nav, o)[0])
    q = jax.jit(lambda s: nav.query(o, d, v0, t_max=3.0, s_at=s)["vol_at"])
    got = [int(q(np.float32(x + 0.8))) for x in (-0.5, -0.1, 0.1, 0.25, 0.5)]
    assert got == [v0, a, a, b, v0], got
    assert int(locate(nav, [[0.1, 0.0, 0.0]])[0]) == a
