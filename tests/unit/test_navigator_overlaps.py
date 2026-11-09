"""Overlapping siblings navigated as Geant4 10.7.4 navigates them (g4ref's coinc_sib toys), no configuration:

- inside the volume it entered the photon sees none of its overlapping siblings; at that volume's exit it is located
  in the highest-index sibling holding the point (partial overlaps, either order);
- an entry tie (coincident faces) goes to the lowest index;
- siblings placed coincident: the lowest index fills the region; leaving it the photon crosses into the highest (a
  zero-length crossing of that pair's interface), and its zero-length step out of that one is Geant4's StepTooSmall,
  a straight continue event.

The toys are GDML strings read by the importer (the coinc_sib layout: a 4 m world, two 0.4 m cubes); ported from the
baseline's tests/test_overlaps.py, which built them with the scene builder."""

import jax
import jax.numpy as jnp
import numpy as np
from nav_toys import F32, I32, defn, make_scene

from lumen.build import lowering, solids
from lumen.gdml import importer
from lumen.geometry import navigate

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


def walk(toy, n=4):
    """the pencil from x = -0.8 m along +x: (x, class after as 0 world, 1 A, 2 B, cont) of its first n crossings"""
    sc, ids = toy
    nav = navigate.Navigator.from_scene(sc)
    q = jax.jit(lambda o, d, v, l_: nav.query(o, d, v, l_, 10.0))
    d = jnp.asarray([1.0, 0.0, 0.0], F32)
    o = jnp.asarray([-0.8, 0.0, 0.0], F32)
    v, last, out, x = nav.locate(o), I32(-1), [], -0.8
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


def test_coincident_siblings_fill_with_the_first_and_cross_into_the_last(tmp_path):
    nav, w = walk(cubes(tmp_path, "AB"))
    assert nav.coincident == [(1, 2)]  # (the world is volume 0)
    assert w[0] == (-0.2, A_, False)  # the entry tie: the lowest index
    assert w[1] == (0.2, B_, False)  # leaving A: into B (A -> B's interface)
    assert w[2] == (0.2, WORLD, True)  # B's zero-length exit: StepTooSmall, no interface
    _, w2 = walk(cubes(tmp_path, "BA"))
    assert w2[0] == (-0.2, B_, False) and w2[1] == (0.2, A_, False) and w2[2] == (0.2, WORLD, True)


def test_partial_overlap_relocates_at_the_exit_in_either_order(tmp_path):
    """B over [-0.1, 0.3]: entered A at -0.2, B unseen inside A, located in B at A's exit (0.2), out at 0.3"""
    for order in ("AB", "BA"):
        _, w = walk(cubes(tmp_path, order, 0.1))
        assert [(x, c) for x, c, _ in w[:3]] == [(-0.2, A_), (0.2, B_), (0.3, WORLD)], (order, w)
    for order in ("AB", "BA"):  # B over [-0.3, 0.1]: B first, A at B's exit
        _, w = walk(cubes(tmp_path, order, -0.1))
        assert [(x, c) for x, c, _ in w[:3]] == [(-0.3, B_), (0.1, A_), (0.2, WORLD)], (order, w)


def test_a_reflection_at_a_relocation_stays_in_the_volume_located(tmp_path):
    """Geant4 reflects with the photon already located beyond: after a face of the volume entered it steps back
    (StepTooSmall), but a photon relocated into a sibling already holding the point stays in that sibling"""
    for off, x_cross in ((0.1, 0.2), (0.0, 0.2)):  # partial overlap; coincident
        nav = navigate.Navigator.from_scene(cubes(tmp_path, "AB", off)[0])
        q = jax.jit(lambda o, d, v, l_, nav=nav: nav.query(o, d, v, l_, 10.0))
        d = jnp.asarray([1.0, 0.0, 0.0], F32)
        o = jnp.asarray([-0.8, 0.0, 0.0], F32)
        r1 = q(o, d, nav.locate(o), I32(-1))  # the entry into A: a reflection stays in the world
        assert int(r1["vol_reflect"]) == int(r1["vol_before"]) == 0
        o2 = o + r1["t"] * d
        r2 = q(o2, d, r1["vol_after"], r1["srf"])  # A's exit, located in B
        assert abs(float(o2[0] + r2["t"]) - x_cross) < 1e-5
        assert int(r2["vol_after"]) == 2 and int(r2["vol_before"]) == 1 and int(r2["vol_reflect"]) == 2


def test_a_reflection_at_a_lowered_hole_steps_back_into_the_filler():
    """the import keeps a holed solid without its holes and ranks the siblings filling them above it
    (Scene.hole_fillers): a filler's face inside the holed solid's slab is the hole's wall, a touching face in Geant4
    -- a reflection there steps back into the filler; the same pair as a plain overlap (no hole) still stays in the
    sibling located"""

    def at0(name, solid, med, pls=()):
        return defn(name, solid, med, [(p, np.eye(3), np.zeros(3)) for p in pls])

    for holed in (True, False):
        S = "S|holes:00000000" if holed else "S"  # S: a 0.2 m slab across x = 0; F: a 0.4 m cube on it
        defs = {
            "world": at0("world", solids.g4_box(4.0, 4.0, 4.0), WORLD, [S, "F"]),
            S: at0(S, solids.g4_box(0.2, 3.0, 3.0), A_),
            "F": at0("F", solids.g4_box(0.4, 0.4, 0.4), B_),
        }
        nav = navigate.Navigator.from_scene(make_scene(defs, outside=3, hole_fillers=["F"] if holed else []))
        q = jax.jit(lambda o, d, v, l_, nav=nav: nav.query(o, d, v, l_, 10.0))
        o = jnp.asarray([0.0, 0.0, 0.0], F32)
        d = jnp.asarray([0.0, 1.0, 0.0], F32)
        r = q(o, d, I32(2), I32(-1))  # the photon in F; F's face y = 0.2 lies inside S's slab
        assert abs(float(r["t"]) - 0.2) < 1e-5 and int(r["vol_after"]) == 1 and int(r["vol_before"]) == 2
        assert int(r["vol_reflect"]) == (2 if holed else 1), holed
