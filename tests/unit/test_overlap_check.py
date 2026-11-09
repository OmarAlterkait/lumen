"""The overlap check (lumen.validate.overlaps): Geant4's preconditions tested on the navigator -- an overlapping
sibling pair and a protruding daughter found, a clean scene clean, the pieces of one split union passed, a mesh's
pass-through reported. Ported from the check tests of the baseline's tests/test_navigator.py."""

import numpy as np
from nav_toys import nav_of, scene_mesh, scene_split

from lumen.build import lowering, scene
from lumen.validate import overlaps


def test_check_finds_overlap_and_protrusion(toys):
    nav = toys("overlap")[1]
    c = overlaps.check(nav, n_pts=256)
    named = {(nav.defs[a], nav.defs[b]) for a, b in c["pairs"]}
    assert ("a", "b") in named, named
    assert {nav.defs[int(u)] for u in c["protrude"]} == {"p"}, c["protrude"]
    clean = overlaps.check(toys("nest")[1], n_pts=128)
    assert len(clean["pairs"]) == 0 and len(clean["protrude"]) == 0, clean
    lines = overlaps.report(nav, c)
    assert "1 protruding volumes" in lines[0] and "('a', 'b')" in lines[0], lines


def test_check_passes_the_pieces_of_one_split_union():
    """split_unions' pieces of one union overlap or touch by construction: the check reports no pair of them"""
    nav = nav_of(lowering.split_unions(scene_split(), min_leaves=8))
    pcs = [i for i, d_ in enumerate(nav.defs) if "|pc" in d_]
    assert len(pcs) == 20, nav.defs
    c = overlaps.check(nav, n_pts=200)
    both = [tuple(p) for p in c["pairs"] if p[0] in pcs and p[1] in pcs]
    assert not both, both[:5]
    # the same union placed twice, overlapping itself: the pieces of one placement pass, the two placements do not
    sc = scene_split()
    sc.definitions["world"].placements.append(scene.Placement("frame", R=np.eye(3), t=np.array([-0.29, 0.0, 0.0])))
    nav2 = nav_of(lowering.split_unions(sc, min_leaves=8))
    inst = {i: nav2.names[i].rsplit("/", 1)[0] for i, d_ in enumerate(nav2.defs) if "|pc" in d_}
    pr = [tuple(p) for p in overlaps.check(nav2, n_pts=200)["pairs"] if p[0] in inst and p[1] in inst]
    assert pr and all(inst[a] != inst[b] for a, b in pr), (pr[:5], len(inst))


def test_check_meshes_reports_a_pass_through(toys):
    """an opaque mesh through a box (its sibling): the mesh check reports the pass-through"""
    sc = scene_mesh()
    sc.definitions["world"].placements[1].t = np.array([0.25, 0.1, 0.0])  # the box onto the tetrahedron
    nav = nav_of(sc)
    m = overlaps.check_meshes(nav, n_pts=64)
    assert any(nav.defs[v] == "box" for (_, v) in m["pairs"]), m
    assert overlaps.check_meshes(toys("nest")[1])["n_points"] == 0
