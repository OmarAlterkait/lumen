"""The placement check (lumen.build.placements): Geant4's preconditions tested on the navigator -- an overlapping
sibling pair found, a protruding daughter found, a clean scene clean, the pieces of one split union one solid, a mesh's
pass-through reported."""

import numpy as np
import pytest
from nav_toys import nav_of, scene_mesh, scene_overlap, scene_split
from test_api import RECIPE, _toy
from test_importer import write

import lumen
from lumen.build import lowering, placements
from lumen.geometry import scene


def test_check_finds_overlap_and_protrusion(toys):
    nav = toys("overlap")[1]
    c = placements.check(nav, n_pts=256)
    named = {(nav.defs[a], nav.defs[b]) for a, b in c["pairs"]}
    assert ("a", "b") in named, named
    assert {nav.defs[int(u)] for u in c["protrude"]} == {"p"}, c["protrude"]
    # a point of a in b is b's, a point of p beyond its mother m the world's: no overlap across levels
    assert len(c["cross"]) == 0 and [(nav.defs[v], nav.defs[m]) for v, m in nav.clipped] == [("p", "m")], c["cross"]
    clean = placements.check(toys("nest")[1], n_pts=128)
    assert len(clean["pairs"]) == 0 and len(clean["protrude"]) == 0, clean
    lines = placements.report(nav, c)
    assert "1 protruding volumes" in lines[0] and "1 sibling pairs (points per pair" in lines[0], lines
    moved = nav_of(scene_overlap(b_x=0.12))  # b moved, still overlapping a
    first = placements.check(nav, n_pts=256, backend="xla")  # jax.vmap (c ran on tallymap's kernel backend)
    c2 = placements.check(moved, n_pts=256, backend="xla")
    # tallymap's kernel and jax.vmap find the same pairs and protrusions; a point's neighbour that lands on a face may
    # round either side, so a pair's count may differ by one
    assert all(np.array_equal(first[k], c[k]) for k in ("pairs", "protrude", "cross", "n_points"))
    assert np.max(np.abs(first["counts"] - c["counts"])) <= 1, (first["counts"], c["counts"])
    assert ("a", "b") in {(moved.defs[a], moved.defs[b]) for a, b in c2["pairs"]}, c2["pairs"]
    apart = placements.check(nav_of(scene_overlap(b_x=0.35)), n_pts=256)  # b off a: no pair
    assert not len(apart["pairs"]) and len(apart["protrude"]) == 1, apart


def test_check_passes_the_pieces_of_one_split_union():
    """split_unions' pieces of one union may overlap (the first placed owns, one label): the check reports no pair of
    pieces of one placement"""
    nav = nav_of(lowering.split_unions(scene_split(), min_leaves=8))
    pcs = [i for i, g in enumerate(nav.pieces) if g >= 0]
    assert len(pcs) == 20, nav.defs
    c = placements.check(nav, n_pts=200)
    both = [tuple(p) for p in c["pairs"] if p[0] in pcs and p[1] in pcs]
    assert not both, both[:5]
    # the same union placed twice, overlapping itself: the pieces of one placement pass, the two placements overlap
    # (equal volumes: the first placed owns)
    sc = scene_split()
    sc.definitions["world"].placements.append(scene.Placement("frame", R=np.eye(3), t=np.array([-0.29, 0.0, 0.0])))
    nav2 = nav_of(lowering.split_unions(sc, min_leaves=8))
    inst = {i: g for i, g in enumerate(nav2.pieces) if g >= 0}
    pr = [tuple(p) for p in placements.check(nav2, n_pts=200)["pairs"] if p[0] in inst and p[1] in inst]
    assert pr and all(inst[a] != inst[b] for a, b in pr), (pr[:5], len(inst))


def test_check_meshes_reports_a_pass_through(toys):
    """an opaque mesh through a box (its sibling): the mesh check reports the pass-through"""
    sc = scene_mesh()
    sc.definitions["world"].placements[1].t = np.array([0.25, 0.1, 0.0])  # the box onto the tetrahedron
    nav = nav_of(sc)
    m = placements.check_meshes(nav, n_pts=64)
    assert any(nav.defs[v] == "box" for (_, v) in m["pairs"]), m
    assert placements.check_meshes(toys("nest")[1])["n_points"] == 0


def test_convert_records_the_placement_check_and_warns_on_an_overlap(tmp_path, recwarn):
    """The conversion reports Geant4's overlap precondition: a clean toy says so, silently; two PMTs placed into each
    other are named in the report with the owner of the region they share, with a PlacementWarning; the geometry is
    kept as it is."""
    clean = lumen.load(lumen.convert(write(tmp_path, _toy()), RECIPE, tmp_path / "clean.lumen"))
    assert not [w for w in recwarn if issubclass(w.category, placements.PlacementWarning)]
    rep = clean.file.report
    assert any("no overlapping siblings, no protruding daughters" in line for line in rep), rep
    toy = _toy().replace('<position name="q" x="-500" y="0" z="0"/>', '<position name="q" x="60" y="0" z="0"/>')
    with pytest.warns(placements.PlacementWarning, match="overlapping siblings"):
        path = lumen.convert(write(tmp_path, toy, "overlap.gdml"), RECIPE, tmp_path / "overlap.lumen")
    rep = lumen.load(path).file.report
    want = "overlapping siblings, the shared region owned by one of them (pairs): 1 families, 1 in all: "
    assert any(want + "{PmtLV owned by PmtLV: 1}" in line for line in rep), rep


def test_a_lowered_hole_and_its_filler_are_no_overlap():
    """A slab whose hole the lowering dropped (Definition.holed) and the cube filling it share a region in the scene,
    not in the detector file: the check reports no pair, so the report and the export (lumen.compare.ownership.found,
    the same pairs) rewrite nothing; the same two solids as a plain overlap are reported."""
    from nav_toys import defn, make_scene

    from lumen.compare import ownership
    from lumen.geometry import navigate, solids

    def at0(name, solid, med, pls=()):
        return defn(name, solid, med, [(p, np.eye(3), np.zeros(3)) for p in pls])

    for holed in (True, False):
        defs = {
            "world": at0("world", solids.g4_box(4.0, 4.0, 4.0), 0, ["S", "F"]),
            "S": at0("S", solids.g4_box(0.2, 3.0, 3.0), 1),
            "F": at0("F", solids.g4_box(0.4, 0.4, 0.4), 2),
        }
        defs["S"].holed = holed
        nav = navigate.Navigator.from_scene(make_scene(defs, outside=3, hole_fillers=["F"] if holed else []))
        c = placements.check(nav, n_pts=256)
        assert (len(c["pairs"]) == 0) == holed, (holed, c["pairs"])
        assert ownership.found(nav)[0] == {frozenset(map(int, p)) for p in c["pairs"]}
        assert ("1 sibling pairs" in placements.report(nav, c)[0]) != holed
