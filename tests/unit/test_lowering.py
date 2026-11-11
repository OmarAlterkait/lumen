"""The lowering on a small GDML (scene_fixture.py): Geant4's placement conventions (passive physvol rotation, active
boolean rotation, reflection by a negative scale, assemblies), the envelope dissolve, sibling holes, the sensor rule
(one PATH id per world instance), the detector subtree, fail-closed unsupported solids, and the exact union split."""

import math

import helpers
import numpy as np
import pytest
import scene_fixture as fixture

from lumen.build import lowering, scene, solids
from lumen.format import schema
from lumen.format.sensorsets import SensorGroup, SensorSet
from lumen.validate import oracle

CATH = SensorSet("default", (SensorGroup("cath", "any_entry", "lumen:counting", lv="Cath"),)).is_sensor


def _names(L):
    return {v: k for k, v in L.media.items()}


def _labels(L, X):
    orc = oracle.Oracle(L.scene)
    med, sen, inst = orc.label_many(np.asarray(X, float))
    nm = _names(L)
    return orc, [nm.get(int(m), "OUTSIDE") for m in med], sen, inst


def test_rotation_is_the_physvol_matrix():
    x, y, z = 0.3, -0.7, 1.1
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    np.testing.assert_allclose(lowering.rot_from_angles(x, y, z), (Rz @ Ry @ Rx).T, atol=1e-15)
    assert lowering.placement_pose(_placement())[0] is lowering.IDENT  # no rotation: the identity
    m, t = lowering.placement_pose(_placement(scale=(-1.0, 1.0, 1.0)))
    assert np.linalg.det(m) < 0 and t.tolist() == [0.1, 0.2, 0.3]


def _placement(rotation=(0.0, 0.0, 0.0), scale=None):
    return schema.Placement("pv", "V", "W", None, (0.1, 0.2, 0.3), rotation, scale)


def test_geant4_conventions_and_placements(tmp_path):
    L = fixture.load(tmp_path, sensors=CATH)
    X = [p for p, _, _ in fixture.POINTS]
    orc, mats, sen, inst = _labels(L, X)
    for (p, m, pv), got, i in zip(fixture.POINTS, mats, inst, strict=True):
        assert got == (m if pv is not None else "OUTSIDE"), (p, m, got)
        if pv is not None:
            assert orc.inst_placement[i] == pv, (p, pv, orc.inst_placement[i])
    assert L.scene.outside.medium == 0  # the world solid is the outside
    assert sorted(L.sensors.values()) == ["tank_pv/env2_pv/pmt_pv/cath_pv/Cath", "tank_pv/env_pv/pmt_pv/cath_pv/Cath"]
    assert sen[1] != sen[3] and {int(sen[1]), int(sen[3])} == {0, 1}  # one id per world instance
    assert any("holes are sibling placements" in a for a in L.scene.approximations)
    assert any(isinstance(m.solid, scene.Mesh) for D in L.scene.definitions.values() for m in D.members)
    assert L.report[0].startswith("gdml scene.gdml: ")  # the file name, no site path


def test_dissolve_is_exact_and_reported(tmp_path):
    L0 = fixture.load(tmp_path)
    L1 = fixture.load(tmp_path, dissolve=True, sensors=CATH)
    assert any("dissolved 2 placements of the envelope Env" in a for a in L1.scene.approximations)
    tank = L1.scene.definitions["Tank"]
    assert not any(pl.definition == "Env" for pl in tank.placements)
    pmt = [pl for pl in tank.placements if pl.definition == "Pmt"]
    np.testing.assert_allclose(sorted(pl.t[1] for pl in pmt), [0.0, 0.45])  # poses composed
    np.testing.assert_allclose([pl.t[2] for pl in pmt], [0.05, 0.05])
    X = np.random.default_rng(0).uniform(-0.8, 0.8, (20000, 3))
    _, m0, _, _ = _labels(L0, X)
    _, m1, _, _ = _labels(L1, X)
    assert m0 == m1  # the same regions, by name


def test_detector_subtree(tmp_path):
    for spec in ("Tank", "auto:water"):
        L = fixture.load(tmp_path, detector=spec)
        assert "kept only the detector volume Tank" in L.report[1]
        assert _names(L)[L.scene.outside.medium] == "Air"  # outside: the mother's material
        _, mats, _, _ = _labels(L, [(0, 0, 0), (0.95, 0.95, 0.95)])
        assert mats == ["Water", "Air"]  # beyond the detector: its mother's material
    det = lowering.find_detector(fixture.load(tmp_path).detector_file, "auto:water")
    assert (det.lv, det.mother, det.placement) == ("Tank", "World", "tank_pv")


def test_unsupported_solid_fails_closed(tmp_path):
    txt = fixture.GDML.replace(
        '<box name="smallbox" x="50" y="50" z="50" lunit="mm"/>',
        '<twistedbox name="smallbox" PhiTwist="0.5" x="50" y="50" z="50" aunit="rad" lunit="mm"/>',
    )
    with pytest.raises(NotImplementedError, match="twistedbox"):
        fixture.load(tmp_path, txt, "tw.gdml")
    L = fixture.load(tmp_path, txt, "tw.gdml", unsupported="assembly")
    assert any("unsupported, LV kept as a transparent assembly: twistedbox x1" in a for a in L.scene.approximations)
    with pytest.raises(ValueError, match="unsupported="):
        fixture.load(tmp_path, unsupported="ignore")


def test_scene_scale(tmp_path):
    L = fixture.load(tmp_path, detector="Tank")
    assert lowering.scene_scale(L.scene) == pytest.approx(1.6)


def test_check_partition_flags_equal_rank_overlaps():
    D = scene.Definition(
        "d",
        members=[
            scene.Member(solids.g4_orb(0.1), scene.Label(1, -1, "a"), 0),
            scene.Member(solids.g4_orb(0.06), scene.Label(2, -1, "b"), 0),
            scene.Member(solids.g4_box(0.02, 0.02, 0.02), scene.Label(3, -1, "c"), 1),
        ],
    )
    out = helpers.check_partition(scene.Scene({"d": D}, "d", scene.Label(0)), n=4000)
    amb, ranked = out["d"]
    assert amb > 0 and ranked >= amb


def test_envelope_with_a_protruding_daughter_is_kept(tmp_path):
    """Geant4 never sees the part of a daughter outside its mother: lifting a protruding daughter out of its envelope
    would change the detector (MicroBooNE's wires out of their plane), so the envelope stays, and the report says so"""
    txt = fixture.GDML.replace('<orb name="pmtorb" r="100" lunit="mm"/>', '<orb name="pmtorb" r="190" lunit="mm"/>')
    L = fixture.load(tmp_path, txt, "p.gdml", dissolve=True)
    assert any("envelope Env kept (x2)" in a for a in L.scene.approximations)
    assert any(pl.definition == "Env" for pl in L.scene.definitions["Tank"].placements)
    _, m0, _, _ = _labels(fixture.load(tmp_path, txt, "p.gdml"), [(0.5, 0.0, 0.22)])
    _, m1, _, _ = _labels(L, [(0.5, 0.0, 0.22)])
    assert m0 == m1 == ["Water"]  # the glass beyond the envelope is not there


def test_envelope_overlapping_a_sibling_is_kept(tmp_path):
    """where siblings overlap, Geant4's locate picks the later one; dissolving it would expose the earlier one
    (MicroBooNE's volTPCActive over its wire planes), so an envelope not proven disjoint from its siblings stays"""
    txt = fixture.GDML.replace(
        '<position name="e2" x="500" y="450" z="0" unit="mm"/>', '<position name="e2" x="500" y="150" z="0" unit="mm"/>'
    )
    L = fixture.load(tmp_path, txt, "o.gdml", dissolve=True)
    assert any("envelope Env kept (x2): not proven disjoint from its siblings" in a for a in L.scene.approximations)
    X = np.random.default_rng(3).uniform([0.25, -0.25, -0.25], [0.75, 0.4, 0.25], (20000, 3))
    _, m0, _, _ = _labels(fixture.load(tmp_path, txt, "o.gdml"), X)
    _, m1, _, _ = _labels(L, X)
    assert m0 == m1


UNION = "".join(
    f'<union name="u{k}"><first ref="u{k - 1}"/><second ref="cube"/>'
    f'<position name="up{k}" x="{30 * k}" y="0" z="0" unit="mm"/></union>'
    for k in range(1, 18)
)


def test_split_unions_is_exact(tmp_path):
    """a union of 18 cubes minus a slab: placed as its 18 pieces (the subtraction distributed), the labels unchanged"""
    txt = fixture.GDML.replace(
        '<box name="smallbox" x="50" y="50" z="50" lunit="mm"/>',
        '<box name="smallbox" x="50" y="50" z="50" lunit="mm"/><box name="u0" x="20" y="20" z="20" lunit="mm"/>'
        + UNION
        + '<box name="slab" x="1000" y="1000" z="4" lunit="mm"/>'
        '<subtraction name="rail"><first ref="u17"/><second ref="slab"/></subtraction>',
    )
    txt = txt.replace(
        '<volume name="Small"><materialref ref="Steel"/><solidref ref="smallbox"/></volume>',
        '<volume name="Small"><materialref ref="Steel"/><solidref ref="rail"/></volume>',
    )
    L = fixture.load(tmp_path, txt, "u.gdml")
    S = lowering.split_unions(L.scene, 16)
    assert S is not L.scene and any("18 pieces" in a for a in S.approximations)
    assert not S.definitions["Small"].members and len(S.definitions["Small"].placements) == 18
    assert lowering.split_unions(L.scene, 100) is L.scene  # nothing that large: unchanged
    X = np.random.default_rng(5).uniform([-0.05, -0.02, 0.48], [0.65, 0.02, 0.52], (20000, 3))
    m0, _, _ = oracle.Oracle(L.scene).label_many(X)
    m1, _, _ = oracle.Oracle(S).label_many(X)
    np.testing.assert_array_equal(m0, m1)
    assert np.any(m0 == L.media["Steel"]) and np.any(m0 == L.media["Water"])


def test_place_composes_a_part(tmp_path):
    base = fixture.load(tmp_path, detector="Tank")
    part = fixture.load(tmp_path, part=True)
    S = lowering.place(base.scene, part.scene, "__detector__", pose=(np.eye(3), np.array([5.0, 0, 0])), prefix="p")
    assert "p/World" in S.definitions and S.definitions["__detector__"].placements[-1].name == "p"
    m, _, _ = oracle.Oracle(S).label_many(np.array([[5.0, 0, 0], [5.95, 0, 0], [0, 0, 0]]))
    # the part keeps its own medium ids and its world solid (Air beyond its tank)
    assert m.tolist() == [part.media["Water"], part.media["Air"], base.media["Water"]]
