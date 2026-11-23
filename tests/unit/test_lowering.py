"""The lowering on a small GDML (scene_fixture.py): Geant4's placement conventions (passive physvol rotation, active
boolean rotation, reflection by a negative scale, assemblies), sibling holes, the sensor rule
(one PATH id per world instance), the detector subtree, fail-closed unsupported solids, and the exact union split."""

import math

import numpy as np
import pytest
import scene_fixture as fixture

from lumen.build import lowering
from lumen.compare import oracle
from lumen.format import schema
from lumen.format.sensorsets import SensorGroup, SensorSet
from lumen.geometry import scene, solids

CATH = SensorSet((SensorGroup("cath", rule="any_entry", lv="Cath"),)).is_sensor


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


def test_a_scale_places_as_geant4_places_it():
    """G4ReflectionFactory::Place keeps only rotation and reflection: a positive scale's magnitude is dropped (the
    matrix stays orthogonal, so the navigator's R^T is its inverse); a reflecting non-unit scale aborts Geant4."""
    m, _ = lowering.placement_pose(_placement(rotation=(0.3, 0.0, 0.0), scale=(2.0, 2.0, 2.0)))
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(m, lowering.placement_pose(_placement(rotation=(0.3, 0.0, 0.0)))[0])
    m, _ = lowering.placement_pose(_placement(scale=(1.0, 1.0, -1.0)))
    np.testing.assert_allclose(m, np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(lowering.ScaleRefused, match="GeomVol0002"):
        lowering.placement_pose(_placement(scale=(1.0, 1.0, -2.0)))


def _placement(rotation=(0.0, 0.0, 0.0), scale=None):
    return schema.Placement("pv", "V", "W", copy=None, position=(0.1, 0.2, 0.3), rotation=rotation, scale=scale)


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
    assert any(D.holed for D in L.scene.definitions.values())  # sibling holes left out of the solid
    assert any(D.member is not None and isinstance(D.member.solid, scene.Mesh) for D in L.scene.definitions.values())


def test_detector_subtree(tmp_path):
    for spec in ("Tank", "auto:water"):
        L = fixture.load(tmp_path, detector=spec)
        assert L.scene.outside.volume == "World"  # kept only the detector volume Tank
        assert _names(L)[L.scene.outside.medium] == "Air"  # outside: the mother's material
        _, mats, _, _ = _labels(L, [(0, 0, 0), (0.95, 0.95, 0.95)])
        assert mats == ["Water", "Air"]  # beyond the detector: its mother's material
    det = lowering.find_detector(fixture.read(tmp_path), "auto:water")
    assert (det.lv, det.mother, det.placement) == ("Tank", "World", "tank_pv")
    with pytest.raises(ValueError, match="no volume, physvol or physvol path 'Tan'"):  # a name, not the start of one
        lowering.find_detector(fixture.read(tmp_path), "Tan")
    suffixed = fixture.read(tmp_path, fixture.GDML.replace('"Tank"', '"Tank0x1f"'), "sx.gdml")
    assert lowering.find_detector(suffixed, "Tank").lv == "Tank"  # stored without its suffix (canonical_names)


def test_unsupported_solid_fails_closed(tmp_path):
    txt = fixture.GDML.replace(
        '<box name="smallbox" x="50" y="50" z="50" lunit="mm"/>',
        '<twistedbox name="smallbox" PhiTwist="0.5" x="50" y="50" z="50" aunit="rad" lunit="mm"/>',
    )
    with pytest.raises(NotImplementedError, match="volume 'Small': solid 'smallbox' is a twistedbox.*replace it"):
        fixture.load(tmp_path, txt, "tw.gdml")


UNION = "".join(
    f'<union name="u{k}"><first ref="u{k - 1}"/><second ref="cube"/>'
    f'<position name="up{k}" x="{30 * k}" y="0" z="0" unit="mm"/></union>'
    for k in range(1, 18)
)


def test_split_unions_is_exact(tmp_path):
    """a union of 18 cubes minus a slab: placed as its 18 pieces (the subtraction distributed), the labels
    unchanged"""
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
    assert S is not L.scene
    assert S.definitions["Small"].member is None and len(S.definitions["Small"].placements) == 18
    assert lowering.split_unions(L.scene, 100) is L.scene  # nothing that large: unchanged
    X = np.random.default_rng(5).uniform([-0.05, -0.02, 0.48], [0.65, 0.02, 0.52], (20000, 3))
    m0, _, _ = oracle.Oracle(L.scene).label_many(X)
    m1, _, _ = oracle.Oracle(S).label_many(X)
    np.testing.assert_array_equal(m0, m1)
    assert np.any(m0 == L.media["Steel"]) and np.any(m0 == L.media["Water"])


def test_a_split_piece_keeps_only_the_holes_that_meet_it(tmp_path):
    """a union of 18 cubes minus a pin through each cube: each piece is its cube minus its own pin, the labels
    unchanged"""
    holes = "".join(
        f'<subtraction name="h{k}"><first ref="{f"h{k - 1}" if k else "u17"}"/><second ref="pin"/>'
        f'<position name="hp{k}" x="{30 * k}" y="0" z="0" unit="mm"/></subtraction>'
        for k in range(18)
    )
    txt = fixture.GDML.replace(
        '<box name="smallbox" x="50" y="50" z="50" lunit="mm"/>',
        '<box name="smallbox" x="50" y="50" z="50" lunit="mm"/><box name="u0" x="20" y="20" z="20" lunit="mm"/>'
        + UNION
        + '<box name="pin" x="4" y="4" z="40" lunit="mm"/>'
        + holes,
    )
    txt = txt.replace(
        '<volume name="Small"><materialref ref="Steel"/><solidref ref="smallbox"/></volume>',
        '<volume name="Small"><materialref ref="Steel"/><solidref ref="h17"/></volume>',
    )
    L = fixture.load(tmp_path, txt, "h.gdml")
    S = lowering.split_unions(L.scene, 16)
    pieces = S.definitions["Small"].placements
    assert len(pieces) == 18
    assert all(lowering.n_leaves(S.definitions[p.definition].member.solid) == 2 for p in pieces)
    X = np.random.default_rng(6).uniform([0.05, -0.012, 0.48], [0.65, 0.012, 0.52], (20000, 3))
    m0, _, _ = oracle.Oracle(L.scene).label_many(X)
    m1, _, _ = oracle.Oracle(S).label_many(X)
    np.testing.assert_array_equal(m0, m1)
    steel = m0 == L.media["Steel"]
    xr = (X[:, 0] - 0.1 + 0.015) % 0.03 - 0.015  # from the nearest cube's centre (Small at x = 0.1 m)
    near = (X[:, 0] > 0.09) & (X[:, 0] < 0.62) & (np.abs(X[:, 2] - 0.5) < 0.009)
    pin = near & (np.abs(xr) < 0.0019) & (np.abs(X[:, 1]) < 0.0019)
    cube = near & (np.abs(xr) > 0.0021) & (np.abs(xr) < 0.0099) & (np.abs(X[:, 1]) < 0.0099)
    assert np.any(pin) and not np.any(steel[pin])
    assert np.any(cube) and np.all(steel[cube])


def test_a_phi_cut_solid_has_its_sector_box():
    # a 0.1 rad slice of a 10 m polycone (as WCSim's extra tower): its box is the sector's, the full solid's is not
    node = solids.g4_polycone(-0.1, 0.1, [(0.0, 10.0, -1.0), (0.0, 10.0, 1.0)])
    lo, hi = solids.sector_bbox(node)
    assert lo[0] >= -1e-12 and hi[1] <= 1e-12 and lo[1] >= -10.0 * math.sqrt(2) * math.sin(0.1) - 1e-9
    assert hi[0] <= 10.0 + 1e-9 and (lo[2], hi[2]) == (-1.0, 1.0)
    full = solids.node_bbox(node)
    assert full[0][0] < -9.0 and full[0][1] < -9.0  # node_bbox stays the whole radius


def test_a_detector_volume_placed_twice_is_refused_and_a_physvol_selects_one():
    """An LV placed twice is no detector volume: never its first placement silently; the refusal names both
    placements and their common mother, which simulates both (each sensor numbered by its placement path). Its
    physvol name or path selects one."""
    from builder import Builder

    b = Builder("twins", detector_volume="Ball")
    b.material("Water", density=1.0, atoms={"H": 2, "O": 1}, state="liquid", RINDEX=1.33, ABSLENGTH=30.0)
    b.material("Black", density=1.0, atoms={"C": 1})
    b.box("WorldS", 6.0, 3.0, 3.0)
    b.box("HallS", 5.0, 2.5, 2.5)
    b.solid("BallS", "orb", r=1.0)
    b.volume("World", "WorldS", "Black")
    b.volume("Hall", "HallS", "Black")
    b.volume("Ball", "BallS", "Water")
    b.place("Hall", into="World", name="hall_pv")
    b.place("Ball", into="Hall", name="left_pv", position=(-1.2, 0.0, 0.0))
    b.place("Ball", into="Hall", name="right_pv", position=(1.2, 0.0, 0.0))
    df = b.structure()[0]
    with pytest.raises(ValueError, match=r"'Ball' is placed 2 times \(hall_pv/left_pv, hall_pv/right_pv\): .*"
                                         r"give their common mother, detector volume 'Hall'"):  # fmt: skip
        lowering.find_detector(df, "Ball")
    with pytest.raises(ValueError, match="common mother, detector volume 'Hall'"):  # auto: finds Ball, the same rule
        lowering.find_detector(df, "auto:water")
    both = lowering.lower(df, detector="Hall", sensors=lambda lv: lv == "Ball")  # every placement, numbered by path
    assert sorted(both.sensors.values()) == ["hall_pv/left_pv/Ball", "hall_pv/right_pv/Ball"]
    for spec in ("right_pv", "hall_pv/right_pv"):
        det = lowering.find_detector(df, spec)
        assert (det.lv, det.placement, det.mother) == ("Ball", "right_pv", "Hall")
        np.testing.assert_allclose(det.t, [1.2, 0.0, 0.0])
    with pytest.raises(ValueError, match="no volume, physvol or physvol path"):
        lowering.find_detector(df, "hall_pv/nowhere")


def test_the_common_mother_of_placements_under_a_repeated_mother():
    """The innermost volume holding every placement: the world when they sit in two placements of one mother; the
    physvol path of the repeated mother's placement when they sit in one of them."""
    from builder import Builder

    b = Builder("halls", detector_volume="World")
    b.material("Water", density=1.0, atoms={"H": 2, "O": 1}, state="liquid", RINDEX=1.33, ABSLENGTH=30.0)
    b.box("WorldS", 9.0, 3.0, 3.0)
    b.box("HallS", 4.0, 2.5, 2.5)
    b.solid("BallS", "orb", r=0.5)
    b.volume("World", "WorldS", "Water")
    b.volume("Hall", "HallS", "Water")
    b.volume("Ball", "BallS", "Water")
    b.place("Hall", into="World", name="hall_a", position=(-4.5, 0.0, 0.0))
    b.place("Hall", into="World", name="hall_b", position=(4.5, 0.0, 0.0))
    b.place("Ball", into="Hall", name="left_pv", position=(-1.0, 0.0, 0.0))
    b.place("Ball", into="Hall", name="right_pv", position=(1.0, 0.0, 0.0))
    df = b.structure()[0]
    with pytest.raises(ValueError, match="placed 4 times .*detector volume 'World'; or give the physvol"):
        lowering.find_detector(df, "Ball")
    b.place("Ball", into="Hall", name="right_pv", position=(0.0, 1.0, 0.0))  # two placements of one name in each hall
    alike = b.structure()[0]
    alike_paths = r"\(hall_a/left_pv, hall_a/right_pv, hall_b/left_pv, hall_b/right_pv\)"
    with pytest.raises(
        ValueError, match=f"placed 6 times {alike_paths}.*'World'; their physvol paths are not distinct"
    ):
        lowering.find_detector(alike, "Ball")
    with pytest.raises(ValueError, match="placed 2 times .*common mother, detector volume 'World'"):
        lowering.find_detector(df, "right_pv")
    chains = [c for c in schema.placements_of(df, "Ball") if c[0] == schema.placements_of(df, "hall_a")[0][0]]
    assert lowering.common_mother(df, chains) == "hall_a"
