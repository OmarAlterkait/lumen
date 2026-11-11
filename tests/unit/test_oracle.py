"""The float64 oracle: the batched point location equals the per-point one, a daughter is clipped to its mother, the
query's first visible change (the thin-layer contract), an opaque mesh, and the torus roots."""

import helpers
import numpy as np
import scene_fixture as fixture

from lumen.build import scene, solids
from lumen.validate import oracle

T_MIN = 1e-7


def test_label_many_equals_label(tmp_path):
    orc = oracle.Oracle(fixture.load(tmp_path, dissolve=True).scene)
    X = np.random.default_rng(1).uniform(-0.85, 0.85, (3000, 3))
    m, s, _ = orc.label_many(X)
    for k in range(0, len(X), 300):
        m2, s2 = orc.label(X[k : k + 300])
        np.testing.assert_array_equal(m[k : k + 300], m2)
        np.testing.assert_array_equal(s[k : k + 300], s2)


def test_daughter_clipped_to_its_mother():
    """a daughter protruding from its mother is clipped (Geant4's rule): outside the mother it holds no point"""
    defs = {
        "root": scene.Definition(
            "root",
            members=[scene.Member(solids.g4_box(1.0, 1.0, 1.0), scene.Label(1), 0)],
            placements=[scene.Placement("m", np.eye(3), np.zeros(3))],
        ),
        "m": scene.Definition(
            "m",
            members=[scene.Member(solids.g4_orb(0.1), scene.Label(2), 0)],
            placements=[scene.Placement("d", np.eye(3), np.array([0.09, 0.0, 0.0]))],
        ),
        "d": scene.Definition("d", members=[scene.Member(solids.g4_orb(0.05), scene.Label(3), 0)]),
    }
    orc = oracle.Oracle(scene.Scene(defs, "root", scene.Label(0)))
    med, _, _ = orc.label_many(np.array([[0.09, 0, 0], [0.13, 0, 0], [0.3, 0, 0]]))
    assert med.tolist() == [3, 1, 1]  # (0.13: inside the daughter, outside its mother)


def test_oracle_thin_layer_contract():
    """1 um glass over a cathode ball: the first visible change is the glass entry with the cathode after (the push
    point); with push 0 the glass is the medium after"""
    defs = {
        "root": scene.Definition(
            "root",
            members=[
                scene.Member(solids.g4_orb(1.0), scene.Label(1, -1, "water"), 1),
                scene.Member(solids.g4_orb(0.500001), scene.Label(2, -1, "glass"), 2),
                scene.Member(solids.g4_orb(0.5), scene.Label(3, 0, "cathode"), 3),
            ],
        )
    }
    orc = oracle.Oracle(scene.Scene(defs, "root", scene.Label(0, -1, "outside")))
    cos_in = np.array([1.0, 0.8, 0.3, 0.2])
    b = 0.5 * np.sqrt(1 - cos_in**2)
    origins = np.stack([np.full_like(b, -0.8), b, np.zeros_like(b)], 1)
    dirs = np.tile([1.0, 0, 0], (len(b), 1))
    for push, after in ((1e-4, 3), (0.0, 2)):
        for i in range(len(b)):
            r = orc.query(origins[i], dirs[i], T_MIN, 1e3, push=push)
            assert r["hit"] and r["before"][0] == 1 and r["after"][0] == after, (push, i, r)
            t_glass = 0.8 - np.sqrt(0.500001**2 - b[i] ** 2)
            assert abs(r["t"] - t_glass) < 1e-9
    med, sen = orc.label(np.array([[0, 0, 0], [0.5000005, 0, 0], [0.9, 0, 0], [2, 0, 0]]))
    assert med.tolist() == [3, 2, 1, 0] and sen.tolist() == [0, -1, -1, -1]
    rec = helpers.oracle_first_visible(orc, origins, dirs, 1e-4)
    assert rec["hit"].all() and (rec["after"] == 3).all() and (rec["start"] == "ok").all()
    assert not helpers.mismatch(rec, rec, push=1e-4).any()


def test_query_through_the_fixture(tmp_path):
    L = fixture.load(tmp_path, dissolve=True)
    orc = oracle.Oracle(L.scene)
    nm = {v: k for k, v in L.media.items()}
    r = orc.query(np.zeros(3), np.array([1.0, 0, 0]), T_MIN, 10.0, push=1e-5)
    assert r["hit"] and nm[r["before"][0]] == "Water" and nm[r["after"][0]] == "Glass"
    assert abs(r["t"] - (0.5 - np.sqrt(0.1**2 - 0.05**2))) < 1e-12  # the PMT glass (the envelope: invisible)
    r = orc.query(np.zeros(3), np.array([1.0, 0, 0]), T_MIN, 10.0, push=1e-5, medium=L.media["Glass"])
    assert r["start"] == "off"  # a start medium that is not the label there is reported


def test_opaque_mesh_is_a_first_hit_and_holds_no_point(tmp_path):
    L = fixture.load(tmp_path)
    orc = oracle.Oracle(L.scene)
    nm = {v: k for k, v in L.media.items()}
    c = np.array([-0.3, -0.3, 0.3])  # the tetrahedron's corner at the origin of its frame
    o = c + np.array([0.02, 0.02, -0.1])
    r = orc.query(o, np.array([0.0, 0.0, 1.0]), T_MIN, 1.0)
    assert r["hit"] and nm[r["after"][0]] == "Steel" and abs(r["t"] - 0.1) < 1e-12
    med, _, _ = orc.label_many((c + 0.01)[None])  # inside the mesh: the region it sits in
    assert nm[int(med[0])] == "Water"


def test_torus_roots_and_locate():
    """a torus (R 0.3, r 0.1) placed off the origin: the ray along x through its centre crosses it at four points"""
    defs = {
        "root": scene.Definition(
            "root",
            members=[scene.Member(solids.g4_box(4.0, 4.0, 4.0), scene.Label(1), 0)],
            placements=[scene.Placement("t", np.eye(3), np.array([0.5, 0.0, 0.0]), name="torus_pv")],
        ),
        "t": scene.Definition("t", members=[scene.Member(solids.g4_torus(0.0, 0.1, 0.3), scene.Label(2, -1, "T"))]),
    }
    orc = oracle.Oracle(scene.Scene(defs, "root", scene.Label(0)))
    roots = np.sort(orc.solids[orc.sol[1]].roots(np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])))
    np.testing.assert_allclose(roots, [0.6, 0.8, 1.2, 1.4], atol=1e-12)
    inst = orc.locate(np.array([[0.2, 0, 0], [0.5, 0, 0], [5.0, 0, 0]]))
    assert orc.inst_volume[inst].tolist() == ["T", "", ""] and orc.inst_placement[inst[0]] == "torus_pv"
    assert inst[2] == -1  # outside everything: the outside's entry
