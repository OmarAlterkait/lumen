"""The volume tree of a Scene (lumen.build.tree) on small hand-built Scenes: one PV per world instance, mothers through
assemblies, sensor ids in the canonical order, unique names, fail-closed checks."""

import numpy as np
import pytest

from lumen.build import tree
from lumen.geometry import scene, solids

MEDIA = {"Air": 1, "LAr": 2, "Steel": 3}


def _scene():
    """outside Air > world box (LAr) > two LAr disk sensors (PATH ids) and an assembly holding a steel block."""
    disk = scene.Definition(
        "Disk", member=scene.Member(solids.g4_tube(0.0, 0.1, 0.002), scene.Label(2, scene.PATH, "Disk"))
    )
    block = scene.Definition("Block", member=scene.Member(solids.g4_box(0.1, 0.1, 0.1), scene.Label(3, -1, "Block")))
    asm = scene.Definition("asm", placements=[scene.Placement("Block", t=np.array([0.0, 0.5, 0.0]), name="b")])
    world = scene.Definition(
        "World",
        member=scene.Member(solids.g4_box(4.0, 4.0, 4.0), scene.Label(2, -1, "World")),
        placements=[
            scene.Placement("Disk", t=np.array([0.0, 0.0, -0.8]), name="d0"),
            scene.Placement("asm", name="a"),
            scene.Placement("Disk", t=np.array([0.0, 0.0, 0.8]), name="d1"),
        ],
    )
    root = scene.Definition("__root__", placements=[scene.Placement("World", name="world")])
    return scene.Scene(
        {"Disk": disk, "Block": block, "asm": asm, "World": world, "__root__": root},
        "__root__",
        scene.Label(1, -1, "Hall"),
    )


def test_tree_one_pv_per_world_instance():
    t, insts = tree.tree_from_geometry(_scene(), MEDIA)
    want = {
        tree.OUTSIDE_PV: (tree.OUTSIDE_PV, tree.OUTSIDE_MATERIAL, None, None),  # always its own material
        "world/World": ("World", "LAr", tree.OUTSIDE_PV, None),
        "world/d0/Disk": ("Disk", "LAr", "world/World", 0),
        "world/a/b/Block": ("Block", "Steel", "world/World", None),  # the assembly is no volume
        "world/d1/Disk": ("Disk", "LAr", "world/World", 1),
    }
    assert {p.name: (p.lv, p.material, p.mother, p.copy) for p in t.pvs} == want
    assert t.placements("Disk") == ("world/d0/Disk", "world/d1/Disk")
    assert t.order[0] == tree.OUTSIDE_PV and t.root == tree.OUTSIDE_PV
    inst = {x.pv: x for x in insts}
    np.testing.assert_allclose(inst["world/a/b/Block"].t, [0.0, 0.5, 0.0])


def test_sensor_ids_are_the_tree_s_pv_names_when_siblings_share_a_name():
    """two sibling placements of one name (LArSoft's "<lv>_PV"): each sensor id is its own path, the tree's PV name"""
    from lumen.build import lowering
    from lumen.format.sensorsets import SensorGroup, SensorSet

    g = _scene()
    for pl in g.definitions["World"].placements:
        pl.name = "Disk_PV" if pl.definition == "Disk" else pl.name
    g.definitions["Disk"].member.label.sensor = -1  # assign_sensors marks them
    marked, ids = lowering.assign_sensors(g, lambda lv: lv == "Disk")
    assert list(ids.values()) == ["world/Disk_PV#0/Disk", "world/Disk_PV#2/Disk"]
    t, _ = tree.tree_from_geometry(marked, MEDIA)
    assert [t.by_name[p].copy for p in ids.values()] == [0, 1]  # the id is the tree's sensor index of that PV
    SensorSet((SensorGroup("d", rule="any_entry", lv="Disk"),), tuple(ids.values()))
    with pytest.raises(ValueError, match="names 'a' twice"):
        SensorSet((SensorGroup("d", rule="any_entry", lv="Disk"),), ("a", "b", "a"))


def test_tree_names_and_fail_closed():
    g = _scene()
    g.definitions["World"].placements[2].name = "d0"  # duplicate sibling names: made unique
    names = {p.name for p in tree.tree_from_geometry(g, MEDIA)[0].pvs}
    assert {"world/d0#0/Disk", "world/d0#2/Disk"} <= names
    with pytest.raises(tree.TreeError, match="no material name"):
        tree.tree_from_geometry(_scene(), {"Air": 1, "LAr": 2})
    with pytest.raises(tree.TreeError, match="share medium id"):
        tree.tree_from_geometry(_scene(), {**MEDIA, "LAr2": 2})


def test_tree_record_checks():
    with pytest.raises(tree.TreeError, match="duplicate"):
        tree.Tree((tree.PV("w", "W", "Air"), tree.PV("w", "W", "Air")))
    with pytest.raises(tree.TreeError, match="exactly one world"):
        tree.Tree((tree.PV("a", "A", "Air"), tree.PV("b", "B", "Air")))
    with pytest.raises(tree.TreeError, match="materials"):
        tree.Tree((tree.PV("w", "W", "Air"), tree.PV("x", "X", "LAr", "w"), tree.PV("y", "X", "Air", "w")))
