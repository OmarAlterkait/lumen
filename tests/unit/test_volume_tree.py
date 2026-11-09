"""The volume tree of a Scene (lumen.build.tree) on small hand-built Scenes: one PV per world instance, mothers through
assemblies, sensor ids in the canonical order, unique names, fail-closed checks; relabelling by region (clones only
where instances differ). Ported from the tree parts of the baseline's tests/test_adapters_geometry.py."""

import numpy as np
import pytest

from lumen.build import scene, solids, tree

MEDIA = {"Air": 1, "LAr": 2, "Steel": 3}


def _scene():
    """outside Air > world box (LAr) > two LAr disk sensors (PATH ids) and an assembly holding a steel block."""
    disk = scene.Definition(
        "Disk", members=[scene.Member(solids.g4_tube(0.0, 0.1, 0.002), scene.Label(2, scene.PATH, "Disk"), 0)]
    )
    block = scene.Definition("Block", members=[scene.Member(solids.g4_box(0.1, 0.1, 0.1), scene.Label(3, -1, "Block"))])
    asm = scene.Definition("asm", placements=[scene.Placement("Block", t=np.array([0.0, 0.5, 0.0]), name="b")])
    world = scene.Definition(
        "World",
        members=[scene.Member(solids.g4_box(4.0, 4.0, 4.0), scene.Label(2, -1, "World"))],
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
    t = tree.tree_from_geometry(_scene(), MEDIA)
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
    inst = {x.pv: x for x in tree.instances(_scene())}
    np.testing.assert_allclose(inst["world/a/b/Block"].t, [0.0, 0.5, 0.0])


def test_tree_names_and_fail_closed():
    g = _scene()
    g.definitions["World"].placements[2].name = "d0"  # duplicate sibling names: made unique
    names = {p.name for p in tree.tree_from_geometry(g, MEDIA).pvs}
    assert {"world/d0#0/Disk", "world/d0#2/Disk"} <= names
    with pytest.raises(tree.TreeError, match="no material name"):
        tree.tree_from_geometry(_scene(), {"Air": 1, "LAr": 2})
    with pytest.raises(tree.TreeError, match="share medium id"):
        tree.tree_from_geometry(_scene(), {**MEDIA, "LAr2": 2})
    with pytest.raises(tree.TreeError, match="needs media"):
        tree.tree_from_geometry(_scene())
    g = _scene()
    g.definitions["World"].members.append(scene.Member(solids.g4_orb(0.01), scene.Label(3, -1, "Ball"), 1))
    with pytest.raises(tree.TreeError, match="ambiguous"):
        tree.tree_from_geometry(g, MEDIA)
    g.definitions["World"].placements.clear()  # several members, no daughters: one PV each
    t = tree.tree_from_geometry(g, MEDIA)
    assert {p.lv for p in t.pvs} == {tree.OUTSIDE_PV, "World.World", "World.Ball"}


def test_tree_record_checks():
    with pytest.raises(tree.TreeError, match="duplicate"):
        tree.Tree((tree.PV("w", "W", "Air"), tree.PV("w", "W", "Air")))
    with pytest.raises(tree.TreeError, match="exactly one world"):
        tree.Tree((tree.PV("a", "A", "Air"), tree.PV("b", "B", "Air")))
    with pytest.raises(tree.TreeError, match="materials"):
        tree.Tree((tree.PV("w", "W", "Air"), tree.PV("x", "X", "LAr", "w"), tree.PV("y", "X", "Air", "w")))


def _classes(**over):
    c = {tree.OUTSIDE_PV: 0, "world/World": 1, "world/d0/Disk": 2, "world/a/b/Block": 4, "world/d1/Disk": 2}
    c.update(over)
    return c


def _labels(g):
    """instance path -> member label (medium, sensor), by walking the relabelled Scene."""
    return {
        x.pv: (
            g.definitions[x.definition].members[x.member].label.medium,
            g.definitions[x.definition].members[x.member].label.sensor,
        )
        for x in tree.instances(g)
    }


def test_relabel_shares_definitions_when_classes_agree():
    s = _scene()
    g = tree.relabel(s, _classes())
    assert sorted(g.definitions) == ["Block", "Disk", "World", "__root__", "asm"]
    assert _labels(g) == {
        "world/World": (1, -1),
        "world/d0/Disk": (2, scene.PATH),
        "world/a/b/Block": (4, -1),
        "world/d1/Disk": (2, scene.PATH),
    }
    assert g.outside.medium == 0 and g.outside.volume == "Hall"
    assert g.definitions["Disk"].members[0].solid is s.definitions["Disk"].members[0].solid  # solids shared
    assert s.definitions["Disk"].members[0].label.medium == 2  # input untouched


def test_relabel_refuses_a_class_shared_with_the_outside():
    with pytest.raises(tree.TreeError, match="outside"):
        tree.relabel(_scene(), _classes(**{"world/a/b/Block": 0}))


def test_relabel_clones_where_instances_differ():
    g = tree.relabel(_scene(), _classes(**{"world/d1/Disk": 3}))
    assert sorted(g.definitions) == ["Block", "Disk|class0", "Disk|class1", "World", "__root__", "asm"]
    assert [p.definition for p in g.definitions["World"].placements] == ["Disk|class0", "asm", "Disk|class1"]
    assert _labels(g)["world/d1/Disk"] == (3, scene.PATH) and _labels(g)["world/d0/Disk"] == (2, scene.PATH)
    # a difference deep inside a repeated parent clones the parent chain
    s = _scene()
    s.definitions["World"].placements.append(scene.Placement("asm", t=np.array([0.0, -1.0, 0.0]), name="a2"))
    g = tree.relabel(s, _classes(**{"world/a2/b/Block": 5}))  # (the default Block class is 4)
    assert {"asm|class0", "asm|class1", "Block|class0", "Block|class1"} <= set(g.definitions)
    assert _labels(g)["world/a2/b/Block"] == (5, -1) and _labels(g)["world/a/b/Block"] == (4, -1)


def test_relabel_fails_closed():
    with pytest.raises(tree.TreeError, match="misses"):
        tree.relabel(_scene(), {k: v for k, v in _classes().items() if k != "world/d1/Disk"})
    with pytest.raises(tree.TreeError, match="does not have"):
        tree.relabel(_scene(), _classes(extra=1))
    for bad in (-1, 1.0, True):
        with pytest.raises(tree.TreeError, match="non-negative integers"):
            tree.relabel(_scene(), _classes(**{"world/World": bad}))
