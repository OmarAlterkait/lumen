"""The volume tree of a Scene: one physical volume per world instance of a member, and the Scene relabelled by region.

The PV names are the ones every consumer shares (the optics' regions, the navigator's volumes); the relabelled Scene
gives the float64 oracle its labels.

    tree = tree_from_geometry(loaded)        # Tree: the outside PV, then one PV per member world instance
    insts = instances(loaded)                # Instance records in the canonical order
    scene2 = relabel(loaded.scene, class_of) # member labels replaced by region ids

World instances, in the canonical order (from the root, a definition's members by rank, then its placements by index):

- a definition instance with one member is one PV, named by its placement path and the member's volume name (the
  loader's sensor path, ``Loaded.sensors``); its LV is the member's volume name (the GDML LV, also for the lowering's
  ``|mirror`` variants), else the definition name. With several members, one PV each (LV ``<definition>.<volume>``),
  and then the definition may not have placements (their mother volume would be ambiguous: fail closed);
- a member-less definition (an assembly, the lowering's root wrapper) is no PV: its daughters' mother is the enclosing
  PV;
- the outside region (``Scene.outside``) is the world PV ``OUTSIDE_PV``;
- ``PV.copy`` is the instance's sensor id (``scene.PATH`` numbered per world instance, ``scene.PLACEMENT``
  inherited, or explicit), None for a non-sensor.

Notes
-----
Ported from lumen.adapters.geometry and the PV / Tree records of lumen.describe (rebuild/next ec8de54c). The LUCiD
model lookup is gone (lumen reads only its own Scene), as are sensor_tags and pass_pairs (they read the optics).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from lumen.build import scene as sc

OUTSIDE_PV = "<outside>"
# The outside's material is always its own (a region of its own): a query treating every volume labelled with the
# outside's region as background prunes it (MicroBooNE: wires, PMT glass and acrylic sharing the opaque region with
# the outside air vanished, 26 % of rays found no boundary).
OUTSIDE_MATERIAL = "<outside>"
SHOWN = 4  # error messages list at most this many names


class TreeError(ValueError):
    """A volume tree that is not one: duplicate or unreachable PVs, a missing material, ambiguous mothers."""


# ------------------------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class PV:
    """A physical volume as a world instance: its unique name, LV, material, mother PV (None: the world), copy."""

    name: str
    lv: str
    material: str
    mother: str | None = None
    copy: int | None = None


@dataclass(frozen=True, eq=False)
class Tree:
    """World instances of physical volumes: one root, every PV of one LV of that LV's material.

    ``aliases`` maps dissolved placements to the placement now holding their region (GDML surface names that name
    them name it), ``groups`` a dissolved envelope's daughters to the envelope instance (Geant4 adjacency, border
    surfaces, holds within one), ``film_axes`` each split WLS-film LV to its face normal (None: no film split).
    Derived and checked on construction: ``by_name``, ``children`` (by name), ``root``, ``order`` (canonical
    depth-first, children by name) and ``lv_material``.
    """

    pvs: tuple[PV, ...]
    aliases: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    film_axes: dict | None = None

    def __post_init__(self):
        """Check the tree (its indexes are derived here)."""
        _ = self.by_name, self.lv_material  # derives and checks every index, PVs first

    @cached_property
    def by_name(self) -> dict:
        """Return the PVs by name."""
        return _by_name(self.pvs)

    @cached_property
    def root(self) -> str:
        """Return the world PV's name."""
        roots = sorted(p.name for p in self.pvs if p.mother is None)
        if len(roots) != 1:
            raise TreeError(f"tree: exactly one world PV (mother None) required, got {roots}")
        return roots[0]

    @cached_property
    def children(self) -> dict:
        """Return each PV's daughter PV names, sorted."""
        by, children = self.by_name, {n: [] for n in self.by_name}
        for p in self.pvs:
            if p.mother is not None:
                if p.mother not in by:
                    raise TreeError(f"tree: PV {p.name!r} has unknown mother {p.mother!r}")
                children[p.mother].append(p.name)
        return {n: tuple(sorted(c)) for n, c in children.items()}

    @cached_property
    def order(self) -> tuple:
        """Return the PV names in the canonical depth-first order, children by name."""
        order, stack, children = [], [self.root], self.children
        while stack:
            n = stack.pop()
            order.append(n)
            stack.extend(reversed(children[n]))
        if len(order) != len(self.by_name):
            lost = sorted(set(self.by_name) - set(order))
            raise TreeError(f"tree: PVs not reachable from the world (mother cycle): {lost[:8]}")
        return tuple(order)

    @cached_property
    def lv_material(self) -> dict:
        """Return each LV's material (one per LV)."""
        mat = {}
        for n in self.order:
            p = self.by_name[n]
            if mat.setdefault(p.lv, p.material) != p.material:
                raise TreeError(f"tree: LV {p.lv!r} placed with materials {mat[p.lv]!r} and {p.material!r}")
        return mat

    def placements(self, lv):
        """Return the PV names of an LV, in the canonical order."""
        return tuple(n for n in self.order if self.by_name[n].lv == lv)


def _by_name(pvs):
    by = {}
    for p in pvs:
        if not isinstance(p, PV):
            raise TreeError(f"tree: expected PV, got {p!r}")
        if p.name in by:
            raise TreeError(f"tree: duplicate PV name {p.name!r}")
        by[p.name] = p
    return by


@dataclass(frozen=True)
class Instance:
    """One world instance of one member: its PV name, LV, definition and member index, mother PV and world pose.

    ``sensor`` is its sensor id (None: not a sensor); ``group`` the dissolved envelope instance its placement came
    from.
    """

    pv: str
    lv: str
    definition: str
    member: int
    mother: str
    R: np.ndarray
    t: np.ndarray
    sensor: int | None
    mesh: bool
    group: str = ""


@dataclass(frozen=True)
class _Node:
    """One definition instance: its members' PV names and its placements' instances (index order)."""

    definition: str
    pvs: tuple[str, ...]
    children: tuple[_Node, ...]


# ------------------------------------------------------------------------------------------------------- the walk
def _split(loaded, media):
    if hasattr(loaded, "scene") and hasattr(loaded, "media"):
        return loaded.scene, dict(loaded.media) if media is None else dict(media)
    if media is None:
        raise TreeError("a bare Scene needs media= (GDML material name -> medium id)")
    return loaded, dict(media)


def _material_names(media):
    names = {}
    for name, mid in media.items():
        if int(mid) in names:
            raise TreeError(f"media: materials {names[int(mid)]!r} and {name!r} share medium id {mid}")
        names[int(mid)] = name
    return names


def _segments(d):
    """Per placement: its path segment (its name, else '<definition>#<index>'), unique among siblings by '#<index>'."""
    segs = [pl.name or f"{pl.definition}#{k}" for k, pl in enumerate(d.placements)]
    n = Counter(segs)
    return [s if n[s] == 1 else f"{s}#{k}" for k, s in enumerate(segs)]


class _Walk:
    """The canonical walk of a Scene: member instances in order, PATH sensor ids numbered as it goes."""

    def __init__(self, scene):
        self.scene, self.insts, self.path_id = scene, [], 0

    def member_instances(self, dname, path, pose, msen, parent):
        """Record the instances of a definition's members; return their PV names (rank order)."""
        d = self.scene.definitions[dname]
        order = sorted(range(len(d.members)), key=lambda i: d.members[i].rank)
        vols = [d.members[i].label.volume or f"{dname}.m{i}" for i in range(len(d.members))]
        nv = Counter(vols)
        pvs = []
        for i in order:
            m = d.members[i]
            vol = vols[i] if nv[vols[i]] == 1 else f"{vols[i]}#{i}"
            pv = "/".join(path + (vol,))
            lv = (m.label.volume or dname) if len(d.members) == 1 else f"{dname}.{vol}"
            mesh = isinstance(m.solid, sc.Mesh)
            s = int(m.label.sensor)
            if s == sc.PATH:
                if mesh:
                    raise TreeError(f"{pv}: a mesh member with a PATH sensor label (meshes are not numbered)")
                s, self.path_id = self.path_id, self.path_id + 1
            elif s == sc.PLACEMENT:
                s = msen
            self.insts.append(Instance(pv, lv, dname, i, parent[0], pose[0], pose[1], s if s >= 0 else None, mesh,
                                       parent[1]))  # fmt: skip
            pvs.append(pv)
        return pvs

    def visit(self, dname, path, pose, msen, parent):
        """Walk one definition instance at world pose (R, t); ``parent`` is (mother PV, group). Return its _Node."""
        if dname not in self.scene.definitions:
            raise TreeError(f"placement of unknown definition {dname!r}")
        d = self.scene.definitions[dname]
        if len(d.members) > 1 and d.placements:
            raise TreeError(f"definition {dname!r} has {len(d.members)} members and placements: the daughters' "
                            "mother volume is ambiguous")  # fmt: skip
        pvs = self.member_instances(dname, path, pose, msen, parent)
        inner = pvs[0] if pvs else parent[0]
        rw, tw = pose
        kids = []
        for pl, seg in zip(d.placements, _segments(d), strict=True):
            rot, t = np.asarray(pl.R, float), np.asarray(pl.t, float)
            kids.append(self.visit(pl.definition, path + (seg,), (rw @ rot, rw @ t + tw),
                                   int(pl.sensor) if int(pl.sensor) >= 0 else msen,
                                   (inner, getattr(pl, "group", ""))))  # fmt: skip
        return _Node(dname, tuple(pvs), tuple(kids))


def _walk(scene):
    """Return (root _Node, [Instance]) in the canonical order."""
    w = _Walk(scene)
    root = w.visit(scene.root, (), (np.eye(3), np.zeros(3)), -1, (OUTSIDE_PV, ""))
    return root, w.insts


def instances(loaded):
    """Return every world instance of a member (Instance records), canonical order, of a Loaded or a Scene."""
    return _walk(getattr(loaded, "scene", loaded))[1]


def tree_from_geometry(loaded, media=None):
    """Return the Tree of a Loaded (or of a Scene with ``media``: material name -> medium id).

    The outside PV, then one PV per member world instance; material = the GDML material name. The tree also carries
    ``aliases`` (dissolved placements -> the placement now holding their region: GDML surface names that name them
    name it) and ``groups`` (dissolved envelopes' daughters -> the envelope instance: Geant4 adjacency, border
    surfaces, holds within one).
    """
    scene, media = _split(loaded, media)
    names = _material_names(media)
    _, insts = _walk(scene)
    pvs = [PV(OUTSIDE_PV, OUTSIDE_PV, OUTSIDE_MATERIAL)]
    for x in insts:
        mid = int(scene.definitions[x.definition].members[x.member].label.medium)
        if mid not in names:
            raise TreeError(f"{x.pv}: medium id {mid} has no material name in media")
        pvs.append(PV(x.pv, x.lv, names[mid], x.mother, x.sensor))
    return Tree(tuple(pvs), dict(getattr(scene, "aliases", {}) or {}), {x.pv: x.group for x in insts if x.group})


# ------------------------------------------------------------------------------------------------------- relabel
def _check_classes(want, class_of):
    have = set(class_of)
    if want - have:
        raise TreeError(f"class_of misses {len(want - have)} PVs, e.g. {sorted(want - have)[:SHOWN]}")
    shared = sorted(pv for pv in want if pv != OUTSIDE_PV and class_of[pv] == class_of[OUTSIDE_PV])
    if shared:
        raise TreeError(f"{len(shared)} PVs share the outside's region {class_of[OUTSIDE_PV]} (e.g. {shared[:3]}): a "
                        "query prunes them as background; bind the outside material "
                        f"{OUTSIDE_MATERIAL!r} to a medium of its own")  # fmt: skip
    if have - want:
        raise TreeError(f"class_of names {len(have - want)} PVs this Scene does not have, e.g. "
                        f"{sorted(have - want)[:SHOWN]} (a different tree?)")  # fmt: skip
    for pv in want:
        c = class_of[pv]
        if isinstance(c, bool) or not isinstance(c, (int, np.integer)) or int(c) < 0:
            raise TreeError(f"class_of[{pv!r}] = {c!r}: region ids are non-negative integers")


def _variants(root, class_of):
    """Intern (definition, member regions, children's keys) of every definition instance."""
    ids, variants, node_key = {}, {}, {}

    def key(node):
        k = (node.definition, tuple(int(class_of[p]) for p in node.pvs), tuple(key(c) for c in node.children))
        if k not in ids:
            ids[k] = len(ids)
            variants.setdefault(node.definition, []).append(k)
        node_key[id(node)] = ids[k]
        return ids[k]

    key(root)
    return ids, variants, node_key


def relabel(scene, class_of):
    """Return a new Scene whose member labels are region ids (sensor labels and volume names kept).

    A definition is cloned (``<name>|class<k>``) only where its world instances (with their subtrees) need different
    regions; unreachable definitions are dropped. ``class_of`` maps the PV name (``tree_from_geometry``'s) to a
    region id, for exactly the PVs of the tree. The float64 oracle's labels.
    """
    scene = getattr(scene, "scene", scene)
    root, insts = _walk(scene)
    _check_classes({OUTSIDE_PV} | {x.pv for x in insts}, class_of)
    ids, variants, node_key = _variants(root, class_of)
    name_of = {}
    for dname, ks in variants.items():
        for j, k in enumerate(ks):
            name_of[ids[k]] = dname if len(ks) == 1 else f"{dname}|class{j}"
    for ks in variants.values():
        clash = [name_of[ids[k]] for k in ks if len(ks) > 1 and name_of[ids[k]] in scene.definitions]
        if clash:
            raise TreeError(f"clone names {clash} already exist in the Scene")
    defs = {}
    for kid, (dname, classes, children) in sorted({v: k for k, v in ids.items()}.items()):
        d = scene.definitions[dname]
        order = sorted(range(len(d.members)), key=lambda i: d.members[i].rank)
        cls_of_member = dict(zip(order, classes, strict=True))
        members = [sc.Member(m.solid, sc.Label(cls_of_member[i], m.label.sensor, m.label.volume), m.rank)
                   for i, m in enumerate(d.members)]  # fmt: skip
        placements = [sc.Placement(name_of[c], pl.R, pl.t, pl.sensor, pl.name, pl.clip, getattr(pl, "group", ""))
                      for pl, c in zip(d.placements, children, strict=True)]  # fmt: skip
        defs[name_of[kid]] = sc.Definition(name_of[kid], members=members, placements=placements)
    out = scene.outside
    n_clones = sum(len(ks) for ks in variants.values() if len(ks) > 1)
    notes = list(scene.approximations) + [f"relabel: member labels are optical class ids ({n_clones} definition "
                                          "clones for instances of different classes)"]  # fmt: skip
    return sc.Scene(defs, name_of[node_key[id(root)]], sc.Label(int(class_of[OUTSIDE_PV]), out.sensor, out.volume),
                    notes, hole_fillers=list(scene.hole_fillers))  # fmt: skip
