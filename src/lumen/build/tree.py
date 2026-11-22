"""The volume tree of a Scene: one physical volume per world instance of a member.

The PV names are the ones every consumer shares (the optics' regions, the navigator's volumes).

    tree, insts = tree_from_geometry(scene, media)   # the outside PV, then one PV per member world instance;
                                                     # the Instance records, in the canonical order

World instances, in the canonical order (``lumen.geometry.scene.instances``: from the root, placements by index):

- a definition instance with a member is one PV, named by its placement path and the member's volume name (the
  loader's sensor path, ``Loaded.sensors``); its LV is the member's volume name (the GDML LV, also for the lowering's
  ``|mirror`` variants), else the definition name;
- a member-less definition (an assembly, the lowering's root wrapper) is no PV: its daughters' mother is the enclosing
  PV;
- the outside region (``Scene.outside``) is the world PV ``OUTSIDE_PV``;
- ``PV.copy`` is the instance's sensor id (``scene.PATH`` numbered per world instance, or explicit), None for a
  non-sensor.

lumen reads only its own Scene; sensor indices and pass pairs come from the optics (``lumen.build.lookup``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from lumen.geometry import scene as sc

OUTSIDE_PV = "<outside>"
# The outside's material is always its own (a region of its own): a query treating every volume labelled with the
# outside's region as background prunes it (the volumes sharing the outside's region, an opaque one, would vanish from
# the query).
OUTSIDE_MATERIAL = "<outside>"


class TreeError(ValueError):
    """A volume tree that is not one: duplicate or unreachable PVs, a missing material, ambiguous mothers."""


# ------------------------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class PV:
    """A physical volume as a world instance: its unique name, LV, material, mother PV (None: the world), copy.

    ``physvol``: the placement it is an instance of (the detector file's physvol name, what a border surface names;
    "" an unnamed or the root's); None: the PV is its own (a tree built directly).
    """

    name: str
    lv: str
    material: str
    mother: str | None = None
    copy: int | None = None
    physvol: str | None = None


@dataclass(frozen=True, eq=False)
class Tree:
    """World instances of physical volumes: one root, every PV of one LV of that LV's material.

    Derived and checked on construction: ``by_name``, ``children`` (by name), ``root``, ``order`` (canonical
    depth-first, children by name) and ``lv_material``.
    """

    pvs: tuple[PV, ...]

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

    def physvol(self, n) -> str:
        """Return the physvol PV n is an instance of (``PV.physvol``)."""
        p = self.by_name[n].physvol
        return n if p is None else p

    @cached_property
    def instances_of(self) -> dict:
        """Return per physvol name its world instances (PV names, canonical order); unnamed ones are not listed."""
        out: dict = {}
        for n in self.order:
            p = self.physvol(n)
            if p:
                out.setdefault(p, []).append(n)
        return {k: tuple(v) for k, v in out.items()}

    @cached_property
    def instances_by_mother(self) -> dict:
        """Return per physvol name its world instances by their mother PV (each in the canonical order)."""
        out: dict = {}
        for p, names in self.instances_of.items():
            by: dict = {}
            for n in names:
                by.setdefault(self.by_name[n].mother, []).append(n)
            out[p] = {m: tuple(v) for m, v in by.items()}
        return out

    @cached_property
    def rank(self) -> dict:
        """Return each PV's index in the canonical order."""
        return {n: i for i, n in enumerate(self.order)}

    @cached_property
    def depth(self) -> dict:
        """Return each PV's depth in the tree (the world PV's 0)."""
        depth: dict = {}
        for n in self.order:  # mothers first
            m = self.by_name[n].mother
            depth[n] = 0 if m is None else depth[m] + 1
        return depth


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
    """One world instance of a member: its PV name, LV, definition, mother PV and world pose.

    ``sensor`` is its sensor id (None: not a sensor); ``physvol`` the name of the placement that placed it.
    """

    pv: str
    lv: str
    definition: str
    mother: str
    R: np.ndarray
    t: np.ndarray
    sensor: int | None
    mesh: bool
    physvol: str = ""


# ------------------------------------------------------------------------------------------------------- the walk
def _material_names(media):
    names = {}
    for name, mid in media.items():
        if int(mid) in names:
            raise TreeError(f"media: materials {names[int(mid)]!r} and {name!r} share medium id {mid}")
        names[int(mid)] = name
    return names


def tree_from_geometry(scene, media) -> tuple:
    """Return (the Tree, the member world instances) of a lowered Scene and its media (material name -> medium id).

    The Tree holds the outside PV, then one PV per member world instance (material: the GDML material name); the
    instances are ``Instance`` records in the canonical order. One pass over the canonical order: a member's mother PV
    is the nearest instance above it with a member, and its PATH sensor id the count of PATH members before it.
    """
    names = _material_names(media)
    T = sc.instances(scene)
    defs, n = scene.definitions, len(T.defn)
    inner: list = [None] * n  # per instance: the PV its daughters' mother is (its own, or the enclosing one)
    insts, path_id = [], 0
    pvs = [PV(OUTSIDE_PV, OUTSIDE_PV, OUTSIDE_MATERIAL)]
    for i in range(n):
        dname, p = T.defn[i], int(T.parent[i])
        mother = OUTSIDE_PV if p < 0 else inner[p]
        m = defs[dname].member
        if m is None:
            inner[i] = mother
            continue
        pv = T.name(scene, i)
        mesh = isinstance(m.solid, sc.Mesh)
        s = int(m.label.sensor)
        if s == sc.PATH:
            if mesh:
                raise TreeError(f"{pv}: a mesh member with a PATH sensor label (meshes are not numbered)")
            s, path_id = path_id, path_id + 1
        x = Instance(pv, m.label.volume or dname, dname, mother, T.R[i], T.t[i], s if s >= 0 else None, mesh,
                     T.physvol[i])  # fmt: skip
        insts.append(x)
        inner[i] = pv
        mid = int(m.label.medium)
        if mid not in names:
            raise TreeError(f"{pv}: medium id {mid} has no material name in media")
        pvs.append(PV(pv, x.lv, names[mid], mother, x.sensor, x.physvol))
    return Tree(tuple(pvs)), insts
