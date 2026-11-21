"""The lowered scene: what a detector's geometry is, independent of any traversal.

- a solid is a ``lumen.geometry.solids`` CSG tree in its definition's frame (m), or a ``Mesh``;
- a ``Label`` says what a region is: its medium id, its sensor id (-1 none; ``PATH``: one id per world instance) and
  its volume name (the GDML logical volume, for border and skin surfaces and Geant4 checks);
- a ``Member`` is (solid, label): a definition's one solid (a Geant4 logical volume's);
- a ``Placement`` places a definition by a rigid pose (R, t), ``x_mother = R x_daughter + t``; it is a Geant4
  daughter, clipped to its mother;
- a ``Definition`` holds its member (None: an assembly) and placements of daughter definitions; a ``Scene`` the
  definitions, the root, the outside label and the hole fillers;
- ``instances``: every definition instance in the canonical order, the one walk every consumer reads.

Every point lies in exactly one region: across levels by Geant4's locate rule -- a daughter outranks its mother and
is clipped to its mother's effective solid -- and among overlapping siblings by
ownership (``lumen.geometry.ownership``): a solid proved inside its sibling, else the one placed first, holds
the region they share, with its daughters; a non-owner and everything below it hold none of it. Geant4 takes its
placements as overlap-free (daughters inside their mothers, siblings disjoint); where that holds the clip and the
ownership change nothing. ``lumen.compare.oracle`` evaluates this rule in float64.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Label.sensor: one id per world instance of the member, numbered in the canonical order (``instances``) -- so a
# sensor inside a repeated parent (an optical detector in a module placed many times) still gets a unique id.
PATH = -3


class Mesh:
    """A closed triangle surface (tris: (n, 3, 3) m, definition frame): a Geant4 tessellated solid.

    It is kept as an opaque free-form part: only its first hit and facet normal are used. It is never clipped, and its
    facet edges are creases.
    """

    def __init__(self, tris):
        self.tris = np.asarray(tris, float)


@dataclass
class Label:
    """What a region is: medium id, sensor id (-1 none, ``PATH``) and GDML volume name."""

    medium: int
    sensor: int = -1
    volume: str = ""


@dataclass
class Member:
    """The solid (or Mesh) of a definition and its label."""

    solid: object
    label: Label


@dataclass
class Placement:
    """A definition placed by ``x_mother = R x_daughter + t`` (t in m).

    A Geant4 daughter: clipped to the mother's effective solid. ``source``: the detector file's placement it lowers
    (-1: none, a placement the lowering made).
    """

    definition: str
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = ""
    source: int = -1


@dataclass
class Definition:
    """A definition: its member (None: an assembly) and the placements of its daughter definitions.

    ``origin``: the logical volume it was lowered from ("" none: a scene built directly); ``piece_of``: the definition
    a split union's piece was split from (``lumen.build.lowering.split_unions``; "" none); ``holed``: its member's
    solid was lowered without the holes sibling placements fill (``Scene.hole_fillers``). Structure the lowering
    records, so no consumer reads it back from a name.
    """

    name: str
    member: Member | None = None
    placements: list[Placement] = field(default_factory=list)
    origin: str = ""
    piece_of: str = ""
    holed: bool = False


@dataclass
class Scene:
    """Definitions by name, the root definition, the outside label, and the hole fillers.

    ``hole_fillers`` lists the logical volumes (``Definition.origin``, else the name) of the siblings filling the holes
    a lowering left out of a ``Definition.holed`` solid: they own the holes.
    """

    definitions: dict[str, Definition]
    root: str
    outside: Label
    hole_fillers: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ instance paths
# One naming rule for an instance path (a sensor id, a tree PV name): segments unique among siblings, so two siblings
# with one placement name (generated physvol names such as "<lv>_PV") never share a path.
def placement_segments(d: Definition) -> list[str]:
    """Per placement of ``d``: its path segment (its name, else '<definition>#<index>'), made unique among its
    siblings by '#<index>'."""  # noqa: D205, D209
    segs = [pl.name or f"{pl.definition}#{k}" for k, pl in enumerate(d.placements)]
    n = Counter(segs)
    return [s if n[s] == 1 else f"{s}#{k}" for k, s in enumerate(segs)]


def member_segment(d: Definition, name: str) -> str:
    """Return the path segment of ``d``'s member: its volume name, else ``name`` (the definition's key)."""
    return (d.member.label.volume if d.member is not None else "") or name


@dataclass(frozen=True)
class Instances:
    """Every definition instance of a Scene, in the canonical order: depth-first from the root, a definition's
    placements by index. The one walk: the volume tree, the navigator, the oracle, the sensor ids and the world boxes
    all read it.

    Per instance: ``defn`` its definition, ``parent`` its mother instance (-1: the root), ``R``, ``t`` its world pose
    (``x_world = R x + t``), ``path`` its placement path (``placement_segments`` joined by '/'; "" the root),
    ``physvol`` the name of the placement that placed it ("" the root), ``source`` that placement's detector-file
    index (-1 none) and ``depth``.
    """  # noqa: D205

    defn: tuple
    parent: np.ndarray
    R: np.ndarray
    t: np.ndarray
    path: tuple
    physvol: tuple
    source: np.ndarray
    depth: np.ndarray

    def name(self, scene, i) -> str:
        """Return instance i's member's name (its PV: the path and the member's segment)."""
        seg = member_segment(scene.definitions[self.defn[i]], self.defn[i])
        return f"{self.path[i]}/{seg}" if self.path[i] else seg


def instances(scene: Scene) -> Instances:
    """Return the Scene's ``Instances`` (one stack walk; a definition's segments computed once)."""
    defs = scene.definitions
    segs: dict = {}
    dn, par, Rs, ts, path, phys, src, dep = [], [], [], [], [], [], [], []
    stack = [(scene.root, -1, np.eye(3), np.zeros(3), "", "", -1, 0)]
    while stack:
        dname, p, rw, tw, pth, pv, so, k = stack.pop()
        if dname not in defs:
            raise KeyError(f"placement of unknown definition {dname!r}")
        i = len(dn)
        dn.append(dname)
        par.append(p)
        Rs.append(rw)
        ts.append(tw)
        path.append(pth)
        phys.append(pv)
        src.append(so)
        dep.append(k)
        d = defs[dname]
        if dname not in segs:
            segs[dname] = placement_segments(d)
        for pl, seg in reversed(list(zip(d.placements, segs[dname], strict=True))):
            rot, t = np.asarray(pl.R, float), np.asarray(pl.t, float)
            stack.append((pl.definition, i, rw @ rot, rw @ t + tw, f"{pth}/{seg}" if pth else seg, pl.name, pl.source,
                          k + 1))  # fmt: skip
    n = len(dn)
    return Instances(tuple(dn), np.asarray(par, np.int64), np.asarray(Rs, float).reshape(n, 3, 3),
                     np.asarray(ts, float).reshape(n, 3), tuple(path), tuple(phys), np.asarray(src, np.int64),
                     np.asarray(dep, np.int64))  # fmt: skip
