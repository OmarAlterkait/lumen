"""The lowered scene: what a detector's geometry is, independent of any traversal.

- a solid is a ``lumen.build.solids`` CSG tree in its definition's frame (m), or a ``Mesh``;
- a ``Label`` says what a region is: its medium id, its sensor id (-1 none; ``PLACEMENT``: the placement's id;
  ``PATH``: one id per world instance) and its volume name (the GDML logical volume, for border and skin surfaces and
  Geant4 checks);
- a ``Member`` is (solid, label, rank): inside one definition a point belongs to the highest-rank member containing
  it (equal ranks must not overlap);
- a ``Placement`` places a definition by a rigid pose (R, t), ``x_mother = R x_daughter + t``, with a sensor id for
  ``PLACEMENT`` labels; ``clip`` makes it a Geant4 daughter (clipped to its mother) or a priority overlay (not
  clipped);
- a ``Definition`` holds members and placements of daughter definitions; a ``Scene`` the definitions, the root, the
  outside label and the declared approximations.

Every point lies in exactly one region: within a definition by member rank; across levels by Geant4's locate rule --
depth-first pre-order rank, daughters in index order, a daughter outranks its mother and is clipped to its mother's
effective solid. Geant4 takes its placements as overlap-free (daughters inside their mothers, siblings disjoint);
where that holds the clip changes nothing. Overlap resolution beyond it is never silent: an approximation or decision
is listed in ``Scene.approximations``. ``lumen.validate.oracle`` evaluates this rule in float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PLACEMENT = -2  # Label.sensor: take the placement's sensor id
# Label.sensor: one id per world instance of the member, numbered in the canonical order -- depth-first from the
# root, a definition's members by rank, then its placements by index -- so a sensor inside a repeated parent (an
# optical detector in a module placed many times) still gets a unique id. Every consumer counts the same way.
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
    """What a region is: medium id, sensor id (-1 none, ``PLACEMENT``, ``PATH``) and GDML volume name."""

    medium: int
    sensor: int = -1
    volume: str = ""


@dataclass
class Member:
    """A solid (or Mesh) of a definition, its label and its rank (the highest rank containing a point wins)."""

    solid: object
    label: Label
    rank: int = 0


@dataclass
class Placement:
    """A definition placed by ``x_mother = R x_daughter + t``.

    ``clip`` True: a Geant4 daughter (clipped to the mother's effective solid); False: a priority overlay (outranks
    the mother, not clipped). ``group`` names the dissolved navigation envelope instance it came from ("" none):
    Geant4 adjacency (border surfaces) holds only within one envelope instance.
    """

    definition: str
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sensor: int = -1
    name: str = ""
    clip: bool = True
    group: str = ""


@dataclass
class Definition:
    """A definition: its members and the placements of its daughter definitions."""

    name: str
    members: list[Member] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)


@dataclass
class Scene:
    """Definitions by name, the root definition, the outside label, and what was approximated or decided.

    ``hole_fillers`` lists definitions that overlap others by construction (a solid left without its holes, the siblings
    filling them, resolved by rank); ``aliases`` maps a dissolved placement's name to the placement that now holds its
    region (its mother's), for surface names.
    """

    definitions: dict[str, Definition]
    root: str
    outside: Label
    approximations: list[str] = field(default_factory=list)
    hole_fillers: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
