"""The GDML solids lumen reads and writes: each attribute's GDML spelling and dimension, in Geant4's order.

The reader stores a solid's attributes lower-cased (``lowx``); Geant4 compares attribute names case-sensitively
(G4GDMLReadSolids::CutTubeRead reads ``lowX`` and ignores ``lowx``), so the exporter writes each attribute with the
spelling of this table. Dimensions: ``L`` a length (m in the detector file, mm in the GDML, under ``lunit``), ``A``
an angle (rad, under ``aunit``), ``U`` unitless (read with no unit, e.g. elcone's ``dx``/``dy`` semi-axis ratios).

``CHILDREN`` names the child elements a solid carries (written from the record's ``planes``, ``vertices`` and
``sections``, or from its references). The set is the 20 solids the lowering supports, the two generic ones the
reader reads, the booleans, multiUnion, scaledSolid and tessellated.

Notes
-----
Attribute lists from G4GDMLReadSolids.cc (Geant4 11.2.2, the same in 10.7.4): BoxRead, ConeRead, CutTubeRead,
ElconeRead, EllipsoidRead, EltubeRead, GenericPolyconeRead, GenericPolyhedraRead, GenTrapRead (arb8), HypeRead,
OrbRead, ParaRead, ParaboloidRead, PolyconeRead, PolyhedraRead, SphereRead, TetRead, TorusRead, TrapRead, TrdRead,
TubeRead, XtruRead.
"""

from __future__ import annotations

L, A, U = "L", "A", "U"
PHI = (("startphi", A), ("deltaphi", A))

ATTRIBUTES: dict[str, tuple[tuple[str, str], ...]] = {
    "box": (("x", L), ("y", L), ("z", L)),
    "tube": (("rmin", L), ("rmax", L), ("z", L), *PHI),
    "cutTube": (("rmin", L), ("rmax", L), ("z", L), *PHI, ("lowX", U), ("lowY", U), ("lowZ", U), ("highX", U),
                ("highY", U), ("highZ", U)),
    "cone": (("rmin1", L), ("rmax1", L), ("rmin2", L), ("rmax2", L), ("z", L), *PHI),
    "polycone": PHI,
    "genericPolycone": PHI,
    "polyhedra": (*PHI, ("numsides", U)),
    "genericPolyhedra": (*PHI, ("numsides", U)),
    "sphere": (("rmin", L), ("rmax", L), *PHI, ("starttheta", A), ("deltatheta", A)),
    "orb": (("r", L),),
    "ellipsoid": (("ax", L), ("by", L), ("cz", L), ("zcut1", L), ("zcut2", L)),
    "eltube": (("dx", L), ("dy", L), ("dz", L)),
    "elcone": (("dx", U), ("dy", U), ("zmax", L), ("zcut", L)),
    "paraboloid": (("rlo", L), ("rhi", L), ("dz", L)),
    "hype": (("rmin", L), ("rmax", L), ("inst", A), ("outst", A), ("z", L)),
    "torus": (("rmin", L), ("rmax", L), ("rtor", L), *PHI),
    "trd": (("x1", L), ("x2", L), ("y1", L), ("y2", L), ("z", L)),
    "para": (("x", L), ("y", L), ("z", L), ("alpha", A), ("theta", A), ("phi", A)),
    "trap": (("z", L), ("theta", A), ("phi", A), ("y1", L), ("x1", L), ("x2", L), ("alpha1", A), ("y2", L),
             ("x3", L), ("x4", L), ("alpha2", A)),
    "arb8": (("dz", L), *((f"v{i}{c}", L) for i in range(1, 9) for c in "xy")),
    "tet": (),
    "xtru": (),
    "tessellated": (),
    "union": (),
    "subtraction": (),
    "intersection": (),
    "multiUnion": (),
    "scaledSolid": (),
}  # fmt: skip

CHILDREN = {
    "polycone": "zplane", "polyhedra": "zplane", "genericPolycone": "rzpoint", "genericPolyhedra": "rzpoint",
    "xtru": "xtru", "tet": "tet", "tessellated": "facets", "union": "boolean", "subtraction": "boolean",
    "intersection": "boolean", "multiUnion": "nodes", "scaledSolid": "scaled",
}  # fmt: skip
PLANE_ATTRIBUTES = {"zplane": ("rmin", "rmax", "z"), "rzpoint": ("r", "z")}  # all lengths
SECTION_ATTRIBUTES = ("zPosition", "xOffset", "yOffset", "scalingFactor")  # lengths, then the unitless factor


def spelling(solid_type: str) -> dict[str, tuple[str, str]]:
    """Return {lower-cased attribute: (GDML spelling, dimension)} of a solid type; raise KeyError if unknown."""
    return {name.lower(): (name, dim) for name, dim in ATTRIBUTES[solid_type]}
