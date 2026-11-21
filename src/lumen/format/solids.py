"""The solids of the detector structure: Geant4's CSG catalogue as GDML spells it, with each attribute's dimension.

Attributes are in Geant4's order. The structural check, the YAML reader and the GDML exporter read this one table
(``lumen.format.check``, ``lumen.format.yaml``, ``lumen.gdml.exporter``).

A solid's attributes are stored under these names, spelled exactly as Geant4 compares them (G4GDMLReadSolids::
CutTubeRead reads ``lowX`` and ignores ``lowx``): the reader accepts no other spelling and the exporter writes these.
Dimensions: ``L`` a length (m in the detector file, mm in the GDML, under ``lunit``), ``A``
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

import math

from lumen.units import TWO_PI

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
# G4GDMLReadSolids reads an omitted attribute as 0. At deltaphi <= 0 the constructors of these solids abort ("Invalid
# dphi": G4Tubs.icc, G4CutTubs.icc, G4Cons.icc, G4Sphere.icc, G4Torus.cc; and G4Sphere at deltatheta <= 0), while
# G4Polycone, G4Polyhedra and G4GenericPolycone take it as the full turn
OPEN_ANGLE_ABORTS = {"tube": ("deltaphi",), "cutTube": ("deltaphi",), "cone": ("deltaphi",),
                     "sphere": ("deltaphi", "deltatheta"), "torus": ("deltaphi",)}  # fmt: skip
# what a YAML or builder solid that omits its angular range means: the whole solid (stored explicitly, so the record and
# its GDML export say it); a GDML's omitted attribute is Geant4's 0
WHOLE = {"startphi": 0.0, "deltaphi": TWO_PI, "starttheta": 0.0, "deltatheta": math.pi}


def whole(solid_type: str, params: dict) -> dict:
    """Return a YAML or builder solid's attributes with an omitted angular range read as the whole solid (``WHOLE``)."""
    names = {n for n, _ in ATTRIBUTES.get(solid_type, ())}
    return {**params, **{k: v for k, v in WHOLE.items() if k in names and k not in params}}


PLANE_ATTRIBUTES = {"zplane": ("rmin", "rmax", "z"), "rzpoint": ("r", "z")}  # all lengths
SECTION_ATTRIBUTES = ("zPosition", "xOffset", "yOffset", "scalingFactor")  # lengths, then the unitless factor


def dimensions(solid_type: str) -> dict[str, str]:
    """Return {attribute: dimension} of a solid type; raise KeyError if unknown."""
    return dict(ATTRIBUTES[solid_type])
