"""The detector file's records: what Geant4 reads from the GDML, plus the recipe's decisions, with provenance.

Units are m, rad, ns, nm and eV everywhere; fields whose unit is not obvious carry it in their name. The records are
frozen and hold plain numbers, strings and numpy arrays, so a detector file is a value: it is written and read back
bit for bit (``lumen.format.io``).

The geometry keeps the FULL GDML world (the detector volume is cut out at load) in Geant4's conventions:

- a placement's rotation is stored as the GDML physvol angles (passive, rad) and its position in m; the matrix is
  derived (``lumen.build``), never stored, since angles to matrix to angles is not exact;
- a boolean's or multi-union's transforms are stored the same way (GDML's active convention for booleans);
- solid attributes keep GDML's own spelling (lower-cased), in m and rad;
- physical-volume names are always explicit (an unnamed physvol gets Geant4's ``<lv>_PV``) and kept whole: Geant4
  binds a border's physvolref by its full name, so names that differ only in the ``0x`` suffix stay distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA = "lumen-detector/1"
OUTSIDE_ABSORB = "absorb"  # photons leaving the detector volume end there


@dataclass(frozen=True)
class Aux:
    """A GDML auxiliary tag: type, value, unit, and nested tags."""

    type: str
    value: str
    unit: str = ""
    children: tuple[Aux, ...] = ()


@dataclass(frozen=True)
class Transform:
    """A GDML transform: rotation angles (rad, GDML order x, y, z) and position (m)."""

    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Solid:
    """A GDML solid in GDML's spelling.

    ``params`` holds the numeric attributes (lower-cased names; m, rad or unitless). ``refs`` and ``transforms`` are
    a boolean's (first, second), a multi-union's nodes or a scaled solid's child; ``scale`` is a scaled solid's.
    ``planes`` holds a polycone's or polyhedra's (rmin, rmax, z) planes, ``vertices`` an extrusion's (x, y) or a
    tetrahedron's four (x, y, z) points, ``sections`` an extrusion's (z, x offset, y offset, scale); ``mesh`` names
    the array of a tessellated solid's triangles (k, 3, 3), m.
    """

    name: str
    type: str
    params: dict[str, float] = field(default_factory=dict)
    refs: tuple[str, ...] = ()
    transforms: tuple[Transform, ...] = ()
    scale: tuple[float, float, float] | None = None
    planes: tuple[tuple[float, ...], ...] = ()
    vertices: tuple[tuple[float, ...], ...] = ()
    sections: tuple[tuple[float, float, float, float], ...] = ()
    mesh: str | None = None


@dataclass(frozen=True)
class Volume:
    """A logical volume (or an assembly: no solid, no material) and its daughter placements in Geant4's order."""

    name: str
    solid: str | None
    material: str | None
    daughters: tuple[int, ...] = ()
    assembly: bool = False
    aux: tuple[Aux, ...] = ()


@dataclass(frozen=True)
class Placement:
    """A physical volume: the volume it places, its mother volume, its name, copy number, pose and scale."""

    name: str
    volume: str
    mother: str
    copy: int | None
    position: tuple[float, float, float]  # m, in the mother's frame
    rotation: tuple[float, float, float]  # rad, GDML physvol angles (passive)
    scale: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Property:
    """A material or surface property as Geant4 reads it: a table linear in photon energy, or a constant.

    ``energy`` (eV, ascending) is None for a constant. Lengths are in m and times in ns. ``matrix`` names the GDML
    matrix it was read from; ``origin`` is "gdml" or "recipe" (a table the GDML lacks, with its source in ``note``).
    """

    key: str
    value: np.ndarray
    energy: np.ndarray | None = None
    matrix: str = ""
    origin: str = "gdml"
    note: str = ""


@dataclass(frozen=True)
class Component:
    """One part of a composition: an element or material by name, as a mass fraction or a number of atoms."""

    ref: str
    fraction: float | None = None
    atoms: int | None = None


@dataclass(frozen=True)
class Composition:
    """What a material is made of: a density (g/cm3) and its components, or a single element (Z, molar mass g/mol)."""

    density_g_cm3: float
    components: tuple[Component, ...] = ()
    z: float | None = None
    molar_mass_g_mol: float | None = None


@dataclass(frozen=True)
class OpticsModel:
    """The optics model that reads a material's or surface's properties, and its own settings.

    ``name`` is a registered optics model ("geant4": Geant4's reading of the properties); ``settings`` its
    model-specific choices (for a material, ``medium``: the medium it is bound to, when several share one). An
    ``origin`` of "recipe" marks a choice the recipe made, with its reason in ``note``.
    """

    name: str = "geant4"
    version: int = 1
    settings: dict = field(default_factory=dict)
    origin: str = "gdml"
    note: str = ""


@dataclass(frozen=True)
class Element:
    """A GDML element: its symbol, Z, and a molar mass (g/mol) or isotope fractions."""

    name: str
    formula: str
    z: float | None
    molar_mass_g_mol: float | None = None
    isotopes: tuple[Component, ...] = ()


@dataclass(frozen=True)
class Isotope:
    """A GDML isotope: Z, N and its molar mass (g/mol)."""

    name: str
    z: int
    n: int
    molar_mass_g_mol: float


@dataclass(frozen=True)
class Material:
    """A material: state, conditions, composition, its optical properties and the optics model that reads them."""

    name: str
    state: str | None
    composition: Composition
    temperature: float | None = None  # K
    pressure: float | None = None  # Pa
    properties: dict[str, Property] = field(default_factory=dict)
    optics: OpticsModel = OpticsModel()


@dataclass(frozen=True)
class Surface:
    """An optical surface: Geant4's (type, model, finish, value) verbatim, its properties, and its optics model.

    ``inert`` lists the keys read from the GDML that no Geant4 process reads on a surface (kept for export).
    """

    name: str
    type: str
    model: str
    finish: str
    value: float
    properties: dict[str, Property] = field(default_factory=dict)
    optics: OpticsModel = OpticsModel()
    inert: tuple[str, ...] = ()


@dataclass(frozen=True)
class Skin:
    """A skin surface: the surface around every placement of a logical volume."""

    name: str
    surface: str
    volume: str
    origin: str = "gdml"
    note: str = ""


@dataclass(frozen=True)
class Border:
    """A border surface, directional: from placement ``pv1`` into placement ``pv2`` (G4LogicalBorderSurface)."""

    name: str
    surface: str
    pv1: str
    pv2: str
    origin: str = "gdml"


@dataclass(frozen=True)
class DetectorFile:
    """A detector: the full GDML world, its materials and optics, surfaces, the recipe's decisions, provenance.

    ``detector_volume`` is the logical volume that is simulated (light leaving it is absorbed: ``outside``), and
    ``outside_surface`` the surface a photon meets on leaving it, when the recipe names one; ``wavelength_range``
    (nm) is where every optical table must be defined. ``meshes`` holds the tessellated solids'
    triangle arrays by name. ``provenance`` records the source file, its sha256, the importer and recipe, and
    ``report`` every normalisation and decision, in words.
    """

    name: str
    world: str
    detector_volume: str
    wavelength_range: tuple[float, float]
    solids: dict[str, Solid]
    volumes: dict[str, Volume]
    placements: tuple[Placement, ...]
    materials: dict[str, Material]
    elements: dict[str, Element] = field(default_factory=dict)
    isotopes: dict[str, Isotope] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    skins: tuple[Skin, ...] = ()
    borders: tuple[Border, ...] = ()
    meshes: dict[str, np.ndarray] = field(default_factory=dict)
    outside: str = OUTSIDE_ABSORB
    outside_surface: str | None = None
    userinfo: tuple[Aux, ...] = ()
    provenance: dict = field(default_factory=dict)
    report: tuple[str, ...] = ()
    schema: str = SCHEMA
