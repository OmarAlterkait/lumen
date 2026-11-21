"""The detector file's records: what Geant4 reads from the GDML, plus the recipe's decisions, with provenance.

Units are m, rad, ns, nm and eV everywhere (a property's value: ``PROPERTY_UNITS``); fields whose unit is not obvious
carry it in their name. The records are
frozen and hold plain numbers, strings and numpy arrays, so a detector file is a value: it is written and read back
bit for bit (``lumen.format.io``).

The geometry keeps the FULL GDML world (the detector volume is cut out at load) in Geant4's conventions:

- a placement's rotation is stored as the GDML physvol angles (passive, rad) and its position in m; the matrix is
  derived (``lumen.build``), never stored, since angles to matrix to angles is not exact;
- a boolean's or multi-union's transforms are stored the same way (GDML's active convention for booleans);
- solid attributes keep Geant4's GDML spelling, exactly (``lumen.format.solids``), in m and rad;
- physical-volume names are always explicit (an unnamed physvol gets Geant4's ``<lv>_PV``).

A record's ``origin`` says who decided it: "source" (the detector's source: a GDML or YAML file, a builder),
"recipe", "import" (an import rule) or "geant4_default" (a value added at a Geant4 default key).
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

import numpy as np

SCHEMA = "lumen-detector/6"  # the one version of a detector directory and its sensor sets: a file of another version
# is refused (convert the detector again)


@dataclass(frozen=True)
class Transform:
    """A GDML transform: position (m) and rotation angles (rad, GDML order x, y, z)."""

    _: KW_ONLY
    position: tuple[float, ...] = (0.0, 0.0, 0.0)
    rotation: tuple[float, ...] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Solid:
    """A GDML solid in GDML's spelling.

    ``params`` holds the numeric attributes (Geant4's spelling, ``lumen.format.solids``; m, rad or unitless).
    ``refs`` and ``transforms`` are a boolean's (first, second), a multi-union's nodes or a scaled solid's child;
    ``scale`` is a scaled solid's. ``planes`` holds a polycone's or polyhedra's (rmin, rmax, z) planes, ``vertices``
    an extrusion's (x, y) or a tetrahedron's four (x, y, z) points, ``sections`` an extrusion's (z, x offset, y offset,
    scale); ``mesh`` names the array of a tessellated solid's triangles (k, 3, 3), m.
    """

    name: str
    type: str
    _: KW_ONLY
    params: dict[str, float] = field(default_factory=dict)
    refs: tuple[str, ...] = ()
    transforms: tuple[Transform, ...] = ()
    scale: tuple[float, ...] | None = None
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
    _: KW_ONLY
    daughters: tuple[int, ...] = ()
    assembly: bool = False


@dataclass(frozen=True)
class Placement:
    """A physical volume: the volume it places, its mother volume, its name, copy number, pose and scale."""

    name: str
    volume: str
    mother: str
    _: KW_ONLY
    copy: int | None
    position: tuple[float, ...]  # m, in the mother's frame
    rotation: tuple[float, ...]  # rad, GDML physvol angles (passive)
    scale: tuple[float, ...] | None = None


# The unit every property key with a dimension is stored in (lumen's: m, ns, eV for a photon, MeV for a deposited
# energy), and the power of ten that takes Geant4's internal unit (mm, ns, MeV), the unit a GDML matrix holds, into it.
# Every reader converts into these units (the GDML importer, ``lumen.format.yaml``) and the exporter back; a builder
# property is already in them. A key not listed is dimensionless (an index, a fraction, a relative spectrum or yield).
LENGTH_KEYS = ("ABSLENGTH", "RAYLEIGH", "WLSABSLENGTH", "WLSABSLENGTH2", "MIEHG", "COATEDTHICKNESS")
TIME_KEYS = ("WLSTIMECONSTANT", "WLSTIMECONSTANT2", "FASTTIMECONSTANT", "SLOWTIMECONSTANT", "FASTSCINTILLATIONRISETIME",
             "SLOWSCINTILLATIONRISETIME", *(f"SCINTILLATIONTIMECONSTANT{i}" for i in (1, 2, 3)),
             *(f"SCINTILLATIONRISETIME{i}" for i in (1, 2, 3)))  # fmt: skip
PROPERTY_UNITS: dict[str, tuple[str, str, int]] = {  # key -> (quantity, unit, power of ten from Geant4's unit)
    **dict.fromkeys(LENGTH_KEYS, ("length", "m", -3)),
    **dict.fromkeys(TIME_KEYS, ("time", "ns", 0)),
    "GROUPVEL": ("velocity", "m/ns", -3),
    "ISOTHERMAL_COMPRESSIBILITY": ("compressibility", "m3/MeV", -9),
    "SCINTILLATIONYIELD": ("yield", "1/MeV", 0),
}


@dataclass(frozen=True)
class Property:
    """A material or surface property as Geant4 reads it: a table linear in photon energy, or a constant.

    ``energy`` (eV, ascending) is None for a constant, whose ``value`` is a scalar (a 0-d array). Values are in the
    unit ``PROPERTY_UNITS`` gives the key (lengths in m, times in ns, GROUPVEL in m / ns, ...). ``origin``: "source",
    or who added it ("recipe": a table the source lacks, with where it comes from in ``note``; the module docstring).
    """

    key: str
    value: np.ndarray
    _: KW_ONLY
    energy: np.ndarray | None = None
    origin: str = "source"
    note: str = ""


COMPONENT_KINDS = ("element", "material", "isotope")


@dataclass(frozen=True)
class Component:
    """One part of a composition: an element, material or isotope by name (``kind``: which table ``ref`` names), as a
    mass fraction or a number of atoms. The GDML importer resolves the kind as Geant4 does (``lumen.gdml.importer``)."""  # noqa: D205, D209

    ref: str
    _: KW_ONLY
    fraction: float | None = None
    atoms: int | None = None
    kind: str = ""  # "": not yet resolved (a YAML or builder record: ``lumen.format.elements.with_elements``)


@dataclass(frozen=True)
class Composition:
    """What a material is made of: a density (g/cm3) and its components, or a single element (Z, molar mass g/mol)."""

    density_g_cm3: float
    _: KW_ONLY
    components: tuple[Component, ...] = ()
    z: float | None = None
    molar_mass_g_mol: float | None = None


def _number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _vector(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3 and all(_number(x) for x in v)


VALUE_TYPES = {
    "a name": lambda v: isinstance(v, str),
    "a number": _number,
    "true or false": lambda v: isinstance(v, bool),
    "a 3-vector": _vector,
}
# the settings an optics model records (lumen.build.library reads them), by kind
OPTICS_SETTINGS = {
    "wls_time_profile": "a name", "scint_finite_rise_time": "true or false",
}  # fmt: skip


def check_settings(what: str, settings: dict, known: dict) -> None:
    """Refuse a setting that is not one of ``known`` (key -> kind) or not of its kind."""
    for k, v in settings.items():
        if k not in known:
            raise ValueError(f"{what}: unknown setting {k!r} (known: {sorted(known)})")
        if not VALUE_TYPES[known[k]](v):
            raise ValueError(f"{what}: setting {k!r} must be {known[k]}, got {v!r}")


@dataclass(frozen=True)
class OpticsModel:
    """The optics model that reads a material's or surface's properties, and its own settings.

    ``name`` is a registered optics model ("geant4": Geant4's reading of the properties); ``settings`` its choices,
    each one of ``OPTICS_SETTINGS`` and of its kind (for a material, ``wls_time_profile``: G4OpWLS's time profile in
    the Geant4 run, "delta" or "exponential"; absent: Geant4's default, "delta"; ``scint_finite_rise_time``: Geant4's
    finite-rise-time switch of G4Scintillation, true or false; absent: Geant4's default, false). An ``origin`` of
    "recipe" marks a choice the recipe made, with its reason in ``note``.
    """

    name: str = "geant4"
    _: KW_ONLY
    settings: dict = field(default_factory=dict)
    origin: str = "source"
    note: str = ""

    def __post_init__(self):
        """Check the settings."""
        check_settings(f"optics model {self.name!r}", self.settings, OPTICS_SETTINGS)


@dataclass(frozen=True)
class Element:
    """A GDML element: its symbol, Z, and a molar mass (g/mol) or isotope fractions."""

    name: str
    _: KW_ONLY
    formula: str
    z: float | None
    molar_mass_g_mol: float | None = None
    isotopes: tuple[Component, ...] = ()


@dataclass(frozen=True)
class Isotope:
    """A GDML isotope: Z, N and its molar mass (g/mol)."""

    name: str
    _: KW_ONLY
    z: int
    n: int
    molar_mass_g_mol: float


@dataclass(frozen=True)
class Material:
    """A material: state, conditions, composition, its optical properties and the optics model that reads them.

    ``inert`` keeps, verbatim (Geant4's internal units), the matrices of the properties the recipe declared
    non-optical (read by another process, such as an experiment's own scintillation): never read by lumen, written
    back on export. ``mee_ev`` is the GDML's <MEE> (the mean excitation energy Geant4's ionisation uses), carried for
    export.
    """

    name: str
    _: KW_ONLY
    state: str | None
    composition: Composition
    temperature_k: float | None = None
    pressure_pa: float | None = None
    properties: dict[str, Property] = field(default_factory=dict)
    optics: OpticsModel = OpticsModel()
    inert: dict[str, np.ndarray] = field(default_factory=dict)  # matrices no optical process reads, as read
    mee_ev: float | None = None  # the mean excitation energy (eV) the GDML gives, if any


@dataclass(frozen=True)
class Surface:
    """An optical surface: Geant4's (type, model, finish, value) verbatim, its properties, and its optics model."""

    name: str
    _: KW_ONLY
    type: str
    model: str
    finish: str
    value: float
    properties: dict[str, Property] = field(default_factory=dict)
    optics: OpticsModel = OpticsModel()


@dataclass(frozen=True)
class Skin:
    """A skin surface: the surface around every placement of a logical volume."""

    name: str
    _: KW_ONLY
    surface: str
    volume: str
    origin: str = "source"
    note: str = ""


@dataclass(frozen=True)
class Border:
    """A border surface, directional: from placement ``pv1`` into placement ``pv2`` (G4LogicalBorderSurface)."""

    name: str
    _: KW_ONLY
    surface: str
    pv1: str
    pv2: str


@dataclass(frozen=True)
class DetectorFile:
    """A detector: the full world, its materials and optics, surfaces, the recipe's decisions, provenance.

    ``detector_volume`` is the volume that is simulated (light leaving it is absorbed, after the surface Geant4 finds
    there: ``lumen.build.binding.outside_volume``): a logical volume placed once, or the physvol name or path that
    selects one placement (``placements_of``). Every optical table is read at any wavelength as
    Geant4 reads it (linear in energy, held at its end values), and checked at its own knots. The build's rewrites
    are rules, not options (``lumen.build.import_rules``). ``meshes`` holds the tessellated solids' triangle arrays by
    name.
    ``provenance`` records the source file, its sha256, the importer and recipe, and ``report`` every normalisation
    and decision, in words.
    """

    name: str = field(metadata={"file": "detector.json"})
    _: KW_ONLY
    world: str = field(metadata={"file": "volumes.json"})
    detector_volume: str = field(metadata={"file": "volumes.json"})
    solids: dict[str, Solid] = field(metadata={"file": "solids.json"})
    volumes: dict[str, Volume] = field(metadata={"file": "volumes.json"})
    placements: tuple[Placement, ...] = field(metadata={"file": "placements.json"})
    materials: dict[str, Material] = field(metadata={"file": "materials.json"})
    elements: dict[str, Element] = field(default_factory=dict, metadata={"file": "elements.json"})
    isotopes: dict[str, Isotope] = field(default_factory=dict, metadata={"file": "elements.json"})
    surfaces: dict[str, Surface] = field(default_factory=dict, metadata={"file": "surfaces.json"})
    skins: tuple[Skin, ...] = field(default=(), metadata={"file": "surfaces.json"})
    borders: tuple[Border, ...] = field(default=(), metadata={"file": "surfaces.json"})
    meshes: dict[str, np.ndarray] = field(default_factory=dict, metadata={"file": "solids.json"})
    provenance: dict = field(default_factory=dict, metadata={"file": "detector.json"})
    report: tuple[str, ...] = field(default=(), metadata={"file": "detector.json"})


def placements_of(df: DetectorFile, spec: str) -> list[tuple[int, ...]]:
    """Return every placement chain from the world (placement indices, depth-first) that ``spec`` selects.

    ``spec``: a logical volume (each of its world instances), a physvol name (each instance of that placement) or a
    physvol path, its names from the world's daughter down joined by "/" (one instance). The world itself has none.
    """
    if "/" in spec:
        chain: tuple[int, ...] = ()
        names, lv = spec.split("/"), df.world
        for name in names:
            js = [j for j in df.volumes[lv].daughters if df.placements[j].name == name]
            if len(js) != 1:
                return []
            chain, lv = chain + (js[0],), df.placements[js[0]].volume
        return [chain]
    by_lv = spec in df.volumes
    out: list[tuple[int, ...]] = []

    def walk(lv, chain):
        for j in df.volumes[lv].daughters:
            c = chain + (j,)
            p = df.placements[j]
            if (p.volume if by_lv else p.name) == spec:
                out.append(c)
            walk(p.volume, c)

    walk(df.world, ())
    return out


def placement_path(df: DetectorFile, chain: tuple[int, ...]) -> str:
    """Return a placement chain's physvol path: its names from the world's daughter down, joined by "/"."""
    return "/".join(df.placements[j].name for j in chain)


def detector_lv(df: DetectorFile) -> str:
    """Return the logical volume of the detector volume (``detector_volume``: a logical volume, or the physvol name or
    path that selects one of its placements)."""  # noqa: D205, D209
    if df.detector_volume in df.volumes:
        return df.detector_volume
    chains = placements_of(df, df.detector_volume)
    if not chains:
        raise ValueError(f"the detector volume {df.detector_volume!r} is no volume, physvol or physvol path placed "
                         "under the world")  # fmt: skip
    return df.placements[chains[0][-1]].volume
