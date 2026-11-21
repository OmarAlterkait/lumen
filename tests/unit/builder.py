"""The tests' detector builder: detectors built in Python, without GDML, as the records an importer writes.

    from builder import Builder, table_nm, patterns
    b = Builder("ball", detector_volume="Ball")
    b.material("Water", density=1.0, atoms={"H": 2, "O": 1}, state="liquid", RINDEX=1.33, ABSLENGTH=30.0)
    b.material("Black", density=1.0, atoms={"C": 1})                       # no RINDEX: opaque
    b.box("WorldS", 3.0, 3.0, 3.0); b.solid("BallS", "orb", r=1.0); b.tube("PmtS", rmax=0.04, z=0.002)
    b.volume("World", "WorldS", "Black"); b.volume("Ball", "BallS", "Water"); b.volume("PMT", "PmtS", "Black")
    b.place("Ball", into="World", name="ball_pv")
    b.place_many("PMT", into="Ball", poses=patterns.sphere(64, 0.999), name="pmt_{copy}")
    b.surface("Cathode", "dielectric_metal", "glisur", "polished", 1.0, REFLECTIVITY=0.0, EFFICIENCY=1.0)
    b.border("ToPMT", "Cathode", "ball_pv", "pmt_*")                       # one border per matching placement
    b.sensors("pmt", "border_into", lv="PMT")
    builder.save(b, "ball.lumen")                                          # or b.structure(): (DetectorFile, SensorSet)

The builder holds no physics and lowers nothing: it writes ``lumen.format.schema`` records, exactly what the GDML
importer would write from the equivalent GDML (Geant4's solids and attribute spelling, lengths in m, angles in rad,
energies in eV), and ``structure`` refuses what ``lumen.format.check`` refuses. ``lumen.convert`` then applies the
general import rules and numbers the sensors, as for any source (``lumen.recipe.import_structure``).
``save`` converts and writes it as ``lumen.convert`` writes any source.

- **Names** are the user's and become the parameter keys (``materials/Water/ABSLENGTH``) and the sensor paths.
- **Solids** are Geant4's, by GDML's attributes (``lumen.format.solids``): ``solid(name, type, **attributes)``, with
  ``box`` and ``tube`` as shorthands; a box's sides and a tube's ``z`` are full lengths, as in GDML; an omitted
  angular range is the whole solid, stored explicitly (``lumen.format.solids.whole``).
- **Placements** take an active rotation matrix (the placed volume's axes in its mother: its columns), stored as
  Geant4's angles (``lumen.format.yaml.angles``); placements keep the order they are made in, which is Geant4's
  daughter order. ``patterns`` makes the poses of PMT arrays (local +z facing inward).
- **Materials** name their composition (a density in g/cm3 and ``atoms`` or mass ``fractions``; elements of
  ``lumen.format.elements.ELEMENTS`` need no definition). A property is a number (a constant) or a table
  (``table(energy_eV, values)``, ``table_nm(wavelength_nm, values)``), in its key's unit
  (``lumen.format.schema.PROPERTY_UNITS``: lengths in m, times in ns, GROUPVEL in m / ns, ISOTHERMAL_COMPRESSIBILITY
  in m3 / MeV, yields per MeV).
- **Sensors** are ``SensorGroup`` records (``lumen.format.sensorsets``), the detector's one sensor set.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import math

import numpy as np

from lumen import units
from lumen.format import elements, schema, sensorsets
from lumen.format import solids as catalogue
from lumen.format.check import POSE_ATOL, StructureError, check
from lumen.format.yaml import angles
from lumen.recipe import import_structure

IMPORTER = "tests.builder"
WAVELENGTH_NOTE = "energies from wavelengths (nm), hc = {hc} eV nm"


class BuildError(StructureError):
    """A detector the builder cannot make: a name given twice, a reference to nothing, a structural rule broken."""


def table(energy_ev, values, note: str = "") -> schema.Property:
    """Return a table property on strictly ascending photon energies (eV); its key is set where it is used."""
    return schema.Property("", np.asarray(values, np.float64), energy=np.asarray(energy_ev, np.float64), note=note)


def table_nm(wavelength_nm, values, note: str = "") -> schema.Property:
    """Return a table property given on wavelengths (nm), stored on ascending energies with ``lumen.units.HC``."""
    lam = np.asarray(wavelength_nm, np.float64)
    if np.any(lam <= 0.0):
        raise BuildError("table_nm: wavelengths are positive")
    e, v = units.HC / lam[::-1], np.asarray(values, np.float64)[::-1]
    return table(e, v, "; ".join(x for x in (note, WAVELENGTH_NOTE.format(hc=units.HC)) if x))


def _property(key: str, v) -> schema.Property:
    if isinstance(v, schema.Property):
        return dataclasses.replace(v, key=key)
    if np.ndim(v):
        raise BuildError(f"property {key}: a table is table(energy_eV, values) or table_nm(wavelength_nm, values)")
    return schema.Property(key, np.asarray(float(v), np.float64))


class Builder:
    """A detector under construction (see the module docstring)."""

    def __init__(self, name: str, detector_volume: str | None = None, world: str | None = None):
        """Start detector ``name``.

        ``detector_volume``: the volume simulated (the world when None); ``world``: the volume no placement places
        (found when None).
        """
        self.name = name
        self.detector_volume, self.world = detector_volume, world
        self.solids: dict = {}
        self.volumes: dict = {}
        self.placements: list = []
        self.materials: dict = {}
        self.elements: dict = {}
        self.surfaces: dict = {}
        self.skins: list = []
        self.borders: list = []
        self.groups: list = []

    @staticmethod
    def _new(table: dict, name: str, what: str) -> None:
        if name in table:
            raise BuildError(f"{what} {name!r} is defined twice")

    # ------------------------------------------------------------------------------------------- geometry
    def solid(self, name: str, type: str, **attributes) -> str:
        """Add a Geant4 solid by its GDML attributes (m, rad); return its name."""
        self._new(self.solids, name, "solid")
        self.solids[name] = schema.Solid(
            name, type, params=catalogue.whole(type, {k: float(v) for k, v in attributes.items()})
        )
        return name

    def box(self, name: str, x: float, y: float, z: float) -> str:
        """Add a box of full side lengths x, y, z (m)."""
        return self.solid(name, "box", x=x, y=y, z=z)

    def tube(self, name: str, rmax: float, z: float, rmin: float = 0.0) -> str:
        """Add a full-turn tube of radii rmin, rmax and full length z (m)."""
        return self.solid(name, "tube", rmin=rmin, rmax=rmax, z=z)

    def volume(self, name: str, solid: str, material: str) -> str:
        """Add a logical volume of a solid and a material; return its name."""
        self._new(self.volumes, name, "volume")
        self.volumes[name] = (solid, material)
        return name

    def place(self, volume: str, into: str, name: str, position=(0.0, 0.0, 0.0), rotation=None,
              copy: int | None = None) -> str:  # fmt: skip
        """Place a volume in its mother ``into``; ``rotation``: the active 3x3 matrix (its columns, the placed
        volume's axes in the mother), None for none. Return the placement's name."""
        rot = (0.0, 0.0, 0.0)
        if rotation is not None:
            m = np.asarray(rotation, np.float64)
            if m.shape != (3, 3) or not (np.allclose(m @ m.T, np.eye(3), atol=POSE_ATOL) and np.linalg.det(m) > 0):
                raise BuildError(f"placement {name!r}: a rotation is an orthonormal 3x3 matrix with determinant +1")
            rot = angles(m.T)  # GDML places with the inverse (lumen.format.yaml.angles)
        self.placements.append(
            schema.Placement(name, volume, into, copy=copy, position=tuple(float(x) for x in position), rotation=rot)
        )
        return name

    def place_many(self, volume: str, into: str, poses, name: str = "{volume}_{copy}") -> list[str]:
        """Place a volume at each pose (rotation matrix, position) of ``poses``; ``name`` is formatted with
        ``volume`` and ``copy`` (the pose's index, also the copy number). Return the names."""
        return [self.place(volume, into, name.format(volume=volume, copy=k), t, r, k)
                for k, (r, t) in enumerate(poses)]  # fmt: skip

    # ------------------------------------------------------------------------------------------- materials
    def material(self, name: str, *, density: float, atoms: dict | None = None,
                 fractions: dict | None = None, state: str | None = None, temperature: float | None = None,
                 pressure: float | None = None, **properties) -> str:  # fmt: skip
        """Add a material: its density (g/cm3), composition (``atoms`` {ref: count} or mass ``fractions`` {ref:
        fraction}; a ref is a material, an element of ``element``, or a symbol of ``lumen.format.elements``) and its
        optical properties by Geant4 key (a number or a table). Return its name."""
        self._new(self.materials, name, "material")
        if (atoms is None) == (fractions is None):
            raise BuildError(f"material {name!r}: give its composition as atoms or as fractions")
        comps = tuple(schema.Component(r, atoms=int(n)) for r, n in (atoms or {}).items()) + \
            tuple(schema.Component(r, fraction=float(f)) for r, f in (fractions or {}).items())  # fmt: skip
        props = {k: _property(k, v) for k, v in properties.items()}
        self.materials[name] = schema.Material(
            name,
            state=state,
            composition=schema.Composition(float(density), components=comps),
            temperature_k=temperature,
            pressure_pa=pressure,
            properties=props,
        )
        return name

    def element(self, name: str, z: float, molar_mass_g_mol: float, formula: str | None = None) -> str:
        """Add an element (Z, molar mass g/mol) a composition may name; return its name."""
        self._new(self.elements, name, "element")
        self.elements[name] = schema.Element(
            name, formula=formula or name, z=float(z), molar_mass_g_mol=float(molar_mass_g_mol)
        )
        return name

    # -------------------------------------------------------------------------------------------- surfaces
    def surface(self, name: str, type: str, model: str, finish: str, value: float, **properties) -> str:
        """Add an optical surface by Geant4's four attributes and its properties by key; return its name."""
        self._new(self.surfaces, name, "surface")
        props = {k: _property(k, v) for k, v in properties.items()}
        self.surfaces[name] = schema.Surface(
            name, type=type, model=model, finish=finish, value=float(value), properties=props
        )
        return name

    def skin(self, name: str, surface: str, volume: str) -> None:
        """Bind a surface around every placement of a logical volume (G4LogicalSkinSurface)."""
        self.skins.append(schema.Skin(name, surface=surface, volume=volume))

    def border(self, name: str, surface: str, pv1: str, pv2: str) -> None:
        """Bind a surface from placement ``pv1`` into ``pv2`` (G4LogicalBorderSurface). Either may be a pattern
        (``fnmatch``: ``"pmt_*"``): one border per matching pair, named ``<name>:<k>``."""
        names = [p.name for p in self.placements]
        a, b = ([n for n in names if fnmatch.fnmatchcase(n, x)] for x in (pv1, pv2))
        if not a or not b:
            raise BuildError(f"border {name!r}: {pv1 if not a else pv2!r} matches no placement")
        pairs = [(x, y) for x in a for y in b]
        for k, (x, y) in enumerate(pairs):
            self.borders.append(
                schema.Border(name if len(pairs) == 1 else f"{name}:{k}", surface=surface, pv1=x, pv2=y)
            )

    # --------------------------------------------------------------------------------------------- sensors
    def sensors(self, name: str, rule: str, **fields) -> None:
        """Add a sensor group (``lumen.format.sensorsets.SensorGroup``: ``lv`` or ``lv_pattern``, ``values``,
        ``readout``, ...) to the sensor set."""
        self.groups.append(sensorsets.SensorGroup(name, rule=rule, **fields))

    # ----------------------------------------------------------------------------------------------- build
    def structure(self):
        """Return ``(DetectorFile, SensorSet)``, checked (``lumen.format.check``; ``BuildError``).

        The sensor set has no ids yet: ``lumen.convert`` numbers them (``lumen.recipe.import_structure``).
        """
        world, detector_volume = self.world, self.detector_volume
        placed = {p.volume for p in self.placements}
        if world is None:
            roots = [v for v in self.volumes if v not in placed]
            if len(roots) != 1:
                raise BuildError(f"the world is the one volume no placement places; found {roots}: pass world=")
            world = roots[0]
        pls = _gdml_order(self.placements, world)
        daughters: dict = {}
        for i, p in enumerate(pls):
            daughters.setdefault(p.mother, []).append(i)
        vols = {n: schema.Volume(n, s, m, daughters=tuple(daughters.get(n, ()))) for n, (s, m) in self.volumes.items()}
        df = schema.DetectorFile(
            name=self.name, world=world, detector_volume=detector_volume or world,
            solids=dict(self.solids), volumes=vols,
            placements=pls, materials=dict(self.materials), elements=dict(self.elements),
            surfaces=dict(self.surfaces), skins=tuple(self.skins), borders=tuple(self.borders),
            provenance=dict(importer=IMPORTER, source=dict(file=f"<builder {self.name}>")),
        )  # fmt: skip
        df = elements.with_elements(df)
        check(df, BuildError)
        return df, sensorsets.SensorSet(tuple(self.groups))


def _gdml_order(placements: list, world: str) -> tuple:
    """Return the placements in the order a GDML file holds them (the GDML importer's): each volume's daughters
    together, in the order they were made, a volume's after those of the volumes placed in it (depth first from the
    world, in placement order)."""
    by: dict = {}
    for p in placements:
        by.setdefault(p.mother, []).append(p)
    order: list = []
    seen: set = set()
    stack = [(world, False)]
    while stack:
        v, done = stack.pop()
        if done:
            order.append(v)
            continue
        if v in seen:
            continue
        seen.add(v)
        stack.append((v, True))
        stack += [(p.volume, False) for p in reversed(by.get(v, ()))]
    order += [v for v in by if v not in seen]
    return tuple(p for v in order for p in by.get(v, ()))


# ======================================================================================================= patterns
def facing(n) -> np.ndarray:
    """Return a rotation whose local +z is the unit vector along ``n`` (columns: the local axes; x chosen as the
    component of the world x, or y, orthogonal to n)."""
    z = np.asarray(n, np.float64) / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = a - (a @ z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


class patterns:  # noqa: N801 -- a namespace of functions, used as builder.patterns.sphere(...)
    """Poses (rotation, position) of sensor arrays, each with the local +z facing inward (or as ``facing`` says)."""

    @staticmethod
    def sphere(n: int, radius: float) -> list:
        """Return ``n`` poses on a sphere of ``radius`` (m) about the origin (Fibonacci lattice), facing its centre."""
        out = []
        for k in range(n):
            z = 1.0 - (2.0 * k + 1.0) / n
            r, phi = math.sqrt(max(0.0, 1.0 - z * z)), k * math.pi * (3.0 - math.sqrt(5.0))
            p = radius * np.array([r * math.cos(phi), r * math.sin(phi), z])
            out.append((facing(-p), p))
        return out

    @staticmethod
    def ring(n: int, radius: float, z: float) -> list:
        """Return ``n`` poses on a circle of ``radius`` (m) at height ``z``, facing the axis."""
        out = []
        for k in range(n):
            phi = units.TWO_PI * k / n
            p = np.array([radius * math.cos(phi), radius * math.sin(phi), z])
            out.append((facing(-p * np.array([1.0, 1.0, 0.0])), p))
        return out

    @staticmethod
    def cylinder(radius: float, zs, n_per_ring: int) -> list:
        """Return the poses of rings of ``n_per_ring`` at heights ``zs`` on a cylinder wall, facing the axis."""
        return [pose for z in zs for pose in patterns.ring(n_per_ring, radius, float(z))]

    @staticmethod
    def disk(radius: float, pitch: float, z: float, facing_z: float) -> list:
        """Return the poses of a square grid of ``pitch`` (m) clipped to a disk of ``radius`` at height ``z``, the
        local +z along ``facing_z`` (+1 or -1) times the world z."""
        k = int(radius // pitch)
        r = facing((0.0, 0.0, facing_z))
        return [(r, np.array([i * pitch, j * pitch, z])) for j in range(-k, k + 1) for i in range(-k, k + 1)
                if math.hypot(i * pitch, j * pitch) <= radius]  # fmt: skip

    @staticmethod
    def grid(nx: int, ny: int, pitch: float, z: float, facing_z: float) -> list:
        """Return the poses of an nx x ny grid of ``pitch`` (m) centred at height ``z``, facing ``facing_z`` z."""
        r = facing((0.0, 0.0, facing_z))
        return [(r, np.array([(i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch, z]))
                for j in range(ny) for i in range(nx)]  # fmt: skip


__all__ = ["BuildError", "Builder", "facing", "patterns", "table", "table_nm"]


def save(b: Builder, out):
    """Convert a builder's structure and write it at ``out`` as ``lumen.convert`` writes a detector; return the path."""
    from pathlib import Path

    from lumen import api

    return api._place(api._placement_report(import_structure(*b.structure())), Path(out), "raise")
