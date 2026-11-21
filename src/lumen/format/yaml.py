"""A detector structure written by hand in YAML: an input, converted, never loaded (lumen writes no YAML).

    df, sensor_set = lumen.format.yaml.read("toy/")                     # a directory of .yaml files, or the files

``lumen.convert`` turns YAML into a detector file (``lumen.recipe.import_yaml``); ``lumen.load`` reads only the
``.lumen`` directory, because YAML reads 25-30 times slower than its JSON.

**The mapping is the records' own, one to one.** Each top-level section is a field of ``lumen.format.schema
.DetectorFile`` (but its ``provenance`` and ``report``, which conversion writes), plus ``sensors`` (the sensor set's
groups, a list, ``lumen.format.sensorsets``; the sensors are numbered at conversion). A record
is a mapping of its fields by their names; a record kept by name (a solid, a material, ...) is keyed by its name; a
field at its default may be left out. An array may be given in a ``.npy`` file beside the YAML (``{npy: <path>}``).

**Files combine.** A detector may be split across files (``geometry.yaml``, ``materials.yaml``, ...), and a file need
not hold everything: named records merge by name (one name defined twice is refused), lists (placements, skins,
borders) join in file order, and ``name``, ``world`` and ``detector_volume`` may be given once (or repeated with one
value).

**What may be left out** (read as):

- ``name``: the first file's name (without its suffix), or its directory's;
- ``world``: the one volume that no placement places;
- ``detector_volume``: the world;
- a placement's ``position`` and ``rotation``: zero; ``copy``: none;
- a solid's angular range (``startphi``, ``deltaphi``, ``starttheta``, ``deltatheta``): the whole solid, stored
  explicitly (``lumen.format.solids.whole``; a GDML's omitted attribute is Geant4's 0);
- an optics model: Geant4's (``geant4``);
- an element of ``lumen.format.elements.ELEMENTS`` (H, C, N, O, ...) a composition names: its symbol, Z and
  standard atomic weight;

A component names an element first, else a material, as Geant4 reads a GDML fraction
(``lumen.format.elements.with_elements``); a volume's daughters are its placements, in the order they are listed.

**Input written by hand** is resolved once to the canonical values:

- a number written with its unit, ``"12.5*mm"``, ``"2.5*eV"``, ``"90*deg"``, read by exact decimal with one rounding
  (``lumen.units.quantity``); a bare number is in lumen's unit (m, eV, ns, rad; a property in its key's unit,
  ``lumen.format.schema.PROPERTY_UNITS``: GROUPVEL in m / ns, ISOTHERMAL_COMPRESSIBILITY in m3 / MeV, ...);
- a table on wavelengths (``wavelength_nm:`` instead of ``energy:``), converted to ascending photon energies with
  ``lumen.units.HC`` (recorded in the property's note);
- a constant property as a bare number (``RINDEX: 1.33``);
- a rotation as a matrix (``rotation: {matrix: [[...], [...], [...]]}``: the rotation that takes the placed volume's
  frame, or a boolean's second solid's, into its mother's), stored as Geant4's angles.

The result passes the structural rules every input meets (``lumen.format.check``). Every error is a ``YAMLError``
that names the file or the record and the field.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
import typing
from pathlib import Path

import numpy as np
import yaml as pyyaml  # type: ignore[import-untyped]

from lumen import units
from lumen.format import elements, io, schema, sensorsets
from lumen.format import solids as catalogue
from lumen.format.check import GIMBAL_COS, POSE_ATOL, check

SCHEMA = "lumen-detector-yaml/3"  # the first line of every file, written by hand too (a file without one is refused)
_TYPES = {
    k: h for k, h in io.field_types(schema.DetectorFile).items() if k not in ("provenance", "report")
}  # the sections: DetectorFile's fields, by kind, and the sensor set
HEAD = tuple(k for k, h in _TYPES.items() if h is str)  # given once
NAMED = tuple(k for k, h in _TYPES.items() if h is dict or typing.get_origin(h) is dict)  # merged by name
LISTS = (*(k for k, h in _TYPES.items() if typing.get_origin(h) is tuple), "sensors")  # joined in file order
SECTIONS = ("schema", *HEAD, *NAMED, *LISTS)
_LOADER = getattr(pyyaml, "CSafeLoader", pyyaml.SafeLoader)


class YAMLError(ValueError):
    """A YAML detector that cannot be read: its syntax, an unknown or missing field, a bad value, a broken rule."""


# ====================================================================================================== reading
def sources(paths) -> list[Path]:
    """Return the YAML files of ``paths`` (files, or directories read as their .yaml and .yml files, sorted)."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix in (".yaml", ".yml"))
        elif p.is_file():
            out.append(p)
        else:
            raise YAMLError(f"{p}: no such file or directory")
    if not out:
        raise YAMLError(f"no YAML files in {[str(p) for p in paths]}")
    return out


def _arrays(v, base: Path, where: str):
    """Return plain data with each ``{npy: <path>}`` replaced by the array (the path relative to its file)."""
    if isinstance(v, dict):
        if set(v) == {"npy"}:
            f = base / str(v["npy"])
            if not f.is_file():
                raise YAMLError(f"{where}: array file {v['npy']!r} not found beside the YAML")
            return np.load(f, allow_pickle=False)
        return {k: _arrays(x, base, f"{where}.{k}") for k, x in v.items()}
    if isinstance(v, list):
        return [_arrays(x, base, where) for x in v]
    return v


def _load(f: Path) -> dict:
    try:
        doc = pyyaml.load(f.read_text(), Loader=_LOADER)
    except pyyaml.YAMLError as err:
        raise YAMLError(f"{f.name}: not valid YAML: {err}") from None
    doc = {} if doc is None else doc
    if not isinstance(doc, dict):
        raise YAMLError(f"{f.name}: a detector file is a mapping of sections ({', '.join(SECTIONS)})")
    _known(doc, SECTIONS, f.name)
    if "schema" not in doc:
        raise YAMLError(f"{f.name}: no schema line: a detector file starts with 'schema: {SCHEMA}'")
    if doc["schema"] != SCHEMA:
        raise YAMLError(f"{f.name}: written by another lumen (schema {doc['schema']!r}); this lumen reads {SCHEMA}")
    return _arrays(doc, f.parent, f.name)


def _known(d: dict, known, where: str) -> None:
    for k in d:
        if k not in known:
            near = difflib.get_close_matches(str(k), [str(x) for x in known], n=1)
            hint = f" (did you mean {near[0]!r}?)" if near else f" (known: {', '.join(map(str, known))})"
            raise YAMLError(f"{where}: unknown field {k!r}{hint}")


def merge(files) -> dict:
    """Return the sections of several YAML files combined (see the module docstring), with each file's name kept."""
    out: dict = {"_files": [f.name for f in files]}
    where: dict = {}
    for f in files:
        doc = _load(f)
        for k, v in doc.items():
            if k == "schema":
                continue
            if k in HEAD:
                if k in out and out[k] != v:
                    raise YAMLError(f"{k} is {out[k]!r} in {where[k]} and {v!r} in {f.name}")
                out[k], where[k] = v, f.name
            elif k in LISTS:
                if not isinstance(v, list):
                    raise YAMLError(f"{f.name}: {k} is a list")
                out.setdefault(k, []).extend(v)
            else:
                if not isinstance(v, dict):
                    raise YAMLError(f"{f.name}: {k} is a mapping by name")
                into = out.setdefault(k, {})
                for name, rec in v.items():
                    if name in into:
                        raise YAMLError(f"{k} {name!r} is defined in {where[(k, name)]} and in {f.name}")
                    into[name], where[(k, name)] = rec, f.name
    return out


class _Reader:
    """Builds the records from merged plain sections, naming the record and field of any bad value."""

    def __init__(self, sections: dict):
        self.s = sections

    # ------------------------------------------------------------------------------------------- values
    def fields(self, d, where: str, required=(), optional=()) -> dict:
        if not isinstance(d, dict):
            raise YAMLError(f"{where}: a mapping of fields ({', '.join((*required, *optional))}), got {d!r}")
        _known(d, (*required, *optional), where)
        missing = [k for k in required if k not in d]
        if missing:
            raise YAMLError(f"{where}: missing {', '.join(missing)}")
        return d

    @staticmethod
    def num(v, what: str, where: str, unit: str | None = None) -> float:
        try:
            return units.quantity(v, what, unit)
        except (ValueError, TypeError) as err:
            raise YAMLError(f"{where}: {err}") from None

    def plain(self, v, where: str) -> float:
        """Return a number with no unit (a ratio, a fraction, Z, a molar mass), given as a number or its text."""
        if isinstance(v, (list, tuple, dict, bool)) or v is None:
            raise YAMLError(f"{where}: a plain number (no unit), got {v!r}")
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise YAMLError(f"{where}: a plain number (no unit), got {v!r}") from None
        return self.num(v, "length", where)

    def plains(self, v, where: str, n: int | None = None) -> tuple:
        if not isinstance(v, (list, tuple, np.ndarray)) or (n is not None and len(v) != n):
            raise YAMLError(f"{where}: {n or 'a list of'} numbers, got {v!r}")
        return tuple(self.plain(x, where) for x in v)

    def nums(self, v, what: str, where: str, n: int | None = None, unit: str | None = None) -> tuple:
        if not isinstance(v, (list, tuple, np.ndarray)) or (n is not None and len(v) != n):
            raise YAMLError(f"{where}: {n or 'a list of'} numbers, got {v!r}")
        return tuple(self.num(x, what, where, unit) for x in v)

    @staticmethod
    def text(v, where: str) -> str:
        if not isinstance(v, str):
            raise YAMLError(f"{where}: a name (text), got {v!r}")
        return v

    @staticmethod
    def integer(v, where: str) -> int:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise YAMLError(f"{where}: a whole number, got {v!r}")
        return int(v)

    def opt(self, d: dict, k: str, fn, default=None):
        return default if d.get(k) is None else fn(d[k])

    # ---------------------------------------------------------------------------------------- rotations
    def rotation(self, v, where: str, placement: bool) -> tuple:
        """Return Geant4's angles of a rotation: three angles, or ``{matrix: 3x3}`` (see the module docstring)."""
        if isinstance(v, dict):
            self.fields(v, where, ("matrix",))
            m = np.array([self.plains(r, f"{where}.matrix", 3) for r in self.listed(v["matrix"], where, 3)])
            if not (np.allclose(m @ m.T, np.eye(3), atol=POSE_ATOL) and np.linalg.det(m) > 0):
                raise YAMLError(f"{where}: a rotation matrix is orthonormal with determinant +1 (a reflection is a "
                                "placement's scale), got {m.tolist()}")  # fmt: skip
            return angles(m.T if placement else m)
        return self.nums(v, "angle", where, 3)

    def listed(self, v, where: str, n: int) -> list:
        if not isinstance(v, (list, tuple)) or len(v) != n:
            raise YAMLError(f"{where}: {n} rows, got {v!r}")
        return list(v)

    # ------------------------------------------------------------------------------------------ geometry
    def solid(self, name: str, d) -> schema.Solid:
        w = f"solids.{name}"
        d = self.fields(d, w, ("type",), ("params", "refs", "transforms", "scale", "planes", "vertices", "sections",
                                          "mesh"))  # fmt: skip
        kind = self.text(d["type"], f"{w}.type")
        if kind not in catalogue.ATTRIBUTES:
            near = difflib.get_close_matches(kind, list(catalogue.ATTRIBUTES), n=1)
            raise YAMLError(f"{w}.type: {kind!r} is not a solid lumen reads" + (f" (did you mean {near[0]!r}?)"
                                                                                  if near else ""))  # fmt: skip
        dims = catalogue.dimensions(kind)
        params = d.get("params") or {}
        if not isinstance(params, dict):
            raise YAMLError(f"{w}.params: a mapping of attributes ({', '.join(dims)})")
        _known(params, list(dims), f"{w}.params")
        quantity = {"L": "length", "A": "angle"}
        p = {k: self.num(v, quantity[dims[k]], f"{w}.params.{k}") if dims[k] in quantity
             else self.plain(v, f"{w}.params.{k}") for k, v in params.items()}  # fmt: skip
        return schema.Solid(
            name,
            kind,
            params=catalogue.whole(kind, p),
            refs=tuple(self.text(r, f"{w}.refs") for r in d.get("refs") or ()),
            transforms=tuple(
                self.transform(t, f"{w}.transforms[{i}]") for i, t in enumerate(d.get("transforms") or ())
            ),
            scale=self.opt(d, "scale", lambda v: self.plains(v, f"{w}.scale", 3)),
            planes=tuple(self.nums(x, "length", f"{w}.planes") for x in d.get("planes") or ()),
            vertices=tuple(self.nums(x, "length", f"{w}.vertices") for x in d.get("vertices") or ()),
            sections=tuple(self.section(x, f"{w}.sections") for x in d.get("sections") or ()),
            mesh=self.opt(d, "mesh", lambda v: self.text(v, f"{w}.mesh")),
        )

    def section(self, x, where: str) -> tuple:
        if not isinstance(x, (list, tuple)) or len(x) != 4:
            raise YAMLError(f"{where}: (z, x offset, y offset, scale), got {x!r}")
        z, dx, dy = self.nums(x[:3], "length", where, 3)
        return (z, dx, dy, self.plain(x[3], where))

    def transform(self, d, where: str) -> schema.Transform:
        d = self.fields(d, where, (), ("position", "rotation"))
        return schema.Transform(
            position=self.opt(d, "position", lambda v: self.nums(v, "length", f"{where}.position", 3), (0.0, 0.0, 0.0)),
            rotation=self.opt(d, "rotation", lambda v: self.rotation(v, f"{where}.rotation", False), (0.0, 0.0, 0.0)),
        )

    def placement(self, i: int, d) -> schema.Placement:
        w = f"placements[{i}]"
        d = self.fields(d, w, ("name", "volume", "mother"), ("copy", "position", "rotation", "scale"))
        w = f"placement {d['name']!r}"
        return schema.Placement(
            self.text(d["name"], w), self.text(d["volume"], f"{w}.volume"), self.text(d["mother"], f"{w}.mother"),
            copy=self.opt(d, "copy", lambda v: self.integer(v, f"{w}.copy")),
            position=self.opt(d, "position", lambda v: self.nums(v, "length", f"{w}.position", 3), (0.0, 0.0, 0.0)),
            rotation=self.opt(d, "rotation", lambda v: self.rotation(v, f"{w}.rotation", True), (0.0, 0.0, 0.0)),
            scale=self.opt(d, "scale", lambda v: self.plains(v, f"{w}.scale", 3)),
        )  # fmt: skip

    def volume(self, name: str, d, derived: dict) -> schema.Volume:
        w = f"volumes.{name}"
        d = self.fields(d, w, (), ("solid", "material", "assembly"))
        daughters = derived.get(name, [])
        if not d.get("assembly") and (d.get("solid") is None or d.get("material") is None):
            raise YAMLError(f"{w}: a volume names its solid and its material (an assembly, assembly: true, neither)")
        return schema.Volume(
            name,
            self.opt(d, "solid", lambda v: self.text(v, f"{w}.solid")),
            self.opt(d, "material", lambda v: self.text(v, f"{w}.material")),
            daughters=tuple(daughters),
            assembly=bool(d.get("assembly", False)),
        )

    # ---------------------------------------------------------------------------------- materials, optics
    def prop(self, key: str, d, where: str) -> schema.Property:
        w = f"{where}.{key}"
        if not isinstance(d, dict):
            d = {"value": d}  # a constant written as a bare number
        d = self.fields(d, w, ("value",), ("energy", "wavelength_nm", "note"))
        if "energy" in d and "wavelength_nm" in d:
            raise YAMLError(f"{w}: give energy or wavelength_nm, not both")
        note = str(d.get("note", ""))
        value = d["value"]
        if "energy" not in d and "wavelength_nm" not in d:
            if isinstance(value, (list, tuple, np.ndarray)) and np.ndim(value):
                raise YAMLError(f"{w}: a table gives its energies (energy, eV) or wavelengths (wavelength_nm)")
            return schema.Property(key, np.asarray(self.prop_value(key, value, w), np.float64), note=note)
        v = self.array(value, w, lambda x, ww: self.prop_value(key, x, ww))
        if "energy" in d:
            e = self.array(d["energy"], f"{w}.energy", lambda x, ww: self.num(x, "energy", ww))
        else:
            lam = self.array(d["wavelength_nm"], f"{w}.wavelength_nm", lambda x, ww: self.num(x, "length", ww, "nm"))
            if np.any(lam <= 0.0):
                raise YAMLError(f"{w}.wavelength_nm: wavelengths are positive")
            e, v = units.HC / lam[::-1], v[::-1]
            note = "; ".join(x for x in (note, f"energies from wavelengths (nm), hc = {units.HC} eV nm") if x)
        if np.shape(e) != np.shape(v):
            raise YAMLError(f"{w}: {np.size(e)} energies but {np.size(v)} values")
        return schema.Property(
            key,
            np.ascontiguousarray(v),
            energy=np.ascontiguousarray(e),
            note=note,
        )

    def array(self, v, where: str, one) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        if not isinstance(v, (list, tuple)):
            raise YAMLError(f"{where}: a list of numbers, got {v!r}")
        return np.array([one(x, where) for x in v], np.float64)

    def prop_value(self, key: str, x, where: str) -> float:
        """Return a value of property ``key`` in its unit (``schema.PROPERTY_UNITS``): a bare number is in it, a text
        with a unit is converted ("2.5*cm", "0.2*m/ns"); a key without a unit there takes a plain number."""  # noqa: D205, D209
        quantity = schema.PROPERTY_UNITS.get(key, (None,))[0]
        if quantity in units.QUANTITIES:
            return self.num(x, quantity, where)
        return self.plain(x, where)

    def optics(self, d, where: str) -> schema.OpticsModel:
        d = self.fields(d, where, (), ("name", "settings", "note"))
        try:
            return schema.OpticsModel(str(d.get("name", "geant4")), settings=dict(d.get("settings") or {}),
                                      note=str(d.get("note", "")))  # fmt: skip
        except ValueError as err:
            raise YAMLError(f"{where}: {err}") from None

    def component(self, d, where: str) -> schema.Component:
        d = self.fields(d, where, ("ref",), ("fraction", "atoms"))
        if ("fraction" in d) == ("atoms" in d):
            raise YAMLError(f"{where}: a component gives a mass fraction or a number of atoms (one of the two)")
        return schema.Component(
            self.text(d["ref"], f"{where}.ref"),
            fraction=self.opt(d, "fraction", lambda v: self.plain(v, f"{where}.fraction")),
            atoms=self.opt(d, "atoms", lambda v: self.integer(v, f"{where}.atoms")),
        )

    def material(self, name: str, d) -> schema.Material:
        w = f"materials.{name}"
        d = self.fields(d, w, ("composition",), ("state", "temperature_k", "pressure_pa", "properties", "optics"))
        c = self.fields(d["composition"], f"{w}.composition", ("density_g_cm3",), ("components", "z",
                                                                                    "molar_mass_g_mol"))  # fmt: skip
        comp = schema.Composition(
            self.num(c["density_g_cm3"], "density", f"{w}.composition.density_g_cm3"),
            components=tuple(
                self.component(x, f"{w}.composition.components[{i}]") for i, x in enumerate(c.get("components") or ())
            ),
            z=self.opt(c, "z", lambda v: self.plain(v, f"{w}.composition.z")),
            molar_mass_g_mol=self.opt(
                c, "molar_mass_g_mol", lambda v: self.plain(v, f"{w}.composition.molar_mass_g_mol")
            ),
        )
        props = d.get("properties") or {}
        return schema.Material(
            name, state=self.opt(d, "state", lambda v: self.text(v, f"{w}.state")), composition=comp,
            temperature_k=self.opt(d, "temperature_k", lambda v: self.num(v, "temperature", f"{w}.temperature_k")),
            pressure_pa=self.opt(d, "pressure_pa", lambda v: self.num(v, "pressure", f"{w}.pressure_pa")),
            properties={str(k): self.prop(str(k), p, f"{w}.properties")
             for k, p in self.mapping(props, f"{w}.properties").items()},
            optics=self.optics(d["optics"], f"{w}.optics") if "optics" in d else schema.OpticsModel(),
        )  # fmt: skip

    @staticmethod
    def mapping(v, where: str) -> dict:
        if not isinstance(v, dict):
            raise YAMLError(f"{where}: a mapping by key, got {v!r}")
        return v

    def surface(self, name: str, d) -> schema.Surface:
        w = f"surfaces.{name}"
        d = self.fields(d, w, ("type", "model", "finish", "value"), ("properties", "optics"))
        return schema.Surface(
            name, type=self.text(d["type"], f"{w}.type"), model=self.text(d["model"], f"{w}.model"),
            finish=self.text(d["finish"], f"{w}.finish"),
            value=(self.plain(d["value"], f"{w}.value") if d["model"] == "glisur" else  # polish, else sigma_alpha
                   self.num(d["value"], "angle", f"{w}.value")),
            properties={str(k): self.prop(str(k), p, f"{w}.properties")
             for k, p in self.mapping(d.get("properties") or {}, f"{w}.properties").items()},
            optics=self.optics(d["optics"], f"{w}.optics") if "optics" in d else schema.OpticsModel(),
        )  # fmt: skip

    # --------------------------------------------------------------------------------------------- sensors
    def sensor_set(self, d) -> sensorsets.SensorSet:
        optional = tuple(f.name for f in dataclasses.fields(sensorsets.SensorGroup) if f.name not in ("name", "rule"))
        groups = [sensorsets.SensorGroup(**self.fields(g, f"sensors[{i}]", ("name", "rule"), optional))
                  for i, g in enumerate(d or ())]  # fmt: skip
        try:
            return sensorsets.SensorSet(tuple(groups))
        except ValueError as err:
            raise YAMLError(f"sensors: {err}") from None

    # ------------------------------------------------------------------------------------------ the whole
    def detector(self) -> tuple[schema.DetectorFile, sensorsets.SensorSet]:
        s = self.s
        placements = tuple(self.placement(i, d) for i, d in enumerate(s.get("placements") or ()))
        derived: dict = {}
        for i, p in enumerate(placements):
            derived.setdefault(p.mother, []).append(i)
        volumes = {str(n): self.volume(str(n), d, derived) for n, d in self.mapping(s.get("volumes") or {},
                                                                                  "volumes").items()}  # fmt: skip
        world = s.get("world") or self.world(volumes, placements)
        df = schema.DetectorFile(
            name=str(s.get("name") or self.default_name()),
            world=str(world),
            detector_volume=str(s.get("detector_volume") or world),
            solids={str(n): self.solid(str(n), d) for n, d in self.mapping(s.get("solids") or {}, "solids").items()},
            volumes=volumes,
            placements=placements,
            materials={str(n): self.material(str(n), d)
                       for n, d in self.mapping(s.get("materials") or {}, "materials").items()},
            elements={str(n): self.element(str(n), d)
                      for n, d in self.mapping(s.get("elements") or {}, "elements").items()},
            isotopes={str(n): self.isotope(str(n), d)
                      for n, d in self.mapping(s.get("isotopes") or {}, "isotopes").items()},
            surfaces={str(n): self.surface(str(n), d)
                      for n, d in self.mapping(s.get("surfaces") or {}, "surfaces").items()},
            skins=tuple(self.skin(i, d) for i, d in enumerate(s.get("skins") or ())),
            borders=tuple(self.border(i, d) for i, d in enumerate(s.get("borders") or ())),
            meshes={str(n): self.mesh(str(n), v) for n, v in self.mapping(s.get("meshes") or {}, "meshes").items()},
        )  # fmt: skip
        return df, self.sensor_set(s.get("sensors"))

    def default_name(self) -> str:
        files = self.s.get("_files") or ["detector"]
        return Path(files[0]).stem

    @staticmethod
    def world(volumes: dict, placements) -> str:
        placed = {p.volume for p in placements}
        roots = [n for n in volumes if n not in placed]
        if len(roots) != 1:
            raise YAMLError(f"world is missing and {len(roots)} volumes are placed by no placement ({roots[:5]}): "
                            "name the world")  # fmt: skip
        return roots[0]

    def element(self, name: str, d) -> schema.Element:
        w = f"elements.{name}"
        d = self.fields(d, w, ("formula",), ("z", "molar_mass_g_mol", "isotopes"))
        return schema.Element(
            name,
            formula=self.text(d["formula"], f"{w}.formula"),
            z=self.opt(d, "z", lambda v: self.plain(v, f"{w}.z")),
            molar_mass_g_mol=self.opt(d, "molar_mass_g_mol", lambda v: self.plain(v, f"{w}.molar_mass_g_mol")),
            isotopes=tuple(self.component(x, f"{w}.isotopes[{i}]") for i, x in enumerate(d.get("isotopes") or ())),
        )

    def isotope(self, name: str, d) -> schema.Isotope:
        w = f"isotopes.{name}"
        d = self.fields(d, w, ("z", "n", "molar_mass_g_mol"))
        return schema.Isotope(name, z=self.integer(d["z"], f"{w}.z"), n=self.integer(d["n"], f"{w}.n"),
                              molar_mass_g_mol=self.plain(d["molar_mass_g_mol"], f"{w}.molar_mass_g_mol"))  # fmt: skip

    def skin(self, i: int, d) -> schema.Skin:
        d = self.fields(d, f"skins[{i}]", ("name", "surface", "volume"), ("note",))
        name, surface, volume = (self.text(d[k], f"skins[{i}].{k}") for k in ("name", "surface", "volume"))
        return schema.Skin(name, surface=surface, volume=volume, note=str(d.get("note", "")))

    def border(self, i: int, d) -> schema.Border:
        d = self.fields(d, f"borders[{i}]", ("name", "surface", "pv1", "pv2"))
        name, surface, pv1, pv2 = (self.text(d[k], f"borders[{i}].{k}") for k in ("name", "surface", "pv1", "pv2"))
        return schema.Border(name, surface=surface, pv1=pv1, pv2=pv2)

    def mesh(self, name: str, v) -> np.ndarray:
        a = v if isinstance(v, np.ndarray) else np.asarray(v, np.float64)
        if a.ndim != 3 or a.shape[1:] != (3, 3):
            raise YAMLError(f"meshes.{name}: triangles (k, 3, 3) in m, got shape {a.shape}")
        return a


def angles(a: np.ndarray) -> tuple[float, float, float]:
    """Return Geant4's (x, y, z) angles (rad) of a rotation ``a = Rz(z) Ry(y) Rx(x)``, each 0.0 rather than -0.0.

    A placement's matrix (the placed volume's frame into its mother's) is the transpose of ``a`` (G4GDML places with
    the inverse); a boolean's second solid's matrix is ``a`` itself (``lumen.build.lowering.placement_pose``,
    ``boolean_pose``).
    """
    y = math.asin(max(-1.0, min(1.0, -float(a[2, 0]))))
    if math.cos(y) > GIMBAL_COS:
        x, z = math.atan2(a[2, 1], a[2, 2]), math.atan2(a[1, 0], a[0, 0])
    else:  # gimbal lock: x folds into z
        x, z = 0.0, math.atan2(-a[0, 1], a[1, 1])
    return (x + 0.0, y + 0.0, z + 0.0)


def read(*paths) -> tuple[schema.DetectorFile, sensorsets.SensorSet]:
    """Return ``(detector file, SensorSet)`` of YAML files or directories of them, combined and checked.

    The structure passes ``lumen.format.check`` (the decisions included); the sensor set gets its ids at conversion
    (``lumen.recipe.import_yaml``).
    """
    df, sset = _Reader(merge(sources(paths))).detector()
    df = elements.with_elements(df)
    check(df, YAMLError)
    return df, sset


__all__ = ["SCHEMA", "YAMLError", "angles", "merge", "read"]
