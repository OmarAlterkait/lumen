"""GDML export: a detector file written as the GDML Geant4 10.7 and 11.2 read as lumen does, plus its sidecar.

    report = lumen.gdml.export("sk.lumen", "sk_export.gdml", scope="full")    # ExportReport
    result = lumen.gdml.read_export("sk_export.gdml")                         # ImportResult: detector file, sensors

The exporter is the exact inverse of the reader and the importer: ``read_export(export(df, scope="full"))`` is ``df``
bit for bit on every content file of ``lumen.format.io.to_files`` and on the head fields, and two exports of one
detector file are byte-identical (the order of everything written is a function of the content). What Geant4 can
read goes into the GDML; the recipe's layer goes into the sidecar ``<path>.lumen.json`` (canonical JSON), which
``read_export`` re-applies: the detector volume, wavelength range, outside, build options, provenance and report,
every optics model that is not the default, the recipe's properties' origin, the recipe's own surfaces and skins, the
borders' binding, the sensor sets with their id maps, what was omitted and the g4ref arguments.

Numbers (``lumen.gdml.units``): each value is the shortest repr of the stored double with its decimal point moved to
Geant4's internal unit by exact decimal arithmetic (lengths m to mm, energies eV to MeV), a bare literal: lengths
under ``lunit="mm"`` / ``unit="mm"``, angles in rad under ``aunit="rad"`` / ``unit="rad"``, energies as bare MeV
tokens in matrices. No expressions, no constants, no matrix ``unit``. Geant4 reads each literal as the correctly
rounded double (strtod; the unit is 1), and every literal is checked to read back to the stored double exactly as
lumen's reader reads it.

Order (Geant4 reads in one pass, G4GDMLRead*): matrices, then the tet and tessellated vertices, in <define>;
isotopes, elements, then materials after their components; solids after their constituents, then the optical
surfaces (in <solids>, after the matrices they use); logical volumes daughters first, so that the placements come
back in their stored order; skins, then borders, in stored order; userinfo; one setup named Default. Physvol
transforms are written inline; a rotation, position or boolean transform only when some value is not +0.0. All four
opticalsurface attributes are written, by name (Geant4 falls through to dichroic, Detector_LUT and x_ray on a missing
or unknown string, G4GDMLReadSolids::OpticalSurfaceRead).

Fails closed (``ExportError``): a border on a physvol name that is not unique (Geant4 binds the first one), a repeated
border pair, two skins on one logical volume, a coated surface (GDML has no coated type), a model, finish or type
Geant4 does not know, a non-finite or denormal value, a solid type the exporter has no spelling for, and placements
in an order no GDML file can hold. Lossy (``on_lossy``): a surface whose optics model has no Geant4 form (e.g.
"lumen:larsoft_reflector") raises under "error"; under "omit" it is not written, its record and those of its skins and
borders go to the sidecar, the export is marked ``lossless: false`` and the g4ref arguments carry ``--refuse-lossy``
(an option g4ref does not know, so it exits 2 rather than serve as a reference). A lowering of content the GDML holds
natively ("lumen:wls_film": the TPB volume and the foil's skin are in the GDML) is not written and is not lossy.

``scope="active"`` writes the detector volume in a shell (gdml_export.md section 4.4): its mother, made of the
mother's material, with the mother's skin (or the detector's outside surface), as a box 1 mm larger than the
detector's bounding box, inside boxes of a material with no RINDEX (Geant4 kills there), every transform as stored.
``scope="full"`` writes the whole world.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np

from lumen.format import io, schema, sensorsets
from lumen.format import overlay as _overlay
from lumen.format.sensorsets import SensorSet
from lumen.gdml import importer, reader
from lumen.gdml import solids_schema as ss
from lumen.gdml.units import shift

EXPORTER_VERSION = "1"
SIDECAR_SCHEMA = "lumen-gdml-sidecar/1"
SIDECAR_SUFFIX = ".lumen.json"
TARGETS = ("10.7", "11.2")
TINY = sys.float_info.min  # Geant4's evaluator refuses a denormal literal (strtod sets ERANGE)
MM_PLACES, MEV_PLACES = 3, -6  # m -> mm and eV -> MeV, as decimal shifts
SHELL_M = 1e-3  # scope "active": the shell is the detector's bounding box grown by this (m)
OUTSIDE = "lumen_outside"  # scope "active": the material (no RINDEX) around the shell
REFUSE_LOSSY = "--refuse-lossy"  # unknown to g4ref (it exits 2): a lossy export is not a reference
SCHEMA_LOCATION = "http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd"
NATIVE_MODELS = ("geant4",)  # optics models a GDML opticalsurface expresses exactly
LOWERED_MODELS = ("lumen:wls_film",)  # a lowering of what the GDML holds natively: not written, not lossy
SENSDET = "SensDet"  # the auxtype of a sensor volume's aux value (lumen.format.sensorsets.SensorVolumes)

# Geant4's optical-surface enums by name (G4OpticalSurface.hh, G4SurfaceProperty.hh; 10.7.4 and 11.2.2 agree), in
# enum order: a stored number is its position. G4GDMLReadSolids::OpticalSurfaceRead accepts each name (model
# "dichroic" and type "x_ray" by falling through); GDML has no coated type (Geant4 11's "coated" falls to x_ray).
MODELS = ("glisur", "unified", "LUT", "DAVIS", "dichroic")
FINISHES = (
    "polished polishedfrontpainted polishedbackpainted ground groundfrontpainted groundbackpainted "
    "polishedlumirrorair polishedlumirrorglue polishedair polishedteflonair polishedtioair polishedtyvekair "
    "polishedvm2000air polishedvm2000glue etchedlumirrorair etchedlumirrorglue etchedair etchedteflonair etchedtioair "
    "etchedtyvekair etchedvm2000air etchedvm2000glue groundlumirrorair groundlumirrorglue groundair groundteflonair "
    "groundtioair groundtyvekair groundvm2000air groundvm2000glue Rough_LUT RoughTeflon_LUT RoughESR_LUT "
    "RoughESRGrease_LUT Polished_LUT PolishedTeflon_LUT PolishedESR_LUT PolishedESRGrease_LUT"
).split()
TYPES = ("dielectric_metal", "dielectric_dielectric", "dielectric_LUT", "dielectric_LUTDAVIS", "dielectric_dichroic",
         "firsov", "x_ray")  # fmt: skip
# the Geant4-10 scintillation keys: user keys no process reads under Geant4 11 (kept equal to lumen.rules.optics)
G4_10_KEYS = ("FASTCOMPONENT", "SLOWCOMPONENT", "FASTTIMECONSTANT", "SLOWTIMECONSTANT", "FASTSCINTILLATIONRISETIME",
              "SLOWSCINTILLATIONRISETIME", "YIELDRATIO")  # fmt: skip
# the property keys Geant4 11.3 knows (G4MaterialPropertiesTable fMatPropNames, fMatConstPropNames; kept equal to
# lumen.rules.optics TABLE_KEYS | CONST_KEYS) and the Geant4-10 ones: a recipe table under one of them is written
GEANT4_KEYS = frozenset(
    "RINDEX REFLECTIVITY REALRINDEX IMAGINARYRINDEX EFFICIENCY TRANSMITTANCE SPECULARLOBECONSTANT "
    "SPECULARSPIKECONSTANT BACKSCATTERCONSTANT GROUPVEL MIEHG RAYLEIGH WLSCOMPONENT WLSABSLENGTH WLSCOMPONENT2 "
    "WLSABSLENGTH2 ABSLENGTH "
    "PROTONSCINTILLATIONYIELD DEUTERONSCINTILLATIONYIELD TRITONSCINTILLATIONYIELD ALPHASCINTILLATIONYIELD "
    "IONSCINTILLATIONYIELD ELECTRONSCINTILLATIONYIELD SCINTILLATIONCOMPONENT1 SCINTILLATIONCOMPONENT2 "
    "SCINTILLATIONCOMPONENT3 COATEDRINDEX SURFACEROUGHNESS WLSMEANNUMBERPHOTONS WLSTIMECONSTANT WLSMEANNUMBERPHOTONS2 "
    "WLSTIMECONSTANT2 SCINTILLATIONYIELD RESOLUTIONSCALE FERMIPOT DIFFUSION SPINFLIP LOSS LOSSCS ABSCS SCATCS "
    "COATEDTHICKNESS COATEDFRUSTRATEDTRANSMISSION MIEHG_FORWARD MIEHG_BACKWARD MIEHG_FORWARD_RATIO".split()
    + [f"{p}SCINTILLATION{q}{i}" for p in ("", "PROTON", "DEUTERON", "TRITON", "ALPHA", "ION", "ELECTRON")
       for q in ("YIELD", "TIMECONSTANT") for i in (1, 2, 3)]
    + [f"SCINTILLATIONRISETIME{i}" for i in (1, 2, 3)] + list(G4_10_KEYS)
)  # fmt: skip


class ExportError(ValueError):
    """A detector file the exporter cannot write as a GDML Geant4 reads as lumen does: the message says why."""


@dataclass(frozen=True)
class ExportReport:
    """What an export wrote and decided.

    ``lossless`` is False iff something was omitted (``on_lossy="omit"``); ``omitted`` lists it, ``warnings`` the
    known differences in Geant4's reading, ``annotations`` what went to the sidecar only, and ``g4ref_args`` the
    arguments g4ref needs to read the sensors as lumen does (with ``--refuse-lossy`` when the export is lossy).
    """

    path: str
    sha256: str
    sidecar: str | None
    detector_digest: str
    scope: str
    names: str
    target: tuple[str, ...]
    lossless: bool
    calibration: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    annotations: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    g4ref_args: tuple[str, ...] = ()

    def __str__(self) -> str:
        """Summarise the export: file, scope, losslessness, warnings and the g4ref arguments."""
        lines = [f"{self.path}  ({self.sha256}; scope {self.scope}, names {self.names}, "
                 f"Geant4 {', '.join(self.target)})",
                 f"  lossless: {'yes' if self.lossless else 'NO'}; {len(self.warnings)} warning(s), "
                 f"{len(self.annotations)} annotation(s); calibration: {' > '.join(self.calibration) or 'none'}",
                 f"  g4ref {' '.join(self.g4ref_args)}"]  # fmt: skip
        lines += [f"  omitted: {x}" for x in self.omitted] + [f"  warning: {x}" for x in self.warnings]
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- numbers
@lru_cache(maxsize=1 << 20)
def _literal(text: str, places: int) -> str:
    """Return the literal of a double's repr shifted by ``places``, checked both ways (see the module docstring)."""
    v = float(text)
    if not math.isfinite(v):
        raise ExportError(f"non-finite value {text}: Geant4 reads no inf or nan")
    if v != 0.0 and abs(v) < TINY:
        raise ExportError(f"denormal value {text}: Geant4's reader refuses it (strtod sets ERANGE)")
    lit = shift(text, places)
    g4 = float(lit)
    if not math.isfinite(g4) or (g4 != 0.0 and abs(g4) < TINY):
        raise ExportError(f"value {text} is {lit} in Geant4's units: not a normal double")
    back = float(Decimal(lit).scaleb(-places))
    if back != v or math.copysign(1.0, back) != math.copysign(1.0, v):
        raise ExportError(f"value {text}: the literal {lit} does not read back exactly")
    return lit


def _mm(v) -> str:
    return _literal(repr(float(v)), MM_PLACES)


def _mev(v) -> str:
    return _literal(repr(float(v)), MEV_PLACES)


def _bare(v) -> str:
    return _literal(repr(float(v)), 0)


def _plus_zero(values) -> bool:
    """Whether every value is +0.0 bit for bit (a transform Geant4 and lumen both take as absent)."""
    return all(x == 0.0 and math.copysign(1.0, x) > 0 for x in values)


# ------------------------------------------------------------------------------------------------------- XML text
def _q(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _attrs(pairs) -> str:
    return " ".join(f'{k}="{_q(v)}"' for k, v in pairs)


def _el(tag: str, pairs=(), inner: str = "") -> str:
    a = _attrs(pairs)
    head = f"<{tag} {a}" if a else f"<{tag}"
    return f"{head}>{inner}</{tag}>" if inner else f"{head}/>"


def _vec(tag: str, v, unit: str | None, fmt) -> str:
    pairs = ([("unit", unit)] if unit else []) + [(c, fmt(x)) for c, x in zip("xyz", v, strict=True)]
    return _el(tag, pairs)


# ---------------------------------------------------------------------------------------------------- the inputs
@dataclass
class _Input:
    """A detector file and what travels with it: sensor sets, their id-map columns, shipped data, its directory."""

    df: schema.DetectorFile
    sets: dict = field(default_factory=dict)
    columns: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    root: Path | None = None


def _read_input(src) -> _Input:
    if isinstance(src, schema.DetectorFile):
        return _Input(src)
    if isinstance(src, importer.ImportResult):
        cols = {n: dict(src.sensor_columns) for n in src.sensor_sets}
        return _Input(src.detector_file, dict(src.sensor_sets), cols, dict(src.calibrations))
    root = Path(src)
    df = io.read(root)
    sets, cols, data = {}, {}, {}
    for f in sorted((root / "sensors").glob("*.ids.json")):
        name = f.name[: -len(".ids.json")]
        sets[name] = sensorsets.read(root, name)
        cols[name] = {k: v for k, v in json.loads(f.read_text()).items() if k != "ids"}
    for f in sorted((root / "data").glob("*.json")):
        data[f.stem] = json.loads(f.read_text())
    return _Input(df, sets, cols, data, root)


def _calibrate(inp: _Input, calibration) -> tuple[schema.DetectorFile, tuple, dict]:
    """Return (the detector file with the calibration applied, the chain's names, the sensors/ values)."""
    if calibration is None:
        return inp.df, (), {}
    from lumen import _env
    from lumen.format import calibrations as cal

    search = ([inp.root / "calibrations"] if inp.root else []) + list(_env.detector_path())
    if isinstance(calibration, cal.Calibration):
        parents = cal.chain(calibration.parent, search) if calibration.parent != cal.NOMINAL else []
        chain = [*parents, calibration]
    else:
        chain = cal.chain(str(calibration), search)
    return _overlay.apply(inp.df, chain), tuple(c.name for c in chain), _overlay.sensor_values(chain)


# ---------------------------------------------------------------------------------------------------- the selection
@dataclass
class _Selection:
    """What goes into the GDML and what into the sidecar only."""

    surfaces: list = field(default_factory=list)  # written, sorted by name
    skins: list = field(default_factory=list)  # indices into df.skins written
    borders: list = field(default_factory=list)  # indices into df.borders written
    sidecar_surfaces: dict = field(default_factory=dict)  # name -> "lowered" | "omitted"
    sidecar_props: dict = field(default_factory=dict)  # material -> keys kept in the sidecar only
    warnings: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    omitted: list = field(default_factory=list)


def _select_surfaces(df: schema.DetectorFile, on_lossy: str, sel: _Selection) -> None:
    for name in sorted(df.surfaces):
        model = df.surfaces[name].optics.name
        if model in NATIVE_MODELS:
            sel.surfaces.append(name)
        elif model in LOWERED_MODELS:
            sel.sidecar_surfaces[name] = "lowered"
            sel.annotations.append(f"surface {name!r} ({model}) is a lowering of what the GDML holds natively: "
                                   "sidecar only")  # fmt: skip
        elif on_lossy == "omit":
            sel.sidecar_surfaces[name] = "omitted"
            sel.omitted.append(f"surface {name!r}: optics model {model!r} has no Geant4 form")
        else:
            raise ExportError(f"surface {name!r}: optics model {model!r} has no Geant4 form; pass on_lossy='omit' "
                              "to write the GDML without it (marked lossy, not a Geant4 reference)")  # fmt: skip
    written = set(sel.surfaces)
    sel.skins = [i for i, k in enumerate(df.skins) if k.surface in written]
    sel.borders = [i for i, b in enumerate(df.borders) if b.surface in written]
    for k in df.skins:
        if k.surface not in written and sel.sidecar_surfaces.get(k.surface) == "omitted":
            sel.omitted.append(f"skin {k.name!r} on {k.volume!r} (surface {k.surface!r})")
    for b in df.borders:
        if b.surface not in written and sel.sidecar_surfaces.get(b.surface) == "omitted":
            sel.omitted.append(f"border {b.name!r} {b.pv1!r} -> {b.pv2!r} (surface {b.surface!r})")


def _select_materials(df: schema.DetectorFile, sel: _Selection) -> None:
    for name in sorted(df.materials):
        m = df.materials[name]
        recipe = sorted(k for k, p in m.properties.items() if p.origin != "gdml")
        geant4 = [k for k in recipe if k in GEANT4_KEYS]
        sel.sidecar_props[name] = [k for k in recipe if k not in GEANT4_KEYS]
        if geant4:
            sel.warnings.append(f"material {name!r}: the recipe's tables {geant4} ({m.optics.name}) are written as "
                                "Geant4 tables; Geant4 reads them with its own semantics")  # fmt: skip
        if sel.sidecar_props[name]:
            sel.annotations.append(f"material {name!r}: recipe properties {sel.sidecar_props[name]} are not Geant4 "
                                   "keys: sidecar only")  # fmt: skip
        if m.optics != schema.OpticsModel():
            sel.annotations.append(f"material {name!r}: optics model {m.optics.name!r} (sidecar)")


def _enum_names(s: schema.Surface) -> tuple[str, str, str]:
    """Return a surface's (model, finish, type) by name; refuse what Geant4's GDML reader does not know."""
    out = []
    for what, v, names in (("model", s.model, MODELS), ("finish", s.finish, FINISHES), ("type", s.type, TYPES)):
        t = str(v).strip()
        if t.isdigit() and int(t) < len(names):
            t = names[int(t)]
        if t == "coated" or (what == "type" and t.isdigit() and int(t) == len(TYPES)):
            raise ExportError(f"surface {s.name!r}: a coated surface; GDML has no coated type (it reads as x_ray)")
        if t not in names:
            raise ExportError(f"surface {s.name!r}: {what} {v!r} is not one Geant4 knows (one of {list(names)})")
        out.append(t)
    return out[0], out[1], out[2]


def _check(df: schema.DetectorFile, sel: _Selection, target: tuple) -> None:
    """Refuse what Geant4 would read differently (the module docstring); add the target warnings."""
    bad = [t for t in target if t not in TARGETS]
    if bad:
        raise ExportError(f"unknown Geant4 target(s) {bad}: the exporter knows {list(TARGETS)}")
    counts: dict = {}
    for p in df.placements:
        counts[p.name] = counts.get(p.name, 0) + 1
    pairs: set = set()
    for i in sel.borders:
        b = df.borders[i]
        for r in (b.pv1, b.pv2):
            if counts.get(r, 0) != 1:
                raise ExportError(f"border {b.name!r}: physvol {r!r} names {counts.get(r, 0)} placements; Geant4 "
                                  "binds a border to the first one only")  # fmt: skip
        if (b.pv1, b.pv2) in pairs:
            raise ExportError(f"border {b.name!r}: the pair {b.pv1!r} -> {b.pv2!r} has a border already (Geant4 keeps "
                              "the first, lumen the last)")  # fmt: skip
        pairs.add((b.pv1, b.pv2))
    lvs: dict = {}
    for i in sel.skins:
        k = df.skins[i]
        if k.volume in lvs:
            raise ExportError(f"logical volume {k.volume!r} would carry two skins ({lvs[k.volume]!r}, {k.name!r})")
        lvs[k.volume] = k.name
    for name in sel.surfaces:
        _enum_names(df.surfaces[name])
    legacy = sorted({k for m in df.materials.values() for k in [*m.properties, *m.inert] if k in G4_10_KEYS})
    if legacy and any(t.startswith("11") for t in target):
        sel.warnings.append(f"Geant4-10 scintillation keys {legacy}: user keys no process reads under Geant4 11")
    n0x = sum("0x" in n for n in [*df.solids, *df.volumes, *df.materials, *(p.name for p in df.placements)])
    if n0x:
        sel.warnings.append(f"{n0x} name(s) contain '0x': Geant4 strips them there (g4ref's --dump, sensors.csv and "
                            "--sens-regex see the stripped names)")  # fmt: skip


def _check_structure(df: schema.DetectorFile, sel: _Selection) -> None:
    """Refuse a volume of an undefined material that is no NIST name, and a border on an assembly's placements.

    Geant4 builds an undefined ``G4_*`` material from its NIST tables (G4GDMLReadMaterials::GetMaterial) and fails on
    any other; it imprints an assembly, naming the imprinted placements itself, so a border cannot name them.
    """
    for v in df.volumes.values():
        if v.material and v.material not in df.materials and not v.material.startswith("G4_"):
            raise ExportError(f"volume {v.name!r}: material {v.material!r} is not defined (and is no G4_ NIST name)")
    assembly = {p.name for p in df.placements if df.volumes[p.mother].assembly or df.volumes[p.volume].assembly}
    for i in sel.borders:
        b = df.borders[i]
        if b.pv1 in assembly or b.pv2 in assembly:
            raise ExportError(f"border {b.name!r}: a physvol of an assembly ({b.pv1!r}, {b.pv2!r}); Geant4 renames "
                              "the imprinted placements, so the border cannot be bound")  # fmt: skip


# ----------------------------------------------------------------------------------------------------- matrices
def _table_text(key: str, p: schema.Property) -> tuple[int, str]:
    vf = _mm if key in importer.LENGTH_KEYS else _bare
    if p.energy is None:
        return 1, vf(np.asarray(p.value).reshape(()))
    e, v = np.asarray(p.energy, np.float64).tolist(), np.asarray(p.value, np.float64).tolist()
    return 2, " ".join(f"{_mev(a)} {vf(b)}" for a, b in zip(e, v, strict=True))


def _sha10(coldim: int, text: str) -> str:
    return hashlib.sha256(f"{coldim}:{text}".encode()).hexdigest()[:10]


def _matrices(df: schema.DetectorFile, sel: _Selection) -> tuple[dict, dict, list]:
    """Name the matrices (gdml_export.md section 4.1, step 3).

    Return (defines {name: (coldim, values)}, refs {(kind, owner, key): name}, renamed [[kind, owner, key, stored]]).
    A property keeps its stored matrix name when every property stored under that name has the same table; any other
    gets ``<KEY>_<sha10>`` of its content (deterministic), and its stored name goes to the sidecar.
    """
    users = []  # (kind, owner, key, stored name, coldim, text)
    for name in sorted(df.materials):
        m = df.materials[name]
        for key in sorted(m.properties):
            if key not in sel.sidecar_props.get(name, ()):
                users.append(("materials", name, key, m.properties[key].matrix, *_table_text(key, m.properties[key])))
        for key in sorted(m.inert):
            a = np.asarray(m.inert[key], np.float64)
            if a.size:
                users.append(("inert", name, key, "", a.shape[1], " ".join(_bare(x) for x in a.ravel().tolist())))
    for name in sel.surfaces:
        s = df.surfaces[name]
        users += [("surfaces", name, key, s.properties[key].matrix, *_table_text(key, s.properties[key]))
                  for key in sorted(s.properties)]  # fmt: skip
    contents: dict = {}
    for _, _, _, stored, cd, text in users:
        if stored:
            contents.setdefault(stored, set()).add((cd, text))
    defines: dict = {}
    refs: dict = {}
    renamed: list = []
    for kind, owner, key, stored, cd, text in users:
        name = stored if stored and len(contents[stored]) == 1 else f"{key}_{_sha10(cd, text)}"
        if defines.setdefault(name, (cd, text)) != (cd, text):
            raise ExportError(f"matrix name {name!r} would name two tables")
        refs[(kind, owner, key)] = name
        if name != stored and kind != "inert":
            renamed.append([kind, owner, key, stored])
    return defines, refs, renamed


def _check_constants(defines: dict, positions: list) -> None:
    """Refuse a name Geant4's DefineMatrix would define twice (it defines ``<name>_<i>`` and ``<name>_<i>_<j>``)."""
    seen = set(defines) | {n for n, _ in positions}
    if len(seen) != len(defines) + len(positions):
        raise ExportError("a matrix and a position share a name")
    for name, (cd, text) in defines.items():
        n = text.count(" ") + 1
        rows = n // cd
        consts = [f"{name}_{i}" for i in range(n)] if (n == cd or cd == 1) else \
            [f"{name}_{i}_{j}" for i in range(rows) for j in range(cd)]  # fmt: skip
        for c in consts:
            if c in seen:
                raise ExportError(f"matrix {name!r}: Geant4 would define the constant {c!r} twice")
            seen.add(c)


# ------------------------------------------------------------------------------------------------------- ordering
def _post_order(items: dict, deps) -> list:
    """Return the names of ``items`` sorted, each after the names ``deps(name)`` gives (defined ones only)."""
    out, done = [], set()
    for root in sorted(items):
        stack = [(root, False)]
        while stack:
            n, expanded = stack.pop()
            if n in done or n not in items:
                continue
            if expanded:
                done.add(n)
                out.append(n)
                continue
            stack.append((n, True))
            stack += [(d, False) for d in reversed(list(deps(n))) if d not in done]
    return out


def _volume_order(df: schema.DetectorFile) -> list:
    """Return the volumes in an order that gives the placements back in their stored order (daughters first)."""
    first = {n: min(v.daughters) for n, v in df.volumes.items() if v.daughters}
    order, done = [], set()

    def visit(n):
        stack = [(n, False)]
        while stack:
            x, expanded = stack.pop()
            if x in done:
                continue
            if expanded:
                done.add(x)
                order.append(x)
                continue
            stack.append((x, True))
            stack += [(df.placements[i].volume, False) for i in reversed(df.volumes[x].daughters)]

    for n in sorted(first, key=lambda k: (first[k], k)):
        visit(n)
    for n in sorted(df.volumes):
        visit(n)
    flat = [i for n in order for i in df.volumes[n].daughters]
    if flat != list(range(len(df.placements))):
        raise ExportError("the placements are not in an order a GDML file can hold (each volume's daughters one "
                          "contiguous run, in the order of their volumes): re-import cannot reproduce "
                          "them")  # fmt: skip
    return order


# ------------------------------------------------------------------------------------------------------ the writer
def _vertices(df: schema.DetectorFile, solid_order: list) -> tuple[list, dict]:
    """Return the vertex defines [(name, (x, y, z) m)] and each tessellated solid's facet vertex names."""
    defines, facets = [], {}
    for name in solid_order:
        s = df.solids[name]
        if s.type == "tet":
            defines += [(f"{name}__v{i + 1}", tuple(v)) for i, v in enumerate(s.vertices)]
        elif s.type == "tessellated":
            v = np.ascontiguousarray(np.asarray(df.meshes[str(s.mesh)], np.float64).reshape(-1, 3))
            key = v.view(np.dtype((np.void, v.dtype.itemsize * 3))).ravel()  # by bits: -0.0 is not +0.0
            _, first, inv = np.unique(key, return_index=True, return_inverse=True)
            order = np.argsort(first, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            ids = rank[inv.ravel()].reshape(-1, 3)
            defines += [(f"{name}__v{k}", tuple(row)) for k, row in enumerate(v[first[order]].tolist())]
            facets[name] = ids
    return defines, facets


def _solid_params(s: schema.Solid) -> list:
    try:
        spell = ss.spelling(s.type)
    except KeyError:
        raise ExportError(f"solid {s.name!r}: type {s.type!r} has no GDML spelling in the exporter (supported: "
                          f"{sorted(ss.ATTRIBUTES)})") from None  # fmt: skip
    unknown = sorted(set(s.params) - set(spell))
    if unknown:
        raise ExportError(f"solid {s.name!r} ({s.type}): attributes {unknown} are not {s.type} attributes")
    fmts = {ss.L: _mm, ss.A: _bare, ss.U: _bare}
    pairs = [("name", s.name)]
    pairs += [(gdml, fmts[dim](s.params[low])) for low, (gdml, dim) in spell.items() if low in s.params]
    dims = {dim for _, dim in spell.values()}
    if ss.L in dims or s.type in ("polycone", "polyhedra", "genericPolycone", "genericPolyhedra", "xtru"):
        pairs.append(("lunit", "mm"))
    if ss.A in dims:
        pairs.append(("aunit", "rad"))
    return pairs


def _transform_lines(tr: schema.Transform, pos: str, rot: str) -> str:
    out = ""
    if not _plus_zero(tr.position):
        out += _vec(pos, tr.position, "mm", _mm)
    if not _plus_zero(tr.rotation):
        out += _vec(rot, tr.rotation, "rad", _bare)
    return out


def _solid_children(s: schema.Solid, facets: dict) -> list:
    kind = ss.CHILDREN.get(s.type)
    if kind in ("zplane", "rzpoint"):
        return [_el(kind, [(a, _mm(x)) for a, x in zip(ss.PLANE_ATTRIBUTES[kind], p, strict=True)]) for p in s.planes]
    if kind == "xtru":
        out = [_el("twoDimVertex", [("x", _mm(x)), ("y", _mm(y))]) for x, y in s.vertices]
        out += [
            _el(
                "section",
                [("zOrder", str(i))]
                + [(a, f(x)) for a, f, x in zip(ss.SECTION_ATTRIBUTES, (_mm, _mm, _mm, _bare), sec, strict=True)],
            )
            for i, sec in enumerate(s.sections)
        ]
        return out  # fmt: skip
    if kind == "facets":
        n = s.name
        return [f'<triangular vertex1="{_q(n)}__v{a}" vertex2="{_q(n)}__v{b}" vertex3="{_q(n)}__v{c}" type="ABSOLUTE"/>'
                for a, b, c in facets[n].tolist()]  # fmt: skip
    if kind == "boolean":
        first, second = s.transforms
        return [_el("first", [("ref", s.refs[0])]), _el("second", [("ref", s.refs[1])]),
                _transform_lines(second, "position", "rotation") + _transform_lines(first, "firstposition",
                                                                                    "firstrotation")]  # fmt: skip
    if kind == "nodes":
        return [_el("multiUnionNode", [("name", f"{s.name}__n{k}")], _el("solid", [("ref", r)])
                    + _transform_lines(tr, "position", "rotation"))
                for k, (r, tr) in enumerate(zip(s.refs, s.transforms, strict=True))]  # fmt: skip
    if kind == "scaled":
        return [_el("solidref", [("ref", s.refs[0])]), _vec("scale", s.scale, None, _bare)]
    return []


def _solid(s: schema.Solid, facets: dict) -> list:
    pairs = _solid_params(s)
    if s.type == "tet":
        pairs = [pairs[0]] + [(f"vertex{i}", f"{s.name}__v{i}") for i in range(1, 5)] + [("lunit", "mm")]
    kids = [k for k in _solid_children(s, facets) if k]
    if not kids:
        return ["  " + _el(s.type, pairs)]
    return ["  " + f"<{s.type} {_attrs(pairs)}>", *("   " + k for k in kids), f"  </{s.type}>"]


def _material(m: schema.Material, refs: dict, sel: _Selection) -> list:
    pairs = [("name", m.name)] + ([("state", m.state)] if m.state else [])
    c = m.composition
    if c.z is not None:
        pairs.append(("Z", _bare(c.z)))
    kids = []
    if m.temperature is not None:
        kids.append(_el("T", [("unit", "K"), ("value", _bare(m.temperature))]))
    if m.pressure is not None:
        kids.append(_el("P", [("unit", "pascal"), ("value", _bare(m.pressure))]))
    if m.mee_ev is not None:
        kids.append(_el("MEE", [("unit", "eV"), ("value", _bare(m.mee_ev))]))
    kids.append(_el("D", [("unit", "g/cm3"), ("value", _bare(c.density_g_cm3))]))
    if c.molar_mass_g_mol is not None:
        kids.append(_el("atom", [("unit", "g/mole"), ("value", _bare(c.molar_mass_g_mol))]))
    for x in c.components:
        if x.atoms is not None:
            kids.append(_el("composite", [("n", str(int(x.atoms))), ("ref", x.ref)]))
        else:
            kids.append(_el("fraction", [("n", _bare(x.fraction)), ("ref", x.ref)]))
    for kind, keys in (("materials", sorted(m.properties)), ("inert", sorted(m.inert))):
        kids += [_el("property", [("name", k), ("ref", refs[(kind, m.name, k)])]) for k in keys
                 if (kind, m.name, k) in refs]  # fmt: skip
    return ["  " + f"<material {_attrs(pairs)}>", *("   " + k for k in kids), "  </material>"]


def _materials(df: schema.DetectorFile, refs: dict, sel: _Selection) -> list:
    out = [" <materials>"]
    for name in sorted(df.isotopes):
        i = df.isotopes[name]
        out.append("  " + _el("isotope", [("name", name), ("Z", str(int(i.z))), ("N", str(int(i.n)))],
                              _el("atom", [("unit", "g/mole"), ("value", _bare(i.molar_mass_g_mol))])))  # fmt: skip
    for name in sorted(df.elements):
        e = df.elements[name]
        pairs = [("name", name)] + ([("formula", e.formula)] if e.formula else []) + \
            ([("Z", _bare(e.z))] if e.z is not None else [])  # fmt: skip
        inner = "".join(_el("fraction", [("n", _bare(x.fraction)), ("ref", x.ref)]) for x in e.isotopes)
        if e.molar_mass_g_mol is not None:
            inner = _el("atom", [("unit", "g/mole"), ("value", _bare(e.molar_mass_g_mol))]) + inner
        out.append("  " + _el("element", pairs, inner))
    order = _post_order(df.materials, lambda n: [x.ref for x in df.materials[n].composition.components])
    for name in order:
        out += _material(df.materials[name], refs, sel)
    return out + [" </materials>"]


def _aux_lines(a: schema.Aux, depth: int) -> list:
    pairs = [("auxtype", a.type), ("auxvalue", a.value)] + ([("auxunit", a.unit)] if a.unit else [])
    if not a.children:
        return [" " * depth + _el("auxiliary", pairs)]
    out = [" " * depth + f"<auxiliary {_attrs(pairs)}>"]
    for c in a.children:
        out += _aux_lines(c, depth + 1)
    return out + [" " * depth + "</auxiliary>"]


def _physvol(p: schema.Placement) -> str:
    pairs = [("name", p.name)] + ([("copynumber", str(int(p.copy)))] if p.copy is not None else [])
    inner = _el("volumeref", [("ref", p.volume)])
    if not _plus_zero(p.position):
        inner += _vec("position", p.position, "mm", _mm)
    if not _plus_zero(p.rotation):
        inner += _vec("rotation", p.rotation, "rad", _bare)
    if p.scale is not None:
        inner += _vec("scale", p.scale, None, _bare)
    return _el("physvol", pairs, inner)


def _structure(df: schema.DetectorFile, sel: _Selection) -> list:
    out = [" <structure>"]
    for name in _volume_order(df):
        v = df.volumes[name]
        tag = "assembly" if v.assembly else "volume"
        out.append("  " + f"<{tag} {_attrs([('name', name)])}>")
        if not v.assembly:
            out += ["   " + _el("materialref", [("ref", v.material)]), "   " + _el("solidref", [("ref", v.solid)])]
        out += ["   " + _physvol(df.placements[i]) for i in v.daughters]
        for a in v.aux:
            out += _aux_lines(a, 3)
        out.append(f"  </{tag}>")
    for i in sel.skins:
        k = df.skins[i]
        out.append("  " + _el("skinsurface", [("name", k.name), ("surfaceproperty", k.surface)],
                              _el("volumeref", [("ref", k.volume)])))  # fmt: skip
    for i in sel.borders:
        b = df.borders[i]
        out.append("  " + _el("bordersurface", [("name", b.name), ("surfaceproperty", b.surface)],
                              _el("physvolref", [("ref", b.pv1)]) + _el("physvolref", [("ref", b.pv2)])))  # fmt: skip
    return out + [" </structure>"]


def _comment(text: str) -> str:
    return "<!-- " + text.replace("--", "- -") + " -->"


def _gdml_text(df: schema.DetectorFile, sel: _Selection, digest: str) -> tuple[str, list]:
    """Return the GDML text and the matrix renames (gdml_export.md section 4.1, steps 4 and 5)."""
    defines, refs, renamed = _matrices(df, sel)
    solid_order = _post_order(df.solids, lambda n: df.solids[n].refs)
    positions, facets = _vertices(df, solid_order)
    _check_constants(defines, positions)
    src = df.provenance.get("source", {})
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           _comment(f"lumen GDML export (lumen.gdml.exporter {EXPORTER_VERSION}) of detector {df.name}, content "
                    f"{digest}; source {src.get('file', '?')} {src.get('sha256', '')}. Numbers are bare literals in "
                    "Geant4's internal units (mm, MeV, ns, rad); read it back with lumen.gdml.read_export."),
           '<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
           f'xsi:noNamespaceSchemaLocation="{SCHEMA_LOCATION}">',
           " <define>"]  # fmt: skip
    out += ["  " + _el("matrix", [("name", n), ("coldim", str(cd)), ("values", t)]) for n, (cd, t) in
            sorted(defines.items())]  # fmt: skip
    out += ["  " + _vec("position", v, "mm", _mm).replace("<position ", f'<position name="{_q(n)}" ', 1)
            for n, v in positions]  # fmt: skip
    out += [" </define>"] + _materials(df, refs, sel) + [" <solids>"]
    for name in solid_order:
        out += _solid(df.solids[name], facets)
    for name in sel.surfaces:
        s = df.surfaces[name]
        model, finish, typ = _enum_names(s)
        pairs = [("name", name), ("model", model), ("finish", finish), ("type", typ), ("value", _bare(s.value))]
        props = "".join(_el("property", [("name", k), ("ref", refs[("surfaces", name, k)])]) for k in
                        sorted(s.properties))  # fmt: skip
        out.append("  " + _el("opticalsurface", pairs, props))
    out += [" </solids>"] + _structure(df, sel)
    if df.userinfo:
        out += [" <userinfo>"] + [x for a in df.userinfo for x in _aux_lines(a, 2)] + [" </userinfo>"]
    out += [" " + _el("setup", [("name", "Default"), ("version", "1.0")], _el("world", [("ref", df.world)])),
            "</gdml>"]  # fmt: skip
    return "\n".join(out) + "\n", renamed


# ------------------------------------------------------------------------------------------------- scope "active"
def _chain(df: schema.DetectorFile) -> list:
    """Return the placement indices from the world down to the detector volume's first placement.

    The first depth-first path, the one ``lumen.build.lowering.find_detector`` keeps (a volume placed in several
    mothers, some of them never placed, has one path from the world).
    """
    stack: list = [(df.world, [])]
    seen: set = set()
    while stack:
        lv, path = stack.pop()
        if lv in seen:
            continue
        seen.add(lv)
        if lv == df.detector_volume:
            return path
        stack += [(df.placements[i].volume, [*path, i]) for i in reversed(df.volumes[lv].daughters)]
    raise ExportError(f"scope 'active': {df.detector_volume!r} is not placed under the world {df.world!r}")


def _subtree(df: schema.DetectorFile, lv: str) -> list:
    out, stack = [], [lv]
    while stack:
        x = stack.pop()
        if x not in out:
            out.append(x)
            stack += [df.placements[i].volume for i in df.volumes[x].daughters]
    return out


def _box_of(df: schema.DetectorFile, lv: str) -> tuple:
    """Return the (low, high) corners of a volume's solid in its own frame (m)."""
    from lumen.build import lowering, solids

    s = df.solids[str(df.volumes[lv].solid)]
    if s.type == "tessellated":
        v = np.asarray(df.meshes[str(s.mesh)]).reshape(-1, 3)
        return v.min(0), v.max(0)
    lo, hi = solids.node_bbox(lowering.SolidMapper(df).solid(s.name))
    return np.asarray(lo, float), np.asarray(hi, float)


def _shell_solid(name: str, lo, hi) -> schema.Solid:
    """Return an axis-aligned box from ``lo`` to ``hi`` (m) as an extrusion (its centre need not be the origin)."""
    poly = ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1]), (hi[0], lo[1]))  # clockwise, as G4ExtrudedSolid wants
    return schema.Solid(name, "xtru", {}, vertices=tuple(tuple(float(c) for c in v) for v in poly),
                        sections=((float(lo[2]), 0.0, 0.0, 1.0), (float(hi[2]), 0.0, 0.0, 1.0)))  # fmt: skip


def _fresh(name: str, taken) -> str:
    while name in taken:
        name += "_"
    return name


def _shells(df: schema.DetectorFile, chain: list, outside: str) -> tuple[dict, dict, tuple]:
    """Return the ancestors of the detector volume as shells: (volumes, solids, the top shell's corners).

    Each ancestor's solid is its kept child's box, in its frame, grown by ``SHELL_M``; the detector's mother keeps its
    material, the others are of ``outside``.
    """
    from lumen.build import lowering

    vols, sols = {}, {}
    lo, hi = _box_of(df, df.detector_volume)
    for i in reversed(chain):
        p = df.placements[i]
        m, t = lowering.placement_pose(p)
        cs = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ m.T + t
        lo, hi = cs.min(0) - SHELL_M, cs.max(0) + SHELL_M
        name = _fresh(f"{p.mother}__shell", df.solids)
        sols[name] = _shell_solid(name, lo, hi)
        mat = df.volumes[p.mother].material if i == chain[-1] else outside
        vols[p.mother] = schema.Volume(p.mother, name, mat, (), False, df.volumes[p.mother].aux)
    return vols, sols, (lo, hi)


def _active(df: schema.DetectorFile) -> tuple[schema.DetectorFile, list]:
    """Return the detector volume in its shell (the module docstring) and the notes."""
    if df.detector_volume == df.world:
        return df, []
    chain = _chain(df)
    keep = _subtree(df, df.detector_volume)
    old = sorted(chain + [i for i, p in enumerate(df.placements) if p.mother in keep])
    outside, element = _fresh(OUTSIDE, df.materials), _fresh(OUTSIDE + "_H", df.elements)
    shells, sols, (lo, hi) = _shells(df, chain, outside)
    pls, world = [df.placements[o] for o in old], df.world
    mother = df.placements[chain[-1]].mother
    if mother == df.world:  # a box of the outside material around the shell, as the new world
        world = _fresh(OUTSIDE, df.volumes)
        name = _fresh(f"{world}__shell", {**df.solids, **sols})
        sols[name] = _shell_solid(name, lo - SHELL_M, hi + SHELL_M)
        shells[world] = schema.Volume(world, name, outside)
        pls.append(schema.Placement(f"{df.world}_PV", df.world, world, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    vols = {**{lv: df.volumes[lv] for lv in keep}, **shells}
    vols = {n: dataclasses.replace(v, daughters=tuple(k for k, p in enumerate(pls) if p.mother == n))
            for n, v in vols.items()}  # fmt: skip
    sols.update(_solids_closure(df, {v.solid for v in vols.values()} - set(sols)))
    mats = _materials_closure(df, {v.material for v in vols.values() if v.material})
    mats[outside] = schema.Material(outside, "gas", schema.Composition(1e-20, (schema.Component(element, 1.0),)))
    els = {**df.elements, element: schema.Element(element, "H", 1.0, 1.00794)}
    names = {p.name for p in pls}
    skins = [k for k in df.skins if k.volume in vols and not (df.outside_surface and k.volume == mother)]
    if df.outside_surface:
        skins.append(schema.Skin(_fresh("lumen_outside_skin", {k.name for k in df.skins}), df.outside_surface, mother))
    borders = [b for b in df.borders if b.pv1 in names and b.pv2 in names]
    used = {k.surface for k in skins} | {b.surface for b in borders}
    notes = [
        f"scope 'active': {df.detector_volume!r} in a {SHELL_M} m shell of {mother!r} "
        f"({df.volumes[mother].material}) inside {outside!r} (no RINDEX)"
    ]
    return dataclasses.replace(df, world=world, solids=sols, volumes=vols, placements=tuple(pls), materials=mats,
                               elements=els, surfaces={n: s for n, s in df.surfaces.items() if n in used},
                               skins=tuple(skins), borders=tuple(borders),
                               meshes={s.mesh: df.meshes[s.mesh] for s in sols.values() if s.mesh}), notes  # fmt: skip


def _solids_closure(df: schema.DetectorFile, names: set) -> dict:
    out, stack = {}, list(names)
    while stack:
        n = stack.pop()
        if n not in out:
            out[n] = df.solids[n]
            stack += list(df.solids[n].refs)
    return out


def _materials_closure(df: schema.DetectorFile, names: set) -> dict:
    out, stack = {}, list(names)
    while stack:
        n = stack.pop()
        if n in df.materials and n not in out:
            out[n] = df.materials[n]
            stack += [c.ref for c in df.materials[n].composition.components]
    return out


# ------------------------------------------------------------------------------------------------ names "geant4"
def _strip_map(names, what: str) -> dict:
    out: dict = {}
    for n in dict.fromkeys(names):  # a name given twice is one name
        s = reader.g4_strip(n)
        if s in out.values():
            raise ExportError(f"names='geant4': two {what} are {s!r} once stripped at '0x'")
        out[n] = s
    return out


def _geant4_names(df: schema.DetectorFile) -> schema.DetectorFile:
    """Return the detector file with solid, volume, placement, material and element names stripped at '0x'."""
    so, vo = _strip_map(df.solids, "solids"), _strip_map(df.volumes, "logical volumes")
    pv = _strip_map([p.name for p in df.placements], "physical volumes")
    ma, el = _strip_map(df.materials, "materials"), _strip_map(df.elements, "elements")
    if set(ma.values()) & set(el.values()):
        raise ExportError("names='geant4': a material and an element share a name once stripped at '0x'")
    comp = {**el, **ma}

    def comps(cs):
        return tuple(dataclasses.replace(c, ref=comp.get(c.ref, c.ref)) for c in cs)

    def solid(s):
        return dataclasses.replace(s, name=so[s.name], refs=tuple(so[r] for r in s.refs),
                                   mesh=None if s.mesh is None else so[s.mesh])  # fmt: skip

    mats = {ma[n]: dataclasses.replace(m, name=ma[n], composition=dataclasses.replace(
        m.composition, components=comps(m.composition.components))) for n, m in df.materials.items()}  # fmt: skip
    vols = {
        vo[n]: dataclasses.replace(
            v, name=vo[n], solid=so.get(v.solid, v.solid), material=ma.get(v.material, v.material)
        )
        for n, v in df.volumes.items()
    }
    pls = tuple(
        dataclasses.replace(p, name=pv[p.name], volume=vo[p.volume], mother=vo[p.mother]) for p in df.placements
    )
    return dataclasses.replace(
        df, world=vo[df.world], detector_volume=vo.get(df.detector_volume, df.detector_volume),
        solids={so[n]: solid(s) for n, s in df.solids.items()}, volumes=vols, placements=pls, materials=mats,
        elements={el[n]: dataclasses.replace(e, name=el[n]) for n, e in df.elements.items()},
        skins=tuple(dataclasses.replace(k, volume=vo[k.volume]) for k in df.skins),
        borders=tuple(dataclasses.replace(b, pv1=pv[b.pv1], pv2=pv[b.pv2]) for b in df.borders),
        meshes={so[n]: a for n, a in df.meshes.items()})  # fmt: skip


# ------------------------------------------------------------------------------------------------------ the sidecar
def _array(a) -> dict:
    a = np.asarray(a)
    return dict(dtype=str(a.dtype), shape=list(a.shape), data=a.ravel().tolist())


def _array_back(d: dict) -> np.ndarray:
    return np.asarray(d["data"], dtype=d["dtype"]).reshape(d["shape"])


def _prop(p: schema.Property) -> dict:
    return dict(key=p.key, value=_array(p.value), energy=None if p.energy is None else _array(p.energy),
                matrix=p.matrix, origin=p.origin, note=p.note)  # fmt: skip


def _prop_back(d: dict) -> schema.Property:
    return schema.Property(d["key"], _array_back(d["value"]), None if d["energy"] is None else _array_back(d["energy"]),
                           d["matrix"], d["origin"], d["note"])  # fmt: skip


def _optics(o: schema.OpticsModel) -> dict:
    return dict(name=o.name, version=o.version, settings=o.settings, origin=o.origin, note=o.note)


def _optics_back(d: dict) -> schema.OpticsModel:
    return schema.OpticsModel(d["name"], d["version"], d["settings"], d["origin"], d["note"])


def _surface_record(s: schema.Surface) -> dict:
    return dict(type=s.type, model=s.model, finish=s.finish, value=s.value, optics=_optics(s.optics),
                inert=list(s.inert), properties={k: _prop(p) for k, p in s.properties.items()})  # fmt: skip


def _surface_back(name: str, d: dict) -> schema.Surface:
    return schema.Surface(name, d["type"], d["model"], d["finish"], d["value"],
                          {k: _prop_back(p) for k, p in d["properties"].items()}, _optics_back(d["optics"]),
                          tuple(d["inert"]))  # fmt: skip


def _material_restore(df: schema.DetectorFile, sel: _Selection) -> tuple[dict, dict]:
    """Return (the sidecar's per-material records, the material keys read_export reads as optical)."""
    out, keys = {}, {}
    for name in sorted(df.materials):
        m, r = df.materials[name], {}
        if m.optics != schema.OpticsModel():
            r["optics"] = _optics(m.optics)
        side = sel.sidecar_props.get(name, [])
        meta = {k: [p.origin, p.note] for k, p in m.properties.items() if k not in side and (p.origin, p.note)
                != ("gdml", "")}  # fmt: skip
        if meta:
            r["properties"] = meta
        if side:
            r["sidecar_properties"] = {k: _prop(m.properties[k]) for k in side}
        empty = {k: list(np.shape(a)) for k, a in m.inert.items() if not np.size(a)}
        if empty:
            r["inert_empty"] = empty
        if any(np.size(a) for a in m.inert.values()):
            keys[name] = sorted(k for k in m.properties if k not in side)
        if r:
            out[name] = r
    return out, keys


def _surface_restore(df: schema.DetectorFile, sel: _Selection) -> dict:
    out = {}
    for name in sel.surfaces:
        s, r = df.surfaces[name], dict[str, object]()
        if s.optics != schema.OpticsModel():
            r["optics"] = _optics(s.optics)
        if s.inert:
            r["inert"] = list(s.inert)
        model, finish, typ = _enum_names(s)
        if (s.type, s.model, s.finish) != (typ, model, finish):
            r["enums"] = [s.type, s.model, s.finish]
        meta = {k: [p.origin, p.note] for k, p in s.properties.items() if (p.origin, p.note) != ("gdml", "")}
        if meta:
            r["properties"] = meta
        if r:
            out[name] = r
    return out


def _records(df: schema.DetectorFile, sel: _Selection) -> tuple[list, list]:
    """Return the skins and borders the GDML does not hold as they are: [index, record, written]."""
    sk, bo = set(sel.skins), set(sel.borders)
    skins = [[i, [k.name, k.surface, k.volume, k.origin, k.note], i in sk] for i, k in enumerate(df.skins)
             if i not in sk or (k.origin, k.note) != ("gdml", "")]  # fmt: skip
    borders = [
        [i, [b.name, b.surface, b.pv1, b.pv2, b.origin, b.bound, b.note], i in bo]
        for i, b in enumerate(df.borders)
        if i not in bo or (b.origin, b.bound, b.note) != ("gdml", True, "")
    ]
    return skins, borders


def _g4ref_args(sset: SensorSet | None, lossless: bool, sel: _Selection) -> list:
    """Return the g4ref arguments that read the sensors as lumen does (and the refuse mark of a lossy export)."""
    args: list = []
    if sset is not None:
        if sset.volumes.names:
            args += ["--sens-regex", sset.volumes.names]
        if sset.volumes.aux:
            args += ["--sens-aux", f"{SENSDET}={sset.volumes.aux[0]}"]
            if len(sset.volumes.aux) > 1:
                sel.warnings.append(f"g4ref takes one --sens-aux; the sensor set has {list(sset.volumes.aux)}")
        if any(g.rule == "border_into" and g.settings.get("efficiency_on_entry") == 1.0 for g in sset.groups):
            args.append("--wcsim-hits")  # a hit is any photon killed entering the sensor, as WCSim's sensitive detector
    if not lossless:
        args.append(REFUSE_LOSSY)
    return args


@dataclass
class _Run:
    """One export's inputs and decisions, for the sidecar and the report."""

    df: schema.DetectorFile
    sel: _Selection
    sset: SensorSet | None
    columns: dict
    data: dict
    chain: tuple
    sensor_values: dict
    options: dict
    digest: str = ""
    renamed: list = field(default_factory=list)


def _sidecar(run: _Run, gdml: Path, sha: str, args: list) -> dict:
    df, sel = run.df, run.sel
    mats, keys = _material_restore(df, sel)
    skins, borders = _records(df, sel)
    head = dict(
        name=df.name,
        world=df.world,
        detector_volume=df.detector_volume,
        wavelength_range_nm=list(df.wavelength_range),
        outside=df.outside,
        outside_surface=df.outside_surface,
        build_options=df.build_options,
        provenance=df.provenance,
        report=list(df.report),
        digest=run.digest,
    )
    sets = (
        {}
        if run.sset is None
        else {
            run.sset.name: dict(
                set=json.loads(sensorsets.to_json(run.sset)[0]), ids=list(run.sset.ids), columns=run.columns
            )
        }
    )
    cal = dict(chain=list(run.chain), sensors={k: _array(v) for k, v in sorted(run.sensor_values.items())})
    return dict(schema=SIDECAR_SCHEMA, exporter_version=EXPORTER_VERSION, gdml=dict(file=gdml.name, sha256=sha),
                lossless=not sel.omitted, options=run.options, detector=head, materials=mats,
                surfaces=_surface_restore(df, sel),
                recipe_surfaces={n: dict(kind=k, record=_surface_record(df.surfaces[n]))
                                 for n, k in sorted(sel.sidecar_surfaces.items())},
                matrices=run.renamed, skins=skins, borders=borders, read=dict(material_keys=keys), sensor_sets=sets,
                data=run.data, calibration=cal, omitted=sel.omitted, warnings=sel.warnings,
                annotations=sel.annotations, g4ref=dict(args=args, refuse=None if not sel.omitted else
                                                        "lossy: omitted surfaces (see omitted)"))  # fmt: skip


def _write(path: Path, data: bytes) -> None:
    """Write a file atomically (a temporary file moved into place)."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _prepare(df_or_path, calibration, sensors, options: dict) -> _Run:
    inp = _read_input(df_or_path)
    df, chain, values = _calibrate(inp, calibration)
    notes: list = []
    if options["scope"] == "active":
        df, notes = _active(df)
    if options["names"] == "geant4":
        df = _geant4_names(df)
    sel = _Selection(annotations=list(notes))
    _select_surfaces(df, options["on_lossy"], sel)
    _select_materials(df, sel)
    _check(df, sel, tuple(options["target"]))
    _check_structure(df, sel)
    if isinstance(sensors, SensorSet) or sensors is None:
        sset = sensors
    elif sensors in inp.sets:
        sset = inp.sets[sensors]
    elif inp.sets:
        raise ExportError(f"no sensor set {sensors!r} (the detector file has {sorted(inp.sets)})")
    else:
        sset = None
        sel.annotations.append("no sensor set: the detector file was given without one")
    cols = inp.columns.get(sset.name, {}) if sset is not None else {}
    return _Run(df, sel, sset, cols, inp.data, chain, values, options)


def export(df_or_path, path, *, calibration=None, sensors="default", scope="active", names="as_stored",  # noqa: PLR0913 -- the documented signature (DESIGN.md 4.7)
           target=TARGETS, on_lossy="error", sidecar=True) -> ExportReport:  # fmt: skip
    """Write a detector file (a ``.lumen`` path, a DetectorFile or an ImportResult) as GDML at ``path``.

    See the module docstring. ``calibration``: a calibration name (found beside the detector file and on the
    detector search path) or a ``Calibration``, applied before writing (``lumen.format.apply``); its sensor values
    go to the sidecar. ``sensors``: the sensor set (name or SensorSet) the sidecar and the g4ref arguments carry.
    ``target``: the Geant4 versions every element must load in. ``sidecar``: write ``<path>.lumen.json``.
    """
    options = dict(scope=scope, names=names, on_lossy=on_lossy, target=list(target))
    for k, allowed in (
        ("scope", ("active", "full")),
        ("names", ("as_stored", "geant4")),
        ("on_lossy", ("error", "omit")),
    ):
        if options[k] not in allowed:
            raise ValueError(f"{k}={options[k]!r}: one of {allowed}")
    run = _prepare(df_or_path, calibration, sensors, options)
    run.digest = io.content_digest(io.to_files(run.df))
    text, run.renamed = _gdml_text(run.df, run.sel, run.digest)
    path = Path(path)
    data = text.encode()
    _write(path, data)
    sha = io.sha256_bytes(data)
    lossless = not run.sel.omitted
    args = _g4ref_args(run.sset, lossless, run.sel)
    side = None
    if sidecar:
        side = Path(str(path) + SIDECAR_SUFFIX)
        _write(side, io.dumps(_sidecar(run, path, sha, args)).encode())
    return ExportReport(str(path), sha, None if side is None else str(side), run.digest, scope, names, tuple(target),
                        lossless, run.chain, tuple(run.sel.warnings), tuple(run.sel.annotations),
                        tuple(run.sel.omitted), tuple(args))  # fmt: skip


# ------------------------------------------------------------------------------------------------------ read back
def _restore_materials(df: schema.DetectorFile, side: dict) -> dict:
    mats = dict(df.materials)
    for name, r in side["materials"].items():
        m = mats[name]
        props = dict(m.properties)
        for k, (origin, note) in r.get("properties", {}).items():
            props[k] = dataclasses.replace(props[k], origin=origin, note=note)
        props.update({k: _prop_back(d) for k, d in r.get("sidecar_properties", {}).items()})
        inert = {**m.inert, **{k: np.zeros(shape) for k, shape in r.get("inert_empty", {}).items()}}
        optics = _optics_back(r["optics"]) if "optics" in r else m.optics
        mats[name] = dataclasses.replace(m, properties=props, inert=inert, optics=optics)
    return mats


def _restore_surfaces(df: schema.DetectorFile, side: dict) -> dict:
    surfs = dict(df.surfaces)
    for name, r in side["surfaces"].items():
        s = surfs[name]
        props = {k: dataclasses.replace(p, origin=r["properties"][k][0], note=r["properties"][k][1])
                 if k in r.get("properties", {}) else p for k, p in s.properties.items()}  # fmt: skip
        typ, model, finish = r.get("enums", (s.type, s.model, s.finish))
        surfs[name] = dataclasses.replace(s, type=typ, model=model, finish=finish, properties=props,
                                          optics=_optics_back(r["optics"]) if "optics" in r else s.optics,
                                          inert=tuple(r.get("inert", s.inert)))  # fmt: skip
    for name, d in side["recipe_surfaces"].items():
        surfs[name] = _surface_back(name, d["record"])
    return surfs


def _merge(read: tuple, entries: list, make) -> tuple:
    """Return the records in stored order: those the GDML holds (``read``, in order) and the sidecar's by index."""
    by_index = {i: (rec, written) for i, rec, written in entries}
    out, it = [], iter(read)
    for i in range(len(read) + sum(1 for _, w in by_index.values() if not w)):
        if i in by_index and not by_index[i][1]:
            out.append(make(*by_index[i][0]))
            continue
        rec = next(it)
        out.append(make(*by_index[i][0]) if i in by_index else rec)
    return tuple(out)


def _restore(df: schema.DetectorFile, side: dict) -> schema.DetectorFile:
    mats, surfs = _restore_materials(df, side), _restore_surfaces(df, side)
    for kind, owner, key, stored in side["matrices"]:
        table = mats if kind == "materials" else surfs
        rec = table[owner]
        table[owner] = dataclasses.replace(rec, properties={**rec.properties, key: dataclasses.replace(
            rec.properties[key], matrix=stored)})  # fmt: skip
    h = side["detector"]
    return dataclasses.replace(
        df, name=h["name"], world=h["world"], detector_volume=h["detector_volume"],
        wavelength_range=tuple(h["wavelength_range_nm"]), outside=h["outside"], outside_surface=h["outside_surface"],
        build_options=h["build_options"], provenance=h["provenance"], report=tuple(h["report"]), materials=mats,
        surfaces=surfs, skins=_merge(df.skins, side["skins"], schema.Skin),
        borders=_merge(df.borders, side["borders"], schema.Border))  # fmt: skip


def read_export(path) -> importer.ImportResult:
    """Read an exported GDML back (``read_gdml``) and re-apply its sidecar: the detector file and its sensor set.

    Without a sidecar it is ``read_gdml``'s detector file. A sidecar whose GDML sha256 differs is refused.
    """
    path = Path(path)
    side_path = Path(str(path) + SIDECAR_SUFFIX)
    if not side_path.exists():
        df = importer.read_gdml(path)
        return importer.ImportResult(df, report=df.report)
    side = json.loads(side_path.read_text())
    if side.get("schema") != SIDECAR_SCHEMA:
        raise ExportError(f"{side_path}: schema {side.get('schema')!r}; this lumen reads {SIDECAR_SCHEMA}")
    if side["gdml"]["sha256"] != io.sha256_bytes(path.read_bytes()):
        raise ExportError(f"{side_path} belongs to another GDML (its sha256 differs from {path.name}'s)")
    keys = {m: frozenset(k) for m, k in side["read"]["material_keys"].items()}
    df = _restore(importer.read_gdml(path, material_keys=keys or None), side)
    sets = {n: sensorsets.from_json(s["set"], s["ids"]) for n, s in side["sensor_sets"].items()}
    cols: dict = next((s["columns"] for s in side["sensor_sets"].values()), {})
    return importer.ImportResult(df, sets, dict(side["data"]), df.report, cols)


__all__ = ["ExportError", "ExportReport", "export", "read_export"]
