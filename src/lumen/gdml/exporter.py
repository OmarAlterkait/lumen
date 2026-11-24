"""GDML export: a detector file written as a GDML that Geant4 10.7 and 11.2 read as lumen does.

A scintillating material carries both releases' scintillation keys (Geant4 10.7 reads only FAST/SLOWCOMPONENT, ...,
YIELDRATIO; 11 only SCINTILLATIONCOMPONENTi, ...: ``lumen.rules.keys.geant4_10_scintillation``); lumen's import reads
the pair back as one (``lumen.build.import_rules``).

    report = lumen.gdml.export("my_detector.lumen", "my_detector.gdml")    # ExportReport: what was written, warnings

The export is the detector as lumen simulates it -- the whole world's geometry, its materials with their optical
properties and its optical surfaces -- for Geant4 to run (a Geant4 control of lumen) or for another tool to read. It
holds what Geant4 can read; what it cannot (a surface model with no Geant4 form, a recipe property no Geant4 key
names) is left out and listed. Two exports of one detector file are byte-identical.

Numbers (``lumen.gdml.units``): each value is the shortest repr of the stored double moved to Geant4's internal unit
by exact decimal arithmetic, written as a bare literal (no unit expression, which Geant4's evaluator reads otherwise:
``lumen.gdml.reader``) and checked to read back exactly, so lumen's reader gives back the stored doubles. Order:
what Geant4's one-pass reader (G4GDMLRead*) needs, placements in their stored order. Every opticalsurface attribute is
written by name (G4GDMLReadSolids::OpticalSurfaceRead falls through to dichroic, Detector_LUT and x_ray on a missing
one).

Fails closed (``ExportError``; the messages: docs/reference/errors.md, "Exporting to GDML"). Lossy (``on_lossy``): a
surface whose optics model has no Geant4 form raises under "error"; under "omit" it is left out and the export is
marked ``lossless: false``. The export names no sensors: Geant4 reads the sensitive volumes from its own run
configuration, by the names written here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np

from lumen import _env
from lumen.format import atomic, io, schema, sensorsets
from lumen.format import calibrations as cal
from lumen.format import solids as ss
from lumen.format.apply import GEANT4_DEFAULT, apply
from lumen.format.check import check_bindings
from lumen.gdml.lexical import TINY
from lumen.gdml.units import shift
from lumen.rules.keys import FINISHES, GEANT4_KEYS, MODELS, TYPES

MM_PLACES, MEV_PLACES = 3, -6  # m -> mm and eV -> MeV, as decimal shifts
SCHEMA_LOCATION = "http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd"
NATIVE_MODELS = ("geant4",)  # optics models a GDML opticalsurface expresses exactly


class ExportError(ValueError):
    """A detector file the exporter cannot write as a GDML Geant4 reads as lumen does: the message says why."""


@dataclass(frozen=True)
class ExportReport:
    """What an export wrote and decided.

    ``omitted`` lists what was left out (``on_lossy="omit"``; ``lossless``: nothing was), ``warnings`` the known
    differences in Geant4's reading, and ``annotations`` what the GDML does not hold and why.
    """

    path: str
    sha256: str
    calibration: str | None = None
    warnings: tuple[str, ...] = ()
    annotations: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()

    @property
    def lossless(self) -> bool:
        """Return whether nothing was left out."""
        return not self.omitted

    def __str__(self) -> str:
        """Summarise the export: file, scope, losslessness and warnings."""
        lines = [
            f"{self.path}  ({self.sha256})",
            f"  lossless: {'yes' if self.lossless else 'NO'}; {len(self.warnings)} warning(s), "
            f"{len(self.annotations)} annotation(s); calibration: {self.calibration or 'none'}",
        ]
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


def _pos(tag: str, v) -> str:
    """Return a position element in mm."""
    return _vec(tag, v, "mm", _mm)


def _vec(tag: str, v, unit: str | None, fmt) -> str:
    pairs = ([("unit", unit)] if unit else []) + [(c, fmt(x)) for c, x in zip("xyz", v, strict=True)]
    return _el(tag, pairs)


# ---------------------------------------------------------------------------------------------------- the inputs
def _calibrate(df, root: Path | None, calibration) -> tuple[schema.DetectorFile, str | None]:
    """Return (the detector file with the calibration applied, its name), the calibration checked against the
    detector file and its sensor set (``detector_digest``)."""  # noqa: D205, D209
    if calibration is None:
        return df, None
    if root is None:
        raise ExportError("calibration=: give the detector file's path, so the calibration is checked against it "
                          "(a DetectorFile in memory has no sensor set to check it with)")  # fmt: skip
    c = calibration if isinstance(calibration, cal.Calibration) else \
        cal.read(cal.find(str(calibration), [root / "calibrations", *_env.detector_path()]))  # fmt: skip
    cal.check_digest(c, io.content_digest(io.to_files(df), sensorsets.files(sensorsets.read(root))), root)
    from lumen.rules.optics import VALUE_RULES

    return apply(df, c.values, VALUE_RULES), c.name


# ---------------------------------------------------------------------------------------------------- the selection
@dataclass
class _Selection:
    """What goes into the GDML, and what is left out."""

    surfaces: list = field(default_factory=list)  # written, sorted by name
    skins: list = field(default_factory=list)  # indices into df.skins written
    borders: list = field(default_factory=list)  # indices into df.borders written
    unwritten_props: dict = field(default_factory=dict)  # material -> recipe keys that are no Geant4 key
    warnings: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    omitted: list = field(default_factory=list)


def _select_surfaces(df: schema.DetectorFile, on_lossy: str, sel: _Selection) -> None:
    for name in sorted(df.surfaces):
        model = df.surfaces[name].optics.name
        if model in NATIVE_MODELS:
            sel.surfaces.append(name)
        elif on_lossy == "omit":
            sel.omitted.append(f"surface {name!r}: optics model {model!r} has no Geant4 form")
        else:
            raise ExportError(f"surface {name!r}: optics model {model!r} has no Geant4 form; pass on_lossy='omit' "
                              "to write the GDML without it (marked lossy, not a Geant4 reference)")  # fmt: skip
    written = set(sel.surfaces)
    sel.skins = [i for i, k in enumerate(df.skins) if k.surface in written]
    sel.borders = [i for i, b in enumerate(df.borders) if b.surface in written]
    for k in df.skins:
        if k.surface not in written:
            sel.omitted.append(f"skin {k.name!r} on {k.volume!r} (surface {k.surface!r})")
    for b in df.borders:
        if b.surface not in written:
            sel.omitted.append(f"border {b.name!r} {b.pv1!r} -> {b.pv2!r} (surface {b.surface!r})")


def _select_materials(df: schema.DetectorFile, sel: _Selection) -> None:
    for name in sorted(df.materials):
        m = df.materials[name]
        recipe = sorted(k for k, p in m.properties.items() if p.origin != "source")
        geant4 = [k for k in recipe if k in GEANT4_KEYS]
        sel.unwritten_props[name] = [k for k in recipe if k not in GEANT4_KEYS]
        if geant4:
            sel.warnings.append(f"material {name!r}: the recipe's tables {geant4} ({m.optics.name}) are written as "
                                "Geant4 tables; Geant4 reads them with its own semantics")  # fmt: skip
        if sel.unwritten_props[name]:
            sel.annotations.append(f"material {name!r}: recipe properties {sel.unwritten_props[name]} are not Geant4 "
                                   "keys: not written")  # fmt: skip
        if (m.optics.name, m.optics.settings) != ("geant4", {}):
            sel.annotations.append(f"material {name!r}: optics model {m.optics.name!r} is not written")


def _enum_names(s: schema.Surface) -> tuple[str, str, str]:
    """Return a surface's (model, finish, type) by name; refuse what Geant4's GDML reader does not know."""
    out = []
    for what, v, names in (("model", s.model, MODELS), ("finish", s.finish, FINISHES), ("type", s.type, TYPES)):
        t = str(v).strip()
        if t == "coated":
            raise ExportError(f"surface {s.name!r}: a coated surface; GDML has no coated type (it reads as x_ray)")
        if t not in names:
            raise ExportError(f"surface {s.name!r}: {what} {v!r} is not one Geant4 knows (one of {list(names)})")
        out.append(t)
    return out[0], out[1], out[2]


def _check(df: schema.DetectorFile, sel: _Selection) -> None:
    """Refuse what Geant4 would read differently (the module docstring); add the warnings."""
    check_bindings([(df.skins[i].name, df.skins[i].volume) for i in sel.skins],
                   [(df.borders[i].name, df.borders[i].pv1, df.borders[i].pv2) for i in sel.borders],
                   [p.name for p in df.placements], ExportError)  # fmt: skip
    for name in sel.surfaces:
        _enum_names(df.surfaces[name])
    expo = sorted(n for n, m in df.materials.items() if m.optics.settings.get("wls_time_profile") == "exponential")
    if expo:
        sel.warnings.append(f"materials {expo} record G4OpWLS's exponential time profile: run Geant4 with it "
                            "(G4OpWLS::UseTimeProfile('exponential')); its default is a fixed delay")  # fmt: skip
    n0x = sum("0x" in n for n in [*df.solids, *df.volumes, *df.materials, *(p.name for p in df.placements)])
    if n0x:
        sel.warnings.append(f"{n0x} name(s) contain '0x': Geant4's GDML reader strips the name from there on (its "
                            "volumes and placements carry the stripped names)")  # fmt: skip


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
FLAT_EV = (1.0, 20.0)  # eV: the knots of the flat table a calibrated Geant4-default constant is written as (any span
# covering optical energies: Geant4 clamps a table outside its knots)


def _vector_form(p: schema.Property) -> schema.Property:
    """Return a Geant4-default constant a calibration set (``lumen.format.apply``) as the flat table Geant4 reads.

    G4OpBoundaryProcess reads REFLECTIVITY, TRANSMITTANCE and the unified constants with GetProperty, a table: a
    constant (a coldim 1 matrix) would be a constant property it ignores. Geant4 clamps the table outside its knots.
    """
    if p.origin != GEANT4_DEFAULT or p.energy is not None:
        return p
    v = float(p.value)
    return dataclasses.replace(p, value=np.array([v, v]), energy=np.array(FLAT_EV))


def _table_text(key: str, p: schema.Property) -> tuple[int, str]:
    """Return a property's (coldim, values): energies in MeV, values in Geant4's internal units.

    Values are in Geant4's internal units (lengths in mm: ``schema.PROPERTY_UNITS``).
    """
    p_key = schema.PROPERTY_UNITS[key][2] if key in schema.PROPERTY_UNITS else 0
    vals = [_literal(repr(float(v)), -p_key) for v in np.asarray(p.value, np.float64).reshape(-1).tolist()]
    if p.energy is None:
        return 1, vals[0]
    es = [_mev(e) for e in np.asarray(p.energy, np.float64).tolist()]
    return 2, " ".join(f"{a} {b}" for a, b in zip(es, vals, strict=True))


def _sha10(coldim: int, text: str) -> str:
    return hashlib.sha256(f"{coldim}:{text}".encode()).hexdigest()[:10]


def _matrices(df: schema.DetectorFile, sel: _Selection) -> tuple[dict, dict]:
    """Name the matrices: return (defines {name: (coldim, values)}, refs {(kind, owner, key): name}).

    A matrix is named ``<KEY>_<sha10>`` of its content (deterministic; properties with one table share it).
    """
    users = []  # (kind, owner, key, coldim, text)
    for name in sorted(df.materials):
        m = df.materials[name]
        users += [("materials", name, key, *_table_text(key, m.properties[key])) for key in sorted(m.properties)
                  if key not in sel.unwritten_props.get(name, ())]  # fmt: skip
        for key in sorted(m.inert):
            a = np.asarray(m.inert[key], np.float64)
            if a.size:
                users.append(("inert", name, key, a.shape[1], " ".join(_bare(x) for x in a.ravel().tolist())))
    for name in sel.surfaces:
        s = df.surfaces[name]
        users += [("surfaces", name, key, *_table_text(key, _vector_form(s.properties[key])))
                  for key in sorted(s.properties)]  # fmt: skip
    defines: dict = {}
    refs: dict = {}
    for kind, owner, key, cd, text in users:
        mname = f"{key}_{_sha10(cd, text)}"
        defines[mname] = (cd, text)
        refs[(kind, owner, key)] = mname
    return defines, refs


def _check_constants(defines: dict, positions: list) -> None:
    """Refuse a name Geant4's DefineMatrix would define twice (it defines ``<name>_<i>`` and ``<name>_<i>_<j>``)."""
    seen = set(defines) | {p[0] for p in positions}
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
def _post_order(items: dict, deps, key=None) -> list:
    """Return the names of ``items`` sorted (by ``key``), each after the names ``deps(name)`` gives (defined ones)."""
    out, done = [], set()
    for root in sorted(items, key=key):
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
    """Return the vertex defines and each tessellated solid's facet vertex names (a tet's lunit).

    A define is (name, (x, y, z) m); a tet's points are written in mm (lunit "mm": Geant4 scales them by it).
    """
    defines, facets = [], {}
    for name in solid_order:
        s = df.solids[name]
        if s.type == "tet":
            facets[name] = "mm"
            defines += [(f"{name}__v{i + 1}", tuple(v)) for i, v in enumerate(s.vertices)]
        elif s.type == "tessellated":
            v = np.ascontiguousarray(np.asarray(df.meshes[str(s.mesh)], np.float64).reshape(-1, 3))
            key = v.view(np.dtype((np.void, v.dtype.itemsize * 3))).ravel()  # by bits: -0.0 is not +0.0
            _, first, inv = np.unique(key, return_index=True, return_inverse=True)
            order = np.argsort(first, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            ids = rank[inv.ravel()].reshape(-1, 3)
            at = first[order]
            defines += [(f"{name}__v{k}", tuple(row)) for k, row in enumerate(v[at].tolist())]
            facets[name] = ids
    return defines, facets


def _solid_params(s: schema.Solid) -> list:
    try:
        spell = ss.dimensions(s.type)
    except KeyError:
        raise ExportError(f"solid {s.name!r}: type {s.type!r} has no GDML spelling in the exporter (supported: "
                          f"{sorted(ss.ATTRIBUTES)})") from None  # fmt: skip
    unknown = sorted(set(s.params) - set(spell))
    if unknown:
        raise ExportError(f"solid {s.name!r} ({s.type}): attributes {unknown} are not {s.type} attributes")
    pairs = [("name", s.name)]
    pairs += [(name, (_mm if dim == ss.L else _bare)(s.params[name])) for name, dim in spell.items()
              if name in s.params]  # fmt: skip
    dims = set(spell.values())
    if ss.L in dims or s.type in ("polycone", "polyhedra", "genericPolycone", "genericPolyhedra", "xtru"):
        pairs.append(("lunit", "mm"))
    if ss.A in dims:
        pairs.append(("aunit", "rad"))
    return pairs


def _transform_lines(tr: schema.Transform, pos: str, rot: str) -> str:
    out = ""
    if not _plus_zero(tr.position):
        out += _pos(pos, tr.position)
    if not _plus_zero(tr.rotation):
        out += _vec(rot, tr.rotation, "rad", _bare)
    return out


def _solid_children(s: schema.Solid, facets: dict) -> list:
    kind = ss.CHILDREN.get(s.type)
    if kind in ("zplane", "rzpoint"):
        return [_el(kind, list(zip(ss.PLANE_ATTRIBUTES[kind], map(_mm, p), strict=True))) for p in s.planes]
    if kind == "xtru":
        out = [_el("twoDimVertex", [("x", _mm(x)), ("y", _mm(y))]) for x, y in s.vertices]
        out += [_el("section", [("zOrder", str(i)), *zip(ss.SECTION_ATTRIBUTES[:3], map(_mm, sec[:3]), strict=True),
                                (ss.SECTION_ATTRIBUTES[3], _bare(sec[3]))])
                for i, sec in enumerate(s.sections)]  # fmt: skip
        return out
    if kind == "facets":
        n = s.name
        return [f'<triangular vertex1="{_q(n)}__v{a}" vertex2="{_q(n)}__v{b}" vertex3="{_q(n)}__v{c}" type="ABSOLUTE"/>'
                for a, b, c in facets[n].tolist()]  # fmt: skip
    if kind == "boolean":
        first, second = s.transforms
        return [_el("first", [("ref", s.refs[0])]), _el("second", [("ref", s.refs[1])]),
                _transform_lines(second, "position", "rotation")
                + _transform_lines(first, "firstposition", "firstrotation")]  # fmt: skip
    if kind == "nodes":
        return [_el("multiUnionNode", [("name", f"{s.name}__n{k}")], _el("solid", [("ref", r)])
                    + _transform_lines(tr, "position", "rotation"))
                for k, (r, tr) in enumerate(zip(s.refs, s.transforms, strict=True))]  # fmt: skip
    if kind == "scaled":
        return [_el("solidref", [("ref", s.refs[0])]), _vec("scale", s.scale, None, _bare)]
    return []


def _solid(s: schema.Solid, facets: dict) -> list:
    pairs = _solid_params(s)  # refuses an unknown type or attribute
    if s.type == "tet":
        pairs = [pairs[0]] + [(f"vertex{i}", f"{s.name}__v{i}") for i in range(1, 5)] + [("lunit", facets[s.name])]
    kids = [k for k in _solid_children(s, facets) if k]
    if not kids:
        return ["  " + _el(s.type, pairs)]
    return ["  " + f"<{s.type} {_attrs(pairs)}>", *("   " + k for k in kids), f"  </{s.type}>"]


def _material(m: schema.Material, refs: dict) -> list:
    pairs = [("name", m.name)] + ([("state", m.state)] if m.state else [])
    c = m.composition
    if c.z is not None:
        pairs.append(("Z", _bare(c.z)))
    kids = []
    if m.temperature_k is not None:
        kids.append(_el("T", [("unit", "K"), ("value", _bare(m.temperature_k))]))
    if m.pressure_pa is not None:
        kids.append(_el("P", [("unit", "pascal"), ("value", _bare(m.pressure_pa))]))
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


def _referenced_first(targets):
    """Return a sort key: the names in ``targets`` first, then by name.

    Geant4 names a material, element or isotope stripped of its pointer suffix and resolves a reference to the first
    record of that stripped name (G4GDMLReadMaterials::GetIsotope, GetElement, GetMaterial), as the import did
    (``importer._first_by_stripped``): each reference's target is written before the others of its name, so the file
    reads back to the same references.
    """
    return lambda name: (name not in targets, name)


def _materials(df: schema.DetectorFile, refs: dict) -> list:
    out = [" <materials>"]
    comps = [x for m in df.materials.values() for x in m.composition.components]
    used = {kind: {x.ref for x in comps if x.kind == kind} for kind in ("element", "material")}
    used["isotope"] = {x.ref for e in df.elements.values() for x in e.isotopes} | {x.ref for x in comps
                                                                                  if x.kind == "isotope"}  # fmt: skip
    for name in sorted(df.isotopes, key=_referenced_first(used["isotope"])):
        i = df.isotopes[name]
        out.append("  " + _el("isotope", [("name", name), ("Z", str(int(i.z))), ("N", str(int(i.n)))],
                              _el("atom", [("unit", "g/mole"), ("value", _bare(i.molar_mass_g_mol))])))  # fmt: skip
    for name in sorted(df.elements, key=_referenced_first(used["element"])):
        e = df.elements[name]
        pairs = [("name", name)] + ([("formula", e.formula)] if e.formula else []) + \
            ([("Z", _bare(e.z))] if e.z is not None else [])  # fmt: skip
        inner = "".join(_el("fraction", [("n", _bare(x.fraction)), ("ref", x.ref)]) for x in e.isotopes)
        if e.molar_mass_g_mol is not None:
            inner = _el("atom", [("unit", "g/mole"), ("value", _bare(e.molar_mass_g_mol))]) + inner
        out.append("  " + _el("element", pairs, inner))

    def parts(n):
        return [x.ref for x in df.materials[n].composition.components if x.kind == "material"]

    order = _post_order(df.materials, parts, _referenced_first(used["material"]))
    for name in order:
        out += _material(df.materials[name], refs)
    return out + [" </materials>"]


def _physvol(p: schema.Placement) -> str:
    pairs = [("name", p.name)] + ([("copynumber", str(int(p.copy)))] if p.copy is not None else [])
    inner = _el("volumeref", [("ref", p.volume)])
    if not _plus_zero(p.position):
        inner += _pos("position", p.position)
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


def _gdml_text(df: schema.DetectorFile, sel: _Selection, digest: str) -> str:
    """Return the GDML text."""
    defines, refs = _matrices(df, sel)
    solid_order = _post_order(df.solids, lambda n: df.solids[n].refs)
    positions, facets = _vertices(df, solid_order)
    _check_constants(defines, positions)
    src = df.provenance.get("source", {})
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           _comment(f"lumen GDML export of detector {df.name}, content "
                    f"{digest}; source {src.get('file', '?')} {src.get('sha256', '')}. Numbers are bare literals in "
                    "Geant4's internal units (mm, MeV, ns, rad)."),
           '<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
           f'xsi:noNamespaceSchemaLocation="{SCHEMA_LOCATION}">',
           " <define>"]  # fmt: skip
    out += ["  " + _el("matrix", [("name", n), ("coldim", str(cd)), ("values", t)]) for n, (cd, t) in
            sorted(defines.items())]  # fmt: skip
    out += ["  " + _pos("position", v).replace("<position ", f'<position name="{_q(n)}" ', 1) for n, v in positions]
    out += [" </define>"] + _materials(df, refs) + [" <solids>"]
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
    out += [" " + _el("setup", [("name", "Default"), ("version", "1.0")], _el("world", [("ref", df.world)])),
            "</gdml>"]  # fmt: skip
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------------ the export
def _geant4_10_scintillation(df: schema.DetectorFile, sel: _Selection) -> schema.DetectorFile:
    """Return the detector with each scintillating material's Geant4-10 keys added beside its Geant4-11 keys
    (``lumen.rules.keys.geant4_10_scintillation``): Geant4 10.7's G4Scintillation reads only the 10 keys, 11's only
    the 11 keys, so both read the light lumen emits. Each is an annotation; a material with no 10.7 form a warning."""  # noqa: D205, D209
    from lumen.rules import keys

    mats = dict(df.materials)
    for name, m in df.materials.items():
        extra, note = keys.geant4_10_scintillation(m.properties)
        if extra:
            mats[name] = dataclasses.replace(m, properties={**m.properties, **extra})
            sel.annotations.append(f"material {name!r}: {note}")
        elif note:
            sel.warnings.append(f"material {name!r}: {note}")
    return dataclasses.replace(df, materials=mats)


def export(df_or_path, path, *, calibration=None, on_lossy="error") -> ExportReport:
    """Write a detector file (a ``.lumen`` path or a DetectorFile) as GDML at ``path``; see the module docstring.

    ``calibration``: a calibration name (found beside the detector file and on the detector search path) or a
    ``Calibration``, applied before writing (``lumen.format.apply``). The file is written atomically.
    """
    if on_lossy not in ("error", "omit"):
        raise ValueError(f"on_lossy={on_lossy!r}: one of ('error', 'omit')")
    root = None if isinstance(df_or_path, schema.DetectorFile) else Path(df_or_path)
    df, name = _calibrate(io.read(root) if root else df_or_path, root, calibration)
    digest = io.content_digest(io.to_files(df))
    sel = _Selection()
    df = _geant4_10_scintillation(df, sel)
    _select_surfaces(df, on_lossy, sel)
    _select_materials(df, sel)
    _check(df, sel)
    _check_structure(df, sel)
    data = _gdml_text(df, sel, digest).encode()
    path = Path(path)
    atomic.write_bytes(path, data)
    return ExportReport(str(path), io.sha256_bytes(data), name, tuple(sel.warnings), tuple(sel.annotations),
                        tuple(sel.omitted))  # fmt: skip


__all__ = ["ExportError", "ExportReport", "export"]
