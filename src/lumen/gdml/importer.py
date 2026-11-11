"""The GDML importer: a GDML file and a recipe to a detector file (``lumen.format``), the full world kept.

    result = import_gdml("sk.gdml", recipe)      # ImportResult(detector_file, sensor_sets, calibrations, report)

A plain function: pure and deterministic (the same bytes and recipe give a byte-identical detector file). It reads
Geant4 semantics only and makes no physics decision; the recipe's decisions are applied after, each recorded with
``origin="recipe"``. Units are converted by exact decimal scaling of plain literals (``lumen.gdml.units``).

Fails closed (``ImportRefused``) where Geant4's reading is ambiguous or lumen's would differ:

- an opticalsurface without all four of model, finish, type and value;
- a repeated border (the same ordered pair of placements twice);
- a border physvolref whose FULL name is carried by more than one physvol (Geant4 binds only the first; names that
  repeat only once the ``0x`` suffix is stripped are distinct to both Geant4 and lumen and are accepted);
- two skins on one logical volume;
- a material without a composition (density and components, or Z and molar mass).

Before these, the reader repairs the exporter defects Geant4 refuses (``reader.REPAIRS``), each with a warning
(``reader.GDMLRepairWarning``); the kinds repaired are in the provenance (``repairs``) and every repair in the report.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np

from lumen.format import schema
from lumen.gdml import reader, tokens
from lumen.gdml.units import ENERGY_EV, LENGTH_M, LITERAL, MM_PER_M, PRODUCT

IMPORTER_VERSION = "1"
LENGTH_KEYS = {"ABSLENGTH", "RAYLEIGH", "WLSABSLENGTH", "WLSABSLENGTH2", "MIEHG", "COATEDTHICKNESS"}
NON_FINITE = 1e30  # a non-finite table value is stored as +-1e30 (Geant4 does not read inf), and reported
MEV_TO_EV = "1000000"


class InertKeysWarning(UserWarning):
    """Material keys no Geant4 optical process reads, kept inert (unread, stored verbatim)."""


class ImportRefused(ValueError):  # noqa: N818 (the name of the documented error catalogue)
    """The GDML has a construct whose Geant4 reading is ambiguous or that lumen reads differently: refused."""


@dataclass(frozen=True)
class ImportResult:
    """What an importer returns: the detector file, its sensor sets and shipped calibrations, and the report."""

    detector_file: schema.DetectorFile
    sensor_sets: dict = field(default_factory=dict)
    calibrations: dict = field(default_factory=dict)
    report: tuple[str, ...] = ()
    sensor_columns: dict = field(default_factory=dict)  # per-sensor columns of the id map (channels, types)


def property_key(name: str) -> str:
    """Return a property name as Geant4 reads it (G4GDMLRead::StripName): everything from '0x' on dropped."""
    i = name.find("0x")
    return name if i < 0 else name[:i]


# ------------------------------------------------------------------------------------------------------ geometry
def _transform(t: reader.Transform) -> schema.Transform:
    return schema.Transform(tuple(float(x) for x in t.rotation), tuple(float(x) for x in t.position))


def _mesh(doc: reader.GDMLDocument, s: reader.Solid) -> np.ndarray:
    """Return a tessellated solid's triangles (k, 3, 3) in m: relative facets resolved, quadrangles split."""
    tris = []
    for refs, ftype in s.facets:
        p = [np.asarray(doc.positions[r], np.float64) for r in refs]
        if ftype == "RELATIVE":
            p = [p[0]] + [p[0] + q for q in p[1:]]
        tris.append(p[:3])
        if len(p) == 4:
            tris.append([p[0], p[2], p[3]])
    if not tris:
        raise ImportRefused(f"tessellated {s.name!r}: no facets")
    return np.asarray(tris, np.float64)


def solids(doc: reader.GDMLDocument) -> tuple[dict, dict, list]:
    """Return (solids, meshes, report lines) of the document."""
    out, meshes, rep = {}, {}, []
    for name, s in doc.solids.items():
        kw: dict = {}
        if s.type in reader.BOOLEAN or s.type == "multiUnion":
            kw.update(refs=tuple(s.refs), transforms=tuple(_transform(t) for t in s.transforms))
        elif s.type == "scaledSolid":
            kw.update(refs=tuple(s.refs), scale=tuple(float(x) for x in s.transforms[0].scale))
        elif s.type == "tessellated":
            meshes[name] = _mesh(doc, s)
            kw["mesh"] = name
        elif s.type == "tet":
            refs, lunit_mm = s.tet
            # G4GDMLReadSolids::TetRead multiplies the (already converted) vertex positions by lunit again: kept, so
            # a tet with lunit="cm" is 10 times larger, as in Geant4
            kw["vertices"] = tuple(tuple(float(c) * lunit_mm for c in doc.positions[r]) for r in refs)
            if lunit_mm != 1.0:
                rep.append(f"tet {name!r}: vertices scaled by its lunit ({lunit_mm} mm) a second time, as Geant4 does")
        elif s.type == "xtru":
            kw.update(vertices=tuple(s.vertices), sections=tuple(s.sections))
        if s.planes:
            kw["planes"] = tuple(tuple(p) for p in s.planes)
        out[name] = schema.Solid(name, s.type, dict(s.params), **kw)
    return out, meshes, rep


def structure(doc: reader.GDMLDocument) -> tuple[dict, tuple]:
    """Return (volumes, placements): every placement in document order, each volume's daughters in Geant4's order."""
    placements: list = []
    volumes: dict = {}
    for name, v in doc.volumes.items():
        idx = []
        for pv in v.placements:
            t = pv.transform
            idx.append(len(placements))
            placements.append(
                schema.Placement(
                    pv.name,
                    pv.child,
                    name,
                    pv.copy,
                    tuple(float(x) for x in t.position),
                    tuple(float(x) for x in t.rotation),
                    None if t.scale is None else tuple(float(x) for x in t.scale),
                )
            )
        aux = tuple(_aux(a) for a in v.aux)
        volumes[name] = schema.Volume(name, v.solid, v.material, tuple(idx), v.assembly, aux)
    return volumes, tuple(placements)


def _aux(a: reader.Aux) -> schema.Aux:
    return schema.Aux(a.type, a.value, a.unit, tuple(_aux(c) for c in a.children))


# ------------------------------------------------------------------------------------------------------ properties
def _energy_ev(tok: str, internal: float) -> float:
    m = PRODUCT.match(tok)
    if LITERAL.match(tok):
        return float(Decimal(tok) * Decimal(MEV_TO_EV))
    if m and m.group(2) in ENERGY_EV:
        return float(Decimal(m.group(1)) * Decimal(ENERGY_EV[m.group(2)]))
    return internal * 1.0e6


def _length_m(tok: str, internal: float) -> float:
    m = PRODUCT.match(tok)
    if LITERAL.match(tok):
        return float(Decimal(tok) * Decimal(LENGTH_M["mm"]))
    if m and m.group(2) in LENGTH_M:
        return float(Decimal(m.group(1)) * Decimal(LENGTH_M[m.group(2)]))
    return internal / MM_PER_M


def read_property(doc: reader.GDMLDocument, key: str, ref: str, what: str, rep: list) -> schema.Property:
    """One property as Geant4 reads it: energies in eV, lengths in m, other values as Geant4 holds them.

    Notes
    -----
    G4GDMLReadDefine::MatrixRead and G4GDMLReadMaterials / Solids::PropertyRead: coldim 1 is a constant (its first
    value), coldim 2 an (energy, value) table; the matrix ``unit`` attribute is not read.
    """
    try:
        m = doc.matrix(ref)
    except reader.GDMLError as err:
        raise ImportRefused(f"{what}.{key}: {err}") from None
    toks = doc.matrix_text.get(ref, [])
    length = key in LENGTH_KEYS
    if m.shape[1] == 1:
        c = float(m[0, 0])
        if length:
            c = _length_m(toks[0], c) if toks else c / MM_PER_M
        return schema.Property(key, np.asarray(c, np.float64), None, ref)
    if m.shape[1] != 2:
        raise ImportRefused(f"{what}.{key}: matrix {ref!r} has {m.shape[1]} columns (a property needs 1 or 2)")
    e = np.array([_energy_ev(toks[2 * i], m[i, 0]) for i in range(m.shape[0])], np.float64)
    if length:
        v = np.array([_length_m(toks[2 * i + 1], m[i, 1]) for i in range(m.shape[0])], np.float64)
    else:
        v = m[:, 1].astype(np.float64)
    if len(e) < 2 or np.any(np.diff(e) <= 0.0):
        raise ImportRefused(f"{what}.{key}: energies must be strictly ascending with >= 2 knots (got {e})")
    bad = ~np.isfinite(v)
    if bad.any():
        rep.append(f"{what}.{key}: {int(bad.sum())} non-finite value(s) stored as +-{NON_FINITE:g}")
        v = np.where(bad, np.sign(np.nan_to_num(v, nan=1.0, posinf=1.0, neginf=-1.0)) * NON_FINITE, v)
    return schema.Property(key, v, e, ref)


@dataclass
class _Inert:
    """The optical keys to read (``keep``) and the other properties' matrices, kept verbatim (``found``)."""

    keep: frozenset
    found: dict = field(default_factory=dict)


def _properties(doc, props, what, rep, skip=(), inert: _Inert | None = None):
    """Read properties; with ``inert``, the keys it does not keep go verbatim into ``inert.found``, unread."""
    out: dict = {}
    for name, ref in props:
        key = property_key(name)
        if key in out or (inert is not None and key in inert.found):
            raise ImportRefused(f"{what}: property {key} given twice")
        if key in skip:
            continue
        if inert is not None and key not in inert.keep:
            inert.found[key] = doc.matrix(ref)  # the reader has refused (or, repaired, dropped) a ref to no matrix
            continue
        out[key] = read_property(doc, key, ref, what, rep)
    return out


# ------------------------------------------------------------------------------------------------------- materials
def materials(doc: reader.GDMLDocument, rep: list, keys=None) -> tuple[dict, dict, dict]:
    """Return (materials, elements, isotopes); every material must name its composition.

    ``keys``: the optical property keys to read (None: ``lumen.rules.optics.MATERIAL_KEYS``, every key a Geant4 optical
    process reads), or a mapping from material name to such keys (a material it does not name reads the default; the
    GDML export's reader uses it); a material's other properties are kept inert, unread.
    """
    isotopes = {n: schema.Isotope(n, i.z, i.n, i.molar_mass_g_mol) for n, i in doc.isotopes.items()}
    elements = {n: schema.Element(n, e.formula, e.z, e.molar_mass_g_mol,
                             tuple(schema.Component(r, fraction=f) for r, f in e.fractions))
                for n, e in doc.elements.items()}  # fmt: skip
    out = {}
    for name, m in doc.materials.items():
        comps = tuple(schema.Component(r, fraction=f) for r, f in m.fractions) + \
            tuple(schema.Component(r, atoms=n) for r, n in m.composites)  # fmt: skip
        if m.density_g_cm3 is None or not (comps or (m.z is not None and m.molar_mass_g_mol is not None)):
            raise ImportRefused(f"material {name!r} names no composition: add <D> and <fraction> or <composite> "
                                "elements, or Z and <atom>")  # fmt: skip
        for c in comps:
            if c.ref not in doc.elements and c.ref not in doc.materials:
                raise ImportRefused(f"material {name!r}: component {c.ref!r} is neither an element nor a material")
        comp = schema.Composition(m.density_g_cm3, comps, m.z, m.molar_mass_g_mol)
        from lumen.rules import optics  # (lazily: the rules import the build, which imports the importer)

        keep = keys.get(name) if isinstance(keys, Mapping) else keys
        sink = _Inert(frozenset(optics.MATERIAL_KEYS if keep is None else keep))
        props = _properties(doc, m.props, f"material {name!r}", rep, inert=sink)
        inert = sink.found
        if inert:
            rep.append(f"material {name!r}: {len(inert)} non-optical propert(ies) kept inert: {sorted(inert)}")
        out[name] = schema.Material(name, m.state, comp, m.temperature, m.pressure, props, inert=inert,
                                    mee_ev=m.mee_ev)  # fmt: skip
    return out, elements, isotopes


# -------------------------------------------------------------------------------------------------------- surfaces
SURFACE_INERT = ("GROUPVEL",)  # read by no Geant4 process on a surface (exports write Geant4's computed table)


def surfaces(doc: reader.GDMLDocument, placements: tuple, rep: list) -> tuple[dict, tuple, tuple]:
    """Return (surfaces, skins, borders), refusing the ambiguous cases of the module docstring."""
    out = {}
    for name, s in doc.surfaces.items():
        missing = [a for a in ("model", "finish", "type", "value") if getattr(s, a) is None]
        if missing:
            raise ImportRefused(f"opticalsurface {name!r} has no {', '.join(repr(a) for a in missing)} attribute. "
                                "Geant4 would read a default and lumen cannot guess: add it in the GDML and convert "
                                "again.")  # fmt: skip
        inert = tuple(sorted({property_key(k) for k, _ in s.props} & set(SURFACE_INERT)))
        props = _properties(doc, s.props, f"opticalsurface {name!r}", rep, skip=SURFACE_INERT)
        out[name] = schema.Surface(name, s.type, s.model, s.finish, float(doc.num(s.value, f"opticalsurface {name!r}")),
                              props, inert=inert)  # fmt: skip
    skins: list = []
    seen_lv: dict = {}
    for name, surf, lvs in doc.skins:
        _known(out, surf, name)
        for lv in lvs:
            if lv in seen_lv:
                raise ImportRefused(f"logical volume {lv!r} has two skin surfaces ({seen_lv[lv]!r}, {name!r}): Geant4 "
                                    "keeps the first registered; refusing the ambiguity")  # fmt: skip
            seen_lv[lv] = name
            skins.append(schema.Skin(name, surf, lv))
    names = Counter(p.name for p in placements)
    borders, pairs = [], set()
    for name, surf, refs in doc.borders:
        _known(out, surf, name)
        if len(refs) != 2:
            raise ImportRefused(f"bordersurface {name!r}: needs exactly two physvolref")
        for r in refs:
            if names.get(r, 0) != 1:
                raise ImportRefused(f"bordersurface {name!r}: physvolref {r!r} names {names.get(r, 0)} physical "
                                    "volumes; Geant4 binds only the first, so the border is ambiguous")  # fmt: skip
        if tuple(refs) in pairs:
            raise ImportRefused(f"bordersurface {name!r}: the pair {refs[0]!r} -> {refs[1]!r} has a border already")
        pairs.add(tuple(refs))
        borders.append(schema.Border(name, surf, refs[0], refs[1]))
    by_stripped = Counter(reader.g4_strip(p.name) for p in placements)
    stripped = sum(1 for b in borders for r in (b.pv1, b.pv2) if by_stripped[reader.g4_strip(r)] > 1)
    if stripped:
        rep.append(f"{stripped} border reference(s) name a physical volume whose name repeats once the 0x suffix is "
                   "stripped: bound by the full name, as Geant4 binds them; a GDML export with stripped names would "
                   "be ambiguous and is refused")  # fmt: skip
    return out, tuple(skins), tuple(borders)


def _known(surfaces_, surf, name):
    if surf not in surfaces_:
        raise ImportRefused(f"surface {name!r} refers to {surf!r}, which is not an opticalsurface")


# ----------------------------------------------------------------------------------------------------- the import
def read_gdml(path, material_keys=None, warn=True) -> schema.DetectorFile:
    """Return the detector file of a GDML file before any recipe decision (the detector volume is the world).

    ``material_keys``: the material property keys to read as optical properties (None: the Geant4 optical keys,
    ``lumen.rules.optics.MATERIAL_KEYS``); see ``materials``. With ``warn``, the keys kept inert are named in one
    ``InertKeysWarning`` (keys x materials).
    The exporter defects found are repaired with warnings (``reader.REPAIRS``; recorded in the provenance and report).
    """
    import hashlib

    doc = reader.read(path, warn=warn)
    rep = list(doc.report)
    sol, meshes, r2 = solids(doc)
    rep += r2
    vols, pls = structure(doc)
    mats, els, isos = materials(doc, rep, material_keys)
    held = {k for m in mats.values() for k in m.inert}
    if warn and held:
        n, nm = sum(len(m.inert) for m in mats.values()), sum(bool(m.inert) for m in mats.values())
        warnings.warn(
            f"gdml {Path(path).name}: {len(held)} material key(s) in {nm} "
            f"material(s) ({n} properties) are read by no Geant4 optical process and are kept inert: "
            f"{sorted(held)[:12]}; Recipe(material_keys=...) reads others (docs/reference/errors.md)",
            InertKeysWarning,
            stacklevel=2,
        )
    surfs, skins, borders = surfaces(doc, pls, rep)
    toks, r3 = tokens.collect(doc, sol, meshes, pls, mats, surfs)
    rep += r3
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    prov = dict(importer="lumen.gdml.import_gdml", importer_version=IMPORTER_VERSION,
                source=dict(file=Path(path).name, sha256=f"sha256:{sha}"))  # fmt: skip
    if doc.repairs:
        prov["repairs"] = sorted(doc.repairs)
    world = str(doc.world)
    return schema.DetectorFile(name=Path(path).stem, world=world, detector_volume=world,
                          wavelength_range=(0.0, 0.0), solids=sol, volumes=vols, placements=pls, materials=mats,
                          elements=els, isotopes=isos, surfaces=surfs, skins=skins, borders=borders, meshes=meshes,
                          userinfo=tuple(_aux(a) for a in doc.userinfo), provenance=prov,
                          report=tuple(rep), tokens=toks)  # fmt: skip


def import_gdml(path, recipe, inputs: dict | None = None) -> ImportResult:
    """Return the detector file, sensor sets and calibrations of a GDML file under a recipe (``lumen.Recipe``).

    ``inputs`` gives the extra files the recipe's decisions read, by name.
    """
    df = read_gdml(path, getattr(recipe, "material_keys", None))
    return recipe.apply(df, Path(path), inputs)
