"""The GDML importer: a GDML file and a recipe to a detector file (``lumen.format``), the full world kept.

    result = import_gdml("tank.gdml", recipe)    # ImportResult(detector_file, sensor_set): lumen.convert saves it

A plain function: pure and deterministic (the same bytes and recipe give a byte-identical detector file). It reads
Geant4 semantics only and makes no physics decision; the recipe's decisions are applied after, each recorded with
``origin="recipe"``. Units are converted by exact decimal scaling of plain literals (``lumen.gdml.units``).

The reader refuses what Geant4 refuses (``lumen.gdml.reader``); a file that needs a repair is repaired first, apart
(``python -m lumen.gdml.repair``), and the repaired copy is the source the provenance names with its sha256.

A property is read in the form Geant4 stores it (G4GDMLReadMaterials / Solids::PropertyRead, on materials and
surfaces alike): a coldim 1 matrix a constant, a wider one a table. A Geant4 key given in the other form -- a table
key (``lumen.rules.keys.TABLE_KEYS``: RINDEX, REFLECTIVITY, ...) as a constant, or a constant key as a table -- is one
no Geant4 process reads (GetProperty / GetConstProperty find nothing under it): it is left out, with a report line.

Fails closed (``ImportRefused``) where Geant4's reading is ambiguous or lumen's would differ: an opticalsurface
without all four of model, finish, type and value, a border without exactly two physvolrefs, and every structural rule
of ``lumen.format.check`` (the rules a YAML detector meets too), among them:

- a repeated border (the same ordered pair of placements twice);
- a border physvolref whose FULL name is carried by more than one physvol (Geant4 binds only the first; names that
  repeat only once the ``0x`` suffix is stripped are distinct to both Geant4 and lumen and are accepted);
- two skins on one logical volume;
- a material without a composition (density and components, or Z and molar mass);
- a table whose energies are not strictly ascending.

Names are stored without Geant4's pointer suffix (``Water0x557c198f8b80`` as ``Water``: G4GDMLRead::StripName, the
first ``0x`` on) when no other name of its kind strips to the same text, so parameter keys and sensor paths stay the
same across exports of one detector; any other name is stored as written, so two records never share a name.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np

from lumen.format import schema
from lumen.format.check import check
from lumen.format.io import sha256_file
from lumen.format.result import ImportResult
from lumen.gdml import reader
from lumen.gdml.units import LITERAL, PRODUCT
from lumen.rules.keys import CONST_KEYS, FINISHES, MODELS, TABLE_KEYS, TYPES
from lumen.units import ENERGY_EV, EV_PER_MEV, QUANTITIES


class InertKeysWarning(UserWarning):
    """Material keys no Geant4 optical process reads, kept inert (unread, stored verbatim)."""


class ImportRefused(ValueError):  # noqa: N818 (the name of the documented error catalogue)
    """The GDML has a construct whose Geant4 reading is ambiguous or that lumen reads differently: refused."""


# ------------------------------------------------------------------------------------------------------ geometry
def _transform(t: reader.Transform) -> schema.Transform:
    return schema.Transform(position=tuple(float(x) for x in t.position), rotation=tuple(float(x) for x in t.rotation))


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
        out[name] = schema.Solid(name, s.type, params=dict(s.params), **kw)
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
                    copy=pv.copy,
                    position=tuple(float(x) for x in t.position),
                    rotation=tuple(float(x) for x in t.rotation),
                    scale=None if t.scale is None else tuple(float(x) for x in t.scale),
                )
            )
        volumes[name] = schema.Volume(name, v.solid, v.material, daughters=tuple(idx), assembly=v.assembly)
    return volumes, tuple(placements)


# ------------------------------------------------------------------------------------------------------ properties
def _energy_ev(tok: str, internal: float) -> float:
    m = PRODUCT.match(tok)
    if LITERAL.match(tok):
        return float(Decimal(tok) * Decimal(ENERGY_EV["MeV"]))  # Geant4's internal unit, exactly
    if m and m.group(2) in ENERGY_EV:
        return float(Decimal(m.group(1)) * Decimal(ENERGY_EV[m.group(2)]))
    return internal * EV_PER_MEV


def _power(key: str) -> int:
    """Return the power of ten from Geant4's internal unit to the unit a property key is stored in (0: none)."""
    return schema.PROPERTY_UNITS[key][2] if key in schema.PROPERTY_UNITS else 0


def _stored(key: str, tok: str | None, internal: float) -> float:
    """Return a matrix value in the unit ``key`` is stored in (``schema.PROPERTY_UNITS``).

    A plain literal is in Geant4's internal unit and shifted by exact decimal arithmetic; a product with a unit of the
    key's quantity that is a decimal multiple of the stored unit ("1.5*cm") is exact; anything else (``internal``,
    the reader's value) is converted by one operation.
    """
    p = _power(key)
    if tok is not None:
        quantity = schema.PROPERTY_UNITS[key][0]
        m, table = PRODUCT.match(tok), QUANTITIES.get(quantity, {})
        if LITERAL.match(tok):
            return float(Decimal(tok.strip()).scaleb(p))
        if m and table.get(m.group(2)):
            return float(Decimal(m.group(1)) * Decimal(table[m.group(2)]))
    return internal / 10.0**-p if p < 0 else internal * 10.0**p


def read_property(doc: reader.GDMLDocument, key: str, ref: str, what: str) -> schema.Property:
    """One property as Geant4 reads it: energies in eV, values in the unit ``schema.PROPERTY_UNITS`` gives the key.

    The GDML holds Geant4's internal units, converted here (lengths mm to m, GROUPVEL mm / ns to m / ns, ...).

    Notes
    -----
    G4GDMLReadDefine::MatrixRead and G4GDMLReadMaterials / Solids::PropertyRead: coldim 1 is a constant (its first
    value), coldim 2 an (energy, value) table; the matrix ``unit`` attribute is not read.
    """
    m = doc.matrix(ref)
    toks = doc.matrix_text[ref]
    scaled = _power(key) != 0
    if m.shape[1] == 1:
        c = float(m[0, 0])
        return schema.Property(key, np.asarray(_stored(key, toks[0], c) if scaled else c, np.float64), energy=None)
    if m.shape[1] != 2:
        raise ImportRefused(f"{what}.{key}: matrix {ref!r} has {m.shape[1]} columns (a property needs 1 or 2)")
    e = np.array([_energy_ev(toks[2 * i], m[i, 0]) for i in range(m.shape[0])], np.float64)
    if scaled:
        v = np.array([_stored(key, toks[2 * i + 1], m[i, 1]) for i in range(m.shape[0])], np.float64)
    else:
        v = m[:, 1].astype(np.float64)
    return schema.Property(key, v, energy=e)


def _unread_form(key: str, coldim: int) -> str | None:
    """Return why Geant4 reads no property of a Geant4 key in this form (None: it reads it); see the docstring."""
    if coldim == 1 and key in TABLE_KEYS:
        return (
            "a table key given as a constant (coldim 1): Geant4 stores it as a constant property, which no "
            "process reads"
        )
    if coldim > 1 and key in CONST_KEYS:
        return "a constant key given as a table: Geant4 stores it as a table property, which no process reads"
    return None


@dataclass
class _Inert:
    """The optical keys to read (``keep``) and the other properties' matrices, kept verbatim (``found``)."""

    keep: frozenset
    found: dict = field(default_factory=dict)


def _properties(doc, props, what, rep, skip=(), inert: _Inert | None = None):
    """Read properties; with ``inert``, the keys it does not keep go verbatim into ``inert.found``, unread."""
    out: dict = {}
    seen: set = set()
    for name, ref in props:
        key = reader.g4_strip(name)  # a property name as Geant4 reads it
        if key in seen:
            raise ImportRefused(f"{what}: property {key} given twice")
        seen.add(key)
        if key in skip:
            rep.append(f"{what}.{key}: left out: Geant4 reads it on no surface (matrix {ref!r})")
            continue
        if inert is not None and key not in inert.keep:
            inert.found[key] = doc.matrix(ref)
            continue
        why = _unread_form(key, doc.matrix(ref).shape[1])
        if why:
            rep.append(f"{what}.{key}: left out: {why} (matrix {ref!r})")
            continue
        out[key] = read_property(doc, key, ref, what)
    return out


# ------------------------------------------------------------------------------------------------------- materials
def _first_by_stripped(names) -> dict:
    """Return {name without its pointer suffix: the first record of it}: Geant4 names its records stripped
    (G4GDMLReadMaterials: ``new G4Element(Strip(name), ...)``) and a lookup returns the first of a name."""  # noqa: D205, D209
    out: dict = {}
    for n in names:
        out.setdefault(reader.g4_strip(n), n)
    return out


def materials(doc: reader.GDMLDocument, rep: list, keys=None) -> tuple[dict, dict, dict]:
    """Return (materials, elements, isotopes); every material must name its composition.

    ``keys``: the optical property keys to read (None: ``lumen.rules.optics.MATERIAL_KEYS``, every key a Geant4 optical
    process reads); a material's other properties are kept inert, unread.
    """
    isotopes = {
        n: schema.Isotope(n, z=i.z, n=i.n, molar_mass_g_mol=i.molar_mass_g_mol) for n, i in doc.isotopes.items()
    }
    first = {kind: _first_by_stripped(names) for kind, names in (("isotope", doc.isotopes), ("element", doc.elements),
                                                                  ("material", doc.materials))}  # fmt: skip

    def component(ref, kinds, **amount):
        """Return a component as Geant4 resolves it: the first record of the first kind of its stripped name."""
        s = reader.g4_strip(ref)
        kind = next((k for k in kinds if s in first[k]), kinds[0])
        return schema.Component(first[kind].get(s, ref), kind=kind, **amount)

    elements = {n: schema.Element(n, formula=e.formula, z=e.z, molar_mass_g_mol=e.molar_mass_g_mol,
                             isotopes=tuple(component(r, ("isotope",), fraction=f) for r, f in e.fractions))
                for n, e in doc.elements.items()}  # fmt: skip
    out = {}
    for name, m in doc.materials.items():
        # G4GDMLReadMaterials::MixtureRead: a fraction names an element if one has that name, else a material; a
        # composite an element; both looked up by the name without its pointer suffix (GenerateName(ref, true))
        comps = tuple(component(r, ("element", "material"), fraction=f) for r, f in m.fractions) + \
            tuple(component(r, ("element",), atoms=n) for r, n in m.composites)  # fmt: skip
        comp = schema.Composition(m.density_g_cm3, components=comps, z=m.z, molar_mass_g_mol=m.molar_mass_g_mol)
        from lumen.rules import optics  # (here: it loads the media and surfaces, which `import lumen` does not)

        sink = _Inert(frozenset(optics.MATERIAL_KEYS if keys is None else keys))
        props = _properties(doc, m.props, f"material {name!r}", rep, inert=sink)
        inert = sink.found
        if inert:
            rep.append(f"material {name!r}: {len(inert)} non-optical propert(ies) kept inert: {sorted(inert)}")
        out[name] = schema.Material(
            name,
            state=m.state,
            composition=comp,
            temperature_k=m.temperature,
            pressure_pa=m.pressure,
            properties=props,
            inert=inert,
            mee_ev=m.mee_ev,
        )
    return out, elements, isotopes


# -------------------------------------------------------------------------------------------------------- surfaces
# GROUPVEL on a surface: Geant4's own table (G4MaterialPropertiesTable::AddProperty computes it with RINDEX and
# G4GDMLWriteMaterials writes it), read on no surface (G4Track and G4OpBoundaryProcess read a material's): not kept
SURFACE_DROPPED = ("GROUPVEL",)


ENUMS = (("model", MODELS), ("finish", FINISHES), ("type", TYPES))


def _enum_name(surface: str, attr: str, names, v: str) -> str:
    """Return an opticalsurface attribute by name: a number (a GDML export's enum value; Geant4 reads both alike) is
    the name at that position of Geant4's enum (``lumen.rules.keys``), refused beyond it."""  # noqa: D205, D209
    t = str(v).strip()
    if not t.isdigit():
        return t
    if int(t) >= len(names):
        raise ImportRefused(f"opticalsurface {surface!r}: {attr} {t} is no value of Geant4's enum (0 to "
                            f"{len(names) - 1}: {', '.join(names)})")  # fmt: skip
    return names[int(t)]


def surfaces(doc: reader.GDMLDocument, placements: tuple, rep: list) -> tuple[dict, tuple, tuple]:
    """Return (surfaces, skins, borders), refusing the ambiguous cases of the module docstring."""
    out = {}
    for name, s in doc.surfaces.items():
        missing = [a for a in ("model", "finish", "type", "value") if getattr(s, a) is None]
        if missing:
            raise ImportRefused(f"opticalsurface {name!r} has no {', '.join(repr(a) for a in missing)} attribute. "
                                "Geant4 would read a default and lumen cannot guess: add it in the GDML and convert "
                                "again.")  # fmt: skip
        props = _properties(doc, s.props, f"opticalsurface {name!r}", rep, skip=SURFACE_DROPPED)
        model, finish, type_ = (_enum_name(name, a, names, getattr(s, a)) for a, names in ENUMS)
        out[name] = schema.Surface(
            name,
            type=type_,
            model=model,
            finish=finish,
            value=float(doc.num(s.value, f"opticalsurface {name!r}")),
            properties=props,
        )
    skins: list = []
    for name, surf, lvs in doc.skins:
        skins += [schema.Skin(name, surface=surf, volume=lv) for lv in lvs]
    borders = []
    for name, surf, refs in doc.borders:
        if len(refs) != 2:
            raise ImportRefused(f"bordersurface {name!r}: needs exactly two physvolref")
        borders.append(schema.Border(name, surface=surf, pv1=refs[0], pv2=refs[1]))
    by_stripped = Counter(reader.g4_strip(p.name) for p in placements)
    stripped = sum(1 for b in borders for r in (b.pv1, b.pv2) if by_stripped[reader.g4_strip(r)] > 1)
    if stripped:
        rep.append(f"{stripped} border reference(s) name a physical volume whose name repeats once the 0x suffix is "
                   "stripped: bound by the full name, as Geant4 binds them; a GDML export with stripped names would "
                   "be ambiguous and is refused")  # fmt: skip
    return out, tuple(skins), tuple(borders)


# ----------------------------------------------------------------------------------------------------------- names
def _stripped(names) -> dict:
    """Return {written: stored} of the names stored without their pointer suffix (see the module docstring)."""
    names = list(dict.fromkeys(names))
    count = Counter(reader.g4_strip(n) for n in names)
    return {n: s for n in names if (s := reader.g4_strip(n)) != n and s and count[s] == 1}


def canonical_names(df: schema.DetectorFile) -> schema.DetectorFile:
    """Return the detector file with each kind's names stored without the pointer suffix where unique (module doc)."""
    so, vo = _stripped(df.solids), _stripped(df.volumes)
    pv = _stripped(p.name for p in df.placements)
    ma, el, iso = _stripped(df.materials), _stripped(df.elements), _stripped(df.isotopes)
    su, sk, bo = _stripped(df.surfaces), _stripped(k.name for k in df.skins), _stripped(b.name for b in df.borders)
    if not any((so, vo, pv, ma, el, iso, su, sk, bo)):
        return df
    by_kind = {"material": ma, "element": el, "isotope": iso}

    def n(m, x):
        return m.get(x, x) if x is not None else None

    def comps(cs):
        return tuple(dataclasses.replace(c, ref=n(by_kind.get(c.kind, {}), c.ref)) for c in cs)

    out = dataclasses.replace(
        df,
        world=n(vo, df.world),
        detector_volume=n(vo, df.detector_volume),
        solids={n(so, k): dataclasses.replace(s, name=n(so, s.name), refs=tuple(n(so, r) for r in s.refs),
                                              mesh=n(so, s.mesh)) for k, s in df.solids.items()},
        volumes={n(vo, k): dataclasses.replace(v, name=n(vo, v.name), solid=n(so, v.solid), material=n(ma, v.material))
                 for k, v in df.volumes.items()},
        placements=tuple(dataclasses.replace(p, name=n(pv, p.name), volume=n(vo, p.volume), mother=n(vo, p.mother))
                         for p in df.placements),
        materials={n(ma, k): dataclasses.replace(m, name=n(ma, m.name), composition=dataclasses.replace(
            m.composition, components=comps(m.composition.components))) for k, m in df.materials.items()},
        elements={n(el, k): dataclasses.replace(e, name=n(el, e.name), isotopes=comps(e.isotopes))
                  for k, e in df.elements.items()},
        isotopes={n(iso, k): dataclasses.replace(i, name=n(iso, i.name)) for k, i in df.isotopes.items()},
        surfaces={n(su, k): dataclasses.replace(s, name=n(su, s.name)) for k, s in df.surfaces.items()},
        skins=tuple(dataclasses.replace(k, name=n(sk, k.name), surface=n(su, k.surface), volume=n(vo, k.volume))
                    for k in df.skins),
        borders=tuple(dataclasses.replace(b, name=n(bo, b.name), surface=n(su, b.surface), pv1=n(pv, b.pv1),
                                          pv2=n(pv, b.pv2)) for b in df.borders),
        meshes={n(so, k): a for k, a in df.meshes.items()},
    )  # fmt: skip
    renamed = sum(map(len, (so, vo, pv, ma, el, iso, su, sk, bo)))
    return dataclasses.replace(out, report=(*out.report, f"names: {renamed} stored without Geant4's pointer suffix"))


# ----------------------------------------------------------------------------------------------------- the import
def read_gdml(path, material_keys=None, warn=True) -> schema.DetectorFile:
    """Return the detector file of a GDML file before any recipe decision (the detector volume is the world).

    ``material_keys``: the material property keys to read as optical properties (None: the Geant4 optical keys,
    ``lumen.rules.optics.MATERIAL_KEYS``); see ``materials``. With ``warn``, the keys kept inert are named in one
    ``InertKeysWarning`` (keys x materials).
    """
    doc = reader.read(path)
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
    prov = dict(importer="lumen.convert", source=dict(file=Path(path).name, sha256=sha256_file(path)))
    world = str(doc.world)
    df = schema.DetectorFile(name=Path(path).stem, world=world, detector_volume=world, solids=sol, volumes=vols,
                             placements=pls, materials=mats, elements=els, isotopes=isos, surfaces=surfs, skins=skins,
                             borders=borders, meshes=meshes, provenance=prov, report=tuple(rep))  # fmt: skip
    check(df, ImportRefused, decisions=False)  # the structural rules every input meets (lumen.format.check)
    return canonical_names(df)


def import_gdml(path, recipe, inputs: dict | None = None) -> ImportResult:
    """Return the detector file, sensor set and recipe data of a GDML file under a recipe (``lumen.Recipe``).

    ``inputs`` gives the extra files the recipe's decisions read, by name.
    """
    df = read_gdml(path, recipe.material_keys)
    return recipe.apply(df, inputs)
