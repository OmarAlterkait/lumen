"""The viewer's scene: a GDML read as lumen reads it, with lumen's reading of every material, surface and volume.

    view = build("tank.gdml", recipe="my_recipes.py:TANK")   # or build("lumen_wc:sk"): a package's detector
    view.payload                                               # the JSON the page draws (``lumen-view/3``)

The detector is converted and loaded (``lumen.convert``, ``lumen.load``, into a temporary directory), and the page
shows what was built: the media and surfaces of its library, the surface of each crossing (``Optics.pair``), its
sensors and their groups. Without a recipe it is converted with the ``auto:`` detector volume of its transparent
materials and no sensors. What lumen refuses is reported in ``errors``, and the geometry is still drawn, with the
materials lumen can read. Lengths m, energies eV, wavelengths nm.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import importlib.util
import sys
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from lumen import api
from lumen.build import build as detector_build
from lumen.build import library, lookup, lowering
from lumen.build.tree import OUTSIDE_PV
from lumen.format import io as detector_io
from lumen.format import schema, sensorsets
from lumen.format.solids import ATTRIBUTES
from lumen.gdml import importer, reader
from lumen.geometry import scene as sc
from lumen.recipe import Recipe
from lumen.rules import optics
from lumen.rules.keys import SCINTILLATION_KEYS
from lumen.sources.scintillation import ScintillationPoint
from lumen.surfaces import types
from lumen.units import HC, MM_PER_M
from lumen.viewer import mesh

SCHEMA = "lumen-view/3"
MAX_INSTANCES = 3_000_000  # placed copies drawn at most; deeper copies past it are left out, and reported
SPECTRA = ("WLSCOMPONENT", *SCINTILLATION_KEYS["spectra"])  # emission spectra: no light outside their knots
BORDERS_SHOWN = 200  # a surface's borders listed at most; their number is always given
AXES = ("x", "y", "z")


@dataclasses.dataclass(frozen=True)
class View:
    """A built scene: ``payload`` (JSON-ready) and the steps that failed (``errors``: [(stage, message)])."""

    payload: dict
    errors: list


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode("ascii")


def short(name):
    """Return a GDML name as Geant4 shows it (the '0x...' pointer suffix dropped)."""
    return reader.g4_strip(name) if name else name


def load_recipe(spec):
    """Return the recipe ``NAME`` of a Python file given as ``path/to/file.py:NAME`` (a recipe of your own)."""
    path, _, attr = str(spec).rpartition(":")
    if not path or not attr:
        raise ValueError(f"recipe {spec!r}: give FILE.py:NAME")
    mod_spec = importlib.util.spec_from_file_location(f"lumen_viewer_recipe_{Path(path).stem}", path)
    if mod_spec is None or mod_spec.loader is None:
        raise ValueError(f"recipe {spec!r}: {path} is not a Python file")
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_spec.name] = module  # dataclasses and pickling look their module up there
    mod_spec.loader.exec_module(module)
    return getattr(module, attr)


@contextlib.contextmanager
def resolve(source, recipe=None, data=None):
    """Yield (GDML path, recipe or None, recipe inputs, name) of a GDML file or of a package's ``"package:name"``.

    ``recipe``: a ``lumen.Recipe`` or ``FILE.py:NAME`` (a GDML file read with a recipe of your own); a package's
    detector comes with its own (``lumen.DetectorEntry``), its relative paths under ``data`` (else ``LUMEN_DATA``) and
    the package's ``prepare`` applied as ``lumen.convert`` applies it.
    """
    if api._is_package(source):
        prepare = api._package_prepare(source)
        gdml, rec, inputs = api._package_detector(source, None, None, data)
        with prepare(rec, gdml, inputs) as (rec, gdml, inputs):
            yield Path(gdml), rec, dict(inputs or {}), str(source)
        return
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    p = Path(str(source))
    yield p, rec, {}, rec.name if rec is not None else p.stem


def _caught(fn, stage, errors, notes):
    """Run ``fn`` recording its warnings in ``notes``; return its result, or None with the error in ``errors``."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
        except Exception as err:  # every failure is shown, the rest of the scene still built
            errors.append((stage, f"{type(err).__name__}: {err}"))
            out = None
    for x in w:  # a warning two steps give is shown once
        note = dict(stage=stage, kind=type(x.message).__name__, text=str(x.message))
        if all(note["text"] != n["text"] for n in notes):
            notes.append(note)
    return out


def _fallback(path, errors, notes):
    """Return the geometry of a GDML the importer refuses: solids and structure only, materials without optics."""
    doc = _caught(lambda: reader.read(path), "read", errors, notes)
    if doc is None:
        return None
    sol, meshes, rep = importer.solids(doc)
    vols, pls = importer.structure(doc)
    mats = {}
    for n, m in doc.materials.items():
        comp = schema.Composition(m.density_g_cm3 or 0.0, z=m.z, molar_mass_g_mol=m.molar_mass_g_mol)
        mats[n] = schema.Material(n, state=m.state, composition=comp, temperature_k=m.temperature,
                                  pressure_pa=m.pressure)  # fmt: skip
    world = str(doc.world)
    return schema.DetectorFile(name=Path(path).stem, world=world, detector_volume=world, solids=sol, volumes=vols,
                               placements=pls, materials=mats, meshes=meshes,
                               report=tuple(doc.report) + tuple(rep))  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Reading:
    """A detector as lumen reads it: its file, its sensor set, the built parts.

    ``parts``: None when lumen refused it; ``optical``: the file holds the GDML's optics (False: the importer refused
    the GDML and only its geometry was read).
    """

    file: schema.DetectorFile
    sensor_set: sensorsets.SensorSet
    parts: detector_build.Parts | None = None
    optical: bool = True


def read(path, recipe, inputs, errors, notes):
    """Return the ``Reading`` of a GDML under a recipe (None: no recipe); what failed is in ``errors``."""
    df = _caught(lambda: importer.read_gdml(path, getattr(recipe, "material_keys", None)), "import", errors, notes)
    if df is None:
        df = _fallback(path, errors, notes)
        return None if df is None else Reading(df, sensorsets.SensorSet(()), optical=False)
    rec = recipe or guess(df)
    if rec is None:
        errors.append(("detector volume", "no material is transparent: name the detector volume in a recipe"))
        return Reading(df, sensorsets.SensorSet(()))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "detector.lumen"
        if _caught(lambda: api.convert(path, rec, out, inputs=inputs), "conversion", errors, notes) is None:
            spec = rec.detector_volume
            found = _caught(lambda: lowering.find_detector(df, spec), "detector volume", [], [])
            lv = spec if found is None or not spec.startswith("auto:") else found.lv
            return Reading(dataclasses.replace(df, detector_volume=lv) if found else df,
                           sensorsets.SensorSet(getattr(rec, "groups", ())))  # fmt: skip
        det = _caught(lambda: api.load(out), "load", errors, notes)
        if det is None:
            return Reading(detector_io.read(out), sensorsets.read(out))
        return Reading(det.file, det.sensor_set, det.parts)


def guess(df):
    """Return a recipe of the ``auto:`` detector volume over every transparent material, without sensors; None when no
    material is transparent."""  # noqa: D205, D209
    media = [m for m, rec in df.materials.items() if _status(rec)[0] == "medium"]
    return Recipe(df.name, "auto:" + ",".join(media), groups=()) if media else None


def _status(rec):
    """Return (status, message, medium entry, sources, warnings) of a material as its optics model reads it."""
    errors, notes = [], []
    got = _caught(lambda: library.material_medium(rec), "medium", errors, notes)
    if got is None:
        return "refused", errors[0][1], None, {}, []
    _, entry, sources = got
    return "opaque" if entry.medium.opaque else "medium", "", entry, sources, [n["text"] for n in notes]


def _tree_instances(parts):
    """Return ({(mother instance, placement): instance}, the PV name of each) of the built geometry's instances."""
    T = sc.instances(parts.scene)
    child = {(int(p), int(k)): i for i, (p, k) in enumerate(zip(T.parent, T.source, strict=True))}
    names = [T.name(parts.scene, i) if parts.scene.definitions[d].member is not None else None
             for i, d in enumerate(T.defn)]  # fmt: skip
    return child, names


def instances(df, chain, parts, max_instances=MAX_INSTANCES):
    """Return every placed copy from the world, depth-first: lv index, parent, placement, depth, in-detector, matrix.

    ``chain``: the placements from the world to the detector volume (() the world, None: unknown). ``pv``: per copy
    the built detector's placed volume (None: not simulated, or an assembly). The matrix is (n, 12): the 3x3 rotation
    rows then the position (m), mapping the volume's frame to the world's.
    """
    lvs = list(df.volumes)
    index = {lv: i for i, lv in enumerate(lvs)}
    child, names = _tree_instances(parts) if parts is not None else ({}, [])
    root = child.get((0, -1), 0) if names else -1  # the detector volume's instance: below the lowering's root wrapper
    rows: dict = dict(lv=[], parent=[], placement=[], depth=[], inside=[], matrix=[])
    pvs = []
    stack = [(df.world, np.eye(3), np.zeros(3), (-1, -1), (), root if chain == () else -1)]
    poses: dict = {}
    cut = 0
    while stack:
        lv, lin, t, (par, p), path, ti = stack.pop()
        if len(pvs) >= max_instances:
            cut += 1
            continue
        g = len(pvs)
        inside = chain is not None and path[: len(chain)] == chain
        for k, v in zip(rows, (index[lv], par, p, len(path), inside, np.concatenate([lin.reshape(-1), t])),
                        strict=True):  # fmt: skip
            rows[k].append(v)
        pvs.append(names[ti] if ti >= 0 else None)
        for k in reversed(df.volumes[lv].daughters):
            if k not in poses:
                poses[k] = lowering.placement_pose(df.placements[k])
            lc, tc = poses[k]
            below = (*path, k)
            kid = child.get((ti, k), -1) if ti >= 0 else (root if below == chain else -1)
            stack.append((df.placements[k].volume, lin @ lc, lin @ tc + t, (g, k), below, kid))
    out = {k: np.asarray(rows[k], np.int32) for k in ("lv", "parent", "placement", "depth")}
    return dict(out, inside=np.asarray(rows["inside"], bool), matrix=np.asarray(rows["matrix"], float).reshape(-1, 12),
                pv=pvs, cut=cut, lvs=lvs)  # fmt: skip


def group_members(reading):
    """Return {group: its sensors' placed volumes} of a built detector, as lumen groups its sensors ({}: not built)."""
    if reading.parts is None:
        return {}
    pvs = reading.parts.optics.sensor_pvs
    return {g.name: [pvs[t] for t in detector_build.group_tags(g, reading.parts, reading.sensor_set)]
            for g in reading.sensor_set.groups}  # fmt: skip


def _unit(key):
    return schema.PROPERTY_UNITS.get(key, ("", "", 0))[1]


def _property(p: schema.Property, reads, lam):
    """Return a property for the page: its table (energies eV, wavelengths nm) or constant, and what reads it."""
    out = dict(key=p.key, origin=p.origin, note=p.note, unit=_unit(p.key), reads=reads)
    if p.energy is None:
        out.update(kind="constant", value=float(np.asarray(p.value)))
        return out
    e, v = np.asarray(p.energy, float), np.asarray(p.value, float)
    out.update(kind="table", energy=e.tolist(), value=v.tolist(), n=len(e), min=float(v.min()), max=float(v.max()),
               lam=[HC / float(e[-1]), HC / float(e[0])], spectrum=p.key in SPECTRA)  # fmt: skip
    if lam and lam[1] > lam[0] and p.key not in SPECTRA:
        lo, hi = np.nextafter(HC / lam[1], np.inf), np.nextafter(HC / lam[0], -np.inf)  # one ulp: an end on the knot
        out["covers"] = bool(e[0] <= lo and e[-1] >= hi)
    return out


def _source_reads(sources):
    """Return {property key: [read, ...]} of a library entry's sources (the lumen parameter each key feeds)."""
    reads: dict = {}
    for path, src in (sources or {}).items():
        if src.key:
            prop = src.key.rsplit("/", 1)[-1]
            reads.setdefault(prop, []).append(dict(param="/".join(path), column=src.column, note=src.note,
                                                   key=src.key))  # fmt: skip
    return reads


def _channels(entry):
    """Return the medium's processes ("refraction", "absorption", then its channels) and its emitters' profiles."""
    med = entry.medium
    procs = (["refraction"] if med.index is not None else []) + (["absorption"] if med.absorption is not None else [])
    emission = {c.name: "+".join(c.emission.profile.components) for c in med.channels if c.emission is not None}
    return procs + [c.name for c in med.channels], emission


def _scintillation(rec):
    """Return the properties a scintillation source of the material reads (none: lumen has no such source for it)."""
    at = f"materials/{rec.name}/"
    try:
        source = ScintillationPoint.of({at + k: p.value for k, p in rec.properties.items()}, rec.name)
    except KeyError:
        return []
    return [key[len(at) :] for key in source.reads()]


def material_info(rec, state, lam):
    """Return a material for the page. ``state``: "" the detector volume holds it: as its optics model reads it;
    else why nothing is read ("unused", "unread")."""  # noqa: D205, D209
    status, message, entry, sources, warned = (state, "", None, {}, []) if state else _status(rec)
    reads, channels, emission, derived = {}, [], {}, []
    scint = None
    if status == "medium":
        reads = _source_reads({k: v for k, v in sources.items() if k[0] in entry.params})
        channels, emission = _channels(entry)
        for k in _scintillation(rec):
            reads.setdefault(k, []).append(dict(param="source", column="scintillation", note="", key=""))
            scint = dict(yield_per_MeV=float(np.asarray(rec.properties["SCINTILLATIONYIELD"].value)))
        ray = optics.geant4_rayleigh(rec) if "rayleigh" in channels and "RAYLEIGH" not in rec.properties else None
        if ray is not None:
            derived = [_property(ray, [dict(param="rayleigh/length", column="derived", note=ray.note, key="")], lam)]
    o, comp = rec.optics, rec.composition
    return dict(
        name=rec.name, short=short(rec.name), state=rec.state, density=comp.density_g_cm3,
        temperature=rec.temperature_k, pressure=rec.pressure_pa,
        composition=[dict(ref=short(c.ref), fraction=c.fraction, atoms=c.atoms) for c in comp.components],
        optics=dict(model=o.name, origin=o.origin, note=o.note, settings={k: str(v) for k, v in o.settings.items()}),
        status=status, message=message, warnings=warned, channels=channels, emission=emission,
        properties=[_property(p, reads.get(k, []), lam) for k, p in sorted(rec.properties.items())],
        derived=derived, inert=sorted(rec.inert), scintillation=scint,
    )  # fmt: skip


def _built_uses(parts, name):
    """Return [(how, type name, {param path: Source})] of a surface's entries in the built library."""
    lib, out = parts.library, []
    if name in lib.surfaces:
        out.append(("surface", "", lib.surfaces[name].surface_type, ("surfaces", name)))
    for k, e in lib.sensors.items():
        if e.surface == name:
            out.append(("sensor", f"group {e.group}, " + ("leaving it" if e.exit else "entering it"), e.surface_type,
                        ("sensors", k)))  # fmt: skip
    return [(how, text, stype.name, {k[2:]: v for k, v in parts.key_rows.items() if k[:2] == at})
            for how, text, stype, at in out]  # fmt: skip


def _fixed_notes(sources):
    """Return {surface property key: why lumen does not read it} of the parameters fixed by the surface's reading."""
    from lumen.rules.keys import SURFACE_KEYS

    return {SURFACE_KEYS[path[0]]: f"not read: {src.note}" for path, src in sources.items()
            if src.column == "fixed" and path and path[0] in SURFACE_KEYS and src.note}  # fmt: skip


def surface_info(reading, rec, lam, lv_of, held):
    """Return a surface for the page: its Geant4 form, and each way the built library reads it (none: "unused").

    Of a detector lumen refused: "refused" with the message when no reading of a surface the detector volume holds
    (``held``: its LVs) accepts it, else "unread". ``lv_of``: a placement's LV by its name.
    """
    df, parts = reading.file, reading.parts
    skins = [k.volume for k in df.skins if k.surface == rec.name]
    borders = [(b.pv1, b.pv2) for b in df.borders if b.surface == rec.name]
    lvs = {*skins, *(lv_of.get(x) for b in borders for x in b)}
    uses, message = [], ""
    if parts is not None:
        uses = _built_uses(parts, rec.name)
        status = ("sensor" if any(u[0] == "sensor" for u in uses) else "ok") if uses else "unused"
    else:
        errors: list = []
        sensor = any(g.selects(lv) for g in reading.sensor_set.groups for lv in lvs if lv)
        if reading.optical and lvs & held and not sensor:  # a sensor's surface is read by its group, which is not built
            _caught(lambda: library.surface_entry(rec), "surface", errors, [])
        status, message = ("refused", errors[0][1]) if errors else ("unread", "")
    reads: dict = {}
    why: dict = {}
    for _, _, _, sources in uses:
        for k, v in _source_reads(sources).items():
            reads[k] = reads.get(k, []) + v
        why.update(_fixed_notes(sources))
    geant4 = dict(type=rec.type, model=rec.model, finish=rec.finish, value=rec.value, value_name=optics.value_key(rec))
    return dict(
        name=rec.name, short=short(rec.name), geant4=geant4, status=status, message=message,
        optics=dict(model=rec.optics.name, origin=rec.optics.origin, note=rec.optics.note),
        uses=[dict(how=how, text=text, type=stype,
                   params=[dict(param="/".join(p), key=s.key, column=s.column, note=s.note) for p, s in src.items()])
              for how, text, stype, src in uses],
        properties=[dict(_property(p, reads.get(k, []), lam), why=why.get(k, "")) for k, p in
                    sorted(rec.properties.items())],
        skins=skins, borders=borders[:BORDERS_SHOWN], n_borders=len(borders),
    )  # fmt: skip


def _crossing(parts, meshes, pre, post):
    """Return one crossing pre -> post of the built optics for the page (``Optics.pair``)."""
    if pre in meshes:
        return dict(outcome="none", surface=None, text="lumen follows no light inside a tessellated solid")
    o = parts.optics
    if o.media[o.medium_of[o.class_of[pre]]].opaque:
        return dict(outcome="none", surface=None, text="no photon travels on this side (opaque)")
    pair = o.pair(pre, post)
    if pair is None or pair.surface_type is types.PASS:
        return dict(outcome="pass", surface=None, text="no boundary: the photon goes on")
    what, _, name = (pair.origin or "").partition(":")
    surface = parts.library.sensors[name].surface if what == "sensor" else (name or None)
    outcome = {types.ABSORBER: "absorb", types.DIELECTRIC: "fresnel"}.get(pair.surface_type, "surface")
    if pair.surface_type.detects:
        outcome = "sensor"
    text = f"{pair.surface_type.name}" + (f", sensor {name}" if what == "sensor" else "")
    return dict(outcome=outcome, surface=surface, text=f"{text} ({'; '.join(pair.reasons)})")


def crossings(reading, inst):
    """Return {lv: (placed volume, [crossing, ...])}: the built optics' surfaces at each volume's first simulated copy.

    With its mother, in then out, then with each sibling a border surface names. A sibling it touches without a border
    is not listed, nor is a coincident face.
    """
    parts = reading.parts
    if parts is None:
        return {}
    tree, meshes = parts.tree, {x.pv for x in parts.instances if x.mesh}
    partners: dict = {}
    for b in reading.file.borders:
        partners.setdefault(b.pv1, set()).add(b.pv2)
        partners.setdefault(b.pv2, set()).add(b.pv1)
    out: dict = {}
    for li, pv in zip(inst["lv"], inst["pv"], strict=True):
        lv = inst["lvs"][li]
        if pv is None or lv in out:
            continue
        around = [tree.by_name[pv].mother]
        for other in sorted(partners.get(tree.physvol(pv), ())):
            sib = [x for x in tree.instances_of.get(other, ()) if lookup.relation(tree, pv, x) == "sibling"]
            around += sib[:1]
        rows = []
        for x in around:
            name = "outside the detector volume" if x == OUTSIDE_PV else short(tree.by_name[x].lv)
            rows += [dict(_crossing(parts, meshes, x, pv), dir="in", partner=name),
                     dict(_crossing(parts, meshes, pv, x), dir="out", partner=name)]  # fmt: skip
        out[lv] = (pv, rows)
    return out


def _solid_info(s):
    """Return a solid's parameters for the page: lengths in mm, angles in degrees."""
    dims = dict(ATTRIBUTES.get(s.type, ()))
    params = {}
    for k, v in s.params.items():
        dim = next((d for a, d in dims.items() if a.lower() == k.lower()), "U")
        params[k] = [float(v) * MM_PER_M, "mm"] if dim == "L" else ([float(np.degrees(v)), "deg"] if dim == "A"
                                                                     else [float(v), ""])  # fmt: skip
    out = dict(type=s.type, params=params, refs=list(s.refs))
    if s.planes:
        out["planes"] = [[float(x) * MM_PER_M for x in row] for row in s.planes]
    return out


def _sensor_groups(sset, members):
    return [dict(name=g.name, rule=g.rule, lv=g.lv, lv_pattern=g.lv_pattern, where=g.where,
                 values=g.values if isinstance(g.values, str) or g.values is None else "in the detector file",
                 settings={k: str(v) for k, v in g.settings.items()},
                 sensors=len(members[g.name]) if g.name in members else None)
            for g in sset.groups]  # fmt: skip


def build(source, recipe=None, *, data=None, max_instances=MAX_INSTANCES, up="z", wavelength_range=None) -> View:
    """Return the scene (``View``) of a GDML file, or of a detector package's ``"package:name"``.

    ``recipe``: a ``lumen.Recipe`` or ``FILE.py:NAME`` for a GDML file (without one, the detector volume is guessed by
    the ``auto:`` rule); ``data``: a package detector's data directory; ``up`` (x, y or z): the axis the page first
    draws vertical; ``wavelength_range`` (nm, low and high): the light's range, shaded on every table and checked
    against each table's extent.
    """
    if up not in AXES:
        raise ValueError(f"up: {up!r} is not one of {AXES}")
    mesh.require()
    with resolve(source, recipe, data) as (path, rec, inputs, name):
        return _build(path, rec, inputs, name, (max_instances, up, wavelength_range))


def _build(path, recipe, inputs, name, opts) -> View:
    max_instances, up, lam = opts
    t0 = time.time()
    errors: list = []
    notes: list = []
    reading = read(path, recipe, inputs, errors, notes)
    if reading is None:
        failed: dict = dict(schema=SCHEMA, meta=dict(name=name, source=str(path)), notes=notes,
                            errors=[dict(stage=s, text=t) for s, t in errors])  # fmt: skip
        return View(failed, errors)
    df, sset, members = reading.file, reading.sensor_set, group_members(reading)
    lam = [float(lam[0]), float(lam[1])] if lam is not None else None
    chain = _detector_chain(df, known=reading.parts is not None or df.detector_volume != df.world)
    inst = instances(df, chain, reading.parts, max_instances)
    held = _held(df, chain)
    payload: dict = dict(schema=SCHEMA, materials=_materials(reading, held, lam))
    lv_of = {p.name: p.volume for p in df.placements}
    payload["surfaces"] = {s: surface_info(reading, r, lam, lv_of, held) for s, r in df.surfaces.items()}
    payload.update(_geometry(reading, inst, members, crossings(reading, inst)))
    payload["placements"] = [dict(name=p.name, short=short(p.name), volume=p.volume, copy=p.copy)
                             for p in df.placements]  # fmt: skip
    payload["sensor_groups"] = _sensor_groups(sset, members)
    payload["sensor_ids"] = len(reading.parts.optics.sensor_pvs) if reading.parts is not None else None
    meta = _meta(df, path, name, recipe, lam, dict(inst=inst, seconds=time.time() - t0))
    det = None if chain is None else (df.placements[chain[-1]].volume if chain else df.world)
    meta.update(detector_volume=det, detector_guessed=recipe is None and det is not None, up=up,
                built=reading.parts is not None)  # fmt: skip
    payload.update(meta=meta, report=list(df.report), notes=notes, errors=[dict(stage=s, text=t) for s, t in errors])
    return View(payload, errors)


def _detector_chain(df, known):
    """Return the placements from the world to the detector volume (() the world; None: not known)."""
    if not known:
        return None
    chains = [()] if df.detector_volume == df.world else schema.placements_of(df, df.detector_volume)
    return chains[0] if len(chains) == 1 else None


def _held(df, chain):
    """Return the LVs of the detector volume and everything in it (every one when it is not known)."""
    if chain is None:
        return set(df.volumes)
    held, stack = set(), [df.placements[chain[-1]].volume if chain else df.world]
    while stack:
        lv = stack.pop()
        if lv not in held:
            held.add(lv)
            stack += [df.placements[k].volume for k in df.volumes[lv].daughters]
    return held


def _materials(reading, held, lam):
    """Return the materials for the page: those the detector volume holds are read."""
    df = reading.file
    used = {df.volumes[lv].material for lv in held}
    state = lambda m: "unread" if not reading.optical else ("" if m in used else "unused")  # noqa: E731
    return {m: material_info(r, state(m), lam) for m, r in df.materials.items()}


def _meta(df, path, name, recipe, lam, run):
    import lumen

    inst, seconds = run["inst"], run["seconds"]
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    counts = dict(solids=len(df.solids), volumes=len(df.volumes), placements=len(df.placements),
                  materials=len(df.materials), surfaces=len(df.surfaces), instances=int(len(inst["lv"])),
                  skins=len(df.skins), borders=len(df.borders))  # fmt: skip
    return dict(name=name, source=str(path), file=Path(path).name, sha256=sha, lumen=lumen.__version__,
                world=df.world, recipe=getattr(recipe, "name", None), wavelength_range=lam,
                provenance=_jsonable(df.provenance), counts=counts, instances_cut=int(inst["cut"]),
                seconds=round(seconds, 2))  # fmt: skip


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _attachments(df):
    """Return ({lv: border surfaces}, {lv: skin surfaces}) of a detector file."""
    pv_lv = {p.name: p.volume for p in df.placements}
    borders: dict = {}
    for b in df.borders:
        for side in (b.pv1, b.pv2):
            borders.setdefault(pv_lv.get(side), set()).add(b.surface)
    skins: dict = {}
    for k in df.skins:
        skins.setdefault(k.volume, []).append(k.surface)
    return borders, skins


def _geometry(reading, inst, members, cross):
    """Return the meshes, solids, volumes and instances of the page."""
    df, lvs = reading.file, inst["lvs"]
    tree = reading.parts.tree.by_name if reading.parts is not None else {}
    sensors = {g: Counter(tree[pv].lv for pv in members[g]) for g in members}
    counts = np.bincount(inst["lv"], minlength=len(lvs))
    tess = mesh.Tessellator(df, {lv: int(counts[i]) for i, lv in enumerate(lvs)})  # coarser faces for repeated solids
    meshes: list = []
    mesh_of: dict = {}
    volumes: list = []
    lv_borders, skins = _attachments(df)
    group_of = {pv: k for k, g in enumerate(members) for pv in members[g]}
    group = np.array([group_of.get(pv, -1) for pv in inst["pv"]], np.int32)
    for i, lv in enumerate(lvs):
        v = df.volumes[lv]
        if v.solid and v.solid not in mesh_of and counts[i] > 0:
            m = tess.solid(v.solid)
            mesh_of[v.solid] = len(meshes)
            meshes.append(dict(solid=v.solid, v=_b64(m.vertices, np.float32), t=_b64(m.triangles, np.uint32),
                               nv=len(m.vertices), nt=len(m.triangles), exact=m.exact, note=m.note,
                               refused=m.refused))  # fmt: skip
        sel = inst["lv"] == i
        pv, rows = cross.get(lv, (None, None))
        volumes.append(dict(
            name=lv, short=short(lv), solid=v.solid, solid_info=_solid_info(df.solids[v.solid]) if v.solid else None,
            mesh=mesh_of.get(v.solid), material=v.material, assembly=v.assembly, instances=int(counts[i]),
            inside=int(np.count_nonzero(inst["inside"][sel])), depth=int(inst["depth"][sel].min()) if counts[i] else -1,
            daughters=len(v.daughters), skins=skins.get(lv, []), borders=sorted(lv_borders.get(lv, [])),
            sensors={g: n[lv] for g, n in sensors.items() if lv in n},
            crossings=rows, first=pv,
        ))  # fmt: skip
    solids = {n: dict(type=s.type, mesh=mesh_of.get(n)) for n, s in df.solids.items()}
    data = dict(lv=_b64(inst["lv"], np.int32), parent=_b64(inst["parent"], np.int32),
                placement=_b64(inst["placement"], np.int32), group=_b64(group, np.int32),
                depth=_b64(inst["depth"], np.int32), inside=_b64(inst["inside"], np.uint8),
                matrix=_b64(inst["matrix"], np.float32), n=int(len(inst["lv"])))  # fmt: skip
    return dict(meshes=meshes, solids=solids, volumes=volumes, instances=data)
