"""The viewer's scene: a GDML read as lumen reads it, with lumen's reading of every material, surface and volume.

    view = build("tank.gdml", recipe="my_recipes.py:TANK")   # or build("lumen_wc:sk"): a package's detector
    view.payload                                               # the JSON the page draws (``lumen-view/2``)

The GDML is imported and, with a recipe, the recipe is applied. Each material is read by its optics model: opaque, a
medium and its channels, or refused with lumen's message; each property is marked with what reads it. Each surface
gets its lumen surface type (or the sensor reader's, on a sensor), its parameters and where it is attached. Every
placed copy of every volume becomes an instance with its world matrix. A step that fails is reported in ``errors`` and
the rest is still built. Lengths m, energies eV, wavelengths nm.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np

from lumen.format import schema
from lumen.format.solids import ATTRIBUTES
from lumen.gdml import importer, reader
from lumen.units import HC, MM_PER_M
from lumen.viewer.mesh import Tessellator

SCHEMA = "lumen-view/2"
MAX_INSTANCES = 3_000_000  # placed copies drawn at most; deeper copies past it are left out, and reported
SCINT_PREFIXES = ("SCINTILLATION", "FAST", "SLOW", "RESOLUTIONSCALE", "PROTON", "DEUTERON", "TRITON", "ALPHA", "ION",
                  "ELECTRON")  # fmt: skip
SPECTRA = ("WLSCOMPONENT", "WLSCOMPONENT2", "SCINTILLATIONCOMPONENT1", "SCINTILLATIONCOMPONENT2",
           "SCINTILLATIONCOMPONENT3", "FASTCOMPONENT", "SLOWCOMPONENT")  # fmt: skip
# emission spectra: never marked as not covering the wavelength range
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
    mod_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if mod_spec is None or mod_spec.loader is None:
        raise ValueError(f"recipe {spec!r}: {path} is not a Python file")
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return getattr(module, attr)


@contextlib.contextmanager
def resolve(source, recipe=None, data=None):
    """Yield (GDML path, recipe or None, recipe inputs, name) of a GDML file or of a package's ``"package:name"``.

    ``recipe``: a ``lumen.Recipe`` or ``FILE.py:NAME`` (a GDML file read with a recipe of your own); a package's
    detector comes with its own (``lumen.DetectorEntry``), its relative paths under ``data`` (else ``LUMEN_DATA``) and
    the package's ``prepare`` applied as ``lumen.convert`` applies it.
    """
    from lumen import api

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
    notes += [dict(stage=stage, kind=type(x.message).__name__, text=str(x.message)) for x in w]
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


def read(path, recipe, inputs, errors, notes):
    """Return (detector file, sensor set or None): the recipe applied when it can be; what failed is in errors."""
    keys = recipe.material_keys if recipe is not None else None
    df = _caught(lambda: importer.read_gdml(path, keys), "import", errors, notes)
    if df is None:
        return _fallback(path, errors, notes), None
    if recipe is None:
        return df, None
    res = _caught(lambda: recipe.apply(df, inputs), "recipe", errors, notes)
    if res is None:
        return _with_detector(df, recipe, errors, notes), None
    return res.detector_file, res.sensor_set


def guess_detector(df, materials):
    """Return the volume lumen's ``auto:`` rule picks over every transparent material, for a GDML without a recipe.

    None when no material has a refractive index.
    """
    from lumen.build import lowering

    media = [m for m, info in materials.items() if info["status"] == "medium"]
    if not media:
        return None
    try:
        return lowering.find_detector(df, "auto:" + ",".join(media)).lv
    except Exception:  # a geometry the rule cannot read: no detector volume shown
        return None


def _with_detector(df, recipe, errors, notes):
    from lumen.build import lowering

    dv = _caught(lambda: lowering.find_detector(df, recipe.detector_volume), "detector volume", errors, notes)
    return df if dv is None else dataclasses.replace(df, detector_volume=dv.lv)


def instances(df, detector_lv, max_instances=MAX_INSTANCES):
    """Return every placed copy from the world, depth-first: lv index, parent, placement, depth, in-detector, matrix.

    The matrix is (n, 12): the 3x3 rotation rows then the position (m), mapping the volume's frame to the world's.
    """
    from lumen.build.lowering import placement_pose

    lvs = list(df.volumes)
    index = {lv: i for i, lv in enumerate(lvs)}
    rows: dict = dict(lv=[], parent=[], pv=[], depth=[], inside=[], matrix=[])
    stack = [(df.world, np.eye(3), np.zeros(3), -1, -1, 0, df.world == detector_lv)]
    poses: dict = {}
    cut = 0
    while stack:
        lv, lin, t, par, p, d, ins = stack.pop()
        if len(rows["lv"]) >= max_instances:
            cut += 1
            continue
        g = len(rows["lv"])
        for k, v in zip(rows, (index[lv], par, p, d, ins, np.concatenate([lin.reshape(-1), t])), strict=True):
            rows[k].append(v)
        for k in reversed(df.volumes[lv].daughters):
            if k not in poses:
                poses[k] = placement_pose(df.placements[k])
            lc, tc = poses[k]
            child = df.placements[k].volume
            stack.append((child, lin @ lc, lin @ tc + t, g, k, d + 1, ins or child == detector_lv))
    out = {k: np.asarray(rows[k], np.int32) for k in ("lv", "parent", "pv", "depth")}
    return dict(out, inside=np.asarray(rows["inside"], bool), matrix=np.asarray(rows["matrix"], float).reshape(-1, 12),
                cut=cut, lvs=lvs)  # fmt: skip


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
               lam=[HC / float(e[-1]), HC / float(e[0])])  # fmt: skip
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
    if entry is None:
        return [], {}
    med = entry.medium
    procs = (["refraction"] if med.index is not None else []) + (["absorption"] if med.absorption is not None else [])
    emission = {c.name: "+".join(c.emission.profile.components) for c in med.channels if c.emission is not None}
    return procs + [c.name for c in med.channels], emission


def _read_medium(rec):
    """Return (status, target, entry, reads, message, warnings) of a material's medium as lumen reads it."""
    from lumen.build import library

    errors: list = []
    notes: list = []
    got = _caught(lambda: library.material_medium(rec), "medium", errors, notes)
    target, entry, sources = got if got is not None else (None, None, None)
    if errors:
        status = "refused"
    elif entry is None:
        status = "bound"  # another material's medium
    else:
        status = "opaque" if entry.medium.opaque else "medium"
    live = set(entry.params) if entry is not None else set()
    reads = _source_reads({k: v for k, v in (sources or {}).items() if k and k[0] in live})
    return status, target, entry, reads, errors[0][1] if errors else "", [n["text"] for n in notes]


def material_info(rec, lam):
    """Return a material as lumen reads it: its optics model, status, channels and every property's reader."""
    from lumen.rules import optics

    status, target, entry, reads, message, warned = _read_medium(rec)
    if "SCINTILLATIONYIELD" in rec.properties:  # read by the scintillation source, not by the medium
        for k in rec.properties:
            if k.startswith(SCINT_PREFIXES):
                reads.setdefault(k, []).append(dict(param="source", column="scintillation",
                                                    note="lumen.sources.ScintillationSteps", key=""))  # fmt: skip
    derived = []
    if status == "medium" and "RAYLEIGH" not in rec.properties:
        ray = optics.geant4_rayleigh(rec)
        if ray is not None:
            read = dict(param="rayleigh/length", column="derived", note=ray.note, key="")
            derived.append(_property(ray, [read], lam))
    channels, emission = _channels(entry)
    o, comp = rec.optics, rec.composition
    yld = rec.properties.get("SCINTILLATIONYIELD")
    scint = dict(yield_per_MeV=float(np.asarray(yld.value).reshape(-1)[0])) if yld is not None else None
    return dict(
        name=rec.name, short=short(rec.name), state=rec.state, density=comp.density_g_cm3,
        temperature=rec.temperature_k, pressure=rec.pressure_pa,
        composition=[dict(ref=short(c.ref), fraction=c.fraction, atoms=c.atoms) for c in comp.components],
        optics=dict(model=o.name, origin=o.origin, note=o.note, settings={k: str(v) for k, v in o.settings.items()}),
        status=status, target=target, message=message, warnings=warned, channels=channels, emission=emission,
        properties=[_property(p, reads.get(k, []), lam) for k, p in sorted(rec.properties.items())],
        derived=derived, inert=sorted(rec.inert), scintillation=scint,
    )  # fmt: skip


def _attached(df, name, groups, pv_lv):
    """Return (skins, borders, the sensor group reading it or None) of a surface: where Geant4 finds it."""
    skins = [k.volume for k in df.skins if k.surface == name]
    borders = [b for b in df.borders if b.surface == name]
    lvs = skins + [pv_lv.get(b.pv2) for b in borders] + [pv_lv.get(b.pv1) for b in borders]
    group = next((g for g in groups for lv in lvs if lv is not None and g.selects(lv)), None)
    return skins, borders, group


def _sensor_reading(df, rec, group):
    """Return (sensor entry, sources) of a surface read by a border sensor group, as the build reads it."""
    from lumen.build import library
    from lumen.rules import sensors as sensor_rules

    how = library.BorderSensor(eff_in=sensor_rules.entry_efficiency(group), prefix=group.name,
                               per_normal=sensor_rules.after_tracking(group))  # fmt: skip
    entry = library._sensor_entry(rec, how, eff=how.eff_in)
    return entry, library._sensor_sources(df, group, entry)


def _fixed_notes(sources):
    """Return {surface property key: why lumen does not read it} of the parameters fixed by the surface's reading."""
    from lumen.rules.keys import SURFACE_KEYS

    return {SURFACE_KEYS[path[0]]: f"not read: {src.note}" for path, src in sources.items()
            if src.column == "fixed" and path and path[0] in SURFACE_KEYS and src.note}  # fmt: skip


def surface_info(df, rec, groups, lam, pv_lv):
    """Return a surface as lumen reads it: its Geant4 form, lumen's surface type (or sensor reader's) and params."""
    from lumen.build import library
    from lumen.rules import optics

    errors: list = []
    notes: list = []
    skins, borders, group = _attached(df, rec.name, groups, pv_lv)
    entry = _caught(lambda: library.surface_entry(rec), "surface", errors, notes)
    status, stype = ("ok", entry.surface_type.name) if entry is not None else ("refused", None)
    sources = library._surface_sources(rec, entry.params) if entry is not None else {}
    sensor_type = None
    if group is not None and group.rule == "border_into":
        got = _caught(lambda: _sensor_reading(df, rec, group), "sensor", [], [])
        if got is not None:
            status, sensor_type, sources = "sensor", got[0].surface_type.name, got[1]
    reads = _source_reads(sources)
    why = _fixed_notes(sources)
    geant4 = dict(type=rec.type, model=rec.model, finish=rec.finish, value=rec.value, value_name=optics.value_key(rec))
    return dict(
        name=rec.name, short=short(rec.name), geant4=geant4,
        optics=dict(model=rec.optics.name, origin=rec.optics.origin, note=rec.optics.note), status=status,
        surface_type=stype, sensor_type=sensor_type, message=errors[0][1] if (errors and status != "sensor") else "",
        refused_as_surface=errors[0][1] if errors else "",
        params=[dict(param="/".join(p), key=s.key, column=s.column, note=s.note) for p, s in sources.items()],
        properties=[dict(_property(p, reads.get(k, []), lam), **({"why": why[k]} if k in why and not reads.get(k)
                                                                  else {})) for k, p in sorted(rec.properties.items())],
        skins=skins, borders=[[b.pv1, b.pv2] for b in borders[:200]], n_borders=len(borders),
    )  # fmt: skip


def _default(pre, post, materials):
    """Return what lumen does where no surface applies."""
    a, b = materials.get(pre or ""), materials.get(post or "")
    if pre == post:
        return "pass", "no boundary: the same material on both sides"
    if b is None or b["status"] == "opaque":
        return "absorb", "absorbed: no RINDEX beyond (Geant4's NoRINDEX)"
    if b["status"] == "refused" or (a is not None and a["status"] == "refused"):
        return "refused", "a material lumen refuses"
    return "fresnel", "Fresnel reflection or refraction between the two indices"


def _side(found, pre, post, ctx):
    """Return one crossing (pre -> post materials) given the surface found by the border and skin rules, if any."""
    materials, surfaces = ctx
    via, name = found
    a = materials.get(pre or "")
    if pre is not None and (a is None or a["status"] == "opaque"):  # None: an unnamed neighbour, taken transparent
        return dict(via="material", surface=None, outcome="none", text="no photon travels on this side (no RINDEX)")
    s = surfaces.get(name) if name else None
    if s is None:
        out, what = _default(pre, post, materials)
        return dict(via="material", surface=None, outcome=out, text=what)
    g4 = s["geant4"]
    if pre == post and g4["type"] == "dielectric_dielectric" and g4["finish"] in ("polished", "ground"):
        out, what = _default(pre, post, materials)
        return dict(via=via, surface=name, outcome=out, text=f"skipped (the same material on both sides): {what}")
    if s["status"] == "sensor":
        return dict(via=via, surface=name, outcome="sensor", text=f"read by the sensor's reader: {s['sensor_type']}")
    if s["status"] == "refused":
        return dict(via=via, surface=name, outcome="refused", text="a surface lumen refuses")
    return dict(via=via, surface=name, outcome="surface", text=f"surface type {s['surface_type']}")


@dataclasses.dataclass(frozen=True)
class _Boundaries:
    """A detector file's surfaces by where Geant4 looks them up: border pairs, borders by placement, skins by volume."""

    border: dict
    by_pv: dict
    skin: dict
    pv_lv: dict

    @classmethod
    def of(cls, df):
        """Return the lookup tables of a detector file."""
        by_pv: dict = {}
        for b in df.borders:
            by_pv.setdefault(b.pv1, []).append(b)
            by_pv.setdefault(b.pv2, []).append(b)
        return cls({(b.pv1, b.pv2): b.surface for b in df.borders}, by_pv, {k.volume: k.surface for k in df.skins},
                   {p.name: p.volume for p in df.placements})  # fmt: skip

    def find(self, pair, lv, mother):
        """Return (via, surface) of a crossing between a volume and its mother: the border, else the skins."""
        if pair in self.border:
            return "border", self.border[pair]
        for who in (lv, mother):
            if who in self.skin:
                return f"skin of {short(who)}", self.skin[who]
        return None, None


def _rows(df, bd, copy, ctx, group):
    """Return the crossings of one volume's first copy ``copy`` = (lv, pv, mother, mother's pv)."""
    lv, pv, mother, mpv = copy
    mat, mmat = df.volumes[lv].material, df.volumes[mother].material
    enter = dict(_side(bd.find((mpv, pv), lv, mother), mmat, mat, ctx), dir="in", partner=short(mother))
    if group is not None and group.rule == "any_entry":
        enter.update(via="sensor", surface=None, outcome="sensor", text=f"counted on entry (group {group.name})")
    rows = [enter, dict(_side(bd.find((pv, mpv), lv, mother), mat, mmat, ctx), dir="out", partner=short(mother))]
    if lv in bd.skin:  # the volume's own skin, met when a touching volume other than its mother enters it
        row = _side((f"skin of {short(lv)}", bd.skin[lv]), None, mat, ctx)
        text = row["text"] + " (unless that volume has a skin or a border of its own)"
        rows.append(dict(row, dir="in", partner="a touching volume", text=text))
    for b in bd.by_pv.get(pv, []):
        if mpv in (b.pv1, b.pv2):
            continue  # with the mother: shown above
        other = b.pv1 if b.pv2 == pv else b.pv2
        pre, post = df.volumes[bd.pv_lv[b.pv1]].material, df.volumes[bd.pv_lv[b.pv2]].material
        row = _side(("border", b.surface), pre, post, ctx)
        rows.append(dict(row, dir="in" if b.pv2 == pv else "out", partner=short(bd.pv_lv.get(other, other))))
    return rows


def crossings(df, inst, materials, surfaces, sset):
    """Return {lv: [crossing, ...]}: lumen's boundary rule at each volume's first copy, every border included.

    Geant4's order: an any-entry sensor; the border surface of the ordered pair of placed volumes; the skins (the
    daughter's, then the mother's, both ways); the material default. The crossings with the mother come first (in,
    then out), then the volume's own skin met from a touching volume, then every border surface the copy takes part in,
    with its partner. Coincident faces are not shown.
    """
    bd = _Boundaries.of(df)
    first: dict = {}
    for g, li in enumerate(inst["lv"]):
        first.setdefault(int(li), g)
    groups = sset.groups if sset is not None else ()
    out = {}
    for li, g in first.items():
        par = int(inst["parent"][g])
        if par < 0:
            continue
        lv, mother = inst["lvs"][li], inst["lvs"][int(inst["lv"][par])]
        mpv = df.placements[int(inst["pv"][par])].name if inst["pv"][par] >= 0 else None
        copy = (lv, df.placements[int(inst["pv"][g])].name, mother, mpv)
        out[lv] = _rows(df, bd, copy, (materials, surfaces), next((x for x in groups if x.selects(lv)), None))
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


def _sensor_groups(sset):
    if sset is None:
        return []
    return [dict(name=g.name, rule=g.rule, lv=g.lv, lv_pattern=g.lv_pattern, where=g.where,
                 values=g.values if isinstance(g.values, str) or g.values is None else "inline",
                 settings={k: str(v) for k, v in g.settings.items()})
            for g in sset.groups]  # fmt: skip


def _group_of(sset, lv):
    if sset is None:
        return None
    return next((g.name for g in sset.groups if g.selects(lv)), None)


def build(source, recipe=None, *, data=None, max_instances=MAX_INSTANCES, up="z", wavelength_range=None) -> View:
    """Return the scene (``View``) of a GDML file, or of a detector package's ``"package:name"``.

    ``recipe``: a ``lumen.Recipe`` or ``FILE.py:NAME`` for a GDML file (without one, the detector volume is guessed by
    the ``auto:`` rule); ``data``: a package detector's data directory; ``up`` (x, y or z): the axis the page first
    draws vertical; ``wavelength_range`` (nm, low and high): the light's range, shaded on every table and checked
    against each table's extent.
    """
    if up not in AXES:
        raise ValueError(f"up: {up!r} is not one of {AXES}")
    with resolve(source, recipe, data) as (path, rec, inputs, name):
        return _build(path, rec, inputs, name, (max_instances, up, wavelength_range))


def _build(path, recipe, inputs, name, opts) -> View:
    max_instances, up, lam = opts
    t0 = time.time()
    errors: list = []
    notes: list = []
    df, sset = read(path, recipe, inputs, errors, notes)
    if df is None:
        failed: dict = dict(schema=SCHEMA, meta=dict(name=name, source=str(path)), notes=notes,
                            errors=[dict(stage=s, text=t) for s, t in errors])  # fmt: skip
        return View(failed, errors)
    lam = [float(lam[0]), float(lam[1])] if lam is not None else None
    payload: dict = dict(schema=SCHEMA)
    payload["materials"] = {m: material_info(r, lam) for m, r in df.materials.items()}
    det_lv = schema.detector_lv(df) if recipe is not None else guess_detector(df, payload["materials"])
    inst = instances(df, det_lv, max_instances)
    groups = sset.groups if sset is not None else ()
    pv_lv = {p.name: p.volume for p in df.placements}
    payload["surfaces"] = {s: surface_info(df, r, groups, lam, pv_lv) for s, r in df.surfaces.items()}
    payload.update(_geometry(df, inst, sset, payload["materials"]))
    cross = crossings(df, inst, payload["materials"], payload["surfaces"], sset)
    for v in payload["volumes"]:
        v["crossings"] = cross.get(v["name"])
    payload["placements"] = [dict(name=p.name, short=short(p.name), volume=p.volume, copy=p.copy)
                             for p in df.placements]  # fmt: skip
    payload["sensor_groups"] = _sensor_groups(sset)
    payload["sensor_ids"] = len(sset.ids) if sset is not None and sset.ids else None
    meta = _meta(df, path, name, recipe, lam, dict(inst=inst, seconds=time.time() - t0))
    meta.update(detector_volume=det_lv, detector_guessed=recipe is None and det_lv is not None, up=up)
    payload.update(meta=meta, report=list(df.report), notes=notes, errors=[dict(stage=s, text=t) for s, t in errors])
    return View(payload, errors)


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


def _geometry(df, inst, sset, materials):
    """Return the meshes, solids, volumes and instances of the page."""
    lvs = inst["lvs"]
    counts = np.bincount(inst["lv"], minlength=len(lvs))
    tess = Tessellator(df, {lv: int(counts[i]) for i, lv in enumerate(lvs)})  # coarser faces for repeated solids
    meshes: list = []
    mesh_of: dict = {}
    volumes: list = []
    lv_borders, skins = _attachments(df)
    for i, lv in enumerate(lvs):
        v = df.volumes[lv]
        if v.solid and v.solid not in mesh_of and counts[i] > 0:
            m = tess.solid(v.solid)
            mesh_of[v.solid] = len(meshes)
            meshes.append(dict(solid=v.solid, v=_b64(m.vertices, np.float32), t=_b64(m.triangles, np.uint32),
                               nv=len(m.vertices), nt=len(m.triangles), exact=m.exact, note=m.note))  # fmt: skip
        sel = inst["lv"] == i
        mat = materials.get(v.material) if v.material else None
        volumes.append(dict(
            name=lv, short=short(lv), solid=v.solid, solid_info=_solid_info(df.solids[v.solid]) if v.solid else None,
            mesh=mesh_of.get(v.solid), material=v.material, assembly=v.assembly, instances=int(counts[i]),
            inside=int(np.count_nonzero(inst["inside"][sel])), depth=int(inst["depth"][sel].min()) if counts[i] else -1,
            daughters=len(v.daughters), skins=skins.get(lv, []), borders=sorted(lv_borders.get(lv, [])),
            sensor=_group_of(sset, lv), status=mat["status"] if mat else None,
        ))  # fmt: skip
    solids = {n: dict(type=s.type, mesh=mesh_of.get(n)) for n, s in df.solids.items()}
    data = dict(lv=_b64(inst["lv"], np.int32), parent=_b64(inst["parent"], np.int32), pv=_b64(inst["pv"], np.int32),
                depth=_b64(inst["depth"], np.int32), inside=_b64(inst["inside"], np.uint8),
                matrix=_b64(inst["matrix"], np.float32), n=int(len(inst["lv"])))  # fmt: skip
    return dict(meshes=meshes, solids=solids, volumes=volumes, instances=data)
