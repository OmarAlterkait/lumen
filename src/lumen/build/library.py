"""A detector file's recorded optics decisions as the surface lookup's Library and Bindings.

    tree, library, bindings, sources = library_and_bindings(df, tree, instances, sensor_set)
    optics = surfaces.compile(library, tree, bindings, df.wavelength_range)

Each material, surface and sensor group names the optics model that reads it (``OpticsModel.name``):

Materials
    ``geant4`` (the default)  Geant4's reading (``lumen.rules.optics.medium``); the medium is named
                              ``settings["medium"]`` (default the material's name); without RINDEX it is opaque.
    ``lumen:opaque``          an opaque medium named ``settings["medium"]`` (default the material's name); several
                              materials may share one.
    ``lumen:same_as``         bound to the medium of material ``settings["material"]`` (an index-matched film lowered
                              onto a surface).
    ``lumen:larsoft_lar``     LArSoft's LAr from the material's RINDEX, RAYLEIGH and ABSLENGTH tables
                              (``lumen.detectors.larsoft_optics``), named ``settings["medium"]`` (default "lar");
                              ``scatter`` / ``absorb`` switch Rayleigh scattering and absorption.
Surfaces
    ``geant4`` (the default)  Geant4's reading (``lumen.rules.optics.surface``).
    ``lumen:larsoft_reflector``  LArSoft's reflector from the surface's REFLECTIVITY and DIFFUSE_FRACTION tables.
    ``lumen:wls_film``        a WLS film over another surface: settings ``film_material``, ``front_material``,
                              ``backing`` (a surface name), ``thickness_m`` and ``local_axis`` (the film's face normal
                              in the volume's frame). The skin's logical volume is split by each placement's film axis
                              (world rotation times ``local_axis``, rounded to 6 decimals) into ``<lv>|film<axis>``,
                              each with the surface ``<surface>:<split lv>``.
Skins and borders
    The GDML's skins are bound where their logical volume is in the tree; a recipe's or import rule's skin on a
    volume replaces the GDML's. ``df.outside_surface`` is the skin of the outside region. The GDML's borders are read
    with the sensor set's ``border_into`` group (``border_library``).
Sensor groups
    ``border_into`` with model ``lumen:geant4``: the borders into the group's volumes are sensors through the
    surface's own detection (``settings["efficiency_on_entry"]`` replaces the efficiency on entry, the sensors are
    named ``<settings["name_prefix"] or the group name>:<surface>``). ``any_entry`` with model ``lumen:counting``: a
    counting sensor of efficiency ``settings["efficiency"]`` on every placement of the group's volumes.
    ``border_into`` with model ``lumen:pmt``: tracked as ``lumen:geant4`` with every absorption entering counted
    (efficiency 1 on entry): the PMT's QE, CE and efficiency apply after tracking (``lumen.sensors``).

The outside material is always its own opaque medium, "outside". ``sources`` maps every parameter leaf of the
library (the paths of ``surfaces.named_params``, e.g. ("media", "Water", "index", "value")) to the
detector-file key it came from, as data (``Source``).
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from lumen.build import surfaces as regions
from lumen.build.tree import OUTSIDE_MATERIAL, OUTSIDE_PV, PV, Tree
from lumen.format import schema
from lumen.media import bulk, phase
from lumen.rules import optics as rules
from lumen.surfaces import types as legacy

AXIS_DECIMALS = 6  # a film axis is rounded to this many decimals to name its side
OUTSIDE_MEDIUM = "outside"
LAR_MEDIUM = "lar"  # lumen:larsoft_lar's default medium name


class LibraryError(ValueError):
    """A recorded optics decision that cannot be turned into the region compiler's library."""


class Source(NamedTuple):
    """Where a parameter leaf comes from.

    ``key`` is the detector-file key ("materials/<material>/<PROPERTY>", "surfaces/<surface>/<PROPERTY>",
    "surfaces/<surface>/sigma_alpha" or "/polish", "surfaces/<surface>/settings/<name>", "sensors/<group>/<setting>"),
    None for a value the file does not give. ``column``: "energy" or "value" of a table, "constant" (a constant property
    or the surface value), "setting", "derived" (computed from the key: ``note`` says how), "default" (Geant4's value of
    a property the file omits and the model reads: a parameter at that value) or "fixed" (a value the model never reads
    for this type, model and finish: folded into the surface type, ``fold_fixed``).
    """

    key: str | None
    column: str
    note: str = ""


# ------------------------------------------------------------------------------------------------------- the films
def _film_skins(df):
    out = {}
    for k in df.skins:
        s = df.surfaces.get(k.surface)
        if s is not None and s.optics.name == "lumen:wls_film":
            out[k.volume] = s
    return out


def split_films(df: schema.DetectorFile, tree, instances):
    """Return (tree, films): each WLS-film skin's volume split by film side, films {split lv: (surface, axis)}.

    A placement's film axis is its world rotation times the surface's ``local_axis``, rounded to 6 decimals (+ 0.0:
    no negative zero); its LV becomes ``<lv>|film<axis>``.
    """
    film_lvs = _film_skins(df)
    if not film_lvs:
        return tree, {}
    rot = {x.pv: x.R for x in instances}
    films, pvs = {}, []
    for p in tree.pvs:
        if p.lv in film_lvs:
            s = film_lvs[p.lv]
            a = np.asarray(rot[p.name], np.float64) @ np.asarray(s.optics.settings["local_axis"], np.float64)
            key = tuple(np.round(a, AXIS_DECIMALS) + 0.0)
            lv = f"{p.lv}|film{key}"
            films.setdefault(lv, (s, np.array(key)))
            p = PV(p.name, lv, p.material, p.mother, p.copy)
        pvs.append(p)
    return Tree(tuple(pvs), dict(tree.aliases), dict(tree.groups), {lv: ax for lv, (_, ax) in films.items()}), films


# ------------------------------------------------------------------------------------------------------- the media
def _opaque(name):
    return regions.MediumEntry(bulk.Medium(name, None), {})


def _material_medium(rec: schema.Material):
    """Return (library name, MediumEntry, sources by param path) of one material, or (target material, None, None)."""
    o, m = rec.optics, rec.name
    if o.name == "lumen:same_as":
        return o.settings["material"], None, None
    if o.name == "lumen:opaque":
        return o.settings.get("medium", m), _opaque(o.settings.get("medium", m)), {}
    if o.name == "geant4":
        name = o.settings.get("medium", m)
        return name, rules.medium(rec, name), _medium_sources(rec)
    if o.name == "lumen:larsoft_lar":
        from lumen.detectors import larsoft_optics

        name = o.settings.get("medium", LAR_MEDIUM)
        med, p = larsoft_optics.lar_medium_from(rec, name, o.settings.get("scatter", True),
                                                o.settings.get("absorb", True))  # fmt: skip
        return name, regions.MediumEntry(med, p), _medium_sources(rec)
    raise LibraryError(f"material {m!r}: optics model {o.name!r} is not one this library reads")


def _add_medium(media, name, entry, m):
    if name not in media:
        media[name] = entry
    elif not (media[name].medium.opaque and entry.medium.opaque):
        raise LibraryError(f"material {m!r}: medium {name!r} is defined by another material already (bind it with "
                           "lumen:same_as)")  # fmt: skip


def _media(df, tree):
    mats = sorted(set(tree.lv_material.values()) - {OUTSIDE_MATERIAL})
    media, bind, same, sources = {OUTSIDE_MEDIUM: _opaque(OUTSIDE_MEDIUM)}, {OUTSIDE_MATERIAL: OUTSIDE_MEDIUM}, {}, {}
    for m in mats:
        if m not in df.materials:
            raise LibraryError(f"material {m!r} of the tree is not in the detector file")
        name, entry, src = _material_medium(df.materials[m])
        if entry is None:
            same[m] = name
            continue
        _add_medium(media, name, entry, m)
        bind[m] = name
        sources.setdefault(name, src)
    for m, target in same.items():
        if target not in bind:
            raise LibraryError(f"material {m!r} is the same as {target!r}, which is not a medium of the tree")
        bind[m] = bind[target]
    return media, bind, sources


# ---------------------------------------------------------------------------------------------------- the surfaces
def surface_entry(s: schema.Surface) -> regions.SurfaceEntry:
    """Return the library surface of a surface record under its optics model (not a WLS film: see ``film_entry``)."""
    if s.optics.name == "geant4":
        return rules.surface(s)
    if s.optics.name == "lumen:larsoft_reflector":
        from lumen.detectors import larsoft_optics

        return regions.SurfaceEntry(*larsoft_optics.reflector_from(s.properties))
    raise LibraryError(f"surface {s.name!r}: optics model {s.optics.name!r} is not one this library reads")


def film_entry(df, s: schema.Surface, axis, types) -> regions.SurfaceEntry:
    """Return the library surface of a ``lumen:wls_film`` surface on the face of normal ``axis``."""
    st = s.optics.settings
    return rules.wls_film(df.materials[st["film_material"]], df.materials[st["front_material"]],
                          df.surfaces[st["backing"]], float(st["thickness_m"]), axis, types)  # fmt: skip


def _effective_skins(df):
    """Return {LV: surface}: the GDML's skins, a recipe's or import rule's skin replacing the GDML's on its volume."""
    out, recipe = {}, {}
    for k in df.skins:
        if k.origin != "gdml":
            if k.volume in recipe:
                raise LibraryError(f"logical volume {k.volume!r} has two recipe skins ({recipe[k.volume]!r}, "
                                   f"{k.surface!r})")  # fmt: skip
            recipe[k.volume] = k.surface
        else:
            out[k.volume] = k.surface
    out.update(recipe)
    return out


def _surfaces(df, tree, films):
    lvs = set(tree.lv_material)
    surfaces, skins, sources = {}, {}, {}
    for lv, s in _effective_skins(df).items():
        rec = df.surfaces[s]
        if rec.optics.name == "lumen:wls_film" or lv not in lvs:
            continue
        surfaces[s], skins[lv] = surface_entry(rec), s
        sources[s] = _surface_sources(rec, surfaces[s].params)
    if df.outside_surface is not None:
        rec = df.surfaces[df.outside_surface]
        surfaces[rec.name] = surface_entry(rec)
        sources[rec.name] = _surface_sources(rec, surfaces[rec.name].params)
        skins[tree.by_name[OUTSIDE_PV].lv] = rec.name
    types = {}
    for lv, (rec, axis) in films.items():
        name = f"{rec.name}:{lv}"
        surfaces[name], skins[lv] = film_entry(df, rec, axis, types), name
        sources[name] = _film_sources(df, rec)
    return surfaces, skins, sources


# ----------------------------------------------------------------------------------------------------- the sensors
BORDER_MODELS = ("lumen:geant4", "lumen:pmt")  # lumen:pmt: tracked as lumen:geant4 at efficiency 1 on entry


def _border_groups(df, tree, groups):
    border = [g for g in groups if g.rule == "border_into"]
    if len(border) > 1:
        raise LibraryError(f"sensor groups {[g.name for g in border]}: one border_into group is supported")
    if not border:
        return set(), BorderSensor(), None
    g = border[0]
    if g.model not in BORDER_MODELS:
        raise LibraryError(f"sensor group {g.name!r}: border_into reads a model of {BORDER_MODELS}, not {g.model!r}")
    pvs = {p for p in tree.order if g.selects(tree.by_name[p].lv)}
    eff_in = 1.0 if g.model == "lumen:pmt" else g.settings.get("efficiency_on_entry")
    how = BorderSensor(None, eff_in, g.settings.get("name_prefix", g.name))
    return pvs, how, g


def _counting(tree, groups):
    sensors, sbind, sources = {}, [], {}
    lvs = sorted(set(tree.lv_material))
    for g in groups:
        if g.rule != "any_entry":
            continue
        if g.model != "lumen:counting":
            raise LibraryError(f"sensor group {g.name!r}: any_entry reads the model 'lumen:counting', not {g.model!r}")
        sensors[g.name] = regions.SensorEntry(legacy.COUNTING, {"eff": float(g.settings["efficiency"])})
        sources[g.name] = {("eff",): Source(f"sensors/{g.name}/efficiency", "setting")}
        for lv in (x for x in lvs if g.selects(x)):
            sbind.append(regions.SensorBinding(g.name, len(tree.placements(lv)), lv=lv))
    return sensors, sbind, sources


def library_and_bindings(df: schema.DetectorFile, tree, instances, sensor_set):
    """Return (tree, Library, Bindings, sources) of a detector file's recorded optics (see the module docstring).

    ``tree`` is the volume tree of the lowered detector (``lumen.build.tree.tree_from_geometry``), ``instances`` its
    world instances (``instances``: their rotations split the WLS-film volumes), ``sensor_set`` the detector's
    ``SensorSet``. The returned tree has the film volumes split (the input tree when there are none).
    """
    tree, films = split_films(df, tree, instances)
    media, bind, msrc = _media(df, tree)
    surfaces, skins, ssrc = _surfaces(df, tree, films)
    pvs, how, group = _border_groups(df, tree, sensor_set.groups)
    sensors, bsurf, borders, sbind = border_library(df, tree, pvs, df.wavelength_range, how)
    for name, e in bsurf.items():
        if name not in surfaces:
            surfaces[name], ssrc[name] = e, _surface_sources(df.surfaces[name], e.params)
    csens, cbind, csrc = _counting(tree, sensor_set.groups)
    bsrc = {k: _sensor_sources(df, k, group, e) for k, e in sensors.items()}
    sensors.update(csens)
    library = regions.Library(media=media, surfaces=surfaces, sensors=sensors)
    bindings = regions.Bindings(materials=bind, skins=skins, borders=borders, sensors=tuple(sbind) + tuple(cbind))
    sources = _flatten(library, {"media": msrc, "surfaces": ssrc, "sensors": {**bsrc, **csrc}})
    return tree, fold_fixed(library, sources), bindings, sources


def _fixed(sources, section, name, param):
    """Return whether every leaf of an entry's parameter is one the model never reads, or the film's geometry.

    Structure decides, never values (docs/explanation/parameters.md): a value the model reads stays a parameter
    whatever it is, Geant4's default of an omitted property included ("default"); the film's thickness and side
    ("/settings/") are geometry.
    """
    src = [v for k, v in sources.items() if k[:3] == (section, name, param)]
    return bool(src) and all(v.column == "fixed" or "/settings/" in (v.key or "") for v in src)


def fold_fixed(library, sources):
    """Return the library with every value the model never reads folded into its surface type (``legacy.fixed``).

    OUT is structural (``_fixed``): a frontpainted finish's specular fraction, LArSoft's reflector's lobe and facet
    width, a film's WLS yield without WLSMEANNUMBERPHOTONS, the film's thickness and side.
    """

    def one(section, name, e):
        consts = {k: v for k, v in e.params.items() if not isinstance(v, tuple) and _fixed(sources, section, name, k)}
        if not consts:
            return e
        kind, rest = legacy.fixed(e.surface_type, consts), {k: v for k, v in e.params.items() if k not in consts}
        if section == "sensors":
            return regions.SensorEntry(kind, rest)
        return regions.SurfaceEntry(kind, rest, e.ignored_between_same_material)

    return regions.Library(media=library.media,
                           surfaces={n: one("surfaces", n, e) for n, e in library.surfaces.items()},
                           sensors={n: one("sensors", n, e) for n, e in library.sensors.items()})  # fmt: skip


# ----------------------------------------------------------------------------------------------------- the sources
def _table_or_constant(key, p: schema.Property | None, default_note, absent="fixed"):
    if p is None:
        return {(): Source(None, absent, default_note)}
    if rules.is_table(p):
        return {("energy",): Source(key, "energy"), ("value",): Source(key, "value")}
    return {(): Source(key, "constant")}


def _medium_sources(rec: schema.Material):
    m, pr = f"materials/{rec.name}", rec.properties
    out = {}
    for param, prop in (("index", "RINDEX"), ("absorption", "ABSLENGTH")):
        if prop in pr:
            out.update({(param,) + k: v for k, v in _table_or_constant(f"{m}/{prop}", pr[prop], "").items()})
    if "RAYLEIGH" in pr:
        out.update({("rayleigh", "length") + k: v for k, v in _table_or_constant(f"{m}/RAYLEIGH", pr["RAYLEIGH"],
                                                                                  "").items()})  # fmt: skip
    else:
        note = "Geant4's own Rayleigh table (G4OpRayleigh) from RINDEX and the temperature"
        out.update({("rayleigh", "length", c): Source(f"{m}/RINDEX", "derived", note) for c in ("energy", "value")})
    if "MIEHG" in pr:
        out.update({("mie", "length") + k: v for k, v in _table_or_constant(f"{m}/MIEHG", pr["MIEHG"], "").items()})
        fixed = rules.mie_constants(pr, f"material {rec.name!r}")[1]
        for name, key in zip(phase.MIE_PARAMS, rules.MIE_KEYS, strict=True):
            out[("mie", "lobe", name)] = (Source(f"{m}/{key}", "constant") if key in pr and key not in fixed else
                                          Source(None, "fixed", f"{key}: never read by G4OpMieHG here"))  # fmt: skip
    if "WLSABSLENGTH" in pr:
        out.update({("wls", "length") + k: v for k, v in _table_or_constant(f"{m}/WLSABSLENGTH", pr["WLSABSLENGTH"],
                                                                             "").items()})  # fmt: skip
        out[("wls", "yield")] = Source(None, "fixed", "one photon per absorption (G4OpWLS)")
        out.update(
            {
                ("wls", "tau") + k: v
                for k, v in _table_or_constant(f"{m}/WLSTIMECONSTANT", pr.get("WLSTIMECONSTANT"), "0 ns").items()
            }
        )
    return out


SURFACE_KEYS = {"R": "REFLECTIVITY", "T": "TRANSMITTANCE", "p_spike": "SPECULARSPIKECONSTANT",
                "p_lobe": "SPECULARLOBECONSTANT", "p_back": "BACKSCATTERCONSTANT", "eff": "EFFICIENCY",
                "qe": "EFFICIENCY"}  # fmt: skip
VALUE_PARAMS = ("polish", "sigma_alpha")  # the opticalsurface value
INDEX_KEYS = {"nr": ("REALRINDEX", "RINDEX"), "nk": ("IMAGINARYRINDEX", "KINDEX")}


def _surface_param(rec: schema.Surface, param: str, value):
    s, pr = f"surfaces/{rec.name}", rec.properties
    if rec.optics.name == "lumen:larsoft_reflector" and param in ("f_glossy", "sigma_alpha"):
        # larsim OpBoundaryProcessSimple: specular about the global normal or Lambertian, no lobe and no facets
        return {(): Source(None, "fixed", f"never read by LArSoft's OpBoundaryProcessSimple ({param})")}
    if rec.optics.name == "lumen:larsoft_reflector" and param in ("R", "f_spec"):
        key = "REFLECTIVITY" if param == "R" else "DIFFUSE_FRACTION"
        return {(): Source(f"{s}/{key}", "derived", "LArSoft's table at 128 nm (f_spec = 1 - the diffuse fraction)")}
    if param in SURFACE_KEYS:
        p = pr.get(SURFACE_KEYS[param])
        if p is not None and rules.is_table(p) and not isinstance(value, tuple):
            return {(): Source(f"{s}/{SURFACE_KEYS[param]}", "value", "a flat table read as its constant")}
        # G4OpBoundaryProcess::PostStepDoIt starts each step at R 1, T 0, EFFICIENCY 0, spike / lobe / back 0 and
        # reads them whether or not the surface gives them: an omitted one is a parameter at Geant4's default
        return _table_or_constant(f"{s}/{SURFACE_KEYS[param]}", p, "Geant4's default", "default")
    if param in VALUE_PARAMS:
        from lumen.build.spec import value_read

        if not value_read(rec):  # a polished or painted finish samples no facet
            return {(): Source(None, "fixed", f"no facet is sampled: {param} is never read")}
        from lumen.rules.optics import value_key

        return {(): Source(f"{s}/{value_key(rec)}", "constant")}
    if param in INDEX_KEYS:
        key = next((k for k in INDEX_KEYS[param] if k in pr), INDEX_KEYS[param][0])
        return {(): Source(f"{s}/{key}", "derived", "at the wavelength range's centre")}
    return {(): Source(None, "fixed", f"set by the surface's model, finish and type ({param})")}


def _surface_sources(rec: schema.Surface, params: dict):
    """Return the sources of a surface or sensor entry's params (by param path), from its record."""
    return {(param,) + k: v for param, value in params.items() for k, v in _surface_param(rec, param, value).items()}


def _film_sources(df, rec: schema.Surface):
    st, s = rec.optics.settings, f"surfaces/{rec.name}"
    film = df.materials[st["film_material"]]
    m = f"materials/{film.name}"
    out = {("L",) + k: v for k, v in _table_or_constant(f"{m}/WLSABSLENGTH", film.properties["WLSABSLENGTH"],
                                                        "").items()}  # fmt: skip
    out[("d",)] = Source(f"{s}/settings/thickness_m", "setting")
    out.update(
        {
            ("eta",) + k: v
            for k, v in _table_or_constant(
                f"{m}/WLSMEANNUMBERPHOTONS",
                film.properties.get("WLSMEANNUMBERPHOTONS"),
                "G4OpWLS emits one photon when WLSMEANNUMBERPHOTONS is absent: no mean is read",
            ).items()
        }
    )
    out.update(
        {
            ("tau",) + k: v
            for k, v in _table_or_constant(
                f"{m}/WLSTIMECONSTANT", film.properties.get("WLSTIMECONSTANT"), "0 ns"
            ).items()
        }
    )
    out[("axis",)] = Source(f"{s}/settings/local_axis", "derived", "the placement's world rotation times local_axis")
    backing = df.surfaces[st["backing"]]
    out.update({(f"b:{k[0]}",) + k[1:]: v for k, v in _surface_sources(backing, rules.surface(backing).params).items()})
    return out


def _sensor_sources(df, name, group, entry):
    surface = name.split(":", 1)[1].removesuffix("|out")
    out = _surface_sources(df.surfaces[surface], entry.params)
    eff = "eff" if "eff" in entry.params else "qe"
    if group is not None and not name.endswith("|out") and group.model == "lumen:pmt":
        return {k: v for k, v in out.items() if k[0] != eff}  # 1 on entry, fixed: the PMT's values apply after
    if group is not None and not name.endswith("|out") and group.settings.get("efficiency_on_entry") is not None:
        out = {k: v for k, v in out.items() if k[0] != eff}
        out[(eff,)] = Source(f"sensors/{group.name}/efficiency_on_entry", "setting")
    return out


def _leaves(params, path=()):
    for k, v in params.items():
        v = v[1] if isinstance(v, tuple) else v
        if isinstance(v, dict):
            yield from _leaves(v, path + (k,))
        else:
            yield path + (k,)


def _flatten(library, sources):
    """Return {(section, entry, *param path): Source} for every parameter leaf of the library."""
    out = {}
    for section, entries in (("media", library.media), ("surfaces", library.surfaces),
                             ("sensors", library.sensors)):  # fmt: skip
        for name, e in entries.items():
            src = sources[section].get(name, {})
            for leaf in _leaves(e.params):
                out[(section, name) + leaf] = src.get(leaf, src.get(leaf[:1], Source(None, "fixed", "unrecorded")))
    return out


# ---------------------------------------------------------------------------------------- the GDML's borders
@dataclass(frozen=True)
class BorderSensor:
    """How border_library reads the borders into a sensor volume.

    ``model`` names the sensor model (None: the default, stock Geant4's reading); ``eff_in`` replaces the detection
    efficiency on entry (a sensitive detector that counts every absorption: WCSim's, 1); ``prefix`` names the
    sensors ("<prefix>:<surface>").
    """

    model: str | None = None
    eff_in: float | None = None
    prefix: str = "sensor"


def _segments(tree):
    seg = {}
    for pv in tree.order:
        parts = pv.split("/")
        if len(parts) >= 2:  # a physvol placed several times in one mother: '<name>#<index>'
            seg.setdefault(re.sub(r"#\d+$", "", parts[-2]), []).append(pv)
    return seg


def _aliased(tree):
    al = getattr(tree, "aliases", {}) or {}

    def alias(n):  # a dissolved envelope's name -> the placement holding its region
        seen = set()
        while n in al and n not in seen:
            seen.add(n)
            n = al[n]
        return n

    return alias


def _border_pairs(tree, seg, a, b, inside):
    """Return the adjacent tree PV pairs of one border a -> b (GDML physvol names)."""
    grp = getattr(tree, "groups", {}) or {}  # dissolved envelopes: siblings touch within one envelope instance
    bs = set(seg.get(b, ()))
    b_in, b_sib = {}, {}  # b's instances by mother (x's daughters), and by (mother, envelope instance) (siblings)
    for y in bs:
        b_in.setdefault(tree.by_name[y].mother, []).append(y)
        b_sib.setdefault((tree.by_name[y].mother, grp.get(y, "")), []).append(y)
    pairs = []
    for x in seg.get(a, ()):  # x's neighbours only (daughters, mother, siblings): linear, not |a| |b|
        m = tree.by_name[x].mother
        sib = b_sib.get((m, grp.get(x, "")), ()) if m is not None else ()
        near = set(b_in.get(x, ())) | ({m} & bs) | set(sib)
        pairs += [(x, y) for y in sorted(near) if y != x and regions.relation(tree, x, y) is not None]
        # the kept detector against its GDML mother, which the tree holds as the outside: a border from the
        # detector volume into the physvol that contains it
        if m == OUTSIDE_PV and not bs and inside(a, b):
            pairs.append((x, OUTSIDE_PV))
    if not seg.get(a) and seg.get(b) and inside(b, a):  # ... and from that mother into the detector
        pairs += [(OUTSIDE_PV, y) for y in seg[b] if tree.by_name[y].mother == OUTSIDE_PV]
    return pairs


def border_bindings(df, tree) -> dict:
    """Return the GDML border surfaces on a tree: {(pre tree PV, post tree PV): surface name}.

    A GDML physvol name matches every world instance whose placement segment (the path segment before the member
    volume) is that name (or that name + '#k': one physvol placed several times in one mother); a border applies to
    each adjacent (mother/daughter or sibling) instance pair. Fail closed when a border matches nothing; a border
    whose two volumes both lie outside the kept detector is dropped with a warning.
    """
    seg = _segments(tree)
    lv_of = {p.name: p.volume for p in df.placements}  # physvol -> its LV

    def inside(dname, mname):  # is physvol dname a daughter of physvol mname's LV
        return mname in lv_of and any(df.placements[i].name == dname for i in df.volumes[lv_of[mname]].daughters)

    alias = _aliased(tree)
    out, dropped = {}, []
    for border in df.borders:
        a, b = alias(border.pv1), alias(border.pv2)
        pairs = _border_pairs(tree, seg, a, b, inside)
        if not pairs and not seg.get(a) and not seg.get(b):
            dropped.append(border.name)  # both volumes lie outside the kept detector
            continue
        if not pairs:
            raise regions.OpticsError(f"bordersurface {border.name!r} ({a!r} -> {b!r}): no adjacent world instances")
        for pr in pairs:
            out[pr] = border.surface
    if dropped:
        warnings.warn(f"{len(dropped)} border surfaces lie wholly outside the kept detector and are dropped (e.g. "
                      f"{dropped[:3]})", stacklevel=2)  # fmt: skip
    return out


def border_library(df, tree, sensor_pvs, lam_domain=None, how: BorderSensor | None = None):
    """Return the GDML's border surfaces with its sensors: (sensors, surfaces, borders, sensor bindings).

    Exactly in the directions the GDML defines them (G4LogicalBorderSurface: ordered pv1 -> pv2). A border INTO a
    sensor volume is that sensor (the photon absorbed-and-detected there by the surface's sensor model,
    ``lumen.rules.optics.sensor``); a border OUT OF one whose surface detects is a sensor too, its hit credited to the
    volume left (Geant4's Detection leaving a sensitive volume, e.g. WCSim's ProcessHits_boundary); any other border
    is its surface. A sensor volume detects only across its borders (SensorBinding entry "border"). ``how``: the
    sensor model, the efficiency on entry and the sensors' name prefix (BorderSensor).

    Returns sensors {name: SensorEntry}, surfaces {name: SurfaceEntry}, borders {(pre, post): name} and the sensor
    bindings.
    """
    how = BorderSensor() if how is None else how
    n_lv = Counter(tree.by_name[p].lv for p in sensor_pvs)
    sensors, surfaces, borders, sbind, bound = {}, {}, {}, [], set()
    for (pre, post), s in sorted(border_bindings(df, tree).items()):
        rec = df.surfaces[s]
        if post in sensor_pvs:
            k = f"{how.prefix}:{s}"
            if k not in sensors:
                sensors[k] = rules.sensor(rec, lam_domain, how.model, eff=how.eff_in)
            lv = tree.by_name[post].lv
            if (k, lv) not in bound:
                bound.add((k, lv))
                sbind.append(regions.SensorBinding(k, n_lv[lv], lv=lv, entry="border"))
            borders[(pre, post)] = k
        elif pre in sensor_pvs and rules.nonzero(rec.properties, "EFFICIENCY"):
            k = f"{how.prefix}:{s}|out"
            if k not in sensors:
                sensors[k] = rules.sensor(rec, lam_domain, how.model)
            borders[(pre, post)] = k
        else:
            if s not in surfaces:
                surfaces[s] = rules.surface(rec)
            borders[(pre, post)] = s
    return sensors, surfaces, borders, tuple(sbind)
