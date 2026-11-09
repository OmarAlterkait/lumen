"""A detector file's recorded optics decisions as the legacy region compiler's Library and Bindings.

    tree, library, bindings, sources = library_and_bindings(df, tree, instances, sensor_set)
    optics = _legacy_regions.compile(library, tree, bindings, df.wavelength_range)

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
    The GDML's skins are bound where their logical volume is in the tree; a recipe skin on a volume replaces the
    GDML's. ``df.outside_surface`` is the skin of the outside region. The GDML's borders are read with the sensor
    set's ``border_into`` group (``_legacy_regions.border_library``).
Sensor groups
    ``border_into`` with model ``lumen:geant4``: the borders into the group's volumes are sensors through the
    surface's own detection (``settings["efficiency_on_entry"]`` replaces the efficiency on entry, the sensors are
    named ``<settings["name_prefix"] or the group name>:<surface>``). ``any_entry`` with model ``lumen:counting``: a
    counting sensor of efficiency ``settings["eff"]`` on every placement of the group's volumes.

The outside material is always its own opaque medium, "outside". ``sources`` maps every parameter leaf of the
library (the paths of ``_legacy_regions.named_params``, e.g. ("media", "Water", "index", "value")) to the
detector-file key it came from, as data (``Source``).

Notes
-----
These are the baseline recipes' libraries (lumen rebuild/next ec8de54c: lumen.detectors.wcsim.library,
sbnd_full.tree / library / film_surfaces / bindings, microboone.library / bindings) as recorded decisions.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

from lumen.build import _legacy_regions as regions
from lumen.build.tree import OUTSIDE_MATERIAL, OUTSIDE_PV, PV, Tree
from lumen.format import schema
from lumen.media import bulk, phase
from lumen.rules import optics as rules
from lumen.surfaces import _legacy as legacy

AXIS_DECIMALS = 6  # a film axis is rounded to this many decimals to name its side
OUTSIDE_MEDIUM = "outside"
LAR_MEDIUM = "lar"  # lumen:larsoft_lar's default medium name


class LibraryError(ValueError):
    """A recorded optics decision that cannot be turned into the region compiler's library."""


class Source(NamedTuple):
    """Where a parameter leaf comes from.

    ``key`` is the detector-file key ("materials/<material>/<PROPERTY>", "surfaces/<surface>/<PROPERTY>",
    "surfaces/<surface>/value", "surfaces/<surface>/settings/<name>", "sensors/<group>/<setting>"), None for a value
    lumen fixes. ``column``: "energy" or "value" of a table, "constant" (a constant property or the surface value),
    "setting", "derived" (computed from the key: ``note`` says how) or "fixed".
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
    no negative zero); its LV becomes ``<lv>|film<axis>``, as the baseline's sbnd_full.tree names it.
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
    _check_axes(film_lvs, films)
    out = Tree(tuple(pvs))
    for k in ("aliases", "groups"):
        object.__setattr__(out, k, dict(getattr(tree, k, {}) or {}))
    object.__setattr__(out, "film_axes", {lv: ax for lv, (_, ax) in films.items()})
    return out, films


AXIS_MATCH = 1e-6  # a film axis matches an expected one to this


def _check_axes(film_lvs, films):
    """Refuse films whose world axes are not the set the recipe expects (``expected_axes``), when it gives one.

    The film lowering assumes each film lies on one known face: a placement rotated onto another axis would put the
    film on a face the recipe did not check (the baseline's SBND recipe refused anything but films facing +-x).
    """
    for lv, s in film_lvs.items():
        want = s.optics.settings.get("expected_axes")
        if want is None:
            continue
        found = [ax for sl, (fs, ax) in films.items() if fs is s and sl.split("|film")[0] == lv]
        ok = len(found) == len(want) and all(any(np.allclose(f, w, atol=AXIS_MATCH) for f in found) for w in want)
        if not ok:
            raise LibraryError(f"film {s.name!r} on {lv!r}: placements face {sorted(map(tuple, found))}, the recipe "
                               f"expects {sorted(map(tuple, want))}")  # fmt: skip


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
    """Return {LV: surface}: the GDML's skins, a recipe skin replacing the GDML's on its volume."""
    out, recipe = {}, {}
    for k in df.skins:
        if k.origin == "recipe":
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
def _matches(group, lv):
    return lv == group.lv if group.lv is not None else re.search(group.lv_pattern, lv) is not None


def _border_groups(df, tree, groups):
    border = [g for g in groups if g.rule == "border_into"]
    if len(border) > 1:
        raise LibraryError(f"sensor groups {[g.name for g in border]}: one border_into group is supported")
    if not border:
        return set(), regions.BorderSensor(), None
    g = border[0]
    if g.model != "lumen:geant4":
        raise LibraryError(f"sensor group {g.name!r}: border_into reads the model 'lumen:geant4', not {g.model!r}")
    pvs = {p for p in tree.order if _matches(g, tree.by_name[p].lv)}
    how = regions.BorderSensor(None, g.settings.get("efficiency_on_entry"), g.settings.get("name_prefix", g.name))
    return pvs, how, g


def _counting(tree, groups):
    sensors, sbind, sources = {}, [], {}
    lvs = sorted(set(tree.lv_material))
    for g in groups:
        if g.rule != "any_entry":
            continue
        if g.model != "lumen:counting":
            raise LibraryError(f"sensor group {g.name!r}: any_entry reads the model 'lumen:counting', not {g.model!r}")
        sensors[g.name] = regions.SensorEntry(legacy.COUNTING, {"eff": float(g.settings["eff"])})
        sources[g.name] = {("eff",): Source(f"sensors/{g.name}/eff", "setting")}
        for lv in (x for x in lvs if _matches(g, x)):
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
    sensors, bsurf, borders, sbind = regions.border_library(df, tree, pvs, df.wavelength_range, how)
    for name, e in bsurf.items():
        if name not in surfaces:
            surfaces[name], ssrc[name] = e, _surface_sources(df.surfaces[name], e.params)
    csens, cbind, csrc = _counting(tree, sensor_set.groups)
    bsrc = {k: _sensor_sources(df, k, group, e) for k, e in sensors.items()}
    sensors.update(csens)
    library = regions.Library(media=media, surfaces=surfaces, sensors=sensors)
    bindings = regions.Bindings(materials=bind, skins=skins, borders=borders, sensors=tuple(sbind) + tuple(cbind))
    sources = _flatten(library, {"media": msrc, "surfaces": ssrc, "sensors": {**bsrc, **csrc}})
    return tree, library, bindings, sources


# ----------------------------------------------------------------------------------------------------- the sources
def _table_or_constant(key, p: schema.Property | None, default_note):
    if p is None:
        return {(): Source(None, "fixed", default_note)}
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
    if rec.optics.name == "lumen:larsoft_reflector" and param in ("R", "f_spec"):
        key = "REFLECTIVITY" if param == "R" else "DIFFUSE_FRACTION"
        return {(): Source(f"{s}/{key}", "derived", "LArSoft's table at 128 nm (f_spec = 1 - the diffuse fraction)")}
    if param in SURFACE_KEYS:
        p = pr.get(SURFACE_KEYS[param])
        if p is not None and rules.is_table(p) and not isinstance(value, tuple):
            return {(): Source(f"{s}/{SURFACE_KEYS[param]}", "value", "a flat table read as its constant")}
        return _table_or_constant(f"{s}/{SURFACE_KEYS[param]}", p, "Geant4's default")
    if param in VALUE_PARAMS:
        return {(): Source(f"{s}/value", "constant")}
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
                "1 photon per absorption (G4OpWLS)",
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
