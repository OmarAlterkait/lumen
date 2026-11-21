"""A detector file's recorded optics decisions as the surface lookup's Library and Bindings.

    library, bindings, key_rows = library_and_bindings(df, tree, sensor_set)
    optics = lookup.compile(library, tree, bindings)

Each material and surface names the optics model that reads it (``OpticsModel.name``). A material's is looked up in
the build's model table: lumen's own (``MEDIA``) and the medium models passed to the build as values
(``lumen.models``); a surface is read by Geant4's reading, the one surface model:

Materials
    ``geant4`` (the default)  Geant4's reading (``lumen.rules.optics.medium``), the medium named after the material;
                              without RINDEX it is opaque.
    ``lumen:opaque``          an opaque medium.
    ``<distribution:name>``   a medium model passed to the build (``lumen.load(..., parts=)``).
    Every opaque material shares one medium, ``OPAQUE_MEDIUM`` (docs/dev/decisions.md, 19).
Surfaces
    ``geant4`` (the only one)  Geant4's reading (``lumen.rules.optics.surface``).
Skins and borders
    The GDML's skins are bound where their logical volume is in the tree; a recipe's or import rule's skin on a
    volume replaces the GDML's. The outside region carries the skin of the volume a photon leaving the detector volume
    enters, as Geant4 finds it (``lumen.build.binding.outside_volume``). The GDML's borders are read
    with the sensor set's ``border_into`` groups (``border_library``; a volume several select is read by the first).
Sensor groups
    ``border_into``: the borders into the group's volumes are sensors through the surface's own detection, as stock
    Geant4 reads it (``lumen.rules.optics.sensor``; the sensors are named ``<group>:<surface>``).
    ``any_entry``: a counting sensor on every placement of the group's volumes (no surface is read).
    The efficiency on entry is one rule for both (``entry_efficiency``): 1 when the group has a value set (its sensor
    model counts every absorption entering; the PMT's QE, CE and efficiency apply after tracking, ``lumen.sensors``),
    else ``settings["entry_efficiency"]``, else the surface's own reading (``lumen.rules.sensors``).

The outside material is always its own opaque medium, "outside". ``sources`` maps every parameter leaf of the
library (the paths of ``lookup.named_params``, e.g. ("media", "Water", "index", "value")) to the
detector-file key it came from, as data (``Source``).
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from lumen import models
from lumen._core import checks
from lumen._core.errors import LibraryError
from lumen.build import binding, lookup
from lumen.build.tree import OUTSIDE_MATERIAL, OUTSIDE_PV
from lumen.format import schema
from lumen.media import bulk, phase
from lumen.media.spec import MediumSpec
from lumen.rules import optics as rules
from lumen.rules import sensors as sensor_rules
from lumen.rules.keys import SURFACE_KEYS  # each surface-type parameter's Geant4 key
from lumen.surfaces import types

OUTSIDE_MEDIUM = "outside"
OPAQUE_MEDIUM = "opaque"  # every opaque material's medium: one entry, so opaque materials never split a class


class Source(NamedTuple):
    """Where a parameter leaf comes from.

    ``key`` is the detector-file key ("materials/<material>/<PROPERTY>", "surfaces/<surface>/<PROPERTY>",
    "surfaces/<surface>/sigma_alpha" or "/polish", "sensors/<group>/entry_efficiency"),
    None for a value the file does not give. ``column``: "energy" or "value" of a table, "constant" (a constant property
    or the surface value), "setting", "derived" (computed from the key: ``note`` says how; ``derive``, a JAX function
    of the key's value, computes it where the key is a parameter), "default" (Geant4's value of a property the file
    omits and the model reads: a parameter at that value) or "fixed" (a value the model never reads for this type,
    model and finish: folded into the surface type, ``fold_fixed``).
    """

    key: str | None
    column: str
    note: str = ""
    derive: object = None


DIRECT = ("energy", "value", "constant", "setting", "default")  # a packed value that is the key's value itself


# ------------------------------------------------------------------------------------------------------- the media
def _opaque(name):
    return lookup.MediumEntry(bulk.Medium(name, None), {})


def geant4_medium(rec):
    """``geant4``: Geant4's reading (``lumen.rules.optics.medium``), named after the material."""
    return rec.name, rules.medium(rec, rec.name), _medium_sources(rec)


def opaque_medium(rec):
    """``lumen:opaque``: an opaque medium named after the material."""
    return rec.name, _opaque(rec.name), {}


def _add_medium(media, name, entry, m):
    if name not in media:
        media[name] = entry
    elif not (media[name].medium.opaque and entry.medium.opaque):
        raise LibraryError(f"material {m!r}: medium {name!r} is defined by another material already")


def _media(df, tree, user):
    mats = sorted(set(tree.lv_material.values()) - {OUTSIDE_MATERIAL})
    media, bind, sources = {OUTSIDE_MEDIUM: _opaque(OUTSIDE_MEDIUM)}, {OUTSIDE_MATERIAL: OUTSIDE_MEDIUM}, {}
    for m in mats:
        if m not in df.materials:
            raise LibraryError(f"material {m!r} of the tree is not in the detector file")
        name, entry, src = material_medium(df.materials[m], user)
        if entry.medium.opaque:  # one opaque medium for every material without a refractive index
            name, entry = OPAQUE_MEDIUM, _opaque(OPAQUE_MEDIUM)
        _add_medium(media, name, entry, m)
        bind[m] = name
        sources.setdefault(name, src)
    return media, bind, sources


# ---------------------------------------------------------------------------------------------------- the surfaces
def surface_entry(s: schema.Surface) -> lookup.SurfaceEntry:
    """Return the library surface of a surface record: Geant4's reading (``lumen.rules.optics.surface``)."""
    if s.optics.name != "geant4":
        raise LibraryError(f"surface {s.name!r}: optics model {s.optics.name!r}; a surface is read by Geant4's "
                           "reading, 'geant4'")  # fmt: skip
    return rules.surface(s)


def _surfaces(df, tree):
    lvs = set(tree.lv_material)
    surfaces, skins, sources = {}, {}, {}
    for lv, s in binding.effective_skins(df).items():
        rec = df.surfaces[s]
        if lv not in lvs:
            continue
        surfaces[s], skins[lv] = surface_entry(rec), s
        sources[s] = _surface_sources(rec, surfaces[s].params)
    outside = binding.outside_skin(df)
    if outside is not None:
        rec = df.surfaces[outside]
        surfaces[rec.name] = surface_entry(rec)
        sources[rec.name] = _surface_sources(rec, surfaces[rec.name].params)
        skins[tree.by_name[OUTSIDE_PV].lv] = rec.name
    return surfaces, skins, sources


# ----------------------------------------------------------------------------------------------------- the sensors
def _border_groups(df, tree, groups):
    """Return ({sensor pv: BorderSensor}, {sensor prefix: group}) of the ``border_into`` groups.

    A volume several groups select (by the id map's columns) is read by the first; they must read it alike.
    """
    hows, by_prefix = {}, {}
    for g in (g for g in groups if g.rule == "border_into"):
        how = BorderSensor(eff_in=sensor_rules.entry_efficiency(g), prefix=g.name,
                           per_normal=sensor_rules.after_tracking(g))  # fmt: skip
        by_prefix[g.name] = g
        for p in (p for p in tree.order if g.selects(tree.by_name[p].lv)):
            first = hows.setdefault(p, how)
            if (first.eff_in, first.per_normal) != (how.eff_in, how.per_normal):
                raise LibraryError(f"sensor groups {first.prefix!r} and {g.name!r} share the volume "
                                   f"{tree.by_name[p].lv!r} but read its surfaces differently")  # fmt: skip
    return hows, by_prefix


def _sensor_entry(rec, how, eff=None, exit_=False):
    """Return a border sensor's SensorEntry (``lumen.rules.optics.sensor``); it records its surface, group and direction
    (``SensorEntry``)."""  # noqa: D205, D209
    e = rules.sensor(rec, eff=eff, per_normal=how.per_normal)
    return dataclasses.replace(e, surface=rec.name, group=how.prefix, exit=exit_)


def _counting(tree, groups):
    sensors, sbind, sources, bound = {}, [], {}, set()
    lvs = sorted(set(tree.lv_material))
    for g in groups:
        if g.rule != "any_entry":
            continue
        eff = sensor_rules.entry_efficiency(g)
        if eff is None:
            raise LibraryError(f"sensor group {g.name!r}: any_entry counts with its setting 'entry_efficiency' or a "
                               "value set (its sensor model applies the efficiency after tracking)")  # fmt: skip
        sensors[g.name] = lookup.SensorEntry(types.COUNTING, {"eff": eff}, born=sensor_rules.counts_born(g))
        sources[g.name] = {("eff",): Source(None, "fixed", "1 on entry: the sensor model's efficiency applies after "
                                            "tracking") if sensor_rules.after_tracking(g) else
                           Source(f"sensors/{g.name}/entry_efficiency", "setting")}  # fmt: skip
        for lv in (x for x in lvs if g.selects(x) and x not in bound):  # a volume groups share: counted once
            bound.add(lv)
            sbind.append(lookup.SensorBinding(g.name, len(tree.placements(lv)), lv=lv))
    return sensors, sbind, sources


def library_and_bindings(df: schema.DetectorFile, tree, sensor_set, models=None):
    """Return (Library, Bindings, key rows) of a detector file's recorded optics (see the module docstring).

    ``tree`` is the volume tree of the lowered detector (``lumen.build.tree.tree_from_geometry``), ``sensor_set`` the
    detector's ``SensorSet``, ``models`` the medium models passed to the build ({name: MediumModel},
    ``lumen.models.model_table``). The key rows map each packed leaf to the key it reads (``Source``).
    """
    for lv in sorted(set(tree.lv_material)):
        sel = [g for g in sensor_set.groups if g.selects(lv)]
        shared = (
            all(g.where for g in sel) and len({g.rule for g in sel}) == 1 and all(map(sensor_rules.after_tracking, sel))
        )
        if len(sel) > 1 and not shared:
            raise LibraryError(f"volume {lv!r} is selected by the sensor groups {[g.name for g in sel]}: a sensor is "
                               "in one group (groups sharing a volume select its sensors by the id map's columns, "
                               "count them by one rule and apply the efficiency after tracking)")  # fmt: skip
    media, bind, msrc = _media(df, tree, models)
    surfaces, skins, ssrc = _surfaces(df, tree)
    hows, groups = _border_groups(df, tree, sensor_set.groups)
    sensors, bsurf, borders, sbind = border_library(df, tree, hows)
    for name, e in bsurf.items():
        if name not in surfaces:
            surfaces[name], ssrc[name] = e, _surface_sources(df.surfaces[name], e.params)
    csens, cbind, csrc = _counting(tree, sensor_set.groups)
    bsrc = {k: _sensor_sources(df, groups.get(e.group), e) for k, e in sensors.items()}
    sensors.update(csens)
    library = lookup.Library(media=media, surfaces=surfaces, sensors=sensors, domain=rules.light_domain(df))
    bindings = lookup.Bindings(materials=bind, skins=skins, borders=borders, sensors=tuple(sbind) + tuple(cbind))
    sources = _flatten(library, {"media": msrc, "surfaces": ssrc, "sensors": {**bsrc, **csrc}})
    return fold_fixed(library, sources), bindings, sources


def _fixed(sources, section, name, param):
    """Return whether every leaf of an entry's parameter is one the model never reads.

    Structure decides, never values (docs/explanation/parameters.md): a value the model reads stays a parameter
    whatever it is, Geant4's default of an omitted property included ("default").
    """
    src = [v for k, v in sources.items() if k[:3] == (section, name, param)]
    return bool(src) and all(v.column == "fixed" for v in src)


def fold_fixed(library, sources):
    """Return the library with every value the model never reads folded into its surface type (``types.fixed``).

    OUT is structural (``_fixed``): a frontpainted finish's specular fraction, a reflector type's lobe and facet width.
    """

    def one(section, name, e):
        consts = {k: v for k, v in e.params.items() if not isinstance(v, tuple) and _fixed(sources, section, name, k)}
        if not consts:
            return e
        stype, rest = types.fixed(e.surface_type, consts), {k: v for k, v in e.params.items() if k not in consts}
        return dataclasses.replace(e, surface_type=stype, params=rest)

    return lookup.Library(media=library.media,
                           surfaces={n: one("surfaces", n, e) for n, e in library.surfaces.items()},
                           sensors={n: one("sensors", n, e) for n, e in library.sensors.items()},
                           domain=library.domain)  # fmt: skip


# ----------------------------------------------------------------------------------------------------- the sources
def _table_or_constant(key, p: schema.Property | None, default_note, absent="fixed"):
    if p is None:  # Geant4's default of an omitted property is a parameter under its key; anything else has none
        return {(): Source(key if absent == "default" else None, absent, default_note)}
    if rules.is_table(p):
        return {("energy",): Source(key, "energy"), ("value",): Source(key, "value")}
    return {(): Source(key, "constant")}


def _medium_sources(rec: schema.Material):
    m, pr = f"materials/{rec.name}", rec.properties
    out = {}
    for param, prop in (("index", "RINDEX"), ("absorption", "ABSLENGTH")):
        if prop in pr:
            out.update({(param,) + k: v for k, v in _table_or_constant(f"{m}/{prop}", pr[prop], "").items()})
    if rules.user_groupvel(pr):
        gv = _table_or_constant(f"{m}/GROUPVEL", pr["GROUPVEL"], "")
        out.update({("index", "groupvel_energy"): gv[("energy",)], ("index", "groupvel"): gv[("value",)]})
    if "RAYLEIGH" in pr:
        out.update({("rayleigh", "length") + k: v for k, v in _table_or_constant(f"{m}/RAYLEIGH", pr["RAYLEIGH"],
                                                                                  "").items()})  # fmt: skip
    elif (c1 := rules.rayleigh_constant(rec)) is not None:
        note = "Geant4's own Rayleigh table (G4OpRayleigh) from RINDEX and the temperature"
        out[("rayleigh", "length", "energy")] = Source(f"{m}/RINDEX", "energy", note)  # RINDEX's knots
        out[("rayleigh", "length", "value")] = Source(f"{m}/RINDEX", "derived", note,
                                                      functools.partial(_rayleigh, c1=c1))  # fmt: skip
    if "MIEHG" in pr:
        out.update({("mie", "length") + k: v for k, v in _table_or_constant(f"{m}/MIEHG", pr["MIEHG"], "").items()})
        fixed = rules.mie_constants(pr, f"material {rec.name!r}")[1]
        for name, key in zip(phase.MIE_PARAMS, rules.MIE_KEYS, strict=True):
            out[("mie", "lobe", name)] = (Source(f"{m}/{key}", "constant") if key in pr and key not in fixed else
                                          Source(None, "fixed", f"{key}: never read by G4OpMieHG here"))  # fmt: skip
    if "WLSABSLENGTH" in pr:
        out.update({("wls", "length") + k: v for k, v in _table_or_constant(f"{m}/WLSABSLENGTH", pr["WLSABSLENGTH"],
                                                                             "").items()})  # fmt: skip
        out.update({("wls", "yield") + k: v for k, v in _table_or_constant(
            f"{m}/WLSMEANNUMBERPHOTONS", pr.get("WLSMEANNUMBERPHOTONS"),
            "G4OpWLS emits one photon when WLSMEANNUMBERPHOTONS is absent: no mean is read").items()})  # fmt: skip
        out.update(
            {
                ("wls", "tau") + k: v
                for k, v in _table_or_constant(f"{m}/WLSTIMECONSTANT", pr.get("WLSTIMECONSTANT"), "0 ns").items()
            }
        )
        wle = _table_or_constant(f"{m}/WLSCOMPONENT", pr.get("WLSCOMPONENT"), "")
        out.update({("wls", "spectrum") + k: v for k, v in wle.items()})
    return out


def _rayleigh(t, c1):
    return rules.rayleigh_length(t.energy, t.value, c1)


VALUE_PARAMS = ("polish", "sigma_alpha")  # the opticalsurface value
INDEX_KEYS = {"nr": "REALRINDEX", "nk": "IMAGINARYRINDEX"}  # a complex index (rules.optics.complex_index)


def _surface_param(rec: schema.Surface, param: str):
    s, pr = f"surfaces/{rec.name}", rec.properties
    if param in SURFACE_KEYS:
        p = pr.get(SURFACE_KEYS[param])
        # G4OpBoundaryProcess::PostStepDoIt starts each step at R 1, T 0, EFFICIENCY 0, spike / lobe / back 0 and
        # reads them whether or not the surface gives them: an omitted one is a parameter at Geant4's default
        return _table_or_constant(f"{s}/{SURFACE_KEYS[param]}", p, "Geant4's default", "default")
    if param in VALUE_PARAMS:
        if param != rules.value_key(rec) or not rules.reads_value(rec):  # no facet is sampled
            return {(): Source(None, "fixed", f"no facet is sampled: {param} is never read")}
        return {(): Source(f"{s}/{param}", "constant")}
    if param in INDEX_KEYS:
        key = INDEX_KEYS[param]
        return _table_or_constant(f"{s}/{key}", pr[key], "")  # read as it is, at the photon's energy
    return {(): Source(None, "fixed", f"set by the surface's model, finish and type ({param})")}


def _surface_sources(rec: schema.Surface, params: dict):
    """Return the sources of a surface or sensor entry's params (by param path), from its record."""
    return {(param,) + k: v for param in params for k, v in _surface_param(rec, param).items()}


def _sensor_sources(df, group, entry):
    out = _surface_sources(df.surfaces[entry.surface], entry.params)
    eff = "eff"
    if group is None or entry.exit or sensor_rules.entry_efficiency(group) is None:
        return out
    out = {k: v for k, v in out.items() if k[0] != eff}
    if sensor_rules.after_tracking(group):
        note = "1 on entry: the sensor model's efficiency applies after tracking"
        declared = df.surfaces[entry.surface].properties.get("EFFICIENCY")
        if declared is not None and not np.all(np.asarray(declared.value, np.float64) == 1.0):
            note += (f"; the surface's EFFICIENCY ({_short(declared.value)}) is not read: the sensor model's QE is the "
                     "detection at normal incidence (docs/dev/decisions.md, 11)")  # fmt: skip
        out[(eff,)] = Source(None, "fixed", note)
    else:
        out[(eff,)] = Source(f"sensors/{group.name}/entry_efficiency", "setting")
    return out


def _short(v):
    """Return a property's values in brief: one value, or its range."""
    v = np.ravel(np.asarray(v, np.float64))
    return f"{v[0]:g}" if np.all(v == v[0]) else f"{v.min():g} to {v.max():g}"


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
                found = src.get(leaf, src.get(leaf[:1]))
                if found is None:  # every parameter's source is recorded where its entry is read: none is a gap
                    raise LibraryError(
                        f"{section} {name!r}: parameter {'/'.join(map(str, leaf))} has no recorded "
                        "source (a reader that adds a parameter must record where it comes from)"
                    )
                out[(section, name) + leaf] = found
    return out


# ---------------------------------------------------------------------------------------- the GDML's borders
@dataclass(frozen=True)
class BorderSensor:
    """How border_library reads the borders into a sensor volume.

    ``eff_in`` replaces the detection efficiency on entry (a sensitive detector that counts every absorption: WCSim's,
    1); ``prefix`` names the sensors ("<prefix>:<surface>"); ``per_normal``: the group's sensor model applies the
    datasheet QE after tracking, so the detection is taken relative to the cathode type's own normal-incidence
    absorptance (``types.per_normal``).
    """

    eff_in: float | None
    prefix: str
    per_normal: bool


def border_bindings(df, tree) -> dict:
    """Return the GDML border surfaces on a tree: {(pre physvol, post physvol): surface name}.

    A GDML physvol name is every world instance placed by that physvol (``Tree.physvol``); a border applies to each
    adjacent (mother/daughter or sibling) instance pair of its two physvols (``lookup.border_instances``: never
    enumerated), and between two physvols that never touch directly to their nearest instances, where they meet
    across coincident faces (``lookup.nearest_instances``). At the detector volume's boundary this is Geant4's own
    binding: a daughter -> the detector placement, the detector placement -> its mother (or the sibling enclosing
    it, ``outside_volume``) and that volume -> the detector placement are bound, the mother as the tree's outside
    (the last never crossed: the outside holds no photon). A border whose first volume lies outside the kept
    detector is dropped with a warning: Geant4 consults a border only for a photon in its first volume, and lumen
    tracks no light outside. Fail closed otherwise when a border matches nothing (the detector placement -> a
    sibling outside it: the outside is one region, see ``outside_volume``).
    """
    lv_of = {p.name: p.volume for p in df.placements}  # physvol -> its LV
    x, det_lv = binding.outside_volume(df), schema.detector_lv(df)
    have = tree.instances_of

    def inside(dname, mname):  # does physvol mname hold physvol dname: dname is its LV's daughter, or encloses it
        if mname in lv_of and lv_of[mname] == x and lv_of.get(dname) == det_lv:
            return True
        return mname in lv_of and any(df.placements[i].name == dname for i in df.volumes[lv_of[mname]].daughters)

    def key(a, b):  # the tree's physvol pair of a border, None: not both in the tree
        if a in have and b not in have and inside(a, b):
            a, b = a, OUTSIDE_PV  # the kept detector against its GDML mother, the tree's outside
        elif a not in have and b in have and inside(b, a):
            a, b = OUTSIDE_PV, b  # ... and from that mother into the detector
        if a not in have or b not in have:
            return None
        # touching instances, or volumes that meet only across coincident faces (``lookup.nearest_instances``;
        # one that no coincident face joins is refused when the optics compile)
        return (a, b) if lookup.border_instances(tree, a, b)[0] or lookup.nearest_instances(tree, a, b) else None

    out, dropped = {}, []
    for border in df.borders:
        a, b = border.pv1, border.pv2
        k = key(a, b)
        if k is None and a not in have:
            dropped.append(border.name)  # its first volume lies outside the kept detector: never consulted
            continue
        if k is None:
            raise lookup.OpticsError(f"bordersurface {border.name!r} ({a!r} -> {b!r}): no adjacent world instances "
                                      "(a border from the detector volume to a volume outside it other than its "
                                      "mother or an enclosing sibling: lumen's outside is one region)")  # fmt: skip
        out[k] = border.surface
    if dropped:
        warnings.warn(f"{len(dropped)} border surfaces start outside the kept detector and are dropped (e.g. "
                      f"{dropped[:3]}): Geant4 consults them only for light outside the detector volume, which lumen "
                      "does not track. This is correct; the recipe needs no action", stacklevel=2)  # fmt: skip
    return out


def _one_reading(s, pvs, hows):
    """Return the first of a border's touching instances ``pvs`` read as a sensor (``hows``), None: none is; fail
    closed when some are and some are not, or two ways (one border is one surface)."""  # noqa: D205, D209
    sel = [n for n in pvs if n in hows]
    if sel and (len(sel) != len(pvs) or len({hows[n] for n in sel}) != 1):
        raise lookup.OpticsError(f"border surface {s!r}: its instances are read two ways ({len(sel)} of {len(pvs)} "
                                  f"as a sensor, {len({hows[n] for n in sel})} ways)")  # fmt: skip
    return sel[0] if sel else None


def border_library(df, tree, hows: dict):
    """Return the GDML's border surfaces with its sensors: (sensors, surfaces, borders, sensor bindings).

    Exactly in the directions the GDML defines them (G4LogicalBorderSurface: ordered pv1 -> pv2). A border INTO a
    sensor volume is that sensor (the photon absorbed-and-detected there by the surface's sensor model,
    ``lumen.rules.optics.sensor``); a border OUT OF one whose surface detects is a sensor too, its hit credited to the
    volume left (Geant4's Detection leaving a sensitive volume, e.g. WCSim's ProcessHits_boundary); any other border
    is its surface. A sensor volume detects only across its borders (SensorBinding entry "border"). ``hows``:
    {sensor placement: BorderSensor} (its group's efficiency on entry and sensors' name prefix).

    Returns sensors {name: SensorEntry}, surfaces {name: SurfaceEntry}, borders {(pre, post): name} and the sensor
    bindings.
    """
    n_lv = Counter(tree.by_name[p].lv for p in hows)
    sensors, surfaces, borders, sbind, bound = {}, {}, {}, [], set()
    for (pre, post), s in sorted(border_bindings(df, tree).items()):
        rec = df.surfaces[s]
        xs, ys = lookup.border_instances(tree, pre, post)
        into, out_of = _one_reading(s, ys, hows), _one_reading(s, xs, hows)
        if into is not None:
            h = hows[into]
            k = f"{h.prefix}:{s}"
            if k not in sensors:
                sensors[k] = _sensor_entry(rec, h, eff=h.eff_in)
            for y in ys:
                lv = tree.by_name[y].lv
                if (k, lv) not in bound:
                    bound.add((k, lv))
                    sbind.append(lookup.SensorBinding(k, n_lv[lv], lv=lv, entry="border"))
            borders[(pre, post)] = k
        elif out_of is not None and "EFFICIENCY" in rec.properties:  # the file's structure, never the value
            h = hows[out_of]
            k = f"{h.prefix}:{s}|out"
            if k not in sensors:  # detected with the surface's EFFICIENCY (the exit rule: lumen.rules.sensors)
                sensors[k] = _sensor_entry(rec, h, exit_=True)
            borders[(pre, post)] = k
        else:
            if s not in surfaces:
                surfaces[s] = surface_entry(rec)
            borders[(pre, post)] = s
    return sensors, surfaces, borders, tuple(sbind)


# ---------------------------------------------------------------------------------------------- the model table
def model_medium_entry(m, rec, name):
    """Return (MediumEntry, sources {param path: Source}) of a material under the medium model ``m`` (a value)."""
    base = rules.medium(rec, name)
    spec = m.fn(rec, MediumSpec(base.medium, dict(base.params), m.values(rec)))
    if not isinstance(spec, MediumSpec):
        raise models.ModelError(f"medium model {m.name!r}: return a lumen.media.MediumSpec, not {type(spec).__name__}")
    _check_medium(m, spec)
    sources = _medium_sources(rec)
    keys = m.keys(rec)
    for path, k in spec.used.items():
        sources[path] = Source(*keys[k])
    return lookup.MediumEntry(spec.medium, spec.params), sources


def _check_medium(m, spec):
    for ch in spec.lobes:
        lobe = next(c.lobe for c in spec.medium.channels if c.name == ch)
        if (m, lobe.name) in models._CHECKED:
            continue
        checks.check_lobe(lobe, spec.params[ch].get("lobe", {}), f"medium model {m.name!r} ({lobe.name})")
        models._CHECKED.add((m, lobe.name))


# lumen's own medium models, by the name a material's record gives (``schema.OpticsModel.name``)
MEDIA = MappingProxyType({"geant4": geant4_medium, "lumen:opaque": opaque_medium})


def _model(table, name, what, kind):
    if name not in table:
        raise LibraryError(f"{what}: optics model {name!r} is not one this build reads as a {kind} ({sorted(table)}; "
                           "a model of your own is passed as a value: lumen.load(..., parts=))")  # fmt: skip
    return table[name]


def material_medium(rec: schema.Material, user=None):
    """Return (library name, MediumEntry, sources by param path) of a material: lumen's own model of its name
    (``MEDIA``), else the medium model of that name in ``user`` ({name: MediumModel}, ``models.model_table``)."""  # noqa: D205, D209
    name = rec.optics.name
    if name in MEDIA:
        return MEDIA[name](rec)
    m = _model(user or {}, name, f"material {rec.name!r}", "medium")
    return (rec.name, *model_medium_entry(m, rec, rec.name))
