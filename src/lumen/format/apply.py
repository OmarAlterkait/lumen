"""Parameter values applied to a detector file: the one place a value replaces one of the file's values.

    df2 = lumen.format.apply(df, values)   # values by key: a lumen.Params, or calibrations.merged(chain)

Pure: returns a new detector file. The keys are those of ``lumen.Params``: ``materials/<M>/<KEY>`` and
``surfaces/<S>/<KEY>`` replace a property (a ``Table`` replaces a table, a number a constant), ``surfaces/<S>/value``
the opticalsurface's value (sigma_alpha or polish); a replaced value loses its source tokens (``lumen.gdml.tokens``),
so an export writes it canonically. ``sensors/<group>/<setting>`` keys are values after tracking: they leave the
detector file as it is (``sensor_set`` applies them to a sensor set's settings).

A table keeps its knot count: a new knot count is a new detector file. One property may be added: a surface property
the file omits whose Geant4 default the model reads (``lumen.rules.optics.surface_defaults``: TRANSMITTANCE on a
polished dielectric_dielectric surface, ...), set to a number; it is a constant of origin ``GEANT4_DEFAULT``, which the
exporter writes as the flat table Geant4 reads (an uncalibrated detector file never has one). Fails closed
(ValueError): an unknown material, surface or property (any other added one), a table given for a constant or a number
for a table, a new knot count, energies that are not strictly ascending, and any other kind of key.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen.format import schema
from lumen.format.calibrations import Table

SENSOR_PREFIX = "sensors/"
GEANT4_DEFAULT = "geant4_default"  # the origin of a property a parameter value added at a Geant4 default key


def check(key: str, old, new) -> None:
    """Refuse a value ``new`` that changes the structure of ``old`` (a Property, a Table or a number)."""
    old_e = getattr(old, "energy", None)
    if isinstance(new, Table) != (old_e is not None):
        raise ValueError(f"{key}: a {'table' if old_e is not None else 'constant'} property needs a "
                         f"{'Table' if old_e is not None else 'number'}")  # fmt: skip
    if old_e is not None:
        e, v = np.asarray(new.energy), np.asarray(new.value)
        if e.shape != np.shape(old_e) or v.shape != e.shape:
            raise ValueError(f"{key}: {len(old_e)} knots expected, got {v.shape} values at {e.shape} energies; a new "
                             "knot count is a new detector file")  # fmt: skip
        if np.any(np.diff(e) <= 0.0):
            raise ValueError(f"{key}: the energies must be strictly ascending")
    elif np.ndim(new):
        raise ValueError(f"{key}: a constant property needs a number")


def _property(p: schema.Property, v, key: str) -> schema.Property:
    check(key, p, v)
    if isinstance(v, Table):
        return dataclasses.replace(p, value=np.asarray(v.value, np.float64), energy=np.asarray(v.energy, np.float64))
    return dataclasses.replace(p, value=np.asarray(float(v), np.float64))


def _changes(df: schema.DetectorFile, values) -> dict:
    """Return {section: {name: {key: value}}} of the material and surface keys of ``values``, checked."""
    out: dict = {"materials": {}, "surfaces": {}}
    defaults = None
    for key, v in values.items():
        if key.startswith((SENSOR_PREFIX, "source/")):  # sensor settings, and a source's own parameters
            continue
        parts = key.split("/")
        if len(parts) != 3 or parts[0] not in out:
            raise ValueError(f"parameter key {key!r}: not materials/<M>/<KEY>, surfaces/<S>/<KEY> or sensors/...")
        section, name, prop = parts
        rec = getattr(df, section).get(name)
        if rec is None:
            raise ValueError(f"parameter key {key!r}: no {section[:-1]} {name!r}")
        if section == "surfaces" and prop == "value":
            if np.ndim(v) or isinstance(v, Table):
                raise ValueError(f"{key}: the opticalsurface value is a number")
        elif prop not in rec.properties:
            if defaults is None:
                from lumen.rules.optics import detector_defaults

                defaults = detector_defaults(df)
            if section != "surfaces" or prop not in defaults.get(name, {}):
                raise ValueError(f"parameter key {key!r}: {name!r} has no property {prop} (adding one is a new "
                                 "detector file, except a Geant4 default the model reads)")  # fmt: skip
            if np.ndim(v) or isinstance(v, Table):
                raise ValueError(f"{key}: a Geant4 default property is set to a number")
        out[section].setdefault(name, {})[prop] = v
    return out


def apply(df: schema.DetectorFile, values) -> schema.DetectorFile:
    """Return the detector file with every material and surface value of ``values`` (a mapping by key) applied."""
    out, toks = {}, {k: dict(d) for k, d in df.tokens.items()}
    for section, changes in _changes(df, values).items():
        recs = dict(getattr(df, section))
        for name, props in changes.items():
            rec, keys = recs[name], set(props)
            extra = {"value": float(props.pop("value"))} if section == "surfaces" and "value" in props else {}
            reps = {k: _property(rec.properties[k], v, f"{section}/{name}/{k}") if k in rec.properties else
                    schema.Property(k, np.asarray(float(v), np.float64), origin=GEANT4_DEFAULT,
                                    note="a Geant4 default key set by a parameter value")
                    for k, v in props.items()}  # fmt: skip
            recs[name] = dataclasses.replace(rec, properties={**rec.properties, **reps}, **extra)
            if name in toks.get(section, {}):
                toks[section][name] = {k: t for k, t in toks[section][name].items() if k not in keys}
        out[section] = recs
    toks = {k: {n: t for n, t in d.items() if t} for k, d in toks.items()}
    return dataclasses.replace(df, materials=out["materials"], surfaces=out["surfaces"],
                               tokens={k: d for k, d in toks.items() if d})  # fmt: skip


def sensor_set(sset, values):
    """Return the sensor set with each group setting that ``values`` holds (``sensors/<group>/<setting>``) replaced."""
    groups = tuple(dataclasses.replace(g, settings={k: float(values[f"{SENSOR_PREFIX}{g.name}/{k}"])
                                                    if f"{SENSOR_PREFIX}{g.name}/{k}" in values else v
                                                    for k, v in g.settings.items()})
                   for g in sset.groups)  # fmt: skip
    return dataclasses.replace(sset, groups=groups)
