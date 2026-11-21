"""Parameter values applied to a detector file: the one place a value replaces one of the file's values.

    df2 = lumen.format.apply(df, values, VALUE_RULES)   # values by key: a lumen.Params, or calibrations.merged(...)
                                                        # (VALUE_RULES from lumen.rules.optics)

Pure: returns a new detector file. The keys are those of ``lumen.Params``: ``materials/<M>/<KEY>`` and
``surfaces/<S>/<KEY>`` replace a property (a ``Table`` replaces a table, a number a constant),
``surfaces/<S>/sigma_alpha`` or ``surfaces/<S>/polish`` the opticalsurface's value (``rules.value_key``).
``sensors/<group>/<setting>`` keys are values after tracking: they leave the detector file as it is (``sensor_set``
applies them to a sensor set's settings).

A table keeps its knot count: a new knot count is a new detector file. One property may be added: a surface property
the file omits whose Geant4 default the model reads (``lumen.rules.optics.surface_defaults``: TRANSMITTANCE on a
polished dielectric_dielectric surface, ...: ``rules.defaults``), set to a number; it is a constant of origin
``GEANT4_DEFAULT``, which the exporter writes as the flat table Geant4 reads (an uncalibrated detector file never has
one). ``rules`` (a ``ValueRules``) carries the two lookups of the optics rules, so the format does not import them.
Fails closed (ValueError): an unknown material, surface or property (any other added one), a table given for a
constant or a number for a table, a new knot count, energies that are not strictly ascending, and any other kind of
key.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import numpy as np

from lumen.format import schema
from lumen.format.calibrations import Table

SENSOR_PREFIX = "sensors/"
VALUE_KEYS = ("sigma_alpha", "polish")  # the opticalsurface value's key, by model (rules.optics.value_key)
GEANT4_DEFAULT = "geant4_default"  # the origin of a property a parameter value added at a Geant4 default key


class ValueRules(NamedTuple):
    """What applying values reads from the optics rules (``lumen.rules.optics.VALUE_RULES``).

    ``value_key(surface)``: the key of an opticalsurface's value, "sigma_alpha" or "polish"; ``defaults(df)``:
    {surface: {KEY: ...}} of the Geant4 defaults each surface's reading reads.
    """

    value_key: Callable
    defaults: Callable


def check(key: str, old, new) -> None:
    """Refuse a value ``new`` that changes the structure of ``old`` (a Property, a Table or a number)."""
    old_e = getattr(old, "energy", None)
    if isinstance(new, Table) != (old_e is not None):
        raise ValueError(f"{key}: a {'table' if old_e is not None else 'constant'} property needs a "
                         f"{'Table' if old_e is not None else 'number'}")  # fmt: skip
    if old_e is not None:
        es, vs = np.shape(new.energy), np.shape(new.value)
        if es != np.shape(old_e) or vs != es:
            raise ValueError(f"{key}: {len(old_e)} knots expected, got {vs} values at {es} energies; a new "
                             "knot count is a new detector file")  # fmt: skip
        traced = isinstance(new.energy, jax.core.Tracer)  # a traced value is checked where it is concrete
        if not traced and np.any(np.diff(np.asarray(new.energy)) <= 0.0):
            raise ValueError(f"{key}: the energies must be strictly ascending")
    elif np.ndim(new):
        raise ValueError(f"{key}: a constant property needs a number")


def _property(p: schema.Property, v, key: str) -> schema.Property:
    check(key, p, v)
    if isinstance(v, Table):
        return dataclasses.replace(p, value=np.asarray(v.value, np.float64), energy=np.asarray(v.energy, np.float64))
    return dataclasses.replace(p, value=np.asarray(float(v), np.float64))


def _surface_value(key, rec, prop, v, rules: ValueRules) -> str:
    """Check a key of the opticalsurface's value (``rules.value_key``: sigma_alpha or polish); return "value"."""
    want = rules.value_key(rec)
    if prop != want:
        raise ValueError(f"parameter key {key!r}: surface {rec.name!r} (model {rec.model}) reads its value as {want}")
    if np.ndim(v) or isinstance(v, Table):
        raise ValueError(f"{key}: the opticalsurface value is a number")
    return "value"


def _changes(df: schema.DetectorFile, values, rules: ValueRules) -> dict:
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
        if section == "surfaces" and prop in VALUE_KEYS:
            prop = _surface_value(key, rec, prop, v, rules)
        elif prop not in rec.properties:
            if defaults is None:
                defaults = rules.defaults(df)
            if section != "surfaces" or prop not in defaults.get(name, {}):
                raise ValueError(f"parameter key {key!r}: {name!r} has no property {prop} (adding one is a new "
                                 "detector file, except a Geant4 default the model reads)")  # fmt: skip
            if np.ndim(v) or isinstance(v, Table):
                raise ValueError(f"{key}: a Geant4 default property is set to a number")
        out[section].setdefault(name, {})[prop] = v
    return out


def _added(key: str, v) -> schema.Property:
    """Return the property a value adds at a Geant4 default key."""
    return schema.Property(key, np.asarray(float(v), np.float64), origin=GEANT4_DEFAULT,
                           note="a Geant4 default key set by a parameter value")  # fmt: skip


def apply(df: schema.DetectorFile, values, rules: ValueRules) -> schema.DetectorFile:
    """Return the detector file with every material and surface value of ``values`` (a mapping by key) applied.

    ``rules``: the optics rules' and the model table's lookups (``lumen.rules.optics.value_rules``).
    """
    out = {}
    for section, changes in _changes(df, values, rules).items():
        recs = dict(getattr(df, section))
        for name, props in changes.items():
            rec = recs[name]
            extra = {"value": float(props.pop("value"))} if section == "surfaces" and "value" in props else {}
            reps = {k: _property(rec.properties[k], v, f"{section}/{name}/{k}") if k in rec.properties else
                    _added(k, v) for k, v in props.items()}  # fmt: skip
            recs[name] = dataclasses.replace(rec, properties={**rec.properties, **reps}, **extra)
        out[section] = recs
    return dataclasses.replace(df, materials=out["materials"], surfaces=out["surfaces"])


def sensor_set(sset, values):
    """Return the sensor set with each group setting that ``values`` holds (``sensors/<group>/<setting>``) replaced."""
    groups = tuple(dataclasses.replace(g, settings={k: float(values[f"{SENSOR_PREFIX}{g.name}/{k}"])
                                                    if f"{SENSOR_PREFIX}{g.name}/{k}" in values else v
                                                    for k, v in g.settings.items()})
                   for g in sset.groups)  # fmt: skip
    return dataclasses.replace(sset, groups=groups)
