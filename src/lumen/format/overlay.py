"""A calibration chain applied to a detector file's values: the detector file Geant4 would read with it.

    df2 = lumen.format.apply(df, [calibration, ...])      # oldest first, as lumen.format.calibrations.chain gives

Pure: returns a new detector file. The keys are those of ``lumen.Params``: ``materials/<M>/<KEY>`` and
``surfaces/<S>/<KEY>`` replace a property (a ``Table`` replaces a table, its knot count free; a number replaces a
constant), ``surfaces/<S>/value`` the opticalsurface's value (sigma_alpha or polish); a replaced property loses
its source tokens (``lumen.gdml.tokens``), so an export writes it canonically. ``sensors/...`` keys are values after
tracking: they do not change the detector file and are returned by ``sensor_values``.

Fails closed (ValueError): an unknown material, surface or property, a table given for a constant or a number for a
table (a change of structure is a new detector file), and any other kind of key.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen.format import schema
from lumen.format.calibrations import Table

SENSOR_PREFIX = "sensors/"


def _property(p: schema.Property, v, key: str) -> schema.Property:
    if isinstance(v, Table):
        if p.energy is None:
            raise ValueError(f"{key}: a table given for a constant property")
        e, val = np.asarray(v.energy, np.float64), np.asarray(v.value, np.float64)
        if e.shape != val.shape or e.ndim != 1 or len(e) < 2 or np.any(np.diff(e) <= 0.0):
            raise ValueError(f"{key}: a table needs >= 2 strictly ascending energies, one value each")
        return dataclasses.replace(p, value=val, energy=e)
    if np.ndim(v) or p.energy is not None:
        raise ValueError(f"{key}: a {'table' if p.energy is not None else 'constant'} property needs a "
                         f"{'Table' if p.energy is not None else 'number'}")  # fmt: skip
    return dataclasses.replace(p, value=np.asarray(float(v), np.float64))


def _one(key: str, v, mats: dict, surfs: dict) -> None:
    parts = key.split("/")
    if len(parts) != 3 or parts[0] not in ("materials", "surfaces"):
        raise ValueError(f"calibration key {key!r}: not materials/<M>/<KEY>, surfaces/<S>/<KEY> or sensors/...")
    kind, name, prop = parts
    table = mats if kind == "materials" else surfs
    if name not in table:
        raise ValueError(f"calibration key {key!r}: no {kind[:-1]} {name!r}")
    rec = table[name]
    if kind == "surfaces" and prop == "value":
        if np.ndim(v) or isinstance(v, Table):
            raise ValueError(f"{key}: the opticalsurface value is a number")
        table[name] = dataclasses.replace(rec, value=float(v))
        return
    if prop not in rec.properties:
        raise ValueError(
            f"calibration key {key!r}: {name!r} has no property {prop} (adding one is a new detector file)"
        )
    table[name] = dataclasses.replace(rec, properties={**rec.properties, prop: _property(rec.properties[prop], v, key)})


def apply(df: schema.DetectorFile, calibrations) -> schema.DetectorFile:
    """Return the detector file with every material and surface value of the calibrations (oldest first) applied."""
    mats, surfs = dict(df.materials), dict(df.surfaces)
    toks = {k: dict(v) for k, v in df.tokens.items()}
    for c in calibrations:
        for key in sorted(c.values):
            if not key.startswith(SENSOR_PREFIX):
                _one(key, c.values[key], mats, surfs)
                kind, name, prop = key.split("/")
                if prop in toks.get(kind, {}).get(name, {}):  # a replaced table is written canonically
                    toks[kind][name] = {k: v for k, v in toks[kind][name].items() if k != prop}
    toks = {k: {n: v for n, v in d.items() if v} for k, d in toks.items()}
    return dataclasses.replace(df, materials=mats, surfaces=surfs, tokens={k: d for k, d in toks.items() if d})


def sensor_values(calibrations) -> dict:
    """Return the ``sensors/...`` values of a calibration chain, the later calibration winning."""
    out: dict = {}
    for c in calibrations:
        out.update({k: v for k, v in c.values.items() if k.startswith(SENSOR_PREFIX)})
    return out
