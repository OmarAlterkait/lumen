"""Calibrations: named sets of parameter values that replace a detector file's values by GDML name, never editing it.

A calibration is ``<name>.params.json`` plus ``<name>/*.npy`` for its arrays, written anywhere on the detector
search path (by default the user's own directory), never inside a shared detector file::

    {"schema": "lumen-params/1", "name": "calib_2026_10", "detector": "sk", "detector_digest": "sha256:...",
     "sensor_set": "default", "parent": "nominal", "note": "...", "values": {
        "materials/Water/ABSLENGTH": {"energy": "calib_2026_10/0.energy.npy", "value": "calib_2026_10/0.value.npy"},
        "surfaces/BlackSheet/REFLECTIVITY": 0.051}}

``parent`` names the calibration it starts from; ``nominal`` is the detector file itself. A table keeps its knot
count (a new knot count is a new detector file).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lumen.format.io import dumps

PARAMS_SCHEMA = "lumen-params/1"
NOMINAL = "nominal"


@dataclass(frozen=True)
class Table:
    """A tabulated parameter: energies (eV) and values, both arrays of one length."""

    energy: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class Calibration:
    """A named set of parameter values (constants, arrays or Tables) keyed by GDML names."""

    name: str
    detector: str
    detector_digest: str
    values: dict = field(default_factory=dict)
    parent: str = NOMINAL
    sensor_set: str = "default"
    note: str = ""


def save(c: Calibration, directory) -> Path:
    """Write ``<directory>/<name>.params.json`` and its arrays; refuse to overwrite an existing calibration."""
    d = Path(directory)
    path = d / f"{c.name}.params.json"
    if path.exists():
        raise FileExistsError(f"calibration {path} exists: choose a new name")
    arrays = d / c.name
    values: dict[str, object] = {}
    for i, (key, v) in enumerate(sorted(c.values.items())):
        if isinstance(v, Table):
            arrays.mkdir(parents=True, exist_ok=True)
            np.save(arrays / f"{i}.energy.npy", np.asarray(v.energy, np.float64))
            np.save(arrays / f"{i}.value.npy", np.asarray(v.value, np.float64))
            values[key] = dict(energy=f"{c.name}/{i}.energy.npy", value=f"{c.name}/{i}.value.npy")
        elif np.ndim(v):
            arrays.mkdir(parents=True, exist_ok=True)
            np.save(arrays / f"{i}.npy", np.asarray(v, np.float64))
            values[key] = f"{c.name}/{i}.npy"
        else:
            values[key] = float(v)
    head = dict(schema=PARAMS_SCHEMA, name=c.name, detector=c.detector, detector_digest=c.detector_digest,
                sensor_set=c.sensor_set, parent=c.parent, note=c.note, values=values)  # fmt: skip
    d.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(head))
    return path


def read(path) -> Calibration:
    """Read a calibration file."""
    path = Path(path)
    head = json.loads(path.read_text())
    if head.get("schema") != PARAMS_SCHEMA:
        raise ValueError(f"{path}: schema {head.get('schema')!r}; this lumen reads {PARAMS_SCHEMA}")
    values = {}
    for key, v in head["values"].items():
        if isinstance(v, dict):
            values[key] = Table(np.load(path.parent / v["energy"]), np.load(path.parent / v["value"]))
        elif isinstance(v, str):
            values[key] = np.load(path.parent / v)
        else:
            values[key] = v
    return Calibration(head["name"], head["detector"], head["detector_digest"], values, head["parent"],
                       head["sensor_set"], head["note"])  # fmt: skip


def find(name: str, search) -> Path:
    """Return the path of calibration ``name`` on a search path of directories."""
    for d in search:
        p = Path(d) / f"{name}.params.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"no calibration {name!r} in {[str(d) for d in search]}")


def chain(name: str, search) -> list[Calibration]:
    """Return the calibration ``name`` and its parents, oldest first (``nominal`` is the detector file itself)."""
    out, seen = [], set()
    while name != NOMINAL:
        if name in seen:
            raise ValueError(f"calibration parent cycle through {name!r}")
        seen.add(name)
        c = read(find(name, search))
        out.append(c)
        name = c.parent
    return out[::-1]
