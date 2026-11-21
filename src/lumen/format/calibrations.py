"""Calibrations: named sets of parameter values that replace a detector file's values by GDML name, never editing it.

A calibration is ``<name>.params.json`` plus ``<name>/*.npy`` for its arrays, written anywhere on the detector
search path (by default the user's own directory), never inside a shared detector file::

    {"schema": "lumen-calibration/1", "name": "calib_2026_10", "detector": "my_detector",
     "detector_digest": "sha256:...", "note": "...", "values": {
        "materials/Water/ABSLENGTH": {"energy": "calib_2026_10/0.energy.npy", "value": "calib_2026_10/0.value.npy"},
        "surfaces/BlackSheet/REFLECTIVITY": 0.051}}

A calibration is flat: its values are every value that differs from the detector file's own (``nominal``), and
loading it applies them to the file's values; it never starts from another calibration. A table keeps its knot count
(a new knot count is a new detector file). ``detector_digest`` (the detector file's content and its sensor set) ties
a calibration to its detector; ``detector`` is a label for the reader, never checked.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np

from lumen.format import atomic
from lumen.format.io import check_format, dumps

SCHEMA = "lumen-calibration/1"  # a calibration's own format, tied to a detector by detector_digest alone

NOMINAL = "nominal"


@dataclass(frozen=True)
class Table:
    """A tabulated parameter: energies (eV) and values, both arrays of one length (a JAX pytree of the two)."""

    energy: np.ndarray
    value: np.ndarray


jax.tree_util.register_pytree_node(Table, lambda t: ((t.energy, t.value), None), lambda _, c: Table(*c))


@dataclass(frozen=True)
class Calibration:
    """A named set of parameter values (constants, arrays or Tables) keyed by GDML names, over the nominal values."""

    name: str
    detector: str
    detector_digest: str
    values: dict = field(default_factory=dict)
    note: str = ""


def save(c: Calibration, directory) -> Path:
    """Write ``<directory>/<name>.params.json`` and its arrays; refuse to overwrite an existing calibration.

    Atomic (``lumen.format.atomic``): the arrays directory ``<name>/`` is placed whole, then the head, which names
    them, is linked into place; either step finding another calibration of the name there raises ``FileExistsError``,
    also when two processes save one name at once. A reader that finds the head finds its arrays.
    """
    d = Path(directory)
    path = d / f"{c.name}.params.json"
    if path.exists():
        raise FileExistsError(f"calibration {path} exists: choose a new name")
    arrays: dict[str, np.ndarray] = {}
    values: dict[str, object] = {}
    for i, (key, v) in enumerate(sorted(c.values.items())):
        if isinstance(v, Table):
            arrays.update({f"{i}.energy.npy": v.energy, f"{i}.value.npy": v.value})
            values[key] = dict(energy=f"{c.name}/{i}.energy.npy", value=f"{c.name}/{i}.value.npy")
        elif np.ndim(v):
            arrays[f"{i}.npy"] = v
            values[key] = f"{c.name}/{i}.npy"
        else:
            values[key] = float(v)
    head = dict(schema=SCHEMA, name=c.name, detector=c.detector, detector_digest=c.detector_digest, note=c.note,
                values=values)  # fmt: skip
    placed = None
    try:
        if arrays:
            with atomic.directory(d / c.name, exists="raise") as tmp:
                for name, a in arrays.items():
                    np.save(tmp / name, np.asarray(a, np.float64))
            placed = d / c.name
        atomic.write_bytes(path, dumps(head).encode(), exclusive=True)
    except FileExistsError:
        if placed is not None:  # this save's own arrays, named by no head: another head took the name
            shutil.rmtree(placed, ignore_errors=True)
        raise FileExistsError(f"calibration {path} exists: choose a new name") from None
    return path


def read(path) -> Calibration:
    """Read a calibration file."""
    path = Path(path)
    head = json.loads(path.read_text())
    check_format(head.get("schema"), SCHEMA, path,
                 "save the calibration again with Detector.save_calibration")  # fmt: skip
    values = {}
    for key, v in head["values"].items():
        if isinstance(v, dict):
            values[key] = Table(np.load(path.parent / v["energy"]), np.load(path.parent / v["value"]))
        elif isinstance(v, str):
            values[key] = np.load(path.parent / v)
        else:
            values[key] = v
    return Calibration(head["name"], head["detector"], head["detector_digest"], values, head["note"])


class CalibrationMismatch(ValueError):  # noqa: N818 -- the documented error catalogue's name
    """A calibration made for another detector file (its ``detector_digest`` is not the detector's)."""


def check_digest(c: Calibration, digest: str, path) -> None:
    """Refuse a calibration whose ``detector_digest`` is not ``digest`` (the detector file's content digest)."""
    if c.detector_digest != digest:
        raise CalibrationMismatch(f"calibration {c.name!r} was made for detector file {c.detector_digest}, but "
                                  f"{path} is {digest}: its values may name keys or knots this file does not have. "
                                  "Remake the calibration on this file")  # fmt: skip


def find(name: str, search) -> Path:
    """Return the path of calibration ``name`` on a search path of directories."""
    for d in search:
        p = Path(d) / f"{name}.params.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"no calibration {name!r} in {[str(d) for d in search]}")
