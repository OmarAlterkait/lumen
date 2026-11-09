"""Sensor sets: which volumes are sensors, how a photon is counted there, and which sensor model reads them.

A sensor set holds structure only, never numbers (values live in the sensor model's defaults and in calibrations).
It is stored beside the detector file as ``sensors/<set>.json`` with its id map ``sensors/<set>.ids.json``: the
placed-volume paths of the sensors in id order, fixed at import.

- ``volumes`` says which logical volumes are sensor volumes for the geometry (they are never merged into their
  mother, and each placed copy gets its own sensor id): by an auxiliary tag value and by a name pattern.
- Each ``SensorGroup`` binds the sensors of one logical volume (or a name pattern) to a counting rule and a model:

  ``border_into``  counted across the GDML border surfaces into the volume, by the surface's own detection
                   (Geant4's G4OpBoundaryProcess Detection); a border out of the volume whose surface detects counts
                   too, credited to the volume left;
  ``any_entry``    counted on every entry into the volume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lumen.format.io import dumps

SENSOR_SCHEMA = "lumen-sensors/1"
RULES = ("border_into", "any_entry")


@dataclass(frozen=True)
class SensorVolumes:
    """Which logical volumes are sensor volumes.

    Those with one of the auxiliary values ``aux`` (auxtype SensDet), or whose name matches the regular expression
    ``names``.
    """

    aux: tuple[str, ...] = ()
    names: str | None = None


@dataclass(frozen=True)
class SensorGroup:
    """One group of sensors: the logical volume(s) it binds, the counting rule, the sensor model and its settings.

    ``lv`` is an exact logical-volume name, ``lv_pattern`` a regular expression on it (one of the two). ``settings``
    are the model's structural choices (for "lumen:geant4", ``efficiency_on_entry``: a sensitive detector that counts
    every absorption entering the volume, as WCSim's does, sets 1). ``defaults`` names the model's default values.
    """

    name: str
    rule: str
    model: str
    lv: str | None = None
    lv_pattern: str | None = None
    settings: dict = field(default_factory=dict)
    defaults: str | None = None
    readout: dict | None = None


@dataclass(frozen=True)
class SensorSet:
    """A named sensor set: the sensor volumes, the groups, and the id map (sensor paths in id order)."""

    name: str
    volumes: SensorVolumes
    groups: tuple[SensorGroup, ...]
    ids: tuple[str, ...] = ()

    def __post_init__(self):
        """Check the rules and that every group names a volume."""
        for g in self.groups:
            if g.rule not in RULES:
                raise ValueError(f"sensor group {g.name!r}: rule {g.rule!r} is not one of {RULES}")
            if (g.lv is None) == (g.lv_pattern is None):
                raise ValueError(f"sensor group {g.name!r}: give exactly one of lv and lv_pattern")


def to_json(s: SensorSet) -> tuple[str, str]:
    """Return the sensor set's JSON and its id map's JSON."""
    groups = {g.name: dict(rule=g.rule, model=g.model, lv=g.lv, lv_pattern=g.lv_pattern, settings=g.settings,
                           defaults=g.defaults, readout=g.readout) for g in s.groups}  # fmt: skip
    head = dict(schema=SENSOR_SCHEMA, name=s.name, volumes=dict(aux=list(s.volumes.aux), names=s.volumes.names),
                groups=groups, group_order=[g.name for g in s.groups], ids=f"{s.name}.ids.json")  # fmt: skip
    return dumps(head), dumps(dict(ids=list(s.ids)))


def save(s: SensorSet, root) -> Path:
    """Write the sensor set into ``<root>/sensors/`` (beside a detector file's content)."""
    d = Path(root) / "sensors"
    d.mkdir(parents=True, exist_ok=True)
    head, ids = to_json(s)
    (d / f"{s.name}.json").write_text(head)
    (d / f"{s.name}.ids.json").write_text(ids)
    return d / f"{s.name}.json"


def from_json(head: dict, ids) -> SensorSet:
    """Return the sensor set of its JSON (``to_json``'s first part, parsed) and its id map's paths."""
    if head.get("schema") != SENSOR_SCHEMA:
        raise ValueError(
            f"sensor set {head.get('name')!r}: schema {head.get('schema')!r}, this lumen reads {SENSOR_SCHEMA}"
        )
    g = head["groups"]
    groups = tuple(SensorGroup(n, g[n]["rule"], g[n]["model"], g[n]["lv"], g[n]["lv_pattern"], g[n]["settings"],
                               g[n]["defaults"], g[n]["readout"]) for n in head["group_order"])  # fmt: skip
    v = head["volumes"]
    return SensorSet(head["name"], SensorVolumes(tuple(v["aux"]), v["names"]), groups, tuple(ids))


def read(root, name: str = "default") -> SensorSet:
    """Read the sensor set ``name`` of a detector file."""
    d = Path(root) / "sensors"
    head = json.loads((d / f"{name}.json").read_text())
    if head.get("schema") != SENSOR_SCHEMA:
        raise ValueError(f"{d / name}.json: schema {head.get('schema')!r}, this lumen reads {SENSOR_SCHEMA}")
    return from_json(head, json.loads((d / head["ids"]).read_text())["ids"])
