"""Sensor sets: which volumes are sensors, how a photon is counted there, and which readers and models read them.

A sensor set holds structure only, never numbers (values live in the sensor model's defaults and in calibrations).
It is stored beside the detector file as ``sensors/<set>.json`` with its id map ``sensors/<set>.ids.json``: the
placed-volume paths of the sensors in id order, fixed at import.

Each ``SensorGroup`` selects logical volumes, one by its name (``lv``) or those whose name matches a regular expression
(``lv_pattern``), and binds them to a counting rule, the surface reader that decides how its detecting surface behaves
during tracking (``surface``: ``lumen:geant4``, ``lumen:counting``) and, optionally, the sensor model that turns the
light into photoelectrons after tracking (``model``: ``lumen:pmt``). The volumes the groups select are the sensor
volumes of the geometry (never merged into their mother; each placed copy gets its own sensor id). The rules:

  ``border_into``  counted across the GDML border surfaces into the volume, by the surface's own detection
                   (Geant4's G4OpBoundaryProcess Detection); a border out of the volume whose surface detects counts
                   too, credited to the volume left;
  ``any_entry``    counted on every entry into the volume.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lumen.format.io import dumps

SENSOR_SCHEMA = "lumen-sensors/2"
READABLE = ("lumen-sensors/1", SENSOR_SCHEMA)  # /1: one "model" for both (migrated on read, from_json)
RULES = ("border_into", "any_entry")


@dataclass(frozen=True)
class SensorGroup:
    """One group of sensors: the logical volume(s) it binds, the counting rule, the surface reader, its settings and
    the sensor model.

    ``surface`` names the surface reader (during tracking: ``lumen:geant4`` for ``border_into``, ``lumen:counting``
    for ``any_entry``). ``lv`` is an exact logical-volume name, ``lv_pattern`` a regular expression on it (one of the
    two). ``settings`` are structural choices (for ``lumen:geant4``, ``efficiency_on_entry``: a sensitive detector
    that counts every absorption entering the volume, as WCSim's does, sets 1; for ``lumen:counting``,
    ``efficiency``). ``model`` names the sensor model after tracking (``lumen:pmt``; None: none) and ``defaults`` its
    default values.
    """  # noqa: D205

    name: str
    rule: str
    surface: str
    lv: str | None = None
    lv_pattern: str | None = None
    settings: dict = field(default_factory=dict)
    model: str | None = None
    defaults: str | None = None
    readout: dict | None = None

    def selects(self, lv: str) -> bool:
        """Return whether the group binds the logical volume ``lv``."""
        return lv == self.lv if self.lv is not None else re.search(str(self.lv_pattern), lv) is not None


@dataclass(frozen=True)
class SensorSet:
    """A named sensor set: the groups and the id map (sensor paths in id order)."""

    name: str
    groups: tuple[SensorGroup, ...]
    ids: tuple[str, ...] = ()

    def __post_init__(self):
        """Check the rules and that every group names a volume."""
        for g in self.groups:
            if g.rule not in RULES:
                raise ValueError(f"sensor group {g.name!r}: rule {g.rule!r} is not one of {RULES}")
            if (g.lv is None) == (g.lv_pattern is None):
                raise ValueError(f"sensor group {g.name!r}: give exactly one of lv and lv_pattern")
            if "eff" in g.settings:  # its name before lumen 0.1
                raise ValueError(f"sensor group {g.name!r}: the setting 'eff' is now 'efficiency' (rename it, or "
                                 "convert the detector again)")  # fmt: skip

    def is_sensor(self, lv: str) -> bool:
        """Return whether the logical volume ``lv`` is a sensor volume: one that a group selects."""
        return any(g.selects(lv) for g in self.groups)


def to_json(s: SensorSet) -> tuple[str, str]:
    """Return the sensor set's JSON and its id map's JSON."""
    groups = {g.name: dict(rule=g.rule, surface=g.surface, model=g.model, lv=g.lv, lv_pattern=g.lv_pattern,
                           settings=g.settings, defaults=g.defaults, readout=g.readout) for g in s.groups}  # fmt: skip
    head = dict(schema=SENSOR_SCHEMA, name=s.name, groups=groups, group_order=[g.name for g in s.groups],
                ids=f"{s.name}.ids.json")  # fmt: skip
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
    """Return the sensor set of its JSON (``to_json``'s first part, parsed) and its id map's paths.

    A ``volumes`` entry (written before the groups' selectors became the only one) is ignored. Schema
    ``lumen-sensors/1`` named one ``model`` for both roles; it is read as ``/2`` says it: ``lumen:pmt`` is the sensor
    model over the ``lumen:geant4`` surface reader, any other name the surface reader with no sensor model.
    """
    if head.get("schema") not in READABLE:
        raise ValueError(f"sensor set {head.get('name')!r}: schema {head.get('schema')!r}, this lumen reads {READABLE}")
    g, v1 = head["groups"], head["schema"] == READABLE[0]
    groups = []
    for n in head["group_order"]:
        surface, model = (("lumen:geant4", "lumen:pmt") if g[n]["model"] == "lumen:pmt" else (g[n]["model"], None)) \
            if v1 else (g[n]["surface"], g[n]["model"])  # fmt: skip
        groups.append(SensorGroup(n, g[n]["rule"], surface, g[n]["lv"], g[n]["lv_pattern"], g[n]["settings"], model,
                                  g[n]["defaults"], g[n]["readout"]))  # fmt: skip
    return SensorSet(head["name"], tuple(groups), tuple(ids))


def read(root, name: str = "default") -> SensorSet:
    """Read the sensor set ``name`` of a detector file."""
    d = Path(root) / "sensors"
    head = json.loads((d / f"{name}.json").read_text())
    if head.get("schema") not in READABLE:
        raise ValueError(f"{d / name}.json: schema {head.get('schema')!r}, this lumen reads {READABLE}")
    return from_json(head, json.loads((d / head["ids"]).read_text())["ids"])
