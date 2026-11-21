"""The sensor set: which volumes are sensors, how a photon is counted there, and the value sets that read them.

A detector file has one sensor set. It holds structure, and the value set each modelled group reads (its sensor
model's nominal values, copied in when the detector is converted, so the file, not the installed lumen, defines them;
calibrations change them). It is stored beside the detector file's content as ``sensors.json`` with its id map
``sensors.ids.json``: the placed-volume paths of the sensors in id order, fixed at import. Both are in the
detector's digest (``lumen.Detector.digest``).

Each ``SensorGroup`` selects logical volumes, one by its name (``lv``) or those whose name matches a regular expression
(``lv_pattern``), optionally only the sensors whose id-map columns hold given values (``where``: {"type":
"pmt_coated"}, for sensors of one volume that differ by type), and binds them to a counting rule and, optionally, a
value set (``values``): a group with a value set is read by the sensor model after tracking (``lumen.sensors``),
which turns the light into photoelectrons. The volumes the groups select are the sensor volumes of the geometry
(never merged into their mother; each placed copy gets its own sensor id). The rules:

  ``border_into``  counted across the GDML border surfaces into the volume, by the surface's own detection as stock
                   Geant4 reads it (G4OpBoundaryProcess Detection, ``lumen.rules.optics.sensor``); a border out of the
                   volume whose surface detects counts too, credited to the volume left;
  ``any_entry``    counted on every entry into the volume.
"""

from __future__ import annotations

import json
import re
from dataclasses import KW_ONLY, MISSING, dataclass, field, fields
from pathlib import Path

from lumen.format import atomic
from lumen.format.io import FormatError, check_format, dumps
from lumen.format.schema import SCHEMA  # the detector directory's one version

RULES = ("border_into", "any_entry")
COUNTS = ("volume", "entry")  # an any_entry group's setting "count"; the first is the default
SETTINGS = ("entry_efficiency", "count")  # a group's structural settings
GROUP_NAME = re.compile(r"[A-Za-z_]\w*")  # a group's name is one segment of its keys, sensors/<group>/<field>
HEAD, IDS = "sensors.json", "sensors.ids.json"  # the sensor set's files in a detector file


@dataclass(frozen=True)
class SensorGroup:
    """One group of sensors: the logical volume(s) it binds, the counting rule, its settings and its value set.

    ``lv`` is an exact logical-volume name, ``lv_pattern`` a regular expression on it (one of the two). ``settings``
    are structural choices (``entry_efficiency``: the detection efficiency on entry of a group without a value set, for
    both rules; a sensitive detector that counts every absorption entering the volume, as WCSim's does, sets 1;
    absent: the surface's EFFICIENCY; ``count``, for ``any_entry``: "volume", the default, also counts a photon emitted
    inside the volume at its emission as Geant4's sensitive detector does, "entry" counts only entries,
    ``lumen.rules.sensors.counts_born``). ``values`` is the group's value set: a reference (``<package>:<set>`` or a
    path) in a recipe, its content once converted; a group with one is read by the sensor model after tracking, which
    applies the efficiency itself (every absorption entering is counted). None: no sensor model.
    ``where``: {column: value} of the id map a sensor of the selected volumes must hold to be in the group (None:
    every sensor of them); groups sharing a volume this way share its counting.
    """

    name: str
    _: KW_ONLY
    rule: str
    lv: str | None = None
    lv_pattern: str | None = None
    settings: dict = field(default_factory=dict)
    readout: dict | None = None
    where: dict | None = None
    values: str | dict | None = None

    def selects(self, lv: str) -> bool:
        """Return whether the group binds the logical volume ``lv``."""
        return lv == self.lv if self.lv is not None else re.search(str(self.lv_pattern), lv) is not None


@dataclass(frozen=True)
class SensorSet:
    """The sensor set of a detector: the groups and the id map (sensor paths in id order)."""

    groups: tuple[SensorGroup, ...]
    ids: tuple[str, ...] = ()
    columns: dict = field(default_factory=dict)  # per-sensor columns of the id map, in id order

    def __post_init__(self):
        """Check the groups: names (unique, one key segment), rules, settings, the volume they name."""
        object.__setattr__(self, "_pos", {pv: i for i, pv in enumerate(self.ids)})
        if len(self._pos) != len(self.ids):  # type: ignore[attr-defined]
            dup = next(pv for i, pv in enumerate(self.ids) if self._pos[pv] != i)  # type: ignore[attr-defined]
            raise ValueError(f"the sensor set's id map names {dup!r} twice (each sensor has its own path): convert "
                             "the detector again")  # fmt: skip
        names = [g.name for g in self.groups]
        for g in self.groups:
            if not GROUP_NAME.fullmatch(str(g.name)):
                raise ValueError(f"sensor group {g.name!r}: a name is letters, digits and '_', not starting with a "
                                 "digit (it is a segment of the keys sensors/<group>/<field>)")  # fmt: skip
            if names.count(g.name) > 1:
                raise ValueError(f"two sensor groups are named {g.name!r}")
            if g.rule not in RULES:
                raise ValueError(f"sensor group {g.name!r}: rule {g.rule!r} is not one of {RULES}")
            settings = g.settings or {}
            unknown = sorted(k for k in settings if k not in SETTINGS)
            if unknown:
                raise ValueError(f"sensor group {g.name!r}: unknown settings {unknown} (the settings are "
                                 f"{list(SETTINGS)})")  # fmt: skip
            if g.values is not None and "entry_efficiency" in settings:
                raise ValueError(f"sensor group {g.name!r}: a group with a value set has every absorption entering "
                                 "counted and its value set's efficiency applied after tracking: drop the setting "
                                 "entry_efficiency")  # fmt: skip
            count = settings.get("count")
            if count is not None and (g.rule != "any_entry" or count not in COUNTS):
                raise ValueError(f"sensor group {g.name!r}: the setting count is 'volume' or 'entry', for an "
                                 "any_entry group (a border_into group counts at its surfaces), got "
                                 f"{count!r} on {g.rule}")  # fmt: skip
            if (g.lv is None) == (g.lv_pattern is None):
                raise ValueError(f"sensor group {g.name!r}: give exactly one of lv and lv_pattern")
            missing = [c for c in g.where or {} if self.ids and c not in self.columns]
            if missing:
                raise ValueError(f"sensor group {g.name!r}: selects by the id-map columns {missing}, which the sensor "
                                 f"set does not have (it has {sorted(self.columns)})")  # fmt: skip

    def is_sensor(self, lv: str) -> bool:
        """Return whether the logical volume ``lv`` is a sensor volume: one that a group selects."""
        return any(g.selects(lv) for g in self.groups)

    def in_group(self, group: SensorGroup, pv: str, lv: str) -> bool:
        """Return whether the sensor at placed-volume path ``pv`` (logical volume ``lv``) is in ``group``."""
        if not group.selects(lv):
            return False
        if not group.where:
            return True
        i = self._pos[pv]  # type: ignore[attr-defined]
        return all(self.columns[c][i] == v for c, v in group.where.items())


def _default(f):
    return f.default_factory() if f.default_factory is not MISSING else f.default


def to_json(s: SensorSet) -> tuple[str, str]:
    """Return the sensor set's JSON and its id map's JSON."""
    groups = [{f.name: getattr(g, f.name) for f in fields(g) if getattr(g, f.name) != _default(f)} for g in s.groups]
    return dumps(dict(schema=SCHEMA, groups=groups)), dumps(dict(ids=list(s.ids), **s.columns))


def files(s: SensorSet) -> dict[str, bytes]:
    """Return the sensor set's files by their path in a detector file (``sensors.json`` and its id map)."""
    head, ids = to_json(s)
    return {HEAD: head.encode(), IDS: ids.encode()}


def save(s: SensorSet, root) -> Path:
    """Write the sensor set into ``root`` (beside a detector file's content), each file atomically and the id map
    first, so a reader that finds the set finds its id map."""  # noqa: D205, D209
    head, ids = to_json(s)
    atomic.write_bytes(Path(root) / IDS, ids.encode())
    atomic.write_bytes(Path(root) / HEAD, head.encode())
    return Path(root) / HEAD


def from_json(head: dict, ids, columns: dict | None = None) -> SensorSet:
    """Return the sensor set of its JSON (``to_json``'s first part, parsed), its id map's paths and columns."""
    check_format(head.get("schema"), SCHEMA, "the sensor set", "convert the detector again")
    try:
        groups = [SensorGroup(**g) for g in head["groups"]]
    except TypeError as err:
        raise FormatError(f"the sensor set: {err}: convert the detector again") from None
    return SensorSet(tuple(groups), tuple(ids), dict(columns or {}))


def read(root) -> SensorSet:
    """Read the sensor set of a detector file."""
    root = Path(root)
    head = json.loads((root / HEAD).read_text())
    check_format(head.get("schema"), SCHEMA, root / HEAD, "convert the detector again")
    idmap = json.loads((root / IDS).read_text())
    return from_json(head, idmap.pop("ids"), idmap)
