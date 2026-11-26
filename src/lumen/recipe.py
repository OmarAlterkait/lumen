"""A recipe: what a GDML file cannot say about a detector, applied once, when the detector is converted.

A recipe names the detector volume (a logical volume, or ``auto:<materials>``: the outermost volume of the main bath of
one of those materials), the sensor groups (which volumes, how they count, which value set reads them) and any
further decisions, each a function that returns the detector file
with its choice recorded (``origin="recipe"`` and a note); the general import rules follow
(``lumen.build.import_rules``). No recipe code runs when a detector file is loaded.

    RECIPE = Recipe(name="my_detector", detector_volume="Tank",
                    groups=(SensorGroup("pmt", rule="border_into", lv_pattern="^PMT"),))
    path = lumen.convert("my_detector.gdml", recipe=RECIPE, out="my_detector.lumen")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

from lumen.format import schema
from lumen.format.check import check
from lumen.format.io import sha256_file
from lumen.format.result import ImportResult
from lumen.format.sensorsets import SensorGroup, SensorSet

Decision = Callable[[schema.DetectorFile, dict], schema.DetectorFile]


@dataclass(frozen=True)
class Recipe:
    """A detector's recipe (see the module docstring).

    ``inputs`` names the extra files a decision reads (e.g. a channel map), given at conversion time and recorded in
    the provenance by name and sha256; ``sensor_columns(df, sensor_paths, inputs) -> {column: [value per sensor]}``
    adds per-sensor columns to the id map (``sensor_paths``: the sensors' placed-volume paths in id order).
    ``precision`` is the navigator precision the detector needs, "float32" (the default) or "float64" for a detector
    whose layers are thinner than float32's band (``lumen.load``, which uses it when the caller passes none).
    A GDML that Geant4 refuses is repaired apart and the repaired copy converted (``python -m lumen.gdml.repair``).
    """

    name: str
    detector_volume: str
    _: KW_ONLY
    groups: tuple[SensorGroup, ...]
    decisions: tuple[Decision, ...] = ()
    inputs: tuple[str, ...] = ()
    sensor_columns: Callable | None = None
    material_keys: frozenset | None = None  # the material keys to read (None: the Geant4 optical keys); others inert
    precision: str = "float32"  # the navigator precision this detector needs ("float64" for sub-band layers)

    def apply(self, df: schema.DetectorFile, inputs: dict | None = None):
        """Return the ImportResult of a detector file read from its source under this recipe.

        The recipe's choices (its name, detector volume, inputs and decisions), then the structure is finished as every
        source's is (``import_structure``: the import rules, the checks, the sensor ids and the id map's columns). The
        build is needed here, never at load: the ``auto:`` detector volume is found in the lowered geometry, the import
        rules are build rules, and the sensor ids are numbered in the lowered geometry's canonical walk.
        """
        from lumen.build import lowering
        from lumen.rules import navigation

        inputs = dict(inputs or {})
        missing = [k for k in self.inputs if k not in inputs]
        if missing:
            raise ValueError(f"recipe {self.name!r} needs the inputs {missing}")
        found = lowering.find_detector(df, self.detector_volume)
        # stored: the LV (auto resolved), or the physvol name or path that selects its placement
        spec = self.detector_volume
        lv = found.lv if spec.startswith("auto:") or spec == found.lv else spec
        rep = list(df.report) + [f"detector volume: {found.lv} (recipe: {spec})"]
        prov = dict(df.provenance, recipe=self.name, recipe_inputs={k: _input_record(k, v) for k, v in inputs.items()})
        df = dataclasses.replace(df, name=self.name, detector_volume=lv, provenance=prov, report=tuple(rep),
                                 precision=navigation.precision(self.precision).name)  # fmt: skip
        for decide in self.decisions:
            df = decide(df, inputs)
        cols = self.sensor_columns
        columns = (lambda d, paths: cols(d, paths, inputs)) if cols else None
        return import_structure(df, SensorSet(self.groups), columns)


def _with_values(groups) -> tuple:
    """Return the groups with each value set a group names copied in (``SensorGroup.values``), read once here."""
    from lumen.sensors import read_values

    return tuple(dataclasses.replace(g, values=read_values(g.values)) if isinstance(g.values, str) else g
                 for g in groups)  # fmt: skip


def _input_record(name: str, path) -> dict:
    """Return a recipe input's provenance: its file name and sha256 (an input is one file)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"recipe input {name!r}: {p} is not a file (an input is one file, recorded by its sha256)")
    return dict(file=p.name, sha256=sha256_file(p))


def import_yaml(*paths) -> ImportResult:
    """Return the detector file and sensor set of a detector written in YAML (files, or directories of them).

    The YAML holds the structure and its decisions (``lumen.format.yaml``): no recipe is needed. It is finished as
    every structure is (``import_structure``). The provenance names the YAML files and their sha256.
    """
    from lumen.format import yaml as yaml_format

    files = yaml_format.sources(paths)
    df, sset = yaml_format.read(*files)
    sha = [sha256_file(f) for f in files]
    prov = dict(importer="lumen.recipe.import_yaml", source=dict(file=", ".join(f.name for f in files), sha256=sha))
    rep = (f"read from YAML: {', '.join(f.name for f in files)}",)
    return import_structure(dataclasses.replace(df, provenance=prov, report=rep), sset)


def import_structure(df: schema.DetectorFile, sensor_set: SensorSet | None = None,
                     sensor_columns: Callable | None = None) -> ImportResult:  # fmt: skip
    """Return a detector structure finished as every source is: the conversion rules applied
    (``lumen.build.import_rules``: e.g. the Geant4-10 scintillation keys a recipe's decision added), checked
    (``lumen.format.check``) and its sensor set numbered.

    The sensor ids are numbered in the canonical walk; with no sensor set, the detector has an empty one.
    ``sensor_columns(df, sensor_paths)`` gives the id-map columns from the sensors' paths (a recipe's; None: the
    set's own). A recipe (``Recipe.apply``) and the YAML reader finish here. The detector volume is closed as Geant4
    transports light (``lumen.build.binding.detector_volume_closed``): a recipe's that light leaves into a volume with
    RINDEX grows to it, each step in the report.
    """  # noqa: D205
    from lumen.build import binding, import_rules, lowering

    df = import_rules.apply(df)
    check(df)
    closed, grown = binding.detector_volume_closed(df)
    if grown:
        df = dataclasses.replace(df, detector_volume=closed, report=tuple(df.report) + grown)
    sset = sensor_set or SensorSet(())
    loaded = lowering.lower(df, detector=df.detector_volume, sensors=sset.is_sensor)
    ids = tuple(loaded.sensors[i] for i in sorted(loaded.sensors))
    columns = sensor_columns(df, ids) if sensor_columns else sset.columns
    return ImportResult(df, SensorSet(_with_values(sset.groups), ids, columns))  # checks the groups' columns
