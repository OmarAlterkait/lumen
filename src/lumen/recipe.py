"""A recipe: what a GDML file cannot say about a detector, applied once, when the detector is converted.

A recipe names the detector volume (a logical volume, or ``auto:<materials>``: the outermost volume of the main bath
of one of those materials), the wavelength range of the optics, the sensors (which volumes, how they count, which
model reads them) and any further decisions, each a function that returns the detector file with its choice
recorded (``origin="recipe"`` and a note). No recipe code runs when a detector file is loaded.

    RECIPE = Recipe(name="sk", detector_volume="WCBarrel", wavelength_range=(300.0, 600.0),
                    sensor_volumes=SensorVolumes(names=r"^InteriorWCPMT"),
                    groups=(SensorGroup("pmt", "border_into", "lumen:geant4", lv_pattern="^InteriorWCPMT"),))
    result = lumen.gdml.import_gdml("sk.gdml", RECIPE)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lumen.format import schema
from lumen.format.sensorsets import SensorGroup, SensorSet, SensorVolumes

Decision = Callable[[schema.DetectorFile, dict], schema.DetectorFile]


@dataclass(frozen=True)
class Recipe:
    """A detector's recipe (see the module docstring).

    ``inputs`` names the extra files a decision reads (e.g. a channel map), given at conversion time;
    ``sensor_columns`` adds per-sensor columns to the id map and ``calibrations`` the data shipped with the file.
    """

    name: str
    detector_volume: str
    wavelength_range: tuple[float, float]
    sensor_volumes: SensorVolumes
    groups: tuple[SensorGroup, ...]
    decisions: tuple[Decision, ...] = ()
    inputs: tuple[str, ...] = ()
    sensor_columns: Callable | None = None
    calibrations: Callable | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)
    material_keys: frozenset | None = None  # the optical material keys (None: all); the others are kept inert

    def apply(self, df: schema.DetectorFile, source: Path, inputs: dict | None = None):
        """Return the ImportResult of a detector file read from ``source`` under this recipe."""
        from lumen.build import lowering, tree
        from lumen.gdml.importer import ImportResult

        inputs = dict(inputs or {})
        missing = [k for k in self.inputs if k not in inputs]
        if missing:
            raise ValueError(f"recipe {self.name!r} needs the inputs {missing}")
        lv = lowering.find_detector(df, self.detector_volume)
        lv = lv if isinstance(lv, str) else lv[0]
        rep = list(df.report) + [f"detector volume: {lv} (recipe: {self.detector_volume})"]
        prov = dict(df.provenance, recipe=self.name, recipe_inputs={k: Path(v).name for k, v in inputs.items()})
        df = dataclasses.replace(df, name=self.name, detector_volume=lv, wavelength_range=self.wavelength_range,
                                 provenance=prov, report=tuple(rep))  # fmt: skip
        for decide in self.decisions:
            df = decide(df, inputs)
        loaded = lowering.lower(
            df, detector=lv, sensors=lowering.Sensors(aux=self.sensor_volumes.aux, names=self.sensor_volumes.names)
        )
        ids = tuple(loaded.sensors[i] for i in sorted(loaded.sensors))
        sset = SensorSet("default", self.sensor_volumes, self.groups, ids)
        columns = self.sensor_columns(df, loaded, tree, inputs) if self.sensor_columns else {}
        cals = self.calibrations(df, sset, inputs) if self.calibrations else {}
        return ImportResult(df, {"default": sset}, cals, df.report, columns)
