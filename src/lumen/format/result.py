"""What an importer returns: the detector file and its sensor set.

The GDML importer (``lumen.gdml.importer.import_gdml``), ``Recipe.apply`` and ``lumen.recipe.import_yaml`` return one;
``lumen.convert`` saves it as a ``.lumen`` directory. It lives in the format, so a recipe needs nothing of an importer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lumen.format import schema
from lumen.format.sensorsets import SensorSet


@dataclass(frozen=True)
class ImportResult:
    """What an importer returns: the detector file (its report in ``detector_file.report``) and its sensor set."""

    detector_file: schema.DetectorFile
    sensor_set: SensorSet = field(default_factory=lambda: SensorSet(()))
