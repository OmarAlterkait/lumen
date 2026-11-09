"""The public surface: convert a GDML file into a detector file, and load a detector.

    path = lumen.convert("sk.gdml", lumen.detectors.wcsim.SK, "sk.lumen")     # once
    det = lumen.load("sk.lumen")                                             # or a registered name: "sk"
    p = det.params()                                                         # every parameter, by GDML name
    arrays = det.pack(p)                                                     # the engine's packed arrays

Loading reads the detector file, cuts out the detector volume, lowers the geometry to its scene, builds the
volume tree, the surfaces and media, and resolves the parameters (the detector file's values, the sensor models'
defaults, then the calibration chain). No recipe code runs at load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lumen import _env
from lumen.format import calibrations as cal
from lumen.format import io, sensorsets
from lumen.format import schema as fmt


@dataclass(frozen=True, eq=False)
class Detector:
    """A loaded detector: its file, sensor set, the built parts (scene, volume tree, surfaces and media), parameters.

    ``sensors`` holds the id map: the sensors' placed-volume paths in id order, and its per-sensor columns.
    """

    name: str
    path: Path
    file: fmt.DetectorFile
    sensor_set: sensorsets.SensorSet
    parts: object
    sensors: dict
    calibration: str = cal.NOMINAL
    _params: object = None

    @property
    def scene(self):
        """Return the lowered scene (the detector volume cut out)."""
        return self.parts.scene

    @property
    def tree(self):
        """Return the volume tree."""
        return self.parts.tree

    def params(self):
        """Return the resolved parameters (``lumen.Params``)."""
        return self._params

    def pack(self, params=None):
        """Return the engine's packed parameter arrays for ``params`` (default: this detector's)."""
        from lumen.build import pack

        return pack.pack(self, self._params if params is None else params)

    def __repr__(self):
        """Summarise the detector: name, detector volume, counts of volumes, materials, surfaces and sensors."""
        f = self.file
        return (f"{self.name}  (from {f.provenance.get('source', {}).get('file', '?')})\n"
                f"  detector volume  {f.detector_volume}; {len(f.volumes)} volumes, {len(f.materials)} materials, "
                f"{len(f.surfaces)} optical surfaces\n"
                f"  sensors          {len(self.sensor_set.ids)} in {len(self.sensor_set.groups)} group(s): "
                f"{', '.join(g.name for g in self.sensor_set.groups)}\n"
                f"  calibration      {self.calibration}")  # fmt: skip


def convert(source, recipe, out, *, inputs: dict | None = None, overwrite: bool = False) -> Path:
    """Convert a GDML file under a recipe into a detector file at ``out``; return its path."""
    from lumen.gdml.importer import import_gdml

    result = import_gdml(source, recipe, inputs)
    path = io.save(result.detector_file, out, overwrite=overwrite)
    for sset in result.sensor_sets.values():
        sensorsets.save(sset, path)
        if result.sensor_columns:
            ids = path / "sensors" / f"{sset.name}.ids.json"
            ids.write_text(io.dumps(dict(ids=list(sset.ids), **result.sensor_columns)))
    if result.calibrations:
        d = path / "data"
        d.mkdir(exist_ok=True)
        for name, data in result.calibrations.items():
            (d / f"{name}.json").write_text(io.dumps(data))
    return path


def _resolve(name_or_path) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    for d in _env.detector_path():
        cand = d / f"{name_or_path}.lumen"
        if cand.exists():
            return cand
    cand = _env.cache_dir() / "detectors" / f"{name_or_path}.lumen"
    if cand.exists():
        return cand
    return _convert_registered(str(name_or_path), cand)


def _convert_registered(name: str, out: Path) -> Path:
    from lumen import detectors, plugins

    detectors.register_builtins()
    pkg = plugins.detectors.get(name)
    data = _env.data_dir()
    if data is None:
        raise FileNotFoundError(f"{name!r}: no detector file found, and LUMEN_DATA is not set to convert it from "
                                f"its dataset {pkg.source_dataset!r}")  # fmt: skip
    inputs = {k: data / v for k, v in pkg.inputs.items()}
    return convert(data / pkg.source_dataset, pkg.recipe, out, inputs=inputs)


def load(name_or_path, *, sensors: str = "default", calibration: str = cal.NOMINAL) -> Detector:
    """Load a detector by path or registered name, with a sensor set and a calibration (see the module docstring)."""
    from lumen.build import build

    path = _resolve(name_or_path)
    df = io.read(path)
    sset = sensorsets.read(path, sensors)
    columns = json.loads((path / "sensors" / f"{sensors}.ids.json").read_text())
    search = [path / "calibrations", *_env.detector_path()]
    chain = cal.chain(calibration, search) if calibration != cal.NOMINAL else []
    parts = build.build(df, sset)
    params = build.parameters(df, sset, parts, chain)
    return Detector(df.name, path, df, sset, parts, columns, calibration, params)


__all__ = ["Detector", "convert", "load"]
