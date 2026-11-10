"""The public surface: convert a GDML file into a detector file, and load a detector.

    from lumen.detectors import wcsim
    path = lumen.convert("sk.gdml", wcsim.SK, "sk.lumen")                    # once
    det = lumen.load("sk.lumen")                                             # or a registered name: "sk"
    p = det.params()                                                         # every parameter, by GDML name
    arrays = det.pack(p)                                                     # the engine's packed arrays

Loading reads the detector file, cuts out the detector volume, lowers the geometry to its scene, builds the
volume tree, the surfaces and media, and resolves the parameters (the detector file's values, the sensor models'
defaults, then the calibration and its parents). No recipe code runs at load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
    parents = cal.lineage(calibration, search) if calibration != cal.NOMINAL else []
    parts = build.build(df, sset)
    params = build.parameters(df, sset, parts, parents)
    return Detector(df.name, path, df, sset, parts, columns, calibration, params)


# ------------------------------------------------------------------------------------------------------ simulate
EXACT_CELL_FACTOR = 5  # device bytes per histogram byte: the GPU's exact sums keep a u64 sum and a u64 scratch per cell
_MODELS: dict = {}  # (id(det), hit angle, moving source, origins) -> (det, Model): one compile per detector and layout
SOURCE_STREAM = np.uint32(0xFFFFFFFF)  # the source's key: fold_in(key, SOURCE_STREAM), never a photon's id (< 2^31)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Light:
    """What ``simulate`` returns: per sensor the expected photon arrivals at the sensor (before its model), their
    Monte Carlo error (over the replicas), the mean arrival time (ns), the group's histograms (``lumen.readout``) and
    the run's health; ``provenance`` (static) says how it was made. Readout functions take ``(light, params)``."""  # noqa: D205, D209

    arrivals: Any
    arrivals_error: Any
    mean_arrival_time: Any
    histograms: Any
    health: Any
    provenance: tuple = ()

    def tree_flatten(self):
        """Return the arrays and the provenance."""
        return (self.arrivals, self.arrivals_error, self.mean_arrival_time, self.histograms, self.health), \
            self.provenance  # fmt: skip

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from the arrays and the provenance."""
        return cls(*children, aux)


def _sensor_group(det):
    from lumen import plugins

    groups = [g for g in det.sensor_set.groups if g.model in plugins.sensor_models.names()]
    if len(groups) != 1:
        raise ValueError(f"simulate: one sensor group with a sensor model (lumen.sensors) is supported, the sensor "
                         f"set has {[g.name for g in groups]}")  # fmt: skip
    return groups[0], plugins.sensor_models.get(groups[0].model)(groups[0].defaults)


def _model(det, hit_angle: bool, moving=False, origins=None):
    from lumen.geometry import clearance as ray_clear
    from lumen.geometry import tracked
    from lumen.transport import step as steps

    k = (id(det), hit_angle, moving, None if origins is None else origins.tobytes())
    if k not in _MODELS:
        geom = tracked.from_parts(det.parts, cache=True)
        field = None if origins is None else ray_clear.Clearance(origins, ray_clear.ray_clearance(geom.nav, origins))
        _MODELS[k] = (det, steps.Model(geom, det.parts.optics, steps.StepConfig(moving_source=moving), field,
                                       hit_angle=hit_angle))  # fmt: skip
    return _MODELS[k][1]


def _moving(source, params, wrt) -> tuple:
    """Return (the moving-source method is on, the source's origins (m) for its clearance, None when traced)."""
    from lumen import sources

    if any(not k.startswith(sources.PREFIX) for k in wrt):
        raise ValueError(f"simulate: wrt takes the source's keys ({sources.PREFIX}<field>), got {list(wrt)}")
    moved = [k for k in wrt if k.removeprefix(sources.PREFIX) in getattr(source, "MOVING", ())]
    if not moved:
        return False, None
    if not source.drawn_direction:
        raise sources.GradientNotAvailable(f"simulate: {moved} move the photons of a source whose directions are "
                                           "not drawn from a density (drawn_direction False)")  # fmt: skip
    v = source.values(params)
    try:
        pts = [np.asarray(v[f], np.float32).reshape(-1, 3) for f in ("x", "start", "end") if f in v]
    except jax.errors.TracerArrayConversionError:  # traced origins: no co-moving radius (exact, without the field)
        return True, None
    return True, np.unique(np.concatenate(pts), axis=0)


def _budget() -> float:
    try:
        limit = float(jax.devices()[0].memory_stats()["bytes_limit"])
    except Exception:  # a device without memory statistics (the CPU): no device budget
        return float("inf")
    return limit / EXACT_CELL_FACTOR


def simulate(det, source, params=None, *, samples: int, seed: int, replicas: int = 16, gate=None,  # noqa: PLR0913
             chunk: int = 1 << 20, backend: str = "auto", wrt=()) -> Light:  # fmt: skip
    """Return the Light of ``samples`` photons of ``source`` through ``det`` at ``params`` (default: its own).

    ``source``: a ``lumen.sources`` source, its fields from ``params``' ``source/`` keys where given (its random numbers
    under ``fold_in(key(seed), SOURCE_STREAM)``), or an Emission batch of at least ``samples`` rows, or a function
    ``(first_id, count) -> Emission``. ``wrt``: the source keys a derivative is taken in; one that moves the photons (a
    position) turns on the moving-source method, with the clearance about the source's origins (none when they are
    traced), and needs a source whose directions are drawn. The photons run in chunks of ``chunk``
    (``lumen.readout.photon_chunks``), photon i under ``fold_in(key(seed), i)``. The histograms are those of the sensor
    set's group with a sensor model (``lumen:pmt``), on its static grid (``lumen.readout.make_grid`` with ``gate`` (ns),
    default the group's ``readout["gate"]``); ``simulate`` refuses before launch (MemoryError) when they exceed the
    device's budget.
    """
    from lumen import readout as ro
    from lumen import run as runner
    from lumen.build import build
    from lumen.transport import item
    from lumen.transport import step as steps

    group, sensor_model = _sensor_group(det)
    gate = tuple(float(x) for x in (gate or (group.readout or {}).get("gate", (0.0, 0.0))))
    grid = ro.make_grid(sensor_model, gate)
    n = len(det.parts.optics.sensor_pvs)
    if build.group_tags(group, det.parts) != list(range(n)):
        raise ValueError(f"simulate: sensor group {group.name!r} must contain every sensor of the detector")
    r = item.HistogramLayout(n, grid, int(replicas))
    ro.check_memory(grid, n, int(replicas), 0, _budget())
    params = det.params() if params is None else params
    moving, origins = _moving(source, params, wrt) if hasattr(source, "emit") else (False, None)
    model = _model(det, grid.theta_knots is not None, moving, origins)
    packed = jax.tree_util.tree_map(jnp.asarray, det.pack(params))
    ops = steps.operands(model, packed, None, steps.Controls(), None)
    key = jax.random.key_data(jax.random.PRNGKey(int(seed)))
    if hasattr(source, "emit"):
        src, skey = source, jax.random.fold_in(key, SOURCE_STREAM)
        source = lambda first, count: src.emit(params, skey, count, first, int(samples))  # noqa: E731
    size, acc = min(int(chunk), int(samples)), None
    for first, count in ro.photon_chunks(int(samples), size):
        rows = source(first, count) if callable(source) else \
            jax.tree_util.tree_map(lambda a, f=first, c=count: a[f:f + c], source)  # fmt: skip
        if count < size:  # the last chunk keeps the shape: padded, its extra rows inactive
            pad = lambda a, c=count: jnp.concatenate([a, jnp.repeat(a[-1:], size - c, 0)])  # noqa: E731
            rows = jax.tree_util.tree_map(pad, rows)
        t = runner.run(item.readout(r), model, ops, rows, key, replicas=int(replicas), first_id=first, backend=backend,
                       n_active=count)  # fmt: skip
        out = {k: jnp.asarray(v, jnp.float32) for k, v in t.outputs.items()}
        acc = out if acc is None else {k: acc[k] + v for k, v in out.items()}
    per = acc["arrival"].reshape(int(replicas), n)
    arrivals = per.sum(0)
    err = jnp.sqrt(replicas / max(replicas - 1, 1) * jnp.sum((per - per.mean(0)) ** 2, 0))
    mean_t = jnp.where(arrivals > 0.0, acc["arrival_t"] / jnp.where(arrivals > 0.0, arrivals, 1.0), jnp.nan)
    health = {k: acc[k] for k in ("lost", "abandoned", "escaped", "retired", "truncated", "n_truncated", "stuck")}
    prov = (("seed", int(seed)), ("samples", int(samples)), ("replicas", int(replicas)), ("chunk", size),
            ("backend", backend), ("detector", det.name), ("calibration", det.calibration),
            ("wrt", tuple(wrt)))  # fmt: skip
    hist = item.histograms(r, acc, sensor_model, group.name, gate)
    return Light(arrivals, err, mean_t, hist, health, prov)


__all__ = ["Detector", "Light", "convert", "load", "simulate"]
