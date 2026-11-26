"""The public API: ``convert``, ``load``, ``simulate``, ``Detector`` and ``Light``.

    path = lumen.convert("tank.gdml", recipe, "tank.lumen")                 # once (recipe: lumen.Recipe)
    det = lumen.load("tank.lumen")                                           # a path
    p = det.params()                                                         # every parameter, by GDML name
    src = lumen.sources.Point(position=(0.0, 0.0, 0.0), lam=400.0, n_photons=1e6)
    light = lumen.simulate(det, src, p, samples=2**20, seed=0)               # arrivals per sensor, and errors

Loading reads the detector file, cuts out the detector volume, lowers the geometry to its scene, builds the
volume tree, the surfaces and media, and resolves the parameters (the detector file's values, the sensor models'
defaults, then the calibration and its parents). No recipe code runs at load.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import json
import math
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lumen import _cache, _env, sources
from lumen import speed as spd
from lumen._core.errors import refusing
from lumen._core.rng import MAX_PHOTONS, SOURCE_STREAM
from lumen._core.safe import traced
from lumen.format import calibrations as cal
from lumen.format import io, sensorsets
from lumen.format import schema as fmt
from lumen.models import MediumModel
from lumen.params import Params
from lumen.readout import grid as ro
from lumen.readout import truth as tr
from lumen.sources._base import bucket
from lumen.transport import estimator

try:  # the error of a custom_jvp differentiated through a value it closes over
    from jax._src.custom_derivatives import CustomJVPException
except ImportError:  # pragma: no cover
    CustomJVPException = jax.errors.UnexpectedTracerError


@dataclass(frozen=True, eq=False)
class Detector:
    """A loaded detector: its file, sensor set, the built parts (scene, volume tree, surfaces and media), parameters.

    ``sensor_table()`` lists the sensors, in the order of ``Light.arrivals``, with the id map's columns. ``precision``:
    the navigator's, "float32" or "float64" (``lumen.load``).
    """

    name: str
    path: Path
    file: fmt.DetectorFile
    sensor_set: sensorsets.SensorSet
    parts: object
    _columns: dict
    calibration: str = cal.NOMINAL
    _params: object = None
    precision: str = "float32"

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

    @property
    def digest(self) -> str:
        """Return the detector's digest: its file's content and its sensor set (groups, value sets, id map); a
        calibration records it as ``detector_digest``."""  # noqa: D205, D209
        return io.content_digest(io.to_files(self.file), sensorsets.files(self.sensor_set))

    def save_calibration(self, params, name: str, directory=None, *, note: str = "") -> Path:
        """Write the values of ``params`` that differ from the detector file's own as calibration ``name``; return its
        path.

        A calibration is flat: the values this detector's calibration set (when it was loaded with one) are written
        with the changes, so it loads on the detector file alone. Written to ``directory`` (default: the first
        directory of the detector search path, ``LUMEN_DETECTORS``), never inside the detector file; it records this
        file's digest, checked when it is loaded (``lumen.load(..., calibration=name)``).
        """  # noqa: D205
        base = self._params
        layer = f"calibration {self.calibration}"
        changed = {k: params[k] for k in params if base.explain(k) == layer or not _same(params[k], base[k])}
        d = Path(directory) if directory is not None else next(iter(_env.detector_path()), None)
        if d is None:
            raise ValueError("save_calibration: give a directory (LUMEN_DETECTORS is not set)")
        return cal.save(cal.Calibration(name, self.name, self.digest, changed, note), d)

    def with_responses(self, *responses):
        """Return this detector with the parameters of sensor responses (``lumen.response``) added at their defaults.

        Each response's keys ``sensors/<group>/<name>`` join ``params()``; a key already there keeps its value. A
        calibration that sets them is loaded with the responses: ``lumen.load(..., responses=)``.
        """
        from lumen.build import build

        return dataclasses.replace(self, _params=build.add_responses(self._params, self.sensor_set, responses))

    def pack(self, params=None):
        """Return the engine's packed parameter arrays for ``params`` (default: this detector's), its values checked
        (``lumen.build.values``: a concrete value outside what the detector holds raises ``InvalidValue``)."""  # noqa: D205, D209
        from lumen.build import pack, values

        params = self._params if params is None else params
        return pack.pack(self, values.check(self, params)[0])

    def sensor_table(self) -> SensorTable:
        """Return the sensors in the order of ``Light.arrivals`` (``SensorTable``: placed volume, sensor entry, world
        position (m), axis, and the id map's columns)."""  # noqa: D205, D209
        o = self.parts.optics
        pose = {x.pv: x for x in self.parts.instances}
        pvs = list(o.sensor_pvs)
        at = {pv: i for i, pv in enumerate(self._columns["ids"])}
        cols = {k: [v[at[pv]] for pv in pvs] for k, v in self._columns.items() if k != "ids"}
        return SensorTable(pvs, list(o.sensor_entries), np.array([pose[p].t for p in pvs], np.float64),
                           np.array([np.asarray(pose[p].R)[:, 2] for p in pvs], np.float64), cols)  # fmt: skip

    def __repr__(self):
        """Summarise the detector: name, detector volume, counts of volumes, materials, surfaces and sensors."""
        f = self.file
        return (f"{self.name}  (from {f.provenance.get('source', {}).get('file', '?')})\n"
                f"  detector volume  {f.detector_volume}; {len(f.volumes)} volumes, {len(f.materials)} materials, "
                f"{len(f.surfaces)} optical surfaces\n"
                f"  sensors          {len(self.sensor_set.ids)} in {len(self.sensor_set.groups)} group(s): "
                f"{', '.join(g.name for g in self.sensor_set.groups)}\n"
                f"  calibration      {self.calibration}\n"
                f"  precision        {self.precision}")  # fmt: skip


def _same(a, b) -> bool:
    if isinstance(a, cal.Table) or isinstance(b, cal.Table):
        return isinstance(a, cal.Table) and isinstance(b, cal.Table) and _same(a.energy, b.energy) and \
            _same(a.value, b.value)  # fmt: skip
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def convert(source, recipe=None, out=None, *, data=None, inputs: dict | None = None,
            overwrite: bool = False) -> Path:  # fmt: skip
    """Convert a detector into a detector file at ``out``; return its path.

    ``source`` is one of:

    - a GDML file, with its ``recipe`` (``lumen.Recipe``: what a GDML file cannot say) and the recipe's ``inputs``;
    - a detector written in YAML (a ``.yaml`` file, a directory of them, or a list of either; ``lumen.format.yaml``);
    - a detector package's detector, ``"package:name"`` (``lumen.convert("lumen_wc:sk", out=...)``): the package's
      ``DETECTORS[name]`` (``lumen.DetectorEntry``: its recipe, its GDML and the recipe's inputs, relative paths under
      ``data``, default ``LUMEN_DATA``, else ``data`` under lumen's cache root). A package that ships or checks files
      defines ``prepare(recipe, gdml, inputs)``, a context manager yielding the three as converted (a shipped
      compressed GDML decompressed and its sha256 checked, a recipe made from its checked inputs): lumen's one pipeline,
      ``prepare, expected = lumen.inputs.package(REFERENCES, GDML_DIR)``.

    YAML carries its own decisions: pass ``recipe=None``. Their structure is finished as every source is:
    the general import rules, the structural checks, the sensor ids (``lumen.recipe.import_structure``).

    The detector file and its sensor set are written under a temporary name and renamed into place whole
    (``lumen._cache.directory``): a reader never sees part of one.
    """
    if out is None:
        raise TypeError("convert: out, the detector file's path, is required")
    out = Path(out)
    if out.exists() and not overwrite:
        raise io.FormatError(f"{out} exists: a detector file is never overwritten (pass overwrite=True)")
    mode = "replace" if overwrite else "raise"
    if is_package_detector(source):
        if recipe is not None or inputs:
            raise ValueError(f"convert: {source!r} names its own recipe and inputs (the package's DETECTORS)")
        with package_detector(source, data=data) as entry:
            return _place(_import(entry.gdml, entry.recipe, entry.inputs), out, mode)
    if data is not None:
        raise ValueError('convert: data is the data directory of a detector package\'s detector ("package:name")')
    return _place(_import(source, recipe, inputs), out, mode)


def is_package_detector(source) -> bool:
    """Return whether ``source`` names a detector package's detector, ``"package:name"`` (not an existing path)."""
    import re

    return (isinstance(source, str) and re.fullmatch(r"[A-Za-z_][\w.]*:[\w.-]+", source) is not None
            and not Path(source).exists())  # fmt: skip


@dataclass(frozen=True)
class DetectorEntry:
    """A detector package's detector (an entry of its ``DETECTORS``, ``lumen.convert("package:name")``).

    ``recipe``: a ``lumen.Recipe``, or a function ``(gdml, inputs) -> Recipe`` (a recipe made from its inputs, by the
    package's ``prepare``); ``gdml``: the GDML file's path; ``inputs``: {name: path} the recipe reads. A relative path
    is under the data directory (``convert``'s ``data``, else ``LUMEN_DATA``, else ``data`` under lumen's cache).
    A package may still give an entry as the tuple (recipe, gdml, inputs).
    """

    recipe: Any
    gdml: str | Path
    inputs: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def of(cls, value) -> DetectorEntry:
        """Return an entry, or the entry of a (recipe, gdml, inputs) tuple."""
        return value if isinstance(value, cls) else cls(*value)


@contextlib.contextmanager
def package_detector(source: str, *, data=None):
    """Yield the ``DetectorEntry`` of a detector package's detector, ``"package:name"``, as ``convert`` converts it.

    The package's ``DETECTORS[name]``, its relative paths under ``data`` (else ``LUMEN_DATA``, else ``data`` under
    lumen's cache root), passed through the package's ``prepare`` when it defines one: the entry's recipe is then a
    ``lumen.Recipe``, and its GDML file and inputs are the files converted (a shipped compressed GDML decompressed and
    its sha256 checked), which exist while the context is open.

        with lumen.api.package_detector("lumen_wc:sk") as entry:
            print(entry.gdml, entry.recipe.name)
    """
    import importlib

    pkg, name = source.split(":", 1)
    module = importlib.import_module(pkg)
    detectors = getattr(module, "DETECTORS", None)
    if not isinstance(detectors, dict) or name not in detectors:
        raise ValueError(f"convert: {pkg} has no detector {name!r} (its DETECTORS: {sorted(detectors or {})})")
    e = DetectorEntry.of(detectors[name])
    root = Path(data) if data is not None else _env.data_dir()
    prepare = getattr(module, "prepare", None) or _as_given
    with prepare(e.recipe, root / e.gdml, {k: root / v for k, v in e.inputs.items()}) as (recipe, gdml, inputs):
        yield DetectorEntry(recipe=recipe, gdml=gdml, inputs=inputs)


@contextlib.contextmanager
def _as_given(recipe, gdml, inputs):
    yield recipe, gdml, inputs


def _convert_once(source, recipe, out: Path, inputs: dict | None = None) -> Path:
    """Convert into a cache entry keyed by its content: of processes converting it at once, the first to finish
    places it and the others use it (``lumen._cache.directory``, ``exists="keep"``)."""  # noqa: D205, D209
    if not out.is_dir():
        _place(_import(source, recipe, inputs), out, "keep")
    return out


def _import(source, recipe, inputs):
    """Return the import result of a source (``convert``), with the placement check's lines in its report."""
    if not _is_gdml(source) and recipe is not None:
        raise ValueError("convert: only a GDML file takes a recipe; a YAML detector holds its own decisions: pass "
                         "recipe=None")  # fmt: skip
    if _is_yaml(source):
        from lumen.recipe import import_yaml

        result = import_yaml(*(source if isinstance(source, list) else [source]))
    elif _is_gdml(source):
        from lumen.gdml.importer import import_gdml

        if recipe is None:
            raise ValueError("convert: a GDML file needs its recipe (lumen.Recipe)")
        result = import_gdml(source, recipe, inputs)
    else:
        raise TypeError(f"convert: a GDML file or a YAML detector, not {type(source).__name__}")
    return _placement_report(result)


def _place(result, out: Path, exists: str) -> Path:
    """Write an import result's detector file and sensor set as one directory placed whole at ``out``."""
    with _cache.directory(out, exists=exists) as tmp:
        io.write_into(result.detector_file, tmp)
        sensorsets.save(result.sensor_set, tmp)  # with its id map's columns
    return out


def _placement_report(result):
    """Return the import result with the placement check's lines in its report (``lumen.build.placements``).

    Geant4's overlap precondition, tested on the detector volume as simulated: overlapping siblings (with the owner of
    each shared region), protruding daughters, meshes through a volume are recorded and warned about
    (``PlacementWarning``), never changed; a contact is no overlap.
    """
    import warnings

    from lumen.build import placements

    df = result.detector_file
    lines, found = placements.conversion(df, result.sensor_set)
    if found:
        warnings.warn(
            f"{df.name}: " + "; ".join(lines[1:]) + ": recorded in the detector file's report",
            placements.PlacementWarning,
            stacklevel=4,  # the caller of convert
        )
    df = dataclasses.replace(df, report=(*df.report, *lines))
    return dataclasses.replace(result, detector_file=df)


def _is_gdml(source) -> bool:
    return isinstance(source, (str, Path)) and not _is_yaml(source)


def _is_yaml(source) -> bool:
    if not isinstance(source, (str, Path, list)):
        return False
    paths = [Path(p) for p in (source if isinstance(source, list) else [source])]
    return all(p.suffix in (".yaml", ".yml") or p.is_dir() for p in paths)


def _check_x64(what: str, precision: str):
    """Refuse a float64 navigator while JAX's 64-bit mode is off: its arrays would be truncated to float32 with float64
    tolerances."""  # noqa: D205, D209
    if precision == "float64" and not jax.config.jax_enable_x64:
        raise ValueError(f'{what}: precision="float64" needs JAX\'s 64-bit mode, which is off: set JAX_ENABLE_X64=1 '
                         'before starting Python, or call jax.config.update("jax_enable_x64", True) before any JAX '
                         "array is made")  # fmt: skip


class PrecisionWarning(UserWarning):
    """A detector is loaded at another precision than its recipe asks for (``load``)."""


@functools.cache  # once per detector and precision
def _warn_precision(name: str, want: str, got: str) -> None:
    warnings.warn(f"{name}: its recipe asks for precision {want!r} and it is loaded as {got!r}: the layers it needs "
                  f"{want} for are not tracked (docs/explanation/known-limits.md)",
                  PrecisionWarning, stacklevel=3)  # fmt: skip


def load(path, *, calibration: str = cal.NOMINAL, parts: dict | None = None, responses=(),
         precision: str | None = None) -> Detector:  # fmt: skip
    """Load the detector file at ``path`` (a ``.lumen`` directory) with a calibration.

    ``calibration`` "nominal" (the default) is the detector file's own values; another is a calibration's name, found
    beside the detector file (``<path>/calibrations``) or on the detector search path, its digest checked.
    ``parts``: the replaceable parts bound by their path, {"materials/<M>": a medium model (a value,
    ``lumen.medium_model``) or the name of one of lumen's own; "materials/<M>/<channel>": a phase function for that
    channel of Geant4's reading (``lumen.media.phase_function``), its keys ``materials/<M>/<channel>/<name>``}: the
    materials read by those models instead of the file's (``lumen.models``). The build's model table holds the values;
    the loaded detector's file carries their names (never code: a file naming a part not passed here is refused), so
    its digest (and a calibration saved from it) records the choice. Which parts are hooks: docs/howto/extending.md.
    ``responses``: sensor responses (``lumen.response``) whose parameters join before the calibration is applied, so a
    calibration saved after ``with_responses`` loads.
    ``precision``: the navigator's, "float32" or "float64": its coordinates, frames, roots and intervals
    and the photon's position in that precision, the physics in float32 (``lumen.rules.navigation``). float64 tracks
    volumes thinner than float32's band (a coating of a few um at metres from the origin) for every photon, at about
    a quarter more time on a GPU with fast float64 (an A100), several times more on one without; it needs JAX's 64-bit
    mode, on before any array is made (``JAX_ENABLE_X64=1``, or ``jax.config.update("jax_enable_x64", True)``).
    Left out, the detector's own precision is used: the one its recipe states (``lumen.recipe.Recipe``), stored in the
    detector file, else "float32". Passing it wins over the file's, and loading a detector whose recipe asks for
    float64 in float32 warns (``PrecisionWarning``), since that is the detector the recipe says lumen gets wrong.
    """
    from lumen.build import build
    from lumen.build import values as value_checks
    from lumen.rules import navigation

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"no detector file at {path} (a detector is its file: convert it first, lumen.convert)")
    models = _part_models(parts or {})
    df = _with_models(io.read(path), models)
    prec = navigation.precision(df.precision if precision is None else precision)
    if precision is not None and df.precision == "float64" and prec.name != "float64":
        _warn_precision(df.name, df.precision, prec.name)
    _check_x64("load", prec.name)
    values = [m for m in models.values() if isinstance(m, MediumModel)]
    sset = sensorsets.read(path)
    columns = json.loads((path / sensorsets.IDS).read_text())
    c = None
    if calibration != cal.NOMINAL:
        c = cal.read(cal.find(calibration, [path / "calibrations", *_env.detector_path()]))
        cal.check_digest(c, io.content_digest(io.to_files(df), sensorsets.files(sset)), path)
    parts = build.build(df, sset, values)
    params = build.parameters(df, sset, parts, c, tuple(responses))
    det = Detector(df.name, path, df, sset, parts, columns, calibration, params, prec.name)
    value_checks.check(det, params)  # the values every run checks, at the file's and the calibration's
    return det


def _parts_record(det, fast: bool) -> tuple:
    """Return the run's record of the detector's parts of your own: (path, name, notes) per material a model passed at
    load reads, the notes "unpolarised" (a phase function without the polarised pair) and, on a fast run, "fast walk
    off" (a medium with one channel and no re-emission whose lobe is not lumen's Rayleigh: the diffusion jump does not
    act there, the medium runs the exact walk; ``lumen.transport.fastpath.eligible``)."""  # noqa: D205, D209
    from lumen.media import phase

    optics, out = det.parts.optics, []
    for name, rec in det.file.materials.items():
        m = det.parts.models.get(rec.optics.name)
        if m is None:
            continue
        notes = []
        if m.part is not None and m.part[1].pol_sample is None:
            notes.append("unpolarised")
        if fast and name in optics.media_names:
            med = optics.media[optics.media_names.index(name)]
            one = len(med.channels) == 1 and med.channels[0].emission is None
            if one and med.channels[0].lobe is not phase.RAYLEIGH:
                notes.append("fast walk off")
        path = f"materials/{name}" + ("" if m.part is None else f"/{m.part[0]}")
        out.append((path, m.name, tuple(notes)))
    return tuple(out)


def _part_models(parts: dict) -> dict:
    """Return ``parts`` by material, {"materials/<M>": a model}: a phase function given for one material's channel,
    "materials/<M>/<channel>", is that material's model (``lumen.models.lobe_model``)."""  # noqa: D205, D209
    from lumen.lobes import Lobe
    from lumen.models import lobe_model

    out = {}
    for target, part in parts.items():
        section, _, rest = str(target).partition("/")
        rec_name, _, channel = rest.partition("/")
        if not isinstance(part, Lobe):
            if channel:
                raise ValueError(f"load(parts=): {target!r} is a channel's path: give it a phase function "
                                 "(lumen.media.phase_function)")  # fmt: skip
            whole = target
        elif section != "materials" or not channel or "/" in channel:
            raise ValueError(f"load(parts=): a phase function is given for one material's channel, "
                             f"materials/<M>/<channel>, not {target!r}")  # fmt: skip
        else:
            whole, part = f"materials/{rec_name}", lobe_model(channel, part)
        if whole in out:
            raise ValueError(f"load(parts=): {whole} is given two parts; one per material in 0.2")
        out[whole] = part
    return out


def _with_models(df, models: dict):
    """Return the detector file with each named material's optics model replaced (``load(..., parts=)``)."""
    from lumen._core.errors import LibraryError
    from lumen.build.library import MEDIA

    materials = dict(df.materials)
    for target, model in models.items():
        section, _, rec_name = str(target).partition("/")
        if section != "materials" or rec_name not in materials:
            raise ValueError(f"load(parts=): {target!r} is not materials/<M> of this detector (a surface is read by "
                             "lumen's own surface model: its parts are not hooks in 0.2)")  # fmt: skip
        name = model.name if isinstance(model, MediumModel) else model
        if not isinstance(model, MediumModel) and name not in MEDIA:
            raise LibraryError(f"load(parts=): {target!r}: {model!r} is neither a medium model (lumen.medium_model) "
                               f"nor one of lumen's own ({sorted(MEDIA)})")  # fmt: skip
        rec = materials[rec_name]
        materials[rec_name] = dataclasses.replace(rec, optics=dataclasses.replace(rec.optics, name=name))
    return dataclasses.replace(df, materials=materials) if models else df


# ------------------------------------------------------------------------------------------------------ simulate
# a built detector's Parts -> {(deposit fields carried, analog, speed levers): Model}:
# one compile per detector and layout, shared by the detector's copies (with_responses keeps its Parts), released with
# the Parts (weak keys)
_MODELS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# a Detector -> {(gate, events): its groups' layouts (``_group_layouts``)}: the sensor models and the groups'
# sensor tags (a pattern match per sensor) found once, not per run
_LAYOUTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class Health(NamedTuple):
    """A run's health (``Light.health``, a pytree): what no sensor counted, the record counts and what was refused.

    Every photon's start weight ends in one fate (``FATES``), each the photon's weight where it ends (live, its path
    weighted live / reference as its continuing weight is: equal to the weight at the reference values):
    ``absorbed_bulk`` (in the media: the path's absorption, a WLS yield below 1), ``absorbed_surface`` (at surfaces:
    1 - the continuing shares; the detected share is part of it), ``lost`` (a crossing no surface type joins: a
    navigation failure, or a transparent volume overlapping an opaque one), ``escaped`` (no surface and no bulk event
    within ``max_distance``, at its default the geometry's reach: the photon left the detector), ``distance_cut`` (the
    same within a ``Controls.max_distance`` shorter than the reach), ``abandoned`` (the stuck rule), ``truncated``
    (alive at the vertex guard), ``rouletted`` (the weight the roulette ended minus the weight it added) and
    ``reweighted`` (the expected weight the photons went on with minus the weight they carried: the drawn directions'
    and outcomes' live / proposal factors). The last two are 0 in expectation; the sum of every fate is the start
    weight exactly (to float32 rounding per step) at the reference values (``balance``). ``arrivals`` is a tally of
    the detected share of ``absorbed_surface``, not a fate of its own.

    ``records``: the deposit records its launches appended; ``records_dropped``, ``records_mismatch``: records past
    the capacity and photons whose two passes differed (either makes the outputs NaN; 0 in a run that fits).
    ``nonfinite``: exact-sum cells that are NaN; ``invalid``: traced values a concrete one would be refused for.
    """

    lost: Any
    abandoned: Any
    escaped: Any
    distance_cut: Any
    truncated: Any
    absorbed_bulk: Any
    absorbed_surface: Any
    rouletted: Any
    reweighted: Any
    records: Any
    records_dropped: Any
    records_mismatch: Any
    nonfinite: Any
    invalid: Any

    FATES = ("absorbed_bulk", "absorbed_surface", "lost", "escaped", "distance_cut", "abandoned", "truncated",
             "rouletted", "reweighted")  # fmt: skip
    """The weights every photon's start weight ends in (``balance``)."""

    def balance(self, start):
        """Return ``start`` (the run's start weight: its rows' weights summed) minus every fate (``FATES``): 0 to
        float32 rounding per step at the reference values, and in expectation everywhere (float64)."""  # noqa: D205, D209
        return float(start) - sum(float(np.sum(np.asarray(getattr(self, k), np.float64))) for k in self.FATES)


class SensorTable(NamedTuple):
    """The sensors in the order of ``Light.arrivals`` (``Detector.sensor_table``).

    ``name``: each sensor's placed volume; ``sensor``: its sensor entry; ``position`` (S, 3): its world position (m);
    ``axis`` (S, 3): its placement's local z; ``columns``: the id map's columns ({name: [value per sensor]}, e.g. a
    sensor's type), in the same order.
    """

    name: list
    sensor: list
    position: np.ndarray
    axis: np.ndarray
    columns: dict


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True)
class Light:
    """What ``simulate`` returns: per sensor the expected photon arrivals at the sensor (before its model), their Monte
    Carlo error (from their per-photon second moment, ``lumen.readout.grid.moment_error``), the mean arrival time (ns),
    the histograms (``lumen.readout``) of each group with a
    sensor model and the run's health; ``provenance`` (static) says how it was made (the seed, samples, chunk,
    backend, detector, calibration, the step's controls, each group's gate, and the speed run: "fast", "exact", or
    "exact (<why>)" for a run the fast walk cannot serve, ``lumen.speed``). Readout functions take
    ``(light.group(name), params)``. ``groups``: {group: Histograms} of every modelled group (None without).
    ``diagnostics``: the diagnostics' cells by output name (``simulate(..., diagnostics=)``; None without).
    ``truth``: the truth rows (``simulate(..., truth=)``: ``lumen.readout.TruthHits``; None without). With
    ``simulate(..., events=E)`` every array but the health has a leading event axis (E, ...): ``event(e)`` is event e's
    Light."""  # noqa: D205, D209

    arrivals: Any
    arrivals_error: Any
    mean_arrival_time: Any
    health: Any
    provenance: tuple = ()
    diagnostics: Any = None
    groups: Any = None
    truth: Any = None

    @property
    def events(self) -> int:
        """Return the number of events of an event axis (0: none)."""
        return int(dict(self.provenance).get("events", 0))

    def event(self, e: int) -> Light:
        """Return event ``e``'s Light (of ``simulate(..., events=E)``): its arrays, histograms and truth rows."""
        if not self.events:
            raise ValueError("Light.event: this light has no event axis (simulate(..., events=E))")
        if not 0 <= int(e) < self.events:
            raise IndexError(f"Light.event: event {e} of {self.events}")
        cut = lambda t: jax.tree_util.tree_map(lambda a: a[e], t)  # noqa: E731
        groups = None if self.groups is None else {k: cut(h) for k, h in self.groups.items()}
        prov = tuple((k, v) for k, v in self.provenance if k != "events") + (("event", int(e)),)
        tr = None if self.truth is None else self.truth.select(self.truth.label == e)
        return dataclasses.replace(self, arrivals=self.arrivals[e], arrivals_error=self.arrivals_error[e],
                                   mean_arrival_time=self.mean_arrival_time[e], provenance=prov, groups=groups,
                                   truth=tr)  # fmt: skip

    def group(self, name: str):
        """Return the Histograms of the modelled sensor group ``name``."""
        if not self.groups or name not in self.groups:
            raise KeyError(f"Light: no histograms of sensor group {name!r}; the modelled groups are "
                           f"{list(self.groups or ())}")  # fmt: skip
        return self.groups[name]

    _ARRAYS = ("arrivals", "arrivals_error", "mean_arrival_time", "health", "diagnostics", "groups", "truth")

    def tree_flatten(self):
        """Return the arrays and the provenance."""
        return tuple(getattr(self, k) for k in self._ARRAYS), self.provenance

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from the arrays and the provenance."""
        return cls(provenance=aux, **dict(zip(cls._ARRAYS, children, strict=True)))


class GroupLayout(NamedTuple):
    """One modelled group's readout in a run: the group, its sensor model, its gate (ns), its grid and its
    histograms' layout (``lumen.readout.fill.HistogramLayout``); a never-filled one of no group when none is."""  # noqa: D205, D209

    group: Any
    model: Any
    gate: tuple
    grid: Any
    layout: Any


def _sensor_groups(det) -> list:
    """Return [(group, its SensorValuesModel)] of every modelled group, in the sensor set's order."""
    from lumen.sensors import group_models

    by_name = {g.name: g for g in det.sensor_set.groups}
    return [(by_name[name], model) for name, model in group_models(det.sensor_set).items()]


def _layouts(det, gate, events: int = 0) -> list:
    """Return the GroupLayouts of every modelled group, in the sensor set's order (one never-filled layout of no group
    when there is none). ``gate`` overrides every group's own."""  # noqa: D205, D209
    gate = tuple(float(x) for x in gate) if gate else None
    known = _LAYOUTS.setdefault(det, {})
    if (gate, events) not in known:  # the groups, their models and sensor tags: once per detector
        known[gate, events] = _group_layouts(det, gate, events)
    return known[gate, events]


def _memory(layouts, derivative: bool, row_bytes: int, backend) -> float:
    """Return the device's bytes left for the deposit records and the readout's exact cells, before launch.

    The run holds the launch's emitted rows (``row_bytes``), the readout's float32 cells (the run's: every block of
    every chunk is added into them in place, ``lumen.readout.fill.place``; counted twice: room for a copy XLA makes
    of them, which an 11 GB GPU running Hyper-K ran out of without it) and, for a derivative's run (one
    direction; a batch of directions, jax.jacfwd or jax.vmap, multiplies it and is not seen here), their tangents.
    What is left is shared by the deposit records (``lumen.run.capacity``) and the readout's exact cells, built in as
    many blocks of sensors as they need (``lumen.readout.fill.plan``). MemoryError when nothing is left.
    """
    from lumen import run as runner
    from lumen.readout import fill

    cells = fill.cell_count(tuple(lay.layout for lay in layouts))
    fixed = row_bytes + cells * ro.CELL_BYTES * 2 * (1 + derivative)
    limit = runner.memory_limit(backend)
    if fixed >= limit:
        raise MemoryError(f"the readout's cells need {cells * ro.CELL_BYTES / 1e9:.2f} GB (float32; with the "
                          f"photons in flight {fixed / 1e9:.2f} GB), over the device's "
                          f"{limit / 1e9:.2f} GB: shorten the readout gates, widen the time spreads' declared lower "
                          "ranges (coarser nodes) or give a smaller chunk")  # fmt: skip
    return limit - fixed


def _block_budget(free, held, fields, derivative, layouts, backend) -> float:
    """Return the bytes the readout's exact cells may take after a launch: what the pre-launch share left beside the
    records the launch holds (``_memory``), and no more than the device's free memory now less the run's float32
    cells (made by the first block; counted again once they exist): arrays the caller holds are not the run's to
    take."""  # noqa: D205, D209
    from lumen import run as runner
    from lumen.readout import fill

    left = free - held * runner.record_bytes(fields) * (1 + derivative)
    stats = runner.memory_statistics(backend)
    if stats and "bytes_in_use" in stats:
        cells = fill.cell_count(tuple(lay.layout for lay in layouts))
        now = stats["bytes_limit"] - stats["bytes_in_use"] - cells * ro.CELL_BYTES * (1 + derivative)
        left = min(left, now)
    return left


def _group_layouts(det, gate, events: int) -> list:
    """Return ``_layouts``'s list without the memory check: a function of the detector and the arguments only."""
    from lumen.build import build
    from lumen.readout import fill

    n = len(det.parts.optics.sensor_pvs)
    out = []
    for group, sensor_model in _sensor_groups(det) or [(None, None)]:
        if group is None:  # no histograms: one never-filled cell (no sensor maps into it)
            g_gate, grid, tags = (0.0, 0.0), ro.Grid(), []
        else:
            given = gate or (group.readout or {}).get("gate")
            g_gate = (0.0, 0.0) if given is None else tuple(float(x) for x in given)
            try:
                grid = sensor_model.grid(g_gate)
            except ro.GateError:
                if given is not None:
                    raise
                grid = None
            if given is None and (grid is None or grid.nt):  # a timed model needs a gate to read any light
                raise ValueError(f"simulate: sensor group {group.name!r} has a timed sensor model and no readout gate: "
                                 "pass gate=(start, end) in ns after emission, or set the group's "
                                 'readout={"gate": (start, end)} in its recipe')  # fmt: skip
            tags = build.group_tags(group, det.parts, det.sensor_set)  # disjoint (checked at load)
        pos = {t: i for i, t in enumerate(tags)}
        index = None if tags == list(range(n)) else tuple(pos.get(i, -1) for i in range(n))
        layout = fill.HistogramLayout(max(len(tags), 1), grid, index=index, events=events)
        out.append(GroupLayout(group, sensor_model, g_gate, grid, layout))
    return out


def _model(det, local: tuple = (), analog=False, levers=None):
    """Return the detector's Model of a layout (per deposit fields, analog, speed levers), with
    the rows of the detector's edge rule (``_edge_rows``)."""  # noqa: D205, D209
    from lumen.build import tracked
    from lumen.geometry import reach
    from lumen.transport import step as steps

    _check_x64("simulate", det.precision)
    levers = spd.EXACT if levers is None else levers
    models = _MODELS.setdefault(det.parts, {})
    k = (local, analog, levers, det.precision)
    if k not in models:
        geom = tracked.from_parts(det.parts, cache=True, precision=det.precision)
        geom = reach.reach_of(geom)  # a bulk step's query bounded by its free path where no sensor is reachable (exact)
        jump = None
        if "jump" in levers:  # lumen.speed: the jump's data from this detector (no media: it can act nowhere: off)
            from lumen.transport import fastpath

            jump = fastpath.prepare(geom, det.parts.optics, det.parts.library)
            levers = levers if jump.media else levers - {"jump"}
        cfg = steps.StepConfig(edge_rows=_edge_rows(det), levers=levers)
        models[k] = steps.Model(geom, det.parts.optics, cfg, local=local, analog=analog, jump=jump)  # fmt: skip
    return models[k]


@jax.jit
def _active_w(w, count):
    return jnp.where(jnp.arange(w.shape[0]) < count, w, 0.0)


def _active(rows, count):
    """Return the rows with the weights past ``count`` zero (a padded chunk's inactive rows)."""
    return rows._replace(w=_active_w(rows.w, jnp.asarray(count, jnp.int32)))


class _Inputs:
    """A run's values that may be traced, by entry: ("p", key) a parameter, ("s", field) a source field, ("r", key) a
    reference value, ("seed",), ("controls", name). ``split`` gives the traced ones (the derivative rule's arguments)
    and ``rebuilt`` puts new values in their entries."""  # noqa: D205, D209

    def __init__(self, params, source, reference, seed, controls):
        self.params, self.source, self.reference, self.seed, self.controls = params, source, reference, seed, controls
        slots = {("p", k): v for k, v in params._values.items()}
        slots |= {("s", f): getattr(source, f) for f in source.FIELDS if getattr(source, f, None) is not None}
        if reference is not None:
            slots |= {("r", k): v for k, v in reference._values.items()}
        slots[("seed",)] = seed
        slots |= {("controls", f): v for f, v in zip(controls._fields, controls, strict=True)}
        self.slots = slots

    def split(self):
        """Return (the traced entries, their values)."""
        hot = [k for k, v in self.slots.items() if traced(v)]
        return hot, [self.slots[k] for k in hot]

    def rebuilt(self, hot, values):
        """Return (params, source, reference, seed, controls) with the entries ``hot`` at ``values``."""
        new = dict(zip(hot, values, strict=True))
        pv = {k: new.get(("p", k), v) for k, v in self.params._values.items()}
        src = self.source
        fields = [k[1] for k in hot if k[0] == "s"]
        if fields:
            src = copy.copy(src)
            for f in fields:
                object.__setattr__(src, f, new[("s", f)])
        ref = self.reference
        if ref is not None:
            ref = Params({k: new.get(("r", k), v) for k, v in ref._values.items()}, ref._spec, ref._layers)
        ctl = self.controls._replace(**{k[1]: new[k] for k in hot if k[0] == "controls"})
        return Params(pv, self.params._spec, self.params._layers), src, ref, new.get(("seed",), self.seed), ctl


def _with_rule(run, inputs):
    """Return the run's arrays through one ``custom_jvp(symbolic_zeros=True)`` rule.

    ``run(params, source, reference, seed, controls, derivative)`` is the run's arrays (``derivative`` False: a value;
    True: inside the rule, whose run plans its memory for tangents). The traced inputs are the rule's arguments (every
    other value is concrete, closed over; a traced value closed over, e.g. by a source function, is refused): the
    primal is the plain run; the derivative rule runs ``jax.jvp`` of the run, differentiating only the inputs that
    carry a tangent that is not a symbolic zero, so every guard inside (a method not available, a field without a
    derivative) sees symbolic zeros as symbolic. A value only traced (``jax.jit``, ``jax.vmap``) is a value run.
    """
    hot, vals = inputs.split()
    leaves, tree = jax.tree_util.tree_flatten(vals)
    at_ = [i for i, x in enumerate(leaves) if traced(x)]  # the rule's arguments: the traced leaves

    def values(xs):
        full = list(leaves)
        for i, x in zip(at_, xs, strict=True):
            full[i] = x
        return inputs.rebuilt(hot, jax.tree_util.tree_unflatten(tree, full))

    def plain(*xs):
        return run(*values(xs), False)

    def rule(primals, tangents):
        live = [i for i, t in enumerate(tangents) if not isinstance(t, jax.custom_derivatives.SymbolicZero)]

        def on(*ys):
            xs = list(primals)
            for i, y in zip(live, ys, strict=True):
                xs[i] = y
            return run(*values(xs), True)

        return jax.jvp(on, tuple(primals[i] for i in live), tuple(tangents[i] for i in live))

    f = jax.custom_jvp(plain)
    f.defjvp(rule, symbolic_zeros=True)
    try:  # with no traced input too: a value it closes over that is traced is refused, never differentiated silently
        return f(*(leaves[i] for i in at_))
    except (jax.errors.UnexpectedTracerError, CustomJVPException) as e:
        raise _closure_error(e) from None


def _closure_error(e):
    return sources.GradientNotAvailable(f"simulate: a traced value reaches the run without being given to it (a source "
                                        "function, or a source's attribute that is not a field, closing over it): "
                                        "declare it as a parameter (lumen.Param) and give it as a field or in params "
                                        f"({type(e).__name__})")  # fmt: skip


def _edge_rows(det) -> tuple:
    """Return the defensive rows: every packed row a parameter with a defensive role is read into, whatever its value.

    The detector's structure decides them, never a tangent or a value, so a derivative's run and a value's draw the same
    photons (the same primal, bitwise). A key maps to the packed rows it is read into (``parts.key_rows``): a surface or
    sensor parameter to the surface table's rows of that surface, a medium's to its own entry;
    ``lumen.transport.estimator.edge_rows`` keeps those with a role (the one defensive mixture).
    """
    o, paths = det.parts.optics, []
    for (section, entry, *leaf), src in det.parts.key_rows.items():
        if not src.key or src.column in ("energy", "fixed"):
            continue
        if section == "media":
            paths.append(("media", entry, *leaf))
        else:
            origin = f"{'surface' if section == 'surfaces' else 'sensor'}:{entry}"
            paths += [("surfaces", i, leaf[0]) for i, r in enumerate(o.rows) if r.origin == origin]
    return estimator.edge_rows(o, paths)


def _diagnostics(diagnostics, samples, chunk, reserved) -> tuple:
    """Return (the diagnostics as a tuple, their output names, the deposit fields they read).

    Refused: an output name the run uses itself (``reserved``) or that two diagnostics share, and a probe (a
    diagnostic with ``update``, indexed by the photon's id in its launch) over several chunks. A records readout
    (``emits``) is summed over the chunks like the readout.
    """
    from lumen.sensors.features import FIELDS

    diagnostics = tuple(diagnostics)
    names = [k for dg in diagnostics for k in dg.outputs()]
    clash = sorted({k for k in names if k in reserved or names.count(k) > 1})
    if clash:
        raise ValueError(f"simulate: diagnostic outputs {clash} are names of the run's own outputs or of another "
                         "diagnostic's: rename the diagnostic's outputs")  # fmt: skip
    if any(not hasattr(dg, "emits") for dg in diagnostics) and int(samples) > int(chunk):
        raise ValueError(f"simulate: a probe diagnostic runs in one chunk (its cells are indexed by the photon); "
                         f"samples {samples} > chunk {chunk}: pass chunk >= samples")  # fmt: skip
    reads = {f for dg in diagnostics for f in getattr(dg, "reads", ())}
    if reads - set(FIELDS):
        raise TypeError(f"simulate: a diagnostic reads {sorted(reads - set(FIELDS))}; its reads are deposit fields "
                        f"of {FIELDS}")  # fmt: skip
    return diagnostics, tuple(names), frozenset(reads)


def _to_device(tree):
    """Return a packed tree on the device: its host float32 arrays moved in one transfer (one buffer, split on the
    device), its other values (traced, already on the device) as ``jnp.asarray`` gives them."""  # noqa: D205, D209
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    host = [i for i, x in enumerate(leaves) if isinstance(x, np.ndarray) and x.dtype == np.float32]
    if host:
        flat = jax.device_put(np.concatenate([leaves[i].ravel() for i in host]))
        for i, a in zip(host, _split(flat, tuple(leaves[i].shape for i in host)), strict=True):
            leaves[i] = a
    return treedef.unflatten([jnp.asarray(x) for x in leaves])


@functools.partial(jax.jit, static_argnums=1)
def _split(flat, shapes):
    out, at = [], 0
    for s in shapes:
        n = math.prod(s)
        out.append(flat[at : at + n].reshape(s))
        at += n
    return out


_ADD = jax.jit(lambda a, b: {k: a[k] + b[k] for k in a}, donate_argnums=0)  # chunk accumulation, in place


def _recorded(v):
    """Return a control's value for the provenance, as the loop reads it (float32): a float, or "traced" (under a
    transform it has no value here)."""  # noqa: D205, D209
    try:
        return float(v)
    except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return "traced"


def simulate(det, source, params=None, *, samples: int | None = None, seed: int, gate=None,  # noqa: C901, PLR0913, PLR0915
             events: int | None = None, reference=None, controls=None, backend: str = "auto", chunk: int | None = None,
             speed: str = "fast", records: int | None = None, diagnostics=(), truth=None) -> Light:  # fmt: skip
    """Trace ``samples`` Monte Carlo photons from ``source`` through ``det`` and return the light at each sensor.

    Parameters
    ----------
    det : Detector
        From ``lumen.load``.
    source : lumen.sources.Source
        The light source; its fields can be set through ``params`` keys ``source/<field>`` (``Params.set``); a key
        holding another value than a traced field, or another source's ``parameters()``, is refused. Photons you have:
        ``lumen.sources.FromArrays``; a source of your own: ``@lumen.source``.
    params : Params, optional
        Parameter values (default ``det.params()``). Any value may be traced (``jax.jit``, ``jax.vmap``, ``jax.jvp``,
        ``jax.grad``, ``jax.linearize``), with no declaration of which: changing them never recompiles. The derivative
        follows from which values carry a tangent (one ``custom_jvp`` rule, symbolic zeros kept). A field (or
        ``source/<field>`` key) that moves where the light starts (``source.MOVING``: a position, a direction) has no
        derivative in this version: a tangent on it raises ``GradientNotAvailable`` (docs/explanation/known-limits.md).
        A value-only run and the primal of a derivative are the same, bitwise. A traced value must reach the run as a
        source field or in ``params`` (a source function closing over a tracer is refused). Traced values that a
        concrete one would be refused for are counted in ``health.invalid``.
    samples : int, optional
        Monte Carlo photons (the effort). The physics, the photons emitted, is the source's ``n_photons``. A source of
        one photon per row (``FromArrays``, a ``Point`` with per-row fields) emits exactly its rows: leave it out (or
        give the row count).
    seed : int
        Fixes the random numbers (an int, or a traced int: ``jax.vmap`` over seeds); keep it fixed inside anything
        you differentiate or fit.
    gate : (float, float), optional
        Readout window in ns after emission, for every modelled group (default: each group's ``readout["gate"]``,
        which every shipped recipe sets). A group whose sensor model is timed (hits, waveforms) needs one. The gate
        sets only the time histogram's extent: every photon is followed (no time cut), and light before or after it
        is in the histograms' early / late cells and in every total (``arrivals``, charge, ``pe``).
    chunk : int, optional
        Photons per launch, a memory lever: every photon is the same whatever it is, and the results differ only by
        the float32 sum of the launches' results (about 1e-7 of a value). A launch has a power of two of rows (the
        power of two at or above ``samples``, at most the chunk), so a new ``samples`` in the same power of two
        compiles nothing. Default: at most ``CHUNK`` (2^20), halved until the rows fit half the device memory the
        readout's cells leave; a launch whose deposit records do not fit beside the readout runs again at the size
        that does, kept for the later launches. A given chunk caps the launch, and a launch of it whose records do
        not fit is refused (MemoryError). Under ``jax.jit`` or ``jax.vmap`` the chunks after the first run in one
        ``lax.scan`` (one launch's memory, whatever their number).
    backend : {"auto", "cuda", "cpu", "xla"}
        "auto": tallymap's compiled photon loop on the device JAX runs on: CUDA on a GPU, else the CPU (which needs
        a C++ compiler with OpenMP, such as g++). "cuda": the same, and a GPU must be there. "cpu": the same on the
        CPU, also on a GPU machine (the results come back to JAX's default device). "xla":
        tallymap's plain-JAX backend: no compiler needed, slow, for tests and small photon counts; the same exact sums.
    controls : lumen.Controls, optional
        The step limits ``max_distance``, ``roulette`` (default ``Controls()``). A ``max_distance`` of
        None (the default) is the detector's reach, the diagonal of the box holding it.
        ``roulette`` is one ε for values and derivatives (default ``lumen.Controls().roulette``): a photon whose
        weight at the reference values falls below that share of its own start weight is kept with probability (that
        weight) / ε, else ended.
        Unbiased at any value; only the run time and the noise depend on it (docs/explanation/known-limits.md).
    events : int, optional
        E: every output per event (a leading axis of E), the event of a photon its source's ``label`` in [0, E)
        (``Light.event(e)``). The health stays the run's.
    reference : Params, optional
        The values every random choice is made at, in the photon loop and in the source (a wavelength from a
        spectrum, a scintillation component, a Geant4 step; default: ``params``, the run's own). Kept fixed while
        ``params`` moves, the light at a seed is a smooth function of ``params`` (a finite difference of it is the
        derivative at the reference); its expectation is the light at ``params``. At
        ``reference=params`` the light is the default run's, exactly. Its derivative is never taken. A source's
        spectrum must lie inside the reference's support (``lumen.build.values``).
    speed : {"fast", "exact"}
        "fast" (the default: the diffusion walk, exact in law, validated against "exact") or "exact" (no fast
        estimator). A derivative along a source position runs exact, recorded
        as ``("speed", "exact (source-position derivative)")`` in ``Light.provenance``. See ``lumen.speed``.
    records : int, optional
        An optional override: the most deposit records a launch appends, whose power of two is each launch's slots
        (to shrink the memory a traced run takes). Without it a run with values takes its slots from its launches'
        counts (tallymap's ``appended``, a launch past its slots run again at the count's power of two), and a run
        under ``jax.jit`` or ``jax.vmap`` the most slots that fit the memory left (``lumen.run.capacity``): its
        outputs are the same bitwise while nothing overflows, and a launch past its slots raises
        ``lumen.run.RecordsOverflow`` when the run executes, with the number to give.
    diagnostics : tuple, optional
        Per-photon diagnostics (``lumen.diagnostics``): their cells in ``Light.diagnostics`` by output name. A probe
        (it sees every step: ``lumen.diagnostics.Trace``) runs in one chunk (``samples <= chunk``) and exact, recorded
        as ``("speed", "exact (diagnostics)")``; a records diagnostic is built from the deposit records like the
        readout and follows ``speed``.
    truth : lumen.readout.Truth, optional
        Truth rows in ``Light.truth`` (``lumen.readout.TruthHits``): the analog record of the run, one row per
        detected photon (``lumen.readout.truth``). Each source row stands for at most one photon (``samples`` at least
        the photons emitted), the roulette is a number <= 1, nothing is differentiated (a traced value carrying a
        derivative is refused), and the photons' fates are decided at the run's own values: a ``reference`` other
        than ``params`` is refused.

    Returns
    -------
    Light
        ``arrivals``, ``arrivals_error`` (the Monte Carlo error from each sensor's per-photon second moment, for
        photons drawn independently: ``lumen.readout.grid.moment_error``) and ``mean_arrival_time`` for every
        sensor; the
        histograms of each sensor group with a sensor model (``groups``); ``health``; ``diagnostics`` and ``truth``
        when asked for.

    Raises
    ------
    GradientNotAvailable, MemoryError, DeviceError, RecordsUnknown, RecordsOverflow
        See docs/reference/errors.md.
    """
    from lumen import run as runner
    from lumen.build import pack, values
    from lumen.readout import detection, fill
    from lumen.transport import step as steps

    if not isinstance(source, sources.Source):
        raise TypeError(f"simulate: the source is a {type(source).__name__}, not a lumen.sources.Source: give photons "
                        "you have as FromArrays, a source of your own as a @lumen.source function")  # fmt: skip
    params = det.params() if params is None else params
    source.refuse_overridden(params)
    params, source, invalid = values.check(det, params, source, reference)
    controls = steps.Controls() if controls is None else controls
    samples = _samples(source, params, samples)
    if chunk is not None and int(chunk) < 1:
        raise ValueError(f"simulate: chunk is a number of photons >= 1, got {chunk}")
    # every launch has `size` rows, a power of two (at most the chunk): one compiled loop and one emission per size,
    # whatever `samples`; rows past the run's photons are inactive (n_active) and weigh nothing
    size0 = min(CHUNK if chunk is None else int(chunk), bucket(int(samples)))
    n_ev = _events(events)
    layouts = _layouts(det, gate, n_ev)
    r = tuple(lay.layout for lay in layouts)
    n = len(det.parts.optics.sensor_pvs)
    source.check(params)
    reserved = {*Health._fields, runner.RECORDS, *fill.readout(r).outputs()}
    diagnostics, dnames, dreads = _diagnostics(diagnostics, samples, size0, reserved)
    local = tr.local_fields(det, truth, frozenset().union(dreads, *(lay.grid.reads for lay in layouts)))
    readouts = tuple(dg for dg in diagnostics if hasattr(dg, "emits"))  # built from the deposit records
    probes = tuple(dg for dg in diagnostics if not hasattr(dg, "emits"))  # beside the loop: the photon's state
    tabs = _truth_setup(det, truth, controls, params, reference)
    inputs = _Inputs(params, source, reference, seed, controls)
    if truth is not None:  # not differentiable: every traced value refuses a derivative
        no = refusing("simulate: truth rows are not differentiable; take the derivative of a run without truth=")
        inputs.slots = {k: no(v) if traced(v) else v for k, v in inputs.slots.items()}

    def run(params, source, reference, seed, controls, derivative):  # noqa: C901, PLR0912, PLR0915 -- the run, in order
        """Return the run's arrays; ``derivative``: inside the derivative's rule."""
        on, ran[0] = spd.resolve(speed, probes=bool(probes))
        model = _model(det, local, truth is not None, on)
        packed = _to_device(pack.pack(det, params))
        ref = None if reference is None else jax.tree_util.tree_map(jax.lax.stop_gradient,
                                                                    _to_device(pack.pack(det, reference)))  # fmt: skip
        ops = steps.operands(model, packed, ref, controls)
        read[0] = ops.controls  # as the loop reads them (a max_distance of None: the model's reach)
        key = jax.random.key_data(jax.random.PRNGKey(seed if traced(seed) else int(seed)))
        skey = jax.random.fold_in(key, SOURCE_STREAM)
        acc, nonfinite, bad = None, jnp.int32(0), _roulette_invalid(controls.roulette)
        emit = source.emitter(params, skey, int(samples), reference=reference)
        fields = runner.record_fields(model.local, bool(n_ev))
        size = size0
        if chunk is None and not probes:  # the default: halved until the rows fit half what the readout's cells leave
            row = _row_bytes(emit) * (1 + derivative)
            free0 = _memory(layouts, derivative, 0, backend)
            while size > 1 and size * row > free0 / 2:
                size //= 2
        emit_rows = lambda first, count: _active(emit(first, size), count)  # noqa: E731
        first_rows = emit_rows(0, min(size, int(samples)))  # emitted once: the checks read it, the first launch runs it
        if jax.core.is_concrete(first_rows.t):  # float32 times resolve the time nodes (``check_times``)
            t_rows = np.asarray(jax.lax.stop_gradient(first_rows.t))[: min(size, int(samples))]  # (a host slice)
            ro.check_times([lay.grid for lay in layouts], np.r_[t_rows, [x for lay in layouts for x in lay.gate]])
        # each modelled group's acceptance per charge cell at the values given (an operand: differentiable in them):
        # the time histogram's weight on a weighted grid, and every group's pe moment
        weights = tuple(lay.model.acceptance(lay.model.split(params, lay.group.name)) if lay.model is not None
                        else np.ones(1, np.float32) for lay in layouts)  # fmt: skip
        rows_of = []  # the truth rows of each chunk
        concrete = all(jax.core.is_concrete(x) for x in jax.tree_util.tree_leaves((ops, key, first_rows)))

        def launch(first, count, rows, cap, free):  # one chunk's photon loop at ``cap`` record slots
            labels, analog = rows.label, (_analog_weights(rows.w, count) if truth is not None else None)
            ev_bad = _check_events(labels, count, n_ev) if n_ev else 0  # refused here: before any launch
            rows = rows._replace(label=labels if n_ev else None)  # a label enters the records as the event
            t = runner.trace(model, ops, rows, key, fields=fields, capacity=cap, first_id=first, backend=backend,
                             n_active=count, analog=analog, diagnostics=probes)  # fmt: skip
            return t, labels, ev_bad

        def add(acc, nonfinite, bad, t, free, ev_bad):  # one launch's readout, added into the run's cells
            held = t.records["sensor"].shape[0]
            budget = _block_budget(free, held, fields, derivative, layouts, backend)
            blocks = fill.plan(r, budget, derivative, backend)
            bad = bad + ev_bad
            nonfinite = nonfinite + jnp.asarray(t.overflow, jnp.int32)
            hist = None if acc is None else {k: acc.pop(k) for k in fill.readout(r).outputs()}
            for b in blocks:  # the readout, per block of sensors, from the chunk's records, added into the run's cells
                o, nan = runner.rebuild(fill.readout(r, b), t.records, t.count, weights, backend=backend)
                nonfinite = nonfinite + jnp.asarray(nan, jnp.int32)
                hist = fill.place(r, b, o, hist)
                del o
            cells = t.sums | runner.records_health(t)
            for dg in readouts:  # a diagnostic of the deposits: built from the records like the readout
                o, nan = runner.rebuild(_as_readout(dg), t.records, t.count, (), backend=backend)
                nonfinite, cells = nonfinite + jnp.asarray(nan, jnp.int32), cells | o
            acc = hist | (cells if acc is None else _ADD(acc, cells))  # in place: acc's buffers donated
            return acc, nonfinite, bad

        def free_for(rows):  # the device's bytes left for a launch's records and the readout's exact cells
            rows_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(rows))
            return _memory(layouts, derivative, rows_bytes, backend)

        first, cap = 0, None  # cap: a run with values, the slots its launches' counts ask for so far
        while concrete and first < int(samples):  # with values: each launch sized from the counts (``capacity``)
            count = min(size, int(samples) - first)
            rows = first_rows if first_rows is not None else emit_rows(first, count)
            free = free_for(rows)
            room = _record_room(free, fields, derivative, r, backend)
            start = cap if cap is not None or records is None else runner.pow2(records)  # records=: the first guess
            slots = runner.capacity(room, rows=size, counted=start)
            t, labels, ev_bad = launch(first, count, rows, slots, free)
            appended = int(t.count)
            over, need = runner.overflow(appended, t.dropped, slots)
            if over:  # past its slots: the outputs are poisoned; again at the count's power of two
                if need is not None and need <= room:
                    cap = need
                    continue
                appended = max(appended, 2 * slots) if need is None else appended  # a wrapped count: at least so many
                if chunk is not None or probes or size == 1:
                    raise MemoryError(f"simulate: a launch of {size} photons appends {appended} deposit records, "
                                      f"over the {room} that fit the device beside the readout's cells: give a "
                                      "smaller chunk") from None  # fmt: skip
                size = max(1, min(size // 2, 1 << max(int(size * room / appended), 1).bit_length() - 1))
                first_rows, cap = None, None
                continue  # the same photons again, in launches of the size that fits
            first_rows = None
            cap = max(cap or 0, runner.pow2(appended))
            if truth is not None:
                rows_of.append(tr.chunk_rows(truth, tabs, tr.detection_values(tabs[3], params), t.per_item, labels,
                                             key, first, count))  # fmt: skip
            acc, nonfinite, bad = add(acc, nonfinite, bad, t, free, ev_bad)
            del t, rows  # the next chunk runs with only acc alive (not this chunk's outputs too)
            first += count
        if not concrete:  # under jax.jit or jax.vmap: a static capacity from the memory, the chunks in one lax.scan
            if truth is not None:
                raise ValueError("simulate: truth rows are built on the host from each chunk: run them outside "
                                 "jax.jit and jax.vmap")  # fmt: skip
            spans = ro.photon_chunks(int(samples), size)
            free = free_for(first_rows)
            room = _record_room(free, fields, derivative, r, backend, runner.batch_size((ops, key, first_rows)))
            slots = runner.capacity(room, records=records)

            def body(c, fc):  # one chunk: its launch, refused when it overflows (``guard``), added into the cells
                acc, nonfinite, bad = c
                t, _, ev_bad = launch(fc[0], fc[1], emit_rows(fc[0], fc[1]), slots, free)
                runner.guard(t, slots, size)
                return add(acc, nonfinite, bad, t, free, ev_bad), None

            t, _, ev_bad = launch(0, spans[0][1], first_rows, slots, free)
            runner.guard(t, slots, size)
            acc, nonfinite, bad = add(acc, nonfinite, bad, t, free, ev_bad)
            first_rows = None
            if len(spans) > 1:  # unrolled, XLA may hold every launch's records at once
                fc = (jnp.asarray([f for f, _ in spans[1:]], jnp.int32), jnp.asarray([k for _, k in spans[1:]],
                                                                                      jnp.int32))  # fmt: skip
                (acc, nonfinite, bad), _ = jax.lax.scan(body, (acc, nonfinite, jnp.asarray(bad, jnp.int32)), fc)
        sizes[0] = size
        ev = (n_ev,) if n_ev else ()
        arrivals = acc["arrival"].reshape(*ev, n)
        # the error of each sensor's sum over the photons, from its per-photon second moment (unbiased for i.i.d.
        # photons; no derivative)
        err = ro.moment_error(acc["arrival2"].reshape(*ev, n), arrivals, int(samples))
        arrival_t = acc["arrival_t"].reshape(*ev, n)
        mean_t = jnp.where(arrivals > 0.0, arrival_t / jnp.where(arrivals > 0.0, arrivals, 1.0), jnp.nan)
        health = Health(**{k: acc[k] for k in Health._fields[:-2]}, nonfinite=nonfinite, invalid=bad)
        pe2 = acc["pe2"].reshape(*ev, n)
        groups = {}
        for k, lay in enumerate(layouts):
            if lay.group is None:
                continue
            h = fill.histograms(lay.layout, acc, lay.model, lay.group.name, lay.gate, f":{k}")
            pe_err = detection.pe_moment_error(h, detection.values(h, params), pe2, int(samples))
            groups[lay.group.name] = dataclasses.replace(h, pe_error=pe_err)
        diag = {k: acc[k] for k in dnames} if diagnostics else None
        _truth_rows[0] = None if truth is None else tr.join(rows_of, truth, int(samples), tabs)
        return arrivals, err, mean_t, health, groups or None, diag

    spd.resolve(speed)  # a speed that is neither level is refused before the run
    ran, read, _truth_rows, sizes = [speed], [controls], [None], [size0]
    arrivals, err, mean_t, health, groups, diag = _with_rule(run, inputs)
    health = health._replace(invalid=health.invalid + invalid)
    gates = tuple((lay.group.name, lay.gate) for lay in layouts if lay.group is not None)
    prov = (
        ("seed", _recorded_int(seed)),
        ("samples", int(samples)),
        ("chunk", sizes[0]),
        ("backend", backend),
        ("detector", det.name),
        ("calibration", det.calibration),
        ("controls", tuple(zip(controls._fields, map(_recorded, read[0]), strict=True))),
        ("gate", gates),
        ("speed", ran[0]),
    )
    prov += (("parts", parts),) if (parts := _parts_record(det, ran[0] == "fast")) else ()
    prov += (("events", n_ev),) if n_ev else ()
    prov += (("truth", dataclasses.astuple(truth)),) if truth else ()
    return Light(
        arrivals=arrivals,
        arrivals_error=err,
        mean_arrival_time=mean_t,
        health=health,
        provenance=prov,
        diagnostics=diag,
        groups=groups,
        truth=_truth_rows[0],
    )


@functools.lru_cache(maxsize=64)
def _as_readout(dg):
    """Return a records diagnostic as a readout (``lumen.run.rebuild``): its emits, outputs and moments (None
    without)."""  # noqa: D205, D209
    from lumen.readout import fill

    return fill.Readout(dg.emits, dg.outputs, getattr(dg, "moments", None))


CHUNK = 1 << 20  # simulate's default launch ceiling: device-independent, so a run is the same wherever it fits


def _row_bytes(emit) -> float:
    """Return one emitted row's bytes (from the rows' shapes, not evaluated)."""
    m = 1024
    shapes = jax.eval_shape(lambda: emit(0, m))
    return sum(math.prod(x.shape) * x.dtype.itemsize for x in jax.tree_util.tree_leaves(shapes)) / m


def _record_room(free, fields, derivative, layouts, backend, batch: int = 1) -> int:
    """Return the most record slots a launch may have, a power of two (0: none): those whose launch, at its peak
    (``lumen.run.slot_bytes``: of a jax.vmap of ``batch`` members under one), fits what ``free`` leaves beside the
    readout's smallest block (the arrivals and its largest sensor's exact cells, ``lumen.readout.fill.plan``)."""  # noqa: D205, D209
    from lumen import run as runner
    from lumen.readout import fill

    rs = tuple(layouts)
    n_all = rs[0].n_sensors if rs[0].index is None else len(rs[0].index)
    cells = len(fill.ARRIVALS) * max(rs[0].events, 1) * n_all + max(fill.sensor_cells(x) for x in rs)
    block = runner.readout_bytes(cells, derivative, backend) * batch
    return runner.slots_in(free - block, fields, derivative, batch)


def _recorded_int(v):
    """Return an int for the provenance, or "traced"."""
    return "traced" if traced(v) else int(v)


def _roulette_invalid(roulette):
    """Return the count of a traced roulette that is not a number > 0 (a concrete one is refused by the step)."""
    if not traced(roulette):
        return jnp.int32(0)
    return jnp.sum(~(jnp.asarray(roulette) > 0.0), dtype=jnp.int32)


def _samples(source, params, samples) -> int:
    """Return the run's photons: a one-photon-per-row source's rows (``samples`` None or equal), else ``samples``."""
    rows = source.photon_rows(params)
    if rows is None and samples is None:
        raise ValueError(f"simulate: samples is the number of Monte Carlo photons to draw from the "
                         f"{type(source).__name__} (a positive int), got {samples}")  # fmt: skip
    if rows is not None and samples is not None and int(samples) != rows:
        raise ValueError(f"simulate: a {type(source).__name__} of one photon per row emits its {rows} rows, each once; "
                         f"samples={int(samples)} is not its row count (leave samples out)")  # fmt: skip
    n = int(samples) if rows is None else rows
    if n > MAX_PHOTONS:  # a photon's id is an int32: past it ids and random keys repeat
        raise ValueError(f"simulate: {n} photons; a run traces at most {MAX_PHOTONS} (2^31 - 1, the range of the "
                         "photon ids): split the run into calls of at most that many photons, each with its own "
                         "seed")  # fmt: skip
    return n


def _events(events) -> int:
    """Return the event axis's length (0: none)."""
    if events is None:
        return 0
    if int(events) < 1:
        raise ValueError(f"simulate: events is a positive number of events, got {events}")
    return int(events)


@jax.jit
def _labels_outside(labels, count, n_ev):
    active = jnp.arange(labels.shape[0]) < count
    return jnp.sum((active & ((labels < 0) | (labels >= n_ev))).astype(jnp.int32))


def _check_events(labels, count, n_ev: int):
    """Return the count of photons with a label outside [0, E), refused when concrete (counted in
    ``health.invalid`` when traced); refuse an event axis over photons without labels."""  # noqa: D205, D209
    if labels is None:
        raise ValueError("simulate: events=E reads each photon's event from its source's label: give the source a "
                         "label (an int per photon row, or per Geant4 step of CherenkovSteps and "
                         "ScintillationSteps)")  # fmt: skip
    bad = _labels_outside(labels, jnp.asarray(count, jnp.int32), jnp.asarray(n_ev, jnp.int32))
    if traced(bad):
        return bad
    if int(bad):
        raise ValueError(f"simulate: {int(bad)} photon(s) have a label outside the events [0, {n_ev})")
    return jnp.int32(0)


@jax.jit
def _analog_w(w, count):
    w = jnp.where(jnp.arange(w.shape[0]) < count, jax.lax.stop_gradient(w), 0.0)
    return w, jnp.max(w)


def _analog_weights(w, count):
    """Return the rows' analog weights (the physical photons each stands for), refused above one photon per row."""
    a, top = _analog_w(w, jnp.asarray(count, jnp.int32))
    if float(top) > 1.0:
        raise ValueError(f"simulate: a source row stands for {float(top):.9g} photons; a truth row is one "
                         "photon: run samples >= the photons emitted (or split the rows)")  # fmt: skip
    return a


def _truth_setup(det, truth, controls, params, reference):
    """Return the detector's detection tables of a run with truth rows (else None), refusing what the analog record
    cannot be exact for."""  # noqa: D205, D209
    if truth is None:
        return None
    if not isinstance(truth, tr.Truth) or truth.keep not in tr.KEEPS:
        raise ValueError(f"simulate: truth is a lumen.readout.Truth with keep one of {tr.KEEPS}, got {truth!r}")
    r = _recorded(controls.roulette)
    if r == "traced" or r > 1.0:
        raise ValueError(f"simulate: with truth rows the roulette must be a number <= 1 (got {r})")
    if reference is not None:  # the analog fate is decided on the live values: a reference elsewhere would move it
        try:
            same = sorted(reference) == sorted(params) and all(_same(params[k], reference[k]) for k in params)
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
            same = False
        if not same:
            raise ValueError("simulate: truth rows decide each photon's fate at the run's own values; a reference= "
                             "other than params is refused (leave it out)")  # fmt: skip
    return tr.tables(det)


__all__ = [
    "Detector",
    "DetectorEntry",
    "Health",
    "Light",
    "SensorTable",
    "convert",
    "is_package_detector",
    "load",
    "package_detector",
    "simulate",
]
