"""The ``lumen:sensor`` sensor model: any sensor's detection and electronics after tracking, and its value sets.

    model = lumen.sensors.load_model("my_pmt.json")             # a path, or <package>:<set> (a package's own sets)
    p = model.defaults("pmt", n_sensors)                        # {"sensors/pmt/<field>": array}
    lumen.sensors.write_defaults("examples/data/toy_pmt.json", "my_pmt.json")

A value set (``lumen-sensor-values/1``) holds structure (forms, knot positions, table lengths, the front end's sampling
step and charge grid) and values. Structure is static: it fixes the histograms' grids and the compiled readout.
Values become the parameters ``sensors/<group>/<field>`` and are traced, so changing one never recompiles. File
fields carry their unit as a suffix (``sigma_ns``), parameter keys do not (``sensors/pmt/tts/sigma``, in ns).

Detection: efficiency x the product of the acceptance tables ``qe``, ``ce``, ``acceptance`` (each optional), each a
table over one or more axes. An axis is a feature of the deposit (``lumen.sensors.features``) on knots (Geant4's
linear interpolation, constant beyond the end knots) or between band edges, named in a table record by its key:
``<feature>_<unit>`` knots, ``<feature>_edges_<unit>`` the edges between bands (the end bands open); ``band_edges_nm``
is the wavelength's bands, ``theta_deg`` the cathode position angle (degrees in the file, radians in the deposit).
The values are nested in the order of the axis keys. ``qe`` over wavelength and ``ce`` over theta are
QE(lambda) CE(theta); a joint table (``acceptance``: e.g. ``cos_incidence`` x ``wavelength_nm``) is a table with two
axes. An axis appears with one set of knots; tables sharing an axis multiply on its knots. The ``axes`` record adds
axes no table reads (``{"<feature>_<unit>": knots}``, e.g. ``x_m``), binned for a response
of your own (``lumen.response``). Time: TTS, an optional delay (one table, or one per cell of its own axes, e.g.
``band_edges_nm`` or ``generation_edges``: its ``pdf`` nested over them), the transit offset ``t_offset``, an optional
time walk; electronics: the SPE charge, the threshold, the pulse, the dark rate and the front end (see
``lumen.sensors.components``).
The per-sensor fields (``efficiency``, ``t_offset``, ``gain``, ``dark_rate``) are arrays over the group's sensors.

A form or axis feature named ``"package:Name"`` is reserved for forms and features of your own (the medium models'
naming rule, ``lumen.models``): this lumen refuses one. A record not of the schema (``KEYS``) is refused.
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from lumen._core.domains import FRACTION, POSITIVE
from lumen.format.io import check_format
from lumen.params import Param
from lumen.readout.grid import Axis, GateError, Grid, cells, node_spacing
from lumen.rules.choices import TIME_SPREAD_REACH
from lumen.sensors import components as parts
from lumen.sensors import features as feats

SCHEMA = "lumen-sensor-values/1"
KEYS = (
    "schema",
    "name",
    "model",
    "provenance",
    "readout",
    *("qe", "ce", "acceptance", "axes"),
    "efficiency",
    "t_offset_ns",
    "gain",
    "dark_rate_hz",
    "tts",
    "delay",
    "time_walk",
    "charge",
    "threshold",
    "pulse",
    "frontend",
    "ranges",
)  # a value set's records; any other is refused
STATUSES = ("measured", "transcribed", "unverified")
PER_SENSOR = ("efficiency", "t_offset", "gain", "dark_rate")
SUFFIXES = ("_ns", "_pe", "_hz", "_nm", "_deg")
READOUTS = ("pe", "hits", "waveform")
FRONTEND_VALUES = ("baseline", "noise", "window", "charge_noise", "charge_scale")
IN_READOUT_UNIT = ("gain", "frontend/baseline", "frontend/noise")  # in the front end's unit (pe or ADC): no suffix
UNITS = {"ns": "ns", "pe": "pe", "hz": "Hz", "nm": "nm", "deg": "deg"}
TABLES = ("qe", "ce", "acceptance")  # the acceptance tables a value set may give (each optional)
# the admissible ranges of the fields a value set need not declare (a component's: its form's ``RANGES``); a field
# in none of them is any finite value
DEFAULT_RANGES = {"qe": FRACTION, "ce": FRACTION, "acceptance": FRACTION, "efficiency": FRACTION,
                  "dark_rate": POSITIVE, "frontend/noise": POSITIVE, "frontend/window": POSITIVE,
                  "frontend/charge_noise": POSITIVE}  # fmt: skip
AXIS_UNITS = {"nm": lambda x: x, "deg": lambda x: x * np.pi / 180.0, "m": lambda x: x, "ns": lambda x: x}  # to the
# feature's unit (degrees in the file, radians in the deposit)
GRID_RANGES = ("t_offset",)  # declared ranges a value set must give (with every tts/ and pulse/ value but a width)
WIDTHS = ("tts/sigma",)  # derived ranges: [r, 1 / r] x the set's own width (r: rules.choices.TIME_SPREAD_REACH)


class SensorValuesError(ValueError):
    """A sensor value set that does not follow the ``lumen-sensor-values/1`` rules."""


def _strip(name: str) -> tuple[str, str]:
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)], UNITS[s[1:]]
    return name, ""


def _reserved(name: str, what: str, set_name: str) -> None:
    """Refuse a name of the form ``"package:Name"``: the rule reserved for forms and features of your own."""
    if ":" in str(name):
        raise SensorValuesError(f"value set {set_name!r}: {what} {name!r}: a name 'package:Name' names a form or "
                                "feature of your own, which this lumen does not read (reserved in "
                                f"{SCHEMA}); use one of lumen's own")  # fmt: skip


def _component(spec: dict | None, forms: dict, what: str, set_name: str):
    """Return (component, {field: value}) of a component record, or (None, {}): the form's ``from_spec`` of the
    record's structure keys, the other keys its values."""  # noqa: D205, D209
    if spec is None:
        return None, {}
    form = spec.get("form", next(iter(forms)) if len(forms) == 1 else None)  # a one-form component may omit it
    _reserved(form, f"{what} form", set_name)
    if form not in forms:
        raise SensorValuesError(f"value set {set_name!r}: {what} form {form!r} is not one of {sorted(forms)}")
    comp = forms[form].from_spec(spec)
    vals = {_strip(k)[0]: np.asarray(v, np.float64) for k, v in spec.items() if k != "form" and k not in comp.STATIC}
    if set(vals) != set(comp.fields()):
        raise SensorValuesError(f"value set {set_name!r}: {what} ({form}) needs the fields {sorted(comp.fields())}, "
                                f"got {sorted(vals)}")  # fmt: skip
    for k in spec:
        f, unit = _strip(k)
        if f in vals and unit and unit != comp.fields()[f][0]:
            raise SensorValuesError(f"value set {set_name!r}: {what}.{k} is in {unit}, the field is in "
                                    f"{comp.fields()[f][0]}")  # fmt: skip
    return comp, vals


@dataclass(frozen=True, eq=False)
class SensorValuesModel:
    """The ``lumen:sensor`` model of one value set: static structure plus its default values and declared ranges."""

    set_name: str
    readout: str
    axes: tuple = ()  # (lumen.readout.grid.Axis, ...): the charge histogram's axes
    tables: tuple = ()  # ((table name, (axis name, ...)), ...): the acceptance tables, each over some axes
    tts: object = None
    charge: object = None
    delay: object = None
    time_walk: object = None
    threshold: object = None
    pulse: object = None
    sampling: float = 1.0
    unit: str = "pe"
    charge_step: float = 1.0 / 32.0
    charge_max: float = 128.0
    values: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)  # field -> unit: the file's suffix, a component's, or the front end's
    provenance: dict = field(default_factory=dict)
    name = "lumen:sensor"
    efficiency_after_tracking = True  # every absorption entering is counted; efficiency and acceptance apply after

    @property
    def reads(self) -> frozenset:
        """Return the deposit fields the model's features read (its axes and its delay's)."""
        delay = tuple(self.delay.axes) if self.delay is not None else ()
        return frozenset(f for a in (*self.axes, *delay) for f in feats.reads(a.feature))

    @property
    def detection_reads(self) -> frozenset:
        """Return the deposit fields its acceptance reads (what a truth row needs for ``detection``)."""
        used = {n for _, ax in self.tables for n in ax}
        return frozenset(f for a in self.axes if a.name in used for f in feats.reads(a.feature))

    def axis(self, name: str):
        """Return the charge axis of that name, or None."""
        return next((a for a in self.axes if a.name == name), None)

    def defaults(self, group: str, n_sensors: int) -> dict:
        """Return the default parameter values ``sensors/<group>/<field>`` (per-sensor fields as (n_sensors,))."""
        out = {}
        for k, v in self.values.items():
            v = np.full(n_sensors, float(v)) if k in PER_SENSOR else np.asarray(v, np.float64)
            out[f"sensors/{group}/{k}"] = v
        return out

    def param_info(self, group: str) -> dict:
        """Return ``sensors/<group>/<field>`` -> lumen.Param (unit, range, after-tracking method)."""
        return {f"sensors/{group}/{k}": Param(self.units.get(k, ""), self.range(k), method="after_tracking")
                for k in self.values}  # fmt: skip

    def range(self, k: str) -> tuple:
        """Return a field's admissible range: the set's declared one, its component form's, or ``DEFAULT_RANGES``'."""
        what, _, field = k.partition("/")
        form = getattr(self, what, None) if field else None  # a component's field: "<component>/<field>"
        inf = float("inf")
        return tuple(self.ranges.get(k, getattr(form, "RANGES", {}).get(field, DEFAULT_RANGES.get(k, (-inf, inf)))))

    def grid(self, gate) -> Grid:
        """Return the static grid of the group's histograms for a readout gate (ns): the model's axes, and on a timed
        readout the time nodes from the time spread's declared ranges and the gate (``lumen.readout.grid``)."""  # noqa: D205, D209
        axes = dict(axes=tuple(self.axes), delay_axes=tuple(self.delay.axes) if self.delay is not None else ())
        # the detection applies after tracking, per photon at normal incidence: the deposit over A_n
        norm = bool(self.efficiency_after_tracking)
        if self.readout == "pe":
            return Grid(axes=axes["axes"], normalise=norm)
        g0, g1 = (float(x) for x in gate)
        if not (math.isfinite(g0) and math.isfinite(g1) and g1 > g0):
            raise GateError(f"a readout gate is (start, end) ns with end > start, got {tuple(gate)}: a reversed or "
                            "zero-width gate reads no light")  # fmt: skip
        r = {f: self.ranges[f"tts/{f}"] for f in self.tts.fields() if f"tts/{f}" in self.ranges}
        lo, hi = self.tts.support(r)
        dmax = self.delay.support(None)[1] if self.delay is not None else 0.0
        off = self.ranges["t_offset"]
        dt = node_spacing(self.tts.min_width(r))
        if self.readout == "waveform":  # a whole number of nodes per front-end sample
            dt = self.sampling / math.ceil(self.sampling / dt - 1e-9)
        t_first = gate[0] - off[1] - hi - dmax
        t_last = gate[1] - off[0] - lo  # a photoelectron's time is arrival + t_offset + delay + spread
        nt = int(math.ceil((t_last - t_first) / dt)) + 1
        if self.delay is not None:
            self.delay.taps(dt)  # the delay's static taps, built once with the grid (cached on (edges, dt))
        return Grid(**axes, t0=float(t_first), dt=dt, nt=nt, weighted=True, normalise=norm)

    def acceptance(self, values):
        """Return the acceptance per charge cell (flat, ``Grid.n_cells``): the product of the acceptance tables, each
        broadcast over the cells (``tables``), of ``values`` (the group's split values; arrays, possibly traced).

        One path for concrete and traced values (float32, the tables multiplied in the model's order), so a run at a
        value and the primal of a derivative at it weigh the time histogram alike, bitwise; ``detection`` reads it too.
        """  # noqa: D205
        import jax.numpy as jnp

        shape = tuple(a.size for a in self.axes)
        names = [a.name for a in self.axes]
        w = jnp.ones(shape, jnp.float32)
        for t, tax in self.tables:
            v = jnp.asarray(values[t], jnp.float32)
            order = sorted(range(len(tax)), key=lambda i: names.index(tax[i]))  # the table's axes in the grid's order
            v = jnp.transpose(v, order) if len(tax) > 1 else v
            w = w * jnp.reshape(v, tuple(a.size if a.name in tax else 1 for a in self.axes))
        return w.ravel()

    def detection(self, v: dict, sensor, rows):
        """Return the detection probability of absorbed photons (JAX, per photon): efficiency x the acceptance tables.

        ``v``: the group's split values (``split``); ``sensor`` the photons' positions in the group; ``rows`` the
        deposit fields per photon (an object or dict with ``lam`` and the fields ``detection_reads`` names, leading
        shape that of ``sensor``). The tables are read as the charge histogram reads them: the product of the tables
        on the joint cells of the photon's axes (hat functions on knots, bands), so a truth row is detected with the
        probability the binned readout gives the same deposit.
        """
        import jax
        import jax.numpy as jnp

        f32 = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
        r = rows if isinstance(rows, dict) else {k: getattr(rows, k) for k in ("lam", *feats.FIELDS)
                                                   if getattr(rows, k, None) is not None}  # fmt: skip
        shape = jnp.shape(r["lam"])
        flat = {k: jnp.reshape(x, (-1, *jnp.shape(x)[len(shape) :])) for k, x in r.items()}
        table = jnp.asarray(self.acceptance({t: f32(v[t]) for t, _ in self.tables}), jnp.float32)
        used = {n for _, ax in self.tables for n in ax}
        axes = tuple(a for a in self.axes if a.name in used)
        stride = {a.name: math.prod(b.size for b in self.axes[i + 1 :]) for i, a in enumerate(self.axes)}

        def one(row):
            dep = _Row(**row)
            acc = 0.0
            for ci, cw in cells(axes, dep):  # the flat cell over the used axes -> the full grid's cell (others 0)
                idx = jnp.int32(0)
                rem = ci
                for a in reversed(axes):
                    idx = idx + (rem % a.size) * stride[a.name]
                    rem = rem // a.size
                acc = acc + jnp.where(ci >= 0, cw * table[jnp.maximum(idx, 0)], 0.0)
            return acc

        acc = jnp.reshape(jax.vmap(one)(flat), shape) if axes else jnp.ones(shape, jnp.float32)
        eff = jnp.atleast_1d(f32(v["efficiency"]))[jnp.clip(sensor, 0, jnp.size(f32(v["efficiency"])) - 1)]
        return eff * acc

    def normal_rows(self, lam) -> dict:
        """Return deposit rows at wavelengths ``lam`` at normal incidence on the sensor's axis (pos on +z, the
        direction against the normal, generation 0): where a QE is defined."""  # noqa: D205, D209
        import jax.numpy as jnp

        lam = jnp.asarray(lam, jnp.float32)
        e = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], jnp.float32), (*lam.shape, 3))
        return dict(lam=lam, pos=e, dir=-e, normal=e, gen=jnp.zeros(lam.shape, jnp.int32))

    def split(self, params, group: str) -> dict:
        """Return {component: {field: value}} and the top-level fields of a group read from a parameter mapping."""
        pre = f"sensors/{group}/"
        flat = {k: params[pre + k] for k in self.values}
        out: dict = {}
        for k, v in flat.items():
            head, _, rest = k.partition("/")
            if rest:
                out.setdefault(head, {})[rest] = v
            else:
                out[head] = v
        return out


def _ranges(d: dict, set_name: str, comps: dict, values: dict) -> dict:
    r = {k: tuple(float(x) for x in v) for k, v in d.get("ranges", {}).items()}
    if comps.get("tts") is None:
        return r
    declared = [k for k in WIDTHS if k in r]
    if declared:
        raise SensorValuesError(f"value set {set_name!r}: the range of {declared} is derived, not declared ([r, 1 / r] "
                                f"x the set's own width, r = {TIME_SPREAD_REACH}: lumen.rules.choices)")  # fmt: skip
    for k in WIDTHS:
        if k in values:
            w = float(values[k])
            r[k] = (w * TIME_SPREAD_REACH, w / TIME_SPREAD_REACH)
    need = list(GRID_RANGES) + [f"tts/{f}" for f in comps["tts"].fields() if f != "pdf"]
    if comps.get("pulse") is not None and comps["pulse"].form != "table":
        need += [f"pulse/{f}" for f in comps["pulse"].fields()]
    missing = [k for k in need if k not in r]
    if missing:
        raise SensorValuesError(f"value set {set_name!r}: declare the ranges of {missing} (they set the static "
                                "time grid; a value outside its range is refused)")  # fmt: skip
    return r


def _in_ranges(values: dict, ranges: dict, set_name: str) -> None:
    """Refuse a value set's own value outside its declared range (the time grid is built for the range)."""
    for k, (lo, hi) in ranges.items():
        if k not in values:
            raise SensorValuesError(f"value set {set_name!r}: a range for {k!r}, which is no value of the set (its "
                                    f"values: {sorted(values)})")  # fmt: skip
        v = np.asarray(values[k], np.float64)
        if np.any((v < lo) | (v > hi)):
            raise SensorValuesError(
                f"value set {set_name!r}: {k} = {v.tolist()} is outside its declared range [{lo}, {hi}]; the time "
                "grid resolves only the declared range (a time spread below its declared minimum is not resolved): "
                "declare a range holding it"
            )


class _Row(dict):
    """A deposit row as features read it (attribute access over a dict of fields)."""

    __getattr__ = dict.get


def _axis(key: str, values, name: str):
    """Return the Axis of a table key and its values (``<feature>_<unit>`` knots, ``<feature>_edges_<unit>`` band
    edges; the wavelength's bands are ``band_edges_nm``), or None when the key names no axis."""  # noqa: D205, D209
    _reserved(key, "axis", name)
    if key == "wavelength_edges_nm":
        raise SensorValuesError(f"value set {name!r}: the wavelength's bands are band_edges_nm")
    if key == "band_edges_nm":
        key = "wavelength_edges_nm"
    base, unit = key, ""
    for u in AXIS_UNITS:
        if key.endswith("_" + u):
            base, unit = key[: -len(u) - 1], u
    edges = base.endswith("_edges")
    feature = base[: -len("_edges")] if edges else base
    if feature not in feats.BUILTIN:
        return None
    x = tuple(float(AXIS_UNITS[unit](float(v))) if unit else float(v) for v in values)
    if edges:
        return Axis(feature, feats.BUILTIN[feature], edges=(-math.inf, *x, math.inf))
    try:
        return Axis(feature, feats.BUILTIN[feature], knots=x)
    except ValueError as e:
        raise SensorValuesError(f"value set {name!r}: {key}: {e}") from None


def _add_axis(axes: dict, ax, name: str) -> None:
    if ax.name in axes and axes[ax.name] != ax:
        raise SensorValuesError(f"value set {name!r}: axis {ax.name!r} is declared with two sets of knots or edges; an "
                                "axis appears with one (tables sharing it multiply on its knots)")  # fmt: skip
    ax.check()
    axes.setdefault(ax.name, ax)


def _record_axes(rec: dict, name: str, what: str) -> list:
    """Return the Axes a record declares by its keys, in their order."""
    out = []
    for k, v in rec.items():
        if k in ("values", "form", "pdf", "edges_ns"):
            continue
        ax = _axis(k, v, name)
        if ax is None:
            raise SensorValuesError(f"value set {name!r}: {what}.{k} names no axis (<feature>_<unit> knots or "
                                    f"<feature>_edges_<unit> edges; features {sorted(feats.BUILTIN)})")  # fmt: skip
        out.append(ax)
    return out


def _detection(d: dict, name: str, values: dict) -> tuple:
    """Return (axes, tables) of a value set's acceptance tables and extra axes, their values added to ``values``."""
    axes: dict = {}
    tables = []
    if d.get("qe") is not None and ("wavelength_nm" in d["qe"]) == ("band_edges_nm" in d["qe"]):
        raise SensorValuesError(f"value set {name!r}: qe needs exactly one of wavelength_nm (knots) or band_edges_nm")
    for t in TABLES:
        rec = d.get(t)
        if rec is None:
            continue
        tax = _record_axes(rec, name, t)
        for ax in tax:
            _add_axis(axes, ax, name)
        v = np.asarray(rec["values"], np.float64)
        want = tuple(a.size for a in tax)
        if v.shape != want:
            raise SensorValuesError(f"value set {name!r}: {t} has values of shape {v.shape} for its axes "
                                    f"{[a.name for a in tax]} of sizes {want}")  # fmt: skip
        values[t] = v
        tables.append((t, tuple(a.name for a in tax)))
    for ax in _record_axes(d.get("axes", {}), name, "axes"):
        _add_axis(axes, ax, name)
    return tuple(axes.values()), tuple(tables)


def _delay(spec: dict | None, name: str):
    """Return (DelayTable or None, {pdf: values}) of a delay record (its axes from its keys)."""
    if spec is None:
        return None, {}
    _reserved(spec.get("form", "table"), "delay form", name)
    if spec.get("form", "table") != "table":
        raise SensorValuesError(f"value set {name!r}: delay form {spec.get('form')!r} is not one of ['table']")
    edges = tuple(float(x) for x in spec["edges_ns"])
    if len(edges) < 2 or any(b < a for a, b in zip(edges, edges[1:], strict=False)) or edges[0] < 0:
        raise SensorValuesError(f"value set {name!r}: delay.edges_ns: at least two non-decreasing edges >= 0 (a "
                                "zero-width bin is a point mass)")  # fmt: skip
    comp = parts.DelayTable(edges, tuple(_record_axes(spec, name, "delay")))
    pdf = np.asarray(spec["pdf"], np.float64)
    if pdf.shape != comp.fields()["pdf"][1]:
        raise SensorValuesError(f"value set {name!r}: delay.pdf has shape {pdf.shape}, its axes and bins need "
                                f"{comp.fields()['pdf'][1]}")  # fmt: skip
    return comp, {"pdf": pdf}


def model_from_dict(d: dict) -> SensorValuesModel:
    """Return the SensorValuesModel of a parsed value set (``lumen-sensor-values/1``), checked."""
    name = str(d.get("name"))
    check_format(d.get("schema"), SCHEMA, f"value set {name!r}", "transcribe it again")
    unknown = sorted(set(d) - set(KEYS))
    if unknown:
        raise SensorValuesError(f"value set {name!r}: unknown records {unknown} (a value set's records: {KEYS})")
    if d.get("model") != "lumen:sensor":
        raise SensorValuesError(f"value set {name!r}: for model {d.get('model')!r}; this model is 'lumen:sensor'")
    status = d.get("provenance", {}).get("status")
    if status not in STATUSES or not d["provenance"].get("source"):
        raise SensorValuesError(f"value set {name!r}: provenance needs a source and a status in {STATUSES}")
    if d.get("readout") not in READOUTS:
        raise SensorValuesError(f"value set {name!r}: readout {d.get('readout')!r} is not one of {READOUTS}")
    values: dict = {}
    axes, tables = _detection(d, name, values)
    for k in ("efficiency", "t_offset_ns", "gain", "dark_rate_hz"):
        values[_strip(k)[0]] = float(d[k])
    units = {_strip(k)[0]: _strip(k)[1] for k in ("efficiency", "t_offset_ns", "gain", "dark_rate_hz")}
    forms = dict(tts=parts.TTS_FORMS, time_walk={"table": parts.TimeWalk}, charge=parts.CHARGE_FORMS,
                 threshold=parts.THRESHOLD_FORMS, pulse=parts.PULSE_FORMS)  # fmt: skip
    comps = {}
    for what, fm in forms.items():
        comps[what], vals = _component(d.get(what), fm, what, name)
        values.update({f"{what}/{k}": v for k, v in vals.items()})
    comps["delay"], vals = _delay(d.get("delay"), name)
    values.update({f"delay/{k}": v for k, v in vals.items()})
    if d["readout"] != "pe" and (comps["tts"] is None or comps["charge"] is None):
        raise SensorValuesError(f"value set {name!r}: readout {d['readout']!r} needs a tts and a charge record")
    fe = d["frontend"]
    values.update({f"frontend/{_strip(k)[0]}": float(v) for k, v in fe.items() if _strip(k)[0] in FRONTEND_VALUES})
    units.update({f"frontend/{_strip(k)[0]}": _strip(k)[1] for k in fe if _strip(k)[0] in FRONTEND_VALUES})
    units.update(
        {f"{c}/{f}": u for c, comp in comps.items() if comp is not None for f, (u, _) in comp.fields().items()}
    )
    units.update(dict.fromkeys((k for k in IN_READOUT_UNIT if k in values), fe["unit"]))
    ranges = _ranges(d, name, comps, values)
    _in_ranges(values, ranges, name)
    model = SensorValuesModel(set_name=name, readout=d["readout"], axes=axes, tables=tables,
                     sampling=float(fe["sampling_ns"]), unit=fe["unit"], charge_step=float(fe["charge_step_pe"]),
                     charge_max=float(fe["charge_max_pe"]), values=values, ranges=ranges,
                     provenance=d["provenance"], units=units, **comps)  # fmt: skip
    if status == "unverified" or d["provenance"].get("warning"):
        warnings.warn(f"sensor values {name}: {d['provenance'].get('warning') or 'status unverified'}", stacklevel=3)
    return model


def _path(ref: str) -> Path:
    """Return the file of a value-set reference: a local path, else ``<package>:<set>``, the file ``<set>.json`` in the
    importable package's ``sensor_values`` directory (an experiment's package ships its sets so; lumen ships none)."""  # noqa: D205, D209
    if Path(ref).exists() or ":" not in ref:
        return Path(ref)
    pkg, _, name = ref.partition(":")
    try:
        root = resources.files(pkg)
    except ModuleNotFoundError:
        raise SensorValuesError(f"{ref!r}: no package {pkg!r} to read its value set from (install it, or give a "
                                "path)") from None  # fmt: skip
    p = Path(str(root)) / "sensor_values" / f"{name}.json"
    if not p.exists():
        have = sorted(q.stem for q in p.parent.glob("*.json"))
        raise SensorValuesError(f"no sensor value set {ref!r}; {pkg} ships: {have}")
    return p


def read_values(ref: str) -> dict:
    """Return the parsed value set of a reference (a path to a JSON file, or ``<package>:<set>``)."""
    return json.loads(_path(ref).read_text())


def load_model(ref: str) -> SensorValuesModel:
    """Return the SensorValuesModel of a value set (a path or ``<package>:<set>``), read now (``model_of``)."""
    return model_of(read_values(ref))


def model_of(values: dict) -> SensorValuesModel:
    """Return the SensorValuesModel of a parsed value set: one object per value set's content (a bounded cache keyed
    by its canonical JSON, so an edited file or a fit's new values make a new model, never a stale one)."""  # noqa: D205, D209
    return _model(json.dumps(values, sort_keys=True))


@lru_cache(maxsize=64)
def _model(canonical: str) -> SensorValuesModel:
    return model_from_dict(json.loads(canonical))


def write_defaults(ref: str, out) -> Path:
    """Copy a value set to ``out``, to edit it into a sensor's own values (set ``provenance`` to your source)."""
    out = Path(out)
    shutil.copyfile(_path(ref), out)
    return out
