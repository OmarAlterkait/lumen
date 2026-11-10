"""The ``lumen:pmt`` sensor model: a PMT's detection and electronics after tracking, and its value sets.

    model = lumen.sensors.load_model("lumen:sk_r3600@2")        # a shipped value set, or a path to a JSON file
    p = model.defaults("pmt", n_sensors)                        # {"sensors/pmt/<field>": array}
    lumen.sensors.write_defaults("lumen:sk_r3600@2", "my_r3600.json")

A value set (``lumen-pmt-values/1``) holds structure (forms, knot positions, table lengths, the front end's sampling
step and charge grid) and values. Structure is static: it fixes the histograms' grids and the compiled readout.
Values become the parameters ``sensors/<group>/<field>`` and are traced, so changing one never recompiles. File
fields carry their unit as a suffix (``sigma_ns``), parameter keys do not (``sensors/pmt/tts/sigma``, in ns).

Detection: eta(lambda, theta) = efficiency * QE(lambda) * CE(theta), QE on wavelength knots (Geant4's linear
interpolation, constant beyond the end knots) or per wavelength band, CE on hit-angle knots (degrees in the file,
radians in the deposit). Electronics: TTS, an optional fixed delay, the transit offset ``t_offset``, an optional time
walk, the SPE charge, the threshold, the pulse, the dark rate and the front end (see ``lumen.sensors.components``).
The per-sensor fields (``efficiency``, ``t_offset``, ``gain``, ``dark_rate``) are arrays over the group's sensors.
"""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from lumen.sensors import components as parts

SCHEMA = "lumen-pmt-values/1"
STATUSES = ("measured", "transcribed", "unverified")
PER_SENSOR = ("efficiency", "t_offset", "gain", "dark_rate")
SUFFIXES = ("_ns", "_pe", "_hz", "_nm", "_deg")
READOUTS = ("pe", "hits", "waveform")
FRONTEND_VALUES = ("baseline", "noise", "window", "charge_noise", "charge_scale")
UNITS = {"ns": "ns", "pe": "pe", "hz": "Hz", "nm": "nm", "deg": "deg"}
DEFAULT_RANGES = {"qe": (0.0, 1.0), "ce": (0.0, 1.0), "efficiency": (0.0, 1.0), "dark_rate": (0.0, 1e9)}
GRID_RANGES = ("t_offset",)  # declared ranges a value set must give (with every tts/ and pulse/ value)


class SensorValuesError(ValueError):
    """A sensor value set that does not follow the ``lumen-pmt-values/1`` rules."""


def _strip(name: str) -> tuple[str, str]:
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)], UNITS[s[1:]]
    return name, ""


def _component(spec: dict | None, forms: dict, what: str, set_name: str):
    """Return (component, {field: value}) of a component record, or (None, {})."""
    if spec is None:
        return None, {}
    form = spec.get("form")
    if form not in forms:
        raise SensorValuesError(f"value set {set_name!r}: {what} form {form!r} is not one of {sorted(forms)}")
    static = {k: v for k, v in spec.items() if k in ("edges_ns", "edges_pe", "charge_pe", "t_first_ns", "step_ns")}
    if form == "table" and what == "pulse":
        comp = parts.TablePulse(float(spec["t_first_ns"]), float(spec["step_ns"]), len(spec["shape"]))
    elif form == "table":
        comp = forms[form](tuple(float(x) for x in next(iter(static.values()))))
    elif form == "poly":
        comp = forms[form](len(spec["coefficients"]) - 1)
    else:
        comp = forms[form]()
    vals = {_strip(k)[0]: np.asarray(v, np.float64) for k, v in spec.items() if k != "form" and k not in static}
    if set(vals) != set(comp.fields()):
        raise SensorValuesError(f"value set {set_name!r}: {what} ({form}) needs the fields {sorted(comp.fields())}, "
                                f"got {sorted(vals)}")  # fmt: skip
    return comp, vals


@dataclass(frozen=True, eq=False)
class PMTModel:
    """The ``lumen:pmt`` model of one value set: static structure plus its default values and declared ranges."""

    set_name: str
    readout: str
    qe_knots: tuple | None
    qe_bands: tuple | None
    ce_knots: tuple | None
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
    provenance: dict = field(default_factory=dict)
    name = "lumen:pmt"
    version = 1

    def components(self) -> dict:
        """Return {name: component} of the components present."""
        names = ("tts", "delay", "time_walk", "charge", "threshold", "pulse")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}

    def defaults(self, group: str, n_sensors: int) -> dict:
        """Return the default parameter values ``sensors/<group>/<field>`` (per-sensor fields as (n_sensors,))."""
        out = {}
        for k, v in self.values.items():
            v = np.full(n_sensors, float(v)) if k in PER_SENSOR else np.asarray(v, np.float64)
            out[f"sensors/{group}/{k}"] = v
        return out

    def param_info(self, group: str) -> dict:
        """Return ``sensors/<group>/<field>`` -> lumen.params.ParamInfo (unit, range, after-tracking method)."""
        from lumen.params import ParamInfo

        units = {f"{c}/{f}": u for c, comp in self.components().items() for f, (u, _) in comp.fields().items()}
        units.update(t_offset="ns", dark_rate="Hz", qe="", ce="", efficiency="", gain=self.unit)
        fe = {"window": "ns", "charge_noise": "pe", "charge_scale": ""}
        units.update({f"frontend/{k}": fe.get(k, self.unit) for k in FRONTEND_VALUES})
        inf = float("inf")
        return {f"sensors/{group}/{k}": ParamInfo(f"sensors/{group}/{k}", units.get(k, ""),
                                                  tuple(self.ranges.get(k, DEFAULT_RANGES.get(k, (-inf, inf)))),
                                                  "after_tracking") for k in self.values}  # fmt: skip

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


def _ranges(d: dict, set_name: str, comps: dict) -> dict:
    r = {k: tuple(float(x) for x in v) for k, v in d.get("ranges", {}).items()}
    if comps.get("tts") is None:
        return r
    need = list(GRID_RANGES) + [f"tts/{f}" for f in comps["tts"].fields() if f != "pdf"]
    if comps.get("pulse") is not None and comps["pulse"].form != "table":
        need += [f"pulse/{f}" for f in comps["pulse"].fields()]
    missing = [k for k in need if k not in r]
    if missing:
        raise SensorValuesError(f"value set {set_name!r}: declare the ranges of {missing} (they set the static "
                                "time grid; a value outside its range is refused)")  # fmt: skip
    return r


def _detection(d: dict, name: str, values: dict) -> tuple:
    qe = d["qe"]
    if ("wavelength_nm" in qe) == ("band_edges_nm" in qe):
        raise SensorValuesError(f"value set {name!r}: qe needs exactly one of wavelength_nm (knots) or band_edges_nm")
    knots = tuple(float(x) for x in qe.get("wavelength_nm", ()) or ()) or None
    bands = tuple(float(x) for x in qe.get("band_edges_nm", ()) or ()) or None
    n = len(knots) if knots else len(bands) - 1
    if len(qe["values"]) != n:
        raise SensorValuesError(f"value set {name!r}: qe has {len(qe['values'])} values for {n} knots or bands")
    values["qe"] = np.asarray(qe["values"], np.float64)
    ce = d.get("ce")
    ce_knots = None
    if ce is not None:
        ce_knots = tuple(float(x) * np.pi / 180.0 for x in ce["theta_deg"])
        values["ce"] = np.asarray(ce["values"], np.float64)
    return knots, bands, ce_knots


def model_from_dict(d: dict) -> PMTModel:
    """Return the PMTModel of a parsed value set (``lumen-pmt-values/1``), checked."""
    name = f"{d.get('name')}@{d.get('version')}"
    if d.get("schema") != SCHEMA or d.get("model") != "lumen:pmt":
        raise SensorValuesError(f"value set {name!r}: schema {d.get('schema')!r} for model {d.get('model')!r}; this "
                                f"lumen reads {SCHEMA} for 'lumen:pmt'")  # fmt: skip
    status = d.get("provenance", {}).get("status")
    if status not in STATUSES or not d["provenance"].get("source"):
        raise SensorValuesError(f"value set {name!r}: provenance needs a source and a status in {STATUSES}")
    if d.get("readout") not in READOUTS:
        raise SensorValuesError(f"value set {name!r}: readout {d.get('readout')!r} is not one of {READOUTS}")
    values: dict = {}
    knots, bands, ce_knots = _detection(d, name, values)
    for k in ("efficiency", "t_offset_ns", "gain", "dark_rate_hz"):
        values[_strip(k)[0]] = float(d[k])
    forms = dict(tts=parts.TTS_FORMS, delay={"table": parts.DelayTable}, time_walk={"table": parts.TimeWalk},
                 charge=parts.CHARGE_FORMS, threshold=parts.THRESHOLD_FORMS, pulse=parts.PULSE_FORMS)  # fmt: skip
    comps = {}
    for what, fm in forms.items():
        spec = d.get(what)
        if what == "time_walk" and spec is not None:
            comps[what] = parts.TimeWalk(tuple(float(x) for x in spec["charge_pe"]))
            vals = {"shift": np.asarray(spec["shift_ns"], np.float64), "sigma": np.asarray(spec["sigma_ns"])}
        else:
            comps[what], vals = _component(spec, fm, what, name)
        values.update({f"{what}/{k}": v for k, v in vals.items()})
    if d["readout"] != "pe" and (comps["tts"] is None or comps["charge"] is None):
        raise SensorValuesError(f"value set {name!r}: readout {d['readout']!r} needs a tts and a charge record")
    fe = d["frontend"]
    values.update({f"frontend/{_strip(k)[0]}": float(v) for k, v in fe.items() if _strip(k)[0] in FRONTEND_VALUES})
    model = PMTModel(set_name=name, readout=d["readout"], qe_knots=knots, qe_bands=bands, ce_knots=ce_knots,
                     sampling=float(fe["sampling_ns"]), unit=fe["unit"], charge_step=float(fe["charge_step_pe"]),
                     charge_max=float(fe["charge_max_pe"]), values=values, ranges=_ranges(d, name, comps),
                     provenance=d["provenance"], **comps)  # fmt: skip
    if status == "unverified" or d["provenance"].get("warning"):
        warnings.warn(f"sensor values {name}: {d['provenance'].get('warning') or 'status unverified'}", stacklevel=3)
    return model


def _path(ref: str) -> Path:
    """Return the file of a value-set reference: ``lumen:<set>@<version>`` (package data) or a local path."""
    if ref.startswith("lumen:"):
        stem, _, ver = ref[len("lumen:") :].partition("@")
        frozen = Path(str(resources.files("lumen.sensors") / "data" / f"{stem}@{ver}.json"))
        if ver and frozen.exists():  # an earlier version, kept as shipped
            return frozen
        p = Path(str(resources.files("lumen.sensors") / "data" / f"{stem}.json"))
        if not p.exists():
            have = sorted(q.stem for q in p.parent.glob("*.json"))
            raise SensorValuesError(f"no shipped sensor value set {stem!r}; shipped: {have}")
        if ver and json.loads(p.read_text()).get("version") != int(ver):
            raise SensorValuesError(f"{ref}: the shipped set {stem!r} is at version "
                                    f"{json.loads(p.read_text()).get('version')}")  # fmt: skip
        return p
    if ":" in ref.split("@")[0] and not Path(ref).exists():
        raise SensorValuesError(f"{ref!r}: value sets of other distributions come from their own packages")
    return Path(ref)


@lru_cache(maxsize=64)
def load_model(ref: str) -> PMTModel:
    """Return the PMTModel of a value set (``lumen:<set>@<version>`` or a path); the same reference, the same object."""
    return model_from_dict(json.loads(_path(ref).read_text()))


def write_defaults(ref: str, out) -> Path:
    """Copy a value set to ``out``, to edit it into your own PMT's values (set ``provenance`` to your source)."""
    out = Path(out)
    shutil.copyfile(_path(ref), out)
    return out
