"""The parameter spec, computed at load: every parameter key of a detector, with its unit, range and gradient method.

Keys come from the detector file and from the sensor set (each group's model fields). A file key is a parameter where
the built optics reads it (``Parts.key_rows``: the library's one record of where each packed value comes from), or, on a
material that is not bound opaque, where a light source reads it (``lumen.rules.keys.SOURCE_KEYS``): a key nothing reads
(a surface's EFFICIENCY where no sensor is bound, GROUPVEL) is no parameter, so no calibration can carry a value pack
would refuse. Its derivative is available where its method is (``lumen.params.PENDING``): every read is the value itself
(``library.DIRECT``) or a JAX function of it (a derived Rayleigh table). What each Geant4 property means is in
``lumen.rules.keys.PARAMS``; a key it does not list is a dimensionless value of unknown range. A value the model never
reads is not a parameter (structure decides, never the value: docs/explanation/parameters.md): the Mie constants of a
material without MIEHG, a Mie forward ratio of 0 or 1 whose other lobe the material lacks
(``lumen.rules.optics.mie_constants``), and a surface's sigma_alpha or polish where its reading samples no facet
(``lumen.rules.optics.reads_value``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen.format import schema as fmt
from lumen.format.calibrations import Table
from lumen.params import Param, Params, UnknownParamKey
from lumen.rules import optics
from lumen.rules.keys import PARAMS, SOURCE_KEYS, UNKNOWN
from lumen.rules.optics import reads_value, value_key

OPAQUE = "lumen:opaque"  # the medium no light leaves (lumen.build.library)


def info(prop) -> Param:
    """Return a key's declaration (``lumen.rules.keys.PARAMS``; an undeclared key: any value)."""
    return PARAMS.get(prop, UNKNOWN)


def _frozen(a):
    """Return a read-only copy: a write to a parameter's array raises instead of changing the detector's table."""
    a = np.array(a)
    a.setflags(write=False)
    return a


def _value(p: fmt.Property):
    if p.energy is None:
        return float(np.asarray(p.value))
    return Table(_frozen(p.energy), _frozen(p.value))


def not_parameters(rec) -> frozenset:
    """Return the properties of a material or surface record whose value the model fixes (no parameter key)."""
    if not isinstance(rec, fmt.Material):
        return frozenset()

    return optics.mie_constants(rec.properties, f"material {rec.name!r}")[1]


def read_columns(key_rows) -> dict:
    """Return {key: the columns the built optics reads it through} (``Parts.key_rows``; values it never reads out)."""
    out: dict = {}
    for src in key_rows.values():
        if src.key and src.column != "fixed":
            out.setdefault(src.key, set()).add(src.column)
    return out


def detector_values(df: fmt.DetectorFile, key_rows: dict) -> tuple[dict, dict]:
    """Return (values, spec) of the detector file's own parameters that the built optics (its ``key_rows``) or a light
    source reads (see the module docstring)."""  # noqa: D205, D209
    read = read_columns(key_rows)
    values, spec = {}, {}
    for kind, items in (("materials", df.materials), ("surfaces", df.surfaces)):
        for name, rec in items.items():
            fixed = not_parameters(rec)
            emits = kind == "materials" and rec.optics.name != OPAQUE  # light born in an opaque material ends there
            props = [(prop, p.energy is not None, _value(p)) for prop, p in rec.properties.items() if prop not in fixed]
            if isinstance(rec, fmt.Surface) and reads_value(rec):
                props.append((value_key(rec), False, float(rec.value)))
            for prop, _, v in props:
                key = f"{kind}/{name}/{prop}"
                if key not in read and not (emits and prop in SOURCE_KEYS):
                    continue  # nothing reads it
                values[key], spec[key] = v, info(prop)
    return values, spec


def default_values(df: fmt.DetectorFile) -> tuple[dict, dict]:
    """Return (values, spec) of the surface properties the file omits whose Geant4 default the model reads.

    ``lumen.rules.optics.detector_defaults``: each key at Geant4's default, its Param naming the default and the
    source line; ``lumen.format.apply`` adds the property when a value is set.
    """
    values, spec = {}, {}
    for name, keys in optics.detector_defaults(df).items():
        for prop, (v, src) in keys.items():
            key = f"surfaces/{name}/{prop}"
            values[key] = float(v)
            spec[key] = dataclasses.replace(info(prop), note=f"default {v:g}, {src}")
    return values, spec


def model_values(df: fmt.DetectorFile, models=None) -> tuple[dict, dict, dict]:
    """Return (values, spec, layers) of the parameters the build's medium models declare.

    ``models``: {name: MediumModel}, the build's (``lumen.models``).

    A declared name the record has as a property keeps the file's value (and its layer); else it is the default.
    """
    values, spec, layers = {}, {}, {}
    for rec in df.materials.values():
        m = (models or {}).get(rec.optics.name)
        keys = m.keys(rec) if m is not None else {}  # the model's own key of each name (a part's: under its channel)
        for k, p in m.declared if m is not None else ():
            key, column, _ = keys[k]
            spec[key] = dataclasses.replace(p, method="reweight", note=f"declared by {m.name}")
            if column == "default":
                values[key], layers[key] = p.default, f"model {m.name} default"
    return values, spec, layers


def param_spec_and_values(df: fmt.DetectorFile, sensor_defaults: dict, key_rows: dict, models: dict) -> Params:
    """Return the detector's Params at its own values, then each sensor group's model defaults.

    ``sensor_defaults`` maps ``sensors/<group>/<field>`` to (value, Param); ``key_rows`` is the built optics'
    ``Parts.key_rows`` (the keys it reads); ``models`` the build's medium models.
    """
    values, spec = detector_values(df, key_rows)
    layers = dict.fromkeys(values, "detector file")
    dv, ds = default_values(df)
    values, spec = {**values, **dv}, {**spec, **ds}
    layers.update({k: f"Geant4 {ds[k].note}" for k in dv})
    mv, ms, ml = model_values(df, models)
    values, spec = {**values, **mv}, {**spec, **ms}
    layers.update(ml)
    for key, (v, info) in sensor_defaults.items():
        values[key], spec[key], layers[key] = v, info, "sensor model defaults"
    return Params(values, spec, layers)


def resolve(params: Params, calibration=None) -> Params:
    """Return params with a calibration's values applied (None: the nominal values).

    A key the detector does not have raises ``UnknownParamKey`` naming the nearest keys (a response's key needs the
    response: ``lumen.load(..., responses=)``).
    """
    if calibration is None:
        return params
    c = calibration
    unknown = [k for k in c.values if k not in params]
    if unknown:
        raise UnknownParamKey(f"calibration {c.name!r}: {params._unknown(unknown[0])} ({len(unknown)} unknown "
                              "keys; a response's keys need it at load: lumen.load(..., responses=))")  # fmt: skip
    return params.with_layer(c.values, f"calibration {c.name}")
