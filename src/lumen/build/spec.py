"""The parameter spec, computed at load: every parameter key of a detector, with its unit, range and gradient method.

Keys come from the detector file (every material and surface property, and each surface's sigma_alpha or polish) and
from the sensor set (each group's model fields). What each Geant4 property means is in ``PROPERTIES``; a key it does not
list is kept, as a dimensionless value of unknown range, so a calibration can still carry it. A value the model never
reads is not a parameter (structure decides, never the value: docs/explanation/parameters.md): the Mie constants of a
material without MIEHG, a Mie forward ratio of 0 or 1 whose other lobe the material lacks
(``lumen.rules.optics.mie_constants``), and a surface's sigma_alpha or polish where no facet is sampled
(``value_read``).
"""

from __future__ import annotations

import math

import numpy as np

from lumen.format import schema as fmt
from lumen.format.calibrations import Table
from lumen.params import ParamInfo, Params

INF = math.inf
FRACTION = ("", (0.0, 1.0), "reweight")
LENGTH = ("m", (0.0, INF), "reweight")
MIE_G = ("", (-0.999, 0.999), "reweight")  # Geant4 allows a negative g (a flipped lobe); |g| <= 0.999 for float32
# Geant4 material and surface properties: (unit, allowed range, gradient method). RINDEX moves light paths and changes
# Fresnel weights: its derivative is pending validation.
PROPERTIES = {
    "RINDEX": ("", (1.0, INF), "moving_paths+reweight"),
    "ABSLENGTH": LENGTH, "RAYLEIGH": LENGTH, "WLSABSLENGTH": LENGTH, "MIEHG": LENGTH,
    "REFLECTIVITY": FRACTION, "TRANSMITTANCE": FRACTION, "EFFICIENCY": FRACTION,
    "SPECULARLOBECONSTANT": FRACTION, "SPECULARSPIKECONSTANT": FRACTION, "BACKSCATTERCONSTANT": FRACTION,
    "DIFFUSE_FRACTION": FRACTION,
    "MIEHG_FORWARD": MIE_G, "MIEHG_BACKWARD": MIE_G, "MIEHG_FORWARD_RATIO": FRACTION,
    "WLSCOMPONENT": ("", (0.0, INF), "reweight"), "WLSTIMECONSTANT": ("ns", (0.0, INF), "reweight"),
    "WLSMEANNUMBERPHOTONS": ("", (0.0, INF), "reweight"), "GROUPVEL": ("mm/ns", (0.0, INF), "reweight"),
    "REALRINDEX": ("", (0.0, INF), "reweight"), "IMAGINARYRINDEX": ("", (0.0, INF), "reweight"),
}  # fmt: skip
UNKNOWN = ("", (-INF, INF), "reweight")
VALUE_PARAMS = {"sigma_alpha": ("rad", (0.0, INF)), "polish": ("", (0.0, 1.0))}  # an opticalsurface's value
PENDING = {"moving_paths+reweight"}  # methods not validated yet: asking for their derivative raises


def _info(key, prop, table):
    unit, rng, method = PROPERTIES.get(prop, UNKNOWN)
    return ParamInfo(key, unit, rng, method, method not in PENDING, table)


def _value(p: fmt.Property):
    if p.energy is None:
        return float(np.asarray(p.value))
    return Table(np.asarray(p.energy), np.asarray(p.value))


def not_parameters(rec) -> frozenset:
    """Return the properties of a material or surface record whose value the model fixes (no parameter key)."""
    if not isinstance(rec, fmt.Material):
        return frozenset()
    from lumen.rules import optics

    return optics.mie_constants(rec.properties, f"material {rec.name!r}")[1]


def value_read(rec: fmt.Surface) -> bool:
    """Whether the model reads a surface's value (sigma_alpha or polish): Geant4's dielectric_metal, ground finish.

    Unified reads sigma_alpha in GetFacetNormal for a lobe reflection, glisur its polish. A polished or painted
    finish never samples a facet, and lumen's dielectric_dielectric surfaces are polished or painted; LArSoft's
    OpBoundaryProcessSimple (``lumen:larsoft_reflector``) and a WLS film's record read no value.
    """
    from lumen.rules import optics

    if rec.optics.name != "geant4":
        return False
    _, finish, type_ = optics.surface_names(rec)
    return type_ == "dielectric_metal" and finish == "ground"


def detector_values(df: fmt.DetectorFile) -> tuple[dict, dict]:
    """Return (values, spec) of the detector file's own parameters."""
    values, spec = {}, {}
    for kind, items in (("materials", df.materials), ("surfaces", df.surfaces)):
        for name, rec in items.items():
            fixed = not_parameters(rec)
            for prop, p in rec.properties.items():
                if prop in fixed:
                    continue
                key = f"{kind}/{name}/{prop}"
                values[key] = _value(p)
                spec[key] = _info(key, prop, p.energy is not None)
            if isinstance(rec, fmt.Surface) and value_read(rec):
                from lumen.rules.optics import value_key

                vk = value_key(rec)
                key = f"surfaces/{name}/{vk}"
                values[key] = float(rec.value)
                spec[key] = ParamInfo(key, *VALUE_PARAMS[vk], "reweight")
    return values, spec


def default_values(df: fmt.DetectorFile) -> tuple[dict, dict]:
    """Return (values, spec) of the surface properties the file omits whose Geant4 default the model reads.

    ``lumen.rules.optics.detector_defaults``: each key at Geant4's default, its ParamInfo naming the default and the
    source line; ``lumen.format.apply`` adds the property when a value is set.
    """
    from lumen.rules import optics

    values, spec = {}, {}
    for name, keys in optics.detector_defaults(df).items():
        for prop, (v, src) in keys.items():
            key = f"surfaces/{name}/{prop}"
            values[key] = float(v)
            info = _info(key, prop, False)
            spec[key] = ParamInfo(key, info.unit, info.range, info.method, info.available, False,
                                  f"default {v:g}, {src}")  # fmt: skip
    return values, spec


def model_values(df: fmt.DetectorFile) -> tuple[dict, dict, dict]:
    """Return (values, spec, layers) of the parameters registered optics models declare (``lumen.models``).

    A declared name the record has as a property keeps the file's value (and its layer); else it is the default.
    """
    from lumen import models

    values, spec, layers = {}, {}, {}
    for kind, section, items in (("medium", "materials", df.materials), ("surface", "surfaces", df.surfaces)):
        for name, rec in items.items():
            m = models.registered(rec.optics.name, kind)
            if m is None:
                continue
            for k, p in m.declared:
                key = f"{section}/{name}/{k}"
                spec[key] = p.info(key, "reweight", f"declared by {m.name}")
                if k not in rec.properties:
                    values[key], layers[key] = p.default, f"model {m.name} default"
    return values, spec, layers


def param_spec_and_values(df: fmt.DetectorFile, sensor_defaults: dict | None = None) -> Params:
    """Return the detector's Params at its own values, then each sensor group's model defaults.

    ``sensor_defaults`` maps ``sensors/<group>/<field>`` to (value, ParamInfo).
    """
    values, spec = detector_values(df)
    layers = dict.fromkeys(values, "detector file")
    dv, ds = default_values(df)
    values, spec = {**values, **dv}, {**spec, **ds}
    layers.update({k: f"Geant4 {ds[k].default}" for k in dv})
    mv, ms, ml = model_values(df)
    values, spec = {**values, **mv}, {**spec, **ms}
    layers.update(ml)
    for key, (v, info) in (sensor_defaults or {}).items():
        values[key], spec[key], layers[key] = v, info, "sensor model defaults"
    return Params(values, spec, layers)


def resolve(params: Params, calibrations) -> Params:
    """Return params with calibrations applied, oldest first (``lumen.format.calibrations.lineage``)."""
    for c in calibrations:
        unknown = [k for k in c.values if k not in params]
        if unknown:
            raise KeyError(f"calibration {c.name!r}: unknown keys {unknown[:5]}")
        params = params.with_layer(c.values, f"calibration {c.name}")
    return params
