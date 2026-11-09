"""The parameter spec, computed at load: every parameter key of a detector, with its unit, range and gradient method.

Keys come from the detector file (every material and surface property, and each surface's value) and from the sensor
set (each group's model fields). What each Geant4 property means is in ``PROPERTIES``; a key it does not list is
kept, as a dimensionless value of unknown range, so a calibration can still carry it. A property whose value the
model fixes is not a parameter: the Mie constants of a material without MIEHG, and a Mie forward ratio of 0 or 1
whose other lobe the material lacks (``lumen.rules.optics.mie_constants``).
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
            if isinstance(rec, fmt.Surface):
                key = f"surfaces/{name}/value"
                values[key] = float(rec.value)
                spec[key] = ParamInfo(key, "rad", (0.0, INF), "reweight")
    return values, spec


def param_spec_and_values(df: fmt.DetectorFile, sensor_defaults: dict | None = None) -> Params:
    """Return the detector's Params at its own values, then each sensor group's model defaults.

    ``sensor_defaults`` maps ``sensors/<group>/<field>`` to (value, ParamInfo).
    """
    values, spec = detector_values(df)
    layers = dict.fromkeys(values, "detector file")
    for key, (v, info) in (sensor_defaults or {}).items():
        values[key], spec[key], layers[key] = v, info, "sensor model defaults"
    return Params(values, spec, layers)


def resolve(params: Params, calibrations) -> Params:
    """Return params with a calibration chain applied, oldest first (``lumen.format.calibrations.chain``)."""
    for c in calibrations:
        unknown = [k for k in c.values if k not in params]
        if unknown:
            raise KeyError(f"calibration {c.name!r}: unknown keys {unknown[:5]}")
        params = params.with_layer(c.values, f"calibration {c.name}")
    return params
