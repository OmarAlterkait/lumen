"""Sensor models: the plug-in point between a detecting optical surface and the surface type lumen tracks with.

A model is any callable

    model(surface, lam_domain) -> SensorEntry

``surface`` is a ``lumen.format.schema.Surface`` record and ``lam_domain`` the run's (lo, hi) wavelengths (nm). It
returns a surface type that can DETECT and its parameters (scalars, or (law, parameters) pairs the region compiler
packs). A recipe or a user registers one by name (``register``) and binds it per sensor
(``lumen.rules.optics.sensor(surface, model=...)``); nothing in the core names a detector.

Built in:

geant4 (the default)
    the surface as stock Geant4 (G4OpBoundaryProcess, 10.7 / 11) reads it. dielectric_metal: REFLECTIVITY R (absent:
    the complex index REALRINDEX + i IMAGINARYRINDEX, CalculateReflectivity, a PHOTOCATHODE; neither: R = 1), the
    absorbed 1 - R detected with EFFICIENCY (default 0); the reflection by model and finish: polished (glisur or
    unified) a mirror (SENSOR, f_spec 1), glisur ground the facet chain (GLISUR_METAL_EFF), unified ground spike /
    lobe / back / Lambert (DIELECTRIC_METAL). dielectric_dielectric, polished, with R / T tables:
    DIELECTRIC_RT_SENSOR, eff = EFFICIENCY. Keys no stock process reads on a surface (RINDEX, KINDEX, THICKNESS on
    a dielectric_metal) are ignored, as Geant4 ignores them.
complex_index
    a thin-film photocathode's tables some experiment software writes on the surface (RINDEX, KINDEX), read as a
    semi-infinite absorber of index RINDEX + i KINDEX (PHOTOCATHODE, specular, detected with EFFICIENCY): not the
    film's interference, an approximation of it.

The complex index of either is taken at the range's centre and must be flat over the range (refused otherwise:
narrow the range or supply a spectral model). A sensitive detector that counts every absorption (WCSim's) replaces
the efficiency: ``lumen.rules.optics.sensor(..., eff=1)``.
"""

from __future__ import annotations

import numpy as np

from lumen import plugins
from lumen.build._legacy_regions import DescriptionError, SensorEntry
from lumen.format import schema
from lumen.rules import optics as rules
from lumen.surfaces import _legacy as legacy
from lumen.units import HC

DEFAULT = "lumen:geant4"
PREFIX = "lumen:"  # lumen's own models; a bare name means one of them
FLAT = 1e-3  # the relative variation over the range a complex-index table may have (it is taken at the centre)
TINY = 1e-12  # the floor of the scale of that variation


class SensorModelError(DescriptionError):
    """A sensor model cannot read a surface."""


def register(name: str, model):
    """Register a sensor model in ``lumen.plugins.sensor_models``: ``model(surface, lam_domain) -> SensorEntry``."""
    if not callable(model):
        raise SensorModelError(f"sensor model {name!r}: not callable")
    plugins.sensor_models.register(str(name), model)
    return model


def get(name=None):
    """Return the sensor model of that name (None: the default; a callable is returned as it is)."""
    name = DEFAULT if name is None else name
    if callable(name):
        return name
    for candidate in (name, PREFIX + name) if ":" not in name else (name,):
        if candidate in plugins.sensor_models.names():
            return plugins.sensor_models.get(candidate)
    raise SensorModelError(f"sensor model {name!r}: not registered ({plugins.sensor_models.names()})")


def at_centre(surface: schema.Surface, key: str, lam_domain) -> float:
    """Return a property at the range's centre (Geant4: linear in energy, clamped), refusing one not flat over it."""
    pr = surface.properties[key]
    if not rules.is_table(pr):
        return float(pr.value)
    e = HC / np.asarray(lam_domain, np.float64)
    v = np.interp(np.r_[e, 0.5 * (e[0] + e[1])], pr.energy, pr.value)
    c = float(np.interp(HC / (0.5 * (lam_domain[0] + lam_domain[1])), pr.energy, pr.value))
    if np.ptp(np.r_[v, c]) > FLAT * max(abs(c), TINY):
        raise SensorModelError(f"opticalsurface {surface.name!r}: {key} varies by {np.ptp(np.r_[v, c]):.3g} over "
                               f"{tuple(lam_domain)} nm: a complex index is taken at one wavelength (narrow the range "
                               "or supply a spectral model)")  # fmt: skip
    return c


def flat_fraction(surface: schema.Surface, key: str, default: float):
    """Return a fraction property: a constant, a flat table as its constant, else the spectral (law, parameters)."""
    pr = surface.properties.get(key)
    if pr is not None and rules.is_table(pr) and np.ptp(np.asarray(pr.value)) == 0.0:
        return float(np.asarray(pr.value)[0])
    return rules.fraction(surface.properties, key, default)


def _mirror(refl, eff):
    return SensorEntry(legacy.SENSOR, {"R": refl, "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 0.0, "qe": eff})


def _ground_metal(s: schema.Surface, model: str, what: str, refl, eff):
    if model == "glisur":
        if s.value >= 1.0:
            return _mirror(refl, eff)
        if s.value < 0.0 or 1.0 - s.value < rules.GLISUR_R_MIN:
            raise SensorModelError(f"{what}: polish {s.value} outside [0, {1 - rules.GLISUR_R_MIN}] (or exactly 1)")
        return SensorEntry(legacy.GLISUR_METAL_EFF, {"R": refl, "polish": float(s.value), "eff": eff})
    if not rules.SIGMA_MIN <= s.value <= rules.SIGMA_MAX:
        raise SensorModelError(f"{what}: sigma_alpha {s.value} outside [{rules.SIGMA_MIN}, {rules.SIGMA_MAX}]")
    return SensorEntry(legacy.DIELECTRIC_METAL, {"R": refl, "p_spike": flat_fraction(s, "SPECULARSPIKECONSTANT", 0.0),
                                                 "p_lobe": flat_fraction(s, "SPECULARLOBECONSTANT", 0.0),
                                                 "p_back": flat_fraction(s, "BACKSCATTERCONSTANT", 0.0),
                                                 "sigma_alpha": float(s.value), "eff": eff})  # fmt: skip


def geant4(surface: schema.Surface, lam_domain) -> SensorEntry:
    """Return stock Geant4's reading of a detecting surface (see the module docstring)."""
    s, pr = surface, surface.properties
    model, finish, type_ = rules.surface_names(s)
    what = f"opticalsurface {s.name!r} ({model} {finish} {type_})"
    eff = flat_fraction(s, "EFFICIENCY", 0.0)
    if type_ == "dielectric_dielectric":
        e = rules.surface(s)
        if e.kind is not legacy.DIELECTRIC_RT:
            raise SensorModelError(f"{what}: {e.kind.name} does not detect (Geant4: only a polished "
                                   "dielectric_dielectric with R / T tables reaches DoAbsorption)")  # fmt: skip
        return SensorEntry(legacy.DIELECTRIC_RT_SENSOR, dict(e.params, eff=eff))
    if type_ != "dielectric_metal":
        raise SensorModelError(f"{what}: surface type {type_} is not supported as a sensor")
    if model not in ("glisur", "unified") or finish not in ("polished", "ground"):
        raise SensorModelError(f"{what}: model / finish not supported as a sensor")
    if rules.nonzero(pr, "TRANSMITTANCE"):
        raise SensorModelError(f"{what}: TRANSMITTANCE on a metal would put the photon into it")
    if "REFLECTIVITY" not in pr and "REALRINDEX" in pr and "IMAGINARYRINDEX" in pr:
        if finish != "polished":
            raise SensorModelError(f"{what}: a complex index on a ground finish (the facet's Fresnel) is not supported")
        return SensorEntry(legacy.PHOTOCATHODE, {"nr": at_centre(s, "REALRINDEX", lam_domain),
                                                 "nk": at_centre(s, "IMAGINARYRINDEX", lam_domain), "qe": eff,
                                                 "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 0.0})  # fmt: skip
    refl = flat_fraction(s, "REFLECTIVITY", 1.0)
    if finish == "polished":
        return _mirror(refl, eff)
    return _ground_metal(s, model, what, refl, eff)


def complex_index(surface: schema.Surface, lam_domain) -> SensorEntry:
    """Return RINDEX + i KINDEX on the surface as a semi-infinite absorber (see the module docstring)."""
    s = surface
    if "RINDEX" not in s.properties or "KINDEX" not in s.properties:
        raise SensorModelError(f"opticalsurface {s.name!r}: complex_index needs RINDEX and KINDEX on the surface")
    return SensorEntry(legacy.PHOTOCATHODE, {"nr": at_centre(s, "RINDEX", lam_domain),
                                             "nk": at_centre(s, "KINDEX", lam_domain),
                                             "qe": flat_fraction(s, "EFFICIENCY", 1.0), "f_spec": 1.0,
                                             "f_glossy": 0.0, "sigma_alpha": 0.0})  # fmt: skip


register("lumen:geant4", geant4)
register("lumen:complex_index", complex_index)
