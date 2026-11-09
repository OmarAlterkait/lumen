"""The sensor-model plug-in point (lumen.sensors.models) and the GDML's directional borders
(_legacy_regions.border_library; the baseline's tests/test_sensor_models.py): the default reads a detecting surface as
stock Geant4 does; a registered model replaces it without touching the core; a border applies only in the direction
the GDML defines it, and a sensor detects from whichever side its border surface is defined on."""

import types

import numpy as np
import pytest
from optics_toy import tree

from lumen.build import _legacy_regions as regions
from lumen.build._legacy_regions import SensorEntry
from lumen.format import schema
from lumen.rules import optics as rules
from lumen.sensors import models
from lumen.surfaces import _legacy as legacy

LAM = (410.0, 430.0)
E2 = [2.5, 3.5]  # eV: a flat table over the range


def prop(key, v, e=None):
    return schema.Property(key, np.asarray(v, float), None if e is None else np.asarray(e, float))


def surf(name="cath", model="glisur", finish="polished", type_="dielectric_metal", value=1.0, **props):
    return schema.Surface(name, type_, model, finish, value, dict(props))


def test_geant4_reads_reflectivity_and_efficiency_as_written():
    """REFLECTIVITY 0, EFFICIENCY 1 (a perfect absorber in stock Geant4), whatever else the surface carries"""
    s = surf(REFLECTIVITY=prop("REFLECTIVITY", [0.0, 0.0], E2), EFFICIENCY=prop("EFFICIENCY", [1.0, 1.0], E2),
             RINDEX=prop("RINDEX", [1.92, 1.92], E2), KINDEX=prop("KINDEX", [1.69, 1.69], E2),
             THICKNESS=prop("THICKNESS", [2.6e-5, 2.6e-5], E2))  # fmt: skip
    e = models.get()(s, LAM)
    assert e.kind is legacy.SENSOR and e.params["R"] == 0.0 and e.params["qe"] == 1.0 and e.params["f_spec"] == 1.0


def test_geant4_complex_index_only_without_reflectivity():
    s = surf(REALRINDEX=prop("REALRINDEX", [1.92, 1.92], E2), IMAGINARYRINDEX=prop("IMAGINARYRINDEX", [1.69, 1.69], E2),
             EFFICIENCY=prop("EFFICIENCY", 0.8))  # fmt: skip
    e = models.geant4(s, LAM)
    assert e.kind is legacy.PHOTOCATHODE and (e.params["nr"], e.params["nk"], e.params["qe"]) == (1.92, 1.69, 0.8)
    s2 = surf(REFLECTIVITY=prop("REFLECTIVITY", 0.3), REALRINDEX=prop("REALRINDEX", 1.9),
              IMAGINARYRINDEX=prop("IMAGINARYRINDEX", 1.6))  # fmt: skip
    assert models.geant4(s2, LAM).kind is legacy.SENSOR  # REFLECTIVITY takes precedence (G4OpBoundaryProcess)


def test_geant4_finishes_and_types():
    g = models.geant4(surf(finish="ground", value=0.5, EFFICIENCY=prop("EFFICIENCY", 1.0)), LAM)
    assert g.kind is legacy.GLISUR_METAL_EFF
    u = models.geant4(surf(model="unified", finish="ground", value=0.2, REFLECTIVITY=prop("REFLECTIVITY", 0.5),
                           EFFICIENCY=prop("EFFICIENCY", 1.0)), LAM)  # fmt: skip
    assert u.kind is legacy.DIELECTRIC_METAL and u.params["sigma_alpha"] == 0.2
    rt = models.geant4(surf(type_="dielectric_dielectric", REFLECTIVITY=prop("REFLECTIVITY", 0.1),
                            EFFICIENCY=prop("EFFICIENCY", 0.7)), LAM)  # fmt: skip
    assert rt.kind is legacy.DIELECTRIC_RT_SENSOR and rt.params["eff"] == 0.7
    enum = models.geant4(surf(model="1", finish="0", type_="1", REFLECTIVITY=prop("REFLECTIVITY", 0.1)), LAM)
    assert enum.kind is legacy.DIELECTRIC_RT_SENSOR  # Geant4's enum values: unified, polished, dielectric_dielectric
    with pytest.raises(models.SensorModelError):  # a polished dielectric without tables: no detection
        models.geant4(surf(type_="dielectric_dielectric"), LAM)


def test_complex_index_model_and_its_flatness_rule():
    s = surf(RINDEX=prop("RINDEX", [1.92, 1.92], E2), KINDEX=prop("KINDEX", [1.69, 1.69], E2),
             REFLECTIVITY=prop("REFLECTIVITY", 0.0), EFFICIENCY=prop("EFFICIENCY", 1.0))  # fmt: skip
    e = models.get("complex_index")(s, LAM)
    assert e.kind is legacy.PHOTOCATHODE and (e.params["nr"], e.params["nk"]) == (1.92, 1.69)
    steep = surf(RINDEX=prop("RINDEX", [1.5, 2.5], E2), KINDEX=prop("KINDEX", [1.69, 1.69], E2))
    with pytest.raises(models.SensorModelError):  # one wavelength for a table that varies: refused
        models.complex_index(steep, LAM)


def test_a_registered_model_replaces_the_default_without_touching_the_core():
    calls = []

    def thin_film(s, lam):
        calls.append((s.name, lam))
        return SensorEntry(legacy.COUNTING, {"eff": 0.25})

    models.register("test_thin_film", thin_film)
    cath = surf(EFFICIENCY=prop("EFFICIENCY", 1.0))
    e = rules.sensor(cath, LAM, model="test_thin_film")
    assert e.kind is legacy.COUNTING and calls == [("cath", LAM)]
    assert rules.sensor(cath, LAM, model="test_thin_film", eff=1.0).params["eff"] == 1.0
    with pytest.raises(models.SensorModelError):
        models.get("no_such_model")


def test_borders_apply_in_the_gdml_directions_and_detect_from_either_side(monkeypatch):
    """body -> inner (into the sensor) and inner -> body (out of it, a detecting surface): both sensors, the second
    credited to the volume left; glass -> body (no EFFICIENCY) a plain surface; no border inner -> glass is invented"""
    t = tree(("Out", "OutLV", "Steel", None), ("World", "WorldLV", "Water", "Out"),
             ("Body", "BodyLV", "Acrylic", "World"), ("Inner", "InnerLV", "LAr", "Body"))  # fmt: skip
    cath = surf("cath", REFLECTIVITY=prop("REFLECTIVITY", 0.0), EFFICIENCY=prop("EFFICIENCY", 1.0))
    mirror = surf("mirror", REFLECTIVITY=prop("REFLECTIVITY", 0.9))
    df = types.SimpleNamespace(surfaces={"cath": cath, "mirror": mirror})
    monkeypatch.setattr(regions, "border_bindings", lambda df_, t_: {("Body", "Inner"): "cath",
                                                                     ("Inner", "Body"): "cath",
                                                                     ("World", "Body"): "mirror"})  # fmt: skip
    sensors, surfaces, borders, sb = regions.border_library(df, t, {"Inner"}, LAM)
    assert borders == {("Body", "Inner"): "sensor:cath", ("Inner", "Body"): "sensor:cath|out",
                       ("World", "Body"): "mirror"}  # fmt: skip
    assert set(sensors) == {"sensor:cath", "sensor:cath|out"} and set(surfaces) == {"mirror"}
    assert [(b.sensor, b.lv, b.entry) for b in sb] == [("sensor:cath", "InnerLV", "border")]
    # WCSim's sensitive detector: every absorption on entry counts
    s2, _, _, _ = regions.border_library(df, t, {"Inner"}, LAM, regions.BorderSensor(eff_in=1.0, prefix="pmt"))
    assert s2["pmt:cath"].params["qe"] == 1.0 and s2["pmt:cath|out"].params["qe"] == 1.0
