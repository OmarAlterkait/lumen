"""The readers of a detecting surface (registered in lumen.plugins.optics_models) and the GDML's directional borders
(library.border_library): the default reads a detecting surface as
stock Geant4 does; a reader passed in replaces it without touching the core; a border applies only in the direction
the GDML defines it, and a sensor detects from whichever side its border surface is defined on."""

import types

import numpy as np
import pytest
from optics_toy import tree

from lumen.build import library as glue
from lumen.build.surfaces import SensorEntry
from lumen.format import schema
from lumen.rules import optics as rules
from lumen.surfaces import types as legacy

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
    e = rules.sensor_surface()(s, LAM)
    assert (
        e.surface_type is legacy.SENSOR and e.params["R"] == 0.0 and e.params["qe"] == 1.0 and e.params["f_spec"] == 1.0
    )


def test_geant4_complex_index_only_without_reflectivity():
    s = surf(REALRINDEX=prop("REALRINDEX", [1.92, 1.92], E2), IMAGINARYRINDEX=prop("IMAGINARYRINDEX", [1.69, 1.69], E2),
             EFFICIENCY=prop("EFFICIENCY", 0.8))  # fmt: skip
    e = rules.geant4_sensor(s, LAM)
    assert e.surface_type is legacy.PHOTOCATHODE and (e.params["nr"], e.params["nk"], e.params["qe"]) == (
        1.92,
        1.69,
        0.8,
    )
    s2 = surf(REFLECTIVITY=prop("REFLECTIVITY", 0.3), REALRINDEX=prop("REALRINDEX", 1.9),
              IMAGINARYRINDEX=prop("IMAGINARYRINDEX", 1.6))  # fmt: skip
    assert (
        rules.geant4_sensor(s2, LAM).surface_type is legacy.SENSOR
    )  # REFLECTIVITY takes precedence (G4OpBoundaryProcess)


def test_geant4_finishes_and_types():
    g = rules.geant4_sensor(surf(finish="ground", value=0.5, EFFICIENCY=prop("EFFICIENCY", 1.0)), LAM)
    assert g.surface_type is legacy.GLISUR_METAL_EFF
    u = rules.geant4_sensor(surf(model="unified", finish="ground", value=0.2, REFLECTIVITY=prop("REFLECTIVITY", 0.5),
                           EFFICIENCY=prop("EFFICIENCY", 1.0)), LAM)  # fmt: skip
    assert u.surface_type is legacy.DIELECTRIC_METAL and u.params["sigma_alpha"] == 0.2
    rt = rules.geant4_sensor(surf(type_="dielectric_dielectric", REFLECTIVITY=prop("REFLECTIVITY", 0.1),
                            EFFICIENCY=prop("EFFICIENCY", 0.7)), LAM)  # fmt: skip
    assert rt.surface_type is legacy.DIELECTRIC_RT_SENSOR and rt.params["eff"] == 0.7
    enum = rules.geant4_sensor(surf(model="1", finish="0", type_="1", REFLECTIVITY=prop("REFLECTIVITY", 0.1)), LAM)
    assert (
        enum.surface_type is legacy.DIELECTRIC_RT_SENSOR
    )  # Geant4's enum values: unified, polished, dielectric_dielectric
    # without tables Geant4 reads R 1 and T 0 (its defaults): the same type, absorbing nothing at those values
    bare = rules.geant4_sensor(surf(type_="dielectric_dielectric"), LAM)
    assert bare.surface_type is legacy.DIELECTRIC_RT_SENSOR and (bare.params["R"], bare.params["T"]) == (1.0, 0.0)


def test_complex_index_model_and_its_flatness_rule():
    s = surf(RINDEX=prop("RINDEX", [1.92, 1.92], E2), KINDEX=prop("KINDEX", [1.69, 1.69], E2),
             REFLECTIVITY=prop("REFLECTIVITY", 0.0), EFFICIENCY=prop("EFFICIENCY", 1.0))  # fmt: skip
    e = rules.sensor_surface("complex_index")(s, LAM)
    assert e.surface_type is legacy.PHOTOCATHODE and (e.params["nr"], e.params["nk"]) == (1.92, 1.69)
    steep = surf(RINDEX=prop("RINDEX", [1.5, 2.5], E2), KINDEX=prop("KINDEX", [1.69, 1.69], E2))
    with pytest.raises(rules.OpticsRefused):  # one wavelength for a table that varies: refused
        rules.complex_index_sensor(steep, LAM)


def test_a_reader_passed_in_replaces_the_default_without_touching_the_core():
    calls = []

    def thin_film(s, lam):
        calls.append((s.name, lam))
        return SensorEntry(legacy.COUNTING, {"eff": 0.25})

    cath = surf(EFFICIENCY=prop("EFFICIENCY", 1.0))
    e = rules.sensor(cath, LAM, model=thin_film)
    assert e.surface_type is legacy.COUNTING and calls == [("cath", LAM)]
    assert rules.sensor(cath, LAM, model=thin_film, eff=1.0).params["eff"] == 1.0
    assert rules.sensor_surface("lumen:geant4") is rules.sensor_surface("geant4") is rules.geant4_sensor
    with pytest.raises(rules.OpticsRefused):
        rules.sensor_surface("no_such_model")


def test_borders_apply_in_the_gdml_directions_and_detect_from_either_side(monkeypatch):
    """body -> inner (into the sensor) and inner -> body (out of it, a detecting surface): both sensors, the second
    credited to the volume left; glass -> body (no EFFICIENCY) a plain surface; no border inner -> glass is invented"""
    t = tree(("Out", "OutLV", "Steel", None), ("World", "WorldLV", "Water", "Out"),
             ("Body", "BodyLV", "Acrylic", "World"), ("Inner", "InnerLV", "LAr", "Body"))  # fmt: skip
    cath = surf("cath", REFLECTIVITY=prop("REFLECTIVITY", 0.0), EFFICIENCY=prop("EFFICIENCY", 1.0))
    mirror = surf("mirror", REFLECTIVITY=prop("REFLECTIVITY", 0.9))
    df = types.SimpleNamespace(surfaces={"cath": cath, "mirror": mirror})
    monkeypatch.setattr(glue, "border_bindings", lambda df_, t_: {("Body", "Inner"): "cath",
                                                                     ("Inner", "Body"): "cath",
                                                                     ("World", "Body"): "mirror"})  # fmt: skip
    sensors, surfaces, borders, sb = glue.border_library(df, t, {"Inner"}, LAM)
    assert borders == {("Body", "Inner"): "sensor:cath", ("Inner", "Body"): "sensor:cath|out",
                       ("World", "Body"): "mirror"}  # fmt: skip
    assert set(sensors) == {"sensor:cath", "sensor:cath|out"} and set(surfaces) == {"mirror"}
    assert [(b.sensor, b.lv, b.entry) for b in sb] == [("sensor:cath", "InnerLV", "border")]
    # WCSim's sensitive detector: every absorption on entry counts
    s2, _, _, _ = glue.border_library(df, t, {"Inner"}, LAM, glue.BorderSensor(eff_in=1.0, prefix="pmt"))
    assert s2["pmt:cath"].params["qe"] == 1.0 and s2["pmt:cath|out"].params["qe"] == 1.0
