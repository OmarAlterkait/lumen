"""The reading of a detecting surface (lumen.rules.optics.sensor) and the GDML's directional borders
(library.border_library): a detecting surface is read as stock Geant4 reads it; a border applies only in the direction
the GDML defines it, and a sensor detects from whichever side its border surface is defined on."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from optics_toy import tree

import lumen
from lumen.build import library as glue
from lumen.format import schema
from lumen.rules import optics as rules
from lumen.surfaces import types

E2 = [2.5, 3.5]  # eV: a table's knots


def prop(key, v, e=None):
    return schema.Property(key, np.asarray(v, float), energy=None if e is None else np.asarray(e, float))


def surf(name="cath", model="glisur", finish="polished", type_="dielectric_metal", value=1.0, **props):
    return schema.Surface(name, type=type_, model=model, finish=finish, value=value, properties=dict(props))


def test_geant4_reads_reflectivity_and_efficiency_as_written():
    """REFLECTIVITY 0, EFFICIENCY 1 (a perfect absorber in stock Geant4), whatever else the surface carries"""
    s = surf(REFLECTIVITY=prop("REFLECTIVITY", [0.0, 0.0], E2), EFFICIENCY=prop("EFFICIENCY", [1.0, 1.0], E2),
             RINDEX=prop("RINDEX", [1.92, 1.92], E2), KINDEX=prop("KINDEX", [1.69, 1.69], E2),
             THICKNESS=prop("THICKNESS", [2.6e-8, 2.6e-8], E2))  # fmt: skip
    e = rules.sensor(s)
    # a table stays a table, flat or not: the structure, never the values, decides (one rule for surfaces and sensors)
    assert e.surface_type is types.SPECULAR_METAL and set(e.params) == {"R", "eff"}
    assert np.all(e.params["R"][1]["value"] == 0.0) and np.all(e.params["eff"][1]["value"] == 1.0)


def test_geant4_complex_index_only_without_reflectivity():
    """REALRINDEX + i IMAGINARYRINDEX, read at the photon's wavelength as tables (CalculateReflectivity), whatever
    their spread; REFLECTIVITY takes precedence (G4OpBoundaryProcess)."""
    s = surf(REALRINDEX=prop("REALRINDEX", [1.5, 2.5], E2), IMAGINARYRINDEX=prop("IMAGINARYRINDEX", [1.69, 1.2], E2),
             EFFICIENCY=prop("EFFICIENCY", 0.8))  # fmt: skip
    e = rules.geant4_sensor(s)
    assert e.surface_type is types.PHOTOCATHODE and e.params["eff"] == 0.8
    assert np.all(e.params["nr"][1]["value"] == [1.5, 2.5]) and np.all(e.params["nk"][1]["value"] == [1.69, 1.2])
    s2 = surf(REFLECTIVITY=prop("REFLECTIVITY", 0.3), REALRINDEX=prop("REALRINDEX", 1.9),
              IMAGINARYRINDEX=prop("IMAGINARYRINDEX", 1.6))  # fmt: skip
    assert rules.geant4_sensor(s2).surface_type is types.SPECULAR_METAL
    one = rules.geant4_sensor(surf(REALRINDEX=prop("REALRINDEX", 1.9), IMAGINARYRINDEX=prop("IMAGINARYRINDEX", 1.6)))
    assert (one.params["nr"], one.params["nk"]) == (1.9, 1.6)


def test_geant4_finishes_and_types():
    g = rules.geant4_sensor(surf(finish="ground", value=0.5, EFFICIENCY=prop("EFFICIENCY", 1.0)))
    assert g.surface_type is types.GLISUR_METAL
    u = rules.geant4_sensor(surf(model="unified", finish="ground", value=0.2, REFLECTIVITY=prop("REFLECTIVITY", 0.5),
                           EFFICIENCY=prop("EFFICIENCY", 1.0)))  # fmt: skip
    assert u.surface_type is types.UNIFIED_METAL and u.params["sigma_alpha"] == 0.2
    rt = rules.geant4_sensor(surf(type_="dielectric_dielectric", REFLECTIVITY=prop("REFLECTIVITY", 0.1),
                            EFFICIENCY=prop("EFFICIENCY", 0.7)))  # fmt: skip
    assert rt.surface_type is types.DIELECTRIC_RT and rt.params["eff"] == 0.7
    # without tables Geant4 reads R 1 and T 0 (its defaults): the same type, absorbing nothing at those values
    bare = rules.geant4_sensor(surf(type_="dielectric_dielectric"))
    assert bare.surface_type is types.DIELECTRIC_RT and (bare.params["R"], bare.params["T"]) == (1.0, 0.0)


def test_borders_apply_in_the_gdml_directions_and_detect_from_either_side(monkeypatch):
    """body -> inner (into the sensor) and inner -> body (out of it, a detecting surface): both sensors, the second
    credited to the volume left; glass -> body (no EFFICIENCY) a plain surface; no border inner -> glass is invented"""
    t = tree(("Out", "OutLV", "Steel", None), ("World", "WorldLV", "Water", "Out"),
             ("Body", "BodyLV", "Acrylic", "World"), ("Inner", "InnerLV", "LAr", "Body"))  # fmt: skip
    cath = surf("cath", REFLECTIVITY=prop("REFLECTIVITY", 0.0), EFFICIENCY=prop("EFFICIENCY", 0.25))
    mirror = surf("mirror", REFLECTIVITY=prop("REFLECTIVITY", 0.9))
    df = SimpleNamespace(surfaces={"cath": cath, "mirror": mirror})
    monkeypatch.setattr(glue, "border_bindings", lambda df_, t_: {("Body", "Inner"): "cath",
                                                                     ("Inner", "Body"): "cath",
                                                                     ("World", "Body"): "mirror"})  # fmt: skip
    sensors, surfaces, borders, sb = glue.border_library(df, t, {"Inner": glue.BorderSensor(None, "sensor", False)})
    assert borders == {("Body", "Inner"): "sensor:cath", ("Inner", "Body"): "sensor:cath|out",
                       ("World", "Body"): "mirror"}  # fmt: skip
    assert set(sensors) == {"sensor:cath", "sensor:cath|out"} and set(surfaces) == {"mirror"}
    # what a sensor reads is data on its entry, not read back from its name
    assert [(e.surface, e.group, e.exit) for e in (sensors["sensor:cath"], sensors["sensor:cath|out"])] == [
        ("cath", "sensor", False), ("cath", "sensor", True)]  # fmt: skip
    assert [(b.sensor, b.lv, b.entry) for b in sb] == [("sensor:cath", "InnerLV", "border")]
    # WCSim's sensitive detector: every absorption on entry counts; leaving, Geant4's Detection with EFFICIENCY
    s2, _, _, _ = glue.border_library(df, t, {"Inner": glue.BorderSensor(1.0, "pmt", False)})
    assert s2["pmt:cath"].params["eff"] == 1.0 and s2["pmt:cath|out"].params["eff"] == 0.25


def _crossing(entry, same_material, post_opaque, exit_=False):
    """``boundary.crossing`` of one sensor entry: entering its volume, or leaving it (an exit sensor)."""
    from lumen.rules import boundary

    v = SimpleNamespace(library=SimpleNamespace(sensors={"s": entry}), sensor={"b": None if exit_ else "s"},
                        enters_sensor=lambda a, b: not exit_, border=lambda a, b: "s" if exit_ else None,
                        material=lambda n: "LAr" if same_material or n == "a" else "Steel",
                        medium={"a": 0, "b": 1}, opaque={0: False, 1: post_opaque})  # fmt: skip
    return boundary.crossing(v, "a", "b", "daughter")


def test_a_detecting_surface_is_read_as_any_surface():
    """One reading of a surface, a sensor's or not (G4OpBoundaryProcess.cc, 10.7.4): its refusals (SURFACEROUGHNESS
    on a dielectric_metal cathode), the polished dielectric_dielectric between two volumes of one material skipped
    before any detection (SameMaterial, :406-416), and an exit sensor into a medium without RINDEX killed before the
    absorption draw (NoRINDEX, :417-434), as one entering is."""
    with pytest.raises(rules.OpticsRefused, match="SURFACEROUGHNESS"):
        rules.sensor(surf(EFFICIENCY=prop("EFFICIENCY", 1.0), SURFACEROUGHNESS=prop("SURFACEROUGHNESS", 0.1)))
    with pytest.raises(rules.OpticsRefused, match="SURFACEROUGHNESS"):
        rules.surface(surf(SURFACEROUGHNESS=prop("SURFACEROUGHNESS", 0.1)))
    rt = rules.sensor(surf(type_="dielectric_dielectric", REFLECTIVITY=prop("REFLECTIVITY", 0.1),
                           EFFICIENCY=prop("EFFICIENCY", 0.7)))  # fmt: skip
    assert rt.ignored_between_same_material and not rules.sensor(surf(EFFICIENCY=prop("EFFICIENCY", 1.0))) \
        .ignored_between_same_material  # fmt: skip
    for exit_ in (False, True):
        assert _crossing(rt, True, False, exit_)[0] is types.PASS
        assert _crossing(rt, False, False, exit_)[1] == "sensor:s"
        assert _crossing(rt, False, True, exit_)[0] is types.ABSORBER  # NoRINDEX beyond: no detection
    # a value never decides which reading: an EFFICIENCY of 0 reads the same parameters
    g = surf(model="unified", finish="ground", value=0.2, EFFICIENCY=prop("EFFICIENCY", 0.0))
    assert rules.reads_value(g) and rules.surface_defaults(g) == rules.surface_defaults(
        surf(model="unified", finish="ground", value=0.2, EFFICIENCY=prop("EFFICIENCY", 1.0))
    )


def test_a_datasheet_qe_above_the_cathodes_normal_absorptance_is_refused_wherever_values_enter(tmp_path):
    """A sensor model applying the QE after tracking detects QE / A_n of an absorbed photon: REFLECTIVITY 0.8 on the
    tutorial tank's cathode leaves A_n 0.2, below its value set's QE (0.25 at 380 nm), refused at load; at 0.7 (A_n
    0.3) it loads, and a QE raised to 1.4 times (0.35) is refused by the run's value check, set or traced (counted,
    its derivative NaN), as the cathode's REFLECTIVITY raised to 0.8 is: one predicate at the run's values."""
    import sys

    import jax

    from lumen.build import values

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import example_tank

    def tank(r):
        text = example_tank.GDML.read_text().replace(
            'values="1.5*eV 0.0 5.0*eV 0.0"', f'values="1.5*eV {r} 5.0*eV {r}"'
        )
        (tmp_path / f"tank{r}.gdml").write_text(text)
        return lumen.convert(tmp_path / f"tank{r}.gdml", tmp_path / f"tank{r}.lumen", recipe=example_tank.RECIPE)

    det = lumen.load(tank(0.7))
    from lumen.build.lookup import normal_absorptance

    a_n = normal_absorptance(det.parts.library, det.parts.optics, [380.0, 450.0])  # (sensors, wavelengths)
    assert a_n.shape == (8, 2) and np.allclose(a_n, 0.3, rtol=1e-6)
    message = r"its sensor model's QE 0\.\d* at normal incidence .* absorptance there 0\.\d*:"
    with pytest.raises(values.InvalidValue, match=message):
        lumen.load(tank(0.8))
    p = det.params()
    values.check(det, p)
    with pytest.raises(values.InvalidValue, match=message):
        values.check(det, p.scale("sensors/pmt/qe", 1.4))
    refl = next(k for k in p if k.endswith("/REFLECTIVITY") and "Cathode" in k)
    with pytest.raises(values.InvalidValue, match=message):
        values.check(det, p.scale(refl, 0.8 / 0.7))

    def invalid(s):
        return values.check(det, p.scale("sensors/pmt/qe", s))[2]

    assert int(jax.jit(invalid)(1.4)) == 1 and int(jax.jit(invalid)(1.0)) == 0
    g = jax.grad(lambda s: values.check(det, p.scale("sensors/pmt/qe", s))[0]["sensors/pmt/qe"].sum())
    assert np.isnan(float(g(1.4))) and np.isfinite(float(g(1.0)))


def test_a_derivative_s_values_are_checked_as_its_value_s_once(tmp_path, monkeypatch):
    """Under an eager jax.jvp the run's value check reads the primal values (it reads no tangent): the QE <= A_n check
    is computed once for them and remembered, not re-traced per call (0.3 s of SK's 0.4 s gradient-check call,
    2026-11-26), and a failing value still counts and makes the derivative NaN, as traced."""
    import sys

    import jax

    from lumen.build import lookup, values

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import example_tank

    text = example_tank.GDML.read_text().replace('values="1.5*eV 0.0 5.0*eV 0.0"', 'values="1.5*eV 0.7 5.0*eV 0.7"')
    (tmp_path / "tank.gdml").write_text(text)  # the cathode's A_n 0.3 at REFLECTIVITY 0.7, above the QE 0.25
    det = lumen.load(lumen.convert(tmp_path / "tank.gdml", tmp_path / "tank.lumen", recipe=example_tank.RECIPE))
    p, calls, margins = det.params(), [], lookup.detection_margins
    monkeypatch.setattr(lookup, "detection_margins", lambda *a: calls.append(1) or margins(*a))

    def run(s):
        return jax.jvp(lambda t: values.check(det, p.scale("sensors/pmt/qe", t))[0]["sensors/pmt/qe"], (s,), (1.0,))

    for _ in range(3):
        _, d = run(1.0)
    assert len(calls) == 1 and np.all(np.isfinite(np.asarray(getattr(d, "value", d)))), calls
    _, d = run(1.4)  # QE 0.35 above A_n 0.3: counted, not raised, and the derivative is NaN
    assert np.all(np.isnan(np.asarray(getattr(d, "value", d))))


def test_an_unread_efficiency_under_a_sensor_model_is_recorded_not_warned(tmp_path):
    """The tutorial tank's cathode with EFFICIENCY 0.5 under lumen:sensor: the build records it in the entry's eff row
    (``Parts.key_rows``) and warns nothing."""
    import sys
    import warnings

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import example_tank

    old = '"CATHODE_EFFICIENCY" coldim="2" values="1.5*eV 1.0 5.0*eV 1.0"'
    text = example_tank.GDML.read_text().replace(old, old.replace("1.0", "0.5"))
    assert "0.5 5.0*eV 0.5" in text
    (tmp_path / "tank.gdml").write_text(text)
    path = lumen.convert(tmp_path / "tank.gdml", tmp_path / "tank.lumen", recipe=example_tank.RECIPE)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        det = lumen.load(path)
    assert not [w for w in seen if "EFFICIENCY" in str(w.message)]
    notes = [s.note for k, s in det.parts.key_rows.items() if k[-1] == "eff" and s.column == "fixed"]
    assert notes and all("EFFICIENCY (0.5) is not read" in n for n in notes), notes
