"""The import rule geant4_10 (lumen.build.import_rules): Geant4-10 scintillation keys translated once at conversion,
whatever the source, checked against Geant4 10.7's reading, recorded; an untranslatable set refused naming what to
add."""

import dataclasses

import numpy as np
import pytest
from test_api import _toy
from test_importer import write

import lumen
from lumen.build import import_rules as translate
from lumen.format.schema import Material, Property
from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe
from lumen.rules import keys, optics
from lumen.sources import scintillation as sc

E = np.array([2.0, 3.0, 4.0])
tab = lambda k, v=(0.2, 1.0, 0.4): Property(k, np.asarray(v, float), energy=E.copy(), origin="recipe", note="test")  # noqa: E731
one = lambda k, v: Property(k, np.asarray(v, float), energy=None, origin="recipe", note="test")  # noqa: E731


def g10(*drop, **values):
    """Return a complete two-component Geant4-10 set (YIELDRATIO 0.27, 6 / 1300 ns, 720 / MeV), minus ``drop``."""
    v = dict(YIELDRATIO=0.27, FASTTIMECONSTANT=6.0, SLOWTIMECONSTANT=1300.0, SCINTILLATIONYIELD=720.0) | values
    out = {k: one(k, x) for k, x in v.items()}
    out.update(FASTCOMPONENT=tab("FASTCOMPONENT"), SLOWCOMPONENT=tab("SLOWCOMPONENT", (1.0, 0.5, 0.1)))
    return {k: p for k, p in out.items() if k not in drop}


def scintillating(*drop, **values):
    """A recipe decision adding LArSoft-like Geant4-10 keys to the toy's Water, after the import."""

    def decide(df, inputs):
        w = df.materials["Water"]
        props = dict(w.properties, **g10(*drop, **values))
        return dataclasses.replace(df, materials=dict(df.materials, Water=dataclasses.replace(w, properties=props)))

    return Recipe(name="toy_scint", detector_volume="TankLV", decisions=(decide,),
                  groups=(SensorGroup("pmt", rule="any_entry", lv="PmtLV",
                                      settings={"entry_efficiency": 1.0}),))  # fmt: skip


def test_a_recipe_adding_geant4_10_keys_after_import_is_translated_checked_and_recorded(tmp_path):
    """The keys a decision adds after the GDML read are translated (they were stored as they were and refused at the
    first scintillation), and the light is Geant4 10.7's: shares 0.27 / 0.73, 6 / 1300 ns."""
    det = lumen.load(lumen.convert(write(tmp_path, _toy()), scintillating(), tmp_path / "s.lumen"))
    props = det.file.materials["Water"].properties
    assert not any(k in props for k in keys.G4_10_KEYS)
    assert float(props["SCINTILLATIONTIMECONSTANT2"].value) == 1300.0
    np.testing.assert_array_equal(props["SCINTILLATIONCOMPONENT2"].value, [1.0, 0.5, 0.1])
    (rec,) = det.file.provenance["translations"]
    assert rec["translation"] == "geant4_10_scintillation" and rec["material"] == "Water"
    assert rec["old"] == sorted(["FASTCOMPONENT", "SLOWCOMPONENT", "FASTTIMECONSTANT", "SLOWTIMECONSTANT",
                                 "YIELDRATIO"]) and rec["unread_by_geant4_10"] == []  # fmt: skip
    assert any(line.startswith("translated: material 'Water'") for line in det.file.report)
    p = det.params()
    src = sc.ScintillationPoint.of(p, "Water")
    y, (f, taus, _, _) = src.light_of(p)
    assert float(y) == 720.0 and np.allclose(f, [0.27, 0.73]) and [float(t) for t in taus] == [6.0, 1300.0]


@pytest.mark.parametrize(
    ("drop", "values", "message"),
    [
        (("YIELDRATIO",), {}, r"without \['YIELDRATIO'\].*add \['YIELDRATIO'\]"),
        (("SLOWTIMECONSTANT",), {}, r"add \['SLOWTIMECONSTANT'\]"),
        (("FASTCOMPONENT", "SLOWCOMPONENT"), {}, "without a FASTCOMPONENT or SLOWCOMPONENT spectrum.*add the spectrum"),
        ((), {"SCINTILLATIONTIMECONSTANT1": 6.0}, "and the Geant4 11 keys.*delete one of the two sets"),
        ((), {"YIELDRATIO": -0.1}, r"YIELDRATIO -0.1: .*a finite constant >= 0"),
    ],
)
def test_an_untranslatable_set_is_refused_at_conversion_saying_what_to_add(tmp_path, drop, values, message):
    """Where Geant4 10.7 aborts, emits nothing, or the keys of both releases are given: refused, never dropped."""
    with pytest.raises(translate.TranslationRefused, match=message):
        lumen.convert(write(tmp_path, _toy()), scintillating(*drop, **values), tmp_path / "s.lumen")
    assert not (tmp_path / "s.lumen").exists()


def test_the_reference_is_geant4_10_7s_reading():
    """One component takes all the light with its own time constants (YIELDRATIO unread); two share min(r, 1)."""
    slow = g10("FASTCOMPONENT")
    ((spec, tau, rise, share),) = translate.geant4_10_light(slow)
    assert spec is slow["SLOWCOMPONENT"] and (tau, rise, share) == (1300.0, None, 1.0)
    assert [x[3] for x in translate.geant4_10_light(g10(YIELDRATIO=3.0))] == [1.0, 0.0]
    m = Material("M", state="liquid", composition=None, temperature_k=None, pressure_pa=None, properties=slow)
    out, rec, line = translate.geant4_10_scintillation("M", m)
    assert rec["unread_by_geant4_10"] == ["FASTTIMECONSTANT", "YIELDRATIO"] and "dropped" in line
    assert set(out.properties) >= {"SCINTILLATIONCOMPONENT1", "SCINTILLATIONTIMECONSTANT1"}
    assert "SCINTILLATIONCOMPONENT2" not in out.properties


def test_a_translation_that_changes_the_light_is_never_written(monkeypatch):
    """The check reads the translation back with the source's own reader and compares it with the reference."""
    real = keys.geant4_11_scintillation

    def swapped(props):
        out, note = real(props)
        if note:
            y1, y2 = out["SCINTILLATIONYIELD1"], out["SCINTILLATIONYIELD2"]
            out = dict(out, SCINTILLATIONYIELD1=dataclasses.replace(y1, value=y2.value),
                       SCINTILLATIONYIELD2=dataclasses.replace(y2, value=y1.value))  # fmt: skip
        return out, note

    monkeypatch.setattr(keys, "geant4_11_scintillation", swapped)
    m = Material("M", state="liquid", composition=None, temperature_k=None, pressure_pa=None, properties=g10())
    with pytest.raises(AssertionError, match="component 1's share"):
        translate.geant4_10_scintillation("M", m)


def test_a_stored_file_with_geant4_10_keys_is_refused_convert_again():
    """No upgrader: a detector file written before the stage (its material keeps the 10 keys) is refused at load."""
    m = Material(
        "M",
        state="liquid",
        composition=None,
        temperature_k=None,
        pressure_pa=None,
        properties=dict(g10(), RINDEX=tab("RINDEX", (1.3,) * 3)),
    )
    with pytest.raises(optics.OpticsRefused, match="convert it again from source"):
        optics.medium(m)


def test_an_untranslatable_gdml_set_is_refused_at_conversion(tmp_path):
    """The same rule on a GDML's keys: two components without SLOWTIMECONSTANT are refused when it is converted (the
    GDML read keeps the keys as they are, for the recipe's decisions)."""
    from export_toys import TOYS

    from lumen.gdml import importer

    text = TOYS["k_scintillation"].replace('<property name="SLOWTIMECONSTANT" ref="C1"/>', "")
    df = importer.read_gdml(write(tmp_path, text))
    assert any("FASTCOMPONENT" in m.properties for m in df.materials.values())
    with pytest.raises(translate.TranslationRefused, match=r"add \['SLOWTIMECONSTANT'\]"):
        translate.apply(df)
