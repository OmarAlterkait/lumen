"""Geant4-default keys: a surface property the file omits whose default the model reads is a parameter.

The type never depends on whether the property is present (PAINTED_T, DIELECTRIC_RT with T at Geant4's 0); Params
lists the key at Geant4's default with its source line; ``lumen.format.apply`` adds the property when a value is set,
and only then does the exporter write it (as the flat table Geant4 reads); its gradient family is the kernel's T.
"""

import dataclasses

import numpy as np
import pytest
from test_api import RECIPE, _toy
from test_importer import write

import lumen
from lumen.format.apply import GEANT4_DEFAULT, apply
from lumen.format.calibrations import Table
from lumen.gdml import exporter, importer
from lumen.rules import optics as rules
from lumen.validate import gradients

WIN = """  <opticalsurface name="Win" model="unified" finish="polished" type="dielectric_dielectric" value="0">
    <property name="REFLECTIVITY" ref="REF"/>
  </opticalsurface>
"""
GLASS = """  <material name="Glass" state="solid"> <property name="RINDEX" ref="RIX0x1"/> <D value="2.2"/>
    <fraction n="1.0" ref="O"/> </material>
"""
KEY = "surfaces/Win/TRANSMITTANCE"
ROW = ("dielectric_rt", "surface:Win")  # a polished surface without TRANSMITTANCE: DIELECTRIC_RT, T at Geant4's 0


def _win_toy():
    """The unit toy with its tubes of glass, each behind a polished window (a skin): T has a transparent far side."""
    text = _toy().replace("</solids>", WIN + "</solids>", 1).replace("</materials>", GLASS + "</materials>", 1)
    text = text.replace('<volume name="PmtLV"> <materialref ref="Steel"/>', '<volume name="PmtLV"> <materialref '
                        'ref="Glass"/>', 1)  # fmt: skip
    return text.replace("</structure>", '<skinsurface name="WinSkin" surfaceproperty="Win"> <volumeref ref="PmtLV"/> '
                        "</skinsurface>\n</structure>", 1)  # fmt: skip


@pytest.fixture(scope="module")
def det(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("paint")
    recipe = dataclasses.replace(RECIPE, groups=())  # no sensors: the windows are plain surfaces
    return lumen.load(lumen.convert(write(tmp, _win_toy()), recipe, tmp / "win.lumen"))


def test_params_list_the_default_with_its_source(det):
    p = det.params()
    assert p[KEY] == 0.0 and p.info(KEY).default == "default 0, G4OpBoundaryProcess.cc:310"
    assert p.explain(KEY) == "Geant4 default 0, G4OpBoundaryProcess.cc:310"
    # the unbound unified ground metal: its omitted constants are read at 0 (G4OpBoundaryProcess.cc:405-413)
    assert p["surfaces/Liner/SPECULARSPIKECONSTANT"] == 0.0 and "surfaces/Liner/TRANSMITTANCE" not in p
    assert "surfaces/Win/value" not in p  # a polished finish samples no facet


def test_setting_a_default_key_adds_the_property(det):
    df = apply(det.file, {KEY: 0.25})
    t = df.surfaces["Win"].properties["TRANSMITTANCE"]
    assert t.energy is None and float(t.value) == 0.25 and t.origin == GEANT4_DEFAULT
    assert rules.surface(df.surfaces["Win"]).params["T"] == 0.25
    assert "TRANSMITTANCE" not in det.file.surfaces["Win"].properties  # pure


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("surfaces/Win/EFFICIENCY", 0.1, "no property EFFICIENCY"),  # a sensor's, not a surface default
        ("surfaces/Liner/TRANSMITTANCE", 0.1, "no property TRANSMITTANCE"),  # T on a metal: refused, not a default
        ("materials/Water/RAYLEIGH", 1.0, "no property RAYLEIGH"),  # a material's omitted property
        (KEY, Table(np.array([2.0, 3.0]), np.array([0.1, 0.1])), "a number"),
    ],
)
def test_any_other_added_property_is_refused(det, key, value, match):
    with pytest.raises(ValueError, match=match):
        apply(det.file, {key: value})


def test_the_export_writes_a_default_only_when_set(det, tmp_path):
    plain = exporter.export(det.file, tmp_path / "a" / "x.gdml", scope="full")
    again = exporter.export(det.file, tmp_path / "b" / "x.gdml", scope="full")
    assert plain.sha256 == again.sha256 and b"TRANSMITTANCE" not in (tmp_path / "a" / "x.gdml").read_bytes()
    exporter.export(apply(det.file, {KEY: 0.25}), tmp_path / "c" / "x.gdml", scope="full")
    back = importer.read_gdml(tmp_path / "c" / "x.gdml").surfaces["Win"].properties["TRANSMITTANCE"]
    np.testing.assert_array_equal(back.value, [0.25, 0.25])  # a flat table: what G4OpBoundaryProcess reads
    assert rules.surface(exporter.read_export(tmp_path / "c" / "x.gdml").detector_file.surfaces["Win"]).params[
        "T"
    ] == 0.25  # fmt: skip


def test_the_gradient_family_is_the_params_key(det):
    fams = [f for f in gradients.families(det.parts.optics, det.pack()) if f[1][-1] == "T"]
    assert [(f[1][1], str(det.parts.optics.rows[f[1][1]][f[2]])) for f in fams] == [ROW]
    assert fams[0][3] == "edge"  # at 0: one-sided, inward
    packed = det.pack(det.params().set(KEY, 0.25))
    assert float(np.asarray(packed["kernel"][ROW[0]]["T"])[fams[0][2]]) == 0.25
