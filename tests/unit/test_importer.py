"""The GDML importer and the detector file: exact unit conversion, the fail-closed rules, and the round trip."""

import json

import numpy as np
import pytest

from lumen.format import io
from lumen.gdml import importer

GDML = """<?xml version="1.0" encoding="UTF-8" ?>
<gdml>
<define>
  <constant name="HALF" value="0.5"/>
  <position name="p1" unit="cm" x="1.25" y="0" z="HALF*2"/>
  <rotation name="r1" unit="deg" x="90" y="0" z="0"/>
  <matrix name="RIX0x1" coldim="2" values="2.0*eV 1.33 3.0*eV 1.34 4.5e-06 1.35"/>
  <matrix name="ABS0x2" coldim="2" values="2.0*eV 1000.0 4.5*eV 20*m"/>
  <matrix name="REF" coldim="2" values="2.0*eV 0.9 4.5*eV 0.8"/>
  <matrix name="GV" coldim="2" values="2.0*eV 220 4.5*eV inf"/>
</define>
<materials>
  <element name="H" formula="H" Z="1"> <atom value="1.008"/> </element>
  <element name="O" formula="O" Z="8"> <atom value="15.999"/> </element>
  <material name="Water" state="liquid">
    <property name="RINDEX" ref="RIX0x1"/>
    <property name="ABSLENGTH" ref="ABS0x2"/>
    <D value="1.0" unit="g/cm3"/>
    <composite n="2" ref="H"/> <composite n="1" ref="O"/>
  </material>
  <material name="Steel" state="solid"> <D value="7.9"/> <fraction n="1.0" ref="O"/> </material>
</materials>
<solids>
  <box name="World" x="4" y="4" z="4" lunit="m"/>
  <box name="Tank" x="2000" y="2000" z="2000" lunit="mm"/>
  <tube name="Pmt" rmax="12.5" z="10" deltaphi="360" aunit="deg" lunit="cm"/>
  <opticalsurface name="Liner" model="unified" finish="ground" type="dielectric_metal" value="0.1">
    <property name="REFLECTIVITY" ref="REF"/>
    <property name="GROUPVEL" ref="GV"/>
  </opticalsurface>
</solids>
<structure>
  <volume name="PmtLV"> <materialref ref="Steel"/> <solidref ref="Pmt"/> </volume>
  <volume name="TankLV"> <materialref ref="Water"/> <solidref ref="Tank"/>
    <physvol name="pmt0x10"> <volumeref ref="PmtLV"/> <positionref ref="p1"/> <rotationref ref="r1"/> </physvol>
    <physvol name="pmt0x20" copynumber="2"> <volumeref ref="PmtLV"/> <position name="q" x="-500" y="0" z="0"/>
    </physvol>
  </volume>
  <volume name="WorldLV"> <materialref ref="Water"/> <solidref ref="World"/>
    <physvol name="tank"> <volumeref ref="TankLV"/> </physvol>
    <auxiliary auxtype="Region" auxvalue="R"> <auxiliary auxtype="cut" auxvalue="1" auxunit="mm"/> </auxiliary>
  </volume>
  <skinsurface name="LinerSkin" surfaceproperty="Liner"> <volumeref ref="TankLV"/> </skinsurface>
  <bordersurface name="B1" surfaceproperty="Liner"> <physvolref ref="tank"/> <physvolref ref="pmt0x10"/>
  </bordersurface>
</structure>
<setup name="Default" version="1.0"> <world ref="WorldLV"/> </setup>
</gdml>
"""


def write(tmp_path, text, name="toy.gdml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_units_are_exact_and_the_world_is_kept(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    assert df.world == "WorldLV" and set(df.volumes) == {"PmtLV", "TankLV", "WorldLV"}
    assert df.solids["Tank"].params["x"] == 2.0 and df.solids["World"].params["x"] == 4.0
    assert df.solids["Pmt"].params["rmax"] == 0.125  # exactly, not 125 * 0.001
    assert df.solids["Pmt"].params["deltaphi"] == 360 * (np.pi / 180)
    p0, p1, tank = df.placements
    assert p0.position == (0.0125, 0.0, 0.01) and p0.rotation == (90 * (np.pi / 180), 0.0, 0.0)
    assert p1.position == (-0.5, 0.0, 0.0) and p1.copy == 2 and p0.copy is None
    assert tank.mother == "WorldLV" and df.volumes["TankLV"].daughters == (0, 1)
    water = df.materials["Water"]
    np.testing.assert_array_equal(water.properties["RINDEX"].energy, [2.0, 3.0, 4.5])
    np.testing.assert_array_equal(water.properties["ABSLENGTH"].value, [1.0, 20.0])  # mm and m to m
    assert water.composition.density_g_cm3 == 1.0 and [c.atoms for c in water.composition.components] == [2, 1]
    liner = df.surfaces["Liner"]
    assert (liner.type, liner.model, liner.finish, liner.value) == ("dielectric_metal", "unified", "ground", 0.1)
    assert liner.inert == ("GROUPVEL",) and "GROUPVEL" not in liner.properties
    assert df.volumes["WorldLV"].aux[0].children[0].unit == "mm"
    assert any("0x suffix" in r for r in df.report)


def test_round_trip_is_byte_identical(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    path = io.save(df, tmp_path / "toy.lumen")
    back = io.read(path)
    assert io.equal(df, back) and back.placements == df.placements
    assert (
        io.to_files(importer.read_gdml(write(tmp_path, GDML, "again.gdml")))["solids.json"]
        == io.to_files(df)["solids.json"]
    )
    with pytest.raises(io.FormatError, match="never overwritten"):
        io.save(df, path)


def test_a_changed_file_is_refused(tmp_path):
    path = io.save(importer.read_gdml(write(tmp_path, GDML)), tmp_path / "toy.lumen")
    s = json.loads((path / "solids.json").read_text())
    s["solids"]["Tank"]["params"]["x"] = 3.0
    (path / "solids.json").write_text(json.dumps(s))
    with pytest.raises(io.FormatError, match="manifest"):
        io.read(path)


@pytest.mark.parametrize(
    "old,new,match",
    [
        ('value="0.1">', ">", "no 'value' attribute"),
        ('<physvolref ref="pmt0x10"/>', '<physvolref ref="pmtX"/>', "names 0 physical volumes"),
        ('name="pmt0x20"', 'name="pmt0x10"', "names 2 physical volumes"),
        (
            "</structure>",
            '<bordersurface name="B2" surfaceproperty="Liner"> <physvolref ref="tank"/> '
            '<physvolref ref="pmt0x10"/> </bordersurface></structure>',
            "has a border already",
        ),
        (
            "</structure>",
            '<skinsurface name="S2" surfaceproperty="Liner"> <volumeref ref="TankLV"/> </skinsurface>' "</structure>",
            "two skin surfaces",
        ),
        ('<D value="7.9"/> <fraction n="1.0" ref="O"/>', "", "names no composition"),
    ],
)
def test_ambiguous_geant4_readings_are_refused(tmp_path, old, new, match):
    assert old in GDML
    with pytest.raises(importer.ImportRefused, match=match):
        importer.read_gdml(write(tmp_path, GDML.replace(old, new, 1)))
