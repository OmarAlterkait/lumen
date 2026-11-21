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
  <matrix name="GV" coldim="2" values="2.0*eV 220 4.5*eV 1e30"/>
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
    assert "GROUPVEL" not in liner.properties  # Geant4's own table, read on no surface: not kept
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


def test_non_optical_keys_are_kept_inert_and_unbound_borders_round_trip(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML), material_keys=frozenset({"RINDEX"}))
    water = df.materials["Water"]
    assert set(water.properties) == {"RINDEX"} and set(water.inert) == {"ABSLENGTH"}  # Water's Rayleigh: at load
    np.testing.assert_array_equal(water.inert["ABSLENGTH"][:, 1], [1000.0, 20000.0])  # as read: Geant4's mm
    back = io.read(io.save(df, tmp_path / "inert.lumen"))
    assert io.equal(df, back) and back.borders == df.borders


def test_a_user_material_key_is_kept_inert_with_one_warning(tmp_path):
    """A key no Geant4 optical process reads (here a one-row table Geant4 never reads) is kept inert, unread, with one
    grouped warning; Recipe.material_keys reads it explicitly (and refuses the one-row table)."""
    import warnings

    from lumen.gdml.importer import ImportRefused, InertKeysWarning, read_gdml

    text = GDML.replace(
        '<matrix name="REF"', '<matrix name="MYKEY" coldim="2" values="1.5*MeV 3.0"/>\n  <matrix name="REF"'
    )
    prop = '<property name="ABSLENGTH" ref="ABS0x2"/>'
    text = text.replace(prop, prop + '\n    <property name="MYCONSTANT" ref="MYKEY"/>')
    path = write(tmp_path, text)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        df = read_gdml(path)
    held = [w for w in seen if issubclass(w.category, InertKeysWarning)]
    assert len(held) == 1 and "1 material key(s) in 1 material(s)" in str(held[0].message)
    assert "MYCONSTANT" in df.materials["Water"].inert and "MYCONSTANT" not in df.materials["Water"].properties
    with pytest.raises(ImportRefused, match="MYCONSTANT"):
        read_gdml(path, material_keys=frozenset({"RINDEX", "ABSLENGTH", "MYCONSTANT"}))


def test_every_property_key_is_stored_in_its_unit_by_every_reader(tmp_path):
    """One unit table (schema.PROPERTY_UNITS): the GDML's Geant4 units (GROUPVEL mm / ns, ISOTHERMAL_COMPRESSIBILITY
    mm3 / MeV, COATEDTHICKNESS mm) and a YAML value with its unit are stored alike (m / ns, m3 / MeV, m); the export
    writes Geant4's units back, exactly."""
    from lumen.format import schema
    from lumen.format.yaml import _Reader
    from lumen.gdml import importer, reader

    assert schema.PROPERTY_UNITS["GROUPVEL"] == ("velocity", "m/ns", -3)
    for key, tok, want in (("GROUPVEL", "220.5", 0.2205), ("GROUPVEL", "0.2205*m/ns", 0.2205),
                           ("ISOTHERMAL_COMPRESSIBILITY", "76.58", 7.658e-8),
                           ("COATEDTHICKNESS", "2.6e-5", 2.6e-8), ("ABSLENGTH", "1.5*cm", 0.015),
                           ("WLSTIMECONSTANT", "2.5", None)):  # fmt: skip
        if want is None:
            assert importer._power(key) == 0
            continue
        internal = reader.Evaluator().eval(tok)  # the reader's value, in Geant4's units
        assert importer._stored(key, tok, internal) == pytest.approx(want, rel=1e-15), (key, tok)
    r = _Reader.__new__(_Reader)
    assert r.prop_value("GROUPVEL", "220.5*mm/ns", "w") == pytest.approx(0.2205, rel=1e-15)
    assert r.prop_value("GROUPVEL", 0.2205, "w") == 0.2205  # a bare number is in the key's unit
    assert r.prop_value("ISOTHERMAL_COMPRESSIBILITY", "7.658e-23*m3/MeV", "w") == 7.658e-23
    assert r.prop_value("ABSLENGTH", "30*cm", "w") == pytest.approx(0.3)
    with pytest.raises(Exception, match="plain number"):
        r.prop_value("RINDEX", "1.33*mm", "w")


def test_the_declared_parameter_units_are_the_stored_units():
    """keys.PARAMS declares a property's unit; it is the unit the property is stored in (schema.PROPERTY_UNITS)."""
    from lumen.format import schema
    from lumen.rules.keys import PARAMS

    for key, (_, unit, _) in schema.PROPERTY_UNITS.items():
        if key in PARAMS:
            assert PARAMS[key].unit == unit, key
    for key, p in PARAMS.items():
        if p.unit and key.isupper():
            assert key in schema.PROPERTY_UNITS, key


# the schema's string fields that name another record: each renamed by importer.canonical_names
NAME_REFERENCES = {("DetectorFile", "world"), ("DetectorFile", "detector_volume"), ("Volume", "solid"),
                   ("Volume", "material"), ("Placement", "volume"), ("Placement", "mother"), ("Skin", "surface"),
                   ("Skin", "volume"), ("Border", "surface"), ("Border", "pv1"), ("Border", "pv2"),
                   ("Component", "ref"), ("Solid", "refs"), ("Solid", "mesh")}  # fmt: skip
# and those that do not (a record's own name, renamed with it, or a value)
NOT_REFERENCES = {("DetectorFile", "name"), ("DetectorFile", "report"), ("Element", "formula"), ("Material", "state"),
                  ("Component", "kind"), ("Solid", "type"), ("Surface", "type"), ("Surface", "model"),
                  ("Surface", "finish"), ("OpticsModel", "name"), ("OpticsModel", "origin"), ("OpticsModel", "note"),
                  ("Property", "key"), ("Property", "origin"), ("Property", "note"), ("Skin", "origin"),
                  ("Skin", "note")} | {(c, "name") for c in ("Border", "Element", "Isotope", "Material", "Placement",
                                                              "Skin", "Solid", "Surface", "Volume")}  # fmt: skip


def test_every_name_reference_of_the_schema_is_renamed():
    """Every string field of the detector-file schema is classified, so a reference field added to the schema fails
    here until importer.canonical_names renames it (a suffixed reference to a stripped name dangles)."""
    import dataclasses
    import inspect

    from lumen.format import schema

    fields = {(c.__name__, f.name) for c in vars(schema).values() if isinstance(c, type) and
              dataclasses.is_dataclass(c) and c.__module__ == schema.__name__ for f in dataclasses.fields(c)
              if str(f.type).startswith(("str", "tuple[str"))}  # fmt: skip
    assert fields == NAME_REFERENCES | NOT_REFERENCES, "classify the schema's new string field (and rename it)"
    src = inspect.getsource(importer.canonical_names)
    assert all(f"{f}=n(" in src or f"{f}=tuple(n(" in src or f"{f}=comps(" in src or (f == "ref" and "ref=n(" in src)
               for _, f in NAME_REFERENCES), "canonical_names renames every reference field"  # fmt: skip
