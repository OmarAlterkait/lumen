"""lumen.build.library: a detector file's recorded optics decisions as the region compiler's library and
bindings -- every material, surface and sensor encoding, the outside surface, a recipe skin
replacing the GDML's, and the parameter sources."""

import dataclasses

import numpy as np
import pytest
from optics_toy import read_toy, tree
from readout_toys import TOY_SET, toy_set

from lumen.build import binding
from lumen.build import library as glue
from lumen.build import lookup as regions
from lumen.format import schema
from lumen.format.calibrations import Table
from lumen.format.sensorsets import SensorGroup, SensorSet
from lumen.rules import sensors as sensor_rules
from lumen.surfaces import types


def model(name, **settings):
    return schema.OpticsModel(name, settings=settings, origin="recipe", note="test")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return read_toy(tmp_path_factory)


def decided(df):
    mats = dict(df.materials)
    mats["Steel"] = dataclasses.replace(mats["Steel"], optics=model("lumen:opaque"))
    return dataclasses.replace(df, materials=mats)


TREE = (("<outside>", "<outside>", "<outside>"), ("w/W", "W", "Liq", "<outside>"),
        ("w/blk1/Blk", "Blk", "Steel", "w/W"), ("w/other/X", "X", "Liq", "w/W"),
        ("w/f1/Foil", "Foil", "Steel", "w/W"), ("w/f2/Foil", "Foil", "Steel", "w/W"),
        ("w/sh/S", "S", "Shifter", "w/W"), ("w/sen/Sen", "Sen", "Liq", "w/W", 0))  # fmt: skip
COUNTING = SensorSet((SensorGroup("sen", rule="any_entry", lv="Sen", settings={"entry_efficiency": 0.5}),))


def test_every_encoding_on_the_toy(toy):
    lib, b, src = glue.library_and_bindings(decided(toy), tree(*TREE), COUNTING)
    assert b.materials == {"<outside>": "outside", "Liq": "Liq", "Steel": "opaque", "Shifter": "Shifter"}
    assert set(lib.media) == {"outside", "Liq", "opaque", "Shifter"} and lib.media["opaque"].medium.opaque
    # the world is the detector volume here: the outside region has no volume beyond it, so no skin
    assert b.skins == {"Blk": "glis"}
    assert (
        b.borders == {("blk1", "other"): "paint"}
        and lib.surfaces["paint"].surface_type.name == "painted_t{eff=0}{f_spec=0}"
    )
    assert lib.sensors["sen"].params == {"eff": 0.5} and b.sensors == (regions.SensorBinding("sen", 1, lv="Sen"),)
    o = regions.compile(lib, tree(*TREE), b)
    assert o.class_of["w/sh/S"] != o.class_of["w/W"]  # the WLS volume is its own medium (bulk WLS)
    # the sources, for every parameter leaf of the library
    named = regions.named_params(o, lib)
    assert src[("media", "Liq", "index", "value")] == glue.Source("materials/Liq/RINDEX", "value")
    assert src[("media", "Liq", "rayleigh", "length", "energy")] == glue.Source("materials/Liq/RAYLEIGH", "energy")
    assert src[("surfaces", "glis", "polish")] == glue.Source("surfaces/glis/polish", "constant")
    # the re-emission spectrum is a parameter (its derivative: lumen.media.emission)
    assert src[("media", "Shifter", "wls", "spectrum", "value")] == glue.Source(
        "materials/Shifter/WLSCOMPONENT", "value"
    )
    assert src[("sensors", "sen", "eff")] == glue.Source("sensors/sen/entry_efficiency", "setting")
    leaves = {(s, n) + p for s in ("surfaces", "sensors") for n, ps in named[s].items() for p in _paths(ps)}
    assert leaves <= set(src) and not [k for k, v in src.items() if v.note == "unrecorded"]


def _paths(p, path=()):
    for k, v in p.items():
        yield from (_paths(v, path + (k,)) if isinstance(v, dict) else [path + (k,)])


def test_a_recipe_skin_replaces_the_gdml_skin(toy):
    df = decided(toy)
    df = dataclasses.replace(
        df, skins=df.skins + (schema.Skin("mirror", surface="glispol", volume="Blk", origin="recipe"),)
    )
    lib, b, _ = glue.library_and_bindings(df, tree(*TREE), COUNTING)
    name = lib.surfaces["glispol"].surface_type.name
    assert b.skins["Blk"] == "glispol" and name.startswith(types.SPECULAR_METAL.name)


def test_border_sensors(toy):
    df = dataclasses.replace(decided(toy), borders=(schema.Border("bs", surface="pdieR", pv1="blk1", pv2="other"),))
    group = SensorGroup("pmt", rule="border_into", lv_pattern="^X$", settings={"entry_efficiency": 1.0})
    sset = SensorSet((group,) + COUNTING.groups)
    lib, b, src = glue.library_and_bindings(df, tree(*TREE), sset)
    e = lib.sensors["pmt:pdieR"]
    # Geant4 reads T (its default 0 when the surface gives none): a parameter, not folded (structure decides)
    assert e.surface_type is types.DIELECTRIC_RT and e.params["eff"] == 1.0 and e.params["T"] == 0.0
    assert src[("sensors", "pmt:pdieR", "T")] == glue.Source(
        "surfaces/pdieR/TRANSMITTANCE", "default", "Geant4's default"
    )
    assert b.borders == {("blk1", "other"): "pmt:pdieR"}
    assert regions.SensorBinding("pmt:pdieR", 1, lv="X", entry="border") in b.sensors
    assert src[("sensors", "pmt:pdieR", "eff")] == glue.Source("sensors/pmt/entry_efficiency", "setting")


def test_several_border_groups(toy):
    """Each border_into group reads the borders into its own volumes, with its own efficiency and sensor names."""
    df = dataclasses.replace(
        decided(toy),
        borders=(
            schema.Border("bs", surface="pdieR", pv1="blk1", pv2="other"),
            schema.Border("bs2", surface="pdieR", pv1="blk1", pv2="sen"),
        ),
    )
    a = SensorGroup("a", rule="border_into", lv="X", settings={"entry_efficiency": 1.0})
    b = SensorGroup("b", rule="border_into", lv="Sen", settings={"entry_efficiency": 0.5})
    lib, bind, src = glue.library_and_bindings(df, tree(*TREE), SensorSet((a, b)))
    assert bind.borders == {("blk1", "other"): "a:pdieR", ("blk1", "sen"): "b:pdieR"}
    assert lib.sensors["a:pdieR"].params["eff"] == 1.0 and lib.sensors["b:pdieR"].params["eff"] == 0.5
    assert src[("sensors", "b:pdieR", "eff")] == glue.Source("sensors/b/entry_efficiency", "setting")
    assert regions.SensorBinding("b:pdieR", 1, lv="Sen", entry="border") in bind.sensors


def _lar_tables():
    """An index, a Rayleigh and an absorption table, as a recipe records them (Property records in eV and m)."""
    e = np.array([2.0, 5.0, 9.0])
    return {k: schema.Property(k, np.asarray(v, np.float64), energy=e, origin="recipe", note="toy")
            for k, v in (("RINDEX", [1.23, 1.28, 1.40]), ("RAYLEIGH", [9.0, 1.0, 0.6]),
                         ("ABSLENGTH", [20.0, 20.0, 20.0]))}  # fmt: skip


def test_a_table_medium_reads_its_tables(toy):
    mats = dict(toy.materials)
    mats["Liq"] = dataclasses.replace(mats["Liq"], properties=_lar_tables(), optics=model("geant4"))
    df = dataclasses.replace(toy, materials=mats, borders=(), skins=())
    lib, b, src = glue.library_and_bindings(df, tree(*TREE[:4]), SensorSet(()))
    assert b.materials["Liq"] == "Liq" and [c.name for c in lib.media["Liq"].medium.channels] == ["rayleigh"]
    assert src[("media", "Liq", "rayleigh", "length", "value")] == glue.Source("materials/Liq/RAYLEIGH", "value")


def test_refusals(toy):
    df = decided(toy)
    bad = dataclasses.replace(df, materials={**df.materials, "Liq": dataclasses.replace(
        df.materials["Liq"], optics=model("lumen:mystery"))})  # fmt: skip
    with pytest.raises(glue.LibraryError, match="mystery"):
        glue.library_and_bindings(bad, tree(*TREE), COUNTING)


def test_sensor_groups_are_checked(toy):
    """Names are unique key segments, and a volume is in one group (to_json would collapse a duplicate name)."""
    one = SensorGroup("pmt", rule="any_entry", lv_pattern="^S", settings={"entry_efficiency": 1.0})
    for bad, match in ((dataclasses.replace(one, name="1 bad/name"), "a name is letters"),
                       (one, "two sensor groups are named")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            SensorSet((one, bad) if bad is one else (bad,))
    other = dataclasses.replace(one, name="other", lv="Sen", lv_pattern=None)
    with pytest.raises(glue.LibraryError, match="selected by the sensor groups"):
        glue.library_and_bindings(decided(toy), tree(*TREE), SensorSet((one, other)))


def test_the_efficiency_on_entry_is_one_rule():
    border = SensorGroup("pmt", rule="border_into", lv="X")
    assert sensor_rules.entry_efficiency(border) is None  # the surface's own reading (EFFICIENCY)
    assert sensor_rules.entry_efficiency(dataclasses.replace(border, settings={"entry_efficiency": 0.5})) == 0.5
    # a value set applies the efficiency after tracking: every absorption entering counts
    pmt = dataclasses.replace(border, values=TOY_SET)
    assert sensor_rules.after_tracking(pmt) and sensor_rules.entry_efficiency(pmt) == 1.0
    counting = SensorGroup("sen", rule="any_entry", lv="Sen", values=TOY_SET)
    assert sensor_rules.entry_efficiency(counting) == 1.0


def _every_value_read(df):
    """Key rows as if the optics read every surface's value key that Geant4's reading reads."""
    from lumen.rules import optics as rules

    return {("surfaces", n, "value"): glue.Source(f"surfaces/{n}/{rules.value_key(s)}", "value")
            for n, s in df.surfaces.items() if rules.reads_value(s)}  # fmt: skip


def test_a_value_key_takes_the_range_the_reader_accepts(toy):
    from lumen.build import spec
    from lumen.params import Params
    from lumen.rules import optics as rules

    values, sp = spec.detector_values(toy, _every_value_read(toy))
    p = Params(values, sp, {})
    assert p.info("surfaces/uni/sigma_alpha").range == (rules.SIGMA_MIN, rules.SIGMA_MAX)
    with pytest.raises(ValueError, match="outside"):  # refused here, not when the library packs it
        p.set("surfaces/uni/sigma_alpha", 1.5)
    with pytest.raises(ValueError, match="outside"):
        p.set("surfaces/glis/polish", 1.0 - rules.GLISUR_R_MIN / 2)
    assert p.set("surfaces/uni/sigma_alpha", 0.3)["surfaces/uni/sigma_alpha"] == 0.3


def test_a_key_is_a_parameter_where_the_optics_reads_it(toy):
    """The spec and the library agree on one record (``Parts.key_rows``): a key nothing reads is no parameter (a
    surface's EFFICIENCY with no sensor bound, GROUPVEL), a light source's material key stays, and a key read through a
    JAX function of it (a derived Rayleigh table) has its method's derivative."""
    from lumen.build import spec
    from lumen.format.schema import Property

    mats = dict(toy.materials)
    liq = mats["Liq"]
    props = {
        **liq.properties,
        "GROUPVEL": Property("GROUPVEL", np.array(0.2), energy=None, origin="source", note=""),
        "SCINTILLATIONYIELD": Property("SCINTILLATIONYIELD", np.array(100.0), energy=None, origin="source", note=""),
    }
    df = dataclasses.replace(toy, materials={**mats, "Liq": dataclasses.replace(liq, properties=props)})
    sources = {
        ("media", "Liq", "index", "value"): glue.Source("materials/Liq/RINDEX", "value"),
        ("surfaces", "uni", "R"): glue.Source("surfaces/uni/REFLECTIVITY", "derived", "a conversion"),
    }
    values, sp = spec.detector_values(df, sources)
    assert isinstance(values["materials/Liq/RINDEX"], Table) and "materials/Liq/SCINTILLATIONYIELD" in values
    assert "materials/Liq/GROUPVEL" not in values and "surfaces/eff/EFFICIENCY" not in values
    assert sp["surfaces/uni/REFLECTIVITY"].available


def test_a_value_key_exists_only_where_the_reading_reads_it(toy):
    from lumen.build import spec
    from lumen.rules import optics as rules

    values, _ = spec.detector_values(toy, _every_value_read(toy))
    assert "surfaces/glis/polish" in values and "surfaces/uni/sigma_alpha" in values
    # glisur ground at polish 1: Geant4's reading is a mirror (no facet), so no key reads the polish
    assert rules.surface(toy.surfaces["glis1"]).params.keys() == {"R"} and not rules.reads_value(toy.surfaces["glis1"])
    assert "surfaces/glis1/polish" not in values and "surfaces/glispol/polish" not in values
    # ... and read as a sensor, the mirror's facet width is no key either (folded into its type)
    src = glue._surface_sources(toy.surfaces["glis1"], rules.sensor(toy.surfaces["glis1"]).params)
    assert ("sigma_alpha",) not in src and set(src) == {("R",), ("eff",)}  # a mirror reads no facet width


def test_the_binding_rules_are_checked_once(toy):
    from lumen.build.binding import check_bindings

    with pytest.raises(ValueError, match="two skin surfaces"):
        check_bindings([("a", "LV"), ("b", "LV")], (), (), ValueError)
    with pytest.raises(ValueError, match="names 2 physical volumes"):
        check_bindings((), [("b", "p", "q")], ["p", "p", "q"], ValueError)
    with pytest.raises(ValueError, match="has a border already"):
        check_bindings((), [("b", "p", "q"), ("c", "p", "q")], ["p", "q"], ValueError)
    check_bindings([("a", "LV")], [("b", "p", "q"), ("c", "q", "p")], ["p", "q"], ValueError)
    df = decided(toy)  # the library: two recipe skins on one volume, the same rule
    two = (
        schema.Skin("m1", surface="glispol", volume="Blk", origin="recipe"),
        schema.Skin("m2", surface="uni", volume="Blk", origin="recipe"),
    )
    with pytest.raises(glue.LibraryError, match="two skin surfaces"):
        glue.library_and_bindings(dataclasses.replace(df, skins=df.skins + two), tree(*TREE), COUNTING)


def test_a_file_of_another_format_is_refused_once(tmp_path, toy):
    """Every file lumen writes is refused in another format by one error and one message (only the remedy differs)."""
    import json

    from lumen.format import calibrations, io
    from lumen.format.sensorsets import from_json

    generic = "written by another lumen \\(format 'old/1'\\); this lumen reads .*: "
    with pytest.raises(io.FormatError, match=generic + "convert the detector again"):
        from_json(dict(schema="old/1", group_order=[], groups={}), ())
    path = io.save(toy, tmp_path / "t.lumen")
    head = path / "detector.json"
    head.write_text(head.read_text().replace(io.schema.SCHEMA, "old/1"))
    with pytest.raises(io.FormatError, match=generic + "convert the detector again"):
        io.read(path)
    cal = tmp_path / "c.params.json"
    cal.write_text(json.dumps({"schema": "old/1"}))
    with pytest.raises(io.FormatError, match=generic + "save the calibration again"):
        calibrations.read(cal)
    from lumen.compare import references
    from lumen.sensors import value_sets

    with pytest.raises(io.FormatError, match=generic + "transcribe it again"):
        value_sets.model_from_dict({"schema": "old/1", "name": "x", "model": "lumen:sensor"})
    (tmp_path / "r.json").write_text(json.dumps({"schema": "old/1"}))
    with pytest.raises(io.FormatError, match=generic + "write the manifest again"):
        references._read(tmp_path / "r.json")


def test_one_range_per_key():
    from lumen.build import build, spec
    from lumen.rules import keys

    assert spec.info("REFLECTIVITY").range == keys.PARAMS["REFLECTIVITY"].range == types.DIELECTRIC_RT.range("R")
    assert spec.info("sigma_alpha").range == (keys.SIGMA_MIN, keys.SIGMA_MAX)
    assert spec.info("NOT_A_KEY").range == (-np.inf, np.inf)
    ((_, info),) = build.group_settings(SensorGroup("sen", rule="any_entry", lv="Sen",
                                                   settings={"entry_efficiency": 0.5})).values()  # fmt: skip
    assert info.range == keys.PARAMS["entry_efficiency"].range


def test_a_value_set_suffix_names_its_unit():
    from lumen.sensors import value_sets

    d = toy_set()
    m = value_sets.model_from_dict(d)
    info = m.param_info("pmt")
    assert info["sensors/pmt/t_offset"].unit == "ns" and info["sensors/pmt/dark_rate"].unit == "Hz"
    assert info["sensors/pmt/gain"].unit == m.unit and info["sensors/pmt/efficiency"].unit == ""
    sigma = next(k for k in d["tts"] if k.startswith("sigma"))
    d["tts"]["sigma_pe"] = d["tts"].pop(sigma)  # a time spread in photoelectrons: refused
    with pytest.raises(value_sets.SensorValuesError, match="is in pe, the field is in ns"):
        value_sets.model_from_dict(d)


def test_default_keys_are_what_the_reading_reads(toy):
    from lumen.rules import optics as rules

    s = toy.surfaces
    assert set(rules.surface_defaults(s["glis1"])) == {"REFLECTIVITY"}  # a mirror reads R only, no facet constant
    assert set(rules.surface_defaults(s["uni"])) == {"SPECULARSPIKECONSTANT", "BACKSCATTERCONSTANT"}
    assert rules.surface_defaults(s["glispol"]) == {"REFLECTIVITY": (1.0, "G4OpBoundaryProcess.cc:308")}
    assert set(rules.surface_defaults(s["paint"])) == {"TRANSMITTANCE"}


OUTSIDE_GDML = """<?xml version="1.0" encoding="UTF-8" ?>
<gdml>
<define> <matrix name="RIX" coldim="2" values="2.0*eV 1.33 4.5*eV 1.35"/> </define>
<materials>
  <element name="O" formula="O" Z="8"> <atom value="15.999"/> </element>
  <material name="Liq" state="liquid"> <property name="RINDEX" ref="RIX"/> <D value="1.0"/> <fraction n="1" ref="O"/>
  </material>
  <material name="Steel" state="solid"> <D value="7.9"/> <fraction n="1" ref="O"/> </material>
</materials>
<solids>
  <box name="World" x="4" y="4" z="4" lunit="m"/> <box name="Hall" x="3" y="3" z="3" lunit="m"/>
  <box name="Det" x="1" y="1" z="1" lunit="m"/> <box name="Hole" x="1" y="1" z="1" lunit="m"/>
  <box name="Outer" x="1.2" y="1.2" z="1.2" lunit="m"/>
  <subtraction name="Membrane"> <first ref="Outer"/> <second ref="Hole"/> </subtraction>
  <opticalsurface name="HallSkin" model="unified" finish="polished" type="dielectric_metal" value="0"/>
  <opticalsurface name="MemSkin" model="glisur" finish="ground" type="dielectric_metal" value="0.5"/>
</solids>
<structure>
  <volume name="DetLV"> <materialref ref="Liq"/> <solidref ref="Det"/> </volume>
  <volume name="MemLV"> <materialref ref="Steel"/> <solidref ref="Membrane"/> </volume>
  <volume name="HallLV"> <materialref ref="Liq"/> <solidref ref="Hall"/>
    <physvol name="det"> <volumeref ref="DetLV"/> <position name="p" x="100" y="0" z="0" unit="mm"/> </physvol>
    MEMBRANE
  </volume>
  <volume name="WorldLV"> <materialref ref="Liq"/> <solidref ref="World"/>
    <physvol name="hall"> <volumeref ref="HallLV"/> </physvol>
  </volume>
  <skinsurface name="s1" surfaceproperty="HallSkin"> <volumeref ref="HallLV"/> </skinsurface>
  <skinsurface name="s2" surfaceproperty="MemSkin"> <volumeref ref="MemLV"/> </skinsurface>
</structure>
<setup name="Default" version="1.0"> <world ref="WorldLV"/> </setup>
</gdml>
"""


CASES = [(100, ("MemLV", "MemSkin")), (130, ("HallLV", "HallSkin")), (None, ("HallLV", "HallSkin"))]


@pytest.mark.parametrize("x_mm,want", CASES)
def test_the_outside_carries_the_skin_of_the_volume_geant4_enters(tmp_path, x_mm, want):
    """Leaving the detector volume, Geant4 enters a sibling that holds it in a hole (every face touches it), else the
    mother: that volume's skin is the outside's. A hole off the detector's pose, or no such sibling: the mother."""
    from optics_toy import read_toy as read

    mem = "" if x_mm is None else (f'<physvol name="mem"> <volumeref ref="MemLV"/> <position name="q" x="{x_mm}" '
                                   'y="0" z="0" unit="mm"/> </physvol>')  # fmt: skip
    df = dataclasses.replace(read(_Factory(tmp_path), OUTSIDE_GDML.replace("MEMBRANE", mem)), detector_volume="DetLV")
    assert (binding.outside_volume(df), binding.outside_skin(df)) == want


CLOSED = [(100, "DetLV", 0), (130, "WorldLV", 2), (None, "WorldLV", 2)]


@pytest.mark.parametrize("x_mm,want,steps", CLOSED)
def test_the_detector_volume_grows_while_light_leaves_it_into_a_volume_with_rindex(tmp_path, x_mm, want, steps):
    """Geant4 transports a photon on into a volume with RINDEX: the detector volume grows to it (DetLV in the Hall's
    liquid, then the world's); into the steel membrane holding it (no RINDEX: NoRINDEX) it stays."""
    from optics_toy import read_toy as read

    mem = "" if x_mm is None else (f'<physvol name="mem"> <volumeref ref="MemLV"/> <position name="q" x="{x_mm}" '
                                   'y="0" z="0" unit="mm"/> </physvol>')  # fmt: skip
    df = dataclasses.replace(read(_Factory(tmp_path), OUTSIDE_GDML.replace("MEMBRANE", mem)), detector_volume="DetLV")
    closed, report = binding.detector_volume_closed(df)
    assert (closed, len(report)) == (want, steps)
    assert all("with RINDEX" in r for r in report)


def test_a_volume_absorbing_light_at_entry_closes_the_detector(tmp_path):
    """The Hall's material with RINDEX but an ABSLENGTH of 1e-11 m (ANNIE's steel): the detector volume grows to the
    Hall, for the reflection at its face, and stops there (no light crosses it into the world's liquid)."""
    from optics_toy import read_toy as read

    absorbing = '<matrix name="ABS" coldim="2" values="2.0*eV 1e-8*mm 4.5*eV 1e-8*mm"/> </define>'
    hall = '<material name="Abs" state="solid"> <property name="RINDEX" ref="RIX"/> <property name="ABSLENGTH" ' \
           'ref="ABS"/> <D value="7.9"/> <fraction n="1" ref="O"/> </material>\n  <material name="Steel"'  # fmt: skip
    text = (OUTSIDE_GDML.replace("MEMBRANE", "").replace("</define>", absorbing, 1)
            .replace('<material name="Steel"', hall, 1)
            .replace('<volume name="HallLV"> <materialref ref="Liq"/>',
                     '<volume name="HallLV"> <materialref ref="Abs"/>'))  # fmt: skip
    df = dataclasses.replace(read(_Factory(tmp_path), text), detector_volume="DetLV")
    closed, report = binding.detector_volume_closed(df)
    assert closed == "HallLV" and len(report) == 1 and "absorbs it at entry" in report[0]


class _Factory:
    def __init__(self, d):
        self.d = d

    def mktemp(self, name):
        p = self.d / name
        p.mkdir(exist_ok=True)
        return p
