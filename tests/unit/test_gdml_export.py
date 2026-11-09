"""The GDML exporter: exact numbers, the round trip and byte determinism on toys, the recipe layer in the sidecar."""

import dataclasses
import struct
from decimal import Decimal

import numpy as np
import pytest
from export_toys import TOYS
from test_importer import GDML

from lumen.format import calibrations, io, schema, sensorsets
from lumen.format.apply import apply
from lumen.gdml import exporter, importer, reader, units
from lumen.gdml import solids_schema as ss
from lumen.recipe import Recipe

N_RANDOM = 20000  # random doubles per dimension in the literal test


def write(tmp_path, text, name="toy.gdml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def contents(df):
    """Every content file and the head fields the round trip compares."""
    f = io.to_files(df)
    head = (df.world, df.detector_volume, tuple(df.wavelength_range), df.outside, df.outside_surface,
            io.dumps(dict(b=df.build_options)))  # fmt: skip
    return {k: v for k, v in f.items() if k != "detector.json"}, head


def round_trip(tmp_path, src, **kw):
    """Export, read back, export again: return (first report, read-back result, both exports' bytes)."""
    kw.setdefault("scope", "full")
    rep = exporter.export(src, tmp_path / "a" / "x.gdml", **kw)
    back = exporter.read_export(tmp_path / "a" / "x.gdml")
    exporter.export(back, tmp_path / "b" / "x.gdml", **kw)
    files = [(tmp_path / d / "x.gdml").read_bytes() + (tmp_path / d / "x.gdml.lumen.json").read_bytes() for d in "ab"]
    return rep, back, files


# ----------------------------------------------------------------------------------------------------- numbers
def _random_doubles(rng, lo_exp, hi_exp, n):
    """Random doubles of random bit patterns with binary exponents in [lo_exp, hi_exp), both signs."""
    mant = rng.integers(0, 1 << 52, n, dtype=np.uint64)
    exp = rng.integers(lo_exp + 1023, hi_exp + 1023, n).astype(np.uint64)
    sign = rng.integers(0, 2, n).astype(np.uint64)
    bits = (sign << np.uint64(63)) | (exp << np.uint64(52)) | mant
    return bits.view(np.float64).tolist()


@pytest.mark.parametrize("places,factor", [(3, "0.001"), (-6, "1000000"), (0, "1")])
def test_every_literal_reads_back_exactly(places, factor):
    rng = np.random.default_rng(places + 7)
    smallest = 2.2250738585072014e-308 if places >= 0 else 1e-290  # the smallest normal double, shifted up
    vals = _random_doubles(rng, -900, 900, N_RANDOM) + [
        0.0,
        -0.0,
        1.0,
        0.1,
        123456.789,
        1.7976931348623157e300,
        smallest,
    ]
    for v in vals:
        lit = exporter._literal(repr(v), places)
        assert units.LITERAL.match(lit) and "+" not in lit
        back = float(Decimal(lit) * Decimal(factor))  # the importer's exact scaling
        assert struct.pack("<d", back) == struct.pack("<d", v), (v, lit)
        assert float(lit) == float(Decimal(repr(v)).scaleb(places))  # Geant4's strtod: the correctly rounded value


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 5e-324, -1e-310])
def test_non_finite_and_denormal_values_are_refused(bad):
    with pytest.raises(exporter.ExportError, match="non-finite|denormal"):
        exporter._mm(bad)


def test_a_value_that_becomes_denormal_in_geant4_units_is_refused():
    with pytest.raises(exporter.ExportError, match="not a normal double"):
        exporter._mev(1e-303)  # eV -> MeV leaves the normal range


def test_reader_reads_pressure_molar_mass_and_mee_exactly(tmp_path):
    df = importer.read_gdml(write(tmp_path, TOYS["a_angles"]))
    w = df.materials["Water"]
    assert w.pressure == 101325.1 and w.mee_ev == 78.0000000001 and w.temperature == 293.15
    assert df.isotopes["U235"].molar_mass_g_mol == 235.0439 and df.elements["H"].molar_mass_g_mol == 1.00794
    assert df.materials["Graphite"].composition.molar_mass_g_mol == 12.011
    back = io.read(io.save(df, tmp_path / "t.lumen"))
    assert back.materials["Water"].mee_ev == 78.0000000001 and back.materials["Air"].mee_ev is None


# ------------------------------------------------------------------------------------------------- round trips
@pytest.mark.parametrize("name", sorted(TOYS) + ["importer"])
def test_toys_round_trip_bit_for_bit_and_byte_for_byte(tmp_path, name):
    df = importer.read_gdml(write(tmp_path, GDML if name == "importer" else TOYS[name]))
    rep, back, (first, second) = round_trip(tmp_path, df)
    assert contents(back.detector_file) == contents(df)
    assert io.equal(back.detector_file, df)  # provenance and report too
    assert first == second and rep.lossless
    assert exporter.export(df, tmp_path / "c" / "x.gdml", scope="full").sha256 == rep.sha256  # two exports


def test_export_order_is_a_function_of_the_content(tmp_path):
    df = importer.read_gdml(write(tmp_path, TOYS["all_solids"]))
    saved = io.read(io.save(df, tmp_path / "t.lumen"))  # dicts in sorted order, not the document's
    a = exporter.export(df, tmp_path / "a.gdml", scope="full")
    b = exporter.export(saved, tmp_path / "b.gdml", scope="full")
    assert a.sha256 == b.sha256


def test_inert_matrices_round_trip_and_refs_geant4_cannot_read_are_refused(tmp_path):
    """A property naming no matrix, or a matrix that does not evaluate, is refused, as Geant4 refuses both."""
    text = GDML.replace('<property name="ABSLENGTH" ref="ABS0x2"/>',
                        '<property name="ABSLENGTH" ref="ABS0x2"/><property name="USERKEY" ref="NOPE"/>')  # fmt: skip
    keys = frozenset({"RINDEX"})
    with pytest.raises(reader.GDMLDefectError, match="undefined_matrix_property"):  # Matrix 'NOPE' was not found!
        importer.read_gdml(write(tmp_path, text), material_keys=keys)
    broken = text.replace("</define>", '<matrix name="NOPE" coldim="1" values="UNKNOWN"/>\n</define>', 1)
    with pytest.raises(reader.GDMLError, match="'NOPE'"):  # Error in expression: UNKNOWN
        importer.read_gdml(write(tmp_path, broken, "broken.gdml"), material_keys=keys)
    df = importer.read_gdml(write(tmp_path, text), material_keys=keys, repairs={"undefined_matrix_property"})
    water = df.materials["Water"]
    assert set(water.inert) == {"ABSLENGTH"} and any("USERKEY names 'NOPE'" in ln for ln in df.report)
    _, back, (first, second) = round_trip(tmp_path, df)
    assert contents(back.detector_file) == contents(df) and first == second
    assert b'name="USERKEY"' not in first


def test_unbound_borders_and_surface_inert_keys_round_trip(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    assert df.surfaces["Liner"].inert == ("GROUPVEL",)
    df = dataclasses.replace(df, borders=tuple(dataclasses.replace(b, bound=False, note="outside") for b in df.borders))
    _, back, _ = round_trip(tmp_path, df)
    assert back.detector_file.borders == df.borders and back.detector_file.surfaces["Liner"].inert == ("GROUPVEL",)


def _toy_recipe():
    def decide(df, inputs):
        o = schema.OpticsModel(
            "lumen:opaque", 1, {"medium": "steel", "expected_axes": [[1.0, 0.0, 0.0]]}, "recipe", "toy"
        )
        steel = dataclasses.replace(
            df.materials["Steel"],
            optics=o,
            properties={
                "RINDEX": schema.Property(
                    "RINDEX", np.array([1.5, 1.6]), np.array([2.0, 3.0]), "", "recipe", "a table"
                ),
                "LUMENONLY": schema.Property("LUMENONLY", np.asarray(0.5), None, "", "recipe", "not a Geant4 key"),
            },
        )
        film = schema.Surface(
            "film",
            "",
            "",
            "",
            0.0,
            {},
            schema.OpticsModel("lumen:wls_film", 1, {"thickness_m": 2e-8}, "recipe", "lowered"),
        )
        brdf = schema.Surface(
            "brdf",
            "dielectric_metal",
            "unified",
            "ground",
            0.0,
            {
                "REFLECTIVITY": schema.Property(
                    "REFLECTIVITY", np.array([0.5, 0.6]), np.array([2.0, 3.0]), "", "recipe", "r"
                )
            },
            schema.OpticsModel("acme:brdf", 2, {}, "recipe", "no Geant4 form"),
        )
        skins = (
            (schema.Skin("film", "film", "PmtLV", "recipe", "a film"),)
            + df.skins
            + (schema.Skin("brdf", "brdf", "WorldLV", "recipe", "lossy"),)
        )
        borders = df.borders + (schema.Border("b2", "brdf", "pmt0x10", "tank", "recipe", True, "lossy border"),)
        return dataclasses.replace(df, materials={**df.materials, "Steel": steel}, skins=skins, borders=borders,
                                   surfaces={**df.surfaces, "film": film, "brdf": brdf}, outside_surface="Liner",
                                   build_options={"split_unions": 16})  # fmt: skip

    group = sensorsets.SensorGroup("pmt", "border_into", "lumen:geant4", lv="PmtLV",
                                   settings={"efficiency_on_entry": 1.0}, readout={"gate": [-1.0, 2.0]})  # fmt: skip
    return Recipe("toy", "TankLV", (300.0, 600.0), (group,), (decide,),
                  sensor_columns=lambda df, loaded, tree, inputs: {"channel": [3, 4]})  # fmt: skip


def test_the_recipe_layer_and_the_sensor_set_round_trip_through_the_sidecar(tmp_path):
    result = importer.import_gdml(write(tmp_path, GDML), _toy_recipe())
    with pytest.raises(exporter.ExportError, match="no Geant4 form"):
        exporter.export(result, tmp_path / "x.gdml", scope="full")
    rep, back, (first, second) = round_trip(tmp_path, result, on_lossy="omit")
    df = result.detector_file
    assert contents(back.detector_file) == contents(df) and first == second
    assert not rep.lossless and rep.g4ref_args == ("--sens-regex", "^PmtLV$", "--wcsim-hits", "--refuse-lossy")
    assert any("brdf" in x for x in rep.omitted) and any("Geant4 reads them" in w for w in rep.warnings)
    s0, s1 = result.sensor_sets["default"], back.sensor_sets["default"]
    assert sensorsets.to_json(s0) == sensorsets.to_json(s1) and back.sensor_columns == {"channel": [3, 4]}
    gdml = (tmp_path / "a" / "x.gdml").read_text()
    assert "film" not in gdml and "brdf" not in gdml and "LUMENONLY" not in gdml and "<opticalsurface" in gdml


def test_a_calibration_is_applied_before_writing(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    c = calibrations.Calibration("c1", "toy", "sha256:x", {
        "materials/Water/RINDEX": calibrations.Table(np.array([1.5, 2.5, 4.5]), np.array([1.3, 1.31, 1.4])),
        "surfaces/Liner/value": 0.25, "surfaces/Liner/REFLECTIVITY": calibrations.Table(np.array([2.0, 4.5]),
                                                                                         np.array([0.5, 0.4])),
        "sensors/pmt/efficiency": np.array([0.9, 0.8])})  # fmt: skip
    rep = exporter.export(df, tmp_path / "c.gdml", calibration=c, scope="full")
    back = exporter.read_export(tmp_path / "c.gdml")
    assert contents(back.detector_file)[0] == contents(apply(df, c.values))[0] and rep.calibration == ("c1",)
    side = (tmp_path / "c.gdml.lumen.json").read_text()
    assert '"sensors/pmt/efficiency"' in side
    with pytest.raises(ValueError, match="needs a Table"):
        apply(df, {"materials/Water/RINDEX": 1.3})
    with pytest.raises(ValueError, match="a new knot count is a new detector file"):  # the one rule, also on export
        exporter.export(df, tmp_path / "k.gdml", scope="full", calibration=dataclasses.replace(c, values={
            "materials/Water/RINDEX": calibrations.Table(np.array([1.5, 4.5]), np.array([1.3, 1.4]))}))  # fmt: skip


def test_names_geant4_strips_at_0x_and_refuses_collisions(tmp_path):
    df = importer.read_gdml(write(tmp_path, TOYS["j_names"]))
    exporter.export(df, tmp_path / "g.gdml", scope="full", names="geant4")
    text = (tmp_path / "g.gdml").read_text()
    assert "0x" not in text.split("-->", 1)[1]
    with pytest.raises(exporter.ExportError, match="once stripped"):
        exporter.export(importer.read_gdml(write(tmp_path, GDML)), tmp_path / "h.gdml", scope="full", names="geant4")


def test_names_geant4_renames_the_sensor_paths(tmp_path):
    text = GDML.replace('name="pmt0x20"', 'name="qmt0x20"').replace('ref="PmtLV"', 'ref="Pmt0xLV"')
    text = text.replace('<volume name="PmtLV">', '<volume name="Pmt0xLV">')
    result = importer.import_gdml(
        write(tmp_path, text),
        dataclasses.replace(
            _toy_recipe(),
            decisions=(),
            groups=(sensorsets.SensorGroup("pmt", "any_entry", "lumen:counting", lv="Pmt0xLV"),),
            sensor_columns=lambda *a: {"lv": ["Pmt0xLV"] * 2},
        ),
    )
    assert all("0x" in p for p in result.sensor_sets["default"].ids)
    exporter.export(result, tmp_path / "g.gdml", scope="full", names="geant4", on_lossy="omit")
    back = exporter.read_export(tmp_path / "g.gdml")
    s = back.sensor_sets["default"]
    assert s.ids == tuple(exporter._strip_path(p) for p in result.sensor_sets["default"].ids)
    assert not any("0x" in p for p in s.ids) and s.groups[0].lv == "Pmt" and back.sensor_columns["lv"] == ["Pmt"] * 2
    names = {p.name for p in back.detector_file.placements} | set(back.detector_file.volumes)
    assert all(part.split("#")[0] in names for p in s.ids for part in p.split("/"))


def test_scope_active_writes_the_detector_in_a_shell(tmp_path):
    df = dataclasses.replace(importer.read_gdml(write(tmp_path, GDML)), detector_volume="TankLV")
    rep = exporter.export(df, tmp_path / "act.gdml")
    assert rep.scope == "active"
    shell = importer.read_gdml(tmp_path / "act.gdml")
    top = shell.volumes[shell.world]
    assert shell.world == exporter.OUTSIDE and top.material == exporter.OUTSIDE
    assert "RINDEX" not in shell.materials[exporter.OUTSIDE].properties
    old_world = shell.placements[top.daughters[0]]
    assert (old_world.name, old_world.volume) == ("WorldLV_PV", "WorldLV")
    assert shell.volumes["WorldLV"].material == "Water"  # the mother's material, around the detector
    tank = shell.placements[shell.volumes["WorldLV"].daughters[0]]
    assert (tank.name, tank.volume, tank.position) == ("tank", "TankLV", (0.0, 0.0, 0.0))
    s = shell.solids[shell.volumes["WorldLV"].solid]
    assert s.type == "xtru" and s.sections[0][0] == pytest.approx(-1.0 - exporter.SHELL_M)
    assert [k.volume for k in shell.skins] == ["TankLV"] and shell.borders == df.borders


def test_scope_active_follows_the_first_path_from_the_world(tmp_path):
    spare = '<volume name="Spare"><materialref ref="Water"/><solidref ref="World"/><physvol name="tank2"><volumeref ' \
        'ref="TankLV"/></physvol></volume>\n  <volume name="WorldLV">'  # fmt: skip
    df = importer.read_gdml(write(tmp_path, GDML.replace('<volume name="WorldLV">', spare, 1)))
    df = dataclasses.replace(df, detector_volume="TankLV")
    exporter.export(df, tmp_path / "act.gdml")
    shell = importer.read_gdml(tmp_path / "act.gdml")
    assert "Spare" not in shell.volumes and [p.name for p in shell.placements if p.volume == "TankLV"] == ["tank"]


def test_a_sidecar_of_another_gdml_is_refused(tmp_path):
    exporter.export(importer.read_gdml(write(tmp_path, GDML)), tmp_path / "x.gdml", scope="full")
    (tmp_path / "x.gdml").write_text((tmp_path / "x.gdml").read_text() + " ")
    with pytest.raises(exporter.ExportError, match="another GDML"):
        exporter.read_export(tmp_path / "x.gdml")


# ----------------------------------------------------------------------------------------------- the tables
def test_solid_spellings_agree_with_the_reader():
    for t in ss.ATTRIBUTES:
        for low, (gdml, dim) in ss.spelling(t).items():
            angle = low in reader.ANGLE_ATTRS
            unitless = low in reader.UNITLESS_ATTRS or (t == "elcone" and low in ("dx", "dy"))
            assert (dim == ss.A) == angle and (dim == ss.U) == unitless, (t, gdml)
    assert ss.spelling("cutTube")["lowx"][0] == "lowX"


def test_enum_and_key_tables_agree_with_the_optics_rules():
    from lumen.rules import optics

    assert exporter.MODELS == optics.MODELS and exporter.TYPES == optics.TYPES
    assert tuple(exporter.FINISHES[: len(optics.FINISHES)]) == optics.FINISHES
    assert set(exporter.G4_10_KEYS) == set(optics.G4_10_KEYS)
    assert (optics.TABLE_KEYS | optics.CONST_KEYS) <= exporter.GEANT4_KEYS


# --------------------------------------------------------------------------------------------------- tokens
TOKENS = GDML.replace('values="2.0*eV 1.33 3.0*eV 1.34 4.5e-06 1.35"', 'values="2.0*eV 1.33 3.4*eV 1.34 4.5e-06 1.35"')
TOKENS = TOKENS.replace('<matrix name="REF" coldim="2" values="2.0*eV 0.9 4.5*eV 0.8"/>',
                        '<matrix name="REF" coldim="2" values="2.0 0.9 4.5 0.8" unit="eV"/>')  # fmt: skip


def test_source_tokens_are_written_while_the_value_is_unchanged(tmp_path):
    df = importer.read_gdml(write(tmp_path, TOKENS))
    w = df.tokens["materials"]["Water"]
    assert w["RINDEX"]["energy"] == ["2.0*eV", "3.4*eV", None]  # 4.5e-06 is the canonical literal already
    assert df.materials["Water"].properties["RINDEX"].energy[1] == 3.4  # lumen keeps the exact decimal value
    assert df.tokens["surfaces"]["Liner"]["REFLECTIVITY"]["matrix_unit"] == "eV"  # kept; Geant4 does not read it
    assert df.tokens["solids"]["Pmt"]["unit"] == "cm" and df.tokens["placements"]["0"] == [
        "cm",
        ["1.25", "0", "HALF*2".replace("HALF*2", "(1.0)")],
    ]
    exporter.export(df, tmp_path / "t.gdml", scope="full")
    text = (tmp_path / "t.gdml").read_text()
    assert 'values="2e-6 1.33 3.4*eV 1.34 4.5e-6 1.35"' in text.replace("2.0*eV", "2e-6")
    assert '<tube name="Pmt" rmax="12.5" z="10"' in text and 'lunit="cm"' in text
    assert 'values="2 0.9 4.5 0.8" unit="eV"' in text  # written back beside its tokens; Geant4 does not read it
    g4 = reader.Evaluator().eval("3.4*eV")  # Geant4's product, in MeV
    assert g4 == 3.4 * 1e-6 and g4 != float("3.4e-6")  # one ulp from the decimal value
    back = exporter.read_export(tmp_path / "t.gdml").detector_file
    assert back.tokens == df.tokens and contents(back) == contents(df)
    c = calibrations.Calibration(
        "c",
        "toy",
        "sha256:x",
        {"materials/Water/RINDEX": calibrations.Table(np.array([2.0, 3.4, 4.5]), np.array([1.3, 1.31, 1.32]))},
    )
    exporter.export(df, tmp_path / "c.gdml", scope="full", calibration=c)
    ctext = (tmp_path / "c.gdml").read_text()
    assert 'values="2e-6 1.3 3.4e-6 1.31 4.5e-6 1.32"' in ctext  # a calibrated table is written canonically
    assert contents(exporter.read_export(tmp_path / "c.gdml").detector_file) == contents(apply(df, c.values))


def test_a_changed_value_is_written_canonically_and_tokens_round_trip_deterministically(tmp_path):
    df = importer.read_gdml(write(tmp_path, TOKENS))
    pls = (dataclasses.replace(df.placements[0], position=(0.02, 0.0, 0.01)), *df.placements[1:])
    exporter.export(dataclasses.replace(df, placements=pls), tmp_path / "m.gdml", scope="full")
    assert '<position unit="mm" x="20" y="0" z="10"/>' in (tmp_path / "m.gdml").read_text()
    _, back, (first, second) = round_trip(tmp_path, df)
    assert back.detector_file.tokens == df.tokens and first == second
    assert importer.read_gdml(write(tmp_path, TOKENS, "again.gdml")).tokens == df.tokens


def test_a_larsoft_reflector_is_written_as_its_geant4_form(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    e = np.array([7.0, 9.0, 10.0])
    props = {
        "REFLECTIVITY": schema.Property("REFLECTIVITY", np.full(3, 0.25), e, "", "recipe", "R"),
        "DIFFUSE_FRACTION": schema.Property("DIFFUSE_FRACTION", np.full(3, 0.5), e, "", "recipe", "f"),
    }
    steel = schema.Surface(
        "steel",
        "dielectric_metal",
        "unified",
        "ground",
        0.0,
        props,
        schema.OpticsModel("lumen:larsoft_reflector", 1, {}, "recipe", "LArSoft"),
    )
    toks = {"surfaces": {"steel": {k: {"energy": ["7.0*eV", "9.0*eV", "10.0*eV"], "matrix_unit": None, "value": None}
                                   for k in props}}}  # fmt: skip
    df = dataclasses.replace(
        df,
        surfaces={**df.surfaces, "steel": steel},
        tokens={**df.tokens, **toks},
        skins=df.skins + (schema.Skin("steel:PmtLV", "steel", "PmtLV", "recipe", "steel"),),
    )
    rep, back, (first, second) = round_trip(tmp_path, df)
    assert rep.lossless and "--refuse-lossy" not in rep.g4ref_args and first == second
    assert contents(back.detector_file) == contents(df) and back.detector_file.tokens == df.tokens
    g4 = importer.read_gdml(tmp_path / "a" / "x.gdml")  # what Geant4 reads
    s = g4.surfaces["steel"]
    assert sorted(s.properties) == ["BACKSCATTERCONSTANT", "REFLECTIVITY", "SPECULARLOBECONSTANT",
                                    "SPECULARSPIKECONSTANT"]  # fmt: skip
    assert list(s.properties["SPECULARSPIKECONSTANT"].value) == [0.5] * 3 and s.value == 0.0
    assert "7.0*eV 0.25" in (tmp_path / "a" / "x.gdml").read_text()
