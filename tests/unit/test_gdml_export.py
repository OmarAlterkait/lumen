"""The GDML exporter: exact numbers, the round trip and byte determinism on toys, the recipe layer in the sidecar."""

import dataclasses
import struct
from decimal import Decimal

import numpy as np
import pytest
from export_toys import TOYS
from test_importer import GDML

from lumen.format import calibrations, io, schema, sensorsets
from lumen.format import overlay as fo
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


def test_inert_matrices_and_unreadable_ones_round_trip(tmp_path):
    text = GDML.replace('<property name="ABSLENGTH" ref="ABS0x2"/>',
                        '<property name="ABSLENGTH" ref="ABS0x2"/><property name="USERKEY" ref="NOPE"/>')  # fmt: skip
    df = importer.read_gdml(write(tmp_path, text), material_keys=frozenset({"RINDEX"}))
    water = df.materials["Water"]
    assert set(water.inert) == {"ABSLENGTH", "USERKEY"} and water.inert["USERKEY"].shape == (0, 1)
    _, back, (first, second) = round_trip(tmp_path, df)
    assert contents(back.detector_file) == contents(df) and first == second
    assert b'name="USERKEY"' not in first  # an empty matrix is fatal in Geant4 (G4GDMLEvaluator::DefineMatrix)


def test_unbound_borders_and_surface_inert_keys_round_trip(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    assert df.surfaces["Liner"].inert == ("GROUPVEL",)
    df = dataclasses.replace(df, borders=tuple(dataclasses.replace(b, bound=False, note="outside") for b in df.borders))
    _, back, _ = round_trip(tmp_path, df)
    assert back.detector_file.borders == df.borders and back.detector_file.surfaces["Liner"].inert == ("GROUPVEL",)


def _toy_recipe():
    def decide(df, inputs):
        o = schema.OpticsModel("lumen:opaque", 1, {"medium": "steel", "axes": [[1.0, 0.0, 0.0]]}, "recipe", "toy")
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
            "film", "", "", "", 0.0, {}, schema.OpticsModel("lumen:wls_film", 1, {"d": 2e-8}, "recipe", "lowered")
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
    return Recipe("toy", "TankLV", (300.0, 600.0), sensorsets.SensorVolumes(names="^PmtLV"), (group,), (decide,),
                  sensor_columns=lambda df, loaded, tree, inputs: {"channel": [3, 4]})  # fmt: skip


def test_the_recipe_layer_and_the_sensor_set_round_trip_through_the_sidecar(tmp_path):
    result = importer.import_gdml(write(tmp_path, GDML), _toy_recipe())
    with pytest.raises(exporter.ExportError, match="no Geant4 form"):
        exporter.export(result, tmp_path / "x.gdml", scope="full")
    rep, back, (first, second) = round_trip(tmp_path, result, on_lossy="omit")
    df = result.detector_file
    assert contents(back.detector_file) == contents(df) and first == second
    assert not rep.lossless and rep.g4ref_args == ("--sens-regex", "^PmtLV", "--wcsim-hits", "--refuse-lossy")
    assert any("brdf" in x for x in rep.omitted) and any("Geant4 reads them" in w for w in rep.warnings)
    s0, s1 = result.sensor_sets["default"], back.sensor_sets["default"]
    assert sensorsets.to_json(s0) == sensorsets.to_json(s1) and back.sensor_columns == {"channel": [3, 4]}
    gdml = (tmp_path / "a" / "x.gdml").read_text()
    assert "film" not in gdml and "brdf" not in gdml and "LUMENONLY" not in gdml and "<opticalsurface" in gdml


def test_a_calibration_is_applied_before_writing(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    c = calibrations.Calibration("c1", "toy", "sha256:x", {
        "materials/Water/RINDEX": calibrations.Table(np.array([1.5, 2.5, 3.5, 4.5]), np.array([1.3, 1.31, 1.32, 1.4])),
        "surfaces/Liner/value": 0.25, "surfaces/Liner/REFLECTIVITY": calibrations.Table(np.array([2.0, 4.5]),
                                                                                         np.array([0.5, 0.4])),
        "sensors/pmt/efficiency": np.array([0.9, 0.8])})  # fmt: skip
    rep = exporter.export(df, tmp_path / "c.gdml", calibration=c, scope="full")
    back = exporter.read_export(tmp_path / "c.gdml")
    assert contents(back.detector_file)[0] == contents(fo.apply(df, [c]))[0] and rep.calibration == ("c1",)
    side = (tmp_path / "c.gdml.lumen.json").read_text()
    assert '"sensors/pmt/efficiency"' in side
    with pytest.raises(ValueError, match="a table given for a constant|needs a Table"):
        fo.apply(df, [dataclasses.replace(c, values={"materials/Water/RINDEX": 1.3})])


def test_names_geant4_strips_at_0x_and_refuses_collisions(tmp_path):
    df = importer.read_gdml(write(tmp_path, TOYS["j_names"]))
    exporter.export(df, tmp_path / "g.gdml", scope="full", names="geant4")
    text = (tmp_path / "g.gdml").read_text()
    assert "0x" not in text.split("-->", 1)[1]
    with pytest.raises(exporter.ExportError, match="once stripped"):
        exporter.export(importer.read_gdml(write(tmp_path, GDML)), tmp_path / "h.gdml", scope="full", names="geant4")


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
