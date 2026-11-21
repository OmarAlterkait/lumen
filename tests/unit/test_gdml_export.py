"""The GDML exporter: exact numbers, what lumen reads back from an export, byte determinism, what is left out."""

import dataclasses
import importlib.util
import struct
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from export_toys import TOYS
from readout_toys import TOY_SET
from test_importer import GDML

from lumen.format import calibrations, io, schema, sensorsets
from lumen.format import solids as ss
from lumen.format.apply import apply
from lumen.gdml import exporter, importer, reader, units
from lumen.recipe import Recipe
from lumen.rules.optics import VALUE_RULES

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "g4ref" / "sensor_args.py"
_spec = importlib.util.spec_from_file_location("sensor_args", _TOOL)
g4ref_args = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g4ref_args)  # tools/g4ref/sensor_args.py: the g4ref arguments of a sensor set

N_RANDOM = 20000  # random doubles per dimension in the literal test


def write(tmp_path, text, name="toy.gdml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def contents(df):
    """Every content file (the records' values, origins and notes), the world and the detector volume; a surface's
    model, finish and type by name (Geant4 reads "1" and "unified" alike; the export writes the name)."""
    named = {}
    for n, s in df.surfaces.items():
        model, finish, typ = exporter._enum_names(s) if s.optics.name == "geant4" else (s.model, s.finish, s.type)
        named[n] = dataclasses.replace(s, model=model, finish=finish, type=typ)
    f = io.to_files(dataclasses.replace(df, surfaces=named))
    return {k: v for k, v in f.items() if k != "detector.json"}, (df.world, df.detector_volume)


def round_trip(tmp_path, src, **kw):
    """Export, read the export back (``read_gdml``), export that again: (report, read-back, both exports' bytes)."""
    rep = exporter.export(src, tmp_path / "a" / "x.gdml", **kw)
    back = importer.read_gdml(tmp_path / "a" / "x.gdml", warn=False)
    exporter.export(back, tmp_path / "b" / "x.gdml", **kw)
    return rep, back, [(tmp_path / d / "x.gdml").read_bytes() for d in "ab"]


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
    assert w.pressure_pa == 101325.1 and w.mee_ev == 78.0000000001 and w.temperature_k == 293.15
    assert df.isotopes["U235"].molar_mass_g_mol == 235.0439 and df.elements["H"].molar_mass_g_mol == 1.00794
    assert df.materials["Graphite"].composition.molar_mass_g_mol == 12.011
    back = io.read(io.save(df, tmp_path / "t.lumen"))
    assert back.materials["Water"].mee_ev == 78.0000000001 and back.materials["Air"].mee_ev is None


# ------------------------------------------------------------------------------------------------- round trips
@pytest.mark.parametrize("name", sorted(TOYS) + ["importer"])
def test_an_export_reads_back_to_the_same_values_and_bytes(tmp_path, name):
    df = importer.read_gdml(write(tmp_path, GDML if name == "importer" else TOYS[name]))
    rep, back, (first, second) = round_trip(tmp_path, df)
    assert contents(back) == contents(df)  # every stored double read back
    assert first.split(b"-->", 1)[1] == second.split(b"-->", 1)[1] and rep.lossless  # the comment names the source
    assert exporter.export(df, tmp_path / "c" / "x.gdml").sha256 == rep.sha256  # two exports


def test_export_order_is_a_function_of_the_content(tmp_path):
    df = importer.read_gdml(write(tmp_path, TOYS["all_solids"]))
    saved = io.read(io.save(df, tmp_path / "t.lumen"))  # dicts in sorted order, not the document's
    a = exporter.export(df, tmp_path / "a.gdml")
    b = exporter.export(saved, tmp_path / "b.gdml")
    assert a.sha256 == b.sha256


def test_inert_matrices_are_written_and_refs_geant4_cannot_read_are_refused(tmp_path):
    """A property naming no matrix and a matrix that does not evaluate are refused, as Geant4 refuses them; a key no
    optical process reads is kept inert and written."""
    text = GDML.replace('<property name="ABSLENGTH" ref="ABS0x2"/>',
                        '<property name="ABSLENGTH" ref="ABS0x2"/><property name="USERKEY" ref="NOPE"/>')  # fmt: skip
    keys = frozenset({"RINDEX"})
    with pytest.raises(reader.GDMLError, match="names 'NOPE', not defined"):  # Matrix 'NOPE' was not found!
        importer.read_gdml(write(tmp_path, text, "undefined.gdml"), material_keys=keys)
    broken = text.replace("</define>", '<matrix name="NOPE" coldim="1" values="UNKNOWN"/>\n</define>', 1)
    with pytest.raises(reader.GDMLError, match="'NOPE'"):  # Error in expression: UNKNOWN
        importer.read_gdml(write(tmp_path, broken, "broken.gdml"), material_keys=keys)
    fine = text.replace("</define>", '<matrix name="NOPE" coldim="2" values="1 2 3 4"/>\n</define>', 1)
    df = importer.read_gdml(write(tmp_path, fine), material_keys=keys)
    assert set(df.materials["Water"].inert) == {"ABSLENGTH", "USERKEY"}
    exporter.export(df, tmp_path / "x.gdml")
    back = importer.read_gdml(tmp_path / "x.gdml", material_keys=keys, warn=False)
    assert contents(back) == contents(df)


def test_borders_read_back(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))
    _, back, _ = round_trip(tmp_path, df)
    assert back.borders == df.borders


def _toy_recipe():
    def decide(df, inputs):
        o = schema.OpticsModel("lumen:opaque", settings={}, origin="recipe", note="toy")
        steel = dataclasses.replace(
            df.materials["Steel"],
            optics=o,
            properties={
                "RINDEX": schema.Property(
                    "RINDEX", np.array([1.5, 1.6]), energy=np.array([2.0, 3.0]), origin="recipe", note="a table"
                ),
                "LUMENONLY": schema.Property(
                    "LUMENONLY", np.asarray(0.5), energy=None, origin="recipe", note="not a Geant4 key"
                ),
            },
        )
        brdf = schema.Surface(
            "brdf",
            type="dielectric_metal",
            model="unified",
            finish="ground",
            value=0.0,
            properties={
                "REFLECTIVITY": schema.Property(
                    "REFLECTIVITY", np.array([0.5, 0.6]), energy=np.array([2.0, 3.0]), origin="recipe", note="r"
                )
            },
            optics=schema.OpticsModel("acme:brdf", settings={}, origin="recipe", note="no Geant4 form"),
        )
        skins = df.skins + (schema.Skin("brdf", surface="brdf", volume="WorldLV", origin="recipe", note="lossy"),)
        borders = df.borders + (schema.Border("b2", surface="brdf", pv1="pmt0x10", pv2="tank"),)
        return dataclasses.replace(df, materials={**df.materials, "Steel": steel}, skins=skins, borders=borders,
                                   surfaces={**df.surfaces, "brdf": brdf})  # fmt: skip

    group = sensorsets.SensorGroup("pmt", rule="border_into", lv="PmtLV",
                                   settings={"entry_efficiency": 1.0}, readout={"gate": [-1.0, 2.0]})  # fmt: skip
    return Recipe("toy", "TankLV", groups=(group,), decisions=(decide,),
                  sensor_columns=lambda df, paths, inputs: {"channel": [3, 4]})  # fmt: skip


def test_what_geant4_cannot_read_is_left_out_and_listed(tmp_path):
    result = importer.import_gdml(write(tmp_path, GDML), _toy_recipe())
    path = io.save(result.detector_file, tmp_path / "toy.lumen")
    sensorsets.save(result.sensor_set, path)
    with pytest.raises(exporter.ExportError, match="no Geant4 form"):
        exporter.export(path, tmp_path / "x.gdml")
    rep = exporter.export(path, tmp_path / "x.gdml", on_lossy="omit")
    assert not rep.lossless
    assert g4ref_args.args(result.sensor_set, rep.lossless) == ["--sens-regex", "^PmtLV$", "--wcsim-hits",
                                                                "--refuse-lossy"]  # fmt: skip
    assert any("brdf" in x for x in rep.omitted) and any("Geant4 reads them" in w for w in rep.warnings)
    assert any("LUMENONLY" in a and "not written" in a for a in rep.annotations)
    gdml = (tmp_path / "x.gdml").read_text()
    assert "brdf" not in gdml and "LUMENONLY" not in gdml and "<opticalsurface" in gdml
    back = importer.read_gdml(tmp_path / "x.gdml", warn=False)
    df = result.detector_file
    np.testing.assert_array_equal(back.materials["Steel"].properties["RINDEX"].value, [1.5, 1.6])  # a Geant4 key
    assert back.solids == df.solids and back.placements == df.placements and back.volumes == df.volumes


def test_a_calibration_is_checked_and_applied_before_writing(tmp_path):
    from lumen.format import io, sensorsets

    df = importer.read_gdml(write(tmp_path, GDML))
    root, sset = io.save(df, tmp_path / "d.lumen"), sensorsets.SensorSet(())
    sensorsets.save(sset, root)
    digest = io.content_digest(io.to_files(df), sensorsets.files(sset))
    c = calibrations.Calibration("c1", "toy", digest, {
        "materials/Water/RINDEX": calibrations.Table(np.array([1.5, 2.5, 4.5]), np.array([1.3, 1.31, 1.4])),
        "surfaces/Liner/sigma_alpha": 0.25, "surfaces/Liner/REFLECTIVITY": calibrations.Table(np.array([2.0, 4.5]),
                                                                                         np.array([0.5, 0.4])),
        "sensors/pmt/efficiency": np.array([0.9, 0.8])})  # fmt: skip
    with pytest.raises(exporter.ExportError, match="give the detector file's path"):
        exporter.export(df, tmp_path / "m.gdml", calibration=c)
    with pytest.raises(calibrations.CalibrationMismatch, match="made for detector file"):
        exporter.export(root, tmp_path / "w.gdml", calibration=dataclasses.replace(c, detector_digest="sha256:x"))
    rep = exporter.export(root, tmp_path / "c.gdml", calibration=c)
    back = importer.read_gdml(tmp_path / "c.gdml", warn=False)
    assert contents(back)[0] == contents(apply(df, c.values, VALUE_RULES))[0] and rep.calibration == "c1"
    with pytest.raises(ValueError, match="needs a Table"):
        apply(df, {"materials/Water/RINDEX": 1.3}, VALUE_RULES)
    with pytest.raises(ValueError, match="a new knot count is a new detector file"):  # the one rule, also on export
        exporter.export(root, tmp_path / "k.gdml", calibration=dataclasses.replace(c, values={
            "materials/Water/RINDEX": calibrations.Table(np.array([1.5, 4.5]), np.array([1.3, 1.4]))}))  # fmt: skip


# ----------------------------------------------------------------------------------------------- the tables
def test_solid_spellings_are_geant4s():
    assert ss.dimensions("cutTube")["lowX"] == ss.U and "lowx" not in ss.dimensions("cutTube")


def test_enum_and_key_tables_agree_with_the_optics_rules():
    from lumen.rules import optics

    assert optics.MATERIAL_KEYS >= exporter.GEANT4_KEYS  # one table: lumen.rules.keys


def test_the_opticalsurface_value_is_keyed_by_what_the_model_reads(tmp_path):
    df = importer.read_gdml(write(tmp_path, GDML))  # Liner: unified, ground: its value is sigma_alpha
    assert apply(df, {"surfaces/Liner/sigma_alpha": 0.25}, VALUE_RULES).surfaces["Liner"].value == 0.25
    with pytest.raises(ValueError, match="reads its value as sigma_alpha"):
        apply(df, {"surfaces/Liner/polish": 0.5}, VALUE_RULES)


def test_a_sensor_model_that_applies_the_efficiency_after_tracking_exports_wcsim_hits():
    # the library counts every absorption entering under lumen:sensor (efficiency 1 on entry): so does the reference
    group = sensorsets.SensorGroup("pmt", rule="border_into", lv="PmtLV", values=TOY_SET)  # fmt: skip
    args = g4ref_args.args(sensorsets.SensorSet((group,)))
    assert args == ["--sens-regex", "^PmtLV$", "--wcsim-hits"]
    plain = dataclasses.replace(group, values=None)
    assert "--wcsim-hits" not in g4ref_args.args(sensorsets.SensorSet((plain,)))


def test_a_value_written_with_a_unit_is_exported_as_the_bare_literal_lumen_read(tmp_path):
    """Geant4's GDML evaluator reads "24000/MeV" as 23999.9896 (its MeV is the 1993 electron charge's: 1.0000004344);
    lumen reads 24000 (CLHEP's MeV, lumen.gdml.reader). The export writes the bare literal 24000, which both read
    alike, so Geant4 runs the detector lumen simulates."""
    text = GDML.replace(
        '<property name="ABSLENGTH" ref="ABS0x2"/>',
        '<property name="ABSLENGTH" ref="ABS0x2"/><property name="SCINTILLATIONYIELD" ref="Y"/>',
    )
    text = text.replace("</define>", '<matrix name="Y" coldim="1" values="24000/MeV"/>\n</define>', 1)
    df = importer.read_gdml(write(tmp_path, text))
    assert float(df.materials["Water"].properties["SCINTILLATIONYIELD"].value) == 24000.0
    exporter.export(df, tmp_path / "y.gdml")
    assert 'coldim="1" values="24000"' in (tmp_path / "y.gdml").read_text()
    assert contents(importer.read_gdml(tmp_path / "y.gdml", warn=False)) == contents(df)


def test_records_of_one_stripped_name_read_back_to_the_same_references(tmp_path):
    """Geant4 resolves a reference by its stripped name to the first record of it (G4Isotope::GetIsotope, ...), and so
    does the import. Two isotopes and two elements share a stripped name, the later-sorted one first in the file: the
    export writes each reference's target first, so lumen (and Geant4) read the same references back."""
    iso = ('<isotope name="H10xb" Z="1" N="1"><atom value="1.00782503081372"/></isotope>'
           '<isotope name="H10xa" Z="1" N="1"><atom value="1.00782503081372"/></isotope>')  # fmt: skip
    els = ('<element name="Hy0xb"><fraction n="1" ref="H10xa"/></element>'
           '<element name="Hy0xa"><fraction n="1" ref="H10xa"/></element>')  # fmt: skip
    text = GDML.replace('<element name="H" formula="H" Z="1"> <atom value="1.008"/> </element>', iso + els, 1)
    text = text.replace('<composite n="2" ref="H"/>', '<composite n="2" ref="Hy0xa"/>', 1)
    df = importer.read_gdml(write(tmp_path, text))
    assert [x.ref for x in df.elements["Hy0xa"].isotopes] == ["H10xb"]  # the first H1 in the file
    assert df.materials["Water"].composition.components[0].ref == "Hy0xb"  # the first Hy in the file
    _, back, _ = round_trip(tmp_path, df)
    assert contents(back) == contents(df)
