"""lumen's reader refuses what stock Geant4 refuses, and repairs nothing; ``python -m lumen.gdml.repair`` is the one
repair, run apart, and its copy is what both Geant4 and lumen read.

tests/data/gdml_repairs holds one GDML toy per case and geant4.json, what Geant4 10.7.4 and 11.2.2 did with each
(g4ref, load and exit) and with each toy repaired by python -m lumen.gdml.repair (the repaired text's sha256).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lumen.gdml import importer, reader
from lumen.gdml import repair as gdml_repair
from lumen.units import HUGE

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "gdml_repairs"
G4 = json.loads((DATA / "geant4.json").read_text())["toys"]
REPAIRABLE = sorted(n for n, t in G4.items() if "repaired" in t)


def test_the_toys_and_both_geant4_versions_agree():
    assert set(G4) == {p.stem for p in DATA.glob("*.gdml")}
    for name, t in G4.items():
        a, b = t["geant4"]["10.7.4"], t["geant4"]["11.2.2"]
        assert a == b, name  # the same verdict and message in both versions
        assert a["loads"] == (t["case"] == "accepted"), name
        assert t.get("repaired", {}).get("geant4", {"10.7.4": True, "11.2.2": True}) == {"10.7.4": True, "11.2.2": True}
    assert set(gdml_repair.REPAIRS) == {t["case"] for t in G4.values() if "repaired" in t}


@pytest.mark.parametrize("name", sorted(G4))
def test_lumen_refuses_what_geant4_refuses(name):
    """A file Geant4 refuses is a GDMLError naming the repair (when there is one); a file it loads is read."""
    t, path = G4[name], DATA / f"{name}.gdml"
    if t["case"] == "accepted":
        importer.read_gdml(path, warn=False)
        return
    with pytest.raises(reader.GDMLError) as err:
        importer.read_gdml(path, warn=False)
    if "repaired" in t:
        assert f"the repair {t['case']!r}" in str(err.value) and "python -m lumen.gdml.repair" in str(err.value)


@pytest.mark.parametrize("name", REPAIRABLE)
def test_the_repaired_copy_is_what_geant4_loaded_and_lumen_reads(tmp_path, name):
    t = G4[name]
    src = DATA / f"{name}.gdml"
    text, lines = gdml_repair.repair(src.read_text())  # every repair: each edits only its own defect
    assert lines and hashlib.sha256(text.encode()).hexdigest() == t["repaired"]["sha256"]
    (tmp_path / src.name).write_text(text)
    df = importer.read_gdml(tmp_path / src.name, warn=False)
    assert df.provenance["source"]["sha256"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _repaired(tmp_path, name):
    (tmp_path / f"{name}.gdml").write_text(gdml_repair.repair((DATA / f"{name}.gdml").read_text())[0])
    return importer.read_gdml(tmp_path / f"{name}.gdml", warn=False)


def test_what_each_repair_writes(tmp_path):
    water = _repaired(tmp_path, "matrix_tail").materials["Water"].properties
    np.testing.assert_array_equal(water["ABSLENGTH"].value, [1.0, 2.0])  # m: the tail (5e-06) dropped
    water = _repaired(tmp_path, "constant_property").materials["Water"]
    assert "SCINTILLATIONYIELD" not in water.properties and "SCINTILLATIONYIELD" not in water.inert
    water = _repaired(tmp_path, "constant_property_optical").materials["Water"]
    assert "RAYLEIGH" not in water.properties  # the GDML's constant deleted
    from lumen.rules import optics

    assert optics.geant4_rayleigh(water) is not None  # Geant4's own table of Water, derived at load
    assert _repaired(tmp_path, "denormal_attribute").placements[0].position == (0.0, 0.0, 0.0)
    abs_ = _repaired(tmp_path, "denormal_matrix").materials["Water"].properties["ABSLENGTH"].value
    np.testing.assert_array_equal(abs_, [1.0, 0.0])
    rix = _repaired(tmp_path, "neg_inf_matrix").materials["Water"].properties["RINDEX"].value
    np.testing.assert_array_equal(rix, [1.33, -HUGE])


def test_a_constant_on_a_table_key_is_left_out_on_surfaces_and_materials_alike(tmp_path):
    """G4GDMLReadSolids / Materials::PropertyRead store a coldim 1 matrix as a constant property, which no process
    reads under a table key (GetProperty): Geant4 loads the file and the key is absent, on both owners."""
    df = importer.read_gdml(DATA / "constant_surface_table.gdml", warn=False)
    win = df.surfaces["Win"]
    assert "TRANSMITTANCE" not in win.properties and "REFLECTIVITY" in win.properties  # Geant4's T: its default 0
    assert any("opticalsurface 'Win'.TRANSMITTANCE: left out" in ln for ln in df.report)
    text = (DATA / "constant_surface_table.gdml").read_text()
    text = text.replace('<property name="RINDEX" ref="RIX"/>', '<property name="RINDEX" ref="STRANS"/>')
    (tmp_path / "m.gdml").write_text(text)
    df = importer.read_gdml(tmp_path / "m.gdml", warn=False)
    assert "RINDEX" not in df.materials["Water"].properties
    assert any("material 'Water'.RINDEX: left out" in ln for ln in df.report)


def test_geant4s_value_count_and_redefinitions_it_accepts():
    """A trailing blank is one more value (0) to Geant4; an identical position re-definition is accepted."""
    doc = reader.read(DATA / "matrix_trailing_space_odd.gdml")
    np.testing.assert_array_equal(doc.matrix("ABS"), [[2e-6, 1000.0], [4e-6, 2000.0], [5e-6, 0.0]])
    assert any("end in whitespace" in ln for ln in doc.report)
    doc = reader.read(DATA / "redefinition_position.gdml")
    assert doc.positions["P1"] == (0.001, 0.0, 0.0) and any("Geant4 keeps the last" in ln for ln in doc.report)


def test_attribute_names_are_matched_exactly(tmp_path):
    """Geant4 reads ``x`` and ignores ``X`` (it would read x as 0): lumen refuses the file."""
    text = (DATA / "clean.gdml").read_text().replace('<box name="Tank" x=', '<box name="Tank" X=', 1)
    (tmp_path / "upper.gdml").write_text(text)
    with pytest.raises(reader.GDMLError, match="attribute 'X' is not one Geant4 reads for box"):
        importer.read_gdml(tmp_path / "upper.gdml", warn=False)


def test_the_tool_checks_the_repair_names():
    with pytest.raises(ValueError, match="unknown repairs"):
        gdml_repair.repair("<gdml/>", ["tail"])


def test_the_tool_keeps_every_other_byte():
    src = (DATA / "denormal_matrix.gdml").read_text()
    text, _ = gdml_repair.repair(src, ["denormal"])
    assert text == src.replace("2.2305392e-316", "0")
    clean = (DATA / "clean.gdml").read_text()
    assert gdml_repair.repair(clean, list(gdml_repair.REPAIRS)) == (clean, [])


def test_the_reader_the_repair_and_the_exporter_share_one_set_of_lexical_rules():
    from lumen.gdml import exporter, lexical, reader, repair

    for name in ("SKIP_LITERALS", "NUMBER", "BIG_EXPONENT", "TINY"):
        assert getattr(reader, name) is getattr(lexical, name) and getattr(repair, name) is getattr(lexical, name)
    assert exporter.TINY is lexical.TINY
