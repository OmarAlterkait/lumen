"""The exporter defects stock Geant4 refuses: lumen repairs each, always, with a warning naming the object, what Geant4
does and the repair, and the repair is the one lumen.gdml.repair applies to a Geant4 reference input.

tests/data/gdml_repairs holds one GDML toy per case and geant4.json, what Geant4 10.7.4 and 11.2.2 did with each
(g4ref, load and exit) and with each toy repaired by refs/gdml_repair.py (the repaired text's sha256).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lumen.format import io
from lumen.gdml import importer, reader
from lumen.gdml import repair as gdml_repair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "gdml_repairs"
G4 = json.loads((DATA / "geant4.json").read_text())["toys"]
REPAIRABLE = sorted(n for n, t in G4.items() if "repaired" in t)


def _contents(df):
    """Every content file, and the report without the repair lines."""
    f = {k: v for k, v in io.to_files(df).items() if k != "detector.json"}
    return f, [ln for ln in df.report if not ln.startswith("repaired (")]


def test_the_toys_and_both_geant4_versions_agree():
    assert set(G4) == {p.stem for p in DATA.glob("*.gdml")}
    for name, t in G4.items():
        a, b = t["geant4"]["10.7.4"], t["geant4"]["11.2.2"]
        assert a == b, name  # the same verdict and message in both versions
        assert a["loads"] == (t["case"] == "accepted" or "geant4_ignores" in t), name  # loads, reads it otherwise
        assert t.get("repaired", {}).get("geant4", {"10.7.4": True, "11.2.2": True}) == {"10.7.4": True, "11.2.2": True}
    assert set(gdml_repair.REPAIRS) == set(reader.REPAIRS) == {t["case"] for t in G4.values() if "repaired" in t}


@pytest.mark.parametrize("name", sorted(G4))
def test_lumen_repairs_what_geant4_refuses_with_a_warning(name):
    """A repairable defect is read with one warning per repair naming the object, Geant4's behaviour and the tool;
    a case Geant4 refuses that has no repair is still refused, and an accepted file gives no repair warning."""
    t, path = G4[name], DATA / f"{name}.gdml"
    if "repaired" in t:
        with pytest.warns(reader.GDMLRepairWarning) as seen:
            df = importer.read_gdml(path)
        msgs = [str(w.message) for w in seen if issubclass(w.category, reader.GDMLRepairWarning)]
        assert all(f"exporter defect {t['case']!r}" in m and "Stock Geant4" in m and str(path) in m
                   and "python -m lumen.gdml.repair SRC DST" in m for m in msgs), msgs  # fmt: skip
        assert df.provenance["repairs"] == [t["case"]]
        return
    if t["case"] == "accepted":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", reader.GDMLRepairWarning)
            importer.read_gdml(path)
        return
    with pytest.raises((reader.GDMLError, importer.ImportRefused)):
        importer.read_gdml(path)


@pytest.mark.parametrize("name", REPAIRABLE)
def test_the_repair_is_the_reference_side_repair(tmp_path, name):
    """lumen reads the toy with its repair as it reads refs/gdml_repair.py's output (the file Geant4 loaded) without."""
    t = G4[name]
    src = DATA / f"{name}.gdml"
    text, lines = gdml_repair.repair(src.read_text())  # every repair: each edits only its own defect
    assert lines and hashlib.sha256(text.encode()).hexdigest() == t["repaired"]["sha256"]
    (tmp_path / src.name).write_text(text)
    with pytest.warns(reader.GDMLRepairWarning):
        a = importer.read_gdml(src)
    b = importer.read_gdml(tmp_path / src.name)
    assert _contents(a) == _contents(b)
    assert any(ln.startswith(f"repaired ({t['case']})") for ln in a.report)
    assert a.provenance["repairs"] == [t["case"]] and "repairs" not in b.provenance


def test_what_each_repair_reads():
    def read(name, kind):
        with pytest.warns(reader.GDMLRepairWarning, match=kind):
            return importer.read_gdml(DATA / f"{name}.gdml")

    water = read("matrix_tail", "matrix_tail").materials["Water"].properties
    np.testing.assert_array_equal(water["ABSLENGTH"].value, [1.0, 2.0])  # m: the tail (5e-06) dropped
    water = read("constant_property", "constant_property").materials["Water"]
    assert "SCINTILLATIONYIELD" not in water.properties and "SCINTILLATIONYIELD" not in water.inert
    assert "RAYLEIGH" not in read("constant_property_optical", "constant_property").materials["Water"].properties
    assert read("denormal_attribute", "denormal").placements[0].position == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(read("denormal_matrix", "denormal").materials["Water"].properties["ABSLENGTH"].value,
                                  [1.0, 0.0])  # fmt: skip
    rix = read("neg_inf_matrix", "non_finite").materials["Water"].properties["RINDEX"].value
    np.testing.assert_array_equal(rix, [1.33, -1e30])
    win = read("constant_surface_table", "constant_surface_table").surfaces["Win"]
    assert "TRANSMITTANCE" not in win.properties and "REFLECTIVITY" in win.properties  # Geant4's T: its default 0


def test_geant4s_value_count_and_redefinitions_it_accepts():
    """A trailing blank is one more value (0) to Geant4; an identical position re-definition is accepted."""
    doc = reader.read(DATA / "matrix_trailing_space_odd.gdml")
    np.testing.assert_array_equal(doc.matrix("ABS"), [[2e-6, 1000.0], [4e-6, 2000.0], [5e-6, 0.0]])
    assert any("end in whitespace" in ln for ln in doc.report)
    doc = reader.read(DATA / "redefinition_position.gdml")
    assert doc.positions["P1"] == (0.001, 0.0, 0.0) and any("Geant4 keeps the last" in ln for ln in doc.report)


def test_the_tool_checks_the_repair_names():
    with pytest.raises(ValueError, match="unknown repairs"):
        gdml_repair.repair("<gdml/>", ["tail"])


def test_the_tool_keeps_every_other_byte():
    src = (DATA / "denormal_matrix.gdml").read_text()
    text, _ = gdml_repair.repair(src, ["denormal"])
    assert text == src.replace("2.2305392e-316", "0")
    clean = (DATA / "clean.gdml").read_text()
    assert gdml_repair.repair(clean, list(gdml_repair.REPAIRS)) == (clean, [])
