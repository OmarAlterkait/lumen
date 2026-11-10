"""LArSoft's public optics (lumen.detectors.larsoft_optics): the tables as detector-file Property records give the
baseline's LAr medium and steel reflector exactly; and the compile-level part of the baseline's
tests/test_lartpc_toy.py (an LAr sensor in LAr has its own class, a same-material active volume is no boundary)."""

from decimal import Decimal

import numpy as np

from lumen.build import surfaces as regions
from lumen.build.tree import PV, Tree
from lumen.detectors import larsoft_optics as lar
from lumen.format import schema
from lumen.media import bulk, laws, phase
from lumen.surfaces import _legacy as legacy
from lumen.units import HC

STEEL = "STEEL_STAINLESS_Fe7Cr2Ni"


def test_tables_are_property_records_in_detector_file_units():
    d = lar.data()["lar"]["larsoft_standard"]
    t = lar.lar_tables()
    assert set(t) == {"RINDEX", "RAYLEIGH", "ABSLENGTH"} and all(p.origin == "recipe" for p in t.values())
    assert np.array_equal(t["RINDEX"].energy, d["rindex"]["energy"]) and np.all(np.diff(t["RAYLEIGH"].energy) > 0)
    for key, name in (("RAYLEIGH", "rayleigh_length"), ("ABSLENGTH", "abs_length")):  # cm -> m, exact decimals
        want = np.array([float(Decimal(repr(float(v))) * Decimal("0.01")) for v in d[name]["values"]])
        assert np.array_equal(t[key].value, want[np.argsort(d[name]["energy"], kind="stable")])


def test_tables_decode_from_their_source_tokens():
    from lumen.gdml import tokens

    t, toks = lar.lar_tables(), lar.lar_tokens()
    for key, p in t.items():
        assert [tokens.energy(x) for x in toks[key]["energy"]] == list(p.energy)
        if toks[key]["value"] is not None:
            assert [tokens.length_value(x) for x in toks[key]["value"]] == list(p.value)
    r, rt = lar.reflector_tables("STEEL_STAINLESS_Fe7Cr2Ni"), lar.reflector_tokens("STEEL_STAINLESS_Fe7Cr2Ni")
    assert all([tokens.energy(x) for x in rt[k]["energy"]] == list(r[k].energy) for k in r)


def test_medium_from_records_is_the_shipped_medium():
    a, pa = lar.lar_medium()
    rec = schema.Material("LAr", "liquid", schema.Composition(1.39), properties=lar.lar_tables())
    b, pb = lar.lar_medium_from(rec, "lar")
    assert a.name == b.name == "lar" and a.index.form == b.index.form == "g4_rindex"
    assert [c.name for c in b.channels] == ["rayleigh"] and b.channels[0].lobe is phase.RAYLEIGH
    for k in ("index", "absorption"):
        for q in ("energy", "value"):
            assert pa[k][q].tobytes() == pb[k][q].tobytes()
    assert pa["rayleigh"]["length"]["value"].tobytes() == pb["rayleigh"]["length"]["value"].tobytes()
    bulk.validate(b, pb, 100.0, 180.0)
    m, p = lar.lar_medium(scatter=False, absorb=False)
    assert not m.channels and m.absorption is None and set(p) == {"index"}
    # the index is Geant4's: n at 128 nm is the table linear in energy
    n = float(m.index({"energy": p["index"]["energy"], "value": p["index"]["value"]}, np.float32(128.0)))
    assert abs(n - np.interp(HC / 128.0, p["index"]["energy"], p["index"]["value"])) < 1e-5


def test_reflector_from_records():
    kind, p = lar.reflector(STEEL)
    props = lar.reflector_tables(STEEL)
    assert set(props) == {"REFLECTIVITY", "DIFFUSE_FRACTION"} and kind is legacy.REFLECTOR
    assert lar.reflector_from(props) == (kind, p)
    assert p == {"R": 0.25, "f_spec": 0.5, "f_glossy": 0.0, "sigma_alpha": 0.1}


def test_scintillation_overrides():
    s = lar.lar_scintillation(**{"yield": 40000.0, "f_fast": 0.23})
    assert s["yield"] == 40000.0 and s["f_fast"] == 0.23 and s["tau_fast"] == 6.0 and s["tau_slow"] == 1590.0
    lam = np.asarray(s["spectrum"].sample(np.float32(0.5)))
    assert 100.0 < float(lam) < 180.0


def test_lar_sensor_in_lar_is_its_own_class_and_the_active_volume_is_invisible():
    lar_m = bulk.Medium("lar", laws.constant("index"), laws.constant("length"))
    library = regions.Library(media={"lar": regions.MediumEntry(lar_m, {"index": {"value": 1.36},
                                                                        "absorption": {"value": 1e9}}),
                                     "steel": regions.MediumEntry(bulk.Medium("steel", None), {})},
                              sensors={"arapuca": regions.SensorEntry(legacy.COUNTING, {"eff": 1.0})})  # fmt: skip
    pvs = (PV("Hall", "HallLV", "Steel"), PV("Cryo", "CryoLV", "LAr", "Hall"), PV("det0", "DetLV", "LAr", "Cryo", 0),
           PV("TPC", "TPCLV", "LAr", "Cryo"))  # fmt: skip
    b = regions.Bindings(
        materials={"LAr": "lar", "Steel": "steel"},
        sensors=(regions.SensorBinding("arapuca", 1, lv="DetLV"),),
        disjoint=(("TPC", "det0"),),
    )
    o = regions.compile(library, Tree(pvs), b, (120.0, 140.0))
    c = o.class_of
    assert c["TPC"] == c["Cryo"] != c["det0"] and o.kernel("Cryo", "TPC") is None
    assert o.kernel("Cryo", "det0").kind is legacy.COUNTING and o.tag_of["det0"] == 0
