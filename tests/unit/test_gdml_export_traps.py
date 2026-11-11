"""The Geant4 traps the exporter avoids: one assertion each, on the written GDML, and the refusals.

The traps are checked on the canonical writer (the toys' source tokens dropped); test_gdml_export checks the tokens.
"""

import dataclasses
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from export_toys import TOYS
from test_importer import GDML

from lumen.format import schema
from lumen.gdml import exporter, importer, units


@pytest.fixture
def written(tmp_path):
    """Export every toy; return {name: (detector file, the GDML's root element, its text)}."""
    out = {}
    for name, text in {**TOYS, "importer": GDML}.items():
        (tmp_path / f"{name}.gdml").write_text(text)
        df = dataclasses.replace(importer.read_gdml(tmp_path / f"{name}.gdml"), tokens={})
        exporter.export(df, tmp_path / f"x_{name}.gdml", scope="full")
        t = (tmp_path / f"x_{name}.gdml").read_text()
        out[name] = (df, ET.fromstring(t), t)
    return out


def test_matrices_have_no_unit_and_bare_literal_values(written):
    for _, root, _ in written.values():
        for m in root.iter("matrix"):
            assert "unit" not in m.attrib
            v = m.get("values")
            assert v == v.strip() and "  " not in v and all(units.LITERAL.match(t) for t in v.split())


def test_no_expressions_constants_or_default_namespace(written):
    for _, root, text in written.values():
        assert root.tag == "gdml" and "xmlns=" not in text  # no default namespace
        assert not list(root.iter("constant")) and not list(root.iter("variable")) and not list(root.iter("quantity"))
        for el in root.iter():
            for k, v in el.attrib.items():
                if k in ("x", "y", "z", "value", "n") or k in ("rmin", "rmax", "lowX", "zPosition", "deltaphi"):
                    assert units.LITERAL.match(v), (el.tag, k, v)


def test_lengths_and_angles_name_their_units(written):
    for _, root, _ in written.values():
        for el in root.iter():
            if el.tag in ("position", "firstposition"):
                assert el.get("unit") == "mm"
            if el.tag in ("rotation", "firstrotation"):
                assert el.get("unit") == "rad"
            if el.tag in ("box", "tube", "cutTube", "cone", "sphere", "polycone", "tet", "xtru", "trap"):
                assert el.get("lunit") == "mm"


def test_optical_surfaces_carry_all_four_attributes_by_name(written):
    seen = 0
    for _, root, _ in written.values():
        for s in root.iter("opticalsurface"):
            seen += 1
            assert s.get("model") in exporter.MODELS and s.get("finish") in exporter.FINISHES
            assert s.get("type") in exporter.TYPES and units.LITERAL.match(s.get("value"))
            assert not any(s.get(a).isdigit() for a in ("model", "finish", "type"))  # names, never enum positions
    assert seen
    df, root, _ = written["h_borders"]
    assert df.surfaces["Sensor"].model == "1"  # stored as the GDML gave it, written as "unified"
    assert {s.get("name"): s.get("model") for s in root.iter("opticalsurface")}["Sensor"] == "unified"


def test_cut_tube_spelling_and_tet_lunit(written):
    _, root, _ = written["all_solids"]
    ct = next(root.iter("cutTube"))
    assert {"lowX", "lowY", "lowZ", "highX", "highY", "highZ"} <= set(ct.attrib) and "lowx" not in ct.attrib
    tet = next(root.iter("tet"))
    assert tet.get("lunit") == "mm"  # G4GDMLReadSolids::TetRead scales the vertices by lunit a second time
    assert next(root.iter("section")).get("zPosition") is not None


def test_single_pass_order(written):
    for name, (df, root, _) in written.items():
        sections = [c.tag for c in root]
        assert sections == ["define", "materials", "solids", "structure", *(["userinfo"] if df.userinfo else []),
                            "setup"], name  # fmt: skip
        define = [c.tag for c in root.find("define")]
        assert define == sorted(define, key=lambda t: t != "matrix")  # matrices, then positions
        defined = set()
        for el in root.find("materials"):
            refs = [c.get("ref") for c in el if c.tag in ("fraction", "composite")]
            assert all(r in defined for r in refs), (name, el.get("name"))
            defined.add(el.get("name"))
        solids = root.find("solids")
        defined = set()
        for el in solids:
            refs = [c.get("ref") for c in el.iter() if c.tag in ("first", "second", "solid", "solidref")]
            assert all(r in defined for r in refs), (name, el.get("name"))
            defined.add(el.get("name"))
        tags = [el.tag for el in solids]
        assert "opticalsurface" not in tags or tags.index("opticalsurface") > max(
            i for i, t in enumerate(tags) if t != "opticalsurface"
        )
        vols = set()
        st = root.find("structure")
        for el in st:
            if el.tag in ("volume", "assembly"):
                assert all(pv.find("volumeref").get("ref") in vols for pv in el.iter("physvol"))
                vols.add(el.get("name"))
            else:
                assert len(vols) == len(df.volumes)  # skins and borders after every volume
        setups = list(root.iter("setup"))
        assert len(setups) == 1 and setups[0].get("name") == "Default"


def test_physvol_transforms_are_inline_and_only_when_not_plus_zero(written):
    _, root, _ = written["d_tessellated"]
    for pv in root.iter("physvol"):
        assert pv.find("positionref") is None and pv.find("rotationref") is None
    _, root, _ = written["j_names"]
    pvs = list(root.iter("physvol"))
    assert [pv.get("name") for pv in pvs] == ["lv0x55aa_PV", "lv0x55aa_PV"]  # Geant4's default names, written
    assert pvs[1].get("copynumber") == "7" and all(pv.find("rotation") is None for pv in pvs)


def test_tessellated_facets_are_absolute_triangles_on_one_position_per_vertex(written):
    df, root, _ = written["d_tessellated"]
    tess = next(root.iter("tessellated"))
    tris = list(tess)
    assert all(t.tag == "triangular" and t.get("type") == "ABSOLUTE" for t in tris)
    assert len(tris) == len(df.meshes["s_tessellated"])
    names = [p.get("name") for p in root.find("define").iter("position")]
    assert len(names) == len(set(names)) == 9  # the cube's 8 corners, and the -0.0 copy of one as its own position


# ------------------------------------------------------------------------------------------------- refusals
def _df(tmp_path, text=GDML):
    (tmp_path / "t.gdml").write_text(text)
    return importer.read_gdml(tmp_path / "t.gdml")


def _refused(tmp_path, df, match, **kw):
    with pytest.raises(exporter.ExportError, match=match):
        exporter.export(df, tmp_path / "x.gdml", scope="full", **kw)


def test_a_border_on_a_repeated_physvol_name_is_refused(tmp_path):
    df = _df(tmp_path)
    pls = tuple(dataclasses.replace(p, name="pmt0x10") if p.name == "pmt0x20" else p for p in df.placements)
    _refused(tmp_path, dataclasses.replace(df, placements=pls), "names 2 placements")


def test_a_repeated_border_pair_is_refused(tmp_path):
    df = _df(tmp_path)
    _refused(tmp_path, dataclasses.replace(df, borders=df.borders * 2), "has a border already")


def test_two_skins_on_one_volume_are_refused(tmp_path):
    df = _df(tmp_path)
    _refused(tmp_path, dataclasses.replace(df, skins=df.skins + (schema.Skin("S2", "Liner", "TankLV"),)), "two skins")


@pytest.mark.parametrize("field,value,match", [("type", "coated", "coated"), ("type", "7", "coated"),
                                               ("model", "foo", "not one Geant4 knows"),
                                               ("finish", "99", "not one Geant4 knows")])  # fmt: skip
def test_a_surface_geant4_cannot_read_is_refused(tmp_path, field, value, match):
    df = _df(tmp_path)
    s = dataclasses.replace(df.surfaces["Liner"], **{field: value})
    _refused(tmp_path, dataclasses.replace(df, surfaces={**df.surfaces, "Liner": s}), match)


@pytest.mark.parametrize("value", [np.inf, np.nan, 1e-310])
def test_non_finite_and_denormal_values_are_refused(tmp_path, value):
    df = _df(tmp_path)
    w = df.materials["Water"]
    p = dataclasses.replace(w.properties["RINDEX"], value=np.array([1.33, value, 1.35]))
    w = dataclasses.replace(w, properties={**w.properties, "RINDEX": p})
    _refused(tmp_path, dataclasses.replace(df, materials={**df.materials, "Water": w}), "non-finite|denormal")


def test_a_lossy_surface_raises_or_is_omitted(tmp_path):
    df = _df(tmp_path)
    s = dataclasses.replace(df.surfaces["Liner"], optics=schema.OpticsModel("acme:brdf"))
    df = dataclasses.replace(df, surfaces={**df.surfaces, "Liner": s})
    _refused(tmp_path, df, "no Geant4 form", on_lossy="error")
    rep = exporter.export(df, tmp_path / "o.gdml", scope="full", on_lossy="omit")
    text = (tmp_path / "o.gdml").read_text()
    assert not rep.lossless and rep.g4ref_args[-1] == "--refuse-lossy"
    assert "<opticalsurface" not in text and "<skinsurface" not in text and "<bordersurface" not in text
    assert exporter.read_export(tmp_path / "o.gdml").detector_file.skins == df.skins


def test_an_unsupported_solid_and_an_unknown_target_are_refused(tmp_path):
    df = _df(tmp_path)
    tw = schema.Solid("Tank", "twistedbox", {"phitwist": 0.1, "x": 1.0, "y": 1.0, "z": 1.0})
    _refused(tmp_path, dataclasses.replace(df, solids={**df.solids, "Tank": tw}), "no GDML spelling")
    _refused(tmp_path, df, "unknown Geant4 target", target=("9.6",))


def test_placements_no_gdml_order_can_hold_are_refused(tmp_path):
    df = _df(tmp_path)  # the world's placement first, before the tank's: no document gives that order
    pls = (df.placements[2], df.placements[0], df.placements[1])
    vols = {**df.volumes, "TankLV": dataclasses.replace(df.volumes["TankLV"], daughters=(1, 2)),
            "WorldLV": dataclasses.replace(df.volumes["WorldLV"], daughters=(0,))}  # fmt: skip
    _refused(tmp_path, dataclasses.replace(df, placements=pls, volumes=vols), "not in an order")


def test_a_name_with_0x_is_warned(tmp_path):
    rep = exporter.export(_df(tmp_path), tmp_path / "x.gdml", scope="full")
    assert any("'0x'" in w for w in rep.warnings)


def test_an_undefined_material_and_a_border_on_an_assembly_are_refused(tmp_path):
    df = _df(tmp_path)
    vols = {**df.volumes, "PmtLV": dataclasses.replace(df.volumes["PmtLV"], material="Unobtainium")}
    _refused(tmp_path, dataclasses.replace(df, volumes=vols), "is not defined")
    vols = {**df.volumes, "TankLV": dataclasses.replace(df.volumes["TankLV"], assembly=True)}
    _refused(tmp_path, dataclasses.replace(df, volumes=vols), "assembly")
