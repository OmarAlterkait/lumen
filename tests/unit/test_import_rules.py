"""The general import rules (lumen.build.import_rules): a WLS box thinner than the band refused, and absorbed at
entry."""

import dataclasses

import numpy as np
import pytest
from film_toy import RECIPE, film_gdml
from optics_toy import read_toy

import lumen
from lumen.build import import_rules
from lumen.format import schema

# holder poses (mm, GDML deg): unrotated, 180 deg about y (SBND's East foils), -+90 about z, 90 about y, and 90
# about x then z (a composite: the order of the angles matters)
HOLDERS = (((-500, 0, -500), None), ((500, 0, -500), (0, 180, 0)), ((-500, 0, 500), (0, 0, 90)),
           ((500, 0, 500), (0, 0, -90)), ((0, 0, 0), (0, 90, 0)), ((0, 450, 0), (90, 0, 90)))  # fmt: skip


def convert(tmp_path, text):
    src = tmp_path / "film.gdml"
    src.write_text(text)
    return lumen.load(lumen.convert(src, tmp_path / "film.lumen", recipe=RECIPE))


def test_a_wls_box_thinner_than_the_band_is_refused(tmp_path):
    """A 20 nm WLS box (thinner than the band at its coordinates) is refused: lumen lowers no film; the 25.4 um slab
    (SBND's split TPB) is tracked as a volume, its material Geant4's (bulk WLS), the foil keeping its skin."""
    with pytest.raises(import_rules.ImportRuleError, match="thinner than the band"):
        convert(tmp_path, film_gdml(HOLDERS[:2], film=2e-5))
    det = convert(tmp_path, film_gdml(HOLDERS))
    assert det.file.materials["Shifter"].optics.name == "geant4"
    assert [k.surface for k in det.file.skins if k.volume == "FoilLV"] == ["paint"]
    assert [c.name for c in det.parts.library.media["Shifter"].medium.channels] == ["wls"]


def test_the_import_rules_apply_once(tmp_path):
    """a converted detector read back (from YAML) meets the rules again and is unchanged"""
    from lumen.format import io

    df = convert(tmp_path, film_gdml(HOLDERS)).file
    again = import_rules.apply(df)
    assert io.content_digest(io.to_files(again)) == io.content_digest(io.to_files(df))


def _mesh_toy(df, absorption):
    mats = dict(df.materials)
    a = schema.Property("ABSLENGTH", np.array([absorption, absorption]), energy=np.array([1.0, 20.0]))
    mats["Steel"] = dataclasses.replace(mats["Steel"], properties={"ABSLENGTH": a})
    tris = np.array(
        [
            [[0, 0, 0], [0.01, 0, 0], [0, 0.02, 0]],
            [[0, 0, 0], [0, 0.02, 0], [0, 0, 0.03]],
            [[0, 0, 0], [0, 0, 0.03], [0.01, 0, 0]],
            [[0.01, 0, 0], [0, 0, 0.03], [0, 0.02, 0]],
        ],
        float,
    )
    solids = {**df.solids, "b": dataclasses.replace(df.solids["b"], type="tessellated", mesh="b")}  # Blk is a mesh
    return dataclasses.replace(df, materials=mats, solids=solids, meshes={**df.meshes, "b": tris})


@pytest.mark.parametrize(("absorption", "opaque"), [(1e-11, True), (1e-6, False)])
def test_a_mesh_material_absorbing_within_the_band_where_it_is_is_bound_opaque(tmp_path_factory, absorption, opaque):
    df = import_rules.absorbed_at_entry(_mesh_toy(read_toy(tmp_path_factory), absorption))
    o = df.materials["Steel"].optics
    assert (o.name == "lumen:opaque" and o.origin == "import") is opaque
    assert df.materials["Liq"].optics.name == "geant4"  # a box: tracked


@pytest.mark.parametrize(("absorption", "opaque"), [(1e-11, True), (1e-6, False)])
def test_an_opaque_material_offers_no_light_to_sources(tmp_path_factory, absorption, opaque):
    """A mesh holds no point: light born in an opaque mesh started outside it and reflected off its face (ANNIE's
    steel frame). Its RINDEX is no parameter, so a Cherenkov source in it is refused and an event's steps in it are
    left out."""
    from lumen.build import spec

    toy = _mesh_toy(read_toy(tmp_path_factory), absorption)
    n = schema.Property("RINDEX", np.array([1.33, 1.34]), energy=np.array([1.0, 20.0]))
    steel = dataclasses.replace(toy.materials["Steel"], properties={**toy.materials["Steel"].properties, "RINDEX": n})
    df = import_rules.absorbed_at_entry(dataclasses.replace(toy, materials={**toy.materials, "Steel": steel}))
    assert ("materials/Steel/RINDEX" not in spec.detector_values(df, {})[0]) is opaque


def test_a_write_to_a_parameters_table_raises(tmp_path_factory):
    from lumen.build import spec

    df = read_toy(tmp_path_factory)
    t = next(v for v in spec.detector_values(df, {})[0].values() if hasattr(v, "energy"))
    with pytest.raises(ValueError, match="read-only"):
        t.value[:] = 1.0
