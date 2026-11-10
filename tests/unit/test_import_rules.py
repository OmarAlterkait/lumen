"""The general import rules (lumen.build.import_rules): the film axis from the contact face per placement (D51), its
fail-closed cases, and absorbed at entry."""

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
AXES = {(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)}


def convert(tmp_path, text):
    src = tmp_path / "film.gdml"
    src.write_text(text)
    return lumen.load(lumen.convert(src, RECIPE, tmp_path / "film.lumen"))


def test_the_film_axis_comes_from_the_contact_face_in_each_placements_frame(tmp_path):
    det = convert(tmp_path, film_gdml(HOLDERS))
    s = det.file.surfaces["FilmLV"]
    assert s.optics.name == "lumen:wls_film" and s.optics.origin == "import"
    assert s.optics.settings == dict(film_material="Shifter", front_material="Liq", backing="paint",
                                     thickness_m=2e-8, local_axis=[1.0, 0.0, 0.0])  # fmt: skip
    assert det.file.materials["Shifter"].optics.settings == {"material": "Liq"}
    # the oracle: in the world, the film lies on the foil's side its axis points to (the lowering's world poses)
    insts = {x.pv: x for x in lumen.build.tree.instances(det.parts.loaded)}
    seen = set()
    for pv, x in insts.items():
        if x.lv != "FoilLV":
            continue
        film = next(y for y in insts.values() if y.lv == "FilmLV" and y.mother == x.mother)
        want = (film.t - x.t) / np.linalg.norm(film.t - x.t)
        got = det.tree.film_axes[det.tree.by_name[pv].lv]
        assert np.allclose(got, want, atol=1e-6), (pv, got, want)
        seen.add(tuple(np.round(want, 6) + 0.0))
    assert seen == AXES  # +-x, +-y, +z: a rotation convention error cannot hide


@pytest.mark.parametrize(("kw", "match"), [
    (dict(films="both"), "touches 2 faces|both faces"),
    (dict(film_side=300.0), "covers part of a face"),
    (dict(bare=True), "some placements only"),
])  # fmt: skip
def test_a_film_whose_contact_is_not_one_complete_face_per_placement_is_refused(tmp_path, kw, match):
    with pytest.raises(import_rules.ImportRuleError, match=match):
        convert(tmp_path, film_gdml(HOLDERS[:2], **kw))


def test_a_film_turned_off_its_mothers_axes_is_refused(tmp_path):
    text = film_gdml(HOLDERS[:1]).replace('<physvol name="film0"><volumeref ref="FilmLV"/>',
                                          '<physvol name="film0"><volumeref ref="FilmLV"/>'
                                          '<rotation name="rf" unit="deg" x="0" y="0" z="30"/>')  # fmt: skip
    with pytest.raises(import_rules.ImportRuleError, match="turned off the axes"):
        convert(tmp_path, text)


def _mesh_toy(df, absorption):
    mats = dict(df.materials)
    a = schema.Property("ABSLENGTH", np.array([absorption, absorption]), np.array([1.0, 20.0]))
    mats["Steel"] = dataclasses.replace(mats["Steel"], properties={"ABSLENGTH": a})
    solids = {**df.solids, "b": dataclasses.replace(df.solids["b"], type="tessellated")}  # Blk (Steel) is a mesh
    return dataclasses.replace(df, materials=mats, solids=solids)


@pytest.mark.parametrize(("absorption", "opaque"), [(1e-11, True), (1e-6, False)])
def test_a_mesh_material_absorbing_within_the_band_floor_is_bound_opaque(tmp_path_factory, absorption, opaque):
    df = import_rules.absorbed_at_entry(_mesh_toy(read_toy(tmp_path_factory), absorption))
    o = df.materials["Steel"].optics
    assert (o.name == "lumen:opaque" and o.origin == "import") is opaque
    assert df.materials["Liq"].optics.name == "geant4"  # a box: tracked
