"""A detector in YAML (lumen.format.yaml, input only): the hand-written toy, input written by hand, and the
errors a bad file gets."""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

import lumen
from lumen import units
from lumen.build import lowering
from lumen.format import yaml as fy
from lumen.format.yaml import YAMLError

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TOY = EXAMPLES / "data" / "yaml_toy"


@pytest.fixture(scope="module")
def tank(tmp_path_factory):
    """The tutorial tank (examples/data/example_tank.gdml) converted under its recipe (test_pack reads it too)."""
    import sys

    sys.path.insert(0, str(EXAMPLES))
    import example_tank

    tmp = tmp_path_factory.mktemp("tank")
    return lumen.convert(example_tank.GDML, tmp / "example_tank.lumen", recipe=example_tank.RECIPE)


# ----------------------------------------------------------------------------------------------------- units
def test_a_quantity_written_as_text_is_read_by_exact_decimal():
    assert units.quantity("12.5*mm", "length") == 0.0125  # one rounding, not 12.5 * 0.001
    assert units.quantity("400*nm", "length", "nm") == 400.0 and units.quantity("0.4 um", "length", "nm") == 400.0
    assert units.quantity("2.5*eV", "energy") == 2.5 and units.quantity("1e-5", "length") == 1e-05
    assert units.quantity("90*deg", "angle") == 90.0 * math.pi / 180.0
    assert units.quantity(3, "time") == 3.0
    with pytest.raises(ValueError, match="'eV' is not a unit of length"):
        units.quantity("3*eV", "length")
    for what in units.QUANTITIES:  # every quantity names an example unit of its own
        with pytest.raises(ValueError, match=f"not a number or a quantity of {what} such as"):
            units.quantity("three metres", what)


# ----------------------------------------------------------------------------------------- the hand toy
def test_the_hand_written_toy_converts_and_loads(tmp_path):
    path = lumen.convert(TOY, tmp_path / "yaml_toy.lumen")
    det = lumen.load(path)
    assert det.file.world == "World" and det.file.detector_volume == "Tank"
    assert det.file.volumes["Tank"].daughters == (1, 2, 3, 4, 5, 6)  # the placements, in their order
    assert det.sensor_set.ids == tuple(f"TankPV/PMT_{s}/PMT" for s in ("top", "bottom", "east", "west", "north",
                                                                       "south"))  # fmt: skip
    rindex = det.file.materials["Water"].properties["RINDEX"]
    np.testing.assert_array_equal(rindex.energy, units.HC / np.array([650.0, 550.0, 450.0, 350.0, 250.0]))
    np.testing.assert_array_equal(rindex.value, [1.331, 1.335, 1.338, 1.349, 1.366])  # reversed with the energies
    assert f"hc = {units.HC} eV nm" in rindex.note
    assert det.file.materials["Water"].properties["ABSLENGTH"].value.tolist() == [5.0, 40.0, 60.0, 30.0, 2.0]
    axes = det.sensor_table().axis
    np.testing.assert_allclose(axes[5], [0.0, -1.0, 0.0], atol=1e-15)  # the matrix: PMT_south's axis onto -y
    assert "sensors/pmt/qe" in det.params()


def test_a_rotation_matrix_is_stored_as_geant4_s_angles():
    rng = np.random.default_rng(3)
    for x, y, z in [*rng.uniform(-3.0, 3.0, (20, 3)), (0.3, math.pi / 2, 0.0), (0.0, -math.pi / 2, 1.0)]:
        m, _ = lowering.placement_pose(
            lumen.format.schema.Placement("p", "V", "W", copy=None, position=(0, 0, 0), rotation=(x, y, z))
        )
        back, _ = lowering.placement_pose(
            lumen.format.schema.Placement(
                "p", "V", "W", copy=None, position=(0, 0, 0), rotation=fy.angles(np.asarray(m).T)
            )
        )
        np.testing.assert_allclose(back, m, atol=1e-12)
    assert fy.angles(np.eye(3)) == (0.0, 0.0, 0.0) and all(math.copysign(1, a) > 0 for a in fy.angles(np.eye(3)))


# ---------------------------------------------------------------------------------------------- bad input
def _toy_copy(tmp_path, edit=None, extra=None):
    d = tmp_path / "toy"
    shutil.copytree(TOY, d)
    if edit:
        f, old, new = edit
        text = (d / f).read_text()
        assert old in text, old
        (d / f).write_text(text.replace(old, new, 1))
    if extra:  # a file of its own starts with its schema line (a list is refused before it is looked at)
        (d / extra[0]).write_text(("" if extra[1].startswith("-") else f"schema: {fy.SCHEMA}\n") + extra[1])
    return d


@pytest.mark.parametrize(
    "edit,extra,match",
    [
        (
            ("geometry.yaml", "params: {rmax: 40*mm", "params: {rmaxx: 40*mm"),
            None,
            r"unknown field 'rmaxx' \(did you " r"mean 'rmax'\?\)",
        ),  # fmt: skip
        (
            ("geometry.yaml", "rmax: 500*mm", "rmax: 500*eV"),
            None,
            r"solids.TankTube.params.rmax: .*'eV' is not a unit " "of length",
        ),  # fmt: skip
        (("geometry.yaml", "detector_volume: Tank", "wavelength_range_nm: [300, 600]"), None, "wavelength_range_nm"),
        (
            ("geometry.yaml", "type: box", "type: boxx"),
            None,
            r"'boxx' is not a solid lumen reads \(did you mean " "'box'",
        ),  # fmt: skip
        (
            None,
            ("more.yaml", "materials:\n  Water: {composition: {density_g_cm3: 1.0}}\n"),
            "materials 'Water' is defined in materials.yaml and in more.yaml",
        ),
        (None, ("more.yaml", "detector_volume: World\n"), "detector_volume is 'Tank' in geometry.yaml and 'World'"),
        (
            ("geometry.yaml", "[[1, 0, 0], [0, 0, -1], [0, 1, 0]]", "[[1, 0, 0], [0, 0, -1], [0, 1, 1]]"),
            None,
            "orthonormal with determinant",
        ),
        (
            ("materials.yaml", "        energy: [1.5, 2.5, 3.0, 3.5, 5.0]\n", ""),
            None,
            "Water.properties.ABSLENGTH: a table gives its energies",
        ),
        (
            ("materials.yaml", "energy: [1.5, 2.5, 3.0, 3.5, 5.0]", "energy: [1.5, 2.5, 3.0, 3.5]"),
            None,
            "4 energies but 5 values",
        ),
        (
            ("materials.yaml", "energy: [1.5, 2.5, 3.0, 3.5, 5.0]", "energy: [1.5, 3.0, 2.5, 3.5, 5.0]"),
            None,
            "'Water'.ABSLENGTH: energies must be strictly ascending",
        ),
        (
            ("materials.yaml", "{ref: Hydrogen, atoms: 2}", "{ref: Hydrogen, atoms: 2, fraction: 0.1}"),
            None,
            "a mass fraction or a number of atoms",
        ),
        (
            ("materials.yaml", "{ref: Oxygen, atoms: 1}", "{ref: Oxygn, atoms: 1}"),
            None,
            "component 'Oxygn' is neither an element nor a material",
        ),
        (("surfaces.yaml", "pv2: PMT_top}", "pv2: PMT_tpo}"), None, "physvol 'PMT_tpo' names 0 physical volumes"),
        (("surfaces.yaml", "    finish: ground\n", ""), None, "surfaces.Liner: missing finish"),
        (("sensors.yaml", "rule: border_into", "rule: border"), None, "rule 'border' is not one of"),
        (("geometry.yaml", "# world: left out", "world: Nowhere #"), None, "the world 'Nowhere' is not a volume"),
        (
            None,
            ("more.yaml", "volumes:\n  Spare: {solid: PMTDisc, material: Glass}\n"),
            "world is missing and 2 volumes are placed by no placement",
        ),
        (None, ("more.yaml", "- just a list\n"), "a detector file is a mapping of sections"),
        (None, ("more.yaml", "solids: {Bad: {type: box, params: {x: [1}}\n"), "not valid YAML"),
    ],
)
def test_bad_hand_input_is_refused_with_a_clear_error(tmp_path, edit, extra, match):
    with pytest.raises(YAMLError, match=match):
        fy.read(_toy_copy(tmp_path, edit, extra))


def test_a_file_without_its_schema_line_is_refused(tmp_path):
    d = _toy_copy(tmp_path)
    (d / "more.yaml").write_text("materials: {}\n")
    with pytest.raises(YAMLError, match="more.yaml: no schema line"):
        fy.read(d)


def test_a_recipe_is_refused(tmp_path):
    d = _toy_copy(tmp_path)
    with pytest.raises(ValueError, match="only a GDML file takes a recipe"):
        lumen.convert(d, tmp_path / "x.lumen", recipe=object())


def test_an_omitted_angular_range_is_the_solid_geant4_builds_from_either_input():
    """A tube whose startphi is left out: GDML stores the record without it (Geant4 reads its 0), YAML reads the
    omission as the whole solid and stores startphi 0 explicitly. Two records of one solid: the lowering reads the same
    values. (The YAML writer's round trip of SBND and MicroBooNE, whose GDML omits startphi on 700 and 1681 solids,
    failed on this record difference; it went with the writer.)"""
    from lumen.format import schema
    from lumen.format import solids as catalogue

    gdml = schema.Solid("t", "tube", params={"rmin": 0.1, "rmax": 0.2, "z": 0.4, "deltaphi": 2 * math.pi})
    yml = schema.Solid("t", "tube", params=catalogue.whole("tube", {"rmin": 0.1, "rmax": 0.2, "z": 0.4}))
    assert "startphi" not in gdml.params and yml.params["startphi"] == 0.0
    keys = ("rmin", "rmax", "z", "startphi", "deltaphi")  # what the lowering builds the tube from
    assert lowering._params(gdml, *keys) == lowering._params(yml, *keys) == [0.1, 0.2, 0.4, 0.0, 2 * math.pi]
