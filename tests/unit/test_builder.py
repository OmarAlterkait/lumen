"""The tests' Python builder (builder.py): the records an importer writes, checked by the same rules; the toys
(builder_toys) round trip through GDML; the element table; the poses."""

import dataclasses

import builder as lb
import builder_toys as toys
import numpy as np
import pytest

from lumen.build import lowering
from lumen.format import elements, io
from lumen.gdml import exporter, importer
from lumen.recipe import import_structure


def _geometry(df):
    """The files GDML holds as they are: every content file but the materials' and surfaces' (whose origins, notes
    and optics models are lumen's) and the head."""
    return {k: v for k, v in io.to_files(df).items() if k not in ("detector.json", "materials.json", "surfaces.json")}


@pytest.mark.parametrize("make", list(toys.TOYS.values()), ids=list(toys.TOYS))
def test_a_toy_round_trips_through_gdml(tmp_path, make):
    result = import_structure(*make().structure())
    df = result.detector_file
    assert df.provenance["importer"] == lb.IMPORTER
    exporter.export(df, tmp_path / "t.gdml", on_lossy="omit")
    back = importer.read_gdml(tmp_path / "t.gdml", warn=False)
    back = dataclasses.replace(back, detector_volume=df.detector_volume)  # the recipe's choice, not the GDML's
    assert _geometry(back) == _geometry(df)  # the geometry and every array, bit for bit
    for name, m in df.materials.items():
        assert sorted(back.materials[name].properties) == sorted(m.properties), name


def test_the_sphere_s_sensors_face_its_centre():
    df, sets = toys.water_sphere(n_pmt=16).structure()
    result = import_structure(df, sets)
    assert len(result.sensor_set.ids) == 16 and result.sensor_set.ids[0] == "ball_pv/pmt_0/PMT"
    for p in df.placements:
        if p.volume == "PMT":
            m, t = lowering.placement_pose(p)
            np.testing.assert_allclose(m[:, 2], -t / np.linalg.norm(t), atol=1e-12)  # local +z to the centre


def test_a_rotation_is_stored_as_geant4_s_angles_and_read_back():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    b = lb.Builder("rot")
    b.material("M", density=1.0, atoms={"H": 1})
    b.box("B", 1.0, 1.0, 1.0)
    b.box("S", 0.1, 0.1, 0.1)
    b.volume("W", "B", "M")
    b.volume("D", "S", "M")
    b.place("D", "W", "d", (0.1, 0.0, 0.0), q)
    m, t = lowering.placement_pose(b.structure()[0].placements[0])
    np.testing.assert_allclose(m, q, atol=1e-15)


def test_elements_come_from_the_table():
    df, _ = toys.cube_wall().structure()
    assert df.elements["H"].z == 1.0 and df.elements["O"].molar_mass_g_mol == elements.ELEMENTS["O"][1]
    assert set(df.elements) == {"H", "O", "C"}


def test_the_builder_fails_closed():
    b = lb.Builder("bad")
    b.material("M", density=1.0, atoms={"H": 1})
    with pytest.raises(lb.BuildError, match="defined twice"):
        b.material("M", density=1.0, atoms={"H": 1})
    with pytest.raises(lb.BuildError, match="atoms or as fractions"):
        b.material("N", density=1.0)
    b.box("B", 1.0, 1.0, 1.0)
    b.volume("W", "B", "M")
    b.volume("V", "B", "Nothing")
    b.place("V", "W", "v")
    with pytest.raises(lb.BuildError, match="'Nothing' is not a material"):
        b.structure()
    with pytest.raises(lb.BuildError, match="matches no placement"):
        b.border("x", "S", "v", "pmt_*")
    with pytest.raises(lb.BuildError, match="a table is table"):
        b.surface("S", "dielectric_metal", "unified", "ground", 0.0, REFLECTIVITY=[0.1, 0.2])


def test_a_table_on_wavelengths_is_stored_on_ascending_energies():
    p = lb.table_nm([300.0, 400.0, 600.0], [1.0, 2.0, 3.0])
    assert np.all(np.diff(p.energy) > 0) and list(p.value) == [3.0, 2.0, 1.0] and "hc =" in p.note
