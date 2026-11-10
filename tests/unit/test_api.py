"""Convert, load, parameters and calibrations, on a small toy detector."""

import numpy as np
import pytest
from test_importer import GDML, write

import lumen
from lumen.format import calibrations as cal
from lumen.format.calibrations import Table
from lumen.format.sensorsets import SensorGroup
from lumen.params import UnknownParamKey
from lumen.recipe import Recipe

RECIPE = Recipe(
    name="toy",
    detector_volume="TankLV",
    wavelength_range=(300.0, 600.0),
    groups=(SensorGroup("pmt", "any_entry", "lumen:counting", lv="PmtLV", settings={"efficiency": 1.0}),),
)


def _toy():
    start = GDML.index('  <bordersurface name="B1"')
    end = GDML.index("</bordersurface>") + len("</bordersurface>\n")
    return GDML[:start] + GDML[end:]  # counting sensors take no border surface


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("toy")
    return lumen.convert(write(tmp, _toy()), RECIPE, tmp / "toy.lumen")


@pytest.mark.filterwarnings("error::DeprecationWarning")  # the path runs as under -W error::DeprecationWarning
def test_a_gdml_constant_is_stored_as_one_value_and_handed_on_as_a_scalar(tmp_path):
    """A coldim="1" matrix is a constant: a scalar in memory, a one-element array in the detector file; the parameters
    and the packing read it with no NumPy array-to-scalar conversion (a deprecation that becomes an error)."""
    toy = _toy().replace('  <matrix name="GV"', '  <matrix name="SY" coldim="1" values="100"/>\n  <matrix name="GV"')
    toy = toy.replace(
        '<property name="ABSLENGTH" ref="ABS0x2"/>',
        '<property name="ABSLENGTH" ref="ABS0x2"/> <property name="SCINTILLATIONYIELD" ref="SY"/>',
    )
    path = lumen.convert(write(tmp_path, toy), RECIPE, tmp_path / "c.lumen")
    assert np.load(path / "arrays/materials/Water/SCINTILLATIONYIELD.value.npy").shape == (1,)  # the stored form
    det = lumen.load(path)
    assert det.file.materials["Water"].properties["SCINTILLATIONYIELD"].value.shape == ()
    p = det.params()
    assert p["materials/Water/SCINTILLATIONYIELD"] == 100.0
    det.pack(p.set("materials/Water/SCINTILLATIONYIELD", 50.0))


def test_convert_and_load(converted):
    det = lumen.load(converted)
    assert det.file.detector_volume == "TankLV" and det.file.wavelength_range == (300.0, 600.0)
    assert len(det.sensor_set.ids) == 2 and det.sensors["ids"] == list(det.sensor_set.ids)
    assert "recipe" in det.file.provenance and "toy" in repr(det)
    assert det.parts.optics.n_classes >= 2


def test_params_keys_info_and_precedence(converted):
    det = lumen.load(converted)
    p = det.params()
    assert "materials/Water/ABSLENGTH" in p and "surfaces/Liner/REFLECTIVITY" in p and "sensors/pmt/efficiency" in p
    info = p.info("materials/Water/ABSLENGTH")
    assert info.unit == "m" and info.method == "reweight" and info.available and info.table
    assert not p.info("materials/Water/RINDEX").available  # refractive-index derivatives: pending validation
    assert p.explain("materials/Water/ABSLENGTH") == "detector file"
    assert p.explain("sensors/pmt/efficiency") == "sensor model defaults"
    with pytest.raises(UnknownParamKey, match="Did you mean"):
        p["materials/Water/ABSLENGHT"]
    q = p.scale("materials/Water/ABSLENGTH", 2.0)
    np.testing.assert_array_equal(q["materials/Water/ABSLENGTH"].value, [2.0, 40.0])
    with pytest.raises(ValueError, match="outside"):
        p.set("surfaces/Liner/REFLECTIVITY", Table(np.array([2.0, 4.5]), np.array([1.2, 0.8])))
    with pytest.raises(ValueError, match="knot"):
        p.set("materials/Water/ABSLENGTH", Table(np.array([2.0, 3.0, 4.5]), np.array([1.0, 2.0, 3.0])))


def test_a_calibration_changes_values_never_the_file(converted):
    det = lumen.load(converted)
    path = cal.save(
        cal.Calibration(
            "abs_x2",
            "toy",
            "sha256:-",
            {"materials/Water/ABSLENGTH": Table(np.array([2.0, 4.5]), np.array([2.0, 40.0]))},
        ),
        converted / "calibrations",
    )
    assert path.exists()
    det2 = lumen.load(converted, calibration="abs_x2")
    assert det2.params().explain("materials/Water/ABSLENGTH") == "calibration abs_x2"
    a, b = det.pack(), det2.pack()
    water = [k for k in a["media"] if "Water" in k][0]
    np.testing.assert_allclose(b["media"][water]["absorption"]["value"], 2 * a["media"][water]["absorption"]["value"])
    assert lumen.load(converted).params()["materials/Water/ABSLENGTH"].value.tolist() == [1.0, 20.0]
