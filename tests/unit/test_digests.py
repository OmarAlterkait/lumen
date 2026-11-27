"""The digests that decide reuse: a calibration's detector digest covers the recipe's decisions in the head (detector
volume, wavelength range) and the sensor value sets; the navigator cache's key holds the code it reads."""

import dataclasses

import pytest
from readout_toys import TOY_SET
from test_api import RECIPE, _toy
from test_importer import write

import lumen
from lumen.format import io


def test_the_content_digest_covers_the_detector_volume(tmp_path):
    df = io.read(lumen.convert(write(tmp_path, _toy()), tmp_path / "toy.lumen", recipe=RECIPE))
    d0 = io.content_digest(io.to_files(df))
    assert io.content_digest(io.to_files(dataclasses.replace(df, detector_volume="WorldLV"))) != d0
    assert io.content_digest(io.to_files(dataclasses.replace(df, report=("another line",)))) == d0  # words only


def test_the_content_digest_leaves_out_the_name(tmp_path):
    """One detector converted under two names has one digest, so a calibration transfers."""
    df = io.read(lumen.convert(write(tmp_path, _toy()), tmp_path / "toy.lumen", recipe=RECIPE))
    d0 = io.content_digest(io.to_files(df))
    assert io.content_digest(io.to_files(dataclasses.replace(df, name="another"))) == d0
    (tmp_path / "toy.lumen" / "solids.json").write_text("{}\n")
    with pytest.raises(io.FormatError, match="does not match the manifest"):
        io.read(tmp_path / "toy.lumen")


def test_the_detector_digest_covers_the_sensor_value_sets(tmp_path):
    """A group's value set is copied into the detector file at conversion and is in the detector's digest: nominal
    sensor values cannot change under an unchanged digest (a newer lumen's value set, or an edited copy)."""
    from lumen.format import sensorsets

    groups = tuple(dataclasses.replace(g, values=TOY_SET, settings={}) for g in RECIPE.groups)
    path = lumen.convert(write(tmp_path, _toy()), tmp_path / "toy.lumen",
                         recipe=dataclasses.replace(RECIPE, groups=groups))  # fmt: skip
    det = lumen.load(path)
    modelled = [g for g in det.sensor_set.groups if g.values is not None]
    assert modelled and all(isinstance(g.values, dict) for g in modelled)
    g = modelled[0]
    edited = dict(g.values, gain=float(g.values["gain"]) * 2.0)
    s2 = dataclasses.replace(det.sensor_set, groups=tuple(dataclasses.replace(x, values=edited) if x is g else x
                                                             for x in det.sensor_set.groups))  # fmt: skip
    assert io.content_digest(io.to_files(det.file), sensorsets.files(s2)) != det.digest


def test_the_code_a_key_holds_is_read_from_the_imports():
    """The navigator cache's key holds every module its entry modules import, and nothing of the conversion."""
    from lumen.geometry import cache as gcache

    nav = {str(f.relative_to(gcache.ROOT)) for f in gcache.code_closure(gcache.NAV_CODE)}
    assert "geometry/navigator_build.py" in nav and "gdml/reader.py" not in nav


def test_every_code_digest_entry_names_a_module():
    """Each caller's entries resolve to a closure; a path (no module name) is refused, not dropped."""
    from lumen.geometry import cache as gcache

    for entries in (gcache.NAV_CODE, ("lumen.geometry.clearance", "lumen.transport.fastpath")):
        assert gcache.code_closure(entries)
    with pytest.raises(ValueError, match="names no lumen module"):
        gcache.code_digest(("geometry/clearance.py",))
