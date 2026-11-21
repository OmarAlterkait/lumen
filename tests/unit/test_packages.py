"""A detector package's detector by name: ``lumen.convert("package:name")`` reads the package's DETECTORS, its GDML
and inputs under the data directory; ``python -m lumen.compare --package`` reads its REFERENCES."""

import sys
import types

import pytest

from lumen import api
from lumen.compare import cli


@pytest.fixture
def package(monkeypatch, tmp_path):
    mod = types.ModuleType("toy_package")
    mod.DETECTORS = {"toy": ("RECIPE", "gdml/toy.gdml", {"mapping": "gdml/map.json"})}
    mod.REFERENCES = tmp_path / "references.json"
    monkeypatch.setitem(sys.modules, "toy_package", mod)
    return mod


def test_a_package_name_resolves_to_its_gdml_recipe_and_inputs(package, tmp_path):
    assert api._is_package("toy_package:toy") and not api._is_package(str(tmp_path))
    gdml, recipe, inputs = api._package_detector("toy_package:toy", None, None, tmp_path)
    assert (
        gdml == tmp_path / "gdml/toy.gdml" and recipe == "RECIPE" and inputs == {"mapping": tmp_path / "gdml/map.json"}
    )


def test_a_package_name_refuses_a_recipe_an_unknown_name_and_a_stray_data(package, tmp_path):
    with pytest.raises(ValueError, match="names its own recipe"):
        api._package_detector("toy_package:toy", "R", None, tmp_path)
    with pytest.raises(ValueError, match="no detector 'nope'"):
        api._package_detector("toy_package:nope", None, None, tmp_path)
    with pytest.raises(ValueError, match="data directory of a detector package"):
        api.convert(tmp_path / "x.gdml", "R", tmp_path / "x.lumen", data=tmp_path)


def test_compare_reads_a_package_s_manifest(package):
    assert cli.package_references(["toy_package"]) == [str(package.REFERENCES)]


def test_a_package_s_prepare_gives_what_is_converted(package, tmp_path, monkeypatch):
    """A package's ``prepare`` (recipe, GDML, inputs) yields the three convert reads (e.g. a decompressed shipped GDML);
    without one they are converted as resolved."""
    import contextlib

    seen = []

    @contextlib.contextmanager
    def prepare(recipe, gdml, inputs):
        yield f"made from {recipe}", tmp_path / "decompressed.gdml", inputs

    def record(source, recipe, inputs):
        seen.append((source, recipe, inputs))
        raise RuntimeError("stop")

    monkeypatch.setattr(api, "_import", record)
    with pytest.raises(RuntimeError, match="stop"):
        api.convert("toy_package:toy", out=tmp_path / "a.lumen", data=tmp_path)
    package.prepare = prepare
    with pytest.raises(RuntimeError, match="stop"):
        api.convert("toy_package:toy", out=tmp_path / "b.lumen", data=tmp_path)
    inputs = {"mapping": tmp_path / "gdml/map.json"}
    assert seen == [(tmp_path / "gdml/toy.gdml", "RECIPE", inputs),
                    (tmp_path / "decompressed.gdml", "made from RECIPE", inputs)]  # fmt: skip


def test_one_input_pipeline_makes_a_recipe_from_the_gdml_as_converted(tmp_path):
    """lumen.inputs.package, every package's prepare: a shipped .gdml.xz is decompressed and checked, and a recipe made
    from its GDML receives the decompressed file (one package once called it on the compressed path); a file of no
    reference product and not recorded is refused."""
    import hashlib
    import json
    import lzma

    from lumen import inputs

    text = b"<gdml/>\n"
    gdml_dir = tmp_path / "gdml"
    gdml_dir.mkdir()
    (gdml_dir / "d.gdml.xz").write_bytes(lzma.compress(text))
    files = {"d.gdml": {"url": "test", "sha256": hashlib.sha256(text).hexdigest()}}
    (gdml_dir / "SOURCES.json").write_text(json.dumps({"files": files}))
    refs = tmp_path / "pkg" / "references.json"
    refs.parent.mkdir()
    refs.write_text(json.dumps({"products": {}}))
    prepare, expected = inputs.package(refs, gdml_dir)
    seen = []
    with prepare(lambda gdml, given: seen.append(gdml) or "recipe", gdml_dir / "d.gdml.xz", {}) as (r, source, _):
        assert r == "recipe" and source.read_bytes() == text
    assert seen[0].name == "d.gdml"
    with pytest.raises(inputs.InputError, match="pkg records no sha256"):
        expected(tmp_path / "other.json")
