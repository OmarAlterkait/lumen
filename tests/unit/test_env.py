"""The environment reads and the atomic cache writes."""

import os
import threading
from pathlib import Path

from lumen import _cache, _env


def test_cache_dir_order(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path / "c"))
    assert _env.cache_dir() == tmp_path / "c"
    monkeypatch.delenv("LUMEN_CACHE")
    (tmp_path / "x").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "x"))
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)  # read TMPDIR again
    root = _env.cache_dir()
    assert root.parent == tmp_path / "x" and root.name.startswith("lumen-")  # one rule: never under HOME by default
    assert Path.home().resolve() not in root.resolve().parents


def test_detector_path_and_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_DETECTORS", os.pathsep.join([str(tmp_path / "a"), "", str(tmp_path / "b")]))
    assert _env.detector_path() == (tmp_path / "a", tmp_path / "b")
    monkeypatch.delenv("LUMEN_DATA", raising=False)
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path / "c"))
    assert _env.data_dir() == tmp_path / "c" / "data"  # unset: under the cache root
    monkeypatch.setenv("LUMEN_DATA", str(tmp_path / "d"))
    assert _env.data_dir() == tmp_path / "d"


def test_atomic_write_leaves_no_partial_file(tmp_path):
    p = tmp_path / "e" / "blob.bin"
    _cache.write_bytes(p, b"x" * 1000)
    assert p.read_bytes() == b"x" * 1000
    assert [q.name for q in p.parent.iterdir()] == ["blob.bin"]


def test_concurrent_directory_builds_keep_one_complete_entry(tmp_path):
    target = tmp_path / "build" / "key"
    errors = []

    def build(i):
        try:
            with _cache.directory(target) as d:
                (Path(d) / "a.npy").write_bytes(b"same content")
                (Path(d) / "b.json").write_text("{}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=build, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sorted(q.name for q in target.iterdir()) == ["a.npy", "b.json"]
    assert [q.name for q in target.parent.iterdir()] == ["key"]


def test_the_tallymap_cache_is_under_the_cache_root(monkeypatch, tmp_path):
    monkeypatch.delenv("TALLYMAP_CACHE", raising=False)
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path / "c"))
    assert _env.tallymap_cache() == tmp_path / "c" / "tallymap"
    assert os.environ["TALLYMAP_CACHE"] == str(tmp_path / "c" / "tallymap")  # noqa: TID251 -- what tallymap reads
    monkeypatch.setenv("TALLYMAP_CACHE", str(tmp_path / "mine"))
    assert _env.tallymap_cache() == tmp_path / "mine"  # a directory named by the user is used as named


def test_importing_lumen_leaves_jax_s_compilation_cache_alone(tmp_path):
    """``import lumen`` sets nothing in JAX's configuration: the persistent cache is JAX's own (its variables)."""
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_")}  # noqa: TID251 -- the child's env
    env["JAX_PLATFORMS"] = "cpu"
    code = ("import jax; c = jax.config; before = (c.jax_compilation_cache_dir, "
            "c.jax_persistent_cache_min_compile_time_secs); import lumen; "
            "print(before == (c.jax_compilation_cache_dir, c.jax_persistent_cache_min_compile_time_secs), "
            "c.jax_compilation_cache_dir)")  # fmt: skip
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["True", "None"], r.stdout


def test_the_version_is_the_source_tree_s_not_another_install_s(monkeypatch):
    """An installed lumen whose metadata describes another tree (an editable install elsewhere) does not name the
    version of the lumen imported: the tree's own version does (git describe, else its archive's describe-name)."""
    import json

    import lumen

    class Other:
        version = "9.9.9"

        def read_text(self, name):
            return json.dumps({"url": "file:///nowhere/else", "dir_info": {"editable": True}})

        def locate_file(self, name):
            return Path("/nowhere/else") / name

    monkeypatch.setattr(lumen, "distribution", lambda name: Other())
    v = lumen._version()
    assert v != "9.9.9" and (v == "0.0.0" or v[0].isdigit()), v
