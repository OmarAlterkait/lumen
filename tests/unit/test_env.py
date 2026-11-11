"""The environment reads and the atomic cache writes."""

import os
import threading
from pathlib import Path

from lumen import _cache, _env


def test_cache_dir_order(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path / "c"))
    assert _env.cache_dir() == tmp_path / "c"
    monkeypatch.delenv("LUMEN_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
    assert _env.cache_dir() == tmp_path / "x" / "lumen"


def test_detector_path_and_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_DETECTORS", os.pathsep.join([str(tmp_path / "a"), "", str(tmp_path / "b")]))
    assert _env.detector_path() == (tmp_path / "a", tmp_path / "b")
    monkeypatch.delenv("LUMEN_DATA", raising=False)
    assert _env.data_dir() is None


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


def test_the_tallymap_cache_is_never_under_home(monkeypatch, tmp_path):
    import pytest

    monkeypatch.delenv("TALLYMAP_CUDA_CACHE", raising=False)
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path / "c"))
    assert _env.tallymap_cache() == tmp_path / "c" / "tallymap"
    assert os.environ["TALLYMAP_CUDA_CACHE"] == str(tmp_path / "c" / "tallymap")  # noqa: TID251 -- what tallymap reads
    monkeypatch.setenv("TALLYMAP_CUDA_CACHE", str(Path.home() / ".cache" / "tallymap"))
    with pytest.raises(RuntimeError, match="under HOME"):
        _env.tallymap_cache()


def test_the_xla_cache_keeps_every_program(monkeypatch, tmp_path):
    """lumen's programs each compile in under JAX's default 1 s threshold, so the cache takes every program."""
    import jax

    old = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs)
    try:
        monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "xla"))
        monkeypatch.delenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", raising=False)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        assert _env.xla_cache() == tmp_path / "xla"
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
        monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2.5")  # an explicit choice is kept
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.5)
        _env.xla_cache()
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 2.5
    finally:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", old[1])
