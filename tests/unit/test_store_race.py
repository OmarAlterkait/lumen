"""Concurrent conversions of one detector into one cache entry (a Slurm array's tasks starting together).

Each writer of a shared store places its entry whole (``lumen._cache``): of N processes converting one detector at
once into a cache entry (``api._convert_once``), one places it, the others discard their copies and use it, and no
reader sees part of an entry. (lumen 0.2 at
fb54cf4 wrote the detector file, then its sensor sets beside it, and a later writer deleted the placed entry before
renaming its own: a task failed on a missing ``RAYLEIGH.value.npy`` while another deleted the entry it read.)
"""

import multiprocessing
import os

import pytest
from test_api import RECIPE, _toy
from test_importer import write

import lumen
from lumen import _cache, api
from lumen.format import calibrations as cal
from lumen.format import io, sensorsets

N = 8
NAME = "race_toy"
GRID = range(-800, 801, 320)  # 6 x 6 x 6 tubes in the 2 m tank, none touching: N conversions overlap


def _many_tubes():
    """The toy with 216 more tubes (its two, and one on each node of a 320 mm grid)."""
    nodes = [(x, y, z) for x in GRID for y in GRID for z in GRID]
    tubes = "".join(
        f'    <physvol name="g{i}"> <volumeref ref="PmtLV"/> <position name="g{i}p" x="{x}" y="{y}" z="{z}"/>\n'
        "    </physvol>\n"
        for i, (x, y, z) in enumerate(nodes)
    )
    toy = _toy()
    at = toy.index('  </volume>\n  <volume name="WorldLV">')
    return toy[:at] + tubes + toy[at:]


def _entry(root):
    """Return (the toy's GDML, the cache entry every task converts it into)."""
    return root / "data" / f"{NAME}.gdml", root / "cache" / "detectors" / f"{NAME}.lumen"


def _load(barrier, out, kernels, root):
    """One task: start with the others, convert the detector into the entry, load and read back every file of it.

    Each task has its own tallymap kernel cache: only lumen's store is shared here.
    """
    os.environ["TALLYMAP_CACHE"] = kernels  # noqa: TID251 -- what tallymap reads
    gdml, entry = _entry(root)
    barrier.wait()
    try:
        det = lumen.load(api._convert_once(gdml, RECIPE, entry))
        io.read(det.path)  # every content file against the manifest
        out.put(("ok", str(det.path), len(det.sensor_set.ids)))
    except BaseException as e:  # the parent reports it
        out.put(("error", f"{type(e).__name__}: {e}", 0))


@pytest.fixture
def store(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write(data, _many_tubes(), f"{NAME}.gdml")
    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path / "cache"))  # empty: every task converts
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    (tmp_path / "cache" / "detectors").mkdir(parents=True)
    return tmp_path / "cache" / "detectors"


def test_n_processes_load_one_detector_into_an_empty_store(store):
    ctx = multiprocessing.get_context("spawn")  # fresh interpreters, as Slurm tasks are
    barrier, out = ctx.Barrier(N), ctx.Queue()
    root = store.parent.parent
    procs = [ctx.Process(target=_load, args=(barrier, out, str(root / f"tm{i}"), root)) for i in range(N)]
    for p in procs:
        p.start()
    got = [out.get(timeout=1200) for _ in procs]  # a loaded 4-core runner
    for p in procs:
        p.join(timeout=60)
    errors = [g[1] for g in got if g[0] != "ok"]
    assert not errors, errors
    assert len({g[1] for g in got}) == 1 and {g[2] for g in got} == {2 + len(GRID) ** 3}  # one entry, read whole
    assert [q.name for q in store.iterdir()] == [os.path.basename(got[0][1])]  # no temporary left behind


def test_an_entry_is_placed_only_once_complete(store, monkeypatch):
    """The partial-read window, deterministically: while the sensor set (written after the detector file) is being
    written, no entry may be visible at the store's name yet."""
    seen = []
    real = sensorsets.save

    def watch(sset, root):
        if store.is_dir():
            seen.extend(p.name for p in store.iterdir() if not p.name.startswith("."))
        return real(sset, root)

    monkeypatch.setattr(sensorsets, "save", watch)
    gdml, entry = _entry(store.parent.parent)
    path = api._convert_once(gdml, RECIPE, entry)
    assert seen == [] and (path / "sensors.ids.json").is_file()


def test_a_losing_writer_keeps_the_winner_s_entry_and_a_replace_swaps_it(tmp_path):
    target = tmp_path / "e"
    with _cache.directory(target) as a:
        (a / "f").write_text("first")
    with _cache.directory(target) as b:  # keep: the entry in place wins
        (b / "f").write_text("second")
    assert (target / "f").read_text() == "first"
    with pytest.raises(FileExistsError), _cache.directory(target, exists="raise") as c:
        (c / "f").write_text("third")
    with _cache.directory(target, exists="replace") as d:
        (d / "f").write_text("fourth")
    assert (target / "f").read_text() == "fourth"
    assert [p.name for p in tmp_path.iterdir()] == ["e"]  # no temporary or old copy left


def test_an_exclusive_file_write_never_replaces(tmp_path):
    p = tmp_path / "h.json"
    _cache.write_bytes(p, b"a", exclusive=True)
    with pytest.raises(FileExistsError):
        _cache.write_bytes(p, b"b", exclusive=True)
    assert p.read_bytes() == b"a" and [q.name for q in tmp_path.iterdir()] == ["h.json"]


def test_a_calibration_of_a_taken_name_is_refused_whole(tmp_path):
    import numpy as np

    c = cal.Calibration("mine", "toy", "sha256:x", {"materials/W/RINDEX": cal.Table(np.ones(2), np.ones(2))})
    cal.save(c, tmp_path)
    (tmp_path / "mine.params.json").unlink()  # a head gone, its arrays left: the arrays' name is still taken
    with pytest.raises(FileExistsError):
        cal.save(c, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mine"]
