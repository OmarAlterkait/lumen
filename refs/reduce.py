"""Reduce a reference run (g4ref's chunk files, refs/g4ref/run.sh) into a store product; print its manifest entry.

    python refs/reduce.py hits RUN_DIR STORE ID --config SOURCES.cfg --producer PRODUCER.json [--input NAME=FILE ...]
    python refs/reduce.py toys RUN_DIR STORE ID --producer PRODUCER.json --gdml-dir DIR
    python refs/reduce.py files STORE ID NAME=FILE ...

``hits``: every chunk's hits (source, sensor, time, energy, and the WLS flag when the producer records one) on one
sensor table (the chunks' tables joined by placement path: g4ref appends replicated sensors on first hit), the photons
launched per source, the source config. ``toys``: one ``<toy>.npz`` (times, energies) per overlap toy run directory
with its photons launched and the toy's GDML. ``files``: plain copies (datasets such as GDMLs). Each product gets
``MANIFEST.json`` (sha256 per file and the producer); files and directory are made read-only, and an existing product
is never overwritten. The printed entry goes into src/lumen/detectors/references.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import sys
import time
from pathlib import Path

import numpy as np

CHUNK = re.compile(r"^chunk(\d+)_hits\.npy$")


def sha256(path) -> str:
    """Return a file's sha256 as "sha256:<hex>"."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def launched_of(summary: Path, out: dict) -> None:
    """Add a summary file's photons launched per source to ``out`` (both producers' forms)."""
    for ln in summary.read_text().splitlines():
        w = ln.split()
        if len(w) == 2 and w[0].isdigit():  # to_hits.py (WCSim): "s n"
            out[int(w[0])] = out.get(int(w[0]), 0) + int(float(w[1]))
        elif len(w) > 2 and w[0] == "source" and "launched" in w:  # g4ref: "source s x y z N E launched n"
            out[int(w[1])] = out.get(int(w[1]), 0) + int(float(w[w.index("launched") + 1]))


def read_sensors(path: Path):
    """Return {sensor id: (placement path, logical volume, centre mm)} of a sensors.csv."""
    rows = [r.split(",") for r in path.read_text().splitlines()[1:]]
    return {int(r[0]): (r[1], r[2], tuple(float(v) for v in r[3:6])) for r in rows}


def reduce_hits(run: Path, out: Path) -> dict:
    """Write a run's hits, sensor table and photons launched into ``out``; return its counts."""
    files = sorted((p for p in run.iterdir() if CHUNK.match(p.name)), key=lambda p: int(CHUNK.match(p.name).group(1)))
    if not files:
        raise SystemExit(f"{run}: no chunk<k>_hits.npy")
    table, rows, launched, cols = {}, [], {}, None
    for f in files:
        base = str(f)[: -len("_hits.npy")]
        h = np.load(f)
        sens = read_sensors(Path(base + "_sensors.csv"))
        lut = np.full(max(sens) + 1, -1, np.int64)
        for sid, (path, lv, centre) in sens.items():
            lut[sid] = table.setdefault(path, (len(table), lv, centre))[0]
        if np.any(h["sensor"] > max(sens)) or np.any(lut[h["sensor"]] < 0):
            raise SystemExit(f"{f}: a hit on a sensor its table does not list")
        cols = cols or [c for c in ("src", "t", "E", "wls") if c in h.dtype.names]
        rows.append({c: h[c] for c in cols} | {"sensor": lut[h["sensor"]]})
        launched_of(Path(base + "_summary.txt"), launched)
    cat = {c: np.concatenate([r[c] for r in rows]) for c in rows[0]}
    np.savez(out / "hits.npz", src=cat["src"].astype(np.int32), sensor=cat["sensor"].astype(np.int32),
             t=cat["t"].astype(np.float32), E=cat["E"].astype(np.float32),
             **({"wls": cat["wls"].astype(np.int8)} if "wls" in cat else {}))  # fmt: skip
    with open(out / "sensors.csv", "w") as f:
        f.write("sensor,path,lv,cx_mm,cy_mm,cz_mm\n")
        for path, (i, lv, c) in sorted(table.items(), key=lambda kv: kv[1][0]):
            f.write(f"{i},{path},{lv},{c[0]!r},{c[1]!r},{c[2]!r}\n")
    n = max(launched) + 1
    (out / "launched.json").write_text(json.dumps([launched.get(s, 0) for s in range(n)]))
    return dict(chunks=len(files), hits=int(len(cat["t"])), sensors=len(table), sources=n)


def reduce_toys(run: Path, out: Path, gdml_dir: Path) -> dict:
    """Write each overlap toy's hit times and energies, its GDML and the photons launched into ``out``."""
    launched = {}
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        h = np.load(d / "chunk0_hits.npy")
        n = {}
        launched_of(d / "chunk0_summary.txt", n)
        launched[d.name] = int(sum(n.values()))
        np.savez(out / f"{d.name}.npz", t=h["t"].astype(np.float32), E=h["E"].astype(np.float32))
        shutil.copyfile(gdml_dir / f"{d.name}.gdml", out / f"{d.name}.gdml")
    (out / "launched.json").write_text(json.dumps(launched, indent=1, sort_keys=True))
    return dict(toys=len(launched))


def publish(out: Path, entry: dict) -> dict:
    """Write the product's MANIFEST.json (every file's sha256) and make it read-only; return the entry."""
    files = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != "MANIFEST.json"}
    entry = dict(entry, files=files, reduced=time.strftime("%Y-%m-%d"))
    (out / "MANIFEST.json").write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
    for p in out.iterdir():
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    out.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return entry


def main(argv) -> int:
    """Reduce one run (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("hits", "toys", "files"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--config")
    ap.add_argument("--producer", help="JSON: who made the run (tool, commit, image, flags)")
    ap.add_argument("--gdml-dir")
    ap.add_argument("--input", action="append", default=[], help="NAME=FILE: an input copied into the product")
    a = ap.parse_args(argv)
    if a.mode == "files":
        store, pid, *items = a.args
        run = None
    else:
        run, store, pid = Path(a.args[0]), a.args[1], a.args[2]
        items = a.input
    out = Path(store) / pid
    if out.exists():
        raise SystemExit(f"{out} exists: a store product is never overwritten (give a new id)")
    out.mkdir(parents=True)
    entry = dict(id=pid, producer=json.loads(Path(a.producer).read_text()) if a.producer else None)
    if a.mode == "hits":
        entry["run"] = str(run)
        entry |= reduce_hits(run, out)
        shutil.copyfile(a.config, out / "sources.cfg")
    elif a.mode == "toys":
        entry["run"] = str(run)
        entry |= reduce_toys(run, out, Path(a.gdml_dir))
    inputs = {}
    for it in items:
        name, src = it.split("=", 1)
        shutil.copyfile(src, out / name)
        inputs[name] = src
    entry["inputs"] = inputs
    entry = publish(out, entry)
    sys.stdout.write(json.dumps(entry, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
