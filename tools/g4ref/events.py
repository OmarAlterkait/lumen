"""Geant4 events for lumen's multi-event comparison (``lumen.compare.event.compare_events``): g4ref's particle gun,
one event per run, at vertices and directions drawn uniformly in a fiducial region, each written as lumen's event file.

    python tools/g4ref/events.py table OUT --name sk --particle mu- --mev 600 --n 100 --seed 1 \
        --region cylinder:z,14.9,-16.1,16.1          # or box:x0,x1,y0,y1,z0,z1 (m)
    python tools/g4ref/events.py run OUT --g4ref BIN --gdml GDML --sens-regex RE --task K --tasks N [g4ref flags]

``table`` writes OUT/events.json: one row per event (its name, particle, energy, vertex m, unit direction, seed),
reproducible from ``--seed``. ``run`` runs rows K, K + N, ... (a Slurm array task K of N): g4ref ``--gun`` with
``--events 1`` and the row's seed, then ``lumen.compare.event.extract`` to OUT/<name>.npz with the row in its meta;
a row whose file exists is skipped. The run's own flags (``--cherenkov --scintillation --split-ev 6.19``, ...) follow
after the options; they are recorded in each event's meta.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _region(spec: str, n: int, rng) -> np.ndarray:
    kind, _, args = spec.partition(":")
    v = args.split(",")
    if kind == "box":
        lo, hi = np.array(v[0::2], float), np.array(v[1::2], float)
        return lo + (hi - lo) * rng.uniform(size=(n, 3))
    if kind == "cylinder":  # axis, radius, low, high (m)
        k, (r, a, b) = "xyz".index(v[0]), map(float, v[1:])
        rr, f = r * np.sqrt(rng.uniform(size=n)), rng.uniform(0, 2 * np.pi, n)
        out = np.empty((n, 3))
        out[:, k] = rng.uniform(a, b, n)
        out[:, (k + 1) % 3], out[:, (k + 2) % 3] = rr * np.cos(f), rr * np.sin(f)
        return out
    raise SystemExit(f"region {spec!r}: box:x0,x1,y0,y1,z0,z1 or cylinder:axis,radius,low,high (m)")


def table(a):
    rng = np.random.default_rng(a.seed)
    pos = _region(a.region, a.n, rng)
    d = rng.normal(size=(a.n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rows = [dict(name=f"{a.name}_{a.particle.strip('+-')}{a.mev:g}_{i:03d}", particle=a.particle, mev=a.mev,
                 vertex_m=[round(float(x), 6) for x in p], direction=[round(float(x), 9) for x in u],
                 seed=int(a.seed) * 100000 + i) for i, (p, u) in enumerate(zip(pos, d, strict=True))]  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "events.json"
    old = json.loads(f.read_text()) if f.exists() else {"rows": []}
    names = {r["name"] for r in old["rows"]}
    old["rows"] += [r for r in rows if r["name"] not in names]
    old.setdefault("regions", {})[f"{a.name} {a.particle} {a.mev:g}"] = a.region
    f.write_text(json.dumps(old, indent=1))
    print(f"{f}: {len(old['rows'])} rows")


def run(a, flags):
    from lumen.compare import event

    out = Path(a.out)
    rows = json.loads((out / "events.json").read_text())["rows"][a.task :: a.tasks]
    for r in rows:
        dst = out / f"{r['name']}.npz"
        if dst.exists():
            continue
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            mm = ",".join(f"{1e3 * x:.3f}" for x in r["vertex_m"])
            cmd = [a.g4ref, "--gdml", a.gdml, "--sens-regex", a.sens_regex, "--gun", r["particle"],
                   "--gun-mev", str(r["mev"]), "--gun-pos", mm, "--gun-dir", ",".join(map(str, r["direction"])),
                   "--events", "1", "--seed", str(r["seed"]), "--out", f"{tmp}/chunk0", *flags]  # fmt: skip
            log = Path(tmp) / "g4ref.log"
            with log.open("w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                sys.stderr.write(log.read_text()[-2000:])
                raise SystemExit(f"{r['name']}: g4ref exited {rc}")
            meta = dict(case=r["name"], particle=r["particle"], kinetic_energy_mev=r["mev"],
                        position_mm=mm, direction=",".join(map(str, r["direction"])), flags=flags)  # fmt: skip
            event.extract(tmp, dst, meta=meta)
        print(r["name"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("out")
    t.add_argument("--name", required=True)
    t.add_argument("--particle", required=True)
    t.add_argument("--mev", type=float, required=True)
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--region", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    for k in ("--g4ref", "--gdml", "--sens-regex"):
        r.add_argument(k, required=True)
    r.add_argument("--task", type=int, default=0)
    r.add_argument("--tasks", type=int, default=1)
    a, rest = ap.parse_known_args()
    table(a) if a.cmd == "table" else run(a, rest)


if __name__ == "__main__":
    main()
