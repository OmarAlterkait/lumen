"""Reduce boundary (boundary.cc) outputs to the histograms lumen's facet-path tests compare with.

    python tools/g4ref/reduce_boundary.py OUT_DIR CASE=RUN.npy [CASE=RUN.npy ...]

Each RUN.npy (with its RUN.json sidecar) becomes OUT_DIR/g4_<CASE>.npz: ``hist``, the counts of the last direction
in (dz, |phi|) on ``edges_z`` x ``edges_phi``; ``rings``, the counts in 36 polar rings of 5 degrees about +z;
``launched``. OUT_DIR/MANIFEST.json lists each case's settings, the npz's sha256 and boundary.cc's.
"""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

EDGES_Z = np.linspace(-1.0, 1.0, 41)
EDGES_PHI = np.linspace(0.0, math.pi, 13)
RINGS = np.linspace(0.0, 180.0, 37)
SOURCE = Path(__file__).with_name("boundary.cc")


def geant4_version(name: str) -> str:
    """Return the release of a G4Version string ("$Name: geant4-11-03-patch-01 [MT]$": "11.3.1")."""
    m = re.search(r"geant4-(\d+)-(\d+)(?:-patch-(\d+))?", name)
    if m is None:
        raise ValueError(f"not a Geant4 version string: {name!r}")
    return f"{int(m[1])}.{int(m[2])}.{int(m[3] or 0)}"


def reduce(run: Path) -> tuple[dict, dict]:
    """Return the histograms of one run and its sidecar."""
    rows = np.load(run)
    meta = json.loads(run.with_suffix(".json").read_text())
    d = rows[:, :3]
    hist = np.histogram2d(d[:, 2], np.abs(np.arctan2(d[:, 1], d[:, 0])), bins=[EDGES_Z, EDGES_PHI])[0]
    rings = np.histogram(np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0))), bins=RINGS)[0]
    arrays = dict(hist=hist, edges_z=EDGES_Z, edges_phi=EDGES_PHI, rings=rings, edges_rings=RINGS,
                  launched=np.int64(len(rows)), alive=np.int64(rows[:, 7].sum()))  # fmt: skip
    return arrays, meta


def main(out: str, *cases: str) -> None:
    """Write OUT_DIR/g4_<CASE>.npz for each CASE=RUN.npy and add them to OUT_DIR/MANIFEST.json."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"cases": {}}
    manifest["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for spec in cases:
        name, run = spec.split("=", 1)
        arrays, meta = reduce(Path(run))
        npz = out_dir / f"g4_{name}.npz"
        np.savez_compressed(npz, **arrays)
        o, srf = meta["options"], meta["surface"]
        value = srf["polish"] if o["model"] == "glisur" else srf["sigma_alpha_rad"]  # as G4OpticalSurface holds it
        manifest["cases"][name] = dict(file=npz.name, sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                                       geant4=geant4_version(meta["geant4_version"]), n1=o["n1"], n2=o["n2"],
                                       model=o["model"], finish=o["finish"], type=o["type"], value=value,
                                       theta=o["theta_deg"],
                                       pol=o["pol"], seed=o["seed"], R=o.get("R"), T=o.get("T"),
                                       spike=o.get("spike"), lobe=o.get("lobe"), back=o.get("back"),
                                       launched=int(arrays["launched"]))  # fmt: skip
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
