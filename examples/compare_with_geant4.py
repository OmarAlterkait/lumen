"""Check lumen against Geant4 on your own detector with g4ref: the last section of docs/howto/your-own-detector.md.

    python examples/your_own_detector.py                       # first: makes the detector file and its GDML export
    G4REF=/path/to/g4ref python examples/compare_with_geant4.py

g4ref is a small Geant4 10.7.4 application that reads a GDML file, shoots optical photons from point sources and
records which sensor volumes they reach. This script runs it on the GDML lumen exported (so both codes read the same
file), runs lumen on the same source, and compares the fraction of photons each sensor counts. It also runs lumen's
geometry-import check, which asks Geant4's navigator where random points are and compares with lumen's answers.
"""
# ruff: noqa: T201, TID251, D103  (an example script: it prints, reads its output directory from the environment)
# fmt: off

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import lumen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from example_tank import sensor_positions  # noqa: E402

OUT = Path(os.environ.get("LUMEN_EXAMPLES_OUT", "lumen-examples"))
G4REF = os.environ["G4REF"]
HC = 1239.84198  # eV nm
N, LAM, X_MM = 1_000_000, 400.0, (100.0, -50.0, 200.0)

det = lumen.load(OUT / "demo_barrel.lumen")
gdml = OUT / "demo_barrel_export.gdml"

# --- Geant4: N photons at 400 nm from one point; a hit is a photon absorbed on its way into a PMT volume
cfg = OUT / "g4ref_sources.cfg"
cfg.write_text(f"{X_MM[0]} {X_MM[1]} {X_MM[2]} {N} {HC / LAM:.6f}\n")
subprocess.run([G4REF, "--gdml", str(gdml), "--config", str(cfg), "--sens-regex", "^PMT$", "--wcsim-hits",
                "--seed", "1", "--no-hits", "--out", str(OUT / "g4ref")], check=True, stdout=subprocess.DEVNULL)
with open(OUT / "g4ref_sensors.csv") as f:
    sensors = list(csv.DictReader(f))
centres = np.array([[float(r[c]) for c in ("cx_mm", "cy_mm", "cz_mm")] for r in sensors])
row = {int(r["sensor"]): k for k, r in enumerate(sensors)}
hits = np.zeros(len(sensors))
with open(OUT / "g4ref_totals.csv") as f:
    for r in csv.DictReader(f):
        hits[row[int(r["sensor"])]] += float(r["n"])

# --- lumen: the same source; arrivals are the photons the detection rule counts, before quantum efficiency
src = lumen.sources.Point(position=tuple(x / 1000.0 for x in X_MM), lam=LAM, n_photons=float(N))
light = lumen.simulate(det, src, samples=2**21, seed=0)
arr, err = np.asarray(light.arrivals, np.float64), np.asarray(light.arrivals_error, np.float64)

# --- match sensors by centre, compare per sensor and in total
_, idx = cKDTree(sensor_positions(det) * 1000.0).query(centres)
z = (arr[idx] - hits) / np.sqrt(err[idx] ** 2 + hits)
print(" sensor              Geant4   lumen          z")
for r, h, i, zi in zip(sensors, hits, idx, z, strict=True):
    print(f" {r['path'][-18:]:18s} {h:8.0f} {arr[i]:9.1f} +- {err[i]:5.1f} {zi:+6.2f}")
tot_g, tot_l, tot_e = hits.sum(), arr[idx].sum(), np.sqrt((err[idx] ** 2).sum())
ratio = tot_l / tot_g
print(f"total: lumen / Geant4 = {ratio:.4f} +- {ratio * np.hypot(tot_e / tot_l, 1 / np.sqrt(tot_g)):.4f}; "
      f"chi2/ndf = {np.sum(z**2):.1f}/{len(z)}")

# --- the geometry-import check: lumen's point location against Geant4's navigator on the GDML it was converted from
source_gdml = OUT / "demo_barrel_fixed.gdml"
subprocess.run([sys.executable, "-m", "lumen.validate.geometry", str(det.path), "--gdml", str(source_gdml),
                "--g4ref", G4REF, "--out", str(OUT / "geometry_check")], check=True)
print((OUT / "geometry_check" / "geometry.json").read_text()[:1500])
