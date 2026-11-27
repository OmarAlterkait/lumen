"""Your own detector, end to end: the code of docs/howto/your-own-detector.md.

    export LUMEN_DETECTORS=$PWD/lumen-examples/calibrations
    python examples/your_own_detector.py

Converts examples/data/demo_barrel.gdml (written with two deliberate defects) under examples/demo_barrel_recipe.py:
the refusal, the repaired copy, the fixed file; then simulates, calibrates (a liner-reflectivity scan by value and its
derivative with ``jax.jvp``), saves the calibration and exports the detector back to GDML. Outputs go to
./lumen-examples (or $LUMEN_EXAMPLES_OUT).
"""
# ruff: noqa: T201, TID251, D103  (an example script: it prints, reads its output directory from the environment)
# fmt: off

import os
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import lumen
import lumen.gdml
from lumen.gdml import importer, reader, repair

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from demo_barrel_recipe import RECIPE  # noqa: E402

OUT = Path(os.environ.get("LUMEN_EXAMPLES_OUT", "lumen-examples"))
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def section(title):
    print(f"\n=== {title}   ({time.time() - T0:.0f} s)", flush=True)


def convert(gdml, out):
    """Convert ``gdml`` under the recipe; print what Geant4 would not read and the import left out."""
    path = lumen.convert(gdml, out, recipe=RECIPE, overwrite=True)
    det = lumen.load(path)
    for line in det.file.report:
        if "left out" in line:
            print("   ", line)
    return det


# --------------------------------------------------------------------------------------- 1. import, repair
section("1. Convert the GDML: Geant4 refuses it, and so does lumen; repair it apart")
try:
    convert(HERE / "data" / "demo_barrel.gdml", OUT / "demo_barrel_as_given.lumen")
except reader.GDMLError as err:
    print(" ", err)
text, edits = repair.repair((HERE / "data" / "demo_barrel.gdml").read_text())
print("  repairs:", edits)
(OUT / "demo_barrel_repaired.gdml").write_text(text)
det = convert(OUT / "demo_barrel_repaired.gdml", OUT / "demo_barrel_as_given.lumen")
key = "surfaces/LinerSurface/REFLECTIVITY"
p = det.params()
print(f"  {key} = {p[key]} (from: {p.explain(key)}; {p.info(key).note or 'in the file'})")

section("1b. Fix the file where Geant4's reading is not what was meant, and convert again")
text = (HERE / "data" / "demo_barrel.gdml").read_text()
text = text.replace('<matrix name="LINER_REFLECTIVITY" coldim="1" values="0.1"/>',
                    '<matrix name="LINER_REFLECTIVITY" coldim="2" values="1.5*eV 0.1 5.0*eV 0.1"/>')
text = re.sub(r'(5\.0\*eV 2\.0\*m) 7\.0\*eV"', r'\1"', text)       # and drop the unpaired tail ourselves
fixed = OUT / "demo_barrel_fixed.gdml"
fixed.write_text(text)
det = convert(fixed, OUT / "demo_barrel.lumen")
p0 = det.params()
print(f"  {key} = {p0[key]} (from: {p0.explain(key)})")
print(det)

# ------------------------------------------------------------------------------------------ 2. simulate
section("2. Forward simulation")
src = lumen.sources.Point(position=(0.0, 0.0, 0.0), lam=400.0, n_photons=1e5)
light = lumen.simulate(det, src, samples=2**18, seed=0)
pe = lumen.readout.pe(light, p0)
for i, name in enumerate(det.sensor_table().name):
    print(f"  {i:2d} {name:28s} arrivals {float(light.arrivals[i]):8.1f} +- {float(light.arrivals_error[i]):5.1f}"
          f"   PE {float(pe[i]):7.1f}")
print("  health:", {k: round(float(np.sum(v)), 1) for k, v in light.health._asdict().items()})

# ------------------------------------------------------------------------------ 3. calibration and fits
section("3a. Fit the liner's reflectivity by value (no recompilation)")
KEY = "surfaces/LinerSurface/REFLECTIVITY"
info = p0.info(KEY)
print(f"  {KEY}: unit {info.unit or '-'}, range {info.range}, gradient method {info.method}")
truth_p = p0.scale(KEY, 1.5)                                  # pseudo-data: the sheet reflects 15 %, not 10 %
n_flash = 50                                                  # 50 flashes of the central source, summed per PMT
light_t = lumen.simulate(det, src, truth_p, samples=2**21, seed=11)
data = sum(jnp.asarray(lumen.readout.sample(light_t, truth_p, seed=200 + k).pe) for k in range(n_flash))
scales, nlls = np.linspace(0.5, 2.5, 9), []
for s in scales:
    t = time.time()
    ps = p0.scale(KEY, s)                                     # every knot of the table times s
    mu = n_flash * lumen.readout.pe(lumen.simulate(det, src, ps, samples=2**20, seed=0), ps)
    nlls.append(float(jnp.sum(mu - data * jnp.log(mu))))
    print(f"  REFLECTIVITY {0.1 * s:.3f}: NLL {nlls[-1]:.2f}   ({time.time() - t:.2f} s)")
k = int(np.clip(np.argmin(nlls), 2, len(nlls) - 3))           # a parabola through the 5 points about the minimum
c2, c1, _ = np.polyfit(scales[k - 2:k + 3], nlls[k - 2:k + 3], 2)
s_hat, s_err = -c1 / (2 * c2), 1 / np.sqrt(2 * c2)
print(f"  fitted REFLECTIVITY {0.1 * s_hat:.4f} +- {0.1 * s_err:.4f} (truth 0.150)")


def nll(s):
    ps = p0.scale(KEY, s)
    mu = n_flash * lumen.readout.pe(lumen.simulate(det, src, ps, samples=2**20, seed=0), ps)
    return jnp.sum(mu - data * jnp.log(mu))


v, slope = jax.jvp(nll, (float(s_hat),), (1.0,))               # the gradient through simulate, at the fit
step = float(slope) / (2 * c2)                                # a Newton step on the parabola's curvature
print(f"  dNLL/d(scale) at the fit: {float(slope):.2f} (curvature {2 * c2:.1f}: a step of {step:+.4f})")

section("3b. Save the calibration and load it")
name = f"liner_{int(time.time())}"
fitted = p0.scale(KEY, float(s_hat))
note = f"liner REFLECTIVITY {0.1 * s_hat:.4f} +- {0.1 * s_err:.4f} from {n_flash} simulated flashes"
print("  wrote", det.save_calibration(fitted, name, note=note))
det_cal = lumen.load(det.path, calibration=name)
print(f"  {KEY} now from '{det_cal.params().explain(KEY)}'")

# ---------------------------------------------------------------------------------------------- 4. export
section("4. Export to GDML")
nominal = OUT / "demo_barrel_export.gdml"
report = lumen.gdml.export(det.path, nominal)
print("  wrote", nominal, "; lossless:", report.lossless)
back = importer.read_gdml(nominal, warn=False)
print("  read back with the same geometry:", back.solids == det.file.solids and back.placements == det.file.placements)
calibrated = OUT / "demo_barrel_calibrated.gdml"
lumen.gdml.export(det.path, calibrated, calibration=name)
print("  wrote", calibrated, "(the calibration's values written into the GDML)")

section("done")
