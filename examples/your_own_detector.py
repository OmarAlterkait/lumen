"""Your own detector, end to end: the code of docs/howto/your-own-detector.md.

    export LUMEN_DETECTORS=$PWD/lumen-examples/calibrations
    python examples/your_own_detector.py

Converts examples/data/demo_barrel.gdml (written with two deliberate defects) under examples/demo_barrel_recipe.py,
reads the repair warnings, fixes the file, simulates, calibrates (a source-position fit with ``wrt=`` and a
liner-reflectivity scan by value), saves the calibration and exports the detector back to GDML. Outputs go to
./lumen-examples (or $LUMEN_EXAMPLES_OUT).
"""
# ruff: noqa: T201, TID251, D103  (an example script: it prints, reads its output directory from the environment)
# fmt: off

import os
import re
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import lumen
import lumen.gdml
from lumen.format import io

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from demo_barrel_recipe import RECIPE  # noqa: E402

OUT = Path(os.environ.get("LUMEN_EXAMPLES_OUT", "lumen-examples"))
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def section(title):
    print(f"\n=== {title}   ({time.time() - T0:.0f} s)", flush=True)


def convert(gdml, out):
    """Convert ``gdml`` under the recipe; print every warning and the repairs the detector file records."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = lumen.convert(gdml, RECIPE, out, overwrite=True)
    for w in caught:
        print(f"  warning: {w.message}")
    det = lumen.load(path)
    repaired = [line for line in det.file.report if line.startswith("repaired")]
    print(f"  {len(repaired)} repair(s) recorded in {path.name}:")
    for line in repaired:
        print("   ", line)
    return det


# --------------------------------------------------------------------------------------- 1. import, warnings
section("1. Convert the GDML: repairs and their warnings")
det = convert(HERE / "data" / "demo_barrel.gdml", OUT / "demo_barrel_as_given.lumen")
key = "surfaces/LinerSurface/REFLECTIVITY"
p = det.params()
print(f"  {key} = {p[key]} (from: {p.explain(key)}; {p.info(key).default or 'in the file'})")

section("1b. Fix the file where the repair changes the physics, and convert again")
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
for i, name in enumerate(det.sensors["ids"]):
    print(f"  {i:2d} {name:28s} arrivals {float(light.arrivals[i]):8.1f} +- {float(light.arrivals_error[i]):5.1f}"
          f"   PE {float(pe[i]):7.1f}")
print("  health:", {k: round(float(np.sum(v)), 1) for k, v in light.health.items()})

# ------------------------------------------------------------------------------ 3. calibration and fits
section("3a. Fit a calibration source's position (wrt=)")


def expected_pe_at(x):
    s = lumen.sources.Point(position=x, lam=400.0, n_photons=1e5)
    return lumen.readout.pe(lumen.simulate(det, s, samples=2**18, seed=0, wrt=["source/position"]), p0)


x_true = jnp.asarray([0.10, -0.05, 0.20], jnp.float32)
truth = lumen.sources.Point(position=x_true, lam=400.0, n_photons=1e5)
n_obs = jnp.asarray(lumen.readout.sample(lumen.simulate(det, truth, samples=2**20, seed=7), p0, seed=1).pe)
x = jnp.zeros(3, jnp.float32)
for it in range(8):                                 # Gauss-Newton (Fisher scoring) on the Poisson likelihood
    mu, J = expected_pe_at(x), jax.jacfwd(expected_pe_at)(x)    # J (sensors, 3): one derivative per coordinate
    fisher = (J / mu[:, None]).T @ J
    step = jnp.linalg.solve(fisher, J.T @ ((n_obs - mu) / mu))
    x = x + step * jnp.minimum(1.0, 0.1 / jnp.linalg.norm(step))   # at most 10 cm per step: stay inside the tank
    print(f"  iteration {it}: x = {np.round(np.asarray(x), 4).tolist()} m")
err = np.sqrt(np.diag(np.linalg.inv(np.asarray(fisher))))
print(f"  fitted {np.round(np.asarray(x), 4).tolist()} +- {np.round(err, 4).tolist()} m; truth {x_true.tolist()}")

section("3b. Fit the liner's reflectivity by value (no recompilation)")
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
    mu = n_flash * lumen.readout.pe(lumen.simulate(det, src, ps, samples=2**20, seed=0, wrt=[KEY]), ps)
    return jnp.sum(mu - data * jnp.log(mu))


v, slope = jax.jvp(nll, (float(s_hat),), (1.0,))               # the gradient through simulate, at the fit
step = float(slope) / (2 * c2)                                # a Newton step on the parabola's curvature
print(f"  dNLL/d(scale) at the fit: {float(slope):.2f} (curvature {2 * c2:.1f}: a step of {step:+.4f})")

section("3c. Save the calibration and load it")
name = f"liner_{int(time.time())}"
fitted = p0.scale(KEY, float(s_hat))
note = f"liner REFLECTIVITY {0.1 * s_hat:.4f} +- {0.1 * s_err:.4f} from {n_flash} simulated flashes"
print("  wrote", det.save_calibration(fitted, name, note=note))
det_cal = lumen.load(det.path, calibration=name)
print(f"  {KEY} now from '{det_cal.params().explain(KEY)}'")

# ---------------------------------------------------------------------------------------------- 4. export
section("4. Export to GDML")
nominal = OUT / "demo_barrel_export.gdml"
report = lumen.gdml.export(det.path, nominal, scope="full")
print("  wrote", nominal, "and its sidecar; lossless:", report.lossless, "; g4ref arguments:", report.g4ref_args)
back = lumen.gdml.read_export(nominal)
print("  re-imported detector file equals the original:", io.equal(back.detector_file, det.file))
calibrated = OUT / "demo_barrel_calibrated.gdml"
lumen.gdml.export(det.path, calibrated, calibration=name, scope="full")
print("  wrote", calibrated, "(the calibration's values written into the GDML)")

section("done")
