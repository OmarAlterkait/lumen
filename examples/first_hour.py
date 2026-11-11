"""The first hour with lumen, as one script: the code of docs/tutorial/first-hour.md, section by section.

    export LUMEN_DETECTORS=$PWD/lumen-examples/calibrations     # where your calibrations go (any directory)
    python examples/first_hour.py

Writes the converted detector, the calibration and two figures (if matplotlib is installed) into ./lumen-examples
(or $LUMEN_EXAMPLES_OUT). Run it on a GPU node; on a CPU it works too, but compiles slowly.
"""
# ruff: noqa: T201, TID251, D103  (an example script: it prints, reads its output directory from the environment)
# fmt: off

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import lumen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from example_tank import OUT, detector, sensor_positions

T0 = time.time()


def section(title):
    print(f"\n=== {title}   ({time.time() - T0:.0f} s)", flush=True)


# ---------------------------------------------------------------------------------------------------- 1. the toy
section("1. The shipped toy and its sensors")
print(f"lumen {lumen.__version__}, jax {jax.__version__}, backend {jax.default_backend()}")
det = detector()
print(det)
for i, path in enumerate(det.sensors["ids"]):
    print(f"  sensor {i}: {path}")
p0 = det.params()
print(f"{len(p0)} parameters, for example:")
for key in ("materials/Water/ABSLENGTH", "materials/Water/RINDEX", "sensors/pmt/efficiency", "sensors/pmt/tts/sigma"):
    info = p0.info(key)
    print(f"  {key:30s} unit {info.unit or '-':4s} range {info.range}  method {info.method}  "
          f"from: {p0.explain(key)}")

# ------------------------------------------------------------------------------------------------- 2. simulate
section("2. Simulate, and what the numbers mean")
src = lumen.sources.Point(position=(0.0, 0.0, 0.1), lam=400.0, n_photons=2e5, time=0.0)
light = lumen.simulate(det, src, samples=2**18, seed=0)
pe = lumen.readout.pe(light, p0)
print(" id  arrivals   error    mean time (ns)   expected PE")
for i in range(len(pe)):
    print(f"{i:3d} {float(light.arrivals[i]):9.2f} {float(light.arrivals_error[i]):7.2f} "
          f"{float(light.mean_arrival_time[i]):12.3f} {float(pe[i]):14.1f}")
print("health (photons):", {k: float(np.sum(v)) for k, v in light.health.items()})

# choosing the photon count: the error falls as 1/sqrt(samples)
quick = lumen.simulate(det, src, samples=2**16, seed=0)
r0 = float(quick.arrivals_error[1] / quick.arrivals[1])
target = 0.002
print(f"PMT 1: relative error {r0:.4f} at 2**16 samples; for {target} use about {int(2**16 * (r0 / target) ** 2):,}")

# ------------------------------------------------------------------------------------------------- 3. hit map
section("3. A hit map")
xyz = sensor_positions(det)
for i, (x, y, z) in enumerate(xyz):
    print(f"  PMT {i} at ({x:+.3f}, {y:+.3f}, {z:+.3f}) m: {float(pe[i]):8.1f} PE")
try:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib is not installed: no figure")


def unfold(xyz, h=0.5):
    """Return 2-D positions (m) on the unfolded cube of half side h: the y = -h face in the middle (x, z)."""
    x, y, z = xyz.T
    k = np.argmax(np.abs(xyz), axis=1)                    # the face: 2k for -x, -y, -z; 2k + 1 for +x, +y, +z
    face = 2 * k + (xyz[np.arange(len(xyz)), k] > 0)
    u = np.select([face == 0, face == 1, face == 3, face == 4, face == 5], [-2 * h - y, 2 * h + y, 4 * h - x, x, x], x)
    v = np.select([face == 4, face == 5], [-2 * h - y, 2 * h + y], z)
    return np.stack([u, v], 1)


if plt is not None:
    h, uv = 0.5, unfold(xyz)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cu, cv in ((-2 * h, 0), (0, 0), (2 * h, 0), (4 * h, 0), (0, 2 * h), (0, -2 * h)):   # the six faces
        ax.add_patch(plt.Rectangle((cu - h, cv - h), 2 * h, 2 * h, fill=False, lw=0.8, color="0.6"))
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=np.asarray(pe), s=250, cmap="viridis")
    for i, (u, v) in enumerate(uv):
        ax.annotate(str(i), (u, v), ha="center", va="center", color="w", fontsize=8)
    ax.set(xlim=(-3.1 * h, 5.1 * h), ylim=(-3.1 * h, 3.1 * h), aspect="equal", xticks=[], yticks=[],
           title="expected PE on the unfolded tank (front face y = -0.5 m in the middle)")
    fig.colorbar(sc, label="photoelectrons", shrink=0.8)
    fig.savefig(OUT / "hitmap.png", dpi=120, bbox_inches="tight")
    print("wrote", OUT / "hitmap.png")

# ----------------------------------------------------------------------------- 4. derivatives and a small fit
section("4a. Derivative with respect to the source position")


def expected_pe_at(z, seed=0):
    """Return the expected PE per PMT for a source at (0, 0, z); the moving-source method needs wrt."""
    s = lumen.sources.Point(position=jnp.stack([0.0, 0.0, z]), lam=400.0, n_photons=2e5)
    light = lumen.simulate(det, s, samples=2**18, seed=seed, wrt=["source/position"])
    return lumen.readout.pe(light, p0)


mu, dmu = jax.jvp(expected_pe_at, (jnp.float32(0.1),), (jnp.float32(1.0),))
print(" id   PE     dPE/dz (1/m)")
for i in range(len(mu)):
    print(f"{i:3d} {float(mu[i]):8.1f} {float(dmu[i]):12.1f}")

# check: the derivative against a central finite difference (h = 2 cm), each on the same random numbers, 8 seeds
h, ad, fd = 0.02, [], []
for seed in range(8):
    ad.append(jax.jvp(lambda z, s=seed: expected_pe_at(z, s), (jnp.float32(0.1),), (jnp.float32(1.0),))[1])
    fd.append((expected_pe_at(jnp.float32(0.1 + h), seed) - expected_pe_at(jnp.float32(0.1 - h), seed)) / (2 * h))
ad, fd = np.asarray(ad, np.float64), np.asarray(fd, np.float64)
for i in (0, 1):
    print(f"PMT {i}: derivative {ad[:, i].mean():8.1f} +- {ad[:, i].std(ddof=1) / 8**0.5:5.1f}   "
          f"finite difference {fd[:, i].mean():8.1f} +- {fd[:, i].std(ddof=1) / 8**0.5:5.1f}   (1/m, 8 seeds)")

section("4b. Fit the source height to a simulated event")
truth = lumen.sources.Point(position=(0.0, 0.0, 0.1), lam=400.0, n_photons=2e5)
event = lumen.readout.sample(lumen.simulate(det, truth, samples=2**20, seed=7), p0, seed=1)
n_obs = jnp.asarray(event.pe, jnp.float32)
print("observed PE:", np.asarray(n_obs).astype(int).tolist())
z = jnp.float32(0.0)
for it in range(5):                                       # Fisher scoring on the Poisson likelihood
    mu, dmu = jax.jvp(expected_pe_at, (z,), (jnp.float32(1.0),))
    z = z + jnp.sum((n_obs - mu) * dmu / mu) / jnp.sum(dmu**2 / mu)
    print(f"  iteration {it}: z = {float(z):+.5f} m")
sigma = 1.0 / jnp.sqrt(jnp.sum(dmu**2 / mu))
print(f"fitted z = {float(z):+.4f} +- {float(sigma):.4f} m (truth +0.1000)")

section("4c. Sensor values: derivative after tracking, and a calibration")
KEY = "sensors/pmt/efficiency"
true_eff = jnp.asarray([0.97, 0.92, 1.0, 0.88, 0.95, 0.9, 0.85, 0.99], jnp.float32)
n_events = 20                                             # 20 flashes of the same source, summed per PMT
truth_p = p0.set(KEY, true_eff)
rng = np.random.default_rng(100)                         # PE counts as readout.pe describes them: no electronics
data = rng.poisson(n_events * np.asarray(lumen.readout.pe(light, truth_p), np.float64))


def nll(p):
    """Poisson negative log-likelihood of the observed PE (constant terms dropped)."""
    m = n_events * lumen.readout.pe(light, p)
    return jnp.sum(m - data * jnp.log(m))


g = jax.grad(nll)(p0)                                     # a Params of derivatives, one per parameter
print("dNLL/d efficiency at 1:", [round(float(v), 1) for v in g[KEY]])
mu1 = n_events * np.asarray(lumen.readout.pe(light, p0), np.float64)        # expected PE at efficiency 1
eff = np.clip(np.asarray(data) / mu1, 0.0, 1.0)            # the Poisson maximum-likelihood efficiency
err = np.sqrt(np.asarray(data)) / mu1
print("fitted efficiency:", [round(float(v), 3) for v in eff])
print("its error:        ", [round(float(v), 3) for v in err])
print("true efficiency:  ", [round(float(v), 3) for v in true_eff])

name = f"eff_fit_{int(time.time())}"
path = det.save_calibration(p0.set(KEY, eff), name, note=f"PMT efficiencies from {n_events} simulated flashes")
print("wrote", path)
det2 = detector(calibration=name)
print(f"reloaded with calibration {det2.calibration}: efficiency from '{det2.params().explain(KEY)}':",
      np.round(np.asarray(det2.params()[KEY]), 3).tolist())

section("4d. An optical value: its derivative through simulate")
KEY_A = "materials/Water/ABSLENGTH"


def total_pe(s):
    p = p0.scale(KEY_A, s)                                    # every knot of the table times s
    return lumen.readout.pe(lumen.simulate(det, src, p, samples=2**18, seed=0, wrt=[KEY_A]), p).sum()


value, slope = jax.jvp(total_pe, (1.0,), (1.0,))
h = 1.0 / 16
fd = (total_pe(1.0 + h) - total_pe(1.0 - h)) / (2 * h)       # the same photons on both sides
print(f"total PE {float(value):.2f}; d/d(ABSLENGTH scale): {float(slope):.4f} (finite difference {float(fd):.4f})")

# ------------------------------------------------------------------------------------------------- 5. waveform
section("5. A waveform")
wf = lumen.readout.expected_waveform(light, p0)
print("mean", wf.mean.shape, "unit", wf.unit, "t0", wf.t0, "ns, dt", wf.dt, "ns")
t = wf.t0 + wf.dt * np.arange(wf.mean.shape[1])
k = int(np.argmax(np.asarray(wf.mean[1])))
print(f"PMT 1: baseline {float(wf.mean[1, 0]):.2f}, peak {float(wf.mean[1, k]):.1f} {wf.unit} at {t[k]:.1f} ns, "
      f"standard deviation there {float(jnp.sqrt(wf.variance[1, k])):.2f}")
ev = lumen.readout.sample(light, p0, seed=1)
print("one event: PE", np.asarray(ev.pe).astype(int).tolist(), "; waveforms", ev.waveforms.shape,
      "; hits", len(ev.hits["sensor"]))
if plt is not None:
    fig, ax = plt.subplots(figsize=(5, 3))
    sd = np.sqrt(np.asarray(wf.variance[1]))
    ax.plot(t, np.asarray(wf.mean[1]), label="expected")
    ax.fill_between(t, np.asarray(wf.mean[1]) - sd, np.asarray(wf.mean[1]) + sd, alpha=0.3, label="±1 s.d.")
    ax.plot(t, ev.waveforms[1], lw=0.8, label="one event")
    ax.set(xlabel="time after emission (ns)", ylabel=wf.unit, title="PMT 1")
    ax.legend()
    fig.savefig(OUT / "waveform.png", dpi=120, bbox_inches="tight")
    print("wrote", OUT / "waveform.png")

section("done")
