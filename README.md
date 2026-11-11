# lumen

A differentiable optical-photon simulation for neutrino detectors, written in JAX.

lumen reads a detector from GDML and traces optical photons through it with Geant4's optical physics. It records the
light that reaches each sensor and turns it into photoelectrons, hit times, charges and PMT waveforms. It also gives
derivatives of those outputs with respect to the detector's parameters: absorption and scattering lengths, surface
reflectivities, PMT efficiencies and timing, the position of the light source, and so on. The geometry is the one
thing that is not a parameter.

What that is for:

- **Calibration.** Fit optical and PMT parameters to calibration data with gradients instead of scanning them.
- **Reconstruction.** Fit a vertex, a time or a light yield directly to the hit pattern.
- **Detector studies.** Change a value and simulate again without recompiling.

lumen is built for water Cherenkov detectors (Super-Kamiokande, WCTE, ANNIE) and liquid-argon TPCs (SBND,
MicroBooNE). It also works with any other detector you can describe in GDML. Its results are checked against Geant4
and WCSim. The [validation report](docs/validation/report.md) lists the checks and what each one measured.

**Status: 0.1.** The interfaces can still change, and lumen is not on PyPI. `simulate` differentiates optical values
(absorption and scattering lengths, reflectivities, surface roughness), the source position and, through the readout,
the PMT values; refractive indices are calibrated by value only.

## Requirements

- Linux with an **NVIDIA GPU** and a **CUDA 12** driver. lumen runs its photon loop through
  [tallymap](https://github.com/OmarAlterkait/tallymap), which compiles a CUDA kernel for it.
- Python 3.10 or newer.
- Without a GPU, lumen falls back to tallymap's plain-JAX CPU backend. Its results agree with the GPU's to about
  2.5e-4, but it compiles slowly and runs slowly. Use it for tests and for small photon counts (up to about 2^16
  photons).

## Install

lumen pins exact versions of JAX and tallymap. Other versions change the photon loop's results bit for bit, or break
its reverse-mode derivatives.

```bash
git clone --branch v0.1.1 https://github.com/OmarAlterkait/lumen.git
python -m venv /path/to/envs/lumen && source /path/to/envs/lumen/bin/activate   # anywhere; not inside the clone
pip install "jax[cuda12]==0.6.2"
pip install "tallymap @ git+https://github.com/OmarAlterkait/tallymap.git@v0.6.0rc1"
pip install -e lumen
pip install matplotlib                                   # optional: the tutorial's figures
python -c "import jax, lumen; print(lumen.__version__, jax.devices())"
```

`jax[cuda12]` brings JAX's CUDA plugin and NVRTC, the compiler tallymap's CUDA backend uses; that is all
`tallymap[cuda]` would add, so plain `tallymap` after `jax[cuda12]` is the same install. Without `jax[cuda12]` (plain
`jax`), tallymap runs on its CPU backend only. The CUDA backend also builds a small C++ launch shim once, so a C++17
compiler must be on the PATH the first time.

The last line should list a `CudaDevice`. If it lists only `CpuDevice`, JAX did not find the CUDA driver. lumen still
runs in that case, on the CPU backend.

lumen keeps compiled kernels and converted detectors in a cache. Point the cache at a disk with some space, and name a
directory for your calibrations. The variables, in one place:

| Variable | What | Default |
|---|---|---|
| `LUMEN_CACHE` | lumen's cache root | `$XDG_CACHE_HOME/lumen` (`~/.cache/lumen`) |
| `TALLYMAP_CUDA_CACHE` | tallymap's compiled CUDA kernels | `$LUMEN_CACHE/tallymap` (lumen sets it; tallymap's own default is `~/.cache/tallymap`) |
| `JAX_COMPILATION_CACHE_DIR` | JAX's persistent compilation cache | `$LUMEN_CACHE/xla` (never under HOME) |
| `CUDA_CACHE_PATH` | the NVIDIA driver's own cache | `~/.nv/ComputeCache` (set it if HOME is small) |
| `LUMEN_DETECTORS` | where detector files and calibrations are saved and found (a search path) | none |
| `LUMEN_DATA` | shared reference data (the validation checks) | none |

```bash
export LUMEN_CACHE=/path/with/space/lumen-cache
export LUMEN_DETECTORS=/path/to/your/calibrations
```

The first run of a detector compiles its photon loop, which takes a few minutes. Later runs reuse the cache, in new
processes too. A measured large detector (43,212 sensors, one NVIDIA L40S): convert and first load about 10 minutes,
the first `simulate` 31 s, later ones 2.2 s at 2^18 photons. At 2^18 photons from a point at the centre, 45 % of
those sensors saw no light: choose the photon count from the light per sensor you need (the Monte Carlo error of a
sensor's expected light falls as 1 / sqrt(photons)), and check `Light.arrivals_error`.

## Quick start

The repository ships a toy detector: a 1 m cube of water with 8 small PMTs (`examples/data/example_tank.gdml`). From
the repository's top directory:

```python
import sys; sys.path.insert(0, "examples")
import lumen
from example_tank import detector

det = detector()                                          # converted from its GDML on first use, then loaded
src = lumen.sources.Point(position=(0.0, 0.0, 0.1), lam=400.0, n_photons=2e5)    # isotropic, 400 nm
light = lumen.simulate(det, src, samples=2**18, seed=0)   # 2^18 Monte Carlo photons

print(light.arrivals)                                     # expected photons reaching each PMT
print(light.arrivals_error)                               # their Monte Carlo standard errors
print(lumen.readout.pe(light, det.params()))              # expected photoelectrons per PMT
```

Then follow the tutorial.

## Documentation

- [The first hour](docs/tutorial/first-hour.md): the toy detector, what the numbers mean, a hit map, derivatives,
  a fit and a waveform.
- [Your own detector](docs/howto/your-own-detector.md): import your GDML, read the repair warnings, simulate,
  calibrate, export, and check against Geant4.
- [Errors](docs/reference/errors.md): the errors lumen raises, what each means, and the fix.
- [Vocabulary](docs/reference/vocabulary.md): the words lumen uses.
- Background: [parameters](docs/explanation/parameters.md), [known limits](docs/explanation/known-limits.md) and
  [how the checks decide](docs/explanation/validation.md).

## Citing

If you use lumen in a publication, please cite this repository. A `CITATION.cff` file will come with the 0.1 release.

## Licence

lumen is released under the [Apache License 2.0](LICENSE).
