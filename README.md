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
git clone https://github.com/OmarAlterkait/lumen.git
cd lumen
python -m venv .venv && source .venv/bin/activate
pip install "jax[cuda12]==0.6.2"
# tallymap: PLACEHOLDER pin until its v0.6.0rc1 tag is published
pip install "tallymap @ git+https://github.com/OmarAlterkait/tallymap.git@v0.6.0rc1"
pip install -e .
pip install matplotlib                                   # optional: the tutorial's figures
python -c "import jax, lumen; print(lumen.__version__, jax.devices())"
```

The last line should list a `CudaDevice`. If it lists only `CpuDevice`, JAX did not find the CUDA driver. lumen still
runs in that case, on the CPU backend.

lumen keeps compiled kernels and converted detectors in a cache. Point the cache at a disk with some space, and name a
directory for your calibrations:

```bash
export LUMEN_CACHE=/path/with/space/lumen-cache          # default: ~/.cache/lumen
export LUMEN_DETECTORS=/path/to/your/calibrations        # where calibrations are saved and found
```

The first run of a detector compiles its photon loop, which takes a few minutes. Later runs reuse the cache, in new
processes too.

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
