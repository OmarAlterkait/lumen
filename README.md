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

**Status: 0.1.2.** The interfaces can still change, and lumen is not on PyPI. `simulate` differentiates optical
values (absorption and scattering lengths, reflectivities, surface roughness), the source position and, through the
readout, the PMT values; refractive indices are calibrated by value only. What changed between releases is in the
[changelog](CHANGELOG.md).

## Install

lumen needs Linux, Python 3.10 or newer and, for full speed, an NVIDIA GPU with a CUDA 12 driver. In a fresh
environment:

```bash
python -m venv /path/to/envs/lumen && source /path/to/envs/lumen/bin/activate
pip install "lumen[cuda] @ git+https://github.com/OmarAlterkait/lumen.git"
python -c "import jax, lumen; print(lumen.__version__, jax.devices())"
export LUMEN_CACHE=/path/with/space/lumen-cache         # compiled kernels and converted detectors
export LUMEN_DETECTORS=/path/to/your/calibrations       # where your calibrations are saved
```

This also installs lumen's photon-loop engine, tallymap 0.6.0, from its tag (`tallymap @
git+https://github.com/OmarAlterkait/tallymap.git@v0.6.0`). The last Python line should list a `CudaDevice`. [Installing
lumen](docs/howto/install.md) has the details: the pinned versions, running without a GPU, every cache variable, compile
times and a fixed-version install.

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

- [Installing lumen](docs/howto/install.md): GPU or CPU, the pinned versions, the caches and their variables.
- [The first hour](docs/tutorial/first-hour.md): the toy detector, what the numbers mean, a hit map, derivatives,
  a fit and a waveform.
- [Your own detector](docs/howto/your-own-detector.md): import your GDML, read the repair warnings, simulate,
  calibrate, export, and check against Geant4.
- [Light sources](docs/howto/sources.md) and [the shipped detectors](docs/howto/shipped-detectors.md).
- [Extending lumen](docs/howto/extending.md): your own source, sensor response, medium or surface model.
- Reference: [the API](docs/reference/api.md), [PMT value sets](docs/reference/value-sets.md).
- [Errors](docs/reference/errors.md): the errors lumen raises, what each means, and the fix.
- [Vocabulary](docs/reference/vocabulary.md): the words lumen uses.
- Background: [parameters](docs/explanation/parameters.md), [known limits](docs/explanation/known-limits.md) and
  [how the checks decide](docs/explanation/validation.md).

## Citing

If you use lumen in a publication, please cite it as [`CITATION.cff`](CITATION.cff) says (GitHub's "Cite this
repository" button reads it).

## Licence

lumen is released under the [Apache License 2.0](LICENSE).
