# Light sources

A source says where the light starts, in which direction, when and at what wavelength. `lumen.simulate(det, source,
...)` takes any of the sources below. Every field is a parameter under `source/<field>`, so you can set it through
`params` or differentiate with respect to it (`wrt=["source/position"]`). Units are m, ns and nm.

`n_photons` (or the physics of the source: a track's length, a deposited energy) is how much light the source emits.
`samples` in `simulate` is how many Monte Carlo photons stand for it; the results are expectations, so doubling
`samples` only shrinks the errors.

The examples use the tutorial's toy detector, from the repository's top directory:

```python
import sys; sys.path.insert(0, "examples")
import numpy as np
import jax.numpy as jnp
import lumen
from example_tank import detector

det = detector()
```

## A point: `Point`

Isotropic light from a point at one wavelength, or a pencil beam with `direction`:

```python
lamp = lumen.sources.Point(position=(0.0, 0.0, 0.1), lam=400.0, n_photons=2e5)
beam = lumen.sources.Point(position=(0.0, 0.0, 0.1), lam=400.0, n_photons=2e5, direction=(0.0, 0.0, 1.0))
light = lumen.simulate(det, lamp, samples=2**16, seed=0)
print(light.arrivals.shape, float(light.arrivals.sum()))
```

A spectrum instead of one wavelength: `spectrum=` a table of photon energies (eV) and intensities, as
`lumen.format.calibrations.Table(energy, value)`. The position derivative (`wrt=["source/position"]`) needs the
directions to be drawn at random, which an isotropic `Point` does and a beam does not.

## Charged particles: `CherenkovSegment`, `ScintillationPoint`, `ScintillationSegment`

`CherenkovSegment(material, start, direction, length, beta, charge, ...)` is the Cherenkov light of a straight
charged segment in a material, with the material's RINDEX read from the detector. Its light covers
`[lam_min, lam_max]` (nm):

```python
mu = lumen.sources.CherenkovSegment("Water", start=(0.0, 0.0, -0.3), direction=(0.0, 0.0, 1.0), length=0.5,
                                    beta=0.99, lam_min=300.0, lam_max=600.0)
light = lumen.simulate(det, mu, samples=2**16, seed=0)
print(float(light.arrivals.sum()))
```

`ScintillationPoint.of(params, material, position=, energy=)` and `ScintillationSegment` give the scintillation of an
energy deposit (MeV), reading the material's yield, spectra and time constants from the detector's parameters; the
toy tank's water does not scintillate, so they need a detector with a scintillator.

## A Geant4 step dump: `Geant4Steps`

`Geant4Steps.of(params, material, light, start=, end=, t_start=, t_end=, beta_start=, beta_end=, charge=, edep=)` turns
the charged steps of a Geant4 run (arrays of shape (n, 3) for positions in m, (n,) for the rest) into their Cherenkov
(`light="cherenkov"`) or scintillation (`light="scintillation"`) light:

```python
n = 4
start = np.array([[0.0, 0.0, -0.3 + 0.1 * i] for i in range(n)])
steps = lumen.sources.Geant4Steps.of(det.params(), "Water", "cherenkov", start=start, end=start + [0.0, 0.0, 0.1],
                                     t_start=np.arange(n) * 0.34, t_end=np.arange(n) * 0.34 + 0.34,
                                     beta_start=np.full(n, 0.99), beta_end=np.full(n, 0.99),
                                     charge=np.ones(n), edep=np.full(n, 2e7), lam_min=300.0, lam_max=600.0)
light = lumen.simulate(det, steps, samples=2**16, seed=0)
print(float(light.arrivals.sum()))
```

## Photons you already have: `FromArrays`

`FromArrays(x, d, t, lam, w)` traces a list of photons as given: positions (N, 3) m, unit directions (N, 3), times
(N,) ns, wavelengths (N,) nm and weights (N,). Photon i of the run is row i, so pass `samples=N`:

```python
N = 4096
rng = np.random.default_rng(0)
d = rng.normal(size=(N, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
photons = lumen.sources.FromArrays(np.zeros((N, 3)) + [0.0, 0.0, 0.1], d, np.zeros(N), np.full(N, 400.0),
                                   np.full(N, 2e5 / N))
light = lumen.simulate(det, photons, samples=N, seed=0)
print(float(light.arrivals.sum()))
```

A derivative that moves such photons needs to know how their directions were drawn: pass `drawn=True` and the
gradient of the directions' log density, `dlogp=` (N, 3); for isotropic directions that is
`dlogp=jnp.zeros((N, 3))`. Otherwise `simulate` raises `GradientNotAvailable`.

## A source of your own

An LED with a beam profile, a laser spot, several emitters at once: write the source as one function with
`@lumen.source`, its parameters declared as keys. [Extending lumen](extending.md), section 2, shows how, with worked
examples in `examples/extending.py`.
