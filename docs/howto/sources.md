# Light sources

A source says where the light starts, in which direction, when and at what wavelength. `lumen.simulate(det, source,
...)` takes any of the sources below. Every field is a parameter under `source/<field>`, so you can set it through
`params` or differentiate with respect to it (trace it: a tangent on `source/position`). Units are m, ns and nm.

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
`lumen.format.calibrations.Table(energy, value)`. The position has no derivative in this version: a tangent on
`source/position` is refused ([known limits](../explanation/known-limits.md#derivatives-in-the-source-position)).

## Charged particles: `CherenkovSegment`, `ScintillationPoint`, `ScintillationSegment`

`CherenkovSegment(material, start, direction, length, beta, charge, ...)` is the Cherenkov light of a straight
charged segment in a material, with the material's RINDEX read from the detector. Its light covers the wavelengths of
the material's RINDEX table, as in Geant4 (G4Cerenkov emits over the table's energies):

```python
mu = lumen.sources.CherenkovSegment("Water", start=(0.0, 0.0, -0.3), direction=(0.0, 0.0, 1.0), length=0.5,
                                    beta=0.99)
light = lumen.simulate(det, mu, samples=2**16, seed=0)
print(float(light.arrivals.sum()))
```

`ScintillationPoint.of(params, material, position=, energy=)` and `ScintillationSegment` give the scintillation of an
energy deposit (MeV), reading the material's yield, spectra and time constants from the detector's parameters; the
toy tank's water does not scintillate, so they need a detector with a scintillator.

## A Geant4 step dump: `CherenkovSteps`, `ScintillationSteps`

`CherenkovSteps(material, start=, end=, t_start=, t_end=, beta_start=, beta_end=, charge=)` turns the charged steps of
a Geant4 run (arrays of shape (n, 3) for positions in m, (n,) for the rest) into their Cherenkov light;
`ScintillationSteps.of(params, material, start=, end=, t_start=, t_end=, edep=)` into their scintillation, the energy
deposited `edep` in MeV:

```python
n = 4
start = np.array([[0.0, 0.0, -0.3 + 0.1 * i] for i in range(n)])
steps = lumen.sources.CherenkovSteps("Water", start=start, end=start + [0.0, 0.0, 0.1],
                                     t_start=np.arange(n) * 0.34, t_end=np.arange(n) * 0.34 + 0.34,
                                     beta_start=np.full(n, 0.99), beta_end=np.full(n, 0.99), charge=np.ones(n))
light = lumen.simulate(det, steps, samples=2**16, seed=0)
print(float(light.arrivals.sum()))
```

## Photons you already have: `FromArrays`

`FromArrays(position, direction, time, lam, weight)` traces a list of photons as given: positions (N, 3) m, unit
directions (N, 3), times (N,) ns, wavelengths (N,) nm and weights (N,). Photon i of the run is row i, and each row is
traced once, so the list has no sample count of its own: leave `samples` out. Every per-row field has the same N rows
(a field given once is every row's); a list with columns of different lengths is refused.

```python
N = 4096
rng = np.random.default_rng(0)
d = rng.normal(size=(N, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
photons = lumen.sources.FromArrays(np.zeros((N, 3)) + [0.0, 0.0, 0.1], d, np.zeros(N), np.full(N, 400.0),
                                   np.full(N, 2e5 / N))
light = lumen.simulate(det, photons, seed=0)
print(float(light.arrivals.sum()))
```

A tangent on their positions, directions or wavelengths moves where the light starts: `simulate` raises
`GradientNotAvailable` (no source-position derivative in this version). Their weights can carry a derivative.

**Give the photons equal weights.** Every photon is followed with the same effort, so a list whose weights are spread
spends most of it on photons that carry little light, and its noise is set by the few heavy ones. Resample such a list
with probability proportional to the weight, each drawn photon at the mean weight (or split heavy photons into rows
of equal weight). On lists spread over three decades, or with 1 % of the rows 100 times brighter, that was 1.6 to 51
times more efficient at equal run time. The same holds for a `@lumen.source` function.

## A source of your own

An LED with a beam profile, a laser spot, several emitters at once: write the source as one function with
`@lumen.source`, its parameters declared as keys. [Extending lumen](extending.md), section 2, shows how, with worked
examples in `examples/extending.py`.
