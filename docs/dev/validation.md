# Validation internals

How the gradient check decides, and the measurements behind the known limits. The user-facing summary is
`docs/explanation/validation.md`; the code is `lumen.compare.gradients` and `lumen.tools.stats`.

## The gradient comparisons

The gradient check runs `lumen.simulate`, as production does, per parameter family on K seeds, and gates three tests
on the same seeds (`lumen.compare.gradients`):

- **exact**: the forward derivative (AD) at the default roulette against the finite differences of the
  fixed-reference function, F(θ; reference θ0) near θ0 (`simulate(reference=)`), per seed and output. The roulette
  decides on the reference weight, so with the reference held no draw and no decision moves. The differences are taken
  on a ladder of four steps, h0 / 2^k (central from h0 = 1/64; one-sided inward at an edge from 1/1024), and their
  order is measured (`stats.fd_order`): where the differences contract at least at the stencil's order all along the
  ladder (2 central, 1 one-sided; up to the sides' float32 rounding, 1e-6 |F| / h) the cell is in its asymptotic range and
  AD must agree with the extrapolated difference within 1e-4 of AD, the rounding and the extrapolation's own size.
  A cell that does not contract so is not tested, and its measured order is reported. Two kinds do not: a smooth
  realisation not yet asymptotic at these steps (a photon reflected many times at R near 1 carries R^n: on ANNIE's
  frame the one-sided difference reads −15988, −31510, −50357, −54792 and −55268 at h = 1/64 ... 1/16384 against AD
  −55285), and a realisation that is not smooth at any fixed step (GLISUR's polish moves the facet cone's edge: terms
  sqrt(δ_i ± Δr) with δ_i arbitrarily small, FD converging like h^1/2). An AD that misses a term differs from a
  difference that converges at the stencil's order, so it is caught. A row with no cell tested is gated on the other
  two tests. The time-weighted arrivals are summed from the deposits (each charge times its time), not formed as the
  arrivals times their mean time: that product is 0 with derivative 0 on a sensor with no light at the values, so it
  misses light an edge row opens (WCTE's absorber at R = 0).
- **value**: F(θ0 + h; reference θ0) against F(θ0 + h) run at its own values, paired on the same K seeds: the
  reweighted estimate is the light at the new values, so the derivative of the fixed-reference function is the
  expectation's. Sharing the seeds, the two sides share their uniforms and F's own seed fluctuation cancels; on
  disjoint seeds the test measured that fluctuation instead. The variance of the roulette's flips, which the shared
  uniforms cannot cancel, is the test's floor (`gradients.disagreement`).
- **production**: AD at the default ε against AD at ε = 1e-6, paired by seed on 64 seeds (at least K): the roulette
  moves no derivative.

The value and production tests are `lumen.tools.stats.mean_test`, the comparisons' one test (also the charge check's
per-sensor test and `python -m lumen.compare mc agree`), each at ALPHA (the two-sided tail beyond 3σ) divided by the
check's statistical tests, two per family (the look-elsewhere over families, which share seeds). Per output (each
sensor group's Q and Tw, and the totals): the mean difference over its standard error from the seeds, referred to
Student's t with as many degrees of freedom as the seeds and the photons carrying the output support (min(K, n) - 1,
n Kish's effective count, `stats.effective_count`, from the photon sums the recorder keeps), as a normal score; the
row's statistic is the largest |score| over the outputs with power, against the largest of normals correlated as
the outputs are over the seeds (Q and Tw of a group are the same light; they count once, by the data). An output has
power only when its effective count n has e^-n below the level: otherwise a sample showing none of its carriers, the
light wholly missing, is not significant, and no difference can be. The cut is ln(tests / ALPHA), the level's own.
A test with no output with power gives no result. A row's results combine by one rule (`stats.verdict`): any
failure fails it, the exactness test's too (it is deterministic and needs no power); with none passing it is reported
(no power is never agreement); else it passes. A failing row's note names the test and its statistic. The summary row,
and every check's exit status, pass only when some row passes and none fails (`stats.gate`): a check never passes on
nothing.

Reverse mode (`jax.grad` of a fixed random weighting of the arrivals, every family traced) is compared with the same
fixed-reference finite difference under the exact rule; a family with no cell in its asymptotic range is not tested,
and a row with no family tested is reported. The row's value is AD at the default roulette, the number the
tests gate.

### The defensive mixture: a fixed 1 % share

AD at an edge is drawn through the one defensive mixture (`lumen.transport.estimator`, `DEFENSIVE_SHARE`): the
outcomes the edge closes are proposed with a fixed share of the draws (1 %), so the path they open is sampled; such a
photon's weight is 0 in value and its reference weight carries the density drawn from over itself, so the roulette
keeps it (a reference weight scaled by the share instead thinned exactly those photons: x51 the variance of WCTE's
AbsorberSurface reflectivity derivative, gradfam_p3.md section 7). A share of 1 % of the absorbed fraction was
tried first: it was 0.1 % at a reflectivity of 0.9; the derivative's positive part then came from a draw in a
thousand at a large weight, and its replicas' spread understated the error (a closed unified box: AD sat one to three
standard errors below the difference at every h, and agreed at a larger share). The fixed share keeps the estimand
and cuts that error about three times.

### A family that is exactly zero

ANNIE's LAPPD glass transmittance has an AD of exactly 0 (its LAPPDs are not sensors: lumen-wc's docs): with 2^20 photons from
the gradient check's sources, 93,382 enter an LAPPD interior at T = 0 and 127,019 at T = 1 − R = 0.68, and the charge
they deposit afterwards is exactly 0 in both, with no lost weight. With the edge rows on the transmitted outcome is
drawn (481 times in 92,740 crossings, 2^19 photons) and AD stays exactly 0. The fixed-reference difference is then 0
too, so the exact test passes on it; an edge-rows bug (the outcome never proposed) gives AD 0 against a fixed-reference
difference that is not, and fails. `tests/physics/test_edge_rows.py` checks the same estimator on a toy with a closed
form (dQ/dT = exp(−0.75 / L) per photon at T = 0+).

## The MC gate and its re-baseline

The MC gate is one command per detector package:

```bash
python -m lumen.compare gate --package lumen_wc [sk wcte annie] --out validation/mc
python -m lumen.compare gate --package lumen_tpc --out validation/mc
```

It runs every `mc` entry of the package's manifest (`REFERENCES`): the entry's reference product (photon list and
reference record) is fetched with `lumen.compare.references.product`, which checks every file's sha256 against the
manifest; the package's detector is converted with its recipe; the record (16 seeds x 65,536 photons: the arrivals
and the forward derivatives along the reference record's keys) is written with the run's environment
(`lumen.compare.records.environment`: lumen and its tree, tallymap, jax, jaxlib, the CUDA runtime, NVRTC, the
device) and compared bit for bit. The output names the environment entries that differ from the reference record's
before the comparison, so a difference that comes from a new jax, CUDA or GPU model is told from one in the code.
Exit 0 when every array of every entry is bitwise identical.

Bitwise identity holds on one environment: the same GPU model, CUDA runtime, NVRTC, jax/jaxlib and tallymap. A change
of any of these, or a change that is meant to move results, needs a new reference, made as follows:

1. Run the gate on the code before the change and on the code after it, in the new environment, and record both:
   `python -m lumen.compare gate --package PKG --out before` (and `--out after`); `--against before` compares the two
   bit for bit.
2. A change meant to keep results must be bitwise identical to `before`. Otherwise test agreement in expectation on
   independent seeds: `python -m lumen.compare mc run DET.lumen PHOTONS.npz new.npz --keys-from REF.npz --seed0 1000`
   against the reference, then `python -m lumen.compare mc agree REF.npz new.npz`; every array agrees, or the change
   is a physics change whose expected effect is stated and measured.
3. Pack the new records with the same photon lists as a new product (`tools/g4ref/mc_store.sh`:
   `<store>/mc-gate-<family>-<lumen commit>/`, its files' sha256 and its producer, the lumen and family commits and
   the environment), add it to the package's manifest, point the `mc` entries at it, and make its release asset
   (`tools/release_assets.py`: the archive, its URL and sha256 in the manifest). Keep the old product: a record is
   never overwritten.
4. Run the gate once more against the manifest, from an empty store: it fetches the new product, checks its sha256,
   and is bitwise identical.

## The absorbing Fresnel derivative

Compared per nk from 1e-6 to 1.5 over 815 (incidence, index) points: how many have a float32 dR/dnk more than 1e-3
from float64, away from and at grazing incidence. Away from grazing incidence: 10 points
above 1e-3 at nk = 1e-6 (largest 6 %), 7 at 1e-5 (0.4 %), none from 1e-4 up. At grazing incidence (cos_i ≤ 1e-3) the
absolute error reaches 0.15 to 0.28 for nk ≤ 1e-4, where the derivative itself is up to ~1e5, and falls to 6e-8 at
nk = 1.5.

## Compiled programs and the persistent cache

JAX's persistent compilation cache is JAX's own setting (`JAX_COMPILATION_CACHE_DIR`); `import lumen` sets nothing in
JAX's configuration. Bitwise checks rely on a program loaded from that cache computing exactly
what a fresh compile computes. On jax 0.5.3 it did not: the navigator missed faces on the CPU in programs loaded from
the cache on XLA's thunk runtime; with the cache off, no process missed. On jax 0.6.2, the pinned version, a load from
the cache is bitwise equal to a fresh compile on the CPU and the GPU (a photon-loop record and leaf-module outputs,
three separate processes: cache off, writing, loading). The GPU tier checks it on every run
(`tests/physics/test_persistent_cache.py`). See decision 7 in `decisions.md`. JAX writes a program to the cache
only when it took longer than `jax_persistent_cache_min_compile_time_secs` (1 s by default) to compile; every lumen
program compiles faster, so set `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` with the directory.
