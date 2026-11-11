# Validation internals

How the gradient check decides, and the measurements behind the known limits. The user-facing summary is
`docs/explanation/validation.md`; the code is `lumen.validate.gradients` and `lumen.validate.stats`.

## The gradient comparisons

Automatic derivatives (AD) are compared with finite differences on the same random numbers (FD, the expectation) and
with the fixed-reference difference (FDX: the same weighted estimator on the same draws). A family whose derivative
is below float32 resolution is reported, not gated; a family that moves no random draw is gated on the exactness
comparison only. Reverse mode is compared with forward mode on a fixed random weighting of the outputs.

### The fixed-reference difference's own float32 error

FDX is itself a float32 computation, and its rounding is not always small against the observable. On surfaces with a
facet path (unified and GLISUR lobes) it evaluates the path's log density at σ(1 ± h); those terms (−2 log σ,
−α²/2σ², log F, log ζ) round the same way on every photon, so the difference carries a bias of one sign, about
2.5e-4 per lobe reflection at h = 1/1024 and scaling as 1/h, which adds up along a path and does not average down.
Found on ANNIE's inner-structure steel σ_α (1.2 % at h = 1/1024): the float64 difference equals AD to 1e-10, so the
derivative is right. The exactness test therefore estimates the difference's own error from the difference at h/4 (a
1/h error gives |FDX(h/4) − FDX(h)| / 3) and allows three times it; an AD that misses a term still disagrees at every
h.

### The edge step scan: two neighbours

A family at the edge of its domain (a probability at 0 or 1) takes a one-sided finite difference, and its step h is
chosen by scanning down by four: the step is accepted where its difference agrees with both neighbours, at 4h and at
h/4, within two standard errors. One agreement is not enough: on ANNIE's R7081 photocathode
(`dielectric_rt_sensor.T`, 16x the default photons) the differences at h = 1/16 and 1/64 agreed, yet the one at 1/64
was 2.2 standard errors off AD (−0.18129 against −0.18201); at 1/256 and 1/1024 it fell on it (−0.18285 and
−0.18195). The truncation at 1/64 was below the noise of one comparison and above that of the final test. With two
agreements required, the scan stops at the first step whose neighbours both confirm it.

### The spread band, widened by the kurtosis

The groups' spread is tested against a band calibrated to the noise it sees: the mean of the groups' z^2 must lie
below its null plus three standard deviations, and that standard deviation is sqrt((κ − 1) / ndf), κ the kurtosis of
the per-replica finite-difference values pooled over the family's groups (never of their differences from AD or of
the groups' z, so a real bias cannot widen its own band; at least 3, the Gaussian value, so a light-tailed family
keeps the Pearson band). A finite difference whose sides change photons' paths has heavy-tailed replicas: on ANNIE's
two PMT photocathode transmittances (`dielectric_rt_sensor.T`, at their edge T = 0) the Gaussian band (1.67 for 50
groups) failed one or the other in turn at the default photon count, χ² 1.86 and 1.87, while their totals agreed (z
+0.44 and −0.51) and both passed at 4x and 16x the photons. The record carries κ for every family.

### The edge floor: a fixed 1 % share

AD at an edge is drawn through the edge floor (`lumen.transport.estimator`): the outcomes the edge closes are proposed
with a fixed share of the draws (1 %), so the path they open is sampled. A share of 1 % of the absorbed fraction was
tried first: it was 0.1 % at a reflectivity of 0.9; the derivative's positive part then came from a draw in a
thousand at a large weight, and its replicas' spread understated the error (a closed unified box: AD sat one to three
standard errors below the difference at every h, and agreed at a larger share). The fixed share keeps the estimand
and cuts that error about three times.

### A family that is exactly zero

The check reports a family as "zero" (`judge_zero`) when its AD is exactly 0 although the edge rows propose the
outcome, and gates it on the FD and FDX totals being consistent with 0. ANNIE's LAPPD glass transmittance is one
(`docs/explanation/known-limits.md`): with 2^20 photons from the gradient check's sources, 93,382 enter an LAPPD
interior at T = 0 and 127,019 at T = 1 − R = 0.68, and the charge they deposit afterwards is exactly 0 in both, with
no lost weight. With the edge rows on, the transmitted outcome is drawn 481 times in 92,740 crossings of that surface
(2^19 photons) and AD stays exactly 0; the FD is −7.0e-4 ± 4.4e-4 per photon. This differs from an edge-rows bug (an
exactly-zero AD with the outcome never proposed), which the check still fails. `tests/physics/test_edge_rows.py`
checks the same estimator on a toy with a closed form (dQ/dT = exp(−0.75 / L) per photon at T = 0+).

## The absorbing Fresnel derivative

A toy (kept outside the repository) prints, per nk from 1e-6 to 1.5, how many of 815 (incidence, index) points have a
float32 dR/dnk more than 1e-3 from float64, away from and at grazing incidence. Away from grazing incidence: 10 points
above 1e-3 at nk = 1e-6 (largest 6 %), 7 at 1e-5 (0.4 %), none from 1e-4 up. At grazing incidence (cos_i ≤ 1e-3) the
absolute error reaches 0.15 to 0.28 for nk ≤ 1e-4, where the derivative itself is up to ~1e5, and falls to 6e-8 at
nk = 1.5.

## Compiled programs and the persistent cache

lumen turns on JAX's persistent compilation cache by default (`<LUMEN_CACHE>/xla`, never under HOME; an explicit
`JAX_COMPILATION_CACHE_DIR` is respected). Bitwise checks rely on a program loaded from that cache computing exactly
what a fresh compile computes. On jax 0.5.3 it did not: the navigator missed faces on the CPU in programs loaded from
the cache on XLA's thunk runtime; with the cache off, no process missed. On jax 0.6.2, the pinned version, a load from
the cache is bitwise equal to a fresh compile on the CPU and the GPU (a photon-loop record and leaf-module outputs,
three separate processes: cache off, writing, loading). The GPU tier checks it on every run
(`tests/physics/test_persistent_cache.py`). See decision 7 in `decisions.md`. JAX writes a program to the cache
only when it took longer than `jax_persistent_cache_min_compile_time_secs` (1 s by default) to compile; every lumen
program compiles faster, so `_env.xla_cache` sets the threshold to 0.
