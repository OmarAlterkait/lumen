# How the validation checks decide

`lumen validate` writes one record per quantity (value, error, z, pass) and the report is generated from them. This
page states the pass rules that are choices rather than plain statistics.

## Charge

- **Total charge.** The ratio of lumen's detected charge to the reference's, with both Monte Carlo errors, passes at
  |z| ≤ 3.
- **Per-sensor agreement.** The Pearson χ²/ndf over the sensors expecting at least 5 hits passes when

      χ²/ndf ≤ 1 + max(3·√(2/ndf), 0.05)

  that is, three standard deviations of χ²/ndf, but never tighter than a 5 % excess. With thousands of sensors the
  pure 3σ band is narrower than any per-sensor model difference worth acting on: the baseline's own SK check had
  χ²/ndf = 1.042 at ndf = 11,096, where 3σ allows only 1.040, while its total charge agreed to 0.1 %.

## Gradients

Automatic derivatives are compared with finite differences on the same random numbers (the expectation) and with the
fixed-reference difference (the same weighted estimator on the same draws). A family whose derivative is below float32
resolution is reported, not gated; a family that moves no random draw is gated on the exactness comparison only.
Reverse mode is compared with forward mode on a fixed random weighting of the outputs.

The fixed-reference difference is itself a float32 computation, and its rounding is not always small against the
observable. On surfaces with a facet chain (unified and GLISUR lobes) it evaluates the chain's log density at
σ(1 ± h); those terms (−2 log σ, −α²/2σ², log F, log ζ) round the same way on every photon, so the difference carries
a bias of one sign, about 2.5e-4 per lobe reflection at h = 1/1024 and scaling as 1/h, which adds up along a path
and does not average down. Found on ANNIE's inner-structure steel σ_α (1.2 % at h = 1/1024): the float64 difference
equals the automatic derivative to 1e-10, so the derivative is right. The exactness test therefore estimates the
difference's own error from the difference at h/4 (a 1/h error gives |FDX(h/4) − FDX(h)| / 3) and allows three times
it; an automatic derivative that misses a term still disagrees at every h.

A family at the edge of its domain (a probability at 0 or 1) takes a one-sided finite difference, and its step h is
chosen by scanning down by four: the step is accepted where its difference agrees with both neighbours, at 4h and at
h/4, within two standard errors. One agreement is not enough: on ANNIE's R7081 photocathode
(`dielectric_rt_sensor.T`, 16x the default photons) the differences at h = 1/16 and 1/64 agreed, yet the one at 1/64
was 2.2 standard errors off the automatic derivative (−0.18129 against −0.18201); at 1/256 and 1/1024 it fell on it
(−0.18285 and −0.18195). The truncation at 1/64 was below the noise of one comparison and above that of the final
test. With two agreements required, the scan stops at the first step whose neighbours both confirm it.

The groups' spread is tested against a band calibrated to the noise it sees: the mean of the groups' z^2 must lie
below its null plus three standard deviations, and that standard deviation is sqrt((κ − 1) / ndf), κ the kurtosis
of the per-replica finite-difference values pooled over the family's groups (never of their differences from the
automatic derivative or of the groups' z, so a real bias cannot widen its own band; at least 3, the Gaussian value,
so a light-tailed family keeps the Pearson band). A finite difference whose sides change photons' paths has heavy-tailed replicas: on
ANNIE's two PMT photocathode transmittances (`dielectric_rt_sensor.T`, at their edge T = 0) the Gaussian band
(1.67 for 50 groups) failed one or the other in turn at the default photon count, χ² 1.86 and 1.87, while their
totals agreed (z +0.44 and −0.51) and both passed at 4x and 16x the photons. The record carries κ for every family.

The automatic derivative at an edge is drawn through the edge floor (`lumen.transport.estimator`): the outcomes the
edge closes are proposed with a fixed share of the draws (1 %), so the path they open is sampled. On a reflector the
baseline's share, 1 % of the absorbed fraction, was 0.1 % at a reflectivity of 0.9; the derivative's positive part then
came from a draw in a thousand at a large weight, and its replicas' spread understated the error (a closed unified box:
the automatic derivative sat one to three standard errors below the difference at every h, and agreed at a larger
share). The fixed share keeps the estimand and cuts that error about three times.

## Compiled programs and the persistent cache

lumen turns on JAX's persistent compilation cache by default (`<LUMEN_CACHE>/xla`, never under HOME; an explicit
`JAX_COMPILATION_CACHE_DIR` is respected). Bitwise checks rely on a program loaded from that cache computing exactly
what a fresh compile computes. On jax 0.5.3 it did not: the navigator's face misses on the CPU (the A5 probe, 16 of
20 processes) came from programs loaded from the cache on XLA's thunk runtime; with the cache off, no process missed.
On jax 0.6.2, the pinned version, a load from the cache is bitwise equal to a fresh compile on the CPU and the GPU
(a photon-loop record and leaf-module outputs, three separate processes: cache off, writing, loading). The GPU tier
checks it on every run (`tests/physics/test_persistent_cache.py`), and `tools/port/xla_runtime_probe.py check
--cache` checks the navigator on the CPU.

## References

Every reference product is listed with its sha256, its producer (binary, flags, seed, chunking) and any patch, in
`src/lumen/detectors/references.json`. The g4ref products in use carry a three-line patch
(`refs/g4ref/velocity_after_relocation.patch`): after a reflection at a relocation into an overlapping sibling volume
the photon speed is that of the volume it continues in. The patch changes no charge reference and moves only the
arrival times of overlap toys (the table below, z of lumen against Geant4's mean arrival time).

| overlap toy | z against stock g4ref | z against patched g4ref |
|---|---|---|
| partT_AB | −5.31 | −0.82 |
| partTm_AB | −4.63 | −1.08 |
| vx3_ACB | −6.03 | −0.23 |
| partT_BA | −1.38 | −0.93 |
| partTm_BA | −1.21 | −0.68 |
| vx3_ABC | −1.18 | +0.79 |
| xl_AS | +2.29 | +0.58 |
| xl_SA | +0.29 | +0.00 |

The other toys' times are unchanged by the patch.
