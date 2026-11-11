# How the validation checks decide

`lumen validate` writes one record per quantity (value, error, z, pass) and the report is generated from them. This
page states the pass rules that are choices rather than plain statistics.

## Charge

- **Total charge.** The ratio of lumen's detected charge to the reference's, with both Monte Carlo errors, passes at
  |z| ≤ 3.
- **Per-sensor agreement.** The Pearson χ²/ndf over the sensors expecting at least 5 hits passes when

      χ²/ndf ≤ 1 + max(3·√(2/ndf), 0.05)

  that is, three standard deviations of χ²/ndf, but never tighter than a 5 % excess. With thousands of sensors the
  pure 3σ band is narrower than any per-sensor model difference worth acting on: an SK comparison had χ²/ndf = 1.042
  at ndf = 11,096, where 3σ allows only 1.040, while its total charge agreed to 0.1 %.

## Gradients

Each derivative is compared with a finite difference taken on the same random numbers. A derivative below float32
resolution is reported but not graded. At the edge of a range (a probability at 0 or 1) the finite difference is
one-sided. Reverse mode is compared with forward mode. The detailed pass bands and why they were chosen are in
[the developer notes on validation](../dev/validation.md).

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
