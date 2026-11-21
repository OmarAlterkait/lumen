# How the validation checks decide

`python -m lumen.compare` writes one record per quantity (value, error, z, pass) and the report is generated from them. This
page states the pass rules that are choices rather than plain statistics.

## Charge

- **Total charge.** The ratio of lumen's detected charge to the reference's, with both Monte Carlo errors, passes at
  |z| ≤ 3.
- **Per-sensor agreement.** The Pearson χ²/ndf over the sensors expecting at least 5 hits passes when it is within
  three standard deviations above its null:

      χ²/ndf ≤ null + 3·√(Var/ndf)

  Each sensor's term is (lumen − reference)² over the expected hits (the two samples' mean) plus lumen's Monte Carlo
  variance. That variance is each sensor's per-photon second moment (the sum over lumen's photons of the square of
  what each added to the sensor), known to the precision of its many photons, so the null and Var are Pearson's 1 and
  2. (An earlier estimate from 8 replicas put the null near 1.03: its noise sat in the denominator.)

## Gradients

Each derivative is checked through `simulate` on the same seeds three ways: against a finite difference of the
light with the reference fixed (`simulate(reference=)`), which must agree exactly; the light reweighted from the
reference against the light run at the new values, which must agree within the Monte Carlo error; and the derivative
at the default roulette against one at a tiny roulette, likewise. At the edge of a range (a probability at 0 or 1)
the finite difference is one-sided. Reverse mode is compared with the same finite difference. The detailed pass
rules and why they were chosen are in [the developer notes on validation](../dev/validation.md).

## References

lumen follows stock Geant4 10.7.4, run by g4ref. One rule holds for every detector: the g4ref reference that gates
runs on the detector's overlap-free GDML, which `python -m lumen.compare export DETECTOR.lumen OUT.gdml` writes
from the detector file. lumen resolves overlapping volumes statically (a volume proved inside a sibling owns the region
they share, else the one placed first) and the export writes that resolution in Geant4's terms, so Geant4 navigates
the geometry lumen simulates. It rewrites only what really overlaps, the pairs and protrusions the conversion's
placement check reports (a contact within the tolerance is none); everything else is the detector file's own solids
and placements, so a detector without overlaps exports as it is. On a GDML whose volumes really overlap, Geant4's answer depends on the path each photon
takes, so the same sources run on the original GDML are reported beside the gate, never gated: the charge check shows
lumen against Geant4 on the original GDML, and Geant4 on the export against Geant4 on the original (the overlap
effect), and the geometry check probes both files. An
export product names its original run (`producer.export.original` in the manifest). MicroBooNE's wire planes overlap
`volTPCActive`: Geant4 on the export sees 0.784 of the light Geant4 on the original GDML sees. On a detector without real overlaps the ratio is consistent
with 1.

Any other simulation of a detector (the experiment's own software on another Geant4 build) is reported beside a
g4ref reference on the export as a cross-check, and gates where the detector has none (SK, the Hyper-K
configurations). ANNIE's WCSim runs on Geant4 10.1.3, which loses 7 to 8 % of the light reflected off the
inner-structure frame where g4ref and lumen agree (a navigation difference of that Geant4 build, measured with a
minimal Geant4 application): lumen's charge is 1.076 times WCSim's there. lumen 0.1.3 reported 1.003 because its
manifest scaled the frame's reflectivity by 0.8182 to match that loss; no reference is scaled to agree now. A
detector outside lumen's families, a private one included, uses the same commands and rule with its own manifest.

Every reference product is listed with its sha256, its producer (binary, flags, seed, chunking) and any patch, in
`src/lumen/compare/references.json`. The g4ref products in use carry a three-line patch
(`tools/g4ref/velocity_after_relocation.patch`): after a reflection at a relocation into an overlapping sibling volume
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
