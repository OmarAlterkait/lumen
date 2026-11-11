# Known limits

Measured limits of lumen 0.1: where a value or a derivative is less accurate than float32 suggests, with the toy that
shows it and the reason it is accepted.

## The absorbing Fresnel reflectance's derivative in the absorption index, at small index

**What.** The reflectance R of a surface into an absorbing medium nr + i nk (a complex-index photocathode, a metal
read from REALRINDEX and IMAGINARYRINDEX) has a float32 derivative dR/dnk that is off by up to a few per cent where
nk is below about 1e-5, near the critical angle (n1 > nr), and at grazing incidence onto an index-matched absorber.
The value R itself is accurate to float32 there.

**Why.** The transmitted cosine squared, 1 - (n1 / n2)^2 (1 - cos_i^2), cancels from order 1 to order nk in those
regions, and the derivative goes as one over its square root. In complex64 its real part keeps an absolute rounding
error of about 1e-7, so at nk = 1e-6 its relative error is a few per cent. No rearrangement of the formula helps:
(nr^2 - n1^2) + n1^2 cos_i^2 cancels at the critical angle from the float32 inputs themselves. Away from those
regions the derivative is accurate to 1e-3 for every nk. At grazing incidence the derivative also goes to 0 as R goes
to 1, so its float32 value is an absolute ~1e-7 against a float64 ~1e-16: harmless.

**The toy** (the port's gate tooling, kept outside the repository) prints, per nk from 1e-6 to 1.5, how many of
815 (incidence, index) points have a float32 derivative more than 1e-3 from float64, away from and at grazing
incidence, and for each detector file the smallest IMAGINARYRINDEX or KINDEX of its surfaces. Measured (2026-11-09):
away from grazing incidence, 10 points above 1e-3 at nk = 1e-6 (largest 6 %), 7 at 1e-5 (0.4 %), none from 1e-4
up; at grazing incidence (cos_i <= 1e-3) the absolute error reaches 0.15 to 0.28 for nk <= 1e-4, where the derivative
itself is up to ~1e5, and falls to 6e-8 at nk = 1.5.

**Who is affected.** None of the public detectors (SK, WCTE, ANNIE, SBND, MicroBooNE) has a complex-index surface. A
detector with a photocathode or metal of nk below 1e-4 should not differentiate nk near the critical angle.

## ANNIE's LAPPDs are not sensors: the LAPPD glass's transmittance has a zero derivative

**What.** In lumen 0.1 ANNIE's sensor set reads its PMTs only (`lumen:pmt`); the LAPPDs are not sensors, because
their electronics (not a PMT model) come with the non-PMT sensor models of 0.2. A photon that enters an LAPPD's
interior (`InteriorWCLAPPD`) is therefore never detected. The LAPPD glass face's surface
(`GlassCathodeSurface`, a DIELECTRIC_RT with TRANSMITTANCE 0) has a T family whose derivative is exactly 0:
T only sends photons into that interior. The gradient check reports it as "zero" (`lumen.validate.gradients`,
`judge_zero`): its AD is exactly 0 although the edge rows propose the transmitted outcome, and it is gated on the
FD and FDX totals being consistent with 0. That is the physics of this sensor set, not the "edge rows missing" bug
(an exactly-zero AD with the outcome never proposed, which the check still fails).

**The measurement** (the port's gate tooling, 2^20 photons): of 2^20 photons from the
gradient check's sources, 93,382 enter an LAPPD interior at T = 0 and 127,019 at T = 1 - R = 0.68; the charge they
deposit afterwards is exactly 0 in both, with no lost weight. With the edge rows on, the transmitted outcome is
drawn 481 times in 92,740 crossings of that surface (2^19 photons) and the AD stays exactly 0; the FD is
-7.0e-4 +- 4.4e-4 per photon. `tests/physics/test_edge_rows.py` checks the same estimator on a toy with a closed
form (dQ/dT = exp(-0.75 / L) per photon at T = 0+).

**Who is affected.** ANNIE until its LAPPDs are sensors (0.2); any detector whose transmitting surface leads only into
volumes that are not sensors and from which light does not return.

## A full detector on the CPU

Without a GPU, lumen runs the photon loop on tallymap's plain-JAX CPU backend. There, building and compiling a full
detector's loop (SBND-sized: 312 sensors) took over 30 minutes before the first photon ran (a 30-minute job at 2^10
photons did not finish its first simulate). The CPU backend is meant for toys and tests; simulate full detectors on a
GPU.

## The moving-source method (derivatives in the source position)

A source-position derivative moves every light path with the source while its random numbers stay fixed. Three
situations are outside what it resolves well, and the tutorial's check against a finite difference is the way to see
it on your detector:

- **The shadow edge** between a source and an object close to it: a path that grazes the object's edge switches between
  hitting and missing it as the source moves, and that switch has no derivative along the path.
- **Light focused behind a refracting PMT window:** the paths converge, the pinned-end Jacobian becomes small, and the
  derivative's variance grows.
- **A source a few cm from a reflector:** the co-moving region around the source is limited by the distance to the
  nearest surface, so the derivative is noisier there.

A photon that bounces many times between convex windows makes its path's sensitivities grow exponentially; lumen
rescales them at every bounce, which changes no value or derivative.

## A user GROUPVEL

Geant4 times photons in a material with a user-supplied GROUPVEL table (it replaces the one Geant4 computes from
RINDEX). lumen 0.1 does not read it: photons are timed by RINDEX's group index, with a warning. The counts are Geant4's;
the arrival times in that material are not. A GROUPVEL that a GDML export wrote (Geant4's own, from RINDEX) agrees with
lumen's and gives no warning.

## The shipped SK value set's collection efficiency

`lumen:sk_r3600` carries a collection efficiency against the hit position's angle on the PMT: WCSim's 2011 table,
which WCSim replaced by a flat 100 % in 2015. Its source and its coordinate are not documented, so it is marked
unverified. If you need an absolute collection efficiency for SK, supply your own value set.

## Lost light at a junction of touching volumes

Where a medium without RINDEX (a steel reflector, say) touches a transparent volume (a PMT's glass), a photon leaving
the glass at the junction is relocated into the steel, as Geant4's navigator does, and the steel's skin surface
reflects it. Its next step, back into the glass, starts in a medium without RINDEX. lumen counts its weight as `lost`
in `Light.health`, where stock Geant4 kills the photon (G4OpBoundaryProcess, NoRINDEX). Measured on a collaboration
detector with 43,212 PMTs (a point at the centre, 420 nm): about 1e-4 of the weight (110.6 of 10^6 photons at 2^20
samples), nearly all at the PMT reflector against its glass. Physically the reflected photon continues in the glass;
lumen will follow it there in 0.2, and is low by about that 1e-4 until then. `lumen.diagnostics.Trace` shows the pair
of volumes for a lost photon (docs/reference/errors.md, Health).
