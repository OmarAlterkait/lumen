# Known limits

Measured limits of lumen 0.1.x: where a value or a derivative is less accurate than float32 suggests, with the toy that
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

**How large.** Away from grazing incidence, at nk = 1e-6 the largest error is 6 %, at 1e-5 0.4 %, and from 1e-4 up
below 1e-3. The measurement is in [the developer notes on validation](../dev/validation.md#the-absorbing-fresnel-derivative).

**Who is affected.** None of the public detectors (SK, WCTE, ANNIE, SBND, MicroBooNE) has a complex-index surface. A
detector with a photocathode or metal of nk below 1e-4 should not differentiate nk near the critical angle.

## ANNIE's LAPPDs are not sensors in 0.1

ANNIE's sensor set reads its PMTs only. LAPPD electronics come with the non-PMT sensor models of 0.2. Light that enters
an LAPPD is therefore never detected. As a result, the derivative with respect to the LAPPD glass's TRANSMITTANCE
(`surfaces/GlassCathodeSurface/TRANSMITTANCE`) is exactly 0: more transmission only sends light where nothing counts
it. The gradient check reports this parameter as "zero". This is correct for this sensor set, not a bug.

**Who is affected.** ANNIE until 0.2; any detector whose transmitting surface leads only into volumes that are not
sensors and from which light does not return.

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

On a very large detector (about 40,000 sensors) a position derivative can run out of GPU memory before tracking, in the
step that estimates the clearance about the source (`lumen.geometry.clearance`, 256 rays per batch). Smaller batches,
sized to the free memory, are planned for 0.1.3.

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
