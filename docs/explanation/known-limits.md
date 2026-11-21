# Known limits

Measured limits of lumen: where a value or a derivative is less accurate than float32 suggests, with the toy that
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

## A full detector on the CPU

Without a GPU, lumen compiles the photon loop for the CPU with the system C++ compiler and OpenMP (tallymap 0.7). A
full detector is usable there; it needs a C++ compiler with OpenMP (g++), and its speed depends on the number of cores.
Measured on SBND (312 sensors), a point source and 2^18 photons, forward only, lumen with tallymap 0.7.0:

| | 48 cores of an AMD EPYC 7542 | one NVIDIA RTX 2080 Ti |
|---|---|---|
| first process, empty caches: load + first `simulate` | 37 s + 36 s (20 s of it compiling the loop) | 57 s + 50 s |
| a later process, caches kept: first `simulate` | 11 s | 15-16 s |
| each further `simulate` | 2.0-2.2 s | 1.8-1.9 s |

The CPU's results agree with the GPU's within the Monte Carlo error, not bit for bit: on SBND and Super-Kamiokande, 16
independent seeds of 65,536 photons on each side, the totals of the arrivals and of three derivatives differ by at most
1.1 standard errors, and 20 sensor groups give chi2/ndf at most 21.5/20. Larger photon counts, reverse-mode
derivatives and detectors with tens of thousands of sensors were not timed on the CPU.

`backend="xla"` is tallymap's plain-JAX backend, which needs no compiler: on it, building and compiling SBND's loop
took over 30 minutes before the first photon ran. Use it for toys and tests only.

## A NaN in the exact sums

The photon loop's sums are exact: each cell is the correctly rounded float32 of the exact sum of what was added to it,
in the value, every tangent and the reverse cotangents, whatever the order. A cell is exact or NaN: a non-finite value
added to it (a quantity singular at the point, its derivative first), or a sum beyond the accumulator's top (about
2^166), makes it NaN, and `Light.health.nonfinite` counts such cells (under `jax.jit` too). On the `jax.grad` /
`jax.linearize` path the derivative's sums run after `simulate` returns, so their NaN cells show only in the
derivative: check a gradient with `jnp.isfinite`. The chunks' results are still added in float32 (several chunks:
`samples` above `chunk`), so a run of several chunks differs from one launch by up to about 3e-7 of a cell.
The default `chunk` is sized from the device's memory, so the same run on devices of different memory differs at that
level (pass `chunk` to fix it).

**Who is affected.** A derivative at a point where the quantity is singular; a comparison of runs with different chunk
sizes, or with the default chunk on devices of different memory, at the 1e-7 level.

## jax_enable_x64

Every result is bitwise reproducible for a given `jax_enable_x64` setting, and turning it on changes no photon: the
photon loop and the exact sums are float32 either way. Measured on SK and SBND (2048 photons, the CUDA backend, x64
off vs on): every histogram, the arrivals and their errors, the forward and reverse derivatives in ABSLENGTH and
RAYLEIGH are bit for bit the same. The one difference is a group's `pe_error`, which follows the dtype of the sensor
values it is weighted by: float64 under x64 (the value sets' tables are float64), equal to the float32 one within 1.4e-7
relative (its rounding).

**Who is affected.** Code comparing `pe_error` arrays across the two settings bit for bit, or expecting float32. A
detector loaded with `precision="float64"` needs the setting on (`lumen.load` refuses it otherwise); with it on, a
float32 detector's photons are unchanged (NEXT-100, measured).

## The roulette's ε

A photon's weight is the number of real photons it stands for: its source's photons over the Monte Carlo samples, 1
when each sample is one real photon. As it reflects and is absorbed in part, lumen carries that weight down rather than
ending the photon. Below weight ε it stops carrying a fractional weight and plays Geant4's survive-or-absorb lottery:
it survives with probability (its weight) / ε and then carries weight ε, else it is absorbed (`Controls.roulette`;
the only way a weight ends a photon). With photons of weight 1, ε = 1 is analog transport, as Geant4 does it, and
ε → 0 is fully weighted. The test reads the weight itself, not its share of the start weight: a photon that starts
above 1 (more real photons than samples) is weighted until its weight falls below ε, whatever ε. The weight the
lottery reads is the reference weight (the weight had the live values been the reference: `simulate(reference=)`, by
default the run's own values). The result is unbiased at any ε > 0, for the light and its derivatives; ε trades run
time against noise. The default is 0.1. Deciding on the reference weight keeps the decision out of
the live values: with the reference fixed, the light at a seed is a smooth function of them, so a fixed-reference
finite difference is the derivative `simulate` gives, at the same ε. One ε then serves values, derivatives, fits and
checks.

Among rules that look at the weight alone, p = w / ε is the most efficient form (variance × time), and the best ε for
a quantity q satisfies ε² = v / (τ ȳ): v is q's variance per photon, τ the mean vertices per photon, and ȳ the second
moment per unit weight² of the steps the roulette acts on. In words: at the best ε, the share of q's second moment
made below ε equals the share of the vertices spent there. That depends on the detector and on the quantity, so no
constant is best everywhere. ε is a run knob, not a detector property: the default, 0.01 (`lumen.Controls().roulette`),
is the same for every detector, at or below every detector's best ε for derivatives in the table below. Measured
with lumen 0.2 (2^21 photons drawn from each detector's validation photons, 2^22 for SK; efficiency the worst of the
per-sensor median, 10th percentile, lowest-occupancy decile and total), the best ε per detector:

| detector | value: best ε | derivatives: best ε | the derivative that binds |
|---|---|---|---|
| SK | 0.5 | 0.05 | the black sheet's REFLECTIVITY |
| WCTE | 0.5 | 0.03 | the black sheet's REFLECTIVITY |
| ANNIE | 1 | 0.15-0.2 | the cathode glass's REFLECTIVITY |
| SBND | 0.3 | 0.02 | a steel surface's REFLECTIVITY |
| MicroBooNE | 0.5 | 0.3 | LAr ABSLENGTH (no surface key) |

A surface-reflectivity derivative wants a lower ε where its light is carried at low weight (the reflection off a
mostly absorbing surface: SK, WCTE, SBND); the value wants a higher one. Where a run is long and far from its best, set
`controls=lumen.Controls(roulette=...)`; keep the same value for the runs you compare (a finite difference, a fit's
values and its gradient).

**Who is affected.** Run time and noise only.

## Derivatives in the source position

A derivative that moves where the light starts (a source's position or direction, a photon list's positions,
directions or wavelengths, a source function's parameter declared `method="moving_paths"`) is not available in this
version: a tangent on one raises `GradientNotAvailable` (`... moves where the light starts: a source-position derivative
is not available in this version`). Moving the source with the random numbers fixed moves every light path, and the
plain derivative of a Monte Carlo sample misses the terms where a sliding path crosses an edge (a sensor's rim, a
shadow, the switch between scattering first and reaching a wall first). Use a finite difference of runs at the two
positions with the same seed. A wavelength derivative (`source/lam` of a `Point`, and RINDEX) is refused for the same
reason: a refraction vertex slides as the index changes (in a slab toy a pathwise derivative missed 19 % of the
wavelength derivative, from paths carrying 4 % of the light).

## A reference far from the live values

Every random choice is drawn at the reference values and weighted to the live ones (`simulate(reference=)`; by
default the reference is the run's own values, where none of this applies). The weight is unbiased wherever the
reference draw can reach every outcome the live values can. Where a parameter can zero a probability, lumen's one
defensive mixture covers it (`DEFENSIVE_SHARE` of the draws from a covering density). Four cases are covered only
near the reference, so a run whose live values move far from a held reference needs a reference closer to them:

- **GLISUR polish moved down (a wider facet ball).** The facet is drawn from the reference ball, mixed with its shell
  (radius 1 - polish at the reference). That covers the derivative at the reference, but not facets outside the
  reference ball that a smaller live polish reaches.
- **An index moved at a GLISUR dielectric surface (facet refraction).** A refraction drawn at the reference implies,
  at the live indices, another facet; the live refractions whose facets lie outside the reference ball are never
  drawn. The facet density vanishes at the ball's edge, so the light missed grows as the move to the power 1.5 and a
  finite difference's bias vanishes with its step (at polish 0.6, gas into an index of 1.5 at 50 degrees, raising the
  index 3 % misses 1.6 % of the light transmitted).
- **A photocathode's straight share with only its complex index keyed.** Where the reflectance from (nr, nk) reaches
  1 (no absorption, total reflection), the straight share is 0 at the reference and not covered. It is covered
  whenever TRANSMITTANCE or the efficiency is keyed too.
- **A unified facet width much larger than the reference's.** The float32 tilt draw reaches at most about 5.8 times
  the reference width, so a much wider live lobe is not reached.

A fit that moves such a value far should redraw at the new values (a reference per iteration, the default).

## Light lost where an opaque volume overlaps a transparent one

**What.** Where a volume of a medium without RINDEX (steel, say) overlaps a transparent one, a photon leaving the
transparent volume at a point the opaque one holds is located in the opaque one, as Geant4's navigator does, and its
skin surface reflects or absorbs it. A reflected photon is then inside the opaque volume (Geant4 keeps a reflected
photon in the volume it was located in), travels through it and leaves it with no surface between a medium without
RINDEX and the next: lumen counts its weight as `lost` in `Light.health`. Stock Geant4 kills the same photon there
(G4OpBoundaryProcess, NoRINDEX: 10.7.4 `G4OpBoundaryProcess.cc` lines 260-286), so the light is the same as
Geant4's; the cause is the detector's overlap. Where the two volumes only touch, the step back has zero length:
Geant4 continues the photon (StepTooSmall, lines 189-195, tested before NoRINDEX), and so does lumen.

**Measured** (toys against Geant4 10.7.4, 2e5 photons each: glass with a polished R = 1 steel reflector touching or
overlapping it): touching plane, cylinder and sphere faces and a 1 um water gap lose nothing in either code
(lumen / Geant4 0.9993-1.0009); a 0.1 mm and a 1 um overlap lose 0.63 of the light in both (lumen / Geant4
1.0011-1.0048, |z| < 1.1).

**The limit.** An overlap thinner than lumen's float32 band (about 1e-6 of the coordinates: 0.1 um at 0.1 m; 1 nm in
float64, `lumen.load(..., precision="float64")`) reads as touching: lumen continues the photon where stock Geant4 kills it (a 1 nm overlap: lumen 1.78 times Geant4 on the toy).
Such an overlap is a geometry defect; `lumen.compare.geometry` and Geant4's overlap check find it.

**Who is affected.** Overlapping siblings no longer lose light this way: the region they share has one owner (the
next section), so MicroBooNE's end caps, which overlap its TPC, are entered at their own face (65,536 photons from
its active volume: a lost weight of 1.5 before, 0 since). What remains is a volume protruding from its mother, or a
mesh. Measured on 2^18 of the shipped detectors' test photons (one seed, before ownership): 3.2 on ANNIE (its LAPPDs overlap the tessellated frame: a photon leaving an LAPPD is inside the frame, which
holds no point, so the next crossing leaves the frame's medium). On a collaboration detector with 43,212 PMTs (a point
at the centre, 420 nm, 2^20 samples) 1.0e-4 of the emitted weight is lost this way, on steps of 1e-5 to 1e-1 m inside
the opaque volume: real overlaps, which stock Geant4 kills too. `lumen.diagnostics.Trace` shows the pair of volumes
for a lost photon (docs/reference/errors.md, Health).

## Thin layers

**What.** The navigator decides at a face within its band, 8 ulps of its precision at a coordinate scale
(`lumen.rules.navigation.band`). In float32, the default, the band at a decision is that of the photon's ray: its
start's largest coordinate plus the distance travelled, 9.5e-7 per metre. A coating a few bands thick where it sits
can still be stepped over, its far face taken for the near one, or a photon re-emitted in it put in the volume
behind it, by photons whose rays started far away: whether a coating is tracked depends on each photon's path.

**Measured** (the geometry of `tests/unit/test_precision.py`, against exact float64 geometry: a coating on a backing
block, its top face shared with a gas volume; 288 configurations of height 0.05-5 m, thickness 0.75-8 float32 bands
where it sits, path 1 mm-5 m, angle 0-75 deg, 48 rays each). In float32, coatings of at least 2.5 bands at 0.5 m or
more are entered in every configuration whose rays started at most 0.3 m away, and in 65 % of those from 1.5 to 5 m;
whether a photon is located inside follows step * cos(theta) < T, with step the band at the photon's path, in 214 of
216 configurations. A photon re-emitted at 10 % of the depth heading out at 3 deg is put in the backing in every
configuration. A coating standing in the gas is entered in 71 % of the configurations from 1.5 to 5 m away. NEXT-100's
TPB (2 to 5.4 um, 2.5 to 8.5 bands where it sits) left its S2 light 8 % low at the PMTs and 10 % at the SiPMs against
nexus. In float64 every configuration is right but those with a 36 nm coating and a re-emission 3.6 nm from its face,
and NEXT-100 agrees with nexus to 0.4 %.

**The limit.** In float32, a volume a few bands thick at the scale of the photons' paths (4 bands: 3.8 um per metre
of start coordinate plus path) is not tracked for every photon. Load a detector with such coatings in float64
(`lumen.load(..., precision="float64")`): its band is its 1 nm floor at any detector scale, so volumes down to tens of
nm are tracked, at about +20 to 27 % time on an A100 and 3.4 times on an RTX 2080 Ti (docs/dev/decisions.md, 29). The
conversion still refuses a WLS box thinner than float32's band where it sits (decision 9).

## The hit likelihood's no-record integral

**What.** `hits_nll` integrates the expected light times its recorded share r over each stretch without a hit (a
sensor with no hit, the gap before a first hit, the gaps between hits), Lambda_r. The share is taken per node mass, in
effect a midpoint rule in u, so Lambda_r carries a quadrature error that the time grid's cubic-split bound
(`lumen.readout.grid.node_spacing`) does not cover.

**Measured** (the readout test's toy: 20 sensors, Gaussian TTS 0.8 ns, a record window of 200 ns, events drawn from a
grid 8 times finer): the grid and the fine grid's Lambda_r differ by -111 to +96 ppm per sensor, and an event's NLL by
up to about 1e-3, about 10 times the error the cubic split alone gives. Not measured on a full detector.

**The limit.** Hit likelihoods, and fits on them, carry this error where a window goes unrecorded. Integrating r over
each piece of the piecewise-linear node mass would remove it; that changes likelihood values and is not in 0.2.

## Overlapping or protruding placements: a point has one owner

**What.** Geant4 takes sibling volumes as disjoint and daughters as inside their mothers; where a detector's geometry
breaks either, Geant4's result depends on the order of the placements and on where the photon came from (a photon in
one sibling never sees the other). lumen gives every point one owner instead: descending from the world, at each level
a sibling proved to lie inside the other (from the solids' parameters) owns the region they share, else the one placed
first, with its daughters, and a daughter ends at its mother's boundary. A tessellated solid follows
the same rule. A contact (a face shared within the float32 tolerance) is neither an overlap nor a protrusion. The
conversion's report names each overlapping family with its owner and each daughter kept inside its mother, and warns
(`PlacementWarning`); the geometry is kept as written. There lumen differs from Geant4 run on that GDML, and agrees
with Geant4 run on the GDML `lumen.compare.ownership.export` writes, which says the same in Geant4's terms (a
non-owner less its owners, a protruding daughter within its mother).

**Who is affected.** MicroBooNE's public GDML places its three wire planes as siblings of `volTPCActive`, inside it
but for 1 mm at each end in z: the planes own that region, so their 8,254 wires are seen from the active volume. Of
100,000 rays uniform in `volTPCActive`, isotropic, 4,501 meet a wire first (a wire solid lies before Geant4's first
opaque volume on 4,503 of the same rays), and the light reaching the PMTs from a uniform 128 nm source falls by 22 %.
So lumen differs from Geant4 on MicroBooNE's own GDML by 0.784 in charge; the charge check gates against Geant4 on the
overlap-free GDML, where it agrees, and reports Geant4 on the two files, which differ by the same 0.784
(validation.md, References). The geometry check likewise gates on the export (every point agrees) and reports the
original, where the planes' points are Titanium for lumen and LAr for Geant4.
The PMT lens owns the coincident underside, the end caps own their overlaps with the TPC, and the
TPC and the end caps, which stick 8 to 19 cm out of the cryostat, end at the cryostat's wall. Geant4 10.7.4's
CheckOverlaps finds nothing on the overlap-free GDML, and on 100,000 rays from the TPC's box its first volume is
lumen's on 99,994. SBND's ullage owns its overlap with the field cage. ANNIE's holders, LAPPDs and two black sheets
own their overlaps with the tessellated frame (CheckOverlaps: nothing on the export; 99,995 of 100,000 rays through
the frame region agree).
