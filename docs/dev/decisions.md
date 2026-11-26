# Decisions

Why lumen is built the way it is: one short entry per decision, numbered in the order they were written down. Cite an
entry as "decisions.md, 7" or by its title. A decision that is replaced keeps its number and says what replaced it.

Each entry: **Decision** (what holds), **Why** (the reason, with the measurement when there is one), **Where** (the
code that carries it), and when useful **Revisit** (what would change it).

## 1. Geant4 semantics as general rules, never per-detector flags

**Decision.** Every Geant4 behaviour lumen reproduces is a general rule (`rules/`, `build/import_rules.py`), applied
to every detector, never a recipe switch. Literal Geant4 ports exist only as test references.
**Why.** A collaborator's GDML must behave as Geant4 runs it without anyone knowing which switch to set.
**Where.** `rules/navigation.py`, `rules/boundary.py`, `rules/optics.py`, `build/import_rules.py`.

## 2. Geometry is not a parameter; a value change never recompiles

**Decision.** Every value except the geometry is a traced parameter, and changing one never recompiles. This outranks
a 10 % speed loss.
**Why.** Calibration and reconstruction call `simulate` thousands of times with new values.
**Where.** `transport/step.py` (`Operands`), `run.py` (`ClosedOverValues`), `tests/physics/test_no_recompile.py`.

## 3. float32 physics, float64 references

**Decision.** The photon loop's physics runs in float32; float64 or closed forms appear only as test references. The
navigator's precision is decision 29 (float32 by default, float64 on request: the geometry and the photon's position,
never the physics), so this entry is about the physics alone.
**Why.** Speed and memory on GPUs; the known float32 limits are measured and listed
(`docs/explanation/known-limits.md`).
**Where.** `transport/` (the physics); `compare/oracle.py` is the float64 reference. The navigator: decision 29.

## 4. Fail closed

**Decision.** Anything lumen cannot treat as Geant4 would is refused, naming the object, the Geant4 rule and the fix.
**Why.** A silent approximation in a detector file is found, if ever, only after a physics result.
**Where.** The importer, the library, `rules/boundary.py` (`BoundaryError`); the messages are catalogued in
`docs/reference/errors.md`.

## 5. One import of tallymap

**Decision.** Only `run.py` imports tallymap; the physics never does.
**Why.** The photon-loop engine can change without touching the physics.
**Where.** `run.py`.

## 6. Plain words for users

**Decision.** Everything a user reads uses plain physics words, one meaning each (`docs/reference/vocabulary.md`).
**Why.** Users are physicists new to lumen, not readers of its code or its history.
**Where.** Style rule 10 (enforced by review); `docs/dev/vocabulary.md` maps the code's words to the user's.

## 7. jax pinned to 0.6.2

**Decision.** jax and jaxlib are pinned exactly, and tallymap by tag.
**Why.** On jax 0.5.3 a program loaded from the persistent compilation cache was not bitwise equal to a fresh compile
on the CPU (navigator face misses); on 0.6.2 it is, on the CPU and the GPU. tallymap 0.7.0 and 0.7.1 are tested on 0.6.2.
**Where.** `pyproject.toml` dependencies; `tests/physics/test_persistent_cache.py`;
`docs/dev/validation.md`.
**Revisit.** When tallymap supports a newer jax and the cache check passes on it.

## 8. Absorbed at entry, on meshes only

**Decision.** A mesh volume's material whose ABSLENGTH is below the band floor at every knot is bound opaque.
Elsewhere such a medium is tracked as Geant4 tracks it.
**Why.** Geant4 refracts the photon in and absorbs it within a length the navigator cannot resolve, and the navigator
does not track inside a mesh. Applied to every volume, the rule bound the PMT interiors' Air in SK and ANNIE, whose
builds then failed (measured).
**Where.** `build/import_rules.py`.

## 9. A WLS film is a volume the navigator resolves; a thinner one is refused

**Decision (0.2, user 2026-11-19).** A box of a WLS material thinner than the float32 band where it is placed is
refused at conversion; lumen lowers no film onto a surface (no layered surface, no `lumen:wls_film`). A recipe makes
such a film a volume: SBND's 23 nm TPB on its CPA foil becomes the front half of the foil (25.4 um of TPB, bulk WLS
with TPB's tables) on a 25.4 um vm2000 half that keeps the Surface_vm2000 skin (lumen-tpc, origin "recipe").
**Why.** Below the band two crossings are one. The split slab is G4OpWLS's own law on the film (TPB absorbs the VUV
within ~1e-14 m and is index-matched to LAr; d/L ~2.5e-7 in the visible; WLSTIMECONSTANT 0); it differs from the 23 nm
volume by at most 0.2 ps of flight, ~50 um x tan(theta) laterally and ~1e-4 of side-face entries, and the slab is
1.7 contact tolerances thick at a 4 m scale (tested: `tests/unit/test_navigator.py`). It removes the film lowering
from six modules, the layered kernel branch and the photon's layer state.
**Where.** `build/import_rules.py`, `rules.navigation.band`; lumen-tpc's SBND recipe.

## 10. The hit angle at the crossing point, in the sensor's own frame

**Decision.** The CE coordinate is θ = acos(z / r) of the crossing point on the sensor, in its placement frame.
**Why.** WCSim evaluates the formula at the step's start point in the glass's frame (WCSimWCSD.cc:75-84, 220-224), a
proxy off by the glass thickness: lumen's SK expected PE is 0.2-0.3 % below that proxy's. WCSim's point would make CE
depend on the photon's origin for a sensor placed straight in the detector volume, with a derivative singular at that
volume's centre.
**Where.** `transport/step.py` (`sensor_angle`).

## 11. One detection rule: the detection at normal incidence is the data's number

**Decision.** "Detected" is one quantity, the step's detect deposit `q`, for arrivals, the charge and time histograms
and the checks. A surface's EFFICIENCY means what its reader's reference code means: stock Geant4 detects EFFICIENCY of
the absorbed light. (GLG4's thin-film photocathode and the `lumen:thin_film` reader were deleted in 0.2: no detector
used them; either can return as an interaction model, `surfaces.types.Interaction`, whose detect deposit is
independent of its continuing shares.) A sensor
model that applies its QE after tracking (`lumen:sensor`) defines the detection itself: its QE is the datasheet QE, the
detection per photon at normal incidence; the sensor is read with every absorption counted, the deposit is the
physical absorbed share, and the sensor model's readout divides it by the cathode type's own A_n(λ) at the crossing's
indices (`surfaces.types.per_normal`; the deposit carries 1/A_n, `Deposit.inv_an`, and `readout.grid.Grid.normalise`
divides), refused where the QE exceeds A_n at the run's values (`build/values.py`). The surface's EFFICIENCY is then not read; where it is not 1 the build records so in the
entry's provenance (the eff row's note in `Parts.key_rows`), not as a warning. The analog photon of a truth run is
absorbed with the physical share and carries 1/A_n (`Analog.scale`), so its photoelectron probability is QE·CE/A_n.
**Why.** Before, arrivals counted the absorbed share while the charge counted it over A_n. One convention for the data: the data hold the measured, sourced number, and the cathode
model that introduces the absorption divides by it. lumen's SK expected PE stays 1/0.907 of WCSim's (0.75/0.68):
WCSimWCSD counts (1 − rgcff)·QE·4/3 of the photons reaching the cathode (WCSimWCSD.cc:209-214, rgcff 0.32 on SK, so
0.907·QE), lumen counts QE. Arrivals count the physical absorbed share on every sensor; only the sensor model's charge
and time histograms are over A_n. If a table's QE was measured in air it also holds the air-glass
Fresnel loss (about 4 %), which is not removed.
**Where.** the families' `sensor_values/<set>.json` and `tools/transcribe.py` (lumen-wc, lumen-tpc), `surfaces/types.py` (`per_normal`), `rules/optics.py` (`sensor`), `build/lookup.py` (`detection_margins`), `build/values.py`, `build/library.py`,
`transport/step.py` (the DETECT deposit).

## 12. The per-sensor χ²/ndf is tested against its own null (the 5 % floor is gone)

**Decision.** The charge check's per-sensor χ²/ndf passes below null + 3·sqrt(Var/ndf), the term's null mean and
variance those of a term whose variance is known; the reference's Poisson variance is the two samples' mean. It
replaced a floor, 1 + max(3·sqrt(2/ndf), 0.05). Lumen's variance was first estimated from 8 replicas, with a null
above 1 computed for it; it is now each sensor's per-photon second moment (N photons: Pearson's 1 and 2, decision 28).
**Why.** The floor was set so that SK's 1.042 (ndf 11,096; 3σ 1.040) passed. That excess is the statistic's own null:
lumen's variance is estimated from 8 replicas, and the estimate's noise in the denominator lifts the mean of a term
to about 1.03 (and lumen's own count in the denominator, correlated with the numerator, lifted it further at low
counts). The moment's own noise over N photons is negligible. Measured by Monte Carlo in
`tests/unit/test_validate.py`.
**Where.** `tools/stats.py`, `compare/charge.py`; `docs/explanation/validation.md`.

## 13. The gradient check's gates (replaced: the bands)

**Decision.** The gradient check runs `simulate` and gates three things per family on the same seeds: exactness, AD
against the fixed-reference Richardson difference per seed and output (`simulate(reference=)` holds every draw, so
the light at a seed is smooth); the value, the reweighted light at a moved value against a run there on disjoint
seeds; and production, AD at the default ε against AD at a tiny ε. Reverse mode is gated by the same exactness rule.
The edge floor still proposes an edge row's closed outcomes with a fixed 1 % share.
**Why.** The earlier check had 26 thresholds (the edge h-scan, the kurtosis-widened band, a median rule, a truncation
scan), each added for a case that failed, because a fixed-seed difference through `simulate` was not the derivative's
function (the roulette moved with the live values: 30 to 1800 times off; review R3). With the reference held, AD is
exact per seed and one rule replaces them.
**Where.** `compare/gradients.py`, `tools/stats.py`, `transport/estimator.py`.

## 14. Sibling holes are lowered out of the subtraction

**Decision.** A hole of a subtraction (A − B) that is also a sibling placement (the same solid at the same pose in
the same mother) is left out of A, and A is placed before those siblings, which own the holes (23).
**Why.** WCSim's black sheets subtract the PMTs that sit in them. The rewrite is exact: every point of the hole lies
in the sibling, which owns it whatever the volumes; a hole matching no sibling stays subtracted.
**Where.** `build/lowering.py` (`_Lowering.lowering`; always on).

## 15. A sensor group's value set is its sensor model

**Decision.** A sensor group's detecting surface behaves during tracking as stock Geant4 reads it (a `border_into`
group; an `any_entry` group reads none), and a group with a value set (`values`) is read after tracking by the sensor
model, `lumen:sensor`. There is no surface-reader or model name to give: each had one legal value. The efficiency on
entry of a group without a value set is its setting `entry_efficiency` (`sensors/<group>/entry_efficiency`), apart
from the value set's per-sensor efficiency after tracking (`sensors/<group>/efficiency`).
**Why.** Fields with one legal value, and one key with two meanings decided by another field, are removed.
**Where.** `format/sensorsets.py`, `rules/sensors.py`, `sensors/__init__.py`.

## 16. A GROUPVEL table times the photons

**Decision.** A material's GROUPVEL table times the photons in that material when the file has one: n_g = c /
GROUPVEL(E), linear in energy and clamped, as G4Track reads it, a parameter as any table. Without one, the photons are
timed by RINDEX's group index (Geant4's CalculateGROUPVEL). A constant GROUPVEL is inert, as in Geant4. (A table
within 0.2 % of RINDEX's used to be taken for an export and replaced by RINDEX's: removed, no tolerance.)
**Why.** Geant4 does it (0.1 kept the table inert, with a warning).
**Where.** `rules/optics.py` (`user_groupvel`), `media/laws.py` (`geant4_rindex_groupvel`), `build/library.py`.

## 17. The GDML reader follows Geant4; a repair is one explicit step

**Decision.** What stock Geant4 refuses (non-finite or denormal values, matrix tails, identical re-definitions, a
property naming no matrix) the reader refuses, naming the repair. `python -m lumen.gdml.repair` is the one repair:
its copy is converted (the provenance records its name and sha256) and given to Geant4. A property keeps the form
Geant4 stores (coldim 1 a constant), on materials and surfaces alike. (0.2 consolidation; it replaced repairs on
import with a warning.)
**Why.** A repair inside the reader made lumen read files Geant4 refuses, and invented values (inf as 1e27 m); a
separate, recorded copy keeps lumen and Geant4 reading the same bytes.
**Where.** `gdml/repair.py`, `gdml/reader.py`, `gdml/importer.py`; the public sources in the store's `gdml-repaired`.

## 18. Only `_env.py` reads the environment; no switches in the environment

**Decision.** The environment holds paths, caches and the device expectation, never a physics or behaviour switch.
**Why.** A result must not depend on a variable nobody recorded.
**Where.** `_env.py`; style rule 8 (ruff's banned API).

## 19. One opaque medium for every material without a refractive index

**Decision.** Every material the build reads as opaque (no RINDEX, or bound `lumen:opaque`) shares one medium,
"opaque"; the outside keeps its own. No recipe binds opaque materials itself.
**Why.** Geant4 kills a photon at any material without RINDEX, so which opaque material it is never changes the light;
separate media only split the surface lookup's classes. The SBND and MicroBooNE recipes bound their opaque materials
to one medium by hand; the general rule does it for every detector.
**Where.** `build/library.py` (`OPAQUE_MEDIUM`).

## 20. One detector structure; GDML and YAML are sources of it

**Decision.** `lumen.format`'s records are the detector, independent of where it came from, and `lumen.format`
imports nothing above it. The structural rules live with the records (`format/check.py`) and every input meets
them. YAML maps the records one to one, may be split across files and need not hold everything (defaults, partial
files combined); it is read at conversion only, and `load` keeps reading the JSON + npy directory. Rotations stay
Geant4's angles, a matrix accepted as input only.
**Why.** A detector written by hand or handed over should be checked as a GDML one is and give the same detector file,
so calibrations transfer; GDML specifics (pointer-suffixed names, source text) should not decide what a detector is.
YAML reads 25-30x slower than the JSON, so it is not a load format. Angles keep the GDML round trip exact.
**Where.** `format/check.py`, `format/yaml.py`, `tests/unit/builder.py`, `recipe.import_structure`; names without
Geant4's pointer suffix (`gdml/importer.py`); `convert` for GDML, YAML and a builder (`api.py`). (The GDML source block
that kept the source text for an ulp-exact round trip was removed in the 0.2 consolidation: the export writes the
stored doubles as bare literals.)

## 21. GDML names are stored without Geant4's pointer suffix where unique

**Decision.** A GDML name is stored without its `0x...` suffix when no other name of its kind (solids, volumes,
placements, materials with elements and isotopes, surfaces, skins, borders) strips to the same text; otherwise whole.
The export writes the stored names (Geant4 strips the suffix itself).
**Why.** Every Geant4 re-export changes the pointers, so parameter keys and calibrations would not survive one. Keeping
the full name on a collision keeps every name, and every border's physvol reference, unique.
**Where.** `gdml/importer.py` (`canonical_names`), `gdml/exporter.py`.
**Revisit.** Placement names repeat across WCSim's PMTs, so sensor paths keep the suffix; a per-mother rule with the
sibling `#k` segments would strip them (a border would then need its own exception).

## 22. Geant4's Rayleigh rule, a material named Water included

**Decision.** A material without RAYLEIGH gets the table G4OpRayleigh computes, derived at load like every other
derived value (`rules.optics.geant4_rayleigh`): a material named Water (its GDML pointer suffix stripped, as Geant4
strips it) uses Geant4's compressibility of water at 283.15 K whatever its own compressibility and temperature; any
other uses its ISOTHERMAL_COMPRESSIBILITY at its temperature; an explicit RAYLEIGH table wins over both.
**Why.** Geant4 is the reference (G4OpRayleigh.cc, 10.7.4 and 11.2: the Water branch and its 283.15 K). The earlier
rule wrote the table at import at the material's temperature (293.15 K): 3.4 % short of Geant4's (measured against a
Geant4 10.7.4 probe), and frozen, so a RINDEX calibration did not move it. Users who want another table give RAYLEIGH,
or ISOTHERMAL_COMPRESSIBILITY on a material not named Water.
**Where.** `rules/optics.py` (`geant4_rayleigh`, `rayleigh_table`).

## 23. A point has one owner, level by level: a solid proved inside its sibling, else the one placed first

**Decision.** A point's owner is found by descending from the world: at each level it belongs to the first sibling
containing it in the rule's order -- the placement order, each solid proved inside a sibling moved ahead of it. So a
solid lying inside its sibling owns the region they share (the more specific description), and where neither lies
inside the other the one placed first does; a daughter holds only what lies inside its mother. A tessellated solid
takes part like any other (nothing is proved inside it); a solid whose holes the import lowers out of a subtraction
(14) ranks after the siblings filling them, so the holes are theirs. "Proved" has one meaning
(`geometry/ownership.Proofs`): interval bounds of the CSG over bisected boxes, a convex leaf holding a box when it holds
its corners -- never a sample. The navigator is given no list of overlapping pairs: per volume, every sibling (of it
or of an ancestor) ahead of it in the rule's order whose box meets its box deeper than the contact tolerance and that
is not proved apart from it is a candidate, and every ancestor it is not proved inside (its box shrunk by the
tolerance, or the proof) is a candidate whose face is an exit; the containment tests decide every point at run time.
The rule's order is one integer per volume (the depth-first order with each mother's daughters in that order), and it
decides every tie: an entry at coincident faces, a relocation. The pieces of a split union are one volume (a crossing
between them is none).
**Why.** Geant4's own behaviour on an overlap (a photon in one sibling does not see the other; at its exit the
higher-index sibling takes the point) depends on the declaration order and on where the photon came from:
MicroBooNE's wire planes, 99.98 % inside their sibling TPCActive, are never seen from it, so 8,254 wires cast no
shadow. Descending from the world is Geant4's own locate (LocateGlobalPointAndSetup), so the rule is Geant4's wherever
placements neither overlap nor protrude, and Geant4 agrees with it on the GDML that writes it out
(`lumen.compare.ownership`, from the navigator's own lists). An earlier version ranked by volume, estimated for a
boolean from 65,536 Halton points: a 0.1 mm shell came out 8 % small and a rotated rod 3 % large (review R1), so
congruent booleans tied by noise; containment is decided exactly or not at all. On the five public detectors every
pair whose owner this changed (SK 11,096, WCTE 9,455, ANNIE 132, SBND 48, MicroBooNE 2) shares no point (20,000
points each), so no region changes hands.
**Cost.** A detector with overlaps or protrusions no longer matches Geant4 run on its raw GDML there; the conversion
says so (`PlacementWarning`, the placement report). Candidates that never overlap cost run time; the proofs cost build
time. A holed solid ranks after its fillers even where it
overlaps a sibling of a volume between theirs (none on the public detectors is known).
**Where.** `geometry/ownership.py`, `geometry/navigator_build.py`, `geometry/navigate.py`, `rules/navigation.py`,
`compare/oracle.py`, `build/placements.py`, `compare/ownership.py`.

## 24. A photon born inside a sensitive volume is counted there

**Decision.** An `any_entry` group declares a sensitive volume, and by default counts it as a Geant4 sensitive
detector does: every step in the volume, the first step of a photon emitted there included, so such a photon deposits
its entry's detection at its emission point and time and ends (`rules/sensors.counts_born`, `Optics.born_pair`, the
step's `_born`). The group's setting `count: "entry"` keeps the border convention (only crossings into the volume
count). A `border_into` group counts at its surfaces only (its detection is the surface's).
**Why.** lumen counted at borders only, so light emitted inside a sensitive volume (scintillation in a liquid-argon
sensitive volume) left uncounted where Geant4 and LArSoft count it (LArSoft's OpDetSensitiveDetector records any
optical step in the volume); another team measured Geant4/lumen = 1.253 on SBND's coated PMTs. One rule from the
declaration, no detector case.
**Where.** `format/sensorsets.py`, `rules/sensors.py`, `rules/entries.py`, `build/library.py`, `build/lookup.py`,
`transport/step.py`.

## 25. The roulette decides on the reference; one ε, a run knob; no time cut

**Decision.** A photon carries its weight at the reference values (`Photon.wr`, stop-gradient) beside its live weight;
the roulette reads only `wr` (kept with probability wr / ε). There is no time cut: every photon is followed and every
output counts all its light. Deposits carry the live weight. `simulate(reference=params0)` holds
every random choice at `params0`. ε is `Controls.roulette` (default 0.01 for every detector; 0.1 before 0.2), not a detector property.
The edge rows are the detector's (every row of a parameter with a domain), so `wrt` never changes the photons.
**Why.** A decision read from the live values moves with them, so a fixed-seed difference was not the derivative's
function, and a run with `wrt` traced other photons than one without (lumen-compare #3, #4, #6; review R3 C1, C2).
On the reference, the run is one function of the live values at a seed and its derivative is exact; ε sets the run
time and the noise, never the expectation, so it belongs to the run, outside the detector's digest (review A5).
**Cost.** A reference of its own evaluates each event twice on one query (+6 to +12 % time, transport_p3 study); the
always-on edge rows cost up to +5 % on value runs (WCTE).
**Where.** `transport/step.py` (`_event`, `_after`), `transport/state.py`, `api.py` (`simulate`, `_edge_rows`).

## 26. The charge check counts what the reference counts

**Decision.** A reference hit is an absorbed photon (`response` "absorbed", the default): lumen's deposit is the
physical absorbed share (11), so the check counts it as it is. With `response` "qe" the reference also applied its
sensors' QE, read from the detector's value sets. `compare/event.py` counts a deposit by the same function
(`charge.response`): one rule for every comparison with a reference's hits.
**Why.** WCSim's and g4ref's hits are absorptions, as lumen's deposit is; ANNIE's WCSim applies its QE after
absorption. (Until 0.2's deposit change the check multiplied by A_n, and `compare.event` kept doing so after it: SK's
g4ref events came out at 0.686 and 0.678 of Geant4's hits, A_n = 0.68.) One declared field replaces a
per-detector hook ("module:function") and the WCSim C++ parse it read.
**Where.** `compare/charge.py` (`response`), `compare/event.py`, `compare/references.json`.

## 27. The detector volume selects one placement

**Decision.** A recipe's `detector_volume` is a logical volume placed once, a physvol name, or a physvol path (names
from the world down, joined by "/"); a logical volume placed more than once is refused, naming its placements.
**Why.** The first placement was taken silently; which of several identical volumes is simulated is a decision.
**Where.** `format/schema.py` (`placements_of`), `build/lowering.py` (`find_detector`), `recipe.py`.

## 28. Errors from per-photon second moments, not replicas

**Decision.** The readout builds beside each charge cell and each sensor's arrival a per-photon second moment,
sum_i x_i^2 with x_i photon i's adds to the cell summed over its records before the square, in the same exact sums
(`readout.fill.Readout.moments`, `run.rebuild`); and each sensor's pe at the run's values ("pe2"). `arrivals_error` and
`readout.pe_error` are N / (N - 1) (sum x^2 - S^2 / N) under the square root (`grid.moment_error`). `simulate` has no
`replicas`; the charge histogram has no replica axis. The pe error at other values is not kept, nor a derivative's
error (seeds answer both). The checks read a diagnostic's moments (`compare.photons.Detected`), the totals from a
total's own cell.
**Why.** Photons are independent (a photon's weight, roulette and reference ratio are functions of its own key), so
the estimate is unbiased for i.i.d. rows; squaring each record instead of each photon's sum biases it low by 1-5 %.
On 64 seeds of 65,536 photons (SK, SBND) its scatter was 0.14-0.18 on bright cells against 0.41 for 4 replicas, at no
measured cost. A moment is diagonal (a sum of cells' moments under-states the sum's variance: SK's total 0.93 of it),
and for fixed rows (`FromArrays`) it includes the spread between the rows (SK's gate rows: 7.5 times the seed spread).
Launches still add in float32: `chunk` changes results at that rounding only.
**Where.** `readout/fill.py`, `readout/grid.py` (`moment_error`), `readout/detection.py` (`pe_error`), `run.py`,
`api.py`, `compare/photons.py`, `compare/charge.py`, `compare/overlaps.py`, `compare/event.py`, `tools/stats.py`.

## 29. The navigator's precision is a load option

**Proposed (0.2, the contributor's proposal of 2026-11-21; the user has not confirmed it).**
`lumen.load(..., precision="float64")` runs the navigator in float64: its coordinates and frames, CSG tables, roots
and intervals, its tolerances (the band, 8 eps64 |x| plus the 1 nm floor: the floor at any detector scale) and the
photon's position. The physics stays float32. float32 is the default. float64 needs JAX's 64-bit mode, on before any
array is made; `load` refuses it otherwise. Conversion is unchanged: decision 9's refusal stays at float32's band.
A recipe states the precision its detector needs (`Recipe(..., precision="float64")`), which is stored in the detector
file and is what `load` uses when the caller passes none, so a detector that needs float64 is not loaded wrong by
default; an explicit `precision=` wins over it, and float32 on such a detector warns once (`PrecisionWarning`).
**Why.** float32's band at a decision is that of the photon's ray, its start's largest coordinate plus the distance
travelled (`geometry/navigate.py`, `geometry/csg.py`), so a coating a few bands thick where it sits is tracked or not
depending on where each photon came from. Measured on a coating under a gas volume against exact float64 geometry,
coatings of 2.5 to 8 bands at 0.5 to 5 m were entered in 65 % of the configurations whose rays started 1.5 to 5 m
away; tightening the on-face tolerances moved that threshold, did not remove the path dependence, and stepped over
more coatings at grazing angles. NEXT-100's TPB (2 to 5.4 um) left its S2 light 8 % low at the PMTs and 10 % at the
SiPMs against nexus. In float64 every configuration is right and NEXT-100 agrees with nexus to 0.4 % (stage 3), at
+20 % (S1) and +19 % (S2) time against a float32 that tracks those coatings on an A100 (14.3 s and 13.8 s against 11.9 s
and 11.6 s), 3.4 times float32's S2 time on an RTX 2080 Ti (float64 at 1/32 of float32's rate). JAX's 64-bit mode alone
leaves a float32 run bit for bit.
**Why the navigator and not only the position.** Measured on the same geometry at 5 m: float32's band there is
4.8 um and the on-face tolerance 19 um, while a 3 um TPB coating is 0.63 band thick, and rays from 5 m away at 0 and
45 deg to its face are located in the backing (in float64, in the coating). The float32 crossing distance itself is
accurate to 0 ulps of the coordinate at normal incidence and 1 ulp at 75 deg, but its error grows with the ray's
length to the face: 15 ulps (7.2 um) at 89 deg and 54 (26 um, 8.6 coatings) at 89.9 deg. So no float32 tolerance both
resolves such a coating and still recognises the photon's own face at a grazing angle, and a float64 position with
float32 roots would not help: the roots, the intervals and the tolerances are what must be float64. Keeping the
position relative to a nearby frame would shrink the band with the local coordinates, but not the 1 / sin(theta)
growth, and it would have to reach every frame, source and deposit.
**Where.** `rules/navigation.py` (`Precision`, `band`), `geometry/navigator_build.py`, `geometry/csg_build.py`,
`geometry/ownership.py`, `geometry/cache.py`, `geometry/navigate.py`, `geometry/csg.py`, `transport/step.py`,
`transport/state.py`, `api.py` (`load`); `tests/unit/test_precision.py`.
**Revisit.** If a float32 navigator re-solves each crossing on its face, so that its band no longer depends on the
photon's path: float32 could then track such coatings on GPUs without fast float64.

## 30. A length of 0 is a value: the infinite rate Geant4 reads

**Decision.** The range of a length (`domains.LENGTH`, so `ABSLENGTH`, `RAYLEIGH`, `MIEHG`, `WLSABSLENGTH`) is
[0, inf), not (0, inf). A length of 0 is the infinite rate Geant4 gets from it, so a file's 0 knot is a value at the
load, in `Params.set` and in every run alike, and the gradient family multiplies such a table as any other (a 0 knot
scaled stays 0; the derivative comes from the other knots). A negative length is still refused.
**Why.** Geant4 reads `ABSLENGTH` as a mean free path (`G4OpAbsorption::GetMeanFreePath`) and
`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength` makes the step `n_left * mfp`, which is 0 for a mean free
path of 0: the photon is absorbed where it stands. `G4PhysicsVector::Value` holds a table's end value beyond its
knots, so a 0 at an end knot means that at every energy past it. Real files have such tables: Hyper-K's and IWCD's
mPMT gel (`materials/SilGel/ABSLENGTH`, WCSim e2dc7a4) is 0 at both ends. Measured on a toy box of a material whose
`ABSLENGTH` is 0 at 2 and 7 eV, 10^6 pencil photons over 500 mm: Geant4 detects 0 of 10^6 at a 0 knot and at both
energies outside the table, and lumen's detected fraction is 0 there too, with no non-finite value; at the seven
energies in between the ratio Geant4 / lumen is 0.998 to 1.018, every one within 1.5 sigma, which also shows both
interpolate the length and not the rate (at 2.1 eV, one tenth of the way from the 0 knot, the length is 100 mm and
0.69 % of the photons arrive, not none). The range excluding 0 refused that table in three different places and
agreed with Geant4 in none: the load admitted it only because its zeros fell outside that detector's light, a 0
inside the light refused the detector outright, and `Params.set` refused it always, which crashed the gradient check.
In a run the rate of a length of 0 is 1 / (the smallest normal float32), about 8.5e37 per metre (`bulk.rate`), not
1 / 0: it absorbs every path longer than about 1e-37 m, as Geant4's at-once absorption does, but a path of 0 still
attenuates by 1 and the derivative of a scaled 0 is 0, where inf gave inf * 0 = NaN (measured on that toy: the
table's forward derivative was NaN at 1.5, 2 and 8 eV with 1 / 0, and 0 with the finite rate).
**Where.** `_core/domains.py` (`LENGTH`), `media/bulk.py` (`rate`), `media/laws.py`, `tools/space.py`;
`tests/unit/test_laws.py`, `tests/unit/test_validate.py`.
