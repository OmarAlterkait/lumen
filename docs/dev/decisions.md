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

## 3. float32 compute, float64 references

**Decision.** The photon loop runs in float32; float64 or closed forms appear only as test references.
**Why.** Speed and memory on GPUs; the known float32 limits are measured and listed
(`docs/explanation/known-limits.md`).
**Where.** Everywhere in `transport/`, `geometry/`; `validate/oracle.py` is the float64 reference.

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
**Where.** Style rule 10, `tools/hooks/jargon.py`; `docs/dev/vocabulary.md` maps the code's words to the user's.

## 7. jax pinned to 0.6.2

**Decision.** jax and jaxlib are pinned exactly, and tallymap by tag.
**Why.** On jax 0.5.3 a program loaded from the persistent compilation cache was not bitwise equal to a fresh compile
on the CPU (navigator face misses); on 0.6.2 it is, on the CPU and the GPU. tallymap 0.6.0 is tested on 0.6.2.
**Where.** `pyproject.toml` dependencies; `_env.xla_cache`; `tests/physics/test_persistent_cache.py`;
`docs/dev/validation.md`.
**Revisit.** When tallymap supports a newer jax and the cache check passes on it.

## 8. Absorbed at entry, on meshes only

**Decision.** A mesh volume's material whose ABSLENGTH is below the band floor at every knot is bound opaque.
Elsewhere such a medium is tracked as Geant4 tracks it.
**Why.** Geant4 refracts the photon in and absorbs it within a length the navigator cannot resolve, and the navigator
does not track inside a mesh. Applied to every volume, the rule bound the PMT interiors' Air in SK and ANNIE, whose
builds then failed (measured).
**Where.** `build/import_rules.py`.

## 9. One-sided WLS films, by the thinness band

**Decision.** A box of a WLS material with no surface of its own is a zero-thickness film when its smallest dimension
is below the float32 band at its first placement; a thicker box is a volume.
**Why.** Below the band two crossings are one, so the film cannot be tracked as a volume. Because the band grows with
the distance from the origin, the classification can depend on placement: a 20 nm film is a film anywhere, a 1 mm
slab never is.
**Where.** `build/import_rules.py`, `rules.navigation.band`.

## 10. The hit angle at the crossing point, in the sensor's own frame

**Decision.** The CE coordinate is θ = acos(z / r) of the crossing point on the sensor, in its placement frame.
**Why.** WCSim evaluates the formula at the step's start point in the glass's frame (WCSimWCSD.cc:75-84, 220-224), a
proxy off by the glass thickness: lumen's SK expected PE is 0.2-0.3 % below that proxy's. WCSim's point would make CE
depend on the photon's origin for a sensor placed straight in the detector volume, with a derivative singular at that
volume's centre.
**Where.** `transport/step.py` (`sensor_angle`).

## 11. QE per absorbed photon (value sets @2)

**Decision.** A shipped PMT value set's QE is per photon absorbed at the cathode: WCSim's table divided by the
cathode absorptance at normal incidence that lumen tracks (0.68 for SK's 20-inch PMT). The @1 sets stay loadable.
**Why.** lumen tracks the window and cathode reflection itself; WCSim's QE already includes it, so @1 counted that
loss twice. lumen's SK expected PE is 1/0.907 of WCSim's (0.75 / 0.68). If a table's QE was measured in air it also
holds the air-glass Fresnel loss (about 4 %), which is not removed.
**Where.** `sensors/data/*@2.json`, `sensors/pmt.py`; `tests/unit/test_sensor_sets.py`.

## 12. The per-sensor χ²/ndf band has a 5 % floor

**Decision.** The charge check's per-sensor χ²/ndf passes below 1 + max(3·sqrt(2/ndf), 0.05).
**Why.** With thousands of sensors the pure 3σ band is narrower than any per-sensor model difference worth acting on
(SK: χ²/ndf 1.042 at ndf 11,096, where 3σ allows 1.040, with the total charge agreeing to 0.1 %).
**Where.** `validate/stats.py`; `docs/explanation/validation.md`.

## 13. The gradient check's bands

**Decision.** Edge steps are accepted when both neighbours (4h, h/4) agree; the spread band is widened by the
replicas' kurtosis; the edge floor proposes closed outcomes with a fixed 1 % share; the fixed-reference difference's
own float32 error is estimated at h/4.
**Why.** Each was measured to fail without it; the cases are in `docs/dev/validation.md`.
**Where.** `validate/gradients.py`, `validate/stats.py`, `transport/estimator.py`.

## 14. Sibling holes are lowered out of the subtraction

**Decision.** A hole of a subtraction (A − B) that is also a sibling placement (the same solid at the same pose in
the same mother) is left out of A, and A is placed before those siblings, which then outrank it.
**Why.** WCSim's black sheets subtract the PMTs that sit in them. The rewrite is exact: every point of the hole lies
in the sibling, which wins it either way (Geant4's locate takes later siblings first); a hole matching no sibling
stays subtracted.
**Where.** `build/lowering.py` (`lower_holes`, on by default).

## 15. The surface reader is apart from the sensor model

**Decision.** A sensor group names how its detecting surface behaves during tracking (`surface`: `lumen:geant4`,
`lumen:complex_index`, `lumen:counting`) separately from the after-tracking sensor model (`model`: `lumen:pmt`).
**Why.** One `model` field meant two things; schema `lumen-sensors/2` keeps them apart.
**Where.** `format/sensorsets.py`, `models.py`.

## 16. A user GROUPVEL is not read

**Decision.** Photons are timed by RINDEX's group index; a user GROUPVEL table is kept inert, with a warning.
**Why.** Geant4 lets a user GROUPVEL replace the computed one; lumen does not model that yet. The counts are Geant4's;
the arrival times in that material are not (`docs/explanation/known-limits.md`).
**Where.** `rules/optics.py`, the importer.
**Revisit.** 0.2 (`roadmap.md`).

## 17. GDML exporter defects are always repaired, with a warning

**Decision.** Defects that stock Geant4 refuses or misreads (non-finite or denormal values, matrix tails, ...) are
repaired on import, each with a warning, and recorded in the detector file's report.
**Why.** Refusing them sent every user to hand-edit exporter output; a warning keeps the edit visible. The geometry
check gives Geant4 the same repaired copy.
**Where.** `gdml/repair.py`, `gdml/reader.py`; `validate/geometry.py`.

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
