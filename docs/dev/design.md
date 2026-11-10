# lumen design

How lumen is built and why, for people changing it. The words users read are in `docs/explanation/glossary.md`;
the developer words and the terms they replace are in `docs/dev/vocabulary.md`. Where this file and the code disagree,
the code is wrong or this file is stale: fix one of them in the same change.

## 1. Scope

lumen is a differentiable optical-photon simulation in JAX for neutrino detectors: water Cherenkov (SK, WCTE, ANNIE)
and LArTPC (SBND, MicroBooNE) detectors, and others through detector packages. It converts a detector from GDML with
a small recipe, tracks photons with Geant4's optical physics, turns the light reaching each PMT into photoelectrons,
charges, times and waveforms, and gives derivatives with respect to every parameter except geometry.

It does three jobs:
- convert a physicist's GDML and simulate per-sensor charge, time and PMT waveforms;
- give correct derivatives for every calibration parameter, in forward and reverse mode;
- reproduce its validation checks (§14) and generate their report.

In 0.1:
- one Geant4-style navigator, with Geant4's overlap, facet and stuck rules;
- the detector file (schema `lumen-detector/1.1`), GDML import, and GDML export (frozen as built);
- Geant4's surface rule (border, then skin, then material default) over the legacy Geant4 surface types (§10);
- bulk absorption, Rayleigh, WLS and Mie (G4OpMieHG);
- sources: point, Cherenkov segment, scintillation, Geant4 step dump, external photon list;
- everything except geometry a traced parameter, with a declared gradient method per parameter and `wrt=`;
- the edge rule: a parameter is OUT of the model or WITHIN it, with one-sided derivatives at domain edges;
- reweighting for optics (forward mode the default), after-tracking derivatives for sensor values, the float32
  moving-source method for the source position (with stated limits), reverse mode validated on two detectors;
- PMT sensors: QE(λ), CE(θ), TTS, SPE charge, threshold; expected and sampled waveforms; the hit likelihood;
- one event per call, photon chunking and a memory refusal;
- tallymap as the photon-loop engine: CUDA on a GPU, its plain-JAX CPU backend elsewhere;
- three registries; plug-in protocols named but provisional.

Not in 0.1: §16.

## 2. Principles

| # | Principle |
|---|---|
| P1 | Geant4 semantics as general rules, never per-detector flags; match Geant4 in distribution; literal Geant4 ports only as test references |
| P2 | Derive optics from the GDML: the detector file holds what Geant4 reads, verbatim; everything lumen-specific is derived |
| P3 | One way to do each thing: one geometry, one GDML reader, one path that applies parameter values, one photon loop, one gradient method per kind of parameter, one parameter type |
| P4 | Everything except geometry is a parameter, and changing a value never recompiles; this outranks a 10 % speed loss |
| P5 | Float32 compute; float64 or closed-form references |
| P6 | Fail closed, naming the object, the Geant4 rule and the fix |
| P7 | Slow means redesign; every module has a line budget and stops for review at 1.5x it |
| P8 | The physics never imports tallymap; one runner (`run.py`) does |
| P9 | Users are human physicists: plain physics words in everything a user reads |
| P10 | Extensible without touching the core; a plug-in needs no Geant4 counterpart |
| P11 | Nothing private enters the repo |

Units are **m, ns, nm, eV** everywhere. The API takes plain numbers in these units; unit suffixes appear only in
file field names. In transport, `t` is a time (ns) and a distance along a ray is `dist` or `s` (m); the navigator's
ray parameter is `t` (m) inside `geometry/` only.

## 3. Words

`docs/dev/vocabulary.md` is the table. One meaning per word; in particular:
- **chain**: the moving-source chain only. GLOSSY's repeated facet reflections are the **facet path**; a
  calibration's ancestors are its **parents**.
- **pinned**: a vertex the moving-source method holds fixed while the source moves (never "held").
- **late / early**: light outside the time grid (the histograms' overflow); **retired**: a photon stopped past
  `t_end`.
- **surface type**: one of the legacy Geant4 surface behaviours (§10); there is no "kind" or "kernel" in the API.
- **sensor model**: the after-tracking PMT model only; how a detecting surface behaves during tracking is its
  **surface reader** (`SensorGroup.surface`).

## 4. Package layout

```
src/lumen/
├── api.py, __init__.py        load, simulate, convert; Detector, Light
├── params.py                  Params; unit, range and gradient method per key; precedence
├── recipe.py, plugins.py      Recipe; the three registries
├── cli.py                     lumen validate
├── diagnostics.py             per-photon probes run beside the loop (Trace)
├── _core/                     frames, safe math, rng
├── units.py, _env.py, _cache.py
├── format/                    schema, io, apply (THE value-application path), calibrations, sensorsets
├── gdml/                      reader, importer, units, solids_schema, tokens, exporter (frozen)
├── rules/                     navigation, boundary, optics (incl. the surface readers)
├── build/                     lowering + solids, scene + tree, surfaces (the surface lookup), library,
│                              spec + pack, cache
├── geometry/                  bvh, navigator_build, csg, navigate, surface_id, clearance, tracked
├── media/                     laws, bulk, phase (Rayleigh, Mie), spectra
├── surfaces/                  outcomes, fresnel, polar, lobes, facets, types (the legacy surface types)
├── transport/                 state, step, medium, surface_event, bulk_event, estimator, moving_source, item
├── run.py                     the tallymap runner: the only `import tallymap`
├── sources/                   point, cherenkov, scintillation, steps, arrays
├── readout/, sensors/         grid, detection, waveform, hits, events; the PMT model, components, default sets
├── detectors/                 wcsim, larsoft, their optics tables and the reference manifest (data)
└── validate/                  oracle, geometry, overlaps, charge, gradients, toys, stats, records, report
```

Line budgets count code lines only (no blanks, comments or docstrings). Outside `src/`: `tests/` (unit, physics,
toys), `refs/` (reference production, never packaged), `docs/`, `cluster/s3df/`, `tools/` (hooks). Detector files,
caches and reference outputs are not in the repo.

## 5. Data flow and signatures

GDML + recipe → `import_gdml` → detector file → `load` → build products (cached) and `Params` → `Model` (static) +
`Operands` (traced) → `run.py` (tallymap) → per-sensor histograms and health → readout functions → `pe`, waveforms,
hits, samples, likelihoods.

```python
# import
def import_gdml(path, recipe, inputs=None) -> ImportResult       # detector_file, sensor_sets, calibrations, report
def convert(source, recipe, out, *, inputs=None, overwrite=False) -> Path

# public
def load(name_or_path, *, sensors="default", calibration="nominal") -> Detector
def simulate(det, source, params=None, *, samples, seed, replicas=16, gate=None, chunk=1 << 20,
             backend="auto", wrt=()) -> Light
class Params(Mapping):  set, scale, info, explain        # an immutable pytree keyed by GDML names

# model and loop (developer level)
class Model: geometry, tables, step: StepConfig, clearance, hit_angle
class Operands(NamedTuple): params, reference, controls, nav, tables, grid
class Emission(NamedTuple): x, d, t, lam, w, polarisation, drawn_direction
def step(model, ops, ph: Photon, key) -> tuple[Photon, Deposit]
class PhotonLoop(NamedTuple): init, alive, emits, outputs, per_item
def run(loop, model, ops, rows, key, *, max_vertices, replicas, diagnostics=(), backend="auto",
        first_id=0, n_active=None) -> RunResult

# readout (after tracking; pathwise in every sensor value)
def pe(light, params); def pe_density(light, params); def expected_waveform(light, params)
def expected_hits(light, params); def sample(light, params, *, seed)          # sample: not differentiable
def hits_nll(eh, data); def waveform_nll(w, data)
```

- `Light` (a pytree): `arrivals (S,)`, `arrivals_error (S,)`, `mean_arrival_time (S,)`, the histograms, `health`;
  static `provenance` (seed, samples, replicas, chunk, backend, detector, calibration, wrt). Readout functions need
  only `(light, params)`.
- Photons run in chunks with `first_id` offsets, photon i under `fold_in(key(seed), i)`; the GPU's exact sums are
  order-independent, so a run is bitwise independent of the chunk size on one backend. `simulate` refuses before
  launch when the histograms exceed the device budget.
- `max_vertices` (2^22) is a guard against a photon that cannot end, not a physics cut; it exists only on `run`, and
  a photon it stops is reported (`truncated`), never dropped.

**Static and traced.** A static change recompiles; a traced one never does.

| | Static | Traced |
|---|---|---|
| geometry | navigator array shapes, solid programs | navigator array values (geometry is not a parameter) |
| surfaces, media | which surface types and channels exist; table knot counts | every property value |
| sources, sensors | source type; sensor models and components per group | every field and value |
| step | `StepConfig(moving_source, edge_rows)`, derived from `wrt`; whether a layered surface exists | `Controls(max_distance, t_end, roulette)` |
| histograms | their shape (sensors, λ knots, θ knots, time nodes); the gate | grid positions |
| run | backend, `max_vertices`, replicas, number of tangents | photons, seed (raw key data), samples, `Params`, `reference` |

**Memory at SK scale** (11,096 sensors, 1.5 µs gate, 0.5 ns nodes): the time histogram is 266 MB per replica, so
replicas live on the charge histogram (2 MB each) and the time histogram carries two half-replicas. Each forward
tangent adds the histograms again.

## 6. The detector file

### 6.1 Layout and schema `lumen-detector/1.1`

Canonical JSON (sorted keys, one-space indent, shortest round-trip floats, one record per line) and one `.npy` per
array, its sha256 in the manifest. The full GDML world is kept; the detector volume is cut out at load.

```
<name>.lumen/
  detector.json        schema, name, units, detector_volume, outside: "absorb", provenance, manifest
  materials.json       {state, T_K, P_Pa, composition, optics: {lumen_model, model_version, channels}, properties}
  elements.json, isotopes.json
  solids.json          GDML spelling and meaning (m, rad): 20 solid types + boolean, multiUnion, scaled, tessellated
  volumes.json         LV -> {solid, material, aux tree, daughters in Geant4 order}
  placements.json      ordered physvols {name, lv, mother, copy}; transforms in arrays/ as GDML angles
  surfaces.json        {lumen_model, model_version, geant4: {type, model, finish, value}, properties, inert, origin};
                       skins; borders (pv1 -> pv2, file order kept)
  tokens.json          the source token (literal and unit, as written) of each imported value
  userinfo.json, arrays/*.npy, sensors/<set>.json, sensors/<set>.ids.json, calibrations/ (read-only), sources/
```

- Values are exact decimals in m, ns, nm, eV. Each imported value keeps its source token, and the exporter writes
  the token for every uncalibrated value, so Geant4 on an export equals Geant4 on the original.
- Placements and boolean transforms are stored as GDML angles, positions and `scale` ±1; the builder takes matrices.
- Every material names its composition (a NIST name, or density and elements).
- Materials name their optics model (`geant4`) and its channels; surfaces name `lumen_model` and keep Geant4's enums
  verbatim under `geant4:`. A surface's behaviour is derived, never stored.
- Physvol names are explicit; sensor ids are fixed at import (placement order or copy number, optionally remapped to
  experiment ids), never taken from a build product.
- A reader refuses a newer major schema version, naming the lumen version that reads it.

### 6.2 Parameters

| Key | Value |
|---|---|
| `materials/<Material>/<PROPERTY>` | a Geant4 material property (table `{energy, value}` or constant) |
| `surfaces/<Surface>/<PROPERTY>` | a surface property; the opticalsurface's `value` is `surfaces/<Surface>/sigma_alpha` (unified) or `surfaces/<Surface>/polish` (glisur) |
| `sensors/<group>/<field>` | an array over the group's sensors in id order |
| `source/<field>` | a field of the source passed to `simulate` (`position`, `direction`, `time`, `n_photons`, `lam`, `spectrum`, …) |

Group names match `[A-Za-z_]\w*`, so they are never ids.

**OUT or WITHIN.** A value the model fixes is not a parameter: a lowering-implied constant (groundfrontpainted's
specular fraction 0; LArSoft's reflector glossy fraction 0) or a property absent from the GDML has no key. Keys exist
only for properties the tables define and for declared parameters of chosen models; a property Geant4 reads at a
default when the file omits it is a key at that default (`ParamInfo.default` names the Geant4 source line). A real
property at its domain edge (REFLECTIVITY 0 or 1) is WITHIN, and its derivative is one-sided toward the interior.

**Each key declares** a unit, an allowed range, a gradient method (`reweight`, `moving_paths`,
`moving_paths+reweight`, `after_tracking`) and whether that method is available. Asking for an unavailable one raises
`GradientNotAvailable`.

**Precedence**, later wins: the detector file's values; each group's sensor-model default values; the calibration's
parents, oldest first; the calibration; for `source/`, the source passed in. `params.explain(key)` names the layer.

**Calibration files** (`<name>.params.json`) name their detector and its content digest, and live in the user's
directories on `LUMEN_DETECTORS`, never inside a shared detector file. The loader refuses an unknown key (naming the
nearest), a value outside its range, a structural field, and a table whose knot count differs from the detector
file's. `format/apply.py` is the one path that applies values, for loading and for export.

### 6.3 Sensor sets

Structure only: groups, each with one selector (`lv` or `lv_pattern`), a counting rule (`border_into`, `any_entry`), a
surface reader (`surface`: how the detecting surface behaves during tracking, `lumen:geant4` or `lumen:counting`), an
optional sensor model (`model`: after tracking, `lumen:pmt`), the name of its default values and its readout (gate,
time bands). The sensor volumes are the union of the groups' selectors. Efficiency during tracking is derived: 1 when
a sensor model applies QE after tracking, Geant4's EFFICIENCY otherwise. Sensor positions and axes are derived at
load.

### 6.4 Digests and caches

| What | Key | Where |
|---|---|---|
| build products (tree, scene, navigator arrays, surface lookup, surface types, media) | sha256 of the content digest, the sensor-set digest, the code digest of `format, gdml, rules, build, geometry`, numpy/jax versions and optics-model versions | `$LUMEN_CACHE/build/<key>/` |
| parameter spec, model tables, sensor positions, `Params` | computed at load, not cached | memory |
| compiled kernels | JAX's key on the HLO; tallymap's on the CUDA source (no values in either) | `$LUMEN_CACHE/xla`, `$LUMEN_CACHE/tallymap` |
| public detectors converted on first `load` | source dataset sha256 + code digest | `$LUMEN_CACHE/detectors/` |

Caches are per user and never under `$HOME`; writes go to a temporary directory, then `os.replace`.

### 6.5 Importer and loader

**Import fails closed** on: an `opticalsurface` missing model, finish, type or value; a repeated border pair; a
border `physvolref` whose full name repeats (stripped-name repeats are fine: Geant4 and lumen bind by full name);
two skins on one LV; a missing `positionref`; an unsupported solid (the message lists the supported ones); a material
without composition; a surface form outside the legacy types; a property Geant4 would read that the reading ignores;
a sensor selector matching no volume.

**Recipe:** the detector volume, sensor groups, an id remap, extra tables GDML lacks (recorded with `origin`), the
recorded films (§10), shipped calibrations; typed fields only.

**`lumen.load`:** a path; else `$LUMEN_DETECTORS`; else `$LUMEN_CACHE/detectors`; else a `lumen.detectors` entry,
converted into the user's cache on first use. It checks the schema, cuts the detector volume, builds or fetches the
build products, computes the parameter spec and resolves the calibration. No recipe code runs at load.

### 6.6 GDML export (frozen as built)

`lumen.gdml.export(df, path, *, calibration=None, sensors="default", scope="active"|"full", on_lossy="error"|"omit")`.
Contract: `import(export(df)) == df` on every content file, and `import(export(df, c)) == apply(df, c)`; the source
tokens make Geant4-on-export equal Geant4-on-original. `on_lossy="omit"` writes no Geant4 surface where an optics
model has no Geant4 form, records it in the sidecar and marks the file `lossless: false`. Geant4 10.7.4 is the
reference reader; the Geant4 11.2.2 load check is advisory.

## 7. Geometry and the Geant4 rules

**The navigator** tracks the placed volume (`vol`), the surface the photon sits on (`srf`), its normal (`sn`) and the
count of zero-length steps (`nz`). One BVH per volume over its daughters; the crossing is the nearer of the own
solid's exit and the first daughter entry. Traversal is detached; only the winning surface is live (a custom JVP,
`dt = -g·(δo + t δd) / (g·d)`). Navigator arrays enter the loop as operands. No global flags in device code.

| Module | Rules | Geant4 source |
|---|---|---|
| `rules/navigation.py` | the float32 tolerance (three sites, two forms); surface identity (`K_ON`, `GRAZE`); entry tie → the lowest-index daughter; exit → relocate to the highest-index sibling holding the point; coincident partners both ways; a volume holds a point only if no hole filler or overlap link does; StepTooSmall continuation; facet rule; stuck rule | G4Navigator, G4VoxelNavigation, G4TessellatedSolid |
| `rules/boundary.py` | the surface at an ordered crossing (pre, post): (1) entering an "any entry" sensor volume → its surface; (2) the border surface for (pre, post); (3) skins: post's then pre's if post is pre's daughter, else pre's then post's; (4) material default: both with RINDEX → Fresnel, post without → absorbed. SameMaterial skip for polished or ground finishes only; exit sensors credited to the volume left; counting sensors count on every face | G4OpBoundaryProcess, G4LogicalBorderSurface, G4LogicalSkinSurface |
| `rules/optics.py` | units; tables linear in photon energy and clamped; Geant4's group velocity; Water's automatic Rayleigh; WLS and scintillation keys (Geant4 10 and 11); MIEHG and its constants; the (type, model, finish) → surface type table; the surface readers | G4MaterialPropertiesTable, G4OpBoundaryProcess, G4OpRayleigh, G4OpMieHG, G4OpWLS |

| Rule | Statement |
|---|---|
| Facet | a mesh crossing's t comes from the facet plane, 2 ulps short; a back facet met first means the photon is inside the opaque solid |
| Stuck | after 10 consecutive zero-length steps the photon is pushed 8 float32 tolerances along d and relocated; at 25 it is abandoned, its weight reported as `abandoned` |

**The surface lookup** (`build/surfaces.py`) evaluates `rules/boundary.py` on the host over every ordered pair of
volumes the navigator can report, including the coincident-face rule (p → q through a zero-thickness b, outward and
inward, taking the other leg's interface across a pass). At run time, classes (the volumes that share every crossing)
are an internal compression: the step reads (class, class) tables, the navigator a class per volume, and a crossing
whose class pair resolves to a pass is invisible. An unsupported surface fails closed only where it is reachable. A
frontpainted transmission into a medium without RINDEX is absorbed where it lands: painted surfaces are PAINTED_T,
with T a calibration key.

**Import rules** (`build/import_rules.py`; general, never recipe flags, recorded with origin "import"):
- `split_unions` by a leaf-count rule (16 leaves).
- **Absorbed at entry, on meshes only.** A mesh volume's material whose ABSLENGTH is below the band floor
  (`BAND_FLOOR`, 1e-9 m) at every knot is bound opaque: Geant4 refracts the photon in and absorbs it within a length
  the navigator cannot resolve, and the navigator does not track inside a mesh. Elsewhere such a medium is tracked as
  Geant4 tracks it. Applied to every volume, the rule would bind the PMT interiors' Air in SK and ANNIE, whose builds
  then fail (measured), so it stays on meshes.
- **One-sided WLS films, by the thinness band.** A box of a WLS material with no surface of its own is a film when its
  smallest box dimension is below the band at its first placement: `BAND_ULPS` float32 ulps of that placement's largest coordinate plus
  `BAND_FLOOR` (`rules.navigation.band`; 0.5 um at 0.5 m), the length below which two crossings are one. A thicker WLS
  box is tracked as a volume. The importer derives the film's axis from its contact face, per placement in its own
  frame, and fails closed unless the contact set is exactly one complete flat face of one backing per placement; a
  film on both faces, partial coverage or a bare placement is rejected. Because the band grows with the distance from
  the origin, a film's classification can depend on where it is placed: a 20 nm film is a film anywhere, a 1 mm
  slab never is.

## 8. Photon state, step and estimator

| Part | Fields | Present when |
|---|---|---|
| core | `x, d` (m), `t` (ns), `lam` (nm), `w`, `alive`, `vertex` (step count and random-number counter), `er, ei` (Jones vector), `vol`, `srf`, `sn`, `nz` | always |
| surface lookup | `cls` (class), `sensor` (sensor index) | always |
| `Stack` | `layer`, the volume, class and sensor beyond the surface | a layered surface exists |
| `MovingSource` | `drawn`, the moving-source chain | `wrt` holds a moving-path parameter |
| `z` | the edge-row factor | `wrt` holds a parameter at a domain edge |

Diagnostics keep their own per-photon memory; they are never state. Sources fill an `Emission`; the core builds the
`Photon` and its optional parts in one place (`state.start`). `Deposit(sensor, q, time, lam, …, hit)` carries the hit
angle θ on the PMT for groups whose model reads CE.

`step`: draw the step's random numbers → recover a stuck photon → evaluate the medium (live and reference) → draw the
free path → the next crossing → the moving-source pins (or the photon itself) → a bulk event if the free path is
shorter, else a surface event → roulette and `t_end` → advance (`vol, srf, sn, nz`) → the deposit (expected
detection, hit angle, health).

**Reweighting.** Every random choice is drawn at the reference values under stop-gradient and weighted by live /
reference: the free path; the bulk channel; the surface outcome (drawn from the reference outcome probabilities with
terminal outcomes zeroed, weight `p_live / p_ref`); the lobe direction (live / reference density); the whole facet
path; roulette (`w / p_s`). Detection is not sampled: the detected share is added at every surface hit (expected
detection). `reference=None` means the live values.

**The edge rule.** When an outcome probability, or Mie's forward ratio, is a `wrt` parameter at its domain edge, its
sampling reference moves just inside (q = ε > 0, ε chosen so the floored outcomes take a fixed share `EDGE_BUDGET` of
that row's draws) and the live weight p/q is kept. Values are unchanged; nothing changes when no such parameter is
in `wrt`. Edge rows are derived from `wrt` and the domains, never a user option.

**Health:** `escaped` (no surface and no bulk event within the maximum step distance), `abandoned` (stuck rule), `truncated` / `n_truncated` (alive at
`max_vertices`), `retired` (stopped past `t_end`), `stuck` (counts of pushes and abandonments), `lost` (a crossing no
surface type joins; with the surface lookup only a navigation failure, and 0 is a test invariant). Light outside the time grid is the histograms'
`late` / `early`.

**No recompile per value.** Every geometry array and value enters the loop as an operand, random keys as raw key
data; the runner refuses a loop body that closes over a concrete float array (`ClosedOverValues`). `test_no_recompile`
changes every kind of value (materials, surfaces, sensors, electronics, source fields, calibration, controls, grid
positions, seed, samples) and asserts zero new compilations.

## 9. Gradients

| Method | Parameters | How | 0.1 |
|---|---|---|---|
| reweight | absorption, Rayleigh, Mie and WLS lengths, Mie g and ratio, reflectivities and fractions, σα, polish, film values, source time, intensity and spectrum | reweighting, with the edge rule | forward mode (`jacfwd` per key) the default |
| moving_paths | source position and direction | the moving-source method in float32, re-pinning at every vertex | available, with known limits |
| moving_paths+reweight | a medium's RINDEX | – | `GradientNotAvailable` |
| after_tracking | QE, CE, efficiency, time offset, TTS, gain, pulse, dark rate, threshold, front end | plain JAX on the readout functions | available |
| – | geometry | not a parameter | – |

- `wrt=` lists what a call differentiates. It switches on the moving-source part and the edge rows when needed. A
  tangent on a moving field of a source whose directions are not drawn from a density raises `GradientNotAvailable`
  (a symbolic-zero custom JVP on each such input).
- **Both AD modes** are supported: forward (`jax.jvp`, `jax.jacfwd`) is the default for optics; reverse (`jax.grad`
  through `simulate`) costs about 3x the value call and is validated against finite differences on SK and SBND.
- A GLISUR polish derivative is unbiased with infinite variance.

**The moving-source method** (`transport/moving_source.py`). The expected count is an integral over paths; its
derivative is the expectation of the integrand's derivative at fixed coordinates plus a boundary term for every
discontinuity that moves in them. The method pins the vertices instead of the sampled coordinates, so a fixed
discontinuity stays fixed and its term is exactly zero. A drawn direction starts a chain with two free coordinates
u; deterministic outcomes after it are functions of u, and the photon carries their sensitivities, propagated by
forward mode of the step's own rules. At the chain's end u is moved so the end stays where it was drawn, and the
weight takes the change of measure. Every correction is zero in value. u is re-anchored at every deterministic vertex
so carried tangents stay O(1) in float32.

| Known limit of the moving-source method | Status |
|---|---|
| the shadow edge swept between a moving source and its first pinned vertex | not handled |
| the caustic at a pinned end (refraction through a PMT window) | infinite-variance tail |
| near a reflector (a source a few cm before a black sheet) | known limit |
| the float32 floor (e.g. SBND TPB film values) | reported "below float32 resolution" |

User-facing text for these limits: `docs/explanation/known-limits.md`.

## 10. Surfaces

0.1 ships the **legacy Geant4 surface types** (`surfaces/types.py`) behind the surface lookup. The `geant4` optics
model maps each `opticalsurface` (type, model, finish) to one of them.

| Surface type | Geant4 form it reads | Used by |
|---|---|---|
| PASS, ABSORBER, DIELECTRIC | material defaults (same medium; two indices; NoRINDEX) | every detector |
| DIELECTRIC_RT, DIELECTRIC_RT_SENSOR | dielectric_dielectric polished with REFLECTIVITY / TRANSMITTANCE (a survival cut ahead of Fresnel); the sensor variant detects with EFFICIENCY | SK, WCTE, ANNIE cathodes |
| PAINTED, PAINTED_T | polishedfrontpainted / groundfrontpainted (mirror or Lambert after the cut) | SK, WCTE, ANNIE black sheet; SBND foil |
| SPECULAR_METAL | dielectric_metal polished | WCTE reflector, absorber |
| UNIFIED_METAL, DIELECTRIC_METAL | dielectric_metal unified ground (spike, lobe facet path, backscatter, Lambert) | ANNIE holders and liner |
| GLISUR_METAL, GLISUR_METAL_EFF | dielectric_metal glisur ground (facet path) | SBND steel |
| REFLECTOR, SENSOR, PHOTOCATHODE, COUNTING | LArSoft reflector; detecting reflector; complex-index cathode; counting sensor | MicroBooNE, SBND |
| coated | a WLS film over a backing (§7) | SBND TPB |

Refused at import with a message: ground dielectric_dielectric, backpainted finishes, complex-index ground metals,
Geant4's `coated` type, LUT, LUTDAVIS, dichroic, SURFACEROUGHNESS. The surface outcomes have ten indices
(`surfaces/outcomes.py`). Each surface type is one `lax.switch` branch; its tables enter as operands. The surface
readers `lumen:geant4` and `lumen:complex_index`, which choose a detecting surface type for `border_into` groups,
live in `rules/optics.py` (`SENSOR_SURFACES`); `any_entry` groups use `lumen:counting`. "Sensor model" means only
the after-tracking PMT model.

## 11. Bulk processes and sources

| Channel | Geant4 process and keys | Notes |
|---|---|---|
| absorption | G4OpAbsorption, ABSLENGTH | a weight, never an event |
| Rayleigh | G4OpRayleigh, RAYLEIGH; Water's automatic table when RAYLEIGH is absent | polarised |
| Mie | G4OpMieHG: MIEHG (length table), MIEHG_FORWARD, MIEHG_BACKWARD, MIEHG_FORWARD_RATIO | below |
| WLS | G4OpWLS: WLSABSLENGTH, WLSCOMPONENT, WLSTIMECONSTANT | bulk `WLSMEANNUMBERPHOTONS` and WLS2 refused |

Channels are drawn in a fixed order (Rayleigh, Mie, WLS); a detector without MIEHG gets no Mie channel and is
bitwise unchanged.

**Mie**, Geant4's convention exactly: P(forward lobe) = MIEHG_FORWARD_RATIO; the forward lobe is HG(MIEHG_FORWARD),
the backward lobe HG(−MIEHG_BACKWARD); after scattering the polarisation is projected onto the plane transverse to the
new direction. Sampler: Geant4's rational inverse CDF rewritten without cancellation (`c₋ = 2(1−g)²(1−r)(1+gr)/D²`,
`c₊ = 2r(1+g)²(1−g+gr)/D²`, `D = 1−g+2gr`), 3 random numbers. Density: the marginal mixture
`r·f(g_f, ½|wo−wi|²) + (1−r)·f(g_b, ½|wo+wi|²)`, `f(g, c) = (1−g²)/(4π((1−g)²+2gc)^{3/2})`.

| Key | Range | Rule |
|---|---|---|
| `materials/<M>/MIEHG` | (0, ∞) m, a table | a constant is refused |
| `MIEHG_FORWARD`, `MIEHG_BACKWARD` | [−0.999, 0.999] (the cap is float32 resolution) | required only when Geant4 reads them (FORWARD unless ratio 0, BACKWARD unless ratio 1) |
| `MIEHG_FORWARD_RATIO` | [0, 1], required with MIEHG | at 0 or 1 it is WITHIN only if both lobes are defined, then the edge rule applies; otherwise OUT |

The three constants without MIEHG are inert and not parameters.

**Sources** (`source.emit(params, key, n, first_id, samples) -> Emission`): `Point` (isotropic or beam; wavelength or
spectrum), `CherenkovSegment`, `ScintillationPoint` / `ScintillationSegment` (yields and components from the GDML,
profiles, Birks), `Geant4Steps`, `FromArrays`. Every field is a parameter under `source/`. Weights sum in expectation
to `n_photons` using `samples` Monte Carlo photons; `drawn_direction` is true exactly when a direction was drawn from
a density (a sharp Cherenkov cone and an external list are not, so they give no source-position derivative); random
numbers come from `lumen.sources.uniforms(key, k, n, first_id)`. Sources run before the loop.

Geant4 step dump: arrays `start (n,3)` m, `end (n,3)` m, `t_start`, `t_end` ns, `beta_start`, `beta_end`, `charge`,
`edep` eV.

## 12. Sensors and readout (PMTs)

```
during tracking: surface type → DETECT → deposit (sensor, q, t, λ, θ)
after tracking:  detection  η_s(λ, θ) = ε_s · QE(λ) · CE(θ)                       linear in the histograms
                 electronics TTS ⊛, transit offset, time walk, SPE charge, pulse, dark rate, threshold, front end
                 → pe · pe_density · expected_waveform · expected_hits · sample
```

- **One sensor model: `lumen:pmt`.** SBND's X-ARAPUCAs and MicroBooNE's discs stay counting sensors read by the
  `lumen:counting` surface reader, with their reference efficiencies applied as constants.
- **Hit angle**: θ = acos(|z_local| / r_local) of the hit position on the PMT, WCSim's definition
  (WCSimWCSD.cc ProcessHits), so CE(θ) means the same in both codes.
- **Two histograms per group**: charge (sensor × λ knots × θ knots, exact in QE and CE) and time (sensor × one λ band
  × time nodes, scaled per sensor to the charge histogram). The time histogram is weighted at the reference QE: exact
  for a QE scale, approximate for a QE shape. Grids are static, derived from the components' declared supports and
  the gate, never from current values.
- **Timing**: a physical per-photoelectron TTS plus time walk from the PMT values.

| Output | Content |
|---|---|
| `pe` | expected photoelectrons per sensor |
| `pe_density` | PE per ns on the fine grid: node masses ⊛ (emission profile ⊛ TTS) shifted by the transit offset, plus dark rate |
| `expected_waveform` | `mean (C, n)`, `variance (C, n)` (shot plus electronic noise), `t0`, `dt` (ns), `unit`, channels |
| `expected_hits` | per sensor: PE mean, hit probability after the threshold, charge distribution on `charge_grid` (compound Poisson), first-hit time terms |
| `sample` | a simulated event: PE counts, hits (sensor, t, q), waveforms; not differentiable |
| `hits_nll`, `waveform_nll` | likelihoods for measured hits and waveforms |

Components: TTS (`gaussian`, `emg`, `table`), SPE charge (`gauss_exp`, `table`), pulse (`bigauss`, `exp2`, `table`),
dark rate, threshold (acceptance vs charge), front end (sampling, baseline, noise, integrator). Each has `kernel`,
`moments`, `charfn` or `sample`, and time kernels declare `support`.

| Default values | Source |
|---|---|
| `sk_r3600` | physical QE: WCSim's PMT20inch table without the ×4/3 of `SensitiveDetector_Only`; the SKI threshold acceptance; SPE table and dark rate from WCSim; CE: the SKDetSim table {100, 100, 99, 95, 90, 85, 80, 69, 35, 13} % in 10° steps, 0 at θ ≥ 90° (WCSimPMTObject.cc:70-71) |
| `wcte_r14374` | the same rule for WCTE's 3-inch PMT (own QE table, CE 95 % flat in WCSim) |
| `wcsim_annie_<type>` | ANNIE's PMT tables from annie WCSim |
| `sbnd_pmt` | `digi_pmt_sbnd.fcl`: efficiency 0.0392 coated (VUV), 0.026 coated (visible), 0.0357 uncoated; TTS 2.4 ns FWHM; transit 55.1 ns; SPE template; 2 ns sampling |

Default values are package data named `<distribution>:<set>@<version>`; a local path is accepted.

## 13. Extensibility

| Registry | Resolves |
|---|---|
| `lumen.detectors` | `lumen.load("<name>")` → `DetectorPackage` |
| `lumen.sensor_models` | a group's `model` → the after-tracking model |
| `lumen.optics_models` | a surface or material record's `lumen_model` → its behaviour (`geant4` in 0.1) |

Importers are plain functions; experiment software may write the format directly. Names are `distribution:model`;
duplicates are refused; an unknown model fails closed, naming the distribution. The plug-in protocols (`Lobe`, sensor
components, `Source`) are documented and **provisional in 0.1**. A plug-in needs no Geant4 counterpart, never
branches in Python on a parameter value, never closes over a parameter array and declares every value it reads.

## 14. Validation and tests

| Check | Reference | Pass |
|---|---|---|
| charge | WCSim (SK, WCTE); Geant4 via g4ref (WCTE, ANNIE, SBND, MicroBooNE) | ratios within quoted errors, \|z\| ≤ 3, per-sensor χ²/ndf |
| gradients | same-random-number and fixed-reference finite differences | K = 32 replicas, the (K−1)/(K−3) null, roulette off, automatic edge h-scans with a fixed-reference stencil, "FD unresolved" by the median rule, declared parameters only; forward on six detectors, reverse on SK and SBND |
| geometry import | g4ref `--probe` point location | 100 % |
| overlaps | the 18 g4ref overlap toys; the overlap census | detected fraction and mean time within errors |
| WCSim digits, SK consistency | digits from the charge-check runs; QE/CE-on runs with the patched SKDetSim CE table | yields 1 %, timing 0.02 ns and 1 %, threshold loss; digit times truncated on a 0.4 ns grid, applied on the reference side |
| SBND PMT waveforms | sbndcode OpDetSim; photon lists from the SBND reference | mean and variance per channel |
| Mie | §11 | §11 |
| export | Geant4 10.7.4 round trip on toys | `import(export(df)) == df`; Geant4 hits equal |

| Tier | Content | Target |
|---|---|---|
| unit and contract | units, rng, laws, lobes, surface types, format and digests, importer fixtures, round trips, navigator vs the float64 oracle, declared methods, `wrt` errors, no-recompile, readout toys, smoke toys | < 3 min, CPU |
| physics (closed form) | toys on the navigator: spheres, films, coincident pairs, overlaps through the step, border directions, a stuck photon, mesh vs box, the moving-source method on curved and refracting sensors, the Mie sphere, GPU vs CPU agreement (2.5e-4) | ~5 min, one GPU |
| reference | charge, geometry, overlaps, digits, waveforms, export | ~1 min per detector |
| gradients | the gradient check; the per-deposit replay against float64 at fixed random numbers | as measured |

References are reduced products with sha256, read-only in the reference store under `LUMEN_DATA`, listed in
`lumen/detectors/references.json`; production code lives in `refs/`. Check outputs never overwrite reference
products. The report is generated from the JSON records every `lumen validate` run writes.

## 15. Engineering

| Item | Rule |
|---|---|
| tallymap | pinned by SHA (`pyproject.toml` range, `cluster/s3df/constraints-develop.txt`), installed from one wheel; FFMA contraction off (it changes bits) |
| backend | `backend="auto"`: CUDA on a GPU, the CPU backend elsewhere; CPU sums are inexact, compared with the GPU at ~2.5e-4 |
| environment | the library reads only `LUMEN_DATA`, `LUMEN_DETECTORS`, `LUMEN_CACHE` (in `_env.py`) |
| tooling | src layout, hatchling + hatch-vcs, ruff at 120 columns, pre-commit, mypy on host code, type hints on the public API |
| style | plain physics words in anything a user reads (the jargon hook); functions ≤ 50 statements and ≤ 6 positional arguments; NamedTuples or frozen dataclasses built by keyword; Protocols, no `getattr` discovery; numpydoc with units; comments say why, never when; named constants with unit and source; no env reads outside `_env.py`; no site paths in `src/` or `tests/` |
| docstrings | what it is, its units, the Geant4 source it follows (class::method or file:line), what each non-obvious argument means; point here or to `docs/explanation/` instead of repeating; no port history |
| privacy | non-public detectors are separate entry-point packages; a pre-commit hook greps content, paths and messages against `LUMEN_PRIVATE_PATTERNS`, kept outside the repo |
| bitwise | results are bitwise reproducible on one CPU or GPU model only |

## 16. Not built yet, and next

Planned for 0.1, not in the code yet: `lumen.estimate` (cost estimate); `Light.check()` and the eager health warning;
`Params.sensor(id, field)`; `simulate(reference=)`; the `lumen convert` and `lumen doctor` commands; a hit-map and
waveform plot module; the error catalogue of the first-hour guide (`RangeError`, `NoSensorsMatched`,
`ModelNotInstalled`, `FormatVersionError`, `IncompleteResultError`, `GradientSetupError`, `MemoryEstimateError`; today
these raise `ValueError`, `MemoryError` or `GradientNotAvailable`).

Next (0.2): factorised surfaces and the public plug-in kit; deposit labels (labelled histograms and arrival records,
history bits); the event axis and sensor subsets; electronics other than PMTs (SiPMs for X-ARAPUCA, LAPPDs with
strips, afterpulses, crosstalk, saturation); refractive-index derivatives; the shadow-edge term; bulk
`WLSMEANNUMBERPHOTONS` and WLS2; LUT, LUTDAVIS, dichroic and latent-facet finishes; tracks and LAr recombination;
geometry calibration; tolerance unification.
