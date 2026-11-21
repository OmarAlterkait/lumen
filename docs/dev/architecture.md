# lumen architecture

How lumen is built, for people changing it. Why each choice was made is in `decisions.md`; the coding rules are in
`style.md`; what is not built yet is in `roadmap.md`. The words users read are in `docs/reference/vocabulary.md`; the
developer words and the terms they replace are in `vocabulary.md`. Cite a section of this file by its name (for
example "architecture.md, Gradients"), never by a number. Where this file and the code disagree, the code is wrong or
this file is stale: fix one of them in the same change.

## Scope

lumen is a differentiable optical-photon simulation in JAX for neutrino detectors: water Cherenkov (SK, WCTE, ANNIE) and
LArTPC (SBND, MicroBooNE) detectors, and others through detector packages. It converts a detector from GDML with a small
recipe or from YAML, tracks photons with Geant4's optical physics, turns the light reaching each
PMT into photoelectrons, charges, times and waveforms, and gives derivatives with respect to every parameter except
geometry.

It does three jobs:
- convert a physicist's GDML (or a YAML or built detector) and simulate per-sensor charge, time and PMT waveforms;
- give correct derivatives for every calibration parameter, in forward and reverse mode;
- reproduce its validation checks (Validation and tests) and generate their report.

Built:
- one Geant4-style navigator, with Geant4's overlap, facet and stuck rules;
- one detector structure (`lumen.format`, schema `lumen-detector/6`) with GDML import and export (frozen as built),
  YAML in both directions; the tests build their toys (`toy_sphere`, `toy_cube`, `toy_tank`) with a builder of their
  own (`tests/unit/builder.py`);
- Geant4's surface rule (border, then skin, then material default) over the Geant4 surface types (Surfaces);
- bulk absorption, Rayleigh, WLS and Mie (G4OpMieHG);
- sources: point, Cherenkov segment, scintillation, Geant4 step dump, external photon list;
- everything except geometry a traced parameter, with a declared gradient method per parameter, the estimator inferred from which values carry a tangent;
- the edge rule: a parameter is OUT of the model or WITHIN it, with one-sided derivatives at domain edges;
- reweighting for optics (forward mode the default), after-tracking derivatives for sensor values, reverse mode
  validated on two detectors (no source-position derivative in 0.2: refused);
- PMT sensors: QE(λ), CE(θ), TTS, SPE charge, threshold; expected and sampled waveforms; the hit likelihood;
- one event per call, photon chunking and a memory refusal; several modelled sensor groups;
- tallymap as the photon-loop engine: its kernel backend on the device JAX uses (CUDA on a GPU, C++/OpenMP on the
  CPU), its plain-JAX backend on request (`backend="xla"`);
- no registries: a model is a value passed to the build (its model table holds lumen's own by name).

Not built yet: `roadmap.md`.

## Principles

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

## Words

`docs/dev/vocabulary.md` is the table. One meaning per word; in particular:
- **facet path**: GLOSSY's repeated facet reflections.
- **late / early**: light beyond the time grid's ends (the histograms' overflow cells); no light is dropped.
- **surface type**: one of the Geant4 surface behaviours (Surfaces); there is no "kind" or "kernel" in the API.
- **sensor model**: the after-tracking sensor model only, read from a group's value set (`SensorGroup.values`); a
  detecting surface behaves during tracking as stock Geant4 reads it.

## Package layout

```
src/lumen/
├── api.py, __init__.py        load, simulate, convert; Detector, Light
├── params.py                  Params, Param; unit, range and gradient method per key; precedence
├── recipe.py                 Recipe, import_structure (every source ends there)
├── inputs.py                  a detector package's input files, checked by sha256 (its prepare: one pipeline)
├── record.py, speed.py, testing.py   the deposit and photon records; simulate's speed; the checks packages share
├── models.py                  medium models of your own: values (MediumModel), the build's table of them
├── diagnostics.py             per-photon diagnostics run beside the loop (Trace)
├── _core/                     frames, safe math, rng, the errors two layers share, checks (plug-in code checks),
│                              domains (the ranges and their one predicate)
├── units.py, _env.py, _cache.py
├── format/                    THE detector structure, importing nothing above it: schema, solids (the catalogue),
│                              check (the structural rules, for every input), io (the .lumen container), yaml
│                              (hand-written and handed-over detectors),
│                              elements (common elements by symbol), result (ImportResult), apply (THE
│                              value-application path), calibrations, sensorsets
├── gdml/                      reader, lexical (GDML's tokens and numbers), repair, importer (pointer suffixes
│                              included), units, exporter (frozen)
├── rules/                     navigation, boundary, optics (incl. detecting surfaces), entries (what a reading
│                              returns), keys (every value's unit, range, method), sensors, tolerances
│                              (every geometry and navigation tolerance, with the error it covers or the choice it is)
├── build/                     lowering, tree, library (incl. the models' entries and lumen's own models), binding
│                              (skins and borders as Geant4 binds them), lookup (the
│                              surface lookup), spec + pack, values (the value checks of every run), import_rules,
│                              tracked, placements (the conversion's placement check)
├── geometry/                  scene, solids, bvh, ownership (overlapping siblings), navigator_build, csg, csg_build,
│                              leaf_kernels, navigate, surface_id, clearance, cache (the navigator cache and the
│                              digests)
├── lobes.py                   direction lobes (shared by the media's phase functions and the surfaces' facets)
├── media/                     laws, bulk, phase (Rayleigh, Mie), spectra, spec
├── surfaces/                  outcomes, fresnel, polar, facets, types (the Geant4 surface types)
├── transport/                 state, step, medium, surface_event, estimator, item, fastpath
├── run.py                     the tallymap runner: the only `import tallymap`
├── sources/                   point, cherenkov, scintillation, steps, arrays, function (source functions)
├── readout/, sensors/         grid (the histograms' cells and memory), fill, detection, waveform, hits, events,
│                              custom, truth (truth rows); the PMT model, components, features, value sets
├── tools/                     check (the gradient check), stats (its pass rules), extend (the checks of extension
│                              code: source draws, lobes), fit, scan, space, losses, plot
└── compare/                   the comparisons against reference simulations: charge, event, geometry, overlaps,
                               gradients, mc (per family), ownership (the overlap-free GDML), references (manifest
                               and store), photons (a detector's loop for the checks), records, report, cli (python
                               -m lumen.compare), oracle (the float64 geometry referee)
```

The detectors' recipes live in their packages (lumen-wc, lumen-tpc).

Line budgets count code lines only (no blanks, comments or docstrings). Outside `src/`: `tests/` (unit, physics,
toys), `refs/` (reference production, never packaged), `docs/`, `tools/` (hooks, the readout transcription, the
Geant4 runner `tools/g4ref`). The
repository map and where new code goes: `README.md` in this directory. Detector files,
caches and reference outputs are not in the repo.

## Data flow and signatures

GDML + recipe → `lumen.convert` (`gdml.importer`; YAML → `recipe.import_structure`) → detector file → `load` → build
products (cached) and `Params` → `Model` (static) +
`Operands` (traced) → `run.py` (tallymap) → per-sensor histograms and health → readout functions → `pe`, waveforms,
hits, samples, likelihoods.

```python
# import
def convert(source, recipe=None, out=None, *, data=None, inputs=None, overwrite=False) -> Path
                                       # GDML (+ recipe), YAML, or "package:name" (a lumen.DetectorEntry)
def import_structure(df, sensor_set=None) -> ImportResult        # import rules, checks, sensor ids

# public
def load(name_or_path, *, calibration="nominal", parts=None, responses=()) -> Detector
def simulate(det, source, params=None, *, samples, seed, gate=None, events=None,
             reference=None, controls=None, backend="auto", chunk=None, speed="fast", records=None,
             diagnostics=(), truth=None) -> Light                # records: required under jax.jit / jax.vmap
class Params(Mapping):  set, scale, info, explain        # an immutable pytree keyed by GDML names

# model and loop (developer level)
class Model: geometry, tables, step: StepConfig, local, analog   # reach, device_tables: cached
class Operands(NamedTuple): params, reference, controls, nav, tables
class Photons(NamedTuple): x, d, t, lam, w, pol, label   # lumen.record: sources' and the loop's one record
def step(model, ops, ph: Photon, key) -> tuple[Photon, Deposit]
def trace(model, ops, rows, key, *, fields, capacity=None, max_vertices, diagnostics=(), backend="auto",
          first_id=0, n_active=None, analog=None) -> Traced    # deposit records + health (the photon loop)
class Readout(NamedTuple): emits, outputs, moments             # readout.fill: what is built from the records
def rebuild(readout, records, n, weights=(), *, backend="auto") -> (cells, nan_cells)
def run(readout, model, ops, rows, key, *, max_vertices, ...) -> RunResult   # trace + rebuild

# readout (after tracking; pathwise in every sensor value)
def pe(light, params); def pe_density(light, params); def expected_waveform(light, params)
def expected_hits(light, params); def sample(light, params, *, seed)          # sample: not differentiable
def hits_nll(eh, data); def waveform_nll(w, data)
```

- `Light` (a pytree): `arrivals (S,)`, `arrivals_error (S,)` (from the per-photon second moment), `mean_arrival_time
  (S,)`, each modelled group's histograms (`groups`, `Light.group(name)`: the charge histogram, its per-photon second
  moment `charge2`, the time histogram, each sensor's `pe_error`), `health`; static `provenance` (seed, samples,
  chunk, backend, detector, calibration, the controls as the loop read them, each group's gate), `truth` (the truth rows of `simulate(..., truth=Truth())`: one
  row per detected photon, drawn from the photon's analog record in the same loop, `lumen.readout.truth`). With
  `events=E` every array but the health has a leading event axis (each photon's source label its event;
  `Light.event(e)`). Readout functions need only `(light, params)`.
- Photons run in chunks with `first_id` offsets, photon i under `fold_in(key(seed), i)`, so every photon is the same
  whatever the chunk size. Within a launch the GPU's exact sums are order-independent; the chunks' results are added
  in float32 (`api._ADD`), so a run of several chunks differs from one launch by that rounding (up to about 3e-7 of a
  cell). The chunk is a memory lever: by default at most 2^20 (`api.CHUNK`), halved until the rows fit half the
  memory the readout's cells leave; a launch whose count pass reports more records than fit beside the readout's
  smallest block (`api._record_room`) is refused before its fill (`run.records_limit`, through tallymap's count hook)
  and, with the default chunk, run again at the size that fits; a given chunk is refused (MemoryError).
- **Errors from per-photon moments.** Beside each charge cell and each sensor's arrival the readout builds sum_i
  x_i^2, x_i photon i's adds to the cell summed over its records before the square (`Readout.moments`; in
  `run.rebuild` the record that first adds to a (photon, cell) sums the photon's later adds from its neighbouring
  records, which are in (photon, step) order, and adds the square into the same exact sums). N / (N - 1) (sum x^2 -
  S^2 / N) is unbiased for the variance of S over i.i.d. photons (`grid.moment_error`): `arrivals_error`, and the pe
  error at the run's values (`detection.pe_moment_error`, its "pe2" moment weighted by the run's acceptance). A
  moment is diagonal (no covariance between cells), has no derivative, and for fixed rows (`FromArrays`) includes the
  spread between the rows. A photon's records are in one launch, so the moments of several launches add.
- **Deposit records, then the readout per block.** The photon loop's only output of the light is its deposit records
  (`tallymap.Records`, count-then-fill: a deterministic (photon, step) order): per deposit the sensor, q relative to
  the row weight, the row weight, time, wavelength, the photon's id, its label and the fields the readout's features
  read (`run.record_fields`). No histogram, grid or axis lives in the loop; its sums are the health. Every binned
  output (charge, time, overflow, arrivals, a records diagnostic) is built from the records by `run.rebuild`, one
  tallymap item per record into exact sums, per block of sensors (`readout.fill.plan`); each add is formed as the
  loop formed its own (value x row weight), so the cells are bitwise those of an in-loop fill. Before launch
  (`api._memory`) the device's memory (`run.memory_limit`) pays the rows in flight and the readout's float32 cells
  (and a derivative's tangents) first. The records' slots are the launch's own count, from tallymap's count pass,
  rounded up to a power of two (`run.capacity` None; the same on a GPU and on the host), so their memory follows the
  light and the fill compiles once per power of two. Under an outer jax.jit or jax.vmap the count cannot reach the
  host before the shapes are fixed, so the caller gives the number there (`simulate(records=)`, bounded by
  `health.records` of the run with values; its power of two), and a traced launch without it is refused
  (`run.RecordsUnknown`), never launched at a guess. A launch past given slots is poisoned and counted in
  `health.records_dropped`. The readout's exact cells (tallymap's bytes of them, `tm.call_memory` / `tm.max_cells`) take what is left, and no more
  than the device's free memory after the launch, in as many blocks of sensors as they need (`api._block_budget`).
- `max_vertices` (2^22) is a guard against a photon that cannot end, not a physics cut; it exists only on `run`, and
  a photon it stops is reported (`truncated`), never dropped.

**Static and traced.** A static change recompiles; a traced one never does.

| | Static | Traced |
|---|---|---|
| geometry | navigator array shapes, solid programs | navigator array values (geometry is not a parameter) |
| surfaces, media | which surface types and channels exist; table knot counts | every property value |
| sources, sensors | source type; sensor models and components per group | every field and value |
| step | `StepConfig(edge_rows, levers, moving_source)` (`moving_source` dormant: always False) | `Controls(max_distance, roulette)` |
| histograms | their shape (sensors, λ knots, θ knots, time nodes); the gate | grid positions |
| run | backend, `max_vertices`, the records' fields and capacity, the readout's blocks | photons, seed (raw key data), samples, `Params`, `reference` |

**Memory at SK scale** (11,096 sensors, 1.5 µs gate, 0.5 ns nodes): the time histogram is 266 MB, the charge
histogram and its moment 2 MB each. Each forward tangent adds the histograms again. The photon loop holds no histogram: the
exact accumulators (44 B a cell) exist only for one block of sensors at a time in the readout. `Controls.max_distance`
None is the geometry's reach (`Model.reach`: the diagonal of the box holding every volume).

## The detector file

### Layout and schema `lumen-detector/6`

Canonical JSON (sorted keys, one-space indent, shortest round-trip floats, one record per line) and one `.npy` per
array, its sha256 in the manifest. The full world is kept; the detector volume is cut out at load. The files, the
records and every departure from "a field is a JSON key of its name" are `docs/reference/format.md`, generated from
the records and the layout spec (`format/io.py`: one walk writes and reads them).

- Values are float64 in m, ns, nm, eV, each rounded once from the exact decimal of its source literal and unit
  (`lumen.units.exact`). The GDML reader follows Geant4: what Geant4 refuses it refuses; `lumen.gdml.repair` is the
  one repair, run apart, and its copy is the source the provenance records (with recipe inputs, by sha256).
- Names: a GDML name is stored without Geant4's pointer suffix (`0x...`) where that keeps it unique within its kind,
  else whole (`gdml.importer.canonical_names`); the export writes the stored names.
- A record's `origin` is "source" (whatever the source: GDML or YAML), "recipe", "import",
  "geant4_default" (a Geant4 default key a value set) or "model_default". The one rule that reads a material's name
  is G4OpRayleigh's for Water (decision 22).
- Placements and boolean transforms are stored as GDML angles, positions and `scale` ±1; YAML takes matrices.
- Every material names its composition (a NIST name, or density and elements).
- Materials and surfaces name their optics model (`optics.name`, `geant4` by default); surfaces keep Geant4's
  `type`, `model`, `finish` and `value` verbatim. A surface's behaviour is derived, never stored.
- Physvol names are explicit; sensor ids are fixed at import, numbered in the canonical depth-first walk of the
  lowered detector volume (`lumen.geometry.scene.instances`: placements in order), never by copy number. A
  sensor's id-map path is its volume-tree PV name: placement names joined by "/", a sibling repeating a name told apart
  by `#<index>` (`lumen.geometry.scene.placement_segments`). Experiment ids go in id-map columns.
- A reader refuses a file of any other format version with one message, naming how to write it again
  (`lumen.format.io.check_format`).

### Parameters

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
default when the file omits it is a key at that default (`Param.note` names the Geant4 source line). A real
property at its domain edge (REFLECTIVITY 0 or 1) is WITHIN, and its derivative is one-sided toward the interior.
The one exception is a value Geant4 itself branches on: an opticalsurface's sigma_alpha exactly 0 (GetFacetNormal
returns the normal, G4OpBoundaryProcess.cc:626-628, 10.7.4) and polish exactly 1 (no facet below it, :653) select
another reading, so there the file's value decides that the surface value has no key; above them the key's range is
the facet samplers' ([SIGMA_MIN, SIGMA_MAX], polish up to 1 - GLISUR_R_MIN), and a fit does not reach the edge.

**Each key declares** a unit, an allowed range, a gradient method (`reweight`, `moving_paths`,
`moving_paths+reweight`, `after_tracking`) and whether that method is available. Asking for an unavailable one raises
`GradientNotAvailable`.

**Precedence**, later wins: the detector file's values; each group's sensor-model default values; the calibration
(flat: every value it changes from the nominal ones); for `source/`, a key set (`Params.set`) over the source's field,
and a key that disagrees with the field otherwise (a traced field, or the copy `parameters()` made of another source's)
refused (`Source.refuse_overridden`). `params.explain(key)` names the layer.

**Calibration files** (`<name>.params.json`) name their detector and its digest (`Detector.digest`: the manifest of
the content files, the name left out, plus the head's `detector_volume` and the sensor set's files with each group's
value set, so a GDML- and a YAML-converted copy of one detector share it, and nominal sensor values cannot change
under it), and live in the user's
directories on `LUMEN_DETECTORS`, never inside a shared detector file. The loader refuses an unknown key (naming the
nearest), a value outside its range, a structural field, and a table whose knot count differs from the detector
file's. `format/apply.py` is the one path that applies values, for loading and for export.

### Sensor sets

One per detector file (`sensors.json`, `sensors.ids.json`). Structure only: groups, each with one selector (`lv` or
`lv_pattern`), a counting rule (`border_into`: the detecting surface read as stock Geant4 reads it; `any_entry`: no
surface read), an optional value set (`values`: a reference in a recipe, its content once converted; a group with
one is read by the sensor model after tracking) and its readout (gate, time bands). The sensor volumes are the union
of the groups' selectors. Efficiency during tracking is derived: 1 when the group has a value set (the efficiency
applies after tracking), else the group's `entry_efficiency` setting (`sensors/<group>/entry_efficiency`), else
Geant4's EFFICIENCY (`lumen.rules.sensors`). Sensor positions and axes are derived at
load.

### Digests and caches

| What | Key | Where |
|---|---|---|
| the navigator arrays (`geometry/cache.py`; the tree, scene, surface lookup and media are built at load) | sha256 of the scene's digest, the build options and the code digest of `geometry.cache.NAV_CODE` and every lumen module they import (`code_closure`, read from the import statements) with the numpy and jax versions | `$LUMEN_CACHE/build/<key>/` |
| the scene's ownership (the navigator build's class-free, costly stage: `navigator_build.prepare`; every set of classes and settings finishes from it) | the same key without build options | `$LUMEN_CACHE/build/own-<key>/` |
| the leaf map of the packed parameters (each leaf: a key it reads, a JAX function of one, or a constant of the build) | per loaded detector (`build/pack.py`): pack is a gather, nothing is rebuilt | memory |
| parameter spec, model tables, sensor positions, `Params` | computed at load, not cached | memory |
| compiled kernels | JAX's key on the HLO; tallymap's on the kernel source, and on the CPU also the compiler and CPU (no values in either) | `$JAX_COMPILATION_CACHE_DIR` (JAX's own, when set), `$LUMEN_CACHE/tallymap` |

Caches are per user, under one root (`LUMEN_CACHE`, by default in the system's temporary directory, never under
`$HOME`); a cache directory the user names is used as named. Writes go to a temporary directory, then `os.replace`.

### Importer and loader

**Import fails closed** on: an `opticalsurface` missing model, finish, type or value; a repeated border pair; a
border `physvolref` whose full name repeats (stripped-name repeats are fine: Geant4 and lumen bind by full name);
two skins on one LV; a missing `positionref`; an unsupported solid (the message lists the supported ones); a material
without composition; a surface form outside the Geant4 surface types; a property Geant4 would read that the reading ignores;
a sensor selector matching no volume.

**One structure, several sources.** `lumen.format`'s records are the detector, whatever it was read from: the GDML
importer (`gdml/importer.py`, then a recipe), the YAML reader (`format/yaml.py`: the YAML holds the recipe's
decisions) and the tests' builder (`tests/unit/builder.py`) all end in `format.check`, the import rules and the sensor ids
(`recipe.import_structure`), and `convert` saves the result. YAML maps the records one to one (exact both ways); it is
never what `load` reads (25-30x slower than the JSON). A built detector exported to GDML reads back with the same
geometry and values.

**Recipe:** the detector volume, sensor groups, an id remap, extra tables GDML lacks (recorded with `origin`), the
recorded films (Surfaces), shipped calibrations; typed fields only.

**`lumen.load`:** a path only. Detectors are not registered and not searched for: a detector is its file, converted
once (`lumen.convert`). It checks the schema, cuts the detector volume, builds or fetches the
build products, computes the parameter spec and resolves the calibration. No recipe code runs at load.

### GDML export (frozen as built)

`lumen.gdml.export(df, path, *, calibration=None, on_lossy="error"|"omit")`: the whole world,
its materials and surfaces, as bare literals in Geant4's units that read back to the same doubles (the GDML a Geant4
control runs on). Contract: lumen's reader gives back every stored value of what the GDML holds, and
`read(export(df, c))` the values of `apply(df, c)`. `on_lossy="omit"` writes no Geant4 surface where an optics
model has no Geant4 form, lists it and marks the file `lossless: false`. Geant4 10.7.4 is the
reference reader; the Geant4 11.2.2 load check is advisory.

## Geometry and the Geant4 rules

**The navigator** tracks the placed volume (`vol`), the surface the photon sits on (`srf`), its normal (`sn`) and the
count of zero-length steps (`nz`). One BVH per volume over its daughters; the crossing is the nearer of the own
solid's exit and the first daughter entry. Traversal is detached; only the winning surface is live (a custom JVP,
`dt = -g·(δo + t δd) / (g·d)`). Navigator arrays enter the loop as operands. No global flags in device code.

| Module | Rules | Geant4 source |
|---|---|---|
| `rules/navigation.py` | the float32 tolerance (one form, `band(scale)`, with named multiples); surface identity (`K_ON`, `GRAZE`); a point → one owner, level by level (`geometry/ownership.py`: a solid proved inside its sibling, else the one placed first): a sibling (of a volume or an ancestor) ahead of it in the rule's order whose box may meet it and that is not proved apart is a candidate, an ancestor it is not proved inside a candidate whose face is an exit, the containment tests deciding at run time; every tie (entry, relocation) → the first in the rule's order (`prio`); a volume holds a point only if none of its owners does; facet rule; stuck rule | G4Navigator, G4VoxelNavigation, G4TessellatedSolid |
| `rules/boundary.py` | the surface at an ordered crossing (pre, post): (1) entering an "any entry" sensor volume → its surface; (2) the border surface for (pre, post); (3) skins: post's then pre's if post is pre's daughter, else pre's then post's; (4) material default: both with RINDEX → Fresnel, post without → absorbed. SameMaterial skip for polished or ground finishes only; exit sensors credited to the volume left; counting sensors count on every face | G4OpBoundaryProcess, G4LogicalBorderSurface, G4LogicalSkinSurface |
| `rules/optics.py` | units; tables linear in photon energy and clamped; Geant4's group velocity; Rayleigh as G4OpRayleigh computes it (Water's, or from ISOTHERMAL_COMPRESSIBILITY); WLS and scintillation keys (Geant4 10 and 11); MIEHG and its constants; the (type, model, finish) → surface type table; the surface readers | G4MaterialPropertiesTable, G4OpBoundaryProcess, G4OpRayleigh, G4OpMieHG, G4OpWLS |

| Rule | Statement |
|---|---|
| Facet | a mesh holds no point; a facet met from either side is an entry into it; a mesh crossing's t comes from the facet plane, 2 ulps short |
| Stuck | after 10 consecutive zero-length steps the photon is pushed 8 float32 tolerances along d and relocated; at 25 it is abandoned, its weight reported as `abandoned` |

**The surface lookup** (`build/lookup.py`) evaluates `rules/boundary.py` on the host over every ordered pair of
volumes the navigator can report, including the coincident-face rule (p → q through a zero-thickness b, outward and
inward, taking the other leg's interface across a pass). At run time, classes (the volumes that share every crossing)
are an internal compression: the step reads (class, class) tables, the navigator a class per volume, and a crossing
whose class pair resolves to a pass is invisible. A crossing between two volumes of one class (touching siblings of one
LV with a skin or an any-entry sensor) is the table's diagonal: the navigator stops at a crossing into another volume of
a class whose diagonal has a surface, and the deposit goes to the volume entered. An unsupported surface fails closed only where it is reachable. A
frontpainted transmission into a medium without RINDEX is absorbed where it lands: painted surfaces are PAINTED_T,
with T a calibration key.

**Import rules** (`build/import_rules.py`; general, never recipe flags, run once after the recipe's decisions):
- **Geant4-10 scintillation keys are translated** to Geant4 11's, checked against Geant4 10.7.4's reading of the
  original (recorded in the report and `provenance["translations"]`); keys with no single reading are refused.
- `split_unions` by a leaf-count rule (16 leaves).
- **Absorbed at entry, on meshes only** (recorded with origin "import"). A mesh volume's material whose ABSLENGTH is below the band floor
  (`BAND_FLOOR`, 1e-9 m) at every knot is bound opaque: Geant4 refracts the photon in and absorbs it within a length
  the navigator cannot resolve, and the navigator does not track inside a mesh. Elsewhere such a medium is tracked as
  Geant4 tracks it (decision 8).
- **Thin WLS boxes are refused.** A box of a WLS material whose smallest dimension is below the band where it is placed
  (`BAND_ULPS` float32 ulps of its largest coordinate plus `BAND_FLOOR`, `rules.navigation.band`; 0.5 um at 0.5 m, the
  length below which two crossings are one) is refused at conversion; lumen lowers no film onto a surface. A recipe
  makes such a film a volume the navigator resolves (SBND's TPB: lumen-tpc splits the CPA foil into a 25.4 um TPB
  half, bulk WLS, and a vm2000 half keeping its skin; decision 9).

## Photon state, step and estimator

| Part | Fields | Present when |
|---|---|---|
| core | `x, d` (m), `t` (ns), `lam` (nm), `w`, `wr` (the weight at the reference values, stop-gradient: the roulette reads it), `alive`, `vertex` (step count and random-number counter), `er, ei` (Jones vector), `vol`, `srf`, `sn`, `nz` | always |
| surface lookup | `cls` (class), `sensor` (sensor index) | always |
| `moving` | a source-position derivative's per-photon state: a dormant slot (`StepConfig.moving_source`) | never in 0.2 |

Diagnostics keep their own per-photon memory; they are never state. Sources fill a `Photons` row (`lumen.record`); the core builds the
`Photon` and its optional parts in one place (`state.start`). `Deposit(sensor, q, time, lam, …, hit)` carries the hit
angle θ on the PMT for groups whose model reads CE.

`step`: draw the step's random numbers → recover a stuck photon → evaluate the medium (live and reference) → draw the
free path → the next crossing → a bulk event if the free path is
shorter, else a surface event → roulette (on the reference weight `wr`) →
advance (`vol, srf, sn, nz`) → the deposit (expected detection, hit angle, health).

**Reweighting.** Every random choice is drawn at the reference values under stop-gradient and weighted by live /
reference: the free path; the bulk channel; the surface outcome (drawn from the reference outcome probabilities with
terminal outcomes zeroed, weight `p_live / p_ref`); the lobe direction (live / reference density); the whole facet
path; roulette (`p_s = wr / ε` on the reference weight `wr`, the photon's weight had the live values been the
reference, relative to its own start: the loop multiplies every weight emit by the row's weight, `lumen.run`; a kept
photon divides `w` and `wr` by `p_s`; a roulette loser's weight is 0, the only way a weight ends a photon). The
roulette is the only decision that read the weight; on `wr` it reads the reference alone, so with the reference held
(`simulate(reference=)`) a realisation is smooth in the live values, and one ε (`Controls.roulette`, default
`step.ROULETTE`) serves values, derivatives, fits and checks. With a reference of its own (`simulate(reference=)`,
or a bulk lobe's edge row) `wr` takes the event's factor at the reference values, a closed form in what the step
already has (every live / reference ratio of a choice drawn at the reference is 1 there): a surface outcome
`att_r p_draw[o] / q[o] w_drawn` (the attenuation at the reference, the outcome at the probability it is drawn from,
the lobe's density drawn from over itself), a bulk event the absorption, the emission's weight and the jump's factor
at the reference. When the run's values are the reference the same formula is `w` bitwise, except after an edge row's floored outcome (`w` 0 in
value).
Detection is not sampled: the detected share is added at every surface hit (expected detection). `reference=None`
means the live values.

**The one defensive mixture** (`lumen.transport.estimator`). A sampled part declares the parameters that can make a
draw at the reference fail to cover the live density (role "zero": a share that can zero a probability; role
"support": a parameter that moves a support's edge) and a covering density: a surface outcome
(`Interaction.defensive`; uniform over the outcomes of reference probability 0), GLISUR's facet (the shell,
`facets.covered`), Mie's lobe choice (`Lobe.defensive`, `Lobe.cover`), a scintillation component and a spectrum's
knot intervals without mass (`spectra.covered`). When such a parameter is keyed (the detector's structure:
`api._edge_rows`, never a tangent, a value or a user option; a source's shares and spectra always) the part is drawn
from (1 - e) p_ref + e c, e = `DEFENSIVE_SHARE`, and from p_ref where nothing is covered. The live weight is p/q (0
in value at a covered draw, dp/q in derivative: a value at the domain's edge gets its one-sided derivative); the
reference weight carries q / q, so the roulette keeps the photons that carry the derivative. A photon of weight 0
is never ended for it: a second covered draw on its path carries the second derivative. Expectations are unchanged; a value-only run and the primal of a derivative are
the same run, bitwise.

**Health** (every photon's start weight ends in one fate, each the weight where the photon ends; their sum is the start
weight at the reference values, `Health.balance`): `absorbed_bulk`, `absorbed_surface` (the detected share is part of
it), `rouletted` and `reweighted` (0 in expectation), `escaped` (no surface and no bulk event within the maximum step
distance, at its default the reach), `distance_cut` (the same within a shorter `max_distance`), `abandoned` (stuck rule), `truncated` (alive at `max_vertices`), `lost` (a
crossing no surface type joins; with the surface lookup only a navigation failure or a transparent volume overlapping an
opaque one, where Geant4 kills the photon too (known limits; a zero-length step out of the opaque volume goes on,
Geant4's StepTooSmall); 0 is a test invariant on the toys, and on every detector the charge check gates it with
`abandoned`: their weight may not move the total by its Monte Carlo error). Light outside the time grid is the
histograms' `late` /
`early`.

**No recompile per value.** Every geometry array and value enters the loop as an operand, random keys as raw key
data; the runner refuses a loop body that closes over a concrete float array (`ClosedOverValues`). `test_no_recompile`
changes every kind of value (materials, surfaces, sensors, electronics, source fields, calibration, controls, grid
positions, seed, samples) and asserts zero new compilations.

## Gradients

| Method | Parameters | How | 0.1 |
|---|---|---|---|
| reweight | absorption, Rayleigh, Mie and WLS lengths, Mie g and ratio, reflectivities and fractions, σα, polish, source time, intensity and spectrum | reweighting, with the edge rule | forward mode (`jacfwd` per key) the default |
| moving_paths | source position and direction | – | `GradientNotAvailable` (a tangent on a field that moves where the light starts: `sources._base.MOVES`) |
| moving_paths+reweight | a medium's RINDEX, a point source's wavelength (`Point.lam`) | – | `GradientNotAvailable` (one rule, `lumen.params.PENDING`, for detector keys and source fields) |
| after_tracking | QE, CE, efficiency, time offset, TTS, gain, pulse, dark rate, threshold, front end | plain JAX on the readout functions | available |
| – | geometry | not a parameter | – |

- No call declares what it differentiates: one `custom_jvp(symbolic_zeros=True)` rule around the run
  (`api._with_rule`) takes the traced values as its arguments and differentiates only those that carry a tangent that
  is not a symbolic zero, so every guard inside sees symbolic zeros as symbolic; a value only traced is a value run.
  A tangent on a field that moves where the light starts raises `GradientNotAvailable` at the source (a refusing
  custom JVP on each such input).
- **Both AD modes** are supported: forward (`jax.jvp`, `jax.jacfwd`) is the default for optics; reverse (`jax.grad`
  through `simulate`) costs about 3x the value call and is gated against the fixed-reference finite difference on SK
  and SBND (the gradient check's reverse row).
- A GLISUR polish derivative moves the edge of the facet cone, where the density vanishes: its model declares polish
  with role "support" (drawn with the shell as cover, a bounded per-facet derivative), and the gradient check reports
  rather than gates its exactness (a fixed-reference finite difference sees square-root terms at every step).

**Source-position derivatives** are not in 0.2 (the moving-source method was removed; decision of 2026-11-20). What a
return as a module of its own needs is kept dormant: the static switch `StepConfig.moving_source` (True is refused),
the per-photon slot `Photon.moving`, the navigator's `surface_at` (re-evaluate the winning surface without walking)
and the one `custom_jvp` rule.

User-facing text for these limits: `docs/explanation/known-limits.md`.

## Surfaces

lumen ships the **Geant4 surface types** (`surfaces/types.py`) behind the surface lookup. The `geant4` optics
model maps each `opticalsurface` (type, model, finish) to one of them.

| Surface type | Geant4 form it reads | Used by |
|---|---|---|
| PASS, ABSORBER, DIELECTRIC | material defaults (same medium; two indices; NoRINDEX) | every detector |
| DIELECTRIC_RT | dielectric_dielectric polished with REFLECTIVITY / TRANSMITTANCE (a survival cut ahead of Fresnel); the absorbed share detected with EFFICIENCY (`eff`) | SK, WCTE, ANNIE cathodes |
| PAINTED_T (PAINTED: T fixed at 0) | polishedfrontpainted / groundfrontpainted (mirror or Lambert after the cut, TRANSMITTANCE straight on), the absorbed share detected with `eff` (DoAbsorption) | SK, WCTE, ANNIE black sheet; SBND foil |
| SPECULAR_METAL (UNIFIED_METAL_NORMAL with the spike fixed at 1) | dielectric_metal polished (and glisur at polish 1): a mirror, the absorbed share detected with `eff` | WCTE reflector, absorber |
| UNIFIED_METAL | dielectric_metal unified ground (spike, lobe facet path, backscatter, Lambert), `eff` | ANNIE holders and liner |
| UNIFIED_METAL_NORMAL | the same at sigma_alpha 0 (GetFacetNormal returns the normal: the lobe is the spike) | the Geant4 form of LArSoft's reflector |
| `transmitting(type)` | any dielectric_metal type above with TRANSMITTANCE: T straight on, as DielectricMetal draws it | |
| GLISUR_METAL | dielectric_metal glisur ground (facet path), `eff` | SBND steel |
| REFLECTOR (PAINTED with `eff` fixed at 0), PHOTOCATHODE, COUNTING | LArSoft reflector; complex-index metal (REALRINDEX + i IMAGINARYRINDEX, tables read at the wavelength); an `any_entry` group's sensors (every entry counted, no surface read) | MicroBooNE, SBND |

One type per Geant4 reading: a sensor is the type with `eff` read from EFFICIENCY (default 0), a surface that is no
sensor the same type with `eff` fixed at 0 (`types.no_detection`, named `<type>{eff=0}`). A table stays a table,
flat or not. A type that gives the same probabilities as another with constants fixed is that type (`types.fixed`;
the Geant4 finish names stay in the files).

Refused at import with a message: ground dielectric_dielectric, backpainted finishes, complex-index ground metals,
Geant4's `coated` type, LUT, LUTDAVIS, dichroic, SURFACEROUGHNESS. The surface outcomes have ten indices
(`surfaces/outcomes.py`). A surface type is the structure of a row, never code: every crossed (type, surface) is a row
of one parameter table (`build/lookup.SurfaceTable`: a column per parameter, R, T, eff, the reflection split, the lobe
widths and a complex index; each row's constants where
it does not read a column as a parameter, decided by key presence at load, never by a value at run time), read by
one kernel (`types.kernel`). Each row has an interaction model (`types.Interaction`: a value with its parameters,
its shares and its drawn outcomes' direction laws; Geant4's models share Geant4's cuts (R, T), the reflection split
or the Fresnel interface, the detection and the far side), and the kernel is one `lax.switch` over the models present
in the build (a single model compiles no switch). A model gives six continuing shares (`lumen.surfaces.outcomes`) and,
separately, its detected share (a scalar deposit, `TypeEval.detect`); the loss is the rest. A type detects when it
reads an efficiency `eff` (`SurfaceType.detects`). The table's columns enter as
operands; `tests/unit/test_surface_table.py` checks the kernel against the
per-type formulas it replaced, bit for bit, on every row. The reading
of a detecting surface (stock Geant4's, `rules/optics.sensor`), which chooses a detecting surface type for
`border_into` groups, is defined in `rules/optics.py`; `any_entry` groups read no surface (they count every entry). "Sensor model" means only
the after-tracking sensor model.

## Bulk processes and sources

| Channel | Geant4 process and keys | Notes |
|---|---|---|
| absorption | G4OpAbsorption, ABSLENGTH | a weight, never an event |
| Rayleigh | G4OpRayleigh, RAYLEIGH; without it, Geant4's table of a material named Water (283.15 K), else from ISOTHERMAL_COMPRESSIBILITY | polarised |
| Mie | G4OpMieHG: MIEHG (length table), MIEHG_FORWARD, MIEHG_BACKWARD, MIEHG_FORWARD_RATIO | below |
| WLS | G4OpWLS: WLSABSLENGTH, WLSCOMPONENT, WLSMEANNUMBERPHOTONS, WLSTIMECONSTANT | below; WLS2 refused |

Channels are drawn in a fixed order (Rayleigh, Mie, WLS); a detector without MIEHG gets no Mie channel and is
bitwise unchanged.

**WLS**, the bulk channel, as G4OpWLS::PostStepDoIt: the emission energy is
drawn from WLSCOMPONENT's sampler truncated at the absorbed photon's energy (Geant4 redraws up to 100 times until it
is not above it, and drops the photon if none is), by the exact truncated inverse CDF, with Geant4's chance of
success 1 - (1 - F)^100 a weight (F: the spectrum's mass at or below the absorbed energy); WLSMEANNUMBERPHOTONS the
expected yield (Geant4: Poisson; 1 when absent), a weight; isotropic and unpolarised; the delay WLSTIMECONSTANT
under the material's ``wls_time_profile``, "delta" (Geant4's default: the delay itself) or "exponential". Every
re-emission samples one emission record (``lumen.media.emission.EmissionRecord``: yield, spectrum, time profile built
from components: delta, decay, rise or a table); the photon goes on at its new wavelength, nothing is spawned. The
yield, the absorption length, the delay and the bulk spectrum (WLSCOMPONENT, a parameter: drawn at the reference,
weighted live / reference) are differentiable. Scintillation sources draw their delays from the same time profiles:
the decay, and SCINTILLATIONRISETIMEi as a rise component only under the material's optics setting
``scint_finite_rise_time`` (Geant4's finite-rise-time switch; a rise time without it is refused).

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

**Sources** (`source.emit(params, key, n, first_id, samples) -> Photons`): `Point` (isotropic or beam; wavelength or
spectrum), `CherenkovSegment`, `ScintillationPoint` / `ScintillationSegment` (yields and components from the GDML,
profiles, Birks), `CherenkovSteps` / `ScintillationSteps` (Geant4 steps, MeV), `FromArrays`. Every field is a parameter under `source/`. Weights sum in expectation
to `n_photons` using `samples` Monte Carlo photons; random
numbers come from `lumen.sources.uniforms(key, k, n, first_id)`. Sources run before the loop. A Cherenkov source's
unset wavelength bounds are the span of its medium's RINDEX table, the range G4Cerenkov emits over. Every optical
table is read at any wavelength as Geant4 reads it, held at its end values beyond its knots, and checked at its own
knots.

Geant4 step dump: arrays `start (n,3)` m, `end (n,3)` m, `t_start`, `t_end` ns, `beta_start`, `beta_end`, `charge`,
`edep` eV.

## Sensors and readout (PMTs)

```
during tracking: surface type → DETECT → deposit (sensor, q, t, λ, θ)
after tracking:  detection  η_s(λ, θ) = ε_s · QE(λ) · CE(θ)                       linear in the histograms
                 electronics TTS ⊛, transit offset, time walk, SPE charge, pulse, dark rate, threshold, front end
                 → pe · pe_density · expected_waveform · expected_hits · sample
```

- **One sensor model: `lumen:sensor`,** read from a group's value set. Each modelled group has its own value set, keys and histograms (`simulate`
  reads out every modelled group: `Light.groups`; the readout builds each record through each group's index map, a
  every sensor in exactly one group: the groups that share a volume partition its sensors, checked at load). The experiments' value sets are their packages' (lumen-wc, lumen-tpc).
- **Hit angle**: θ = acos(z_local / r_local) of the crossing point on the sensor, in the sensor's own placement
  frame: the hit position on the PMT about its axis, on which the CE table is defined. It is signed; WCSim's |z|
  agrees bitwise on SK and WCTE, whose cathode crossings all have z > 0. Why not WCSim's point: decision 10.
- **Two histograms per group**: charge (sensor × λ knots × θ knots, exact in QE and CE) and time (sensor × one λ band
  × time nodes, scaled per sensor to the charge histogram). The time histogram is weighted by the QE × CE of the
  values the run is given (an operand of the loop, read at each deposit's charge cells: `Grid.weighted`), so its shape
  is that of the detected light at those values and differentiable in them. A QE band table is held at its end bands beyond its edges, as every table is held at its end
  values; a cutoff is an explicit 0 in the data. Grids are static, derived from the components' declared supports and
  the gate, never from current values.
- **Timing**: a physical per-photoelectron TTS plus time walk from the PMT values.

| Output | Content |
|---|---|
| `pe` | expected photoelectrons per sensor |
| `pe_density` | PE per ns on the fine grid: node masses ⊛ (emission profile ⊛ TTS) shifted by the transit offset, plus dark rate |
| `expected_waveform` | `mean (C, n)`, `variance (C, n)` (shot plus electronic noise), `t0`, `dt` (ns), `unit`, channels |
| `expected_hits` | per sensor: PE mean, the probability of a recorded hit, the recording probability of a window of j photoelectrons; what `hits_nll` needs (every hit of a sensor's integration windows) |
| `sample` | a simulated event: PE counts, hits (sensor, t, q), waveforms; not differentiable |
| `hits_nll`, `waveform_nll` | likelihoods for measured hits and waveforms |

Components: TTS (`gaussian`, `emg`, `table`), SPE charge (`gauss_exp`, `table`), pulse (`bigauss`, `exp2`, `table`),
dark rate, threshold (acceptance vs charge), front end (sampling, baseline, noise, integrator). Each has `kernel`,
`moments`, `charfn` or `sample`, and time kernels declare `support`.

A value set is a JSON file (a path), or `<package>:<set>`: the file `<set>.json` in an installed package's
`sensor_values` directory. lumen ships none (`examples/data/toy_pmt.json` is an illustrative set); lumen-wc and
lumen-tpc ship the experiments' sets, each with its provenance.

## Extensibility

| Table | Resolves |
|---|---|
| the sensor model | a group's value set (`values`): `lumen:sensor`, the value-set model (`lumen.sensors.value_sets`, schema `lumen-sensor-values/1`) |
| the build's model table | a material record's optics model (`optics.name`): lumen's own (`build.library.MEDIA`) or a `MediumModel` value passed to the build (`lumen.load(..., parts=)`, a channel's phase function as `lumen.models.lobe_model`); a surface is read by Geant4's reading (`optics.name` "geant4") |

Importers are plain functions; experiment software may write the format directly. Names are `distribution:model`;
duplicates are refused; an unknown model fails closed, naming the distribution. The plug-in protocols (`Lobe`, sensor
components, `Source`) are documented and **provisional**; the way to extend lumen is `docs/howto/extending.md`. A
plug-in needs no Geant4 counterpart, never
branches in Python on a parameter value, never closes over a parameter array and declares every value it reads.

## Validation and tests

| Check | Reference | Pass |
|---|---|---|
| charge | through `simulate`; WCSim (SK, WCTE; ANNIE a reported cross-check with its QE, `response` "qe"); Geant4 via g4ref (WCTE, ANNIE, SBND, MicroBooNE); a reference hit an absorbed photon (lumen's deposit as it is, `charge.response`) | ratios within quoted errors, \|z\| ≤ 3, per-sensor χ²/ndf |
| gradients | through `simulate`: AD against the fixed-reference finite difference (exact per seed), the reweighted value against the value at its own values, AD at the default ε against ε = 1e-6 | K = 16 seeds, the (K−1)/(K−3) null, the reweighted keys the loop reads; reverse against the fixed-reference finite difference |
| geometry import | g4ref `--probe` point location | 100 % |
| overlaps | the 18 g4ref overlap toys through `simulate`; the placement check (with the sibling pairs closer than the contact tolerance, `thin`) | detected fraction and mean time within errors |
| Mie | Geant4's G4OpMieHG tables (`tests/data/g4ref_mie`) | the scattering-angle distribution within its errors |
| facet paths | Geant4 11.3.0's G4OpBoundaryProcess on one interface (`tools/g4ref/boundary.cc`, `tests/data/g4_boundary`) | the last direction's (dz, \|phi\|) distribution within its errors |
| export | Geant4 10.7.4 round trip on toys | `import(export(df)) == df`; Geant4 hits equal |

| Tier | Content | Target |
|---|---|---|
| unit and contract (CI: `.github/workflows/ci.yml`, with the CPU target) | units, rng, laws, lobes, surface types, format and digests, importer fixtures, the builder and its toys, round trips, navigator vs the float64 oracle, declared methods, derivative refusals, no-recompile, readout toys, smoke toys | 1.5-5 min per shard of 4 on 4 cores, CPU |
| physics (closed form) | toys on the navigator: spheres, a WLS slab, coincident pairs, overlaps through the step, border directions, a stuck photon, mesh vs box, the Mie sphere, GPU vs the plain-JAX backend (2.5e-4), GPU vs the CPU target within MC error (`test_cpu_target.py`, also runs without a GPU: the CPU tier) | ~5 min, one GPU |
| reference | charge, geometry, overlaps, export | ~1 min per detector |
| gradients | the gradient check; the per-deposit replay against float64 at fixed random numbers | as measured |

References are reduced products with sha256, read-only in the reference store under `LUMEN_DATA`, listed in
`lumen/compare/references.json` (a detector package passes its own); the Geant4 runner and the reduction live
in `tools/g4ref/`. Check outputs never overwrite reference
products. The report is generated from the JSON records every `python -m lumen.compare` run writes.

## Engineering

| Item | Rule |
|---|---|
| tallymap | pinned by commit in `pyproject.toml` (decision 7); FFMA contraction off (it changes bits) |
| backend | `run.resolve_backend` is the only place backends are decided: `"auto"`, `"cuda"` and `"cpu"` are tallymap's kernel backend (`"auto"`), on the device JAX places the loop on (CUDA on a GPU, C++/OpenMP on the CPU, exact sums on both); `"cuda"` also expects the GPU (`check_device`), `"cpu"` is put on the CPU, inputs there and outputs back (`run.on_device`); `"xla"` is tallymap's plain-JAX backend, the same exact sums, compared with the GPU at 2.5e-4 (XLA's CPU and GPU math differ in the last bits) |
| environment | only `_env.py` reads the environment (decision 18); what it reads is listed in its docstring |
| tooling | src layout, hatchling + hatch-vcs, ruff at 120 columns, pre-commit, mypy on host code, type hints on the public API |
| style, docstrings | `style.md` (numbered rules; the hooks and `pyproject.toml` cite them by number) |
| privacy | non-public detectors live in their own packages, which convert with `lumen.convert`; a pre-commit hook greps content, paths and messages against `LUMEN_PRIVATE_PATTERNS`, kept outside the repo |
| bitwise | results are bitwise reproducible on one CPU or GPU model only |
