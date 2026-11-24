# Changelog

Changes a user of lumen can see, newest first. Each entry says what changed, what an old file, key or script does
now, and what to do.

## Unreleased

### Results that change

- **A unified ground surface's sigma_alpha reads up to 1 rad** (was 0.707 rad), G4OpticalSurface's default value
  (nexus's SiPM board mask, for one): four more rows of the unified tables, at their stated accuracy. Above 1 rad the
  surface is still refused: there Geant4's own facet draw departs from the distribution it states and lumen draws
  (`lumen.rules.choices`, `SIGMA_MAX`), by at most 8.5e-4 of the facets at 1 rad and 0.15 at 2 rad.

### Added

- **The navigator's precision**: `lumen.load(..., precision="float64")` runs the navigator in float64: its
  coordinates and frames, CSG tables, roots and intervals, its tolerances, and the photon's position. The physics stays
  float32, and float32 stays the default. float32's tolerance at a decision is the band of the photon's ray, its
  start's coordinates plus the distance travelled, so a coating a few um thick at metres from the origin was stepped
  over or mis-tracked by photons that came from far: NEXT-100's TPB left its S2 light 8 % low at the PMTs and 10 % at
  the SiPMs against nexus. In float64 NEXT-100 agrees with nexus to 0.4 %, at +20 % (S1) and +19 % (S2) time on an
  A100 against a float32 that tracks those coatings, and 3.4 times float32's S2 time on an RTX 2080 Ti. float64 needs
  JAX's 64-bit mode (`JAX_ENABLE_X64=1`), and `load` refuses it otherwise; with the mode on, float32 runs are unchanged.
  The build cache keys a navigator by its precision (docs/dev/decisions.md, 29).
- **Skin sensors**: a `border_into` group's volume that no border sensor enters and whose own skin detects has that
  skin as its sensor, met where Geant4 picks it (entering from the volume's mother, or from a volume without a skin);
  leaving through it detects with the surface's EFFICIENCY, credited to the volume left. nexus's photocathodes and
  SiPMs read so (G4OpticalParameters::SetBoundaryInvokeSD). A `border_into` group that binds nothing, neither a border
  into its volumes nor a detecting skin, is refused (it converted with no sensors).
- **Border surfaces across coincident faces**: a border between volumes that never touch as mother and daughter or as
  siblings was refused; it now holds where they meet across coincident faces, as Geant4 crosses from one straight
  into the other (a coating inside a tube against the gas beside it; a coating inside a mask inside a board against
  the gas, the faces composing through opaque volumes). It is read between their nearest instances, and refused when
  the optics compile if no coincident face joins them. NEXT-100's TPB coatings read.
- **GLISUR facet refraction**: a glisur ground dielectric_dielectric surface reads (it was refused), as Geant4's
  DielectricDielectric draws it: after the REFLECTIVITY / TRANSMITTANCE cut, a facet path whose facets reflect or
  refract with the Fresnel shares of the photon's own state, ending on either side of the surface. Its polish and both
  indices are weighted as a path (`surfaces/<S>/polish` is a key). NEXT-100's TPB surfaces read. Between matched
  indices a facet refraction keeps the direction (`lumen.rules.choices.INDEX_MATCH`); a unified ground
  dielectric_dielectric surface is still refused. lumen keeps the sign of a state's part in the plane of incidence
  where Geant4's DielectricDielectric takes its magnitude (`lumen.surfaces.polar`): unpolarised light is reflected and
  refracted as Geant4 draws it; a definite linear state that meets several facets of one surface is not.
- **The facet paths against Geant4 on one interface**: `tools/g4ref/boundary.cc` (Geant4 11.3.0) and its histograms in
  `tests/data/g4_boundary`, which `tests/unit/test_g4_boundary.py` compares with the lobes' own draws.

### Faster

- **Loading a detector of many placements**: a block of cells, a placement in each and a border per cell (a SuperFGD:
  each cube's core inside its drilled coating, fibers through the cubes), loaded in a time quadratic in its cells.
  A border's touching instances were found by scanning every instance of its two physvols, for each border: they are
  now reached from the physvol with fewer instances, through the other's instances by mother. The sibling sweep met
  every box beside a box along its axis (a whole layer of cubes, and every cube for a fiber along it): it now runs
  within tiles across that axis. A drilled solid's hole plan was redone for each placement: it is found once per
  solid. Loading is linear in the placements, and builds the same detector files, parts and navigator arrays. The
  full SuperFGD (192 x 56 x 192 cubes and their 58k fibers, 4.5 million volumes) converts in 18 min, loads in 6 min
  and builds its navigator in 4 min, in 21 GB of memory; a 24 x 16 x 24 block loads in 5 s (was 230 s).
- **Holed solids in the ownership**: a large union with holes was placed as its pieces each minus every hole (the 61
  pieces of NEXT-100's energy-plane copper plate, 181 leaves each), and a containment proof evaluated every hole of a
  holed solid (NEXT-100's 60 PMTs against the plate's 180-leaf disc). A piece now keeps only the holes whose box meets
  its own, and a proof only those meeting the other solid's box at that pose: the same solids and the same ownership.
  NEXT-100 converts in 73 s and builds its float64 navigator in 66 s (were 18 min each).
- **Re-emission spectra**: a wavelength shift rebuilt its spectrum's sampler and densities from the spectrum table at
  every re-emission, inside the photon loop (NEXT-100's TPB: 17 passes over its 119 knot intervals, held in each
  thread's local memory). They are made once per run, outside the loop, with the step's other operands
  (`Operands.emission`), and a re-emission looks them up: the same draws and weights to float32 rounding (a
  normalisation's sum is rounded in another order). NEXT-100's S1 takes 12.7 us per photon on an A100 (was 13.5) and
  its loop 0.7 kB of local memory per thread (was 4.7 kB).
- **The navigator's BVH walk**: a node's two children were each tested when the walk reached them, the farther pushed
  whether or not the ray met its box: 42 % of the node visits of a NEXT-100 query in its gas volume (1340 daughters)
  were boxes the ray missed, each reading three boxes. The children are now tested at their parent and only those the
  ray meets within the best event so far go on, the stack keeping each entry's distance to drop it once a nearer
  event is found: the same candidates in the same order, the same outputs bit for bit (every query of the traced
  states of NEXT-100, SK, WCTE and a tank). Per query on an A100: NEXT-100's gas volume 90 -> 79 ns (float32), SK
  123 -> 113 ns, the tank 4.3 -> 3.4 ns.
- **The photon loop's registers**: tallymap's kernel kept 168 registers per thread, so 12 warps fit an SM of an A100;
  lumen now asks tallymap for 8 resident blocks per SM (`lumen.run.MIN_BLOCKS`, tallymap's `min_blocks_per_sm`): 64
  registers, the rest spilled to local memory, and the outputs the same bit for bit. Per photon on an A100: NEXT-100
  (float64) -7.5 %, SK -20 %, WCTE -23 %, a tank -11 % (10, 12 and 16 blocks were slower).
- **Relocation after a crossing**: going down into the daughter it had just found holding the ray, the walk tested
  that daughter's solid again, and going up out of a volume it searched that volume's daughters first. Neither is
  done now: the same volumes, the same outputs bit for bit. Per query on an A100: NEXT-100 70 ns (was 82, float64)
  and 33 ns (was 40, float32), SK 104 ns (was 112), WCTE 96 ns (was 105); NEXT-100's S1 takes 10.5 us per photon
  (was 11.6).
- **A solid's leaves along a ray**: the scan for a solid's next membership change computed every leaf's intervals
  along the ray again at each endpoint it passed, twice (for the next endpoint, then for the membership past it). It
  now computes those of its first 8 leaves once per scan and reads them where the window meets the leaf's box (a torus
  leaf, whose intervals depend on the window, is still solved per window): the same intervals, the same outputs bit
  for bit. Per query on an A100: NEXT-100 61 ns (was 70, float64) and 30 ns (was 33, float32), SK 75 ns (was 104),
  WCTE 62 ns (was 96); NEXT-100's S1 takes 9.6 us per photon (was 10.5).
- **A run's health sums**: each photon added its fates to the health's exact sums as it ended, while the other photons
  of its warp, mid-path, waited for those adds. The loop now returns each photon's fates and a launch of their own
  adds them: the same exact sums. Per photon on an A100: SK -16 %, WCTE -2 %.
- **Locating a point among a volume's daughters**: relocating after a crossing, and locating a photon's start, walked
  the volume's BVH of daughter boxes down to every box holding the point (NEXT-100's gas volume: 1340 daughters). Each
  volume's daughters now also have a uniform grid over their boxes (2 cells per daughter, each cell listing the
  daughters whose box meets it once widened by the widest tolerance a relocation tests boxes with), and the point's
  cell gives the candidates: every daughter whose box holds the point is among them and the winner is picked by rank,
  so the same volumes and the same outputs bit for bit. Per photon on an A100, with tallymap's step loop taking a
  warp's steps together (tallymap d1970a7): NEXT-100's S1 6.6 us (was 7.8), SK -9 %, WCTE -12 %, the others within
  2 %; with tallymap 0.8.1's loop, NEXT-100's S1 -15 % but SK, WCTE and MicroBooNE are slower (+34 %, +61 %, +38 %).

## 0.2.1 (2026-11-23)

### Results that change

- **A photon in a closed cavity of absorbing walls now ends.** At a surface whose reflectivity is exactly 0, the
  covered outcome is proposed at a share of the whole distribution, so the photon's reference weight falls and the
  roulette ends it. In 0.2.0 such a photon (inside a PMT whose walls absorb everything) kept its reference weight and
  was followed until the vertex guard, which could take millions of steps. Values and derivatives are unchanged within
  the Monte Carlo error.
- **The detector volume grows to where Geant4 transports the light**: while a photon leaving it enters a volume with
  RINDEX; it stops at a volume that absorbs light on entry. This is decided at conversion: convert a detector again to
  get it. The conversion summary names the volume.
- **The Geant4 export writes Geant4 10.7's scintillation keys** beside Geant4 11's (10.7 ignores the 11 keys and
  produced no scintillation). A GDML that has both is read as its Geant4 11 keys, giving the same light.

### New

- **Event displays on the detector's own layout** (`lumen.tools.layout`, `lumen.tools.plot.display`): planes for a
  TPC's light detectors, an unrolled cylinder for a tank, an equal-area sphere otherwise.
- **Many Geant4 events against lumen** (`lumen.compare.event.compare_events`, `lumen.tools.plot.compare_events`):
  pulls per sensor and per event, by display surface and sensor type. A group's per-event pulls use only the events in
  which it expects light. `tools/g4ref/events.py` makes the Geant4 events.

### Faster

- The first `simulate` on a warm cache of a detector with 43,212 sensors went from 384 s to 43 s.

### Pinned

- tallymap v0.8.1 (was v0.8.0).

## 0.2.0 (2026-11-20)

lumen 0.2 is a rewrite of the photon loop's outputs, of how a detector is described and of where the experiments'
detectors live. A 0.1 detector file, value set or calibration does not load: convert the detector again and save the
calibration again. The git history keeps the detail of every step.

### Results that change

- **Exact sums.** Every histogram cell, tangent and reverse cotangent is the correctly rounded float32 of the exact sum
  of what was added to it (0.1's fixed point set a tangent below its quantum to 0). A cell is exact or NaN, counted in
  `light.health.nonfinite`. `backend="xla"` sums exactly too.
- **The roulette's ε is 0.01** (was 0.1), on the photon's weight relative to its emission row; one ε, in `Controls`.
  Values and derivatives move within the Monte Carlo error; derivatives of surface reflectivities get tighter.
- **`speed="fast"` is the default**: in a Rayleigh-only medium the photon walks its own scatterings inside a ball the
  geometry proves empty. Exact in law, other random numbers; `speed="exact"` is the plain loop. A probe diagnostic
  runs exact; `Light.provenance` says which ran.
- **Other random numbers, the same law**: the unpolarised linear state is drawn like every other draw, and one
  dielectric reflectance serves every Fresnel branch.
- **One meaning of "detected".** `arrivals`, the histograms and the checks count the same deposit; under a model that
  applies the QE after tracking, SK's, WCTE's and ANNIE's arrivals are 1/0.68 of 0.1's (photoelectrons unchanged).
- **As in Geant4**: facet loops uncapped (G4OpBoundaryProcess); Cherenkov light spans its medium's RINDEX table; a
  user GROUPVEL times the photons.
- **No time cut**: every photon is followed until the roulette, absorption, escape or the vertex guard ends it, and
  every output counts all the light (a readout's gate sets only the time histogram's extent; light before or after
  it is in the early / late cells). `Controls.t_end` and `health.retired` are gone.
- **A time spread's fit range is derived** ([1/2, 2] times the set's width): time grids change, charges do not.
- **Errors from per-photon moments; no replicas**: `simulate` has no `replicas` and the charge histogram no replica
  axis (charge is (S, *axes)). `arrivals_error` and `lumen.readout.pe_error(light)` (at the run's values only) come
  from each cell's per-photon second moment (`Histograms.charge2`, `lumen.readout.grid.moment_error`): unbiased for
  i.i.d. photons, its own scatter about 15 % on a cell of tens of photons (41 % from 4 replicas). Arrivals move by the
  replica sum's float32 rounding. The checks (`compare.charge`, `overlaps`, `event`) read the moments of a
  `photons.Detected`; the gradient check's `--replicas` is `--seeds`.
- **`chunk` is a memory lever**: by default at most 2^20, smaller when the rows or the records do not fit (a launch
  whose records do not fit runs again, smaller); a given chunk whose records do not fit is refused.

### Added

- **One run entry**: `lumen.simulate(det, source, params, *, ..., diagnostics=(), truth=None, records=None)` returns a
  `Light`: `arrivals` and their error, `health` (a `Health` record, a pytree: `lost`, `abandoned`, `escaped`,
  `truncated`, `records`, `records_dropped`, `records_mismatch`, `nonfinite`, `invalid`), each modelled group's
  histograms (`light.group(name)`), `diagnostics` and `truth` (one row per detected photon, `lumen.readout.Truth`); an
  event axis with `events=E` (`light.event(e)`), labels on sources.
- **Deposit records.** The photon loop emits one record per deposit and the readout builds every output from them.
  Under `jax.jit` or `jax.vmap` their number is given, `simulate(records=)`, bounded by `health.records` of the run
  with values (`RecordsUnknown` without it). The readout runs per block of sensors that fits the device: Hyper-K's and
  DUNE FD1's time profiles run on an 11 GB GPU. A value run and the primal of its derivative agree bit for bit.
- **`simulate(..., reference=params0)`**: every random choice at the reference values, the light at the values given,
  so a finite difference converges to the derivative at a fixed seed.
- **Sensors**: `lumen.SensorGroup` (keyword-only past its name); several `border_into` groups; groups selecting on the
  id map's columns (`where=`); a readout gate per timed group; `Detector.sensor_table()`, a `SensorTable` in the
  arrivals' order; `lumen.readout.pe(light, params)`.
- **Sources**: `CherenkovSteps` and `ScintillationSteps` (Geant4's steps, energies in MeV), `FromArrays`,
  `@lumen.source`, `Point` and the other built-ins.
- **Physics**: G4OpWLS's delta time profile and WLSMEANNUMBERPHOTONS, Geant4 11's third scintillation component. Every
  re-emission (bulk WLS, a surface's) samples one emission record; the WLS spectrum is a parameter with a derivative.
- **Parts are values, no registries**: passed at load by their path, `lumen.load(path, parts={...})`: a medium model
  (`@lumen.medium_model`, `"materials/<M>"`) or a channel's phase function (`lumen.media.phase_function`,
  `"materials/<M>/<channel>"`, its keys `materials/<M>/<channel>/<name>`; without the polarised pair it declares
  `unpolarised=True`). A detector file records a part's name, never code; a run records the parts it read
  (`Light.provenance`'s `"parts"`, with "unpolarised" and "fast walk off"). Geant4's surface models are interaction
  values (parameters, continuing shares, deposits) read by one kernel, not hooks in 0.2 (docs/howto/extending.md).
- **Inputs**: Geant4-10 scintillation keys are translated wherever they enter (GDML, recipe, YAML), checked against
  Geant4 10.7.4's reading and recorded in the provenance, or refused (`TranslationRefused`). Unchecked or
  unsupported inputs are refused at conversion, naming what to change; nothing falls back silently.
- **YAML detectors**, converted like a GDML (`lumen.convert` takes a GDML with its recipe, or YAML files).
- **`lumen.tools`**: `Space`, `jacobian`, `fisher`, `fit` (Fisher scoring on a Poisson likelihood), `descend`,
  `profile`, `coverage`, `scan`, `check_gradient`, `lumen.tools.plot` (extra `plot`).
- **`lumen.compare`**, one command: `python -m lumen.compare charge|geometry|gradients|overlaps|report|export|gate`.
  `gate --package PKG` is the bitwise Monte Carlo gate over a package's manifest. The Geant4 comparisons gate on the
  overlap-free export of the detector's GDML (`export`), the original GDML's run reported beside it.
- **Reference data is fetched** from its release asset into `LUMEN_DATA` (default under `LUMEN_CACHE`) on first use and
  checked by sha256; `LUMEN_ASSETS` names a mirror of the assets.
- **For detector packages**: `lumen.DetectorEntry` (a package's `DETECTORS`, `lumen.convert("package:name")`) and
  `lumen.testing` (value-set, readout-gate and manifest checks, the notebooks' runner).
- **The CPU target**: without a GPU the photon loop compiles to C++ with OpenMP (`backend="cpu"`).
- **Overlaps are resolved statically**: a point belongs to one owner, level by level; the conversion reports
  overlapping placements, and the GDML export writes the resolution out.

### Migrating a 0.1 script

- **Formats**: detector files `lumen-detector/5`, calibrations `lumen-calibration/1` (tied to their detector file by
  its digest), value sets `lumen-sensor-values/1`. Older ones are refused: convert again, save calibrations again.
- **No registries.** `lumen.plugins`, the detector registry and `lumen.load("sk")` are gone: a detector is its file
  (`lumen.convert(...)` once, `lumen.load(path)`); one sensor model, `lumen:sensor` (was `lumen:pmt`).
- **A GDML that Geant4 refuses is refused**: run `python -m lumen.gdml.repair SRC DST` and convert `DST`. A detector
  volume placed more than once is refused, naming the common mother of its placements.
- **`simulate` takes a `lumen.sources.Source` only**; `Geant4Steps` is `CherenkovSteps` or `ScintillationSteps`
  (`edep` in MeV); `simulate_truth` and `lumen.diagnostics.run` are `simulate(truth=)` and `simulate(diagnostics=)`.
- **Records by attribute**: `light.health.lost` (was `health["lost"]`; `out_of_range` is `invalid`),
  `det.sensor_table().name`; public records are built by keyword. `Light.histograms` is `light.group(name)`,
  `lumen.tools.pe` is `lumen.readout.pe`.
- **No source-position derivative**: the moving-source method is removed for 0.2. A tangent on a field that moves where
  the light starts (a position, a direction, a photon list's rows, a source function's `method="moving_paths"`
  parameter) raises `GradientNotAvailable`; use a finite difference. With it go `Photons.drawn` and `Photons.dlogp`,
  `Source.drawn_direction`, `FromArrays(drawn=, dlogp=)`, `lumen.source(density=)` and `lumen.sources.grad_logpdf`
  (drop them; a direction density that depends on a parameter enters by `reweight`), and `check_source`'s
  `dlogp_error`.
- **Gone**: layered surfaces and thin-film photocathodes (`lumen:wls_film`, `lumen:thin_film`, `lumen:complex_index`,
  ...), the weight scale and `SumOverflow`, `CallableSource`, `lumen.validate` (now `lumen.tools` and
  `lumen.compare`), YAML writing and the Python builder, `lumen.gdml.import_gdml` (use `lumen.convert`).
- **`import lumen` no longer turns on JAX's persistent compilation cache** (set `JAX_COMPILATION_CACHE_DIR` yourself);
  every cache lives under `LUMEN_CACHE` (never HOME).
- **Pinned**: JAX 0.6.2 and tallymap v0.8.0 (installed from its git tag).

### The detectors left lumen

The experiments' recipes, value sets, reference manifests and comparisons moved to
[lumen-wc](https://github.com/OmarAlterkait/lumen-wc) (Super-K, WCTE, ANNIE, Hyper-K, IWCD) and
[lumen-tpc](https://github.com/OmarAlterkait/lumen-tpc) (SBND, MicroBooNE, ICARUS, DUNE):
`lumen.convert("lumen_wc:sk", out=..., data=...)`; a value set is a path or `<package>:<set>` (`lumen_wc:sk_r3600`).
lumen ships none (`examples/data/toy_pmt.json` is the toy).

## 0.1.3 (2026-11-11)

- tallymap 0.6.0, the final release (rc1 plus four fixes in its CPU backend and process spawning), installed from
  its tag v0.6.0; lumen accepts tallymap 0.6.x. The photon loop's results on the GPU are unchanged bit for bit.

- The developer documentation moved to `docs/dev/` (entry page `docs/dev/README.md`); this file replaces
  `docs/dev/migration.md`.
- Removed, unused by lumen and undocumented: `lumen.readout.grid.window_end`; `lumen.readout.hits.window_mass`,
  `window_horizon`, `poisson_nll_dark` and `first_hit_terms(horizon=)`; `Navigator.query(counts=, max_rounds=)`;
  `lumen.rules.optics.key_report` and `rt_sensor`, `lumen.validate.oracle.oracle_first_visible` and `mismatch`,
  `lumen.build.scene.check_partition`, `lumen.surfaces.polar.jones`, `lumen._core.safe.safe_norm` and
  `Compiled.is_single_convex` (test helpers, now in `tests/unit/helpers.py`).
- The geometry lowering's `dissolve` and `keep` options are removed: no build used them. The geometry check
  (`python -m lumen.validate.geometry`) now checks the geometry as simulated instead of a dissolved one.

Changed:

- Every material without a refractive index shares one opaque medium, by a general rule; the SBND and MicroBooNE
  recipes no longer bind theirs. The light is unchanged bit for bit on the five shipped detectors and no parameter key
  changes. A detector file converted again from SBND's or MicroBooNE's GDML differs in `materials.json` (those
  materials name the `geant4` optics model, not `lumen:opaque`), so a calibration saved against an earlier conversion
  of them is refused as made for another detector file: convert again and recalibrate, or keep the earlier file.

Fixed:

- JAX's persistent compilation cache stayed empty: JAX writes only programs that took over 1 s to compile, and each of
  lumen's compiles faster than that (the photon loop's heavy part is tallymap's CUDA kernel, cached by tallymap). lumen
  now caches every program, unless `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` is set. On the toy tank a new
  process's first `simulate` goes from about 22 s to about 9 s.

- A surface type with a value fixed by its model kept only the outcomes it had with every other parameter at 0.5, so
  under Geant4's chained cuts a unified dielectric_metal surface with a fixed EFFICIENCY lost its backscatter and
  Lambertian outcomes (no shipped detector had such a surface).
- A source-position derivative on a very large detector (about 40,000 sensors) could run out of GPU memory in the
  clearance estimate about the source: its rays now run in batches sized to a memory budget
  (`lumen.geometry.clearance.BYTES`, 1 GiB). The clearance values are unchanged.

## 0.1.2 (2026-11-11)

New:

- **Extending lumen** (`docs/howto/extending.md`, `examples/extending.py`): `lumen.Param` declares a parameter that
  becomes a `Params` key; a source, a sensor response, a medium model or a surface model is one function registered
  by name, with its parameters as keys; `lumen.validate.check_gradient` checks any derivative against finite
  differences on the same random numbers.
- `simulate(controls=, diagnostics=)`: the step limits and the per-photon diagnostics that `run.run` already took.
- One-line install from git: `pip install "lumen[cuda] @ git+https://github.com/OmarAlterkait/lumen.git"`. tallymap
  is installed from its tag (v0.6.0rc1); the `cuda` extra is `jax[cuda12]`.
- `lumen.validate`, `lumen.extend`, `lumen.diagnostics` and `lumen.plugins` are reachable after a bare `import lumen`.

Changed:

- **`FromArrays(drawn=True)` and a derivative that moves its photons.** With `wrt=["source/position"]` such a list now
  raises `GradientNotAvailable` unless it carries its direction density's gradient, `dlogp=` (N, 3). Before, lumen
  assumed every drawn list was isotropic: right for an isotropic list, silently wrong for any other (a cos^8 LED's
  position derivative was 5 times too small). To do: for isotropic directions pass `dlogp=jnp.zeros((N, 3))`, which
  gives the 0.1.1 result; for any other density pass grad_d log p (`lumen.sources.grad_logpdf`), or write the source
  as a function (`@lumen.source`). Every other source, detector file, sensor set and calibration is unchanged.
- **`Recipe.note` removed.** Nothing read it. A recipe written with `note=...` fails with a `TypeError`: drop the
  argument. WCSim's recipes also lost their `cathode_model` decision (a check that could not fail) and the sensor
  group's `name_prefix` setting (it equalled the default). Converted detector files are unchanged except the sensor
  set's JSON, which no longer lists `name_prefix`.
- **A border that starts outside the detector volume is dropped**, with the same warning as a border wholly outside,
  instead of raising "no adjacent world instances": Geant4 consults a border only for a photon in its first volume,
  and lumen tracks no light outside. Every detector that converted before converts to the same file. The reverse
  case, the detector volume into such a neighbour, still raises.
- A source function draws all its samples at once, so photon i is the same whatever the chunk size.

## 0.1.1 (2026-11-11)

Changed:

- The importer reads only Geant4's optical material keys by default. Any other material key is kept inert, with one
  grouped `InertKeysWarning`; `Recipe(material_keys=)` still reads more.
- A user GROUPVEL is no longer refused. It is not read, with a warning: photons are timed by RINDEX's group index
  (see [known limits](docs/explanation/known-limits.md)).
- Repair warnings are grouped by repair and object, with a count, and given once per conversion. Every occurrence
  stays in the detector file's report.
- `lumen.load(calibration=None)` is the nominal calibration.
- `simulate` refuses a callable source (or an array of photons) whose position is differentiated, with
  `GradientNotAvailable`; before, the derivative came out about 140 times too small, silently.
- `simulate` rescales very small source weights (largest below 2^-6) by an exact power of 2 and scales the outputs
  back. A source normalised to one photon was rouletted at its first event and rounded by the GPU's fixed-point sums.
  Weights of order 1 run unchanged, bit for bit.

Documentation: the health keys (`docs/reference/errors.md#health`), lost light at a junction of touching volumes,
split PMT interiors and `lumen:complex_index`, a large-detector timing, `CITATION.cff`.

## 0.1.0 (2026-11-10)

The first public release. Files and keys from the pre-release tag `0.0.1` are still read, with these renames:

| 0.0.1 | 0.1 | What an old file or key does |
|---|---|---|
| `surfaces/<S>/value` | `surfaces/<S>/sigma_alpha` (unified, rad) or `surfaces/<S>/polish` (glisur) | refused, naming the new key |
| a counting group's setting `eff`, key `sensors/<group>/eff` | `efficiency`, `sensors/<group>/efficiency` | a sensor set with `eff` is refused, naming the new setting; convert again |
| `Point(x=)`, `source/x` | `Point(position=)`, `source/position` | rename in scripts; positional use is unchanged |
| `SensorGroup.model` naming the surface reader (`lumen:geant4`, `lumen:counting`) | `SensorGroup.surface`; `model` is only the sensor model (`lumen:pmt`) | a `lumen-sensors/1` file is read as `lumen-sensors/2` |
| `build_options` in `detector.json` | removed (every build rewrite is a rule) | an empty one is read; a non-empty one is refused |
| health key `late` | gone (no time cut); `late` / `early` the time grid's overflow | – |
