# Changelog

Changes a user of lumen can see, newest first. Each entry says what changed, what an old file, key or script does
now, and what to do.

## Unreleased

- The developer documentation moved to `docs/dev/` (entry page `docs/dev/README.md`); this file replaces
  `docs/dev/migration.md`.
- Removed, unused by lumen and undocumented: `lumen.readout.grid.window_end`; `lumen.readout.hits.window_mass`,
  `window_horizon`, `poisson_nll_dark` and `first_hit_terms(horizon=)`; `Navigator.query(counts=, max_rounds=)`;
  `lumen.rules.optics.key_report` and `rt_sensor`, `lumen.validate.oracle.oracle_first_visible` and `mismatch`,
  `lumen.build.scene.check_partition`, `lumen.surfaces.polar.jones`, `lumen._core.safe.safe_norm` and
  `Compiled.is_single_convex` (test helpers, now in `tests/unit/helpers.py`).
- The geometry lowering's `dissolve` and `keep` options are removed: no build used them. The geometry check
  (`python -m lumen.validate.geometry`) now checks the geometry as simulated instead of a dissolved one.

Fixed:

- A surface type with a value fixed by its model kept only the outcomes it had with every other parameter at 0.5, so
  under Geant4's chained cuts a unified dielectric_metal surface with a fixed EFFICIENCY lost its backscatter and
  Lambertian outcomes (no shipped detector had such a surface).

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
| health key `late` | `retired` (a photon stopped past `t_end`); `late` / `early` is light outside the time grid | – |
