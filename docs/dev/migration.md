# Migration notes

Changes to what is stored on disk or typed as a key, newest last. Each says what changed, what an old file or key
does now, and what to do.

## Detector files: `build_options` removed

`detector.json` no longer has a `build_options` entry: every build rewrite is a rule (`lumen.build.import_rules`).
A detector file written before reads as before when its `build_options` is empty, which is every converted
detector's; a non-empty one is refused, naming the options. To do: nothing, or convert the detector again.

## Parameter keys: the opticalsurface's value

`surfaces/<S>/value` is now `surfaces/<S>/sigma_alpha` for a unified surface (rad, at least 0) and
`surfaces/<S>/polish` for a glisur one (dimensionless, in [0, 1]): the name Geant4's GetFacetNormal reads it as
(`lumen.rules.optics.value_key`). Only a dielectric_metal ground surface has the key. A calibration or a `Params.set`
with the old key is refused, naming the new one. To do: rename the key in calibrations (`<name>.params.json`) and
scripts. The detector files do not change (the key is derived at load).

## Sensor sets: a counting group's `eff` is `efficiency`

An `any_entry` (`lumen:counting`) group's setting `eff` is now `efficiency`, so its parameter key is
`sensors/<group>/efficiency` (it was `sensors/<group>/eff`). A sensor set (`sensors/<set>.json`) with the old setting
is refused, naming the new one. To do: convert the detector again (the shipped recipes write `efficiency`), or rename
the setting in the sensor set's JSON and the key in calibrations.

## Sources: a point's `x` is `position`

`Point` and `ScintillationPoint` take `position` (m) instead of `x`, so the parameter key is `source/position` (it
was `source/x`), as in `simulate(..., wrt=["source/position"])`. Positional use (`Point((0, 0, 0), ...)`) is
unchanged. `FromArrays` keeps the field names of the photons it is given (`x`, `d`, `t`, `lam`, `w`). Nothing on disk
changes. To do: rename `x=` and `source/x` in scripts.

## Sensor sets: schema `lumen-sensors/2`, the surface reader apart from the sensor model

A sensor group named one `model` for two things: how its detecting surface behaves during tracking (`lumen:geant4`,
`lumen:counting`) and the sensor model after tracking (`lumen:pmt`). Schema `lumen-sensors/2` keeps them apart:
`surface` is the surface reader (the third argument of `SensorGroup`, as before) and `model` the sensor model (None
for none). A `lumen-sensors/1` file is read as /2 says it: `model: "lumen:pmt"` becomes `surface: "lumen:geant4"`,
`model: "lumen:pmt"`; any other name becomes the surface reader with no sensor model. To do: nothing for files
(convert again to write /2); in scripts, `dataclasses.replace(group, model="lumen:pmt", defaults=...)` is unchanged,
and a group built with `model=` as a keyword for the surface reader takes `surface=`.

## Sources: `FromArrays(drawn=True)` and a position derivative (0.1.2)

The one change of behaviour in 0.1.2. A `FromArrays(..., drawn=True)` list given a derivative that moves its photons
(`wrt=["source/x"]`) now raises `GradientNotAvailable` unless it carries its direction density's gradient,
`dlogp=` (N, 3). Before, lumen assumed every drawn list was isotropic: right for an isotropic list, silently wrong for
any other (a cos^8 LED's position derivative was 5 times too small). To do: for isotropic directions pass
`dlogp=jnp.zeros((N, 3))` and the result is the one 0.1.1 gave; for any other density pass grad_d log p
(`lumen.sources.grad_logpdf`), or write the source as a function (`@lumen.source`, docs/howto/extending.md). Every
other source (`Point`, the scintillation and Cherenkov sources, `Geant4Steps`) and every detector file, sensor set and
calibration is unchanged.
