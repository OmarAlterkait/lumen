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
