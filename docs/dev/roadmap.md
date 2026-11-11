# Roadmap

What is not built yet. A release that builds an item removes it from here and records it in `CHANGELOG.md`.

## Planned for 0.1, not built

- `lumen.estimate`, a cost estimate before a run.
- `Light.check()` and an eager warning on bad health.
- `Params.sensor(id, field)`.
- `simulate(reference=)`: drawing at values other than the live ones (the loop has `Operands.reference`; the public
  call does not take it).
- The `lumen convert` and `lumen doctor` commands (the CLI has `lumen validate` only).
- A hit-map and waveform plot module.
- Named error types for the first-hour guide's cases (`RangeError`, `NoSensorsMatched`, `ModelNotInstalled`,
  `FormatVersionError`, `IncompleteResultError`, `GradientSetupError`, `MemoryEstimateError`); today these raise
  `ValueError`, `MemoryError` or `GradientNotAvailable`.

## 0.2

- Factorised surfaces and the public plug-in kit, on the registered-model API of 0.1.2.
- Several modelled sensor groups (PMT types; SBND's PMTs and X-ARAPUCAs together).
- The hit point on the sensor in the deposit; a declared CE coordinate in value sets; a photocathode model.
- Electronics other than PMTs: SiPMs for X-ARAPUCA, LAPPDs with strips, afterpulses, crosstalk, saturation.
- Deposit labels (labelled histograms and arrival records, history bits); the event axis and sensor subsets.
- A photon reflected at a junction of touching volumes continues in the volume it came from (today counted `lost`).
- A user GROUPVEL read as Geant4 reads it (today inert, with a warning).
- Refractive-index derivatives; the shadow-edge term of the moving-source method.
- Bulk `WLSMEANNUMBERPHOTONS` and WLS2; LUT, LUTDAVIS, dichroic and latent-facet finishes.
- Tracks and LAr recombination; geometry calibration.
- One float32 tolerance form for the navigator, the overlap check and the step (`rules/navigation.py` keeps three
  sites and two forms).
- A Python geometry builder for toy detectors without GDML; a viewer.
- Dropping the 0.0.1 migration paths (the `eff` refusal, the `lumen-sensors/1` reader, the `surfaces/<S>/value`
  message, the `build_options` refusal).
