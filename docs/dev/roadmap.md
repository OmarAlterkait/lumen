# Roadmap

What is not built yet, with no release planned. A release that builds an item removes it from here and records it in
`CHANGELOG.md`.

- Refractive-index derivatives; source-position derivatives (the moving-source method, removed for 0.2).
- WLS2 (G4OpWLS2); the LUT, LUTDAVIS, dichroic and latent-facet finishes.
- Sensor electronics other than PMTs: SiPMs (X-ARAPUCA), LAPPDs with strips; afterpulses, cross-talk, saturation.
- Tracks and LAr recombination; geometry calibration.
- Open extension points: surface models, readout components, features, time profiles, re-emitting media.
- `lumen.estimate` (a cost estimate before a run), `Light.check()` and an eager warning on bad health,
  `Params.sensor(id, field)`.
- A viewer; `lumen convert` and `lumen doctor` commands (the command line has `python -m lumen.compare` only).
