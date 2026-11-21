# Developer vocabulary

The words the code uses, for developers; the user glossary is `docs/reference/vocabulary.md` and the architecture is
in `architecture.md`. Terms marked *dev* stay in developer documentation and code; user-facing text (tutorials, error
messages, `__repr__`, public docstrings) uses only the user's words (style rule 10).

| Word | What it is | Where it lives |
|---|---|---|
| **detector file** | `<name>.lumen`, class `DetectorFile` | `lumen.format` |
| **calibration** | `<name>.params.json`, flat: every value it changes from the detector file's | `lumen.format.calibrations` |
| **sensor set** | the groups of sensors, one per detector file | `lumen.format.sensorsets` |
| **GDML names** | the names in parameter keys (`materials/Water/ABSLENGTH`) | parameter keys |
| **recipe** | the small Python file with what GDML cannot say (detector volume, sensors, sensor models) | `lumen.Recipe` |
| **detector volume** | the volume lumen tracks light in | `Recipe(detector_volume=)` |
| **`Params`** | the user's parameters; *dev*: the parameter spec and the packed arrays | `lumen.params` |
| **unit and allowed range** | `params.info(key)` | `lumen.params` |
| **facet path** | Geant4's repeated facet reflections | `surfaces/facets.py` |
| **drawn** | the direction was drawn from a density | `transport/state.py` |
| **reweighting** | draw at reference values, weight by live / reference (`estimator.reweight`) | `transport/estimator.py` |
| **reweighting / moving paths / after tracking** | the gradient method of a key | `Params.info(key).method` |
| **`reference`** | the values the draws are made at | `Operands.reference` |
| **moment** | per cell, sum over photons of the square of each photon's adds (summed over its records): the errors | `Readout.moments(dep, weights)`, `grid.moment_error` |
| **plain-JAX backend**, **CPU target** | tallymap's `"xla"` backend; tallymap's kernel backend compiled with C++/OpenMP | `backend="xla"`; `backend="cpu"` or `"auto"` |
| **deposit record** | *dev*: one deposit as the photon loop's output (sensor, q relative to the row weight, row weight, time, λ, photon, label, feature fields); every readout output is built from them | `run.record_fields`, `readout/fill.py` |
| **per-sensor histograms** | in time; over the axes a sensor model declares; *dev*: `RunResult`, `HistogramLayout` | `Light`, `run.py`, `readout/fill.py` |
| **response applied after tracking** | a readout function on the histograms | `lumen.readout` |
| **`arrivals`**, `mean_arrival_time` | *dev*: the loop outputs `arrival`, `arrival_t` | `Light`, `readout/fill.py` |
| **sensor model** | the response after tracking, read from a group's value set (`SensorGroup.values`, `lumen:sensor`) | `lumen.sensors` |
| **sensor index** | a sensor's position in every per-sensor array (`sensor`) | `det.sensors`, `Photon.sensor` |
| **`rand`** | the step's random numbers | `_core/rng.py` |
| **surface type** | `surface_types`, `stype`; the packed key `"surface_types"` | `surfaces/types.py` |
| **dist**, **max_distance** | distances (m); in transport **t** is a time (ns); inside `geometry/`, `t` is the ray parameter (m) | `transport/step.py` |
| **late / early** | light beyond the time grid's ends (the overflow cells) | `Light.health`, `Histograms` |
| **StepContext** | *dev*: what one step knows before its event | `transport/step.py` |
| **optical class** | *dev*: the surface lookup's compression (PVs that share every crossing) | `build/lookup.py` |
| **`TypeEval`** | *dev*: a surface type evaluated at a crossing (outcome probabilities, widths, amplitudes) | `transport/surface_event.py` |
| **survival cut** | Geant4's R/T cut ahead of the interface | `surfaces/types.py` |
| **expected detection** | the detected share added at every surface hit, not sampled | `transport/step.py` |
| **diagnostics** | per-photon records beside the loop | `lumen.diagnostics` |
| **float32 tolerance** | the band within which two float32 surfaces are one | `rules/navigation.py` |
| **fixed-reference finite difference**, **same-random-number finite difference** | the gradient check's two differences | `lumen.compare.gradients` |
| `python -m lumen.compare charge`, `gradients`, `geometry`, `overlaps` | the checks | `lumen.compare` |
| **readout toys** | closed-form tests of the readout | `tests/unit/readout_toys.py` |
