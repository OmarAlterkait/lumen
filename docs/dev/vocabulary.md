# Developer vocabulary

lumen's words (the glossary) against the terms of the code it was ported from, its design studies and tallymap.
Terms marked *dev* stay in developer documentation and code; user-facing text (tutorials, error messages,
`__repr__`, public docstrings) uses only the right-hand words, checked by `tools/hooks/jargon.py`.

| Old / internal term | lumen's word | Where it lives |
|---|---|---|
| description | **detector file** (`<name>.lumen`, class `DetectorFile`) | `lumen.format` |
| overlay, parameter set, ParamSet | **calibration** (`<name>.params.json`) | `lumen.format.calibrations` |
| sensor-set overlay | **sensor set** (groups of sensors, structure only) | `lumen.format.sensorsets` |
| authority names | **GDML names** (`materials/Water/ABSLENGTH`) | parameter keys |
| recipe | **recipe**: the small Python file with what GDML cannot say (detector volume, sensors, sensor models) | `lumen.Recipe` |
| active LV | **detector volume** | `Recipe(detector_volume=)` |
| NamedParams, ParamSpec, packed params | **`Params`** (user); *dev*: the parameter spec and the packed engine arrays | `lumen.params` |
| Domain | **unit and allowed range** (`params.info(key)`) | `lumen.params` |
| vertex chart, chart, ChartState, NoChart | **moving-source method** (`MovingSourceState`, `FixedSource`) | `transport/moving_source.py` |
| hold, held vertex | **pinned vertex** | docs, *dev* |
| anchor | **re-pinning at each vertex** | *dev* |
| detached-sampling score, magic | **reweighting** (draw at reference values, weight by live/reference) | `transport/estimator.py` |
| weight / motion / readout parameter | gradient method **reweighting / moving paths / after tracking** | `Params.info(key).method` |
| `params_ref` | **`reference`** | `simulate(reference=)` |
| slots | **replicas** (independent sub-runs that give the errors) | `simulate(replicas=)` |
| lanes, padded engine | (gone) | – |
| lockstep backend | **CPU backend** (tallymap's plain-JAX backend) | `backend="auto"` |
| tallymap item, ItemSpec | *dev*: the per-photon loop body handed to tallymap | `transport/item.py` |
| tally, timed tally, charge tally | **per-sensor histograms** (in time; in wavelength and hit position) | `Light` |
| fold | **response applied after tracking** | `readout/response.py` |
| Q, Tw, `photons` | **`arrivals`**, `mean_arrival_time` | `Light` |
| tag | **sensor index** | `det.sensors` |
| `u` (deposit coordinate) | **hit coordinates** on the sensor | `SensorModel.hit_coordinates` |
| `u` (uniforms) | **`rand`** (the step's random numbers) | `_core/rng.py` |
| kind, Kind | *dev*: **legacy surface type** (B0's 16); new code has **surface models** made of factors | `surfaces/_legacy.py` |
| optical class | *dev*: the surface lookup's internal compression (PVs that share every crossing) | `build/surfaces.py` |
| SurfaceKernel | *dev*: the compiled surface model (outcome probabilities, lobes, amplitudes) | `surfaces/kernel.py` |
| Sheet | **SurvivalCut** (Geant4's R/T cut ahead of the interface) | `surfaces/compose.py` |
| Rao-Blackwellised deposit | **expected detection** (the detected share is added at every surface hit, not sampled) | *dev* |
| probes | **diagnostics** | `lumen.diagnostics` |
| band | **float32 tolerance** | `rules/navigation.py` |
| FDX, CRN FD | **fixed-reference finite difference**, **same-random-number finite difference** | `lumen.testing` |
| E1 | **charge check** (`lumen validate charge`) | `lumen.validate` |
| D1 | **gradient check** (`lumen validate gradients`) | `lumen.validate` |
| G1 / G2 | **geometry-import check** / **overlap check** | `lumen.validate` |
| S1 | **speed check** | `lumen.validate.speed` |
| R1 / R2 / R3 | **WCSim digit check** / **SBND waveform check** / **sensor-set provenance check** | `lumen.validate` |
| T1-T9 | **readout toys** | `tests/unit` |
| G-e1, G-d1, G-bit, G-path | *dev* port gates **G-charge, G-gradient, G-bit, G-path** (§15) | §15 |
| B0, Phase A/B/C | *dev*: the baseline (`rebuild/next` + item-4 + d1) and the port phases | §15 |
