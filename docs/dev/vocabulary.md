# Developer vocabulary

lumen's words (the glossary) against the terms of the code it was ported from, its design studies and tallymap.
The design is in `docs/dev/design.md`. Terms marked *dev* stay in developer documentation and code; user-facing text (tutorials, error messages,
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
| vertex chart, chart, ChartState, NoChart | **moving-source method** (`state.MovingSource`) | `transport/moving_source.py` |
| hold, held vertex, held end | **pinned vertex**, **pinned end** (never "held") | `transport/moving_source.py`, `transport/step.py` |
| chain | only the **moving-source chain** (`moving_source.Chain`): its sensitivities to the free coordinates u | `transport/moving_source.py` |
| chain (GLOSSY), chain lobe | **facet path** (Geant4's repeated facet reflections) | `surfaces/facets.py` |
| chain (calibrations) | the calibration's **parents** | `format/calibrations.py` |
| free (the moving-source flag) | **drawn** (the direction was drawn from a density) | `transport/state.py` |
| anchor | **re-pinning at each vertex** | *dev* |
| detached-sampling score, magic, magic box | **reweighting** (draw at reference values, weight by live/reference; `estimator.reweight`) | `transport/estimator.py` |
| weight / motion / readout parameter | gradient method **reweighting / moving paths / after tracking** | `Params.info(key).method` |
| `params_ref` | **`reference`** | `Operands.reference` |
| slots, `slot` | **replicas** (independent sub-runs that give the errors) | `simulate(replicas=)`, `PhotonLoop.emits(dep, replica, grid)` |
| lanes, padded engine | (gone) | – |
| lockstep backend | **CPU backend** (tallymap's plain-JAX backend) | `backend="auto"` |
| tallymap item, ItemSpec | *dev*: **`PhotonLoop`**, the per-photon loop body handed to tallymap | `transport/item.py` |
| tally, timed tally, charge tally, `Tally`, `TallyShape` | **per-sensor histograms** (in time; in wavelength and hit angle); *dev*: `RunResult`, `HistogramShape` | `Light`, `run.py`, `transport/item.py` |
| fold | **response applied after tracking** | `lumen.readout` |
| Q, Tw, `photons` | **`arrivals`**, `mean_arrival_time`; *dev*: the loop outputs `arrival`, `arrival_t` | `Light`, `transport/item.py` |
| tag | **sensor index** (`sensor`) | `det.sensors`, `Photon.sensor` |
| `u` (uniforms) | **`rand`** (the step's random numbers) | `_core/rng.py` |
| `u` (moving source) | the chain's two **free coordinates** u (the maths keeps the letter) | `transport/moving_source.py` |
| kind, Kind, kidx, kernel (surface) | **surface type** (`surface_types`, `stype`; the packed key `"kernel"` until it is renamed) | `surfaces/_legacy.py` |
| t (a distance), t_if, t_max (m) | **dist**, **max_distance** (m); in transport **t** is a time (ns). Inside `geometry/`, `t` is the ray parameter (m) | `transport/step.py` |
| late (health) | **retired** (a photon stopped past `t_end`); **late / early** is light outside the time grid | `Light.health`, `Histograms` |
| Vertex (the step bundle) | *dev*: **StepContext** (what one step knows before its event) | `transport/step.py` |
| optical class | *dev*: the surface lookup's internal compression (PVs that share every crossing) | `build/surfaces.py` |
| SurfaceKernel | *dev*: a surface type evaluated at a crossing (`TypeEval`: outcome probabilities, widths, amplitudes) | `transport/surface_event.py` |
| Sheet | **survival cut** (Geant4's R/T cut ahead of the interface) | `surfaces/_legacy.py` |
| Rao-Blackwellised deposit | **expected detection** (the detected share is added at every surface hit, not sampled) | *dev* |
| probes | **diagnostics** | `lumen.diagnostics` |
| band | **float32 tolerance** | `rules/navigation.py` |
| FDX, CRN FD | **fixed-reference finite difference**, **same-random-number finite difference** | `lumen.validate.gradients` |
| E1 | **charge check** (`lumen validate charge`) | `lumen.validate` |
| D1 | **gradient check** (`lumen validate gradients`) | `lumen.validate` |
| G1 / G2 | **geometry-import check** / **overlap check** | `lumen.validate` |
| R1 / R2 / R3 | **WCSim digit check** / **SBND waveform check** / **sensor-set provenance check** | `lumen.validate` |
| T1-T9 | **readout toys** | `tests/unit` |
| G-e1, G-d1, G-bit, G-path | *dev* port gates **G-charge, G-gradient, G-bit, G-path** | port records (outside the repo) |
| B0, Phase A/B/C | *dev*: the baseline (`rebuild/next` + item-4 + d1) and the port phases | port records (outside the repo) |
