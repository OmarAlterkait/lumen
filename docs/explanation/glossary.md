# Glossary

The words lumen uses in its API, messages and documentation.

| Word | Meaning | Where you meet it |
|---|---|---|
| **detector file** | A detector saved in lumen's own format: a `<name>.lumen/` directory of JSON and `.npy` files holding what Geant4 reads from the GDML plus the recipe's decisions. Once converted, no GDML is needed. | `lumen.format`, `DetectorFile` |
| **calibration** | A file of parameter values (`<name>.params.json`) that replaces values of a detector file by name, without editing it. | `lumen.format.calibrations` |
| **sensor set** | The groups of sensors of a detector: which volumes are sensors, how a photon is counted, which sensor model reads them. Structure only, no numbers. | `lumen.format.sensorsets` |
| **GDML names** | Parameter keys built from the names in the GDML, such as `materials/Water/ABSLENGTH`. | parameter keys |
| **recipe** | The small Python file that says what GDML cannot: the detector volume, the sensors and their models. | `lumen.Recipe` |
| **detector volume** | The volume that is simulated. Light leaving it is absorbed. | `Recipe(detector_volume=)` |
| **`Params`** | The one mapping of every parameter value, keyed by GDML names. | `lumen.params` |
| **unit and allowed range** | What each parameter declares about its values. | `params.info(key)` |
| **moving-source method** | How lumen differentiates with respect to the source position and direction: light paths move with the source while their random numbers stay fixed. | `wrt=["source/position"]` |
| **pinned vertex** | A point of a light path that stays on its surface while the source moves. | explanation pages |
| **reweighting** | How lumen differentiates with respect to optical values: photons are drawn at reference values and weighted by the ratio of live to reference probabilities. | gradient method `reweight` |
| **gradient method** | How a parameter is differentiated: reweighting, moving paths, or after tracking (sensor and electronics values). | `params.info(key).method` |
| **`reference`** | The parameter values the photons are drawn at; by default the live values. | `simulate(reference=)` |
| **replicas** | Independent sub-runs of one call, whose spread gives the Monte Carlo errors. | `simulate(replicas=)` |
| **CPU backend** | Running the photon loop without a GPU (slower; for tests and small photon counts). | `backend="auto"` |
| **per-sensor histograms** | The light recorded at each sensor, in time and in wavelength and hit position. | `Light` |
| **response applied after tracking** | Quantum and collection efficiency, timing spread and electronics, applied to the recorded light after the photons are traced. | `lumen.readout` |
| **`arrivals`**, **`mean_arrival_time`** | The expected number of photons a sensor counts, and their mean arrival time. | `Light` |
| **sensor index** | The position of a sensor in every per-sensor array. | `det.sensors` |
| **hit coordinates** | Where on a sensor a photon arrived (for example the polar angle on a PMT). | `SensorModel.hit_coordinates` |
| **`rand`** | The random numbers one photon step uses. | sources and plug-ins |
| **surface model** | How a surface reflects, transmits, absorbs and detects light, built from independent factors. | `lumen.surfaces` |
| **SurvivalCut** | Geant4's reflectivity and transmittance cut, applied before the interface physics. | `lumen.surfaces` |
| **expected detection** | The detected share of a photon's weight is added at every sensor surface it meets, instead of being drawn. | explanation pages |
| **diagnostics** | Per-photon records used by checks and debugging. | `lumen.diagnostics` |
| **float32 tolerance** | The distance below which two surfaces count as the same in single precision. | geometry |
| **fixed-reference finite difference**, **same-random-number finite difference** | The two finite differences lumen's derivatives are checked against. | `lumen.testing` |
| **charge check**, **gradient check** | Validation of the expected charge per sensor against Geant4 or WCSim, and of derivatives against finite differences. | `lumen validate charge`, `lumen validate gradients` |
| **geometry-import check**, **overlap check** | Validation of the imported geometry against Geant4, and of overlapping volumes. | `lumen validate` |
| **speed check** | The time per photon and the compile time, per detector. | `lumen validate speed` |
| **WCSim digit check**, **SBND waveform check**, **sensor-set provenance check** | Validation of the sensor and electronics models against WCSim digits, SBND's waveform simulation and the sources of each set of values. | `lumen validate` |
| **readout toys** | Small closed-form tests of the response applied after tracking. | tests |
