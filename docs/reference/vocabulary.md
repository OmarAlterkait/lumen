# Vocabulary

The words lumen uses in its API, its messages and these pages, one meaning each. Units are m, ns, nm and eV
everywhere. The API takes plain numbers in these units; only file field names carry a unit suffix (`sigma_ns`).

## Detectors and files

| Word | Meaning | Where you meet it |
|---|---|---|
| **detector file** | A converted detector: a `<name>.lumen/` directory of JSON and `.npy` files holding what Geant4 reads from the GDML, plus the recipe's decisions. Once converted, no GDML is needed. | `lumen.convert`, `lumen.load`, `det.file` |
| **recipe** | The small Python file that says what GDML cannot: the detector volume, the sensor groups, the wavelength range of the optics. It runs once, at conversion. | `lumen.Recipe` |
| **detector volume** | The logical volume lumen simulates. Light that leaves it is absorbed. | `Recipe(detector_volume=)` |
| **repair** | A fix lumen applies to a GDML construct that stock Geant4 refuses or reads differently from what it says (for example an unpaired value at the end of a matrix). Each repair is reported with a warning and recorded in the detector file. | conversion warnings, `det.file.report` |
| **calibration** | A named set of parameter values, `<name>.params.json`, that replaces values of a detector file by name without editing it. | `lumen.load(..., calibration=)` |
| **parents** (of a calibration) | The calibrations a calibration starts from, oldest first. `nominal` is the detector file itself. | `Calibration(parent=)` |
| **GDML names** | Parameter keys built from the names in the GDML, such as `materials/Water/ABSLENGTH`. | parameter keys |

## Sensors

| Word | Meaning | Where you meet it |
|---|---|---|
| **sensor set** | The sensor groups of a detector: which volumes are sensors, how a photon is counted there and what reads it. Structure only, no numbers. | `det.sensor_set` |
| **sensor group** | One type of sensor: a selector (a logical-volume name or pattern), a counting rule, a surface reader, a sensor model, its default values and its readout gate. | `SensorGroup` |
| **counting rule** | How a photon is counted at a sensor: `border_into` (across the GDML border surface into the sensor, by that surface's own detection) or `any_entry` (every entry). | `SensorGroup(rule=)` |
| **surface reader** | How the detecting surface of a sensor behaves while photons are tracked: `lumen:geant4` (as stock Geant4 reads it), `lumen:complex_index`, `lumen:counting`. | `SensorGroup(surface=)` |
| **sensor model** | The PMT model applied after tracking: quantum and collection efficiency, timing, charge, pulse and front end. 0.1 has one, `lumen:pmt`. | `SensorGroup(model=)` |
| **value set** | A sensor model's default values, shipped as `lumen:<set>@<version>` or your own JSON file. | `SensorGroup(defaults=)` |
| **sensor index** | The position of a sensor in every per-sensor array. Sensors are numbered at conversion. | `det.sensors["ids"]` |
| **hit angle** | The angle θ of a photon's arrival on a PMT, as WCSim defines it; the collection efficiency is a function of it. | `ce` in a value set |
| **efficiency** | A per-sensor scale on the detection probability, applied after tracking: detection = efficiency × QE(λ) × CE(θ). | `sensors/<group>/efficiency` |

## Parameters

| Word | Meaning | Where you meet it |
|---|---|---|
| **`Params`** | The one mapping of every parameter value, keyed by GDML names. Immutable, and a JAX pytree. | `det.params()` |
| **unit and allowed range** | What each parameter declares about its values. | `params.info(key)` |
| **gradient method** | How a parameter is differentiated: `reweight` (optical values), `moving_paths` (source position and direction), `after_tracking` (sensor and electronics values). | `params.info(key).method` |
| **`wrt`** | The parameter keys a `simulate` call differentiates: optical keys (absorption lengths, reflectivities, ...), sensor keys and source keys. Required for the source position, which turns on the moving-source method. | `simulate(wrt=["source/position"])` |
| **`sigma_alpha`**, **`polish`** | The opticalsurface's roughness: σα of a unified surface (rad) or the polish of a glisur surface. Geant4's GDML calls both `value`. | `surfaces/<Surface>/sigma_alpha`, `surfaces/<Surface>/polish` |
| **`source/position`** | The parameter key of a source's position (m). | `wrt=["source/position"]` |
| **out of the model / within it** | A value the model fixes (such as the specular share of a ground painted surface) is not a parameter at all. A real property at the edge of its range (a REFLECTIVITY of exactly 0) is a parameter, and its derivative is one-sided. | `params.info(key)` |

## Simulation

| Word | Meaning | Where you meet it |
|---|---|---|
| **samples** | The number of Monte Carlo photons a call traces: the effort. | `simulate(samples=)` |
| **`n_photons`** | The expected number of photons the source emits: the physics. The weights of the samples add up to it. | `Point(n_photons=)` |
| **replicas** | Independent sub-runs of one call. Their spread gives the Monte Carlo errors. | `simulate(replicas=)` |
| **arrivals** | The expected number of photons the counting rule counts at each sensor, before QE and collection efficiency. | `light.arrivals` |
| **`mean_arrival_time`** | The arrivals-weighted mean time of arrival after emission, before the PMT's time spread (ns). | `light.mean_arrival_time` |
| **health** | What happened to the photons that were not counted or absorbed: `escaped`, `abandoned`, `truncated`, `n_truncated`, `retired`, `lost`, `stuck`, `overflow`. Each key and what to do about it: [health](errors.md#health). | `light.health` |
| **late / early** | Light outside the time grid of the histograms. Not the same as `retired`. | the histograms |
| **per-sensor histograms** | The light recorded at each sensor, in wavelength and hit angle, and in time. | `light.histograms` |
| **expected detection** | The detected share of a photon's weight is added at every sensor surface it meets, instead of being drawn at random. | how arrivals are computed |
| **reweighting** | How lumen differentiates with respect to optical values: photons are drawn at reference values and weighted by live / reference probabilities. | gradient method `reweight` |
| **moving-source method** | How lumen differentiates with respect to the source position and direction: light paths move with the source while their random numbers stay fixed. | `wrt=["source/position"]` |
| **pinned vertex** | A point of a light path that stays where it is while the source moves (the moving-source method pins it). | known limits |
| **moving-source chain** | The run of vertices after a drawn direction that move together with the source. | known limits |
| **facet path** | The repeated microfacet reflections of a rough (GLISUR or unified) surface, weighted as one path. | surfaces |
| **surface type** | One of the Geant4 surface behaviours lumen implements (for example PAINTED, DIELECTRIC_RT, SPECULAR_METAL), chosen from each opticalsurface's type, model and finish. | errors, the design notes |
| **float32 tolerance** | The distance below which two surfaces count as the same in single precision. | geometry errors |
| **CPU backend** | Running the photon loop without a GPU, on tallymap's plain-JAX backend. Slower; for tests and small photon counts. | `simulate(backend="auto")` |

## Readout

| Word | Meaning | Where you meet it |
|---|---|---|
| **readout** | What happens after tracking: photoelectrons, hits, waveforms, simulated events and their likelihoods. Every readout function takes `(light, params)`. | `lumen.readout` |
| **gate** | The time window of a sensor group's readout, in ns after emission. | `SensorGroup(readout={"gate": ...})` |
| **expected waveform** | The mean and variance of each sensor's front-end signal, sample by sample. | `lumen.readout.expected_waveform` |
| **sampled event** | One random event drawn from the expected light: PE counts, hits and waveforms. Not differentiable. | `lumen.readout.sample` |

## Validation

| Word | Meaning | Where you meet it |
|---|---|---|
| **charge check** | Expected charge per sensor against Geant4 or WCSim. | `lumen validate charge` |
| **gradient check** | Derivatives against finite differences. | `lumen validate gradients` |
| **geometry-import check** | lumen's point location against Geant4's navigator on the same GDML. | `python -m lumen.validate.geometry` |
| **overlap check** | Overlapping and coincident volumes through lumen's step against Geant4. | `lumen validate overlaps` |
| **validation report** | The page generated from the checks' records. | `lumen validate report` |
| **fixed-reference finite difference**, **same-random-number finite difference** | The two finite differences lumen's derivatives are checked against. | the validation report |
| **g4ref** | The small Geant4 10.7.4 application that produces lumen's Geant4 references from a GDML file. | the how-to guide |

Names that changed between releases are in the [changelog](../../CHANGELOG.md).
