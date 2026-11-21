# Vocabulary

The words lumen uses in its API, its messages and these pages, one meaning each. Units are m, ns, nm and eV
everywhere. The API takes plain numbers in these units; only file field names carry a unit suffix (`sigma_ns`).

## Detectors and files

| Word | Meaning | Where you meet it |
|---|---|---|
| **detector file** | A converted detector: a `<name>.lumen/` directory of JSON and `.npy` files holding what Geant4 reads from the GDML, plus the recipe's decisions. Once converted, no GDML is needed. | `lumen.convert`, `lumen.load`, `det.file` |
| **recipe** | The small Python file that says what GDML cannot: the detector volume, the sensor groups. It runs once, at conversion. | `lumen.Recipe` |
| **detector volume** | The volume lumen simulates: a logical volume placed once, or the physvol name or path (names from the world down, joined by "/") of one placement. Light that leaves it is absorbed. | `Recipe(detector_volume=)` |
| **repair** | A fix for a GDML construct that stock Geant4 refuses or reads differently from what it says (for example an unpaired value at the end of a matrix), written by `python -m lumen.gdml.repair` into a copy you then convert. lumen itself refuses such a GDML and names the construct. | `python -m lumen.gdml.repair`, `lumen.gdml.repair.repair` |
| **calibration** | A named set of parameter values, `<name>.params.json`, that replaces values of a detector file by name without editing it: every value it changes from the file's own (`nominal`). | `lumen.load(..., calibration=)` |
| **GDML names** | Parameter keys built from the names in the GDML, such as `materials/Water/ABSLENGTH`. | parameter keys |

## Sensors

| Word | Meaning | Where you meet it |
|---|---|---|
| **sensor set** | The sensor groups of a detector (one per detector file): which volumes are sensors, how a photon is counted there and what reads it. | `det.sensor_set` |
| **sensor group** | One type of sensor: a selector (a logical-volume name or pattern), a counting rule, its value set and its readout gate. | `SensorGroup` |
| **counting rule** | How a photon is counted at a sensor: `border_into` (across the GDML border surface into the sensor, by that surface's own detection) or `any_entry` (every entry, and an emission inside the volume unless its setting `count` is "entry"). | `SensorGroup(rule=)` |
| **sensor model** | The sensor response applied after tracking: quantum and collection efficiency, timing, charge, pulse and front end, read from a group's value set. lumen has one, `lumen:sensor`. | `SensorGroup(values=)` |
| **value set** | A sensor model's default values, shipped as `<package>:<set>` or your own JSON file; a group with one is read by the sensor model. | `SensorGroup(values=)` |
| **sensor index** | The position of a sensor in every per-sensor array. Sensors are numbered at conversion. | `det.sensor_table().name` |
| **hit angle** | The angle θ of a photon's arrival on a PMT, as WCSim defines it; the collection efficiency is a function of it. | `ce` in a value set |
| **efficiency** | A per-sensor scale on the detection probability, applied after tracking: detection = efficiency × QE(λ) × CE(θ). | `sensors/<group>/efficiency` |
| **entry efficiency** | The detection probability on entry of a group without a value set (a setting; absent: the surface's EFFICIENCY). | `sensors/<group>/entry_efficiency` |

## Parameters

| Word | Meaning | Where you meet it |
|---|---|---|
| **`Params`** | The one mapping of every parameter value, keyed by GDML names. Immutable, and a JAX pytree. | `det.params()` |
| **unit and allowed range** | What each parameter declares about its values. | `params.info(key)` |
| **gradient method** | How a parameter is differentiated: `reweight` (optical values), `moving_paths` (source position and direction), `after_tracking` (sensor and electronics values). | `params.info(key).method` |
| **`sigma_alpha`**, **`polish`** | The opticalsurface's roughness: σα of a unified surface (rad) or the polish of a glisur surface. Geant4's GDML calls both `value`. | `surfaces/<Surface>/sigma_alpha`, `surfaces/<Surface>/polish` |
| **`source/position`** | The parameter key of a source's position (m); a tangent on it is refused (no source-position derivative in this version). | `p.set("source/position", x)` |
| **out of the model / within it** | A value the model fixes (such as the specular share of a ground painted surface) is not a parameter at all. A real property at the edge of its range (a REFLECTIVITY of exactly 0) is a parameter, and its derivative is one-sided. | `params.info(key)` |

## Simulation

| Word | Meaning | Where you meet it |
|---|---|---|
| **samples** | The number of Monte Carlo photons a call traces: the effort. | `simulate(samples=)` |
| **`n_photons`** | The expected number of photons the source emits: the physics. The weights of the samples add up to it. | `Point(n_photons=)` |
| **moment** | Per cell, the sum over photons of the square of what each photon added to it. It gives the Monte Carlo errors. | `Histograms.charge2`, `lumen.readout.grid.moment_error` |
| **arrivals** | The expected number of photons the counting rule counts at each sensor, before QE and collection efficiency. | `light.arrivals` |
| **`mean_arrival_time`** | The arrivals-weighted mean time of arrival after emission, before the PMT's time spread (ns). | `light.mean_arrival_time` |
| **health** | Where the photons' weight went (`Health.FATES`: `absorbed_bulk`, `absorbed_surface`, `lost`, `escaped`, `distance_cut`, `abandoned`, `truncated`, `rouletted`, `reweighted`; their sum is the start weight) and what was refused: `nonfinite`, `invalid`. Each key and what to do about it: [health](errors.md#health). | `light.health` |
| **late / early** | Light outside the time grid of the histograms (counted in every total). | the histograms |
| **per-sensor histograms** | The light recorded at each sensor, in wavelength and hit angle, and in time. | `light.group(name)` |
| **expected detection** | The detected share of a photon's weight is added at every sensor surface it meets, instead of being drawn at random. | how arrivals are computed |
| **reweighting** | How lumen differentiates with respect to optical values: photons are drawn at reference values and weighted by live / reference probabilities. | gradient method `reweight` |
| **facet path** | The repeated microfacet reflections of a rough (GLISUR or unified) surface, weighted as one path. | surfaces |
| **surface type** | One of the Geant4 surface behaviours lumen implements (for example PAINTED, DIELECTRIC_RT, SPECULAR_METAL), chosen from each opticalsurface's type, model and finish. | errors, the design notes |
| **float32 tolerance** | The distance below which two surfaces count as the same in single precision. | geometry errors |
| **CPU target** | Running the photon loop on the CPU: tallymap compiles it with the system C++ compiler and OpenMP (g++). The default on a machine without a GPU. | `simulate(backend="cpu")`, or `"auto"` without a GPU |
| **plain-JAX backend** | tallymap's reference backend in plain JAX: no compiler needed, but slow to compile and run. For tests and small photon counts. | `simulate(backend="xla")` |

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
| **charge check** | Expected charge per sensor against Geant4 or WCSim. | `python -m lumen.compare charge` |
| **gradient check** | Derivatives against finite differences. | `python -m lumen.compare gradients` |
| **geometry-import check** | lumen's point location against Geant4's navigator on the same GDML. | `python -m lumen.compare geometry` |
| **overlap check** | Overlapping and coincident volumes through lumen's step against Geant4. | `python -m lumen.compare overlaps` |
| **validation report** | The page generated from the checks' records. | `python -m lumen.compare report` |
| **fixed-reference finite difference**, **same-random-number finite difference** | The two finite differences lumen's derivatives are checked against. | the validation report |
| **g4ref** | The small Geant4 10.7.4 application that produces lumen's Geant4 references from a GDML file. | the how-to guide |

Names that changed between releases are in the [changelog](../../CHANGELOG.md).
