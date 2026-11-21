# Sensor value sets

A value set holds the numbers of the sensor model (`lumen:sensor`): quantum and collection efficiency, timing, charge, pulse,
dark rate and front end. A sensor group names one with `values=`: the path of a JSON file, or `<package>:<set>`,
the file `<set>.json` in an installed package's `sensor_values` directory (lumen ships none; the experiments' packages
ship theirs). Its numbers become the parameters `sensors/<group>/<field>`, so you can calibrate and
differentiate them; its structure (the forms, the knot positions, the sampling) is fixed when the detector loads.

## The experiments' sets

lumen-wc ships the WCSim detectors' sets (`lumen_wc:sk_r3600`, `lumen_wc:wcte_r14374`, the ANNIE, Hyper-K and IWCD
tubes), lumen-tpc SBND's and MicroBooNE's (`lumen_tpc:sbnd_pmt`, ...). A set's QE is the datasheet QE as
published, per photon at normal incidence; the cathode's model divides by its own normal-incidence absorptance at each
wavelength (`docs/dev/decisions.md`, 11). Beyond its knots or bands a table keeps its end values; a cutoff is written as
an explicit 0. `examples/data/toy_pmt.json` is an illustrative set, not a real PMT.

## Writing your own

Start from a set, edit it, and name the file in your recipe:

```python
import json
import lumen.sensors

path = lumen.sensors.write_defaults("examples/data/toy_pmt.json", "my_pmt.json")
d = json.loads(path.read_text())
d["name"] = "my_pmt"
d["provenance"] = {"source": "my lab's measurement, 2026", "status": "measured", "fields": {}}
d["efficiency"] = 0.9
path.write_text(json.dumps(d, indent=1))
model = lumen.sensors.load_model(str(path))          # checks the file; a recipe names it with values="my_pmt.json"
print(model.set_name, model.readout, sorted(model.values)[:4])
```

`load_model` refuses a file with a missing record or an inconsistent table, naming what is wrong
([errors](errors.md), Sensor value sets).

## The fields

File fields carry their unit as a suffix (`_ns`, `_pe`, `_hz`, `_nm`, `_deg`); parameter keys do not
(`sensors/pmt/tts/sigma`, in ns).

| Field | What | Parameter keys |
|---|---|---|
| `schema`, `model` | `lumen-sensor-values/1`, `lumen:sensor` | – |
| `name` | the set's name | – |
| `readout` | what the readout computes: `pe`, `hits` or `waveform` (hits and waveforms need `tts` and `charge`) | – |
| `provenance` | `source` (where the numbers come from), `status` (`measured`, `transcribed` or `unverified`; unverified warns at load), optional `warning` and per-field notes | – |
| `qe` | `wavelength_nm` knots (linear between them, constant beyond) or `band_edges_nm` bands (the edges between them; the first and last bands go on beyond them; `[]` is one QE at every wavelength), and `values` | `qe` |
| `ce` | `theta_deg` hit-angle knots and `values`, or null (no collection-efficiency table) | `ce` |
| `efficiency` | a per-sensor scale: detection = efficiency x QE(λ) x CE(θ) | `efficiency` (one per sensor) |
| `t_offset_ns` | the transit time, per sensor | `t_offset` |
| `tts` | the per-photoelectron time spread: `gaussian` (`sigma_ns`), `emg` (`sigma_ns`, `tau_ns`, `shift_ns`) or `table` | `tts/<field>` |
| `delay`, `time_walk` | optional: a fixed delay table (a wavelength shifter's emission time); a charge-dependent shift and spread | `delay/...`, `time_walk/...` |
| `charge` | the single-photoelectron charge: `gauss_exp` (`sigma_pe`, `fraction`, `tau_pe`; mean 1 pe) or `table` | `charge/<field>` |
| `gain` | the charge per photoelectron in front-end units, per sensor | `gain` |
| `threshold` | acceptance against charge: `table` or `poly` (WCSim's form), or null | `threshold/<field>` |
| `pulse` | the single-photoelectron pulse: `bigauss` (`t_peak_ns`, `sigma_rise_ns`, `sigma_fall_ns`), `exp2` or `table` | `pulse/<field>` |
| `dark_rate_hz` | the dark rate, per sensor | `dark_rate` |
| `frontend` | `sampling_ns`, `unit`, `baseline`, `noise`, `window_ns`, `charge_noise_pe`, `charge_scale`, `charge_step_pe`, `charge_max_pe` | `frontend/<field>` for the values |
| `ranges` | the allowed range of a value, `"<field>": [lo, hi]`; required for `t_offset`, every `tts/` value but `sigma` and every `pulse/` value of a non-table pulse, because they set the time grid; the set's own values must lie in their ranges (refused at load). The width `tts/sigma` has a derived range, never declared: [r, 1 / r] times the set's own width, r = `lumen.rules.choices.TIME_SPREAD_REACH` (1/2). A range naming no value of the set is refused (a misspelt field). A field without a declared range has its component form's (a width, time constant or table probability >= 0, a fraction in [0, 1]: `lumen.sensors.components`, each form's `RANGES`) or the defaults' (`value_sets.DEFAULT_RANGES`: QE, CE, acceptance and efficiency in [0, 1], dark rate and the front end's noise and window >= 0); any other is any finite value | – |

| `acceptance`, `axes` | optional: a joint acceptance table over several axes; axes no table reads, binned for a response of your own (`{"<feature>_<unit>": knots}`, e.g. `x_m`) | `acceptance` |

The component forms and their fields are listed in `lumen.sensors.components`. A record not in this table is refused.
A form or an axis feature named `"package:Name"` (the naming rule of the medium models, `lumen.medium_model`) is
reserved for forms and features of your own: `lumen-sensor-values/1` keeps the name free, and this lumen refuses
one.
