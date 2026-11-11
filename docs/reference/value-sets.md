# PMT value sets

A value set holds the numbers of the PMT model (`lumen:pmt`): quantum and collection efficiency, timing, charge, pulse,
dark rate and front end. A sensor group names one with `defaults=`: a shipped set, `lumen:<set>@<version>`, or a path
to your own JSON file. Its numbers become the parameters `sensors/<group>/<field>`, so you can calibrate and
differentiate them; its structure (the forms, the knot positions, the sampling) is fixed when the detector loads.

## Shipped sets

`lumen:sk_r3600@2` (SK), `lumen:wcte_r14374@2` (WCTE), `lumen:wcsim_annie_<type>@2` (ANNIE's tube types),
`lumen:sbnd_pmt@1` and `lumen:sbnd_pmt_uncoated@1` (SBND), `lumen:microboone_qe@1` (MicroBooNE). The `@1` versions of
the WCSim sets stay loadable; `@2` is what the recipes use (`docs/dev/decisions.md`, 11).

## Writing your own

Start from a shipped set, edit it, and name the file in your recipe:

```python
import json
import lumen.sensors

path = lumen.sensors.write_defaults("lumen:wcte_r14374@2", "my_pmt.json")
d = json.loads(path.read_text())
d["name"], d["version"] = "my_pmt", 1
d["provenance"] = {"source": "my lab's measurement, 2026", "status": "measured", "fields": {}}
d["efficiency"] = 0.9
path.write_text(json.dumps(d, indent=1))
model = lumen.sensors.load_model(str(path))          # checks the file; a recipe names it with defaults="my_pmt.json"
print(model.set_name, model.readout, sorted(model.values)[:4])
```

`load_model` refuses a file with a missing record or an inconsistent table, naming what is wrong
([errors](errors.md), PMT value sets).

## The fields

File fields carry their unit as a suffix (`_ns`, `_pe`, `_hz`, `_nm`, `_deg`); parameter keys do not
(`sensors/pmt/tts/sigma`, in ns).

| Field | What | Parameter keys |
|---|---|---|
| `schema`, `model`, `model_version` | `lumen-pmt-values/1`, `lumen:pmt`, 1 | – |
| `name`, `version` | the set's name and version | – |
| `readout` | what the readout computes: `pe`, `hits` or `waveform` (hits and waveforms need `tts` and `charge`) | – |
| `provenance` | `source` (where the numbers come from), `status` (`measured`, `transcribed` or `unverified`; unverified warns at load), optional `warning` and per-field notes | – |
| `qe` | `wavelength_nm` knots (linear between them, constant beyond) or `band_edges_nm` bands, and `values` | `qe` |
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
| `ranges` | the allowed range of a value, `"<field>": [lo, hi]`; required for `t_offset`, every `tts/` value and every `pulse/` value of a non-table pulse, because they set the time grid | – |

The component forms and their fields are listed in `lumen.sensors.components`.
