# Your own detector

This guide takes a GDML file that is not in lumen's repository from conversion to a calibration and a comparison
with Geant4. It uses `examples/data/demo_barrel.gdml` as the example: a water cylinder (radius 0.6 m, height 1.2 m)
lined with a black sheet, with 12 disc PMTs. The file was written with two deliberate defects like those real
exporters produce, so that you can see how lumen reports them.

Every code block comes from `examples/your_own_detector.py` and `examples/compare_with_geant4.py`, which run as they
stand. The outputs shown in sections 3 and 4 are from lumen 0.2 on an NVIDIA RTX 2080 Ti; the others from lumen 0.1.1
on an NVIDIA A100.

```bash
export LUMEN_DETECTORS=$PWD/lumen-examples/calibrations && mkdir -p $LUMEN_DETECTORS
python examples/your_own_detector.py
G4REF=/path/to/g4ref python examples/compare_with_geant4.py
```

## 1. Write the recipe

A GDML file holds what Geant4 reads. It does not say which volume to simulate, which volumes are sensors, or what
reads them. A **recipe**, a small Python file you keep beside your GDML (not in lumen), says that. For the example
(`examples/demo_barrel_recipe.py`):

```python
from lumen import Recipe
from lumen.format.sensorsets import SensorGroup

RECIPE = Recipe(
    name="demo_barrel",
    detector_volume="BarrelLV",              # the logical volume lumen simulates; light leaving it is absorbed
    groups=(
        SensorGroup(
            "pmt",                           # the group's name: parameter keys are sensors/pmt/<field>
            rule="border_into",              # counted across a border surface into the sensor volume
            lv="PMT",                        # or lv_pattern=r"^PMT" (a regular expression on logical-volume names)
            values="examples/data/toy_pmt.json",    # a PMT value set: a JSON file (or <package>:<set>)
            readout={"gate": (-20.0, 80.0)}, # ns after emission
        ),
    ),
)
```

How to fill it in for your detector:

- **The detector volume.** The logical volume that holds everything optical: usually the water or argon volume, or
  the tank around it. Everything outside is absorbed. Pick the smallest volume that still holds every surface light
  can reach. A logical volume placed more than once is refused (the message names its placements and their common
  mother): give the common mother to simulate every placement, or the physvol name or path (`"hall_pv/left_pv"`) of
  the one to simulate.
- **The sensor volumes.** One sensor group selects logical volumes, by name (`lv="PMT"`) or by a regular
  expression on the name (`lv_pattern=r"^PMT"`). Each placement of a selected volume becomes one sensor, numbered in
  placement order. The names are as your GDML writes them, including any `0x...` suffix an exporter added. Sensors of
  one volume that differ by type are split into groups by a column of the id map (`where={"type": "coated"}`, with
  the recipe's `sensor_columns` giving that column): such groups count by one rule and each has a sensor model.
- **The counting rule.** `border_into` (PMTs with a photocathode surface): a photon counts when it crosses the GDML
  border surface into the sensor volume and that surface detects it (with a value set, `values=`, every absorption
  there counts and the QE is applied after tracking). `any_entry` (sensitive volumes without a detecting surface): every entry counts,
  and a photon emitted inside the volume counts where it is emitted, as a Geant4 sensitive detector counts it
  (`settings={"count": "entry"}` counts only entries).
- **A PMT interior split into several volumes:** select the volume on the far side of the photocathode border, that
  is, the second `physvolref` of its `<bordersurface>`. Use one `lv_pattern` that matches it in every PMT type.
- **The detecting surface** is read as stock Geant4 reads it. With a value set, the QE applies after tracking: it is
  the detection at normal incidence, and the surface's `EFFICIENCY` is not read. Without one, the group's setting
  `entry_efficiency` (else the surface's `EFFICIENCY`) is the detection on entry.
- **The value set** (`values=`): a JSON file, or `<package>:<set>`, a set an experiment's package ships (lumen-wc's
  `lumen_wc:sk_r3600`, lumen-tpc's `lumen_tpc:sbnd_pmt`, ...), read by the sensor model after tracking. To start
  your own, copy one with `lumen.sensors.write_defaults("examples/data/toy_pmt.json", "my_pmt.json")`, edit it, set
  `provenance.source` and `provenance.status`, and pass the file's path as `values`.

Every group with a value set has its own keys
(`sensors/<group>/<field>`) and histograms: `light.group("<group>")` (and `light.groups`) holds them, and the readout
functions take `(light.group(name), params)`.
Groups without a sensor model (counting sensors) give arrivals only.

### Borders and skins at the detector volume's boundary

lumen cuts the GDML world at the detector volume and binds the borders there as Geant4 binds them, so a recipe never
needs to unbind or rebind them:

- a border from a daughter into the detector volume's placement, or from that placement into its mother, is bound
  and acts at that crossing as in Geant4;
- a border from the mother into the detector volume is bound but never crossed: lumen's outside holds no photon;
- a border whose first volume lies outside the detector volume (wholly outside, or from a neighbour into the detector
  volume) is dropped, with a warning saying so. Geant4 consults a border only for a photon in its first volume, and
  lumen tracks no light outside, so the drop changes nothing;
- a border from the detector volume into a neighbour other than its mother (or a sibling that encloses it) is refused
  (`no adjacent world instances`): lumen's outside is one region.

Skins bind wherever their logical volume is in the kept tree, as in Geant4. Leaving the detector volume, a photon
meets the skin of the volume Geant4 enters there: a sibling that holds the detector volume in a hole of its solid
(every face touches it), otherwise the mother.

## 2. Convert: what Geant4 refuses, lumen refuses

```python
from lumen.gdml import reader, repair

def convert(gdml, out):
    path = lumen.convert(gdml, RECIPE, out, overwrite=True)
    det = lumen.load(path)
    for line in det.file.report:
        if "left out" in line:
            print("   ", line)
    return det

try:
    convert("examples/data/demo_barrel.gdml", "lumen-examples/demo_barrel_as_given.lumen")
except reader.GDMLError as err:
    print(err)
```

lumen reads a GDML file as stock Geant4 (10.7 and 11.2) does, and refuses what Geant4 refuses. The example has an
unpaired tail on `WATER_ABSLENGTH` (11 values for a two-column table), which Geant4 refuses:

```
gdml examples/data/demo_barrel.gdml: matrix 'WATER_ABSLENGTH': 11 values (as Geant4 counts them) for coldim 2: Geant4 refuses it too; the repair 'matrix_tail' (the unpaired tail of the matrix dropped) writes a copy Geant4 and lumen both read: python -m lumen.gdml.repair SRC DST, then convert DST
```

The repair is a separate step, and its copy is what you convert (the detector file records its name and sha256) and
what you give Geant4 if you compare with it later:

```python
text, edits = repair.repair(open("examples/data/demo_barrel.gdml").read())
print(edits)
open("lumen-examples/demo_barrel_repaired.gdml", "w").write(text)
det = convert("lumen-examples/demo_barrel_repaired.gdml", "lumen-examples/demo_barrel_as_given.lumen")
p = det.params()
key = "surfaces/LinerSurface/REFLECTIVITY"
print(p[key], p.explain(key))
```
```
["matrix 'WATER_ABSLENGTH': 11 values for coldim 2, the unpaired tail (1) dropped"]
    opticalsurface 'LinerSurface'.REFLECTIVITY: left out: a table key given as a constant (coldim 1): Geant4 stores it as a constant property, which no process reads (matrix 'LINER_REFLECTIVITY')
1.0 Geant4 default 1, G4OpBoundaryProcess.cc:308
```

The second line is the file's other surprise. The liner's `REFLECTIVITY` is a one-column constant: Geant4 loads it
as a constant property, but its boundary process reads REFLECTIVITY only as a table, so it uses its default, 1.
lumen reads it the same way, so both codes agree, and the "black" sheet is a perfect reflector. Fix the file, here
by writing the reflectivity as a two-column table (and dropping the tail by hand), and convert again:

```xml
<matrix name="LINER_REFLECTIVITY" coldim="2" values="1.5*eV 0.1 5.0*eV 0.1"/>
```
```
  surfaces/LinerSurface/REFLECTIVITY = Table(energy=array([1.5, 5. ]), value=array([0.1, 0.1])) (from: detector file)
demo_barrel  (from demo_barrel_fixed.gdml)
  detector volume  BarrelLV; 4 volumes, 4 materials, 2 optical surfaces
  sensors          12 in 1 group(s): pmt
  calibration      nominal
```

The repairs, with what each means, are in [errors](../reference/errors.md#repairs).

Conversion refuses what it cannot convert faithfully, with a message that names the object and the fix: a material
without a composition, an opticalsurface missing an attribute, a surface form lumen does not implement, two skins
on one volume, a border on a placement name that is not unique. See [errors](../reference/errors.md).

Check the sensors once converted: `det.sensor_table().name` lists them in order, and a quick simulation of a source at a
symmetric point should light symmetric sensors equally. In writing this guide, two side PMTs of the example first
saw 40 % of the light of the other two from a central source. Their rotations in the GDML were wrong, so they faced
along the wall; a quick simulation found it.

## 3. Simulate

```python
src = lumen.sources.Point(position=(0.0, 0.0, 0.0), lam=400.0, n_photons=1e5)
light = lumen.simulate(det, src, samples=2**18, seed=0)
pe = lumen.readout.pe(light, p0)
```
```
   0 BarrelPV/WaterPV/PMT_top_0/PMT arrivals    169.8 +-   4.2   PE    43.2
   1 BarrelPV/WaterPV/PMT_bottom_0/PMT arrivals    171.8 +-   8.5   PE    43.7
   ...
   8 BarrelPV/WaterPV/PMT_side_0/PMT arrivals    223.6 +-   8.1   PE    56.8
   9 BarrelPV/WaterPV/PMT_side_1/PMT arrivals    224.1 +-   8.2   PE    57.0
  10 BarrelPV/WaterPV/PMT_side_2/PMT arrivals    206.1 +-   9.8   PE    52.4
  11 BarrelPV/WaterPV/PMT_side_3/PMT arrivals    211.8 +-   4.8   PE    53.8
  health: {'lost': 0.0, 'abandoned': 0.0, 'escaped': 0.0, 'truncated': 0.0, 'nonfinite': 0.0, 'invalid': 0.0}
```

The tutorial explains each number: [the first hour, section 2](../tutorial/first-hour.md#2-simulate-and-what-the-numbers-mean).

## 4. Calibrate

Every value except the geometry is a parameter, keyed by its GDML name. A calibration is a named set of values that
replaces the detector file's values by name, without editing it.

### An optical value

Changing a value never recompiles the photon loop, so a scan is cheap: each call below took under 0.1 s after the
first.
Here the liner's reflectivity is fitted to pseudo-data
made with the sheet reflecting 15 % instead of 10 % (50 flashes of the central source, summed per PMT), by a parabola
through the likelihood near its minimum:

```python
KEY = "surfaces/LinerSurface/REFLECTIVITY"
truth_p = p0.scale(KEY, 1.5)
light_t = lumen.simulate(det, src, truth_p, samples=2**21, seed=11)
data = sum(jnp.asarray(lumen.readout.sample(light_t, truth_p, seed=200 + k).pe) for k in range(50))
for s in np.linspace(0.5, 2.5, 9):
    ps = p0.scale(KEY, s)                            # every knot of the table times s
    mu = 50 * lumen.readout.pe(lumen.simulate(det, src, ps, samples=2**20, seed=0), ps)
    nlls.append(float(jnp.sum(mu - data * jnp.log(mu))))
```
```
  surfaces/LinerSurface/REFLECTIVITY: unit -, range (0.0, 1.0), gradient method reweight
  REFLECTIVITY 0.050: NLL -208761.50   (0.41 s)
  REFLECTIVITY 0.075: NLL -208898.47   (0.09 s)
  REFLECTIVITY 0.100: NLL -209000.05   (0.08 s)
  REFLECTIVITY 0.125: NLL -209060.50   (0.09 s)
  REFLECTIVITY 0.150: NLL -209076.06   (0.08 s)
  REFLECTIVITY 0.175: NLL -209043.78   (0.08 s)
  REFLECTIVITY 0.200: NLL -208958.75   (0.08 s)
  REFLECTIVITY 0.225: NLL -208817.41   (0.09 s)
  REFLECTIVITY 0.250: NLL -208609.22   (0.10 s)
  fitted REFLECTIVITY 0.1449 +- 0.0036 (truth 0.150)
```

The fit is 1.4 errors from the truth. The error is the statistical one of the data only; the Monte Carlo error of
the model (2^20 photons at one seed) is not included.

The derivative is available through `simulate` too: trace the value. At the fitted scale
the script takes it with `jax.jvp` (one run) and turns it into a Newton step on the parabola's curvature, which should
be small against the error:

```python
def nll(s):
    ps = p0.scale(KEY, s)
    mu = n_flash * lumen.readout.pe(lumen.simulate(det, src, ps, samples=2**20, seed=0), ps)
    return jnp.sum(mu - data * jnp.log(mu))
value, slope = jax.jvp(nll, (s_hat,), (1.0,))
```
```
  dNLL/d(scale) at the fit: -8.72 (curvature 774.4: a step of -0.0113)
```

The step (−0.012 in the scale, −0.0012 in REFLECTIVITY) is a third of the fit's error.

A water absorption length is a poor example in a detector this small: at 60 m against a 1.2 m tank, a 10 % change
moves the light by about 0.1 %.

A source-position derivative (a tangent on `source/position`) is not available in this version ([known
limits](../explanation/known-limits.md#derivatives-in-the-source-position)).

`p0.info(KEY)` gives the key's unit, range and gradient method; a value outside the range is refused when you set it.

### Save the calibration

```python
fitted = p0.scale(KEY, float(s_hat))
det.save_calibration(fitted, name, note=note)      # <name>.params.json in the first directory on LUMEN_DETECTORS
det_cal = lumen.load(det.path, calibration=name)
```

A calibration keeps only the values that differ from the detector file's own, and records the detector file's digest:
loading it with another detector file raises `CalibrationMismatch`. It keeps each table's energies; a table with a
different number of knots is a new detector file. It is flat: saved from a detector loaded with a calibration, it
holds that calibration's values too, so it loads on the detector file alone.

## 5. Export to GDML

`lumen.gdml.export` writes the detector as lumen simulates it -- geometry, materials with their optical properties,
optical surfaces -- as a GDML that Geant4 10.7 and 11.2 read as lumen does: the file to run Geant4 on. Every number is
a bare literal in Geant4's units that reads back to the same double.

```python
import lumen.gdml

report = lumen.gdml.export(det.path, "lumen-examples/demo_barrel_export.gdml")
back = lumen.gdml.importer.read_gdml("lumen-examples/demo_barrel_export.gdml")
back.solids == det.file.solids and back.placements == det.file.placements   # True
lumen.gdml.export(det.path, "lumen-examples/demo_barrel_calibrated.gdml", calibration=name)
```
```
  wrote lumen-examples/demo_barrel_export.gdml; lossless: True
  read back with the same geometry: True
```

With `calibration=`, the calibration's values are written into the GDML, so a Geant4 run on it uses your fitted
values. What GDML cannot express (the detector volume, the sensor groups, an optics model with no Geant4 form) stays
in the detector file; `report.annotations` lists it. The arguments g4ref reads the sensors with come from the
detector file: `python tools/g4ref/sensor_args.py DETECTOR.lumen` prints them (`--sens-regex '^PMT$' --wcsim-hits`
here).

## 6. Check against Geant4 with g4ref

g4ref is a small Geant4 10.7.4 application that reads a GDML file, shoots optical photons from point sources and
records which sensor volumes they reach; lumen's validation uses it for every Geant4 comparison. Its source is
`tools/g4ref/g4ref.cc`; build it where Geant4 10.7.4 with GDML is installed (`geant4-config` on the PATH):

```bash
tools/g4ref/build.sh /path/to/g4ref
```

By default g4ref corrects two points where Geant4 10.7's own processes are wrong. It computes the mean number of
Cherenkov photons per step exactly (`--cerenkov-count exact`); stock 10.7 assumes RINDEX rises with energy, which
argon's does not. It also splits a step's scintillation photons between the fast and slow components binomially
(`--scint-split binomial`). lumen does both the same way. `--geant4-stock` gives Geant4 10.7's own behaviour.

**Charge.** Run g4ref and lumen on the same source, and compare the photons each sensor counts. With
`--wcsim-hits`, g4ref counts a photon absorbed on its way into a sensor volume, which is what lumen's `border_into`
rule with a value set counts as arrivals:

```bash
echo "100 -50 200 1000000 3.099605" > g4ref_sources.cfg          # x y z (mm), photons, energy (eV) = 400 nm
g4ref --gdml lumen-examples/demo_barrel_export.gdml --config g4ref_sources.cfg \
      --sens-regex '^PMT$' --wcsim-hits --seed 1 --no-hits --out lumen-examples/g4ref
```
```python
src = lumen.sources.Point(position=(0.1, -0.05, 0.2), lam=400.0, n_photons=1e6)
light = lumen.simulate(det, src, samples=2**21, seed=0)
# match the sensors by centre, then z = (lumen - Geant4) / sqrt(lumen error^2 + Geant4 counts)
```
```
 sensor              Geant4   lumen          z
 :0@0/PMT_top_0:0@0     2872    2884.5 +-  26.7  +0.21
 0/PMT_bottom_0:1@1     1160    1144.6 +-  15.1  -0.41
 :0@0/PMT_top_1:2@2     1998    1963.9 +-  24.4  -0.67
 0/PMT_bottom_1:3@3      947     976.0 +-  18.8  +0.81
 :0@0/PMT_top_2:4@4     2284    2344.8 +-  33.7  +1.04
 0/PMT_bottom_2:5@5     1050    1007.7 +-  24.0  -1.05
 :0@0/PMT_top_3:6@6     3735    3750.2 +-  40.9  +0.21
 0/PMT_bottom_3:7@7     1164    1223.0 +-  19.8  +1.50
 0@0/PMT_side_0:8@8     2516    2638.5 +-  36.1  +1.98
 0@0/PMT_side_1:9@9     1706    1676.3 +-  35.5  -0.54
 0/PMT_side_2:10@10     1564    1474.3 +-  25.1  -1.92
 0/PMT_side_3:11@11     2203    2154.6 +-  26.3  -0.90
total: lumen / Geant4 = 1.0017 +- 0.0078; chi2/ndf = 14.5/12
```

**Geometry.** lumen's geometry-import check asks Geant4's navigator where random points are, and compares with
lumen's answers, point by point:

```bash
python -m lumen.compare geometry lumen-examples/demo_barrel.lumen --gdml lumen-examples/demo_barrel_fixed.gdml \
       --g4ref /path/to/g4ref --out lumen-examples/geometry_check
```
```
 "points": 40000,
 "material_mismatch": 0,
 "placement_compared": 35077,
 "placement_mismatch": 0,
 "as_used_mismatch": 0,
```

It needs the GDML the detector was converted from. For a detector converted with repairs, it gives Geant4 the
repaired copy.

For larger comparisons (many sources, the per-sensor χ², the derivatives), `python -m lumen.compare charge` and
`python -m lumen.compare gradients` run lumen's own checks from a reference manifest; see
[how the checks decide](../explanation/validation.md).

## Sharing

The detector file, your recipe and your calibrations are yours: nothing about them has to go into lumen. To share a
detector with a colleague, give them the GDML and the recipe (or the converted `.lumen` directory); they need only
the public lumen.
