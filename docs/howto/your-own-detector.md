# Your own detector

This guide takes a GDML file that is not in lumen's repository from conversion to a calibration and a comparison
with Geant4. It uses `examples/data/demo_barrel.gdml` as the example: a water cylinder (radius 0.6 m, height 1.2 m)
lined with a black sheet, with 12 disc PMTs. The file was written with two deliberate defects like those real
exporters produce, so that you can see how lumen reports them.

Every code block comes from `examples/your_own_detector.py` and `examples/compare_with_geant4.py`, which run as they
stand. The outputs shown are from one run of them (job Slurm job 44135330 on an NVIDIA A100, before the 0.1 naming pass, so the health key printed as `late` is `retired` after it).

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
    wavelength_range=(300.0, 600.0),         # nm
    groups=(
        SensorGroup(
            "pmt",                           # the group's name: parameter keys are sensors/pmt/<field>
            rule="border_into",              # counted across a border surface into the sensor volume
            surface="lumen:geant4", model="lumen:pmt",
            lv="PMT",                        # or lv_pattern=r"^PMT" (a regular expression on logical-volume names)
            defaults="lumen:wcte_r14374@1",  # a shipped PMT value set; or the path of your own JSON file
            readout={"gate": (-20.0, 80.0)}, # ns after emission
        ),
    ),
)
```

How to fill it in for your detector:

- **The detector volume.** The logical volume that holds everything optical: usually the water or argon volume, or
  the tank around it. Everything outside is absorbed. Pick the smallest volume that still holds every surface light
  can reach.
- **The wavelength range** over which your optics tables are read. Every material table must cover it.
- **The sensor volumes.** One sensor group selects logical volumes, by name (`lv="PMT"`) or by a regular
  expression on the name (`lv_pattern=r"^PMT"`). Each placement of a selected volume becomes one sensor, numbered in
  placement order. The names are as your GDML writes them, including any `0x...` suffix an exporter added.
- **The counting rule.** `border_into`: the photon is counted when it crosses the GDML border surface into the
  sensor and that surface detects it (Geant4's own detection; with `model="lumen:pmt"` every photon absorbed on its
  way in is counted, and the PMT's QE is applied after tracking). This is the rule for PMTs with a photocathode
  surface. `any_entry`: every entry into the volume counts, for sensitive volumes without a detecting surface.
- **The surface reader**: `lumen:geant4` reads the detecting surface as stock Geant4 does.
- **The sensor model and its values.** `lumen:pmt` with a shipped value set (`lumen:sk_r3600@1`,
  `lumen:wcte_r14374@1`, `lumen:sbnd_pmt@1`, the ANNIE sets, ...) or your own. To start your own, copy one with
  `lumen.sensors.write_defaults("lumen:wcte_r14374@1", "my_pmt.json")`, edit it, set `provenance.source` and
  `provenance.status`, and pass the file's path as `defaults`.

0.1 supports one `border_into` group per detector, and `simulate` reads out one group with a sensor model, which must
contain every sensor.

## 2. Convert, and read the repair warnings

```python
def convert(gdml, out):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = lumen.convert(gdml, RECIPE, out, overwrite=True)
    for w in caught:
        print(f"  warning: {w.message}")
    det = lumen.load(path)
    for line in det.file.report:
        if line.startswith("repaired"):
            print("   ", line)
    return det

det = convert("examples/data/demo_barrel.gdml", "lumen-examples/demo_barrel_as_given.lumen")
```

Some GDML constructs are refused by stock Geant4, or loaded but read differently from what they say. lumen repairs
each one so that it reads the file exactly as a working Geant4 would, prints a warning naming the object, how many
there are and what Geant4 does, and records the repair in the detector file. The example has two.

```
  warning: gdml examples/data/demo_barrel.gdml: matrix 'WATER_ABSLENGTH': 11 values (as Geant4 counts them) for coldim 2: exporter defect 'matrix_tail', a <matrix> whose value count is not a multiple of its coldim: repaired, the unpaired tail (1) dropped. Stock Geant4 refuses the file (G4GDMLEvaluator::DefineMatrix: InvalidSize, 'Matrix ... is not filled correctly!'); give a Geant4 reference the same file: python -m lumen.gdml.repair SRC DST
  warning: gdml examples/data/demo_barrel.gdml: opticalsurface 'LinerSurface': property REFLECTIVITY names the constant (coldim 1) matrix 'LINER_REFLECTIVITY': exporter defect 'constant_surface_table', an opticalsurface REFLECTIVITY, TRANSMITTANCE, EFFICIENCY or unified constant given as a constant (a coldim 1 matrix): repaired, dropped: Geant4's default applies. Stock Geant4 loads the file and ignores the value (G4GDMLReadSolids.cc:3556 adds it as a constant property; G4OpBoundaryProcess reads these only as tables, GetProperty, G4OpBoundaryProcess.cc:381-413); give a Geant4 reference the same file: python -m lumen.gdml.repair SRC DST
  2 repair(s) recorded in demo_barrel_as_given.lumen:
    repaired (matrix_tail): matrix 'WATER_ABSLENGTH': 11 values (as Geant4 counts them) for coldim 2: the unpaired tail (1) dropped
    repaired (constant_surface_table): opticalsurface 'LinerSurface': property REFLECTIVITY names the constant (coldim 1) matrix 'LINER_REFLECTIVITY': dropped: Geant4's default applies
```

Read each warning and decide whether the file says what its author meant:

1. **`matrix_tail`**: `WATER_ABSLENGTH` has 11 values for a two-column table. Stock Geant4 refuses the file; lumen
   drops the unpaired last value. That is almost certainly what was meant, but check the table, and fix the file
   when you can.
2. **`constant_surface_table`**: the liner's `REFLECTIVITY` is a one-column constant. Geant4 loads the file but its
   boundary process reads REFLECTIVITY only as a table, so it uses its default, 1. lumen does the same, so both codes
   agree, and the "black" sheet is a perfect reflector:

   ```python
   p = det.params()
   key = "surfaces/LinerSurface/REFLECTIVITY"
   print(p[key], p.explain(key))
   ```
   ```
   1.0 Geant4 default 1, G4OpBoundaryProcess.cc:308
   ```

   This repair changes the physics. Fix the file, here by writing the reflectivity as a two-column table, and
   convert again:

   ```xml
   <matrix name="LINER_REFLECTIVITY" coldim="2" values="1.5*eV 0.1 5.0*eV 0.1"/>
   ```
   ```
     0 repair(s) recorded in demo_barrel.lumen:
  surfaces/LinerSurface/REFLECTIVITY = Table(energy=array([1.5, 5. ]), value=array([0.1, 0.1])) (from: detector file)
demo_barrel  (from demo_barrel_fixed.gdml)
  detector volume  BarrelLV; 4 volumes, 4 materials, 2 optical surfaces
  sensors          12 in 1 group(s): pmt
  calibration      nominal
   ```

The full list of repairs, with what each means, is in [errors](../reference/errors.md#repairs). If you compare
with Geant4 later, give Geant4 the same repaired file: `python -m lumen.gdml.repair SRC DST REPAIR ...` writes it.

Conversion refuses what it cannot convert faithfully, with a message that names the object and the fix: a material
without a composition, an opticalsurface missing an attribute, a surface form lumen does not implement, two skins
on one volume, a border on a placement name that is not unique. See [errors](../reference/errors.md).

Check the sensors once converted: `det.sensors["ids"]` lists them in order, and a quick simulation of a source at a
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
   0 BarrelPV/WaterPV/PMT_top_0/PMT arrivals    169.4 +-   6.5   PE    43.1
   1 BarrelPV/WaterPV/PMT_bottom_0/PMT arrivals    171.6 +-   8.1   PE    43.6
   ...
   8 BarrelPV/WaterPV/PMT_side_0/PMT arrivals    223.6 +-   9.1   PE    56.8
   9 BarrelPV/WaterPV/PMT_side_1/PMT arrivals    224.0 +-   8.2   PE    57.0
  10 BarrelPV/WaterPV/PMT_side_2/PMT arrivals    205.8 +-   9.1   PE    52.3
  11 BarrelPV/WaterPV/PMT_side_3/PMT arrivals    211.5 +-   8.2   PE    53.8
  health: {'lost': 0.0, 'abandoned': 0.0, 'escaped': 0.0, 'late': 0.0, 'truncated': 0.0, 'n_truncated': 0.0, 'stuck': 0.0}
```

The tutorial explains each number: [the first hour, section 2](../tutorial/first-hour.md#2-simulate-and-what-the-numbers-mean).

## 4. Calibrate

Every value except the geometry is a parameter, keyed by its GDML name. A calibration is a named set of values that
replaces the detector file's values by name, without editing it.

### A source position, with `wrt=`

The derivative with respect to the source position needs `wrt=["source/position"]`. `jax.jacfwd` gives the three
derivatives at once, and a few Gauss-Newton steps on the Poisson likelihood find the position of a calibration source
from one event (the truth is (0.10, −0.05, 0.20) m):

```python
def expected_pe_at(x):
    s = lumen.sources.Point(position=x, lam=400.0, n_photons=1e5)
    return lumen.readout.pe(lumen.simulate(det, s, samples=2**18, seed=0, wrt=["source/position"]), p0)

x = jnp.zeros(3, jnp.float32)
for it in range(8):
    mu, J = expected_pe_at(x), jax.jacfwd(expected_pe_at)(x)    # J (sensors, 3)
    fisher = (J / mu[:, None]).T @ J
    step = jnp.linalg.solve(fisher, J.T @ ((n_obs - mu) / mu))
    x = x + step * jnp.minimum(1.0, 0.1 / jnp.linalg.norm(step))   # at most 10 cm per step
```
```
  iteration 0: x = [0.0034, -0.0033, 0.0999] m
  iteration 1: x = [0.063, -0.043, 0.1697] m
  iteration 2: x = [0.0955, -0.0631, 0.2223] m
  ...
  iteration 7: x = [0.0945, -0.0532, 0.2271] m
  fitted [0.0945, -0.0532, 0.2271] +- [0.0209, 0.0207, 0.0183] m; truth [0.1, -0.05, 0.2]
```

The fit ends within 1.5 errors of the truth. Two things to know:

- **Limit the step.** From the centre, the first Gauss-Newton step was far larger than the tank (z = 3.3 m in an
  earlier run without the limit), so the loop caps each step at 10 cm. A fit that puts the source outside the
  detector volume gives no light and no derivative.
- **Check the derivative on your detector.** While this guide was written, a check of the derivative against a
  central finite difference (h = 2 cm, 4 seeds, source on the axis) found the derivative below the finite difference
  by about 30 % in this detector (for example 217.5 against 314.5 per m for PMT_top_0 at z = 0.1 m), and by more at
  the exact centre. The same check in the tutorial's tank agreed within two standard errors. The size of the
  disagreement relative to its noise was not measured; it is under investigation. Until it is resolved, compare the
  derivative with a finite difference, as the tutorial does, before you rely on it.

### An optical value

Changing a value never recompiles the photon loop, so a scan is cheap: each call below took 0.04 to 0.06 s after the
first. Here the liner's reflectivity is fitted to pseudo-data
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
  REFLECTIVITY 0.050: NLL -208495.67   (0.17 s)
  REFLECTIVITY 0.075: NLL -208633.20   (0.04 s)
  REFLECTIVITY 0.100: NLL -208732.64   (0.04 s)
  REFLECTIVITY 0.125: NLL -208791.00   (0.05 s)
  REFLECTIVITY 0.150: NLL -208804.86   (0.05 s)
  REFLECTIVITY 0.175: NLL -208770.30   (0.05 s)
  REFLECTIVITY 0.200: NLL -208683.03   (0.05 s)
  REFLECTIVITY 0.225: NLL -208538.09   (0.06 s)
  REFLECTIVITY 0.250: NLL -208329.89   (0.06 s)
  fitted REFLECTIVITY 0.1438 +- 0.0036 (truth 0.150)
```

The fit is 1.7 errors from the truth. The error is the statistical one of the data only; the Monte Carlo error of
the model (2^20 photons at one seed) is not included.

The derivative is available through `simulate` too: name the key in `wrt` and trace the value. At the fitted scale
the script takes it with `jax.jvp` (one run) and turns it into a Newton step on the parabola's curvature, which should
be small against the error:

```python
def nll(s):
    ps = p0.scale(KEY, s)
    mu = n_flash * lumen.readout.pe(lumen.simulate(det, src, ps, samples=2**20, seed=0, wrt=[KEY]), ps)
    return jnp.sum(mu - data * jnp.log(mu))
value, slope = jax.jvp(nll, (s_hat,), (1.0,))
```
```
  dNLL/d(scale) at the fit: -8.60 (curvature 776.4: a step of -0.0111)
```

The step (−0.011 in the scale, −0.0011 in REFLECTIVITY) is a third of the fit's error.

A water absorption length is a poor example in a detector this small: at 60 m against a 1.2 m tank, a 10 % change
moves the light by about 0.1 %.

`p0.info(KEY)` gives the key's unit, range and gradient method; a value outside the range is refused when you set it.

### Save the calibration

```python
fitted = p0.scale(KEY, float(s_hat))
det.save_calibration(fitted, name, note=note)      # <name>.params.json in the first directory on LUMEN_DETECTORS
det_cal = lumen.load(det.path, calibration=name)
```

A calibration keeps only the values that differ from the detector's, and records the detector file's digest: loading
it with another detector file raises `CalibrationMismatch`. It keeps each table's energies; a table with a different
number of knots is a new detector file. Its parent is the calibration the detector was loaded with.

## 5. Export to GDML

`lumen.gdml.export` writes the detector file back as GDML that Geant4 10.7 and 11.2 read exactly as lumen does, plus a
sidecar (`<file>.lumen.json`) with what GDML cannot express: the detector volume and the sensor groups.

```python
import lumen.gdml

report = lumen.gdml.export(det.path, "lumen-examples/demo_barrel_export.gdml", scope="full")
back = lumen.gdml.read_export("lumen-examples/demo_barrel_export.gdml")
io.equal(back.detector_file, det.file)                 # True: the export is exact
lumen.gdml.export(det.path, "lumen-examples/demo_barrel_calibrated.gdml", calibration=name, scope="full")
```
```
  wrote lumen-examples/demo_barrel_export.gdml and its sidecar; lossless: True ; g4ref arguments: ('--sens-regex', '^PMT$')
  re-imported detector file equals the original: True
```

With `calibration=`, the calibration's values are written into the GDML, so a Geant4 run on it uses your fitted
values. `scope="active"` writes only the detector volume, in a shell.

## 6. Check against Geant4 with g4ref

g4ref is a small Geant4 10.7.4 application that reads a GDML file, shoots optical photons from point sources and
records which sensor volumes they reach. lumen's validation uses it for every Geant4 comparison. It is not published
yet; the commands below assume you have a build of it.

**Charge.** Run g4ref and lumen on the same source, and compare the photons each sensor counts. With
`--wcsim-hits`, g4ref counts a photon absorbed on its way into a sensor volume, which is what lumen's `border_into`
rule with `lumen:pmt` counts as arrivals:

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
python -m lumen.validate.geometry lumen-examples/demo_barrel.lumen --gdml lumen-examples/demo_barrel_fixed.gdml \
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

For larger comparisons (many sources, the per-sensor χ², the derivatives), `lumen validate charge` and
`lumen validate gradients` run lumen's own checks from a reference manifest; see
[how the checks decide](../explanation/validation.md).

## Sharing

The detector file, your recipe and your calibrations are yours: nothing about them has to go into lumen. To share a
detector with a colleague, give them the GDML and the recipe (or the converted `.lumen` directory); they need only
the public lumen.
