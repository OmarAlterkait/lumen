# Your own detector

This guide takes a GDML file that is not in lumen's repository from conversion to a calibration and a comparison
with Geant4. It uses `examples/data/demo_barrel.gdml` as the example: a water cylinder (radius 0.6 m, height 1.2 m)
lined with a black sheet, with 12 disc PMTs. The file was written with two deliberate defects like those real
exporters produce, so that you can see how lumen reports them.

Every code block comes from `examples/your_own_detector.py` and `examples/compare_with_geant4.py`, which run as they
stand. The outputs shown are from lumen 0.1.1 on an NVIDIA A100.

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
            defaults="lumen:wcte_r14374@2",  # a shipped PMT value set; or the path of your own JSON file
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
- **The counting rule.** `border_into` (PMTs with a photocathode surface): a photon counts when it crosses the GDML
  border surface into the sensor volume and that surface detects it (with `model="lumen:pmt"`, every absorption
  there counts and the QE is applied after tracking). `any_entry` (sensitive volumes without a detecting surface): every entry counts.
- **A PMT interior split into several volumes:** select the volume on the far side of the photocathode border, that
  is, the second `physvolref` of its `<bordersurface>`. Use one `lv_pattern` that matches it in every PMT type.
- **The surface reader**: `lumen:geant4` reads the detecting surface as stock Geant4 does. `lumen:complex_index` is
  for a photocathode surface whose GDML carries a complex refractive index (`RINDEX` and `KINDEX` tables, as some
  experiments' software writes for a thin-film cathode, which stock Geant4 10.7 ignores on a surface): it treats the
  cathode as a thick absorbing layer of index RINDEX + i KINDEX, with Fresnel reflection off it and the absorbed
  share detected with `EFFICIENCY`. It does not model the thin film's interference; it is an approximation of it, and
  it needs both tables flat over the wavelength range.
- **The sensor model and its values.** `lumen:pmt` with a shipped value set (`lumen:sk_r3600@2`,
  `lumen:wcte_r14374@2`, `lumen:sbnd_pmt@1`, the ANNIE sets, ...) or your own. To start your own, copy one with
  `lumen.sensors.write_defaults("lumen:wcte_r14374@2", "my_pmt.json")`, edit it, set `provenance.source` and
  `provenance.status`, and pass the file's path as `defaults`.

0.1 supports one `border_into` group per detector, and `simulate` reads out at most one group with a sensor model;
other groups (counting sensors) give arrivals only. Several PMT types in one group share one value set in 0.1
(per-type values in one group come in 0.2).

### Borders and skins at the detector volume's boundary

lumen cuts the GDML world at the detector volume and binds the borders there as Geant4 binds them, so a recipe never
needs to unbind or rebind them:

- a border from a daughter into the detector volume's placement, or from that placement into its mother, is bound
  and acts at that crossing as in Geant4;
- a border from the mother into the detector volume is bound but never crossed: lumen's outside holds no photon;
- a border whose first volume lies outside the detector volume (wholly outside, or from a neighbour into the detector
  volume) is dropped, with a warning saying so. Geant4 consults a border only for a photon in its first volume, and
  lumen tracks no light outside, so the drop changes nothing;
- a border from the detector volume into a neighbour other than its mother is refused (`no adjacent world
  instances`): lumen's outside is one region, so give the surface to every exit with the recipe's `outside_surface`.

Skins bind wherever their logical volume is in the kept tree, as in Geant4.

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
  health: {'lost': 0.0, 'abandoned': 0.0, 'escaped': 0.0, 'retired': 0.0, 'truncated': 0.0, 'n_truncated': 0.0, 'stuck': 0.0, 'overflow': 0.0}
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
  iteration 0: x = [0.037, -0.0266, 0.089] m
  iteration 1: x = [0.0767, -0.0534, 0.1768] m
  iteration 2: x = [0.094, -0.0635, 0.2239] m
  ...
  iteration 7: x = [0.0926, -0.0541, 0.2255] m
  fitted [0.0926, -0.0541, 0.2255] +- [0.0213, 0.0206, 0.0183] m; truth [0.1, -0.05, 0.2]
```

The fit ends within 1.4 errors of the truth. Two things to know:

- **Limit the step.** A Gauss-Newton step can leave the detector, which then gives no light and no derivative; the
  loop caps each step at 10 cm. From the centre the first step points at the source and is capped.
- **The derivative agrees with a finite difference.** With 32 seeds of 2^18 photons and a central difference of
  h = 2 cm, the source-position derivative of every PMT's expected PE agrees with the finite difference within
  1.8 standard errors at the centre and at z = 0.1 m, with the liner as given and with a black liner (for example
  PMT_top_0 with a black liner: 94.7 against 95.3 per m at the centre, 129.0 against 129.4 at z = 0.1 m). The
  collection efficiency is read at the photon's crossing point on the PMT, in the PMT's own frame, so the derivative
  is smooth through the centre of the tank.

### An optical value

Changing a value never recompiles the photon loop, so a scan is cheap: each call below took 0.02 s after the first.
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
  REFLECTIVITY 0.050: NLL -209265.45   (0.13 s)
  REFLECTIVITY 0.075: NLL -209406.34   (0.02 s)
  REFLECTIVITY 0.100: NLL -209509.16   (0.02 s)
  REFLECTIVITY 0.125: NLL -209570.97   (0.02 s)
  REFLECTIVITY 0.150: NLL -209588.33   (0.02 s)
  REFLECTIVITY 0.175: NLL -209557.33   (0.02 s)
  REFLECTIVITY 0.200: NLL -209473.66   (0.02 s)
  REFLECTIVITY 0.225: NLL -209332.44   (0.02 s)
  REFLECTIVITY 0.250: NLL -209127.94   (0.02 s)
  fitted REFLECTIVITY 0.1456 +- 0.0036 (truth 0.150)
```

The fit is 1.2 errors from the truth. The error is the statistical one of the data only; the Monte Carlo error of
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
  dNLL/d(scale) at the fit: -8.89 (curvature 775.6: a step of -0.0115)
```

The step (−0.012 in the scale, −0.0012 in REFLECTIVITY) is a third of the fit's error.

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

g4ref is not published yet: this section assumes you have a build of it. g4ref is a small Geant4 10.7.4 application
that reads a GDML file, shoots optical photons from point sources and records which sensor volumes they reach.
lumen's validation uses it for every Geant4 comparison.

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
