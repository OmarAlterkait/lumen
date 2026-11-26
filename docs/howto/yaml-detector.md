# A detector written in YAML

This guide writes a small detector by hand, without any GDML, converts it, loads it and simulates it.

YAML is an input for writing a detector by hand; lumen writes none. It is not what `lumen.load` reads: `lumen.convert`
turns the YAML into a detector file (the `.lumen` directory of JSON and `.npy` files), exactly as it does a GDML file,
and `load` reads that (reading YAML is 25-30 times slower). The YAML holds the same records as the detector file, one
to one.

The physics tier runs every code block of this page as it stands; the outputs shown are from an
NVIDIA RTX 2080 Ti. The example detector is `examples/data/yaml_toy/`: a water cylinder (radius 0.5 m, height 1 m)
lined with a black sheet, with six disc PMTs.

## 1. The files

A detector may be one file or several; here it is four, one per part, and a directory of them reads as one detector:

| File | Sections |
|---|---|
| `geometry.yaml` | `name`, `world`, `detector_volume`, `solids`, `volumes`, `placements` (and `meshes`) |
| `materials.yaml` | `materials`, `elements` (and `isotopes`) |
| `surfaces.yaml` | `surfaces`, `skins`, `borders` |
| `sensors.yaml` | `sensors`: the sensor groups (one list) |

The split is yours: any section may go in any file. Named records (solids, volumes, materials, surfaces, ...) are
combined by name, and a name defined in two files is refused; lists (placements, skins, borders) are joined in file
order, which is the order of the placements in their mother. So a file of materials can be shared by several
detectors.

Every file starts with the format it is written in, `schema: lumen-detector-yaml/3`; a file without it is refused.
The geometry of the example:

```yaml
schema: lumen-detector-yaml/3
name: yaml_toy
detector_volume: Tank            # light leaving the water is absorbed (after the Liner skin)

solids:                          # Geant4's solids as GDML spells them: a box's x, y, z and a tube's z are full lengths
  WorldBox: {type: box, params: {x: 3*m, y: 3*m, z: 3*m}}
  TankTube: {type: tube, params: {rmax: 500*mm, z: 1*m}}       # rmin 0 and a full turn when left out
  PMTDisc: {type: tube, params: {rmax: 40*mm, z: 20*mm}}

volumes:                         # daughters left out: the placements below, in their order
  World: {solid: WorldBox, material: Air}
  Tank: {solid: TankTube, material: Water}
  PMT: {solid: PMTDisc, material: Glass}

placements:                      # position and rotation default to zero
  - {name: TankPV, volume: Tank, mother: World}
  - {name: PMT_top, volume: PMT, mother: Tank, copy: 0, position: [0, 0, 480*mm]}
  - {name: PMT_east, volume: PMT, mother: Tank, copy: 2, position: [460*mm, 0, 0], rotation: [0, 90*deg, 0]}
  - name: PMT_south
    volume: PMT
    mother: Tank
    copy: 5
    position: [0, -460*mm, 0]
    rotation: {matrix: [[1, 0, 0], [0, 0, -1], [0, 1, 0]]}
```

The rules, field by field:

- **Units.** A number may be written with its unit, `480*mm`, `2.5*eV`, `90*deg`, `10*ns`, `1.0*g/cm3`; lumen reads
  the number and the unit's size as exact decimals and rounds once, so `12.5*mm` is the double nearest 0.0125 m. A
  bare number is in lumen's units: m, eV, ns, rad. A unit of the wrong quantity (`500*eV` for a radius) is
  refused.
- **Solids** are Geant4's, with GDML's attribute names, lower-cased: `box` (x, y, z), `tube` (rmin, rmax, z,
  startphi, deltaphi), `cone`, `sphere`, `polycone` (with `planes: [[rmin, rmax, z], ...]`), `union`,
  `subtraction` and `intersection` (with `refs: [first, second]` and `transforms: [{}, {position: ...}]`), and the
  rest of `lumen.format.solids`. A box's sides and a tube's `z` are full lengths, as in GDML.
- **Rotations** are Geant4's three angles (x, y, z) of a physvol, or a matrix: `rotation: {matrix: ...}` is the
  rotation that turns the placed volume's frame into its mother's (its columns are the volume's axes in the mother).
  A matrix is stored as the three angles. A mirror image is a placement's `scale`, not a rotation.
- **What may be left out:** `world` (the one volume nothing places), `detector_volume` (the world), `name` (the
  first file's name), a volume's `daughters` (its placements, in order), a placement's `position`, `rotation` and
  `copy`, and an optics model (`geant4`, which reads the properties as Geant4 does).

The materials:

```yaml
materials:
  Water:
    state: liquid
    composition:
      density_g_cm3: 1.0
      components: [{ref: Hydrogen, atoms: 2}, {ref: Oxygen, atoms: 1}]
    properties:
      RINDEX:                    # a table on wavelengths (nm): stored on ascending photon energies (eV)
        wavelength_nm: [250, 350, 450, 550, 650]
        value: [1.366, 1.349, 1.338, 1.335, 1.331]
      ABSLENGTH:                 # a table on photon energies (eV); lengths with their unit
        energy: [1.5, 2.5, 3.0, 3.5, 5.0]
        value: [5*m, 40*m, 60*m, 30*m, 2*m]
```

- **Every material names its composition**: a density and components (each an element or a material, by
  `atoms` or by mass `fraction`), or Z and a molar mass. An element is defined in `elements` (`{formula: H, z:
  1.0, molar_mass_g_mol: 1.008}`), or left out when it is a common one named by its symbol (`H`, `C`, `O`, `Si`,
  ...: `lumen.format.elements`, with IUPAC's standard atomic weights).
- **Properties** use Geant4's keys (`RINDEX`, `ABSLENGTH`, `RAYLEIGH`, `REFLECTIVITY`, `EFFICIENCY`, ...). A table
  is linear in photon energy, as Geant4 reads it, on at least two strictly ascending energies; a table on
  wavelengths is converted once with lumen.units.HC, h c in eV nm from the SI defining constants (recorded in the property's note). A constant is a bare
  number: `RINDEX: 1.33`. A bare number is in the key's unit (lengths in m, times in ns, `GROUPVEL` in m / ns,
  `ISOTHERMAL_COMPRESSIBILITY` in m3 / MeV), or write the unit: `"0.22*m/ns"`. Every key becomes a parameter,
  `materials/Water/ABSLENGTH`. A material without `RAYLEIGH` scatters as Geant4's G4OpRayleigh makes it: one named
  Water gets Geant4's table of water (at 283.15 K), one with `ISOTHERMAL_COMPRESSIBILITY` its own; give `RAYLEIGH` to
  choose another.

The surfaces give Geant4's four attributes, always (`type`, `model`, `finish`, `value`), their properties, and where
they are: a skin around every placement of a logical volume, a border from one placement into another (by placement
name, in that order). The sensors are a recipe's groups, written as data (one list; files join it in order):

```yaml
sensors:
  - name: pmt                # parameter keys are sensors/pmt/<field>
    rule: border_into        # counted across the Cathode borders into each PMT
    lv: PMT
    values: examples/data/toy_pmt.json   # a PMT value set: a JSON file (or <package>:<set>)
    readout: {gate: [-10.0, 60.0]}  # ns after emission
```

The fields are those of a recipe's `SensorGroup` ([your own detector](your-own-detector.md), section 1). The
sensor ids are numbered when the detector is converted.

## 2. Convert, load, simulate

```python
import os
import sys
from pathlib import Path

import numpy as np

import lumen
import lumen.format.yaml

OUT = Path(os.environ.get("LUMEN_EXAMPLES_OUT", "lumen-examples"))  # where the examples write
path = lumen.convert("examples/data/yaml_toy", OUT / "yaml_toy.lumen", overwrite=True)
det = lumen.load(path)
print(det)
for name, pos in zip(det.sensor_table().name, det.sensor_table().position, strict=True):
    print(f"  {name:22s} {np.round(pos, 3)}")

src = lumen.sources.Point(position=(0.1, 0.0, 0.2), lam=400.0, n_photons=1e5)
light = lumen.simulate(det, src, det.params(), samples=2**18, seed=0)
print("arrivals:", np.round(np.asarray(light.arrivals), 1))
```

The source may be a directory, one file, or a list of files and directories. No recipe is passed (`None`): the YAML
holds the decisions a recipe would make. The conversion applies lumen's general import rules and the same checks
as for a GDML file, and numbers the sensors:

```
yaml_toy  (from geometry.yaml, materials.yaml, sensors.yaml, surfaces.yaml)
  detector volume  Tank; 3 volumes, 3 materials, 2 optical surfaces
  sensors          6 in 1 group(s): pmt
  calibration      nominal
  TankPV/PMT_top/PMT     [0.   0.   0.48]
  TankPV/PMT_bottom/PMT  [ 0.    0.   -0.48]
  TankPV/PMT_east/PMT    [0.46 0.   0.  ]
  TankPV/PMT_west/PMT    [-0.46  0.    0.  ]
  TankPV/PMT_north/PMT   [0.   0.46 0.  ]
  TankPV/PMT_south/PMT   [ 0.   -0.46  0.  ]
arrivals: [518.1 112.9 264.9 148.4 196.5 193.8]
```

A sensor's name is its path of placement names; a placement name that repeats among its siblings gets `#<index>`.

## 3. When the YAML is wrong

Every error names the file or the record, and the field. A few of them:

| Message | What to do |
|---|---|
| `solids.TankTube.params.rmax: '500*eV': 'eV' is not a unit of length (one of mm, ...)` | write a length |
| `solids.PMTDisc.params: unknown field 'rmaxx' (did you mean 'rmax'?)` | fix the name |
| `materials 'Water' is defined in materials.yaml and in more.yaml` | keep one definition |
| `materials.Water.properties.ABSLENGTH: a table gives its energies (energy, eV) or wavelengths (wavelength_nm)` | add `energy:` or `wavelength_nm:` |
| `material 'Water'.ABSLENGTH: energies must be strictly ascending with >= 2 knots (got ...)` | sort the table |
| `border surface 'Cathode_top': physvol 'PMT_tpo' names 0 physical volumes; ...` | name an existing placement |
| `placement 'PMT_south'.rotation: a rotation matrix is orthonormal with determinant +1 ...` | fix the matrix; a mirror image is `scale` |

The structural checks are those of a GDML file ([errors](../reference/errors.md), "Converting the GDML"); a YAML
detector raises them as `lumen.format.yaml.YAMLError`.
