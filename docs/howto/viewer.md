# Look at a GDML file as lumen reads it

`python -m lumen.viewer` opens a GDML file in your browser: the geometry in 3D, and next to it what lumen does with
every material and every optical surface. Use it before you write a recipe, to see what lumen will refuse and why;
after, to check that the detector volume and the sensors are the ones you meant; and to look up a table (an
absorption length, a reflectivity) against wavelength without writing a script.

It needs the `view` extra (`pip install "lumen[view]"`, which adds `manifold3d` for the solids' booleans) and a browser
with WebGL. Nothing it shows changes how a detector is converted or simulated.

## Open a detector

```bash
python -m lumen.viewer my_detector.gdml                              # any GDML file, without a recipe
python -m lumen.viewer tank.gdml --recipe my_recipes.py:TANK         # with a recipe (the Recipe named TANK there)
python -m lumen.viewer lumen_wc:sk                                   # a detector package's detector, with its recipe
```

The command reads the file the way `lumen.convert` does (the importer, then the recipe's decisions and the general
import rules; a package's detector through the package's own preparation, its files under `LUMEN_DATA` or `--data`),
prints how long that took, and serves the page on `http://127.0.0.1:8765/` until you stop it with Ctrl-C. `--open`
opens a browser, `--port` picks another number. On a cluster node, run it there and forward 8765 from your laptop
(`ssh -L 8765:127.0.0.1:8765 <node>`), then open the address on the laptop.

A large detector takes a while to read. To look again later without reading the file again, write the scene once and
serve it from the file:

```bash
python -m lumen.viewer lumen_wc:sk --out sk.view.json
python -m lumen.viewer --scene sk.view.json
```

A GDML file that Geant4 refuses is refused here too, with lumen's message: it names the repair that writes a copy both
read (`python -m lumen.gdml.repair SRC DST`), whose copy you then open. A file the importer reads but a recipe does not
accept still shows its geometry, with the recipe's error in the REPORT tab.

## What you see

**The 3D view** draws every placed copy of every volume. The colours (COLOR in the toolbar) say:

| COLOR | Each volume coloured by |
|---|---|
| LUMEN | how lumen reads its material: a transparent medium, a wavelength shifter, a scintillator, a sensor of the recipe, opaque (no RINDEX: lumen, as Geant4, stops a photon at its boundary), or refused |
| MATERIAL | its material |
| SURFACE | the optical surface on it (its skin, or a border it takes part in), red when lumen refuses that surface |
| DEPTH | its depth in the volume tree |

UP picks the axis drawn vertical, since a GDML file does not say which one is up. The default is z, as in WCSim's
tanks; `--up y` suits a geometry built with y up (nexus's, for instance). The triad in the corner shows the x (red), y
(green) and z (blue) axes as you turn the view.

SHOW DETECTOR draws only the detector volume and what it contains, which is what lumen simulates; without a recipe the
viewer guesses it as lumen's `auto:` rule would (the outermost volume of a transparent material's main bath) and marks
it GUESSED in the REPORT tab. CLIP cuts the view along x, y or z (the slider moves the plane, the arrows keep the other
side), and MEDIA sets how transparent the media are. Volumes placed more than 50,000 times (a mesh's holes, say) are
not drawn until you tick them in the VOLUMES list; the limit is in the settings (the gear).

**The sidebar** lists the volumes (as a tree), the materials, the optical surfaces, the sensor groups of the recipe,
and the REPORT: the file, the detector volume, everything lumen refuses with its message, the warnings of the import
(the GDML repairs among them) and the import report.

**The right panel** shows what you click, in the 3D view or in a list:

- a volume: the path of the copy you clicked, its solid and its parameters (mm and degrees), its material, its skin and
  borders, and whether it is a sensor;
- a material: its optics model and status, the channels lumen gives it (refraction, absorption, Rayleigh, Mie,
  wavelength shifting, with a wavelength shifter's delay profile), and each property with what reads it: which lumen
  parameter (its key, for a calibration), the scintillation source, or nothing, with the reason (GROUPVEL, for
  instance: lumen times photons by the group index of RINDEX). Keys no Geant4 optical process reads are listed apart,
  as kept inert;
- a surface: its Geant4 type, model, finish and value, the lumen surface type it becomes (on a sensor, the one the
  build reads for the sensor's group, with its efficiency on entry or after tracking), each lumen parameter and the
  property it comes from, the reason a property is not read, and where the surface is attached.

Click a table in the properties to plot it against wavelength, with the flat continuation lumen and Geant4 use outside
the knots dashed. `--range LOW HIGH` (nm) gives the light's wavelength range: it is shaded on every plot, and a table
whose knots do not cover it is marked CLAMPED. The settings switch the plot to a log scale or to photon energy.

## From Python

```python
import lumen.viewer

view = lumen.viewer.build("examples/data/example_tank.gdml", recipe="examples/example_tank.py:RECIPE")
water = next(m for m in view.payload["materials"].values() if m["short"] == "Water")
print(water["status"], water["channels"])         # medium ['refraction', 'absorption', 'rayleigh']
```

`view.payload` is the JSON the page draws; `view.errors` lists the steps that failed (the page still shows the rest).
