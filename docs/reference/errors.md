# Errors

lumen fails closed: when it cannot do what Geant4 would do, or cannot do it exactly, it stops with an error that
names the object, the rule and the fix. This page lists every error and every message the code raises, grouped
by when you meet them, with what each means and what to do.

In the messages below, `<...>` stands for a value filled in at run time (a volume name, a number).

## Reading an error

Most lumen errors are subclasses of Python's own, so `except ValueError` still catches them:

| Error | Base | Raised when |
|---|---|---|
| `GDMLError` | `ValueError` | the GDML cannot be read as Geant4 reads it |
| `GDMLRepairWarning` | `UserWarning` | a GDML construct Geant4 refuses or misreads, repaired on import (a warning, not an error) |
| `ImportRefused` | `ValueError` | the GDML is readable but lumen cannot convert it faithfully |
| `OpticsRefused` | `DescriptionError` | a material or surface uses optics lumen does not implement, or is inconsistent |
| `LibraryError` | `ValueError` | a recipe's sensor groups or optics choices do not fit the detector |
| `FormatError` | `ValueError` | a detector file is damaged, from a newer lumen, or would be overwritten |
| `SensorValuesError` | `ValueError` | a PMT value set is malformed or cannot be found |
| `UnknownParamKey` | `KeyError` | a parameter key that the detector does not have |
| `GradientNotAvailable` | `ValueError` | a derivative lumen cannot give in this version |
| `NotRegistered` | `KeyError` | a detector, sensor model or optics model that is not installed |
| `LawError` | `ValueError` | a material table does not cover the wavelength range, or has bad values |
| `ExportError` | `ValueError` | a detector file cannot be written as GDML that Geant4 reads the same way |
| `DeviceError` | `RuntimeError` | a GPU was expected but JAX is not using it |
| `ClosedOverValues` | `ValueError` | a plug-in's code captured a parameter value instead of taking it as an input |
| `ReferenceStoreError` | `LookupError` | a validation check cannot find its reference data |
| `CalibrationMismatch` | `ValueError` | a calibration was made for another detector file (its digest differs) |
| `SumOverflow` | `RuntimeError` | the GPU's exact sums dropped a value or derivative too large to represent (`Light.health["overflow"]` counts them under `jax.jit`); the result is wrong, usually because a derivative is taken at a point where the quantity is singular |
| `BoundaryError`, `TreeError`, `GeometryError`, `DescriptionError`, `OpticsError` | `ValueError` | internal consistency checks of the geometry and surface build; see [the last section](#internal-consistency-checks) |

Python's own `ValueError`, `KeyError`, `MemoryError`, `FileNotFoundError`, `FileExistsError`, `PermissionError`,
`TypeError` and `RuntimeError` are also raised, with lumen messages, where no lumen type is needed.

## Converting a GDML file

### Repairs

Some GDML files, often written by detector software's own exporters, contain constructs that stock Geant4 refuses
or reads differently from what they say. lumen repairs each of them, warns once per repair (the object, how many,
and what Geant4 does), and records the repair in the detector file. Read the warnings: a repair keeps lumen and
Geant4 reading the same file, but it can mean the file does not say what its author meant. The guide
[Your own detector](../howto/your-own-detector.md#2-read-the-repair-warnings) shows the loop.

| Repair | The construct | What stock Geant4 does | lumen's repair | What to do |
|---|---|---|---|---|
| `matrix_tail` | a `<matrix>` whose value count is not a multiple of its `coldim` | refuses the file (`Matrix ... is not filled correctly!`) | drops the unpaired tail values | check the table is the one you meant, then fix the file |
| `identical_redefinition` | a constant, variable, quantity, expression or matrix defined twice, identically | refuses the file (`Redefinition of constant or variable`) | keeps the first definition | remove the duplicate |
| `constant_property` | a `<property>` whose `ref` names a define that is not a `<matrix>` | refuses the file (`Matrix ... was not found!`) | drops the property | point the property at a matrix, if you meant it |
| `undefined_matrix_property` | a `<property>` whose `ref` names nothing | refuses the file (`Matrix ... was not found!`) | drops the property | define the matrix, if you meant the property |
| `constant_surface_table` | an opticalsurface REFLECTIVITY, TRANSMITTANCE, EFFICIENCY or unified constant given as a one-column constant | loads it but ignores it: its boundary process reads these only as tables, so Geant4's default applies | drops it, so Geant4's default applies in lumen too | **usually a real mistake**: the default REFLECTIVITY is 1. Write it as a two-column table and convert again |
| `denormal` | a nonzero number below the smallest normal double | refuses the file (`Error in expression`) | writes it as 0 | none needed, usually |
| `non_finite` | an `inf` or `-inf` matrix value | refuses the file | writes it as ±1e30 | replace it by a large finite number |

Each repair's warning reads `gdml <file>: <object>: exporter defect '<repair>', <construct>: repaired, <what was
done>. Stock Geant4 refuses the file (<Geant4's message>); give a Geant4 reference the same file: python -m
lumen.gdml.repair SRC DST`.

The reference-input tool, `python -m lumen.gdml.repair SRC DST`, writes the same repairs into a copy of the
GDML for Geant4. It raises `ValueError`:

| Message | Fix |
|---|---|
| `unknown repairs <names>; known: <list>` | spell the repair as in the table above |
| `<where>: <literal> overflows a double` | the number cannot be repaired; fix it in the file |
| `the elements found in the text are not the parsed ones (an unusual layout): not repaired` | the file's layout defeats the text edit; fix the construct by hand |

### Reading the GDML (`GDMLError`)

These mean the file is not GDML as Geant4 10.7 or 11 reads it. Geant4 refuses each of them too. Fix the file.

| Message | Meaning and fix |
|---|---|
| `gdml <path>: XML error (external entity includes are not supported): <detail>` | not well-formed XML, or it uses `<!ENTITY ... SYSTEM>` includes. Inline the included files |
| `gdml <path>: <<tag>> <name> is not supported` | a GDML element lumen does not read (for example a parameterised volume). Replace it with plain placements |
| `gdml <path>: structure element <<tag>> is not supported` | as above, inside `<structure>` |
| `volume <name>: a physvol <file> (<name>) is not supported` | a placement that loads another GDML file. Merge the files |
| `gdml <path>: world volume <name> is not defined` | the `<setup>`'s world names no volume |
| `volume <name>: solid <solid> is not defined` · `<type> <name>: solid <ref> is not defined` | a reference to a missing solid |
| `volume <name>: physvol <pv> places <lv>, not defined before it` | GDML is read in one pass: define the daughter volume before its mother |
| `<where>: <label> <ref> is not defined` · `matrix <name> is not defined` | a reference to a missing define (position, rotation, matrix) |
| `<what>: scaleref <ref> is not defined` · `<what>: vertex <ref> is not defined` · `<what>: facet vertex <ref> is not defined` | a reference to a missing define in a scaled or tessellated solid |
| `gdml <path>: conflicting re-definition of <tag> <name>` | one name defined twice with different values. Rename one |
| `gdml <path>: <where>: <literal> overflows a double (Geant4 refuses it)` | a number too large for a double |
| `gdml <path>: matrix <name>: <detail>` · `coldim <n>` | a malformed matrix |
| `<what>: <detail>` | an expression in an attribute did not evaluate; the detail is one of the next rows |
| `cyclic definition of <name>` · `unknown identifier <name>` · `missing value` · `bad token in <text> at <pos>` · `trailing tokens in <text>` · `bad call of <name>` · `unknown function <name>` · `unknown matrix <name>` · `index out of range: <name><index>` · `missing )` · `unexpected token <token>` | the expression evaluator's errors, as Geant4's CLHEP evaluator would give them. Fix the expression |

### Converting the GDML (`ImportRefused`)

The file reads, but converting it would change what Geant4 does.

| Message | Meaning and fix |
|---|---|
| `material <name> names no composition: add <D> and <fraction> or <composite> elements, or Z and <atom>` | every material must say what it is made of |
| `material <name>: component <ref> is neither an element nor a material` | fix the reference |
| `opticalsurface <name> has no '<attribute>' attribute. Geant4 would read a default and lumen cannot guess: add it in the GDML and convert again.` | write all four of `model`, `finish`, `type` and `value` |
| `logical volume <lv> has two skin surfaces (<a>, <b>): Geant4 keeps the first registered; refusing the ambiguity` | remove one skin |
| `bordersurface <name>: needs exactly two physvolref` | a border joins two placements, in order |
| `bordersurface <name>: physvolref <pv> names <n> physical volumes; Geant4 binds only the first, so the border is ambiguous` | give the placements unique names |
| `bordersurface <name>: the pair <pv1> -> <pv2> has a border already` | remove the repeated border |
| `surface <name> refers to <ref>, which is not an opticalsurface` | point the skin or border at an `<opticalsurface>` |
| `<what>: property <key> given twice` | remove one |
| `<what>.<key>: matrix <ref> has <n> columns (a property needs 1 or 2)` | a property is a constant (1 column) or a table (energy, value) |
| `<what>.<key>: energies must be strictly ascending with >= 2 knots (got <energies>)` | sort the table, remove repeated energies |
| `<what>.<key>: <detail>` | a property value that does not evaluate |
| `tessellated <name>: no facets` | an empty mesh |

### Optics lumen does not implement, or that is inconsistent (`OpticsRefused`)

lumen implements Geant4's legacy surface types and bulk processes (see the [vocabulary](vocabulary.md)). Anything
else is refused when the detector is converted or loaded, never silently approximated. `<what>` names the material
or surface.

| Message | Meaning and fix |
|---|---|
| `<what>: model <model> is not supported` · `<what>: finish <finish> is not supported` · `<what>: surface type <type> is not supported` | a surface form outside 0.1's legacy types (refused: ground dielectric_dielectric, backpainted finishes, Geant4's `coated` type, LUT, LUTDAVIS, dichroic). Use a supported form |
| `<what>: ground dielectric_dielectric (facet refraction) is not supported` | as above |
| `<what>: <key> is not supported` · `<what>: <key> (<why>) is not supported` | a property lumen does not read (SURFACEROUGHNESS, for example). Remove it, or ask for it |
| `<what>: GROUPVEL (a user GROUPVEL, not Geant4's own from RINDEX) is not supported` | remove GROUPVEL; lumen computes it from RINDEX as Geant4 does |
| `<what>: polish <value> outside [0, <max>] (or exactly 1)` · `<what>: sigma_alpha <value> outside [<min>, <max>]` | the opticalsurface `value` is outside what float32 resolves |
| `<what>: TRANSMITTANCE on a metal would put the photon into it` | remove TRANSMITTANCE from the dielectric_metal surface |
| `<what>: EFFICIENCY > 0 makes it a detecting surface; bind a sensor instead` | put the volume behind it in a sensor group, or set EFFICIENCY to 0 |
| `<what>: <key> must be a table` · `<what>: <key> must be a constant (G4OpMieHG reads a constant property)` | the property has the wrong shape for what Geant4 reads |
| `<what>: MIEHG without MIEHG_FORWARD_RATIO: G4OpMieHG aborts (mat202) at the first Mie event; add MIEHG_FORWARD_RATIO` | add the property |
| `<what>: MIEHG_FORWARD_RATIO <r> without <key>: G4OpMieHG aborts (mat202) at the first <lobe> Mie event; add <key>` | add MIEHG_FORWARD or MIEHG_BACKWARD |
| `<what>: MIEHG_FORWARD_RATIO <r> outside [0, 1]` · `<what>: <key> <g> outside [-0.999, 0.999] (a narrower lobe than float32 directions resolve)` | fix the value |
| `<what>: WLSABSLENGTH without WLSCOMPONENT (G4OpWLS absorbs and emits nothing)` | add WLSCOMPONENT, or remove WLSABSLENGTH |
| `<what>: WLSTIMECONSTANT <tau> ns: Geant4's default WLS time profile is 'delta' (a fixed delay); lumen's delay is exponential: declare wls_time_profile='exponential' if that is the Geant4 configuration` | say which time profile your Geant4 run uses |
| `material <name>: no <keys> for Geant4 <version><hint>` · `geant4 must be '11' or '10.7', got <value>` | scintillation keys differ between Geant4 10.7 and 11; give the ones of your version |
| `index must be 'geant4' or 'tabulated', got <value>` | a recipe option spelt wrongly |
| `opticalsurface <name>: <key> varies by <amount> over <range> nm: a complex index is taken at one wavelength (narrow the range or supply a spectral reader)` | a complex-index cathode is read at one wavelength in 0.1 |
| `opticalsurface <name>: complex_index needs RINDEX and KINDEX on the surface` | add them, or use the `lumen:geant4` reader |
| `<what>: a complex index on a ground finish (the facet's Fresnel) is not supported` | use a polished finish |
| `<what>: <surface type> does not detect (Geant4: only a polished dielectric_dielectric reaches DoAbsorption)` · `<what>: surface type <type> is not supported as a sensor` · `<what>: model / finish not supported as a sensor` · `opticalsurface <name>: <surface type>, not a polished dielectric_dielectric` | the surface in front of a `border_into` sensor cannot detect the way Geant4 would. Use a detecting form, or the `any_entry` rule |
| `detecting surface reader <name>: not one of [...]` · `opticalsurface <name>: reader <name> returned no detecting SensorEntry` | spell the sensor group's `surface` as `lumen:geant4` or `lumen:complex_index` |
| `<what>: TRANSMITTANCE on a WLS film's backing (no transmitting backing)` · `<what> is not index-matched to the medium in front (equal RINDEX tables): ...` · `<what>: no <key>` · `<what>: <key> is not lowered onto a zero-thickness film` · `<what>: a nonzero WLSTIMECONSTANT needs the time profile (G4OpWLS delta / exponential), not lowered yet` · `<what>: WLSMEANNUMBERPHOTONS must be a constant >= 0` · `<what>: WLSCOMPONENT has no emission` · `<what>: absorbs its own emission (d / L up to <value>)` | a thin wavelength-shifting film that lumen cannot treat as a zero-thickness film (0.1 supports an index-matched film over an opaque backing). Model it as a volume, or report it |

### The recipe and the sensor groups

| Message | Error | Meaning and fix |
|---|---|---|
| `gdml: no volume <name>` | `ValueError` | the recipe's `detector_volume` names no logical volume |
| `gdml: no volume of material matching <materials>` | `ValueError` | `detector_volume="auto:<materials>"` found no such volume |
| `recipe <name> needs the inputs <names>` | `ValueError` | pass `inputs={...}` to `lumen.convert` |
| `sensor group <name>: rule <rule> is not one of ('border_into', 'any_entry')` | `ValueError` | spell the rule |
| `sensor group <name>: give exactly one of lv and lv_pattern` | `ValueError` | select by name or by pattern, not both |
| `sensor groups <names>: one border_into group is supported` | `LibraryError` | 0.1 has one `border_into` group per detector |
| `sensor group <name>: border_into reads a model of ('lumen:geant4', 'lumen:pmt'), not <model>` | `LibraryError` | the surface reader is `SensorGroup.surface` |
| `sensor group <name>: any_entry reads the model 'lumen:counting', not <model>` | `LibraryError` | an `any_entry` group is read by `lumen:counting` |
| `sensor group <name>: model <model> needs its value set (defaults)` | `ValueError` | give `defaults=` (a shipped `lumen:<set>@<version>` or a JSON path) |
| `<what>: unknown setting <key> (known: <keys>)` · `<what>: setting <key> must be <type>, got <value>` | `ValueError` | a misspelt sensor-group or optics setting (`efficiency`, not `eff`) |
| `material <name>: optics model <model> is not one this library reads` · `surface <name>: optics model <model> is not one this library reads` | `LibraryError` | a recipe bound an optics model that is not installed |
| `material <name>: medium <medium> is defined by another material already (bind it with lumen:same_as)` · `material <name> is the same as <target>, which is not a medium of the tree` · `material <name> of the tree is not in the detector file` | `LibraryError` | a recipe decision about materials that does not fit the file |
| `logical volume <lv> has two recipe skins (<a>, <b>)` | `LibraryError` | a recipe added a second skin |
| `film <name> on <lv>: placements face <axes>, the recipe expects <axes>` | `LibraryError` | a one-sided film's placements do not all face the same way |
| `WCSim photocathode model <n>: only model 0 is implemented` · `WCSim PMT <type> (<class>): QE tables of <n> / <m> points, not 20` · `tube_types: <path>: interior LV <lv> has glass faces of types <types>` · `ANNIE: PMTwiseQE on (pmt_qe.txt by WCSim tube id) is not implemented` | `ValueError` | the shipped WCSim recipes (SK, WCTE, ANNIE) check the GDML and the WCSim source they read. Use the GDML and WCSim version they were written for |
| `<lv>: solid <name> is a <type>, expected a box` · `<lv> / <lv> carry an optical surface: the film lowering does not apply` · `<lv>: expected LAr holding exactly <a> and <b>` · `TPB film geometry is not a <t> m layer on one face of the foil filling <lv>` · `<lv> is not of material <material>` · `TPB RINDEX differs from LAr's: the film is not index-matched` · `<lv> is not enclosed by <lv>'s inner box` · `<mother>: <n> placements of <child> (want 1)` · `sbnd: OpDet channels are not unique` · `LArSoft table <key>: length unit <unit> is not one of <units>` · `material <name>: LArSoft's LAr needs the tables <names>` | `ValueError` | the same for the shipped LArSoft recipes (SBND, MicroBooNE) |

### Materials over the wavelength range (`LawError`)

| Message | Meaning and fix |
|---|---|
| `<name>: defined on [<lo>, <hi>] nm, the detector uses [<min>, <max>] nm (extend the data or declare outside='clamp')` | a table does not cover the recipe's `wavelength_range`. Extend the table, or narrow the range |
| `<name>: missing parameters <names>` · `<name>: non-finite values in [<min>, <max>] nm` · `<name>: <quantity> out of range at <lam> nm: <value> (allowed [<a>, <b>])` | a table with missing, infinite or out-of-range values |
| `<medium>.<channel>: a channel length law must output a length` · `outside must be 'error' or 'clamp', got <value>` · `group_index needs an index law` | a plug-in's material law is malformed |

## Loading a detector

| Message | Error | Meaning and fix |
|---|---|---|
| `<path> is schema <schema>; this lumen reads lumen-detector/1.1. Upgrade lumen, or open it with the version that wrote it.` | `FormatError` | the detector file was written by another lumen version |
| `<path>: <file> does not match the manifest (changed since it was written)` | `FormatError` | a file inside the detector file was edited. Convert again; never edit a detector file by hand |
| `<path> exists: a detector file is never overwritten (pass overwrite=True)` | `FormatError` | `lumen.convert(..., overwrite=True)`, or another path |
| `non-finite number <x>: the detector file stores finite numbers only` · `cannot store <type> in JSON` · `two arrays at <path>` | `FormatError` | a recipe decision produced a value the format cannot store |
| `sensor set <name>: schema <schema>, this lumen reads lumen-sensors/2` | `ValueError` | the sensor set was written by another lumen version. Convert again |
| `<name>: no detector file found, and LUMEN_DATA is not set to convert it from its dataset <dataset>` | `FileNotFoundError` | a registered detector name needs its source data: set `LUMEN_DATA`, or pass a path |
| `<group>: no entry <name>; registered: <names>. Install the package that provides it.` | `NotRegistered` | a detector, sensor model or optics model that no installed package provides |
| `<group>: <name> is registered already` | `ValueError` | two packages register the same name |
| `tallymap's CUDA cache <path> is under HOME (<home>): set LUMEN_CACHE (or TALLYMAP_CUDA_CACHE) to a directory outside it` | `RuntimeError` | point the cache elsewhere (`LUMEN_CACHE`) |

### Calibrations

| Message | Error | Meaning and fix |
|---|---|---|
| `no calibration <name> in <directories>` | `FileNotFoundError` | save it in a directory on `LUMEN_DETECTORS`, or check the name |
| `calibration <path> exists: choose a new name` | `FileExistsError` | calibrations are never overwritten |
| `calibration parent cycle through <name>` | `ValueError` | two calibrations name each other as parent |
| `calibration <name>: unknown keys <keys>` | `KeyError` | the calibration was made for another detector, or for a key that was renamed (`surfaces/<S>/value` is now `sigma_alpha` or `polish`; `source/x` is `source/position`; see docs/dev/migration.md) |
| `<path>: schema <schema>; this lumen reads lumen-params/1` | `ValueError` | another lumen version wrote it |

### PMT value sets (`SensorValuesError`)

| Message | Meaning and fix |
|---|---|
| `no shipped sensor value set <name>; shipped: <names>` | spell the set; the shipped ones are listed |
| `<ref>: the shipped set <name> is at version <n>` | change the `@<version>` |
| `<ref>: value sets of other distributions come from their own packages` | install the package that ships it, or give a file path |
| `value set <name>: schema <schema> for model <model>; this lumen reads lumen-pmt-values/1 for 'lumen:pmt'` | the file is not a `lumen:pmt` value set |
| `value set <name>: provenance needs a source and a status in ('measured', 'transcribed', 'unverified')` | say where the values come from |
| `value set <name>: readout <readout> is not one of ('pe', 'hits', 'waveform')` · `value set <name>: readout <readout> needs a tts and a charge record` | a timed readout needs a time spread and a charge model |
| `value set <name>: qe needs exactly one of wavelength_nm (knots) or band_edges_nm` · `value set <name>: qe has <n> values for <m> knots or bands` | fix the QE table |
| `value set <name>: <component> form <form> is not one of <forms>` · `value set <name>: <component> (<form>) needs the fields <fields>, got <fields>` | fix the component record |
| `value set <name>: declare the ranges of <keys> (they set the static time grid; a value outside its range is refused)` | add `ranges` for the timing values |

A value set whose status is `unverified`, or which carries a `warning`, loads with a warning (`sensor values
<name>: <warning>`): do not trust what it has not verified.

## Parameters

| Message | Error | Meaning and fix |
|---|---|---|
| `'<key>' is not a parameter of this detector. Did you mean [<keys>]?` | `UnknownParamKey` | a misspelt key, or a property the GDML does not define (adding a property means a new detector file) |
| `<key> has a value outside [<lo>, <hi>] <unit>` | `ValueError` | the value is outside the key's allowed range (`params.info(key).range`); a probability is in [0, 1], a length positive |
| `<key>: shape <shape> expected, got <shape>` | `ValueError` | a per-sensor value must have one entry per sensor of the group |
| `<key>: a table property needs a Table` · `<key>: a constant property needs a number` | `ValueError` | give the value in the shape the GDML gave it |
| `<key>: <n> knots expected, got <shape> values at <shape> energies; a new knot count is a new detector file` | `ValueError` | a calibration keeps the table's energies; to change them, edit the GDML and convert again |
| `<key>: the energies must be strictly ascending` | `ValueError` | sort the table |
| `parameter key <key>: not materials/<M>/<KEY>, surfaces/<S>/<KEY> or sensors/...` · `parameter key <key>: no <material or surface> <name>` · `parameter key <key>: <name> has no property <prop> (adding one is a new detector file, except a Geant4 default the model reads)` | `ValueError` | the key does not name something in the detector file |
| `<key>: the opticalsurface value is a number` · `<key>: a Geant4 default property is set to a number` | `ValueError` | give a number |

## Simulating

| Message | Error | Meaning and fix |
|---|---|---|
| `simulate: one sensor group with a sensor model (lumen.sensors) is supported, the sensor set has <names>` | `ValueError` | 0.1 simulates one sensor group with a sensor model per detector |
| `simulate: sensor group <name> must contain every sensor of the detector` | `ValueError` | as above |
| `simulate: wrt takes the source's keys (source/<field>), got <keys>` | `ValueError` | in this build `wrt` takes only source keys |
| `simulate: <keys> move the photons of a source whose directions are not drawn from a density (drawn_direction False)` | `GradientNotAvailable` | a beam, a sharp Cherenkov cone or a photon list has no source-position derivative; use an isotropic source |
| `<source> moves photons whose directions are not drawn from a density (drawn_direction False): no derivative (docs/explanation/known-limits.md)` | `GradientNotAvailable` | as above |
| `the histograms need <x> GB, over the device budget of <y> GB: <n> sensors x <m> time nodes (<span> ns at <dt> ns), <k> replicas, <d> derivative directions. Shorten the readout gate, widen the time spread's declared lower range (coarser nodes), use fewer replicas or fewer derivative directions (wrt).` | `MemoryError` | lumen refuses before launch. Do what the message lists |
| `the photon loop expects the GPU (<why>) but JAX runs on <backend>: the CUDA plugin did not load (see JAX's log above); run on a working GPU node, or pass backend='cpu'` | `DeviceError` | a GPU job whose JAX did not start CUDA. Check `jax.devices()` |
| `<source> reads <key>: pass the detector's parameters` | `KeyError` | a source that reads detector values (scintillation yields) was emitted without them |
| `<material>: no <keys> (Geant4 <version> emits none)` | `KeyError` | a scintillation source in a material without scintillation properties |
| `Point: give a wavelength (lam) or a spectrum, not both` | `ValueError` | give exactly one |
| `Geant4Steps: light is 'cherenkov' or 'scintillation', got <value>` | `ValueError` | spell the light type |
| `time spread <name>: a scintillation emission profile combines in closed form only with the 'gaussian' time spread; use form 'gaussian' for scintillation light` | `ValueError` | use a gaussian time spread in the value set for scintillation light |
| `<name>: ascending wavelengths and a non-negative, non-zero spectrum of the same length` · `spectrum <name>: needs a non-negative intensity with positive integral` · `spectrum <name>: negative intensities` | `ValueError` | fix the source spectrum |
| `the photon loop (<name>) closes over <n> float array(s) <names>: pass every value through the Operands (lumen.transport.step.operands)` | `ClosedOverValues` | a plug-in captured a number in a closure; such a number would silently get a zero derivative. Declare it as a parameter |
| `max_vertices must be >= 1, got <n>` · `photon_chunks: samples >= 0 and chunk > 0, got <a>, <b>` | `ValueError` | invalid run settings |

## Readout

| Message | Error | Meaning and fix |
|---|---|---|
| `readout: expected lumen.readout.Histograms (or a light holding one), got <type>` | `TypeError` | pass the `Light` that `simulate` returned |
| `readout: sensor group <name> has no sensor model` | `ValueError` | the group has no `model`, so there is nothing to read out |
| `pe_error: the charge histogram has one replica; simulate with replicas >= 2` | `ValueError` | errors need at least two replicas |
| `sensor group <name>: no time histogram (readout 'pe'); simulate with a timed readout` | `ValueError` | the value set's readout is `pe`; use one with `hits` or `waveform` |
| `sensor group <name>: value set <set> has no waveform readout (readout <readout>); use a value set with a pulse and readout 'waveform'` | `ValueError` | as the message says |
| `waveform_nll: data of shape <shape>, the expected waveform is <shape>` | `ValueError` | the data must have one row per channel and one column per sample |
| `readout: the item was made for <n> replicas, run with <m>` | `ValueError` | internal; report it |
| `node_spacing: the narrowest time spread must be > 0 ns; declare its range in the value set` | `ValueError` | a value set whose timing ranges do not define a time grid |
| `Grid: give lam_edges or lam_knots, not both` · `Grid.<name>: at least two ascending values` · `Grid: nt >= 0 and dt > 0` · `Grid.split: one of <splits>, not <split>` | `ValueError` | a histogram grid built by hand with invalid values |

## Exporting to GDML (`ExportError`)

| Message | Meaning and fix |
|---|---|
| `surface <name>: optics model <model> has no Geant4 form; pass on_lossy='omit' to write the GDML without it (marked lossy, not a Geant4 reference)` | a surface lumen models that Geant4 cannot express |
| `surface <name>: a coated surface; GDML has no coated type (it reads as x_ray)` · `surface <name>: <field> <value> is not one Geant4 knows (one of <names>)` | the surface cannot be written as Geant4 would read it |
| `border <name>: physvol <pv> names <n> placements; Geant4 binds a border to the first one only` · `border <name>: the pair <pv1> -> <pv2> has a border already (Geant4 keeps the first, lumen the last)` · `border <name>: a physvol of an assembly (<pv1>, <pv2>); Geant4 renames the imprinted placements, so the border cannot be bound` | the borders would bind differently in Geant4 |
| `logical volume <lv> would carry two skins (<a>, <b>)` | as on import |
| `volume <name>: material <material> is not defined (and is no G4_ NIST name)` | the detector file references a missing material |
| `non-finite value <text>: Geant4 reads no inf or nan` · `denormal value <text>: Geant4's reader refuses it (strtod sets ERANGE)` · `value <text> is <literal> in Geant4's units: not a normal double` · `value <text>: the literal <literal> does not read back exactly` | a value that cannot be written so that Geant4 reads the same double |
| `solid <name>: type <type> has no GDML spelling in the exporter (supported: <types>)` · `solid <name> (<type>): attributes <names> are not <type> attributes` | a solid the exporter cannot write |
| `the placements are not in an order a GDML file can hold (...)` | (a verbatim message) a detector file built in an order GDML cannot express |
| `scope 'active': <volume> is not placed under the world <world>` | use `scope="full"` |
| `names='geant4': two <things> are <name> once stripped at '0x'` · `names='geant4': a material and an element share a name once stripped at '0x'` | use `names="as_stored"` |
| `matrix name <name> would name two tables` · `a matrix and a position share a name` · `matrix <name>: Geant4 would define the constant <c> twice` | a name clash in the written file; report it |
| `surface <name>: lumen:larsoft_reflector needs REFLECTIVITY and DIFFUSE_FRACTION tables on the same energies` · `surface <name>: lumen:larsoft_reflector is a dielectric_metal, unified, ground surface with sigma_alpha 0` (`ValueError`) | a LArSoft reflector surface whose record cannot be written in its Geant4 form |
| `unknown Geant4 target(s) <names>: the exporter knows <names>` · `<option>=<value>: one of <allowed>` (`ValueError`) | an export option spelt wrongly |
| `no sensor set <name> (the detector file has <names>)` | pass an existing sensor set |
| `<sidecar>: schema <schema>; this lumen reads <schema>` · `<sidecar> belongs to another GDML (its sha256 differs from <file>'s)` | `read_export` found a sidecar that does not match its GDML |

## Validation checks

| Message | Error | Meaning and fix |
|---|---|---|
| `no reference store: set LUMEN_DATA to the directory holding the reference products` | `ReferenceStoreError` | set `LUMEN_DATA` |
| `no <check> check named <name>; the manifests list <names>` · `no reference product <id> in the manifests` · `<path>: missing from the store` · `<path>: sha256 differs from the manifest (<digest>)` · `<path>: schema <schema>, this lumen reads <schema>` | `ReferenceStoreError` | the reference data is missing, or not the one the manifest names |
| `<path> is inside a reference store: write check outputs elsewhere (--out)` | `PermissionError` | outputs never go into the reference store |
| `the geometry check needs Geant4's answers: give the g4ref binary (--g4ref)` · `<gdml> is not the GDML <path> was converted from (<sha256>)` | `ValueError` | the geometry check compares against the GDML the detector was converted from |
| `g4ref failed (<code>): <output>` · `<csv>: <n> answers for <m> points (probed on other points?)` | `RuntimeError` | g4ref did not run, or answered for other points |
| `sensor match: <n> reference sensors farther than <tol> mm from any lumen sensor (largest <d> mm)` | `ValueError` | the reference was made from another geometry |
| `<n> sensors but <m> sensor volumes in the navigator` | `ValueError` | internal; report it |

## Internal consistency checks

These errors come from checks inside the geometry and surface build. They should not happen with a detector file
that converted cleanly. If you see one, report it with the GDML, the recipe and the full message.

- `TreeError` (`build/tree.py`): `tree: exactly one world PV (mother None) required, got <pvs>`; `tree: PV <name> has
  unknown mother <name>`; `tree: PVs not reachable from the world (mother cycle): <pvs>`; `tree: LV <lv> placed with
  materials <a> and <b>`; `tree: expected PV, got <x>`; `tree: duplicate PV name <name>`; `a bare Scene needs media=
  (GDML material name -> medium id)`; `media: materials <a> and <b> share medium id <id>`; `<pv>: a mesh member with a
  PATH sensor label (meshes are not numbered)`; `placement of unknown definition <name>`; `definition <name> has <n>
  members and placements: the daughters' mother volume is ambiguous`; `<pv>: medium id <id> has no material name in
  media`; `class_of misses <n> PVs, e.g. <pvs>`; `<n> PVs share the outside's region <id> (e.g. <pvs>): a query prunes
  them as background; bind the outside material <name> to a medium of its own`; `class_of names <n> PVs this Scene does
  not have, e.g. <pvs> (a different tree?)`; `class_of[<pv>] = <c>: region ids are non-negative integers`; `clone names
  <names> already exist in the Scene`.
- `DescriptionError` and `OpticsError` (`build/surfaces.py`, `build/library.py`): the surface lookup checks that
  every material has a medium, every skin and border names a volume of the tree, and every sensor binding covers its
  placements. Messages: `<what>: surface type <name> takes parameters <names>, got <names> (...)`; `medium <name>: expected a MediumEntry(bulk.Medium, params)`; `surface <name>: expected a
  SurfaceEntry(SurfaceType, params)`; `surface <name>: surface type <name> detects; detection needs a sensor binding
  (library.sensors), not a surface`; `sensor <name>: expected a SensorEntry(SurfaceType, params)`; `sensor <name>:
  surface type <name> cannot detect`; `<what> pair <pair>: unknown PV <name>`; `<what> pair <pair>: adjacency facts are
  declared between distinct siblings only`; `materials without a binding: <names>`; `bindings for materials that do
  not occur in the tree: <names>`; `material <m> bound to <medium>, which is not a library medium`; `skin surface <s>
  on LV <lv>, which does not occur in the tree`; `skin on LV <lv>: <s> is not a library surface`; `disjoint placements
  of LV <lv>, which does not occur in the tree`; `sibling pairs declared both touching and disjoint: <pairs>`; `border
  surface <s> (<pre> -> <post>): unknown PV <name>`; `border (<pre> -> <post>): <s> is <what>`; `border surface <s>
  (<pre> -> <post>): the volumes never touch (not mother/daughter, not touching siblings)`; `sensor binding <b>: give
  exactly one of lv= or pv=`; `sensor binding <b>: entry must be 'any' or 'border'`; `sensor binding <b>: <name> is not
  a library sensor`; `sensor binding <b>: LV <lv> does not occur in the tree`; `sensor binding <b>: PV <pv> does not
  occur in the tree`; `sensor <name> on <volume>: expected <n> instances, found <m>`; `PV <name> bound to two sensors
  (<a>, <b>)`; `sensor PV <name> also has a skin surface <s> on its LV: ambiguous (the sensor's surface type is the boundary of a sensor)`
  (remove the skin from the sensor's logical volume); `border surface <s> (<pre> -> <post>) into sensor PV <post>: ambiguous (the sensor's surface type is the boundary of a sensor; a
  border may name the sensor itself when its binding has entry='border')`; `border
  sensor <s> (<pre> -> <post>): <pre> is not a sensor PV, so a detection leaving it has no sensor to go to`; `sensor PV
  <name> has entry='border' but no border bound to <sensor> leads into it: it would never detect` (add a border
  surface into the sensor, or use `any_entry`); `bordersurface <name> (<a> -> <b>): no adjacent world instances`;
  `class <class> touches itself across a <surface type> boundary (<reason>; e.g. <pre> -> <post>): the crossing would be invisible. Declare the placements disjoint (bindings.disjoint) if they do not touch, or give the ...`; `class pair <a> -> <b> resolves to two surface types: ... The signature missed a distinction ...`
 ; `two different surface types share a name: <names>`;
  `wavelength domain must be finite with 0 < lam_min < lam_max, got <range>`; `medium <name>: missing params <names>`;
  `<surface type>.<name>: rows use different Laws`; `<surface type>.<name>: only (energy, value) tables are packed`.
- `BoundaryError` (`rules/boundary.py`): `<pre> -> <post>: type <surface type> (<reason>) would put the photon into
  the opaque medium <medium>`. A surface transmits into a material without RINDEX. Remove TRANSMITTANCE, or give the
  material a RINDEX.
- `GeometryError` (`geometry/tracked.py`): `the scene's PVs differ from the optics': ...`; `navigator volumes <n> /
  meshes <m> vs PVs <a> / <b>`; `coincident siblings <pvs>: no interface between them (...); Geant4 crosses one
  leaving the first`.
- The navigator's limits (`NotImplementedError` and `ValueError` in `geometry/`, `build/solids.py`,
  `build/lowering.py`): `navigator: definition <name> has <n> members (one solid per volume: the lowering gives one member per logical volume)`; `navigator: mesh definition <name> with
  daughters`; `navigator: mesh definition <name> is a sensor`; `arb8 with a twisted (non-planar) face`;
  `polytope_from_faces: the vertices are not convex in position (a concave arb8/trap base or a twisted face)`; `xtru
  polygon: ear clipping failed (self-touching or duplicate vertices)`; `csg tables: an unbounded solid (no finite
  box)`; `csg tables: a boolean deeper than <n> operands`; `navigator: srf codes overflow int32 (<n>)`; `navigator: a
  scene without volumes`; `navigator: a BVH of depth <n> > the traversal stack (<m>)`; `bvh.build_many: empty or
  unordered box ranges`; `Op: unknown operation <op> (one of <ops>)`; `unsupported=<value> (one of <modes>)`; and the
  lowering's `NotImplementedError` naming an unsupported solid type. A mesh used as a sensor, or a mesh with daughters,
  is not supported in 0.1: make the sensor a solid, or move the daughters out of the mesh.
- `edge rows: no surface type <name> (types <names>)`; `coated <name>: backing <name> has outcomes <names>; only an
  opaque backing (reflect / absorb / detect) is supported`; `not a finite decimal: <value>`: internal; report them.
