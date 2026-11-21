# Errors

lumen fails closed: when it cannot do what Geant4 would do, or cannot do it exactly, it stops with an error that
names the object, the rule and the fix. The sections below explain the messages you meet most, grouped by when you
meet them, with what each means and what to do; the last section, generated from the code, lists every message.

In the messages below, `<...>` stands for a value filled in at run time (a volume name, a number).

## Reading an error

Most lumen errors are subclasses of Python's own, so `except ValueError` still catches them:

| Error | Base | Raised when |
|---|---|---|
| `GDMLError` | `ValueError` | the GDML cannot be read as Geant4 reads it |
| `ImportRefused` | `ValueError` | the GDML is readable but lumen cannot convert it faithfully |
| `OpticsRefused` | `DescriptionError` | a material or surface uses optics lumen does not implement, or is inconsistent |
| `LibraryError` | `ValueError` | a recipe's sensor groups or optics choices do not fit the detector |
| `FormatError` | `ValueError` | a detector file is damaged, from a newer lumen, or would be overwritten |
| `SensorValuesError` | `ValueError` | a sensor value set is malformed or cannot be found |
| `UnknownParamKey` | `KeyError` | a parameter key that the detector does not have |
| `GradientNotAvailable` | `ValueError` | a derivative lumen cannot give in this version |
| `LawError` | `ValueError` | a material table has bad values at its knots |
| `ExportError` | `ValueError` | a detector file cannot be written as GDML that Geant4 reads the same way |
| `DeviceError` | `RuntimeError` | a GPU was expected but JAX is not using it |
| `ClosedOverValues` | `ValueError` | a plug-in's code captured a parameter value instead of taking it as an input |
| `ReferenceStoreError` | `LookupError` | a validation check cannot find its reference data |
| `SourceError` | `ValueError` | a source function's output breaks its declaration (docs/howto/extending.md) |
| `ModelError` | `ValueError` | a registered optics model's output breaks the rules of model code (docs/howto/extending.md) |
| `InertKeysWarning` | `UserWarning` | the GDML's materials carry keys no Geant4 optical process reads (a collaboration's own constants, for example); they are kept inert, unread, as Geant4 ignores them. `Recipe(material_keys=frozenset({...}))` chooses the keys to read instead |
| `CalibrationMismatch` | `ValueError` | a calibration was made for another detector file (its digest differs) |
| `BoundaryError`, `TreeError`, `GeometryError`, `DescriptionError`, `OpticsError` | `ValueError` | internal consistency checks of the geometry and surface build; see [the last section](#internal-consistency-checks) |

Python's own `ValueError`, `KeyError`, `MemoryError`, `FileNotFoundError`, `FileExistsError`, `PermissionError`,
`TypeError` and `RuntimeError` are also raised, with lumen messages, where no lumen type is needed.

## Converting a GDML file

### Repairs

Some GDML files, often written by detector software's own exporters, contain constructs that stock Geant4 refuses.
lumen refuses them too (a `GDMLError` naming the object and the repair) and repairs nothing on import. The repair is
one explicit step, `python -m lumen.gdml.repair SRC DST`: convert the repaired copy (the detector file records its
name and sha256 as the source) and give the same copy to Geant4. The guide
[Your own detector](../howto/your-own-detector.md#2-convert-what-geant4-refuses-lumen-refuses) shows the loop.

| Repair | The construct | What stock Geant4 does | The repair | What to do |
|---|---|---|---|---|
| `matrix_tail` | a `<matrix>` whose value count is not a multiple of its `coldim` | refuses the file (`Matrix ... is not filled correctly!`) | drops the unpaired tail values | check the table is the one you meant, then fix the file |
| `identical_redefinition` | a constant, variable, quantity, expression or matrix defined twice, identically | refuses the file (`Redefinition of constant or variable`) | keeps the first definition | remove the duplicate |
| `constant_property` | a `<property>` whose `ref` names a define that is not a `<matrix>` | refuses the file (`Matrix ... was not found!`) | deletes the property | point the property at a matrix, if you meant it |
| `undefined_matrix_property` | a `<property>` whose `ref` names nothing | refuses the file (`Matrix ... was not found!`) | deletes the property | define the matrix, if you meant the property |
| `denormal` | a nonzero number below the smallest normal double | refuses the file (`Error in expression`) | writes it as 0 | none needed, usually |
| `non_finite` | an `inf` or `-inf` matrix value | refuses the file | writes it as ±1e30 | replace it by a large finite number |

The error reads `gdml <file>: <object>: Geant4 refuses it too; the repair '<repair>' (<what it does>) writes a copy
Geant4 and lumen both read: python -m lumen.gdml.repair SRC DST, then convert DST`.

A property is read in the form Geant4 stores it, on materials and surfaces alike: a one-column matrix is a constant
property, a wider one a table. A table key (REFLECTIVITY, RINDEX, ...) given as a constant, or a constant key given as
a table, is one no Geant4 process reads: Geant4 loads the file and the key is absent, so lumen leaves it out too, with
a report line (`<owner>.<KEY>: left out: ...`). **This is usually a real mistake**: a surface's REFLECTIVITY then is
Geant4's default, 1. Write it as a two-column table and convert again.

`python -m lumen.gdml.repair SRC DST` raises `ValueError`:

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
| `gdml <path>: <solid> <name>: attribute <a> is not one Geant4 reads for <solid> (names are case-sensitive; ...): Geant4 ignores it` | an attribute spelt otherwise than Geant4 spells it (`RMAX` for `rmax`): Geant4 would ignore it and read 0. Spell it as Geant4 does |
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

The file reads, but converting it would change what Geant4 does. The structural rules (`lumen.format.check`) are the
same for a detector written in YAML, raised there as `StructureError`.

| Message | Meaning and fix |
|---|---|
| `material <name> names no composition: give a density and its components (mass fractions or atom counts), or Z and a molar mass` | every material must say what it is made of: in GDML, `<D>` and `<fraction>` or `<composite>` elements, or Z and `<atom>` |
| `material <name>: component <ref> is neither an element nor a material` | fix the reference |
| `opticalsurface <name> has no '<attribute>' attribute. Geant4 would read a default and lumen cannot guess: add it in the GDML and convert again.` | write all four of `model`, `finish`, `type` and `value` |
| `logical volume <lv> has two skin surfaces (<a>, <b>): Geant4 keeps the first registered; refusing the ambiguity` | remove one skin |
| `bordersurface <name>: needs exactly two physvolref` | a border joins two placements, in order |
| `border surface <name>: physvol <pv> names <n> physical volumes; Geant4 binds a border to the first one only, so the border is ambiguous` | give the placements unique names |
| `border surface <name>: the pair <pv1> -> <pv2> has a border already (Geant4 keeps the first)` | remove the repeated border |
| `skin <name> refers to <ref>, which is not a surface` · `border <name> refers to <ref>, which is not a surface` | point the skin or border at an `<opticalsurface>` |
| `<what>: property <key> given twice` | remove one |
| `<what>.<key>: matrix <ref> has <n> columns (a property needs 1 or 2)` | a property is a constant (1 column) or a table (energy, value) |
| `<what>.<key>: energies must be strictly ascending with >= 2 knots (got <energies>)` | sort the table, remove repeated energies. A key Geant4's optical processes never read (a collaboration's own constant) is kept inert by default; this error then comes only from a key you asked for with `Recipe(material_keys=...)`: drop it from that set |
| `<what>.<key>: <detail>` | a property value that does not evaluate |
| `tessellated <name>: no facets` | an empty mesh |

### Optics lumen does not implement, or that is inconsistent (`OpticsRefused`)

lumen implements Geant4's surface types and bulk processes (see the [vocabulary](vocabulary.md)). Anything
else is refused when the detector is converted or loaded, never silently approximated. `<what>` names the material
or surface.

| Message | Meaning and fix |
|---|---|
| `<what>: model <model> is not supported` · `<what>: finish <finish> is not supported` · `<what>: surface type <type> is not supported` | a surface form outside lumen's Geant4 surface types (refused: unified ground dielectric_dielectric, backpainted finishes, Geant4's `coated` type, LUT, LUTDAVIS, dichroic). Use a supported form |
| `<what>: unified ground dielectric_dielectric (facet refraction) is not supported` | as above (a glisur ground one reads: GLISUR's facet refraction) |
| `<what>: <key> is not supported` · `<what>: <key> (<why>) is not supported` | a property lumen does not read (SURFACEROUGHNESS, for example). Remove it, or ask for it |
| `<what>: polish <value> outside [0, <max>] (or exactly 1)` · `<what>: sigma_alpha <value> outside [<min>, <max>] (or exactly 0)` | the opticalsurface `value` is outside the range lumen's facet samplers cover (sigma_alpha 0 is read as Geant4 reads it: the lobe is the spike; above 1 rad, G4OpticalSurface's default, Geant4's own facet draw departs from the unified distribution: `lumen.rules.choices`, `SIGMA_MAX`) |
| `<what>: GROUPVEL must hold positive speeds (m / ns)` | a GROUPVEL table times the photons, as in Geant4; fix its values |
| `<what>: <key> must be a table` · `<what>: <key> must be a constant (G4OpMieHG reads a constant property)` | the property has the wrong shape for what Geant4 reads |
| `<what>: MIEHG without MIEHG_FORWARD_RATIO: G4OpMieHG aborts (mat202) at the first Mie event; add MIEHG_FORWARD_RATIO` | add the property |
| `<what>: MIEHG_FORWARD_RATIO <r> without <key>: G4OpMieHG aborts (mat202) at the first <lobe> Mie event; add <key>` | add MIEHG_FORWARD or MIEHG_BACKWARD |
| `<what>: MIEHG_FORWARD_RATIO <r> outside [0, 1]` · `<what>: <key> <g> outside [-0.999, 0.999] (a narrower lobe than float32 directions resolve)` | fix the value |
| `<what>: WLSABSLENGTH without WLSCOMPONENT (G4OpWLS absorbs and emits nothing)` | add WLSCOMPONENT, or remove WLSABSLENGTH |
| `<what>: wls_time_profile <profile> is not one of G4OpWLS's ('delta', 'exponential')` | record the time profile your Geant4 run uses (none: Geant4's default, delta) |
| `<what>: [<keys>] without the optics setting scint_finite_rise_time: ...` · `<what>: scint_finite_rise_time without [<keys>]: ...` | record Geant4's finite-rise-time switch as your run sets it (material optics setting `scint_finite_rise_time`), with a rise time per emitted component, or drop the rise times |
| `<what>: <key> must be a constant >= 0` (WLSMEANNUMBERPHOTONS, WLSTIMECONSTANT) | fix the value |
| `material <name>: no <keys> for Geant4 <version><hint>` | the material lacks Geant4 11's scintillation keys (a GDML's complete Geant4-10 keys are read as them at import) |
| `<what>: a complex index on a ground finish (the facet's Fresnel) is not supported` | use a polished finish |
| `opticalsurface <name>: its reading detects nothing` | the surface in front of a `border_into` sensor is read as any surface is (the same refusals) and cannot detect the way Geant4 would. Use a detecting form, or the `any_entry` rule |
| `opticalsurface <name>: its reader returned no detecting SensorEntry` | a reader that is not a detecting reading; the sensor group's `surface` is `lumen:geant4` or a registered reader |
| `WLS boxes <lvs> are thinner than the band where they are placed: ...` | a wavelength-shifting film the navigator cannot resolve: lumen lowers no film; thicken it in the recipe (SBND's CPA foil split, lumen-tpc) |

### The recipe and the sensor groups

| Message | Error | Meaning and fix |
|---|---|---|
| `no volume, physvol or physvol path <name> placed under the world` | `ValueError` | the recipe's `detector_volume` names no logical volume, physvol or physvol path placed under the world (names are stored without Geant4's pointer suffix where unique) |
| `detector volume <name> is placed <n> times (<paths>): give the physvol path of the one to simulate` | `ValueError` | the recipe's `detector_volume` is a logical volume placed more than once: give one placement's physvol path |
| `gdml: no volume of material matching <materials>` | `ValueError` | `detector_volume="auto:<materials>"` found no such volume |
| `recipe <name> needs the inputs <names>` | `ValueError` | pass `inputs={...}` to `lumen.convert` |
| `sensor group <name>: rule <rule> is not one of ('border_into', 'any_entry')` | `ValueError` | spell the rule |
| `sensor group <name>: give exactly one of lv and lv_pattern` | `ValueError` | select by name or by pattern, not both |
| `sensor groups <a> and <b> share the volume <lv> but read its surfaces differently` | `LibraryError` | `border_into` groups that share a volume (by the id map's columns) read it with one surface reader and one efficiency on entry |
| `sensor group <name>: border_into reads its surfaces with one of <readers>, not <surface>` | `LibraryError` | the surface reader is `SensorGroup.surface`: `lumen:geant4` or a registered reader |
| `sensor group <name>: any_entry counts every entry and reads no surface (got surface <surface>; lumen:counting is removed): leave surface unset` | `ValueError` | an `any_entry` group takes no surface reader |
| `sensor group <name>: a group with a value set has every absorption entering counted and its value set's efficiency applied after tracking: drop the setting entry_efficiency` | `ValueError` | drop the setting; the value set's `efficiency` applies after tracking |
| `sensor group <name>: a name is letters, digits and '_', not starting with a digit (it is a segment of the keys sensors/<group>/<field>)` · `two sensor groups are named <name>` | `ValueError` | rename the group |
| `volume <lv> is selected by the sensor groups <names>: a sensor is in one group` | `LibraryError` | narrow the groups' `lv` or `lv_pattern` |
| `sensor group <name>: any_entry counts with its setting 'entry_efficiency' or a value set (its sensor model applies the efficiency after tracking)` | `LibraryError` | give the group `settings={"entry_efficiency": ...}`, or a value set (`values=`) |
| `sensor group <name>: unknown settings <unknown> (the settings are <SETTINGS>)` · `<what>: unknown setting <key> (known: <keys>)` · `<what>: setting <key> must be <type>, got <value>` | `ValueError` | a misspelt sensor-group or optics setting (`entry_efficiency`, not `efficiency`) |
| `material <name>: optics model <model> is not one this library reads` · `surface <name>: optics model <model>; a surface is read by Geant4's reading, 'geant4'` | `LibraryError` | a recipe bound an optics model that is not installed |
| `material <name>: medium <medium> is defined by another material already` · `material <name> of the tree is not in the detector file` | `LibraryError` | a recipe decision about materials that does not fit the file |
| `logical volume <lv> has two skin surfaces (<a>, <b>): Geant4 keeps the first registered; refusing the ambiguity` | `LibraryError` | a recipe added a second skin |
| `WCSim PMT <type> (<class>): QE tables of <n> / <m> points, not 20` · `glass_faces: <path>: interior LV <lv> has glass faces of <faces>` · `ANNIE: PMTwiseQE on (pmt_qe.txt by WCSim tube id) is not implemented` | `ValueError` | the shipped WCSim recipes (SK, WCTE, ANNIE) check the GDML and the WCSim source they read. Use the GDML and WCSim version they were written for |
| `sbnd: OpDet channels are not unique` · `LArSoft table <key>: length unit <unit> is not one of <units>` | `ValueError` | the same for the shipped LArSoft recipes (SBND, MicroBooNE) |

### Material tables (`LawError`)

A table is checked at its own knots: it is linear in energy between them and keeps its end values beyond them, so
its values there bound it at every wavelength.

| Message | Meaning and fix |
|---|---|
| `<name>: missing parameters <names>` · `<name>: non-finite values at its points` · `<name>: <quantity> out of range at <lam> nm: <value> (allowed [<a>, <b>])` | a table with missing, infinite or out-of-range values |
| `<medium>.<channel>: a channel length law must output a length` · `<name>: the <form> law declares no points to be checked at (Law.points)` · `group_index needs an index law` | a plug-in's material law is malformed |

## Loading a detector

| Message | Error | Meaning and fix |
|---|---|---|
| `<path>: written by another lumen (format <schema>); this lumen reads <format>: <remedy>` | `FormatError` | a detector file, sensor set or calibration written by another lumen version: lumen reads no other format. The remedy says how to write it again: convert the detector again (detector files and sensor sets), or save the calibration again from a detector this lumen loads |
| `<path>: <file> does not match the manifest (changed since it was written)` | `FormatError` | a file inside the detector file was edited. Convert again; never edit a detector file by hand |
| `load(parts=): <target> is not materials/<M> of this detector` | `ValueError` | name a material record of the detector (a surface's parts are not hooks in 0.2) |
| `<record>: optics model <name> is not lumen's own nor registered (...)` | `LibraryError` | import the module that registers the model (or install its package) before `lumen.load` |
| `<record>: optics model <name> is not a medium model` (or surface) | `LibraryError` | a surface model bound to a material, or the reverse |
| `<what>: outcome probabilities outside [0, 1] or not summing to 1 at <params>` | `ModelError` | the model's `probs` over the six outcomes, checked at its first build over incidences and its parameters' range corners |
| `<what>: its sampler does not draw its density (chi2/ndf <x>)` | `ModelError` | the lobe's `sample` and `logpdf` disagree; fix one |
| `with_responses: <name> declares <keys>, keys the detector has already (rename them)` | `ValueError` | a response parameter named like a sensor model's value (lumen:sensor has `gain`, `qe`, ...): rename it |
| `<path> exists: a detector file is never overwritten (pass overwrite=True)` | `FormatError` | `lumen.convert(..., overwrite=True)`, or another path |
| `non-finite number <x>: the detector file stores finite numbers only` · `cannot store <type> in JSON` · `two arrays at <path>` | `FormatError` | a recipe decision produced a value the format cannot store |
| `no detector file at <path> (a detector is its file: convert it first, lumen.convert)` | `FileNotFoundError` | pass the detector file's path, or convert it first |
| `<group>: <name> is registered already` | `ValueError` | two packages register the same name |

### Calibrations

| Message | Error | Meaning and fix |
|---|---|---|
| `no calibration <name> in <directories>` | `FileNotFoundError` | save it in a directory on `LUMEN_DETECTORS`, or check the name |
| `calibration <path> exists: choose a new name` | `FileExistsError` | calibrations are never overwritten |
| `calibration parent cycle through <name>` | `ValueError` | two calibrations name each other as parent |
| `calibration <name>: unknown keys <keys>` | `KeyError` | the calibration was made for another detector, or for a key it does not have |

### Sensor value sets (`SensorValuesError`)

| Message | Meaning and fix |
|---|---|
| `no sensor value set <ref>; <package> ships: <names>` | spell the set; the package's sets are listed |
| `<ref>: no package <package> to read its value set from (install it, or give a path)` | install the package that ships it, or give a file path |
| `value set <name>: for model <model>; this model is 'lumen:sensor'` | the file is not a `lumen:sensor` value set (a file naming `lumen:pmt`, the old name, is refused: set its `model` to `lumen:sensor`) |
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
| `<key> has a value outside [<lo>, <hi>] <unit> or not finite` | `ValueError` | the value is outside the key's allowed range (`params.info(key).range`) or is NaN or infinite; a probability is in [0, 1], a length positive (the file's check, `Params.set` and every run apply one predicate, `lumen._core.domains.admits`) |
| `<key>: shape <shape> expected, got <shape>` | `ValueError` | a per-sensor value must have one entry per sensor of the group |
| `<key>: a table property needs a Table` · `<key>: a constant property needs a number` | `ValueError` | give the value in the shape the GDML gave it |
| `<key>: <n> knots expected, got <shape> values at <shape> energies; a new knot count is a new detector file` | `ValueError` | a calibration keeps the table's energies; to change them, edit the GDML and convert again |
| `<key>: the energies must be strictly ascending` | `ValueError` | sort the table |
| `parameter key <key>: not materials/<M>/<KEY>, surfaces/<S>/<KEY> or sensors/...` · `parameter key <key>: no <material or surface> <name>` · `parameter key <key>: <name> has no property <prop> (adding one is a new detector file, except a Geant4 default the model reads)` | `ValueError` | the key does not name something in the detector file |
| `<key>: the opticalsurface value is a number` · `<key>: a Geant4 default property is set to a number` | `ValueError` | give a number |

## Simulating

| Message | Error | Meaning and fix |
|---|---|---|
| `simulate: one sensor group with a sensor model (lumen.sensors) is supported, the sensor set has <names>` | `ValueError` | lumen reads at most one group with a sensor model; the other groups (counting) give arrivals only, and a detector without one gives arrivals and no histograms. Several PMT types in one group share one value set |
| `simulate: the source is a <type>, not a lumen.sources.Source: give photons you have as FromArrays, a source of your own as a @lumen.source function` | `TypeError` | `simulate` takes a source only |
| `the histograms need <x> GB (<each> GB by sensor group) and the photons in flight <r> GB, over the device's <y> GB: shorten the readout gates, widen the time spreads' declared lower ranges (coarser nodes) or give a smaller chunk` | `MemoryError` | lumen refuses before launch (a derivative run holds the histograms twice). Do what the message lists |
| `the photon loop expects the GPU (<why>) but JAX runs on <backend>: the CUDA plugin did not load (see JAX's log above); run on a working GPU node, or pass backend='cpu'` | `DeviceError` | a GPU job whose JAX did not start CUDA. Check `jax.devices()` |
| `tallymap: the kernel backend cannot run on the CPU here: <why> Fix: install g++ (or point $CXX at GCC or Clang with OpenMP; Linux only); or pass backend='xla' (plain JAX, much slower)` | `tallymap.UnsupportedError` | from tallymap, running on the CPU (`backend="auto"` without a GPU, or `"cpu"`): no C++ compiler with OpenMP. Install g++ (or set `CXX`), or pass `backend="xla"` |
| `simulate: a launch of <n> photons appends <records> deposit records, over the <slots> that fit the device beside the readout's cells: give a smaller chunk` | `MemoryError` | a given `chunk` whose launch's records do not fit beside the readout (the default chunk runs such a launch again, smaller). Give a smaller chunk, or leave it out |
| `<source> emits its <n> rows, one photon each: simulate it with samples=<n> (or None), not <samples>` | `ValueError` | a source of one photon per row (`FromArrays`, a `Point` with per-row fields) is traced row by row, each once: fewer samples would drop rows, more would trace empty ones. Leave `samples` out |
| `<source>: its per-row fields have different numbers of rows <counts>: give each the same rows (or one value for all)` | `ValueError` | the columns of a per-row source differ in length. Give every column the same rows |
| `source <name>: a parameter that moves its photons (declare it moves=True and give it in wrt)` | `GradientNotAvailable` | a tangent reached the photons' position, direction or wavelength without the moving-source method: declare the parameter `moves=True` and list its key in `wrt`; a wavelength or intensity parameter enters by `reweight` |
| `source <name>: Photons.<field> has shape <s>, expected <s> float (n = 256)` | `SourceError` | the function's output, checked at first use with n = 256 |
| `source <name>: host callbacks [...] (a source is plain JAX)` | `SourceError` | remove `jax.pure_callback` / `io_callback` from the source |
| `simulate: '<key>' carries a derivative but is not in wrt: name every key you differentiate in wrt (it decides the derivative's roulette, its one-sided derivative at a range's edge and the moving-source method)` | `GradientNotAvailable` | a key carries a tangent (`jax.jvp`, `jax.grad`) but `wrt` does not name it: add it to `wrt`, and give the same `wrt` to the runs you compare with |
| `simulate: diagnostics run in one chunk (their cells are indexed by the photon); samples <n> > chunk <c>` | `ValueError` | give `chunk >= samples` with `diagnostics=` |
| `<source> reads <key>: pass the detector's parameters` | `KeyError` | a source that reads detector values (scintillation yields) was emitted without them |
| `<material>: no <keys> (Geant4 <version> emits none)` | `KeyError` | a scintillation source in a material without scintillation properties |
| `Point: give a wavelength (lam) or a spectrum, not both` | `ValueError` | give exactly one |
| `time spread <name>: a scintillation emission profile combines in closed form only with the 'gaussian' time spread; use form 'gaussian' for scintillation light` | `ValueError` | use a gaussian time spread in the value set for scintillation light |
| `<name>: ascending wavelengths and a non-negative, non-zero spectrum of the same length` · `spectrum <name>: needs a non-negative intensity with positive integral` · `spectrum <name>: negative intensities` | `ValueError` | fix the source spectrum |
| `the photon loop (<name>) closes over <n> float array(s) <names>: pass every value through the Operands (lumen.transport.step.operands)` | `ClosedOverValues` | a plug-in captured a number in a closure; such a number would silently get a zero derivative. Declare it as a parameter |
| `max_vertices must be >= 1, got <n>` · `photon_chunks: samples >= 0 and chunk > 0, got <a>, <b>` | `ValueError` | invalid run settings |

## Health

`Light.health` says where the light went. Each weight is in the units of the source's photons (`nonfinite`, `invalid`
and the record counts: counts). It is a `Health` record: read a key by attribute, `light.health.lost`.

Every photon's start weight ends in one fate (`Health.FATES`), each the photon's weight where it ends:
start = `absorbed_bulk` + `absorbed_surface` + `lost` + `escaped` + `distance_cut` + `abandoned` + `truncated` +
`rouletted` + `reweighted`, exactly (to float32 rounding per step) at the reference values;
`light.health.balance(start)` returns start minus the fates. The detected light (`arrivals`) is a tally of the detected
share of `absorbed_surface`, not a fate of its own. A healthy run has the fates below `absorbed_*` at zero or tiny;
`rouletted` and `reweighted` are 0 in expectation.

| Key | Meaning | What to do |
|---|---|---|
| `absorbed_bulk` | absorbed in the media (the path's absorption, a WLS yield below 1) | none: the physics |
| `absorbed_surface` | absorbed at surfaces (1 - the continuing shares; the detected share is part of it) | none: the physics |
| `escaped` | met no surface and no bulk event within the maximum step distance (its default, the geometry's reach): left the detector | a hole in the detector volume, or a detector volume chosen too small |
| `distance_cut` | the same within a `Controls.max_distance` shorter than the geometry's reach: cut there | expected with a short `max_distance` |
| `rouletted` | the weight the roulette ended minus the weight it added (0 in expectation) | none; its spread is the roulette's noise |
| `reweighted` | the expected weight the photons went on with minus the weight they carried: the drawn outcomes' and directions' live / proposal factors (0 in expectation) | none; a mean away from 0 over seeds is an estimator bias |
| `abandoned` | stopped on a surface after repeated zero-length steps (Geant4's stuck-track rule) | none if tiny; many: a degenerate surface |
| `truncated` | the weight still bouncing at the photon loop's guard (`lumen.run.MAX_VERTICES` vertices: a photon that cannot end) | check for a lossless trap (a perfect mirror around a medium that absorbs nothing) |
| `lost` | a crossing whose pair of volumes has no surface in lumen's tables. The known case: a volume without RINDEX overlapping a transparent one puts the photon inside the opaque one, which it then leaves (known limits); stock Geant4 kills it there too (NoRINDEX). A zero-length step out of a volume that only touches goes on, as Geant4's StepTooSmall | about 1e-4 of the weight in a large PMT detector is this case; much more: find where (below) |
| `invalid` | elements of traced values (under `jax.jit`, `jax.grad`, ...) outside their key's declared range (`params.info(key).range`) or not finite; a concrete one is refused when set. The derivative at such a value is NaN | keep the traced values inside their ranges (a fit: bound or transform the parameter) |
| `records` | the deposit records the run's launches appended | under `jax.jit` or `jax.vmap`, give `simulate(records=)` at least this number of the run with values |
| `records_dropped`, `records_mismatch` | records past the capacity given, and photons whose two passes differed; either makes the outputs NaN | 0 in a run that fits: give a larger `records=` |
| `nonfinite` | cells of the exact sums made NaN (tallymap's count: a non-finite value emitted, or a sum past the accumulator's top, about 2^166) | the NaN cells are in the outputs; a non-finite value comes from a quantity singular at the point ([known limits](../explanation/known-limits.md#a-nan-in-the-exact-sums)) |

**Where `lost` photons end.** `lumen.diagnostics.Trace(steps, n)` passed to `simulate(..., diagnostics=)` records each photon's
first `steps` steps (position, direction, volume, the pair of volumes crossed, and the lost weight). The pair crossed
at the step with nonzero lost weight names the two volumes; the position says where. `python -m lumen.compare geometry` checks
lumen's point location against Geant4's on the same GDML: if it finds no mismatch, the geometry reads the same in
both, and the lost light comes from an overlap of an opaque volume and a transparent one (known limits).

## Readout

| Message | Error | Meaning and fix |
|---|---|---|
| `readout: expected lumen.readout.Histograms (or a light of one modelled group), got <type>` | `TypeError` | pass the `Light` that `simulate` returned, or `light.group(name)` |
| `readout: sensor group <name> has no sensor model` | `ValueError` | the group has no `model`, so there is nothing to read out |
| `sensor group <name>: no time histogram (readout 'pe'); simulate with a timed readout` | `ValueError` | the value set's readout is `pe`; use one with `hits` or `waveform` |
| `sensor group <name>: value set <set> has no waveform readout (readout <readout>); use a value set with a pulse and readout 'waveform'` | `ValueError` | as the message says |
| `waveform_nll: data of shape <shape>, the expected waveform is <shape>` | `ValueError` | the data must have one row per channel and one column per sample |
| `node_spacing: the narrowest time spread must be > 0 ns; declare its range in the value set` | `ValueError` | a value set whose timing ranges do not define a time grid |
| `Grid: give lam_edges or lam_knots, not both` · `Grid.<name>: at least two ascending values` · `Grid: nt >= 0 and dt > 0` | `ValueError` | a histogram grid built by hand with invalid values |

## Exporting to GDML (`ExportError`)

| Message | Meaning and fix |
|---|---|
| `surface <name>: optics model <model> has no Geant4 form; pass on_lossy='omit' to write the GDML without it (marked lossy, not a Geant4 reference)` | a surface lumen models that Geant4 cannot express |
| `surface <name>: a coated surface; GDML has no coated type (it reads as x_ray)` · `surface <name>: <field> <value> is not one Geant4 knows (one of <names>)` | the surface cannot be written as Geant4 would read it |
| `border surface <name>: physvol <pv> names <n> physical volumes; Geant4 binds a border to the first one only, so the border is ambiguous` · `border surface <name>: the pair <pv1> -> <pv2> has a border already (Geant4 keeps the first)` · `border <name>: a physvol of an assembly (<pv1>, <pv2>); Geant4 renames the imprinted placements, so the border cannot be bound` | the borders would bind differently in Geant4 |
| `logical volume <lv> has two skin surfaces (<a>, <b>): Geant4 keeps the first registered; refusing the ambiguity` | as on import |
| `volume <name>: material <material> is not defined (and is no G4_ NIST name)` | the detector file references a missing material |
| `non-finite value <text>: Geant4 reads no inf or nan` · `denormal value <text>: Geant4's reader refuses it (strtod sets ERANGE)` · `value <text> is <literal> in Geant4's units: not a normal double` · `value <text>: the literal <literal> does not read back exactly` | a value that cannot be written so that Geant4 reads the same double |
| `solid <name>: type <type> has no GDML spelling in the exporter (supported: <types>)` · `solid <name> (<type>): attributes <names> are not <type> attributes` | a solid the exporter cannot write |
| `the placements are not in an order a GDML file can hold (...)` | (a verbatim message) a detector file built in an order GDML cannot express |
| `a matrix and a position share a name` · `matrix <name>: Geant4 would define the constant <c> twice` | a name clash in the written file; report it |
| `surface <name>: lumen:larsoft_reflector needs REFLECTIVITY and DIFFUSE_FRACTION tables on the same energies` · `surface <name>: lumen:larsoft_reflector is a dielectric_metal, unified, ground surface with sigma_alpha 0` (`ValueError`) | a LArSoft reflector surface whose record cannot be written in its Geant4 form |
| `unknown Geant4 target(s) <names>: the exporter knows <names>` · `<option>=<value>: one of <allowed>` (`ValueError`) | an export option spelt wrongly |
| `no sensor set <name> (the detector file has <names>)` | pass an existing sensor set |

## Validation checks

| Message | Error | Meaning and fix |
|---|---|---|
| `<url>: cannot fetch the release asset (<reason>)` | `ReferenceStoreError` | the reference data is fetched on first use from its release asset: check the network, set `LUMEN_ASSETS` to a mirror of the release assets, or set `LUMEN_DATA` to a directory that already holds the products |
| `no <check> check named <name>; the manifests list <names>` · `no reference product <id> in the manifests` · `<path>: missing from the store` · `<path>: sha256 differs from the manifest (<digest>)` · `<url>: sha256 differs from the manifest (<digest>)` · `<path>: schema <schema>, this lumen reads <schema>` | `ReferenceStoreError` | the reference data is missing, or not the one the manifest names |
| `<path> is inside a reference store: write check outputs elsewhere (--out)` | `PermissionError` | outputs never go into the reference store |
| `the geometry check needs Geant4's answers: give the g4ref binary (--g4ref)` · `<gdml> is not the GDML <path> was converted from (<sha256>)` | `ValueError` | the geometry check compares against the GDML the detector was converted from |
| `g4ref failed (<code>): <output>` · `<csv>: <n> answers for <m> points (probed on other points?)` | `RuntimeError` | g4ref did not run, or answered for other points |
| `sensor match: <n> reference sensors farther than <tol> mm from any lumen sensor (largest <d> mm)` | `ValueError` | the reference was made from another geometry |
| `<n> sensors but <m> sensor volumes in the navigator` | `ValueError` | internal; report it |

## Internal consistency checks

`TreeError`, `DescriptionError`, `OpticsError`, `GeometryError`, and `NotImplementedError` or `ValueError` from
`geometry/` or `build/`, come from checks inside the geometry and surface build. They should not happen for a detector
that converted cleanly. If you see one, open an issue with the GDML, the recipe and the full message.

Four have a user-side fix:

- `BoundaryError: <pre> -> <post>: type <surface type> (<reason>) would put the photon into the opaque medium
  <medium>`: a surface of a registered surface model (no Geant4 reading says what happens there) transmits into a
  material without RINDEX. Give the material a RINDEX, or bind a surface Geant4 reads (lumen follows Geant4's rule for
  its type and finish).
- `navigator: mesh definition <name> is a sensor` or `navigator: mesh definition <name> with daughters`: not
  supported. Make the sensor a solid, or move the daughters out of the mesh.
- `sensor PV <name> also has a skin surface <s> on its LV`: remove the skin from the sensor's logical volume.
- `bordersurface <name> (<a> -> <b>): no adjacent world instances (...)`: a border from the detector volume into a
  volume outside it other than its mother or a sibling that encloses it. lumen's outside is one region: choose a
  detector volume whose whole boundary meets one volume. A border that starts outside the detector volume is dropped
  with a warning instead.

<!-- every message: generated by tools/docs/errors_reference.py; do not edit by hand -->
## Every message

Every message a raise site in lumen builds, with its error and the module that raises it, generated from the code.
`<name>` stands for a value filled in at run time.

| Message | Error | Module |
|---|---|---|
| `module 'lumen' has no attribute <name>` | `AttributeError` | `lumen` |
| `<what>: <n_rand> uniforms; a lobe draws at most <LOBE_UNIFORMS>` | `ModelError` | `lumen._core.checks` |
| `<what>: a bulk lobe has a density (logpdf) and no surface normal` | `ModelError` | `lumen._core.checks` |
| `<what>: host callbacks <bad> (model code is plain JAX, per photon)` | `ModelError` | `lumen._core.checks` |
| `<what>: its density integrates to <sum>, not 1` | `ModelError` | `lumen._core.checks` |
| `<what>: its sampler does not draw its density (chi2/ndf <chi2>)` | `ModelError` | `lumen._core.checks` |
| `(a message built at run time)` | `GradientNotAvailable` | `lumen._core.errors` |
| `lumen: a jitted function traced with an object that no longer exists` | `ReferenceError` | `lumen._core.static` |
| `(a message built at run time)` | `_closure_error` | `lumen.api` |
| `<out> exists: a detector file is never overwritten (pass overwrite=True)` | `FormatError` | `lumen.api` |
| `Light.event: event <e> of <events>` | `IndexError` | `lumen.api` |
| `Light.event: this light has no event axis (simulate(..., events=E))` | `ValueError` | `lumen.api` |
| `Light: no histograms of sensor group <name>; the modelled groups are <groups>` | `KeyError` | `lumen.api` |
| `convert: <pkg> has no detector <name> (its DETECTORS: <detectors>)` | `ValueError` | `lumen.api` |
| `convert: <source> names its own recipe and inputs (the package's DETECTORS)` | `ValueError` | `lumen.api` |
| `convert: a GDML file needs its recipe (lumen.Recipe)` | `ValueError` | `lumen.api` |
| `convert: a GDML file or a YAML detector, not <__name__>` | `TypeError` | `lumen.api` |
| `convert: data is the data directory of a detector package's detector ("package:name")` | `ValueError` | `lumen.api` |
| `convert: only a GDML file takes a recipe; a YAML detector holds its own decisions: pass recipe=None` | `ValueError` | `lumen.api` |
| `convert: out, the detector file's path, is required` | `TypeError` | `lumen.api` |
| `load(parts=): <target> is a channel's path: give it a phase function (lumen.media.phase_function)` | `ValueError` | `lumen.api` |
| `load(parts=): <target> is not materials/<M> of this detector (a surface is read by lumen's own surface model: its parts are not hooks in 0.2)` | `ValueError` | `lumen.api` |
| `load(parts=): <target>: <model> is neither a medium model (lumen.medium_model) nor one of lumen's own (<MEDIA>)` | `LibraryError` | `lumen.api` |
| `load(parts=): <whole> is given two parts; one per material in 0.2` | `ValueError` | `lumen.api` |
| `load(parts=): a phase function is given for one material's channel, materials/<M>/<channel>, not <target>` | `ValueError` | `lumen.api` |
| `no detector file at <path> (a detector is its file: convert it first, lumen.convert)` | `FileNotFoundError` | `lumen.api` |
| `save_calibration: give a directory (LUMEN_DETECTORS is not set)` | `ValueError` | `lumen.api` |
| `simulate: <bad> photon(s) have a label outside the events [0, <n_ev>)` | `ValueError` | `lumen.api` |
| `simulate: <n> photons; a run traces at most <MAX_PHOTONS> (2^31 - 1, the range of the photon ids): split the run into calls of at most that many photons, each with its own seed` | `ValueError` | `lumen.api` |
| `simulate: a <__name__> of one photon per row emits its <rows> rows, each once; samples=<samples> is not its row count (leave samples out)` | `ValueError` | `lumen.api` |
| `simulate: a diagnostic reads <reads>; its reads are deposit fields of <FIELDS>` | `TypeError` | `lumen.api` |
| `simulate: a launch of <size> photons appends <appended> deposit records, over the <room> that fit the device beside the readout's cells: give a smaller chunk` | `MemoryError` | `lumen.api` |
| `simulate: a probe diagnostic runs in one chunk (its cells are indexed by the photon); samples <samples> > chunk <chunk>: pass chunk >= samples` | `ValueError` | `lumen.api` |
| `simulate: a source row stands for <top> photons; a truth row is one photon: run samples >= the photons emitted (or split the rows)` | `ValueError` | `lumen.api` |
| `simulate: chunk is a number of photons >= 1, got <chunk>` | `ValueError` | `lumen.api` |
| `simulate: diagnostic outputs <clash> are names of the run's own outputs or of another diagnostic's: rename the diagnostic's outputs` | `ValueError` | `lumen.api` |
| `simulate: events is a positive number of events, got <events>` | `ValueError` | `lumen.api` |
| `simulate: events=E reads each photon's event from its source's label: give the source a label (an int per photon row, or per Geant4 step of CherenkovSteps and ScintillationSteps)` | `ValueError` | `lumen.api` |
| `simulate: samples is the number of Monte Carlo photons to draw from the <__name__> (a positive int), got <samples>` | `ValueError` | `lumen.api` |
| `simulate: sensor group <name> has a timed sensor model and no readout gate: pass gate=(start, end) in ns after emission, or set the group's readout={"gate": (start, end)} in its recipe` | `ValueError` | `lumen.api` |
| `simulate: the source is a <__name__>, not a lumen.sources.Source: give photons you have as FromArrays, a source of your own as a @lumen.source function` | `TypeError` | `lumen.api` |
| `simulate: truth is a lumen.readout.Truth with keep one of <KEEPS>, got <truth>` | `ValueError` | `lumen.api` |
| `simulate: truth rows are built on the host from each chunk: run them outside jax.jit and jax.vmap` | `ValueError` | `lumen.api` |
| `simulate: truth rows decide each photon's fate at the run's own values; a reference= other than params is refused (leave it out)` | `ValueError` | `lumen.api` |
| `simulate: with truth rows the roulette must be a number <= 1 (got <r>)` | `ValueError` | `lumen.api` |
| `the readout's cells need <cells> GB (float32; with the photons in flight <fixed> GB), over the device's <limit> GB: shorten the readout gates, widen the time spreads' declared lower ranges (coarser nodes) or give a smaller chunk` | `MemoryError` | `lumen.api` |
| `<free> sensors are in no sensor group, e.g. <free> (id-map columns <cols>): the groups that select a volume select each of its sensors, in exactly one group` | `ValueError` | `lumen.build.build` |
| `sensor <sensor_pvs> is in the sensor groups <owner> and <name>: a sensor is in one group` | `ValueError` | `lumen.build.build` |
| `with_responses: <name> declares <taken>, keys the detector has already (rename them)` | `ValueError` | `lumen.build.build` |
| `with_responses: <name> reads group <group>; the groups are <groups>` | `ValueError` | `lumen.build.build` |
| `<what> has YIELDRATIO <tolist>: Geant4 10.7 gives the fast component min(YIELDRATIO, 1) of the light; give a finite constant >= 0 (the fast share)` | `TranslationRefused` | `lumen.build.import_rules` |
| `<what> has the Geant4-10 scintillation keys <old> and the Geant4 11 keys <new>: Geant4 10.7 reads only the first, Geant4 11 only the second; delete one of the two sets` | `TranslationRefused` | `lumen.build.import_rules` |
| `<what> has the Geant4-10 scintillation keys <old> without <need> (Geant4 10.7 aborts at its first scintillating step): add <need>` | `TranslationRefused` | `lumen.build.import_rules` |
| `<what> has the Geant4-10 scintillation keys <old> without a FASTCOMPONENT or SLOWCOMPONENT spectrum (Geant4 10.7 emits no light from them): add the spectrum, or delete the keys` | `TranslationRefused` | `lumen.build.import_rules` |
| `<what>: Geant4 10.7's <ref> components read as <comps>` | `AssertionError` | `lumen.build.import_rules` |
| `<what>: Geant4-10 keys left after the translation` | `AssertionError` | `lumen.build.import_rules` |
| `<what>: SCINTILLATIONYIELD changed` | `AssertionError` | `lumen.build.import_rules` |
| `<what>: component <i>'s decay time <value> != <tau>` | `AssertionError` | `lumen.build.import_rules` |
| `<what>: component <i>'s rise time <got> != <rise>` | `AssertionError` | `lumen.build.import_rules` |
| `<what>: component <i>'s share <f> != Geant4 10.7's <share>` | `AssertionError` | `lumen.build.import_rules` |
| `<what>: component <i>'s spectrum differs from Geant4 10.7's` | `AssertionError` | `lumen.build.import_rules` |
| `WLS boxes <thin> are thinner than the band where they are placed: the navigator does not resolve them, and lumen lowers no film; thicken them in the recipe (a coated foil: split it into a coating half and a substrate half)` | `ImportRuleError` | `lumen.build.import_rules` |
| `<section> <name>: parameter <join> has no recorded source (a reader that adds a parameter must record where it comes from)` | `LibraryError` | `lumen.build.library` |
| `<what>: optics model <name> is not one this build reads as a <kind> (<table>; a model of your own is passed as a value: lumen.load(..., parts=))` | `LibraryError` | `lumen.build.library` |
| `border surface <s>: its instances are read two ways (<sel> of <pvs> as a sensor, <hows> ways)` | `OpticsError` | `lumen.build.library` |
| `bordersurface <name> (<a> -> <b>): no adjacent world instances (a border from the detector volume to a volume outside it other than its mother or an enclosing sibling: lumen's outside is one region)` | `OpticsError` | `lumen.build.library` |
| `material <m> of the tree is not in the detector file` | `LibraryError` | `lumen.build.library` |
| `material <m>: medium <name> is defined by another material already` | `LibraryError` | `lumen.build.library` |
| `medium model <name>: return a lumen.media.MediumSpec, not <__name__>` | `ModelError` | `lumen.build.library` |
| `sensor group <name>: any_entry counts with its setting 'entry_efficiency' or a value set (its sensor model applies the efficiency after tracking)` | `LibraryError` | `lumen.build.library` |
| `sensor groups <prefix> and <name> share the volume <lv> but read its surfaces differently` | `LibraryError` | `lumen.build.library` |
| `surface <name>: optics model <name>; a surface is read by Geant4's reading, 'geant4'` | `LibraryError` | `lumen.build.library` |
| `volume <lv> is selected by the sensor groups <name>: a sensor is in one group (groups sharing a volume select its sensors by the id map's columns, count them by one rule and apply the efficiency after tracking)` | `LibraryError` | `lumen.build.library` |
| `(a message built at run time)` | `OpticsError` | `lumen.build.lookup` |
| `<what>: surface type <name> takes parameters <want>, got <have> (missing <want>, unknown <have>)` | `DescriptionError` | `lumen.build.lookup` |
| `PV <n> bound to two sensors (<sensor>, <sensor>)` | `DescriptionError` | `lumen.build.lookup` |
| `bindings for materials that do not occur in the tree: <dangling>` | `DescriptionError` | `lumen.build.lookup` |
| `border (<pre> -> <post>): <s> is <what>` | `DescriptionError` | `lumen.build.lookup` |
| `border sensor <s> (<a> -> <z>): <pre> is not a sensor PV, so a detection leaving it has no sensor index to go to` | `DescriptionError` | `lumen.build.lookup` |
| `border surface <s> (<a> -> <b>): the volumes neither touch (mother/daughter, siblings) nor meet across a coincident face (a daughter's face on its mother's): Geant4 would never cross it; check the border's physvols` | `OpticsError` | `lumen.build.lookup` |
| `border surface <s> (<a> -> <z>) into sensor PV <post>: ambiguous (the sensor's surface type is the boundary of a sensor; a border may name the sensor itself when its binding has entry='border')` | `DescriptionError` | `lumen.build.lookup` |
| `border surface <s> (<pre> -> <post>): no instance of one has an instance of the other in the tree` | `DescriptionError` | `lumen.build.lookup` |
| `border surface <s> (<pre> -> <post>): unknown physvol <n>` | `DescriptionError` | `lumen.build.lookup` |
| `column <c>: only (energy, value) tables are packed, not <law>` | `OpticsError` | `lumen.build.lookup` |
| `internal: a group pair counted as touching without a border has no such pair` | `OpticsError` | `lumen.build.lookup` |
| `material <m> bound to <med>, which is not a library medium` | `DescriptionError` | `lumen.build.lookup` |
| `materials without a binding: <unbound>` | `DescriptionError` | `lumen.build.lookup` |
| `medium <name>: expected a MediumEntry(bulk.Medium, params)` | `DescriptionError` | `lumen.build.lookup` |
| `medium <name>: missing params <err>` | `OpticsError` | `lumen.build.lookup` |
| `sensor <name>: expected a SensorEntry(SurfaceType, params)` | `DescriptionError` | `lumen.build.lookup` |
| `sensor <name>: surface type <name> cannot detect` | `DescriptionError` | `lumen.build.lookup` |
| `sensor <sensor> on <lv>: expected <count> instances, found <covered>` | `DescriptionError` | `lumen.build.lookup` |
| `sensor PV <n> also has a skin surface <skin> on its LV: ambiguous (the sensor's surface type is the boundary of a sensor)` | `DescriptionError` | `lumen.build.lookup` |
| `sensor PV <n> has entry='border' but no border bound to <sensor> leads into it: it would never detect` | `DescriptionError` | `lumen.build.lookup` |
| `sensor binding <sb>: <sensor> is not a library sensor` | `DescriptionError` | `lumen.build.lookup` |
| `sensor binding <sb>: LV <lv> does not occur in the tree` | `DescriptionError` | `lumen.build.lookup` |
| `sensor binding <sb>: PV <pv> does not occur in the tree` | `DescriptionError` | `lumen.build.lookup` |
| `sensor binding <sb>: entry must be 'any' or 'border'` | `DescriptionError` | `lumen.build.lookup` |
| `sensor binding <sb>: give exactly one of lv= or pv=` | `DescriptionError` | `lumen.build.lookup` |
| `sensor entry <name>: crossings into it have different normal-incidence absorptance (<by_entry> and <a_n> at <lam> nm)` | `OpticsError` | `lumen.build.lookup` |
| `skin on LV <lv>: <s> is not a library surface` | `DescriptionError` | `lumen.build.lookup` |
| `skin surface <s> on LV <lv>, which does not occur in the tree` | `DescriptionError` | `lumen.build.lookup` |
| `surface <name>: expected a SurfaceEntry(SurfaceType, params)` | `DescriptionError` | `lumen.build.lookup` |
| `surface <name>: surface type <name> detects; detection needs a sensor binding (library.sensors), not a surface` | `DescriptionError` | `lumen.build.lookup` |
| `two different surface types share a name: <name>` | `OpticsError` | `lumen.build.lookup` |
| `volume groups <class_names> -> <class_names> resolve to two surface types: <a> -> <b>: <name> <f> (<f>), <pre> -> <post>: <name> <origin> (<reason>). The signature missed a distinction` | `OpticsError` | `lumen.build.lookup` |
| `detector volume <spec> is placed <chains> times (<shown><more>): lumen simulates one placement of the detector volume; to simulate all of them give their common mother, detector volume <common_mother>; <one>` | `ValueError` | `lumen.build.lowering` |
| `no volume of a material named <names> (ignoring case): name one of the detector's materials (<found><SHOWN>)` | `ValueError` | `lumen.build.lowering` |
| `no volume, physvol or physvol path <spec> placed under the world` | `ValueError` | `lumen.build.lowering` |
| `placement <name>: scale <s> reflects and is not a unit scale: Geant4 aborts (G4ReflectionFactory::CheckScale, GeomVol0002)` | `ScaleRefused` | `lumen.build.lowering` |
| `solid <name> is a <type>, which lumen does not trace (it traces <join>): replace it in the recipe` | `NotImplementedError` | `lumen.build.lowering` |
| `volume <h> of material <material> has no bounded solid: auto: chooses by size; name the detector volume instead` | `ValueError` | `lumen.build.lowering` |
| `volume <lv>: <e>` | `NotImplementedError` | `lumen.build.lowering` |
| `calibration <name>: <_unknown> (<unknown> unknown keys; a response's keys need it at load: lumen.load(..., responses=))` | `UnknownParamKey` | `lumen.build.spec` |
| `navigator volume <i> is <names>, the optics' <paths>: the class and sensor tables would attach to other volumes` | `GeometryError` | `lumen.build.tracked` |
| `navigator volumes <n> / meshes <n_mesh> vs PVs <vols> / <meshes>` | `GeometryError` | `lumen.build.tracked` |
| `the scene's PVs differ from the optics': <have> not in the optics (e.g. <sorted>), <want> missing (e.g. <sorted>)` | `GeometryError` | `lumen.build.tracked` |
| `<pv>: a mesh member with a PATH sensor label (meshes are not numbered)` | `TreeError` | `lumen.build.tree` |
| `<pv>: medium id <mid> has no material name in media` | `TreeError` | `lumen.build.tree` |
| `media: materials <names> and <name> share medium id <mid>` | `TreeError` | `lumen.build.tree` |
| `tree: LV <lv> placed with materials <mat> and <material>` | `TreeError` | `lumen.build.tree` |
| `tree: PV <name> has unknown mother <mother>` | `TreeError` | `lumen.build.tree` |
| `tree: PVs not reachable from the world (mother cycle): <lost>` | `TreeError` | `lumen.build.tree` |
| `tree: duplicate PV name <name>` | `TreeError` | `lumen.build.tree` |
| `tree: exactly one world PV (mother None) required, got <roots>` | `TreeError` | `lumen.build.tree` |
| `tree: expected PV, got <p>` | `TreeError` | `lumen.build.tree` |
| `(a message built at run time)` | `InvalidValue` | `lumen.build.values` |
| `charge entry <name>: <orig_id> does not run <spec>'s sources` | `ValueError` | `lumen.compare.charge` |
| `charge entry <spec>: response 'qe' is read at the photon's energy` | `ValueError` | `lumen.compare.charge` |
| `charge entry <spec>: response 'qe' needs a sensor model with a detection on group <name>` | `ValueError` | `lumen.compare.charge` |
| `charge entry <spec>: response <kind> is not one of <RESPONSES>` | `ValueError` | `lumen.compare.charge` |
| `sensor match: <sum> reference sensors farther than <tol_mm> mm from any lumen sensor (largest <max> mm)` | `ValueError` | `lumen.compare.charge` |
| `compare_events: no events` | `ValueError` | `lumen.compare.event` |
| `event: no light of <lights> lumen can emit (outside: <outside>)` | `ValueError` | `lumen.compare.event` |
| `<csv>: <rows> answers for <points> points (probed on other points?)` | `RuntimeError` | `lumen.compare.geometry` |
| `<gdml> is not the GDML <path> was converted from (<want>)` | `ValueError` | `lumen.compare.geometry` |
| `g4ref failed (<returncode>): <stdout> <stderr>` | `RuntimeError` | `lumen.compare.geometry` |
| `the geometry check probes two GDMLs (the export and the original): give the g4ref binary` | `ValueError` | `lumen.compare.geometry` |
| `gradients <name>: <shards> shard files, not one per shard of one sharding` | `ValueError` | `lumen.compare.gradients` |
| `gradients <name>: a family appears in two shards` | `ValueError` | `lumen.compare.gradients` |
| `gradients <name>: the shards hold <fam> of <nf> families` | `ValueError` | `lumen.compare.gradients` |
| `the reference's keys are not this detector's: <missing>` | `KeyError` | `lumen.compare.mc` |
| `<names> reaches out of an ancestor through an assembly: not written out` | `OwnershipExportError` | `lumen.compare.ownership` |
| `a volume and its owner have different mothers in the detector file (<pv>, <pw>)` | `OwnershipExportError` | `lumen.compare.ownership` |
| `an owner owns part of a volume across an assembly (<pv>, <pw>): not written out` | `OwnershipExportError` | `lumen.compare.ownership` |
| `placement <name> is reflected or scaled: its ownership is not written out` | `OwnershipExportError` | `lumen.compare.ownership` |
| `the ancestor's volume is not the path's mother <lv>` | `OwnershipExportError` | `lumen.compare.ownership` |
| `the placement check finds overlaps the navigator does not link (<lost>): not written out` | `OwnershipExportError` | `lumen.compare.ownership` |
| `the scene walk gives <vols> volumes and <meshes> meshes, the navigator <n> and <n_mesh>` | `OwnershipExportError` | `lumen.compare.ownership` |
| `<n_sensors> sensors but <unique> sensor volumes in the navigator` | `ValueError` | `lumen.compare.photons` |
| `photons.loop: a layout of every sensor in order (index None)` | `ValueError` | `lumen.compare.photons` |
| `<out> is inside a reference store: write check outputs elsewhere (--out)` | `PermissionError` | `lumen.compare.records` |
| `<d>: missing from the store` | `ReferenceStoreError` | `lumen.compare.references` |
| `<d>: sha256 differs from the manifest (<digest>)` | `ReferenceStoreError` | `lumen.compare.references` |
| `<found> manifests name a <check> check <name>: rename one` | `ReferenceStoreError` | `lumen.compare.references` |
| `<url>: <name> missing from the release asset` | `ReferenceStoreError` | `lumen.compare.references` |
| `<url>: cannot fetch the release asset (<e>)` | `ReferenceStoreError` | `lumen.compare.references` |
| `<url>: sha256 differs from the manifest (<entry>)` | `ReferenceStoreError` | `lumen.compare.references` |
| `no <check> check named <name>; the manifests list <known>` | `ReferenceStoreError` | `lumen.compare.references` |
| `no reference product <pid> in the manifests` | `ReferenceStoreError` | `lumen.compare.references` |
| `<key>: <old_e> knots expected, got <vs> values at <es> energies; a new knot count is a new detector file` | `ValueError` | `lumen.format.apply` |
| `<key>: a <old_e> property needs a <old_e>` | `ValueError` | `lumen.format.apply` |
| `<key>: a Geant4 default property is set to a number` | `ValueError` | `lumen.format.apply` |
| `<key>: a constant property needs a number` | `ValueError` | `lumen.format.apply` |
| `<key>: the energies must be strictly ascending` | `ValueError` | `lumen.format.apply` |
| `<key>: the opticalsurface value is a number` | `ValueError` | `lumen.format.apply` |
| `parameter key <key>: <name> has no property <prop> (adding one is a new detector file, except a Geant4 default the model reads)` | `ValueError` | `lumen.format.apply` |
| `parameter key <key>: no <section> <name>` | `ValueError` | `lumen.format.apply` |
| `parameter key <key>: not materials/<M>/<KEY>, surfaces/<S>/<KEY> or sensors/...` | `ValueError` | `lumen.format.apply` |
| `parameter key <key>: surface <name> (model <model>) reads its value as <want>` | `ValueError` | `lumen.format.apply` |
| `(a message built at run time)` | `FileExistsError` | `lumen.format.atomic` |
| `exists=<exists>: one of 'keep', 'raise', 'replace'` | `ValueError` | `lumen.format.atomic` |
| `calibration <name> was made for detector file <detector_digest>, but <path> is <digest>: its values may name keys or knots this file does not have. Remake the calibration on this file` | `CalibrationMismatch` | `lumen.format.calibrations` |
| `calibration <path> exists: choose a new name` | `FileExistsError` | `lumen.format.calibrations` |
| `no calibration <name> in <str>` | `FileNotFoundError` | `lumen.format.calibrations` |
| `<what>: <e> energies but values of shape <shape>` | `error` | `lumen.format.check` |
| `<what>: a constant is one finite number` | `error` | `lumen.format.check` |
| `<what>: a value is not finite` | `error` | `lumen.format.check` |
| `<what>: energies must be strictly ascending with >= 2 knots (got <e>)` | `error` | `lumen.format.check` |
| `border <name> refers to <surface>, which is not a surface` | `error` | `lumen.format.check` |
| `border surface <name>: physvol <r> names <get> physical volumes; Geant4 binds a border to the first one only, so the border is ambiguous` | `error` | `lumen.format.check` |
| `border surface <name>: the pair <pv1> -> <pv2> has a border already (Geant4 keeps the first)` | `error` | `lumen.format.check` |
| `element <name>: isotope <ref> is not an isotope` | `error` | `lumen.format.check` |
| `logical volume <lv> has two skin surfaces (<lvs>, <name>): Geant4 keeps the first registered; refusing the ambiguity` | `error` | `lumen.format.check` |
| `material <name> names no composition: give a density and its components (mass fractions or atom counts), or Z and a molar mass` | `error` | `lumen.format.check` |
| `material <name>: component <ref> has a negative share (<fraction>): give a mass fraction or an atom count >= 0` | `error` | `lumen.format.check` |
| `material <name>: component <ref> is neither an element nor a material` | `error` | `lumen.format.check` |
| `material <name>: density <density_g_cm3> g/cm3: give a positive, finite density` | `error` | `lumen.format.check` |
| `placement <i> is a daughter of <owner> and of <name>` | `error` | `lumen.format.check` |
| `placement <name> (index <i>) is not listed in its mother <mother>'s daughters` | `error` | `lumen.format.check` |
| `placement <name> places the world <world>: the world is placed by no placement` | `error` | `lumen.format.check` |
| `placement <name>: mother <mother> is not a volume` | `error` | `lumen.format.check` |
| `placement <name>: position and rotation are three finite numbers each` | `error` | `lumen.format.check` |
| `placement <name>: volume <volume> is not a volume` | `error` | `lumen.format.check` |
| `skin <name> refers to <surface>, which is not a surface` | `error` | `lumen.format.check` |
| `skin <name>: volume <volume> is not a volume` | `error` | `lumen.format.check` |
| `solid <name> (<type>): <k> <get> (omitted: 0): Geant4's <type> constructor aborts at <k> <= 0 (Invalid dphi / dtheta); give it` | `error` | `lumen.format.check` |
| `solid <name> (<type>): unknown attributes <extra> (known: <known>)` | `error` | `lumen.format.check` |
| `solid <name>: a number is not finite` | `error` | `lumen.format.check` |
| `solid <name>: mesh <mesh> has no triangles` | `error` | `lumen.format.check` |
| `solid <name>: reference <r> is not a solid` | `error` | `lumen.format.check` |
| `surface <name>: its value is not finite` | `error` | `lumen.format.check` |
| `the detector volume <detector_volume> is no volume, physvol or physvol path placed under the world` | `error` | `lumen.format.check` |
| `the world <world> is not a volume` | `error` | `lumen.format.check` |
| `volume <child> holds itself: <join>` | `error` | `lumen.format.check` |
| `volume <name>: daughter <i> names no placement (there are <placements>)` | `error` | `lumen.format.check` |
| `volume <name>: material <material> is not a material` | `error` | `lumen.format.check` |
| `volume <name>: solid <solid> is not a solid` | `error` | `lumen.format.check` |
| `<join>: the record kept under <path> is named <getattr>: key it by its own <key>` | `FormatError` | `lumen.format.io` |
| `<path> exists: a detector file is never overwritten (pass overwrite=True)` | `FormatError` | `lumen.format.io` |
| `<root>: <err>: convert the detector again` | `FormatError` | `lumen.format.io` |
| `<root>: <file> has unknown fields <unknown>: convert the detector again` | `FormatError` | `lumen.format.io` |
| `<root>: <name>[<i>] has <row> fields, not <names> (<names>)` | `FormatError` | `lumen.format.io` |
| `<root>: <rel> does not match the manifest (changed since it was written): convert the detector again` | `FormatError` | `lumen.format.io` |
| `<root>: <where> (<__name__>): unknown fields <unknown>, missing fields <missing>: convert the detector again` | `FormatError` | `lumen.format.io` |
| `<root>: <where> is not a mapping of <__name__>'s fields` | `FormatError` | `lumen.format.io` |
| `<where>: written by another lumen (format <found>); this lumen reads <want>: <remedy>` | `FormatError` | `lumen.format.io` |
| `cannot store <__name__> in JSON` | `FormatError` | `lumen.format.io` |
| `non-finite number <x>: the detector file stores finite numbers only` | `FormatError` | `lumen.format.io` |
| `two arrays at <rel>: rename one of <path>` | `FormatError` | `lumen.format.io` |
| `<what>: setting <k> must be <known>, got <v>` | `ValueError` | `lumen.format.schema` |
| `<what>: unknown setting <k> (known: <known>)` | `ValueError` | `lumen.format.schema` |
| `the detector volume <detector_volume> is no volume, physvol or physvol path placed under the world` | `ValueError` | `lumen.format.schema` |
| `sensor group <name>: a group with a value set has every absorption entering counted and its value set's efficiency applied after tracking: drop the setting entry_efficiency` | `ValueError` | `lumen.format.sensorsets` |
| `sensor group <name>: a name is letters, digits and '_', not starting with a digit (it is a segment of the keys sensors/<group>/<field>)` | `ValueError` | `lumen.format.sensorsets` |
| `sensor group <name>: give exactly one of lv and lv_pattern` | `ValueError` | `lumen.format.sensorsets` |
| `sensor group <name>: rule <rule> is not one of <RULES>` | `ValueError` | `lumen.format.sensorsets` |
| `sensor group <name>: selects by the id-map columns <missing>, which the sensor set does not have (it has <columns>)` | `ValueError` | `lumen.format.sensorsets` |
| `sensor group <name>: the setting count is 'volume' or 'entry', for an any_entry group (a border_into group counts at its surfaces), got <count> on <rule>` | `ValueError` | `lumen.format.sensorsets` |
| `sensor group <name>: unknown settings <unknown> (the settings are <SETTINGS>)` | `ValueError` | `lumen.format.sensorsets` |
| `the sensor set's id map names <dup> twice (each sensor has its own path): convert the detector again` | `ValueError` | `lumen.format.sensorsets` |
| `the sensor set: <err>: convert the detector again` | `FormatError` | `lumen.format.sensorsets` |
| `two sensor groups are named <name>` | `ValueError` | `lumen.format.sensorsets` |
| `(a message built at run time)` | `YAMLError` | `lumen.format.yaml` |
| `<k> <name> is defined in <where> and in <name>` | `YAMLError` | `lumen.format.yaml` |
| `<k> is <out> in <where> and <v> in <name>` | `YAMLError` | `lumen.format.yaml` |
| `<name>: <k> is a list` | `YAMLError` | `lumen.format.yaml` |
| `<name>: <k> is a mapping by name` | `YAMLError` | `lumen.format.yaml` |
| `<name>: a detector file is a mapping of sections (<join>)` | `YAMLError` | `lumen.format.yaml` |
| `<name>: no schema line: a detector file starts with 'schema: <SCHEMA>'` | `YAMLError` | `lumen.format.yaml` |
| `<name>: not valid YAML: <err>` | `YAMLError` | `lumen.format.yaml` |
| `<name>: written by another lumen (schema <doc>); this lumen reads <SCHEMA>` | `YAMLError` | `lumen.format.yaml` |
| `<p>: no such file or directory` | `YAMLError` | `lumen.format.yaml` |
| `<w>.params: a mapping of attributes (<join>)` | `YAMLError` | `lumen.format.yaml` |
| `<w>.wavelength_nm: wavelengths are positive` | `YAMLError` | `lumen.format.yaml` |
| `<w>: <size> energies but <size> values` | `YAMLError` | `lumen.format.yaml` |
| `<w>: a table gives its energies (energy, eV) or wavelengths (wavelength_nm)` | `YAMLError` | `lumen.format.yaml` |
| `<w>: a volume names its solid and its material (an assembly, assembly: true, neither)` | `YAMLError` | `lumen.format.yaml` |
| `<w>: give energy or wavelength_nm, not both` | `YAMLError` | `lumen.format.yaml` |
| `<where>: (z, x offset, y offset, scale), got <x>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: <err>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: <n> numbers, got <v>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: <n> rows, got <v>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a component gives a mass fraction or a number of atoms (one of the two)` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a list of numbers, got <v>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a mapping by key, got <v>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a mapping of fields (<join>), got <d>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a name (text), got <v>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a plain number (no unit), got <v>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a rotation matrix is orthonormal with determinant +1 (a reflection is a placement's scale), got {m.tolist()}` | `YAMLError` | `lumen.format.yaml` |
| `<where>: a whole number, got <v>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: array file <v> not found beside the YAML` | `YAMLError` | `lumen.format.yaml` |
| `<where>: missing <join>` | `YAMLError` | `lumen.format.yaml` |
| `<where>: unknown field <k><hint>` | `YAMLError` | `lumen.format.yaml` |
| `meshes.<name>: triangles (k, 3, 3) in m, got shape <shape>` | `YAMLError` | `lumen.format.yaml` |
| `no YAML files in <str>` | `YAMLError` | `lumen.format.yaml` |
| `sensors: <err>` | `YAMLError` | `lumen.format.yaml` |
| `world is missing and <roots> volumes are placed by no placement (<roots>): name the world` | `YAMLError` | `lumen.format.yaml` |
| `a matrix and a position share a name` | `ExportError` | `lumen.gdml.exporter` |
| `border <name>: a physvol of an assembly (<pv1>, <pv2>); Geant4 renames the imprinted placements, so the border cannot be bound` | `ExportError` | `lumen.gdml.exporter` |
| `calibration=: give the detector file's path, so the calibration is checked against it (a DetectorFile in memory has no sensor set to check it with)` | `ExportError` | `lumen.gdml.exporter` |
| `denormal value <text>: Geant4's reader refuses it (strtod sets ERANGE)` | `ExportError` | `lumen.gdml.exporter` |
| `matrix <name>: Geant4 would define the constant <c> twice` | `ExportError` | `lumen.gdml.exporter` |
| `non-finite value <text>: Geant4 reads no inf or nan` | `ExportError` | `lumen.gdml.exporter` |
| `on_lossy=<on_lossy>: one of ('error', 'omit')` | `ValueError` | `lumen.gdml.exporter` |
| `solid <name> (<type>): attributes <unknown> are not <type> attributes` | `ExportError` | `lumen.gdml.exporter` |
| `solid <name>: type <type> has no GDML spelling in the exporter (supported: <ATTRIBUTES>)` | `ExportError` | `lumen.gdml.exporter` |
| `surface <name>: <what> <v> is not one Geant4 knows (one of <names>)` | `ExportError` | `lumen.gdml.exporter` |
| `surface <name>: a coated surface; GDML has no coated type (it reads as x_ray)` | `ExportError` | `lumen.gdml.exporter` |
| `surface <name>: optics model <model> has no Geant4 form; pass on_lossy='omit' to write the GDML without it (marked lossy, not a Geant4 reference)` | `ExportError` | `lumen.gdml.exporter` |
| `the placements are not in an order a GDML file can hold (each volume's daughters one contiguous run, in the order of their volumes): re-import cannot reproduce them` | `ExportError` | `lumen.gdml.exporter` |
| `value <text> is <lit> in Geant4's units: not a normal double` | `ExportError` | `lumen.gdml.exporter` |
| `value <text>: the literal <lit> does not read back exactly` | `ExportError` | `lumen.gdml.exporter` |
| `volume <name>: material <material> is not defined (and is no G4_ NIST name)` | `ExportError` | `lumen.gdml.exporter` |
| `<what>.<key>: matrix <ref> has <shape> columns (a property needs 1 or 2)` | `ImportRefused` | `lumen.gdml.importer` |
| `<what>: property <key> given twice` | `ImportRefused` | `lumen.gdml.importer` |
| `bordersurface <name>: needs exactly two physvolref` | `ImportRefused` | `lumen.gdml.importer` |
| `opticalsurface <name> has no <join> attribute. Geant4 would read a default and lumen cannot guess: add it in the GDML and convert again.` | `ImportRefused` | `lumen.gdml.importer` |
| `opticalsurface <surface>: <attr> <t> is no value of Geant4's enum (0 to <len>: <join>)` | `ImportRefused` | `lumen.gdml.importer` |
| `tessellated <name>: no facets` | `ImportRefused` | `lumen.gdml.importer` |
| `(a message built at run time)` | `GDMLError` | `lumen.gdml.reader` |
| `<type> <name>: solid <r> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `<what>: <e>` | `GDMLError` | `lumen.gdml.reader` |
| `<what>: <s> is expected to have an integer value (G4GDMLEvaluator::EvaluateInteger)` | `GDMLError` | `lumen.gdml.reader` |
| `<what>: facet vertex <r> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `<what>: scaleref <get> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `<what>: vertex <r> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `<where>: <label> <ref> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `bad token in <s> at <pos>` | `GDMLError` | `lumen.gdml.reader` |
| `coldim <cd>` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: <<tag>> <get> is not supported` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: <what>: attribute <a> is not one Geant4 reads for <tg> (names are case-sensitive; <tg> has <dims>): Geant4 ignores it` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: <where>: <refused>` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: XML error (external entity includes are not supported): <e>` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: conflicting re-definition of <tag> <nm>` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: matrix <nm>: <e>` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: structure element <<tg>> is not supported` | `GDMLError` | `lumen.gdml.reader` |
| `gdml <path>: world volume <world> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `index out of range: <name><idx>` | `GDMLError` | `lumen.gdml.reader` |
| `matrix <name> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `missing )` | `GDMLError` | `lumen.gdml.reader` |
| `missing value` | `GDMLError` | `lumen.gdml.reader` |
| `trailing tokens in <s>` | `GDMLError` | `lumen.gdml.reader` |
| `unexpected token <t>` | `GDMLError` | `lumen.gdml.reader` |
| `volume <name>: a physvol <file> (<get>) is not supported` | `GDMLError` | `lumen.gdml.reader` |
| `volume <name>: physvol <get> places <child>, not defined before it` | `GDMLError` | `lumen.gdml.reader` |
| `volume <name>: solid <solid> is not defined` | `GDMLError` | `lumen.gdml.reader` |
| `<where>: <group> overflows a double` | `ValueError` | `lumen.gdml.repair` |
| `matrix <nm>: coldim <get> is no number literal: write it as one to repair this matrix` | `ValueError` | `lumen.gdml.repair` |
| `the elements found in the text are not the parsed ones (an unusual layout): not repaired` | `ValueError` | `lumen.gdml.repair` |
| `unknown repairs <bad>; known: <REPAIRS>` | `ValueError` | `lumen.gdml.repair` |
| `not a finite decimal: <d>` | `ValueError` | `lumen.gdml.units` |
| `bvh.build_many: empty or unordered box ranges` | `ValueError` | `lumen.geometry.bvh` |
| `code_digest: <missing> names no lumen module; give a module name such as 'lumen.geometry.clearance', not a path` | `ValueError` | `lumen.geometry.cache` |
| `the navigator cache cannot store a <__name__>` | `TypeError` | `lumen.geometry.cache` |
| `the navigator cache names a class outside lumen: <x>` | `ValueError` | `lumen.geometry.cache` |
| `csg tables: a boolean deeper than <MAX_DEPTH> operands` | `ValueError` | `lumen.geometry.csg_build` |
| `csg tables: an unbounded solid (no finite box)` | `ValueError` | `lumen.geometry.csg_build` |
| `navigator: a BVH of depth <dmax> exceeds the traversal stack (<STACK>): a volume has too many daughters (<max>): group them in sub-volumes` | `ValueError` | `lumen.geometry.navigator_build` |
| `navigator: a scene without volumes` | `ValueError` | `lumen.geometry.navigator_build` |
| `navigator: tessellated volume <dname> has daughters: a mesh is opaque (first hit only), so place them in its mother` | `NotImplementedError` | `lumen.geometry.navigator_build` |
| `navigator: tessellated volume <dname> is a sensor: a mesh reports its first hit only; select a volume of a CSG solid` | `NotImplementedError` | `lumen.geometry.navigator_build` |
| `navigator: the surface codes overflow int32 (<sm>): the detector volume holds too many solid leaves and mesh triangles; simplify its largest solids or meshes` | `ValueError` | `lumen.geometry.navigator_build` |
| `placement of unknown definition <dname>` | `KeyError` | `lumen.geometry.scene` |
| `Op: unknown operation <op> (one of <OPS>)` | `ValueError` | `lumen.geometry.solids` |
| `a solid with no part: every section is thinner than the z tolerance (Z_TOL)` | `ValueError` | `lumen.geometry.solids` |
| `arb8 with a twisted (non-planar) face` | `NotImplementedError` | `lumen.geometry.solids` |
| `phi range <dphi> <= 0: Geant4's constructor aborts (Invalid dphi); give the range` | `ValueError` | `lumen.geometry.solids` |
| `polytope_from_faces: the vertices are not convex in position (a concave arb8/trap base or a twisted face)` | `ValueError` | `lumen.geometry.solids` |
| `xtru polygon: ear clipping failed (self-touching or duplicate vertices)` | `ValueError` | `lumen.geometry.solids` |
| `(a message built at run time)` | `InputError` | `lumen.inputs` |
| `<path>: <name> records no sha256 for this file (a conversion reads only checked files)` | `InputError` | `lumen.inputs` |
| `<name>.<name>: a channel length law must output a length` | `LawError` | `lumen.media.bulk` |
| `channel <name>: a re-emission draws its wavelength and one uniform per time-profile component, at most <REEMIT_UNIFORMS> in all; <comps> has <comps> components` | `ValueError` | `lumen.media.bulk` |
| `TimeProfile: components from <COMPONENTS>, got <components>` | `ValueError` | `lumen.media.emission` |
| `<name>: <out> out of range at <lam> nm: <v> (allowed [<a>, <b>])` | `LawError` | `lumen.media.laws` |
| `<name>: missing parameters <missing>` | `LawError` | `lumen.media.laws` |
| `<name>: non-finite values at its points` | `LawError` | `lumen.media.laws` |
| `<name>: the <form> law declares no points to be checked at (Law.points)` | `LawError` | `lumen.media.laws` |
| `group_index needs an index law` | `ValueError` | `lumen.media.laws` |
| `MediumSpec: <name> is not a declared parameter of the model (<values>)` | `KeyError` | `lumen.media.spec` |
| `MediumSpec: a channel's lobe is a lumen Lobe, not <__name__>` | `TypeError` | `lumen.media.spec` |
| `MediumSpec: medium <name> has a <name> already` | `KeyError` | `lumen.media.spec` |
| `MediumSpec: medium <name> has no channel <name> (<chans>)` | `KeyError` | `lumen.media.spec` |
| `MediumSpec: medium <name> is opaque (no RINDEX)` | `ValueError` | `lumen.media.spec` |
| `phase function <name>: give both pol_sample and pol_logpdf, or neither` | `ValueError` | `lumen.media.spec` |
| `phase function <name>: unpolarised=True with a polarised pair; give one` | `ValueError` | `lumen.media.spec` |
| `phase function <name>: without pol_sample and pol_logpdf its direction ignores the photon's polarisation: give the pair, or declare unpolarised=True` | `ValueError` | `lumen.media.spec` |
| `spectrum <name>: needs a non-negative intensity with positive integral` | `ValueError` | `lumen.media.spectra` |
| `a model is a MediumModel (lumen.medium_model), not <__name__>` | `TypeError` | `lumen.models` |
| `a part's channel is one name, not <channel>` | `ValueError` | `lumen.models` |
| `lobe <name>: its parameters <params> are not declared (lumen.media.phase_function)` | `ModelError` | `lumen.models` |
| `model <name>: <name> gives <k> as a table; a declared parameter is one number (a model's parameters cannot be wavelength tables)` | `ModelError` | `lumen.models` |
| `optics model <name>: <k> must be a scalar lumen.Param with a default` | `TypeError` | `lumen.models` |
| `optics model <name>: a 'distribution:name', not one of lumen's own` | `ValueError` | `lumen.models` |
| `optics model <name>: parameter name <k> (no '/')` | `ValueError` | `lumen.models` |
| `optics model <name>: two models of one name` | `ValueError` | `lumen.models` |
| `(a message built at run time)` | `UnknownParamKey` | `lumen.params` |
| `(a message built at run time)` | `ValueError` | `lumen.params` |
| `<key>: shape <shape> expected, got <shape>` | `ValueError` | `lumen.params` |
| `<prefix><name>: declare a lumen.Param, not <__name__>` | `TypeError` | `lumen.params` |
| `Param: default <default> has shape <shape>, declared <shape>` | `ValueError` | `lumen.params` |
| `Param: default <default> is outside its range [<lo>, <hi>]` | `ValueError` | `lumen.params` |
| `Param: range <range> is not (lo, hi) with lo <= hi` | `ValueError` | `lumen.params` |
| `lumen.response: parameter <k> must be a lumen.Param that does not move the light` | `TypeError` | `lumen.readout.custom` |
| `lumen.response: parameter name <k> (no '/')` | `ValueError` | `lumen.readout.custom` |
| `response <name> reads group <group>, the histograms are <group>'s` | `ValueError` | `lumen.readout.custom` |
| `response <name>: <prefix> has no default; set it in the params` | `KeyError` | `lumen.readout.custom` |
| `readout: expected lumen.readout.Histograms (or a light of one modelled group), got <light>` | `TypeError` | `lumen.readout.detection` |
| `readout: sensor group <group> has no sensor model: give the group a value set (SensorGroup.values)` | `ValueError` | `lumen.readout.detection` |
| `readout: the light holds the histograms of the sensor groups <groups>: give light.group(name)` | `TypeError` | `lumen.readout.detection` |
| `sensor group <group>: no time histogram (readout 'pe'); simulate with a timed readout` | `ValueError` | `lumen.readout.detection` |
| `the readout of one sensor needs <row_bytes> GB, over the <budget> GB free for it: shorten the readout gates, widen the time spreads' declared lower ranges (coarser nodes)` | `MemoryError` | `lumen.readout.detection` |
| `sample: the expected density of sensor(s) <tolist> of group <group> is negative beyond <NEGATIVE_SHARE> of its total (at most <max>): not a rounding of the time histogram's signed split` | `ValueError` | `lumen.readout.events` |
| `one sensor's readout is more cells than int32 indices reach (<kinds>): shorten the readout gates or give fewer events per run` | `MemoryError` | `lumen.readout.fill` |
| `readout: a weighted time histogram reads its QE x CE from the run's weights (one per group, lumen.run.run): got <weights> for <rs> groups` | `ValueError` | `lumen.readout.fill` |
| `readout: the groups' layouts differ in events or in the detector's sensors` | `ValueError` | `lumen.readout.fill` |
| `the arrivals of <n_all> sensors over <events> events are more cells than int32 indices reach: fewer events per run` | `MemoryError` | `lumen.readout.fill` |
| `the readout of one sensor is <per> exact cells, over the <cap> the <budget> GB left for it hold: shorten the readout gates, widen the time spreads' declared lower ranges (coarser nodes) or give a smaller chunk` | `MemoryError` | `lumen.readout.fill` |
| `<what>: sensor ids are integers in [0, <n_sensors>), got values in [<min>, <max>]` | `ValueError` | `lumen.readout.grid` |
| `Axis <name>: exactly one of knots or edges, at least two ascending values` | `ValueError` | `lumen.readout.grid` |
| `Grid: distinct axis names, got <name>` | `ValueError` | `lumen.readout.grid` |
| `Grid: nt >= 0 and dt > 0` | `ValueError` | `lumen.readout.grid` |
| `from_photons: <nan> histogram cells are not finite (a non-finite weight or field)` | `ValueError` | `lumen.readout.grid` |
| `from_photons: the photons lack <missing>, which the readout's features read` | `ValueError` | `lumen.readout.grid` |
| `from_photons: the photons' <bad> must be finite` | `ValueError` | `lumen.readout.grid` |
| `node_spacing: the narrowest time spread must be > 0 ns; declare its range in the value set` | `ValueError` | `lumen.readout.grid` |
| `photon_chunks: samples >= 0 and chunk > 0, got <samples>, <chunk>` | `ValueError` | `lumen.readout.grid` |
| `times up to <top> ns are float32 steps of <step> ns, coarser than the time nodes' <min> ns / <TIME_ULPS>: give the source's times and the gate relative to the event (subtract its start)` | `GateError` | `lumen.readout.grid` |
| `hits_nll: the hits are of <shape> sensors, the expected hits of <S> (Hits.from_list(..., n_sensors) of this group)` | `ValueError` | `lumen.readout.hits` |
| `sum_by: the keys <keys> span too many values for one int64 key` | `ValueError` | `lumen.readout.truth` |
| `sensor group <group>: value set <set_name> has no waveform readout (readout <readout>); use a value set with a pulse and readout 'waveform'` | `ValueError` | `lumen.readout.waveform` |
| `waveform_nll: data of shape <shape>, the expected waveform is <shape>` | `ValueError` | `lumen.readout.waveform` |
| `recipe <name> needs the inputs <missing>` | `ValueError` | `lumen.recipe` |
| `recipe input <name>: <p> is not a file (an input is one file, recorded by its sha256)` | `ValueError` | `lumen.recipe` |
| `<pre> -> <post>: type <name> (<reason>) would put the photon into the opaque medium <medium>` | `BoundaryError` | `lumen.rules.boundary` |
| `<key>: an extinction index is >= 0 (an absorbing medium), got <min>` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: <given> without the optics setting scint_finite_rise_time: Geant4 reads a rise time only under its finite-rise-time switch; set it as the Geant4 run does, or drop the key` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: <k> (<why>) is not supported` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: <k> is not supported` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: <k> must be a constant (G4OpMieHG reads a constant property)` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: <k> must be a constant >= 0` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: <k> must be a table` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: <key> <g> outside [-<MIE_G_MAX>, <MIE_G_MAX>] (a narrower lobe than float32 directions resolve)` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: GROUPVEL must hold speeds in (0, c] in m / ns (c = <C_VACUUM> m / ns), got <min> to <max>: in mm / ns it is 1000 times too fast` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: Geant4-10 scintillation keys <old> in a detector file: lumen translates them when a detector is converted (lumen.build.import_rules), so this file was written by an older lumen; convert it again from source` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: MIEHG without MIEHG_FORWARD_RATIO: G4OpMieHG aborts (mat202) at the first Mie event; add MIEHG_FORWARD_RATIO` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: MIEHG_FORWARD_RATIO <r> outside [0, 1]` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: MIEHG_FORWARD_RATIO <r> without <key>: G4OpMieHG aborts (mat202) at the first <lobe_name> Mie event; add <key>` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: WLSABSLENGTH without WLSCOMPONENT (G4OpWLS absorbs and emits nothing)` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: a complex index on a ground finish (the facet's Fresnel) is not supported` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: finish <finish> is not supported` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: model <model> is not supported` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: polish <value> outside [0, <GLISUR_R_MIN>] (or exactly 1)` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: scint_finite_rise_time without <missing>: G4Scintillation reads each emitted component's rise time and aborts without it` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: sigma_alpha <value> outside [<SIGMA_MIN>, <SIGMA_MAX>] (or exactly 0)` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: surface type <type_> is not supported` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: unified ground dielectric_dielectric (facet refraction) is not supported` | `OpticsRefused` | `lumen.rules.optics` |
| `<what>: wls_time_profile <profile> is not one of G4OpWLS's <WLS_TIME_PROFILES>` | `OpticsRefused` | `lumen.rules.optics` |
| `opticalsurface <name>: its reading detects nothing` | `OpticsRefused` | `lumen.rules.optics` |
| `a launch appended <need> deposit records, over its <slots> slots: give records=<need> (or more), or a chunk below <below>` | `RecordsOverflow` | `lumen.run` |
| `a launch appends <count> deposit records (int32), over the <room> slots that fit: give fewer rows` | `MemoryError` | `lumen.run` |
| `backend must be one of <BACKENDS>, got <backend>` | `ValueError` | `lumen.run` |
| `max_vertices must be >= 1, got <max_vertices>` | `ValueError` | `lumen.run` |
| `not one deposit record fits the memory left beside the readout: use a smaller chunk (or fewer jax.vmap members)` | `RecordsUnknown` | `lumen.run` |
| `records=<records> need <slots> slots, over the <room> that fit the memory left for them: use a smaller chunk` | `MemoryError` | `lumen.run` |
| `run: give the rows' analog weights (analog=) exactly when the model carries the analog photon` | `ValueError` | `lumen.run` |
| `the photon loop closes over <floats> float array(s) <floats>: pass every value through the Operands (lumen.transport.step.operands)` | `ClosedOverValues` | `lumen.run` |
| `the photon loop expects the GPU (<why>) but JAX runs on <default_backend>: the CUDA plugin did not load (see JAX's log above); run on a working GPU node, or pass backend='cpu'` | `DeviceError` | `lumen.run` |
| `feature <getattr>: declare the deposit fields it reads as fn.reads, a tuple of <FIELDS> (() for none)` | `TypeError` | `lumen.sensors.features` |
| `<ref>: no package <pkg> to read its value set from (install it, or give a path)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `a readout gate is (start, end) ns with end > start, got <gate>: a reversed or zero-width gate reads no light` | `GateError` | `lumen.sensors.value_sets` |
| `no sensor value set <ref>; <pkg> ships: <have>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: <key>: <e>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: <t> has values of shape <shape> for its axes <name> of sizes <want>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: <what>.<k> names no axis (<feature>_<unit> knots or <feature>_edges_<unit> edges; features <BUILTIN>)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: axis <name> is declared with two sets of knots or edges; an axis appears with one (tables sharing it multiply on its knots)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: delay form <get> is not one of ['table']` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: delay.edges_ns: at least two non-decreasing edges >= 0 (a zero-width bin is a point mass)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: delay.pdf has shape <shape>, its axes and bins need <fields>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: for model <get>; this model is 'lumen:sensor'` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: provenance needs a source and a status in <STATUSES>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: qe needs exactly one of wavelength_nm (knots) or band_edges_nm` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: readout <d> needs a tts and a charge record` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: readout <get> is not one of <READOUTS>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: the wavelength's bands are band_edges_nm` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <name>: unknown records <unknown> (a value set's records: <KEYS>)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: <k> = <tolist> is outside its declared range [<lo>, <hi>]; the time grid resolves only the declared range (a time spread below its declared minimum is not resolved): declare a range holding it` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: <what> (<form>) needs the fields <fields>, got <vals>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: <what> <name>: a name 'package:Name' names a form or feature of your own, which this lumen does not read (reserved in <SCHEMA>); use one of lumen's own` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: <what> form <form> is not one of <forms>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: <what>.<k> is in <unit>, the field is in <f>` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: a range for <k>, which is no value of the set (its values: <values>)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: declare the ranges of <missing> (they set the static time grid; a value outside its range is refused)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `value set <set_name>: the range of <declared> is derived, not declared ([r, 1 / r] x the set's own width, r = <TIME_SPREAD_REACH>: lumen.rules.choices)` | `SensorValuesError` | `lumen.sensors.value_sets` |
| `<__name__> emits its <rows> rows, one photon each: simulate it with samples=<rows> (or None), not <samples>` | `ValueError` | `lumen.sources._base` |
| `<__name__> reads <k>: pass the detector's parameters` | `KeyError` | `lumen.sources._base` |
| `<__name__>: <stale> are given as <PREFIX> in the parameters with other values than the source's (a traced field, or the keys of another source's parameters()): give each value once (set the key, or pass the source its value)` | `ValueError` | `lumen.sources._base` |
| `<name>.<f>: one row axis at most, got shape <shape>` | `ValueError` | `lumen.sources._base` |
| `<name>: a label per row needs a source with rows (give one label for all its photons)` | `ValueError` | `lumen.sources._base` |
| `<name>: its per-row fields have different numbers of rows <n>: give each the same rows (or one value for all)` | `ValueError` | `lumen.sources._base` |
| `<name>: one photon per row, and no field has rows (give (N, ...) arrays)` | `ValueError` | `lumen.sources._base` |
| `a source's label is an integer or a 1-d integer array, got <dtype> <shape>` | `TypeError` | `lumen.sources._base` |
| `concat: no photons` | `ValueError` | `lumen.sources.function` |
| `lumen.source: name <n> (no '/')` | `ValueError` | `lumen.sources.function` |
| `lumen.source: parameter <k> must be a lumen.Param` | `TypeError` | `lumen.sources.function` |
| `source <name>: <prefix> has no default; set it in the params` | `KeyError` | `lumen.sources.function` |
| `source <name>: Photons.<k> has shape <shape> <dtype>, expected <shape> <k> (n = <n>)` | `SourceError` | `lumen.sources.function` |
| `source <name>: Photons.<k> is not finite at the defaults` | `SourceError` | `lumen.sources.function` |
| `source <name>: directions are not unit vectors (\|d\| up to <max>)` | `SourceError` | `lumen.sources.function` |
| `source <name>: host callbacks <bad> (a source is plain JAX)` | `SourceError` | `lumen.sources.function` |
| `source <name>: its function closes over a traced value it is not given: declare it as a parameter (lumen.Param) and give it in params` | `GradientNotAvailable` | `lumen.sources.function` |
| `source <name>: negative weights` | `SourceError` | `lumen.sources.function` |
| `source <name>: return lumen.Photons, not <__name__>` | `SourceError` | `lumen.sources.function` |
| `Point: give a wavelength (lam) or a spectrum, not both` | `ValueError` | `lumen.sources.point` |
| `<material>: no <missing> (Geant4 11 emits none or aborts<hint>)` | `KeyError` | `lumen.sources.scintillation` |
| `<material>: the components' shares <tolist> are not finite and >= 0 (Geant4 divides by the sum of the yields)` | `KeyError` | `lumen.sources.scintillation` |
| `simulate: speed is 'fast' or 'exact', got <speed>` | `ValueError` | `lumen.speed` |
| `into_opaque: <name>'s facet path decides its own side` | `ValueError` | `lumen.surfaces.types` |
| `per_normal: a detecting type with eff, got <name>` | `ValueError` | `lumen.surfaces.types` |
| `(a message built at run time)` | `AssertionError` | `lumen.testing` |
| `check_gradient: more than <power_floor> seeds (fewer cannot reject at the checks' level: stats.mean_test)` | `ValueError` | `lumen.tools.check` |
| `check_gradient: stencil one of <STENCILS>, not <stencil>` | `ValueError` | `lumen.tools.check` |
| `<what> keys <unknown> are not keys of the space` | `ValueError` | `lumen.tools.fit` |
| `<what>: <sum> cells have counts where the model expects none (mu = 0, n > 0): the data are impossible at the start values` | `ValueError` | `lumen.tools.fit` |
| `fisher: <sum> cells with mu = 0 have a nonzero derivative: their information J^2 / mu is infinite` | `ValueError` | `lumen.tools.fit` |
| `layout: a mask of <sel> over <n> sensors` | `ValueError` | `lumen.tools.layout` |
| `layout: kind <kind> is not 'auto' or one of <KINDS>` | `ValueError` | `lumen.tools.layout` |
| `layout: positions must be (S, 3), got <shape>` | `ValueError` | `lumen.tools.layout` |
| `layout: the selection holds no sensor` | `ValueError` | `lumen.tools.layout` |
| `layout: the sensors do not lie in planes` | `ValueError` | `lumen.tools.layout` |
| `layout: the sensors do not lie on a cylinder` | `ValueError` | `lumen.tools.layout` |
| `display: <shape> values for <xy> sensors` | `ValueError` | `lumen.tools.plot` |
| `profile: <key> is not a single-number key of the space` | `ValueError` | `lumen.tools.scan` |
| `scan: <k> is not a single-number key of the space` | `ValueError` | `lumen.tools.scan` |
| `scan: one or two axes` | `ValueError` | `lumen.tools.scan` |
| `Space: <k> has a value <= 0; a key varied in log needs positive values` | `ValueError` | `lumen.tools.space` |
| `Space: <k> is not a positive factor times the space's own value` | `ValueError` | `lumen.tools.space` |
| `Space: scales for <bad>: a scale is for a key of this space without a log transform` | `ValueError` | `lumen.tools.space` |
| `StepConfig.moving_source: a source-position derivative is not available in this version (docs/explanation/known-limits.md)` | `NotImplementedError` | `lumen.transport.step` |
| `operands: the roulette is a number ε > 0 (a photon ends only at weight 0), got <roulette>` | `ValueError` | `lumen.transport.step` |
| `<value> is not a number or a quantity of <what> such as "<example>"` | `ValueError` | `lumen.units` |
| `<value>: <src> is not a unit of <what> (one of <join>)` | `ValueError` | `lumen.units` |
<!-- end of the generated messages -->
