# Notes for 0.2: factorised surfaces, the optics-models registry, the plug-in kit

Factorised surfaces, the plug-in kit (`lumen.testing`) and the example plug-ins moved to 0.2. These notes are what a
read of the code (main at cfbed84) showed before that work stopped. No framework code was written. The design is in
the redesign study's `surface_models.md` and `extensibility.md`.

## The unified and GLISUR facet lobes are not in the repository

The first port step left `lobes/unified.py` and `lobes/glisur.py` waiting for the d1 work. The legacy types
(`surfaces/_legacy.py`) only name their glossy lobe by a string (`SurfaceType.glossy`). Any check that the new
surface models draw the same lobe samples as the legacy types needs these two lobes ported first, bitwise against
the baseline (tag `lumen-b0`, whose lobes match today's `rebuild/next`).

They map onto the factors directly. The baseline's two `chain` functions differ only in `_facet`. So the port is:
- a `G4Unified(sigma_alpha)` microgeometry, with its tables;
- a `G4Glisur(polish)` microgeometry, which owns the map from polish to 1 - polish (this removes the `"sigma_alpha"`
  key trick);
- one facet-chain lobe over either microgeometry.

The unified F and zeta tables (about 270 KB) are closed-over constants today. Making them operands is part of the
operand step.

## Bitwise equality with the legacy types

- The legacy expressions multiply by Python constants in places. Examples: `p.get("eff", 0.0)` in
  `_dielectric_rt` gives `detect = a * 0.0`, and `chained_cuts` runs where a share is statically 1.
- A composer that emits no operation for a static 1 or 0 share therefore matches the legacy values and derivatives
  bit for bit, but not its jaxpr.
- So the migration gate should compare values and jvps on the stored grids. Do not compare jaxprs, or else keep the
  legacy multiplications.
- The baseline's leaf goldens already hold dense surface grids: incidences with the critical angles × index pairs ×
  parameter sets with their edges (`port/goldens/leaf_814e1868`, `surface.*`). They can be reused.
- REFLECTOR, SENSOR and PHOTOCATHODE carry `f_glossy` and `sigma_alpha` that no caller sets. The planned
  composition drops them, so these types change structure, not value.

## Where the build path refuses plug-ins today

| Place | What it does | What 0.2 needs |
|---|---|---|
| `build/legacy_library.surface_entry`, `_material_medium` | refuse any optics model other than the built-in names | fall back to `plugins.optics_models`, failing closed with the distribution's name and minimum version |
| `build/legacy_library._border_groups` | refuses a `border_into` model other than `lumen:geant4` | pass the group's model through to `rules.optics.sensor` (the sensor-model registry is already there) |
| `build/_legacy_regions` | requires `isinstance(kind, legacy.SurfaceType)` | a compiled surface model with an adapter to a `SurfaceType` (all its fields fit), until the step reads models directly |
| `build/spec.PROPERTIES` | gives a key it does not know an unknown, unbounded range | take the unit, range and method from the model's declared parameters |
| `build/pack.with_values` | needed a key for every property | has a fallback now (the Mie change) for properties the model fixes |
| `plugins.Registry.get` | names the registered entries | name the package to install and its minimum version; refuse a newer `model_version` |

## Lobe protocol

- The phase-A lobe is a record of functions, hashed by identity. A 0.2 lobe class should hash by value and keep the
  record's attribute names (`sample`, `logpdf`, `weight`, `axis`, `n_rand`, `delta`, `surface`, `path`,
  `pol_sample`).
- Its declared parameters then need another name (`PARAMS`), because the record's `params` is a tuple of key names.
  With that, the ported step consumes both kinds of lobe.
- The polarisation rule after each lobe (the facet mirror for chains, the implied facet for Lambert, unpolarised for
  Lambert transmission and re-emission, the projection for bulk scattering) lives in the baseline's step today. It
  moves into the lobes.

## Contract-kit traps

- **Declared randomness.** JAX clamps an out-of-range index silently: `rand[3]` on a length-3 array reads
  `rand[2]`. To check how many random numbers a lobe uses, append one more and require the output to be unchanged.
- **No closed-over floats.**
  - A float on `self` used in arithmetic becomes a jaxpr literal, which the jaxpr cannot tell apart from a
    structural constant. So check the factor's fields: no arrays, and no floats unless declared structural.
  - Then check the jaxpr's array constants (size above 1) against the factor's declared static tables.
- **Phase functions.** The repository's `lobe_contract.check_lobe` puts its direction grid around `axis`. It cannot
  resolve a backward lobe 1e-3 rad wide at `-wi`, so phase functions need a grid dense at both poles. `test_mie.py`
  uses one, in the angle from `wi`.
- **Edges of the ratio.** The score of a mixture weight at 0 or 1 has mean 0 but unbounded variance (the Mie forward
  ratio). The kit must skip the z-test there, or apply the edge rule's reference shift.

## Sensors

- An angle-dependent collection efficiency can be a surface detection factor today: the detected share of the
  absorbed light split by the incidence angle at the cathode.
- WCSim's collection efficiency is a function of the hit position on the PMT (`WCSimWCSD.cc:221-224`), not of the
  photon direction. That needs the deposit's hit coordinates. The two are different models, and a plug-in must say
  which one it is.
- The electronics half of `check_sensor_model` (time kernels and charge models) waits for the sensor and
  electronics step.

## Example plug-ins planned (not written)

- **A GGX rough conductor.** The weight form: one reflected lobe with weight F G2 / G1, and the lost share is
  absorbed. It is valid only on a surface that detects nothing.
  - Checks: the weak white-furnace identity, the integral of G1(wi) max(0, wi·m) D(m) dm = cos θi, by quadrature;
    BRDF reciprocity; the α → 0 limit.
- **A measured-BRDF table.** Density linear in cos θo per incidence node, constant in azimuth bins, and a mixture of
  the two nearest nodes, so the density is exactly the sampler's. Its table values can be parameters.
  - The region compiler packs scalars and vectors. A table of 3 or more dimensions as a property value is untested.
- **A tabulated phase function** as a bulk process: density linear in mu, inverse CDF in closed form. Check its
  moments against the table's integrals.
- **An angular collection efficiency** as a detection factor on the WCSim cathode form.
