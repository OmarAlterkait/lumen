# Parameters

Every value lumen reads while tracking is a parameter: you can set it, calibrate it and differentiate it. Two sorts of
value are not parameters:

- **the geometry**;
- **values Geant4 never reads** for that surface or material. Example: a ground-front-painted surface is always
  Lambertian, so it has no "specular fraction" parameter.

Whether Geant4 reads a value decides; the number does not. A value Geant4 reads stays a parameter even when it is 0,
1, or Geant4's default for a property the GDML leaves out. Example: Geant4 always reads TRANSMITTANCE on a polished
dielectric_dielectric surface (default 0, `rand > fReflectivity + fTransmittance`). So
`surfaces/<S>/TRANSMITTANCE` exists at 0 even when the GDML omits it, and `params.explain(key)` says "Geant4 default
0, G4OpBoundaryProcess.cc:310". Setting it adds the property (`lumen.format.apply`); the exporter writes it only when
a calibration has set it, so an uncalibrated detector exports as before. At such an edge (0 or 1) the derivative is
one-sided, and it may legitimately be 0.

An omitted ABSLENGTH has no key: for Geant4 that means "no absorption process", not a value, and calibrating it would
add a process.

Other values Geant4 never reads:

- the surface value (sigma_alpha or polish) on a polished or painted finish, where no facet is sampled, and at the
  values where Geant4 samples none: sigma_alpha exactly 0, polish exactly 1 (G4OpBoundaryProcess::GetFacetNormal).
  There the file's value, not its presence, decides: a surface at sigma_alpha 0.01 has the key, one at 0 has not,
  and a fit of sigma_alpha stays within the facet sampler's range, above 0;
- a WLS yield when the material has no WLSMEANNUMBERPHOTONS: G4OpWLS then emits exactly one photon.

A worked case: MicroBooNE's steel follows LArSoft's reflector, which reflects specularly about the global normal or
Lambertian and never samples a facet. lumen-tpc's recipe writes it in its Geant4 form: a unified, ground
dielectric_metal surface of sigma_alpha 0 with REFLECTIVITY, SPECULARSPIKECONSTANT (1 - LArSoft's diffuse fraction)
and zero lobe and backscatter. Geant4, and lumen, take the lobe at sigma_alpha 0 as the spike: no facet is sampled, so
sigma_alpha is not a parameter there. On a unified ground dielectric_metal surface with sigma_alpha above 0 Geant4
reads it at every lobe reflection, so there it stays a parameter even where every lobe fraction is 0; its zero
derivative is then a measured fact.
