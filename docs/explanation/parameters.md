# Parameters

Every optical value lumen tracks with is a parameter: you can calibrate it, and it has a derivative (D39). The only
exceptions are the geometry, and a value the model never reads. The model never reads a value when Geant4's process,
given the surface's or material's type, model and finish, has no code path that uses it. Such a value is OUT (D68): it
is folded into the surface type (`painted{f_spec=0}`), it is not packed as a parameter, and it is never a reported or
gated gradient family.

The structure of the model decides, never the number. A value the model reads stays a parameter whatever it is,
including 0, 1, and Geant4's own default for a property the file leaves out. For example, G4OpBoundaryProcess starts
every step with TRANSMITTANCE 0 and reads it (`rand > fReflectivity + fTransmittance`), so a polished
dielectric_dielectric surface without TRANSMITTANCE has the parameter T = 0. Its derivative there may legitimately be
0, or one-sided.

MicroBooNE's steel shows the rule. Its surface follows LArSoft's OpBoundaryProcessSimple (`lumen:larsoft_reflector`),
which reflects specularly about the global normal or Lambertian, and never samples a facet or a lobe. So its facet
width sigma_alpha (0.1) and its lobe fraction f_glossy (0) are OUT. The reason is that no code path reads them, not
that 0.1 or 0 are special values. On a unified dielectric_metal surface with a ground finish, Geant4 reads sigma_alpha
in GetFacetNormal for every lobe reflection, so there sigma_alpha stays a parameter even where every lobe fraction is
0. Its zero derivative is then a measured fact, not a reason to drop it.

Other structural cases:

- a frontpainted finish's specular fraction: groundfrontpainted is Lambertian and polishedfrontpainted a mirror;
- the surface value (sigma_alpha or polish) on a polished or painted finish, where no facet is sampled;
- a WLS film's yield when the material has no WLSMEANNUMBERPHOTONS: G4OpWLS then emits exactly one photon and
  reads no mean;
- the film's thickness and side, which are geometry.
