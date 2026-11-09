"""Geant4 forms of lumen surface models: the Geant4 opticalsurface that makes Geant4 simulate what the model does.

The surface-model contract's ``geant4_form`` (gdml_export.md section 4.3), for the models lumen ships: an export writes
the form in place of the model's record (the record itself goes to the sidecar, so ``read_export`` restores it), and
the export stays lossless. A model with no entry here has no Geant4 form: lossy (``on_lossy``).

``lumen:larsoft_reflector`` (LArSoft's OpBoundaryProcessSimple surface, ``lumen.detectors.larsoft_optics``): a
reflectance R, of which a fraction f_diff is Lambertian and the rest specular, the remainder absorbed. Its Geant4 form
is a dielectric_metal, unified, ground surface with sigma_alpha 0 and REFLECTIVITY R, SPECULARSPIKECONSTANT 1 - f_diff,
SPECULARLOBECONSTANT 0, BACKSCATTERCONSTANT 0: G4OpBoundaryProcess::DielectricMetal reflects with probability R and
chooses spike, lobe, backscatter, else Lambertian (the remainder, f_diff); with sigma_alpha 0 the lobe would be the
spike. It is the mapping the MicroBooNE reference uses (g4ref/geom/mk_microboone_optics.py). The tables keep the
model's energies, and its energy tokens (``lumen.gdml.tokens``) carry over to every table of the form.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen.format import schema

REFLECTIVITY, DIFFUSE_FRACTION = "REFLECTIVITY", "DIFFUSE_FRACTION"
SPIKE, LOBE, BACK = "SPECULARSPIKECONSTANT", "SPECULARLOBECONSTANT", "BACKSCATTERCONSTANT"
NOTE = "Geant4 form of lumen:larsoft_reflector (lumen.gdml.geant4_forms)"


def larsoft_reflector(s: schema.Surface, tok: dict | None) -> tuple[schema.Surface, dict | None]:
    """Return the Geant4 form of a ``lumen:larsoft_reflector`` surface and its tokens (see the module docstring)."""
    r, f = s.properties.get(REFLECTIVITY), s.properties.get(DIFFUSE_FRACTION)
    if r is None or f is None or r.energy is None or f.energy is None or not np.array_equal(r.energy, f.energy):
        raise ValueError(f"surface {s.name!r}: lumen:larsoft_reflector needs REFLECTIVITY and DIFFUSE_FRACTION tables "
                         "on the same energies")  # fmt: skip
    if (s.type, s.model, s.finish, s.value) != ("dielectric_metal", "unified", "ground", 0.0):
        raise ValueError(f"surface {s.name!r}: lumen:larsoft_reflector is a dielectric_metal, unified, ground surface "
                         "with sigma_alpha 0")  # fmt: skip
    e = np.asarray(r.energy, np.float64)
    zero = np.zeros_like(e)

    def table(key, value):
        return schema.Property(key, np.asarray(value, np.float64), e, "", "recipe", NOTE)

    props = {REFLECTIVITY: table(REFLECTIVITY, r.value), SPIKE: table(SPIKE, 1.0 - np.asarray(f.value, np.float64)),
             LOBE: table(LOBE, zero), BACK: table(BACK, zero)}  # fmt: skip
    energy = ((tok or {}).get(REFLECTIVITY) or {}).get("energy")
    toks = None if energy is None else {k: {"energy": list(energy), "value": None, "matrix_unit": None} for k in props}
    return dataclasses.replace(s, properties=props, optics=schema.OpticsModel()), toks


FORMS = {"lumen:larsoft_reflector": larsoft_reflector}
