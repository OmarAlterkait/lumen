"""Every value lumen reads from a detector file, declared once: its unit, admissible range and gradient method.

    PARAMS["REFLECTIVITY"].range        # (0.0, 1.0)
    PARAMS["sigma_alpha"].range         # the reader's admissible facet width, rad

``PARAMS`` maps a Geant4 property key (G4MaterialPropertiesTable), an opticalsurface's value (``sigma_alpha``,
``polish``) and a sensor group's value setting (``efficiency``) to a ``lumen.Param``; a property's unit is the
one it is stored in (``lumen.format.schema.PROPERTY_UNITS``, read from there). The
ranges are the readers' admissibility (``lumen.rules.optics`` refuses a file value outside them, and ``Params.set`` a
calibration), so a refusal and a range cannot drift apart. ``SURFACE_KEYS`` names the Geant4 key each surface-type
parameter reads; a key's range is its domain in ``lumen._core.domains``, the one table of ranges the surface types
read too, so the surfaces import nothing of the rules.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen._core import domains
from lumen.format.schema import PROPERTY_UNITS, Property
from lumen.params import Param
from lumen.rules.choices import GLISUR_R_MIN, MIE_G_MAX, SIGMA_MAX, SIGMA_MIN  # the accepted ranges (stated there)
from lumen.units import C_VACUUM

FRACTION = Param("", domains.FRACTION)  # the one table of ranges: lumen._core.domains
POSITIVE = Param("", domains.POSITIVE)
LENGTH = Param("", domains.LENGTH)
MIE_G = Param("", (-MIE_G_MAX, MIE_G_MAX))  # Geant4 allows a negative g (a flipped lobe)
# the Geant4-10 scintillation keys, inert user keys under Geant4 >= 11 (G4Scintillation finds no
# SCINTILLATIONCOMPONENT1 and emits nothing), and what Geant4 11 calls them
G4_10_KEYS = {
    "FASTCOMPONENT": "SCINTILLATIONCOMPONENT1",
    "SLOWCOMPONENT": "SCINTILLATIONCOMPONENT2",
    "FASTTIMECONSTANT": "SCINTILLATIONTIMECONSTANT1",
    "SLOWTIMECONSTANT": "SCINTILLATIONTIMECONSTANT2",
    "FASTSCINTILLATIONRISETIME": "SCINTILLATIONRISETIME1",
    "SLOWSCINTILLATIONRISETIME": "SCINTILLATIONRISETIME2",
    "YIELDRATIO": "(fast fraction of the yield)",  # no Geant4-11 key: read as the two components' yields
}
# the scintillation keys Geant4 11's G4Scintillation reads (lumen's scintillation sources read the same; a GDML's
# Geant4-10 keys are mapped to them at import, ``geant4_11_scintillation``): each component's spectrum and decay time,
# and the keys that share the light among them (lumen.sources.scintillation), and each component's rise time (read
# under the material's optics setting ``scint_finite_rise_time``, Geant4's G4OpticalParameters::SetScintFiniteRiseTime)
SCINTILLATION_KEYS = {
    "spectra": ("SCINTILLATIONCOMPONENT1", "SCINTILLATIONCOMPONENT2", "SCINTILLATIONCOMPONENT3"),
    "times": ("SCINTILLATIONTIMECONSTANT1", "SCINTILLATIONTIMECONSTANT2", "SCINTILLATIONTIMECONSTANT3"),
    "shares": ("SCINTILLATIONYIELD1", "SCINTILLATIONYIELD2", "SCINTILLATIONYIELD3"),
    "rises": ("SCINTILLATIONRISETIME1", "SCINTILLATIONRISETIME2", "SCINTILLATIONRISETIME3"),
}


def components(has) -> tuple[int, ...]:
    """Return the components G4Scintillation emits for a material whose keys ``has(key)`` tells (indices into
    ``SCINTILLATION_KEYS``).

    Geant4 11 (G4Scintillation::PostStepDoIt): three with SCINTILLATIONCOMPONENT3, else two with
    SCINTILLATIONCOMPONENT2, else one with SCINTILLATIONCOMPONENT1, else none.
    """  # noqa: D205
    spectra = SCINTILLATION_KEYS["spectra"]
    n = next((i + 1 for i in reversed(range(len(spectra))) if has(spectra[i])), 0)
    return tuple(range(n))


def geant4_11_scintillation(props: dict) -> tuple[dict, str | None]:
    """Return a material's properties with its Geant4-10 scintillation keys as Geant4 11 names them, and a note.

    Geant4 10.7's G4Scintillation emits each of FASTCOMPONENT and SLOWCOMPONENT the material has, with its
    FAST/SLOWTIMECONSTANT (and rise time), sharing the light min(YIELDRATIO, 1) and the rest when it has both; Geant4
    11 emits SCINTILLATIONCOMPONENT1 (2, 3) in order, shared by SCINTILLATIONYIELD1 (2, 3) over their sum. The
    components the material has are numbered 1, 2 in that order, and YIELDRATIO r becomes the yields (min(r, 1),
    1 - min(r, 1)): the same light. A material with Geant4 11 keys, or whose 10.7 reading is incomplete (a component
    without its time constant, two without YIELDRATIO: Geant4 10.7 aborts), is returned as it is (its 10 keys stay
    inert under Geant4 11, and the source refuses it, naming them). ``props``: {key: format.schema.Property}.
    """
    old = [k for k in props if k in G4_10_KEYS]
    if not old or any(k.startswith("SCINTILLATION") and k != "SCINTILLATIONYIELD" for k in props):
        return props, None
    present = [c for c in ("FAST", "SLOW") if f"{c}COMPONENT" in props]
    if (
        not present
        or any(f"{c}TIMECONSTANT" not in props for c in present)
        or (len(present) == 2 and "YIELDRATIO" not in props)
    ):
        return props, None
    out = {k: v for k, v in props.items() if k not in G4_10_KEYS}
    renamed = (
        ("COMPONENT", "SCINTILLATIONCOMPONENT"),
        ("TIMECONSTANT", "SCINTILLATIONTIMECONSTANT"),
        ("SCINTILLATIONRISETIME", "SCINTILLATIONRISETIME"),
    )
    for i, c in enumerate(present, 1):
        for suffix, key in renamed:
            v = props.get(f"{c}{suffix}")
            if v is not None:
                out[f"{key}{i}"] = Property(f"{key}{i}", v.value, energy=v.energy, origin=v.origin, note=v.note)
    if len(present) == 2:
        r = min(float(props["YIELDRATIO"].value), 1.0)
        y = props["YIELDRATIO"]
        out["SCINTILLATIONYIELD1"] = Property(
            "SCINTILLATIONYIELD1", np.asarray(r), energy=None, origin=y.origin, note=y.note
        )
        out["SCINTILLATIONYIELD2"] = Property(
            "SCINTILLATIONYIELD2", np.asarray(1.0 - r), energy=None, origin=y.origin, note=y.note
        )
    nums = ", ".join(str(i) for i in range(1, len(present) + 1))
    return out, f"Geant4-10 scintillation keys {sorted(old)} read as Geant4 11's (components {present} as {nums})"


# Geant4's optical-surface enums by name (G4OpticalSurface.hh, G4SurfaceProperty.hh; 10.7.4 and 11.2.2 agree), in enum
# order: a stored number is its position (the reader names a value by it; the exporter writes it).
# G4GDMLReadSolids::OpticalSurfaceRead accepts each name (model "dichroic" and type "x_ray" by falling through); GDML
# has no coated type (Geant4 11's "coated" falls to x_ray).
MODELS = ("glisur", "unified", "LUT", "DAVIS", "dichroic")
FINISHES = (
    "polished polishedfrontpainted polishedbackpainted ground groundfrontpainted groundbackpainted "
    "polishedlumirrorair polishedlumirrorglue polishedair polishedteflonair polishedtioair polishedtyvekair "
    "polishedvm2000air polishedvm2000glue etchedlumirrorair etchedlumirrorglue etchedair etchedteflonair etchedtioair "
    "etchedtyvekair etchedvm2000air etchedvm2000glue groundlumirrorair groundlumirrorglue groundair groundteflonair "
    "groundtioair groundtyvekair groundvm2000air groundvm2000glue Rough_LUT RoughTeflon_LUT RoughESR_LUT "
    "RoughESRGrease_LUT Polished_LUT PolishedTeflon_LUT PolishedESR_LUT PolishedESRGrease_LUT"
).split()
TYPES = ("dielectric_metal", "dielectric_dielectric", "dielectric_LUT", "dielectric_LUTDAVIS", "dielectric_dichroic",
         "firsov", "x_ray")  # fmt: skip
# Geant4 11.3 property keys (G4MaterialPropertiesTable: fMatPropNames and fMatConstPropNames)
TABLE_KEYS = frozenset(
    (
        "RINDEX REFLECTIVITY REALRINDEX IMAGINARYRINDEX EFFICIENCY TRANSMITTANCE SPECULARLOBECONSTANT "
        "SPECULARSPIKECONSTANT BACKSCATTERCONSTANT GROUPVEL MIEHG RAYLEIGH WLSCOMPONENT WLSABSLENGTH WLSCOMPONENT2 "
        "WLSABSLENGTH2 ABSLENGTH PROTONSCINTILLATIONYIELD DEUTERONSCINTILLATIONYIELD TRITONSCINTILLATIONYIELD "
        "ALPHASCINTILLATIONYIELD IONSCINTILLATIONYIELD ELECTRONSCINTILLATIONYIELD SCINTILLATIONCOMPONENT1 "
        "SCINTILLATIONCOMPONENT2 SCINTILLATIONCOMPONENT3 COATEDRINDEX"
    ).split()
)
CONST_KEYS = frozenset(
    (
        "SURFACEROUGHNESS WLSMEANNUMBERPHOTONS WLSTIMECONSTANT WLSMEANNUMBERPHOTONS2 WLSTIMECONSTANT2 "
        "SCINTILLATIONYIELD RESOLUTIONSCALE FERMIPOT DIFFUSION SPINFLIP LOSS LOSSCS ABSCS SCATCS COATEDTHICKNESS "
        "COATEDFRUSTRATEDTRANSMISSION MIEHG_FORWARD MIEHG_BACKWARD MIEHG_FORWARD_RATIO"
    ).split()
    + [f"{p}SCINTILLATION{q}{i}" for p in ("", "PROTON", "DEUTERON", "TRITON", "ALPHA", "ION", "ELECTRON")
       for q in ("YIELD", "TIMECONSTANT") for i in (1, 2, 3)]
    + [f"SCINTILLATIONRISETIME{i}" for i in (1, 2, 3)]
)  # fmt: skip
# Geant4's material and surface keys: its tables, its constants and the Geant4-10 scintillation keys (one copy: the
# keys lumen reads by default and the GDML exporter writes)
GEANT4_KEYS = TABLE_KEYS | CONST_KEYS | frozenset(G4_10_KEYS)
# the material keys a light source reads (not the optics): parameters even where no surface or medium reads them
SOURCE_KEYS = frozenset({"RINDEX", "SCINTILLATIONYIELD"}.union(*SCINTILLATION_KEYS.values()))
PARAMS = {
    # RINDEX moves light paths and changes Fresnel weights: its derivative is pending validation
    "RINDEX": Param("", domains.INDEX, method="moving_paths+reweight"),
    "ABSLENGTH": LENGTH, "RAYLEIGH": LENGTH, "WLSABSLENGTH": LENGTH, "MIEHG": LENGTH,
    "REFLECTIVITY": FRACTION, "TRANSMITTANCE": FRACTION, "EFFICIENCY": FRACTION,
    "SPECULARLOBECONSTANT": FRACTION, "SPECULARSPIKECONSTANT": FRACTION, "BACKSCATTERCONSTANT": FRACTION,
    "MIEHG_FORWARD": MIE_G, "MIEHG_BACKWARD": MIE_G, "MIEHG_FORWARD_RATIO": FRACTION,
    "WLSCOMPONENT": POSITIVE, "WLSTIMECONSTANT": POSITIVE, "WLSMEANNUMBERPHOTONS": POSITIVE,
    "GROUPVEL": Param("", (0.0, C_VACUUM)), "RS_SCALE_FACTOR": POSITIVE,
    "ISOTHERMAL_COMPRESSIBILITY": POSITIVE, "SCINTILLATIONYIELD": POSITIVE,
    **dict.fromkeys(SCINTILLATION_KEYS["rises"], POSITIVE),
    # a complex index (lumen.rules.optics.complex_index): its real part read through an index law, its imaginary part
    # an extinction (k >= 0)
    "REALRINDEX": Param("", domains.INDEX), "IMAGINARYRINDEX": POSITIVE,
    "sigma_alpha": Param("rad", (SIGMA_MIN, SIGMA_MAX)),  # unified's opticalsurface value
    "polish": Param("", (0.0, 1.0 - GLISUR_R_MIN)),  # glisur's opticalsurface value
    "entry_efficiency": FRACTION,  # a sensor group's efficiency on entry (its setting)
}  # fmt: skip
# a property's unit is the one it is stored in (one table: lumen.format.schema.PROPERTY_UNITS)
PARAMS = {k: dataclasses.replace(p, unit=PROPERTY_UNITS[k][1]) if k in PROPERTY_UNITS else p for k, p in PARAMS.items()}
UNKNOWN = Param()  # a key no reader declares: dimensionless, any value
SURFACE_KEYS = {"R": "REFLECTIVITY", "T": "TRANSMITTANCE", "p_spike": "SPECULARSPIKECONSTANT",
                "p_lobe": "SPECULARLOBECONSTANT", "p_back": "BACKSCATTERCONSTANT", "eff": "EFFICIENCY"}  # fmt: skip
