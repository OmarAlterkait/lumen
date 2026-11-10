"""The "geant4" optics model: a detector file's material and surface properties read as Geant4 reads them.

    entry = medium(df.materials["LAr"])                 # a MediumEntry (opaque without RINDEX)
    entry = surface(df.surfaces["Surface_Steel_1"])     # a SurfaceEntry, Geant4 11 semantics, fail closed
    entry = sensor(df.surfaces["GlassCathode"], lam)    # a SensorEntry: how a detecting surface reads (SENSOR_SURFACES)

The records are ``lumen.format.schema`` Material and Surface records: energies in eV, lengths in m (tables and
constants), every other value as Geant4 holds it (times in ns, yields per MeV). Geant4 semantics are those of
G4OpBoundaryProcess, G4OpWLS and G4OpRayleigh in Geant4 11.3 (the code paths used here are the same in 10.7).

Tables are linear in photon ENERGY between their knots and clamped to the end values outside
(G4PhysicsVector::Value): ``laws.tabulated(..., outside="clamp")``, one shared law per output quantity; ``coverage``
lists every table whose knots do not cover a wavelength range. RINDEX is ``laws.geant4_rindex``: the same n(E), whose
group index is Geant4's GROUPVEL.

Media (G4OpAbsorption, G4OpRayleigh, G4OpMieHG, G4OpWLS): RINDEX gives the index (none: opaque, Geant4's NoRINDEX
kill), ABSLENGTH the absorption, RAYLEIGH a Rayleigh channel (for a material named Water without one, Geant4's own
table: ``geant4_rayleigh``), MIEHG with MIEHG_FORWARD, MIEHG_BACKWARD and MIEHG_FORWARD_RATIO a Mie channel (a double
Henyey-Greenstein, ``phase.MIE_HG``: ``mie``), WLSABSLENGTH with WLSCOMPONENT a re-emitting channel (one photon,
isotropic, Geant4's sampler: ``spectra.geant4_spectrum``); the channels in the order rayleigh, mie, wls. Refused: the
second WLS, WLSMEANNUMBERPHOTONS on a bulk medium (lowered only on a film: ``wls_film``), a complex index, a user
GROUPVEL (one that is not Geant4's own from RINDEX, which a GDML export writes: ``geant4_groupvel_export``), a nonzero
WLSTIMECONSTANT without a declared time profile, and a Mie constant Geant4 would read but the material lacks.

Surfaces (G4OpBoundaryProcess::PostStepDoIt, DielectricMetal, DoReflection; no table: R = 1, T = 0, efficiency 0):

    dielectric_metal, glisur, ground, polish < 1        GLISUR_METAL: R facet path (r = 1 - polish)
    dielectric_metal, glisur (other finish, polish 1)   SPECULAR_METAL: R mirror
    dielectric_metal, unified, polished                 SPECULAR_METAL
    dielectric_metal, unified, ground                   UNIFIED_METAL: spike / lobe / back / Lambert, Geant4's cuts
    dielectric_dielectric, polishedfrontpainted         PAINTED_T (f_spec 1): R mirror, T straight, rest absorbed
    dielectric_dielectric, groundfrontpainted           PAINTED_T (f_spec 0): R Lambert about the normal
    dielectric_dielectric, polished                     DIELECTRIC_RT: absorbed 1 - R - T, T straight, R to Fresnel,
                                                        skipped between one material

The type does not depend on whether R or T is present on a polished surface: Geant4 reads them at their defaults (R
1, T 0) when the surface omits them (``surface_defaults``), so an omitted one is a parameter at that default
(docs/explanation/parameters.md); so is a frontpainted surface's T (a transmission into a medium without RINDEX is
absorbed where it lands: ``lumen.rules.boundary``). A WLS film's backing models no transmission (PAINTED): a
TRANSMITTANCE on it is refused.

Refused: the LUT, DAVIS and dichroic models and surface types, backpainted finishes, a ground dielectric_dielectric,
a complex index, SURFACEROUGHNESS, a coated surface's keys, a nonzero EFFICIENCY outside DIELECTRIC_RT (a detecting
surface is a sensor: ``sensor``), TRANSMITTANCE on a metal, a unified sigma_alpha outside [SIGMA_MIN, SIGMA_MAX] and
a GLISUR polish below 0 or with 1 - polish below GLISUR_R_MIN.

Detecting surfaces (``sensor``): how a sensor's optical surface behaves during tracking, read by one of
``SENSOR_SURFACES`` (``reader(surface, lam_domain) -> SensorEntry``), named by the sensor group's model (or a callable
of that form). Not the sensor model (``lumen.plugins.sensor_models``): that is what happens after tracking.

    lumen:geant4 (the default)
        the surface as stock Geant4 (G4OpBoundaryProcess, 10.7 / 11) reads it. dielectric_metal: REFLECTIVITY R
        (absent: the complex index REALRINDEX + i IMAGINARYRINDEX, CalculateReflectivity, a PHOTOCATHODE; neither:
        R = 1), the absorbed 1 - R detected with EFFICIENCY (default 0); the reflection by model and finish: polished
        (glisur or unified) a mirror (SENSOR, f_spec 1), glisur ground the facet path (GLISUR_METAL_EFF), unified
        ground spike / lobe / back / Lambert (DIELECTRIC_METAL). dielectric_dielectric, polished (R / T at Geant4's
        defaults if absent): DIELECTRIC_RT_SENSOR, eff = EFFICIENCY. Keys no stock process reads on a surface (RINDEX,
        KINDEX, THICKNESS on a dielectric_metal) are ignored, as Geant4 ignores them.
    lumen:complex_index
        a thin-film photocathode's tables some experiment software writes on the surface (RINDEX, KINDEX), read as a
        semi-infinite absorber of index RINDEX + i KINDEX (PHOTOCATHODE, specular, detected with EFFICIENCY): not the
        film's interference, an approximation of it.

The complex index of either is taken at the range's centre and must be flat over it (refused otherwise: narrow the
range or supply a spectral reader). A sensitive detector that counts every absorption (WCSim's) replaces the
efficiency: ``sensor(..., eff=1)``.

Spectral surface parameters (REFLECTIVITY(E), ...) are (law, law parameters) pairs; the surface lookup
(``lumen.build.surfaces.compile``) packs them.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lumen.build.surfaces import DescriptionError, MediumEntry, SensorEntry, SurfaceEntry
from lumen.format import schema
from lumen.media import bulk, laws, phase, spectra
from lumen.surfaces import types as legacy
from lumen.surfaces.outcomes import DETECT
from lumen.units import HC

# the facet lobes' admissible widths (the surface types name them)
GLISUR_R_MIN = 0.01  # the smallest 1 - polish of a GLISUR facet path
SIGMA_MIN, SIGMA_MAX = 0.005, 0.5  # rad: the unified facet lobe's sigma_alpha range

C_MM_PER_NS = 299.792458  # Geant4's c_light, mm / ns (GROUPVEL is in mm / ns)
GROUPVEL_RTOL = 2e-3  # an exported GROUPVEL's tolerance (6 printed digits, the RINDEX rounding through dn / d log E)
GROUPVEL_ENERGY_RTOL = 1e-5  # an exported GROUPVEL's knots (6 printed digits)
WATER_COMPRESSIBILITY = 7.658e-23  # m3 / MeV: G4OpRayleigh's isothermal compressibility of "Water"
MM3_PER_M3 = 1e9  # Geant4's internal volume unit is mm3
K_BOLTZMANN = 8.617333262e-11  # MeV / K
NTP_TEMPERATURE = 293.15  # K: G4GDMLReadMaterials' default temperature (NTP_Temperature)
HC_EV_MM = 1.239841984e-3  # eV mm: h c
M_PER_MM = 1e-3
FILM_GRID = 1025  # energies at which a film's absorption of its own emission is checked
FILM_SELF_ABSORPTION = 1e-6  # the largest thickness / WLSABSLENGTH a film may have over its own emission band
FLAT = 1e-3  # the relative variation over the range a sensor's complex-index table may have (taken at the centre)
TINY = 1e-12  # the floor of the scale of that variation

# G4OpticalSurfaceModel, G4OpticalSurfaceFinish and G4SurfaceType (Geant4 10.7 / 11): a GDML export writes the enum
# values, G4GDMLReadSolids::OpticalSurfaceRead takes names or values (the LUT finishes past groundbackpainted are
# named by value and refused)
MODELS = ("glisur", "unified", "LUT", "DAVIS", "dichroic")
FINISHES = ("polished", "polishedfrontpainted", "polishedbackpainted", "ground", "groundfrontpainted",
            "groundbackpainted")  # fmt: skip
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
# the Geant4-10 scintillation keys, inert user keys under Geant4 >= 11 (G4Scintillation finds no
# SCINTILLATIONCOMPONENT1 and emits nothing), and what Geant4 11 calls them
G4_10_KEYS = {
    "FASTCOMPONENT": "SCINTILLATIONCOMPONENT1",
    "SLOWCOMPONENT": "SCINTILLATIONCOMPONENT2",
    "FASTTIMECONSTANT": "SCINTILLATIONTIMECONSTANT1",
    "SLOWTIMECONSTANT": "SCINTILLATIONTIMECONSTANT2",
    "FASTSCINTILLATIONRISETIME": "SCINTILLATIONRISETIME1",
    "SLOWSCINTILLATIONRISETIME": "SCINTILLATIONRISETIME2",
    "YIELDRATIO": "(fast fraction of the yield)",
}

REFUSED_MEDIUM = {"WLSCOMPONENT2": "a second WLS (G4OpWLS2)",
                  "WLSABSLENGTH2": "a second WLS (G4OpWLS2)", "WLSMEANNUMBERPHOTONS": "a Poisson WLS count",
                  "WLSMEANNUMBERPHOTONS2": "a Poisson WLS count", "REALRINDEX": "a complex index",
                  "IMAGINARYRINDEX": "a complex index", "COATEDRINDEX": "a coated surface property"}  # fmt: skip
REFUSED_SURFACE = ("REALRINDEX", "IMAGINARYRINDEX", "SURFACEROUGHNESS", "COATEDRINDEX", "COATEDTHICKNESS")
SUPPORTED_FINISHES = ("polished", "ground", "polishedfrontpainted", "groundfrontpainted")
MIE_KEYS = ("MIEHG_FORWARD", "MIEHG_BACKWARD", "MIEHG_FORWARD_RATIO")  # the lobe parameters of phase.MIE_PARAMS


class OpticsRefused(DescriptionError):  # noqa: N818 (named like the importer's ImportRefused)
    """A property or surface Geant4 reads in a way lumen does not reproduce: refused, never approximated."""


# ------------------------------------------------------------------------------------------------- the shared laws
TABLE_LAWS = {out: laws.tabulated(out, 1.0, 2.0, outside="clamp") for out in ("index", "length", "fraction", "scale")}
GEANT4_RINDEX = laws.geant4_rindex()


def is_table(p: schema.Property) -> bool:
    """Return whether a property is a table (energies and values) rather than a constant."""
    return p.energy is not None


def table(p: schema.Property, out: str):
    """Return the shared clamped law of output ``out`` and the parameters of a table property."""
    return TABLE_LAWS[out], {"energy": np.asarray(p.energy, np.float64), "value": np.asarray(p.value, np.float64)}


def fraction(props: dict, key: str, default: float):
    """Return a fraction property: its (law, parameters) for a table, its value for a constant, else ``default``."""
    if key not in props:
        return default
    p = props[key]
    return table(p, "fraction") if is_table(p) else float(p.value)


def nonzero(props: dict, key: str) -> bool:
    """Return whether a property is present with a nonzero value anywhere."""
    p = props.get(key)
    return p is not None and bool(np.any(np.asarray(p.value) != 0.0))


def _enum(names, v):
    v = str(v).strip()
    if not v.isdigit():
        return v
    i = int(v)
    return names[i] if i < len(names) else f"enum{i}"


def surface_names(s: schema.Surface) -> tuple[str, str, str]:
    """Return a surface's (model, finish, type) as names (a GDML export writes Geant4's enum values)."""
    return _enum(MODELS, s.model), _enum(FINISHES, s.finish), _enum(TYPES, s.type)


# ----------------------------------------------------------------------------------------------------- reporting
def coverage(df: schema.DetectorFile, lam_domain) -> list[str]:
    """Return report lines: every table whose knots do not cover ``lam_domain`` (nm); Geant4 clamps there."""
    e_lo, e_hi = HC / max(lam_domain), HC / min(lam_domain)
    owners = [(f"material {m!r}", r.properties) for m, r in df.materials.items()]
    owners += [(f"surface {s!r}", r.properties) for s, r in df.surfaces.items()]
    out = []
    for who, props in owners:
        for k, p in sorted(props.items()):
            if is_table(p) and (p.energy[0] > e_lo + 1e-12 or p.energy[-1] < e_hi - 1e-12):
                out.append(f"  {who}.{k}: knots {p.energy[0]:.4g}..{p.energy[-1]:.4g} eV "
                           f"({HC / p.energy[-1]:.1f}..{HC / p.energy[0]:.1f} nm), clamped outside")  # fmt: skip
    return out


def key_report(materials: dict) -> list[str]:
    """Return report lines naming the Geant4-10 keys and the keys no Geant4 optical process reads, per material."""
    out = []
    for m, rec in sorted(materials.items()):
        legacy_keys = sorted(k for k in rec.properties if k in G4_10_KEYS)
        user = sorted(k for k in rec.properties if k not in TABLE_KEYS | CONST_KEYS | set(G4_10_KEYS))
        if legacy_keys:
            out.append(
                f"  {m!r}: Geant4-10 keys {legacy_keys}: inert user keys under Geant4 >= 11 (scintillation(...))"
            )
        if user:
            out.append(f"  {m!r}: user keys {user}: read by no Geant4 optical process")
    return out


# ---------------------------------------------------------------------------------------------------------- media
def geant4_groupvel_export(props: dict, rtol: float = GROUPVEL_RTOL) -> bool:
    """Return whether a material's GROUPVEL table is Geant4's own CalculateGROUPVEL of its RINDEX.

    G4GDMLWriteMaterials writes every property of the table, the computed GROUPVEL included: an export, not a user
    table. It has the same knots (the first RINDEX knot, the midpoints, the last) and v_g = c / n_g (mm / ns) there.

    Notes
    -----
    ``rtol``: the writer prints 6 digits, and the exported RINDEX's rounding enters n_g through dn / d log E (WCSim's
    WCTE glass and gel: up to 7e-4 at a knot). lumen's timing uses n_g from the RINDEX it reads, within that rounding
    of Geant4's.
    """
    if "RINDEX" not in props or not is_table(props["GROUPVEL"]) or not is_table(props["RINDEX"]):
        return False
    e, n = (np.asarray(x, np.float64) for x in (props["RINDEX"].energy, props["RINDEX"].value))
    gv = props["GROUPVEL"]
    if len(e) < 2:
        return len(gv.energy) == len(e) and np.allclose(gv.value, C_MM_PER_NS / n, rtol=rtol)
    eg, ng = (np.asarray(x, np.float64) for x in laws.geant4_group_index_table(jnp.asarray(e), jnp.asarray(n)))
    return bool(len(gv.energy) == len(eg) and np.allclose(gv.energy, eg, rtol=GROUPVEL_ENERGY_RTOL)
                and np.allclose(gv.value, C_MM_PER_NS / ng, rtol=rtol))  # fmt: skip


def geant4_rayleigh(material: schema.Material) -> schema.Property | None:
    """Return Geant4's own Rayleigh table for a material without RAYLEIGH, or None.

    The Einstein-Smoluchowski length from the isothermal compressibility (7.658e-23 m3/MeV for a material named
    "Water", the name stripped of its '0x...' suffix as G4GDMLRead does; else the constant ISOTHERMAL_COMPRESSIBILITY),
    the material's temperature and RINDEX, times RS_SCALE_FACTOR, on the RINDEX knots (energies eV, lengths m).
    None when there is no compressibility or no RINDEX table.

    Notes
    -----
    G4OpRayleigh::CalculateRayleighMeanFreePaths (10.4+): 1 / L = c1 (2 pi / lambda)^4 ((n^2 - 1)(n^2 + 2) / 3)^2
    with c1 = s beta_T T k_B / (6 pi).
    Notation: T the temperature (K), L the length.
    """
    props = material.properties
    if "RINDEX" not in props or not is_table(props["RINDEX"]):
        return None
    if material.name.split("0x")[0] == "Water":
        beta = WATER_COMPRESSIBILITY * MM3_PER_M3  # mm3 / MeV (Geant4's internal units)
    elif "ISOTHERMAL_COMPRESSIBILITY" in props:
        beta = float(props["ISOTHERMAL_COMPRESSIBILITY"].value)
    else:
        return None
    scale = float(props["RS_SCALE_FACTOR"].value) if "RS_SCALE_FACTOR" in props else 1.0
    T = NTP_TEMPERATURE if material.temperature is None else float(material.temperature)
    c1 = scale * beta * T * K_BOLTZMANN / (6.0 * np.pi)  # mm3
    e = np.asarray(props["RINDEX"].energy, np.float64)
    n2 = np.asarray(props["RINDEX"].value, np.float64) ** 2
    lam = HC_EV_MM / e  # mm
    L_mm = 1.0 / (c1 * (2.0 * np.pi / lam) ** 4 * ((n2 - 1.0) * (n2 + 2.0) / 3.0) ** 2)
    return schema.Property("RAYLEIGH", L_mm * M_PER_MM, e, f"g4:{material.name}", "rule",
                           "Geant4's own Rayleigh table (G4OpRayleigh)")  # fmt: skip


def _check_medium(props: dict, what: str):
    for k, why in REFUSED_MEDIUM.items():
        if k in props:
            raise OpticsRefused(f"{what}: {k} ({why}) is not supported")
    if "GROUPVEL" in props and not geant4_groupvel_export(props):
        raise OpticsRefused(f"{what}: GROUPVEL (a user GROUPVEL, not Geant4's own from RINDEX) is not supported")


def _index_law(props: dict, index: str):
    if index == "geant4":
        return GEANT4_RINDEX, {"energy": props["RINDEX"].energy, "value": props["RINDEX"].value}
    if index == "tabulated":
        return table(props["RINDEX"], "index")
    raise OpticsRefused(f"index must be 'geant4' or 'tabulated', got {index!r}")


def _wls_channel(props: dict, name: str, what: str, wls_time_profile):
    if "WLSCOMPONENT" not in props:
        raise OpticsRefused(f"{what}: WLSABSLENGTH without WLSCOMPONENT (G4OpWLS absorbs and emits nothing)")
    tau = float(props["WLSTIMECONSTANT"].value) if "WLSTIMECONSTANT" in props else 0.0
    if tau != 0.0 and wls_time_profile != "exponential":
        raise OpticsRefused(f"{what}: WLSTIMECONSTANT {tau} ns: Geant4's default WLS time profile is 'delta' (a fixed "
                            "delay); lumen's delay is exponential: declare wls_time_profile='exponential' if that is "
                            "the Geant4 configuration")  # fmt: skip
    law, lp = table(props["WLSABSLENGTH"], "length")
    spec = spectra.geant4_spectrum(f"{name}.WLSCOMPONENT", props["WLSCOMPONENT"].energy, props["WLSCOMPONENT"].value)
    return bulk.Channel("wls", phase.ISOTROPIC, law, reemit=spec), {"length": lp, "yield": 1.0, "tau": tau}


def mie_constants(props: dict, what: str) -> tuple[dict, frozenset]:
    """Return the Mie lobe's parameters read from the constants, and the constants Geant4 never reads.

    G4OpMieHG reads MIEHG_FORWARD_RATIO at every Mie event, MIEHG_FORWARD only when the forward lobe is drawn
    (``rand <= ratio``, never at ratio 0) and MIEHG_BACKWARD only otherwise (never at ratio 1); a constant it reads
    and the material lacks makes it abort (mat202), so that is refused here. A constant it never reads and the
    material lacks is 0 (any value gives the same light). |g| <= ``phase.MIE_G_MAX`` (float32 resolution) and
    0 <= ratio <= 1. Without MIEHG the three constants are never read; an opaque material (no RINDEX) is not read.

    The second set lists the constants that are not parameters: all three without MIEHG; with MIEHG, the ratio when
    it is 0 or 1 and the material lacks the other lobe (moving it would need that lobe).
    """
    if "RINDEX" not in props:
        return {}, frozenset()
    present = [k for k in MIE_KEYS if k in props]
    for k in present:
        if is_table(props[k]):
            raise OpticsRefused(f"{what}: {k} must be a constant (G4OpMieHG reads a constant property)")
    if "MIEHG" not in props:
        return {}, frozenset(present)
    if "MIEHG_FORWARD_RATIO" not in props:
        raise OpticsRefused(f"{what}: MIEHG without MIEHG_FORWARD_RATIO: G4OpMieHG aborts (mat202) at the first Mie "
                            "event; add MIEHG_FORWARD_RATIO")  # fmt: skip
    r = float(np.asarray(props["MIEHG_FORWARD_RATIO"].value).reshape(-1)[0])
    if not 0.0 <= r <= 1.0:
        raise OpticsRefused(f"{what}: MIEHG_FORWARD_RATIO {r} outside [0, 1]")
    lobe, fixed = {"forward_ratio": r}, set()
    for key, name, read, lobe_name in (("MIEHG_FORWARD", "g_forward", r > 0.0, "forward"),
                                       ("MIEHG_BACKWARD", "g_backward", r < 1.0, "backward")):  # fmt: skip
        if key not in props:
            if read:
                raise OpticsRefused(f"{what}: MIEHG_FORWARD_RATIO {r} without {key}: G4OpMieHG aborts (mat202) at the "
                                    f"first {lobe_name} Mie event; add {key}")  # fmt: skip
            lobe[name] = 0.0
            fixed.add("MIEHG_FORWARD_RATIO")
            continue
        g = float(np.asarray(props[key].value).reshape(-1)[0])
        if not abs(g) <= phase.MIE_G_MAX:
            raise OpticsRefused(f"{what}: {key} {g} outside [-{phase.MIE_G_MAX}, {phase.MIE_G_MAX}] (a narrower lobe "
                                "than float32 directions resolve)")  # fmt: skip
        lobe[name] = g
    return lobe, frozenset(fixed)


def mie(props: dict, what: str):
    """Return Geant4's Mie channel of a material with MIEHG, and its parameters (``mie_constants`` checks them)."""
    lobe, _ = mie_constants(props, what)
    law, lp = table(props["MIEHG"], "length")
    return bulk.mie(law), {"length": lp, "lobe": {k: lobe[k] for k in phase.MIE_PARAMS}}


def medium(material: schema.Material, name: str | None = None, index: str = "geant4", wls_time_profile=None):
    """Return the MediumEntry of a material, Geant4's reading (see the module docstring), named ``name``.

    ``index``: "geant4" (Geant4's GROUPVEL timing) or "tabulated" (the group index of the piecewise-linear n).
    ``wls_time_profile``: needed ("exponential") only for a nonzero WLSTIMECONSTANT.
    """
    props = material.properties
    name = name or material.name
    what = f"material {material.name!r}"
    _check_medium(props, what)
    if "RINDEX" not in props:
        return MediumEntry(bulk.Medium(name, None), {})
    for k in ("RINDEX", "ABSLENGTH", "RAYLEIGH", "MIEHG", "WLSABSLENGTH", "WLSCOMPONENT"):
        if k in props and not is_table(props[k]):
            raise OpticsRefused(f"{what}: {k} must be a table")
    p = {}
    ilaw, p["index"] = _index_law(props, index)
    alaw = None
    if "ABSLENGTH" in props:
        alaw, p["absorption"] = table(props["ABSLENGTH"], "length")
    channels = []
    ray = props["RAYLEIGH"] if "RAYLEIGH" in props else geant4_rayleigh(material)
    if ray is not None:
        law, lp = table(ray, "length")
        channels.append(bulk.Channel("rayleigh", phase.RAYLEIGH, law))
        p["rayleigh"] = {"length": lp}
    if "MIEHG" in props:
        ch, p["mie"] = mie(props, what)
        channels.append(ch)
    if "WLSABSLENGTH" in props:
        ch, p["wls"] = _wls_channel(props, name, what, wls_time_profile)
        channels.append(ch)
    return MediumEntry(bulk.Medium(name, ilaw, alaw, tuple(channels)), p)


# ------------------------------------------------------------------------------------------------------- surfaces
def _metal(s: schema.Surface, model: str, finish: str, what: str, refl):
    pr = s.properties
    if nonzero(pr, "TRANSMITTANCE"):
        raise OpticsRefused(f"{what}: TRANSMITTANCE on a metal would put the photon into it")
    if model == "glisur":
        if finish == "ground" and s.value < 1.0:
            if s.value < 0.0 or 1.0 - s.value < GLISUR_R_MIN:
                raise OpticsRefused(f"{what}: polish {s.value} outside [0, {1 - GLISUR_R_MIN}] (or exactly 1)")
            return SurfaceEntry(legacy.GLISUR_METAL, {"R": refl, "polish": float(s.value)})
        return SurfaceEntry(legacy.SPECULAR_METAL, {"R": refl})
    if finish == "polished":
        return SurfaceEntry(legacy.SPECULAR_METAL, {"R": refl})
    if not SIGMA_MIN <= s.value <= SIGMA_MAX:
        raise OpticsRefused(f"{what}: sigma_alpha {s.value} outside [{SIGMA_MIN}, {SIGMA_MAX}]")
    return SurfaceEntry(legacy.UNIFIED_METAL, {"R": refl, "p_spike": fraction(pr, "SPECULARSPIKECONSTANT", 0.0),
                                               "p_lobe": fraction(pr, "SPECULARLOBECONSTANT", 0.0),
                                               "p_back": fraction(pr, "BACKSCATTERCONSTANT", 0.0),
                                               "sigma_alpha": float(s.value)})  # fmt: skip


def _dielectric(s: schema.Surface, finish: str, what: str, refl, transmits: bool):
    # Geant4 reads T at its default 0 when the surface gives none (G4OpBoundaryProcess.cc:310, 463): a parameter
    t = flat_fraction(s, "TRANSMITTANCE", 0.0)
    if finish in ("polishedfrontpainted", "groundfrontpainted"):
        fs = 1.0 if finish == "polishedfrontpainted" else 0.0
        if transmits:  # into a medium without RINDEX the transmission is absorbed (lumen.rules.boundary)
            return SurfaceEntry(legacy.PAINTED_T, {"R": refl, "T": t, "f_spec": fs})
        if "TRANSMITTANCE" in s.properties:
            raise OpticsRefused(f"{what}: TRANSMITTANCE on a WLS film's backing (no transmitting backing)")
        return SurfaceEntry(legacy.PAINTED, {"R": refl, "f_spec": fs})
    if finish == "polished":  # absorption (detected with EFFICIENCY on a sensor) and transmission before Fresnel
        return SurfaceEntry(legacy.DIELECTRIC_RT, {"R": refl, "T": t}, ignored_between_same_material=True)
    raise OpticsRefused(f"{what}: ground dielectric_dielectric (facet refraction) is not supported")


# Geant4's value of a surface property the model reads when the surface omits it (G4OpBoundaryProcess.cc, 11.2.2)
DEFAULT_R = (1.0, "G4OpBoundaryProcess.cc:308")
DEFAULT_T = (0.0, "G4OpBoundaryProcess.cc:310")
DEFAULT_UNIFIED = {"SPECULARLOBECONSTANT": (0.0, "G4OpBoundaryProcess.cc:405-407"),
                   "SPECULARSPIKECONSTANT": (0.0, "G4OpBoundaryProcess.cc:408-410"),
                   "BACKSCATTERCONSTANT": (0.0, "G4OpBoundaryProcess.cc:411-413")}  # fmt: skip


def surface_defaults(s: schema.Surface, backing: bool = False) -> dict:
    """Return {KEY: (Geant4's default, its source line)} of the properties ``s`` omits that its model reads.

    dielectric_dielectric: REFLECTIVITY and TRANSMITTANCE (not on a WLS film's ``backing``);
    dielectric_metal without a complex index: REFLECTIVITY, and with unified and a ground finish the
    three unified constants. EFFICIENCY is not offered: a nonzero one makes a surface a sensor, whose efficiency is its
    sensor model's. A surface another optics model reads has none.
    """
    if s.optics.name != "geant4":
        return {}
    model, finish, type_ = surface_names(s)
    pr, out = s.properties, {}
    if type_ == "dielectric_dielectric" and finish in ("polished", "polishedfrontpainted", "groundfrontpainted"):
        out = {"REFLECTIVITY": DEFAULT_R} | ({} if backing else {"TRANSMITTANCE": DEFAULT_T})
    elif type_ == "dielectric_metal" and not ("REALRINDEX" in pr and "IMAGINARYRINDEX" in pr):
        out = {"REFLECTIVITY": DEFAULT_R} | (DEFAULT_UNIFIED if (model, finish) == ("unified", "ground") else {})
    return {k: v for k, v in out.items() if k not in pr}


def detector_defaults(df: schema.DetectorFile) -> dict:
    """Return {surface: surface_defaults(...)} of every surface of a detector file (a WLS film's backing marked)."""
    backings = {s.optics.settings.get("backing") for s in df.surfaces.values() if s.optics.name == "lumen:wls_film"}
    return {n: d for n, s in df.surfaces.items() if (d := surface_defaults(s, n in backings))}


def surface(s: schema.Surface, transmits: bool = True) -> SurfaceEntry:
    """Return the SurfaceEntry of an optical surface with Geant4 11 semantics (see the module docstring).

    ``transmits=False``: a WLS film's backing (frontpainted only), whose T stays at Geant4's 0 (``wls_film``).
    """
    model, finish, type_ = surface_names(s)
    what = f"opticalsurface {s.name!r} ({model} {finish} {type_})"
    pr = s.properties
    for k in REFUSED_SURFACE:
        if k in pr:
            raise OpticsRefused(f"{what}: {k} is not supported")
    rt = type_ == "dielectric_dielectric" and finish == "polished"
    if nonzero(pr, "EFFICIENCY") and not rt:
        raise OpticsRefused(f"{what}: EFFICIENCY > 0 makes it a detecting surface; bind a sensor instead")
    if model not in ("glisur", "unified"):
        raise OpticsRefused(f"{what}: model {model} is not supported")
    if "backpainted" in finish or finish not in SUPPORTED_FINISHES:
        raise OpticsRefused(f"{what}: finish {finish} is not supported")
    refl = fraction(pr, "REFLECTIVITY", 1.0)
    if type_ == "dielectric_metal":
        return _metal(s, model, finish, what, refl)
    if type_ == "dielectric_dielectric":
        return _dielectric(s, finish, what, refl, transmits)
    raise OpticsRefused(f"{what}: surface type {type_} is not supported")


# ----------------------------------------------------------------------------------------------- detecting surfaces
def at_centre(surface: schema.Surface, key: str, lam_domain) -> float:
    """Return a property at the range's centre (Geant4: linear in energy, clamped), refusing one not flat over it."""
    pr = surface.properties[key]
    if not is_table(pr):
        return float(pr.value)
    e = HC / np.asarray(lam_domain, np.float64)
    v = np.interp(np.r_[e, 0.5 * (e[0] + e[1])], pr.energy, pr.value)
    c = float(np.interp(HC / (0.5 * (lam_domain[0] + lam_domain[1])), pr.energy, pr.value))
    if np.ptp(np.r_[v, c]) > FLAT * max(abs(c), TINY):
        raise OpticsRefused(f"opticalsurface {surface.name!r}: {key} varies by {np.ptp(np.r_[v, c]):.3g} over "
                            f"{tuple(lam_domain)} nm: a complex index is taken at one wavelength (narrow the range or "
                            "supply a spectral reader)")  # fmt: skip
    return c


def flat_fraction(surface: schema.Surface, key: str, default: float):
    """Return a fraction property: a constant, a flat table as its constant, else the spectral (law, parameters)."""
    pr = surface.properties.get(key)
    if pr is not None and is_table(pr) and np.ptp(np.asarray(pr.value)) == 0.0:
        return float(np.asarray(pr.value)[0])
    return fraction(surface.properties, key, default)


def _mirror(refl, eff):
    return SensorEntry(legacy.SENSOR, {"R": refl, "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 0.0, "qe": eff})


def _ground_metal_sensor(s: schema.Surface, model: str, what: str, refl, eff):
    if model == "glisur":
        if s.value >= 1.0:
            return _mirror(refl, eff)
        if s.value < 0.0 or 1.0 - s.value < GLISUR_R_MIN:
            raise OpticsRefused(f"{what}: polish {s.value} outside [0, {1 - GLISUR_R_MIN}] (or exactly 1)")
        return SensorEntry(legacy.GLISUR_METAL_EFF, {"R": refl, "polish": float(s.value), "eff": eff})
    if not SIGMA_MIN <= s.value <= SIGMA_MAX:
        raise OpticsRefused(f"{what}: sigma_alpha {s.value} outside [{SIGMA_MIN}, {SIGMA_MAX}]")
    return SensorEntry(legacy.DIELECTRIC_METAL, {"R": refl, "p_spike": flat_fraction(s, "SPECULARSPIKECONSTANT", 0.0),
                                                 "p_lobe": flat_fraction(s, "SPECULARLOBECONSTANT", 0.0),
                                                 "p_back": flat_fraction(s, "BACKSCATTERCONSTANT", 0.0),
                                                 "sigma_alpha": float(s.value), "eff": eff})  # fmt: skip


def geant4_sensor(s: schema.Surface, lam_domain) -> SensorEntry:
    """Return stock Geant4's reading of a detecting surface (``lumen:geant4``, see the module docstring)."""
    pr = s.properties
    model, finish, type_ = surface_names(s)
    what = f"opticalsurface {s.name!r} ({model} {finish} {type_})"
    eff = flat_fraction(s, "EFFICIENCY", 0.0)
    if type_ == "dielectric_dielectric":
        e = surface(s)
        if e.surface_type is not legacy.DIELECTRIC_RT:
            raise OpticsRefused(f"{what}: {e.surface_type.name} does not detect (Geant4: only a polished "
                                "dielectric_dielectric reaches DoAbsorption)")  # fmt: skip
        return SensorEntry(legacy.DIELECTRIC_RT_SENSOR, dict(e.params, eff=eff))
    if type_ != "dielectric_metal":
        raise OpticsRefused(f"{what}: surface type {type_} is not supported as a sensor")
    if model not in ("glisur", "unified") or finish not in ("polished", "ground"):
        raise OpticsRefused(f"{what}: model / finish not supported as a sensor")
    if nonzero(pr, "TRANSMITTANCE"):
        raise OpticsRefused(f"{what}: TRANSMITTANCE on a metal would put the photon into it")
    if "REFLECTIVITY" not in pr and "REALRINDEX" in pr and "IMAGINARYRINDEX" in pr:
        if finish != "polished":
            raise OpticsRefused(f"{what}: a complex index on a ground finish (the facet's Fresnel) is not supported")
        return SensorEntry(legacy.PHOTOCATHODE, {"nr": at_centre(s, "REALRINDEX", lam_domain),
                                                 "nk": at_centre(s, "IMAGINARYRINDEX", lam_domain), "qe": eff,
                                                 "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 0.0})  # fmt: skip
    refl = flat_fraction(s, "REFLECTIVITY", 1.0)
    if finish == "polished":
        return _mirror(refl, eff)
    return _ground_metal_sensor(s, model, what, refl, eff)


def complex_index_sensor(s: schema.Surface, lam_domain) -> SensorEntry:
    """Return RINDEX + i KINDEX on the surface as a semi-infinite absorber (``lumen:complex_index``)."""
    if "RINDEX" not in s.properties or "KINDEX" not in s.properties:
        raise OpticsRefused(f"opticalsurface {s.name!r}: complex_index needs RINDEX and KINDEX on the surface")
    return SensorEntry(legacy.PHOTOCATHODE, {"nr": at_centre(s, "RINDEX", lam_domain),
                                             "nk": at_centre(s, "KINDEX", lam_domain),
                                             "qe": flat_fraction(s, "EFFICIENCY", 1.0), "f_spec": 1.0,
                                             "f_glossy": 0.0, "sigma_alpha": 0.0})  # fmt: skip


SENSOR_SURFACES = {"lumen:geant4": geant4_sensor, "lumen:complex_index": complex_index_sensor}
DEFAULT_SENSOR_SURFACE = "lumen:geant4"


def sensor_surface(model=None):
    """Return the reader of a detecting surface named ``model`` (None: the default; a bare name is lumen's own)."""
    if callable(model):
        return model
    name = DEFAULT_SENSOR_SURFACE if model is None else str(model)
    for candidate in (name,) if ":" in name else (name, f"lumen:{name}"):
        if candidate in SENSOR_SURFACES:
            return SENSOR_SURFACES[candidate]
    raise OpticsRefused(f"detecting surface reader {name!r}: not one of {sorted(SENSOR_SURFACES)}")


def sensor(s: schema.Surface, lam_domain=None, model=None, eff: float | None = None) -> SensorEntry:
    """Return the SensorEntry of a detecting optical surface, read by ``sensor_surface(model)``.

    ``model`` names the reader (default "lumen:geant4": the surface as stock Geant4 reads it), or is one. ``eff``
    replaces the reading's detection efficiency (a sensitive detector that counts every absorption, WCSim's, passes 1).
    """
    e = sensor_surface(model)(s, lam_domain)
    if not isinstance(e, SensorEntry) or DETECT not in e.surface_type.outcomes:
        raise OpticsRefused(f"opticalsurface {s.name!r}: reader {model!r} returned no detecting SensorEntry")
    if eff is not None:
        k = "eff" if "eff" in e.params else "qe"
        e = SensorEntry(e.surface_type, dict(e.params, **{k: float(eff)}))
    return e


def rt_sensor(s: schema.Surface, eff: float | None = None) -> SensorEntry:
    """Return the sensor form of a polished dielectric_dielectric surface (DIELECTRIC_RT_SENSOR).

    Its absorbed share is detected with ``eff``: by default the surface's EFFICIENCY (Geant4's Detection status); a
    sensitive detector that counts every absorption (WCSim's) passes 1.
    """
    e = sensor(s, eff=eff)
    if e.surface_type is not legacy.DIELECTRIC_RT_SENSOR:
        raise OpticsRefused(f"opticalsurface {s.name!r}: {e.surface_type.name}, not a polished dielectric_dielectric")
    return e


# -------------------------------------------------------------------------------------------------- scintillation
def scintillation(material: schema.Material, geant4: str = "11") -> dict:
    """Return a material's scintillation: spectra, yield (per MeV), time constants (ns) and fast fraction.

    ``geant4="11"`` reads the Geant4 11 keys (SCINTILLATIONCOMPONENT1/2, SCINTILLATIONTIMECONSTANT1/2,
    SCINTILLATIONYIELD1/2 for the fast fraction) and refuses a material without them (Geant4 11 then emits no
    scintillation); ``geant4="10.7"`` reads FASTCOMPONENT, SLOWCOMPONENT, FASTTIMECONSTANT, SLOWTIMECONSTANT and
    YIELDRATIO (the fast fraction of the yield, as G4Scintillation 10.7 reads it).
    """
    pr = material.properties
    if geant4 == "11":
        keys = ("SCINTILLATIONCOMPONENT1", "SCINTILLATIONCOMPONENT2", "SCINTILLATIONTIMECONSTANT1",
                "SCINTILLATIONTIMECONSTANT2")  # fmt: skip
    elif geant4 == "10.7":
        keys = ("FASTCOMPONENT", "SLOWCOMPONENT", "FASTTIMECONSTANT", "SLOWTIMECONSTANT")
    else:
        raise OpticsRefused(f"geant4 must be '11' or '10.7', got {geant4!r}")
    miss = [k for k in keys + ("SCINTILLATIONYIELD",) if k not in pr]
    if miss:
        found = sorted(k for k in pr if k in G4_10_KEYS)
        hint = (f" (it has the Geant4-10 keys {found}: inert under Geant4 >= 11, where G4Scintillation then emits "
                "nothing)") if found and geant4 == "11" else ""  # fmt: skip
        raise OpticsRefused(f"material {material.name!r}: no {miss} for Geant4 {geant4}{hint}")
    cf, cs, tf, ts = (pr[k] for k in keys)
    if geant4 == "11":
        y1 = float(pr["SCINTILLATIONYIELD1"].value) if "SCINTILLATIONYIELD1" in pr else 1.0
        y2 = float(pr["SCINTILLATIONYIELD2"].value) if "SCINTILLATIONYIELD2" in pr else 0.0
        f_fast = y1 / (y1 + y2)
    else:
        f_fast = float(pr["YIELDRATIO"].value) if "YIELDRATIO" in pr else 1.0
    same = bool(np.array_equal(cf.energy, cs.energy) and np.array_equal(cf.value, cs.value))
    return dict(spectrum=spectra.geant4_spectrum(f"{material.name}.fast", cf.energy, cf.value),
                spectrum_slow=spectra.geant4_spectrum(f"{material.name}.slow", cs.energy, cs.value),
                same_spectra=same, yield_per_MeV=float(pr["SCINTILLATIONYIELD"].value), tau_fast=float(tf.value),
                tau_slow=float(ts.value), f_fast=f_fast)  # fmt: skip


# ------------------------------------------------------------------------------------------------------ WLS films
def _check_film(props: dict, front: dict, what: str):
    if (
        "RINDEX" not in props
        or "RINDEX" not in front
        or not (
            np.array_equal(props["RINDEX"].energy, front["RINDEX"].energy)
            and np.array_equal(props["RINDEX"].value, front["RINDEX"].value)
        )
    ):
        raise OpticsRefused(f"{what} is not index-matched to the medium in front (equal RINDEX tables): Geant4 would "
                            "refract / reflect at the film, which a zero-thickness film omits")  # fmt: skip
    for k in ("WLSABSLENGTH", "WLSCOMPONENT"):
        if k not in props:
            raise OpticsRefused(f"{what}: no {k}")
    for k in ("WLSABSLENGTH2", "WLSCOMPONENT2", "ABSLENGTH", "RAYLEIGH", "MIEHG"):
        if k in props:
            raise OpticsRefused(f"{what}: {k} is not lowered onto a zero-thickness film")
    tau = float(props["WLSTIMECONSTANT"].value) if "WLSTIMECONSTANT" in props else 0.0
    if tau != 0.0:
        raise OpticsRefused(f"{what}: a nonzero WLSTIMECONSTANT needs the time profile (G4OpWLS delta / exponential), "
                            "not lowered yet")  # fmt: skip
    return tau


def _film_yield(props: dict, what: str) -> float:
    if "WLSMEANNUMBERPHOTONS" not in props:  # G4OpWLS: Poisson(mean) photons per absorption, 1 without it
        return 1.0
    mean = props["WLSMEANNUMBERPHOTONS"]
    if is_table(mean) or not float(mean.value) >= 0.0:
        raise OpticsRefused(f"{what}: WLSMEANNUMBERPHOTONS must be a constant >= 0")
    return float(mean.value)


def wls_film(film: schema.Material, front: schema.Material, backing: schema.Surface, d: float, axis, types=None):
    """Return a WLS film of thickness ``d`` (m) on one face (normal ``axis``) of a surface, as one coated surface.

    The film material carries WLSABSLENGTH, WLSCOMPONENT and WLSTIMECONSTANT; ``backing`` is the surface behind it.
    It is lowered to ``surfaces.types.coated`` over the backing's own surface: every number comes from the records,
    nothing is averaged, every one is a live parameter (the film's absorption length table, thickness, yield and
    delay; the backing's tables). The yield eta is the film's WLSMEANNUMBERPHOTONS (Geant4 emits Poisson(eta)
    photons), 1 when absent as in Geant4.

    Refused: a film not index-matched to ``front`` (its RINDEX table equal to the front material's), one that absorbs
    its own emission (Geant4 would re-absorb it at a depth the lowering does not track), and any other optics on it.
    ``types``: a dict shared across calls, one coated surface type per (film, backing type).
    """
    props = film.properties
    what = f"WLS film {film.name!r}"
    tau = _check_film(props, front.properties, what)
    wle, wla = props["WLSCOMPONENT"], props["WLSABSLENGTH"]
    # the emission band: the support of Geant4's sampler (knot intervals of positive trapezoid mass)
    pos = np.flatnonzero(0.5 * (wle.value[1:] + wle.value[:-1]) * np.diff(wle.energy) > 0)
    if not len(pos):
        raise OpticsRefused(f"{what}: WLSCOMPONENT has no emission")
    e_lo, e_hi = float(wle.energy[pos[0]]), float(wle.energy[pos[-1] + 1])
    grid = np.linspace(e_lo, e_hi, FILM_GRID)
    worst = float(np.max(d / np.interp(grid, wla.energy, wla.value)))
    if worst > FILM_SELF_ABSORPTION:
        raise OpticsRefused(f"{what}: absorbs its own emission (d / L up to {worst:.1e})")
    b = surface(backing, transmits=False)
    types = {} if types is None else types
    key = (film.name, b.surface_type.name)
    if key not in types:
        spectrum = spectra.geant4_spectrum(film.name, wle.energy, wle.value)
        types[key] = legacy.coated(f"coated:{film.name}:{b.surface_type.name}", b.surface_type, spectrum)
    params = {"L": table(wla, "length"), "d": float(d), "eta": _film_yield(props, what), "tau": tau,
              "axis": np.asarray(axis, np.float64)}  # fmt: skip
    params.update({f"b:{k}": v for k, v in b.params.items()})
    return SurfaceEntry(types[key], params)
