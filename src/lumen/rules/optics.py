"""The "geant4" optics model: a detector file's material and surface properties read as Geant4 reads them.

    entry = medium(df.materials["LAr"])                 # a MediumEntry (opaque without RINDEX)
    entry = surface(df.surfaces["Surface_Steel_1"])     # a SurfaceEntry, Geant4 11 semantics, fail closed
    entry = sensor(df.surfaces["GlassCathode"])         # a SensorEntry: how a detecting surface reads

The records are ``lumen.format.schema`` Material and Surface records: energies in eV, values in the unit
``schema.PROPERTY_UNITS`` gives each key (lengths in m, times in ns, GROUPVEL in m / ns, ISOTHERMAL_COMPRESSIBILITY in
m3 / MeV, yields per MeV). Geant4 semantics are those of G4OpBoundaryProcess, G4OpWLS and G4OpRayleigh in
Geant4 11.3.0. Geant4 10.7.4 differs on one path read here: there a dielectric_dielectric surface with TRANSMITTANCE
replaces the Fresnel transmission coefficient with it (G4OpBoundaryProcess::DielectricDielectric,
"if(theTransmittance > 0.) TransCoeff = theTransmittance"), so TRANSMITTANCE acts twice; Geant4 11 removed that and
keeps the Fresnel ratio, which lumen follows.

Tables are linear in photon ENERGY between their knots and clamped to the end values outside
(G4PhysicsVector::Value): ``laws.tabulated``, one shared law per output quantity. RINDEX is ``laws.geant4_rindex``: the
same n(E), whose group index is Geant4's GROUPVEL.

Media (``medium``; the channels and their keys: docs/dev/architecture.md, "Bulk processes and sources"): no RINDEX is
opaque (Geant4's NoRINDEX kill); a material without RAYLEIGH gets Geant4's own table where G4OpRayleigh computes one
(``geant4_rayleigh``: a material named Water, or one that gives ISOTHERMAL_COMPRESSIBILITY).
Wavelength shifting is G4OpWLS's (``lumen.media.emission``): the emission drawn at or below the absorbed energy,
WLSMEANNUMBERPHOTONS the expected yield (1 when absent), the delay WLSTIMECONSTANT under the time profile the material's
optics setting ``wls_time_profile`` records ("delta", Geant4's default, when absent; or "exponential"). Refused: WLS2, a
complex index, a Mie constant Geant4 would read but the material lacks. A GROUPVEL table times the photons, as in
Geant4 (``user_groupvel``); without one, RINDEX's group velocity.

Surfaces (G4OpBoundaryProcess::PostStepDoIt, DielectricMetal, DoReflection; no table: R = 1, T = 0, efficiency 0):

    dielectric_metal, glisur, ground, polish < 1        GLISUR_METAL: R facet path (r = 1 - polish)
    dielectric_metal, glisur (other finish, polish 1)   SPECULAR_METAL: R mirror
    dielectric_metal, unified, polished                 SPECULAR_METAL
    dielectric_metal, unified, ground                   UNIFIED_METAL: spike / lobe / back / Lambert, Geant4's cuts
    dielectric_metal, unified, ground, sigma_alpha 0    UNIFIED_METAL_NORMAL: the lobe is the spike (GetFacetNormal
                                                        returns the normal)
    dielectric_metal, polished, complex index           PHOTOCATHODE: REALRINDEX + i IMAGINARYRINDEX, read at the
                                                        photon's wavelength (CalculateReflectivity), no REFLECTIVITY
    dielectric_metal with TRANSMITTANCE                 the type above, transmitting T straight on (``transmitting``)
    dielectric_dielectric, polishedfrontpainted         PAINTED_T (f_spec 1): R mirror, T straight, rest absorbed
                                                        (DoAbsorption: detected with EFFICIENCY on a sensor)
    dielectric_dielectric, groundfrontpainted           PAINTED_T (f_spec 0): R Lambert about the normal
    dielectric_dielectric, glisur, ground, polish < 1   GLISUR_DIELECTRIC: absorbed 1 - R - T, T straight, R the GLISUR
                                                        facet path, each facet reflecting or refracting (Fresnel, the
                                                        photon's state), ending on either side; skipped between one
                                                        material; at polish 1 the polished reading
    dielectric_dielectric, polished                     DIELECTRIC_RT: absorbed 1 - R - T, T straight, R to Fresnel,
                                                        skipped between one material

The type does not depend on whether R or T is present on a polished surface: Geant4 reads them at their defaults (R
1, T 0) when the surface omits them (``surface_defaults``), so an omitted one is a parameter at that default
(docs/explanation/parameters.md); so is a frontpainted surface's T (a transmission into a medium without RINDEX is
absorbed where it lands: ``lumen.rules.boundary``).

Refused (the messages: docs/reference/errors.md): LUT, DAVIS, dichroic, backpainted, unified ground
dielectric_dielectric, a complex index on a ground finish, SURFACEROUGHNESS, coated keys, sigma_alpha outside
[SIGMA_MIN, SIGMA_MAX] other than 0, 1 - polish below GLISUR_R_MIN.

Detecting surfaces (``sensor``): how a sensor's optical surface behaves during tracking (not the sensor model,
``lumen.sensors``: that is what happens after tracking). A surface is a sensor where a sensor group binds it
(``lumen.build.library``), never by its values. It is read as stock Geant4 (G4OpBoundaryProcess, 10.7 / 11) reads it
(``geant4_sensor``): the one reading of a surface (``_reading``: the same refusals, types, far side and same-material
skip) with eff = EFFICIENCY instead of 0 (DoAbsorption: every dielectric_metal, and the polished and frontpainted
dielectric_dielectric finishes). Geant4's order holds for both (G4OpBoundaryProcess.cc, 10.7.4): a polished or ground
dielectric_dielectric between two volumes of one material returns SameMaterial (:406-416) before any absorption or
detection (:442-446), whatever its EFFICIENCY; into a material without RINDEX it is NoRINDEX (:417-434), killed
before the absorption draw. Keys no
stock process reads on a surface (RINDEX, KINDEX, THICKNESS on a dielectric_metal) are ignored, as Geant4 ignores
them.

A complex index is read at the photon's energy (linear in energy, clamped), as G4OpBoundaryProcess::
CalculateReflectivity reads REALRINDEX and IMAGINARYRINDEX (G4OpBoundaryProcess.cc:1393-1395, 10.7.4). A sensitive
detector that
counts every absorption (WCSim's) replaces the efficiency: ``sensor(..., eff=1)``.

Spectral surface parameters (REFLECTIVITY(E), ...) are (law, law parameters) pairs; the surface lookup
(``lumen.build.lookup.compile``) packs them.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core.safe import traced
from lumen.format import schema
from lumen.format.apply import ValueRules
from lumen.media import bulk, emission, laws, phase, spectra
from lumen.rules.boundary import FAR_ABSORB, FAR_KILL
from lumen.rules.choices import GLISUR_R_MIN, MIE_G_MAX, SIGMA_MAX, SIGMA_MIN
from lumen.rules.entries import DescriptionError, MediumEntry, SensorEntry, SurfaceEntry
from lumen.rules.keys import G4_10_KEYS, GEANT4_KEYS, SCINTILLATION_KEYS, SURFACE_KEYS, components  # declared once
from lumen.surfaces import types
from lumen.units import C_VACUUM, HC_EV_M, TWO_PI

# G4OpRayleigh::CalculateRayleighMeanFreePaths (10.7.4 and 11.2, G4OpRayleigh.cc): a material named "Water" gets these
# whatever its own ISOTHERMAL_COMPRESSIBILITY and temperature ("Temperature of water is 10 degrees celsius")
WATER = "Water"
WATER_COMPRESSIBILITY = 7.658e-23  # m3 / MeV
WATER_TEMPERATURE = 283.15  # K
K_BOLTZMANN = 8.617333e-11  # MeV / K: CLHEP's k_Boltzmann (PhysicalConstants.h), as G4OpRayleigh reads it
NTP_TEMPERATURE = 293.15  # K: G4Material's default temperature (NTP_Temperature), G4GDMLReadMaterials' too

# the material keys lumen reads by default (Recipe.material_keys None): every key a Geant4 optical process or lumen's
# rules read, implemented or refused; any other key is a user key no Geant4 optical process reads, kept inert
MATERIAL_KEYS = GEANT4_KEYS | {"ISOTHERMAL_COMPRESSIBILITY", "RS_SCALE_FACTOR"}

REFUSED_MEDIUM = {"WLSCOMPONENT2": "a second WLS (G4OpWLS2)",
                  "WLSABSLENGTH2": "a second WLS (G4OpWLS2)", "WLSMEANNUMBERPHOTONS2": "a second WLS (G4OpWLS2)",
                  "REALRINDEX": "a complex index",
                  "IMAGINARYRINDEX": "a complex index", "COATEDRINDEX": "a coated surface property"}  # fmt: skip
REFUSED_SURFACE = ("SURFACEROUGHNESS", "COATEDRINDEX", "COATEDTHICKNESS")
SUPPORTED_FINISHES = ("polished", "ground", "polishedfrontpainted", "groundfrontpainted")
MIE_KEYS = ("MIEHG_FORWARD", "MIEHG_BACKWARD", "MIEHG_FORWARD_RATIO")  # the lobe parameters of phase.MIE_PARAMS


class OpticsRefused(DescriptionError):  # noqa: N818 (named like the importer's ImportRefused)
    """A property or surface Geant4 reads in a way lumen does not reproduce: refused, never approximated."""


# ------------------------------------------------------------------------------------------------- the shared laws
TABLE_LAWS = {out: laws.tabulated(out) for out in laws.RANGES}
GEANT4_RINDEX = laws.geant4_rindex()
GEANT4_RINDEX_GROUPVEL = laws.geant4_rindex_groupvel()


def is_table(p: schema.Property) -> bool:
    """Return whether a property is a table (energies and values) rather than a constant."""
    return p.energy is not None


def table(p: schema.Property, out: str):
    """Return the shared clamped law of output ``out`` and the parameters of a table property."""
    return TABLE_LAWS[out], {"energy": np.asarray(p.energy, np.float64), "value": np.asarray(p.value, np.float64)}


def index(p: schema.Property):
    """Return a complex index's real part: its (law, parameters) for a table, its value for a constant."""
    return table(p, "index") if is_table(p) else float(p.value)


def extinction(p: schema.Property):
    """Return a complex index's imaginary part (>= 0: an absorbing medium's; 0 a dielectric): its (law, parameters)
    for a table, its value for a constant. A negative value (a medium that amplifies) is refused."""  # noqa: D205, D209
    if np.any(np.asarray(p.value, np.float64) < 0.0):
        raise OpticsRefused(f"{p.key}: an extinction index is >= 0 (an absorbing medium), got {np.min(p.value):g}")
    return table(p, "extinction") if is_table(p) else float(p.value)


def fraction(props: dict, key: str, default: float):
    """Return a fraction property: its (law, parameters) for a table, its value for a constant, else ``default``."""
    if key not in props:
        return default
    p = props[key]
    return table(p, "fraction") if is_table(p) else float(p.value)


def value_key(s: schema.Surface) -> str:
    """Return the parameter name of an opticalsurface's ``value``: glisur's ``polish``, else unified's ``sigma_alpha``.

    G4OpBoundaryProcess::GetFacetNormal reads it as the polish (glisur) or as sigma_alpha in rad (unified).
    """
    return "polish" if s.model == "glisur" else "sigma_alpha"


def reads_value(s: schema.Surface) -> bool:
    """Return whether Geant4's reading of a surface reads its opticalsurface value (``value_key``).

    Structure decides: the reading's entry has that parameter, which only a ground finish's facet path has
    (dielectric_metal: unified's sigma_alpha, glisur's polish below 1; dielectric_dielectric: glisur's polish below 1;
    at polish 1 the reading is a mirror or the polished interface and reads no value). A surface reads it whether or
    not it detects (one reading); one the reader refuses, or another optics model's, reads none.
    """
    if s.optics.name != "geant4":
        return False
    finish, type_ = s.finish, s.type
    if type_ not in ("dielectric_metal", "dielectric_dielectric") or finish != "ground":  # no facet sampled
        return False
    try:
        params = _reading(s)[1]
    except OpticsRefused:
        return False
    return value_key(s) in params


# ---------------------------------------------------------------------------------------------------------- media
def rayleigh_constant(material: schema.Material) -> float | None:
    """Return c1 = s beta_T T k_B / (6 pi) (m3) of the Rayleigh table Geant4 computes for a material without RAYLEIGH.

    None where it computes none (``geant4_rayleigh``).
    """
    props = material.properties
    if "RINDEX" not in props or not is_table(props["RINDEX"]):
        return None
    if material.name.split("0x")[0] == WATER:
        beta, T = WATER_COMPRESSIBILITY, WATER_TEMPERATURE
    elif "ISOTHERMAL_COMPRESSIBILITY" in props:
        beta = float(props["ISOTHERMAL_COMPRESSIBILITY"].value)
        T = NTP_TEMPERATURE if material.temperature_k is None else float(material.temperature_k)
    else:
        return None
    scale = float(props["RS_SCALE_FACTOR"].value) if "RS_SCALE_FACTOR" in props else 1.0
    return scale * beta * T * K_BOLTZMANN / (6.0 * np.pi)


def rayleigh_length(energy, rindex, c1):
    """Return the Einstein-Smoluchowski Rayleigh length (m) at ``energy`` (eV) of a medium of index ``rindex``.

    In JAX (float32, the engine's precision): one function for the table built at load and the one a traced RINDEX
    packs.

    Notes
    -----
    G4OpRayleigh::CalculateRayleighMeanFreePaths (10.4+): 1 / L = c1 (2 pi / lambda)^4 ((n^2 - 1)(n^2 + 2) / 3)^2
    with c1 = s beta_T T k_B / (6 pi) (``rayleigh_constant``).
    Notation: L the length, lambda the wavelength.
    """
    return _rayleigh_length(jnp.asarray(energy, jnp.float32), jnp.asarray(rindex, jnp.float32), jnp.float32(c1))


def _rayleigh_length_f32(e, n, c1):
    n2 = n * n
    k = jnp.float32(TWO_PI / HC_EV_M) * e  # 2 pi / lambda, 1 / m
    return 1.0 / (c1 * k**4 * ((n2 - 1.0) * (n2 + 2.0) / 3.0) ** 2)


# compiled as its own program: the table at load and a traced RINDEX's are the same compiled arithmetic, bit for bit
_rayleigh_length = jax.jit(_rayleigh_length_f32)


def geant4_rayleigh(material: schema.Material) -> schema.Property | None:
    """Return the Rayleigh table Geant4 computes for a material without RAYLEIGH, or None (it scatters none).

    G4OpRayleigh::BuildPhysicsTable reads a material's RAYLEIGH table (the caller's first choice) and otherwise
    computes one (CalculateRayleighMeanFreePaths, ``rayleigh_length``) on the RINDEX knots, from RINDEX and:

    - a material named "Water" (the name Geant4 holds: a GDML name without its pointer suffix, as G4GDMLRead strips
      it): Geant4's compressibility of water at 283.15 K (``WATER_COMPRESSIBILITY``, ``WATER_TEMPERATURE``),
      whatever the material's own ISOTHERMAL_COMPRESSIBILITY and temperature;
    - otherwise its ISOTHERMAL_COMPRESSIBILITY (m3 / MeV) at its temperature (293.15 K when it gives none);
    - otherwise none;

    times RS_SCALE_FACTOR (``rayleigh_constant``). The packed table is derived from RINDEX in JAX by the same function
    (``lumen.build.pack``), so a RINDEX value, traced or concrete, moves it as it moves Geant4's.
    """
    c1 = rayleigh_constant(material)
    if c1 is None:
        return None
    r = material.properties["RINDEX"]
    e = np.asarray(r.energy, np.float64)
    L = np.asarray(rayleigh_length(e, r.value, c1), np.float64)
    return schema.Property("RAYLEIGH", L, energy=e, origin="rule", note="Geant4's own Rayleigh table (G4OpRayleigh)")


def _check_medium(props: dict, what: str, settings=None):
    old = sorted(k for k in props if k in G4_10_KEYS)
    if old:  # translated by an import rule at every conversion (lumen.build.import_rules): an older lumen's file
        raise OpticsRefused(f"{what}: Geant4-10 scintillation keys {old} in a detector file: lumen translates them "
                            "when a detector is converted (lumen.build.import_rules), so this file was written by an "
                            "older lumen; convert it again from source")  # fmt: skip
    for k, why in REFUSED_MEDIUM.items():
        if k in props:
            raise OpticsRefused(f"{what}: {k} ({why}) is not supported")
    _check_group_velocity(props, what)
    _check_rise_times(props, what, bool((settings or {}).get("scint_finite_rise_time", False)))


def _check_rise_times(props: dict, what: str, finite: bool):
    """Refuse a scintillation rise time Geant4's switch would ignore, and a missing one it would read.

    G4Scintillation reads SCINTILLATIONRISETIMEi only under the finite-rise-time switch (Geant4 11:
    G4OpticalParameters::SetScintFiniteRiseTime; 10.7: G4Scintillation::SetFiniteRiseTime), recorded as the material's
    optics setting ``scint_finite_rise_time`` (absent: Geant4's default, off). A rise time without the switch is never
    silently ignored: it is refused (set the switch as the Geant4 run does, or drop the key). Under the switch Geant4
    reads each emitted component's rise time and aborts without it (GetConstProperty).
    """
    comps = components(lambda k: k in props)
    rises = SCINTILLATION_KEYS["rises"]
    if not finite:
        given = [k for k in rises if k in props]
        if given:
            raise OpticsRefused(f"{what}: {given} without the optics setting scint_finite_rise_time: Geant4 reads a "
                                "rise time only under its finite-rise-time switch; set it as the Geant4 run does, or "
                                "drop the key")  # fmt: skip
        return
    missing = [rises[i] for i in comps if rises[i] not in props]
    if missing:
        raise OpticsRefused(f"{what}: scint_finite_rise_time without {missing}: G4Scintillation reads each emitted "
                            "component's rise time and aborts without it")  # fmt: skip


def _check_group_velocity(props: dict, what: str):
    """Refuse a user GROUPVEL (table or constant) outside (0, c], ``c`` lumen's exact ``units.C_VACUUM``: one written in
    mm / ns (Geant4's internal unit) is 1000 times too fast. A speed derived from RINDEX is Geant4's, never refused
    (``laws.geant4_group_index``: G4MaterialPropertiesTable::CalculateGROUPVEL replaces a v_g < 0 or > c / n
    by c / n)."""  # noqa: D205, D209
    if "GROUPVEL" in props and not traced(props["GROUPVEL"].value):  # a traced value is checked where concrete
        v = np.asarray(props["GROUPVEL"].value, np.float64)
        if not (np.all(v > 0.0) and np.all(v <= C_VACUUM)):
            raise OpticsRefused(f"{what}: GROUPVEL must hold speeds in (0, c] in m / ns (c = {C_VACUUM} m / ns), "
                                f"got {v.min():g} to {v.max():g}: in mm / ns it is 1000 times too fast")  # fmt: skip


def user_groupvel(props: dict) -> bool:
    """Return whether a material's GROUPVEL table times its photons: a table, with RINDEX (a parameter, as any table).

    Geant4 times photons with the material's GROUPVEL (``laws.geant4_rindex_groupvel``), the one in the file when it
    has one; without one, the group velocity of RINDEX (G4MaterialPropertiesTable::CalculateGROUPVEL). A constant
    GROUPVEL is inert, as in Geant4 (G4Track reads the property vector).
    """
    return "GROUPVEL" in props and is_table(props["GROUPVEL"]) and "RINDEX" in props


WLS_TIME_PROFILES = ("delta", "exponential")  # G4OpWLS::UseTimeProfile's values; "delta" is Geant4's default


def wls_emission(props: dict, what: str, wls_time_profile) -> tuple[float, float, str]:
    """Return G4OpWLS's (expected yield, delay constant, time profile) of a WLS material.

    The yield is WLSMEANNUMBERPHOTONS (Geant4 emits Poisson(mean) photons; 1 when absent), the delay constant
    WLSTIMECONSTANT (0 when absent) and the profile the material's ``wls_time_profile`` setting ("delta" when
    absent, as Geant4's default).
    """
    profile = "delta" if wls_time_profile is None else wls_time_profile
    if profile not in WLS_TIME_PROFILES:
        raise OpticsRefused(f"{what}: wls_time_profile {profile!r} is not one of G4OpWLS's {WLS_TIME_PROFILES}")
    for k in ("WLSMEANNUMBERPHOTONS", "WLSTIMECONSTANT"):
        if k in props and (is_table(props[k]) or not float(props[k].value) >= 0.0):
            raise OpticsRefused(f"{what}: {k} must be a constant >= 0")
    mean = float(props["WLSMEANNUMBERPHOTONS"].value) if "WLSMEANNUMBERPHOTONS" in props else 1.0
    tau = float(props["WLSTIMECONSTANT"].value) if "WLSTIMECONSTANT" in props else 0.0
    return mean, tau, profile


def _wls_channel(props: dict, name: str, what: str, wls_time_profile):
    if "WLSCOMPONENT" not in props:
        raise OpticsRefused(f"{what}: WLSABSLENGTH without WLSCOMPONENT (G4OpWLS absorbs and emits nothing)")
    mean, tau, profile = wls_emission(props, what, wls_time_profile)
    law, lp = table(props["WLSABSLENGTH"], "length")
    wle = props["WLSCOMPONENT"]
    spectra.geant4_spectrum(f"{name}.WLSCOMPONENT", wle.energy, wle.value)  # its refusals
    record = emission.EmissionRecord(f"{name}.wls", emission.wls_profile(profile))
    spec = {"energy": np.asarray(wle.energy, np.float64), "value": np.asarray(wle.value, np.float64)}
    return bulk.Channel("wls", phase.ISOTROPIC, law, emission=record), {
        "length": lp,
        "yield": mean,
        "tau": tau,
        "spectrum": spec,
    }


def mie_constants(props: dict, what: str) -> tuple[dict, frozenset]:
    """Return the Mie lobe's parameters read from the constants, and the constants Geant4 never reads.

    G4OpMieHG reads MIEHG_FORWARD_RATIO at every Mie event, MIEHG_FORWARD only when the forward lobe is drawn
    (``rand <= ratio``, never at ratio 0) and MIEHG_BACKWARD only otherwise (never at ratio 1); a constant it reads
    and the material lacks makes it abort (mat202), so that is refused here. A constant it never reads and the
    material lacks is 0 (any value gives the same light). |g| <= ``MIE_G_MAX`` (float32 resolution) and
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
        if not abs(g) <= MIE_G_MAX:
            raise OpticsRefused(f"{what}: {key} {g} outside [-{MIE_G_MAX}, {MIE_G_MAX}] (a narrower lobe "
                                "than float32 directions resolve)")  # fmt: skip
        lobe[name] = g
    return lobe, frozenset(fixed)


def mie(props: dict, what: str):
    """Return Geant4's Mie channel of a material with MIEHG, and its parameters (``mie_constants`` checks them)."""
    lobe, _ = mie_constants(props, what)
    law, lp = table(props["MIEHG"], "length")
    return bulk.mie(law), {"length": lp, "lobe": {k: lobe[k] for k in phase.MIE_PARAMS}}


def medium(material: schema.Material, name: str | None = None):
    """Return the MediumEntry of a material, Geant4's reading (see the module docstring), named ``name``.

    The index is ``laws.geant4_rindex`` (photons timed by Geant4's GROUPVEL of RINDEX), or with a user GROUPVEL
    ``laws.geant4_rindex_groupvel`` (timed by that table). WLS: ``wls_emission``.
    """
    props = material.properties
    name = name or material.name
    what = f"material {material.name!r}"
    _check_medium(props, what, material.optics.settings)
    if "RINDEX" not in props:
        return MediumEntry(bulk.Medium(name, None), {})
    for k in ("RINDEX", "ABSLENGTH", "RAYLEIGH", "MIEHG", "WLSABSLENGTH", "WLSCOMPONENT"):
        if k in props and not is_table(props[k]):
            raise OpticsRefused(f"{what}: {k} must be a table")
    p = {"index": {"energy": props["RINDEX"].energy, "value": props["RINDEX"].value}}
    ilaw = GEANT4_RINDEX
    if user_groupvel(props):
        gv = props["GROUPVEL"]
        p["index"].update(groupvel_energy=gv.energy, groupvel=gv.value)
        ilaw = GEANT4_RINDEX_GROUPVEL
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
        ch, p["wls"] = _wls_channel(props, name, what, material.optics.settings.get("wls_time_profile"))
        channels.append(ch)
    return MediumEntry(bulk.Medium(name, ilaw, alaw, tuple(channels)), p)


# ------------------------------------------------------------------------------------------------------- surfaces
def complex_index(s: schema.Surface) -> bool:
    """Return whether a dielectric_metal surface's reflectance is its complex index's (REALRINDEX + i
    IMAGINARYRINDEX without REFLECTIVITY: G4OpBoundaryProcess::CalculateReflectivity)."""  # noqa: D205, D209
    pr = s.properties
    return "REFLECTIVITY" not in pr and "REALRINDEX" in pr and "IMAGINARYRINDEX" in pr


def _metal_path(s: schema.Surface, model: str, finish: str, what: str) -> str:
    """Return the path of Geant4's DielectricMetal reading of a surface: "mirror", "glisur" (the facet path at
    1 - polish), "unified" (spike / lobe / back / Lambert) or "unified_normal" (the same at sigma_alpha 0, where
    GetFacetNormal returns the normal: the lobe is the spike), refusing a polish or sigma_alpha outside the range the
    facet samplers cover."""  # noqa: D205, D209
    if complex_index(s):
        if finish != "polished" and not (model == "glisur" and s.value >= 1.0):
            raise OpticsRefused(f"{what}: a complex index on a ground finish (the facet's Fresnel) is not supported")
        return "complex"
    if model == "glisur":
        if finish != "ground" or s.value >= 1.0:
            return "mirror"
        if s.value < 0.0 or 1.0 - s.value < GLISUR_R_MIN:
            raise OpticsRefused(f"{what}: polish {s.value} outside [0, {1 - GLISUR_R_MIN}] (or exactly 1)")
        return "glisur"
    if finish == "polished":
        return "mirror"
    if s.value == 0.0:
        return "unified_normal"
    if not SIGMA_MIN <= s.value <= SIGMA_MAX:
        raise OpticsRefused(f"{what}: sigma_alpha {s.value} outside [{SIGMA_MIN}, {SIGMA_MAX}] (or exactly 0)")
    return "unified"


def _metal_entry(s: schema.Surface, path: str, refl, eff):
    """Return (type, params, far side) of a DielectricMetal path, the absorbed share detected with ``eff``; with a
    TRANSMITTANCE the type transmits straight on (``types.transmitting``), and into a medium without RINDEX that
    share is absorbed where it lands (Geant4 transmits without a RINDEX check, then kills at the next step)."""  # noqa: D205, D209
    pr = s.properties
    lobes = {"p_spike": fraction(pr, "SPECULARSPIKECONSTANT", 0.0), "p_lobe": fraction(pr, "SPECULARLOBECONSTANT", 0.0),
             "p_back": fraction(pr, "BACKSCATTERCONSTANT", 0.0)}  # fmt: skip
    if path == "complex":
        stype = types.PHOTOCATHODE  # a specular semi-infinite absorber of index nr + i nk
        params = {"nr": index(pr["REALRINDEX"]), "nk": extinction(pr["IMAGINARYRINDEX"]), "eff": eff}
    elif path == "glisur":
        stype, params = types.GLISUR_METAL, {"R": refl, "polish": float(s.value), "eff": eff}
    elif path == "mirror":
        stype, params = types.SPECULAR_METAL, {"R": refl, "eff": eff}
    elif path == "unified_normal":
        stype, params = types.UNIFIED_METAL_NORMAL, {"R": refl, **lobes, "eff": eff}
    else:
        stype, params = types.UNIFIED_METAL, {"R": refl, **lobes, "sigma_alpha": float(s.value), "eff": eff}
    if "TRANSMITTANCE" not in pr:  # no TRANSMITTANCE: Geant4's T 0, no straight share
        return stype, params, None
    return types.transmitting(stype), params | {"T": fraction(pr, "TRANSMITTANCE", 0.0)}, FAR_ABSORB


def _ordinary(stype, params: dict, far_side=None, **kw) -> SurfaceEntry:
    """Return the entry of a surface that is no sensor: its type with ``eff`` fixed at 0 (``types.no_detection``)."""
    return SurfaceEntry(types.no_detection(stype), {k: v for k, v in params.items() if k != "eff"}, far_side=far_side,
                        **kw)  # fmt: skip


def _dielectric(s: schema.Surface, finish: str, what: str, refl, eff):
    """Return (type, params, far side, between-same-material skip) of a dielectric_dielectric surface, the absorbed
    share (Geant4's DoAbsorption) detected with ``eff`` (Geant4's default 0, ``DEFAULT_EFFICIENCY``)."""  # noqa: D205, D209
    # Geant4 reads T at its default 0 when the surface gives none (G4OpBoundaryProcess.cc:310, 463): a parameter
    t = fraction(s.properties, "TRANSMITTANCE", 0.0)
    if finish in ("polishedfrontpainted", "groundfrontpainted"):
        fs = 1.0 if finish == "polishedfrontpainted" else 0.0
        # into a medium without RINDEX the transmission is absorbed (lumen.rules.boundary)
        return types.PAINTED_T, {"R": refl, "T": t, "f_spec": fs, "eff": eff}, FAR_ABSORB, False
    # GetFacetNormal returns the normal for glisur at polish 1: the polished reading (G4OpBoundaryProcess.cc, 11.3.0)
    if finish == "polished" or (s.model == "glisur" and s.value >= 1.0):
        # absorption (detected with EFFICIENCY on a sensor) and transmission before Fresnel; NoRINDEX beyond: killed
        # before the R / T draw (lumen.rules.boundary)
        return types.DIELECTRIC_RT, {"R": refl, "T": t, "eff": eff}, FAR_KILL, True
    if s.model == "glisur":  # ground: the facet path, reflecting or refracting at each facet
        if s.value < 0.0 or 1.0 - s.value < GLISUR_R_MIN:
            raise OpticsRefused(f"{what}: polish {s.value} outside [0, {1 - GLISUR_R_MIN}] (or exactly 1)")
        return types.GLISUR_DIELECTRIC, {"R": refl, "T": t, "polish": float(s.value), "eff": eff}, FAR_KILL, True
    raise OpticsRefused(f"{what}: unified ground dielectric_dielectric (facet refraction) is not supported")


# Geant4's value of a surface property the model reads when the surface omits it: the readers insert these values;
# the source lines are the citations a default key shows (G4OpBoundaryProcess.cc, 11.2.2)
DEFAULT_EFFICIENCY = 0.0  # G4OpBoundaryProcess::PostStepDoIt starts every step at EFFICIENCY 0, on any surface
DEFAULT_LINES = {"REFLECTIVITY": "G4OpBoundaryProcess.cc:308", "TRANSMITTANCE": "G4OpBoundaryProcess.cc:310",
                 "SPECULARLOBECONSTANT": "G4OpBoundaryProcess.cc:405-407",
                 "SPECULARSPIKECONSTANT": "G4OpBoundaryProcess.cc:408-410",
                 "BACKSCATTERCONSTANT": "G4OpBoundaryProcess.cc:411-413"}  # fmt: skip


def surface_defaults(s: schema.Surface) -> dict:
    """Return {KEY: (Geant4's default, its source line)} of the properties ``s`` omits that its reading reads.

    Asked of the reader: every parameter of its one reading (``_reading``, a sensor's or not) whose Geant4 key
    (``keys.SURFACE_KEYS``) the surface omits, at the value the reader put there. EFFICIENCY is not offered: a
    sensor's efficiency is its group's or its sensor model's. A surface another optics model reads, or one the reader
    refuses, has none.
    """
    if s.optics.name != "geant4":
        return {}
    pr = s.properties
    try:
        params = _reading(s)[1]
    except OpticsRefused:
        return {}
    out = {}
    for param, v in params.items():
        key = SURFACE_KEYS.get(param)
        if key is not None and key in DEFAULT_LINES and key not in pr and not isinstance(v, tuple):
            out[key] = (float(v), DEFAULT_LINES[key])
    return out


def detector_defaults(df: schema.DetectorFile) -> dict:
    """Return {surface: surface_defaults(...)} of every surface of a detector file."""
    return {n: d for n, s in df.surfaces.items() if (d := surface_defaults(s))}


VALUE_RULES = ValueRules(value_key, detector_defaults)  # what applying parameter values reads (lumen.format.apply)


def _reading(s: schema.Surface):
    """Return (type, params, far side, between-same-material skip) of Geant4 11's reading of an optical surface, its
    absorbed share detected with EFFICIENCY (Geant4's default 0): the one reading of a surface, a sensor's or not, and
    its one list of refusals (see the module docstring)."""  # noqa: D205, D209
    model, finish, type_ = s.model, s.finish, s.type
    what = f"opticalsurface {s.name!r} ({model} {finish} {type_})"
    pr = s.properties
    for k in REFUSED_SURFACE:
        if k in pr:
            raise OpticsRefused(f"{what}: {k} is not supported")
    if model not in ("glisur", "unified"):
        raise OpticsRefused(f"{what}: model {model} is not supported")
    if "backpainted" in finish or finish not in SUPPORTED_FINISHES:
        raise OpticsRefused(f"{what}: finish {finish} is not supported")
    refl, eff = fraction(pr, "REFLECTIVITY", 1.0), fraction(pr, "EFFICIENCY", DEFAULT_EFFICIENCY)
    if type_ == "dielectric_metal":  # DielectricMetal reads no second material (no SameMaterial, no NoRINDEX)
        return (*_metal_entry(s, _metal_path(s, model, finish, what), refl, eff), False)
    if type_ == "dielectric_dielectric":
        return _dielectric(s, finish, what, refl, eff)
    raise OpticsRefused(f"{what}: surface type {type_} is not supported")


def surface(s: schema.Surface) -> SurfaceEntry:
    """Return the SurfaceEntry of an optical surface that is no sensor: its reading without detection."""
    stype, params, far, same = _reading(s)
    return _ordinary(stype, params, far, ignored_between_same_material=same)


# ----------------------------------------------------------------------------------------------- detecting surfaces
def geant4_sensor(s: schema.Surface) -> SensorEntry:
    """Return stock Geant4's reading of a detecting surface: the surface's one reading, detecting (``_reading``)."""
    stype, params, far, same = _reading(s)
    return SensorEntry(stype, params, far, ignored_between_same_material=same)


def sensor(s: schema.Surface, eff: float | None = None, per_normal: bool = False) -> SensorEntry:
    """Return the SensorEntry of a detecting optical surface: the surface as stock Geant4 reads it (``geant4_sensor``).

    ``eff`` is the sensor group's detection on entry, per absorbed photon, in place of the surface's EFFICIENCY
    (``lumen.rules.sensors.entry_efficiency``: a group's ``entry_efficiency`` setting, or 1 under a sensor model that
    applies its QE after tracking). ``per_normal``: that sensor model's QE is the detection at normal incidence, so the
    deposit is taken over the cathode type's own normal-incidence absorptance at each wavelength
    (``types.per_normal``; docs/dev/decisions.md, 11).
    """
    e = geant4_sensor(s)
    if not e.surface_type.detects:
        raise OpticsRefused(f"opticalsurface {s.name!r}: its reading detects nothing")
    if eff is not None:  # every absorption detected with eff: the reading's own efficiency replaced
        e = dataclasses.replace(e, params=dict(e.params, eff=float(eff)))
    return dataclasses.replace(e, surface_type=types.per_normal(e.surface_type)) if per_normal else e


# ------------------------------------------------------------------------------------------------- the light's domain
def emission_support(t) -> tuple[float, float] | None:
    """Return the support (eV) of Geant4's sampler of a spectrum table: its knot intervals of positive mass."""
    e, v = np.asarray(t.energy, np.float64), np.asarray(t.value, np.float64)
    pos = np.flatnonzero(0.5 * (v[1:] + v[:-1]) * np.diff(e) > 0)
    return (float(e[pos[0]]), float(e[pos[-1] + 1])) if len(pos) else None


def light_domain(df) -> tuple[tuple[float, float], ...] | None:
    """Return the photon energies (eV) a detector's light can have: its emitters' supports, as merged intervals.

    G4Cerenkov emits within each material's RINDEX table (its first to last knot); G4Scintillation and G4OpWLS
    within their spectra's supports (``emission_support``). A table is evaluated only there (held at its end values
    beyond its knots), so its checks read it only there: its knots inside the domain and its values at the domain's
    ends (``laws.validate``, ``lumen.build.values``); a source's light must lie inside it. None (no material has a
    RINDEX table): every wavelength.
    """
    iv, rindex = [], []
    for m in df.materials.values():
        props = m.properties
        r = props.get("RINDEX")
        if r is not None and r.energy is not None and len(r.energy):
            rindex.append((float(np.min(r.energy)), float(np.max(r.energy))))
        for k in (*SCINTILLATION_KEYS["spectra"], "WLSCOMPONENT"):
            t = props.get(k)
            if t is not None and t.energy is not None and len(t.energy) > 1:
                sup = emission_support(t)
                if sup is not None:
                    iv.append(sup)
    if not rindex:
        return None
    out: list = []
    for lo, hi in sorted(rindex + iv):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return tuple(out)
