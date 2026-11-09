"""Liquid-argon optics from LArSoft's PUBLIC configuration.

The tables are ``data/lartpc_optics_public.json``: every array with its source URL and commit, compiled 2026-10-22
from LArSoft/lardataalg larproperties.fcl and detector overrides.

    lar_tables()        LAr's RIndexSpectrum, RayleighSpectrum (Babicz et al. 2020, arXiv:2002.09346) and
                        AbsLengthSpectrum as detector-file Property records (RINDEX, RAYLEIGH, ABSLENGTH; energies eV,
                        lengths m, origin "recipe")
    lar_medium_from()   the medium of a material whose properties are those tables (the optics model
                        "lumen:larsoft_lar"): read as Geant4 reads them, linear in photon ENERGY, clamped outside
    reflector_tables()  LArSoft's OpBoundaryProcessSimple surface of a named material as Property records:
                        REFLECTIVITY (LArSoft's reflectance) and DIFFUSE_FRACTION (the diffuse share of the reflected
                        light), each a table in energy
    reflector_from()    such a surface (the optics model "lumen:larsoft_reflector") as a REFLECTOR at LAr's 128 nm
                        line: reflected light Lambertian with the diffuse fraction, else specular; the rest absorbed
    lar_scintillation() the emission spectrum (FastScintSpectrum = SlowScintSpectrum), the time constants, the fast
                        fraction and the yield (standard values; detector overrides by argument)

``lar_medium()`` and ``reflector()`` read the shipped tables directly and give the same results.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from lumen.format import schema
from lumen.media import bulk, laws, phase, spectra
from lumen.surfaces import _legacy as legacy
from lumen.units import CM, HC

DATA = Path(__file__).parent / "data" / "lartpc_optics_public.json"
STANDARD = "larsoft_standard"
LAR_LINE = 128.0  # nm: LAr's scintillation line, where LArSoft's reflector tables are read
REFLECTOR_SIGMA_ALPHA = 0.1  # rad: the facet width the REFLECTOR type carries (unused: f_glossy is 0)
LENGTH_UNITS = {"cm": CM}  # the length units the tables are given in, in m
REFLECTIVITY, DIFFUSE_FRACTION = "REFLECTIVITY", "DIFFUSE_FRACTION"  # the reflector's property keys


@lru_cache(maxsize=1)
def data() -> dict:
    """Return the shipped LArSoft optics tables."""
    return json.loads(DATA.read_text())


def _sorted_table(entry, scale=1.0):
    """Return (energies ascending, values) of a {energy (eV), values} entry, values times ``scale``."""
    e = np.asarray(entry["energy"], np.float64)
    v = np.asarray(entry["values"], np.float64) * scale
    o = np.argsort(e)
    return e[o], v[o]


def _length_scale(entry, key):
    unit = entry.get("value_unit", "cm")
    if unit not in LENGTH_UNITS:
        raise ValueError(f"LArSoft table {key}: length unit {unit!r} is not one of {sorted(LENGTH_UNITS)}")
    return LENGTH_UNITS[unit]


def lar_tables(table: str = STANDARD) -> dict[str, schema.Property]:
    """Return LAr's LArSoft tables as Property records {RINDEX, RAYLEIGH, ABSLENGTH} (energies eV, lengths m)."""
    d = data()["lar"][table]
    out = {}
    for key, name, length in (("RINDEX", "rindex", False), ("RAYLEIGH", "rayleigh_length", True),
                              ("ABSLENGTH", "abs_length", True)):  # fmt: skip
        entry = d[name]
        e, v = _sorted_table(entry, _length_scale(entry, name) if length else 1.0)
        note = f"LArSoft {table} {name}: {entry.get('source', d['source'])}"
        out[key] = schema.Property(key, v, e, "", "recipe", note)
    return out


def lar_medium_from(material: schema.Material, name: str = "lar", scatter: bool = True, absorb: bool = True):
    """Return (bulk.Medium, params) of LAr from a material's RINDEX, RAYLEIGH and ABSLENGTH (LArSoft's tables).

    The index is Geant4's reading (``laws.geant4_rindex``: n linear in energy, photons timed by GROUPVEL's group
    index); ``scatter`` / ``absorb`` switch the Rayleigh channel and the absorption.
    """
    props = material.properties
    missing = [k for k in ("RINDEX",) + (("RAYLEIGH",) if scatter else ()) + (("ABSLENGTH",) if absorb else ())
               if k not in props or props[k].energy is None]  # fmt: skip
    if missing:
        raise ValueError(f"material {material.name!r}: LArSoft's LAr needs the tables {missing}")
    p = {"index": {"energy": props["RINDEX"].energy, "value": props["RINDEX"].value}}
    alaw = None
    if absorb:
        a = props["ABSLENGTH"]
        alaw = laws.tabulated("length", a.energy[0], a.energy[-1], outside="clamp")
        p["absorption"] = {"energy": a.energy, "value": a.value}
    channels = ()
    if scatter:
        r = props["RAYLEIGH"]
        channels = (bulk.Channel("rayleigh", phase.RAYLEIGH, laws.tabulated("length", r.energy[0], r.energy[-1],
                                                                             outside="clamp")),)  # fmt: skip
        p["rayleigh"] = {"length": {"energy": r.energy, "value": r.value}}
    return bulk.Medium(name, laws.geant4_rindex(), alaw, channels), p


def lar_medium(name: str = "lar", table: str = STANDARD, scatter: bool = True, absorb: bool = True):
    """Return (bulk.Medium, params) of LAr with the shipped LArSoft tables (``lar_medium_from`` on them)."""
    lar = schema.Material("LAr", "liquid", schema.Composition(1.0), properties=lar_tables(table))
    return lar_medium_from(lar, name, scatter, absorb)


def reflector_tables(material: str, table: str = STANDARD) -> dict[str, schema.Property]:
    """Return LArSoft's ReflectiveSurface tables of a material as {REFLECTIVITY, DIFFUSE_FRACTION} Property records."""
    s = data()["surfaces"][table]
    e = np.asarray(s["energies_eV"], np.float64)
    m = s["surfaces"][material]
    note = f"LArSoft {table} ReflectiveSurfaces {material}: {s['source']}"
    return {REFLECTIVITY: schema.Property(REFLECTIVITY, np.asarray(m["reflectance"], np.float64), e, "", "recipe",
                                          note),
            DIFFUSE_FRACTION: schema.Property(DIFFUSE_FRACTION, np.asarray(m["diffuse_fraction"], np.float64), e, "",
                                              "recipe", note)}  # fmt: skip


def reflector_from(properties: dict):
    """Return (REFLECTOR, params) of a LArSoft reflector surface's REFLECTIVITY and DIFFUSE_FRACTION tables.

    R and the diffuse fraction are interpolated linearly in energy at LAr's 128 nm line (the tables are flat
    there); the reflected light is specular with probability 1 - diffuse fraction.

    Notation: E0 the line's photon energy (eV), R the reflectance.
    """
    E0 = HC / LAR_LINE
    R = float(np.interp(E0, properties[REFLECTIVITY].energy, properties[REFLECTIVITY].value))
    dfr = float(np.interp(E0, properties[DIFFUSE_FRACTION].energy, properties[DIFFUSE_FRACTION].value))
    return legacy.REFLECTOR, {"R": R, "f_spec": 1.0 - dfr, "f_glossy": 0.0, "sigma_alpha": REFLECTOR_SIGMA_ALPHA}


def reflector(material: str, table: str = STANDARD):
    """Return (REFLECTOR, params) of a material's shipped LArSoft reflector tables (``reflector_from`` on them)."""
    return reflector_from(reflector_tables(material, table))


def lar_scintillation(table: str = STANDARD, **override) -> dict:
    """Return LAr's scintillation: spectrum, tau_fast, tau_slow (ns), f_fast and yield (photons / MeV).

    ``override`` replaces any of the scalars (detector overrides: MicroBooNE yield 40000, f_fast 0.23; SBND tau_slow
    1300 ns). G4Scintillation samples the spectrum per unit energy (``spectra.geant4_spectrum``).
    """
    d = data()["lar"][table]
    e, s = _sorted_table(d["fast_scint_spectrum"])
    out = dict(spectrum=spectra.geant4_spectrum("lar_scint", e, s), tau_fast=float(d["fast_time_constant"]["value"]),
               tau_slow=float(d["slow_time_constant"]["value"]), f_fast=float(d["yield_ratio"]["value"]),
               **{"yield": float(d["scint_yield"]["value"])})  # fmt: skip
    out.update(override)
    return out
