"""Units in GDML: CLHEP's table (Geant4's internal units mm, ns, MeV, rad) and the exact conversion to lumen's units.

``UNITS`` maps every unit name GDML may use to its value in Geant4's internal units, as CLHEP defines it.
``LENGTH_M`` and ``ENERGY_EV`` (``lumen.units``, shared with the YAML reader) give the units that are exact decimal
multiples of a metre and an electronvolt: a plain literal in one of them converts by exact decimal arithmetic
(``exact("12.5", LENGTH_M["mm"])`` is exactly 0.0125, rounded once), and anything else is evaluated with CLHEP's unit
values and converted by one multiplication (Geant4's own evaluator reads some unit expressions a few 1e-7 off:
``lumen.gdml.reader``).

``shift`` is the exporter's side of the same rule: the shortest repr of a double with its decimal point moved, exactly
(``shift("0.0125", 3)`` is "12.5"), so that the importer's exact scaling reads the same double back.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal

from lumen.units import C_M_S, E_CHARGE_C, H_PLANCK_J_S, NUMBER, TWO_PI

_EV = 1.0e-6
_JOULE = _EV / E_CHARGE_C
_S = 1.0e9
_M = 1000.0
_KG = _JOULE * _S * _S / (_M * _M)
_G = 1.0e-3 * _KG
_CM = 10.0
_PASCAL = _JOULE / (_M**3)
# CLHEP's definitions (SystemOfUnits.h, PhysicalConstants.h), in the same arithmetic
_PARSEC = 3.0856775807e16 * _M
_C_LIGHT = C_M_S * _M / _S
_H_PLANCK = H_PLANCK_J_S * _JOULE * _S

UNITS = {
    "mm": 1.0, "millimeter": 1.0, "cm": 10.0, "centimeter": 10.0, "m": 1000.0,
    "meter": 1000.0, "km": 1.0e6, "kilometer": 1.0e6, "um": 1.0e-3,
    "micrometer": 1.0e-3, "nm": 1.0e-6, "nanometer": 1.0e-6,
    "angstrom": 1.0e-7, "fermi": 1.0e-12, "pc": _PARSEC, "parsec": _PARSEC,
    "mm2": 1.0, "cm2": 100.0, "m2": 1.0e6, "mm3": 1.0, "cm3": 1000.0,
    "m3": 1.0e9, "L": 1.0e6, "l": 1.0e6, "liter": 1.0e6, "dL": 1.0e5,
    "cL": 1.0e4, "mL": 1.0e3,
    "rad": 1.0, "radian": 1.0, "mrad": 1.0e-3, "milliradian": 1.0e-3,
    "deg": math.pi / 180.0, "degree": math.pi / 180.0, "sr": 1.0,
    "steradian": 1.0,
    "eV": _EV, "electronvolt": _EV, "keV": 1.0e-3, "kiloelectronvolt": 1e-3,
    "MeV": 1.0, "megaelectronvolt": 1.0, "GeV": 1.0e3, "gigaelectronvolt": 1e3,
    "TeV": 1.0e6, "PeV": 1.0e9, "meV": 1.0e-9, "joule": _JOULE, "J": _JOULE,
    "s": _S, "second": _S, "ms": 1.0e6, "millisecond": 1.0e6, "us": 1.0e3,
    "microsecond": 1.0e3, "ns": 1.0, "nanosecond": 1.0, "ps": 1.0e-3,
    "picosecond": 1.0e-3, "Hz": 1.0 / _S, "hertz": 1.0 / _S,
    "kg": _KG, "kilogram": _KG, "g": _G, "gram": _G, "mg": 1.0e-3 * _G,
    "milligram": 1.0e-3 * _G,
    "mole": 1.0, "mol": 1.0, "kelvin": 1.0, "K": 1.0,
    "pascal": _PASCAL, "Pa": _PASCAL, "bar": 1.0e5 * _PASCAL,
    "atmosphere": 101325.0 * _PASCAL, "atm": 101325.0 * _PASCAL,
    "perCent": 0.01, "percent": 0.01, "perThousand": 0.001,
    "perMillion": 1.0e-6, "candela": 1.0, "cd": 1.0,
    "pi": math.pi, "twopi": TWO_PI, "halfpi": math.pi / 2,
    "pi2": math.pi * math.pi, "e": math.e, "gamma": 0.577215664901532861,
    "c_light": _C_LIGHT, "h_Planck": _H_PLANCK, "hbarc": _H_PLANCK / TWO_PI * _C_LIGHT,
    "hbar_Planck": _H_PLANCK / TWO_PI,
    "universe_mean_density": 1.0e-25 * _G / (_CM**3),
    "inch": 25.4, "in": 25.4, "ft": 304.8, "foot": 304.8,
}  # fmt: skip

LITERAL = re.compile(rf"^\s*{NUMBER}\s*$")  # lumen.units.NUMBER, the YAML reader's too
PRODUCT = re.compile(rf"^\s*({NUMBER})\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")


def is_literal(text: str | None) -> bool:
    """Whether a GDML attribute is a plain decimal number (no expression)."""
    return text is not None and LITERAL.match(text) is not None


PLAIN_LEADS = (-5, 16)  # a literal whose leading digit's exponent is in [-5, 16) is written without an exponent


def plain(d: Decimal) -> str:
    """Return a finite decimal as the shortest plain literal of its digits ("12.5", "0.002", "2e-21", "-0").

    The value is kept exactly: trailing zeros are dropped, and an exponent is written only for a leading digit outside
    ``PLAIN_LEADS`` (as Python's repr does), with no "+" sign. Every literal is one ``LITERAL`` matches.
    """
    sign, digits, exponent = d.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"not a finite decimal: {d}")
    s = "-" if sign else ""
    full = "".join(map(str, digits))  # no leading zeros unless the value is zero
    ds = full.rstrip("0")
    if not ds:
        return s + "0"
    exponent += len(full) - len(ds)
    lead = len(ds) + exponent - 1  # the exponent of the leading digit
    if not PLAIN_LEADS[0] <= lead < PLAIN_LEADS[1]:
        return s + ds[0] + ("." + ds[1:] if len(ds) > 1 else "") + f"e{lead}"
    if exponent >= 0:
        return s + ds + "0" * exponent
    if lead >= 0:
        return s + ds[: lead + 1] + "." + ds[lead + 1 :]
    return s + "0." + "0" * (-lead - 1) + ds


def shift(text: str, places: int) -> str:
    """Return the decimal literal ``text`` times ``10**places``, exactly, as a plain literal (``plain``)."""
    return plain(Decimal(text).scaleb(places))
