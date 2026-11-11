"""Units in GDML: CLHEP's table (Geant4's internal units mm, ns, MeV, rad) and the exact conversion to lumen's units.

``UNITS`` maps every unit name GDML may use to its value in Geant4's internal units, as CLHEP defines it.
``LENGTH_M`` and ``ENERGY_EV`` give the units that are exact decimal multiples of a metre and an electronvolt: a
plain literal in one of them converts by exact decimal arithmetic (``to_metres("12.5", "mm")`` is exactly 0.0125,
rounded once), and anything else is evaluated as Geant4 does and converted by one multiplication.

``shift`` is the exporter's side of the same rule: the shortest repr of a double with its decimal point moved, exactly
(``shift("0.0125", 3)`` is "12.5"), so that the importer's exact scaling reads the same double back.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal

_E_SI = 1.602176634e-19
_EV = 1.0e-6
_JOULE = _EV / _E_SI
_S = 1.0e9
_M = 1000.0
_KG = _JOULE * _S * _S / (_M * _M)
_G = 1.0e-3 * _KG
_CM = 10.0
_PASCAL = _JOULE / (_M**3)

UNITS = {
    "mm": 1.0, "millimeter": 1.0, "cm": 10.0, "centimeter": 10.0, "m": 1000.0,
    "meter": 1000.0, "km": 1.0e6, "kilometer": 1.0e6, "um": 1.0e-3,
    "micrometer": 1.0e-3, "nm": 1.0e-6, "nanometer": 1.0e-6,
    "angstrom": 1.0e-7, "fermi": 1.0e-12, "pc": 3.0857e19, "parsec": 3.0857e19,
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
    "pi": math.pi, "twopi": 2 * math.pi, "halfpi": math.pi / 2,
    "pi2": math.pi * math.pi, "e": math.e, "gamma": 0.577215664901532861,
    "c_light": 299.792458, "h_Planck": 4.135667696e-15 * _EV * _S,
    "hbarc": 197.3269804e-6 * 1.0e-6 * 1000.0, "hbar_Planck": 6.582119569e-22 * _S,
    "universe_mean_density": 1.0e-25 * _G / (_CM**3),
    "inch": 25.4, "in": 25.4, "ft": 304.8, "foot": 304.8,
}  # fmt: skip

LENGTH_M = {
    "mm": "0.001", "millimeter": "0.001", "cm": "0.01", "centimeter": "0.01", "m": "1", "meter": "1",
    "km": "1000", "kilometer": "1000", "um": "0.000001", "micrometer": "0.000001", "nm": "1e-9",
    "nanometer": "1e-9", "angstrom": "1e-10", "fermi": "1e-15", "inch": "0.0254", "in": "0.0254", "ft": "0.3048",
    "foot": "0.3048",
}  # fmt: skip
ENERGY_EV = {
    "eV": "1", "electronvolt": "1", "meV": "0.001", "keV": "1000", "kiloelectronvolt": "1000", "MeV": "1000000",
    "megaelectronvolt": "1000000", "GeV": "1e9", "gigaelectronvolt": "1e9", "TeV": "1e12", "PeV": "1e15",
}  # fmt: skip
MM_PER_M = 1000.0  # Geant4's internal length unit is the mm
LITERAL = re.compile(r"^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$")
PRODUCT = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")


def is_literal(text: str | None) -> bool:
    """Whether a GDML attribute is a plain decimal number (no expression)."""
    return text is not None and LITERAL.match(text) is not None


def exact(text: str, factor: str) -> float:
    """Return float(Decimal(text) * Decimal(factor)): one rounding."""
    return float(Decimal(text.strip()) * Decimal(factor))


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
