"""Units and constants: lengths in m, times in ns, wavelengths in nm, photon energies in eV.

Every number lumen takes or returns is in these units. The constants below convert other units into them, e.g.
``thickness = 2.5 * units.MM``. A quantity written as text, as in a YAML detector, is read by ``quantity``. This is
the one module of physical constants, named units and 2 pi: every other module imports them from here.
"""

import math
import re
from decimal import Decimal

# the SI-2019 defining constants (exact)
C_M_S = 2.99792458e8  # speed of light in vacuum, m / s
H_PLANCK_J_S = 6.62607015e-34  # Planck constant, J s
E_CHARGE_C = 1.602176634e-19  # elementary charge, C (one eV in J)
FINE_STRUCTURE = 1.0 / 137.035999084  # the fine-structure constant (CODATA 2018): Frank-Tamm's photon number

C_VACUUM = 0.299792458  # speed of light in vacuum, m / ns
HC_EV_M = H_PLANCK_J_S * C_M_S / E_CHARGE_C  # h c in eV m, exact to float64 (CLHEP's h_Planck c_light)
HC = HC_EV_M * 1e9  # h c in eV nm (the same constant): the wavelength <-> energy conversion of tables and sources
TWO_PI = 2.0 * math.pi
INF = math.inf  # an unbounded end of a range
# a finite stand-in for infinity: a float32 with room above it (x 3.4e8 before float32 overflows), so a sum or product
# of it stays finite where inf would give NaN (0 x inf); the geometry's "no crossing" and unbounded end, the
# navigator's "no hit", and what an inf GDML matrix entry is written as (python -m lumen.gdml.repair)
HUGE = 1e30

MM = 1e-3  # one millimetre in m
US = 1e3  # one microsecond in ns
MM_PER_M = 1000.0  # Geant4's internal length unit is the mm
NM_PER_M = 1e9
S_PER_NS = 1e-9  # a rate in Hz times this is per ns
EV_PER_MEV = 1e6  # Geant4's internal energy unit is the MeV


def energy_of(wavelength):
    """Return the photon energy (eV) at a wavelength (nm)."""
    return HC / wavelength


def wavelength_of(energy):
    """Return the wavelength (nm) of a photon energy (eV)."""
    return HC / energy


# ---------------------------------------------------------------------------------------- quantities written as text
# A quantity written as text ("12.5*mm", "2.5 eV", "90*deg") converts with one rounding: the number and the unit's
# factor are exact decimals, multiplied exactly, and rounded once to a float ("12.5*mm" is exactly 0.0125 m, rounded
# once). Only an angle in degrees is not a decimal multiple of a radian; it is converted by one multiplication, as
# Geant4 does. Each table gives a unit's size in lumen's unit of that quantity, as a decimal string.
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
TIME_NS = {"ns": "1", "ps": "0.001", "us": "1000", "ms": "1000000", "s": "1000000000"}
ANGLE_RAD = {"rad": "1", "mrad": "0.001", "deg": "", "degree": ""}  # "": not a decimal (pi / 180)
DENSITY_G_CM3 = {"g/cm3": "1", "kg/m3": "0.001", "mg/cm3": "0.001"}
PRESSURE_PA = {"Pa": "1", "pascal": "1", "bar": "100000", "atm": "101325", "atmosphere": "101325"}
TEMPERATURE_K = {"K": "1", "kelvin": "1"}
VELOCITY_M_NS = {"m/ns": "1", "mm/ns": "0.001", "cm/ns": "0.01", "m/s": "1e-9", "km/s": "0.000001"}
COMPRESSIBILITY_M3_MEV = {"m3/MeV": "1", "cm3/MeV": "0.000001", "mm3/MeV": "1e-9", "m3/eV": "1000000",
                          "m3/keV": "1000"}  # fmt: skip
QUANTITIES = {"length": LENGTH_M, "energy": ENERGY_EV, "time": TIME_NS, "angle": ANGLE_RAD,
              "density": DENSITY_G_CM3, "pressure": PRESSURE_PA, "temperature": TEMPERATURE_K,
              "velocity": VELOCITY_M_NS, "compressibility": COMPRESSIBILITY_M3_MEV}  # fmt: skip
NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"  # a decimal literal (YAML text and GDML attributes)
_TEXT = re.compile(rf"^\s*({NUMBER})\s*(?:\*?\s*([A-Za-z][A-Za-z0-9/]*))?\s*$")


def exact(text: str, factor: str) -> float:
    """Return float(Decimal(text) * Decimal(factor)): one rounding."""
    return float(Decimal(text.strip()) * Decimal(factor))


def quantity(value, what: str, unit: str | None = None) -> float:
    """Return a number, or a quantity written as text, in ``unit`` (default: lumen's unit of ``what``).

    ``what`` is a key of ``QUANTITIES`` ("length", "energy", ...); a plain number (or its text, "1e-5") is already
    in ``unit``. Raises ValueError for text that is not a number and a unit of that
    quantity.
    """
    table = QUANTITIES[what]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    m = _TEXT.match(value) if isinstance(value, str) else None
    if m is None:
        example = f"1*{next(iter(table))}"
        raise ValueError(f'{value!r} is not a number or a quantity of {what} such as "{example}"')
    number, src = m.group(1), m.group(2)
    if src is None:
        return float(Decimal(number))
    if src not in table:
        raise ValueError(f"{value!r}: {src!r} is not a unit of {what} (one of {', '.join(table)})")
    a, b = table[src], table[unit] if unit is not None else "1"
    if a and b:
        return float(Decimal(number) * Decimal(a) / Decimal(b))
    deg = math.pi / 180.0
    return float(Decimal(number)) * ((float(a) if a else deg) / (float(b) if b else deg))
