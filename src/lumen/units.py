"""Units: lengths in m, times in ns, wavelengths in nm, photon energies in eV.

Every number lumen takes or returns is in these units. The constants below convert other units into them, e.g.
``thickness = 2.5 * units.MM``.
"""

C_VACUUM = 0.299792458  # speed of light in vacuum, m / ns
HC = 1239.841984  # Planck constant times the speed of light, eV nm

MM = 1e-3  # one millimetre in m
CM = 1e-2  # one centimetre in m
US = 1e3  # one microsecond in ns
KEV = 1e3  # one keV in eV
DEG = 0.017453292519943295  # one degree in rad (pi / 180)


def energy_of(wavelength):
    """Return the photon energy (eV) at a wavelength (nm)."""
    return HC / wavelength


def wavelength_of(energy):
    """Return the wavelength (nm) of a photon energy (eV)."""
    return HC / energy
