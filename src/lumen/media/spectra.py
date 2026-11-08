"""Emission spectra of wavelength shifters and scintillators, drawn by their exact inverse CDF.

A spectrum is static (hashed by identity) and is a FIXED proposal: ``sample(rand)`` maps a uniform in (0, 1) to a
wavelength (nm), and ``rand_of(lam)`` is its inverse, the CDF. Yields and delay times are parameters of the surface or
channel that uses the spectrum (live and differentiable), not of the spectrum.

- ``WavelengthSpectrum``: piecewise linear in wavelength between knots (``wavelength_spectrum``).
- ``EnergySpectrum``: piecewise linear in photon energy on segments, with jumps between them. Geant4's sampler for
  WLSCOMPONENT and SCINTILLATIONCOMPONENTi is the special case of a constant density per knot interval
  (``geant4_spectrum``).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from lumen.units import HC

LINEAR = 1e-12  # a segment whose density slope is below this (relative) is drawn as flat
TINY = 1e-30  # a floor for divisions by a density


@dataclass(frozen=True, eq=False)
class WavelengthSpectrum:
    """An emission spectrum piecewise linear in wavelength; build it with ``wavelength_spectrum``.

    Attributes
    ----------
    name : str
    wl : tuple of float
        Knots (nm), ascending.
    cdf : tuple of float
        The cumulative distribution at the knots, from 0 to 1.
    density : tuple of float
        The normalised density (1/nm) at the knots.
    """

    name: str
    wl: tuple[float, ...]
    cdf: tuple[float, ...]
    density: tuple[float, ...]

    def sample(self, rand):
        """Return the wavelength (nm) for a uniform in (0, 1): the exact inverse of the spectrum's CDF."""
        wl, s, cdf = (jnp.asarray(x, jnp.float32) for x in (self.wl, self.density, self.cdf))
        i = jnp.clip(jnp.searchsorted(cdf, rand, side="right") - 1, 0, len(self.wl) - 2)
        c0, s0, s1 = cdf[i], s[i], s[i + 1]
        h = wl[i + 1] - wl[i]
        r = rand - c0  # within the segment: s0 x + (s1 - s0) x^2 / (2 h) = r
        a2 = (s1 - s0) / (2.0 * h)
        disc = jnp.maximum(s0 * s0 + 4.0 * a2 * r, 0.0)
        flat = jnp.abs(a2) < LINEAR * jnp.maximum(s0, TINY)
        # the root 2 r / (s0 + sqrt(disc)) has no cancellation when a2 is small (s0 > 0 within a segment with mass)
        x = jnp.where(flat, r / jnp.maximum(s0, TINY), 2.0 * r / jnp.maximum(s0 + jnp.sqrt(disc), TINY))
        return wl[i] + jnp.clip(x, 0.0, h)

    def rand_of(self, lam):
        """Return the uniform that ``sample`` maps to wavelength ``lam`` (nm): the CDF (numpy)."""
        return self.cdf_at(lam)

    def cdf_at(self, lam):
        """Return the CDF at ``lam`` (nm), exact for the piecewise-linear density; 0 and 1 outside the knots."""
        wl, s, c = (np.asarray(x, np.float64) for x in (self.wl, self.density, self.cdf))
        lam = np.clip(np.asarray(lam, np.float64), wl[0], wl[-1])
        i = np.clip(np.searchsorted(wl, lam, side="right") - 1, 0, len(wl) - 2)
        h, x = wl[i + 1] - wl[i], lam - wl[i]
        return np.clip(c[i] + s[i] * x + (s[i + 1] - s[i]) * x * x / (2.0 * h), 0.0, 1.0)

    def density_at(self, lam):
        """Return the normalised density (1/nm) at ``lam`` (nm); 0 outside the knots."""
        return np.interp(lam, np.asarray(self.wl), np.asarray(self.density), left=0.0, right=0.0)

    def __repr__(self):
        """Show the spectrum's name."""
        return f"WavelengthSpectrum({self.name})"


def wavelength_spectrum(name, wl, spectrum):
    """Return a ``WavelengthSpectrum`` from a spectrum sampled at wavelengths ``wl`` (nm), in any normalisation."""
    wl = np.asarray(wl, np.float64)
    s = np.clip(np.asarray(spectrum, np.float64), 0.0, None)
    if wl.ndim != 1 or len(wl) < 2 or np.any(np.diff(wl) <= 0) or s.shape != wl.shape or s.sum() <= 0:
        raise ValueError(f"{name}: ascending wavelengths and a non-negative, non-zero spectrum of the same length")
    seg = 0.5 * (s[1:] + s[:-1]) * np.diff(wl)
    norm = seg.sum()
    cdf = np.concatenate([[0.0], np.cumsum(seg)]) / norm
    return WavelengthSpectrum(name, tuple(wl.tolist()), tuple(cdf.tolist()), tuple((s / norm).tolist()))


@dataclass(frozen=True, eq=False)
class EnergySpectrum:
    """An emission spectrum piecewise linear in photon energy on segments, with jumps between them.

    Attributes
    ----------
    name : str
    e0, h : tuple of float
        Segment starts and widths (eV).
    d0, d1 : tuple of float
        The normalised density (1/eV) at each segment's start and end.
    cdf : tuple of float
        The CDF at the segment starts (cdf[0] = 0).
    """

    name: str
    e0: tuple[float, ...]
    h: tuple[float, ...]
    d0: tuple[float, ...]
    d1: tuple[float, ...]
    cdf: tuple[float, ...]

    def sample(self, rand):
        """Return the wavelength (nm) for a uniform in (0, 1): the exact inverse CDF, increasing in energy."""
        e0, h, d0, d1, c = (jnp.asarray(x, jnp.float32) for x in (self.e0, self.h, self.d0, self.d1, self.cdf))
        k = jnp.clip(jnp.searchsorted(c, rand, side="right") - 1, 0, len(self.e0) - 1)
        r = jnp.maximum(rand - c[k], 0.0)
        a = (d1[k] - d0[k]) / h[k]
        den = d0[k] + jnp.sqrt(jnp.maximum(d0[k] * d0[k] + 2.0 * a * r, 0.0))
        x = jnp.where(den > 0.0, 2.0 * r / jnp.where(den > 0.0, den, 1.0), 0.0)
        return HC / (e0[k] + jnp.clip(x, 0.0, h[k]))

    def rand_of(self, lam):
        """Return the uniform that ``sample`` maps to ``lam`` (nm), numpy; decreasing in lam."""
        energy = np.atleast_1d(np.asarray(HC / np.asarray(lam, np.float64), np.float64))
        u = np.zeros_like(energy)
        for e0, h, d0, d1, c in zip(self.e0, self.h, self.d0, self.d1, self.cdf, strict=True):
            x = np.clip(energy - e0, 0.0, h)
            m = energy >= e0
            u[m] = c + d0 * x[m] + 0.5 * (d1 - d0) / h * x[m] ** 2
        return np.clip(u, 0.0, 1.0).reshape(np.shape(lam))

    def density_energy(self, energy):
        """Return the normalised density (1/eV) at photon energies ``energy`` (eV), numpy."""
        energy = np.asarray(energy, np.float64)
        out = np.zeros_like(energy)
        for e0, h, d0, d1 in zip(self.e0, self.h, self.d0, self.d1, strict=True):
            m = (energy >= e0) & (energy < e0 + h)
            out[m] = d0 + (d1 - d0) * (energy[m] - e0) / h
        return out

    def density_at(self, lam):
        """Return the normalised density (1/nm) at wavelengths ``lam`` (nm), numpy."""
        lam = np.asarray(lam, np.float64)
        return self.density_energy(HC / lam) * HC / lam**2

    @property
    def wl(self):
        """Return the wavelength range (nm) the spectrum covers."""
        return (HC / (self.e0[-1] + self.h[-1]), HC / self.e0[0])

    def __repr__(self):
        """Show the spectrum's name."""
        return f"EnergySpectrum({self.name})"


def energy_spectrum(name, segments):
    """Return an ``EnergySpectrum`` from unnormalised segments [(e0, e1, f0, f1)]; segments without mass are dropped."""
    seg = [(a, b, f0, f1) for a, b, f0, f1 in segments if b > a and 0.5 * (f0 + f1) * (b - a) > 0.0]
    if not seg or any(f0 < 0 or f1 < 0 for _, _, f0, f1 in seg):
        raise ValueError(f"spectrum {name!r}: needs a non-negative intensity with positive integral")
    mass = np.array([0.5 * (f0 + f1) * (b - a) for a, b, f0, f1 in seg])
    tot = mass.sum()
    cdf = np.concatenate([[0.0], np.cumsum(mass)[:-1]]) / tot
    return EnergySpectrum(
        name,
        tuple(a for a, *_ in seg),
        tuple(b - a for a, b, *_ in seg),
        tuple(f0 / tot for *_, f0, _ in seg),
        tuple(f1 / tot for *_, f1 in seg),
        tuple(cdf),
    )


def geant4_spectrum(name, energy, intensity):
    """Return Geant4's sampler for a WLSCOMPONENT or SCINTILLATIONCOMPONENTi table, as an ``EnergySpectrum``.

    Notes
    -----
    G4OpWLS / G4Scintillation BuildPhysicsTable and G4PhysicsVector::GetEnergy: the cumulative trapezoid integral at
    the knots is inverted linearly, so the draw is uniform in energy within each knot interval, with that interval's
    trapezoid mass.
    """
    e, s = np.asarray(energy, np.float64), np.asarray(intensity, np.float64)
    if np.any(s < 0):
        raise ValueError(f"spectrum {name!r}: negative intensities")
    seg = []
    for i in range(len(e) - 1):
        d = 0.5 * (s[i] + s[i + 1])
        seg.append((e[i], e[i + 1], d, d))
    return energy_spectrum(name, seg)
