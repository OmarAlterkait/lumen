"""Emission spectra of wavelength shifters and scintillators: Geant4's sampler, one JAX function for every spectrum.

    arrays = g4_arrays(energy, intensity)          # the sampler's tables (float32), from a table, traced or not
    lam = sample(arrays, rand)                     # the wavelength (nm) at a uniform in (0, 1)
    p = density(energy, intensity, e)              # the normalised density (1/eV) at photon energies e

Geant4's sampler of a WLSCOMPONENT or SCINTILLATIONCOMPONENTi table (G4OpWLS / G4Scintillation BuildPhysicsTable and
G4PhysicsVector::GetEnergy) inverts the cumulative trapezoid integral at the knots linearly: the draw is uniform in
energy within each knot interval, with that interval's trapezoid mass. ``g4_arrays`` gives, per knot interval, its
start and width (eV), its normalised density and the CDF at its start; an interval without mass is never drawn (its
CDF does not step). The same tables serve a source's spectrum (drawn at the reference values, ``lumen.sources``) and a
re-emission spectrum (a medium's WLSCOMPONENT: ``lumen.media.emission``; ``Spectrum`` checks a table). The density is 0
outside the table's span, never clipped to its ends.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from lumen.rules.choices import EPS32
from lumen.units import HC

WLS_TRIES = 100  # G4OpWLS::PostStepDoIt's draws of an emission energy below the absorbed one (10.7, 11)
F32 = jnp.float32


def kept(f):
    """Return the probability that one of ``WLS_TRIES`` draws lands where a single draw lands with probability f.

    G4OpWLS's draw: Geant4 redraws (up to ``WLS_TRIES`` times) until the emitted energy is at most the absorbed
    photon's, and drops the photon if every draw fails. That is the spectrum truncated at the absorbed energy (drawn
    by its exact inverse CDF: ``sample`` at a uniform times F, ``cumulative``) and the probability that some draw
    succeeds, 1 - (1 - F)^WLS_TRIES with F the spectrum's mass at or below the absorbed energy, a weight.
    """
    return -jnp.expm1(WLS_TRIES * jnp.log1p(-jnp.clip(f, 0.0, 1.0 - EPS32)))  # f < 1: log1p's derivative is finite


def g4_arrays(energy, intensity):
    """Return Geant4's sampler of a spectrum table as float32 arrays (interval starts e0, widths h, densities d, CDF
    at the starts c): each knot interval drawn with its trapezoid mass, uniform in energy within it."""  # noqa: D205, D209
    e, s = jnp.asarray(energy, F32), jnp.asarray(intensity, F32)
    h = e[1:] - e[:-1]
    mass = 0.5 * (s[1:] + s[:-1]) * h
    tot = jnp.sum(mass)
    d = 0.5 * (s[1:] + s[:-1]) / tot
    c = jnp.concatenate([jnp.zeros(1, F32), jnp.cumsum(mass)[:-1] / tot])
    return e[:-1], h, d, c


def invert(arrays, rand):
    """Return the coordinate (energy eV, or a table's own) of Geant4's sampler at a uniform in (0, 1): the exact
    inverse CDF of ``g4_arrays``' tables."""  # noqa: D205, D209
    e0, h, d, c = arrays
    k = jnp.clip(jnp.searchsorted(c, rand, side="right") - 1, 0, e0.shape[0] - 1)
    x = jnp.where(d[k] > 0.0, jnp.maximum(rand - c[k], 0.0) / jnp.where(d[k] > 0.0, d[k], 1.0), 0.0)
    return e0[k] + jnp.clip(x, 0.0, h[k])


def sample(arrays, rand):
    """Return the wavelength (nm) for a uniform in (0, 1): the exact inverse CDF, increasing in energy."""
    return HC / invert(arrays, rand)


def cumulative(arrays, lam):
    """Return the spectrum's mass at photon energies up to HC / ``lam`` (nm)."""
    e0, h, d, c = arrays
    energy = HC / lam
    k = jnp.clip(jnp.searchsorted(e0, energy, side="right") - 1, 0, e0.shape[0] - 1)
    u = c[k] + d[k] * jnp.clip(energy - e0[k], 0.0, h[k])
    return jnp.where(energy < e0[0], 0.0, jnp.clip(u, 0.0, 1.0))


def density(energy, intensity, at):
    """Return the normalised density (1/eV) of Geant4's sampler of a table at photon energies ``at``: 0 outside the
    table's span (never its end values). Reads the two knots of each energy's interval, not the whole table's
    ``interval_density``, so a caller inside the photon loop (a source's wavelength draw) holds no table of its
    own."""  # noqa: D205, D209
    e, s = jnp.asarray(energy, F32), jnp.asarray(intensity, F32)
    tot = jnp.sum((s[1:] + s[:-1]) * (e[1:] - e[:-1]))
    i = _interval(e, at)
    return jnp.where((at >= e[0]) & (at <= e[-1]), (s[i] + s[i + 1]) / tot, 0.0)


def interval_density(energy, intensity):
    """Return ``density`` on each knot interval of a table."""
    e, s = jnp.asarray(energy, F32), jnp.asarray(intensity, F32)
    tot = jnp.sum((s[1:] + s[:-1]) * (e[1:] - e[:-1]))
    return (s[:-1] + s[1:]) / tot


def on_intervals(energy, values, at):
    """Return the value of the knot interval holding each photon energy ``at`` (``values`` one per interval of the
    knots ``energy``): 0 outside the knots' span."""  # noqa: D205, D209
    e = jnp.asarray(energy, F32)
    return jnp.where((at >= e[0]) & (at <= e[-1]), values[_interval(e, at)], 0.0)


def _interval(e, at):
    """Return the index of the knot interval holding each photon energy ``at``."""
    return jnp.clip(jnp.searchsorted(e, at, side="right") - 1, 0, e.shape[0] - 2)


def _zero_intervals(e, s):
    """Return (each knot interval without mass (both knots 0, a width), their total width Z eV)."""
    h = e[1:] - e[:-1]
    zero = (s[1:] + s[:-1] <= 0.0) & (h > 0.0)
    return zero, jnp.sum(jnp.where(zero, h, 0.0))


def covered(energy, intensity, e):
    """Return Geant4's sampler (``g4_arrays``) of a reference spectrum under the one defensive mixture
    (``lumen.transport.estimator``): (1 - e) p_ref + e c, c uniform in energy over the knot intervals without mass (a
    live spectrum on the same knots can put light there); the sampler itself where there are none (e is not paid)."""  # noqa: D205, D209
    en, s = jnp.asarray(energy, F32), jnp.asarray(intensity, F32)
    e0, h, d, c = g4_arrays(en, s)
    zero, Z = _zero_intervals(en, s)
    d2 = (1.0 - e) * d + e * zero.astype(F32) / jnp.where(Z > 0.0, Z, 1.0)
    c2 = jnp.concatenate([jnp.zeros(1, F32), jnp.cumsum(d2 * h)[:-1]])
    return e0, h, jnp.where(Z > 0.0, d2, d), jnp.where(Z > 0.0, c2, c)


def covered_density(energy, intensity, at, e):
    """Return the density (1/eV) ``covered`` draws from at photon energies ``at`` (``density`` where nothing is
    covered). Reads the knots of each energy's interval, as ``density`` does."""  # noqa: D205, D209
    en, s = jnp.asarray(energy, F32), jnp.asarray(intensity, F32)
    p = density(en, s, at)
    zero, Z = _zero_intervals(en, s)
    inside = (at >= en[0]) & (at <= en[-1]) & zero[_interval(en, at)]
    return jnp.where(Z > 0.0, (1.0 - e) * p + e * inside.astype(F32) / jnp.where(Z > 0.0, Z, 1.0), p)


def covered_interval_density(energy, intensity, e):
    """Return ``covered_density`` on each knot interval of a table."""
    en, s = jnp.asarray(energy, F32), jnp.asarray(intensity, F32)
    p = interval_density(en, s)
    zero, Z = _zero_intervals(en, s)
    return jnp.where(Z > 0.0, (1.0 - e) * p + e * zero.astype(F32) / jnp.where(Z > 0.0, Z, 1.0), p)


@dataclass(frozen=True)
class Spectrum:
    """A spectrum table of the build (a WLSCOMPONENT), hashed by its content."""

    name: str
    energy: tuple
    intensity: tuple

    def __repr__(self):
        """Show the spectrum's name."""
        return f"Spectrum({self.name})"


def geant4_spectrum(name, energy, intensity) -> Spectrum:
    """Return a WLSCOMPONENT table as a ``Spectrum`` (Geant4's sampler: ``g4_arrays``), refusing negative intensities
    or no mass."""  # noqa: D205, D209
    e, s = np.asarray(energy, np.float64), np.asarray(intensity, np.float64)
    if np.any(s < 0) or not np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(e)) > 0:
        raise ValueError(f"spectrum {name!r}: needs a non-negative intensity with positive integral")
    return Spectrum(name, tuple(e.tolist()), tuple(s.tolist()))
