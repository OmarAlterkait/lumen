"""The emission record: what an emitter gives a photon it emits, sampled by one producer.

    record = EmissionRecord("TPB.wls", TimeProfile(("delta",)))
    lam, delay, w = record.sample(live, ref, rand, lam_in)          # a re-emission of a photon absorbed at lam_in

A record is structure: its time profile's components (``TimeProfile``) and whether the emission conserves energy
(G4OpWLS: drawn at or below the absorbed photon's energy). Its values are the emitter's parameters, a dict:
``yield`` (the expected photons per absorbed one, a weight), ``spectrum`` ({"energy", "value"}: a WLSCOMPONENT or
SCINTILLATIONCOMPONENT table) and the profile's constants (``tau`` the decay or delay constant, ns; ``rise`` the rise
time, ns; ``table`` {"time", "value"} a tabulated profile). Every re-emission samples its record: a bulk wavelength
shifter's channel (``lumen.media.bulk.Channel``), a surface's re-emission (``lumen.surfaces.types.SurfaceType``) and
the scintillation sources' time profile (``lumen.sources.scintillation``). The photon continues at its new wavelength
after the delay (no photon is spawned).

The wavelength is drawn at the reference spectrum (Geant4's sampler, ``spectra.g4_arrays``; truncated at the absorbed
energy by its exact inverse CDF for an energy-conserving record) and weighted by the live density over the reference
one (exactly 1 at the reference), so the spectrum has a derivative; Geant4's chance of finding an energy below the
absorbed one (``spectra.kept``) is part of that weight. The delay is reparametrised (a draw is a function of the
uniforms and the live constants), so it is differentiable in them.

Time-profile components (summed, one uniform each, in order):

    "delta"     tau: a fixed delay (G4OpWLS's default profile)
    "decay"     Exp(tau) (G4OpWLS's exponential profile; G4Scintillation's decay)
    "rise"      the formation of a decay tau with rise time ``rise``: Geant4's finite-rise-time profile
                (G4Scintillation::sample_time, density (tau_r + tau) / tau^2 e^(-t / tau) (1 - e^(-t / tau_r))) is the
                sum of Exp(tau) and Exp(tau tau_r / (tau + tau_r)), so "rise" adds the second (a profile ("decay",
                "rise") is the whole of it; rise 0 adds 0: the plain decay, as Geant4 at rise time 0)
    "table"     a tabulated profile {"time", "value"}: drawn as Geant4 draws a table (uniform within each knot
                interval, with its trapezoid mass)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen._core.safe import safe_div
from lumen.media import spectra
from lumen.rules.choices import DEFENSIVE_SHARE
from lumen.units import HC

sg = jax.lax.stop_gradient
F32 = jnp.float32
COMPONENTS = ("delta", "decay", "rise", "table")


def _component(kind, p, u):
    if kind == "delta":
        return p["tau"] + 0.0 * u
    if kind == "decay":
        return -p["tau"] * jnp.log(u)
    if kind == "rise":
        tau, rise = p["tau"], p["rise"]
        s = tau + rise
        return -(tau * rise / jnp.where(s > 0.0, s, 1.0)) * jnp.log(u)
    return spectra.invert(spectra.g4_arrays(p["table"]["time"], p["table"]["value"]), u)  # Geant4's sampler of a table


@dataclass(frozen=True)
class TimeProfile:
    """A time profile: its components (``COMPONENTS``), summed; one uniform each."""

    components: tuple[str, ...]

    def __post_init__(self):
        """Refuse an unknown component."""
        bad = [c for c in self.components if c not in COMPONENTS]
        if bad or not self.components:
            raise ValueError(f"TimeProfile: components from {COMPONENTS}, got {self.components}")

    def sample(self, p, rand):
        """Return the delay (ns) at the profile's constants ``p`` and uniforms ``rand`` (one per component)."""
        t = _component(self.components[0], p, rand[0])
        for k, c in enumerate(self.components[1:], 1):
            t = t + _component(c, p, rand[k])
        return t


class Tables(NamedTuple):
    """What an emission reads of a record's values, none of it the photon's (``EmissionRecord.tables``): the live
    sampler (``spectra.g4_arrays``) and density per knot interval, and the sampler drawn from (the reference under the
    one defensive mixture, ``spectra.covered``) and its density per knot interval, each with its knots (eV)."""  # noqa: D205, D209

    live: tuple
    live_density: jax.Array
    live_knots: jax.Array
    drawn: tuple
    drawn_density: jax.Array
    drawn_knots: jax.Array


class Emitted(NamedTuple):
    """An emission (``EmissionRecord.sample``): wavelength (nm), delay (ns) and weight, and the weight at the reference
    values (stop-gradient: what the reference weight takes)."""  # noqa: D205, D209

    lam: jax.Array
    delay: jax.Array
    w: jax.Array
    w_r: jax.Array


@dataclass(frozen=True, eq=False)
class EmissionRecord:
    """An emitter's record (see the module docstring): ``profile`` its time profile, ``conserving`` drawn at or below
    the absorbed energy (G4OpWLS), ``spectrum`` a constant table of the build (``spectra.Spectrum``) when the emitter
    has no spectrum parameter: the step passes its table as an operand in the values' ``spectrum`` (None: the
    emitter's parameter)."""  # noqa: D205, D209

    name: str
    profile: TimeProfile
    conserving: bool = True
    spectrum: spectra.Spectrum | None = None

    def __repr__(self):
        """Show the record's name."""
        return f"EmissionRecord({self.name})"

    def tables(self, live, ref) -> Tables:
        """Return the Tables of the record's values ``live`` and ``ref`` (the reference's stop-gradient)."""
        spec_l, spec_r = self._spectrum(live), tuple(sg(x) for x in self._spectrum(ref))
        return Tables(live=spectra.g4_arrays(*spec_l), live_density=spectra.interval_density(*spec_l),
                      live_knots=spec_l[0], drawn=spectra.covered(*spec_r, DEFENSIVE_SHARE),
                      drawn_density=spectra.covered_interval_density(*spec_r, DEFENSIVE_SHARE),
                      drawn_knots=spec_r[0])  # fmt: skip

    def sample(self, live, ref, rand, lam_in, tables=None) -> Emitted:
        """Return the Emitted of an emission from a photon absorbed at ``lam_in`` (nm).

        ``live``, ``ref``: the record's values (module docstring) at the live and the reference values; ``rand``:
        (the wavelength's uniform, then one per profile component); ``tables``: their ``tables`` (None: made here).
        The weight is the yield times Geant4's chance of an energy below the absorbed one times the live / reference
        density of the drawn wavelength; the reference's weight is the same function at the reference values
        (stop-gradient).
        """
        t = self.tables(live, ref) if tables is None else tables
        arr_q = t.drawn  # the one defensive mixture over its massless intervals
        f_q = spectra.cumulative(arr_q, sg(lam_in)) if self.conserving else F32(1.0)
        lam = sg(spectra.sample(arr_q, rand[0] * f_q))
        energy = HC / lam
        q = spectra.on_intervals(t.drawn_knots, t.drawn_density, energy)
        dens_q = sg(safe_div(q, f_q))

        def weight(v, arrays, dens):  # the yield times the density over the one drawn from, at the values v
            if self.conserving:  # truncated at the absorbed energy (an absorbed energy below the spectrum: weight 0)
                f = spectra.cumulative(arrays, sg(lam_in))
                dens = spectra.kept(f) * safe_div(dens, f)
            return jnp.asarray(v["yield"], F32) * safe_div(dens, dens_q)

        # the reference weight: the same at the reference values with the density drawn from over itself
        ref_v = jax.tree_util.tree_map(sg, ref)
        w_r = weight(ref_v, arr_q, q)
        return Emitted(lam=lam, delay=self.profile.sample(live, rand[1:]),
                       w=weight(live, t.live, spectra.on_intervals(t.live_knots, t.live_density, energy)),
                       w_r=sg(w_r))  # fmt: skip

    @staticmethod
    def _spectrum(v):
        return jnp.asarray(v["spectrum"]["energy"], F32), jnp.asarray(v["spectrum"]["value"], F32)


def wls_profile(profile: str) -> TimeProfile:
    """Return G4OpWLS's time profile by its name: "delta" (Geant4's default) or "exponential"."""
    return TimeProfile({"delta": ("delta",), "exponential": ("decay",)}[profile])
