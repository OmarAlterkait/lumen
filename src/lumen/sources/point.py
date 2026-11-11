"""A point source: isotropic or a beam, one wavelength or a spectrum, emitted at ``time``.

    src = Point((0.0, 0.0, 0.0), n_photons=1e6, lam=400.0)  # isotropic, monochromatic
    src = Point(position, n_photons=1e4, spectrum=Table(e_ev, s), direction=(0.0, 0.0, 1.0))  # a beam, a spectrum

``Point.draw`` builds the photons ``emit`` returns (``lumen.sources``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from lumen.params import Param
from lumen.sources._base import COUNT, DIRECTION, F32, POSITION, TIME, Source, gaussian, proposal, rows, uniforms
from lumen.sources._base import wavelength as draw_wavelength
from lumen.transport.state import Emission

LAM_STREAM = 1  # a spectrum's uniforms come from fold_in(key, LAM_STREAM)


@dataclasses.dataclass(frozen=True)
class Point(Source):
    """Photons from ``position`` (m) at ``time`` (ns), weights summing in expectation to ``n_photons``.

    ``lam`` (nm), or ``spectrum``: a Table (energies eV, intensities) drawn by Geant4's sampler from the nominal
    table and reweighted to the live one. ``direction`` None: isotropic (a normal triple per photon,
    normalised); else a unit vector, a beam. ``position`` (N, 3), and ``lam``, ``time`` or ``direction``
    with a photon axis, give photon i of a run its own row (a check's several sources).
    """

    position: Any
    n_photons: Any = 1.0
    lam: Any = None
    spectrum: Any = None
    direction: Any = None
    time: Any = 0.0
    structure: Any = dataclasses.field(init=False, default=None)  # (the spectrum's proposal, a beam)
    FIELDS = {"position": POSITION, "direction": DIRECTION, "n_photons": COUNT, "lam": Param("nm", (0.0, float("inf")),
              method="moving_paths+reweight"), "spectrum": COUNT, "time": TIME}  # fmt: skip
    MOVING = ("position", "direction", "lam")

    def __post_init__(self):  # noqa: D105
        if (self.lam is None) == (self.spectrum is None):
            raise ValueError("Point: give a wavelength (lam) or a spectrum, not both")
        object.__setattr__(self, "structure", (proposal("source", self.spectrum), self.direction is not None))

    @property
    def drawn_direction(self) -> bool:
        """Return True for an isotropic point (its directions are drawn), False for a beam."""
        return not self.structure[1]

    def draw(self, v, key, n, first_id, samples):
        """Return the Emission (see Point)."""
        prop, beam = self.structure
        g = None if beam else gaussian(uniforms(key, 3, n, first_id))
        d = rows(v["direction"], n, first_id, 1) if beam else g / jnp.linalg.norm(g, axis=1, keepdims=True)
        w = v["n_photons"] / samples
        if prop is None:
            lam = rows(v["lam"], n, first_id, 0)
        else:
            rand = uniforms(jax.random.fold_in(key, LAM_STREAM), 1, n, first_id)[:, 0]
            lam, f = jax.vmap(lambda r_: draw_wavelength(prop, v["spectrum"], r_))(rand)
            w = w * f
        return Emission(rows(v["position"], n, first_id, 1), d, rows(v["time"], n, first_id, 0), lam,
                        jnp.broadcast_to(w, (n,)), jnp.zeros((n, 2, 3), F32), jnp.full(n, not beam))  # fmt: skip
