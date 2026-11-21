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

from lumen._core.safe import unit
from lumen.params import Param
from lumen.record import Photons
from lumen.sources._base import (
    COUNT,
    DIRECTION,
    F32,
    POSITION,
    ROW_COUNT,
    TIME,
    Source,
    gaussian,
    photon_uniforms,
    rows,
)
from lumen.sources._base import wavelength as draw_wavelength
from lumen.units import INF


@dataclasses.dataclass(frozen=True)
class Point(Source):
    """Photons from ``position`` (m) at ``time`` (ns), weights summing in expectation to ``n_photons``.

    ``lam`` (nm), or ``spectrum``: a Table (energies eV, intensities) drawn by Geant4's sampler of the reference
    table (the run's own by default) and reweighted to the live one (``wavelength``). ``direction`` None: isotropic
    (a normal triple per photon, normalised); else a unit vector, a beam.

    Per-row fields: ``position`` (N, 3), and ``lam``, ``time``, ``direction``, ``n_photons`` or ``label`` with a
    photon axis, make the source N rows, one photon each (a check's several sources): it emits exactly its rows,
    photon i from row i, standing for that row's ``n_photons`` (one value: every row's), and has no sample count of
    its own (``simulate(..., samples=None)``). Without them, ``samples`` photons share ``n_photons``.
    """

    position: Any
    n_photons: Any = 1.0
    lam: Any = None
    spectrum: Any = None
    direction: Any = None
    time: Any = 0.0
    structure: Any = dataclasses.field(init=False, default=None)  # (a spectrum, a beam)
    FIELDS = {"position": POSITION, "direction": DIRECTION, "n_photons": COUNT, "lam": Param("nm", (0.0, INF),
              method="moving_paths+reweight"), "spectrum": COUNT, "time": TIME}  # fmt: skip

    ROWS = {"position": 1, "direction": 1, "n_photons": 0, "lam": 0, "time": 0}
    PHOTON_ROWS = True

    def __post_init__(self):  # noqa: D105
        super().__post_init__()
        if (self.lam is None) == (self.spectrum is None):
            raise ValueError("Point: give a wavelength (lam) or a spectrum, not both")
        object.__setattr__(self, "structure", (self.spectrum is not None, self.direction is not None))

    def spectra(self) -> tuple:
        """Return ("spectrum",) for a spectrum, else ()."""
        return ("spectrum",) if self.structure[0] else ()

    def draw(self, v, ref, key, n, first_id, samples):
        """Return the Photons (see Point): photon i's uniforms are its own (``photon_uniforms``): direction, lam."""
        spectral, beam = self.structure
        rand = photon_uniforms(key, 4, n, first_id)
        d = rows(v["direction"], n, first_id, 1) if beam else unit(gaussian(rand[:, :3]))
        # one photon per row: its row's count; else the run's samples share the count
        w = rows(v["n_photons"], n, first_id, 0) if ROW_COUNT in v else v["n_photons"] / samples
        if not spectral:
            lam = rows(v["lam"], n, first_id, 0)
        else:
            lam, f = jax.vmap(lambda r_: draw_wavelength(v["spectrum"], ref["spectrum"], r_))(rand[:, 3])
            w = w * f
        return Photons(x=rows(v["position"], n, first_id, 1), d=d, t=rows(v["time"], n, first_id, 0), lam=lam,
                       w=jnp.broadcast_to(w, (n,)), pol=jnp.zeros((n, 2, 3), F32))  # fmt: skip
