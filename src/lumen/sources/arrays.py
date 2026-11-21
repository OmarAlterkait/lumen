"""Photons given as arrays, one row per photon: ``FromArrays(position, direction, time, lam, weight)``.

A run traces each row once (``simulate(det, FromArrays(...), samples=None)``): the rows are the photons, so there is
no separate sample count. Every per-row field has the same rows (refused otherwise); a field given once is every row's.

position, direction (N, 3) m and unit, time (N,) ns, lam (N,) nm, weight (N,) the expected photons each stands for (used
as given), ``polarisation`` (N, 2, 3) the Jones vector's real and imaginary parts (None: unpolarised). A tangent on
the positions, directions or wavelengths moves where the light starts: refused (no derivative in this version).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp

from lumen.params import Param
from lumen.record import Photons
from lumen.sources._base import COUNT, DIRECTION, INF, POSITION, TIME, Source, rows


@dataclasses.dataclass(frozen=True)
class FromArrays(Source):
    """Photons from arrays (see the module docstring); emits rows first_id .. first_id + n - 1."""

    position: Any
    direction: Any
    time: Any
    lam: Any
    weight: Any
    polarisation: Any = None
    # lam moves the photons it is given (a tangent on it moves them, as on position and direction); the polarisation
    # is carried, never moved, and has no range
    FIELDS = {"position": POSITION, "direction": DIRECTION, "time": TIME,
              "lam": Param("nm", (0.0, INF), method="moving_paths"), "weight": COUNT,
              "polarisation": Param("")}  # fmt: skip

    def draw(self, v, ref, key, n, first_id, samples):
        """Return the rows (``key`` and ``samples`` are not used)."""
        pol = v.get("polarisation", jnp.zeros((v["position"].shape[0], 2, 3), jnp.float32))
        cols = {"x": ("position", 1), "d": ("direction", 1), "t": ("time", 0), "lam": ("lam", 0), "w": ("weight", 0)}
        return Photons(**{f: rows(v[k], n, first_id, r) for f, (k, r) in cols.items()}, pol=rows(pol, n, first_id, 2))

    ROWS = {"position": 1, "direction": 1, "time": 0, "lam": 0, "weight": 0, "polarisation": 2}
    PHOTON_ROWS = REQUIRES_ROWS = True  # a row is one photon, traced once: the rows are the run (no sample count)

    def _prepared(self, params, reference=None):
        """Return the prepared rows (drawn at no reference)."""
        return super()._prepared(params)
