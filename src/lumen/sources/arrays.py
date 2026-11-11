"""Photons given as arrays (an external list), one row per photon: ``FromArrays(x, d, t, lam, w)``.

x, d (N, 3) m and unit, t (N,) ns, lam (N,) nm, w (N,) the expected photons each stands for (used as given),
``polarisation`` (N, 2, 3) the Jones vector's real and imaginary parts (None: unpolarised). ``drawn``: the list's
directions were drawn from a density in solid angle (default False: an external list gives no source-position
derivative); a derivative that moves them then needs ``dlogp`` (N, 3), that density's log gradient in d (zeros for an
isotropic list).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp

from lumen.sources._base import COUNT, DIRECTION, POSITION, TIME, WAVELENGTH, GradientNotAvailable, Source, rows
from lumen.transport.state import Emission


@dataclasses.dataclass(frozen=True)
class FromArrays(Source):
    """Photons from arrays (see the module docstring); emits rows first_id .. first_id + n - 1."""

    x: Any
    d: Any
    t: Any
    lam: Any
    w: Any
    polarisation: Any = None
    drawn: bool = dataclasses.field(default=False, kw_only=True)
    dlogp: Any = dataclasses.field(default=None, kw_only=True)
    FIELDS = {"x": POSITION, "d": DIRECTION, "t": TIME, "lam": WAVELENGTH, "w": COUNT, "polarisation": DIRECTION,
              "dlogp": DIRECTION}  # fmt: skip
    MOVING = ("x", "d", "lam")

    @property
    def drawn_direction(self) -> bool:
        """Return whether the list's directions were drawn from a density (``drawn``)."""
        return self.drawn

    def draw(self, v, key, n, first_id, samples):
        """Return the rows (``key`` and ``samples`` are not used)."""
        pol = v.get("polarisation", jnp.zeros((v["x"].shape[0], 2, 3), jnp.float32))
        return Emission(*(rows(v[k], n, first_id, r) for k, r in (("x", 1), ("d", 1), ("t", 0), ("lam", 0), ("w", 0))),
                        rows(pol, n, first_id, 2), jnp.full(n, self.drawn),
                        rows(v["dlogp"], n, first_id, 1) if "dlogp" in v else None)  # fmt: skip

    def emit(self, params, key, n: int, first_id=0, samples=None, *, moving=False):
        """Return the rows; a derivative that moves drawn rows needs their density's gradient (``dlogp``)."""
        if moving and self.dlogp is None:
            raise GradientNotAvailable("FromArrays(drawn=True) moved without its direction density's gradient: pass "
                                       "dlogp=jnp.zeros((N, 3)) when the directions are isotropic, else grad_d log p "
                                       "of their density (or write a source function, lumen.source)")  # fmt: skip
        return super().emit(params, key, n, first_id, samples, moving=moving)
