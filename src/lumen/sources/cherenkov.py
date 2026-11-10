"""Cherenkov light of a straight charged segment (Frank-Tamm) with the medium's live index (``materials/<M>/RINDEX``).

    src = CherenkovSegment("Water", start=(0, 0, 0), direction=(0, 0, 1), length=0.1, beta=0.99)

cos theta = 1 / (beta n); the wavelength is drawn from the fixed proposal 1 / lam^2 on [lam_min, lam_max], so the
weight 2 pi alpha z^2 L (1 - 1 / (beta n)^2)_+ (1 / lam_min - 1 / lam_max) / samples carries the index. The cone is
sharp (no direction drawn from a density: no position derivative); the polarisation Geant4's; a photon starts
uniformly along the segment when the charge passes (random numbers: lam, pos, phi).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp

from lumen._core import frames
from lumen._core.safe import safe_sqrt
from lumen.media import laws
from lumen.sources._base import DIRECTION, INF, POSITION, TIME, WAVELENGTH, Source, photon_uniforms
from lumen.surfaces import polar
from lumen.transport.state import Emission

ALPHA, C_M_PER_NS = 1.0 / 137.035999084, 0.299792458
UNIFORMS = 3  # lam, pos, phi


def cone(u, index, start, axis, length, beta, charge, lam_min, lam_max, samples):  # noqa: PLR0913 -- the photon's pieces
    """Return one photon (x, d, lam, w, polarisation, its distance along the segment) at uniforms u.

    ``index``: the medium's RINDEX (energies eV, values); ``samples``: the photons the light is spread over.
    """
    z2 = charge * charge
    zq = 1.0 / lam_min - 1.0 / lam_max
    lam = jax.lax.stop_gradient(1.0 / (1.0 / lam_min - u[0] * (1.0 / lam_min - 1.0 / lam_max)))  # ~ 1 / lam^2
    n = laws.geant4_rindex()(dict(energy=index[0], value=index[1]), lam)
    s = 1.0 - 1.0 / (beta * beta * n * n)
    emits = s > 0.0
    w = 2.0 * math.pi * ALPHA * z2 * length * jnp.maximum(s, 0.0) * zq / samples * 1e9  # lam in nm: 1/nm -> 1/m
    cos_c = jnp.where(emits, 1.0 / (beta * n), 1.0)
    sin_c = safe_sqrt(1.0 - cos_c * cos_c)
    phi = 2.0 * math.pi * u[2]
    d = frames.to_world(jnp.stack([sin_c * jnp.cos(phi), sin_c * jnp.sin(phi), cos_c]), axis)
    xs = u[1] * length
    return start + xs * axis, d, lam, w, polar.cherenkov(d, axis), xs


@dataclasses.dataclass(frozen=True)
class CherenkovSegment(Source):
    """A charge ``charge`` (e) from ``start`` (m) along the unit ``direction`` for ``length`` (m) at ``beta``, at
    ``start`` at ``time`` (ns), in ``material``; its light on [lam_min, lam_max] (nm)."""  # noqa: D205, D209

    material: str
    start: Any = (0.0, 0.0, 0.0)
    direction: Any = (0.0, 0.0, 1.0)
    length: Any = 1.0
    beta: Any = 1.0
    charge: Any = 1.0
    time: Any = 0.0
    lam_min: Any = 200.0
    lam_max: Any = 700.0
    FIELDS = {"start": POSITION, "direction": DIRECTION, "length": ("m", (0.0, INF), "moving_paths"),
              "beta": ("", (0.0, 1.0), "moving_paths"), "charge": ("e", (-INF, INF), "reweight"), "time": TIME,
              "lam_min": WAVELENGTH, "lam_max": WAVELENGTH}  # fmt: skip
    MOVING = ("start", "direction", "length", "beta")

    def reads(self) -> tuple:
        """Return the medium's RINDEX key."""
        return (f"materials/{self.material}/RINDEX",)

    def draw(self, v, key, n, first_id, samples):
        """Return the Emission (see the module docstring)."""
        t = v[self.reads()[0]]
        a = [v[k] for k in ("start", "direction", "length", "beta", "charge", "lam_min", "lam_max")]
        x, d, lam, w, er, xs = jax.vmap(lambda u: cone(u, (t.energy, t.value), *a, samples))(
            photon_uniforms(key, UNIFORMS, n, first_id))  # fmt: skip
        tm = v["time"] + xs / (v["beta"] * C_M_PER_NS)
        return Emission(x, d, tm, lam, w, jnp.stack([er, jnp.zeros_like(er)], 1), jnp.zeros(n, bool))
