"""Cherenkov light of a straight charged segment (Frank-Tamm) with the medium's live index (``materials/<M>/RINDEX``).

    src = CherenkovSegment("Water", start=(0, 0, 0), direction=(0, 0, 1), length=0.1, beta=0.99)

cos theta = 1 / (beta n); the wavelength is drawn from the fixed proposal 1 / lam^2 (uniform in energy) over the span
of the medium's RINDEX table, as G4Cerenkov emits over [RINDEX's first energy, its last] (``bounds``; a table of one
knot spans none: no light, as Geant4's photon number is then 0). The weight
2 pi alpha z^2 L (1 - 1 / (beta n)^2)_+ (1 / lam_min - 1 / lam_max) / samples carries the index. The cone is sharp;
the polarisation Geant4's; a photon starts uniformly along the segment when the charge passes (random numbers: lam,
pos, phi).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from lumen._core import frames, rng
from lumen._core.safe import safe_sqrt, sanitize
from lumen.media import laws
from lumen.params import Param
from lumen.record import Photons
from lumen.sources._base import DIRECTION, INF, POSITION, TIME, Source, photon_uniforms
from lumen.surfaces import polar
from lumen.units import C_VACUUM, FINE_STRUCTURE, HC, NM_PER_M, TWO_PI

LAYOUT = rng.Layout(lam=1, pos=1, phi=1)  # a photon's uniforms: its wavelength, place along the segment, azimuth


def cone(rand, index, start, axis, length, beta, charge, lam_min, lam_max, samples):  # noqa: PLR0913 -- the photon's pieces
    """Return one photon (x, d, lam, w, polarisation, its distance along the segment) at uniforms rand (``LAYOUT``).

    ``index``: the medium's RINDEX (energies eV, values); ``samples``: the photons the light is spread over.
    """
    z2 = charge * charge
    zq = 1.0 / lam_min - 1.0 / lam_max
    lam = jax.lax.stop_gradient(1.0 / (1.0 / lam_min - rand["lam"][0] * zq))  # ~ 1 / lam^2; zq = 0 (one knot): no light
    n = laws.geant4_rindex()(dict(energy=index[0], value=index[1]), lam)
    # below threshold (beta n <= 1, beta 0 included) the photon carries weight 0 and a tangent of exactly 0: 1 / (beta
    # n) is taken only where it emits (at beta n -> 0 its tangent overflows, and a max's mask times it is NaN)
    emits = jax.lax.stop_gradient(1.0 - 1.0 / (beta * beta * n * n)) > 0.0
    b_, n_ = sanitize(emits, beta, 1.0), sanitize(emits, n, 2.0)
    s = jnp.where(emits, 1.0 - 1.0 / (b_ * b_ * n_ * n_), 0.0)
    w = TWO_PI * FINE_STRUCTURE * z2 * length * s * zq / samples * NM_PER_M  # lam in nm: 1/nm -> 1/m
    cos_c = jnp.where(emits, 1.0 / (b_ * n_), 1.0)
    sin_c = safe_sqrt(1.0 - cos_c * cos_c)
    phi = TWO_PI * rand["phi"][0]
    d = frames.to_world(jnp.stack([sin_c * jnp.cos(phi), sin_c * jnp.sin(phi), cos_c]), axis)
    xs = rand["pos"][0] * length
    return start + xs * axis, d, lam, w, polar.cherenkov(d, axis), xs


def bounds(index):
    """Return (lam_min, lam_max) nm: the span of the RINDEX table ``index`` (energies eV, values), the range G4Cerenkov
    emits over (G4Cerenkov 10.7.4 PostStepDoIt: Pmin = Rindex->GetMinLowEdgeEnergy(), Pmax = GetMaxLowEdgeEnergy();
    one knot: Pmin = Pmax, no light)."""  # noqa: D205, D209
    e = jnp.atleast_1d(jax.lax.stop_gradient(jnp.asarray(index[0])))
    return HC / jnp.max(e), HC / jnp.min(e)


@dataclasses.dataclass(frozen=True)
class CherenkovSegment(Source):
    """A charge ``charge`` (e) from ``start`` (m) along the unit ``direction`` for ``length`` (m) at ``beta``, at
    ``start`` at ``time`` (ns), in ``material``; its light over the RINDEX table's span."""  # noqa: D205, D209

    material: str
    start: Any = (0.0, 0.0, 0.0)
    direction: Any = (0.0, 0.0, 1.0)
    length: Any = 1.0
    beta: Any = 1.0
    charge: Any = 1.0
    time: Any = 0.0
    FIELDS = {"start": POSITION, "direction": DIRECTION, "length": Param("m", (0.0, INF), method="moving_paths"),
              "beta": Param("", (0.0, 1.0), method="moving_paths"),
              "charge": Param("e", (-INF, INF), method="reweight"),
              "time": TIME}  # fmt: skip

    def reads(self) -> tuple:
        """Return the medium's RINDEX key."""
        return (f"materials/{self.material}/RINDEX",)

    def draw(self, v, ref, key, n, first_id, samples):
        """Return the Photons (see the module docstring)."""
        t = v[self.reads()[0]]
        a = [v[k] for k in ("start", "direction", "length", "beta", "charge")] + list(bounds((t.energy, t.value)))
        x, d, lam, w, er, xs = jax.vmap(lambda r: cone(LAYOUT.split(r), (t.energy, t.value), *a, samples))(
            photon_uniforms(key, LAYOUT.n, n, first_id))  # fmt: skip
        tm = v["time"] + xs / (v["beta"] * C_VACUUM)
        return Photons(x=x, d=d, t=tm, lam=lam, w=w, pol=jnp.stack([er, jnp.zeros_like(er)], 1))


def photons_per_length(index, beta):
    """Return G4Cerenkov's mean photon number per length at ``beta`` (GetAverageNumberOfPhotons, Geant4 10.7.4), but
    for the constant 369.81 / (eV cm) z^2: dp - CAI / beta^2 over the energies where beta n > 1, CAI the trapezoid
    integral of 1 / n^2 on the RINDEX knots (BuildPhysicsTable), the threshold energy the table's inverse
    interpolation (G4PhysicsVector::GetEnergy, n ascending in energy). Values only (stop-gradient)."""  # noqa: D205, D209
    e, n = (jnp.atleast_1d(jax.lax.stop_gradient(jnp.asarray(x))) for x in index)
    inv2 = 1.0 / (n * n)
    cai = jnp.concatenate([jnp.zeros(1, e.dtype), jnp.cumsum(jnp.diff(e) * 0.5 * (inv2[1:] + inv2[:-1]))])
    binv = 1.0 / beta
    p_thr = jnp.interp(binv, n, e)
    full = jnp.min(n) > binv
    dp = jnp.where(full, e[-1] - e[0], e[-1] - p_thr)
    ge = jnp.where(full, cai[-1], cai[-1] - jnp.interp(p_thr, e, cai))
    return jnp.where((jnp.max(n) < binv) | (beta <= 0.0), 0.0, dp - ge * binv * binv)


def along_step(u, n1, n2):
    """Return the fraction along a step at uniform ``u`` drawn from the density linear between the pre-step and
    post-step photon numbers ``n1``, ``n2`` (G4Cerenkov's rejection sampling, by its inverse CDF)."""  # noqa: D205, D209
    den = n1 + jnp.sqrt(n1 * n1 + u * (n2 * n2 - n1 * n1))
    return jnp.where(den > 0.0, u * (n1 + n2) / jnp.where(den > 0.0, den, 1.0), u)
