"""Scintillation light: a flash at a point, or along a charged segment (G4Scintillation's step model).

    src = ScintillationPoint.of(det.params(), "LAr", position=(0, 0, 0), energy=1.0)  # 1 MeV at t = 0

The material's light from its GDML values: SCINTILLATIONYIELD (per MeV) and two components, each a spectrum, decay
time and share (Geant4 11's SCINTILLATIONCOMPONENTi, ...TIMECONSTANTi, ...YIELDi; ``geant4="10.7"``: FAST/SLOW
COMPONENT, FAST/SLOWTIMECONSTANT, YIELDRATIO). A segment is quenched by Birks' law E / (1 + kB E / L), kB ``birks``
(m/MeV; never set by a GDML: 0); a point is not. Photons are isotropic, uniform along a segment, delayed by their
component's decay (no rise time: Geant4's default). Random numbers: lam, dir (2), comp, time[, pos].
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from lumen._core.safe import safe_sqrt
from lumen.params import Param
from lumen.sources._base import F32, POSITION, TIME, Source, photon_uniforms, proposal, wavelength
from lumen.transport.estimator import reweight
from lumen.transport.state import Emission

KEYS = {"11": ("SCINTILLATIONCOMPONENT1", "SCINTILLATIONCOMPONENT2", "SCINTILLATIONTIMECONSTANT1",
               "SCINTILLATIONTIMECONSTANT2", "SCINTILLATIONYIELD1", "SCINTILLATIONYIELD2"),
        "10.7": ("FASTCOMPONENT", "SLOWCOMPONENT", "FASTTIMECONSTANT", "SLOWTIMECONSTANT", "YIELDRATIO")}  # fmt: skip
ENERGY = Param("MeV", (0.0, float("inf")), method="reweight")
BIRKS = Param("m/MeV", (0.0, float("inf")), method="reweight")
sg = jax.lax.stop_gradient


def visible_energy(edep, length, birks):
    """Return a charged step's visible energy (MeV): Birks' dE / (1 + kB dE / dx); a zero-length step unquenched."""
    q = 1.0 + birks * edep / jnp.where(length > 0.0, length, 1.0)
    return jnp.where((length > 0.0) & (edep > 0.0), edep / q, jnp.maximum(edep, 0.0))


def flash(rand, x0, x1, t0, t1, n_ph, light, props, samples):  # noqa: PLR0913 -- the photon's pieces
    """Return one photon (x, d, t, lam, w) at uniforms rand (lam, dir (2), comp, time, and pos when x1 is given)."""
    f, taus, live = light
    fast = rand[3] < sg(f)
    (lf, wf), (ls, ws) = wavelength(props[0], live[0], rand[0]), wavelength(props[1], live[1], rand[0])
    lam, wt = sg(jnp.where(fast, lf, ls)), jnp.where(fast, wf, ws)
    z, phi = 1.0 - 2.0 * rand[1], 2.0 * math.pi * rand[2]
    r = safe_sqrt(1.0 - z * z)
    d = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z])
    tau = jnp.where(fast, taus[0], taus[1])
    w = n_ph / samples * jnp.where(fast, reweight(f, f), reweight(1.0 - f, 1.0 - f)) * wt
    if x1 is None:
        return x0, d, t0 - tau * jnp.log(rand[4]), lam, w
    return x0 + rand[5] * (x1 - x0), d, t0 + rand[5] * (t1 - t0) - tau * jnp.log(rand[4]), lam, w


@dataclasses.dataclass(frozen=True)
class _Scintillation(Source):
    material: str = ""
    geant4: str = dataclasses.field(default="11", kw_only=True)
    shares: tuple = dataclasses.field(default=(), kw_only=True)  # the share keys the material has
    proposals: tuple = dataclasses.field(default=(), kw_only=True)
    SEGMENT: ClassVar[bool] = False

    @classmethod
    def of(cls, params, material: str, *, geant4: str = "11", **fields):
        """Return the source of ``material``'s light, its spectra's proposals from ``params``' nominal tables."""
        need = [f"materials/{material}/{k}" for k in (*KEYS[geant4][:4], "SCINTILLATIONYIELD")]
        if any(k not in params for k in need):
            raise KeyError(f"{material}: no {[k for k in need if k not in params]} (Geant4 {geant4} emits none)")
        shares = tuple(k for k in KEYS[geant4][4:] if f"materials/{material}/{k}" in params)
        props = tuple(proposal(f"{material}.{i}", params[need[i]]) for i in (0, 1))
        return cls(material, **fields, geant4=geant4, shares=shares, proposals=props)

    @property
    def drawn_direction(self) -> bool:
        """Return True: the directions are isotropic."""
        return True

    def reads(self) -> tuple:
        """Return the material keys the light is read from."""
        return tuple(f"materials/{self.material}/{k}" for k in (*KEYS[self.geant4][:4], "SCINTILLATIONYIELD",
                                                                *self.shares))  # fmt: skip

    def light_of(self, v):
        """Return (yield per MeV, (fast share, decay times, live spectra)) from the values ``v``."""
        m = f"materials/{self.material}/"
        c = [v[m + k] for k in KEYS[self.geant4][:4]]
        if self.geant4 == "11":  # G4Scintillation 11: SCINTILLATIONYIELD1 defaults to 1, 2 to 0
            y1, y2 = (v.get(m + k, F32(d)) for k, d in zip(KEYS["11"][4:], (1.0, 0.0), strict=True))
            f = y1 / (y1 + y2)
        else:
            f = v.get(m + "YIELDRATIO", F32(1.0))
        return v[m + "SCINTILLATIONYIELD"], (f, (c[2], c[3]), (c[0], c[1]))

    def draw(self, v, key, n, first_id, samples):
        """Return the Emission (see the module docstring)."""
        y, light = self.light_of(v)
        if self.SEGMENT:
            x0, x1, t0, t1 = v["start"], v["end"], v["t_start"], v["t_end"]
            n_ph = y * visible_energy(v["energy"], jnp.linalg.norm(x1 - x0), v["birks"])
        else:
            x0, x1, t0, t1, n_ph = v["position"], None, v["time"], None, y * v["energy"]
        x, d, t, lam, w = jax.vmap(lambda rand: flash(rand, x0, x1, t0, t1, n_ph, light, self.proposals, samples))(
            photon_uniforms(key, 6 if self.SEGMENT else 5, n, first_id))  # fmt: skip
        return Emission(x, d, t, lam, w, jnp.zeros((n, 2, 3), F32), jnp.ones(n, bool))


@dataclasses.dataclass(frozen=True)
class ScintillationPoint(_Scintillation):
    """``energy`` (MeV) deposited at ``position`` (m) at ``time`` (ns) in ``material``; build it with ``of``."""

    position: Any = (0.0, 0.0, 0.0)
    energy: Any = 1.0
    time: Any = 0.0
    FIELDS = {"position": POSITION, "energy": ENERGY, "time": TIME}
    MOVING = ("position",)


@dataclasses.dataclass(frozen=True)
class ScintillationSegment(_Scintillation):
    """A charged step from ``start`` (time ``t_start``) to ``end`` (``t_end``) depositing ``energy`` (MeV)."""

    start: Any = (0.0, 0.0, 0.0)
    end: Any = (0.0, 0.0, 0.0)
    t_start: Any = 0.0
    t_end: Any = 0.0
    energy: Any = 1.0
    birks: Any = 0.0
    FIELDS = {"start": POSITION, "end": POSITION, "t_start": TIME, "t_end": TIME, "energy": ENERGY, "birks": BIRKS}
    MOVING = ("start", "end")
    SEGMENT = True
