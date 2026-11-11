"""The light of a Geant4 charged-step dump, Cherenkov or scintillation, one step per photon.

    src = Geant4Steps.of(det.params(), "Water", "cherenkov", start=a, end=b, t_start=t0, t_end=t1,
                         beta_start=b0, beta_end=b1, charge=q, edep=e)   # (n, 3) m, ns, c, e, eV
    src = Geant4Steps.of(..., step_length=L)                             # (n,) m: Geant4's StepLength

A step's speed is the mean of its ends (G4Cerenkov's). Its light counts over ``step_length`` when given (Geant4's
G4Step::GetStepLength, the true path: G4Cerenkov's photon number and the Birks dE/dx use it), else over the chord;
its photons are emitted along the chord. A photon draws step i with probability
p_i proportional to a fixed reference light e_i (Cherenkov L q^2 (1 - 1 / (beta n_max)^2)_+; scintillation the
visible energy) and is one photon of that step as the segment sources emit it, weighted by 1 / (samples p_i).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from lumen._core.safe import safe_div
from lumen.params import Param
from lumen.sources import scintillation as sc
from lumen.sources._base import F32, INF, POSITION, TIME, WAVELENGTH, photon_uniforms
from lumen.sources.cherenkov import UNIFORMS, CherenkovSegment, cone
from lumen.transport.state import Emission

sg = jax.lax.stop_gradient
SPEED = Param("", (0.0, 1.0), method="moving_paths")


@dataclasses.dataclass(frozen=True)
class Geant4Steps(sc._Scintillation):
    """A step dump's light (``light``: "cherenkov" or "scintillation") in ``material``; build it with ``of``."""

    start: Any = None
    end: Any = None
    t_start: Any = None
    t_end: Any = None
    beta_start: Any = None
    beta_end: Any = None
    charge: Any = None
    edep: Any = None
    step_length: Any = None
    birks: Any = 0.0
    lam_min: Any = 200.0
    lam_max: Any = 700.0
    light: str = dataclasses.field(default="cherenkov", kw_only=True)
    FIELDS = {"start": POSITION, "end": POSITION, "t_start": TIME, "t_end": TIME, "beta_start": SPEED,
              "beta_end": SPEED, "charge": CherenkovSegment.FIELDS["charge"],
              "edep": dataclasses.replace(sc.ENERGY, unit="eV"),
              "step_length": Param("m", (0.0, INF), method="reweight"),  # the light count only
              "birks": sc.BIRKS, "lam_min": WAVELENGTH, "lam_max": WAVELENGTH}  # fmt: skip
    MOVING = ("start", "end", "beta_start", "beta_end")

    @classmethod
    def of(cls, params, material: str, light: str = "cherenkov", *, geant4: str = "11", **fields):
        """Return the source; its scintillation is read from ``params`` as ``ScintillationPoint.of`` reads it."""
        if light not in ("cherenkov", "scintillation"):
            raise ValueError(f"Geant4Steps: light is 'cherenkov' or 'scintillation', got {light!r}")
        if light == "cherenkov":
            return cls(material, **fields, light=light)
        return super().of(params, material, geant4=geant4, light=light, **fields)

    @property
    def drawn_direction(self) -> bool:
        """Return True for scintillation (isotropic), False for Cherenkov (a sharp cone)."""
        return self.light == "scintillation"

    def reads(self) -> tuple:
        """Return the material keys the light is read from."""
        return (f"materials/{self.material}/RINDEX",) if self.light == "cherenkov" else super().reads()

    def draw(self, v, key, n, first_id, samples):
        """Return the Emission (see the module docstring)."""
        a, b, t0, t1, q = v["start"], v["end"], v["t_start"], v["t_end"], v["charge"]
        length, beta = jnp.linalg.norm(b - a, axis=1), 0.5 * (v["beta_start"] + v["beta_end"])
        true = v.get("step_length")  # Geant4's step length: the light counts over it (None: the chord)
        path = length if true is None else true
        cher = self.light == "cherenkov"
        if cher:
            index = v[self.reads()[0]]
            n_max = sg(jnp.max(index.value))
            e = sg(path * q * q * jnp.maximum(1.0 - 1.0 / (beta * beta * n_max * n_max), 0.0))
        else:
            y, light = self.light_of(v)
            vis = sc.visible_energy(v["edep"] * 1e-6, path, v["birks"])
            e = sg(vis)
        rand = photon_uniforms(key, 1 + (UNIFORMS if cher else 6), n, first_id)
        i = jnp.clip(jnp.searchsorted(jnp.cumsum(e) / jnp.sum(e), rand[:, 0], side="right"), 0, e.shape[0] - 1)
        scale = safe_div(jnp.sum(e), samples * e[i])
        if cher:
            axis = (b - a) / jnp.where(length > 0.0, length, 1.0)[:, None]
            x, d, lam, w, er, _ = jax.vmap(lambda u_, j: cone(u_, (index.energy, index.value), a[j], axis[j], length[j],
                                                              beta[j], q[j], v["lam_min"], v["lam_max"], 1.0))(
                rand[:, 1:], i)  # fmt: skip
            t, pol = t0[i] + rand[:, 2] * (t1[i] - t0[i]), jnp.stack([er, jnp.zeros_like(er)], 1)
            if true is not None:  # the cone counts over the chord it emits along: rescale to the true length
                w = w * safe_div(true[i], length[i])
        else:
            x, d, t, lam, w = jax.vmap(lambda u_, j: sc.flash(u_, a[j], b[j], t0[j], t1[j], y * vis[j], light,
                                                              self.proposals, 1.0))(rand[:, 1:], i)  # fmt: skip
            pol = jnp.zeros((n, 2, 3), F32)
        return Emission(x, d, t, lam, w * scale, pol, jnp.full(n, not cher))
