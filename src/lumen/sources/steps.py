"""The light of a Geant4 charged-step dump, one step per photon: ``CherenkovSteps`` and ``ScintillationSteps``.

    src = CherenkovSteps("Water", start=a, end=b, t_start=t0, t_end=t1, beta_start=b0, beta_end=b1,
                         charge=q)                                        # (n, 3) m, (n,) ns, c, e
    src = ScintillationSteps.of(det.params(), "LS", start=a, end=b, t_start=t0, t_end=t1, edep=e)   # (n,) MeV
    src = CherenkovSteps(..., step_length=L)                              # (n,) m: Geant4's StepLength

A step's light counts over ``step_length`` when given (Geant4's G4Step::GetStepLength, the true path: G4Cerenkov's
photon number and the Birks dE/dx use it), else over the chord; its photons are emitted along the chord. A photon
draws step i with probability p_i proportional to a fixed reference light e_i (Cherenkov L q^2 (1 - 1 / (beta
n_max)^2)_+, scintillation the visible energy) and is one photon of that step as the segment sources emit it, weighted
by 1 / (samples p_i). A ``label`` per step (int, (n,)) is the label of the photons drawn from it (one label: all of
them).

Cherenkov: a step's speed is the mean of its ends (G4Cerenkov's). Its photons are placed as G4Cerenkov 10.7.4 places
them (PostStepDoIt): at a fraction f of the step drawn from the density linear between the photon numbers at the pre-
and post-step speeds (``cherenkov.photons_per_length``), at time t_start + f L / (v_pre + f (v_post - v_pre) / 2).
Scintillation: the material's light read as ``ScintillationSegment.of`` reads it, the energy deposited ``edep`` (MeV,
Geant4's unit) quenched by Birks' law over the step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from lumen._core import rng
from lumen._core.safe import safe_div
from lumen.params import Param
from lumen.record import Photons
from lumen.sources import scintillation as sc
from lumen.sources._base import F32, INF, POSITION, TIME, Source, lit, photon_uniforms
from lumen.sources.cherenkov import LAYOUT, CherenkovSegment, along_step, bounds, cone, photons_per_length
from lumen.units import C_VACUUM

sg = jax.lax.stop_gradient
SPEED = Param("", (0.0, 1.0), method="moving_paths")
STEP_FIELDS = {
    "start": POSITION,
    "end": POSITION,
    "t_start": TIME,
    "t_end": TIME,
    "step_length": Param("m", (0.0, INF), method="reweight"),
}  # fmt: skip -- step_length: the light count only
STEP_ROWS = {"start": 1, "end": 1, "t_start": 0, "t_end": 0, "step_length": 0}  # a row is a step


def _paths(v):
    """Return (each step's chord length, the length its light counts over: ``step_length``, else the chord) (m)."""
    length = jnp.linalg.norm(v["end"] - v["start"], axis=1)
    true = v.get("step_length")
    return length, length if true is None else true


def _steps(e, u, samples):
    """Return (each photon's step, its weight factor 1 / (samples p_i)): the step drawn by its reference light ``e``.

    Never a step without light (``lit``): one drawn (a step at rest, beta 0) had weight 0, time 0 / 0 = NaN and a NaN
    index tangent, and with a drawn direction deposited NaN (lumen-compare: about 2 photons in 1e6).
    """
    cdf = jnp.cumsum(e)
    i = lit(jnp.searchsorted(cdf, u * cdf[-1], side="right"), e)
    return i, safe_div(jnp.sum(e), samples * e[i])


def _labels(v, i, n):
    """Return each photon's label: its step's (None: no labels)."""
    lab = v.get("label")
    return None if lab is None else jnp.broadcast_to(lab, (n,)) if jnp.ndim(lab) == 0 else jnp.asarray(lab)[i]


@dataclasses.dataclass(frozen=True)
class CherenkovSteps(Source):
    """A step dump's Cherenkov light in ``material`` (see the module docstring): ``start``, ``end`` (n, 3) m,
    ``t_start``, ``t_end`` (n,) ns, ``beta_start``, ``beta_end`` (n,), ``charge`` (n,) e, ``step_length`` (n,) m
    or None; its light over the RINDEX table's span."""  # noqa: D205, D209

    material: str = ""
    start: Any = None
    end: Any = None
    t_start: Any = None
    t_end: Any = None
    beta_start: Any = None
    beta_end: Any = None
    charge: Any = None
    step_length: Any = None
    FIELDS = {**STEP_FIELDS, "beta_start": SPEED, "beta_end": SPEED, "charge": CherenkovSegment.FIELDS["charge"]}
    ROWS = {**STEP_ROWS, "beta_start": 0, "beta_end": 0, "charge": 0}
    LIGHTLESS = ("charge",)  # a padding step: q^2 = 0

    def reads(self) -> tuple:
        """Return the medium's RINDEX key."""
        return (f"materials/{self.material}/RINDEX",)

    def _reference_light(self, r):
        """Return each step's light at the values ``r`` (the reference: the step choice's proposal), stop-gradient:
        L q^2 (1 - 1 / (beta n_max)^2)_+."""  # noqa: D205, D209
        r = sg(r)
        _, path = _paths(r)
        beta, n_max = 0.5 * (r["beta_start"] + r["beta_end"]), jnp.max(r[self.reads()[0]].value)
        return path * r["charge"] ** 2 * jnp.maximum(1.0 - 1.0 / (beta * beta * n_max * n_max), 0.0)

    def draw(self, v, ref, key, n, first_id, samples):
        """Return the Photons (see the module docstring); the step is chosen by its light at the reference ``ref``."""
        a, b, t0, q = v["start"], v["end"], v["t_start"], v["charge"]
        length, path = _paths(v)
        beta = 0.5 * (v["beta_start"] + v["beta_end"])
        lay = rng.Layout(step=1, **dict(LAYOUT.sizes))
        rand = lay.split(photon_uniforms(key, lay.n, n, first_id))
        i, scale = _steps(self._reference_light(ref), rand["step"][:, 0], samples)
        index = v[self.reads()[0]]
        axis = (b - a) / jnp.where(length > 0.0, length, 1.0)[:, None]
        tab = (index.energy, index.value)
        lo, hi = bounds(tab)
        b0, b1 = sg(v["beta_start"]), sg(v["beta_end"])
        n0, n1 = (
            jax.vmap(lambda x: photons_per_length(tab, x))(b0),
            jax.vmap(lambda x: photons_per_length(tab, x))(b1),
        )
        f = along_step(rand["pos"][:, 0], n0[i], n1[i])  # G4Cerenkov's place along the step
        x, d, lam, w, er, _ = jax.vmap(lambda u_, j: cone(u_, tab, a[j], axis[j], length[j], beta[j], q[j], lo, hi,
                                                          1.0))(dict(rand, pos=f[:, None]), i)  # fmt: skip
        vel = C_VACUUM * (b0[i] + f * (b1[i] - b0[i]) * 0.5)
        t, pol = t0[i] + f * path[i] / vel, jnp.stack([er, jnp.zeros_like(er)], 1)
        if (
            v.get("step_length") is not None
        ):  # the cone counts over the chord it emits along: rescale to the true length
            w = w * safe_div(path[i], length[i])
        return Photons(x=x, d=d, t=t, lam=lam, w=w * scale, pol=pol, label=_labels(v, i, n))


@dataclasses.dataclass(frozen=True)
class ScintillationSteps(sc._Scintillation):
    """A step dump's scintillation light in ``material`` (see the module docstring): ``start``, ``end`` (n, 3) m,
    ``t_start``, ``t_end`` (n,) ns, ``edep`` (n,) MeV, ``step_length`` (n,) m or None, Birks' ``birks`` (m/MeV);
    build it with ``of``."""  # noqa: D205, D209

    start: Any = None
    end: Any = None
    t_start: Any = None
    t_end: Any = None
    edep: Any = None
    step_length: Any = None
    birks: Any = 0.0
    FIELDS = {**STEP_FIELDS, "edep": sc.ENERGY, "birks": sc.BIRKS}
    ROWS = {**STEP_ROWS, "edep": 0}
    LIGHTLESS = ("edep",)  # a padding step: no energy

    def _reference_light(self, r):
        """Return each step's visible energy at the values ``r`` (the reference: the step choice's proposal),
        stop-gradient."""  # noqa: D205, D209
        r = sg(r)
        return sc.visible_energy(r["edep"], _paths(r)[1], r["birks"])

    def draw(self, v, ref, key, n, first_id, samples):
        """Return the Photons (see the module docstring); the step is chosen by its light at the reference ``ref``."""
        a, b, t0, t1 = v["start"], v["end"], v["t_start"], v["t_end"]
        y, light = self.light_of(v)
        light_ref = self.light_of(ref)[1]
        vis = sc.visible_energy(v["edep"], _paths(v)[1], v["birks"])
        lay = rng.Layout(step=1, **dict(sc.layout(True, self.rises).sizes))
        rand = lay.split(photon_uniforms(key, lay.n, n, first_id))
        i, scale = _steps(self._reference_light(ref), rand["step"][:, 0], samples)
        x, d, t, lam, w = jax.vmap(lambda u_, j: sc.flash(u_, a[j], b[j], t0[j], t1[j], y * vis[j], light, light_ref,
                                                          1.0))(rand, i)  # fmt: skip
        return Photons(x=x, d=d, t=t, lam=lam, w=w * scale, pol=jnp.zeros((n, 2, 3), F32), label=_labels(v, i, n))
