"""Scintillation light: a flash at a point, or along a charged segment (G4Scintillation's step model).

    src = ScintillationPoint.of(det.params(), "LAr", position=(0, 0, 0), energy=1.0)  # 1 MeV at t = 0

The material's light from its GDML values: SCINTILLATIONYIELD (per MeV) and its components, each a spectrum, decay
time and share, read as G4Scintillation reads them (``components``, ``shares``): Geant4 11's SCINTILLATIONCOMPONENTi,
...TIMECONSTANTi, ...YIELDi, i = 1..3 (a GDML's Geant4-10 keys are read as these at import,
``lumen.rules.keys.geant4_11_scintillation``).
A segment is quenched by Birks' law E / (1 + kB E / L), kB ``birks`` (m/MeV; never set by a GDML: 0); a point is not.
Photons are isotropic, uniform along a segment, delayed by their component's time profile
(``lumen.media.emission.TimeProfile``): its decay, and with the material's SCINTILLATIONRISETIMEi (read only under
the optics setting ``scint_finite_rise_time``, Geant4's finite-rise-time switch: ``lumen.rules.optics``) the rise.
Random numbers: ``layout``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import rng
from lumen._core.safe import ratio, safe_sqrt, traced
from lumen.media.emission import TimeProfile
from lumen.params import Param
from lumen.record import Photons
from lumen.rules.keys import G4_10_KEYS, SCINTILLATION_KEYS, components
from lumen.sources._base import F32, POSITION, TIME, Source, lit, photon_uniforms, wavelength
from lumen.transport.estimator import defensive_q
from lumen.units import INF, TWO_PI

KEYS = SCINTILLATION_KEYS
ENERGY = Param("MeV", (0.0, INF), method="reweight")
BIRKS = Param("m/MeV", (0.0, INF), method="reweight")
sg = jax.lax.stop_gradient
DECAY, RISE = TimeProfile(("decay",)), TimeProfile(("decay", "rise"))  # a component's time profile without, with rise


def visible_energy(edep, length, birks):
    """Return a charged step's visible energy (MeV): Birks' dE / (1 + kB dE / dx); a zero-length step unquenched."""
    q = 1.0 + birks * edep / jnp.where(length > 0.0, length, 1.0)
    return jnp.where((length > 0.0) & (edep > 0.0), edep / q, jnp.maximum(edep, 0.0))


def shares(get, comps: tuple):
    """Return the components' shares of the light, G4Scintillation's (``get(key, default)`` reads a value).

    Geant4 11: yields y_i = SCINTILLATIONYIELDi (defaults 1, 0, 0), s = y1 + y2 + y3; one component takes all, two
    take y1 / s and the rest, three y_i / s.
    """
    if len(comps) == 1:
        return jnp.ones(1, F32)
    y = [get(k, F32(d)) for k, d in zip(KEYS["shares"], (1.0, 0.0, 0.0), strict=True)]
    s = y[0] + y[1] + y[2]
    if len(comps) == 2:
        return jnp.stack([y[0] / s, 1.0 - y[0] / s])
    return jnp.stack(y) / s


def layout(segment: bool, rises: bool) -> rng.Layout:
    """Return a photon's uniforms: its wavelength, direction, component, delay, place along a segment (``segment``) and
    rise (``rises``: the light has rise times)."""  # noqa: D205, D209
    return rng.Layout(lam=1, dir=2, comp=1, time=1, pos=int(segment), rise=int(rises))


def flash(rand, x0, x1, t0, t1, n_ph, light, light_ref, samples):  # noqa: PLR0913 -- the photon's pieces
    """Return one photon (x, d, t, lam, w) at uniforms rand (``layout``: pos read when x1 is given, rise when the
    light has rise times).

    The component and its wavelength are drawn at the reference (``light_ref``: shares and spectra) under the one
    defensive mixture (``lumen.transport.estimator.defensive_q``: a component of share 0 at the reference is covered,
    as each spectrum's knot intervals without mass) and weighted to the live ``light`` (``lumen._core.safe.ratio``;
    exactly 1 at the reference where nothing is covered); its delay is its time profile's (``TimeProfile``: the decay,
    and the rise when ``light`` has rise times), reparametrised."""  # noqa: D205, D209
    f, taus, live, rises = light
    f_ref, _, spec_ref, _ = light_ref
    f_ref = defensive_q(sg(f_ref), jnp.ones(f_ref.shape, bool))
    c = lit(jnp.sum(rand["comp"][0] >= jnp.cumsum(f_ref)[:-1]), f_ref)  # drawn by its share (never one of 0)
    each = [wavelength(s, r, rand["lam"][0]) for s, r in zip(live, spec_ref, strict=True)]
    lam, wt = sg(jnp.stack([x[0] for x in each])[c]), jnp.stack([x[1] for x in each])[c]
    z, phi = 1.0 - 2.0 * rand["dir"][0], TWO_PI * rand["dir"][1]
    r = safe_sqrt(1.0 - z * z)
    d = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z])
    p = {"tau": jnp.stack(taus)[c]}
    if rises is None:
        delay = DECAY.sample(p, rand["time"])
    else:
        delay = RISE.sample(dict(p, rise=jnp.stack(rises)[c]), jnp.stack([rand["time"][0], rand["rise"][0]]))
    w = n_ph / samples * ratio(f[c], f_ref[c]) * wt
    if x1 is None:
        return x0, d, t0 + delay, lam, w
    u = rand["pos"][0]
    return x0 + u * (x1 - x0), d, t0 + u * (t1 - t0) + delay, lam, w


@dataclasses.dataclass(frozen=True)
class _Scintillation(Source):
    material: str = ""
    share_keys: tuple = dataclasses.field(default=(), kw_only=True)  # the share keys the material has
    components: tuple = dataclasses.field(default=(), kw_only=True)  # the components Geant4 emits (``components``)
    rises: bool = dataclasses.field(default=False, kw_only=True)  # the material has its components' rise times
    SEGMENT: ClassVar[bool] = False

    @classmethod
    def of(cls, params, material: str, **fields):
        """Return the source of ``material``'s light, its structure (the components) from the keys ``params`` has.

        Refused (KeyError) where Geant4 emits nothing or aborts: no component, no SCINTILLATIONYIELD, an emitted
        component without its spectrum or decay time (GetConstProperty aborts), or shares that are not finite (yields
        summing to 0; checked when the yields are concrete). The rise times are read where the material has them (the
        load refuses them without the finite-rise-time switch, and their absence under it). Nothing it builds depends on
        a value: the components follow from the keys present (``params`` may carry traced values, e.g. under
        ``jax.grad``).
        """
        m, k = f"materials/{material}/", KEYS
        comps = components(lambda key: m + key in params)
        need = ["SCINTILLATIONYIELD"] + [k[f][i] for i in comps for f in ("spectra", "times")]
        missing = [x for x in need if m + x not in params] if comps else list(k["spectra"][:1])
        if missing:
            g10 = sorted(x for x in G4_10_KEYS if m + x in params)
            hint = f"; it has the Geant4-10 keys {g10}, inert under Geant4 >= 11" if g10 else ""
            raise KeyError(f"{material}: no {missing} (Geant4 11 emits none or aborts{hint})")
        share_keys = tuple(x for x in k["shares"] if m + x in params)
        f = shares(lambda key, d: params.get(m + key, d), comps)
        if not traced(f):  # traced yields (under jax.grad) have no concrete shares to check
            f = np.asarray(f, np.float64)
            if not (np.all(np.isfinite(f)) and np.all(f >= 0.0)):
                raise KeyError(f"{material}: the components' shares {f.tolist()} are not finite and >= 0 (Geant4 "
                               "divides by the sum of the yields)")  # fmt: skip
        rises = bool(comps) and all(m + k["rises"][i] in params for i in comps)
        return cls(material, **fields, share_keys=share_keys, components=comps, rises=rises)

    def reads(self) -> tuple:
        """Return the material keys the light is read from."""
        k = KEYS
        own = [k[f][i] for i in self.components for f in ("spectra", "times", *(("rises",) if self.rises else ()))]
        return tuple(f"materials/{self.material}/{x}" for x in (*own, "SCINTILLATIONYIELD", *self.share_keys))

    def light_of(self, v):
        """Return (yield per MeV, (shares (N,), decay times, live spectra, rise times or None)) of the components
        from the values ``v``.
        """  # noqa: D205
        m, k = f"materials/{self.material}/", KEYS
        f = shares(lambda key, d: v.get(m + key, d), self.components)
        taus = tuple(v[m + k["times"][i]] for i in self.components)
        rises = tuple(v[m + k["rises"][i]] for i in self.components) if self.rises else None
        return v[m + "SCINTILLATIONYIELD"], (f, taus, tuple(v[m + k["spectra"][i]] for i in self.components), rises)

    def spectra(self) -> tuple:
        """Return the keys of the spectra drawn (each component's)."""
        return tuple(f"materials/{self.material}/{KEYS['spectra'][i]}" for i in self.components)

    def draw(self, v, ref, key, n, first_id, samples):
        """Return the Photons (see the module docstring)."""
        y, light = self.light_of(v)
        light_ref = self.light_of(ref)[1]
        if self.SEGMENT:
            x0, x1, t0, t1 = v["start"], v["end"], v["t_start"], v["t_end"]
            n_ph = y * visible_energy(v["energy"], jnp.linalg.norm(x1 - x0), v["birks"])
        else:
            x0, x1, t0, t1, n_ph = v["position"], None, v["time"], None, y * v["energy"]
        lay = layout(self.SEGMENT, self.rises)
        x, d, t, lam, w = jax.vmap(lambda r: flash(lay.split(r), x0, x1, t0, t1, n_ph, light, light_ref, samples))(
            photon_uniforms(key, lay.n, n, first_id))  # fmt: skip
        return Photons(x=x, d=d, t=t, lam=lam, w=w, pol=jnp.zeros((n, 2, 3), F32))


@dataclasses.dataclass(frozen=True)
class ScintillationPoint(_Scintillation):
    """``energy`` (MeV) deposited at ``position`` (m) at ``time`` (ns) in ``material``; build it with ``of``."""

    position: Any = (0.0, 0.0, 0.0)
    energy: Any = 1.0
    time: Any = 0.0
    FIELDS = {"position": POSITION, "energy": ENERGY, "time": TIME}


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
    SEGMENT = True
