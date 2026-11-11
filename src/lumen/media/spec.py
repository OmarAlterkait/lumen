"""What a medium model builds on: ``MediumSpec`` (a medium, its parameters, the model's declared values) and lobes.

    base.with_channel("rayleigh", lobe=henyey_greenstein("g"))    # Geant4's Rayleigh rate, an HG lobe of parameter g
    base.add_channel("dye", lobe=ISOTROPIC_LOBE, length="L_dye")   # a new scatterer, constant mean free path L_dye (m)
    base.with_absorption("L_abs")                                  # a constant absorption length L_abs (m)

Each name given (a lobe's ``params``, a length) is one of the model's declared parameters; the spec records where it
is read (``used``), and the build maps that leaf to its key. A new channel comes after Geant4's (``CHANNEL_ORDER``),
so the existing channels keep their random numbers.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax.numpy as jnp

from lumen._core import frames
from lumen.media import bulk, laws, phase
from lumen.surfaces.lobes import Lobe

ISOTROPIC_LOBE = phase.ISOTROPIC


@dataclasses.dataclass(frozen=True)
class MediumSpec:
    """A medium under construction: ``medium`` (bulk.Medium), ``params`` and the model's ``values`` by name."""

    medium: bulk.Medium
    params: dict
    values: dict
    used: dict = dataclasses.field(default_factory=dict)  # {param path: declared name}
    lobes: dict = dataclasses.field(default_factory=dict)  # {channel: declared lobe parameters}

    def _value(self, name):
        if name not in self.values:
            raise KeyError(f"MediumSpec: {name!r} is not a declared parameter of the model ({sorted(self.values)})")
        return self.values[name]

    def _lobe(self, ch, lobe):
        if not isinstance(lobe, Lobe):
            raise TypeError(f"MediumSpec: a channel's lobe is a lumen Lobe, not {type(lobe).__name__}")
        used = {(ch, "lobe", k): k for k in lobe.params}
        return {k: self._value(k) for k in lobe.params}, used

    def with_channel(self, name, *, lobe=None, length=None) -> MediumSpec:
        """Return the spec with channel ``name``'s lobe and/or its length (a declared constant, m) replaced."""
        chans = {c.name: c for c in self.medium.channels}
        if name not in chans:
            raise KeyError(f"MediumSpec: medium {self.medium.name!r} has no channel {name!r} ({sorted(chans)})")
        c, p, used, lobes = chans[name], dict(self.params[name]), dict(self.used), dict(self.lobes)
        if lobe is not None:
            p["lobe"], u = self._lobe(name, lobe)
            used.update(u)
            lobes[name] = lobe.params
            c = dataclasses.replace(c, lobe=lobe)
        if length is not None:
            p["length"] = {"value": self._value(length)}
            used[(name, "length", "value")] = length
            c = dataclasses.replace(c, length=laws.constant("length"))
        chans[name] = c
        medium = dataclasses.replace(self.medium, channels=tuple(chans.values()))
        return dataclasses.replace(self, medium=medium, params={**self.params, name: p}, used=used, lobes=lobes)

    def add_channel(self, name, *, lobe, length) -> MediumSpec:
        """Return the spec with a new scattering channel: ``lobe``, a constant mean free path ``length`` (declared)."""
        if self.medium.opaque:
            raise ValueError(f"MediumSpec: medium {self.medium.name!r} is opaque (no RINDEX)")
        if name in {c.name for c in self.medium.channels} or name in ("index", "absorption"):
            raise KeyError(f"MediumSpec: medium {self.medium.name!r} has a {name!r} already")
        p, used = {"length": {"value": self._value(length)}}, {**self.used, (name, "length", "value"): length}
        lp, u = self._lobe(name, lobe)
        if lp:
            p["lobe"] = lp
        used.update(u)
        ch = bulk.Channel(name, lobe, laws.constant("length"))
        medium = dataclasses.replace(self.medium, channels=self.medium.channels + (ch,))
        return dataclasses.replace(self, medium=medium, params={**self.params, name: p}, used=used,
                                   lobes={**self.lobes, name: lobe.params})  # fmt: skip

    def with_absorption(self, length) -> MediumSpec:
        """Return the spec with a constant absorption length ``length`` (a declared parameter, m)."""
        medium = dataclasses.replace(self.medium, absorption=laws.constant("length"))
        params = {**self.params, "absorption": {"value": self._value(length)}}
        return dataclasses.replace(
            self, medium=medium, params=params, used={**self.used, ("absorption", "value"): length}
        )


@functools.cache
def henyey_greenstein(g: str) -> Lobe:
    """Return a Henyey-Greenstein phase function whose asymmetry is the declared parameter named ``g``.

    HG(mu; g) = (1 - g^2) / (4 pi (1 + g^2 - 2 g mu)^(3/2)), drawn by Geant4's inverse CDF (``phase.mie_hg_cos``, in a
    form that never cancels in float32), 2 uniforms; |g| <= 0.999 resolves in float32. One lobe per name (a lobe is
    static, hashed by identity: the same lobe on every build).
    """

    def sample(p, wi, n, rand):
        c_minus, c_plus = phase.mie_hg_cos(p[g], rand[..., 0])
        mu = jnp.where(c_minus < c_plus, 1.0 - c_minus, c_plus - 1.0)
        s = jnp.sqrt(jnp.clip(c_minus * c_plus, 0.0, None))
        phi = 2.0 * math.pi * rand[..., 1]
        return frames.to_world(jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), mu], -1), wi)

    def logpdf(p, wi, n, wo):
        c_minus = 0.5 * jnp.sum((wo - wi) ** 2, -1)
        c_plus = 0.5 * jnp.sum((wo + wi) ** 2, -1)
        return jnp.log(phase.hg_density(p[g], c_minus, c_plus))

    return Lobe(f"hg:{g}", delta=False, surface=False, n_rand=2, params=(g,), sample=sample, logpdf=logpdf,
                weight=lambda p, wi, n, wo: jnp.ones(wi.shape[:-1]), axis=lambda p, wi, n: wi)  # fmt: skip
