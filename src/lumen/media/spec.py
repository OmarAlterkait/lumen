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

import jax.numpy as jnp

from lumen.lobes import Lobe
from lumen.media import bulk, laws, phase

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


def phase_function(name: str, *, params: dict, n_rand: int, sample, logpdf, pol_sample=None, pol_logpdf=None,  # noqa: PLR0913
                   unpolarised: bool = False) -> Lobe:  # fmt: skip
    """Return a bulk phase function of your own, a value (a part: ``lumen.lobes.Lobe``).

    ``sample(p, wi, n, rand) -> wo`` and ``logpdf(p, wi, n, wo)`` (``n`` unused; ``rand`` (..., n_rand) uniforms,
    ``n_rand`` <= 3), ``p`` its declared parameters ({name: lumen.Param}, scalars with a default) by name. Pass it at
    load for one material's channel, ``lumen.load(path, parts={"materials/<M>/<channel>": lobe})``: its parameters are
    the keys ``materials/<M>/<channel>/<name>``, differentiable by reweighting (draws at the reference values, weighted
    by the live / reference density).

    ``pol_sample(p, wi, er, ei, rand) -> wo`` and ``pol_logpdf(p, wi, er, ei, wo)``: the direction's law given the
    photon's Jones vector (as lumen's Rayleigh). A phase function without the pair says so, ``unpolarised=True``: its
    direction ignores the polarisation (the state is still projected after the event), and a run records it
    (``Light.provenance``'s ``"parts"``).
    """
    from lumen.models import medium_model

    medium_model(name, params)  # the same checks as a medium model's: "distribution:name", scalar Params, defaults
    if (pol_sample is None) != (pol_logpdf is None):
        raise ValueError(f"phase function {name!r}: give both pol_sample and pol_logpdf, or neither")
    if pol_sample is None and not unpolarised:
        raise ValueError(f"phase function {name!r}: without pol_sample and pol_logpdf its direction ignores the "
                         "photon's polarisation: give the pair, or declare unpolarised=True")  # fmt: skip
    if pol_sample is not None and unpolarised:
        raise ValueError(f"phase function {name!r}: unpolarised=True with a polarised pair; give one")
    return Lobe(name, surface=False, n_rand=n_rand, params=tuple(sorted(params)), sample=sample, logpdf=logpdf,
                pol_sample=pol_sample, pol_logpdf=pol_logpdf, declared=tuple(sorted(params.items())))  # fmt: skip


@functools.cache
def henyey_greenstein(g: str) -> Lobe:
    """Return a Henyey-Greenstein phase function whose asymmetry is the declared parameter named ``g``.

    HG(mu; g) = (1 - g^2) / (4 pi (1 + g^2 - 2 g mu)^(3/2)), drawn by Geant4's inverse CDF (``phase.mie_hg_cos``, in a
    form that never cancels in float32), 2 uniforms; |g| <= ``lumen.rules.choices.MIE_G_MAX`` resolves in float32. One
    lobe per name (a lobe is static, hashed by identity: the same lobe on every build).
    """

    def sample(p, wi, n, rand):
        return phase.hg_direction(*phase.mie_hg_cos(p[g], rand[..., 0]), rand[..., 1], wi)

    def logpdf(p, wi, n, wo):
        return jnp.log(phase.hg_density(p[g], *phase.hg_cosines(wi, wo)))

    return Lobe(f"hg:{g}", surface=False, n_rand=2, params=(g,), sample=sample, logpdf=logpdf)
