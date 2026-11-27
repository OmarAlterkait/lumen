"""Bulk media: what happens to a photon between surfaces.

A ``Medium`` is static: an index law (None for an opaque medium, in which no photon travels: Geant4 kills a photon
entering a material without RINDEX), an absorption-length law (absorption acts as a weight exp(-s / L_abs), never as
an event) and event channels, each a (name, lobe, length law): Rayleigh scattering, re-emission by a wavelength
shifter, and so on. Geant4's channels come in a fixed order, ``CHANNEL_ORDER`` (Rayleigh, Mie, WLS), so a medium's
channel indices and its use of random numbers do not depend on which others it has.

Parameters of one medium: ``{"index": {...}, "absorption": {...}, "<channel>": {"length": {...}, "lobe": {...}}}``.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from lumen._core.rng import REEMIT_UNIFORMS
from lumen.lobes import Lobe
from lumen.media import laws, phase
from lumen.media.emission import EmissionRecord

CHANNEL_ORDER = ("rayleigh", "mie", "wls")


@dataclass(frozen=True, eq=False)
class Channel:
    """An event channel of a medium: rate 1 / length(lam), outgoing direction drawn from ``lobe``.

    With ``emission`` (a ``lumen.media.emission.EmissionRecord``) the channel is absorption by a wavelength shifter
    (G4OpWLS): the photon goes on as the record's emission, with expected yield ``p["yield"]`` (a weight) at a
    wavelength from the spectrum ``p["spectrum"]`` at or below the absorbed energy, after the record's time profile
    (``p["tau"]``). Its parameters are ``{"length": ..., "yield": ..., "tau": ..., "spectrum": {"energy", "value"}}``.
    """

    name: str
    lobe: Lobe
    length: laws.Law
    emission: EmissionRecord | None = None

    def __post_init__(self):
        """Refuse a re-emission that draws more uniforms than the step gives it (``rng.REEMIT_UNIFORMS``)."""
        if self.emission is not None and 1 + len(self.emission.profile.components) > REEMIT_UNIFORMS:
            comps = self.emission.profile.components
            raise ValueError(f"channel {self.name!r}: a re-emission draws its wavelength and one uniform per "
                             f"time-profile component, at most {REEMIT_UNIFORMS} in all; {comps} has {len(comps)} "
                             "components")  # fmt: skip


@dataclass(frozen=True, eq=False)
class Medium:
    """A bulk medium: its index, absorption and event channels (see the module docstring)."""

    name: str
    index: laws.Law | None  # None: opaque
    absorption: laws.Law | None = None  # None: no absorption
    channels: tuple[Channel, ...] = ()

    @property
    def opaque(self):
        """Whether no photon travels in this medium (it has no refractive index)."""
        return self.index is None

    def __repr__(self):
        """Show the medium's name."""
        return f"Medium({self.name})"


def mie(length: laws.Law) -> Channel:
    """Return Geant4's Mie scattering channel (G4OpMieHG): mean free path ``length`` (MIEHG), lobe ``phase.MIE_HG``.

    Its parameters are ``{"length": {...}, "lobe": {"g_forward": ..., "g_backward": ..., "forward_ratio": ...}}``
    (MIEHG_FORWARD, MIEHG_BACKWARD, MIEHG_FORWARD_RATIO).
    """
    return Channel("mie", phase.MIE_HG, length)


def rate(length):
    """Return the rate 1 / L (1/m) of a mean free path ``L`` (m).

    A length of 0 is Geant4's mean free path of 0, the interaction at once: its rate is 1 / (the smallest normal
    float32), about 8.5e37 per metre, finite, so a path of 0 still attenuates by exp(-mu * 0) = 1 and every longer one
    by 0, and its derivative is 0 (scaling a 0 leaves it 0), where 1 / 0 would give inf * 0 = NaN for both.
    """
    tiny = jnp.finfo(jnp.float32).tiny
    ok = length >= tiny
    return jnp.where(ok, 1.0 / jnp.where(ok, length, 1.0), 1.0 / tiny)


def evaluate(medium, p, lam):
    """Return the medium at wavelength ``lam`` (nm).

    Returns
    -------
    dict
        ``n`` and ``n_g`` (refractive and group index; absent for an opaque medium), ``mu_abs`` (1/m), ``mu``
        (1/m, one rate per channel: shape (..., n_channels)) and ``lobe_params`` (one dict per channel).
    """
    out = {}
    if medium.index is not None:
        out["n"] = medium.index(p["index"], lam)
        out["n_g"] = laws.group_index(medium.index)(p["index"], lam)
    out["mu_abs"] = (
        rate(medium.absorption(p["absorption"], lam)) if medium.absorption is not None else jnp.zeros_like(lam)
    )
    out["mu"] = (
        jnp.stack([rate(c.length(p[c.name]["length"], lam)) for c in medium.channels], -1)
        if medium.channels
        else jnp.zeros(jnp.shape(lam) + (0,))
    )
    out["lobe_params"] = tuple(p[c.name].get("lobe", {}) for c in medium.channels)
    return out


def validate(medium, p, domain=None):
    """Check every law of the medium where light evaluates it (``laws.validate``: a table's knots inside the light's
    ``domain``, and its ends), failing closed."""  # noqa: D205, D209
    if medium.index is not None:
        laws.validate(medium.index, p["index"], name=f"{medium.name}.index", domain=domain)
    if medium.absorption is not None:
        laws.validate(medium.absorption, p["absorption"], name=f"{medium.name}.absorption", domain=domain)
    for c in medium.channels:
        if c.length.out != "length":
            raise laws.LawError(f"{medium.name}.{c.name}: a channel length law must output a length")
        laws.validate(c.length, p[c.name]["length"], name=f"{medium.name}.{c.name}.length", domain=domain)
