"""Bulk media: what happens to a photon between surfaces.

A ``Medium`` is static: an index law (None for an opaque medium, in which no photon travels: Geant4 kills a photon
entering a material without RINDEX), an absorption-length law (absorption acts as a weight exp(-s / L_abs), never as
an event) and event channels, each a (name, lobe, length law): Rayleigh scattering, re-emission by a wavelength
shifter, and so on.

Parameters of one medium: ``{"index": {...}, "absorption": {...}, "<channel>": {"length": {...}, "lobe": {...}}}``.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from lumen.media import laws
from lumen.surfaces.lobes import Lobe


@dataclass(frozen=True, eq=False)
class Channel:
    """An event channel of a medium: rate 1 / length(lam), outgoing direction drawn from ``lobe``.

    With ``reemit`` (a spectrum of ``lumen.media.spectra``) the channel is absorption by a wavelength shifter or a
    scintillator component: the photon is re-emitted with probability ``p["yield"]`` (the event's live weight) at a
    new wavelength from the spectrum, after a delay drawn from Exp(``p["tau"]``). Its parameters are
    ``{"length": ..., "yield": ..., "tau": ...}``.
    """

    name: str
    lobe: Lobe
    length: laws.Law
    reemit: object = None


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
        1.0 / medium.absorption(p["absorption"], lam) if medium.absorption is not None else jnp.zeros_like(lam)
    )
    out["mu"] = (
        jnp.stack([1.0 / c.length(p[c.name]["length"], lam) for c in medium.channels], -1)
        if medium.channels
        else jnp.zeros(jnp.shape(lam) + (0,))
    )
    out["lobe_params"] = tuple(p[c.name].get("lobe", {}) for c in medium.channels)
    return out


def validate(medium, p, lam_min, lam_max):
    """Check every law of the medium over the wavelength range [lam_min, lam_max] (nm), failing closed."""
    if medium.index is not None:
        laws.validate(medium.index, p["index"], lam_min, lam_max, name=f"{medium.name}.index")
    if medium.absorption is not None:
        laws.validate(medium.absorption, p["absorption"], lam_min, lam_max, name=f"{medium.name}.absorption")
    for c in medium.channels:
        if c.length.out != "length":
            raise laws.LawError(f"{medium.name}.{c.name}: a channel length law must output a length")
        laws.validate(c.length, p[c.name]["length"], lam_min, lam_max, name=f"{medium.name}.{c.name}.length")
