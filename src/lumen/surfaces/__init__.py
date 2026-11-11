"""Surfaces: the outcomes of a photon meeting a surface, Fresnel physics, polarisation, lobes and surface types.

What a surface model (``lumen.surface_model``) builds with, here by name: ``SurfaceType``, ``outcome_vector``,
``cos_incidence``, ``is_polarised``, ``chained_cuts`` and the ten outcomes (STRAIGHT, REFRACT, SPECULAR, BACKSCATTER,
GLOSSY, DIFFUSE, DIFFUSE_T, DETECT, ABSORB, REEMIT).
"""

_TYPES = ("SurfaceType", "outcome_vector", "cos_incidence", "is_polarised", "chained_cuts")
_OUTCOMES = ("STRAIGHT", "REFRACT", "SPECULAR", "BACKSCATTER", "GLOSSY", "DIFFUSE", "DIFFUSE_T", "DETECT", "ABSORB",
             "REEMIT", "N_OUTCOMES")  # fmt: skip
__all__ = [*_TYPES, *_OUTCOMES]


def __getattr__(name):  # lazily: lumen.surfaces.types imports lumen.media, whose lobes import this package
    if name in _TYPES:
        from lumen.surfaces import types

        return getattr(types, name)
    if name in _OUTCOMES:
        from lumen.surfaces import outcomes

        return getattr(outcomes, name)
    raise AttributeError(f"module 'lumen.surfaces' has no attribute {name!r}")
