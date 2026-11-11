"""Bulk media: laws, phase functions, spectra, and what a medium model (``lumen.medium_model``) builds on.

``MediumSpec`` (``lumen.media.spec``), ``henyey_greenstein(g)`` and ``ISOTROPIC_LOBE``, here by name.
"""

_SPEC = ("MediumSpec", "henyey_greenstein", "ISOTROPIC_LOBE")
__all__ = list(_SPEC)


def __getattr__(name):  # lazily: the media modules import lumen.surfaces, which imports them back
    if name in _SPEC:
        from lumen.media import spec

        return getattr(spec, name)
    raise AttributeError(f"module 'lumen.media' has no attribute {name!r}")
