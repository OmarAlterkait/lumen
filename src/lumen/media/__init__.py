"""Bulk media: laws, phase functions, spectra, and what a medium model (``lumen.medium_model``) builds on.

``MediumSpec`` (``lumen.media.spec``), ``henyey_greenstein(g)``, ``phase_function`` (a part) and ``ISOTROPIC_LOBE``,
here by name.
"""

from lumen.media.spec import ISOTROPIC_LOBE, MediumSpec, henyey_greenstein, phase_function

__all__ = ["ISOTROPIC_LOBE", "MediumSpec", "henyey_greenstein", "phase_function"]
