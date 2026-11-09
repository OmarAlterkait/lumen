"""Pack parameters into the engine's arrays: Params (GDML names) to the legacy optical-class rows.

The values of a ``Params`` are applied to the detector file and its sensor set (``lumen.format.apply``); the surfaces
and media are rebuilt from them and packed into the class compiler's rows. The packing is done on the host here; the
traced packing (every value an operand of the photon loop) comes with the operands.
"""

from __future__ import annotations

from lumen.build import _legacy_regions, build, legacy_library
from lumen.format.apply import apply, sensor_set


def named(detector, params) -> dict:
    """Return the class compiler's named parameters for ``params`` (the baseline's calibration space)."""
    df, sset = apply(build.bound(detector.file), params), sensor_set(detector.sensor_set, params)
    parts = detector.parts
    _, library, _, _ = legacy_library.library_and_bindings(df, parts.base_tree, parts.instances, sset)
    return _legacy_regions.named_params(parts.optics, library)


def pack(detector, params) -> dict:
    """Return the engine's packed parameter arrays (the class compiler's rows) for ``params``."""
    return _legacy_regions.pack(detector.parts.optics, named(detector, params))
