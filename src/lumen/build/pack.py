"""Pack a ``Params`` (GDML names) into the engine's arrays: the surface lookup's per-type rows and the media.

The values are applied to the detector file and its sensor set (``lumen.format.apply``); the surfaces and media are
rebuilt from them and packed, on the host. ``simulate`` passes the packed arrays to the photon loop as operands.
"""

from __future__ import annotations

from lumen.build import build, surfaces
from lumen.build import library as optics_library
from lumen.format.apply import apply, sensor_set


def named(detector, params) -> dict:
    """Return the surface lookup's named parameters for ``params``."""
    df, sset = apply(build.bound(detector.file), params), sensor_set(detector.sensor_set, params)
    parts = detector.parts
    _, library, _, _ = optics_library.library_and_bindings(df, parts.base_tree, parts.instances, sset)
    return surfaces.named_params(parts.optics, library)


def pack(detector, params) -> dict:
    """Return the engine's packed parameter arrays (the class compiler's rows) for ``params``."""
    return surfaces.pack(detector.parts.optics, named(detector, params))
