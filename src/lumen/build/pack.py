"""Pack parameters into the engine's arrays: Params (GDML names) to the legacy optical-class rows.

The values of a ``Params`` replace the detector file's properties, surface values and sensor settings; the surfaces
and media are rebuilt from them and packed into the class compiler's rows. The packing is done on the host here; the
traced packing (every value an operand of the photon loop) comes with the operands.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen.build import _legacy_regions, build, legacy_library
from lumen.format import schema
from lumen.format.calibrations import Table


def _property(p: schema.Property, v) -> schema.Property:
    if isinstance(v, Table):
        return dataclasses.replace(p, energy=np.asarray(v.energy, np.float64), value=np.asarray(v.value, np.float64))
    return dataclasses.replace(p, value=np.asarray(v, np.float64))


def with_values(df: schema.DetectorFile, sensor_set, params):
    """Return (detector file, sensor set) with every parameter's value from ``params``."""
    mats = {n: dataclasses.replace(m, properties={k: _property(p, params[f"materials/{n}/{k}"])
                                                  if f"materials/{n}/{k}" in params else p  # fixed: no parameter
                                                  for k, p in m.properties.items()})
            for n, m in df.materials.items()}  # fmt: skip
    surfs = {n: dataclasses.replace(s, value=float(params[f"surfaces/{n}/value"]),
                                    properties={k: _property(p, params[f"surfaces/{n}/{k}"])
                                                for k, p in s.properties.items()})
             for n, s in df.surfaces.items()}  # fmt: skip
    groups = tuple(dataclasses.replace(g, settings={k: (float(params[f"sensors/{g.name}/{k}"])
                                                        if f"sensors/{g.name}/{k}" in params else v)
                                                    for k, v in g.settings.items()})
                   for g in sensor_set.groups)  # fmt: skip
    return dataclasses.replace(df, materials=mats, surfaces=surfs), dataclasses.replace(sensor_set, groups=groups)


def pack(detector, params) -> dict:
    """Return the engine's packed parameter arrays (the class compiler's rows) for ``params``."""
    df, sset = with_values(build.bound(detector.file), detector.sensor_set, params)
    parts = detector.parts
    _, library, _, _ = legacy_library.library_and_bindings(df, parts.base_tree, parts.instances, sset)
    named = _legacy_regions.named_params(parts.optics, library)
    return _legacy_regions.pack(parts.optics, named)


def named(detector, params) -> dict:
    """Return the class compiler's named parameters for ``params`` (the baseline's calibration space)."""
    df, sset = with_values(build.bound(detector.file), detector.sensor_set, params)
    parts = detector.parts
    _, library, _, _ = legacy_library.library_and_bindings(df, parts.base_tree, parts.instances, sset)
    return _legacy_regions.named_params(parts.optics, library)
