"""Pack a ``Params`` (GDML names) into the engine's arrays: the surface lookup's per-type rows and the media.

The values are applied to the detector file and its sensor set (``lumen.format.apply``); the surfaces and media are
rebuilt from them and packed, on the host. ``simulate`` passes the packed arrays to the photon loop as operands.

A ``Params`` holding traced values (``jax.jvp``, ``jax.grad``, ``jax.jit`` over parameters) packs differentiably: the
rest is packed on the host as above, with each traced key at the detector's own value, and every packed value read
from a traced key directly (``Source`` column "energy", "value", "constant", "setting" or "default", the value equal to
the key's) moves with it, ``named + (traced - its host value)``: exact in value where they are equal, its derivative the
key's. A table's knot energies keep their host values. A traced key the engine reads through a conversion ("derived":
e.g. the Rayleigh length from RINDEX) raises ``GradientNotAvailable``; a key the photon loop does not read (a sensor
model's value, applied after tracking) passes through to the readout.
"""

from __future__ import annotations

import jax
import numpy as np

from lumen.build import build, surfaces
from lumen.build import library as optics_library
from lumen.format.apply import apply, sensor_set
from lumen.format.calibrations import Table

DIRECT = ("energy", "value", "constant", "setting", "default")  # a packed value that is the key's value itself


def named(detector, params) -> dict:
    """Return the surface lookup's named parameters for ``params``."""
    df, sset = apply(build.bound(detector.file), params), sensor_set(detector.sensor_set, params)
    parts = detector.parts
    _, library, _, _ = optics_library.library_and_bindings(df, parts.base_tree, parts.instances, sset)
    return surfaces.named_params(parts.optics, library)


def _traced(v) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(v))


def _leaf(v, column):
    return v.energy if column == "energy" else v.value if isinstance(v, Table) else v


def _get(tree, path):
    for k in path:
        tree = tree[k]
    return tree


def _put(tree, path, v):
    return v if not path else {**tree, path[0]: _put(tree[path[0]], path[1:], v)}


def host_params(detector, params):
    """Return (the traced keys, ``params`` with each traced key at the detector's own value)."""
    from lumen.params import Params

    p0 = detector.params()
    hot = {k for k in params if _traced(params[k])}
    values = {k: (p0[k] if k in hot else params[k]) for k in params}  # the detector's own values are unchecked
    return hot, Params(values, params._spec, params._layers)


def pack(detector, params) -> dict:
    """Return the engine's packed parameter arrays (the surface lookup's rows) for ``params`` (traced: module docs)."""
    if not _traced(params):
        return surfaces.pack(detector.parts.optics, named(detector, params))
    from lumen.sources import GradientNotAvailable

    hot, host = host_params(detector, params)
    unavailable = sorted(k for k in hot if k in params._spec and not params.info(k).available)
    if unavailable:
        method = params.info(unavailable[0]).method
        raise GradientNotAvailable(f"{unavailable}: calibrated by value, but the derivative's method ({method}) is not "
                                   "available in this version")  # fmt: skip
    live = named(detector, host)
    for (section, entry, *leaf), src in detector.parts.sources.items():
        if src.key not in hot or src.column == "energy":  # a table's knot energies stay the host's (no derivative)
            continue
        if src.column not in DIRECT:
            raise GradientNotAvailable(f"{src.key}: the engine reads it through a conversion ({src.column}: "
                                       f"{src.note}), so its derivative is not available")  # fmt: skip
        path = (section, entry, *leaf)
        base = _get(live, path)
        if not np.array_equal(np.asarray(base, np.float32), np.asarray(_leaf(host[src.key], src.column), np.float32)):
            raise GradientNotAvailable(f"{src.key}: the engine's value is not the key's value ({src.column}), so its "
                                       "derivative is not available")  # fmt: skip
        delta = _leaf(params[src.key], src.column) - _leaf(host[src.key], src.column)
        live = _put(live, path, np.asarray(base, np.float32) + delta)
    return surfaces.pack(detector.parts.optics, live)
