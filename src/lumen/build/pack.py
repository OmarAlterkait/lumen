"""Pack a ``Params`` (GDML names) into the engine's arrays: the surface lookup's per-type rows and the media.

    packed = pack(det, params)

A gather: at load each leaf of the surface lookup's named parameters (``lumen.build.lookup.named_params``) is mapped
once at the build (``leaf_map``, from ``Parts.key_rows``) to

- the value of the key it reads directly (``Source`` column "value", "constant", "setting" or "default"), for a key
  of the detector's parameters: ``pack`` reads it from ``params``;
- a JAX function of a key's value (column "derived": a Rayleigh table from RINDEX, ``Source.derive``);
- otherwise a constant of the build: a table's knot energies (structure, as its knot count: a key's must be the
  detector's, ``lumen.build.values``), a value the model never reads (folded into its type), a setting that is no
  parameter.

So nothing is rebuilt at pack: the same arrays whether a value is the detector's, a calibration's, a new concrete
one or a traced one (``jax.jit``, ``jax.vmap``, ``jax.grad``), bit for bit: a concrete
value is cast on the host as at load (``lookup.pack``), a traced one with jnp, and the knot energies are the build's.
The structure (which surface types, rows, knot counts and knots, channels) is the load's: a value that would
contradict it is refused or counted by the value checks (``lumen.build.values``), never re-decided. A derivative is
refused only where its method is not available in this version (``lumen.params.PENDING``), when it is traced (a
``custom_jvp`` whose rule raises for a tangent that is not a symbolic zero). A key the photon loop does not read (a
sensor model's value, applied after tracking) passes through to the readout.
"""

from __future__ import annotations

import numpy as np

from lumen._core.errors import refusing
from lumen._core.safe import traced
from lumen.build import lookup
from lumen.build.library import DIRECT
from lumen.format.calibrations import Table


def leaf_map(optics, library, key_rows) -> tuple[dict, tuple, dict]:
    """Return (the build's named parameters, ((path, key, column, derive), ...) of every leaf read from a key, {key:
    the knot energies the build packs, float64}); a table's knot energies are structure, the build's
    (``lumen.build.values``)."""  # noqa: D205, D209
    base = lookup.named_params(optics, library)
    reads, knots = [], {}
    for path, src in key_rows.items():
        if src.key is None or not _has(base, path):
            continue
        if src.column == "energy":
            leaf = base
            for k in path:
                leaf = leaf[k]
            knots.setdefault(src.key, np.asarray(leaf, np.float64))
        elif src.column in DIRECT or (src.column == "derived" and src.derive is not None):
            reads.append((path, src.key, src.column, src.derive))
    return base, tuple(reads), knots


def _has(tree, path) -> bool:
    for k in path:
        if not isinstance(tree, dict) or k not in tree:
            return False
        tree = tree[k]
    return True


def _put(tree, path, v):
    return v if not path else {**tree, path[0]: _put(tree[path[0]], path[1:], v)}


def _leaf(v):
    return v.value if isinstance(v, Table) else v


def named(detector, params) -> dict:
    """Return the surface lookup's named parameters for ``params``: the build's, each leaf read from a key gathered."""
    out, guarded = detector.parts.named, {}
    for path, key, column, derive in detector.parts.reads:
        if key not in params:  # a key no Params holds (a setting that is no parameter): the build's value
            continue
        if key not in guarded:
            guarded[key] = _guarded(params, key)
        v = guarded[key]
        out = _put(out, path, derive(v) if column == "derived" else _leaf(v))
    return out


def pack(detector, params) -> dict:
    """Return the engine's packed parameter arrays (the surface lookup's rows) for ``params`` (the module docstring)."""
    return lookup.pack(detector.parts.optics, named(detector, params))


def _guarded(params, key):
    """Return a key's value, a traced one's derivative refused where there is none: a method not available in this
    version (``lumen.params.PENDING``)."""  # noqa: D205, D209
    v = params[key]
    info = params.info(key) if key in params._spec else None
    if info is not None and not info.available and traced(v):
        v = refusing(f"{key}: calibrated by value, but the derivative's method ({info.method}) is not available in "
                     "this version")(v)  # fmt: skip
    return v
