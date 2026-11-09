"""Params: the one mapping of every parameter value, keyed by GDML names.

Keys are ``materials/<Material>/<PROPERTY>``, ``surfaces/<Surface>/<PROPERTY>`` (with ``surfaces/<Surface>/value``
for the opticalsurface's sigma_alpha or polish) and ``sensors/<group>/<field>``. A tabulated property's value is a
``Table`` (energies in eV, values); a constant is a float. ``info(key)`` gives each key's unit, allowed range and
gradient method.

Precedence, later wins: (1) the detector file's own values; (2) each sensor group's model defaults; (3) the
calibration's parent chain, oldest first; (4) the calibration itself. ``explain(key)`` says which layer each value
came from.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import jax
import numpy as np

from lumen.format.apply import check
from lumen.format.calibrations import Table

METHODS = ("reweight", "moving_paths", "moving_paths+reweight", "after_tracking")


@dataclass(frozen=True)
class ParamInfo:
    """What a parameter declares: its unit, allowed range, gradient method and whether that method is available."""

    key: str
    unit: str
    range: tuple[float, float]
    method: str
    available: bool = True
    table: bool = False


class UnknownParamKey(KeyError):  # noqa: N818 (the name of the documented error catalogue)
    """A key that is not a parameter of this detector."""


class Params(Mapping):
    """An immutable mapping of parameter values with their declared spec and the layer each value came from.

    A JAX pytree: its values are the leaves (a ``Table`` gives its energies and values), its keys, spec and layers the
    static part, so a ``Params`` passes through ``jax.jit``, ``jax.grad`` and ``wrt=`` as it is.
    """

    def __init__(self, values: dict, spec: Mapping, layers: dict):
        self._values = dict(values)
        self._spec = dict(spec)
        self._layers = dict(layers)

    def __getitem__(self, key):
        """Return a value; an unknown key names the nearest known ones."""
        if key not in self._values:
            raise UnknownParamKey(self._unknown(key))
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys in sorted order."""
        return iter(sorted(self._values))

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._values)

    def _unknown(self, key):
        import difflib

        near = difflib.get_close_matches(key, list(self._values), n=3)
        return f"{key!r} is not a parameter of this detector" + (f". Did you mean {near}?" if near else "")

    def info(self, key) -> ParamInfo:
        """Return the unit, allowed range and gradient method of a key."""
        if key not in self._spec:
            raise UnknownParamKey(self._unknown(key))
        return self._spec[key]

    def explain(self, key) -> str:
        """Return which layer a key's value came from."""
        self[key]
        return self._layers[key]

    def set(self, key, value, layer: str = "user") -> Params:
        """Return a copy with one value replaced, checked against the key's allowed range and shape."""
        old = self[key]
        new = _checked(self.info(key), old, value)
        return Params({**self._values, key: new}, self._spec, {**self._layers, key: layer})

    def scale(self, key, factor) -> Params:
        """Return a copy with a key's values multiplied by ``factor`` (a table's values, not its energies)."""
        v = self[key]
        if isinstance(v, Table):
            return self.set(key, Table(v.energy, np.asarray(v.value) * factor))
        return self.set(key, np.asarray(v) * factor)

    def with_layer(self, values: dict, layer: str) -> Params:
        """Return a copy with several values replaced by one layer (a calibration, sensor defaults)."""
        out = self
        for k, v in values.items():
            out = out.set(k, v, layer)
        return out


def _checked(info: ParamInfo, old, value):
    lo, hi = info.range
    if isinstance(old, Table):
        check(info.key, old, value)  # a table keeps its knot count
        vals = np.asarray(value.value)
    else:
        vals = np.asarray(value)
        if vals.shape != np.shape(old):
            raise ValueError(f"{info.key}: shape {np.shape(old)} expected, got {vals.shape}")
    if not isinstance(vals, np.ndarray) or vals.dtype.kind in "fiu":
        v = np.asarray(vals, np.float64)
        if np.any(v < lo) or np.any(v > hi):
            raise ValueError(f"{info.key} has a value outside [{lo}, {hi}] {info.unit}".rstrip())
    return value


def _flatten(p: Params):
    keys = tuple(sorted(p._values))
    return tuple(p._values[k] for k in keys), (keys, tuple(sorted(p._spec.items())), tuple(sorted(p._layers.items())))


def _unflatten(aux, leaves):
    keys, spec, layers = aux
    return Params(dict(zip(keys, leaves, strict=True)), dict(spec), dict(layers))


jax.tree_util.register_pytree_node(Params, _flatten, _unflatten)
