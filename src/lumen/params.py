"""Params: the one mapping of every parameter value, keyed by GDML names.

Keys are ``materials/<Material>/<PROPERTY>``, ``surfaces/<Surface>/<PROPERTY>`` (with ``surfaces/<Surface>/sigma_alpha``
or ``surfaces/<Surface>/polish`` for the opticalsurface's value) and ``sensors/<group>/<field>``. A tabulated property's
value is a ``Table`` (energies in eV, values); a constant is a float. ``info(key)`` gives each key's unit, allowed range
and gradient method.

Precedence, later wins: (1) the detector file's own values; (2) each sensor group's model defaults; (3) the
calibration (flat: every value it changes from the nominal ones). ``explain(key)`` says which layer each value came
from.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass

import jax
import numpy as np

from lumen._core import domains
from lumen._core.safe import traced
from lumen.format.apply import check
from lumen.format.calibrations import Table

PENDING = {"moving_paths+reweight"}  # methods not validated yet: asking for their derivative raises (any key)
NONE = "none"  # a parameter without a derivative (calibrated by value)


@dataclass(frozen=True)
class Param:
    """A parameter's declaration: its unit, allowed range, shape, default, gradient method and a note.

        Param("m", shape=(3,), method="moving_paths", default=(0.0, 0.0, 0.2))   # a source position
        Param("", (0.0, 50.0), default=4.0)                                       # a dimensionless value in [0, 50]

    Every key's declaration (``Params.info``): lumen's own (``lumen.rules.keys.PARAMS``, a sensor model's), and an
    extension's, whose parameters become ``Params`` keys (``source/<source>/<name>``, ``sensors/<group>/<name>``,
    ``materials/<M>/<name>``, ``surfaces/<S>/<name>``), so ``Params.set``, calibrations and the gradient check take them
    as they take lumen's own. ``method``: "reweight" (the default), "moving_paths" (a source parameter that moves where
    the light starts, a position: no derivative in this version, ``lumen.sources``' refusal), "after_tracking" (applied
    to the readout), "none" (no derivative) or a pending one (``PENDING``). ``note``: where a value comes from when the
    file omits it (e.g. "default 0, G4OpBoundaryProcess.cc:310"), or who declares it. A value is admitted by one
    predicate everywhere (``lumen._core.domains.admits``): inside the range and finite, so NaN is never a value.
    """

    unit: str = ""
    range: tuple = (-float("inf"), float("inf"))
    _: KW_ONLY
    shape: tuple = ()
    default: object = None
    method: str = "reweight"
    note: str = ""

    def __post_init__(self):
        """Check the declaration; store the range, shape and default as hashable tuples of floats."""
        lo, hi = (float(x) for x in self.range)
        if not lo <= hi:
            raise ValueError(f"Param: range {self.range} is not (lo, hi) with lo <= hi")
        object.__setattr__(self, "range", (lo, hi))
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        if self.default is not None:
            d = np.asarray(self.default, np.float64)
            if d.shape != self.shape:
                raise ValueError(f"Param: default {self.default!r} has shape {d.shape}, declared {self.shape}")
            if not np.all(domains.admits(np, d, lo, hi)):
                raise ValueError(f"Param: default {self.default!r} is outside its range [{lo}, {hi}]")
            object.__setattr__(self, "default", float(d) if not d.shape else tuple(d.reshape(-1).tolist()))

    @property
    def moves(self) -> bool:
        """Return whether the parameter moves where the light starts (method "moving_paths..."): no derivative."""
        return self.method.startswith("moving_paths")

    @property
    def available(self) -> bool:
        """Return whether its derivative is available (a method neither pending nor "none")."""
        return self.method not in PENDING and self.method != NONE

    def value(self):
        """Return the default: a float, or a float32 array of the declared shape (None without a default)."""
        if self.default is None:
            return None
        return self.default if not self.shape else np.asarray(self.default, np.float32).reshape(self.shape)


def declared(prefix: str, params: Mapping, method: str | None = None, note: str = "") -> tuple[dict, dict]:
    """Return (values, spec) of declared parameters ``{name: Param}`` as keys ``prefix + name`` (values: defaults).

    A parameter without a default has no value here: set it (``Params.set`` refuses an unknown key, so a source
    function's parameter without a default is given to ``simulate`` in its ``params``).
    """
    values, spec = {}, {}
    for name, p in params.items():
        if not isinstance(p, Param):
            raise TypeError(f"{prefix}{name}: declare a lumen.Param, not {type(p).__name__}")
        spec[prefix + name] = dataclasses.replace(p, method=method or p.method, note=note or p.note)
        if p.default is not None:
            values[prefix + name] = p.value()
    return values, spec


class UnknownParamKey(KeyError):  # noqa: N818 (the name of the documented error catalogue)
    """A key that is not a parameter of this detector."""


class Params(Mapping):
    """An immutable mapping of parameter values with their declared spec and the layer each value came from.

    A JAX pytree: its values are the leaves (a ``Table`` gives its energies and values), its keys and spec the static
    part, so a ``Params`` passes through ``jax.jit`` and ``jax.grad`` as it is; the layers ride along without being part
    of the static part's identity (a value set again from another layer retraces nothing).
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

    def __contains__(self, key) -> bool:
        """Return whether ``key`` is a parameter (without composing the unknown key's message)."""
        return key in self._values

    def _unknown(self, key):
        import difflib

        near = difflib.get_close_matches(key, list(self._values), n=3)
        return f"{key!r} is not a parameter of this detector" + (f". Did you mean {near}?" if near else "")

    def info(self, key) -> Param:
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
        new = _checked(key, self.info(key), old, value)
        return Params({**self._values, key: new}, self._spec, {**self._layers, key: layer})

    def scale(self, key, factor) -> Params:
        """Return a copy with a key's values multiplied by ``factor`` (a table's values, not its energies).

        ``factor`` may be traced (``jax.jvp``, ``jax.grad``): the scaled value is then traced too.
        """
        v = self[key]
        if isinstance(v, Table):
            return self.set(key, Table(v.energy, _values(v.value) * factor))
        return self.set(key, _values(v) * factor)

    def merged(self, values: dict, spec: Mapping, layer: str) -> Params:
        """Return a copy with new keys (their values and Param) from one layer; a key already here keeps its own."""
        new = {k: v for k, v in values.items() if k not in self._values}
        return Params({**self._values, **new}, {**self._spec, **{k: spec[k] for k in new}},
                      {**self._layers, **dict.fromkeys(new, layer)})  # fmt: skip

    def with_layer(self, values: dict, layer: str) -> Params:
        """Return a copy with several values replaced by one layer (a calibration, sensor defaults)."""
        out = self
        for k, v in values.items():
            out = out.set(k, v, layer)
        return out


def _values(v):
    """Return a value as an array: numpy when concrete, as it is when traced."""
    return v if traced(v) else np.asarray(v)


def _checked(key: str, info: Param, old, value):
    """Return ``value`` after the shape check and, for a concrete value, the admissibility check.

    ``lumen._core.domains.admits``: in range and finite; a traced value is checked by every run
    (``lumen.build.values``).
    """
    lo, hi = info.range
    if isinstance(old, Table):
        check(key, old, value)  # a table keeps its knot count
        vals = value.value
    else:
        vals = value
        if np.shape(vals) != np.shape(old):
            raise ValueError(f"{key}: shape {np.shape(old)} expected, got {np.shape(vals)}")
    if traced(vals):
        return value
    vals = np.asarray(vals)
    if vals.dtype.kind in "fiu":
        v = np.asarray(vals, np.float64)
        if not np.all(domains.admits(np, v, lo, hi)):
            raise ValueError(f"{key} has a value outside [{lo}, {hi}] {info.unit} or not finite".rstrip())
    return value


class _Layers:
    """The layer each value came from: carried through a flatten, never part of the treedef's identity (equal and of one
    hash whatever they contain), so a value set again from another layer (``Params.set``) retraces nothing."""  # noqa: D205, D209

    __slots__ = ("layers",)

    def __init__(self, layers: dict):
        self.layers = layers

    def __eq__(self, other):
        return isinstance(other, _Layers)

    def __hash__(self):
        return 0


def _flatten(p: Params):
    keys = tuple(sorted(p._values))
    return tuple(p._values[k] for k in keys), (keys, tuple(sorted(p._spec.items())), _Layers(dict(p._layers)))


def _unflatten(aux, leaves):
    keys, spec, layers = aux
    return Params(dict(zip(keys, leaves, strict=True)), dict(spec), dict(layers.layers))


jax.tree_util.register_pytree_node(Params, _flatten, _unflatten)
