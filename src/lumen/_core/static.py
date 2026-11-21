"""A jitted function's static argument by its structure: hashed and compared by value, never by the object.

    jax.jit(f, static_argnums=0)(Static(model), ...)        # inside f: static() is the model

jax keeps every static argument a jitted function was called with for the function's lifetime (its trace records,
which ``jax.clear_caches`` leaves), and keys its compiled programs on the argument's hash and equality. An object equal
only to itself (a Model, a diagnostic without value equality) would compile once per object and keep each one, with
its arrays, alive. A ``Static`` is equal to another of the same structure (``structure``): two loads of one detector
share one compiled program, and what jax keeps is the structure (shapes, configuration, the arrays' digests, the code's
digest) and a weak reference to the object, never the object or its arrays.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import types
import weakref
from typing import Any

import jax
import numpy as np

_ATOMS = (type(None), bool, int, float, complex, str, bytes, range, type(Ellipsis))


def _name(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


def _digest(a) -> tuple:
    a = np.ascontiguousarray(a)
    return ("array", a.dtype.str, a.shape, hashlib.sha256(a.view(np.uint8).reshape(-1)).hexdigest())


def _cached(t: type) -> frozenset:
    """Return the names of a class's cached properties: values derived from the object, not part of its structure."""
    return frozenset(k for c in t.__mro__ for k, v in vars(c).items() if isinstance(v, functools.cached_property))


def structure(obj: Any) -> Any:  # noqa: C901 -- one case per kind of value
    """Return obj's structure: a hashable value equal for two objects that trace the same program.

    Atoms are themselves (with their type: 1 and 1.0 differ); containers, dataclasses and other objects their type's
    name and their fields' structures (a cached property is derived, not structure); an array (numpy or jax) its
    dtype, shape and the sha256 of its bytes; a function its name, its code (bytecode, constants, names) and its
    defaults' and closure's structures; a class or module its name. A cycle is a reference to the object's position
    on the walk's path (an object met twice elsewhere is the same structure both times: aliasing is not structure).
    """
    done: dict[int, Any] = {}  # an object's structure, once walked (the object held, so its id stays its own)
    path: dict[int, int] = {}  # the objects on the walk's path, by depth
    held: list = []

    def walk(o):
        if isinstance(o, _ATOMS):
            return (type(o).__name__, o)
        k = id(o)
        if k in done:
            return done[k]
        if k in path:
            return ("cycle", len(path) - path[k])
        path[k] = len(path)
        try:
            out = done[k] = one(o)
        finally:
            del path[k]
        held.append(o)
        return out

    def one(o):  # noqa: C901, PLR0912
        if isinstance(o, (type, types.ModuleType)):
            return ("name", _name(o) if isinstance(o, type) else o.__name__)
        if isinstance(o, (np.ndarray, np.generic)):
            return _digest(o)
        if isinstance(o, jax.Array):
            return ("device", *_digest(np.asarray(o))[1:])
        t = type(o)
        if isinstance(o, types.CodeType):
            return ("code", o.co_name, o.co_code, walk(o.co_consts), o.co_names, o.co_varnames, o.co_freevars)
        if isinstance(o, types.FunctionType):
            cells = tuple(walk(c.cell_contents) for c in (o.__closure__ or ()))
            return ("function", o.__module__, o.__qualname__, walk(o.__code__), walk(o.__defaults__),
                    walk(o.__kwdefaults__), cells)  # fmt: skip
        if isinstance(o, types.MethodType):
            return ("method", walk(o.__func__), walk(o.__self__))
        if isinstance(o, functools.partial):
            return ("partial", walk(o.func), walk(o.args), walk(o.keywords))
        if isinstance(o, (types.BuiltinFunctionType, types.MethodWrapperType, types.WrapperDescriptorType)):
            return ("builtin", getattr(o, "__module__", None), getattr(o, "__qualname__", repr(o)))
        if isinstance(o, dict):  # in insertion order: an iteration in the trace follows it
            return (_name(t), tuple((walk(k), walk(v)) for k, v in o.items()))
        if isinstance(o, (set, frozenset)):
            return (_name(t), frozenset(walk(v) for v in o))
        if isinstance(o, (tuple, list)):  # a NamedTuple included (its type's name)
            return (_name(t), tuple(walk(v) for v in o))
        if dataclasses.is_dataclass(o):
            return (_name(t), tuple((f.name, walk(getattr(o, f.name))) for f in dataclasses.fields(o)))
        skip = _cached(t)
        fields = [(k, v) for k, v in getattr(o, "__dict__", {}).items() if k not in skip]
        for c in t.__mro__:
            slots = c.__dict__.get("__slots__", ())
            for s in (slots,) if isinstance(slots, str) else slots:
                if s not in ("__weakref__", "__dict__") and hasattr(o, s):
                    fields.append((s, getattr(o, s)))
        if not fields and t.__hash__ is not None and t.__eq__ is not object.__eq__:
            return (_name(t), "value", o)  # a value of its own equality (an enum member, ...)
        return (_name(t), tuple((k, walk(v)) for k, v in fields))

    return walk(obj)


_GONE = object()


def _hold(o: Any):
    """Return a function giving back ``o`` (``_GONE`` once it is gone), holding it weakly where it allows: a tuple or
    list (not weakly referable) by its items, each held so."""  # noqa: D205, D209
    try:
        r = weakref.ref(o)
        return lambda: _GONE if (x := r()) is None else x
    except TypeError:
        pass
    if isinstance(o, (tuple, list)):
        items, t = [_hold(v) for v in o], type(o)
        make = t._make if hasattr(t, "_make") else t  # a NamedTuple from its fields

        def back():
            vals = [f() for f in items]
            return _GONE if any(v is _GONE for v in vals) else make(vals)

        return back
    return lambda: o  # held (an atom, or an object without weak references: as small as its structure)


class Static:
    """An object as a jitted function's static argument (see the module docstring); ``static()`` is the object.

    Hashed and compared by ``key`` (the object's ``structure``, unless given). It holds the object weakly (a tuple by
    its items): jax calls the function with the Static it was called with, whose object the caller holds, and a later
    Static of the same structure hits the compiled program without a trace.
    """

    __slots__ = ("_back", "_hash", "key")

    def __init__(self, obj: Any, key: Any = None):
        self.key = structure(obj) if key is None else key
        self._hash = hash(self.key)
        self._back = _hold(obj)

    def __call__(self) -> Any:
        o = self._back()
        if o is _GONE:  # a trace runs on the Static of the call, whose object the caller holds
            raise ReferenceError("lumen: a jitted function traced with an object that no longer exists")
        return o

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return isinstance(other, Static) and self._hash == other._hash and self.key == other.key

    def __repr__(self) -> str:
        return f"Static({self._hash:#x})"
