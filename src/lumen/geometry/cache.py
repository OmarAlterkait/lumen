"""The navigator cache, and the digests every cache key is made of.

The navigator's arrays are kept by their content under ``$LUMEN_CACHE/build/<key>/``, written atomically; the digests
are ``code_digest`` and ``scene_digest``.

A key is the sha256 of what the product is built from and of the code that builds it:

- **content**: a canonical digest of the input (``scene_digest``: every definition, member, solid tree, label and
  placement; object sharing is part of it, since the navigator numbers kinds by distinct solid; a definition's
  placements are digested by value, all at once);
- **options**: the build's settings;
- **code**: the source of the modules that build it -- its entry modules and every lumen module they import,
  transitively, lazy imports included (``code_closure``; read from the import statements, never listed by hand) --
  their packages' data files, and the numpy and jax versions (``code_digest``).

So nothing stored can drift: a change of input, settings or code is another entry. Two processes building one entry
at once both write a temporary directory and move it into place (``lumen.format.atomic.directory``); the first one
wins.

    arrays, key = nav_arrays(scene)            # the navigator's NavArrays (read, or built and stored) and their key

Notes
-----
Implements the build-products cache of docs/dev/architecture.md, "Digests and caches", for the navigator's arrays.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np

from lumen import _env
from lumen.format.atomic import directory
from lumen.geometry import navigator_build
from lumen.geometry import scene as sc

ROOT = Path(__file__).resolve().parents[1]  # the lumen package
MODULE = re.compile(r"lumen(\.[A-Za-z_]\w*)*")  # a module name
NAV_CODE = ("lumen.geometry.navigator_build", "lumen.geometry.cache")  # what builds and stores the arrays
KIND = "nav"  # the entry's prefix: build/nav-<key>
OWN = "own"  # the scene's ownership (the class-free stage's costly part): build/own-<key>
PACK_MIN = 16  # a stored sequence of at least this many strings, ints or int sequences is written as arrays


# ------------------------------------------------------------------------------------------------- digests
def _module_file(name: str) -> Path | None:
    """Return the source file of a lumen module (``lumen.a.b``), or None when it is no module of lumen."""
    if not MODULE.fullmatch(name):
        return None
    parts = name.split(".")[1:]
    for f in (ROOT.joinpath(*parts).with_suffix(".py"), ROOT.joinpath(*parts, "__init__.py")):
        if f.is_file():
            return f
    return ROOT / "__init__.py" if not parts else None


def _imports(name: str, f: Path) -> set:
    """Return the lumen modules a module imports: every import statement, inside functions too."""
    out = set()
    package = name if f.name == "__init__.py" else name.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(f.read_text(), str(f))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.split(".")[0] == "lumen"}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(up + ([base] if base else []))
            if base.split(".")[0] != "lumen":
                continue
            for a in node.names:  # ``from lumen.x import y``: the module lumen.x.y if it is one, else lumen.x
                out.add(f"{base}.{a.name}" if _module_file(f"{base}.{a.name}") else base)
    return out


@lru_cache(maxsize=None)
def code_closure(entries: tuple) -> tuple:
    """Return the source files of the lumen modules ``entries`` and every lumen module they import, transitively.

    The import graph is read once per process; an import added or removed after the first call is not seen until the
    next process. The files' bytes are read on every ``code_digest``.
    """
    missing = [e for e in entries if _module_file(e) is None]
    if missing:
        raise ValueError(f"code_digest: {missing} names no lumen module; give a module name such as "
                         f"'lumen.geometry.clearance', not a path")  # fmt: skip
    seen: dict = {}
    stack = list(entries)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        f = _module_file(name)
        assert f is not None, name  # an import that Python resolved
        seen[name] = f
        stack += sorted(_imports(name, f))
    return tuple(sorted(set(seen.values())))


def code_digest(entries: tuple) -> str:
    """Return the sha256 of the code ``entries`` run (``code_closure``), each module's package data files (its
    directory's and its ``data/`` directory's ``.json``) and the numpy and jax versions."""  # noqa: D205, D209
    h = hashlib.sha256()
    files = set(code_closure(entries))
    files |= {j for f in set(files) for d in (f.parent, f.parent / "data") for j in d.glob("*.json")}
    for f in sorted(files):
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    h.update(f"numpy {np.__version__} jax {jax.__version__}".encode())
    return h.hexdigest()


class _Digest:
    """A canonical walk of a plain object graph into a sha256 (arrays by dtype, shape and bytes; shared objects)."""

    def __init__(self):
        self.h = hashlib.sha256()
        self.seen = {}

    def put(self, tag, data=b""):
        self.h.update(tag.encode() + b"\0" + data + b"\1")

    def walk(self, x):
        if x is None or isinstance(x, (bool, np.bool_)):
            self.put("b", repr(x).encode())
        elif isinstance(x, str):
            self.put("s", x.encode())
        elif isinstance(x, (int, np.integer)):
            self.put("i", str(int(x)).encode())
        elif isinstance(x, (float, np.floating)):
            self.put("f", np.float64(x).tobytes())
        elif isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            self.put("a", f"{a.dtype.str}{a.shape}".encode() + (a.tobytes() if a.dtype != object else b""))
            if a.dtype == object:
                for v in a.ravel():
                    self.walk(v)
        elif isinstance(x, dict):
            self.put("d", str(len(x)).encode())
            for k in sorted(x, key=repr):
                self.walk(k)
                self.walk(x[k])
        elif isinstance(x, (list, tuple, set, frozenset)):
            items = sorted(x, key=repr) if isinstance(x, (set, frozenset)) else x
            self.put(type(x).__name__, str(len(items)).encode())
            for v in items:
                self.walk(v)
        else:
            self.obj(x)

    def obj(self, x):
        if id(x) in self.seen:
            self.put("r", str(self.seen[id(x)]).encode())
            return
        self.seen[id(x)] = len(self.seen)
        if dataclasses.is_dataclass(x):
            fields = {f.name: getattr(x, f.name) for f in dataclasses.fields(x)}
        else:
            fields = dict(vars(x))
        self.put("o", type(x).__name__.encode())
        if isinstance(x, sc.Definition):
            self.placements(fields.pop("placements"))
        self.walk(fields)

    def placements(self, pls):
        """Digest a definition's placements by value, each field of all of them at once."""
        self.put("P", str(len(pls)).encode())
        if pls:
            self.put("pd", "\0".join(p.definition for p in pls).encode())
            self.put("pn", "\0".join(p.name for p in pls).encode())
            self.put("pR", np.stack([np.asarray(p.R, float).reshape(3, 3) for p in pls]).tobytes())
            self.put("pt", np.stack([np.asarray(p.t, float).reshape(3) for p in pls]).tobytes())
            self.put("ps", np.asarray([p.source for p in pls], np.int64).tobytes())


def scene_digest(scene) -> str:
    """Return the canonical sha256 of a lowered Scene (its content and its object sharing)."""
    d = _Digest()
    d.walk(scene)
    return d.h.hexdigest()


def key(scene, options, digest=None) -> str:
    """Return the cache key of the navigator's arrays of a scene built with ``options`` (``digest``: the scene's
    ``scene_digest``, when known)."""  # noqa: D205, D209
    h = hashlib.sha256()
    h.update((digest or scene_digest(scene)).encode())
    h.update(json.dumps(options, sort_keys=True).encode())
    h.update(code_digest(NAV_CODE).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- the entry
def _encode(x, arrays: dict):
    """Return a JSON-able tree of a value: arrays (numpy scalars too) into ``arrays`` by key, dataclasses by field."""
    if isinstance(x, (np.ndarray, np.generic)):
        name = f"a{len(arrays)}"
        arrays[name] = np.asarray(x)
        return {"a": name, "scalar": isinstance(x, np.generic)}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        t = type(x)
        return {"dc": f"{t.__module__}:{t.__qualname__}",
                "f": {f.name: _encode(getattr(x, f.name), arrays) for f in dataclasses.fields(x)}}  # fmt: skip
    if isinstance(x, dict):
        return {"m": [[_encode(k, arrays), _encode(v, arrays)] for k, v in x.items()]}
    if isinstance(x, (tuple, list)):
        tag = "t" if isinstance(x, tuple) else "l"
        packed = _pack(x, arrays) if len(x) >= PACK_MIN else None
        return {tag: [_encode(v, arrays) for v in x]} if packed is None else {tag: [], **packed}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    raise TypeError(f"the navigator cache cannot store a {type(x).__name__}")


def _array(arrays: dict, a) -> str:
    name = f"a{len(arrays)}"
    arrays[name] = a
    return name


def _pack(x, arrays: dict) -> dict | None:
    """Return a sequence of strings, of ints, or of int tuples (or lists) stored as arrays; None for any other."""
    if all(type(v) is str for v in x):
        blob = "\0".join(x)
        return {"strs": _array(arrays, np.frombuffer(blob.encode(), np.uint8))} if blob.count("\0") == len(x) - 1 \
            else None  # fmt: skip
    try:
        if all(type(v) is int for v in x):
            return {"ints": _array(arrays, np.asarray(x, np.int64))}
        inner = type(x[0])
        if inner in (tuple, list) and all(type(v) is inner and all(type(w) is int for w in v) for v in x):
            lens = np.fromiter(map(len, x), np.int64, len(x))
            flat = np.fromiter(itertools.chain.from_iterable(x), np.int64, int(lens.sum()))
            return {"rag": [_array(arrays, lens), _array(arrays, flat)], "inner": "t" if inner is tuple else "l"}
    except OverflowError:  # an int beyond int64
        return None
    return None


def _unpack(x, arrays) -> list | None:
    """Return the items of a sequence ``_pack`` stored, None for any other."""
    if "strs" in x:
        return arrays[x["strs"]].tobytes().decode().split("\0")
    if "ints" in x:
        return arrays[x["ints"]].tolist()
    if "rag" in x:
        lens, flat = arrays[x["rag"][0]].tolist(), arrays[x["rag"][1]].tolist()
        inner, items, p = tuple if x["inner"] == "t" else list, [], 0
        for n in lens:
            items.append(inner(flat[p : p + n]))
            p += n
        return items
    return None


def _decode(x, arrays):
    """Return the value of an ``_encode`` tree (a dataclass only of a lumen module)."""
    if not isinstance(x, dict):
        return x
    if "a" in x:
        a = arrays[x["a"]]
        return a[()] if x["scalar"] else a
    if "dc" in x:
        import importlib

        module, name = x["dc"].split(":")
        if not MODULE.fullmatch(module):
            raise ValueError(f"the navigator cache names a class outside lumen: {x['dc']}")
        return getattr(importlib.import_module(module), name)(**{k: _decode(v, arrays) for k, v in x["f"].items()})
    if "m" in x:
        return {_decode(k, arrays): _decode(v, arrays) for k, v in x["m"]}
    items = _unpack(x, arrays)
    if items is None:
        items = [_decode(v, arrays) for v in x.get("t", x.get("l", []))]
    return tuple(items) if "t" in x else items


def _save(arrays, d: Path):
    """Write the navigator's arrays (every field of NavArrays, whatever they are), or an Ownership, into ``d``."""
    store: dict = {}
    tree = _encode(arrays, store)
    np.savez(d / "arrays.npz", **store)
    (d / "meta.json").write_text(json.dumps(tree))


def _load(d: Path):
    with np.load(d / "arrays.npz") as z:
        store = {k: z[k] for k in z.files}
    return _decode(json.loads((d / "meta.json").read_text()), store)


def nav_arrays(scene, **options):
    """Return (the navigator's arrays, their cache key): read from the build cache, else built and stored. Products of
    the same geometry are keyed by that key (the jump's clearance bound, ``lumen.transport.fastpath``).

    A miss builds them from the scene's class-free stage (``navigator_build.prepare``) with its ownership, the costly
    part, read from or stored in its own entry (``own-<key>``: the scene, the precision and the code only), so builds
    of one scene for other classes or settings (the conversion's placement check, then the first load) resolve it once.
    ``precision`` ("float32", the default, or "float64") enters a key only when it is not float32.
    """  # noqa: D205
    digest = scene_digest(scene)
    precision = options.pop("precision", "float32")
    instances = options.pop("instances", None)  # the scene's own listing, if made: not a build option
    extra = {} if precision == "float32" else {"precision": precision}
    k = key(scene, {**options, **extra}, digest)
    path = _env.cache_dir() / "build" / f"{KIND}-{k}"
    if (path / "meta.json").is_file():
        return _load(path), k
    # the ownership's own entry: its contact tolerance is the precision's
    own_path = _env.cache_dir() / "build" / f"{OWN}-{key(scene, extra, digest)}"
    own = _load(own_path) if (own_path / "meta.json").is_file() else None
    stage = navigator_build.prepare(scene, own, precision, instances)
    if own is None:
        with directory(own_path) as tmp:
            _save(stage.own, tmp)
    arrays = navigator_build.finish(stage, **options)
    with directory(path) as tmp:
        _save(arrays, tmp)
    return arrays, k
