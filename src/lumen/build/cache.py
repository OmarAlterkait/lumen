"""The build cache: build products keyed by their content, under ``$LUMEN_CACHE/build/<key>/``, written atomically.

A key is the sha256 of what the product is built from and of the code that builds it:

- **content**: a canonical digest of the input (``scene_digest``: every definition, member, solid tree, label and
  placement; object sharing is part of it, since the navigator numbers kinds by distinct solid);
- **options**: the build's settings;
- **code**: the source of ``lumen/{format,gdml,rules,build,geometry}`` and the numpy and jax versions
  (``code_digest``).

So nothing stored can drift: a change of input, settings or code is another entry. Two processes building one entry
at once both write a temporary directory and move it into place (``lumen._cache.directory``); the first one wins.

    arrays = nav_arrays(scene)                 # the navigator's NavArrays: read, or built and stored

Notes
-----
Implements DESIGN section 4.5's build-products cache for the navigator's arrays.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

import lumen
from lumen import _cache

CODE_PACKAGES = ("format", "gdml", "rules", "build", "geometry")
KIND = "nav"  # the entry's prefix: build/nav-<key>
FORMAT = 1  # the entry's layout version


# ------------------------------------------------------------------------------------------------- digests
def code_digest() -> str:
    """Return the sha256 of the building code's source and the numpy and jax versions."""
    import jax

    h = hashlib.sha256()
    root = Path(lumen.__file__).parent
    for pkg in CODE_PACKAGES:
        for f in sorted((root / pkg).rglob("*.py")):
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    h.update(f"numpy {np.__version__} jax {jax.__version__} format {FORMAT}".encode())
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
        self.walk(fields)


def scene_digest(scene) -> str:
    """Return the canonical sha256 of a lowered Scene (its content and its object sharing)."""
    d = _Digest()
    d.walk(scene)
    return d.h.hexdigest()


def key(scene, options) -> str:
    """Return the cache key of the navigator's arrays of a scene built with ``options``."""
    h = hashlib.sha256()
    h.update(scene_digest(scene).encode())
    h.update(json.dumps(options, sort_keys=True).encode())
    h.update(code_digest().encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- the entry
def _save(arrays, d: Path):
    t = arrays.csg
    flat = {f"host/{k}": v for k, v in arrays.host.items()}
    for part in ("T", "K", "G"):
        flat.update({f"csg{part}/{k}": v for k, v in getattr(t, part).items()})
    kids = arrays.kids
    flat.update(kb_c=np.asarray([b[0] for b in t.kind_box], float).reshape(-1, 3),
                kb_h=np.asarray([b[1] for b in t.kind_box], float).reshape(-1, 3),
                kids_len=np.asarray([len(k) for k in kids], np.int64),
                kids_flat=np.asarray([i for k in kids for i in k], np.int64),
                lo_w=arrays.lo_w, hi_w=arrays.hi_w)  # fmt: skip
    np.savez(d / "arrays.npz", **flat)
    meta = dict(format=FORMAT, n=arrays.n, n_mesh=arrays.n_mesh, chunk=arrays.chunk, ch0=arrays.ch0,
                srf=list(arrays.srf), bg=arrays.bg, names=list(arrays.names), defs=list(arrays.defs),
                stats=arrays.stats, coincident=[list(g) for g in arrays.coincident], n_leaves=list(t.n_leaves),
                n_kinds=t.n_kinds, maxnl=t.maxnl, empty_leaf=t.empty_leaf, has_torus=t.has_torus)  # fmt: skip
    (d / "meta.json").write_text(json.dumps(meta))


def _load(d: Path):
    from lumen.geometry.csg_build import CsgTables
    from lumen.geometry.navigator_build import NavArrays

    meta = json.loads((d / "meta.json").read_text())
    with np.load(d / "arrays.npz") as z:
        a = {k: z[k] for k in z.files}

    def part(prefix):
        return {k.split("/", 1)[1]: v for k, v in a.items() if k.startswith(prefix + "/")}

    boxes = tuple(zip(a["kb_c"], a["kb_h"], strict=True))
    t = CsgTables(T=part("csgT"), K=part("csgK"), G=part("csgG"), kind_box=boxes,
                  n_leaves=tuple(meta["n_leaves"]), n_kinds=meta["n_kinds"], maxnl=meta["maxnl"],
                  empty_leaf=meta["empty_leaf"], has_torus=meta["has_torus"])  # fmt: skip
    ends = np.cumsum(a["kids_len"])
    kids = tuple(tuple(int(i) for i in k) for k in np.split(a["kids_flat"], ends[:-1])) if len(ends) else ()
    return NavArrays(host=part("host"), csg=t, n=meta["n"], n_mesh=meta["n_mesh"], chunk=meta["chunk"],
                     ch0=meta["ch0"], srf=tuple(meta["srf"]), bg=meta["bg"], names=tuple(meta["names"]),
                     defs=tuple(meta["defs"]), stats=meta["stats"], coincident=tuple(map(tuple, meta["coincident"])),
                     kids=kids, lo_w=a["lo_w"], hi_w=a["hi_w"])  # fmt: skip


def entry_path(scene, options) -> Path:
    """Return the cache directory of the navigator's arrays of a scene (it may not exist)."""
    return _cache.entry("build", f"{KIND}-{key(scene, options)}")


def nav_arrays(scene, **options):
    """Return the navigator's arrays of a scene: read from the build cache, else built and stored."""
    from lumen.geometry import navigator_build

    path = entry_path(scene, options)
    if (path / "meta.json").is_file():
        return _load(path)
    arrays = navigator_build.build(scene, **options)
    with _cache.directory(path) as tmp:
        _save(arrays, tmp)
    return arrays
