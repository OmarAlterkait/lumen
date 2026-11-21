"""Reading and writing a detector file: a ``<name>.lumen/`` directory of canonical JSON and one ``.npy`` per array.

Canonical JSON: sorted keys, one-space indent, floats as the shortest repr that reads back exactly, and each record
of a list on one line, so two detector files compare with ``diff``.

**The layout is one rule** (``to_files`` writes it, ``read`` inverts it, both by the records' fields and types):

- each field of ``schema.DetectorFile`` is stored in the file its field metadata names; ``detector.json`` holds the
  labels (``name``, ``provenance``, ``report``), the format version and the manifest;
- a record is a mapping of its fields by name; a field at its default is left out; a record kept in a dict under
  its own name (or a property under its key) does not store that field;
- a list of records at the top level (``placements``, ``skins``, ``borders``) is a table: one row per record, the
  record's fields in field order, except that a vector field (``tuple[float, ...]``: a placement's position and
  rotation) is one (n, k) array for the whole table;
- an array (``np.ndarray``) of one or more dimensions is a float64 ``.npy`` file at ``arrays/<its path of keys>.npy``;
  a 0-d array (a constant) is a JSON number.

The manifest in ``detector.json`` holds every content file's digest: a JSON file's sha256, an array's sha256 of its
dtype, shape and bytes (``array_digest``; independent of numpy's file header). The content digest
(``content_digest``) is the sha256 of the manifest: the labels are not in it, so one detector converted from two
sources, or under two names, has one digest. A save goes to a temporary directory moved into place, so a reader never
sees half a detector file, and an existing detector file is never overwritten unless asked. Reading refuses a file of
another format version, an unknown or missing field and a file whose digest differs from the manifest.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io as _io
import json
import re
import sys
import types
import typing
from pathlib import Path

import numpy as np

from lumen.format import atomic, schema

SAFE = re.compile(r"^[A-Za-z0-9_.+-]+$")


class FormatError(ValueError):
    """A file that cannot be read: another lumen's format, a missing or changed file, a broken reference."""


def check_format(found, want: str, where, remedy: str) -> None:
    """Refuse a file of another format than ``want``: one message for every file lumen writes (no older format is read).

    ``remedy`` says how to write the file again ("convert the detector again", "export again", ...).
    """
    if found != want:
        raise FormatError(f"{where}: written by another lumen (format {found!r}); this lumen reads {want}: {remedy}")


# ------------------------------------------------------------------------------------------------ canonical JSON
def _number(x: float) -> str:
    if not np.isfinite(x):
        raise FormatError(f"non-finite number {x!r}: the detector file stores finite numbers only")
    return repr(float(x))


def _flat(v) -> str:
    """One value on one line."""
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return _number(float(v))
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(str(k), ensure_ascii=False)}: {_flat(v[k])}" for k in sorted(v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_flat(x) for x in v) + "]"
    raise FormatError(f"cannot store {type(v).__name__} in JSON")


def dumps(obj: dict) -> str:
    """Canonical JSON of a dict: top-level keys sorted; a dict value one entry per line, a list one item per line."""
    lines = ["{"]
    keys = sorted(obj)
    for i, k in enumerate(keys):
        v, sep = obj[k], "," if i < len(keys) - 1 else ""
        if isinstance(v, dict) and v:
            inner = sorted(v)
            lines.append(f" {json.dumps(k)}: {{")
            lines += [
                f"  {json.dumps(str(j), ensure_ascii=False)}: {_flat(v[j])}{',' if n < len(inner) - 1 else ''}"
                for n, j in enumerate(inner)
            ]
            lines.append(f" }}{sep}")
        elif isinstance(v, (list, tuple)) and v:
            lines.append(f" {json.dumps(k)}: [")
            lines += [f"  {_flat(x)}{',' if n < len(v) - 1 else ''}" for n, x in enumerate(v)]
            lines.append(f" ]{sep}")
        else:
            lines.append(f" {json.dumps(k)}: {_flat(v)}{sep}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def sha256_bytes(data: bytes) -> str:
    """Return 'sha256:' and the hex digest of data."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    """Return 'sha256:' and the hex digest of a file's bytes, read in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def array_digest(a: np.ndarray) -> str:
    """Return 'sha256:' and the hex digest of an array's dtype, shape and bytes (C order)."""
    a = np.ascontiguousarray(a)
    return sha256_bytes(f"{a.dtype.str}{a.shape}".encode() + b"\0" + a.tobytes())


# ------------------------------------------------------------------------------------------------- the layout
HEAD_FILE = "detector.json"
FILES: dict[str, tuple[str, ...]] = {}  # every file and the DetectorFile fields it stores, from the field metadata
for _f in dataclasses.fields(schema.DetectorFile):
    FILES.setdefault(_f.metadata["file"], ())
    FILES[_f.metadata["file"]] += (_f.name,)
HEAD = FILES.pop(HEAD_FILE)  # the labels: not in the content digest


@functools.cache
def field_types(t) -> dict:
    """Return a record's fields and their types, in field order (each annotation evaluated in the record's module)."""
    module = vars(sys.modules[t.__module__])
    return {f.name: eval(f.type, module) if isinstance(f.type, str) else f.type for f in dataclasses.fields(t)}


def _default(t, name: str):
    """Return a record field's default (``dataclasses.MISSING``: none)."""
    f = next(f for f in dataclasses.fields(t) if f.name == name)
    return f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default


def _at_default(t, name: str, v) -> bool:
    d = _default(t, name)
    return v is d or (d is not dataclasses.MISSING and not isinstance(v, np.ndarray) and type(v) is type(d) and v == d)


def _inner(h):
    """Return the type of an optional (``X | None``) as X; any other type as it is."""
    if typing.get_origin(h) in (typing.Union, types.UnionType):
        return next(a for a in typing.get_args(h) if a is not type(None))
    return h


def _item(h, i: int):
    """Return the type of item ``i`` of a tuple type (``tuple[X, ...]`` or ``tuple[X, Y]``; a bare tuple: object)."""
    args = typing.get_args(h)
    return object if not args else args[0] if args[-1] is Ellipsis else args[i]


def _key_field(t) -> str | None:
    """Return the field a record kept in a dict is keyed by (``name``, or a property's ``key``), if any."""
    return next((k for k in ("name", "key") if k in field_types(t)), None)


def _is_vector(h) -> bool:
    return h == tuple[float, ...]


def _is_table(h) -> bool:
    return typing.get_origin(h) is tuple and dataclasses.is_dataclass(_item(h, 0))


SAFE = re.compile(r"^[A-Za-z0-9_+-][A-Za-z0-9_.+-]*$")  # a name used as it is in an array's path


def _segment(name: str) -> str:
    return name if SAFE.match(name) else "x" + hashlib.sha256(name.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------------- writing
class _Writer:
    """The writer's walk over one detector file: JSON values, its arrays collected by relative path with digests."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.digests: dict[str, str] = {}

    def array(self, path: tuple, a) -> str:
        a = np.ascontiguousarray(a, np.float64)
        rel = "arrays/" + "/".join(_segment(str(k)) for k in path) + ".npy"
        if rel in self.files:
            raise FormatError(f"two arrays at {rel}: rename one of {path}")
        buf = _io.BytesIO()
        np.save(buf, a, allow_pickle=False)
        self.files[rel], self.digests[rel] = buf.getvalue(), array_digest(a)
        return rel

    def encode(self, h, v, path: tuple):
        """Return the JSON value of ``v`` of type ``h``, its arrays written at ``path`` (the keys leading to it)."""
        h = _inner(h)
        if v is None:
            return None
        if h is np.ndarray:
            a = np.asarray(v, np.float64)
            return float(a) if a.ndim == 0 else self.array(path, a)
        if dataclasses.is_dataclass(h):
            return self.record(h, v, path)
        args = typing.get_args(h)
        if typing.get_origin(h) is dict and args and dataclasses.is_dataclass(args[1]):
            return {k: self.record(args[1], x, (*path, k), _key_field(args[1])) for k, x in v.items()}
        if typing.get_origin(h) is dict and args:
            return {k: self.encode(args[1], x, (*path, k)) for k, x in v.items()}
        if typing.get_origin(h) is tuple:
            return [self.encode(_item(h, i), x, (*path, i)) for i, x in enumerate(v)]
        return v

    def record(self, t, x, path: tuple, key: str | None = None) -> dict:
        """Return a record as the mapping of its fields not at their default; ``key``: the field that is the key of
        the dict holding it (not stored)."""  # noqa: D205, D209
        if key and getattr(x, key) != path[-1]:
            raise FormatError(f"{'/'.join(map(str, path))}: the record kept under {path[-1]!r} is named "
                              f"{getattr(x, key)!r}: key it by its own {key}")  # fmt: skip
        return {f: self.encode(h, getattr(x, f), (*path, f)) for f, h in field_types(t).items()
                if f != key and not _at_default(t, f, getattr(x, f))}  # fmt: skip

    def table(self, t, rows, name: str) -> list:
        """Return a table's rows; its vector fields written as one array each."""
        hints = field_types(t)
        for f in (f for f, h in hints.items() if _is_vector(h)):
            self.array((name, f), np.array([getattr(r, f) for r in rows], np.float64).reshape(len(rows), -1))
        return [[self.encode(h, getattr(r, f), (name, i, f)) for f, h in hints.items() if not _is_vector(h)]
                for i, r in enumerate(rows)]  # fmt: skip


def to_files(df: schema.DetectorFile) -> dict[str, bytes]:
    """Return every file of the detector file by relative path (detector.json last, with the manifest)."""
    w, hints = _Writer(), field_types(schema.DetectorFile)
    docs: dict[str, dict] = {file: {} for file in FILES}
    for file, names in FILES.items():
        for k in names:
            v = getattr(df, k)
            if _is_table(hints[k]):
                docs[file][k] = w.table(_item(hints[k], 0), v, k)
            elif not _at_default(schema.DetectorFile, k, v):
                docs[file][k] = w.encode(hints[k], v, (k,))
    files = dict(w.files)
    files.update({k: dumps(v).encode() for k, v in docs.items()})
    manifest = {k: w.digests.get(k) or sha256_bytes(files[k]) for k in sorted(files)}
    head = {k: w.encode(hints[k], getattr(df, k), (k,)) for k in HEAD}
    files[HEAD_FILE] = dumps(dict(head, schema=schema.SCHEMA, manifest=manifest)).encode()
    return files


def content_digest(files: dict[str, bytes], sensors: dict[str, bytes] | None = None) -> str:
    """Return the content digest: sha256 of the canonical manifest of the content files and of ``sensors``' files (a
    sensor set's, ``lumen.format.sensorsets.files``: its groups, value sets and id map) when given."""  # noqa: D205, D209
    head = json.loads(files[HEAD_FILE])
    manifest = dict(head["manifest"], **{k: sha256_bytes(v) for k, v in (sensors or {}).items()})
    return sha256_bytes(dumps(dict(manifest=manifest)).encode())


def save(df: schema.DetectorFile, path, overwrite: bool = False) -> Path:
    """Write the detector file at ``path`` (a ``.lumen`` directory) atomically; refuse to overwrite unless asked.

    Written in full under a temporary name, then renamed into place (``lumen.format.atomic.directory``): a reader sees
    the old file, none, or the new one whole.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FormatError(f"{path} exists: a detector file is never overwritten (pass overwrite=True)")
    with atomic.directory(path, exists="replace" if overwrite else "raise") as tmp:
        write_into(df, tmp)
    return path


def write_into(df: schema.DetectorFile, root: Path) -> None:
    """Write the detector file's files into the directory ``root`` (not atomic: ``save`` places it whole)."""
    for rel, data in to_files(df).items():
        f = Path(root) / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)


# ------------------------------------------------------------------------------------------------------- reading
class _Reader:
    """The reader's walk: the writer's inverted by the records' types, every file checked against the manifest."""

    def __init__(self, root: Path, manifest: dict):
        self.root, self.manifest = root, manifest

    def json(self, rel: str) -> dict:
        data = (self.root / rel).read_bytes()
        if self.manifest.get(rel) != sha256_bytes(data):
            raise FormatError(f"{self.root}: {rel} does not match the manifest (changed since it was written): "
                              "convert the detector again")  # fmt: skip
        return json.loads(data)

    def npy(self, rel: str) -> np.ndarray:
        a = np.load(_io.BytesIO((self.root / rel).read_bytes()), allow_pickle=False)
        if self.manifest.get(rel) != array_digest(a):
            raise FormatError(f"{self.root}: {rel} does not match the manifest (changed since it was written): "
                              "convert the detector again")  # fmt: skip
        return a

    def decode(self, h, v, where: str):
        """Return the value of type ``h`` stored as ``v`` (``where``: its path, for a refusal)."""
        h = _inner(h)
        if v is None:
            return None
        if h is np.ndarray:
            return self.npy(v) if isinstance(v, str) else np.asarray(v, np.float64)
        if dataclasses.is_dataclass(h):
            return self.record(h, v, where)
        args = typing.get_args(h)
        if typing.get_origin(h) is dict and args:
            if dataclasses.is_dataclass(args[1]):
                key = _key_field(args[1])
                return {k: self.record(args[1], v[k], f"{where}/{k}", {key: k}) for k in v}
            return {k: self.decode(args[1], x, f"{where}/{k}") for k, x in v.items()}
        if typing.get_origin(h) is tuple:
            return tuple(self.decode(_item(h, i), x, where) for i, x in enumerate(v))
        return dict(v) if h is dict else v

    def record(self, t, v, where: str, given: dict | None = None):
        if not isinstance(v, dict):
            raise FormatError(f"{self.root}: {where} is not a mapping of {t.__name__}'s fields")
        hints = field_types(t)
        unknown = sorted(set(v) - set(hints) - set(given or {}))
        missing = [f for f in hints if f not in v and f not in (given or {}) and _default(t, f) is dataclasses.MISSING]
        if unknown or missing:
            raise FormatError(f"{self.root}: {where} ({t.__name__}): unknown fields {unknown}, missing fields "
                              f"{missing}: convert the detector again")  # fmt: skip
        return t(**(given or {}), **{f: self.decode(hints[f], x, f"{where}/{f}") for f, x in v.items()})

    def table(self, t, rows: list, name: str) -> tuple:
        hints = field_types(t)
        cols = {f: self.npy(f"arrays/{name}/{f}.npy") for f, h in hints.items() if _is_vector(h)}
        names = [f for f in hints if f not in cols]
        out = []
        for i, row in enumerate(rows):
            if len(row) != len(names):
                raise FormatError(f"{self.root}: {name}[{i}] has {len(row)} fields, not {len(names)} ({names})")
            kw = {f: self.decode(hints[f], x, f"{name}[{i}]/{f}") for f, x in zip(names, row, strict=True)}
            out.append(t(**kw, **{f: tuple(a[i].tolist()) for f, a in cols.items()}))
        return tuple(out)


def read(path) -> schema.DetectorFile:
    """Read a detector file, checking its format version, its fields and every file against the manifest."""
    root = Path(path)
    head = json.loads((root / HEAD_FILE).read_text())
    check_format(head.get("schema"), schema.SCHEMA, root, "convert the detector again")
    r, hints = _Reader(root, head["manifest"]), field_types(schema.DetectorFile)
    kw: dict[str, typing.Any] = {k: r.decode(hints[k], head[k], k) for k in HEAD if k in head}
    for file, names in FILES.items():
        doc = r.json(file)
        unknown = sorted(set(doc) - set(names))
        if unknown:
            raise FormatError(f"{root}: {file} has unknown fields {unknown}: convert the detector again")
        for k in names:
            if _is_table(hints[k]):
                kw[k] = r.table(_item(hints[k], 0), doc.get(k, []), k)
            elif k in doc:
                kw[k] = r.decode(hints[k], doc[k], k)
    try:
        return schema.DetectorFile(**kw)
    except TypeError as err:
        raise FormatError(f"{root}: {err}: convert the detector again") from None


def equal(a: schema.DetectorFile, b: schema.DetectorFile) -> bool:
    """Whether two detector files are the same value (every file byte-identical)."""
    return to_files(a) == to_files(b)


__all__ = ["FormatError", "check_format", "content_digest", "dumps", "equal", "read", "save", "to_files"]
