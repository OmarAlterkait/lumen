"""The reference manifest: which stored products each check compares against, and their sha256.

    spec = references.spec("charge", "sk")          # the check's entry: its detector, reference id and settings
    ref = references.product(spec["reference"])     # the product's directory, every file's sha256 verified

lumen's own manifest is ``lumen/detectors/references.json``. A detector package registered under ``lumen.detectors``
adds its own (``DetectorPackage.references``, the path of a manifest of the same form), so a detector outside lumen
is validated by the same commands with no change here; its manifest may name its own store (``"store"``). Products
live read-only in the store, ``LUMEN_DATA`` unless the manifest names one: ``<store>/<id>/``. A check never writes
into a store: ``inside_store`` tells the record writer where not to write.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

from lumen import _env

SCHEMA = "lumen-references/1"
_VERIFIED: dict = {}


class ReferenceStoreError(LookupError):
    """A check or product the manifests do not list, a missing store, or a file whose sha256 differs."""


def _read(path) -> dict:
    m = json.loads(Path(path).read_text())
    if m.get("schema") != SCHEMA:
        raise ReferenceStoreError(f"{path}: schema {m.get('schema')!r}, this lumen reads {SCHEMA}")
    return m


def manifests(extra=()) -> list:
    """Return every manifest: lumen's, each registered detector package's, then the ``extra`` paths given."""
    from lumen import detectors, plugins

    out = [_read(resources.files("lumen.detectors") / "references.json")]
    detectors.register_builtins()
    for name in plugins.detectors.names():
        path = getattr(plugins.detectors.get(name), "references", None)
        if path and all(m.get("_path") != str(path) for m in out):
            out.append(_read(path) | {"_path": str(path)})
    return out + [_read(p) | {"_path": str(p)} for p in extra]


def spec(check: str, name: str, extra=()) -> dict:
    """Return the entry of ``check`` named ``name`` (a detector, or a reference such as "wcte_g4")."""
    found = [dict(e, _store=m.get("store")) for m in manifests(extra) for e in m["checks"].get(check, ())
             if e["name"] == name]  # fmt: skip
    if not found:
        known = sorted({e["name"] for m in manifests(extra) for e in m["checks"].get(check, ())})
        raise ReferenceStoreError(f"no {check} check named {name!r}; the manifests list {known}")
    return found[-1]


def names(check: str, extra=()) -> list:
    """Return the names the manifests list for a check."""
    return [e["name"] for m in manifests(extra) for e in m["checks"].get(check, ())]


def store(spec_: dict | None = None) -> Path:
    """Return the store a spec's products live in (its manifest's, else ``LUMEN_DATA``)."""
    root = (spec_ or {}).get("_store") or _env.data_dir()
    if root is None:
        raise ReferenceStoreError("no reference store: set LUMEN_DATA to the directory holding the reference products")
    return Path(root)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def product(pid: str, spec_: dict | None = None, extra=()) -> Path:
    """Return the directory of product ``pid`` after checking every file's sha256 against the manifest."""
    entry = next((m["products"][pid] for m in manifests(extra) if pid in m.get("products", {})), None)
    if entry is None:
        raise ReferenceStoreError(f"no reference product {pid!r} in the manifests")
    d = store(spec_) / pid
    if (pid, str(d)) not in _VERIFIED:
        for name, digest in entry["files"].items():
            if not (d / name).exists():
                raise ReferenceStoreError(f"{d / name}: missing from the store")
            if _sha256(d / name) != digest:
                raise ReferenceStoreError(f"{d / name}: sha256 differs from the manifest ({digest})")
        _VERIFIED[(pid, str(d))] = entry
    return d


def provenance(pid: str, extra=()) -> dict:
    """Return the manifest entry of a product (its producer, inputs and files)."""
    for m in manifests(extra):
        if pid in m.get("products", {}):
            return m["products"][pid]
    raise ReferenceStoreError(f"no reference product {pid!r} in the manifests")


def inside_store(path) -> bool:
    """Return whether ``path`` lies inside a reference store (where a check never writes)."""
    roots = {Path(m["store"]) for m in manifests() if m.get("store")}
    if _env.data_dir() is not None:
        roots.add(_env.data_dir())
    p = Path(path).resolve()
    return any(p == r.resolve() or r.resolve() in p.parents for r in roots)
