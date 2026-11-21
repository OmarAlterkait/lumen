"""The reference manifest: which reference products each check compares against, and their sha256.

    spec = references.spec("charge", "sk")          # the check's entry: its detector, reference id and settings
    ref = references.product(spec["reference"])     # the product's directory, fetched once, every file verified

lumen's own manifest is ``lumen/compare/references.json``; any other is given by its path (``extra``: a detector
outside lumen is validated by the same commands with its own manifest of the same form, which may name its own store,
``"store"``). A product lives in the store, ``LUMEN_DATA`` (else ``data`` under lumen's cache root, ``LUMEN_CACHE``)
unless the manifest names one: ``<store>/<id>/``. A product the store lacks is fetched from its release asset into the
store: the entry's ``url`` (a tar archive of the product's files, any URL urllib reads, ``file://`` too) and the
archive's ``sha256``; with ``LUMEN_ASSETS`` set (a mirror's base URL), the archive of the same file name under it.
Every file is checked against its sha256 before a check reads it. A check never writes its records
into a store: ``inside_store`` tells the record writer where not to write.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

from lumen import _env
from lumen.format.io import check_format, sha256_file

SCHEMA = "lumen-references/1"
_VERIFIED: dict = {}


class ReferenceStoreError(LookupError):
    """A check or product the manifests do not list, a missing store, or a file whose sha256 differs."""


def _read(path) -> dict:
    m = json.loads(Path(path).read_text())
    check_format(m.get("schema"), SCHEMA, path, "write the manifest again for this lumen")
    return m


def manifests(extra=()) -> list:
    """Return every manifest: lumen's, then the ``extra`` paths given."""
    return [_read(resources.files("lumen.compare") / "references.json")] + [_read(p) | {"_path": str(p)}
                                                                               for p in extra]  # fmt: skip


def spec(check: str, name: str, extra=()) -> dict:
    """Return the entry of ``check`` named ``name`` (a detector, or a reference such as "wcte_g4")."""
    found = [dict(e, _store=m.get("store")) for m in manifests(extra) for e in m["checks"].get(check, ())
             if e["name"] == name]  # fmt: skip
    if not found:
        known = sorted({e["name"] for m in manifests(extra) for e in m["checks"].get(check, ())})
        raise ReferenceStoreError(f"no {check} check named {name!r}; the manifests list {known}")
    if len(found) > 1:  # a name is one entry across the manifests: none shadows another
        raise ReferenceStoreError(f"{len(found)} manifests name a {check} check {name!r}: rename one")
    return found[0]


REFERENCE_ENGINE = "g4ref"  # the producer of the references lumen follows: stock Geant4 (10.7.4) on the same GDML


def _entry(pid: str, extra=()) -> dict:
    return next((m["products"][pid] for m in manifests(extra) if pid in m.get("products", {})), {})


def followed(pid: str, extra=()) -> bool:
    """Return whether product ``pid`` comes from the reference engine lumen follows (``REFERENCE_ENGINE``)."""
    producer = _entry(pid, extra).get("producer")
    return isinstance(producer, dict) and producer.get("binary") == REFERENCE_ENGINE


def original(pid: str, extra=()) -> str | None:
    """Return the product on the original GDML that product ``pid`` pairs with: ``pid`` the reference engine on the
    overlap-free export of the detector's geometry (its producer's ``export``: ``lumen.compare.ownership``, the
    export's sha256 and ``original``, the engine's run of the same sources on the GDML the detector file was converted
    from), else None."""  # noqa: D205, D209
    export = (_entry(pid, extra).get("producer") or {}).get("export") if followed(pid, extra) else None
    return export.get("original") if isinstance(export, dict) else None


def gated(check: str, spec_: dict, extra=()) -> bool:
    """Return whether an entry's rows gate (pass or fail) or are reported only.

    One rule for every detector: the reference engine (``followed``) on the overlap-free export of the detector's
    geometry (``original``) gates. lumen resolves overlapping volumes statically (a volume proved inside a sibling
    owns the shared region, else the one placed first), and the export writes that resolution out, so Geant4
    navigates the geometry lumen simulates; on a GDML whose volumes really overlap Geant4's answer depends on the
    path a photon takes, so the engine on the original GDML is reported, never gated (the charge check shows
    G4(export) / G4(original), the overlap effect). Another simulation of the detector (an experiment's own software
    on another Geant4 build) gates where the detector has no reference from the engine on its export, and is
    otherwise a reported cross-check: where the two references differ, lumen follows the engine (ANNIE's WCSim on
    Geant4 10.1.3 loses 7 to 8 % of the light at its inner-structure frame against Geant4 10.7.4,
    docs/explanation/validation.md).
    """
    if followed(spec_["reference"], extra):
        return original(spec_["reference"], extra) is not None
    same = [e for m in manifests(extra) for e in m["checks"].get(check, ()) if e["detector"] == spec_["detector"]]
    return not any(original(e["reference"], extra) is not None for e in same)


def names(check: str, extra=()) -> list:
    """Return the names the manifests list for a check."""
    return [e["name"] for m in manifests(extra) for e in m["checks"].get(check, ())]


def store(spec_: dict | None = None) -> Path:
    """Return the store a spec's products live in: its manifest's, else ``LUMEN_DATA`` (``data`` under lumen's cache
    root when unset)."""  # noqa: D205, D209
    return Path((spec_ or {}).get("_store") or _env.data_dir())


def product(pid: str, spec_: dict | None = None, extra=()) -> Path:
    """Return the directory of product ``pid``, fetched from its release asset when the store lacks it (``fetch``),
    after checking every file's sha256 against the manifest."""  # noqa: D205, D209
    entry = next((m["products"][pid] for m in manifests(extra) if pid in m.get("products", {})), None)
    if entry is None:
        raise ReferenceStoreError(f"no reference product {pid!r} in the manifests")
    d = store(spec_) / pid
    if (pid, str(d)) not in _VERIFIED:
        if not d.exists() and entry.get("url"):
            fetch(entry, d)
        for name, digest in entry["files"].items():
            if not (d / name).exists():
                raise ReferenceStoreError(f"{d / name}: missing from the store")
            if sha256_file(d / name) != digest:
                raise ReferenceStoreError(f"{d / name}: sha256 differs from the manifest ({digest})")
        _VERIFIED[(pid, str(d))] = entry
    return d


def fetch(entry: dict, d: Path) -> Path:
    """Fetch a product's release asset (its entry's ``url``, or its file name under the ``LUMEN_ASSETS`` mirror) into
    directory ``d``: the archive checked against the entry's ``sha256``, only the files the entry lists taken from it,
    and ``d`` placed whole (a reader never sees part of it; of processes fetching it at once, the first to finish places
    it)."""  # noqa: D205, D209
    url, mirror = entry["url"], _env.assets_url()
    if mirror:
        url = f"{mirror.rstrip('/')}/{url.rsplit('/', 1)[-1]}"
    d.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=d.parent, prefix=f".{d.name}.") as tmp:
        archive, out = Path(tmp) / "asset.tar", Path(tmp) / d.name
        try:
            with urllib.request.urlopen(url) as r, open(archive, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        except OSError as e:
            raise ReferenceStoreError(f"{url}: cannot fetch the release asset ({e})") from e
        if sha256_file(archive) != entry["sha256"]:
            raise ReferenceStoreError(f"{url}: sha256 differs from the manifest ({entry['sha256']})")
        out.mkdir()
        with tarfile.open(archive) as t:
            names = set(t.getnames())
            for name in entry["files"]:
                member = t.extractfile(name) if name in names else None
                if member is None:
                    raise ReferenceStoreError(f"{url}: {name} missing from the release asset")
                with open(out / Path(name).name, "wb") as g:
                    shutil.copyfileobj(member, g, 1 << 20)
        try:
            out.rename(d)
        except OSError:
            if not d.is_dir():
                raise
    return d


def provenance(pid: str, extra=()) -> dict:
    """Return the manifest entry of a product (its producer, inputs and files)."""
    for m in manifests(extra):
        if pid in m.get("products", {}):
            return m["products"][pid]
    raise ReferenceStoreError(f"no reference product {pid!r} in the manifests")


def inside_store(path, extra=()) -> bool:
    """Return whether ``path`` lies inside a reference store (where a check never writes): ``store()``, or the store
    of lumen's manifest or of an ``extra`` one."""  # noqa: D205, D209
    roots = {Path(m["store"]) for m in manifests(extra) if m.get("store")} | {store()}
    p = Path(path).resolve()
    return any(p == r.resolve() or r.resolve() in p.parents for r in roots)
