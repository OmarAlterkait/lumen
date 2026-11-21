"""A detector package's input files, checked by sha256 before a conversion reads them: one pipeline for every package.

    prepare, expected = inputs.package(REFERENCES, GDML_DIR)   # a package's prepare (lumen.convert("package:name"))
    with inputs.gdml_file(path, GDML_DIR) as gdml: ...          # a .gdml as it is; a shipped .gdml.xz decompressed
    inputs.require({"gdml": (path, sha256, where)})             # refuses a missing file or another sha256

A package ships its GDMLs xz-compressed beside ``<gdml_dir>/SOURCES.json`` (each file's source and the sha256 of the
decompressed file). Every other file a conversion reads is a file of one of the package's reference products
(``<data>/<product>/<file>``, its sha256 in the package's manifest) or a file the package records by name
(``recorded``: an input it may not ship, given by path).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import lzma
import shutil
import tempfile
from pathlib import Path


class InputError(FileNotFoundError):
    """An input file that is missing or is not the one the recipe expects (another sha256)."""


def sha256(path) -> str:
    """Return a file's sha256 (hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(expected: dict) -> None:
    """Refuse unless every input is there with its expected sha256: ``{name: (path, sha256, where it is from)}``."""
    problems = []
    for name, (path, want, where) in expected.items():
        p = Path(path)
        if not p.is_file():
            problems.append(f"  {name}: {p} is missing (expected sha256 {want}; from {where})")
        elif (got := sha256(p)) != want:
            problems.append(f"  {name}: {p} has sha256 {got}, not the expected {want} (from {where})")
    if problems:
        raise InputError("the detector's inputs are not all there:\n" + "\n".join(problems))


def shipped(gdml_dir) -> dict:
    """Return the provenance of a package's shipped GDMLs (``<gdml_dir>/SOURCES.json``: file name -> url, sha256)."""
    return json.loads((Path(gdml_dir) / "SOURCES.json").read_text())["files"]


@contextlib.contextmanager
def gdml_file(path, gdml_dir):
    """Yield the GDML to convert: ``path`` itself, or a shipped ``.gdml.xz`` decompressed into a temporary directory
    (under its own name) and checked against its sha256 in ``<gdml_dir>/SOURCES.json``."""  # noqa: D205, D209
    path = Path(path)
    if path.suffix != ".xz":
        yield path
        return
    name = path.name[: -len(".xz")]
    src = shipped(gdml_dir)[name]
    with tempfile.TemporaryDirectory(prefix="lumen-gdml-") as tmp:
        out = Path(tmp) / name
        with lzma.open(path) as f, open(out, "wb") as g:
            shutil.copyfileobj(f, g, 1 << 20)
        require({name: (out, src["sha256"], src["url"])})
        yield out


def package(references, gdml_dir, recorded=None):
    """Return (prepare, expected) of a detector package: its manifest ``references``, its shipped GDMLs' directory
    and ``recorded`` {file name: (sha256, where it is from)} of the inputs it records by name.

    ``prepare(recipe, gdml, inputs)`` yields them as ``lumen.convert("package:name")`` converts them: every file is
    checked against its sha256 before anything reads it (``expected``), a file the data directory lacks is fetched
    first with its reference product, a shipped ``.gdml.xz`` is decompressed and checked (``gdml_file``), and a recipe
    made from its inputs (a callable) is then made from the GDML as converted and the inputs.
    """  # noqa: D205
    references, recorded = Path(references), dict(recorded or {})

    def expected(path) -> tuple[str, str]:
        """Return (sha256, where it is recorded) of a file a conversion reads; refuse one recorded nowhere."""
        path = Path(path)
        files = json.loads(references.read_text())["products"].get(path.parent.name, {}).get("files", {})
        if path.name in files:
            return files[path.name].removeprefix("sha256:"), f"the reference product {path.parent.name}"
        if path.name in recorded:
            return recorded[path.name]
        raise InputError(f"{path}: {references.parent.name} records no sha256 for this file (a conversion reads only "
                         "checked files)")  # fmt: skip

    @contextlib.contextmanager
    def prepare(recipe, gdml, inputs):
        from lumen.compare import references as refs

        named = {"gdml": Path(gdml)} | {k: Path(v) for k, v in inputs.items()}
        products = json.loads(references.read_text())["products"]
        for p in named.values():  # a missing file of a reference product: fetched, each file's sha256 checked
            if not p.exists() and p.parent.name in products:
                refs.product(p.parent.name, {"_store": str(p.parent.parent)}, extra=[references])
        require({k: (p, *expected(p)) for k, p in named.items() if p.suffix != ".xz"})
        with gdml_file(gdml, gdml_dir) as source:
            yield (recipe(source, inputs) if callable(recipe) else recipe), source, inputs

    return prepare, expected
