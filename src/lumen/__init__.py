"""lumen: a differentiable optical-photon simulation for neutrino detectors, written in JAX."""

from importlib.metadata import PackageNotFoundError, distribution

from lumen import gdml, media, readout, sources
from lumen.api import Detector, DetectorEntry, Light, convert, load, simulate
from lumen.format.sensorsets import SensorGroup
from lumen.models import medium_model
from lumen.params import Param, Params
from lumen.readout import response
from lumen.recipe import Recipe
from lumen.sources import Photons, concat, source

LAZY_MODULES = (
    "compare",
    "diagnostics",
    "sensors",
    "tools",
)  # lumen.<name> after a bare ``import lumen``


def __getattr__(name):  # lazily: these load the transport, the checks' statistics
    if name == "Controls":
        from lumen.transport.step import Controls

        return Controls
    if name in LAZY_MODULES:
        import importlib

        return importlib.import_module(f"lumen.{name}")
    raise AttributeError(f"module 'lumen' has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__, *LAZY_MODULES})


def _version() -> str:
    """Return the version of the lumen imported: the installed metadata's when it was installed from this source tree,
    else the tree's own (hatch-vcs's form from ``git describe``, or a git archive's ``.git_archival.txt``; "0.0.0",
    the fallback, without either)."""  # noqa: D205, D209
    import json
    import subprocess
    from pathlib import Path

    here = Path(__file__).resolve().parent
    try:
        dist = distribution("lumen")
        url = json.loads(dist.read_text("direct_url.json") or "{}")
        src = url.get("url", "").removeprefix("file://") if url.get("dir_info", {}).get("editable") else None
        where = Path(src).resolve() / "src" / "lumen" if src else Path(dist.locate_file("lumen")).resolve()
        if where == here:
            return dist.version
    except (PackageNotFoundError, OSError, ValueError):
        pass
    try:
        out = subprocess.run(["git", "-C", str(here), "describe", "--tags", "--long", "--dirty", "--match", "v[0-9]*"],
                             capture_output=True, text=True, timeout=10).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        out = ""
    archival = here.parents[1] / ".git_archival.txt"  # a git archive of the tree: its describe-name, filled in
    if not out and archival.is_file():
        lines = dict(x.split(": ", 1) for x in archival.read_text().splitlines() if ": " in x)
        out = "" if "$Format" in lines.get("describe-name", "$Format") else lines["describe-name"]
        out = out if out.count("-") >= 2 else out + "-0-g" + lines.get("node", "")[:7]  # on the tag itself
    parts = out.removeprefix("v").split("-")
    if len(parts) < 3 or not parts[0].replace(".", "").isdigit():
        return "0.0.0"
    tag, n, sha, dirty = parts[0], int(parts[1]), parts[2], len(parts) > 3
    if n:  # the next patch's dev release, as hatch-vcs (setuptools-scm's guess-next-dev) numbers it
        *major, patch = tag.split(".")
        tag = ".".join([*major, str(int(patch) + 1)]) + f".dev{n}"
    return tag + (f"+{sha}" if n or dirty else "") + (".dirty" if dirty else "")


__version__ = _version()

__all__ = [
    "Controls",
    "Detector",
    "DetectorEntry",
    "Light",
    "Param",
    "Params",
    "Photons",
    "Recipe",
    "SensorGroup",
    "__version__",
    "concat",
    "convert",
    "gdml",
    "load",
    "media",
    "medium_model",
    "readout",
    "response",
    "simulate",
    "source",
    "sources",
]
