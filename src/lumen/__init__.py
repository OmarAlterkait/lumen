"""lumen: a differentiable optical-photon simulation for neutrino detectors, written in JAX."""

from importlib.metadata import PackageNotFoundError, version

from lumen import gdml, media, readout, sources, surfaces
from lumen.api import Detector, Light, convert, load, simulate
from lumen.models import medium_model, surface_model
from lumen.params import Param, Params
from lumen.readout import response
from lumen.recipe import Recipe
from lumen.sources import Photons, concat, source

LAZY_MODULES = ("diagnostics", "extend", "plugins", "validate")  # lumen.<name> after a bare ``import lumen``


def __getattr__(name):  # lazily: these load the transport, the checks' statistics
    if name == "Controls":
        from lumen.transport.step import Controls

        return Controls
    if name in LAZY_MODULES:
        import importlib

        return importlib.import_module(f"lumen.{name}")
    raise AttributeError(f"module 'lumen' has no attribute {name!r}")


try:
    __version__ = version("lumen")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "Controls",
    "Detector",
    "Light",
    "Param",
    "Params",
    "Photons",
    "Recipe",
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
    "surface_model",
    "surfaces",
]
