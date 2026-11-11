"""lumen: a differentiable optical-photon simulation for neutrino detectors, written in JAX."""

from importlib.metadata import PackageNotFoundError, version

from lumen import gdml, readout, sources
from lumen.api import Detector, Light, convert, load, simulate
from lumen.params import Param, Params
from lumen.readout import response
from lumen.recipe import Recipe
from lumen.sources import Photons, concat, source

try:
    __version__ = version("lumen")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
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
    "readout",
    "response",
    "simulate",
    "source",
    "sources",
]
