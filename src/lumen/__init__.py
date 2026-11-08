"""lumen: a differentiable optical-photon simulation for neutrino detectors, written in JAX."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lumen")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
