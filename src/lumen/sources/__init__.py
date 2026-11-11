"""Light sources: ``emit(params, key, n, first_id, samples) -> Emission``.

    src = lumen.sources.Point((0.0, 0.0, 0.0), n_photons=1e6, lam=400.0)
    light = lumen.simulate(det, src, samples=1 << 20, seed=1)

Weights sum in expectation to the source's photons; fields are traced parameters ``source/<field>``;
``drawn_direction``: directions drawn from a density, the only case with a position derivative.
A source of your own is one function, ``@lumen.source`` (``lumen.sources.function``).
"""

from lumen.sources._base import (
    PREFIX,
    CallableSource,
    GradientNotAvailable,
    Source,
    gaussian,
    photon_uniforms,
    uniforms,
)
from lumen.sources.arrays import FromArrays
from lumen.sources.cherenkov import CherenkovSegment
from lumen.sources.function import (
    FunctionSource,
    Photons,
    SourceError,
    concat,
    cone_about,
    gaussian_spot,
    grad_logpdf,
    reweight,
    source,
)
from lumen.sources.point import Point
from lumen.sources.scintillation import ScintillationPoint, ScintillationSegment
from lumen.sources.steps import Geant4Steps

__all__ = [
    "PREFIX", "CallableSource", "CherenkovSegment", "FromArrays", "FunctionSource", "Geant4Steps",
    "GradientNotAvailable", "Photons", "Point", "ScintillationPoint", "ScintillationSegment", "Source", "SourceError",
    "concat", "cone_about", "gaussian", "gaussian_spot", "grad_logpdf", "photon_uniforms", "reweight", "source",
    "uniforms",
]  # fmt: skip
