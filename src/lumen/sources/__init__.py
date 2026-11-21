"""Light sources: ``emit(params, key, n, first_id, samples) -> Photons`` (``lumen.record``).

    src = lumen.sources.Point((0.0, 0.0, 0.0), n_photons=1e6, lam=400.0)
    light = lumen.simulate(det, src, samples=1 << 20, seed=1)

Weights sum in expectation to the source's photons; fields are traced parameters ``source/<field>``; a field that moves
where the light starts (``Source.MOVING``) has no derivative in this version.
A source of your own is one function, ``@lumen.source`` (``lumen.sources.function``).
"""

from lumen._core.errors import GradientNotAvailable
from lumen.sources._base import (
    PREFIX,
    Source,
    gaussian,
    photon_uniforms,
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
    reweight,
    source,
)
from lumen.sources.point import Point
from lumen.sources.scintillation import ScintillationPoint, ScintillationSegment
from lumen.sources.steps import CherenkovSteps, ScintillationSteps

__all__ = [
    "PREFIX", "CherenkovSegment", "CherenkovSteps", "FromArrays", "FunctionSource",
    "GradientNotAvailable", "Photons", "Point", "ScintillationPoint", "ScintillationSegment", "ScintillationSteps",
    "Source", "SourceError",
    "concat", "cone_about", "gaussian", "gaussian_spot", "photon_uniforms", "reweight", "source",
]  # fmt: skip
