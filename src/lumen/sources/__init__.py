"""Light sources: ``emit(params, key, n, first_id, samples) -> Emission``.

    src = lumen.sources.Point((0.0, 0.0, 0.0), n_photons=1e6, lam=400.0)
    light = lumen.simulate(det, src, samples=1 << 20, seed=1)

Weights sum in expectation to the source's photons; fields are traced parameters ``source/<field>``;
``drawn_direction``: directions drawn from a density, the only case with a position derivative.
"""

from lumen.sources._base import PREFIX, GradientNotAvailable, Source, gaussian, photon_uniforms, uniforms
from lumen.sources.arrays import FromArrays
from lumen.sources.cherenkov import CherenkovSegment
from lumen.sources.point import Point
from lumen.sources.scintillation import ScintillationPoint, ScintillationSegment
from lumen.sources.steps import Geant4Steps

__all__ = [
    "PREFIX", "CherenkovSegment", "FromArrays", "Geant4Steps", "GradientNotAvailable", "Point", "ScintillationPoint",
    "ScintillationSegment", "Source", "gaussian", "photon_uniforms", "uniforms",
]  # fmt: skip
