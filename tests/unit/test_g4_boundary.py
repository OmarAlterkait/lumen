"""The facet paths against Geant4 11.3.0's G4OpBoundaryProcess on one flat interface (tests/data/g4_boundary).

Each case is a Geant4 run of tools/g4ref/boundary.cc (tools/g4ref/boundary_cases.sh): photons along one direction
onto an interface with one optical surface, configured so that the facet path is the only outcome (R = 1, the lobe's
constant 1). The last direction's histogram in (dz, |phi|) is compared with the same histogram of lumen's path, drawn
by the lobe's own sampler (chi2 with both counts' errors).
"""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from lumen.surfaces import facets
from lumen.tools.stats import ALPHA, SUITE_TESTS

F32 = jnp.float32
DATA = Path(__file__).resolve().parents[1] / "data" / "g4_boundary"
MANIFEST = json.loads((DATA / "MANIFEST.json").read_text())
P_MIN = ALPHA / SUITE_TESTS
UP = jnp.array([0.0, 0.0, 1.0], F32)
N = 1 << 20


def _uniforms(seed, n):
    """Return (n, 3) uniforms on the grid ``facets.seed_key`` reads exactly."""
    u = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), F32, 1e-7, 1 - 1e-7)
    return ((u * 2.0**23).astype(jnp.int32).astype(F32) + 0.5) * 2.0**-23


def _cases(kind):
    return sorted(k for k, c in MANIFEST["cases"].items() if c["model"] == kind[0] and c["type"] == kind[1])


def _chi2_p(g4, ours):
    """Return the p-value of two histograms' agreement in shape (chi2, both counts' errors)."""
    g4, ours = np.ravel(g4).astype(np.float64), np.ravel(ours).astype(np.float64)
    n1, n2 = g4.sum(), ours.sum()
    use = (g4 + ours) > 0
    c2 = float(np.sum((g4[use] / n1 - ours[use] / n2) ** 2 / (g4[use] / n1**2 + ours[use] / n2**2)))
    return float(stats.chi2.sf(c2, int(use.sum()) - 1)), c2, int(use.sum()) - 1


def _histogram(wo, z):
    return np.histogram2d(wo[:, 2], np.abs(np.arctan2(wo[:, 1], wo[:, 0])), bins=[z["edges_z"], z["edges_phi"]])[0]


@pytest.mark.parametrize("name", _cases(("unified", "dielectric_metal")))
def test_the_unified_lobe_path_is_geant4s(name):
    """Geant4's unified ground dielectric_metal with R = 1 and SPECULARLOBECONSTANT 1: the facet path, sigma_alpha up
    to SIGMA_MAX (1 rad) and grazing incidence there."""
    case = MANIFEST["cases"][name]
    theta = math.radians(case["theta"])
    wi = jnp.array([math.sin(theta), 0.0, -math.cos(theta)], F32)
    p = {"sigma_alpha": F32(case["value"])}
    draw = jax.jit(jax.vmap(lambda u: facets.UNIFIED.path(p, wi, UP, u, lambda c, w0, w1: c, ())[2]))
    wo = np.asarray(draw(_uniforms(case["seed"], N)), np.float64)
    with np.load(DATA / case["file"]) as z:
        pv, c2, dof = _chi2_p(z["hist"], _histogram(wo, z))
    assert pv > P_MIN, (name, c2, dof)
