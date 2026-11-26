"""The facet paths against Geant4 11.3.0's G4OpBoundaryProcess on one flat interface (tests/data/g4_boundary).

Each case is a Geant4 run of tools/g4ref/boundary.cc (tools/g4ref/boundary_cases.sh): photons along one direction
onto an interface with one optical surface, configured so that the facet path is the only outcome (R = 1, the lobe's
constant 1; T = 0). The last direction's histogram in (dz, |phi|) is compared with the same histogram of lumen's path,
drawn by the lobe's own sampler (the two-sample chi2 on pooled counts).

A definite linear state (the s and p cases) that meets several facets of one GLISUR dielectric surface ends in
another state in Geant4 than in lumen (``lumen.surfaces.polar``: Geant4's DielectricDielectric takes the in-plane
part of the state as a magnitude and mirrors the real vector at total reflection); those cases are compared with
Geant4's two conventions emulated in the path, the random linear ones (unpolarised light) as lumen draws them.
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
N, N_DIELECTRIC = 1 << 20, 1 << 18


def _uniforms(seed, n):
    """Return (n, 3) uniforms on the grid ``facets.seed_key`` reads exactly."""
    u = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), F32, 1e-7, 1 - 1e-7)
    return ((u * 2.0**23).astype(jnp.int32).astype(F32) + 0.5) * 2.0**-23


def _cases(kind):
    return sorted(k for k, c in MANIFEST["cases"].items() if c["model"] == kind[0] and c["type"] == kind[1])


def _chi2_p(g4, ours):
    """Return (p-value, chi2, dof) of two histograms' agreement in shape: the two-sample chi2 on pooled counts,
    sum (a / n1 - b / n2)^2 / ((a + b) / (n1 n2)) (a bin's variance from both samples' counts, so a sparse bin one
    sample misses weighs as it should)."""
    g4, ours = np.ravel(g4).astype(np.float64), np.ravel(ours).astype(np.float64)
    n1, n2 = g4.sum(), ours.sum()
    use = (g4 + ours) > 0
    c2 = float(np.sum((g4[use] / n1 - ours[use] / n2) ** 2 / ((g4[use] + ours[use]) / (n1 * n2))))
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


def _polarisation(case, wi, n, seed):
    """Return (er, ei) per photon as the Geant4 run launched them: s (along y), p (wi x s) or random linear."""
    s = np.array([0.0, 1.0, 0.0])
    p = np.cross(np.asarray(wi, np.float64), s)
    if case["pol"] == "random":
        phi = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, n)[:, None]
        er = np.cos(phi) * p + np.sin(phi) * s
    else:
        er = np.tile(s if case["pol"] == "s" else p, (n, 1))
    return jnp.asarray(er, F32), jnp.zeros((n, 3), F32)


def _geant4_conventions(interface):
    """Return ``facets._interface`` with Geant4's DielectricDielectric conventions for the state after a facet: the
    in-plane part as a magnitude (E1_parl = |E1pl|) and, at total reflection, the real vector's mirror (r_s, r_p) =
    (-1, 1) (G4OpBoundaryProcess.cc, 11.3.0)."""

    def f(w0, m, n_a, n_b, er, ei):
        R, s_ax, (e_s, e_p), amps = interface(w0, m, n_a, n_b, er, ei)
        cos1 = -jnp.sum(w0 * m, -1)
        tir = (n_a / n_b) ** 2 * (1.0 - cos1 * cos1) >= 1.0
        r_s = jnp.where(tir, -1.0 + 0.0j, amps[0]).astype(jnp.complex64)
        r_p = jnp.where(tir, 1.0 + 0.0j, amps[1]).astype(jnp.complex64)
        return R, s_ax, (e_s, jnp.where(tir, e_p, jnp.abs(e_p).astype(e_p.dtype))), (r_s, r_p, amps[2], amps[3])

    return f


@pytest.mark.parametrize("name", _cases(("glisur", "dielectric_dielectric")))
def test_the_glisur_dielectric_path_is_geant4s(name, monkeypatch):
    """Geant4's glisur ground dielectric_dielectric (R = 1, T = 0: the facet loop of DielectricDielectric): the last
    direction on either side, with the photon's polarisation (random linear as lumen draws it; s or p with Geant4's
    conventions for the state between facets); the path's far side is the side its last direction points to."""
    case = MANIFEST["cases"][name]
    if case["pol"] != "random":
        monkeypatch.setattr(facets, "_interface", _geant4_conventions(facets._interface))
    lobe = facets._dielectric_lobe("glisur_dielectric", 0.0)
    theta = math.radians(case["theta"])
    wi = jnp.array([math.sin(theta), 0.0, -math.cos(theta)], F32)
    er, ei = _polarisation(case, wi, N_DIELECTRIC, case["seed"])
    idx = (F32(case["n1"]), F32(case["n2"]))

    def one(u, a, b):
        p = {"sigma_alpha": F32(1.0 - case["value"]), "idx": idx, "er": a, "ei": b}
        _, wo, far, _ = lobe.path(p, wi, UP, u, lambda c, ev: c, ())
        return wo, far

    wo, far = jax.jit(jax.vmap(one))(_uniforms(case["seed"], N_DIELECTRIC), er, ei)
    wo, far = np.asarray(wo, np.float64), np.asarray(far)
    assert np.array_equal(far, wo[:, 2] < 0.0)
    with np.load(DATA / case["file"]) as z:
        pv, c2, dof = _chi2_p(z["hist"], _histogram(wo, z))
    assert pv > P_MIN, (name, c2, dof)
