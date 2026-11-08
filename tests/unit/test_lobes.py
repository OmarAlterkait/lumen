"""Lobes: every drawn lobe passes the lobe contract (lobe_contract.py); the deterministic maps against float64."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lobe_contract import F32, check_lobe

from lumen.media import phase
from lumen.surfaces import lobes, outcomes

NZ = np.array([0.0, 0.0, 1.0])
TILTED = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def incoming(theta_deg, nrm=NZ):
    """The incoming direction at incidence theta onto the facing normal nrm (in a plane containing nrm)."""
    t = math.radians(theta_deg)
    e1 = np.cross(nrm, [0.3, 0.7, 0.1])
    e1 /= np.linalg.norm(e1)
    return math.sin(t) * e1 - math.cos(t) * nrm


@pytest.mark.parametrize("lobe", [lobes.LAMBERT, lobes.LAMBERT_T])
def test_lambert_contract(lobe):
    r = check_lobe(lobe, {}, incoming(40.0, TILTED), TILTED, label=lobe.name)
    print(f"\n[{lobe.name}] integral {r['total']:.5f} chi2/ndf {r['chi2']:.2f}")


@pytest.mark.parametrize("lobe", [phase.RAYLEIGH, phase.ISOTROPIC])
@pytest.mark.parametrize("wi", [NZ, -NZ, np.array([1.0, 0.0, 0.0]), TILTED])
def test_parameter_free_phase_contract(lobe, wi):
    check_lobe(lobe, {}, wi, NZ, label=lobe.name)


def test_specular_and_refraction_maps_against_float64():
    r = np.random.default_rng(1)
    n = TILTED
    wi = np.stack([incoming(t, n) for t in r.uniform(0, 89.9, 500)])
    spec = np.asarray(jax.vmap(lambda w: lobes.SPECULAR.sample({}, w, jnp.asarray(n, F32), None))(jnp.asarray(wi, F32)))
    np.testing.assert_allclose(spec, wi - 2 * (wi @ n)[:, None] * n, atol=2e-6)
    for eta in (1.0, 1.0 / 1.33, 1.33, 1.5, 1.36 / 1.0):
        wo, tir = (
            np.asarray(x)
            for x in jax.vmap(lambda w: outcomes.refract(w, jnp.asarray(n, F32), F32(eta)))(jnp.asarray(wi, F32))
        )
        ci = -(wi @ n)
        s2 = eta**2 * (1 - ci**2)
        far = np.abs(s2 - 1) > 1e-4
        np.testing.assert_array_equal(tir[far], (s2 >= 1)[far])
        ok = far & ~tir
        ct = np.sqrt(1 - s2[ok])
        np.testing.assert_allclose(wo[ok], eta * wi[ok] + (eta * ci[ok] - ct)[:, None] * n, atol=3e-6)
        if eta == 1.0:
            np.testing.assert_allclose(wo, wi, atol=2e-6)
    tc = math.degrees(math.asin(1.0 / 1.5))
    for t in (0.0, tc - 1e-4, tc, tc + 1e-4):  # finite derivatives at normal incidence and at the critical angle
        g = jax.grad(
            lambda w, nn, e: jnp.sum(outcomes.refract(w, nn / jnp.linalg.norm(nn), e)[0] * jnp.array([1.0, 2.0, 3.0])),
            argnums=(0, 1, 2),
        )(jnp.asarray(incoming(t, n), F32), jnp.asarray(n, F32), F32(1.5))
        assert all(np.all(np.isfinite(np.asarray(x))) for x in g), t


def test_outcome_tables_are_consistent():
    assert len(outcomes.NAMES) == outcomes.N_OUTCOMES
    assert [outcomes.INDEX[k] for k in outcomes.NAMES] == list(range(outcomes.N_OUTCOMES))
    for i in range(outcomes.N_OUTCOMES):
        assert not (outcomes.DELTA[i] and outcomes.TERMINAL[i])
