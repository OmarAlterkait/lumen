"""The unified facet lobe's tables over their span, sigma_alpha in [SIGMA_MIN, SIGMA_MAX], and the reason for its top.

What is checked:

- the float32 bicubic Hermite against the float64 quadrature at off-node points of every cell (the errors
  ``lumen.rules.choices`` states for ``UNIFIED_TABLE``);
- the share of Geant4's facing facets drawn from a negative tilt (GetFacetNormal keeps a draw in (-2 pi k,
  -2 pi k + pi), whose sine is positive), which lumen's lobe does not draw: the values ``SIGMA_MAX`` states.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from lumen.rules.choices import SIGMA_MAX, UNIFIED_TABLE
from lumen.surfaces import facets

DF_MAX, DLZ_MAX = 5e-7, 1.4e-6  # lumen.rules.choices, UNIFIED_TABLE's measured errors (4.7e-7, 1.3e-6)


def test_the_tables_match_their_quadrature_over_their_span():
    """Each cell's midpoint and a random point in it, 128 regular and 128 random x: the stated errors hold."""
    rng = np.random.default_rng(1)
    y0, dy = UNIFIED_TABLE["y0"], UNIFIED_TABLE["dy"]
    cells = np.arange(UNIFIED_TABLE["j_lo"], UNIFIED_TABLE["j_hi"])
    ys = (y0 + dy * np.concatenate([cells + 0.5, cells + rng.uniform(0.05, 0.95, cells.size)])).astype(np.float32)
    xs = np.concatenate([(np.arange(128) + 0.5) / 128, rng.uniform(0.0, 1.0, 128)]).astype(np.float32)
    front = jax.jit(jax.vmap(jax.vmap(facets.table_front, (0, None)), (None, 0)))
    F = np.asarray(front(jnp.asarray(xs), jnp.asarray(ys)), np.float64)
    lz = np.asarray(jax.jit(jax.vmap(facets.table_lz))(jnp.asarray(ys)), np.float64)
    x64, y64 = xs.astype(np.float64), ys.astype(np.float64)  # the float32 coordinates, exactly
    df = max(float(np.abs(F[k] - facets.front_fraction_x(x64, math.exp(y))).max()) for k, y in enumerate(y64))
    dlz = max(abs(lz[k] - math.log(facets.tilt_norm(math.exp(y)) / math.exp(2.0 * y))) for k, y in enumerate(y64))
    assert df <= DF_MAX and dlz <= DLZ_MAX, (df, dlz)


def _wrapped_share(sigma, theta):
    """Return the share of Geant4's facets facing a photon at incidence theta that come from a negative tilt draw.

    Geant4's tilt density on the facet's polar angle b in (0, pi): g(b) for b < pi / 2 (the draw alpha = b) and
    sum_k g(2 pi k - b) (alpha = b - 2 pi k), times min(sin b, f_max); the facing share of the azimuths is
    acos(-cot theta cot b) / pi.
    """
    f_max = min(1.0, 4.0 * sigma)

    def g(a):
        return math.exp(-a * a / (2.0 * sigma * sigma))

    def facing(b):
        r = math.cos(theta) * math.cos(b) / (math.sin(theta) * math.sin(b))
        return math.acos(max(-1.0, min(1.0, -r))) / math.pi

    def intended(b):
        return g(b) * min(math.sin(b), f_max) * facing(b)

    def wrapped(b):
        return sum(g(2.0 * math.pi * k - b) for k in range(1, 8)) * min(math.sin(b), f_max) * facing(b)

    a0 = integrate.quad(intended, 0.0, 0.5 * math.pi, limit=400)[0]
    a1 = integrate.quad(wrapped, 0.0, math.pi, limit=400, points=[0.5 * math.pi])[0]
    return a1 / (a0 + a1)


@pytest.mark.parametrize(("sigma", "theta_deg", "share"), [(0.707, 89.0, 2.5e-6), (SIGMA_MAX, 89.0, 8.5e-4),
                                                            (SIGMA_MAX, 50.0, 6.8e-5), (SIGMA_MAX, 1e-3, 4.9e-6),
                                                            (1.2, 89.0, 6.1e-3), (2.0, 89.0, 0.15)])  # fmt: skip
def test_geant4s_negative_tilt_draws_are_the_share_sigma_max_states(sigma, theta_deg, share):
    assert abs(_wrapped_share(sigma, math.radians(theta_deg)) / share - 1.0) < 0.05
