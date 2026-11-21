"""The second derivative through a covered draw of live probability 0 (lumen.transport.estimator), on the YAML toy.

Liner REFLECTIVITY 0.9, cathode REFLECTIVITY R (efficiency 1). At R = 0 the cathodes are defensive rows at their edge:
every cathode hit is a zero-probability choice, and a path through two cathode reflections goes as R^2, so it carries
d2Q/dR2 (an end of the weight-0 photon at its second such draw dropped 44 % of it: measured). The forward-over-forward
AD of Q = the arrivals' sum, draws at the run's values, against the finite difference of the fixed-reference gradient
g(R) (draws at R0), per seed: one-sided second-order Richardson at R0 = 0, central at R0 = 0.01, step eps32^(1/3)
(a second-order difference of a float32 function of O(1) curvature).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lumen
from lumen.compare import gradients as gr
from lumen.format.calibrations import Table
from lumen.tools.stats import z_limit

TOY = Path(__file__).resolve().parents[2] / "examples" / "data" / "yaml_toy"
KEY = "surfaces/Cathode/REFLECTIVITY"
K, N, H = 8, 1 << 15, 5e-3
ONE = jnp.float32(1.0)


@pytest.fixture(scope="module")
def det(tmp_path_factory):
    return lumen.load(lumen.convert(TOY, None, tmp_path_factory.mktemp("ghost") / "toy.lumen"))


@pytest.mark.parametrize("r0", [0.0, 0.01], ids=["at the edge", "off the edge"])
def test_the_second_derivative_is_the_finite_difference_of_the_gradient(det, r0):
    p = det.params()
    for k, val in (("surfaces/Liner/REFLECTIVITY", 0.9), (KEY, r0)):
        v = p[k]
        p = p.set(k, Table(jnp.asarray(v.energy), jnp.full_like(jnp.asarray(v.value), val)))
    fam = (KEY, "add", np.ones(np.shape(p[KEY].value)), True)
    src = lumen.sources.Point((0.05, 0.02, 0.1), n_photons=float(N), lam=420.0)

    def q(th, seed, ref=None):
        return jnp.sum(lumen.simulate(det, src, gr.at(p, fam, th), samples=N, seed=seed, reference=ref,
                                      speed="exact").arrivals)  # fmt: skip

    def g(th, seed, ref=None):
        return float(jax.jvp(lambda t: q(t, seed, ref), (jnp.float32(th),), (ONE,))[1])

    diff = []
    for s in range(K):
        ad = float(jax.jvp(lambda t: jax.jvp(lambda u: q(u, s), (t,), (ONE,))[1], (jnp.float32(0.0),), (ONE,))[1])
        if r0 == 0.0:
            fd = (-3 * g(0.0, s, p) + 4 * g(H, s, p) - g(2 * H, s, p)) / (2 * H)
        else:
            fd = (g(H, s, p) - g(-H, s, p)) / (2 * H)
        diff.append(ad - fd)
    d = np.asarray(diff)
    assert abs(d.mean()) < z_limit(df=K - 1) * max(d.std(ddof=1) / np.sqrt(K), 1e-3), d.tolist()
