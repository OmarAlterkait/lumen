"""The derivative in a surface probability at the edge of its domain, against its closed form (the edge floor).

``toys.rt_boxes``: a pencil through a dielectric_dielectric border with R = 0.5 and T = 0, then a detecting border.
dQ/dT at T = 0+ is exp(-0.75 / L) per photon (float64, the closed form), reached only by proposing the STRAIGHT outcome
T opens (``lumen.transport.estimator``: the edge rows derived from wrt and the type's domain). Without the edge rows
the outcome is never drawn and the AD is exactly 0 (the signature the gradient check's zero rule must not hide).

The physics tier (one GPU).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen import run as runner
from lumen.transport import estimator, item
from lumen.transport import step as steps
from lumen.validate import photons

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="the physics tier runs on a GPU")
N, K, L = 1 << 20, 16, 1e3


def _ad(s, path, row):
    p0 = photons.packed(s.det)

    def q(x):
        t = p0["kernel"][path[1]][path[2]]
        t = {**t, "value": t["value"].at[row].add(x)} if isinstance(t, dict) else t.at[row].add(x)
        p = {**p0, "kernel": {**p0["kernel"], path[1]: {**p0["kernel"][path[1]], path[2]: t}}}
        ref = jax.tree_util.tree_map(jax.lax.stop_gradient, p)
        ops = photons.operands(s, p, ref, steps.Controls(roulette=0.0))
        rows = photons.pencil([-0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 400.0, N)
        return runner.run(item.photons(s.tally), s.model, ops, rows, photons.raw_key(900), replicas=K).outputs["Q"]

    d = np.asarray(jax.jit(lambda x: jax.jvp(q, (x,), (1.0,))[1])(jnp.float32(0.0)), np.float64).reshape(K, -1).sum(-1)
    return d.sum() / N, np.std(d, ddof=1) * np.sqrt(K) / N


def test_the_edge_floor_gives_the_closed_form_derivative_at_t_zero(toy):
    s0 = toy("rt_boxes")
    p0 = photons.packed(s0.det)
    row = 0
    path = ("kernel", "dielectric_rt", "T", row)
    edge = estimator.edge_rows(s0.det.parts.optics, p0, [path])
    assert edge == (("dielectric_rt", row),), edge  # T = 0 is at its domain's edge (the type's domain)
    ad, err = _ad(toy("rt_boxes", step=steps.StepConfig(edge_rows=edge)), path, row)
    exact = float(np.exp(-0.75 / L))
    assert abs(ad - exact) <= 4.0 * err and err < 0.03, (ad, err, exact)
    ad_off, _ = _ad(s0, path, row)
    assert ad_off == 0.0  # the outcome is never proposed without the edge rows
