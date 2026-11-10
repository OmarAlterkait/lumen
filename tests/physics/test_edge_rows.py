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


KEYS = {"p_lobe": "SPECULARLOBECONSTANT", "p_back": "BACKSCATTERCONSTANT"}
M = 1 << 20


def _plate(s, x_param=None, x=0.0, edge=False, seed=900):
    """Return (Q, Tw) per replica of the pencil onto the plate, or their jvp in ``x_param`` at x (edge rows on)."""
    p0 = photons.packed(s.det)
    kind = next(k.name for k in s.det.parts.optics.kinds if "p_lobe" in k.params)
    rows = photons.pencil([-0.6, 0.0, 0.3], [0.6, 0.0, -0.8], 400.0, M)

    def q(y):
        v = p0["kernel"][kind][x_param] if x_param else None
        p = p0 if v is None else {**p0, "kernel": {**p0["kernel"], kind: {**p0["kernel"][kind], x_param: (
            {**v, "value": v["value"].at[0].add(y)} if isinstance(v, dict) else v.at[0].add(y))}}}  # fmt: skip
        ops = photons.operands(s, p, jax.tree_util.tree_map(jax.lax.stop_gradient, p), steps.Controls(roulette=0.0))
        t = runner.run(item.photons(s.tally), s.model, ops, rows, photons.raw_key(seed), replicas=K, max_vertices=2000)
        return t.outputs["Q"].reshape(K, -1).sum(-1), t.outputs["Tw"].reshape(K, -1).sum(-1)

    out = jax.jit(lambda y: jax.jvp(q, (y,), (1.0,))[1])(jnp.float32(x)) if edge else jax.jit(q)(jnp.float32(x))
    return [np.asarray(v, np.float64) for v in out]


@pytest.mark.parametrize("param", ["p_lobe", "p_back"])
def test_a_unified_constant_at_zero_gives_its_exact_derivative(toy, param):
    # one reflection: Q is linear in the constant, so dQ/dp = Q(p = 1) - Q(p = 0) exactly (each its own photons)
    s0, s1 = toy("lobe_plate", 0.2), toy("lobe_plate", 0.2, **{KEYS[param]: 1.0})
    q0, q1 = _plate(s0, seed=901), _plate(s1, seed=901)
    kind = next(k.name for k in s0.det.parts.optics.kinds if "p_lobe" in k.params)
    edge = estimator.edge_rows(s0.det.parts.optics, photons.packed(s0.det), [("kernel", kind, param, 0)])
    assert edge == ((kind, 0),), edge
    ad = _plate(toy("lobe_plate", 0.2, step=steps.StepConfig(edge_rows=edge)), param, edge=True)
    for c in (0, 1):  # Q and the time-weighted Q
        exact, e_exact = q1[c].sum() - q0[c].sum(), np.hypot(*(np.std(v[c], ddof=1) for v in (q0, q1))) * np.sqrt(K)
        a, e_a = ad[c].sum(), np.std(ad[c], ddof=1) * np.sqrt(K)
        assert abs(a - exact) <= 4.0 * np.hypot(e_a, e_exact) and e_a < 0.05 * abs(exact), (param, c, a, e_a, exact)
