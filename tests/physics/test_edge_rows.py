"""The derivative in a surface probability at the edge of its domain, against its closed form (the edge floor).

``toys.rt_boxes``: a pencil through a dielectric_dielectric border with R = 0.5 and T = 0, then a detecting border.
dQ/dT at T = 0+ is exp(-0.75 / L) per photon (float64, the closed form), reached only by proposing the STRAIGHT outcome
T opens (``lumen.transport.estimator``: the edge rows derived from the structure and the type's domain). Without the
edge rows
the outcome is never drawn and the AD is exactly 0 (the signature the gradient check's zero rule must not hide).

The physics tier (one GPU).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import toys
from toys import EXACT

import lumen
from lumen import run as runner
from lumen import sources
from lumen.compare import photons
from lumen.tools import stats
from lumen.transport import estimator
from lumen.transport import step as steps

N, K, L = 1 << 20, 16, 1e3


def _moved(p0, column, row, x):
    """Return the packed params with the surface table's ``column`` moved by x at ``row``."""
    t = p0["surfaces"][column]
    t = {**t, "value": t["value"].at[row].add(x)} if isinstance(t, dict) else t.at[row].add(x)
    return {**p0, "surfaces": {**p0["surfaces"], column: t}}


def _row(det, cond):
    return next(i for i, r in enumerate(det.parts.optics.rows) if cond(r.surface_type))


def _ad(s, column, row):
    p0 = photons.packed(s.det)

    def launch(x, records=None):
        p = _moved(p0, column, row, x)
        ref = jax.tree_util.tree_map(jax.lax.stop_gradient, p)
        ops = photons.operands(s, p, ref, steps.Controls(roulette=EXACT))
        rows = toys.pencil([-0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 400.0, N)
        return toys.stack_seeds(lambda r, j: runner.run(s.loop(), s.model, ops, r, toys.raw_key(900 + j),
                                                        records=records).outputs, rows, K)  # K seeds  # fmt: skip

    rec = 2 * int(launch(jnp.float32(0.0))["records"].max())  # under jax.jit: the records of the runs with values
    q = lambda x: launch(x, rec)["charge:0"]  # noqa: E731
    d = np.asarray(jax.jit(lambda x: jax.jvp(q, (x,), (1.0,))[1])(jnp.float32(0.0)), np.float64).reshape(K, -1).sum(-1)
    return d.sum() / N, np.std(d, ddof=1) * np.sqrt(K) / N


def test_the_edge_floor_gives_the_closed_form_derivative_at_t_zero(toy):
    s0 = toy("rt_boxes")
    row = _row(s0.det, lambda k: k.name.startswith("dielectric_rt"))
    edge = estimator.edge_rows(s0.det.parts.optics, [("surfaces", row, "T")])
    # T has a domain (the type's): its row is an edge row, whatever its value
    assert edge == estimator.EdgeRows(surfaces=(row,)), edge
    ad, err = _ad(toy("rt_boxes", step=steps.StepConfig(edge_rows=edge)), "T", row)
    exact = float(np.exp(-0.75 / L))
    assert abs(ad - exact) <= stats.z_limit() * err and err < 0.03, (ad, err, exact)
    ad_off, _ = _ad(s0, "T", row)
    assert ad_off == 0.0  # the outcome is never proposed without the edge rows


KEYS = {"p_lobe": "SPECULARLOBECONSTANT", "p_back": "BACKSCATTERCONSTANT"}
M = 1 << 20


def _plate(s, x_param=None, x=0.0, edge=False, seed=900, n=M):
    """Return (Q, Tw) per seed of the pencil onto the plate, or their jvp in ``x_param`` at x (edge rows on)."""
    p0 = photons.packed(s.det)
    row = _row(s.det, lambda k: "p_lobe" in k.params)
    rows = toys.pencil([-0.6, 0.0, 0.3], [0.6, 0.0, -0.8], 400.0, n)

    def launch(y, records=None):
        p = p0 if x_param is None else _moved(p0, x_param, row, y)
        ops = photons.operands(s, p, jax.tree_util.tree_map(jax.lax.stop_gradient, p), steps.Controls(roulette=EXACT))

        def one(r, j):
            return runner.run(s.loop(), s.model, ops, r, toys.raw_key(seed + 1000 * j), max_vertices=2000,
                              records=records).outputs  # fmt: skip

        return toys.stack_seeds(one, rows, K)

    rec = 2 * int(launch(jnp.float32(x))["records"].max())  # under jax.jit: the records of the runs with values

    def q(y):
        o = launch(y, rec)
        return o["charge:0"].reshape(K, -1).sum(-1), o["Tw"].reshape(K, -1).sum(-1)

    out = jax.jit(lambda y: jax.jvp(q, (y,), (1.0,))[1])(jnp.float32(x)) if edge else jax.jit(q)(jnp.float32(x))
    return [np.asarray(v, np.float64) for v in out]


@pytest.mark.parametrize("param", ["p_lobe", "p_back"])
def test_a_unified_constant_at_zero_gives_its_exact_derivative(toy, param):
    # one reflection: Q is linear in the constant, so dQ/dp = Q(p = 1) - Q(p = 0) exactly (each its own photons)
    s0, s1 = toy("lobe_plate", 0.2), toy("lobe_plate", 0.2, **{KEYS[param]: 1.0})
    # 2M photons: the edge rows' derivative holds its 5 % precision with margin whatever seed its photons fall in
    q0, q1 = _plate(s0, seed=901, n=2 * M), _plate(s1, seed=901, n=2 * M)
    row = _row(s0.det, lambda k: "p_lobe" in k.params)
    edge = estimator.edge_rows(s0.det.parts.optics, [("surfaces", row, param)])
    assert edge == estimator.EdgeRows(surfaces=(row,)), edge
    ad = _plate(toy("lobe_plate", 0.2, step=steps.StepConfig(edge_rows=edge)), param, edge=True, n=2 * M)
    for c in (0, 1):  # Q and the time-weighted Q
        exact, e_exact = q1[c].sum() - q0[c].sum(), np.hypot(*(np.std(v[c], ddof=1) for v in (q0, q1))) * np.sqrt(K)
        a, e_a = ad[c].sum(), np.std(ad[c], ddof=1) * np.sqrt(K)
        assert abs(a - exact) <= 4.0 * np.hypot(e_a, e_exact) and e_a < 0.05 * abs(exact), (param, c, a, e_a, exact)


def test_simulate_takes_every_row_of_a_key_with_a_domain_as_an_edge_row(toy):
    """Through lumen.simulate: REFLECTIVITY scaled to 0 from the detector's 0.5 (interior) is at its edge; dQ/ds at
    s = 0 is 0.5 exp(-0.75 / L) per photon, eagerly and under jit alike (every row of a key with a domain is an
    edge row, whatever its value: one estimator)."""
    det, key, n = toy("rt_boxes").det, "surfaces/RT/REFLECTIVITY", 1 << 20
    src = sources.FromArrays(np.tile([-0.5, 0.0, 0.0], (n, 1)), np.tile([1.0, 0.0, 0.0], (n, 1)), np.zeros(n),
                             np.full(n, 400.0), np.ones(n))  # fmt: skip
    p = det.params()

    rec = 2 * int(lumen.simulate(det, src, p.scale(key, 0.0), samples=n, seed=900).health.records)

    def q(s):
        return lumen.simulate(det, src, p.scale(key, s), samples=n, seed=900, records=rec).arrivals.sum() / n

    exact = 0.5 * float(np.exp(-0.75 / L))
    for f in (lambda s: jax.jvp(q, (s,), (1.0,)), jax.jit(lambda s: jax.jvp(q, (s,), (1.0,)))):
        value, d = f(jnp.float32(0.0))
        assert float(value) == 0.0 and abs(float(d) - exact) < 0.05 * exact, (float(d), exact)
