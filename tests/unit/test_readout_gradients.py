"""After-tracking derivatives: forward (jvp) and reverse (grad) mode against central finite differences, in float64,
for every sensor value of every component form; and no recompilation when any value changes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import readout_toys as toys
from jax.experimental import enable_x64

from lumen import readout as rd

VARIANTS = {
    "toy": {},
    "emg_table_poly_exp2_delay": dict(
        tts=dict(form="emg", sigma_ns=1.0, tau_ns=1.5, shift_ns=0.2),
        charge=dict(form="table", edges_pe=list(np.linspace(-0.5, 3.0, 15)), pdf=list(np.r_[0.2, 1, 3, 6, 9, 8, 6, 4,
                                                                                          2.5, 1.5, 1, 0.5, 0.3, 0.1])),
        threshold=dict(form="poly", coefficients=[-0.1, 1.2, 0.3], offset_pe=0.1, limit_pe=1.0),
        pulse=dict(form="exp2", t_start_ns=1.0, tau_rise_ns=1.0, tau_fall_ns=5.0),
        delay=dict(form="table", edges_ns=[0.0, 1.0, 2.0, 4.0], pdf=[0.5, 0.3, 0.2])),
    "table_tts_table_pulse": dict(
        tts=dict(form="table", edges_ns=list(np.linspace(-3.0, 3.0, 13)), pdf=list(np.exp(-np.linspace(-2.75, 2.75,
                                                                                                         12) ** 2))),
        pulse=dict(form="table", t_first_ns=0.0, step_ns=1.0, shape=list(np.sin(np.linspace(0, np.pi, 16)) ** 2)),
        t_offset_ns=5.13),  # off the node grid: a table's CDF is only continuous at its bin edges
}  # fmt: skip


def _setup(name):
    m = toys.model(**VARIANTS[name])
    h = toys.hist(m, n=300, n_sensors=3, t0=30.0, spread=2.0, replicas=2)
    p0 = toys.params(m)
    ev = rd.sample(h, p0, seed=2)
    data = rd.Hits.from_list(ev.hits["sensor"], ev.hits["t"], ev.hits["q"], 3)
    rng = np.random.default_rng(1)
    wts = rng.standard_normal((2, 3, 60))
    return m, h, p0, data, wts


def _loss(h, data, wts):
    def f(p):
        w = rd.expected_waveform(h, p)
        e = rd.expected_hits(h, p)
        return (jnp.sum(wts[0] * w.mean) * 1e-2 + jnp.sum(wts[1] * w.variance) * 1e-3 + rd.hits_nll(e, data)
                + jnp.sum(rd.pe_density(h, p).rate))  # fmt: skip

    return f


@pytest.mark.parametrize("name", list(VARIANTS))
def test_gradients_match_finite_differences(name):
    m, h, p0, data, wts = _setup(name)
    with enable_x64():
        h64 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), h)
        p64 = {k: jnp.asarray(v, jnp.float64) for k, v in p0.items()}
        f = jax.jit(_loss(h64, data, wts))
        g = jax.jit(jax.grad(f))(p64)
        fwd_of = jax.jit(lambda p, t: jax.jvp(f, (p,), (t,))[1])
        rng = np.random.default_rng(7)
        checked = []
        for k in sorted(p64):
            x = np.asarray(p64[k])
            d = rng.standard_normal(x.shape) * (np.abs(x) * 0.5 + 0.05)
            tangent = {kk: jnp.zeros_like(v) for kk, v in p64.items()} | {k: jnp.asarray(d)}
            fwd = float(fwd_of(p64, tangent))
            rev = float(np.sum(np.asarray(g[k]) * d))
            hstep = 1e-6

            def at(s, k=k, x=x, d=d):
                return float(f(p64 | {k: jnp.asarray(x + s * d)}))

            fd = (8 * (at(hstep) - at(-hstep)) - (at(2 * hstep) - at(-2 * hstep))) / (12 * hstep)
            tol = 1e-5 * max(1.0, abs(fd)) + 1e-6 * abs(float(f(p64)))
            assert abs(fwd - fd) < tol and abs(rev - fd) < tol, (k, fwd, rev, fd)
            checked.append(k)
    assert len(checked) == len(p64)


def test_no_recompile_on_any_value():
    """Changing every sensor value leaves the compiled readout as it is (no new compilation)."""
    m, h, p0, data, wts = _setup("toy")
    f = jax.jit(_loss(h, data, wts))
    f(p0)
    n0 = f._cache_size()
    rng = np.random.default_rng(3)
    for _ in range(3):
        p = {k: np.asarray(v * (1.0 + 0.05 * rng.standard_normal(v.shape)), np.float32) for k, v in p0.items()}
        f(p)
    assert f._cache_size() == n0 == 1
