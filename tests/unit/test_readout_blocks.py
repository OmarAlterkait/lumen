"""The readout stages after the charge run per block of sensors (and of hits): the same values and derivatives as the
whole group in one piece, when the device's free memory holds only a few sensors at once. In float64: a transform's
bits depend on how many run together, so float32 agrees only to its rounding (1e-7 of a row's largest cell)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import readout_toys as toys
from jax.experimental import enable_x64

from lumen import readout as rd
from lumen.readout import detection

S = 7  # sensors: blocks of 2 leave a short last block (it ends at the last sensor, overlapping its neighbour)


def _setup():
    m = toys.model(delay=dict(form="table", edges_ns=[0.0, 1.0, 2.0, 4.0], pdf=[0.5, 0.3, 0.2]))  # the delay too
    h = toys.hist(m, n=900, n_sensors=S, t0=30.0, spread=2.0)
    p = {k: np.asarray(v, np.float64) for k, v in toys.params(m, n_sensors=S).items()}
    p["sensors/pmt/t_offset"] = np.linspace(4.0, 6.0, S)  # a per-sensor value that differs
    ev = rd.sample(h, p, seed=3)
    data = rd.Hits.from_list(ev.hits["sensor"], ev.hits["t"], ev.hits["q"], S)
    h64 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), h)
    return h64, {k: jnp.asarray(v, jnp.float64) for k, v in p.items()}, data


def _outputs(h, p, data):
    e = rd.expected_hits(h, p)
    w = rd.expected_waveform(h, p)
    return dict(light=rd.pe_density(h, p, dark=False).rate, density=rd.pe_density(h, p).rate, p_hit=e.p_hit,
                nodes=e.nodes, mean=w.mean, variance=w.variance, nll=rd.hits_nll(e, data))  # fmt: skip


def _small(monkeypatch, rows):
    """Leave free memory for ``rows`` rows of the largest stage at once (after its output)."""
    calls = []
    real = detection.rows_map

    def rows_map(f, n, row_bytes, out_row_bytes=0.0):
        calls.append(n)
        free = n * out_row_bytes + detection.BLOCK_SHARE * rows * row_bytes
        monkeypatch.setattr(detection, "free_memory", lambda: free)
        return real(f, n, row_bytes, out_row_bytes)

    for mod in ("detection", "waveform", "hits"):
        monkeypatch.setattr(f"lumen.readout.{mod}.rows_map", rows_map)
    return calls


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_blocks_give_the_whole_groups_values(monkeypatch, rows):
    with enable_x64():
        h, p, data = _setup()
        whole = _outputs(h, p, data)
        calls = _small(monkeypatch, rows)
        blocked = _outputs(h, p, data)
    assert calls and S in calls and len(data.sensor) in calls  # the sensors and the hits ran in blocks
    for k, v in whole.items():
        top = float(np.max(np.abs(np.asarray(v))))
        np.testing.assert_allclose(np.asarray(blocked[k]), np.asarray(v), rtol=1e-10, atol=1e-12 * top, err_msg=k)


def test_blocks_give_the_whole_groups_derivatives(monkeypatch):
    with enable_x64():
        _derivatives(monkeypatch)


def _derivatives(monkeypatch):
    h, p, data = _setup()
    keys = ("sensors/pmt/t_offset", "sensors/pmt/efficiency", "sensors/pmt/tts/sigma")

    def loss(x):
        q = dict(p, **x)
        e = rd.expected_hits(h, q)
        return (
            rd.hits_nll(e, data)
            + jnp.sum(rd.pe_density(h, q, dark=False).rate ** 2)
            + jnp.sum(rd.expected_waveform(h, q).mean)
        )

    x = {k: p[k] for k in keys}
    whole = jax.grad(loss)(x)
    _small(monkeypatch, 2)
    blocked = jax.grad(loss)(x)
    jitted = jax.jit(jax.grad(loss))(x)
    for k in keys:
        tol = dict(rtol=1e-9, atol=1e-9 * float(np.max(np.abs(np.asarray(whole[k])))), err_msg=k)
        np.testing.assert_allclose(np.asarray(blocked[k]), np.asarray(whole[k]), **tol)
        np.testing.assert_allclose(np.asarray(jitted[k]), np.asarray(whole[k]), **tol)


def test_one_sensor_that_does_not_fit_is_refused(monkeypatch):
    with enable_x64():
        h, p, _ = _setup()
        monkeypatch.setattr(detection, "free_memory", lambda: 1.0)
        with pytest.raises(MemoryError, match="the readout of one sensor"):
            rd.pe_density(h, p, dark=False)
