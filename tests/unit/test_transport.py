"""The transport's host-free pieces: the estimator (proposal, choice, edge rows and floor), the photon's parts, the
default item's cells and the trace's integer halves. The step itself is gated against the baseline on six detectors
(tools/port/check_a6.py) and by the physics tier (tests/physics)."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen import diagnostics
from lumen._core.safe import safe_div
from lumen.media import bulk, laws, phase
from lumen.surfaces import types as legacy
from lumen.surfaces.outcomes import ABSORB, DETECT, SPECULAR, TERMINAL
from lumen.transport import estimator, item, state

F32 = jnp.float32


def baseline_defensive(p_ref, mix):
    """The baseline's lumen.estimator.defensive, verbatim: the port is its mix 0."""
    p_ref = jax.lax.stop_gradient(p_ref)
    sup = p_ref > 0.0
    tot = jnp.sum(p_ref)
    n = jnp.maximum(jnp.sum(sup), 1)
    q = (1.0 - mix) * safe_div(p_ref, tot) + mix * sup / n
    return jnp.where(tot > 0.0, q, 0.0)


def test_the_proposal_is_the_baselines_defensive_mix_at_zero_bit_for_bit():
    r = np.random.default_rng(0)
    p = r.uniform(size=(500, 10)).astype(np.float32) * (r.uniform(size=(500, 10)) > 0.4)
    p[0] = 0.0
    a = jax.jit(jax.vmap(estimator.proposal))(jnp.asarray(p))
    b = jax.jit(jax.vmap(lambda x: baseline_defensive(x, 0.0)))(jnp.asarray(p))
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    u = jnp.asarray(r.uniform(size=500), F32)
    k = jax.vmap(estimator.choose)(a, u)
    assert int(k[0]) == -1 and bool(jnp.all(jnp.asarray(p)[jnp.arange(1, 500), k[1:]] > 0))


def _tables():
    """Two rows of one painted type (R 0 and 0.5) and a water with Mie: the estimator reads the types and media."""
    water = bulk.Medium("w", laws.constant("index"), None, (bulk.mie(laws.constant("length")),))
    kp = {"painted": {"R": np.array([0.0, 0.5], np.float32), "f_spec": np.array([1.0, 1.0], np.float32)}}
    return types.SimpleNamespace(surface_types=(legacy.PAINTED,), media=(water,)), kp


def test_edge_rows_come_from_wrt_and_the_domains():
    tables, kp = _tables()
    lobe = {"g_forward": 0.6, "g_backward": 0.3, "forward_ratio": 1.0}
    params = {"surface_types": kp, "media": {"w": {"mie": {"lobe": lobe}}}}
    assert estimator.edge_rows(tables, params, [("surface_types", "painted", "R")]) == (("painted", 0),)
    assert estimator.edge_rows(tables, params, [("surface_types", "painted", "R", 1)]) == ()
    assert estimator.edge_rows(tables, params, [("surface_types", "painted", "f_spec", 0)]) == (("painted", 0),)
    mie = ("media", "w", "mie", "lobe", "forward_ratio")
    assert estimator.edge_rows(tables, params, [mie]) == (("w", "mie", "forward_ratio"),)
    lobe["forward_ratio"] = 0.7
    assert estimator.edge_rows(tables, params, [mie]) == ()
    assert estimator.edge_rows(tables, params, []) == ()


def test_the_edge_floor_proposes_the_zero_outcomes_at_their_share():
    tables, _ = _tables()
    pr = jnp.zeros(10, F32).at[ABSORB].set(1.0)  # R = 0: every continuing outcome has probability 0
    q, fl = estimator.edge_floor(pr, jnp.int32(0), jnp.int32(0), tables, (("painted", 0),))
    support = [o for o in legacy.PAINTED.outcomes if not TERMINAL[o]]
    assert sorted(np.flatnonzero(np.asarray(fl)).tolist()) == sorted(support)
    assert float(jnp.sum(jnp.where(fl, q, 0.0))) == pytest.approx(1.0)  # S = 0: n eps = 1
    pr = jnp.zeros(10, F32).at[ABSORB].set(0.5).at[SPECULAR].set(0.5)
    q, fl = estimator.edge_floor(pr, jnp.int32(0), jnp.int32(0), tables, (("painted", 0),))
    s, e = 0.5, estimator.EDGE_BUDGET  # a fixed share of the draws, whatever S
    assert float(jnp.sum(jnp.where(fl, q, 0.0))) == pytest.approx(s * e / (1 - e), rel=1e-6)
    q2, fl2 = estimator.edge_floor(pr, jnp.int32(0), jnp.int32(1), tables, (("painted", 0),))  # not an edge row
    assert not bool(jnp.any(fl2)) and np.array_equal(np.asarray(q2), np.asarray(pr))
    lobe = {"forward_ratio": jnp.asarray(1.0, F32), "g_forward": 0.6}
    rows = (("w", "mie", "forward_ratio"),)
    assert float(estimator.lobe_reference(lobe, phase.MIE_HG, ("w", "mie"), rows)["forward_ratio"]) == pytest.approx(
        0.99
    )
    assert estimator.lobe_reference(lobe, phase.MIE_HG, ("w", "mie"), ()) is lobe


def _row(n=3):
    z = jnp.zeros(n, F32)
    return state.Emission(jnp.zeros((n, 3), F32), jnp.tile(jnp.asarray([0.0, 0.0, 1.0]), (n, 1)), z, z + 400.0,
                          z + 1.0, jnp.zeros((n, 2, 3), F32), jnp.ones(n, bool))  # fmt: skip


def test_the_photon_has_only_the_parts_asked_for():
    row = jax.tree_util.tree_map(lambda a: a[0], _row())
    ph = state.start(row, 5, 2)
    assert (ph.stack, ph.moving, ph.z) == (None, None, None)
    assert int(ph.vol) == 5 and int(ph.cls) == 2 and int(ph.sensor) == -1 and bool(ph.alive)
    full = state.start(row, -1, 0, stacked=True, moving=True, edge=True)
    assert int(full.stack.layer) == 0 and float(full.z) == 1.0 and bool(full.moving.drawn)
    assert np.asarray(full.moving.chain.B).shape == (3, 2)


def test_the_default_item_tallies_by_sensor_and_band():
    spec = item.photons(item.HistogramShape(3, (300.0, 400.0, 500.0)))
    out = spec.outputs(2)
    assert out["Q"] == (12, True) and out["stuck"] == (2, False)

    def dep(sensor, lam):
        z = F32(0.0)
        fields = dict(sensor=jnp.int32(sensor), lam=F32(lam), q=F32(0.5), time=F32(4.0), lost=z, abandoned=z,
                      escaped=z, retired=z, stuck_pushed=jnp.bool_(True), stuck_abandoned=jnp.bool_(False))  # fmt: skip
        return type("D", (), fields)

    adds = [(k, int(c), float(v)) for k, c, v in spec.emits(dep(2, 450.0), jnp.int32(1), None)]
    assert adds[:2] == [("Q", 2 * 2 + 1 + 6, 0.5), ("Tw", 11, 2.0)]
    assert [a[1:] for a in adds if a[0] == "stuck"] == [(0, 1.0), (-1, 1.0)]  # pushed, not abandoned
    assert {
        k: c
        for k, c, _ in spec.emits(dep(2, 650.0), jnp.int32(0), item.grid(item.HistogramShape(3, (300.0, 400.0, 500.0))))
    }["Q"] == -1  # outside the bands
    assert {k: c for k, c, _ in spec.emits(dep(-1, 450.0), jnp.int32(0), None)}["Q"] == -1  # no sensor


def test_the_traces_integers_are_exact_in_two_halves():
    s = np.array([-1, 0, 65535, 65536, 123456789, 2**31 - 2], np.int32)
    hi, lo = diagnostics.halves(jnp.asarray(s))
    assert np.array_equal(diagnostics.from_halves(np.asarray(hi), np.asarray(lo)), s)
    assert int(diagnostics.last_event(jnp.int32(3), jnp.int32(-2))) == 1
    assert int(diagnostics.last_event(jnp.int32(3), jnp.int32(0))) == 3  # STRAIGHT keeps it
    assert int(diagnostics.last_event(jnp.int32(3), jnp.int32(DETECT))) == 2 + DETECT
    assert phase.MIE_HG.n_rand == 3  # the bulk layout's lobe block holds Mie's three uniforms


def test_the_runner_passes_tallymap_its_schedule(monkeypatch):
    """lumen asks for step scheduling explicitly, so a change of tallymap's default cannot slow it silently."""
    from lumen import run

    seen = {}

    def vmap(fn, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(run.tm, "vmap", vmap)
    spec = item.photons(item.HistogramShape(1, (300.0, 500.0)))
    with pytest.raises(RuntimeError, match="stop"):
        run.run(spec, None, None, _row(2), jnp.zeros(2, jnp.uint32), backend="cpu")
    assert seen["_schedule"] == run.SCHEDULE == "step" and seen["backend"] == run.TALLYMAP_BACKEND["cpu"]


def test_auto_is_the_cpu_backend_without_a_gpu(monkeypatch):
    """backend="auto": tallymap's CPU backend (inexact float sums) when JAX has no GPU, CUDA when it has one."""
    from lumen import run

    seen = {}

    def vmap(fn, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(run.tm, "vmap", vmap)
    monkeypatch.setattr(run.jax, "default_backend", lambda: "cpu")
    spec = item.photons(item.HistogramShape(1, (300.0, 500.0)))
    with pytest.raises(RuntimeError, match="stop"):
        run.run(spec, None, None, _row(2), jnp.zeros(2, jnp.uint32), backend="auto")
    assert seen["backend"] == run.TALLYMAP_BACKEND["cpu"]
