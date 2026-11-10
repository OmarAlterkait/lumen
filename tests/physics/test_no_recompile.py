"""No recompile per value (SPEC section 8): every kind of value enters the photon loop as an operand.

The loop runs once to compile, then again with each kind of value changed: materials, surfaces, sensors, a
calibration, source fields, controls, the grid's positions, the seed and the number of samples. No new XLA
compilation and no new trace may happen. A loop that closes over a float array is refused.

The physics tier (one GPU, ``backend="auto"``): on a CPU the loop's first compile alone takes minutes. The toy is the
unit tier's (tests/unit/test_api.py: a tank and two tubes).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))  # the unit tier's toy
from test_api import RECIPE, _toy
from test_importer import write

import lumen
from lumen import run as runner
from lumen.geometry import tracked
from lumen.transport import item, state
from lumen.transport import step as steps

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="the physics tier runs on a GPU")
N = 64
EVENTS = ("/jax/core/compile/backend_compile_duration", "/jax/core/compile/jaxpr_trace_duration")


def rows(shift=0.0, t=0.0, lam=400.0):
    d = np.random.default_rng(0).normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    z = jnp.zeros(N, jnp.float32)
    x = jnp.tile(jnp.asarray([0.3 + shift, 0.0, 0.0], jnp.float32), (N, 1))
    return state.Emission(x, jnp.asarray(d, jnp.float32), z + t, z + lam, z + 1.0, jnp.zeros((N, 2, 3), jnp.float32),
                          jnp.ones(N, bool))  # fmt: skip


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("norecompile")
    det = lumen.load(lumen.convert(write(tmp, _toy()), RECIPE, tmp / "toy.lumen"))
    model = steps.Model(tracked.from_parts(det.parts), det.parts.optics)
    tally = item.HistogramShape(len(det.parts.optics.tag_pvs), (300.0, 600.0))
    return det, model, item.photons(tally), tally


def test_no_value_recompiles(setup):
    det, model, spec, tally = setup
    p = det.params()

    def go(params=p, controls=None, edges=(300.0, 600.0), r=None, seed=0, n_active=N):
        ops = steps.operands(
            model, det.pack(params), None, controls or steps.Controls(), {"lam_edges": jnp.asarray(edges, jnp.float32)}
        )
        key = jax.random.key_data(jax.random.PRNGKey(seed))
        t = runner.run(spec, model, ops, rows() if r is None else r, key, max_vertices=500, backend="auto",
                       n_active=n_active)  # fmt: skip
        return float(np.asarray(t.outputs["Q"]).sum())

    q0 = go()
    seen = []
    jax.monitoring.register_event_duration_secs_listener(lambda name, *a, **k: seen.append(name))
    absorb = p.scale("materials/Water/ABSLENGTH", 0.5)
    calibration = p.with_layer({"materials/Water/ABSLENGTH": absorb["materials/Water/ABSLENGTH"],
                                "sensors/pmt/eff": 0.5}, "calibration test")  # fmt: skip
    changed = dict(
        materials=go(absorb),
        surfaces=go(p.scale("surfaces/Liner/REFLECTIVITY", 0.5)),
        sensors=go(p.set("sensors/pmt/eff", 0.25)),
        calibration=go(calibration),
        source=go(r=rows(shift=0.1, t=2.0, lam=450.0)),
        controls=go(controls=steps.Controls(0.5, 3.0, 1e-2)),
        grid=go(edges=(420.0, 600.0)),
        seed=go(seed=7),
        samples=go(n_active=N // 2),
    )
    compiles = [e for e in seen if e in EVENTS]
    assert not compiles, compiles
    assert changed["sensors"] < q0 and changed["grid"] == 0.0 and changed["samples"] < q0, (q0, changed)


def test_a_loop_closing_over_a_float_array_is_refused(setup):
    det, model, spec, tally = setup
    ops = steps.operands(model, det.pack(), grid=item.grid(tally))
    baked = jnp.linspace(0.5, 1.0, 4, dtype=jnp.float32)  # a value the item closes over instead of reading an operand
    start = spec.init
    spec = spec._replace(init=lambda m, o, r: start(m, o, r)._replace(w=start(m, o, r).w * baked[1]))
    with pytest.raises(runner.ClosedOverValues, match="closes over"):
        runner.run(spec, model, ops, rows(), jax.random.key_data(jax.random.PRNGKey(0)), max_vertices=50,
                   backend="auto")  # fmt: skip
