"""The photon loop's kernel: tallymap's step schedule engages, and XLA adds no per-photon memory around it.

tallymap refills a lane per step only when the item is one top-level loop; otherwise it falls back to a lane per
photon, silently in this tallymap (ec953da). ``lumen.run.kernel_schedules`` reads the schedule each CUDA kernel was
built for (tallymap's kernel record): a fallback fails here. The memory check compiles the loop for 2^20 photons and
bounds what XLA allocates around the kernel (``lumen.run.loop_memory``): a batched computation of the photons
outside the kernel (a vmapped locate needed ~219 KB per photon) fails it.

The physics tier (one GPU). The toy is the unit tier's (tests/unit/test_api.py: a tank and two tubes).
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
TEMP_PER_PHOTON = 64  # bytes XLA may allocate per photon around the kernel (measured: 5 on SBND's charge path)


def rows(n):
    d = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    z = jnp.zeros(n, jnp.float32)
    x = jnp.tile(jnp.asarray([0.3, 0.0, 0.0], jnp.float32), (n, 1))
    return state.Emission(x, jnp.asarray(d), z, z + 400.0, z + 1.0, jnp.zeros((n, 2, 3), jnp.float32),
                          jnp.ones(n, bool))  # fmt: skip


@pytest.fixture(scope="module")
def loop(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("loopkernel")
    det = lumen.load(lumen.convert(write(tmp, _toy()), RECIPE, tmp / "toy.lumen"))
    model = steps.Model(tracked.from_parts(det.parts), det.parts.optics)
    tally = item.HistogramShape(len(det.parts.optics.sensor_pvs), (300.0, 600.0))
    ops = steps.operands(model, det.pack(), None, steps.Controls(), item.grid(tally))
    return item.photons(tally), model, ops, jax.random.key_data(jax.random.PRNGKey(0))


def test_the_loop_runs_on_the_step_schedule(loop):
    spec, model, ops, key = loop
    t = runner.run(spec, model, ops, rows(256), key, max_vertices=500, backend="auto")
    assert float(np.asarray(t.outputs["Q"]).sum()) > 0.0
    modes = runner.kernel_schedules()
    assert modes and modes[-1] == "step", modes


def test_xla_adds_no_per_photon_memory_around_the_kernel(loop):
    spec, model, ops, key = loop
    n = 1 << 20
    m = runner.loop_memory(spec, model, ops, rows(n), key, max_vertices=500, backend="auto")
    assert m["temp"] <= TEMP_PER_PHOTON * n, m
