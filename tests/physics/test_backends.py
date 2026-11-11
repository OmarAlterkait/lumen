"""The CUDA backend against tallymap's CPU backend: the same toy, photons and keys, within 2.5e-4.

The CPU backend sums in float32 (``exact=False``), the CUDA backend in exact fixed point, and XLA's CPU and GPU
math differ in the last bits, so the two agree to a Monte Carlo tolerance, not bit for bit. The specular sensor
sphere is deterministic per photon (no drawn choice changes its charge), so its value and derivative must agree to
float32 accumulation (each replica's cells summed in float32 on the CPU, the replicas in float64 here);
``backend="auto"`` must pick CUDA on the GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_toys import charge, families, rows_at
from toys import gpu_only

from lumen import run as runner
from lumen.transport import item
from lumen.validate import photons

TOLERANCE = 2.5e-4  # relative, CPU against GPU
N, REPLICAS = 4096, 64  # replicas keep each float32 cell's sum short on the CPU (its sums are plain float32)


@gpu_only
def test_the_cpu_backend_agrees_with_the_gpu(toy):
    s = toy("sphere", r=0.3, absorption=2.0)
    th = jnp.zeros(2, jnp.float32)
    e = th.at[0].set(1.0)
    p0, fams = families(s, ".R[", "Liquid:absorption")
    gpu = [float(np.mean(x)) for x in jax.jvp(charge(s, p0, fams, rows_at(N), REPLICAS, backend="cuda"), (th,), (e,))]
    with jax.default_device(jax.devices("cpu")[0]):
        sc = photons.setup(s.det)  # its own navigator arrays, on the CPU
        p0c, famc = families(sc, ".R[", "Liquid:absorption")
        th_c, e_c = jnp.zeros(2, jnp.float32), jnp.zeros(2, jnp.float32).at[0].set(1.0)
        q = charge(sc, p0c, famc, rows_at(N), REPLICAS, backend="cpu")
        cpu = [float(np.mean(x)) for x in jax.jvp(q, (th_c,), (e_c,))]
    np.testing.assert_allclose(cpu, gpu, rtol=TOLERANCE)


@gpu_only
def test_auto_picks_cuda_on_a_gpu(monkeypatch):
    seen = {}

    def vmap(fn, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(runner.tm, "vmap", vmap)
    spec = item.photons(item.HistogramShape(1, (300.0, 500.0)))
    with pytest.raises(RuntimeError, match="stop"):
        runner.run(spec, None, None, rows_at(2), photons.raw_key(0), backend="auto")
    assert seen["backend"] == runner.TALLYMAP_BACKEND["cuda"]
