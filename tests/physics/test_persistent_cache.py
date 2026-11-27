"""JAX's persistent compilation cache: a program loaded from it is bitwise the one a fresh compile gives.

A user turns the cache on through JAX's own variables (``JAX_COMPILATION_CACHE_DIR``; lumen sets nothing). That is
safe only while a loaded program computes exactly what a fresh compile computes: on jax 0.5.3 it did not (CPU
programs loaded from the cache missed navigator faces; docs/dev/decisions.md, 7). This test runs the photon loop on a
small toy in three fresh processes -- (a) the cache disabled, (b) a cold compile writing a fresh cache, (c) a new
process loading it -- and requires every output to be bitwise equal, so a future jax that breaks it fails here. The
physics tier (one GPU); this file is also the driver its processes run (``python <this file> DET OUT``).
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

N = 64


def _drive(det_path, out):
    """One process: the toy's photon loop on the GPU, every output saved."""
    import jax.numpy as jnp

    import lumen
    from lumen import run as runner
    from lumen.build import pack, tracked
    from lumen.compare import photons
    from lumen.readout import fill
    from lumen.readout.grid import Grid
    from lumen.record import Photons
    from lumen.transport import step as steps

    det = lumen.load(det_path)
    model = steps.Model(tracked.from_parts(det.parts), det.parts.optics)
    d = np.random.default_rng(0).normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    z = jnp.zeros(N, jnp.float32)
    x = jnp.tile(jnp.asarray([0.3, 0.0, 0.0], jnp.float32), (N, 1))
    rows = Photons(x, jnp.asarray(d, jnp.float32), z, z + 400.0, z + 1.0, jnp.zeros((N, 2, 3), jnp.float32))
    ops = steps.operands(model, pack.checked(det, det.params()), None, steps.Controls())
    loop = photons.loop(fill.HistogramLayout(len(det.parts.optics.sensor_pvs), Grid.bands((300.0, 600.0))))
    t = runner.run(loop, model, ops, rows, jax.random.key_data(jax.random.PRNGKey(0)),
                   max_vertices=500, backend="auto")  # fmt: skip
    np.savez(out, **{f"out:{k}": np.asarray(v) for k, v in t.outputs.items()},
             cache=np.array(str(jax.config.jax_compilation_cache_dir)))  # fmt: skip


def _process(det_path, out, env):
    e = {k: v for k, v in os.environ.items() if k not in ("JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE")}  # noqa: TID251 -- the child's environment, not a lumen setting
    e.update(env)
    r = subprocess.run([sys.executable, __file__, str(det_path), str(out)], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


def test_a_program_loaded_from_the_cache_is_bitwise_a_fresh_compile(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))  # the unit tier's toy
    from test_api import RECIPE, _toy
    from test_importer import write

    import lumen

    det_path = lumen.convert(write(tmp_path, _toy()), tmp_path / "toy.lumen", recipe=RECIPE)
    cache_root = tmp_path / "lumen_cache"  # fresh
    every = {
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        "JAX_COMPILATION_CACHE_DIR": str(cache_root / "xla"),
    }
    fresh = _process(det_path, tmp_path / "a.npz", {"JAX_ENABLE_COMPILATION_CACHE": "false",
                                                    "LUMEN_CACHE": str(tmp_path / "no_cache")})  # fmt: skip
    writes = _process(det_path, tmp_path / "b.npz", {"LUMEN_CACHE": str(cache_root), **every})
    entries = sorted(p.name for p in (cache_root / "xla").iterdir())
    loads = _process(det_path, tmp_path / "c.npz", {"LUMEN_CACHE": str(cache_root), **every})
    assert str(writes["cache"]) == str(cache_root / "xla") and entries, (writes["cache"], entries)
    assert sorted(p.name for p in (cache_root / "xla").iterdir()) == entries, "the loading process compiled anew"
    for k in fresh:
        if k.startswith("out:"):
            for name, other in (("writes", writes), ("loads", loads)):
                assert fresh[k].tobytes() == other[k].tobytes() and fresh[k].shape == other[k].shape, (k, name)


if __name__ == "__main__":
    _drive(sys.argv[1], sys.argv[2])
