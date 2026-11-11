"""Ray clearance (lumen.geometry.clearance): the rays run in batches sized by a byte budget, and the clearance does
not depend on the batch."""

import nav_toys
import numpy as np

from lumen.geometry import clearance


def test_a_tiny_budget_runs_one_ray_per_batch_with_the_same_clearance():
    _, nav = nav_toys.toy("nest")
    points = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, -0.2]], np.float32)
    tiny = clearance.ray_clearance(nav, points, n_rays=16, budget=1)  # one ray per batch
    default = clearance.ray_clearance(nav, points, n_rays=16)
    assert np.array_equal(tiny, default)
    assert np.all(np.isfinite(default)) and np.all(default > 0)


def test_ray_chunk_sizes_the_batch_by_its_working_memory():
    import jax
    import jax.numpy as jnp

    program = jax.jit(lambda x, d: jnp.min((d @ x) * jnp.cumsum(d, 0)[:, 0]))
    x, d = jnp.ones(3, jnp.float32), jnp.ones((64, 3), jnp.float32)
    assert clearance.ray_chunk(program, x, d, budget=1) == 1
    assert clearance.ray_chunk(program, x, d, budget=1 << 40) == 64
