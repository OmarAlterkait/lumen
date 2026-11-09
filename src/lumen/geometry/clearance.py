"""Ray clearance: a lower-bound estimate of a point's distance to the nearest surface the light sees.

The moving-source method keeps a co-moving radius about a source's pinned vertices: within it no surface moves the
path. ``ray_clearance`` estimates each point's distance to the nearest crossing (visible, or the first invisible one)
over Fibonacci-sphere directions; ``Clearance`` turns those radii into a field over space by the triangle inequality.

    clear = ray_clearance(nav, points)                 # (n,) m
    field = Clearance(points, clear)                   # field(x): max_i (clear_i - |x - points_i|), at least 0

Notes
-----
Ported from the baseline's lumen.geometry.tracked (ray_clearance, set_clearance, clearance).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

N_RAYS = 4096  # directions per point (~3.5 deg apart): a feature thinner than that at that distance can be missed
SAFETY = 0.9  # the estimate is scaled by this
T_MAX = 1e3  # m: the farthest crossing looked for


def fibonacci_sphere(n):
    """Return n directions (n, 3) float32 on a Fibonacci sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    ph = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1).astype(np.float32)


def ray_clearance(nav, points, n_rays=N_RAYS, safety=SAFETY):
    """Return each point's estimated distance (m) to the nearest surface the light sees, times ``safety``."""
    X = np.asarray(points, np.float32).reshape(-1, 3)
    D = jnp.asarray(fibonacci_sphere(n_rays))

    def one(x):
        v = nav.locate(x)
        q = jax.vmap(lambda d: nav.query(x, d, v, t_max=T_MAX))(D)
        t = jnp.where(q["hit"], q["t"], jnp.inf)
        return jnp.min(jnp.minimum(t, q["t_pass0"]))

    c = np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(X)))
    return safety * c


class Clearance:
    """The clearance field of points with known radii: max_i (clear_i - |x - points_i|), at least 0 (m)."""

    def __init__(self, points, clear):
        self.points = jnp.asarray(np.asarray(points, np.float32).reshape(-1, 3))
        self.clear = jnp.asarray(np.asarray(clear, np.float32).reshape(-1))

    def __call__(self, gp, x):
        """Return the clearance at x (``gp`` is not read)."""
        return jnp.maximum(
            jnp.max(self.clear - jnp.linalg.norm(self.points - jnp.asarray(x, jnp.float32), axis=-1)), 0.0
        )
