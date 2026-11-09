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
RAY_CHUNK = 256  # rays per batch: the vmapped navigator's lanes keep their own stacks (all 4096 at once: ~50 GB on SK)


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
    D = fibonacci_sphere(n_rays)

    @jax.jit
    def nearest(x, d):
        v = nav.locate(x)
        q = jax.vmap(lambda d_: nav.query(x, d_, v, t_max=T_MAX))(d)
        return jnp.min(jnp.minimum(jnp.where(q["hit"], q["t"], jnp.inf), q["t_pass0"]))

    c = [min(float(nearest(jnp.asarray(x), jnp.asarray(D[i : i + RAY_CHUNK]))) for i in range(0, n_rays, RAY_CHUNK))
         for x in X]  # fmt: skip
    return safety * np.asarray(c, np.float32)


class Clearance:
    """The clearance field of points with known radii: max_i (clear_i - |x - points_i|), at least 0 (m)."""

    def __init__(self, points, clear):
        self.points = jnp.asarray(np.asarray(points, np.float32).reshape(-1, 3))
        self.clear = jnp.asarray(np.asarray(clear, np.float32).reshape(-1))

    def operands(self):
        """Return (points, radii), to enter a loop as operands (``gp`` of the call)."""
        return self.points, self.clear

    def __call__(self, gp, x):
        """Return the clearance at x; ``gp``: the operands (``operands``), else this field's own arrays."""
        points, clear = self.operands() if gp is None else gp
        return jnp.maximum(jnp.max(clear - jnp.linalg.norm(points - jnp.asarray(x, jnp.float32), axis=-1)), 0.0)
