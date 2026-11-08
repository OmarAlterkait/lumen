"""Orthonormal frames about a unit vector.

``basis(n)`` is the branchless construction of Duff et al., "Building an Orthonormal Basis, Revisited" (JCGT 2017):
exact to float32 round-off for every unit ``n``, smooth in ``n`` except across the plane n_z = 0, where the tangent
pair flips (no continuous tangent field exists on the whole sphere). Distributions that are azimuthally symmetric
about ``n`` are unaffected by the flip; an anisotropic one must carry its own reference direction.
"""

import jax.numpy as jnp


def basis(n):
    """Return the tangents (t1, t2) that make (t1, t2, n) a right-handed orthonormal frame.

    Parameters
    ----------
    n : array (..., 3)
        Unit vectors.
    """
    sign = jnp.where(n[..., 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    t1 = jnp.stack([1.0 + sign * n[..., 0] ** 2 * a, sign * b, -sign * n[..., 0]], -1)
    t2 = jnp.stack([b, sign + n[..., 1] ** 2 * a, -n[..., 1]], -1)
    return t1, t2


def to_world(local, n):
    """Return the world vector of local coordinates (x, y, z) in the frame (t1, t2, n)."""
    t1, t2 = basis(n)
    return local[..., 0:1] * t1 + local[..., 1:2] * t2 + local[..., 2:3] * n


def to_local(v, n):
    """Return the local coordinates of a world vector in the frame (t1, t2, n)."""
    t1, t2 = basis(n)
    return jnp.stack([jnp.sum(v * t1, -1), jnp.sum(v * t2, -1), jnp.sum(v * n, -1)], -1)


def reflect(w, n):
    """Return w mirrored about the plane with normal n (either orientation of n)."""
    return w - 2.0 * jnp.sum(w * n, -1, keepdims=True) * n
