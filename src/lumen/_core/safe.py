"""Operations whose value and derivative are finite at their singular points.

``jnp.where(c, f(x), g)`` does not protect a derivative: the branch not taken is still differentiated, and a NaN
or inf there poisons the result (0 * nan = nan). Every function here first replaces the input where the function is
singular (the "double where").
"""

import jax
import jax.numpy as jnp


def sanitize(ok, x, fill):
    """Return x where ok, else a harmless fill value; apply it before a function that is singular where not ok."""
    return jnp.where(ok, x, fill)


def safe_sqrt(x):
    """Return sqrt(max(x, 0)), with derivative 0 (not inf) at x <= 0."""
    ok = x > 0.0
    return jnp.where(ok, jnp.sqrt(sanitize(ok, x, 1.0)), 0.0)


def safe_norm(v, axis=-1):
    """Return |v|, with derivative 0 (not NaN) at v = 0."""
    return safe_sqrt(jnp.sum(v * v, axis=axis))


def unit(v, axis=-1, fallback=None):
    """Return v / |v|; at v = 0 return ``fallback`` (default: v itself, i.e. 0), with finite derivatives."""
    s = jnp.sum(v * v, axis=axis, keepdims=True)
    ok = s > 0.0
    u = v / jnp.sqrt(sanitize(ok, s, 1.0))
    return jnp.where(ok, u, v if fallback is None else fallback)


def safe_log(x, floor=1e-30):
    """Return log(max(x, floor)): finite value and derivative for x <= 0 (derivative 0 below the floor)."""
    ok = x > floor
    return jnp.where(ok, jnp.log(sanitize(ok, x, 1.0)), jnp.log(floor))


@jax.custom_jvp
def clip01(x):
    """Return clip(x, 0, 1) with the derivative of the linear model on the closed interval [0, 1].

    The derivative is 1 on [0, 1] and 0 strictly outside. ``jnp.clip``'s derivative at a bound is 0.5 (JAX splits a
    min / max tie between its arguments), which would halve a derivative at a physical value on the bound, such as a
    reflectivity of 1. Where the clipped quantity is a probability whose parameters live on a closed simplex it is
    linear on that simplex, so this is the exact derivative along every direction that stays in it.
    """
    return jnp.clip(x, 0.0, 1.0)


@clip01.defjvp
def _clip01_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return clip01(x), jnp.where((x >= 0.0) & (x <= 1.0), t, jnp.zeros_like(t))


def safe_div(a, b, fill=0.0):
    """Return a / b, and ``fill`` where b == 0, with finite derivatives."""
    ok = b != 0.0
    return jnp.where(ok, a / sanitize(ok, b, 1.0), fill)


def literal(values):
    """Return a float32 vector built from scalar literals: a constant of the code that is never a closed-over array."""
    return jnp.stack([jnp.float32(v) for v in values])
