"""Operations whose value and derivative are finite at their singular points.

``jnp.where(c, f(x), g)`` does not protect a derivative: the branch not taken is still differentiated, and a NaN
or inf there poisons the result (0 * nan = nan). Every function here first replaces the input where the function is
singular (the "double where").
"""

import jax
import jax.numpy as jnp

from lumen.rules.choices import FLOOR32


def traced(v) -> bool:
    """Return whether any leaf of the pytree ``v`` is a JAX tracer (a value inside a transformation), not concrete."""
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(v))


@jax.custom_jvp
def score(logp):
    """Return 1, exactly, with the derivative of ``logp``: the weight of draws kept fixed while their log density moves.

    ``w = w0 * score(logp_live(d))``: directions (or wavelengths, positions) drawn once and kept, their density's
    parameters live; the value is unchanged and the derivative is the score, d logp (``lumen.sources.reweight``; the
    transport's ``estimator.reweight_log(lp, sg(lp))`` for a finite lp).
    """
    return jnp.ones_like(logp)


score.defjvp(lambda primals, tangents: (jnp.ones_like(primals[0]), tangents[0]))


def sanitize(ok, x, fill):
    """Return x where ok, else a harmless fill value; apply it before a function that is singular where not ok."""
    return jnp.where(ok, x, fill)


def safe_sqrt(x):
    """Return sqrt(max(x, 0)), with derivative 0 (not inf) at x <= 0."""
    ok = x > 0.0
    return jnp.where(ok, jnp.sqrt(sanitize(ok, x, 1.0)), 0.0)


def unit(v, axis=-1, fallback=None):
    """Return v / |v|; at v = 0 return ``fallback`` (default: v itself, i.e. 0), with finite derivatives."""
    s = jnp.sum(v * v, axis=axis, keepdims=True)
    ok = s > 0.0
    u = v / jnp.sqrt(sanitize(ok, s, 1.0))
    return jnp.where(ok, u, v if fallback is None else fallback)


TINY_NORMAL = float(jnp.finfo(jnp.float32).tiny)  # float32's smallest normal number


def safe_norm(v, axis=-1):
    """Return |v| = sqrt(max(sum v^2, TINY_NORMAL)): a positive divisor whose derivative is finite (0) at v = 0.

    ``v / jnp.maximum(jnp.linalg.norm(v), floor)`` is NaN in reverse mode at v = 0 (the norm's derivative v / |v|);
    here the floor is on the square, below which the derivative is 0, so v / safe_norm(v) is 0 at v = 0 both ways.
    """
    return jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=axis), TINY_NORMAL))


def safe_log(x, floor=FLOOR32):
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


def ratio(live, ref):
    """Return live / sg(ref): value live / ref (1 when equal), derivative d live / ref; 0 where ref = 0 (the transport's
    and the sources' one weight of a choice drawn at the reference, ``lumen.transport.estimator``)."""  # noqa: D205, D209
    return safe_div(live, jax.lax.stop_gradient(ref))


def safe_div(a, b, fill=0.0):
    """Return a / b, and ``fill`` where b == 0, with finite derivatives."""
    ok = b != 0.0
    return jnp.where(ok, a / sanitize(ok, b, 1.0), fill)


def literal(values):
    """Return a float32 vector built from scalar literals: a constant of the code that is never a closed-over array."""
    return jnp.stack([jnp.float32(v) for v in values])
