"""The worked examples of docs/howto/extending.md, as the guide shows them; tests/physics/test_extend.py checks each.

Each example is user code: it imports lumen and nothing private. The detector is the tutorial's tank
(``example_tank.detector()``); every derivative is checked AD against FD with ``lumen.validate.check_gradient``.
"""
# ruff: noqa: D103  (the guide explains each example)
# fmt: off

import math

import jax
import jax.numpy as jnp

import lumen
from lumen import Param
from lumen.sources import cone_about, gaussian_spot, grad_logpdf, reweight


# ---------------------------------------------------------------- 2. a source function: an LED and a laser in one
# example: flasher
@lumen.source(params={"led_x": Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2)),
                      "led_axis": Param("", shape=(3,), default=(0.0, 0.0, -1.0)), "m": Param("", default=4.0),
                      "beam_x": Param("m", shape=(3,), default=(0.1, 0.0, 0.3)),
                      "beam_sigma": Param("m", default=0.003),
                      "t_led": Param("ns", default=0.0), "t_beam": Param("ns", default=25.0)},
              density="direction")
def flasher(v, key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    h = n // 2
    a = jax.lax.stop_gradient(v["led_axis"])                               # directions fixed; the axis enters by weight
    u = jax.random.uniform(k1, (h, 2), minval=1e-7)
    d = cone_about(a, u[:, 0] ** (1 / (v["m"] + 1)), 2 * math.pi * u[:, 1])

    def logp(d_):                                                          # cos^m about the live axis (+ a constant)
        return v["m"] * jnp.log(jnp.clip(d_ @ v["led_axis"], 1e-12))

    led = lumen.Photons(jnp.broadcast_to(v["led_x"], (h, 3)), d, v["t_led"] + 2.0 * jax.random.exponential(k2, (h,)),
                        jnp.full(h, 450.0), reweight(jax.vmap(logp)(d)) * 1e6 / h, dlogp=grad_logpdf(logp, d))
    beam = gaussian_spot(k3, n - h, centre=v["beam_x"], sigma=v["beam_sigma"], direction=(0.0, 0.0, -1.0),
                         t=v["t_beam"], lam=405.0, photons=1e4)            # a fixed spot, its centre by reweight
    return lumen.concat(led, beam)                                         # beam rows: dlogp 0, no position tangent
# end example


# example: cos_m
def cos_m_led(m):
    """Return an LED of profile cos^m about -z at (0, 0, 0.2), 420 nm, its position a moving parameter."""

    @lumen.source(params={"x": Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2))},
                  density="direction", name=f"cos{m:g}")
    def led(v, key, n):
        axis = jnp.array([0.0, 0.0, -1.0])
        u = jax.random.uniform(key, (n, 2), minval=1e-7)
        d = cone_about(axis, u[:, 0] ** (1.0 / (m + 1.0)), 2 * math.pi * u[:, 1])
        logp = lambda d_: m * jnp.log(jnp.clip(d_ @ axis, 1e-12))          # noqa: E731
        return lumen.Photons(jnp.broadcast_to(v["x"], (n, 3)), d, jnp.zeros(n), jnp.full(n, 420.0),
                             jnp.ones(n), dlogp=grad_logpdf(logp, d))

    return led
# end example


# example: soft_cone
CONE, EDGE = math.radians(20.0), math.radians(1.0)


@lumen.source(params={"x": Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2))}, density="direction")
def soft_cone(v, key, n):
    """Draw a 20 degree cone about -z whose edge is a logistic about 1 degree wide (a hard edge is refused)."""
    axis = jnp.array([0.0, 0.0, -1.0])
    top = CONE + 8 * EDGE                                                  # drawn uniformly out to 8 widths
    u = jax.random.uniform(key, (n, 2), minval=1e-7)
    d = cone_about(axis, 1.0 - u[:, 0] * (1.0 - math.cos(top)), 2 * math.pi * u[:, 1])

    def logp(d_):                                                          # the edge in cos(theta): smooth everywhere
        return jax.nn.log_sigmoid((d_ @ axis - math.cos(CONE)) / (EDGE * math.sin(CONE)))

    return lumen.Photons(jnp.broadcast_to(v["x"], (n, 3)), d, jnp.zeros(n), jnp.full(n, 420.0),
                         jnp.exp(jax.vmap(logp)(d)), dlogp=grad_logpdf(logp, d))
# end example
