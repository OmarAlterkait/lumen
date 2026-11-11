"""The extension API's derivatives (tier b): each extension point, AD against FD with ``lumen.validate.check_gradient``.

All on the shipped toy, ``examples/data/example_tank.gdml`` (a 1 m water cube, 8 disc PMTs, Rayleigh from Geant4's
water formula, lumen:pmt with the tutorial's values), converted once per module. Every derivative is the forward mode
against the finite difference on the same random numbers, over ``SEEDS`` seeds, judged by ``lumen.validate.stats``'s
rules (``GradientReport.passes``). The worked examples of docs/howto/extending.md are ``examples/extending.py``'s
functions, tested here as they are written there.
"""

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

import lumen
from lumen import sources
from lumen.validate import check_gradient

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
F32 = jnp.float32
SEEDS = 8
N = 1 << 18  # photons per side and seed
PMT7 = 7  # the PMT at (0.25, 0, -0.48), off the LED's axis


def _module(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tank(tmp_path_factory):
    """The tutorial's tank, converted with its own recipe."""
    et = _module("example_tank")
    return lumen.load(lumen.convert(et.GDML, et.RECIPE, tmp_path_factory.mktemp("tank") / "tank.lumen"))


def arrivals(det, src, params=None, seed=0, wrt=(), samples=N):
    return lumen.simulate(det, src, params, samples=samples, seed=seed, wrt=wrt).arrivals


# ------------------------------------------------------------------------------------------------- check_gradient
def test_check_gradient_on_a_built_in_point_position(tank):
    """The Point's position derivative (moving-source method), the procedure every test below uses."""

    def f(dx, seed):
        src = sources.Point(jnp.stack([dx, F32(0.0), F32(0.2)]), n_photons=float(N), lam=420.0)
        return arrivals(tank, src, seed=seed, wrt=("source/position",))

    r = check_gradient(f, 0.0, seeds=SEEDS, h=0.01, stencil="central")
    assert r.passes, r


# ------------------------------------------------------------------------------------------------ source functions
@pytest.fixture(scope="module")
def ex():
    return _module("extending")


def along(key, base, direction=(1.0, 0.0, 0.0)):
    """f's parameter: the source key ``key`` at base + theta * direction."""
    b, u = jnp.asarray(base, F32), jnp.asarray(direction, F32)
    return lambda theta: (key, b + theta * u)


def source_check(det, src, move, *, h, wrt=()):
    p0 = src.parameters(det.params())

    def f(theta, seed):
        key, value = move(theta)
        return arrivals(det, src, p0.set(key, value), seed=seed, wrt=wrt)

    return check_gradient(f, 0.0, seeds=SEEDS, h=h, stencil="central")


@pytest.mark.parametrize("m", [2.0, 4.0, 8.0])
def test_a_cos_m_led_position(tank, ex, m):
    """cos^m about -z: with dlogp the position derivative is right (0.1.0's isotropic start: z -38 at m = 8)."""
    led = ex.cos_m_led(m)
    k = f"source/cos{m:g}/x"
    r = source_check(tank, led, along(k, (0.0, 0.0, 0.2)), h=0.01, wrt=(k,))
    assert r.passes, r


def test_a_soft_edged_cone_position(tank, ex):
    k = "source/soft_cone/x"
    r = source_check(tank, ex.soft_cone, along(k, (0.0, 0.0, 0.2)), h=0.005, wrt=(k,))
    assert r.passes, r


@pytest.mark.parametrize("what", ["led_x", "led_axis", "beam_x"])
def test_the_flasher(tank, ex, what):
    """The guide's LED + laser: the LED's position (moving source), its axis and the beam's centre (reweight)."""
    base = dict(led_x=(0.0, 0.0, 0.2), led_axis=(0.0, 0.0, -1.0), beam_x=(0.1, 0.0, 0.3))[what]
    k = f"source/flasher/{what}"
    h = 0.002 if what == "beam_x" else 0.01
    r = source_check(tank, ex.flasher, along(k, base), h=h, wrt=(k,) if what == "led_x" else ())
    assert r.passes, r


def test_simulate_refuses_a_position_derivative_it_cannot_give(tank, ex):
    import jax

    no_dlogp = lumen.source(params={"x": lumen.Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2))},
                            density="direction", name="no_dlogp")(lambda v, key, n: ex.cos_m_led(4.0).fn(v, key, n)
                                                                  ._replace(dlogp=None))  # fmt: skip
    with pytest.raises(sources.SourceError, match="needs dlogp"):
        arrivals(tank, no_dlogp, samples=1024, wrt=("source/no_dlogp/x",))
    arr = sources.FromArrays(jnp.zeros((1024, 3)) + 0.1, jnp.tile(jnp.asarray([0.0, 0.0, -1.0]), (1024, 1)),
                             jnp.zeros(1024), jnp.full(1024, 420.0), jnp.ones(1024), drawn=True)  # fmt: skip
    with pytest.raises(sources.GradientNotAvailable, match="dlogp=jnp.zeros"):
        arrivals(tank, arr, samples=1024, wrt=("source/x",))
    led = ex.cos_m_led(4.0)
    p = led.parameters(tank.params())

    def f(dx):  # a moving parameter's tangent without it in wrt
        return arrivals(tank, led, p.set("source/cos4/x", jnp.stack([dx, F32(0.0), F32(0.2)])), samples=1024).sum()

    with pytest.raises(sources.GradientNotAvailable, match="moves=True"):
        jax.jvp(f, (F32(0.0),), (F32(1.0),))
