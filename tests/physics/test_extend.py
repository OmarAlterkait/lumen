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
import numpy as np
import pytest
from toys import GPU

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


def test_a_source_function_is_the_same_in_any_chunks(tank, ex):
    """Photon i of a source function is the same whatever the chunk size (the function draws all photons once)."""
    whole = lumen.simulate(tank, ex.flasher, samples=1 << 16, seed=4)
    parts = lumen.simulate(tank, ex.flasher, samples=1 << 16, seed=4, chunk=1 << 14)
    np.testing.assert_allclose(np.asarray(parts.arrivals), np.asarray(whole.arrivals), rtol=1e-6 if GPU else 1e-5)


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


# ------------------------------------------------------------------------------------------------ sensor responses
@pytest.mark.parametrize(("response", "key", "theta0"), [("saturating_pe", "sat", 300.0),
                                                          ("saturating_pe", "qe_scale", 1.0),
                                                          ("my_waveform", "tau_r", 2.0),
                                                          ("my_waveform", "tau_f", 8.0)])  # fmt: skip
def test_a_response_is_exact_after_tracking(tank, ex, response, key, theta0):
    """The guide's saturating PE and waveform: their keys join the detector's, and their derivatives (no new
    simulation) are exact: AD equals the Richardson FD on every observable to its float32 resolution. (The seeds'
    z rules of ``passes`` judge Monte Carlo noise; here there is none, so the FD's own rounding would dominate them.)"""
    r = getattr(ex, response)
    det = tank.with_responses(r)
    k = f"sensors/pmt/{key}"
    assert k in det.params() and det.params().explain(k) == f"response {r.name}"
    lights = {}

    def f(theta, seed):
        if seed not in lights:
            src = sources.Point((0.05, 0.02, 0.1), n_photons=1e5, lam=420.0)
            lights[seed] = lumen.simulate(det, src, samples=1 << 16, seed=seed)
        return r(lights[seed], det.params().set(k, theta))

    rep = check_gradient(f, theta0, seeds=4, h=1e-2 * theta0)
    scale = np.max(np.abs(rep.fd[:-1]))
    assert scale > 0.0 and np.max(np.abs(rep.ad - rep.fd)[:-1]) <= 1e-3 * scale, rep
    assert abs(rep.ad[-1] - rep.fd[-1]) <= 1e-4 * abs(rep.fd[-1]), rep


# ------------------------------------------------------------------------------------------------ registered models
@pytest.fixture(scope="module")
def lined(ex, tmp_path_factory):
    """The tank with a liner on the outside (the guide's lined_tank): a surface for a surface model to read."""
    return ex.lined_tank(tmp_path_factory.mktemp("lined"))


@pytest.mark.parametrize(("models", "key", "theta0"), [
    ({"materials/Water": "mylab:water_hg"}, "materials/Water/g", 0.9),
    ({"surfaces/Liner": "mylab:schlick"}, "surfaces/Liner/R0", 0.04),
    ({"surfaces/CathodeSurface": "mylab:schlick_cathode"}, "surfaces/CathodeSurface/R0", 0.04),
])  # fmt: skip
def test_a_registered_model_parameter(lined, ex, models, key, theta0):
    """The guide's HG water, Schlick liner and Schlick cathode: their declared keys are differentiable (reweight)."""
    det = lumen.load(lined, models=models)
    p0 = det.params()
    assert p0[key] == theta0
    src = sources.Point((0.05, 0.02, 0.1), n_photons=float(N), lam=420.0)

    def f(theta, seed):
        return arrivals(det, src, p0.set(key, theta), seed=seed)

    # Richardson: the light reflected twice makes the arrivals curve in the parameter, and the seeds' spread of AD - FD
    # is small enough (the draws stay, the weights move) that a central FD's h^2 term would show
    r = check_gradient(f, theta0, seeds=SEEDS, h=0.02)
    assert r.passes, r
    assert abs(r.ad[-1]) > 3 * r.ad_sem[-1], r  # resolved: the parameter moves the light


# --------------------------------------------------------------------------------------------- controls, diagnostics
def test_simulate_takes_controls_and_diagnostics(tank):
    from lumen.diagnostics import TRACE_FIELDS, Trace

    src = sources.Point((0.05, 0.02, 0.1), n_photons=4096.0, lam=420.0)
    n = 4096
    plain = lumen.simulate(tank, src, samples=n, seed=2)
    same = lumen.simulate(tank, src, samples=n, seed=2, controls=lumen.Controls())
    assert np.asarray(plain.arrivals).tobytes() == np.asarray(same.arrivals).tobytes() and plain.diagnostics is None
    early = lumen.simulate(tank, src, samples=n, seed=2, controls=lumen.Controls(t_end=0.1))
    # a photon past t_end is retired at its first event: only light that reaches a PMT on its first path counts
    assert float(np.asarray(early.arrivals).sum()) < float(np.asarray(plain.arrivals).sum())
    assert float(np.asarray(early.health["retired"]).sum()) > 0.0
    tr = Trace(steps=4, n_traced=n)
    seen = lumen.simulate(tank, src, samples=n, seed=2, diagnostics=(tr,))
    if GPU:  # exact sums: a probe changes nothing (the CPU's sums may round a last bit differently)
        assert np.asarray(seen.arrivals).tobytes() == np.asarray(plain.arrivals).tobytes()
    cells = tr.read(seen.diagnostics["trace"])
    assert cells.shape == (n, 4, len(TRACE_FIELDS)) and float(seen.diagnostics["weight_scale"]) == 1.0
    first = cells[:, 0]
    assert np.all(first[:, TRACE_FIELDS.index("alive")] == 1.0)
    x = first[:, [TRACE_FIELDS.index(k) for k in ("x0", "x1", "x2")]]
    assert np.all(np.abs(x) <= 0.5 + 1e-4)  # the first vertex is in the 1 m cube
    with pytest.raises(ValueError, match="one chunk"):
        lumen.simulate(tank, src, samples=n, seed=2, chunk=1024, diagnostics=(tr,))
