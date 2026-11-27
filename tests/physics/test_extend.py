"""The extension API's derivatives: each extension point, AD against FD with ``lumen.tools.check_gradient``.

All on the shipped toy, ``examples/data/example_tank.gdml`` (a 1 m water cube, 8 disc PMTs, Rayleigh from Geant4's
water formula, lumen:sensor with the tutorial's values), converted once per module. Every derivative is the forward mode
against the finite difference on the same random numbers, over ``SEEDS`` seeds, judged by ``lumen.tools.stats``'s
rules (``GradientReport.passes``). The worked examples of docs/howto/extending.md are ``examples/extending.py``'s
functions, tested here as they are written there.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toys import GPU

import lumen
from lumen import sources
from lumen.build import pack
from lumen.tools import check_gradient

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
    return lumen.load(lumen.convert(et.GDML, tmp_path_factory.mktemp("tank") / "tank.lumen", recipe=et.RECIPE))


@pytest.fixture(scope="module")
def still(tmp_path_factory):
    """The tank with its water renamed: no Geant4 default RAYLEIGH (a material named Water), so no bulk channel."""
    et = _module("example_tank")
    d = tmp_path_factory.mktemp("still")
    text = et.GDML.read_text().replace('name="Water"', 'name="StillWater"').replace('ref="Water"', 'ref="StillWater"')
    (d / "still_tank.gdml").write_text(text)
    return lumen.load(lumen.convert(d / "still_tank.gdml", d / "still_tank.lumen", recipe=et.RECIPE))


def arrivals(det, src, params=None, seed=0, samples=N, reference=None):
    return lumen.simulate(det, src, params, samples=samples, seed=seed, reference=reference).arrivals


# ------------------------------------------------------------------------------------------------- check_gradient
def test_check_gradient_on_the_still_water_absorption(still):
    """The water's ABSLENGTH (a log factor) seen by a Point, the procedure every test below uses."""
    src = sources.Point((0.0, 0.0, 0.2), n_photons=float(N), lam=420.0)

    def f(a, seed):
        return arrivals(still, src, still.params().scale("materials/StillWater/ABSLENGTH", jnp.exp(a)), seed=seed)

    r = check_gradient(f, 0.0, seeds=SEEDS, h=0.05, stencil="central")
    assert r.passes, r


# ------------------------------------------------------------------------------------------------ source functions
@pytest.fixture(scope="module")
def ex():
    _module("example_tank")  # extending.lined_tank imports it
    return _module("extending")


def along(key, base, direction=(1.0, 0.0, 0.0)):
    """f's parameter: the source key ``key`` at base + theta * direction."""
    b, u = jnp.asarray(base, F32), jnp.asarray(direction, F32)
    return lambda theta: (key, b + theta * u)


def source_check(det, src, move, *, h):
    p0 = src.parameters(det.params())

    def f(theta, seed):
        key, value = move(theta)
        return arrivals(det, src, p0.set(key, value), seed=seed)

    return check_gradient(f, 0.0, seeds=SEEDS, h=h, stencil="central")


@pytest.mark.parametrize("what", ["led_axis", "beam_x"])
def test_the_flasher(still, ex, what):
    """The guide's LED + laser: the LED's axis and the beam's centre (reweight; its position moves the light)."""
    base = dict(led_axis=(0.0, 0.0, -1.0), beam_x=(0.1, 0.0, 0.3))[what]
    k = f"source/flasher/{what}"
    h = 0.002 if what == "beam_x" else 0.01
    r = source_check(still, ex.flasher, along(k, base), h=h)
    assert r.passes, r


def test_a_source_function_is_the_same_in_any_chunks(tank, ex):
    """Photon i of a source function is the same whatever the chunk size (the function draws all photons once)."""
    whole = lumen.simulate(tank, ex.flasher, samples=1 << 16, seed=4)
    parts = lumen.simulate(tank, ex.flasher, samples=1 << 16, seed=4, chunk=1 << 14)
    np.testing.assert_allclose(np.asarray(parts.arrivals), np.asarray(whole.arrivals), rtol=1e-6 if GPU else 1e-5)


def test_simulate_refuses_a_position_derivative_it_cannot_give(tank, ex):
    import jax

    def arr(z):  # rows moved (a tangent on their position)
        return arrivals(tank, sources.FromArrays(jnp.zeros((1024, 3)) + z, jnp.tile(jnp.asarray([0.0, 0.0, -1.0]),
                        (1024, 1)), jnp.zeros(1024), jnp.full(1024, 420.0), jnp.ones(1024)),
                        samples=1024).sum()  # fmt: skip

    with pytest.raises(sources.GradientNotAvailable, match="moves where the light starts"):
        jax.jvp(arr, (F32(0.1),), (F32(1.0),))
    led = ex.cos_m_led(4.0)
    p = led.parameters(tank.params())

    def f(dx):  # a moving parameter's tangent: no source-position derivative in this version
        return arrivals(tank, led, p.set("source/cos4/x", jnp.stack([dx, F32(0.0), F32(0.2)])), samples=1024).sum()

    with pytest.raises(sources.GradientNotAvailable, match="moves where the light starts"):
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
    det = lumen.load(tank.path, responses=(r,))
    k = f"sensors/pmt/{key}"
    assert k in det.params() and det.params().explain(k) == f"response {r.name}"
    lights = {}

    def f(theta, seed):
        if seed not in lights:
            src = sources.Point((0.05, 0.02, 0.1), n_photons=1e5, lam=420.0)
            lights[seed] = lumen.simulate(det, src, samples=1 << 16, seed=seed)
        return r(lights[seed], det.params().set(k, theta))

    rep = check_gradient(f, theta0, seeds=6, h=1e-2 * theta0)  # the fewest the check takes
    scale = np.max(np.abs(rep.fd[:-1]))
    assert scale > 0.0 and np.max(np.abs(rep.ad - rep.fd)[:-1]) <= 1e-3 * scale, rep
    assert abs(rep.ad[-1] - rep.fd[-1]) <= 1e-4 * abs(rep.fd[-1]), rep


# ------------------------------------------------------------------------------------------- medium models (values)
@pytest.fixture(scope="module")
def lined(ex, tmp_path_factory):
    """The tank with a liner on the outside (the guide's lined_tank): light that reaches a PMT after a reflection."""
    return ex.lined_tank(tmp_path_factory.mktemp("lined"))


@pytest.mark.parametrize(("model", "key", "theta0"), [("water_hg", "materials/Water/g", 0.9)])
def test_a_medium_model_parameter(lined, ex, model, key, theta0):
    """The guide's HG water: its declared key is differentiable (reweight).

    Per seed the derivative is the exact derivative of the light with every draw held at the reference
    (``simulate(reference=)``: the gradient check's exactness rule against a Richardson difference), and the seeds'
    AD agrees with the difference whose draws move (``check_gradient``). Whether the derivative is resolved from zero
    is not asked: the water's g moves this light by -1.4e-3 of it per unit g (its Rayleigh scattering is Geant4's,
    tens of metres against a 1 m tank), -12.7 +- 4.4 over 64 seeds of 2^18 photons, unresolved at 8 seeds.
    """
    from lumen.compare import gradients
    from lumen.tools import stats

    det = lumen.load(lined, parts={"materials/Water": getattr(ex, model)})
    p0 = det.params()
    assert p0[key] == theta0
    src = sources.Point((0.05, 0.02, 0.1), n_photons=float(N), lam=420.0)

    def f(theta, seed, reference=None):  # the default roulette, as for its values
        return arrivals(det, src, p0.set(key, theta), seed=seed, reference=reference)

    # Richardson: the light reflected twice makes the arrivals curve in the parameter, and the seeds' spread of AD - FD
    # is small enough (the draws stay, the weights move) that a central FD's h^2 term would show
    r = check_gradient(f, theta0, seeds=SEEDS, h=0.02)
    assert r.passes is not False, r  # unresolved at 8 seeds (above): no power is the expected verdict, not a failure
    for seed in (0, 1):
        ad = float(jax.jvp(lambda t, s=seed: f(t, s).sum(), (F32(theta0),), (F32(1.0),))[1])
        steps = [k * gradients.H for k in (-1, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1)]
        side = {x: np.full((1, 1), float(f(F32(theta0 + x), seed, p0).sum())) for x in steps}
        f0 = np.full((1, 1), float(f(F32(theta0), seed).sum()))
        r = stats.fd_order(side, f0, np.full((1, 1), ad), gradients.H, False, gradients.FLOOR_V, gradients.EXACT)
        assert r["applies"] and r["ratio"] <= 1.0, (seed, ad, r)


# --------------------------------------------------------------------------------------------- controls, diagnostics
def test_simulate_takes_controls_and_diagnostics(lined):
    from lumen.diagnostics import TRACE_FIELDS, Trace

    tank = lumen.load(lined)  # a liner (R = 0.5): light that reaches a PMT after its first event
    src = sources.Point((0.05, 0.02, 0.1), n_photons=4096.0, lam=420.0)
    n = 4096
    plain = lumen.simulate(tank, src, samples=n, seed=2)
    same = lumen.simulate(tank, src, samples=n, seed=2, controls=lumen.Controls())
    assert np.asarray(plain.arrivals).tobytes() == np.asarray(same.arrivals).tobytes() and plain.diagnostics is None
    prov = dict(plain.provenance)  # how the run was made: its controls as the loop read them, each group's gate
    reach = lumen.api._model(tank).reach  # max_distance: the tank's reach (its world box's diagonal)
    want = (("max_distance", float(np.float32(reach))), ("roulette", float(np.float32(lumen.Controls().roulette))))
    assert prov["controls"] == want  # the default roulette: a run knob, the same for every detector
    assert 1.0 < reach < 10.0
    assert prov["gate"] == (("pmt", (-10.0, 60.0)),)
    tr = Trace(steps=4, n_traced=n)
    seen = lumen.simulate(tank, src, samples=n, seed=2, diagnostics=(tr,))
    if GPU:  # exact sums: a probe changes nothing (the CPU's sums may round a last bit differently)
        assert np.asarray(seen.arrivals).tobytes() == np.asarray(plain.arrivals).tobytes()
    cells = tr.read(seen.diagnostics["trace"])
    assert cells.shape == (n, 4, len(TRACE_FIELDS))
    first = cells[:, 0]
    assert np.all(first[:, TRACE_FIELDS.index("alive")] == 1.0)
    x = first[:, [TRACE_FIELDS.index(k) for k in ("x0", "x1", "x2")]]
    assert np.all(np.abs(x) <= 0.5 + 1e-4)  # the first vertex is in the 1 m cube
    with pytest.raises(ValueError, match="one chunk"):
        lumen.simulate(tank, src, samples=n, seed=2, chunk=1024, diagnostics=(tr,))


# ----------------------------------------------------------------------------------------------------- the roulette
def _listed(n, weight, seed=0):
    """A FromArrays list of n isotropic photons at one point of the tank, 420 nm, every weight ``weight``."""
    d = np.random.default_rng(seed).normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return sources.FromArrays(np.tile([0.05, 0.02, 0.1], (n, 1)), d, np.zeros(n), np.full(n, 420.0),
                              jnp.broadcast_to(jnp.asarray(weight, F32), (n,)))  # fmt: skip


def test_unit_weights_at_an_explicit_roulette_are_the_photon_loop_run_directly(lined):
    """The weight carry is invisible on unit weights: simulate at an explicit roulette is bitwise the photon loop run
    on the same rows with that roulette (no weight scale, chunks, nor resolution in between)."""
    import jax

    from lumen import run as runner
    from lumen.readout import fill
    from lumen.transport import step as steps

    tank, n = lumen.load(lined), 1 << 14
    src = _listed(n, 1.0)
    light = lumen.simulate(tank, src, samples=n, seed=3, controls=lumen.Controls(roulette=0.03))
    _, _, _, grid, lay = lumen.api._layouts(tank, None, 0)[0]
    model = lumen.api._model(tank, tuple(sorted(grid.reads)))
    ops = steps.operands(model, jax.tree_util.tree_map(jnp.asarray, pack.checked(tank, tank.params())), None,
                         steps.Controls(roulette=0.03))  # fmt: skip
    key = jax.random.key_data(jax.random.PRNGKey(3))
    m = lumen.api._layouts(tank, None, 0)[0][1]
    t = runner.run(fill.readout((lay,)), model, ops, src.emit(None, key, n), key, weights=(m.acceptance(m.values),))
    direct = np.asarray(t.outputs["arrival"], np.float32)
    if GPU:  # exact sums: bitwise
        assert np.asarray(light.arrivals).tobytes() == direct.tobytes()
    else:
        np.testing.assert_allclose(np.asarray(light.arrivals), direct, rtol=1e-5)


def test_a_source_ten_times_brighter_gives_ten_times_the_light(lined):
    """The roulette is relative to a photon's own start weight: w = 10 rows give 10 x the w = 1 rows' light, the same
    photons followed (only the float32 rounding of the weights differs), at the default roulette."""
    tank, n = lumen.load(lined), 1 << 16
    one = lumen.simulate(tank, _listed(n, 1.0), samples=n, seed=5)
    ten = lumen.simulate(tank, _listed(n, 10.0), samples=n, seed=5)
    a1, a10 = np.asarray(one.arrivals, np.float64), np.asarray(ten.arrivals, np.float64)
    assert np.all(np.abs(a10 - 10.0 * a1) <= 3.0 * 10.0 * np.asarray(one.arrivals_error) + 1e-6 * a10)
    np.testing.assert_allclose(a10, 10.0 * a1, rtol=1e-5)  # the same photons: far inside the MC error


def test_the_source_weight_derivative_is_exact(lined):
    """d/dth of the arrivals with half the rows' weights x (1 + th): AD against FD on the same random numbers."""
    tank, n = lumen.load(lined), 1 << 16
    half = jnp.asarray(np.arange(n) % 2 == 0, F32)

    def f(theta, seed):
        src = _listed(n, 1.0 + 9.0 * half * (1.0 + theta), seed=seed)  # bright and dim rows (10 and 1)
        return arrivals(tank, src, seed=seed, samples=n)

    r = check_gradient(f, 0.0, seeds=SEEDS, h=0.05, stencil="central")
    assert r.passes, r
    assert abs(r.ad[-1]) > 3 * r.ad_sem[-1], r


def test_the_default_roulette_is_unbiased_against_1e_4(lined, ex):
    """simulate's default roulette (one ε for values and derivatives alike) against 1e-4, paired by seed:
    the total arrivals and one derivative (the liner's REFLECTIVITY, scaled) agree within |z| < 3."""
    import jax

    det = lumen.load(lined)
    p0, k = det.params(), "surfaces/Liner/REFLECTIVITY"
    src = sources.Point((0.05, 0.02, 0.1), n_photons=float(N), lam=420.0)
    old = lumen.Controls(roulette=1e-4)

    def total(theta, seed, controls):
        return lumen.simulate(det, src, p0.scale(k, theta), samples=1 << 17, seed=seed,
                              controls=controls).arrivals.sum()  # fmt: skip

    dv, dd = [], []
    for seed in range(16):
        dv.append(float(total(F32(1.0), seed, None)) - float(total(F32(1.0), seed, old)))
        tan = [jax.jvp(lambda th, c=c: total(th, seed, c), (F32(1.0),), (F32(1.0),))[1] for c in (None, old)]
        dd.append(float(tan[0]) - float(tan[1]))
    for d in (np.asarray(dv), np.asarray(dd)):
        assert abs(d.mean()) <= 3.0 * d.std(ddof=1) / np.sqrt(len(d)), d


def test_a_roulette_is_a_number_above_zero(lined):
    """A photon ends only at weight 0, by the roulette: an ε of 0 (or below) is refused, so nothing else ends one."""
    tank = lumen.load(lined)
    src = sources.Point((0.05, 0.02, 0.1), n_photons=4096.0, lam=420.0)
    with pytest.raises(ValueError, match="ε > 0"):
        lumen.simulate(tank, src, samples=4096, seed=2, controls=lumen.Controls(roulette=0.0))


def test_several_modelled_groups_fill_what_one_group_fills(tmp_path_factory):
    """The tank's 8 PMTs as one modelled group, then as two (split by an id-map column, the same value set): with the
    same seed every sensor's histograms and arrivals are bitwise the same (the exact sums do not depend on the order of
    the adds), and each group holds its own sensors"""
    import dataclasses
    import re

    from lumen.recipe import Recipe

    et = _module("example_tank")
    g0 = et.RECIPE.groups[0]
    one = dataclasses.replace(g0, rule="any_entry")
    gdml = tmp_path_factory.mktemp("gdml") / "tank_any_entry.gdml"  # every entry counts: no cathode border
    gdml.write_text(re.sub(r"<bordersurface .*?</bordersurface>", "", et.GDML.read_text(), flags=re.S))

    def half(df, paths, inputs):
        return {"half": ["a" if i % 2 == 0 else "b" for i in range(len(paths))]}

    two = tuple(dataclasses.replace(one, name=f"pmt_{h}", where={"half": h}) for h in "ab")
    dets = []
    for name, groups in (("one", (one,)), ("two", two)):
        r = Recipe(name, et.RECIPE.detector_volume, groups=groups, sensor_columns=half)
        dets.append(lumen.load(lumen.convert(gdml, tmp_path_factory.mktemp(name) / f"{name}.lumen", recipe=r)))
    src = sources.Point((0.05, 0.02, 0.1), n_photons=float(N), lam=420.0)
    l1, l2 = (lumen.simulate(d, src, samples=N, seed=3) for d in dets)
    assert set(l2.groups) == {"pmt_a", "pmt_b"} and set(l1.groups) == {"pmt"}
    np.testing.assert_array_equal(np.asarray(l1.arrivals), np.asarray(l2.arrivals))
    ids1, ids2 = list(dets[0].parts.optics.sensor_pvs), list(dets[1].parts.optics.sensor_pvs)
    assert ids1 == ids2
    from lumen.build import build

    h1 = l1.group("pmt")
    for name in ("pmt_a", "pmt_b"):
        g = next(x for x in dets[1].sensor_set.groups if x.name == name)
        tags = build.group_tags(g, dets[1].parts, dets[1].sensor_set)
        h2 = l2.group(name)
        assert h2.charge.shape[0] == len(tags) == 4
        np.testing.assert_array_equal(np.asarray(h2.charge), np.asarray(h1.charge)[tags])
        np.testing.assert_array_equal(np.asarray(h2.charge2), np.asarray(h1.charge2)[tags])
        np.testing.assert_array_equal(np.asarray(h2.time), np.asarray(h1.time)[tags])
        p = dets[1].params()
        np.testing.assert_array_equal(np.asarray(lumen.readout.pe(h2, p)),
                                      np.asarray(lumen.readout.pe(h1, dets[0].params()))[tags])  # fmt: skip


def test_a_vmapped_jvp_over_one_tangent_is_the_plain_jvp(still):
    n = 1 << 14

    src = sources.Point((0.0, 0.0, 0.2), n_photons=float(n), lam=420.0)

    def f(a):
        return arrivals(still, src, still.params().scale("materials/StillWater/ABSLENGTH", a), samples=n, seed=3).sum()

    plain = jax.jvp(f, (F32(1.0),), (F32(1.0),))[1]
    one = jax.vmap(lambda t: jax.jvp(f, (F32(1.0),), (t,))[1])(jnp.ones(1, F32))
    assert float(one[0]) == pytest.approx(float(plain), rel=1e-5)
