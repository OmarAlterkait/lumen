"""The extension API (lumen 0.1.2): declared parameters, source functions, responses, registered models, checks.

The derivative checks (AD against FD on the shipped toys) are the physics tier's ``test_extend.py``; here each piece's
contract: declarations become keys, refusals refuse, built-ins keep their bits.
"""

import numpy as np
import pytest

import lumen
from lumen.params import Param, Params, declared


# ------------------------------------------------------------------------------------------------ declared keys
def test_a_declaration_is_checked_and_hashable():
    p = Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2))
    assert (
        p.default == (0.0, 0.0, 0.2)
        and p.shape == (3,)
        and hash(p) == hash(Param("m", shape=[3], moves=True, default=[0, 0, 0.2]))
    )
    assert Param("", (0, 50), default=4.0).range == (0.0, 50.0)
    with pytest.raises(ValueError, match="shape"):
        Param("m", shape=(3,), default=1.0)
    with pytest.raises(ValueError, match="outside"):
        Param("", (0, 1), default=2.0)
    with pytest.raises(ValueError, match="lo <= hi"):
        Param("", (1, 0))
    assert lumen.Param is Param


def test_declared_parameters_become_keys_with_their_info():
    values, spec = declared("sensors/pmt/", {"sat": Param("pe", (1, 1e4), default=300.0), "k": Param("")})
    assert values == {"sensors/pmt/sat": 300.0} and set(spec) == {"sensors/pmt/sat", "sensors/pmt/k"}
    p = Params({}, {}, {}).merged(values, spec, "response r")
    assert p["sensors/pmt/sat"] == 300.0 and p.explain("sensors/pmt/sat") == "response r"
    info = p.info("sensors/pmt/sat")
    assert (info.unit, info.range, info.method) == ("pe", (1.0, 1e4), "reweight")
    with pytest.raises(ValueError, match="outside"):
        p.set("sensors/pmt/sat", 0.5)
    v, s = declared("source/f/", {"x": Param("m", shape=(3,), moves=True, default=(1, 2, 3))})
    assert s["source/f/x"].method == "moving_paths" and v["source/f/x"].dtype == np.float32
    with pytest.raises(TypeError, match="lumen.Param"):
        declared("x/", {"a": 1.0})


# ------------------------------------------------------------------------------------------------ check_gradient
def _noisy(bias=0.0):
    """f(theta, seed): a noisy quadratic per observable; AD carries ``bias`` (a wrong derivative when nonzero)."""
    import jax
    import jax.numpy as jnp

    @jax.custom_jvp
    def g(th):
        return th**2

    g.defjvp(lambda p, t: (p[0] ** 2, (2.0 * p[0] + bias) * t[0]))

    def f(theta, seed):
        noise = jax.random.normal(jax.random.PRNGKey(seed), (6,)) * 0.1
        return g(theta) * jnp.arange(1.0, 7.0) + noise * (1.0 + theta)

    return f


def test_check_gradient_passes_a_right_derivative_and_fails_a_wrong_one():
    from lumen.validate import check_gradient

    good = check_gradient(_noisy(), 1.5, seeds=8)
    assert good.passes, good
    assert good.ad.shape == (7,) and abs(good.ad[-1] - 21 * (3.0 + 0.1 * 0)) < 3.0
    assert "passes" in str(good)
    bad = check_gradient(_noisy(bias=0.3), 1.5, seeds=8, stencil="central")
    assert not bad.passes and abs(bad.z_total) > 3.0, bad
    with pytest.raises(ValueError, match="stencil"):
        check_gradient(_noisy(), 1.5, stencil="forward")


# ------------------------------------------------------------------------------------------------ source functions
def _led(density="direction", give_g=True, m_default=4.0):
    import math

    import jax
    import jax.numpy as jnp

    from lumen.sources import cone_about, grad_logpdf, reweight

    @lumen.source(params={"x": Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2)),
                          "m": Param("", (0, 50), default=m_default)}, density=density)  # fmt: skip
    def led(v, key, n):
        a = jnp.asarray([0.0, 0.0, -1.0])
        u = jax.random.uniform(key, (n, 2), minval=1e-7, maxval=1.0)
        d = cone_about(a, u[:, 0] ** (1 / (jax.lax.stop_gradient(v["m"]) + 1)), 2 * math.pi * u[:, 1])

        def logp(d_):
            return v["m"] * jnp.log(jnp.clip(d_ @ a, 1e-12)) + jnp.log(v["m"] + 1)

        w = reweight(jax.vmap(logp)(d)) * 1e3 / n
        return lumen.Photons(jnp.broadcast_to(v["x"], (n, 3)), d, jnp.zeros(n), jnp.full(n, 420.0), w,
                             dlogp=grad_logpdf(logp, d) if give_g else None)  # fmt: skip

    return led


def test_a_source_function_is_a_source_with_keys():
    import jax

    led = _led()
    assert led.prefix == "source/led/" and led.MOVING == ("x",) and led.ORIGINS == ("x",) and led.drawn_direction
    p = led.parameters()
    assert set(p) == {"source/led/x", "source/led/m"} and p.info("source/led/x").method == "moving_paths"
    led.check(None)
    e = led.emit(None, jax.random.PRNGKey(1), 64, 0, 256)
    assert e.x.shape == (64, 3) and e.dlogp.shape == (64, 3) and bool(e.drawn_direction.all())
    np.testing.assert_allclose(float(e.w.sum()), 1e3 * 64 / 256, rtol=1e-6)  # the chunk's share of the photons
    want = 4.0 * np.array([0, 0, -1.0]) / np.asarray(e.d @ np.array([0, 0, -1.0]))[:, None]  # m a / (d . a)
    g, d = np.asarray(e.dlogp), np.asarray(e.d)
    tang = lambda v: v - np.sum(v * d, 1, keepdims=True) * d  # noqa: E731 (only the tangent part is read)
    np.testing.assert_allclose(tang(g), tang(want), rtol=1e-4, atol=1e-4)
    e2 = led.emit(None, jax.random.PRNGKey(1), 64, 64, 256)
    assert not np.array_equal(np.asarray(e.d), np.asarray(e2.d))  # each chunk its own photons
    whole = led.emit(None, jax.random.PRNGKey(1), 256, 0, 256)
    for a, b in zip(whole, e2, strict=True):  # photon i is the same whatever the chunk
        assert (a is None and b is None) or np.array_equal(np.asarray(a)[64:128], np.asarray(b))


def test_a_source_function_refuses_what_it_cannot_differentiate():
    import jax
    import jax.numpy as jnp

    from lumen import api, sources
    from lumen.transport.state import Emission

    with pytest.raises(sources.SourceError, match="needs dlogp"):
        _led(give_g=False).check(None)
    with pytest.raises(sources.SourceError, match="has no dlogp"):
        _led(density="isotropic").check(None)
    with pytest.raises(sources.GradientNotAvailable, match="hard angular edge"):
        _led(m_default=0.0).check(None)  # a uniform cone: no gradient anywhere
    led = _led()
    p = led.parameters()

    def x_of(dx, moving):
        q = p.set("source/led/x", jnp.asarray([0.0, 0.0, 0.2]) + dx * jnp.asarray([1.0, 0.0, 0.0]))
        return led.emit(q, jax.random.PRNGKey(0), 16, 0, 16, moving=moving).x

    with pytest.raises(sources.GradientNotAvailable, match="moves=True"):
        jax.jvp(lambda a: x_of(a, False), (0.0,), (1.0,))
    assert float(jax.jvp(lambda a: x_of(a, True), (0.0,), (1.0,))[1][:, 0].min()) == 1.0
    assert api._moving(led, p, ("source/led/x",))[0] and not api._moving(led, p, ("source/led/m",))[0]
    n = 8
    arr = sources.FromArrays(jnp.zeros((n, 3)), jnp.tile(jnp.asarray([0.0, 0.0, 1.0]), (n, 1)), jnp.zeros(n),
                             jnp.full(n, 400.0), jnp.ones(n), drawn=True)  # fmt: skip
    assert api._moving(arr, None, ("source/x",))[0]
    with pytest.raises(sources.GradientNotAvailable, match="dlogp=jnp.zeros"):
        arr.emit(None, jax.random.PRNGKey(0), n, moving=True)  # moved without its density's gradient
    with_g = sources.FromArrays(arr.x, arr.d, arr.t, arr.lam, arr.w, drawn=True, dlogp=jnp.zeros((n, 3)))
    assert with_g.emit(None, jax.random.PRNGKey(0), n, moving=True).dlogp.shape == (n, 3)
    e = arr.emit(None, jax.random.PRNGKey(0), n)
    assert e.dlogp is None and len(e) == len(Emission._fields)  # built-ins: no dlogp, the chain start unchanged
    with pytest.raises(ValueError, match="density"):
        lumen.source(density="cone")


def test_concat_reweight_and_the_beam():
    import jax
    import jax.numpy as jnp

    from lumen.sources import concat, gaussian_spot, reweight

    v, t = jax.jvp(lambda s: reweight(s * jnp.arange(3.0)), (2.0,), (1.0,))
    assert np.array_equal(np.asarray(v), np.ones(3)) and np.array_equal(np.asarray(t), np.arange(3.0))

    def spot(c):
        return gaussian_spot(jax.random.PRNGKey(0), 4096, centre=jnp.stack([c, 0.0, 0.0]), sigma=0.01,
                             direction=(0, 0, -1), lam=405.0, photons=1.0)  # fmt: skip

    b, db = jax.jvp(spot, (0.0,), (1.0,))
    assert float(b.w.sum()) == pytest.approx(1.0, rel=1e-5) and float(np.abs(np.asarray(db.x)).max()) == 0.0
    assert abs(float(db.w.sum())) < 5.0  # the score: mean zero, sem 1 / (sigma sqrt(n)) = 1.6
    led = _led()
    a = led(led.values(), jax.random.PRNGKey(2), 8)
    c = concat(a, spot(0.0))
    assert c.x.shape == (4104, 3) and c.dlogp.shape == (4104, 3) and float(np.abs(c.dlogp[8:]).max()) == 0.0
    assert c.pol is None


# ------------------------------------------------------------------------------------------------ sensor responses
def test_a_response_reads_the_histograms_with_its_keys():
    import jax
    import jax.numpy as jnp
    import readout_toys as rt

    from lumen import readout as rd

    m = rt.model()
    h = rt.hist(m)
    p = rt.params(m)

    @lumen.response(group="pmt", params={"sat": Param("pe", (1, 1e4), default=300.0), "k": Param("", default=1.0)})
    def sat_pe(h_, v):
        pe = v["k"] * rd.pe(h_, v)
        return v["sat"] * (1.0 - jnp.exp(-pe / v["sat"]))

    pe = np.asarray(rd.pe(h, p))
    y = np.asarray(sat_pe(h, p))  # unset keys: the defaults
    np.testing.assert_allclose(y, 300.0 * (1 - np.exp(-pe / 300.0)), rtol=1e-5)
    q = sat_pe.parameters(Params({}, {}, {}))
    assert set(q) == {"sensors/pmt/sat", "sensors/pmt/k"} and q.info("sensors/pmt/sat").method == "after_tracking"
    g = jax.grad(lambda s: sat_pe(h, {**p, "sensors/pmt/sat": s}).sum())(300.0)
    want = np.sum(1 - np.exp(-pe / 300.0) - pe / 300.0 * np.exp(-pe / 300.0))
    assert float(g) == pytest.approx(want, rel=1e-4)
    nodes, late, early = rd.time_nodes(h, p)
    tot = np.asarray(nodes.sum(-1) + late[:, 0] + early[:, 0])
    np.testing.assert_allclose(tot, pe, rtol=1e-5)  # the time nodes carry pe
    wrong = lumen.response(group="other")(lambda h_, v: 0.0)
    with pytest.raises(ValueError, match="reads group"):
        wrong(h, p)


# ------------------------------------------------------------------------------------------------ registered models
def _examples():
    import importlib.util
    import sys
    from pathlib import Path

    ex = Path(__file__).resolve().parents[2] / "examples"
    sys.path.insert(0, str(ex))
    if "extending" not in sys.modules:
        spec = importlib.util.spec_from_file_location("extending", ex / "extending.py")
        sys.modules["extending"] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules["extending"]


@pytest.fixture(scope="module")
def lined(tmp_path_factory):
    return _examples().lined_tank(tmp_path_factory.mktemp("lined"))


def test_a_registered_medium_and_surface_model_build_with_their_keys(lined):
    import jax
    import jax.numpy as jnp

    ex = _examples()
    plain = lumen.load(lined)
    det = lumen.load(lined, models={"materials/Water": "mylab:water_hg", "surfaces/Liner": "mylab:schlick"})
    p = det.params()
    assert p["materials/Water/g"] == 0.9 and p.explain("materials/Water/g") == "model mylab:water_hg default"
    assert p["surfaces/Liner/R0"] == 0.04 and p.info("surfaces/Liner/R0").range == (0.0, 1.0)
    assert "materials/Water/g" not in plain.params() and set(plain.params()) <= set(p)
    lib = det.parts.library
    ray = next(c for c in lib.media["Water"].medium.channels if c.name == "rayleigh")
    assert ray.lobe is lumen.media.henyey_greenstein("g")
    assert lib.media["Water"].params["rayleigh"]["lobe"] == {"g": 0.9}
    assert lib.surfaces["Liner"].surface_type.name == "schlick" and lib.surfaces["Liner"].params == {"R0": 0.04}
    src = det.parts.sources
    assert src[("media", "Water", "rayleigh", "lobe", "g")].key == "materials/Water/g"
    assert src[("surfaces", "Liner", "R0")].key == "surfaces/Liner/R0"
    q = p.set("materials/Water/g", 0.5).set("surfaces/Liner/R0", 0.2)
    named = det.pack(q)  # a value set: applied to the file (the declared property added) and packed
    assert named is not None
    from lumen.format.apply import apply

    df2 = apply(det.file, q)
    assert float(df2.materials["Water"].properties["g"].value) == 0.5
    t = jax.jvp(lambda g: det.pack(p.set("materials/Water/g", g)), (jnp.float32(0.9),), (jnp.float32(1.0),))[1]
    assert any(float(jnp.abs(x).max()) == 1.0 for x in jax.tree_util.tree_leaves(t))  # g reaches the packed rows
    assert ex.water_hg.name == "mylab:water_hg"


def test_a_detecting_model_and_the_refusals(lined):
    import jax.numpy as jnp

    from lumen import models, plugins
    from lumen.build.library import LibraryError

    det = lumen.load(lined, models={"surfaces/CathodeSurface": "mylab:schlick_cathode"})
    sens = det.parts.library.sensors
    e = next(v for k, v in sens.items() if k.endswith("CathodeSurface"))
    # under lumen:pmt every absorption entering is counted: eff 1, folded into the type as lumen's own sensors' is
    assert e.surface_type.name == "schlick_cathode{eff=1}" and e.params == {"R0": 0.04}
    assert "surfaces/CathodeSurface/R0" in det.params()
    with pytest.raises(LibraryError, match="not one this library reads"):
        lumen.load(lined, models={"surfaces/Liner": "mylab:nothing"})
    with pytest.raises(LibraryError, match="as a medium"):
        lumen.load(lined, models={"materials/Water": "mylab:schlick"})
    with pytest.raises(ValueError, match="not materials"):
        lumen.load(lined, models={"materials/Nope": "mylab:water_hg"})

    @lumen.surface_model("test:leaky", params={"a": Param("", (0, 1), default=0.5)})
    def leaky(surface, base):
        def probs(p, ctx):
            return lumen.surfaces.outcome_vector(specular=p["a"] * jnp.ones_like(ctx["lam"]))  # sums to a, not 1

        return lumen.surfaces.SurfaceType("leaky", ("a",), (lumen.surfaces.SPECULAR,), probs)

    with pytest.raises(models.ModelError, match="summing to 1"):
        lumen.load(lined, models={"surfaces/Liner": "test:leaky"})
    with pytest.raises(ValueError, match="registered already"):
        lumen.surface_model("test:leaky")(lambda s, b: None)
    with pytest.raises(ValueError, match="lumen's own"):
        lumen.medium_model("geant4")(lambda m, b: b)
    with pytest.raises(TypeError, match="scalar"):
        lumen.medium_model("test:vec", params={"v": Param("", shape=(2,), default=(0, 0))})
    assert "test:leaky" in plugins.optics_models.names()


def test_a_lobe_is_checked_against_its_density():
    from lumen.extend import check_lobe
    from lumen.models import ModelError

    hg = lumen.media.henyey_greenstein("g")
    for g in (-0.5, 0.0, 0.9):
        check_lobe(hg, {"g": g})
    import dataclasses

    wrong = dataclasses.replace(hg, name="wrong", logpdf=lambda p, wi, n, wo: hg.logpdf({"g": 0.5}, wi, n, wo))
    with pytest.raises(ModelError, match="does not draw its density"):
        check_lobe(wrong, {"g": 0.9})


def test_lumen_s_own_models_are_registered_like_any_other(lined):
    """One dispatch: lumen's own models are registry entries, read by name like a user's (and cannot be replaced)."""
    from lumen import models

    assert {"geant4", "lumen:opaque", "lumen:same_as", "lumen:larsoft_lar"} <= set(models.names("medium"))
    assert {"geant4", "lumen:larsoft_reflector", "lumen:wls_film"} <= set(models.names("surface"))
    assert models.names("reader") == ["lumen:complex_index", "lumen:geant4"] and models.names("counting") == [
        "lumen:counting"
    ]
    det = lumen.load(lined)
    assert det.parts.library.surfaces["Liner"].surface_type.name == "specular_metal"
    with pytest.raises(ValueError, match="lumen's own"):
        lumen.surface_model("lumen:geant4")


def test_with_responses_adds_keys_and_refuses_a_taken_one(lined):
    det = lumen.load(lined)
    ex = _examples()
    d2 = det.with_responses(ex.saturating_pe, ex.my_waveform)
    assert d2.params()["sensors/pmt/adc_per_pe"] == 10.0 and "sensors/pmt/sat" in d2.params()
    assert d2.with_responses(ex.saturating_pe).params()["sensors/pmt/sat"] == 300.0  # the same response again
    taken = lumen.response(group="pmt", params={"gain": Param("ADC", default=1.0)})(lambda h, v: v["gain"])
    with pytest.raises(ValueError, match="keys the detector has already"):
        det.with_responses(taken)


def test_the_guide_shows_the_examples_as_they_are_tested():
    """Every worked example of docs/howto/extending.md is examples/extending.py's code, verbatim."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ex = (root / "examples" / "extending.py").read_text()
    guide = (root / "docs" / "howto" / "extending.md").read_text()
    names = re.findall(r"<!-- example: (\w+) -->", guide)
    assert set(names) == set(re.findall(r"^# example: (\w+)$", ex, re.M)) and len(names) >= 8
    for name in names:
        i = ex.index(f"# example: {name}\n") + len(f"# example: {name}\n")
        code = ex[i : ex.index("# end example", i)].rstrip()
        assert f"<!-- example: {name} -->\n```python\n{code}\n```" in guide, name


def test_check_source_compares_the_draws_with_a_density():
    import jax.numpy as jnp

    from lumen.extend import check_source

    led = _examples().cos_m_led(4.0)
    good = check_source(led, logpdf=lambda d: 4.0 * jnp.log(jnp.clip(-d[2], 1e-12)))
    assert good.passes, good
    bad = check_source(led, logpdf=lambda d: 2.0 * jnp.log(jnp.clip(-d[2], 1e-12)))
    assert not bad.passes and bad.dlogp_error > 0.1, bad


def test_every_module_the_guide_names_resolves_after_a_bare_import():
    """``lumen.X`` as the guide writes it (lumen.validate.check_gradient, lumen.extend, ...), in a fresh interpreter."""
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "howto" / "extending.md").read_text() + (root / "examples" / "extending.py").read_text()
    names = sorted({m for m in re.findall(r"\blumen\.([a-z_]+)\b", text) if m != "optics_models"})  # a group name
    assert {"validate", "extend", "sources", "media", "surfaces", "readout"} <= set(names)
    code = "import lumen\n" + "".join(f"getattr(lumen, {n!r})\n" for n in names)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=None)
    assert r.returncode == 0, r.stderr[-2000:]
