"""The extension API: declared parameters, source functions, responses, registered models, checks.

The derivative checks (AD against FD on the shipped toys) are the physics tier's ``test_extend.py``; here each piece's
contract: declarations become keys, refusals refuse, built-ins keep their bits.
"""

import numpy as np
import pytest

import lumen
from lumen.params import Param, Params, declared


# ------------------------------------------------------------------------------------------------ declared keys
def test_a_declaration_is_checked_and_hashable():
    p = Param("m", shape=(3,), method="moving_paths", default=(0.0, 0.0, 0.2))
    assert (
        p.default == (0.0, 0.0, 0.2)
        and p.shape == (3,)
        and hash(p) == hash(Param("m", shape=[3], method="moving_paths", default=[0, 0, 0.2]))
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
    v, s = declared("source/f/", {"x": Param("m", shape=(3,), method="moving_paths", default=(1, 2, 3))})
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
    from lumen.tools import check_gradient

    good = check_gradient(_noisy(), 1.5, seeds=8)
    assert good.passes, good
    assert good.ad.shape == (7,) and abs(good.ad[-1] - 21 * (3.0 + 0.1 * 0)) < 3.0
    assert "passes" in str(good)
    bad = check_gradient(_noisy(bias=0.3), 1.5, seeds=8, stencil="central")
    assert not bad.passes and abs(bad.z_total) > 3.0, bad
    with pytest.raises(ValueError, match="stencil"):
        check_gradient(_noisy(), 1.5, stencil="forward")


# ------------------------------------------------------------------------------------------------ source functions
def _led(m_default=4.0):
    import math

    import jax
    import jax.numpy as jnp

    from lumen.sources import cone_about, reweight

    @lumen.source(params={"x": Param("m", shape=(3,), method="moving_paths", default=(0.0, 0.0, 0.2)),
                          "m": Param("", (0, 50), default=m_default)})  # fmt: skip
    def led(v, key, n):
        a = jnp.asarray([0.0, 0.0, -1.0])
        u = jax.random.uniform(key, (n, 2), minval=1e-7, maxval=1.0)
        d = cone_about(a, u[:, 0] ** (1 / (jax.lax.stop_gradient(v["m"]) + 1)), 2 * math.pi * u[:, 1])

        def logp(d_):
            return v["m"] * jnp.log(jnp.clip(d_ @ a, 1e-12)) + jnp.log(v["m"] + 1)

        w = reweight(jax.vmap(logp)(d)) * 1e3 / n
        return lumen.Photons(x=jnp.broadcast_to(v["x"], (n, 3)), d=d, t=jnp.zeros(n), lam=jnp.full(n, 420.0), w=w)

    return led


def test_a_source_function_is_a_source_with_keys():
    import jax

    led = _led()
    assert led.prefix == "source/led/" and led.MOVING == ("x",)
    p = led.parameters()
    assert set(p) == {"source/led/x", "source/led/m"} and p.info("source/led/x").method == "moving_paths"
    led.check(None)
    e = led.emit(None, jax.random.PRNGKey(1), 64, 0, 256)
    assert e.x.shape == (64, 3)
    np.testing.assert_allclose(float(e.w.sum()), 1e3 * 64 / 256, rtol=1e-6)  # the chunk's share of the photons
    e2 = led.emit(None, jax.random.PRNGKey(1), 64, 64, 256)
    assert not np.array_equal(np.asarray(e.d), np.asarray(e2.d))  # each chunk its own photons
    whole = led.emit(None, jax.random.PRNGKey(1), 256, 0, 256)
    for a, b in zip(whole, e2, strict=True):  # photon i is the same whatever the chunk
        assert (a is None and b is None) or np.array_equal(np.asarray(a)[64:128], np.asarray(b))


def test_a_source_function_is_called_once_per_run(monkeypatch):
    """simulate reads a run's photons through ``emitter``: a source function draws all of them once, not per chunk."""
    import jax

    from lumen.sources import function

    led, calls = _led(), []
    real = function.FunctionSource.emit
    monkeypatch.setattr(function.FunctionSource, "emit", lambda self, *a, **k: calls.append(a) or real(self, *a, **k))
    rows = led.emitter(None, jax.random.PRNGKey(1), 256)
    chunks = [rows(f, 64) for f in range(0, 256, 64)]
    whole = real(led, None, jax.random.PRNGKey(1), 256, 0, 256)
    assert len(calls) == 1
    np.testing.assert_array_equal(np.concatenate([np.asarray(c.x) for c in chunks]), np.asarray(whole.x))


def test_a_source_function_refuses_what_it_cannot_differentiate():
    import jax
    import jax.numpy as jnp

    from lumen import sources
    from lumen.record import Photons

    led = _led()
    p = led.parameters()

    def x_of(dx):
        q = p.set("source/led/x", jnp.asarray([0.0, 0.0, 0.2]) + dx * jnp.asarray([1.0, 0.0, 0.0]))
        return led.emit(q, jax.random.PRNGKey(0), 16, 0, 16).x

    with pytest.raises(sources.GradientNotAvailable, match="source/led/x|x moves where the light starts"):
        jax.jvp(x_of, (0.0,), (1.0,))  # a moving parameter's tangent: no source-position derivative
    n = 8
    arr = sources.FromArrays(jnp.zeros((n, 3)), jnp.tile(jnp.asarray([0.0, 0.0, 1.0]), (n, 1)), jnp.zeros(n),
                             jnp.full(n, 400.0), jnp.ones(n))  # fmt: skip

    def pos_of(dz):
        a = sources.FromArrays(arr.position + dz * jnp.asarray([0.0, 0.0, 1.0]), arr.direction, arr.time, arr.lam,
                               arr.weight)  # fmt: skip
        return a.emit(None, jax.random.PRNGKey(0), n).x

    with pytest.raises(sources.GradientNotAvailable, match="moves where the light starts"):
        jax.jvp(pos_of, (0.0,), (1.0,))  # rows moved: refused
    e = arr.emit(None, jax.random.PRNGKey(0), n)
    assert len(e) == len(Photons._fields)


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
    assert c.x.shape == (4104, 3)
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
    nodes, late, early = rd.detection.masses(h, rd.detection.values(h, p))
    tot = np.asarray(nodes.sum(-1) + late[:, 0] + early[:, 0])
    np.testing.assert_allclose(tot, pe, rtol=1e-5)  # the time nodes carry pe
    np.testing.assert_allclose(
        np.asarray(rd.pe_density(h, p, dark=False).rate).sum(-1), pe, rtol=1e-4
    )  # the profile: all in the gate
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


def test_a_medium_model_passed_as_a_value_builds_with_its_keys(lined):
    import jax
    import jax.numpy as jnp

    ex = _examples()
    plain = lumen.load(lined)
    det = lumen.load(lined, parts={"materials/Water": ex.water_hg})
    p = det.params()
    assert p["materials/Water/g"] == 0.9 and p.explain("materials/Water/g") == "model mylab:water_hg default"
    assert "materials/Water/g" not in plain.params()
    assert set(p) - set(plain.params()) == {"materials/Water/g"}
    lib = det.parts.library
    ray = next(c for c in lib.media["Water"].medium.channels if c.name == "rayleigh")
    assert ray.lobe is lumen.media.henyey_greenstein("g")
    assert lib.media["Water"].params["rayleigh"]["lobe"] == {"g": 0.9}
    src = det.parts.key_rows
    assert src[("media", "Water", "rayleigh", "lobe", "g")].key == "materials/Water/g"
    q = p.set("materials/Water/g", 0.5)
    named = det.pack(q)  # a value set: applied to the file (the declared property added) and packed
    assert named is not None
    t = jax.jvp(lambda g: det.pack(p.set("materials/Water/g", g)), (jnp.float32(0.9),), (jnp.float32(1.0),))[1]
    assert any(float(jnp.abs(x).max()) == 1.0 for x in jax.tree_util.tree_leaves(t))  # g reaches the packed rows
    assert ex.water_hg.name == "mylab:water_hg"


def test_the_model_refusals(lined):
    """A surface is one of lumen's surface types (no surface model of your own); a medium model is a value or one of
    lumen's own names; nothing is registered: a name the build's table does not hold is refused."""
    import dataclasses

    from lumen.build.library import LibraryError

    ex = _examples()
    assert not hasattr(lumen, "surface_model") and not hasattr(lumen, "plugins")
    with pytest.raises(ValueError, match="not materials"):
        lumen.load(lined, parts={"surfaces/Liner": ex.water_hg})
    with pytest.raises(LibraryError, match="neither a medium model"):
        lumen.load(lined, parts={"materials/Water": "mylab:water_hg"})  # a name alone: no registry to find it in
    with pytest.raises(ValueError, match="not materials"):
        lumen.load(lined, parts={"materials/Nope": ex.water_hg})
    from lumen.build import build

    named = lumen.load(lined, parts={"materials/Water": ex.water_hg}).file
    with pytest.raises(LibraryError, match="not one this build reads as a medium"):
        build.build(named, lumen.load(lined).sensor_set)  # the file names it, the build was not given it
    water = named.materials["Water"]  # a file naming importable code ("module:attr"): a name, never imported
    code = dataclasses.replace(named, materials={**named.materials, "Water": dataclasses.replace(
        water, optics=dataclasses.replace(water.optics, name="json:dumps"))})  # fmt: skip
    with pytest.raises(LibraryError, match="not one this build reads as a medium"):
        build.build(code, lumen.load(lined).sensor_set)

    @lumen.medium_model("test:twice")
    def twice(material, base):
        return base

    other = lumen.medium_model("test:twice")(lambda m, b: None)
    from lumen.models import model_table

    with pytest.raises(ValueError, match="two models of one name"):
        model_table([twice, other])
    assert model_table([twice, twice]) == {"test:twice": twice}
    with pytest.raises(ValueError, match="lumen's own"):
        lumen.medium_model("geant4")(lambda m, b: b)
    with pytest.raises(TypeError, match="scalar"):
        lumen.medium_model("test:vec", params={"v": Param("", shape=(2,), default=(0, 0))})
    assert dataclasses.is_dataclass(twice) and twice.name == "test:twice"


def test_a_lobe_is_checked_against_its_density():
    from lumen._core.errors import ModelError
    from lumen.tools.extend import check_lobe

    hg = lumen.media.henyey_greenstein("g")
    for g in (-0.5, 0.0, 0.9):
        check_lobe(hg, {"g": g})
    import dataclasses

    wrong = dataclasses.replace(hg, name="wrong", logpdf=lambda p, wi, n, wo: hg.logpdf({"g": 0.5}, wi, n, wo))
    with pytest.raises(ModelError, match="does not draw its density"):
        check_lobe(wrong, {"g": 0.9})


def test_lumen_s_own_models_are_the_build_s_table(lined):
    """lumen's own models are the build's table by name (no registry), and a model of your own cannot take a name of
    theirs."""
    from lumen.build import library

    assert set(library.MEDIA) == {"geant4", "lumen:opaque"}
    det = lumen.load(lined)
    from lumen.surfaces import types

    assert det.parts.library.surfaces["Liner"].surface_type == types.no_detection(types.SPECULAR_METAL)
    with pytest.raises(ValueError, match="lumen's own"):
        lumen.medium_model("lumen:geant4")


def test_with_responses_adds_keys_and_refuses_a_taken_one(lined):
    det = lumen.load(lined)
    ex = _examples()
    d2 = det.with_responses(ex.saturating_pe, ex.my_waveform)
    assert d2.params()["sensors/pmt/adc_per_pe"] == 10.0 and "sensors/pmt/sat" in d2.params()
    assert d2.with_responses(ex.saturating_pe).params()["sensors/pmt/sat"] == 300.0  # the same response again
    taken = lumen.response(group="pmt", params={"gain": Param("ADC", default=1.0)})(lambda h, v: v["gain"])
    with pytest.raises(ValueError, match="keys the detector has already"):
        det.with_responses(taken)


def test_a_calibration_of_a_response_loads_with_the_response(lined, tmp_path, monkeypatch):
    """A calibration saved after with_responses holds the response's keys: it loads with load(responses=), and without
    them it names the unknown key (UnknownParamKey, the nearest keys), not a bare KeyError."""
    from lumen.params import UnknownParamKey

    ex = _examples()
    d2 = lumen.load(lined).with_responses(ex.saturating_pe)
    d2.save_calibration(d2.params().set("sensors/pmt/sat", 250.0), "satcal", tmp_path)
    monkeypatch.setenv("LUMEN_DETECTORS", str(tmp_path))
    back = lumen.load(lined, calibration="satcal", responses=(ex.saturating_pe,))
    assert back.params()["sensors/pmt/sat"] == 250.0
    with pytest.raises(UnknownParamKey, match="sensors/pmt/sat"):
        lumen.load(lined, calibration="satcal")


def test_the_guide_shows_the_examples_as_they_are_tested():
    """Every worked example of docs/howto/extending.md is examples/extending.py's code, verbatim."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ex = (root / "examples" / "extending.py").read_text()
    guide = (root / "docs" / "howto" / "extending.md").read_text()
    names = re.findall(r"<!-- example: (\w+) -->", guide)
    assert set(names) == set(re.findall(r"^# example: (\w+)$", ex, re.M)) and len(names) >= 5
    for name in names:
        i = ex.index(f"# example: {name}\n") + len(f"# example: {name}\n")
        code = ex[i : ex.index("# end example", i)].rstrip()
        assert f"<!-- example: {name} -->\n```python\n{code}\n```" in guide, name


def test_check_source_compares_the_draws_with_a_density():
    import jax.numpy as jnp

    from lumen.tools.extend import check_source

    led = _examples().cos_m_led(4.0)
    good = check_source(led, logpdf=lambda d: 4.0 * jnp.log(jnp.clip(-d[2], 1e-12)))
    assert good.passes, good
    bad = check_source(led, logpdf=lambda d: 2.0 * jnp.log(jnp.clip(-d[2], 1e-12)))
    assert not bad.passes, bad


def test_every_module_the_guide_names_resolves_after_a_bare_import():
    """``lumen.X`` as the guide writes it (lumen.tools.check_gradient, lumen.tools.extend, ...), in a new process."""
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "howto" / "extending.md").read_text() + (root / "examples" / "extending.py").read_text()
    names = sorted({m for m in re.findall(r"\blumen\.([a-z_]+)\b", text) if m != "optics_models"})  # a group name
    assert {"tools", "sources", "media", "surfaces", "readout"} <= set(names)
    code = "import lumen\n" + "".join(f"getattr(lumen, {n!r})\n" for n in names)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=None)
    assert r.returncode == 0, r.stderr[-2000:]
