"""The navigator's precision (``lumen.load(..., precision=)``): float64 tracks volumes thinner than float32's band, for
every photon, whatever its path.

What is checked:

- ``lumen.load`` refuses precision="float64" while JAX's 64-bit mode is off, and a precision it does not know;
- a float64 navigator holds float64 arrays and CSG tables, and its band is its 1 nm floor at metres;
- in float64, a coating 2.5 float32 bands thick at 0.5, 1.4 and 5 m from the origin, under a gas volume whose bottom
  face it shares: a photon whose ray started 1 mm or 5 m away, at 0, 45 and 75 deg, enters it, crosses to its far face
  T / cos deep and, reflected there, leaves into the gas T / cos later;
- in float64, a photon re-emitted in that coating at 10 % and 50 % of its depth, at 3, 15 and 60 deg to its face,
  leaves into the gas heading out and into the backing heading in, at its exact distance;
- in float64, a foil 1.5 float32 bands thick standing in the gas is entered by a ray from 5 m away, then crossed;
- through simulate, a skin sensor detects in float64 as in float32, in a medium with Rayleigh scattering too (the
  diffusion jump), and its derivatives are float32's;
- simulate refuses a float64 detector while the 64-bit mode is off; a build cache never serves one precision's
  navigator to the other; siblings that overlap by less than float32's contact tolerance are ordered in float64; a
  step's position is the navigator's own crossing, not its float32 rounding; the jump's and the sensor reach's margins,
  whose arrays are float32, are float32's in either precision.

In float32 the coating is skipped by rays from 5 m away and the re-emissions leave into the wrong volume
(docs/explanation/known-limits.md, "Thin layers").
"""

import math

import numpy as np
import pytest
from jax.experimental import enable_x64
from nav_toys import defn, locate, make_scene, nav_of, query
from test_skin_sensors import SKIN, SLAB

import lumen
from lumen.geometry import solids
from lumen.rules import navigation

F32 = np.float32
ZS = (0.5, 1.4, 5.0)  # m: the coating's height


def band32(scale):
    """float32's band (m) at a coordinate scale (m)."""
    return float(navigation.band(scale, navigation.FLOAT32))


@pytest.fixture
def x64():
    """JAX's 64-bit mode for the test (float64 arrays exist only in it)."""
    with enable_x64():
        yield


def _coating(z0, t, h=8.0):
    """A gas volume (19 m wide, h tall) on a backing block whose top face carries a coating of thickness t (its
    daughter, top faces coincident), the shared face at height z0."""
    backing = defn("backing", solids.g4_box(0.8, 0.8, 0.1), 3, [("film", np.eye(3), (0.0, 0.0, 0.05 - t / 2))])
    return make_scene({"world": defn("world", solids.g4_box(40.0, 40.0, 40.0), 1, [
        ("backing", np.eye(3), (0.0, 0.0, z0 - 0.05)), ("gas", np.eye(3), (0.0, 0.0, z0 + h / 2))]),
        "backing": backing, "film": defn("film", solids.g4_box(0.8, 0.8, t), 4),
        "gas": defn("gas", solids.g4_box(19.0, 19.0, h), 2)})  # fmt: skip


def _rays(z0, path, theta_deg, n=8, seed=0):
    """Return (starts, directions): rays landing on the face at height z0 at theta from its normal, from ``path``
    away."""
    rng = np.random.default_rng(seed)
    th = math.radians(theta_deg)
    land = np.c_[rng.uniform(-0.25, 0.25, n), rng.uniform(-0.25, 0.25, n), np.full(n, z0)]
    phi = rng.uniform(0.0, 2 * math.pi, n)
    d = np.c_[math.sin(th) * np.cos(phi), math.sin(th) * np.sin(phi), np.full(n, -math.cos(th))]
    return land - path * d, d


def _faces(nav, name, t):
    """Return (top, bottom) of a box of thickness t as the navigator holds it (its placement, z)."""
    tz = float(np.asarray(nav.host["t"])[nav.defs.index(name)][2])
    return tz + t / 2, tz - t / 2


@pytest.mark.parametrize("z0", ZS)
def test_a_coating_is_entered_crossed_and_left_whatever_the_path(x64, z0):
    t = 2.5 * band32(z0)
    nav = nav_of(_coating(z0, t), "float64")
    film, backing, gas = (nav.defs.index(s) for s in ("film", "backing", "gas"))
    top, bot = _faces(nav, "film", t)
    for path in (1e-3, 5.0):
        for theta in (0.0, 45.0, 75.0):
            x0, d = _rays(z0, path, theta)
            c = math.cos(math.radians(theta))
            r1 = query(nav, x0, d, locate(nav, x0))
            assert np.all(r1["vol_after"] == film), (path, theta, r1["vol_after"])
            x1 = x0 + r1["t"][:, None] * d
            r2 = query(nav, x1, d, r1["vol_after"], last=r1["srf"], sn=r1["normal"])
            assert np.all(r2["vol_after"] == backing), (path, theta, r2["vol_after"])
            np.testing.assert_allclose(r2["t"], (x1[:, 2] - bot) / c, rtol=1e-6, err_msg=str((path, theta)))
            x2 = x1 + r2["t"][:, None] * d
            dr = d * np.array([1.0, 1.0, -1.0])  # reflected at the far face
            r3 = query(nav, x2, dr, r1["vol_after"], last=r2["srf"], sn=r2["normal"])
            assert np.all(r3["vol_after"] == gas), (path, theta, r3["vol_after"])
            np.testing.assert_allclose(r3["t"], (top - x2[:, 2]) / c, rtol=1e-6, err_msg=str((path, theta)))


@pytest.mark.parametrize("z0", ZS)
def test_a_reemission_in_a_coating_leaves_where_it_heads(x64, z0):
    t = 2.5 * band32(z0)
    nav = nav_of(_coating(z0, t), "float64")
    film, backing, gas = (nav.defs.index(s) for s in ("film", "backing", "gas"))
    top, bot = _faces(nav, "film", t)
    rng = np.random.default_rng(3)
    n = 8
    for depth in (0.1, 0.5):
        xb = np.c_[rng.uniform(-0.25, 0.25, n), rng.uniform(-0.25, 0.25, n), np.full(n, top - depth * t)]
        for elev in (3.0, 15.0, 60.0):
            e, phi = math.radians(elev), rng.uniform(0.0, 2 * math.pi, n)
            for sign, want in ((1.0, gas), (-1.0, backing)):
                d = np.c_[math.cos(e) * np.cos(phi), math.cos(e) * np.sin(phi), np.full(n, sign * math.sin(e))]
                r = query(nav, xb, d, np.full(n, film), last=-1)
                assert np.all(r["vol_after"] == want), (depth, elev, sign, r["vol_after"])
                dist = ((top - xb[:, 2]) if sign > 0 else (xb[:, 2] - bot)) / math.sin(e)
                np.testing.assert_allclose(r["t"], dist, rtol=1e-6, err_msg=str((depth, elev, sign)))


def test_a_foil_in_the_gas_is_entered_from_far_away(x64):
    for z0 in ZS:
        t = 1.5 * band32(z0)
        gas_ = defn("gas", solids.g4_box(19.0, 19.0, 8.0), 2, [("foil", np.eye(3), (0.0, 0.0, -3.5 - t / 2))])
        sc = make_scene({"world": defn("world", solids.g4_box(40.0, 40.0, 40.0), 1,
                                       [("gas", np.eye(3), (0.0, 0.0, z0 + 3.5))]),
                         "gas": gas_, "foil": defn("foil", solids.g4_box(0.8, 0.8, t), 4)})  # fmt: skip
        nav = nav_of(sc, "float64")
        foil, gas = nav.defs.index("foil"), nav.defs.index("gas")
        top, bot = _faces(nav, "foil", t)
        for theta in (0.0, 45.0, 75.0):
            x0, d = _rays(top, 5.0, theta)
            r1 = query(nav, x0, d, locate(nav, x0))
            assert np.all(r1["vol_after"] == foil), (z0, theta, r1["vol_after"])
            x1 = x0 + r1["t"][:, None] * d
            r2 = query(nav, x1, d, r1["vol_after"], last=r1["srf"], sn=r1["normal"])
            assert np.all(r2["vol_after"] == gas), (z0, theta, r2["vol_after"])
            np.testing.assert_allclose(r2["t"], (x1[:, 2] - bot) / math.cos(math.radians(theta)), rtol=1e-6)


def test_a_float64_navigator_holds_float64_arrays_and_its_band_is_its_floor(x64):
    nav = nav_of(_coating(1.4, 3.5e-6), "float64")
    assert nav.prec is navigation.FLOAT64
    assert nav.host["t"].dtype == np.float64 and nav.host["R"].dtype == np.float64
    assert nav.E.tables.T["c"].dtype == np.float64 and nav.E.tables.T["h"].dtype == np.float64
    assert abs(float(navigation.band(1.4, nav.prec)) - navigation.BAND_FLOOR) < 1e-14


def _converted(tmp_path, slab=SLAB):
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe

    (tmp_path / "slab.gdml").write_text(slab.format(skin=SKIN))
    recipe = Recipe("slab", "GasLV", groups=(SensorGroup("cath", rule="border_into", lv="CathLV"),))
    return lumen.convert(tmp_path / "slab.gdml", tmp_path / "slab.lumen", recipe=recipe)


def test_load_refuses_float64_without_64_bit_mode_and_an_unknown_precision(tmp_path):
    path = _converted(tmp_path)
    with pytest.raises(ValueError, match="64-bit mode"):
        lumen.load(path, precision="float64")
    with pytest.raises(ValueError, match="is not one of"):
        lumen.load(path, precision="float16")
    assert lumen.load(path).precision == "float32"


@pytest.mark.slow  # compiles the photon loop twice: about two minutes on 4 CPUs
def test_through_simulate_a_float64_detector_detects_as_a_float32_one(tmp_path):
    from lumen.sources import FromArrays

    path = _converted(tmp_path)
    n, th = 4096, math.radians(30.0)
    src = FromArrays(np.tile([0.0, 0.0, 0.0], (n, 1)), np.tile([math.sin(th), 0.0, -math.cos(th)], (n, 1)),
                     np.zeros(n), np.full(n, 400.0), np.full(n, 1.0 / n))  # fmt: skip
    a32 = float(lumen.simulate(lumen.load(path), src, seed=1).arrivals.sum())
    with enable_x64():
        det = lumen.load(path, precision="float64")
        assert det.precision == "float64"
        a64 = float(lumen.simulate(det, src, seed=1).arrivals.sum())
    assert abs(a32 - 0.4) < 1e-5 and abs(a64 - 0.4) < 1e-5, (a32, a64)


def test_simulate_refuses_a_float64_detector_without_64_bit_mode(tmp_path):
    from lumen.sources import Point

    path = _converted(tmp_path)
    with enable_x64():
        det = lumen.load(path, precision="float64")
    with pytest.raises(ValueError, match="simulate: .* 64-bit mode"):
        lumen.simulate(det, Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0), samples=16, seed=1)


def test_the_build_cache_keeps_the_precisions_apart(x64, tmp_path, monkeypatch):
    from lumen.geometry import cache, navigate

    monkeypatch.setenv("LUMEN_CACHE", str(tmp_path))
    sc = _coating(1.4, 3.5e-6)
    keys = {}
    for p in ("float32", "float64", "float32", "float64"):
        arrays, k = cache.nav_arrays(sc, precision=p)
        assert arrays.host["t"].dtype == np.dtype(p) and arrays.precision == p
        assert navigate.Navigator(arrays).prec.name == p and keys.setdefault(p, k) == k
    assert keys["float32"] != keys["float64"] and keys["float32"] == cache.nav_arrays(sc)[1]


def test_siblings_overlapping_below_float32_contact_are_ordered_in_float64(x64):
    """a (placed first) over b, 5 um deep at 5 m (float32's contact tolerance there: 19 um): the overlap is a's."""
    z, dep = 5.0, 5e-6
    sc = make_scene({"world": defn("world", solids.g4_box(20.0, 20.0, 20.0), 1, [
        ("a", np.eye(3), (0.0, 0.0, z + 0.5)), ("b", np.eye(3), (0.0, 0.0, z - 0.5 + dep))]),
        "a": defn("a", solids.g4_box(1.0, 1.0, 1.0), 2), "b": defn("b", solids.g4_box(1.0, 1.0, 1.0), 3)})  # fmt: skip
    nav = nav_of(sc, "float64")
    x, d = np.array([[0.1, 0.1, z - 0.25]]), np.array([[0.0, 0.0, 1.0]])
    r = query(nav, x, d, locate(nav, x))
    assert r["vol_after"][0] == nav.defs.index("a")
    np.testing.assert_allclose(r["t"], 0.25, atol=1e-8)  # a's bottom face, not b's top (5 um further)


def test_margins_of_float32_arrays_are_float32s_in_either_precision(x64):
    from lumen.geometry import clearance, reach

    sc = _coating(5.0, 2e-5)
    n32, n64 = nav_of(sc), nav_of(sc, "float64")
    none = np.full(n32.n + 1, -1)
    assert reach.build(n64, none, None)[1]["pad"] == reach.build(n32, none, None)[1]["pad"]
    g = clearance.grid(n32, [0], 4.0)
    b = [clearance.bound(n, g["lo"], 4.0, g["n"], 3.0) for n in (n32, n64)]
    np.testing.assert_allclose(b[0], b[1], atol=2e-6)  # the float32 leaf coefficients' rounding
    scale = float(np.max(np.abs(np.r_[g["lo"], g["lo"] + 4.0 * (np.array(g["n"]) - 1)]))) + 3.0
    assert float(np.max(b[1])) <= 3.0 - band32(scale) + 1e-9


@pytest.mark.slow  # compiles one step: half a minute
def test_a_step_moves_to_the_navigators_own_crossing(x64, tmp_path):
    import jax
    import jax.numpy as jnp

    from lumen import api
    from lumen.build import pack
    from lumen.record import Photons
    from lumen.transport import item
    from lumen.transport import step as steps

    det = lumen.load(_converted(tmp_path), precision="float64")
    model = api._model(det)
    ops = steps.operands(model, api._to_device(pack.pack(det, det.params())), None, steps.Controls())
    th = 0.5236
    row = Photons(x=jnp.array([1e-3, 2e-3, 1e-3], F32), d=jnp.array([math.sin(th), 0.0, -math.cos(th)], F32),
                  t=F32(0.0), lam=F32(400.0), w=F32(1.0), pol=jnp.zeros((2, 3), F32))  # fmt: skip
    ph = item.start_photon(model, ops, row)
    assert ph.x.dtype == jnp.float64
    nav = model.geometry.nav
    t = jax.jit(lambda: nav.query(ph.x, ph.d, ph.vol, t_max=10.0, vertex_tab=model.geometry.vertex_tab)["t"])()
    assert float(t) != float(F32(t))
    ph2, _ = jax.jit(lambda p: steps.step(model, ops, p, jax.random.PRNGKey(1)))(ph)
    np.testing.assert_array_equal(np.asarray(ph2.x), np.asarray(ph.x + t * ph.d.astype(jnp.float64)))


@pytest.mark.slow  # compiles the photon loop four times
def test_through_simulate_a_scattering_medium_at_speed_fast(tmp_path):
    """The diffusion walk (a Rayleigh-only medium) in float64, its light and derivative against float32's."""
    import jax

    from lumen.sources import Point

    gas = ('<property name="RINDEX" ref="R1"/>',
           '<property name="RINDEX" ref="R1"/> <property name="RAYLEIGH" ref="RL"/> '
           '<property name="ABSLENGTH" ref="AB"/>')  # fmt: skip
    tab = '<matrix name="RL" coldim="2" values="2.0*eV 1.0 4.5*eV 1.0"/>' \
          '<matrix name="AB" coldim="2" values="2.0*eV 200.0 4.5*eV 200.0"/></define>'  # fmt: skip
    path = _converted(tmp_path, SLAB.replace(*gas).replace("</define>", tab))

    out = {}
    with enable_x64():
        for p in ("float32", "float64"):
            det = lumen.load(path, precision=p)
            p0, src = det.params(), Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0)
            f = lambda t, det=det, p0=p0: lumen.simulate(det, src, p0.scale("materials/Gas/ABSLENGTH", 1.0 + t),  # noqa: E731
                                                          samples=1 << 14, seed=3).arrivals.sum()  # fmt: skip
            out[p] = (float(f(0.0)), float(jax.jvp(f, (0.0,), (1.0,))[1]), float(jax.grad(f)(0.0)))
    a, b = out["float32"], out["float64"]
    assert b[1] == pytest.approx(b[2], rel=1e-4)
    np.testing.assert_allclose(a, b, rtol=0.02)


def test_a_recipe_states_the_precision_its_detector_needs(tmp_path, x64):
    """A recipe's ``precision`` is stored in the detector file, and ``lumen.load`` uses it when the caller passes
    none; an explicit precision wins over it, and float32 on a float64 detector warns once per call site (Python's
    own warning filter, no cache of lumen's own)."""
    import warnings

    from lumen import api
    from lumen.format import io
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe

    (tmp_path / "slab.gdml").write_text(SLAB.format(skin=SKIN))
    recipe = Recipe("slab64", "GasLV", groups=(SensorGroup("cath", rule="border_into", lv="CathLV"),),
                    precision="float64")  # fmt: skip
    path = lumen.convert(tmp_path / "slab.gdml", tmp_path / "slab64.lumen", recipe=recipe)
    assert io.read(path).precision == "float64"
    assert lumen.load(path).precision == "float64"  # the file's, no argument given
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("default")
        for _ in range(2):  # the argument wins, and the warning is said once from one place
            assert lumen.load(path, precision="float32").precision == "float32"
    only = [w for w in said if issubclass(w.category, api.PrecisionWarning)]
    assert len(only) == 1 and "loaded as 'float32'" in str(only[0].message), [str(w.message) for w in said]
