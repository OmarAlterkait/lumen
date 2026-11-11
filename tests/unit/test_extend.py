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
    with pytest.raises(sources.GradientNotAvailable, match="dlogp=jnp.zeros"):
        api._moving(arr, None, ("source/x",))
    with_g = sources.FromArrays(arr.x, arr.d, arr.t, arr.lam, arr.w, drawn=True, dlogp=jnp.zeros((n, 3)))
    assert api._moving(with_g, None, ("source/x",))[0]
    assert with_g.emit(None, jax.random.PRNGKey(0), n).dlogp.shape == (n, 3)
    assert (
        type(arr.emit(None, jax.random.PRNGKey(0), n)) is Emission
    )  # built-ins: a plain Emission, the start unchanged
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
