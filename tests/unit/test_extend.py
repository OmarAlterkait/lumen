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

