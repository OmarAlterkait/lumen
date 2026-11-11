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
