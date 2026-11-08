"""Bulk media: evaluation at a wavelength (index, group index from the same law, absorption, channel rates) and
fail-closed validation."""

import jax.numpy as jnp
import numpy as np
import pytest

from lumen.media import bulk, laws, phase

F32 = jnp.float32


def water():
    ch = (
        bulk.Channel("rayleigh", phase.RAYLEIGH, laws.power("length")),
        bulk.Channel("other", phase.ISOTROPIC, laws.constant("length")),
    )
    m = bulk.Medium("water", laws.cauchy(), laws.constant("length"), ch)
    p = {
        "index": {"A": F32(1.32), "B": F32(0.004), "C": F32(0.0)},
        "absorption": {"value": F32(60.0)},
        "rayleigh": {"length": {"v0": F32(30.0), "lam0": F32(400.0), "k": F32(4.0)}},
        "other": {"length": {"value": F32(100.0)}, "lobe": {"g": F32(0.9)}},
    }
    return m, p


def test_medium_evaluates_every_law_at_the_photon_wavelength():
    m, p = water()
    e = bulk.evaluate(m, p, F32(350.0))
    l2 = 0.35**2
    assert abs(float(e["n"]) - (1.32 + 0.004 / l2)) < 1e-6
    assert abs(float(e["n_g"]) - (1.32 + 3 * 0.004 / l2)) < 1e-5  # n - lam dn/dlam for B / lam^2
    assert abs(float(e["mu_abs"]) - 1 / 60.0) < 1e-8
    np.testing.assert_allclose(np.asarray(e["mu"]), [1 / (30.0 * (350 / 400) ** 4), 1 / 100.0], rtol=1e-6)
    assert e["lobe_params"][1]["g"] == p["other"]["lobe"]["g"]


def test_validation_fails_closed():
    m, p = water()
    bulk.validate(m, p, 300.0, 600.0)
    with pytest.raises(laws.LawError):
        bulk.validate(m, dict(p, absorption={"value": F32(-1.0)}), 300.0, 600.0)
    m2 = bulk.Medium("x", None, None, (bulk.Channel("c", phase.ISOTROPIC, laws.constant("fraction")),))
    with pytest.raises(laws.LawError):
        bulk.validate(m2, {"c": {"length": {"value": F32(0.5)}}}, 300.0, 600.0)
    assert bulk.Medium("steel", None).opaque
