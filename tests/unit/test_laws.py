"""Spectral laws: values against float64 references, domains and ranges fail closed, the group index from the same
law, Geant4's RINDEX and GROUPVEL, and no recompilation when a value changes."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen.media import laws
from lumen.units import HC

F32 = jnp.float32
LAM = np.linspace(200.0, 700.0, 501)


def ev(law, p, lam=LAM):
    return np.asarray(law({k: jnp.asarray(v, F32) for k, v in p.items()}, jnp.asarray(lam, F32)), np.float64)


def test_tabulated_is_linear_in_energy_and_fails_outside_its_knots():
    e = np.array([1.5, 2.0, 3.0, 4.0, 6.5])
    v = np.array([10.0, 20.0, 15.0, 5.0, 1.0])
    law = laws.tabulated("length", e[0], e[-1])
    np.testing.assert_allclose(ev(law, {"energy": e, "value": v}), np.interp(HC / LAM, e, v), rtol=2e-6)
    assert abs(ev(law, {"energy": e, "value": v}, np.array([HC / 2.5]))[0] - 17.5) < 1e-4
    with pytest.raises(laws.LawError):
        laws.validate(law, {"energy": e, "value": v}, 150.0, 600.0)  # 150 nm = 8.3 eV > 6.5 eV
    laws.validate(laws.tabulated("length", e[0], e[-1], outside="clamp"), {"energy": e, "value": v}, 150.0, 900.0)


def test_sellmeier_and_cauchy_against_float64():
    p = dict(B1=0.6961663, B2=0.4079426, B3=0.8974794, C1=0.0684043**2, C2=0.1162414**2, C3=9.896161**2)
    l2 = (LAM * 1e-3) ** 2
    ref = np.sqrt(1 + sum(p[f"B{i}"] * l2 / (l2 - p[f"C{i}"]) for i in (1, 2, 3)))
    np.testing.assert_allclose(ev(laws.sellmeier(), p), ref, rtol=1e-6)
    assert abs(ev(laws.sellmeier(), p, np.array([587.6]))[0] - 1.4585) < 2e-4  # fused silica (Malitson 1965)
    pc = dict(A=1.32, B=0.004, C=1e-5)
    np.testing.assert_allclose(ev(laws.cauchy(), pc), 1.32 + 0.004 / l2 + 1e-5 / l2**2, rtol=1e-6)


def test_group_index_is_n_minus_lambda_dn_dlambda_of_the_same_law():
    p = dict(A=1.32, B=0.004, C=1e-5)
    l2 = (LAM * 1e-3) ** 2
    n = 1.32 + 0.004 / l2 + 1e-5 / l2**2
    dn = (-2 * 0.004 / (l2 * LAM * 1e-3) - 4 * 1e-5 / (l2**2 * LAM * 1e-3)) * 1e-3  # per nm
    np.testing.assert_allclose(ev(laws.group_index(laws.cauchy()), p), n - LAM * dn, rtol=1e-5)
    with pytest.raises(ValueError):
        laws.group_index(laws.constant("length"))


def test_power_law_and_ranges_fail_closed():
    np.testing.assert_allclose(
        ev(laws.power("length"), dict(v0=30.0, lam0=400.0, k=4.0)), 30.0 * (LAM / 400.0) ** 4, rtol=1e-5
    )
    for law, bad in (
        (laws.constant("index"), {"value": 0.9}),
        (laws.constant("fraction"), {"value": 1.2}),
        (laws.constant("length"), {"value": 0.0}),
        (laws.constant("length"), {"value": np.nan}),
    ):
        with pytest.raises(laws.LawError):
            laws.validate(law, bad, 200.0, 700.0)
    with pytest.raises(laws.LawError):
        laws.validate(laws.cauchy(), {"A": 1.3, "B": 0.0}, 200.0, 700.0)  # missing parameter


def test_changing_a_value_does_not_recompile_and_gradients_flow():
    law = laws.power("length")
    f = jax.jit(lambda p, lam: jnp.sum(law(p, lam)))
    lam = jnp.asarray(LAM, F32)
    p = {"v0": F32(30.0), "lam0": F32(400.0), "k": F32(4.0)}
    f(p, lam)
    n0 = f._cache_size()
    for v in (31.0, 29.0, 33.0):
        f({**p, "v0": F32(v)}, lam)
    assert f._cache_size() == n0
    g = jax.grad(lambda q: f({**p, "v0": q}, lam))(F32(30.0))
    np.testing.assert_allclose(float(g), float(np.sum((LAM / 400.0) ** 4)), rtol=1e-5)


def geant4_groupvel(energy, n):
    """G4MaterialPropertiesTable::CalculateGROUPVEL, literally (float64): -> (energies, v_g / c)."""
    E = energy
    eg, vg = [E[0]], [1.0 / (n[0] + (n[1] - n[0]) / math.log(E[1] / E[0]))]
    if vg[0] < 0 or vg[0] > 1.0 / n[0]:
        vg[0] = 1.0 / n[0]
    e0, n0, e1, n1 = E[0], n[0], E[1], n[1]
    for i in range(2, len(E)):
        v = 1.0 / (0.5 * (n0 + n1) + (n1 - n0) / math.log(e1 / e0))
        if v < 0 or v > 1.0 / (0.5 * (n0 + n1)):
            v = 1.0 / (0.5 * (n0 + n1))
        eg.append(0.5 * (e0 + e1))
        vg.append(v)
        e0, n0, e1, n1 = e1, n1, E[i], n[i]
    v = 1.0 / (n1 + (n1 - n0) / math.log(e1 / e0))
    if v < 0 or v > 1.0 / n1:
        v = 1.0 / n1
    eg.append(e1)
    vg.append(v)
    return np.array(eg), np.array(vg)


TABLES = [
    (np.array([2.0, 3.0, 6.0, 9.0]), np.array([1.30, 1.31, 1.35, 1.45])),
    (
        np.array([1.8, 2.2, 2.9, 3.5, 4.1, 5.0, 6.2, 7.3, 8.9, 9.7]),
        np.array([1.329, 1.331, 1.336, 1.341, 1.347, 1.358, 1.377, 1.402, 1.46, 1.51]),
    ),
    (np.array([2.0, 4.0, 6.0, 8.0]), np.array([1.5, 1.4, 1.45, 1.44])),  # anomalous dispersion: clamped knots
]


@pytest.mark.parametrize("energy,n", TABLES)
def test_geant4_rindex_group_index_is_geant4s_groupvel(energy, n):
    law = laws.geant4_rindex()
    p = {"energy": jnp.asarray(energy, F32), "value": jnp.asarray(n, F32)}
    lam = np.linspace(HC / (energy[-1] * 1.05), HC / (energy[0] * 0.95), 4001)
    got_n = np.asarray(law(p, jnp.asarray(lam, F32)), np.float64)
    got_g = np.asarray(laws.group_index(law)(p, jnp.asarray(lam, F32)), np.float64)
    eg, vg = geant4_groupvel(energy, n)
    np.testing.assert_allclose(got_n, np.interp(HC / lam, energy, n), atol=1e-5)
    np.testing.assert_allclose(got_g, 1.0 / np.interp(HC / lam, eg, vg), rtol=1e-5)
    gr = jax.grad(
        lambda v: jnp.sum(laws.group_index(law)({"energy": p["energy"], "value": v}, jnp.asarray(lam[::50], F32)))
    )(p["value"])
    assert np.all(np.isfinite(np.asarray(gr)))
