"""Spectral laws: values against float64 references, domains and ranges fail closed, the group index from the same
law, Geant4's RINDEX and GROUPVEL, and no recompilation when a value changes."""

import math

import helpers
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


def test_tabulated_is_linear_in_energy_held_beyond_and_checked_at_its_knots():
    e = np.array([1.5, 2.0, 3.0, 4.0, 6.5])
    v = np.array([10.0, 20.0, 15.0, 5.0, 1.0])
    law = laws.tabulated("length")
    np.testing.assert_allclose(ev(law, {"energy": e, "value": v}), np.interp(HC / LAM, e, v), rtol=2e-6)
    assert abs(ev(law, {"energy": e, "value": v}, np.array([HC / 2.5]))[0] - 17.5) < 1e-4
    assert np.all(ev(law, {"energy": e, "value": v}, np.array([HC / 9.0, HC / 1.0])) == np.float32([1.0, 10.0]))
    lam, got = laws.validate(law, {"energy": e, "value": v})
    np.testing.assert_allclose(lam, HC / e)
    np.testing.assert_allclose(got, v, rtol=1e-6)
    with pytest.raises(laws.LawError, match="out of range at 190.7 nm"):  # 6.5 eV
        laws.validate(law, {"energy": e, "value": np.array([10.0, 20.0, 15.0, 5.0, -1.0])})
    with pytest.raises(laws.LawError, match="no points"):  # a law in closed form names where it is checked
        laws.validate(helpers.power("length"), dict(v0=30.0, lam0=400.0, k=4.0))


def test_a_table_is_checked_only_where_light_evaluates_it():
    """Hyper-K's mPMT gel (WCSim e2dc7a4): ABSLENGTH 0 at its end knots (177 nm, 12398 nm), outside the water light
    (its RINDEX 200-790 nm). Checked at its knots inside the light's domain and at the domain's ends (held beyond its
    knots), only those knots are read; a value out of range the domain's end reaches by interpolation is refused."""
    law = laws.tabulated("length")
    e = np.array([0.1, 1.0, 2.0, 5.0, 7.0])
    gel = {"energy": e, "value": np.array([0.0, 0.2, 0.9, 0.9, 0.0])}
    dom = ((HC / 790.0, HC / 200.0),)  # 1.57-6.2 eV
    lam, got = laws.validate(law, gel, domain=dom)
    np.testing.assert_allclose(np.sort(lam), np.sort(np.r_[HC / e[2:4], 790.0, 200.0]))
    assert np.all(got > 0.0)  # the end knots are not among them
    with pytest.raises(laws.LawError, match="out of range at 200.0 nm"):  # the domain's end, interpolated
        laws.validate(laws.tabulated("fraction"), {"energy": np.array([1.0, 2.0, 7.0]),
                                                   "value": np.array([0.9, 0.9, 1.5])}, domain=dom)  # fmt: skip
    assert laws.domain_points([500.0], None).tolist() == [500.0]


def test_sellmeier_and_cauchy_against_float64():
    p = dict(B1=0.6961663, B2=0.4079426, B3=0.8974794, C1=0.0684043**2, C2=0.1162414**2, C3=9.896161**2)
    l2 = (LAM * 1e-3) ** 2
    ref = np.sqrt(1 + sum(p[f"B{i}"] * l2 / (l2 - p[f"C{i}"]) for i in (1, 2, 3)))
    np.testing.assert_allclose(ev(helpers.sellmeier(), p), ref, rtol=1e-6)
    assert abs(ev(helpers.sellmeier(), p, np.array([587.6]))[0] - 1.4585) < 2e-4  # fused silica (Malitson 1965)
    pc = dict(A=1.32, B=0.004, C=1e-5)
    np.testing.assert_allclose(ev(helpers.cauchy(), pc), 1.32 + 0.004 / l2 + 1e-5 / l2**2, rtol=1e-6)


def test_group_index_is_n_minus_lambda_dn_dlambda_of_the_same_law():
    p = dict(A=1.32, B=0.004, C=1e-5)
    l2 = (LAM * 1e-3) ** 2
    n = 1.32 + 0.004 / l2 + 1e-5 / l2**2
    dn = (-2 * 0.004 / (l2 * LAM * 1e-3) - 4 * 1e-5 / (l2**2 * LAM * 1e-3)) * 1e-3  # per nm
    np.testing.assert_allclose(ev(laws.group_index(helpers.cauchy()), p), n - LAM * dn, rtol=1e-5)
    with pytest.raises(ValueError):
        laws.group_index(laws.constant("length"))


def test_power_law_and_ranges_fail_closed():
    np.testing.assert_allclose(
        ev(helpers.power("length"), dict(v0=30.0, lam0=400.0, k=4.0)), 30.0 * (LAM / 400.0) ** 4, rtol=1e-5
    )
    for law, bad in (
        (laws.constant("index"), {"value": 0.9}),
        (laws.constant("fraction"), {"value": 1.2}),
        (laws.constant("length"), {"value": -1.0}),
        (laws.constant("length"), {"value": np.nan}),
    ):
        with pytest.raises(laws.LawError):
            laws.validate(law, bad)
    with pytest.raises(laws.LawError):
        laws.validate(helpers.cauchy(points=helpers.VISIBLE), {"A": 1.3, "B": 0.0})  # missing parameter


def test_changing_a_value_does_not_recompile_and_gradients_flow():
    law = helpers.power("length")
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


@pytest.mark.parametrize(
    "energy,n,lam",
    [
        (np.array([2.0, 3.0, 4.0]), np.array([1.30, 1.32, 1.35]), 420.0),  # a synthetic 3-knot table
        # a quantised acrylic-like table (index steps of 1e-3, uneven knots), off the segment's midpoint
        (np.array([2.80, 2.95, 3.03, 3.10, 3.25]), np.array([1.495, 1.497, 1.498, 1.500, 1.502]), 400.10),
    ],
)
def test_geant4_rindex_wavelength_derivative_is_the_segment_slope(energy, n, lam):
    """The index's derivative in wavelength is its forward interpolation's (the segment's slope in energy), not
    Geant4's group index (that is ``group_index``, for timing)"""
    p = {"energy": jnp.asarray(energy, F32), "value": jnp.asarray(n, F32)}
    _, dn = jax.jvp(lambda x: laws.geant4_rindex()(p, x), (jnp.float32(lam),), (jnp.float32(1.0),))
    k = np.searchsorted(energy, HC / lam) - 1
    e32, n32 = energy.astype(F32).astype(float), n.astype(F32).astype(float)  # the knots as the law holds them
    slope = (n32[k + 1] - n32[k]) / (e32[k + 1] - e32[k]) * (-HC / lam**2)
    np.testing.assert_allclose(float(dn), slope, rtol=1e-5)
    flat = {"energy": p["energy"], "value": jnp.full(len(n), 1.5, F32)}
    assert float(jax.jvp(lambda x: laws.geant4_rindex()(flat, x), (jnp.float32(lam),), (jnp.float32(1.0),))[1]) == 0.0


def test_a_length_of_zero_is_an_infinite_rate_not_a_refusal():
    """Geant4 reads an ABSLENGTH of 0 as a mean free path of 0: G4OpAbsorption returns it and G4VDiscreteProcess makes
    the step 0, so the photon is absorbed at once; and G4PhysicsVector holds a table's end value beyond its knots, so
    a 0 end knot means that at every energy past it. A length of 0 is therefore a value: the law check admits it at
    every knot, the rate 1 / L there is infinite, and the LENGTH is what is interpolated between the knots, not the
    rate. A negative length is not a value."""
    law = laws.tabulated("length")
    e = np.array([0.1, 1.0, 2.0, 5.0, 7.0])
    gel = {"energy": e, "value": np.array([0.0, 0.2, 0.9, 0.9, 0.0])}
    _, got = laws.validate(law, gel)  # every knot, no domain: the zeros are values
    assert got[0] == 0.0 and got[-1] == 0.0
    laws.validate(law, dict(gel, value=np.array([0.0, 0.2, 0.0, 0.9, 0.0])), domain=((HC / 790.0, HC / 200.0),))
    assert ev(law, gel, HC / np.array([0.05, 0.1, 9.0])).tolist() == [0.0, 0.0, 0.0]  # held beyond the knots
    with np.errstate(divide="ignore"):
        assert math.isinf(1.0 / ev(law, gel, HC / np.array([0.1]))[0])
    assert ev(law, gel, HC / np.array([0.55]))[0] == pytest.approx(0.1, rel=1e-5)  # the length, not 1 / L
    with pytest.raises(laws.LawError, match="out of range"):
        laws.validate(law, dict(gel, value=np.array([-1.0, 0.2, 0.9, 0.9, 0.0])))


def test_a_length_of_zero_gives_a_finite_rate_with_a_derivative_of_zero():
    """A medium whose ABSLENGTH is 0 at the photon's energy (at a 0 knot, or beyond a 0 end knot): the rate absorbs
    every path longer than 0 and none of a path of 0, and scaling the table (the gradient family: a 0 stays 0) has a
    derivative of 0 there, not NaN (1 / 0 = inf, whose derivative times 0 is NaN); elsewhere 1 / L and its derivative
    are unchanged."""
    from lumen.media import bulk

    gel = bulk.Medium("gel", laws.constant("index"), laws.tabulated("length"))
    e, v = jnp.array([2.0, 3.0, 5.0, 7.0]), jnp.array([0.0, 1.0, 0.5, 0.0])
    for energy, length in ((1.5, 0.0), (2.0, 0.0), (8.0, 0.0), (3.0, 1.0)):

        def mu(t, energy=energy):
            p = {"index": {"value": 1.4}, "absorption": {"energy": e, "value": v * (1.0 + t)}}
            return bulk.evaluate(gel, p, jnp.float32(HC / energy))["mu_abs"]

        m, dm = jax.jvp(mu, (jnp.float32(0.0),), (jnp.float32(1.0),))
        assert np.isfinite(m) and np.isfinite(dm), energy
        if length == 0.0:
            assert float(dm) == 0.0 and float(jnp.exp(-m * 0.0)) == 1.0 and float(jnp.exp(-m * 1e-30)) == 0.0
        else:
            assert float(m) == pytest.approx(1.0) and float(dm) == pytest.approx(-1.0)


def test_one_admissibility_predicate_for_the_file_the_parameters_and_every_run():
    """A negative length is refused by the file's law check, by Params.set and by the run's value check alike (one
    range table, one predicate); NaN is never a value, even of a key with no finite range; a traced NaN counts as
    invalid."""

    from lumen._core import domains
    from lumen.build import values
    from lumen.params import Param, Params
    from lumen.rules.keys import PARAMS

    law_of = {"ABSLENGTH": "length", "RAYLEIGH": "length", "MIEHG": "length", "WLSABSLENGTH": "length",
              "RINDEX": "index", "REALRINDEX": "index", "IMAGINARYRINDEX": "extinction", "REFLECTIVITY": "fraction",
              "TRANSMITTANCE": "fraction", "EFFICIENCY": "fraction"}  # each key's law (lumen.rules.optics)  # fmt: skip
    assert {k: PARAMS[k].range for k in law_of} == {k: laws.RANGES[q] for k, q in law_of.items()}
    with pytest.raises(laws.LawError, match="out of range"):
        laws.validate(laws.constant("length"), {"value": -1.0})
    spec = {"m/W/ABSLENGTH": PARAMS["ABSLENGTH"], "m/W/ANY": Param()}
    p = Params({"m/W/ABSLENGTH": 10.0, "m/W/ANY": 1.0}, spec, {})
    for key, bad in (("m/W/ABSLENGTH", -1.0), ("m/W/ANY", float("nan")), ("m/W/ANY", float("inf"))):
        with pytest.raises(ValueError, match="outside"):
            p.set(key, bad)
        with pytest.raises(values.InvalidValue, match="not finite"):
            values.check(None, Params({**p._values, key: bad}, spec, {}))
    assert float(p.set("m/W/ABSLENGTH", 0.0)["m/W/ABSLENGTH"]) == 0.0  # an infinite rate is a value
    assert bool(domains.admits(np, 0.0, *domains.LENGTH)) and not domains.admits(np, -1e-30, *domains.LENGTH)

    def invalid(x):
        return values.check(None, Params({**p._values, "m/W/ANY": x}, spec, {}))[2]

    assert int(jax.jit(invalid)(float("nan"))) == 1 and int(jax.jit(invalid)(2.0)) == 0
