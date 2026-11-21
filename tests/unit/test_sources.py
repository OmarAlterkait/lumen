"""The light sources (lumen.sources): random numbers, closed forms per source type, chunks, parameters, derivatives.

Each source is checked against what it must be: yields, the cone angle, spectrum moments, time profiles.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.random import threefry2x32_p

from lumen import sources as S  # noqa: N812
from lumen._core import rng
from lumen.format.calibrations import Table
from lumen.tools.stats import z_limit
from lumen.units import FINE_STRUCTURE as ALPHA
from lumen.units import HC

F32 = jnp.float32
KEY = jax.random.PRNGKey(3)
WATER = Table(np.linspace(1.6, 4.6, 31), 1.325 + 0.01 * np.linspace(0.0, 3.0, 31))


def FLAT(n):  # noqa: N802 -- a flat RINDEX table spanning 300 .. 600 nm
    return Table(np.array([HC / 600.0, HC / 300.0]), np.array([n, n]))


E_TAB = np.array([2.0, 2.4, 2.8, 3.1, 3.4, 3.8, 4.2])
FAST, SLOW = np.array([0.0, 0.3, 1.0, 0.8, 0.5, 0.1, 0.0]), np.array([0.1, 0.6, 0.9, 0.4, 0.2, 0.05, 0.0])


def scint_params(y=4000.0, y1=0.3, y2=0.7, tau=(6.0, 1500.0)):
    m = "materials/LS/"
    return {m + "SCINTILLATIONYIELD": y, m + "SCINTILLATIONYIELD1": y1, m + "SCINTILLATIONYIELD2": y2,
            m + "SCINTILLATIONCOMPONENT1": Table(E_TAB, FAST), m + "SCINTILLATIONCOMPONENT2": Table(E_TAB, SLOW),
            m + "SCINTILLATIONTIMECONSTANT1": tau[0], m + "SCINTILLATIONTIMECONSTANT2": tau[1]}  # fmt: skip


def g4_mean_energy(e, v):
    """The mean photon energy of Geant4's sampler: uniform in energy per interval, trapezoid masses."""
    m = 0.5 * (v[1:] + v[:-1]) * np.diff(e)
    return float(np.sum(m * 0.5 * (e[1:] + e[:-1])) / m.sum())


def z(x, want):
    x = np.asarray(x, np.float64)
    return (x.mean() - want) / (x.std(ddof=1) / math.sqrt(len(x)))


# ------------------------------------------------------------------------------------------------- random numbers
def test_uniforms_are_the_threefry_bits_of_their_counters():
    """A key's k uniforms are threefry's bits of the counters (0, j), j < k (the sources' photon_uniforms)."""
    lo = jnp.arange(8, dtype=jnp.uint32)
    b1, b2 = threefry2x32_p.bind(*(jnp.broadcast_to(k, lo.shape) for k in KEY), jnp.zeros_like(lo), lo)
    assert bool(jnp.all(rng.uniforms(KEY, 8) == (((b1 ^ b2) >> 9).astype(F32) + 0.5) * F32(2.0**-23)))


def test_gaussian_of_uniforms_is_jax_random_normal():
    u = ((jax.random.bits(KEY, (512, 3), jnp.uint32) >> 9).astype(F32) + 0.5) * F32(2.0**-23)
    assert bool(jnp.all(S.gaussian(u) == jax.random.normal(KEY, (512, 3), F32)))


@pytest.mark.parametrize("make", ["point", "spectrum", "cherenkov", "scint", "segment", "steps", "arrays"])
def test_a_run_is_the_same_in_any_chunks(make):
    src, p = SOURCES[make]()  # bitwise on the GPU; the CPU's vectorised loop tail may round a last bit differently
    whole = src.emit(p, KEY, 320, 0, 320)
    part = src.emit(p, KEY, 128, 128, 320)
    for a, b in zip(whole, part, strict=True):
        if a is None:  # an optional field none of lumen's own sources fills (a label)
            assert b is None
            continue
        if jax.default_backend() == "gpu" or a.dtype == bool:
            assert bool(jnp.all(a[128:256] == b)), make
        else:
            np.testing.assert_allclose(np.asarray(a[128:256]), np.asarray(b), rtol=3e-7, atol=3e-7, err_msg=make)


def test_a_new_value_or_a_new_source_of_the_same_structure_never_recompiles():
    from lumen.sources._base import _EMIT

    S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0).emit(None, KEY, 96)
    size = _EMIT._cache_size()
    S.Point((1.0, 2.0, 3.0), n_photons=7.0, lam=450.0, time=3.0).emit(None, KEY, 96, 1000)
    p = (
        S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0)
        .parameters()
        .set("source/position", np.array([4.0, 0.0, 0.0]))
    )
    S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0).emit(p, jax.random.PRNGKey(9), 96)
    assert _EMIT._cache_size() == size


# ------------------------------------------------------------------------------------------------------- the point
def test_point_isotropic_and_beam():
    e = S.Point((1.0, 2.0, 3.0), n_photons=50.0, lam=420.0, time=4.0).emit(None, KEY, 20000)
    assert e.x.shape == (20000, 3) and bool(jnp.all(e.x == jnp.asarray([1.0, 2.0, 3.0], F32)))
    assert float(e.w.sum()) == pytest.approx(50.0, rel=1e-6) and bool(jnp.all((e.t == 4.0) & (e.lam == 420.0)))
    assert np.allclose(np.linalg.norm(np.asarray(e.d), axis=1), 1.0, atol=1e-6)
    for c in range(3):  # isotropic: <d_c> = 0, <d_c^2> = 1/3
        assert abs(z(e.d[:, c], 0.0)) < z_limit(6) and abs(z(e.d[:, c] ** 2, 1 / 3)) < z_limit(6)
    b = S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0, direction=(0.0, 0.6, 0.8)).emit(None, KEY, 10)
    assert bool(jnp.all(b.d == jnp.asarray([0.0, 0.6, 0.8], F32)))


def test_point_spectrum_moments_and_its_derivative():
    tab = Table(E_TAB, SLOW)
    src = S.Point((0.0, 0.0, 0.0), n_photons=10.0, spectrum=tab)
    e = src.emit(None, KEY, 1 << 16)
    # at the reference (the run's own table) the weight is n / samples: live / reference density, one float32 ulp
    np.testing.assert_allclose(np.asarray(e.w), 10.0 / (1 << 16), rtol=2 * np.finfo(np.float32).eps)
    assert abs(z(HC / np.asarray(e.lam, np.float64), g4_mean_energy(E_TAB, SLOW))) < z_limit()
    k = 3  # d E[photon energy] / d value[k] through the reweighted spectrum, against the closed form

    def per_photon(t):  # each photon's estimate of E[photon energy] and of the total photons, n = 10
        r = src.emit({"source/spectrum": Table(jnp.asarray(E_TAB, F32), jnp.asarray(SLOW, F32).at[k].add(t))}, KEY,
                     1 << 16)  # fmt: skip
        return r.w * (1 << 16) * HC / r.lam / 10.0, r.w * (1 << 16)

    _, (dv, dtot) = jax.jvp(per_photon, (F32(0.0),), (F32(1.0),))
    h = 1e-4
    up, down = (g4_mean_energy(E_TAB, SLOW + s * h * (np.arange(7) == k)) for s in (1, -1))
    assert abs(z(dv, (up - down) / (2 * h))) < z_limit()
    assert abs(z(dtot, 0.0)) < z_limit()  # the total is the same for any spectrum


def test_point_parameters_and_values():
    src = S.Point((0.0, 0.0, 1.0), n_photons=5.0, lam=400.0)
    p = src.parameters()
    assert not p.info("source/position").available  # it moves where the light starts: no derivative in this version
    assert p.info("source/n_photons").unit == "" and p.explain("source/time") == "source Point"
    p2 = p.set("source/n_photons", 8.0)
    assert float(src.emit(p2, KEY, 100).w.sum()) == pytest.approx(8.0, rel=1e-6)


def test_a_source_field_of_a_pending_method_is_refused_like_rindex():
    """Point.lam declares moving_paths+reweight, RINDEX's method, not validated (refraction vertices sliding across a
    face edge have no boundary term): unavailable, and a tangent through emit raises"""
    src = S.Point((0.0, 0.0, 0.0), lam=400.0)
    p = src.parameters()
    assert p.info("source/lam").method == "moving_paths+reweight" and not p.info("source/lam").available
    with pytest.raises(S.GradientNotAvailable, match=r"moving_paths\+reweight"):
        jax.jvp(lambda lam: src.emit(p.set("source/lam", lam), KEY, 8).w, (F32(400.0),), (F32(1.0),))
    with pytest.raises(ValueError):
        S.Point((0.0, 0.0, 0.0))


def test_a_source_position_derivative_is_refused_and_reweight_fields_are_not():
    def total(x, direction):
        e = S.Point(x, n_photons=3.0, lam=400.0, direction=direction).emit(None, KEY, 64)
        return jnp.sum(e.x[:, 2] * e.w)

    x0 = jnp.asarray([0.0, 0.0, 1.0], F32)
    t = jnp.asarray([0.0, 0.0, 1.0], F32)
    for direction in (None, (0.0, 0.0, 1.0)):  # drawn (isotropic) or a beam: one refusal
        with pytest.raises(S.GradientNotAvailable, match="moves where the light starts"):
            jax.jvp(lambda x, direction=direction: total(x, direction), (x0,), (t,))
    # time and intensity are reweight parameters of every source: d t / d time = 1, d w / d n = w / n
    src = S.Point((0.0, 0.0, 0.0), n_photons=2.0, lam=400.0, direction=(1.0, 0.0, 0.0))
    dt = jax.jvp(lambda tm: src.emit({"source/time": tm}, KEY, 8).t, (F32(1.0),), (F32(1.0),))[1]
    dw = jax.jvp(lambda n: src.emit({"source/n_photons": n}, KEY, 8).w.sum(), (F32(2.0),), (F32(1.0),))[1]
    assert bool(jnp.all(dt == 1.0)) and float(dw) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------------------------------- Cherenkov
def cherenkov(n=1.4, beta=0.95, charge=1.0, length=0.2):
    src = S.CherenkovSegment("W", (0.0, 0.0, 0.0), (0.0, 0.6, 0.8), length, beta, charge, 1.0)
    return src, {"materials/W/RINDEX": FLAT(n)}  # its span: 300 .. 600 nm


def test_cherenkov_yield_cone_spectrum_and_polarisation():
    src, p = cherenkov(charge=2.0)
    e = src.emit(p, KEY, 1 << 15)
    want = 2 * math.pi * ALPHA * 4.0 * 0.2 * (1 - 1 / (0.95 * 1.4) ** 2) * (1 / 300.0 - 1 / 600.0) * 1e9  # Frank-Tamm
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(want, rel=1e-5)
    axis = np.array([0.0, 0.6, 0.8])
    np.testing.assert_allclose(np.asarray(e.d) @ axis, 1 / (0.95 * 1.4), rtol=2e-6)  # the cone
    assert abs(z(1.0 / np.asarray(e.lam, np.float64), 0.5 * (1 / 300.0 + 1 / 600.0))) < z_limit()  # 1/lam^2 spectrum
    er = np.asarray(e.pol[:, 0])
    np.testing.assert_allclose(np.sum(er * np.asarray(e.d), 1), 0.0, atol=2e-6)  # transverse, in the track's plane
    np.testing.assert_allclose(np.sum(np.cross(np.asarray(e.d), axis) * er, 1), 0.0, atol=2e-6)
    s = np.asarray(e.x) @ axis  # uniform along the segment, at the time the charge passes there
    assert s.min() >= 0 and s.max() <= 0.2 + 1e-7 and abs(z(s, 0.1)) < z_limit()
    np.testing.assert_allclose(np.asarray(e.t), 1.0 + s / (0.95 * 0.299792458), rtol=1e-6)


def test_cherenkov_below_threshold_and_with_a_dispersive_index():
    src, p = cherenkov(beta=0.7)
    assert float(src.emit(p, KEY, 256).w.sum()) == 0.0
    src = S.CherenkovSegment("W", length=0.5, beta=0.99)
    e = src.emit({"materials/W/RINDEX": WATER}, KEY, 1 << 16)
    lam = np.linspace(HC / 4.6, HC / 1.6, 200001)  # the table's span
    n = np.interp(HC / lam, WATER.energy, WATER.value)
    want = 2 * math.pi * ALPHA * 0.5 * np.trapezoid(np.maximum(1 - 1 / (0.99 * n) ** 2, 0) / lam**2, lam) * 1e9
    w = np.asarray(e.w, np.float64)
    assert abs((w.sum() - want) / (w.std() * math.sqrt(len(w)))) < z_limit()


def test_a_photon_below_threshold_carries_a_tangent_of_exactly_zero():
    """At beta 0 (a step at rest) and below threshold the photon has weight 0 and, in the index and in beta, a
    tangent of exactly 0 (a max of the yield gave NaN: its mask times the overflowing tangent of 1 / (beta n)^2)."""
    from lumen.sources import cherenkov as ch

    e, n = jnp.asarray([2.0, 4.0], F32), jnp.asarray([1.33, 1.36], F32)
    lo, hi = ch.bounds((e, n))
    for beta in (0.0, 1e-20, 0.5):
        for u in (0.1, 0.5, 0.9):
            rand = ch.LAYOUT.split(jnp.asarray([u, 0.3, 0.7], F32))
            f = lambda nv, b, r=rand: ch.cone(r, (e, nv), jnp.zeros(3), jnp.asarray([0.0, 0.0, 1.0]), 0.1, b, 1.0,  # noqa: E731
                                              lo, hi, 1.0)  # fmt: skip
            out, tan = jax.jvp(f, (n, F32(beta)), (jnp.ones_like(n), F32(1.0)))
            assert float(out[3]) == 0.0 and all(bool(jnp.all(t == 0.0)) for t in (tan[1], tan[3], tan[4]))


def test_cherenkov_derivatives():
    src, p = cherenkov()
    dq = jax.jvp(lambda q: src.emit({**p, "source/charge": q}, KEY, 64).w.sum(), (F32(1.0),), (F32(1.0),))[1]
    assert float(dq) == pytest.approx(2.0 * float(src.emit(p, KEY, 64).w.sum()), rel=1e-5)
    for f in ("start", "beta"):
        x0 = jnp.asarray(getattr(src, f), F32)
        with pytest.raises(S.GradientNotAvailable):
            jax.jvp(lambda x, f=f: src.emit({**p, f"source/{f}": x}, KEY, 8).x, (x0,), (jnp.ones_like(x0),))


# ------------------------------------------------------------------------------------------------ scintillation
def test_scintillation_point_yield_shares_times_and_spectra():
    p = scint_params()
    src = S.ScintillationPoint.of(p, "LS", position=(0.1, 0.2, 0.3), energy=0.5, time=2.0)
    e = src.emit(p, KEY, 1 << 16)
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(4000.0 * 0.5, rel=1e-5)  # exact per photon
    t = np.asarray(e.t, np.float64) - 2.0
    fast = t < 60.0  # the fast (6 ns) and slow (1500 ns) components separate here up to 4 % of the slow
    assert abs(z(fast, 0.3 * (1 - math.exp(-60 / 6)) + 0.7 * (1 - math.exp(-60 / 1500)))) < z_limit()
    assert abs(z(t, 0.3 * 6.0 + 0.7 * 1500.0)) < z_limit()
    e_fast = HC / np.asarray(e.lam, np.float64)[t < 1.0]  # the components' mixture below 1 ns: 1 % slow
    wf, ws = 0.3 * (1 - math.exp(-1 / 6)), 0.7 * (1 - math.exp(-1 / 1500))
    mix = (wf * g4_mean_energy(E_TAB, FAST) + ws * g4_mean_energy(E_TAB, SLOW)) / (wf + ws)
    assert abs(z(e_fast, mix)) < z_limit()
    assert bool(jnp.all(e.x == jnp.asarray([0.1, 0.2, 0.3], F32)))
    dw = jax.jvp(lambda y: src.emit({**p, "materials/LS/SCINTILLATIONYIELD1": y}, KEY, 4096).w,
                 (F32(0.3),), (F32(1.0),))[1]  # fmt: skip
    dw = np.asarray(dw, np.float64)  # each photon's derivative: the total does not depend on the share (mean 0)
    assert np.all(dw == 0.0) or abs(z(dw, 0.0)) < z_limit(), (dw.sum(), z(dw, 0.0))


def test_scintillation_segment_birks_and_positions():
    p = scint_params()
    a, b = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.003, 0.004])
    src = S.ScintillationSegment.of(p, "LS", start=a, end=b, t_start=1.0, t_end=1.05, energy=0.9, birks=1.3e-4)
    e = src.emit(p, KEY, 1 << 14)
    want = 4000.0 * 0.9 / (1 + 1.3e-4 * 0.9 / 0.005)
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(want, rel=1e-5)
    f = np.asarray(e.x)[:, 1] / 0.003
    assert f.min() >= 0 and f.max() <= 1 and abs(z(f, 0.5)) < z_limit()
    with pytest.raises(KeyError):
        S.ScintillationPoint.of({}, "LS")


def test_scintillation_reads_geant4_10_keys_as_geant4_11_s():
    """Geant4-10 keys mapped at import (lumen.rules.keys.geant4_11_scintillation) give Geant4 10.7's light."""
    from lumen.format.schema import Property
    from lumen.rules import keys

    tab = lambda k: Property(k, np.asarray(FAST, float), energy=np.asarray(E_TAB, float))  # noqa: E731
    one = lambda k, v: Property(k, np.asarray(v))  # noqa: E731
    old = {
        "SCINTILLATIONYIELD": one("SCINTILLATIONYIELD", 100.0),
        "YIELDRATIO": one("YIELDRATIO", 0.8),
        "FASTCOMPONENT": tab("FASTCOMPONENT"),
        "SLOWCOMPONENT": tab("SLOWCOMPONENT"),
        "FASTTIMECONSTANT": one("FASTTIMECONSTANT", 2.0),
        "SLOWTIMECONSTANT": one("SLOWTIMECONSTANT", 20.0),
    }
    props, note = keys.geant4_11_scintillation(old)
    assert note and "YIELDRATIO" not in props and float(props["SCINTILLATIONYIELD2"].value) == pytest.approx(0.2)
    m = "materials/LS/"
    p = {m + k: (Table(v.energy, v.value) if v.energy is not None else float(v.value)) for k, v in props.items()}
    src = S.ScintillationPoint.of(p, "LS", energy=2.0)
    t = np.asarray(src.emit(p, KEY, 1 << 15).t, np.float64)
    assert abs(z(t, 0.8 * 2.0 + 0.2 * 20.0)) < z_limit()


# ---------------------------------------------------------------------------------------------- the step dump
def steps(light):
    rng_ = np.random.default_rng(1)
    a = rng_.normal(size=(5, 3))
    d = rng_.normal(size=(5, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    length = np.array([0.01, 0.02, 0.005, 0.03, 0.015])
    beta = np.array([0.99, 0.9, 0.6, 0.95, 0.8])  # one below threshold
    f = dict(start=a, end=a + length[:, None] * d, t_start=np.arange(5.0), t_end=np.arange(5.0) + 0.1)
    p = {"materials/W/RINDEX": FLAT(1.4)} | scint_params()
    if light == "cherenkov":
        f |= dict(beta_start=beta + 0.002, beta_end=beta - 0.002, charge=np.array([1.0, -1.0, 1.0, 2.0, 1.0]))
        return S.CherenkovSteps("W", **f), p, f, d, beta
    f |= dict(edep=np.array([2.0, 1.0, 3.0, 0.5, 1.5]))  # MeV
    return S.ScintillationSteps.of(p, "LS", **f, birks=1e-4), p, f, d, beta


def test_geant4_steps_cherenkov_yield_and_cone():
    src, p, f, d, beta = steps("cherenkov")
    e = src.emit(p, KEY, 1 << 16)
    length = np.linalg.norm(f["end"] - f["start"], axis=1)
    per = 2 * math.pi * ALPHA * f["charge"] ** 2 * length * np.maximum(1 - 1 / (beta * 1.4) ** 2, 0)
    per = per * (1 / 300 - 1 / 600) * 1e9
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(per.sum(), rel=1e-5)  # n constant: tot / M each
    cosines = np.asarray(e.d) @ d.T  # each photon on its step's cone
    assert np.all(np.min(np.abs(cosines - 1 / (beta * 1.4)), 1) < 3e-6)


def test_geant4_steps_scintillation_yield():
    src, p, f, _, _ = steps("scintillation")
    e = src.emit(p, KEY, 1 << 16)
    length = np.linalg.norm(f["end"] - f["start"], axis=1)
    mev = f["edep"]
    want = np.sum(4000.0 * mev / (1 + 1e-4 * mev / length))
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(want, rel=1e-5)  # every photon carries tot / M


def test_geant4_steps_count_light_over_the_true_step_length_when_given():
    """Geant4's photon number and Birks dE/dx use G4Step::GetStepLength (the true path), not the chord."""
    for light in ("cherenkov", "scintillation"):
        src, p, f, _, _ = steps(light)
        chord = np.linalg.norm(f["end"] - f["start"], axis=1)
        true = chord * np.array([1.0, 1.3, 1.05, 2.0, 1.1])
        e0 = src.emit(p, KEY, 1 << 16)
        e1 = dataclasses.replace(src, step_length=true).emit(p, KEY, 1 << 16)
        w0, w1 = (float(np.asarray(e.w, np.float64).sum()) for e in (e0, e1))
        if light == "cherenkov":  # N proportional to the length
            beta = 0.5 * (f["beta_start"] + f["beta_end"])
            per = f["charge"] ** 2 * np.maximum(1 - 1 / (beta * 1.4) ** 2, 0)
            assert w1 / w0 == pytest.approx(np.sum(per * true) / np.sum(per * chord), rel=1e-5)
        else:  # Birks' dE/dx = edep / length
            mev = f["edep"]
            assert w1 == pytest.approx(np.sum(4000.0 * mev / (1 + 1e-4 * mev / true)), rel=1e-5)


# ------------------------------------------------------------------------------------------------------- arrays
def test_from_arrays_rows_and_its_derivative_rule():
    n = 40
    x = jnp.arange(3 * n, dtype=F32).reshape(n, 3)
    d = jnp.tile(jnp.asarray([0.0, 0.0, 1.0], F32), (n, 1))
    src = S.FromArrays(x, d, jnp.zeros(n), jnp.full(n, 400.0), jnp.ones(n))
    e = src.emit(None, KEY, 10, 5)
    assert bool(jnp.all(e.x == x[5:15]))
    with pytest.raises(S.GradientNotAvailable, match="moves where the light starts"):
        jax.jvp(lambda xv: src.emit({"source/position": xv}, KEY, n).x, (x,), (jnp.ones_like(x),))


SOURCES = {
    "point": lambda: (S.Point((0.0, 0.0, 0.0), n_photons=2.0, lam=400.0), None),
    "spectrum": lambda: (S.Point((0.0, 0.0, 0.0), spectrum=Table(E_TAB, FAST)), None),
    "cherenkov": cherenkov,
    "scint": lambda: (S.ScintillationPoint.of(scint_params(), "LS", energy=1.0), scint_params()),
    "segment": lambda: (S.ScintillationSegment.of(scint_params(), "LS", end=(0.0, 0.0, 0.01), energy=1.0),
                        scint_params()),
    "steps": lambda: steps("cherenkov")[:2],
    "arrays": lambda: (S.FromArrays(jnp.ones((320, 3)), jnp.ones((320, 3)), jnp.arange(320.0), jnp.full(320, 400.0),
                                    jnp.ones(320)), None),
}  # fmt: skip


def test_the_moving_fields_are_the_ones_declared_moving():
    import numpy as np

    from lumen.sources.arrays import FromArrays
    from lumen.sources.point import Point

    n = 4
    src = FromArrays(
        np.zeros((n, 3)),
        np.tile([0.0, 0.0, 1.0], (n, 1)),
        np.zeros(n),
        np.full(n, 400.0),
        np.ones(n),
        polarisation=np.zeros((n, 2, 3)),
    )
    assert src.MOVING == ("position", "direction", "lam") == tuple(f for f, p in src.FIELDS.items() if p.moves)
    p = src.parameters()
    assert p.info("source/lam").method == "none"  # a tangent on it moves the photons: no derivative in this version
    assert p.info("source/polarisation").range == (-np.inf, np.inf)
    assert Point((0.0, 0.0, 0.0), lam=400.0).MOVING == ("position", "direction", "lam")


def test_rows_are_bucketed_and_padding_steps_are_lightless():
    """One rule for compiled sizes (the power of two >= n); a padded chunk reads zeros past a source's arrays, never
    a clamped slice that would shift its active rows; padding steps are lightless."""
    from lumen.sources._base import _pad_rows, bucket, rows

    assert [bucket(n) for n in (0, 1, 2, 3, 4, 5, 1000, 1024, 1025)] == [1, 1, 2, 4, 4, 8, 1024, 1024, 2048]
    a = jnp.arange(6, dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(rows(a, 4, 4, 0)), [4.0, 5.0, 0.0, 0.0])
    v = _pad_rows({"charge": a, "start": a, "birks": jnp.float32(1e-4)}, {"charge": 0, "start": 0}, ("charge",))
    np.testing.assert_array_equal(np.asarray(v["charge"]), [0, 1, 2, 3, 4, 5, 0, 0])
    np.testing.assert_array_equal(np.asarray(v["start"]), [0, 1, 2, 3, 4, 5, 5, 5])
    assert float(v["birks"]) == np.float32(1e-4)


def _arrays(n, w=None, **kw):
    return S.FromArrays(np.tile([0.0, 0.0, 0.1], (n, 1)), np.tile([0.0, 0.0, 1.0], (n, 1)), np.zeros(n),
                        np.full(n, 400.0), np.ones(n) if w is None else w, **kw)  # fmt: skip


def test_a_source_of_one_photon_per_row_emits_exactly_its_rows():
    """FromArrays and a Point with per-row fields: no sample count of their own. Each row is one photon at its own
    weight (a Point row stands for its row's n_photons), the padding rows past the last weigh 0 and sit at the last
    row's point (never at zeros), and a run with another sample count is refused."""
    n = 5  # padded to 8
    src = _arrays(n, w=np.arange(1.0, 6.0))
    assert src.photon_rows() == n
    e = src.emit(None, KEY, 8)
    np.testing.assert_array_equal(np.asarray(e.w), [1, 2, 3, 4, 5, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(e.x)[5:], np.tile([0.0, 0.0, 0.1], (3, 1)).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(e.lam), 400.0)
    for samples in (n - 1, 4 * n):  # fewer would drop rows, more would trace empty ones: neither is the rows
        with pytest.raises(ValueError, match="emits its 5 rows"):
            src.emitter(None, KEY, samples)
    np.testing.assert_array_equal(np.asarray(src.emitter(None, KEY, n)(0, 8).w), [1, 2, 3, 4, 5, 0, 0, 0])
    p = S.Point(np.zeros((n, 3)), n_photons=3.0, lam=400.0)
    assert p.photon_rows() == n
    np.testing.assert_array_equal(np.asarray(p.emit(None, KEY, 8).w), [3, 3, 3, 3, 3, 0, 0, 0])
    per = S.Point(np.zeros((n, 3)), n_photons=np.arange(1.0, 6.0), lam=400.0)
    np.testing.assert_array_equal(np.asarray(per.emit(None, KEY, 8).w), [1, 2, 3, 4, 5, 0, 0, 0])
    assert S.Point((0.0, 0.0, 0.0), n_photons=8.0, lam=400.0).photon_rows() is None  # a point: samples share it
    np.testing.assert_array_equal(
        np.asarray(S.Point((0.0, 0.0, 0.0), n_photons=8.0, lam=400.0).emit(None, KEY, 4).w), 2
    )


def test_per_row_fields_of_different_lengths_are_refused():
    """Columns of unequal rows are refused at construction and when a parameter sets one (never read as zeros)."""
    with pytest.raises(ValueError, match="different numbers of rows"):
        S.FromArrays(np.zeros((4, 3)), np.tile([0.0, 0.0, 1.0], (4, 1)), np.zeros(3), np.full(4, 400.0), np.ones(4))
    with pytest.raises(ValueError, match="different numbers of rows"):
        S.Point(np.zeros((4, 3)), lam=np.full(6, 400.0))
    with pytest.raises(ValueError, match="different numbers of rows"):
        S.Point(np.zeros((4, 3)), lam=400.0, label=np.arange(3))
    with pytest.raises(ValueError, match="different numbers of rows"):
        _arrays(4).emit({"source/time": jnp.zeros(6)}, KEY, 4)
    with pytest.raises(ValueError, match="one row axis at most"):
        S.Point(np.zeros((2, 4, 3)), lam=400.0)
    with pytest.raises(ValueError, match="no field has rows"):
        S.FromArrays(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0, 400.0, 1.0)
    with pytest.raises(ValueError, match="a label per row needs a source with rows"):
        S.CherenkovSegment("W", label=np.arange(4))
    a = np.zeros((3, 3))
    with pytest.raises(ValueError, match="different numbers of rows"):
        S.CherenkovSteps("Water", start=a, end=a + 1.0, t_start=np.zeros(3), t_end=np.zeros(2), beta_start=np.ones(3),
                         beta_end=np.ones(3), charge=np.ones(3))  # fmt: skip


def test_cherenkov_emits_over_the_rindex_table_s_span_as_geant4_does():
    """The wavelengths are the span of the medium's RINDEX table (G4Cerenkov emits over [its first energy, its last]),
    with no detector; a one-knot RINDEX spans none: no light, as G4Cerenkov's photon number is then 0."""
    p = {"materials/W/RINDEX": WATER}  # 1.6 .. 4.6 eV: 269.5 .. 774.9 nm
    e = S.CherenkovSegment("W", length=0.5, beta=0.99).emit(p, KEY, 1 << 14)
    lam = np.asarray(e.lam, np.float64)
    assert HC / 4.6 <= lam.min() < HC / 4.6 + 2.0 and HC / 1.6 - 25.0 < lam.max() <= HC / 1.6 * (1 + 1e-6)
    one = S.CherenkovSegment("W", length=0.5, beta=0.99).emit(
        {"materials/W/RINDEX": Table(np.array([3.0]), np.array([1.33]))}, KEY, 4
    )
    assert float(np.asarray(one.w).sum()) == 0.0


def test_a_step_s_cherenkov_photons_are_placed_as_g4cerenkov_places_them():
    """Along a slowing step the photons follow the density linear between the photon numbers at the two speeds
    (G4Cerenkov 10.7.4 PostStepDoIt), timed by the interpolated velocity; at one speed they are uniform."""
    from lumen.sources import cherenkov as ch

    tab = (WATER.energy, WATER.value)
    n0, n1 = (float(ch.photons_per_length(tab, b)) for b in (0.95, 0.76))
    assert n0 > n1 > 0.0
    a, one, L = np.zeros((1, 3)), np.ones(1), 0.01
    src = S.CherenkovSteps("W", start=a, end=a + [0.0, 0.0, L], t_start=0 * one, t_end=one, beta_start=0.95 * one,
                           beta_end=0.76 * one, charge=one)  # fmt: skip
    e = src.emit({"materials/W/RINDEX": WATER}, KEY, 1 << 16)
    f = np.asarray(e.x)[:, 2] / L
    want = (n0 / 2 + (n1 - n0) / 3) / ((n0 + n1) / 2)  # the mean of f under the linear density
    assert abs(f.mean() - want) < 4.0 * f.std() / np.sqrt(len(f)), (f.mean(), want)
    v = 0.299792458 * (0.95 + f * (0.76 - 0.95) / 2)
    np.testing.assert_allclose(np.asarray(e.t), f * L / v, rtol=1e-5)
    assert float(ch.along_step(jnp.float32(0.3), 2.0, 2.0)) == pytest.approx(0.3)


def test_scintillation_reads_three_components_and_one():
    """G4Scintillation 11: three components with SCINTILLATIONCOMPONENT3 (shares y_i / s), one with only the first."""
    m = "materials/LS/"
    p = {m + "SCINTILLATIONYIELD": 100.0, m + "SCINTILLATIONYIELD1": 0.2, m + "SCINTILLATIONYIELD2": 0.3,
         m + "SCINTILLATIONYIELD3": 0.5, **{m + f"SCINTILLATIONCOMPONENT{i}": Table(E_TAB, FAST) for i in (1, 2, 3)},
         m + "SCINTILLATIONTIMECONSTANT1": 2.0, m + "SCINTILLATIONTIMECONSTANT2": 20.0,
         m + "SCINTILLATIONTIMECONSTANT3": 200.0}  # fmt: skip
    src = S.ScintillationPoint.of(p, "LS", energy=2.0)
    assert src.components == (0, 1, 2)
    t = np.asarray(src.emit(p, KEY, 1 << 16).t, np.float64)
    assert abs(z(t, 0.2 * 2.0 + 0.3 * 20.0 + 0.5 * 200.0)) < z_limit()
    one = {k: v for k, v in p.items() if not k.endswith(("2", "3"))}
    t1 = np.asarray(S.ScintillationPoint.of(one, "LS", energy=2.0).emit(one, KEY, 1 << 14).t, np.float64)
    assert abs(z(t1, 2.0)) < z_limit()  # one component takes all the light, whatever SCINTILLATIONYIELD1 is
    with pytest.raises(KeyError, match="SCINTILLATIONTIMECONSTANT3"):
        S.ScintillationPoint.of({k: v for k, v in p.items() if k != m + "SCINTILLATIONTIMECONSTANT3"}, "LS")
    with pytest.raises(KeyError, match="shares"):
        S.ScintillationPoint.of({**p, **{m + f"SCINTILLATIONYIELD{i}": 0.0 for i in (1, 2, 3)}}, "LS")


def test_a_draw_never_lands_on_a_share_of_zero():
    """Interior and trailing zero-light steps are never drawn (``lit``: the first share above zero at or after the
    search's index), whatever the cumulative sum's rounding; a lit index stays."""
    from lumen.sources._base import lit

    e = jnp.asarray([0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0], F32)
    i = jnp.arange(e.shape[0] + 2) - 1  # every index the search can give, the ends' overshoots included
    j = np.asarray(lit(i, e))
    assert np.all(np.asarray(e)[j] > 0) and j.tolist() == [1, 1, 1, 4, 4, 4, 6, 6, 6, 6, 6]
    u = jax.random.uniform(KEY, (1 << 16,))
    cdf = jnp.cumsum(e)
    k = np.asarray(lit(jnp.searchsorted(cdf, u * cdf[-1], side="right"), e))
    assert np.all(np.asarray(e)[k] > 0)


# --------------------------------------------------------------------------------------- silent errors refused
def test_a_direction_that_is_not_a_unit_vector_is_refused():
    """Every field declared a direction is a unit vector, as a source function's draw must be (the value checks,
    ``lumen.build.values``, which ``simulate`` runs): an unnormalised one stretched a Cherenkov track and a beam."""
    from lumen.build import values
    from lumen.params import Params

    n, none = 8, Params({}, {}, {})
    rows = np.tile([0.0, 0.0, 1.0], (n, 1))
    for src in (S.Point((0.0, 0.0, 0.0), lam=400.0, direction=(0.0, 0.0, 2.0)),
                S.CherenkovSegment("W", (0.0, 0.0, 0.0), (0.0, 0.0, 2.0), 0.2, 0.95, 1.0, 1.0),
                S.FromArrays(np.zeros((n, 3)), np.vstack([rows[:-1], [0.0, 0.5, 0.5]]), np.zeros(n),
                             np.full(n, 400.0), np.ones(n))):  # fmt: skip
        with pytest.raises(values.InvalidValue, match="unit vector"):
            values.check(None, none, src)
    beam = S.Point((0.0, 0.0, 0.0), lam=400.0, direction=(0.0, 0.6, 0.8))
    values.check(None, none, beam)
    with pytest.raises(values.InvalidValue, match="unit vector"):  # a direction given as its key is checked too
        values.check(None, beam.parameters().set("source/direction", np.array([0.6, 0.6, 0.6])), beam)
    values.check(None, none, S.FromArrays(np.zeros((n, 3)), rows, np.zeros(n), np.full(n, 400.0), np.ones(n)))


def test_a_key_may_not_replace_a_traced_source_field():
    """A traced field with a ``source/<field>`` key of another value: the key replaced it and its derivative was 0."""
    stale = S.Point((0.0, 0.0, 0.0), n_photons=3.0, lam=400.0).parameters()

    def total(n, p):
        return S.Point((0.0, 0.0, 0.0), n_photons=n, lam=400.0).emit(p, KEY, 16).w.sum()

    with pytest.raises(ValueError, match="give each value once"):
        jax.grad(total)(F32(3.0), stale)
    assert float(jax.grad(total)(F32(3.0), None)) == pytest.approx(1.0, rel=1e-6)

    def own(n):  # the key holding the field itself (parameters() of this source) is one value
        src = S.Point((0.0, 0.0, 0.0), n_photons=n, lam=400.0)
        return src.emit(src.parameters(), KEY, 16).w.sum()

    assert float(jax.grad(own)(F32(3.0))) == pytest.approx(1.0, rel=1e-6)


def test_another_source_s_parameters_may_not_replace_a_field():
    """One precedence for source fields: a key parameters() added is a copy of its source's field, so one holding
    another source's value is refused (it replaced the field passed, silently); a key set is the value given."""
    a = S.Point((0.0, 0.0, 0.0), n_photons=3.0, lam=400.0)
    b = S.Point((0.0, 0.0, 0.0), n_photons=5.0, lam=400.0)
    with pytest.raises(ValueError, match="give each value once"):
        b.emit(a.parameters(), KEY, 16)
    assert float(b.emit(b.parameters(), KEY, 16).w.sum()) == pytest.approx(5.0, rel=1e-6)  # its own keys
    same = S.Point((0.0, 0.0, 0.0), n_photons=3.0, lam=400.0)  # another source of equal values: one value
    assert float(same.emit(a.parameters(), KEY, 16).w.sum()) == pytest.approx(3.0, rel=1e-6)
    assert float(b.emit(a.parameters().set("source/n_photons", 8.0), KEY, 16).w.sum()) == pytest.approx(8.0, rel=1e-6)


def test_scintillation_point_of_works_under_grad():
    """``of`` builds nothing from a traced value: the components from the keys, the shares checked when concrete."""
    p = scint_params()

    def total(y1):
        q = {**p, "materials/LS/SCINTILLATIONYIELD1": y1}
        e = S.ScintillationPoint.of(q, "LS", energy=1.0).emit(q, KEY, 4096)
        return jnp.sum(e.w * (e.t < 20.0))  # the light before 20 ns grows with the fast share

    g = float(jax.grad(total)(F32(0.3)))
    assert np.isfinite(g) and g > 0.0
    with pytest.raises(KeyError, match="not finite"):  # concrete yields summing to 0 are still refused
        S.ScintillationPoint.of({**p, "materials/LS/SCINTILLATIONYIELD1": 0.0,
                                 "materials/LS/SCINTILLATIONYIELD2": 0.0}, "LS")  # fmt: skip


def test_a_new_source_with_the_same_spectrum_never_recompiles():
    """A spectrum is a value, not structure: a new spectrum source, or a new spectrum, compiles nothing."""
    from lumen.sources._base import _EMIT

    S.Point((0.0, 0.0, 0.0), spectrum=Table(E_TAB, SLOW)).emit(None, KEY, 96)
    size = _EMIT._cache_size()
    for v in (SLOW.copy(), FAST):
        S.Point((0.0, 0.0, 0.0), spectrum=Table(E_TAB.copy(), v)).emit(None, KEY, 96)
    assert _EMIT._cache_size() == size
    p = scint_params()
    S.ScintillationPoint.of(p, "LS", energy=1.0).emit(p, KEY, 96)
    size = _EMIT._cache_size()
    S.ScintillationPoint.of(scint_params(), "LS", energy=2.0).emit(p, KEY, 96)
    assert _EMIT._cache_size() == size


def test_every_choice_is_drawn_at_the_reference():
    """A scintillation component and its wavelength are drawn at the reference (default: the run's own values) and
    weighted to the live values: at the reference the emission is the default's, bit for bit; with the reference held
    and a yield moved, every photon is the same (x, d, t, lam) and only its weight moves, by the live share over the
    reference's."""
    p = scint_params()
    src = S.ScintillationPoint.of(p, "LS", energy=1.0)
    base = src.emit(p, KEY, 4096)
    at_ref = src.emit(p, KEY, 4096, reference=p)
    for a, b in zip(base, at_ref, strict=True):
        assert (a is None and b is None) or np.array_equal(np.asarray(a), np.asarray(b))
    q = {**p, "materials/LS/SCINTILLATIONYIELD1": 0.4}  # shares 0.4 / 1.1 against 0.3 / 1.0
    held = src.emit(q, KEY, 4096, reference=p)
    for f in ("x", "d", "t", "lam"):
        assert np.array_equal(np.asarray(getattr(held, f)), np.asarray(getattr(base, f))), f
    ratio = np.asarray(held.w, np.float64) / np.asarray(base.w, np.float64)
    want = np.array([(0.4 / 1.1) / 0.3, (0.7 / 1.1) / 0.7])  # the live share over the reference's, per component
    assert np.all(np.min(np.abs(ratio[:, None] / want[None, :] - 1.0), 1) < 1e-6)


def test_a_spectrum_is_drawn_from_the_reference_and_weighted_to_the_live_table():
    """A Point's spectrum: the density is 0 outside the live table (never its end value), and a live spectrum emitting
    where the reference has no mass is refused (the weight cannot reach it)."""
    from lumen.build import values
    from lumen.params import Params

    wide, narrow = Table(E_TAB, SLOW), Table(E_TAB, np.where(E_TAB > 3.0, 0.0, SLOW))
    src = S.Point((0.0, 0.0, 0.0), n_photons=1.0, spectrum=wide)
    held = S.Point((0.0, 0.0, 0.0), n_photons=1.0, spectrum=narrow).parameters()
    ref = held.set("source/spectrum", wide)
    e = S.Point((0.0, 0.0, 0.0), n_photons=1.0, spectrum=narrow).emit(None, KEY, 1 << 15, reference=ref)
    energy = HC / np.asarray(e.lam, np.float64)
    assert float(np.asarray(e.w)[energy > 3.1].sum()) == 0.0  # outside the live mass: weight 0
    assert abs(float(np.asarray(e.w, np.float64).sum()) - 1.0) < 0.02  # the total is the live spectrum's (1)
    none = Params({}, {}, {})
    values.check(None, none, S.Point((0.0, 0.0, 0.0), n_photons=1.0, spectrum=narrow), ref)
    with pytest.raises(values.InvalidValue, match="no mass"):
        values.check(None, none, src, held.set("source/spectrum", narrow))
