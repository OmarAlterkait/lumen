"""The light sources (lumen.sources): random numbers, closed forms per source type, chunks, parameters, derivatives.

The bitwise comparison with the baseline's emitted photons is tools/port/a9s_check_sources.py on stored goldens;
here each source is checked against what it must be: yields, the cone angle, spectrum moments, time profiles.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.random import threefry2x32_p

from lumen import sources as S  # noqa: N812
from lumen._core import rng
from lumen.format.calibrations import Table
from lumen.sources.cherenkov import ALPHA

F32 = jnp.float32
KEY = jax.random.PRNGKey(3)
Z_MAX = 4.5
WATER = Table(np.linspace(1.6, 4.6, 31), 1.325 + 0.01 * np.linspace(0.0, 3.0, 31))
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
def test_uniforms_are_jax_bits_rows_at_any_offset():
    bits = jax.random.bits(KEY, (64, 3), jnp.uint32)
    want = ((bits >> 9).astype(F32) + 0.5) * F32(2.0**-23)
    assert bool(jnp.all(S.uniforms(KEY, 3, 64, 0) == want))
    assert bool(jnp.all(S.uniforms(KEY, 3, 20, 37) == want[37:57]))
    assert bool(jnp.all(S.uniforms(KEY, 5) == rng.uniforms(KEY, 5)))
    c = (np.arange(1000, dtype=np.uint64)[:, None] + np.uint64((1 << 31) - 500)) * np.uint64(4)
    c = c + np.arange(4, dtype=np.uint64)
    hi, lo = (jnp.asarray(v.astype(np.uint32)) for v in (c >> np.uint64(32), c & np.uint64(0xFFFFFFFF)))
    b1, b2 = threefry2x32_p.bind(*(jnp.broadcast_to(k, hi.shape) for k in KEY), hi, lo)  # counters past 2^32
    want = (((b1 ^ b2) >> 9).astype(F32) + 0.5) * F32(2.0**-23)
    assert bool(jnp.all(S.uniforms(KEY, 4, 1000, (1 << 31) - 500) == want))


def test_gaussian_of_uniforms_is_jax_random_normal():
    assert bool(jnp.all(S.gaussian(S.uniforms(KEY, 3, 512)) == jax.random.normal(KEY, (512, 3), F32)))


@pytest.mark.parametrize("make", ["point", "spectrum", "cherenkov", "scint", "segment", "steps", "arrays"])
def test_a_run_is_the_same_in_any_chunks(make):
    src, p = SOURCES[make]()  # bitwise on the GPU; the CPU's vectorised loop tail may round a last bit differently
    whole = src.emit(p, KEY, 320, 0, 320)
    part = src.emit(p, KEY, 128, 128, 320)
    for a, b in zip(whole, part, strict=True):
        if jax.default_backend() == "gpu" or a.dtype == bool:
            assert bool(jnp.all(a[128:256] == b)), make
        else:
            np.testing.assert_allclose(np.asarray(a[128:256]), np.asarray(b), rtol=3e-7, atol=3e-7, err_msg=make)


def test_a_new_value_or_a_new_source_of_the_same_structure_never_recompiles():
    from lumen.sources._base import _EMIT

    S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0).emit(None, KEY, 96)
    size = _EMIT._cache_size()
    S.Point((1.0, 2.0, 3.0), n_photons=7.0, lam=450.0, time=3.0).emit(None, KEY, 96, 1000)
    p = S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0).parameters().set("source/x", np.array([4.0, 0.0, 0.0]))
    S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0).emit(p, jax.random.PRNGKey(9), 96)
    assert _EMIT._cache_size() == size


# ------------------------------------------------------------------------------------------------------- the point
def test_point_isotropic_and_beam():
    e = S.Point((1.0, 2.0, 3.0), n_photons=50.0, lam=420.0, time=4.0).emit(None, KEY, 20000)
    assert e.x.shape == (20000, 3) and bool(jnp.all(e.x == jnp.asarray([1.0, 2.0, 3.0], F32)))
    assert float(e.w.sum()) == pytest.approx(50.0, rel=1e-6) and bool(jnp.all((e.t == 4.0) & (e.lam == 420.0)))
    assert np.allclose(np.linalg.norm(np.asarray(e.d), axis=1), 1.0, atol=1e-6)
    for c in range(3):  # isotropic: <d_c> = 0, <d_c^2> = 1/3
        assert abs(z(e.d[:, c], 0.0)) < Z_MAX and abs(z(e.d[:, c] ** 2, 1 / 3)) < Z_MAX
    assert bool(jnp.all(e.drawn_direction))
    b = S.Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0, direction=(0.0, 0.6, 0.8)).emit(None, KEY, 10)
    assert bool(jnp.all(b.d == jnp.asarray([0.0, 0.6, 0.8], F32))) and not bool(jnp.any(b.drawn_direction))


def test_point_spectrum_moments_and_its_derivative():
    tab = Table(E_TAB, SLOW)
    src = S.Point((0.0, 0.0, 0.0), n_photons=10.0, spectrum=tab)
    e = src.emit(None, KEY, 1 << 16)
    assert bool(jnp.all(e.w == F32(10.0 / (1 << 16))))  # the nominal spectrum: weight exactly n / samples
    assert abs(z(1239.841984 / np.asarray(e.lam, np.float64), g4_mean_energy(E_TAB, SLOW))) < Z_MAX
    k = 3  # d E[photon energy] / d value[k] through the reweighted spectrum, against the closed form

    def per_photon(t):  # each photon's estimate of E[photon energy] and of the total photons, n = 10
        r = src.emit({"source/spectrum": Table(jnp.asarray(E_TAB, F32), jnp.asarray(SLOW, F32).at[k].add(t))}, KEY,
                     1 << 16)  # fmt: skip
        return r.w * (1 << 16) * 1239.841984 / r.lam / 10.0, r.w * (1 << 16)

    _, (dv, dtot) = jax.jvp(per_photon, (F32(0.0),), (F32(1.0),))
    h = 1e-4
    up, down = (g4_mean_energy(E_TAB, SLOW + s * h * (np.arange(7) == k)) for s in (1, -1))
    assert abs(z(dv, (up - down) / (2 * h))) < Z_MAX
    assert abs(z(dtot, 0.0)) < Z_MAX  # the total is the same for any spectrum


def test_point_parameters_and_values():
    src = S.Point((0.0, 0.0, 1.0), n_photons=5.0, lam=400.0)
    p = src.parameters()
    assert p.info("source/x").method == "moving_paths" and p.info("source/x").available
    assert p.info("source/n_photons").unit == "" and p.explain("source/time") == "source Point"
    p2 = p.set("source/n_photons", 8.0)
    assert float(src.emit(p2, KEY, 100).w.sum()) == pytest.approx(8.0, rel=1e-6)
    beam = S.Point((0.0, 0.0, 0.0), lam=400.0, direction=(0.0, 0.0, 1.0)).parameters()
    assert not beam.info("source/x").available  # a beam's position has no derivative
    with pytest.raises(ValueError):
        S.Point((0.0, 0.0, 0.0))


def test_derivatives_of_drawn_and_undrawn_sources():
    def total(x, direction):
        e = S.Point(x, n_photons=3.0, lam=400.0, direction=direction).emit(None, KEY, 64)
        return jnp.sum(e.x[:, 2] * e.w)

    x0 = jnp.asarray([0.0, 0.0, 1.0], F32)
    t = jnp.asarray([0.0, 0.0, 1.0], F32)
    assert float(jax.jvp(lambda x: total(x, None), (x0,), (t,))[1]) == pytest.approx(3.0, rel=1e-6)
    with pytest.raises(S.GradientNotAvailable):
        jax.jvp(lambda x: total(x, (0.0, 0.0, 1.0)), (x0,), (t,))
    # time and intensity are reweight parameters of every source: d t / d time = 1, d w / d n = w / n
    src = S.Point((0.0, 0.0, 0.0), n_photons=2.0, lam=400.0, direction=(1.0, 0.0, 0.0))
    dt = jax.jvp(lambda tm: src.emit({"source/time": tm}, KEY, 8).t, (F32(1.0),), (F32(1.0),))[1]
    dw = jax.jvp(lambda n: src.emit({"source/n_photons": n}, KEY, 8).w.sum(), (F32(2.0),), (F32(1.0),))[1]
    assert bool(jnp.all(dt == 1.0)) and float(dw) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------------------------------- Cherenkov
def cherenkov(n=1.4, beta=0.95, charge=1.0, length=0.2):
    src = S.CherenkovSegment("W", (0.0, 0.0, 0.0), (0.0, 0.6, 0.8), length, beta, charge, 1.0, 300.0, 600.0)
    return src, {"materials/W/RINDEX": Table(np.array([1.0, 6.0]), np.array([n, n]))}


def test_cherenkov_yield_cone_spectrum_and_polarisation():
    src, p = cherenkov(charge=2.0)
    e = src.emit(p, KEY, 1 << 15)
    want = 2 * math.pi * ALPHA * 4.0 * 0.2 * (1 - 1 / (0.95 * 1.4) ** 2) * (1 / 300.0 - 1 / 600.0) * 1e9  # Frank-Tamm
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(want, rel=1e-5)
    axis = np.array([0.0, 0.6, 0.8])
    np.testing.assert_allclose(np.asarray(e.d) @ axis, 1 / (0.95 * 1.4), rtol=2e-6)  # the cone
    assert abs(z(1.0 / np.asarray(e.lam, np.float64), 0.5 * (1 / 300.0 + 1 / 600.0))) < Z_MAX  # 1/lam^2 spectrum
    er = np.asarray(e.polarisation[:, 0])
    np.testing.assert_allclose(np.sum(er * np.asarray(e.d), 1), 0.0, atol=2e-6)  # transverse, in the track's plane
    np.testing.assert_allclose(np.sum(np.cross(np.asarray(e.d), axis) * er, 1), 0.0, atol=2e-6)
    s = np.asarray(e.x) @ axis  # uniform along the segment, at the time the charge passes there
    assert s.min() >= 0 and s.max() <= 0.2 + 1e-7 and abs(z(s, 0.1)) < Z_MAX
    np.testing.assert_allclose(np.asarray(e.t), 1.0 + s / (0.95 * 0.299792458), rtol=1e-6)
    assert not bool(jnp.any(e.drawn_direction))


def test_cherenkov_below_threshold_and_with_a_dispersive_index():
    src, p = cherenkov(beta=0.7)
    assert float(src.emit(p, KEY, 256).w.sum()) == 0.0
    src = S.CherenkovSegment("W", length=0.5, beta=0.99, lam_min=280.0, lam_max=650.0)
    e = src.emit({"materials/W/RINDEX": WATER}, KEY, 1 << 16)
    lam = np.linspace(280.0, 650.0, 200001)
    n = np.interp(1239.841984 / lam, WATER.energy, WATER.value)
    want = 2 * math.pi * ALPHA * 0.5 * np.trapezoid(np.maximum(1 - 1 / (0.99 * n) ** 2, 0) / lam**2, lam) * 1e9
    w = np.asarray(e.w, np.float64)
    assert abs((w.sum() - want) / (w.std() * math.sqrt(len(w)))) < Z_MAX


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
    src = S.ScintillationPoint.of(p, "LS", x=(0.1, 0.2, 0.3), energy=0.5, time=2.0)
    e = src.emit(p, KEY, 1 << 16)
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(4000.0 * 0.5, rel=1e-5)  # exact per photon
    t = np.asarray(e.t, np.float64) - 2.0
    fast = t < 60.0  # the fast (6 ns) and slow (1500 ns) components separate here up to 4 % of the slow
    assert abs(z(fast, 0.3 + 0.7 * (1 - math.exp(-60 / 1500)))) < Z_MAX
    assert abs(z(t, 0.3 * 6.0 + 0.7 * 1500.0)) < Z_MAX
    e_fast = 1239.841984 / np.asarray(e.lam, np.float64)[t < 1.0]  # nearly all fast: its spectrum
    assert abs(z(e_fast, g4_mean_energy(E_TAB, FAST))) < Z_MAX + 1.0
    assert bool(jnp.all(e.drawn_direction)) and bool(jnp.all(e.x == jnp.asarray([0.1, 0.2, 0.3], F32)))
    d0 = jax.jvp(lambda y: src.emit({**p, "materials/LS/SCINTILLATIONYIELD1": y}, KEY, 4096).w.sum(),
                 (F32(0.3),), (F32(1.0),))[1]  # fmt: skip
    assert abs(float(d0)) < 0.05 * 2000.0  # the total does not depend on the share


def test_scintillation_segment_birks_and_positions():
    p = scint_params()
    a, b = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.003, 0.004])
    src = S.ScintillationSegment.of(p, "LS", start=a, end=b, t_start=1.0, t_end=1.05, energy=0.9, birks=1.3e-4)
    e = src.emit(p, KEY, 1 << 14)
    want = 4000.0 * 0.9 / (1 + 1.3e-4 * 0.9 / 0.005)
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(want, rel=1e-5)
    f = np.asarray(e.x)[:, 1] / 0.003
    assert f.min() >= 0 and f.max() <= 1 and abs(z(f, 0.5)) < Z_MAX
    with pytest.raises(KeyError):
        S.ScintillationPoint.of({}, "LS")


def test_scintillation_reads_geant4_10_keys():
    m = "materials/LS/"
    p = {m + "SCINTILLATIONYIELD": 100.0, m + "YIELDRATIO": 0.8, m + "FASTCOMPONENT": Table(E_TAB, FAST),
         m + "SLOWCOMPONENT": Table(E_TAB, FAST), m + "FASTTIMECONSTANT": 2.0,
         m + "SLOWTIMECONSTANT": 20.0}  # fmt: skip
    src = S.ScintillationPoint.of(p, "LS", energy=2.0, geant4="10.7")
    t = np.asarray(src.emit(p, KEY, 1 << 15).t, np.float64)
    assert abs(z(t, 0.8 * 2.0 + 0.2 * 20.0)) < Z_MAX


# ---------------------------------------------------------------------------------------------- the step dump
def steps(light):
    rng_ = np.random.default_rng(1)
    a = rng_.normal(size=(5, 3))
    d = rng_.normal(size=(5, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    length = np.array([0.01, 0.02, 0.005, 0.03, 0.015])
    beta = np.array([0.99, 0.9, 0.6, 0.95, 0.8])  # one below threshold
    f = dict(start=a, end=a + length[:, None] * d, t_start=np.arange(5.0), t_end=np.arange(5.0) + 0.1,
             beta_start=beta + 0.002, beta_end=beta - 0.002, charge=np.array([1.0, -1.0, 1.0, 2.0, 1.0]),
             edep=np.array([2e6, 1e6, 3e6, 5e5, 1.5e6]), lam_min=300.0, lam_max=600.0)  # fmt: skip
    p = {"materials/W/RINDEX": Table(np.array([1.0, 6.0]), np.array([1.4, 1.4]))} | scint_params()
    return S.Geant4Steps.of(p, "W" if light == "cherenkov" else "LS", light, **f, birks=1e-4), p, f, d, beta


def test_geant4_steps_cherenkov_yield_and_cone():
    src, p, f, d, beta = steps("cherenkov")
    e = src.emit(p, KEY, 1 << 16)
    length = np.linalg.norm(f["end"] - f["start"], axis=1)
    per = 2 * math.pi * ALPHA * f["charge"] ** 2 * length * np.maximum(1 - 1 / (beta * 1.4) ** 2, 0)
    per = per * (1 / 300 - 1 / 600) * 1e9
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(per.sum(), rel=1e-5)  # n constant: tot / M each
    cosines = np.asarray(e.d) @ d.T  # each photon on its step's cone
    assert np.all(np.min(np.abs(cosines - 1 / (beta * 1.4)), 1) < 3e-6)
    assert not bool(jnp.any(e.drawn_direction))


def test_geant4_steps_scintillation_yield():
    src, p, f, _, _ = steps("scintillation")
    e = src.emit(p, KEY, 1 << 16)
    length = np.linalg.norm(f["end"] - f["start"], axis=1)
    mev = f["edep"] * 1e-6
    want = np.sum(4000.0 * mev / (1 + 1e-4 * mev / length))
    assert float(np.asarray(e.w, np.float64).sum()) == pytest.approx(want, rel=1e-5)  # every photon carries tot / M
    assert bool(jnp.all(e.drawn_direction))


# ------------------------------------------------------------------------------------------------------- arrays
def test_from_arrays_rows_and_its_derivative_rule():
    n = 40
    x = jnp.arange(3 * n, dtype=F32).reshape(n, 3)
    d = jnp.tile(jnp.asarray([0.0, 0.0, 1.0], F32), (n, 1))
    src = S.FromArrays(x, d, jnp.zeros(n), jnp.full(n, 400.0), jnp.ones(n))
    e = src.emit(None, KEY, 10, 5)
    assert bool(jnp.all(e.x == x[5:15])) and not bool(jnp.any(e.drawn_direction))
    with pytest.raises(S.GradientNotAvailable):
        jax.jvp(lambda xv: src.emit({"source/x": xv}, KEY, n).x, (x,), (jnp.ones_like(x),))
    drawn = S.FromArrays(x, d, jnp.zeros(n), jnp.full(n, 400.0), jnp.ones(n), drawn=True)
    tx = jax.jvp(lambda xv: drawn.emit({"source/x": xv}, KEY, n).x, (x,), (jnp.ones_like(x),))[1]
    assert bool(jnp.all(tx == 1.0)) and bool(jnp.all(drawn.emit(None, KEY, n).drawn_direction))


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
