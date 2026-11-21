"""The sensor response as readout of the deposit: a joint acceptance table, a feature-indexed delay, the time spread in
the profile, the static taps built once, a stage of your own, and derivatives through the whole chain."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import readout_toys as toys
from jax.experimental import enable_x64

import lumen
from lumen import readout as rd
from lumen.readout import detection
from lumen.readout import grid as tg
from lumen.sensors import components as comp

COS = [0.0, 0.3, 0.7, 1.0]
LAM = [120.0, 300.0, 450.0]
ACC = [[0.0, 0.1, 0.05], [0.2, 0.4, 0.1], [0.5, 0.6, 0.3], [0.7, 0.8, 0.35]]  # (cos_incidence, wavelength)


def _tile(**over):
    """A flat tile's value set: no QE or CE, one joint acceptance over (cos_incidence, wavelength)."""
    return toys.model(qe=None, ce=None, acceptance=dict(cos_incidence=COS, wavelength_nm=LAM, values=ACC), **over)


def _photons(n, n_sensors=3, seed=0, t0=20.0):
    """Photons on a tile (normal +z) from random directions, wavelengths across both bands."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.05, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    s = np.sqrt(1.0 - c * c)
    d = np.c_[s * np.cos(phi), s * np.sin(phi), -c]  # into the tile
    return dict(sensor=rng.integers(0, n_sensors, n), lam=rng.uniform(100.0, 480.0, n), time=np.full(n, t0),
                pos=np.c_[rng.uniform(-0.1, 0.1, (n, 2)), np.zeros(n)], dir=d, normal=np.tile([0.0, 0.0, 1.0], (n, 1)),
                weight=rng.uniform(0.5, 1.5, n), cos=c)  # fmt: skip


def _bilinear(c, lam):
    """The brute-force reference: the acceptance table's bilinear interpolation, constant beyond its end knots."""
    out = np.empty(len(c))
    for k, (ci, li) in enumerate(zip(c, lam, strict=True)):
        col = np.array([np.interp(li, LAM, row) for row in ACC])
        out[k] = np.interp(ci, COS, col)
    return out


def test_a_joint_acceptance_table_is_its_bilinear_interpolation():
    """pe over a (cos_incidence x wavelength) table equals the per-photon bilinear interpolation, and truth rows'
    detection() gives the same probability as the binned readout."""
    m = _tile()
    assert [a.name for a in m.axes] == ["cos_incidence", "wavelength"] and m.reads == {"dir", "normal"}
    ph = _photons(3000, seed=4)
    h = tg.from_photons(m, (0.0, 120.0), 3, ph, group="pmt")
    assert h.charge.shape == h.charge2.shape == (3, 4, 3)
    p = toys.params(m)
    want_each = p["sensors/pmt/efficiency"][ph["sensor"]] * _bilinear(ph["cos"], ph["lam"])
    np.testing.assert_allclose(rd.pe(h, p), np.bincount(ph["sensor"], ph["weight"] * want_each, 3), rtol=2e-6)
    v = detection.values(h, p)
    rows = {k: jnp.asarray(ph[k], jnp.float32) for k in ("lam", "pos", "dir", "normal")}
    got = m.detection(v, jnp.asarray(ph["sensor"]), rows)
    np.testing.assert_allclose(got, want_each, rtol=2e-5, atol=1e-7)


def test_existing_qe_ce_sets_are_two_one_axis_tables():
    """Today's value set (QE over wavelength, CE over the cathode angle): two tables; the charge histogram's cells and
    flat indices are today's (wavelength x angle), and pe is N eta."""
    m = toys.model()
    assert m.tables == (("qe", ("wavelength",)), ("ce", ("theta",))) and m.reads == {"pos"}
    assert m.grid((0.0, 100.0)).shape == (4, 4)


def _banded(**over):
    """The toy with a delay per wavelength band: VUV (< 200 nm) through a 0-12 ns table, blue a point mass at 0."""
    edges = [0.0, 0.0, 2.0, 4.0, 8.0, 12.0]
    pdf = [[0.0, 0.4, 0.3, 0.2, 0.1], [1.0, 0.0, 0.0, 0.0, 0.0]]
    qe = dict(wavelength_nm=[100.0, 300.0, 500.0], values=[0.2, 0.2, 0.2])
    return toys.model(qe=qe, ce=None, delay=dict(form="table", edges_ns=edges, band_edges_nm=[200.0], pdf=pdf),
                      tts=dict(form="gaussian", sigma_ns=0.8), **over)  # fmt: skip


def _one(lam, t, w=10.0):
    return dict(sensor=np.zeros(1, int), lam=np.full(1, lam), time=np.full(1, t), weight=np.full(1, w))


def test_the_delay_is_per_band_and_a_zero_width_bin_is_no_delay():
    """VUV light is delayed by its band's table (the profile's mean moves by the table's mean), blue light by none
    (a zero-width first bin is a point mass at 0): its profile is that of a model without a delay."""
    m = _banded()
    assert m.delay.fields()["pdf"][1] == (2, 5) and m.grid((0.0, 100.0)).delay_shape == (2,)
    p = toys.params(m, 1)
    p["sensors/pmt/dark_rate"] = np.zeros(1, np.float32)
    gate = (0.0, 100.0)
    nodelay = toys.model(qe=dict(wavelength_nm=[100.0, 300.0, 500.0], values=[0.2, 0.2, 0.2]), ce=None,
                         tts=dict(form="gaussian", sigma_ns=0.8))  # fmt: skip
    pn = toys.params(nodelay, 1)
    for lam, mean in ((128.0, 0.4 * 1.0 + 0.3 * 3.0 + 0.2 * 6.0 + 0.1 * 10.0), (420.0, 0.0)):
        h = tg.from_photons(m, gate, 1, _one(lam, 30.0), group="pmt")
        prof = np.asarray(rd.pe_density(h, p, dark=False).rate)[0]
        t = gate[0] + h.grid.dt * (np.arange(prof.size) + 0.5)
        np.testing.assert_allclose(prof.sum(), float(rd.pe(h, p)[0]), rtol=1e-4)
        np.testing.assert_allclose((prof * t).sum() / prof.sum(), 30.0 + 5.0 + mean, atol=2e-3)
        if lam > 200.0:
            hn = tg.from_photons(nodelay, gate, 1, _one(lam, 30.0), group="pmt")
            np.testing.assert_allclose(
                prof, np.asarray(rd.pe_density(hn, pn, dark=False).rate)[0], atol=1e-5 * prof.max()
            )


def test_the_delay_can_key_on_the_re_emission_count():
    """A delay over the generation feature: light shifted in transport (gen >= 1) takes the second table."""
    m = toys.model(ce=None, delay=dict(form="table", edges_ns=[0.0, 0.0, 10.0], generation_edges=[0.5],
                                       pdf=[[1.0, 0.0], [0.0, 1.0]]))  # fmt: skip
    assert m.reads == {"gen"}
    p = toys.params(m, 1)
    means = []
    for gen in (0, 2):
        h = tg.from_photons(m, (0.0, 100.0), 1, _one(400.0, 30.0) | dict(gen=np.full(1, gen)), group="pmt")
        prof = np.asarray(rd.pe_density(h, p, dark=False).rate)[0]
        t = h.gate[0] + h.grid.dt * (np.arange(prof.size) + 0.5)
        means.append((prof * t).sum() / prof.sum())
    np.testing.assert_allclose(np.diff(means), 5.0, atol=2e-3)  # the uniform 0-10 ns delay's mean


def test_the_time_spread_is_in_the_profile():
    """The light-only density of light at one instant is the photoelectrons in each bin of N(t + t_offset, TTS)."""
    from scipy import stats

    m = toys.model(ce=None, tts=dict(form="gaussian", sigma_ns=2.0))
    p = toys.params(m, 1)
    h = tg.from_photons(m, (0.0, 100.0), 1, _one(400.0, 41.3), group="pmt")
    prof = np.asarray(rd.pe_density(h, p, dark=False).rate)[0]
    e = h.gate[0] + h.grid.dt * np.arange(prof.size + 1)
    want = float(rd.pe(h, p)[0]) * np.diff(stats.norm.cdf(e, 41.3 + 5.0, 2.0))
    np.testing.assert_allclose(prof, want, atol=1e-4 * want.max())
    assert prof.max() < 0.25 * want.sum()  # spread over several bins, not a delta


def test_the_delay_taps_are_built_once_with_the_grid():
    """The static taps are built when the grid is made and never again: no host work per readout call."""
    comp.delay_taps.cache_clear()
    comp._device_taps.cache_clear()
    m = _banded()
    h = tg.from_photons(m, (0.0, 100.0), 1, _one(128.0, 30.0), group="pmt")
    assert comp.delay_taps.cache_info().misses == 1
    p = toys.params(m, 1)
    for k in range(3):
        rd.pe_density(h, p | {"sensors/pmt/delay/pdf": np.asarray(p["sensors/pmt/delay/pdf"]) * (1.0 + k)}, dark=False)
        rd.expected_hits(h, p)
    assert comp.delay_taps.cache_info().misses == 1 and comp._device_taps.cache_info().misses == 1


def test_a_stage_of_your_own_composes_lumens_stages():
    """A response written outside lumen: the arrival-weighted mean time of the delayed light, from the library stages,
    with a parameter of its own; differentiable in it and in every sensor value."""
    from lumen.params import Param

    @lumen.response(group="pmt", params={"t_ref": Param("ns", (0.0, 100.0), default=10.0)})
    def late_light(h, v):
        vv = detection.values(h, v)
        prof = detection.density(h, vv)
        t = h.gate[0] + h.grid.dt * (jnp.arange(prof.shape[-1]) + 0.5)
        return jnp.sum(prof * jax.nn.sigmoid(t - v["t_ref"]))

    m = _banded()
    h = tg.from_photons(m, (0.0, 100.0), 1, _one(128.0, 30.0), group="pmt")
    p = toys.params(m, 1) | {"sensors/pmt/t_ref": np.float32(40.0)}  # its key, as det.with_responses adds it
    assert float(late_light(h, p)) > 0.0
    g = jax.grad(lambda q: late_light(h, q))(p)
    assert float(g["sensors/pmt/t_ref"]) < 0.0 and np.all(np.isfinite(np.asarray(g["sensors/pmt/delay/pdf"])))


def test_derivatives_of_the_new_forms_match_finite_differences():
    """Forward and reverse derivatives against central differences in float64 for the joint acceptance's knots, the
    per-band delay table, the time spread and the SPE charge, through pe, the profile and the hit likelihood."""
    m = toys.model(qe=None, ce=None, acceptance=dict(cos_incidence=COS, wavelength_nm=LAM, values=ACC),
                   delay=dict(form="table", edges_ns=[0.0, 0.0, 2.0, 6.0], band_edges_nm=[200.0],
                              pdf=[[0.0, 0.6, 0.4], [1.0, 0.0, 0.0]]))  # fmt: skip
    ph = _photons(300, seed=2)
    ph["time"] = 30.0 + 2.0 * np.random.default_rng(3).standard_normal(300)
    h = tg.from_photons(m, (0.0, 120.0), 3, ph, group="pmt")
    p0 = toys.params(m)
    ev = rd.sample(h, p0, seed=2)
    data = rd.Hits.from_list(ev.hits["sensor"], ev.hits["t"], ev.hits["q"], 3)
    with enable_x64():
        h64 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), h)
        p64 = {k: jnp.asarray(v, jnp.float64) for k, v in p0.items()}

        def f(p):
            light = rd.pe_density(h64, p, dark=False).rate
            return rd.hits_nll(rd.expected_hits(h64, p), data) + jnp.sum(light**2) + jnp.sum(rd.pe(h64, p))

        f = jax.jit(f)
        g = jax.jit(jax.grad(f))(p64)
        rng = np.random.default_rng(7)
        for k in ("sensors/pmt/acceptance", "sensors/pmt/delay/pdf", "sensors/pmt/tts/sigma",
                  "sensors/pmt/charge/sigma"):  # fmt: skip
            x = np.asarray(p64[k])
            d = rng.standard_normal(x.shape) * (np.abs(x) * 0.5 + 0.05)
            fwd = float(
                jax.jvp(f, (p64,), ({kk: jnp.zeros_like(vv) for kk, vv in p64.items()} | {k: jnp.asarray(d)},))[1]
            )
            rev = float(np.sum(np.asarray(g[k]) * d))
            hs = 1e-6

            def at(s, k=k, x=x, d=d):
                return float(f(p64 | {k: jnp.asarray(x + s * d)}))

            fd = (8 * (at(hs) - at(-hs)) - (at(2 * hs) - at(-2 * hs))) / (12 * hs)
            tol = 1e-5 * max(1.0, abs(fd)) + 1e-6 * abs(float(f(p64)))
            assert abs(fwd - fd) < tol and abs(rev - fd) < tol, (k, fwd, rev, fd)


@pytest.fixture(scope="module")
def two_groups(tmp_path_factory):
    from test_api import _two_groups

    return lumen.load(_two_groups(tmp_path_factory.mktemp("resp")))


def test_a_transport_derivative_through_the_whole_chain(two_groups):
    """d/d(absorption length scale) of the profile and pe of both groups after tracking (forward mode through the run)
    against a central difference on the same seed with the draws held at the reference (``reference=``: the light is
    then smooth in the value): the readout chain carries the transport derivative."""
    det = two_groups
    p = det.params()
    key = "materials/Water/ABSLENGTH"
    v0 = p[key]
    src = lumen.sources.Point((0.0, 0.0, 0.3), n_photons=2000.0, lam=400.0)  # in the toy's water

    def out(th):
        from lumen.format.calibrations import Table

        q = p.set(key, Table(jnp.asarray(v0.energy), jnp.asarray(v0.value) * (1.0 + th)))
        light = lumen.simulate(det, src, q, samples=4096, seed=3, gate=(0.0, 100.0), reference=p)
        total = 0.0
        for g in ("near", "far"):
            prof = rd.pe_density(light.group(g), q, dark=False).rate
            total = total + jnp.sum(prof * jnp.linspace(0.5, 1.5, prof.shape[-1])) + jnp.sum(rd.pe(light.group(g), q))
        return total

    tan = float(jax.jvp(out, (jnp.float32(0.0),), (jnp.float32(1.0),))[1])
    hs = 1e-2
    fd = (float(out(jnp.float32(hs))) - float(out(jnp.float32(-hs)))) / (2 * hs)
    assert tan != 0.0 and abs(tan - fd) < 2e-2 * abs(fd) + 1e-6, (tan, fd)


def test_the_re_emission_count_reaches_the_deposit(tmp_path):
    """Transport counts each photon's re-emissions (a bulk WLS event in the toy's slab) into the deposit's ``gen``
    when a feature reads it: VUV light the slab shifts reaches the counting slabs with gen >= 1 as well as 0, blue
    light (which the slab does not absorb) only with gen 0."""
    import dataclasses

    from film_toy import RECIPE, film_gdml

    from lumen import run as runner
    from lumen.compare import photons
    from lumen.readout import fill
    from lumen.sensors import features

    src = tmp_path / "film.gdml"
    src.write_text(film_gdml((((0, 0, 0), None),)))
    det = lumen.load(lumen.convert(src, RECIPE, tmp_path / "film.lumen"))
    s = photons.setup(det)
    model = dataclasses.replace(s.model, local=("gen",))
    grid = tg.Grid(axes=(tg.Axis("generation", features.generation, edges=(-0.5, 0.5, 10.5)),))
    assert grid.reads == {"gen"}
    loop = fill.readout((fill.HistogramLayout(s.n_sensors, grid),))
    ops = photons.operands(dataclasses.replace(s, model=model), photons.packed(det))
    k = jax.random.PRNGKey(5)  # the loop's key operand: the raw key data
    key = jax.random.key_data(k) if jnp.issubdtype(k.dtype, jax.dtypes.prng_key) else k
    n = 1 << 12
    got = {}
    for lam in (128.0, 400.0):
        rows = lumen.sources.Point((0.3, 0.0, 0.0), n_photons=float(n), lam=lam).emit(None, jax.random.PRNGKey(1), n)
        t = runner.run(loop, model, ops, rows, key)
        got[lam] = np.asarray(t.outputs["charge:0"], np.float64).reshape(s.n_sensors, 2).sum(0)
    assert got[128.0][1] > 0.05 * got[128.0].sum() and got[400.0][1] == 0.0 and got[400.0][0] > 0.0, got
