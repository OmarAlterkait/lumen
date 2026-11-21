"""lumen.tools: named keys as a vector, the Fisher information, the fit, the losses and the scans, on a closed-form
model over a toy detector's parameters (no photon loop), and one fit through simulate.

The closed-form model: expected counts mu_i = A_i R^a_i exp(-L_i / ABS_k(i)) (1 + c_i sigma_alpha), from the toy's
REFLECTIVITY table (R, two knots), ABSLENGTH table and sigma_alpha, plus per-sensor efficiencies (the nuisances)."""

import jax.numpy as jnp
import numpy as np
import pytest
from test_api import INSIDE, converted  # noqa: F401 -- the toy detector, converted once

import lumen
from lumen import tools
from lumen.format.calibrations import Table

R, ABS, SA = "surfaces/Liner/REFLECTIVITY", "materials/Water/ABSLENGTH", "surfaces/Liner/sigma_alpha"
AMP = jnp.asarray([400.0, 900.0, 250.0, 1300.0, 700.0, 520.0], jnp.float32)  # six bins
POW = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 3.0], [0.5, 0.5], [0.0, 0.0]], jnp.float32)
DIST = jnp.asarray([1.0, 2.0, 0.5, 1.5, 3.0, 2.5], jnp.float32)
KNOT = jnp.asarray([0, 1, 0, 1, 0, 1])  # the ABSLENGTH knot each bin's path is in


def model(p):
    r, a = p[R].value, p[ABS].value
    return AMP * jnp.prod(r**POW, axis=1) * jnp.exp(-DIST / a[KNOT]) * (1.0 + 3.0 * p[SA] * (1.0 + POW[:, 1]))


@pytest.fixture(scope="module")
def p0(converted):  # noqa: F811
    return lumen.load(converted).params()


def test_a_space_is_its_keys_values_transformed_by_their_range(p0):
    s = tools.Space(p0, [R, ABS, SA], scales={SA: 0.1})
    assert s.sizes == (2, 2, 1) and s.log(R) and s.log(ABS) and not s.log(SA)
    x = s.vector()
    np.testing.assert_allclose(x[:2], np.log(p0[R].value)) and np.testing.assert_allclose(x[4], p0[SA] / 0.1)
    back = s.params(x + np.r_[0.1, 0.0, 0.0, 0.0, 0.5])
    assert isinstance(back[R], Table) and np.array_equal(back[R].energy, p0[R].energy)
    np.testing.assert_allclose(back[R].value[0], p0[R].value[0] * np.exp(0.1), rtol=1e-6)
    np.testing.assert_allclose(back[SA], p0[SA] + 0.05, rtol=1e-6)
    with pytest.raises(ValueError, match="a scale is for a key"):
        tools.Space(p0, [R], scales={R: 2.0})


def test_the_jacobian_is_one_forward_derivative_per_entry(p0):
    import jax

    s = tools.Space(p0, [R, ABS, SA])
    x = s.vector()
    J, mu = tools.jacobian(model, s, x)
    want = np.asarray(jax.jacfwd(lambda v: model(s.params(v)))(jnp.asarray(x, jnp.float32)))
    assert J.shape == (6, 5) and np.allclose(J, want, rtol=1e-5) and np.allclose(mu, np.asarray(model(p0)), rtol=1e-6)


def test_the_fit_recovers_the_values_its_data_were_made_at(p0):
    s = tools.Space(p0, [R, SA], scales={SA: 0.1})
    truth = s.vector() + np.r_[-0.15, 0.1, 0.4]
    data = np.asarray(model(s.params(truth)), np.float64)  # the expectation: no noise, an exact optimum
    res = tools.fit(model, s, data, steps=50)
    np.testing.assert_allclose(res.x, truth, atol=1e-3)  # the model runs in float32
    assert res.history and res.history[-1]["nll"] <= 1e-6 * data.sum()
    # the covariance is the inverse Fisher information at the fit; its standard errors in the keys' own units
    F = tools.fisher(model, s, res.x)
    np.testing.assert_allclose(res.cov, np.linalg.inv(F), rtol=1e-3)
    assert set(res.sigma) == {R, SA} and np.shape(res.sigma[R]) == (2,) and res.sigma[SA] > 0


def test_eliminating_a_nuisance_is_the_schur_complement(p0):
    s = tools.Space(p0, [R, ABS, SA])
    x = s.vector()
    full = tools.fisher(model, s, x)
    reduced = tools.fisher(model, s, x, per_sensor=(ABS,))
    keep = [0, 1, 4]
    # the marginal covariance of the kept entries is the inverse of the Schur complement
    np.testing.assert_allclose(np.linalg.inv(reduced), np.linalg.pinv(full)[np.ix_(keep, keep)], rtol=1e-3)
    with pytest.raises(ValueError, match="not keys of the space"):
        tools.fisher(model, s, x, per_sensor=("sensors/pmt/entry_efficiency",))


def test_the_losses():
    mu, n = jnp.asarray([2.0, 5.0, 0.5]), jnp.asarray([2.0, 5.0, 0.0])
    assert float(tools.poisson_nll(mu, n)) == pytest.approx(0.5)  # 0 where n = mu; mu where n = 0


def test_a_scan_gives_the_loss_its_derivatives_and_their_differences(p0):
    s = tools.Space(p0, [R, ABS, SA])
    data = np.asarray(model(p0), np.float64) * 1.05

    def loss(p):
        return tools.poisson_nll(model(p), data)

    one = tools.scan(loss, s, {SA: np.linspace(0.05, 0.3, 201)})
    assert one.loss.shape == (201,) and one.grad.shape == (1, 201)
    scale = np.max(np.abs(one.grad))  # the central difference's truncation (h^2 / 6 times the third derivative)
    np.testing.assert_allclose(one.grad[0, 1:-1], one.numerical_grad[0, 1:-1], atol=2e-3 * scale)
    spike = "sensors/pmt/entry_efficiency"  # the model does not read it: its derivative is 0
    two = tools.scan(loss, tools.Space(p0, [R, SA, spike]), {SA: [0.1, 0.2, 0.3], spike: [0.5, 0.9]})
    assert two.loss.shape == (3, 2) and two.grad.shape == (2, 3, 2) and np.all(two.grad[1] == 0.0)
    with pytest.raises(ValueError, match="single-number key"):
        tools.scan(loss, s, {R: [0.1, 0.2]})


@pytest.mark.slow  # minutes on 4 CPUs (measured 2.5-9.5 min on roma, 4 workers)
def test_a_fit_through_simulate_recovers_the_reflectivity(converted):  # noqa: F811
    det = lumen.load(converted)
    src = lumen.sources.Point(INSIDE, n_photons=1e4, lam=400.0)
    s = tools.Space(det.params(), [R])

    def light(p):
        return lumen.simulate(det, src, p, samples=4096, seed=1, backend="xla").arrivals

    truth = s.vector() + np.log(0.8)
    data = np.asarray(light(s.params(truth)), np.float64)  # the same seed: the model's own expectation
    res = tools.fit(light, s, data, steps=30)
    # monochromatic light reads the table at 400 nm only: that value is what the data fix (the two knots are not)
    from lumen.units import HC

    energy = np.asarray(det.params()[R].energy)
    at = lambda x: np.interp(HC / 400.0, energy, np.exp(x))  # noqa: E731
    assert at(res.x) == pytest.approx(at(truth), rel=2e-3)
    assert res.history[-1]["nll"] <= 1e-4 * data.sum()


def test_a_factor_key_is_one_number_on_its_whole_value(p0):
    s = tools.Space(p0, [SA], factors=[ABS])
    assert s.keys == (SA, ABS) and s.sizes == (1, 1) and s.log(ABS)
    np.testing.assert_allclose(s.vector()[1], 0.0, atol=1e-12)  # factor 1 at the space's own values
    back = s.params(np.r_[s.vector()[0], np.log(1.25)])
    np.testing.assert_allclose(back[ABS].value, 1.25 * np.asarray(p0[ABS].value), rtol=1e-6)
    np.testing.assert_allclose(s.vector(back)[1], np.log(1.25), rtol=1e-6)
    assert s.values(np.r_[0.2, np.log(1.25)])[ABS] == pytest.approx(1.25)
    assert s.sigma(np.diag([0.0, 0.01]), np.r_[0.0, np.log(2.0)])[ABS] == pytest.approx(0.2)  # at x: 2 x 0.1
    with pytest.raises(ValueError, match="positive factor"):
        s.vector(p0.set(ABS, Table(p0[ABS].energy, np.asarray(p0[ABS].value) * [1.0, 2.0])))


def test_first_and_second_order_reach_the_same_fit_and_record_their_paths(p0):
    s = tools.Space(p0, [SA], scales={SA: 0.1}, factors=[R, ABS])
    truth = s.vector() + np.r_[0.4, -0.1, 0.15]
    data = np.asarray(model(s.params(truth)), np.float64)
    second = tools.fit(model, s, data, steps=50)
    first = tools.descend(model, s, data, steps=400, lr=0.02)
    for r in (first, second):  # each by its own convergence criterion (the Newton decrement), then to one tolerance
        assert r.converged, r.reason
        np.testing.assert_allclose(r.x, truth, atol=1e-3)
    for r in (first, second):
        assert np.array_equal(r.path[0], s.vector()) and np.array_equal(r.path[-1], r.x)
        assert len(r.path) == len(r.history) + 1 and r.nll == pytest.approx(float(tools.poisson_nll(
            np.asarray(model(r.params)), data)), rel=1e-5, abs=1e-6)  # fmt: skip
    held = tools.fit(model, s, data, steps=50, fixed=(R,))
    assert held.x[1] == s.vector()[1] and held.nll > second.nll
    with pytest.raises(ValueError, match="fixed keys"):
        tools.fit(model, s, data, fixed=("sensors/pmt/entry_efficiency",))


def test_the_profile_is_lowest_at_the_fit_and_toys_cover_the_truth(p0):
    s = tools.Space(p0, [SA], scales={SA: 0.1}, factors=[R])
    truth = s.vector() + np.r_[0.3, -0.1]
    rng = np.random.default_rng(3)
    data = rng.poisson(50.0 * np.asarray(model(s.params(truth)), np.float64))
    m50 = lambda p: 50.0 * model(p)  # noqa: E731 -- brighter, so the likelihood is close to Gaussian
    res = tools.fit(m50, s, data, steps=50)
    vals = res.sigma[R] * np.linspace(-2, 2, 9) + s.values(res.x)[R]
    pr = tools.profile(m50, s, data, R, vals, x0=res.x, steps=20)
    assert np.argmin(pr.nll) == 4 and pr.nll[4] == pytest.approx(res.nll, abs=3e-3)  # float32 sums
    np.testing.assert_allclose(pr.nll[[0, 8]] - pr.nll[4], 2.0, rtol=0.25)  # 2 sigma: a rise of 2
    cov = tools.coverage(m50, s, truth, toys=40, seed=5, steps=20)
    assert cov.pulls.shape == (40, 2) and np.all(np.abs(cov.pulls.mean(0)) < 0.6)
    assert np.all((cov.covered > 0.45) & (cov.covered < 0.9))


def test_the_plots_draw(p0, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from lumen.tools import plot

    s = tools.Space(p0, [SA], scales={SA: 0.1}, factors=[R])
    truth = s.vector() + np.r_[0.2, -0.1]
    data = np.asarray(model(s.params(truth)), np.float64)
    loss = lambda p: tools.poisson_nll(model(p), data)  # noqa: E731
    fits = {"first": tools.descend(model, s, data, steps=5), "second": tools.fit(model, s, data, steps=5)}
    grid = tools.scan(loss, s, {SA: np.linspace(0.05, 0.3, 4), R: np.linspace(0.8, 1.05, 5)})
    figs = [plot.scan(grid, space=s, truth=truth, paths=fits), plot.scan(tools.scan(loss, s, {SA: [0.1, 0.2, 0.3]})),
            plot.convergence(s, fits, truth), plot.errors(s, fits, truth, fits["second"].cov),
            plot.profile(tools.profile(model, s, data, SA, [0.1, 0.2], steps=2), fit=fits["second"], space=s),
            plot.pulls(tools.coverage(model, s, truth, toys=3, steps=2), s)]  # fmt: skip
    pos = np.random.default_rng(0).normal(size=(50, 3))
    figs.append(plot.display(pos, np.arange(50.0), log=True).figure)
    figs.append(plot.event(pos, np.arange(50.0) + 1, np.linspace(0, 10, 50)))
    for i, f in enumerate(figs):
        f.savefig(tmp_path / f"{i}.png")
    u, v = plot.sensor_map([[1.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, -3.0], [-1.0, 0.0, 0.0]])  # long axis z
    np.testing.assert_allclose(v, [0.0, 1.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(u[[0, 3]], [0.0, np.pi], atol=1e-12)


def test_a_fit_reports_convergence_a_singular_information_and_refuses_impossible_data(p0):
    """No silent ends: a fit stopped short says so; an information singular along a direction (only the mean of the
    two REFLECTIVITY knots enters) gives its rank and infinite variance on the entries it reaches, not a pseudo-inverse;
    a count where the model expects none is impossible data (the Poisson likelihood is 0 there), refused."""
    s = tools.Space(p0, [SA], scales={SA: 0.1}, factors=[R])
    truth = s.vector() + np.r_[0.3, -0.1]
    data = np.asarray(model(s.params(truth)), np.float64)
    short = tools.fit(model, s, data, steps=1)
    assert not short.converged and "1 steps taken" in short.reason and short.rank == 2
    full = tools.fit(model, s, data, steps=50)
    assert full.converged and full.rank == 2 and np.all(np.isfinite(full.cov)), full.reason
    s2 = tools.Space(p0, [R, SA], scales={SA: 0.1})

    def mean_knots(p):
        r = jnp.mean(p[R].value)
        return AMP * r ** POW[:, 0] * (1.0 + 3.0 * p[SA] * (1.0 + POW[:, 1]))

    res = tools.fit(mean_knots, s2, np.asarray(mean_knots(s2.params(s2.vector())), np.float64), steps=5)
    assert res.rank == 2 and np.all(np.isinf(np.diag(res.cov)[:2])) and np.isfinite(res.cov[2, 2])
    assert np.all(np.isinf(res.sigma[R])) and np.isfinite(res.sigma[SA])
    with pytest.raises(ValueError, match="mu = 0, n > 0"):
        tools.fit(lambda p: jnp.concatenate([model(p), jnp.zeros(1)]), s, np.r_[data, 3.0])
