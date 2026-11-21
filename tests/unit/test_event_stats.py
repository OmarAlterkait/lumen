"""The multi-event comparison's statistics: unbiased with a noisy lumen, and they see a real difference."""

import numpy as np
import pytest

from lumen.compare import event


def _sample(rng, true, noise, scale=1.0):
    """Two lumen halves about ``true`` with Monte Carlo noise (variance ``noise * true``) and Geant4 Poisson counts."""
    halves = [np.maximum(true + rng.normal(0, np.sqrt(noise * true)), 0.0) for _ in range(2)]
    return halves[0], noise * true, halves[1], noise * true, rng.poisson(scale * true)


def test_an_exact_match_is_unbiased_even_with_a_noisy_lumen_and_few_counts():
    rng = np.random.default_rng(3)
    true = np.exp(rng.uniform(np.log(0.5), np.log(3000), size=(200, 400)))  # events x sensors, a wide range of counts
    r = event.statistics(*_sample(rng, true, noise=0.5), seed=1)
    assert abs(r["pull"].mean() - r["pull_toy"].mean()) < 0.01  # Geant4 as an exact match looks like the toys
    assert abs(r["pull"].std() - r["pull_toy"].std()) < 0.01
    assert abs(r["pull"].mean()) < 0.02  # and is centred: one lumen run doing both was -0.26 at this noise
    assert all(abs(q["z"]) < 4 for q in r["bins"])
    assert abs(np.nanmean(r["sensor_z"])) < 0.1 and 0.9 < np.std(r["event_z"]) < 1.1


def test_a_three_percent_difference_in_dim_sensors_is_seen_and_only_there():
    rng = np.random.default_rng(4)
    true = np.exp(rng.uniform(np.log(0.5), np.log(3000), size=(400, 600)))
    a, va, b, vb, n = _sample(rng, true, noise=0.5)
    n = np.where(true < 5, rng.poisson(1.03 * true), n)
    r = event.statistics(a, va, b, vb, n)
    low = [q for q in r["bins"] if q["hi"] <= 5]  # lumen's noise moves sensors between bins: each is diluted a little
    assert all(q["diff"] == pytest.approx(0.03, abs=4 * q["err"]) for q in low)
    assert sum(q["z"] ** 2 for q in low) > 25  # seen at more than 5 sigma together
    assert all(abs(q["z"]) < 4 for q in r["bins"] if q["lo"] >= 20)


def test_the_figure_draws(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    from lumen.tools import plot

    rng = np.random.default_rng(5)
    true = np.exp(rng.uniform(0, 6, size=(20, 300)))
    r = event.statistics(*_sample(rng, true, noise=0.3))
    r |= dict(
        t=np.arange(100) * 4.0,
        times_lumen=np.exp(-np.arange(100) / 20) * 1e4,
        times_geant4=rng.poisson(np.exp(-np.arange(100) / 20) * 1e4),
    )
    plot.compare_events(r, title="toy").savefig(tmp_path / "e.png")
