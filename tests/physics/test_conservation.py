"""The weight accounting: every photon's start weight ends in one of the health's fates (``lumen.Health.balance``).

    start = lost + abandoned + escaped + distance_cut + truncated + absorbed_bulk + absorbed_surface + rouletted
            + reweighted

Each term is an exact sum of per-photon totals (the loop's health: a photon's steps summed in float32, added once as it
ends), so the identity holds to float32 rounding per step at the reference values: the deterministic fates (absorbed in
the media and at surfaces, lost, escaped, abandoned by the stuck rule, truncated by the vertex guard) leave no weight
unaccounted. Two terms are 0 in expectation only: ``rouletted`` (a photon below eps survives with probability p_s = wr /
eps and carries w / p_s, else it ends: per event, w leaves with probability 1 - p_s and w (1 / p_s - 1) is added with
probability p_s, mean 0, variance w^2 (1 - p_s) / p_s) and ``reweighted`` (the drawn outcome's or direction's live /
proposal factor against its expectation: the defensive mixture's draws, a lobe drawn from a proposal); they are tested
over independent seeds, to the same float32 rounding floor (their per-step float32 arithmetic is not unbiased: measured
on the toys, reweighted's mean up to 1.6e-7 of the start weight, 0.15 ulp per vertex, with a spread far smaller).
The detected light is a tally of ``absorbed_surface``'s share at a sensor (detection is expected, not sampled), not a
term of its own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import toys
from toys import raw_key

from lumen import run as runner
from lumen.api import Health, _edge_rows
from lumen.compare import photons
from lumen.tools import stats
from lumen.transport import step as steps

N = 1 << 14  # photons per run
ZERO_MEAN = ("rouletted", "reweighted")  # the fates 0 in expectation, not in each run
K = 8  # independent seeds of the zero-mean terms


def _rows(x, d=None, n=N, w=1.0):
    """Return n rows from x: along d (a pencil), else isotropic; weight w each."""
    rows = toys.pencil(x, d, 400.0, n) if d is not None else photons.isotropic(jnp.tile(jnp.asarray(x), (n, 1)), 400.0,
                                                                              jax.random.PRNGKey(3))  # fmt: skip
    return rows._replace(w=jnp.full(n, w, jnp.float32))


def _fates(s, rows, seed, *, controls=None, max_vertices=20000, params=None):
    """Return ({fate: float64 total}, start weight, vertices per photon) of one run at the reference values."""
    p = photons.packed(s.det) if params is None else params
    ops = photons.operands(s, p, None, controls or steps.Controls())
    r = runner.run(s.loop(), s.model, ops, rows, raw_key(seed), max_vertices=max_vertices)
    tot = {k: float(np.sum(np.asarray(r.outputs[k], np.float64))) for k in Health.FATES}
    return tot, float(np.sum(np.asarray(rows.w, np.float64))), np.asarray(r.per_item["vertices"])


def _balance(tot, start):
    return start - sum(tot.values())


def _edged(toy, name, *args, **kw):
    """Return the toy's Setup with the production edge rows (every keyed parameter's, ``lumen.api._edge_rows``)."""
    s0 = toy(name, *args, **kw)
    return toy(name, *args, step=steps.StepConfig(edge_rows=_edge_rows(s0.det)), **kw)


# (toy, its args, its kwargs, source point, pencil direction or None: isotropic)
TOYS = {
    "sphere_absorbing": ("sphere", (0.3,), {}, (0.1, 0.05, 0.0), None),
    "sphere_rayleigh_ground": ("sphere", (0.6, "ground", 0.2), dict(rayleigh=0.5, absorption=5.0),
                               (0.2, 0.0, 0.1), None),
    "sphere_glass_occluder": ("sphere", (0.5,), dict(glass=True, rayleigh=3.0, occluder=("box", 0.1, 0.4)),
                              (0.0, 0.0, 0.0), None),
    "rt_boxes": ("rt_boxes", (0.5,), dict(absorption=3.0), (-0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "self_pair": ("self_pair", (0.3, 0.5), dict(absorption=3.0), (-0.25, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "lobe_plate": ("lobe_plate", (0.2,), dict(SPECULARLOBECONSTANT=0.3, BACKSCATTERCONSTANT=0.1), (-0.6, 0.0, 0.3),
                   (0.6, 0.0, -0.8)),
    "border_pair": ("border_pair", (), {}, (-0.25, 0.0, 0.0), None),
    "junction": ("junction", (), {}, (0.0, 0.0, 0.3), None),
}  # fmt: skip


@pytest.mark.parametrize("edge", [False, True], ids=["plain", "edge_rows"])
@pytest.mark.parametrize("name", list(TOYS))
def test_every_photon_weight_is_accounted(toy, name, edge):
    fn, args, kw, x, d = TOYS[name]
    s = _edged(toy, fn, *args, **kw) if edge else toy(fn, *args, **kw)
    rows = _rows(x, d)
    runs = [_fates(s, rows, seed) for seed in range(K)]
    for tot, start, vert in runs:
        assert abs(_balance(tot, start)) <= _rounding(start, vert), (name, _balance(tot, start), tot)
        assert tot["truncated"] == 0.0 and np.all(vert < 20000), (name, vert.max())
    limit = stats.z_limit(len(ZERO_MEAN), df=K - 1)
    for k in ZERO_MEAN:
        v = np.array([tot[k] for tot, _, _ in runs])
        floor = np.mean([_rounding(start, vert) for _, start, vert in runs])
        assert abs(v.mean()) <= limit * stats.sem(v) + floor, (name, k, v, floor)


def _rounding(start, vert):
    """Return the float32 rounding floor of a run's sums: 2 ulp of a photon's start weight per vertex (measured on the
    toys: the balance and reweighted's mean within 1 ulp per vertex, at least 4x inside it)."""
    return 2.0 * 2.0**-24 * start / vert.size * float(vert.sum())


def test_the_roulette_term_has_its_known_variance(toy):
    """rouletted's spread over seeds is that of the roulette's events (not a constant offset)."""
    fn, args, kw, x, d = TOYS["sphere_absorbing"]
    s = toy(fn, *args, **kw)
    rows = _rows(x, d)
    runs = [_fates(s, rows, seed) for seed in range(2 * K)]
    v = np.array([tot["rouletted"] for tot, _, _ in runs])
    floor = np.mean([_rounding(start, vert) for _, start, vert in runs])
    assert v.std(ddof=1) > 0.0, v  # the roulette acted (photons end only by it here)
    assert abs(v.mean()) <= stats.z_limit(df=2 * K - 1) * stats.sem(v) + floor, v


def test_born_photons_are_absorbed_at_their_sensor(toy):
    """A photon emitted inside a sensor counting by volume ends there: its whole weight is absorbed_surface."""
    fn, args, kw, _, _ = TOYS["junction"]
    s = toy(fn, *args, **kw)
    rows = _rows((0.405, 0.0, 0.0))  # inside the counting water shell (r 0.4-0.41 m)
    tot, start, vert = _fates(s, rows, 1)
    assert np.all(vert == 1) and tot["absorbed_surface"] == pytest.approx(start, rel=1e-6), (vert.max(), tot)
    assert abs(_balance(tot, start)) <= _rounding(start, vert), tot


def test_the_guard_reports_what_it_stops(toy):
    """A lossless mirror sphere: no photon ends before the vertex guard; all of its weight is ``truncated``.

    Fails on p3/final: ``health.truncated`` reads 0 when the guard stops live photons (review6/I4_notend.md).
    """
    s = toy("sphere", 1.0, absorption=1e9)  # R = 1, no absorption: nothing ends a photon
    rows = _rows((0.1, 0.05, 0.0), n=256)
    tot, start, vert = _fates(s, rows, 2, max_vertices=64)
    assert np.all(vert == 64), vert
    assert tot["truncated"] == pytest.approx(start, rel=1e-5), tot
    assert abs(_balance(tot, start)) <= _rounding(start, vert), tot


def test_a_short_max_distance_is_its_own_fate(toy):
    """A max_distance shorter than the geometry's reach cuts photons: their weight where they stop is distance_cut
    (attenuated, as every fate's), not escaped; the identity holds."""
    fn, args, kw, x, d = TOYS["sphere_absorbing"]
    s = toy(fn, *args, **kw)
    rows = _rows(x, d)
    full, start, _ = _fates(s, rows, 4)
    assert full["distance_cut"] == 0.0 and full["escaped"] == 0.0, full  # a closed sphere: nothing leaves it
    tot, start, vert = _fates(s, rows, 4, controls=steps.Controls(max_distance=0.3))
    assert tot["distance_cut"] > 0.0 and tot["escaped"] == 0.0, tot
    assert abs(_balance(tot, start)) <= _rounding(start, vert), tot
