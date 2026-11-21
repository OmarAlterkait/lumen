"""Hit-level readout: the expected hits of a sensor's integration windows and the likelihood of a measured hit list.

The electronics (``lumen.readout.events``): a sensor's photoelectrons form a Poisson process of intensity rho(t) (the
light's node masses through the time spread and transit offset, plus the dark rate over the gate). A window opens at a
photoelectron when none is open and integrates the photoelectrons in [t, t + W] (W the front end's ``window``, 0: each
photoelectron its own window); the threshold accepts it with probability A(Q); the measured charge is Q + noise,
recorded when positive, times ``charge_scale``; the hit time is the window's first photoelectron plus the time walk at
the measured charge.

The likelihood of a sensor's hits (t_k, q_k), k = 1..K in time order, with tau_k = t_k - walk shift(q_k):

    L = prod_k [ rho~(tau_k) dt  f(q_k | m_k)  exp(-Lambda_r(e_{k-1}, tau_k)) ]  exp(-Lambda_r(e_K, g1))

- rho~ the intensity with the time spread widened by the walk's spread at q_k; dt = ``t_window``, a constant;
- m_k = Lambda(tau_k, tau_k + W), the expected other photoelectrons in the window: the window holds 1 + Poisson(m_k)
  photoelectrons, so its measured charge has the density f(q | m) of that compound sum (SPE charges, the threshold,
  the noise, q > 0), whose total r(m) is the probability the window is recorded;
- e_k = tau_k + W the end of window k (e_0 = g0, the gate's start; g1 its end); between windows no recorded window
  opens: Lambda_r(a, b) = int_a^b rho(t) r(m(t)) dt, the recorded windows' opening intensity.

It is exact when every window is recorded (r = 1) and when W = 0 (a window not recorded holds no time); otherwise an
unrecorded window's W is taken as free time (its blocking is neglected: a window below threshold followed within W by
more light). A sensor with no hit has exp(-Lambda_r(g0, g1)). Lambda is exact (node masses through the time spread's
CDF); r(m(t)) is averaged over each interval on the time nodes, so Lambda_r is exact wherever r is constant.

Charge: the measured charge of j photoelectrons from the characteristic function on a static charge grid (one FFT per
hit, exact in m and the SPE values); a window whose 1 + m exceeds half the grid's range is taken Gaussian.

The light of each sensor (and of each hit) is its own: both run per block of sensors, and of hits
(``detection.rows_map``), so no step holds the whole group's node-by-node intermediates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lumen.readout.detection import (
    CELL_BYTES,
    expected_pe,
    histograms,
    masses,
    masses_row_bytes,
    rows_map,
    spread,
    values,
)
from lumen.readout.grid import check_sensors
from lumen.rules.choices import M_GRID_POINTS, Q_NEGATIVE
from lumen.units import S_PER_NS, TWO_PI

T_WINDOW = 1.0  # ns: the width a hit's time density is taken over (a constant of the likelihood)
FLOOR = 1e-30
# float32 arrays of the time nodes one sensor's (one hit's) light holds: the recorded share's cumulative sums and the
# windows' light through the spread's CDF and survival function (``_record_share``, ``_lambda_r``, ``_mass``)
SENSOR_ARRAYS = 32
HIT_ARRAYS = 96


# ------------------------------------------------------------------------------------------------ the hit list
@dataclass(frozen=True)
class Hits:
    """Measured hits, sorted by sensor then time: ``sensor``, ``t`` (ns) and ``q`` (measured units) (K,) each; per
    hit ``prev`` (its sensor's previous hit, -1 for the first) and ``last`` (it is its sensor's last); ``hit`` (S,)
    whether each sensor has a hit."""  # noqa: D205, D209

    sensor: Any
    t: Any
    q: Any
    prev: Any
    last: Any
    hit: Any

    @classmethod
    def from_list(cls, sensor, t, q, n_sensors: int) -> Hits:
        """Return the hits of a hit list (sensor ids, times in ns, charges), every hit kept."""
        sensor, t, q = (np.asarray(a) for a in (sensor, t, q))
        check_sensors(sensor, n_sensors, "Hits.from_list")
        order = np.lexsort((t, sensor))
        s, tt, qq = sensor[order].astype(np.int32), t[order].astype(np.float32), q[order].astype(np.float32)
        same = np.r_[False, s[1:] == s[:-1]]
        prev = np.where(same, np.arange(len(s)) - 1, -1).astype(np.int32)
        last = np.r_[s[1:] != s[:-1], True] if len(s) else np.zeros(0, bool)
        hit = np.zeros(n_sensors, bool)
        hit[s] = True
        return cls(*(jnp.asarray(a) for a in (s, tt, qq, prev, last, hit)))


@dataclass(frozen=True)
class ExpectedHits:
    """Per sensor: ``pe``, ``mu`` (every photoelectron of the light, whatever its time, plus the dark ones over the
    gate) and ``p_hit`` (the probability of at
    least one recorded hit); the charge grid ``charge_grid`` (N,) in measured units and ``recorded`` (J,), the
    probability a window of j = 1..J photoelectrons is recorded; and what the likelihood needs (node masses, overflow
    cells, the values, the histograms).
    """  # noqa: D205

    pe: Any
    mu: Any
    p_hit: Any
    charge_grid: Any
    recorded: Any
    nodes: Any
    late: Any
    early: Any
    values: Any
    hist: Any


jax.tree_util.register_dataclass(Hits, data_fields=["sensor", "t", "q", "prev", "last", "hit"], meta_fields=[])
_EXPECTED = ["pe", "mu", "p_hit", "charge_grid", "recorded", "nodes", "late", "early", "values", "hist"]
jax.tree_util.register_dataclass(ExpectedHits, data_fields=_EXPECTED, meta_fields=[])


def charge_grid(model):
    """Return (N, n_neg, q (N,)): the static charge grid in pe, in linear order (n_neg bins below 0)."""
    n_neg = int(math.ceil(Q_NEGATIVE / model.charge_step))
    n = 1 << int(math.ceil(math.log2((model.charge_max + Q_NEGATIVE) / model.charge_step)))
    return n, n_neg, (np.arange(n) - n_neg) * model.charge_step


def _spe(model, v):
    """Return the SPE charge masses on the grid, q = 0 at index 0 (circular), and the grid's (n, n_neg, q)."""
    n, n_neg, q = charge_grid(model)
    dq = model.charge_step
    edges = np.r_[q - 0.5 * dq, q[-1] + 0.5 * dq]
    return jnp.roll(model.charge.pmf(v["charge"], edges), -n_neg), (n, n_neg, q)


def _window_transform(model, v, m):
    """Return (the charge transform (Q, N/2 + 1) of windows of 1 + Poisson(m) photoelectrons (Q,): the first
    photoelectron's SPE charge and a compound Poisson sum of m, the grid (n, n_neg, q))."""  # noqa: D205, D209
    spe, grid = _spe(model, v)
    phi1 = jnp.fft.rfft(spe)
    return phi1[None, :] * jnp.exp(jnp.asarray(m)[:, None] * (phi1[None, :] - 1.0)), grid


def window_charge(model, v, m):
    """Return (charge masses (Q, N) on ``charge_grid``, linear order) of windows of 1 + Poisson(m) photoelectrons (Q,),
    before the threshold and noise."""  # noqa: D205, D209
    phi, (n, n_neg, _) = _window_transform(model, v, m)
    return jnp.roll(jnp.fft.irfft(phi, n), n_neg, -1)


def _digitise(model, v, phi, grid):
    """Return the recorded windows' measured-charge masses (..., N) (linear order) of charge transforms ``phi``."""
    n, n_neg, q = grid
    qpdf = jnp.roll(jnp.fft.irfft(phi, n), n_neg, -1)
    acc = model.threshold.accept(v["threshold"], jnp.asarray(q, jnp.float32)) if model.threshold is not None else 1.0
    w = TWO_PI * jnp.fft.rfftfreq(n, model.charge_step)
    noisy = jnp.fft.irfft(jnp.fft.rfft(jnp.roll(qpdf * acc, -n_neg, -1)) * jnp.exp(-0.5 * (v["frontend"][
        "charge_noise"] * w) ** 2), n)  # fmt: skip
    return jnp.roll(noisy, n_neg, -1) * jnp.asarray(q > 0.0, jnp.float32)


def recorded(model, v):
    """Return (J,) the probability that a window of j = 1..J photoelectrons is recorded (J the grid's range in pe)."""
    spe, grid = _spe(model, v)
    phi1 = jnp.fft.rfft(spe)
    j = jnp.arange(1, int(model.charge_max) + 1, dtype=jnp.float32)
    return _digitise(model, v, phi1[None, :] ** j[:, None], grid).sum(-1)


def record_probability(rec, m):
    """Return r(m): the probability that a window of 1 + Poisson(m) photoelectrons is recorded (``recorded`` rec)."""
    J = rec.shape[-1]
    n = jnp.arange(J, dtype=jnp.float32)
    m = jnp.asarray(m)[..., None]
    logp = n * jnp.log(jnp.maximum(m, FLOOR)) - m - jax.lax.lgamma(n + 1.0)
    p = jnp.where(n == 0, jnp.exp(-m), jnp.exp(logp))
    return jnp.sum(p * rec, -1) + jnp.maximum(1.0 - p.sum(-1), 0.0) * rec[-1]


# ------------------------------------------------------------------------------------------------ the light
# windows' expected extra photoelectrons where r(m) is tabulated (its resolution: lumen.rules.choices)
M_GRID = np.r_[0.0, np.geomspace(1e-4, 1e4, M_GRID_POINTS)]


def _mass(Hm, tt, T0, a, b, cdf, sf):  # noqa: N803, PLR0913
    """Return the light of rows Hm (Q, M) at times tt (Q, M) in [a, b] (Q,) after the delay of CDF / SF: a
    difference of CDFs, or of survival functions right of the median (no cancellation)."""  # noqa: D205, D209
    lo = a[:, None] - tt - T0[:, None]
    hi = b[:, None] - tt - T0[:, None]
    Fa, Fb = cdf(lo), cdf(hi)
    m = jnp.where(Fa < 0.5, Fb - Fa, sf(lo) - sf(hi))
    return jnp.sum(Hm * jnp.where(b[:, None] > a[:, None], m, 0.0), -1)


def _rows(e: ExpectedHits, rows):
    """Return the node masses and their times (Q, M) of sensors ``rows``, the overflow cells at their mean times."""
    g, H = e.hist.grid, e.nodes
    tn = jnp.broadcast_to(g.t0 + g.dt * jnp.arange(H.shape[-1]), (len(rows), H.shape[-1]))
    masses, times = [H[rows]], [tn]
    for cell in (e.late, e.early):
        q, qt = cell[rows, 0], cell[rows, 1]
        masses.append(q[:, None])
        times.append((qt / jnp.where(q != 0.0, q, 1.0))[:, None])
    return jnp.concatenate(masses, -1), jnp.concatenate(times, -1)


def _dark(e: ExpectedHits, rows, a, b):
    """Return the dark photoelectrons of sensors ``rows`` in [a, b] within the gate."""
    g0, g1 = e.hist.gate
    rate = jnp.broadcast_to(jnp.asarray(e.values["dark_rate"]) * S_PER_NS, e.pe.shape)[rows]
    return rate * jnp.maximum(jnp.minimum(b, g1) - jnp.maximum(a, g0), 0.0)


def _record_share(e: ExpectedHits):
    """Return (the node cells' edges + offset (S, nt + 1), their light Hd (S, nt), dark included, and r (S, nt)).

    r = r(m), m the light in the W after each node: a window opening there is recorded with probability r(m)
    (``record_probability``, interpolated on ``M_GRID``).
    """
    g, H, v = e.hist.grid, e.nodes, e.values
    S, nt = H.shape
    t = g.t0 + g.dt * jnp.arange(nt) + jnp.reshape(jnp.broadcast_to(v["t_offset"], (S,)), (S, 1))
    rate = jnp.broadcast_to(jnp.asarray(v["dark_rate"]) * S_PER_NS, (S,))[:, None]
    Hd = H + rate * g.dt
    C = jnp.concatenate([jnp.zeros((S, 1), H.dtype), jnp.cumsum(Hd, -1)], -1)  # at the nodes' lower edges
    edges = jnp.concatenate([t - 0.5 * g.dt, t[:, -1:] + 0.5 * g.dt], -1)
    W = v["frontend"]["window"]
    m = jax.vmap(jnp.interp)(t + W, edges, C) - jax.vmap(jnp.interp)(t, edges, C)
    r = jnp.interp(m, jnp.asarray(M_GRID, m.dtype), record_probability(e.recorded, jnp.asarray(M_GRID, m.dtype)))
    return edges, Hd, r


def _mean_r(share, rows, a, b):
    """Return r averaged over [a, b] on the nodes (the share of the light there that opens a recorded window).

    Summed over the cells' overlap with [a, b], not differenced from cumulative sums: a cumulative sum's rounding
    (XLA sums in blocks) leaves steps where the cells hold no light, and the ratio of two such steps is noise.
    """
    edges, Hd, r = (x[rows] for x in share)
    f = jnp.maximum(jnp.minimum(b[:, None], edges[:, 1:]) - jnp.maximum(a[:, None], edges[:, :-1]), 0.0) * Hd
    dc = jnp.sum(f, -1)
    dr = jnp.sum(f * r, -1)
    at = jnp.take_along_axis(r, jnp.clip(jax.vmap(jnp.searchsorted)(edges, a)[:, None] - 1, 0, r.shape[-1] - 1),
                             -1)[:, 0]  # fmt: skip
    return jnp.where(dc > 0.0, dr / jnp.where(dc > 0.0, dc, 1.0), at)


def _lambda_r(e: ExpectedHits, share, rows, a, b, cdf, sf):  # noqa: PLR0913
    """Return Lambda_r(a, b): the recorded windows' opening light of sensors ``rows`` in [a, b]."""
    Hm, tt = _rows(e, rows)
    T0 = jnp.broadcast_to(e.values["t_offset"], e.pe.shape)[rows]
    lam = _mass(Hm, tt, T0, a, b, cdf, sf) + _dark(e, rows, a, b)
    return lam * _mean_r(share, rows, a, b)


def expected_hits(light, params) -> ExpectedHits:
    """Return the expected hits per sensor (see the module docstring)."""
    h = histograms(light)
    m = h.model
    v = values(h, params)
    pe_s = expected_pe(h, v)
    g0, g1 = h.gate
    mu = pe_s + v["dark_rate"] * S_PER_NS * (g1 - g0)
    q = jnp.asarray(charge_grid(m)[2], jnp.float32) * v["frontend"]["charge_scale"]
    H, late, early = masses(h, v) if h.time is not None else (None, None, None)
    e = ExpectedHits(pe_s, mu, None, q, recorded(m, v), H, late, early, v, h)
    if H is None:
        return e
    cdf, sf = spread(h, v)

    def block(s, b):  # the recorded windows' light over the gate of sensors [s, s + b)
        eb = _restrict(e, s, b)
        full = jnp.full(b, g0, jnp.float32), jnp.full(b, g1, jnp.float32)
        return _lambda_r(eb, _record_share(eb), jnp.arange(b), *full, cdf, sf)

    lr = rows_map(block, H.shape[0], _sensor_row_bytes(h), CELL_BYTES)
    return ExpectedHits(pe_s, mu, -jnp.expm1(-lr), q, e.recorded, H, late, early, v, h)


def _sensor_row_bytes(h) -> float:
    """Return one sensor's working memory for its recorded windows' light (``_lambda_r``)."""
    return masses_row_bytes(h) + CELL_BYTES * SENSOR_ARRAYS * (h.grid.nt + 3)


def _restrict(e: ExpectedHits, start, size: int, rows=None) -> ExpectedHits:
    """Return the expected hits of the sensors [start, start + size), or of the sensors ``rows`` (an index array,
    repeats allowed): every per-sensor array and value cut (the whole itself)."""  # noqa: D205, D209
    from lumen.sensors.value_sets import PER_SENSOR

    n = e.pe.shape[0]
    if rows is None and isinstance(start, int) and start == 0 and size == n:
        return e
    rows = start + jnp.arange(size) if rows is None else rows
    take = lambda x: None if x is None else jnp.asarray(x)[rows]  # noqa: E731
    per = lambda k, x: k in PER_SENSOR and jnp.ndim(x) > 0 and jnp.shape(x)[0] == n  # noqa: E731
    v = {k: take(x) if per(k, x) else x for k, x in e.values.items()}
    return ExpectedHits(take(e.pe), take(e.mu), take(e.p_hit), e.charge_grid, e.recorded, take(e.nodes), take(e.late),
                        take(e.early), v, e.hist)  # fmt: skip


def _charge_logp(e: ExpectedHits, q, m):
    """Return log f(q | m) per hit: the density (measured unit) of a recorded window of 1 + Poisson(m) photoelectrons
    with measured charge q (its total over q is the window's recording probability)."""  # noqa: D205, D209
    model, v = e.hist.model, e.values
    s = v["frontend"]["charge_scale"]
    big = 1.0 + m > 0.5 * model.charge_max
    phi, grid = _window_transform(model, v, jnp.where(big, 0.0, m))
    pdf = _digitise(model, v, phi, grid)
    dq = model.charge_step * s
    x = jnp.clip((q - e.charge_grid[0]) / dq, 0.0, e.charge_grid.shape[0] - 1.001)
    i = jnp.floor(x).astype(jnp.int32)
    f = x - i
    pick = jnp.take_along_axis(pdf, jnp.stack([i, i + 1], -1), -1)
    dens = jnp.maximum(((1.0 - f) * pick[:, 0] + f * pick[:, 1]) / dq, FLOOR)
    m1, m2 = model.charge.moments(v["charge"])
    mean = (1.0 + m) * m1 * s
    var = ((m2 - m1 * m1) + m * m2) * s**2 + (v["frontend"]["charge_noise"] * s) ** 2
    gauss = -0.5 * (q - mean) ** 2 / var - 0.5 * jnp.log(TWO_PI * var)
    return jnp.where(big, gauss, jnp.log(dens))


def hits_nll(e: ExpectedHits, data: Hits, t_window: float = T_WINDOW):
    """Return the negative log-likelihood of a measured hit list under the expected hits (the module docstring)."""
    h, m, v = e.hist, e.hist.model, e.values
    g0, g1 = h.gate
    S = e.pe.shape[0]
    if tuple(data.hit.shape) != (S,):
        raise ValueError(f"hits_nll: the hits are of {data.hit.shape[0]} sensors, the expected hits of {S} "
                         "(Hits.from_list(..., n_sensors) of this group)")  # fmt: skip
    cdf, sf = spread(h, v)

    def quiet(s, b):  # the sensors [s, s + b) without a hit: no recorded window over the gate
        eb = _restrict(e, s, b)
        full = jnp.full(b, g0, jnp.float32), jnp.full(b, g1, jnp.float32)
        lam = _lambda_r(eb, _record_share(eb), jnp.arange(b), *full, cdf, sf)
        return jnp.where(jax.lax.dynamic_slice_in_dim(data.hit, s, b), 0.0, lam)

    none = jnp.sum(rows_map(quiet, S, _sensor_row_bytes(h), CELL_BYTES))
    K = data.sensor.shape[0]
    if K == 0:
        return none
    W = v["frontend"]["window"]
    s = v["frontend"]["charge_scale"]
    qpe = data.q / s
    shift, extra = m.time_walk.at(v["time_walk"], qpe) if m.time_walk is not None else (0.0, 0.0)
    tau = data.t - shift
    start = jnp.where(data.prev >= 0, tau[jnp.maximum(data.prev, 0)] + W, g0)
    extra = jnp.asarray(extra) * jnp.ones_like(qpe)  # the walk's spread at each hit's charge
    per = (data.sensor, tau, start, data.q, data.last, extra)

    def hits(i, b):  # the hits [i, i + b), each in its own sensor's light (one row per hit)
        k, tau, start, q, last, extra = (jax.lax.dynamic_slice_in_dim(x, i, b) for x in per)
        eb, rows = _restrict(e, 0, b, rows=k), jnp.arange(b)
        sh = _record_share(eb)
        lo, hi = tau - 0.5 * t_window, tau + 0.5 * t_window
        gap = _lambda_r(eb, sh, rows, start, tau, cdf, sf)  # no recorded opening before the hit's photoelectron
        tail = jnp.where(last, _lambda_r(eb, sh, rows, tau + W, jnp.full_like(tau, g1), cdf, sf), 0.0)
        Hm, tt = _rows(eb, rows)
        T0 = jnp.broadcast_to(eb.values["t_offset"], eb.pe.shape)[rows]
        pt = m.tts.widened(v["tts"], jnp.reshape(extra, (-1, 1)))
        dens = _mass(Hm, tt, T0, lo, hi, lambda x: m.tts.cdf(pt, x), lambda x: m.tts.sf(pt, x))
        dens = jnp.maximum(dens + _dark(eb, rows, lo, hi), FLOOR)
        extra_pe = _mass(Hm, tt, T0, tau, tau + W, cdf, sf) + _dark(eb, rows, tau, tau + W)
        return -jnp.log(dens) - _charge_logp(e, q, extra_pe) + gap + tail

    n_q = e.charge_grid.shape[0]
    per_hit = rows_map(hits, K, CELL_BYTES * (HIT_ARRAYS * (h.grid.nt + 3) + 16 * n_q), CELL_BYTES)
    return none + jnp.sum(per_hit)
