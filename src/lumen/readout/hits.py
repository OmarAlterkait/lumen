"""Hit-level readout: the expected hits (charge and first-hit time) and the likelihood of measured hits.

A hit is one integration window: it opens at a sensor's first photoelectron and integrates the window's charge Q;
the threshold accepts it with probability A(Q); the measured charge is Q + noise (a hit whose measured charge is not
positive is not recorded), times ``charge_scale``; its time is the first photoelectron's plus the time walk at the
measured charge. ``expected_hits`` assumes one window per sensor holding all its light (true when the light's
spread is shorter than the window).

Charge: the number of photoelectrons n in the window is Poisson(mu), mu = pe + dark rate x window, and Q | n is the
n-fold SPE convolution, computed at once from the characteristic function on a static charge grid:
phi_Q = exp(mu (phi_1 - 1)), one FFT per sensor, exact in mu and the SPE values. Sensors with mu above half the
grid's range are marked unresolved (their charge distribution is taken Gaussian, their hit probability 1 - e^-mu).

First-hit time: the windowed first-arrival terms (mu, U, W) of the time histogram's node masses for observed hit
times, with the time spread's CDF and survival function, the transit offset and the overflow cells at their mean
times; the likelihood of the first of n photoelectrons falling in a window of width ``t_window`` about the observed
time, n = the observed charge in photoelectrons (at least 1), with uniform dark light over the gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from lumen.readout.detection import expected_pe, histograms, nodes, values
from lumen.readout.waveform import _pytree

Q_NEGATIVE = 4.0  # pe: the charge grid reaches this far below 0 (SPE spectra with an undershoot)
T_WINDOW = 1.0  # ns: the default window about an observed hit time in the first-hit likelihood
FLOOR = 1e-30


# ------------------------------------------------------------------------------------------------ first-hit terms
def window_mass(a, b):
    """Return Phi(b sqrt 2) - Phi(a sqrt 2) for a <= b without cancellation (each edge in its accurate tail)."""
    left = 0.5 * (jsp.erfc(-b) - jsp.erfc(-a))
    right = 0.5 * (jsp.erfc(a) - jsp.erfc(b))
    mid = 0.5 * (jsp.erf(b) - jsp.erf(a))
    return jnp.where(b <= 0.0, left, jnp.where(a >= 0.0, right, mid))


def first_hit_terms(H, grid, tq, t_window, cdf, sf, T0=0.0, late=None, early=None, horizon=None):  # noqa: PLR0913, N803
    """Return (mu, U, W) per sensor: all the light, the light after tq - t_window / 2, the light in the window.

    H (S, nt) node masses on ``grid``; ``cdf`` / ``sf`` the delay's CDF and survival function (callables of the lag,
    ns); T0 the transit offset (scalar or (S,)); late / early optional (q, q t) overflow cells (S,) each, entered at
    their mean times. The window mass is a difference of CDFs, or of survival functions right of the median.
    ``horizon`` (a static node count): nodes from it on, and the late cell, are light still to come at every window,
    added to mu and U as plain masses (exact when grid.t0 + horizon dt >= max(tq) + t_window / 2 + the spread's
    reach - min(T0); ``window_horizon``).
    """
    tail = 0.0
    if horizon is not None and horizon < grid.nt:
        tail = H[:, horizon:].sum(-1) + (0.0 if late is None else late[0])
        H, late = H[:, :horizon], None
    nt = H.shape[-1]
    tn = grid.t0 + grid.dt * jnp.arange(nt)
    masses, times = [H], [jnp.broadcast_to(tn, H.shape)]
    for cell in (late, early):
        if cell is not None:
            q, tw = cell
            masses.append(q[:, None])
            times.append((tw / jnp.where(q != 0.0, q, 1.0))[:, None])
    Hm = jnp.concatenate(masses, -1)
    tt = jnp.concatenate(times, -1)
    T0 = jnp.reshape(T0, (-1, 1)) if jnp.ndim(T0) == 1 else T0
    a = tq[:, None] - 0.5 * t_window - tt - T0
    b = a + t_window
    Sa, Sb = sf(a), sf(b)
    Fa, Fb = cdf(a), cdf(b)
    m = jnp.where(Fa < 0.5, Fb - Fa, Sa - Sb)
    return Hm.sum(-1) + tail, (Hm * Sa).sum(-1) + tail, (Hm * m).sum(-1)


def window_horizon(grid, tq_max, t_window, reach, T0_min=0.0):  # noqa: N803
    """Return the smallest exact ``first_hit_terms`` horizon (a node count) for windows up to tq_max (ns)."""
    t = float(tq_max) + 0.5 * float(t_window) + float(reach) - min(float(T0_min), 0.0)
    return int(min(grid.nt, max(0, math.ceil((t - grid.t0) / grid.dt) + 1)))


def first_arrival_nll(mu, Ulo, Wwin, obs_counts, dark_rate=0.0, t_lo=None, gate=None, t_window=1.0, floor=FLOOR):  # noqa: PLR0913, N803
    """Return the windowed first-arrival negative log-likelihood per sensor (0 for unlit sensors).

    With n photoelectrons the probability that the first falls in the window is S_lo^n - S_hi^n, S = U / mu and
    S_hi / S_lo = 1 - W / U: nll = -n (log U - log mu) - log(-expm1(n log1p(-W / U))), accurate for W / U down to
    the float floor and up to 1. Dark light (rate per ns, uniform over ``gate`` on the same clock as ``t_lo``) adds
    r (g1 - g0) to mu, r (g1 - t_lo) to U and r t_window to W; ``floor`` guards log(0) without dark light.
    """
    hit = obs_counts > 0
    n = jnp.where(hit, obs_counts, 1.0)
    mu_t, U, W = mu, Ulo, Wwin
    if jnp.ndim(dark_rate) or dark_rate > 0.0:
        g0, g1 = gate
        mu_t = mu_t + dark_rate * (g1 - g0)
        U = U + dark_rate * jnp.maximum(g1 - t_lo, 0.0)
        W = W + dark_rate * t_window
    mu_t = jnp.where(hit, mu_t, 2.0)
    U = jnp.where(hit, U, 1.0)
    W = jnp.where(hit, W, 0.5)
    U = jnp.maximum(U, floor)
    W = jnp.clip(W, floor, U)
    mu_t = jnp.maximum(mu_t, U)
    q = W / U
    lfirst = jnp.log(-jnp.expm1(n * jnp.log1p(-jnp.minimum(q, 1.0 - 1e-7))))
    nll = -n * (jnp.log(U) - jnp.log(mu_t)) - lfirst
    return jnp.where(hit, nll, 0.0)


def poisson_nll_dark(mu, obs_counts, dark_rate=0.0, gate=None):
    """Return the Poisson charge negative log-likelihood (without log n!) with the dark expectation r (g1 - g0)."""
    mu_t = mu if dark_rate <= 0.0 else mu + dark_rate * (gate[1] - gate[0])
    mu_t = jnp.maximum(mu_t, FLOOR)
    return mu_t - obs_counts * jnp.log(mu_t)


# ------------------------------------------------------------------------------------------------ expected hits
@dataclass(frozen=True)
class Hits:
    """Measured hits as per-sensor arrays: ``hit`` (S,) bool, the first hit's time ``t`` (ns) and charge ``q``."""

    hit: Any
    t: Any
    q: Any

    @classmethod
    def from_list(cls, sensor, t, q, n_sensors: int) -> Hits:
        """Return the per-sensor first hits of a hit list (sensor ids, times in ns, charges)."""
        sensor, t, q = (np.asarray(a) for a in (sensor, t, q))
        order = np.lexsort((t, sensor))
        s, first = np.unique(sensor[order], return_index=True)
        hit, tt, qq = np.zeros(n_sensors, bool), np.zeros(n_sensors, np.float32), np.zeros(n_sensors, np.float32)
        hit[s], tt[s], qq[s] = True, t[order][first], q[order][first]
        return cls(jnp.asarray(hit), jnp.asarray(tt), jnp.asarray(qq))


@dataclass(frozen=True)
class ExpectedHits:
    """Per sensor: ``pe``, ``mu`` (photoelectrons in the window, dark included), ``p_hit``, the hits' charge
    distribution ``charge_pdf`` (S, N) as masses on the grid ``charge`` (N,) (in measured units, summing to p_hit),
    ``resolved``, and what the first-hit likelihood needs (node masses, overflow cells and the values).
    """  # noqa: D205

    pe: Any
    mu: Any
    p_hit: Any
    charge: Any
    charge_pdf: Any
    resolved: Any
    nodes: Any
    late: Any
    early: Any
    values: Any
    hist: Any


_pytree(Hits, ("hit", "t", "q"), ())
_pytree(ExpectedHits, ("pe", "mu", "p_hit", "charge", "charge_pdf", "resolved", "nodes", "late", "early", "values",
                       "hist"), ())  # fmt: skip


def charge_grid(model):
    """Return (N, n_neg, q (N,)): the static charge grid in pe, in linear order (n_neg bins below 0)."""
    n_neg = int(math.ceil(Q_NEGATIVE / model.charge_step))
    n = 1 << int(math.ceil(math.log2((model.charge_max + Q_NEGATIVE) / model.charge_step)))
    return n, n_neg, (np.arange(n) - n_neg) * model.charge_step


def compound_charge(model, v, mu):
    """Return (Q pdf of n >= 1 photoelectrons (S, N), digit masses (S, N)) on ``charge_grid`` (linear order)."""
    n, n_neg, q = charge_grid(model)
    dq = model.charge_step
    edges = np.r_[q - 0.5 * dq, q[-1] + 0.5 * dq]
    spe = jnp.roll(model.charge.pmf(v["charge"], edges), -n_neg)  # q = 0 at index 0, circular
    phi = jnp.exp(mu[:, None] * (jnp.fft.rfft(spe) - 1.0))
    qpdf = jnp.fft.irfft(phi, n).at[:, 0].add(-jnp.exp(-mu))
    qpdf = jnp.roll(qpdf, n_neg, -1)
    acc = model.threshold.accept(v["threshold"], jnp.asarray(q, jnp.float32)) if model.threshold is not None else 1.0
    w = 2.0 * jnp.pi * jnp.fft.rfftfreq(n, dq)
    noisy = jnp.fft.irfft(jnp.fft.rfft(jnp.roll(qpdf * acc, -n_neg, -1)) * jnp.exp(-0.5 * (v["frontend"][
        "charge_noise"] * w) ** 2), n)  # fmt: skip
    digit = jnp.roll(noisy, n_neg, -1) * jnp.asarray(q > 0.0, jnp.float32)
    return qpdf, digit


def expected_hits(light, params) -> ExpectedHits:
    """Return the expected hits per sensor (see the module docstring)."""
    h = histograms(light)
    m = h.model
    v = values(h, params)
    pe_s = expected_pe(h, v)
    mu = pe_s + v["dark_rate"] * 1e-9 * jnp.minimum(v["frontend"]["window"], h.gate[1] - h.gate[0])
    resolved = mu <= 0.5 * m.charge_max
    _, digit = compound_charge(m, v, jnp.where(resolved, mu, 0.0))
    p_hit = jnp.where(resolved, digit.sum(-1), -jnp.expm1(-mu))
    q = jnp.asarray(charge_grid(m)[2], jnp.float32) * v["frontend"]["charge_scale"]
    H, late, early = nodes(h, v) if h.time is not None else (None, None, None)
    return ExpectedHits(pe_s, mu, p_hit, q, digit, resolved, H, late, early, v, h)


def _charge_logp(e: ExpectedHits, q):
    """Return log p(hit with measured charge q) per sensor (a density in the measured unit)."""
    m, v = e.hist.model, e.values
    s = v["frontend"]["charge_scale"]
    dq = m.charge_step * s
    x = jnp.clip((q - e.charge[0]) / dq, 0.0, e.charge.shape[0] - 1.001)
    i = jnp.floor(x).astype(jnp.int32)
    f = x - i
    pick = jnp.take_along_axis(e.charge_pdf, jnp.stack([i, i + 1], -1), -1)
    dens = jnp.maximum(((1.0 - f) * pick[:, 0] + f * pick[:, 1]) / dq, FLOOR)
    m1, m2 = m.charge.moments(v["charge"])
    mean, var = e.mu * m1 * s, e.mu * m2 * s**2 + (v["frontend"]["charge_noise"] * s) ** 2
    gauss = -0.5 * (q - mean) ** 2 / var - 0.5 * jnp.log(2.0 * jnp.pi * var)
    return jnp.where(e.resolved, jnp.log(dens), gauss)


def hits_nll(e: ExpectedHits, data: Hits, t_window: float = T_WINDOW):
    """Return the negative log-likelihood of measured hits (charge and first-hit time) under the expected hits."""
    h, m, v = e.hist, e.hist.model, e.values
    s = v["frontend"]["charge_scale"]
    miss = -jnp.log(jnp.maximum(1.0 - e.p_hit, FLOOR))
    charge = -_charge_logp(e, data.q)
    qpe = data.q / s
    shift, extra = m.time_walk.at(v["time_walk"], qpe) if m.time_walk is not None else (0.0, 0.0)
    pt = m.tts.widened(v["tts"], jnp.reshape(jnp.asarray(extra, jnp.float32) * jnp.ones_like(qpe), (-1, 1)))

    def cdf(x):
        return m.tts.cdf(pt, x, h.profile)

    def sf(x):
        return m.tts.sf(pt, x, h.profile)

    tq = data.t - shift
    late, early = (e.late[:, 0], e.late[:, 1]), (e.early[:, 0], e.early[:, 1])
    mu, U, W = first_hit_terms(e.nodes, h.grid, tq, t_window, cdf, sf, v["t_offset"], late, early)
    m1, _ = m.charge.moments(v["charge"])
    n = jnp.where(data.hit, jnp.maximum(qpe / m1, 1.0), 0.0)
    time = first_arrival_nll(mu, U, W, n, v["dark_rate"] * 1e-9, tq - 0.5 * t_window, h.gate, t_window)
    return jnp.sum(jnp.where(data.hit, charge + time, miss))
