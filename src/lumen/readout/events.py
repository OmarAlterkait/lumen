"""A simulated event from the expected light: photoelectrons, hits and waveforms (pseudo-data; not differentiable).

    ev = lumen.readout.sample(light, params, seed=1)            # ev.pe, ev.hits, ev.waveforms

Photoelectron counts are Poisson in each bin of the expected density (dark included), uniform within the bin (the
density already holds the time spread and the transit offset). ``photoelectrons`` turns given photon arrivals into
photoelectrons instead (detection draws, then time spread, delay and offset), and ``electronics`` runs the rest:
SPE charges; integration windows that open at a sensor's first photoelectron and last ``window`` (no dead time; the
next photoelectron after a window opens the next); the threshold's acceptance of the window charge Q; the measured
charge Q + noise, recorded when positive, times ``charge_scale``; the hit time = the window's first photoelectron +
the time walk at the measured charge, recorded when inside the gate. Waveforms: each photoelectron's pulse times
its charge and the gain, at the
front end's samples over the gate, plus the baseline and white noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from lumen.readout.detection import histograms, values
from lumen.readout.waveform import _density

_DENSITY = jax.jit(_density)


@dataclass(frozen=True)
class SampledEvent:
    """One simulated event: photoelectrons per sensor ``pe`` (S,), the hit list ``hits`` and ``waveforms``.

    ``hits`` is a dict of arrays: ``sensor``, ``t`` (ns), ``q`` (measured units), ``n_pe`` (photoelectrons in the
    window) and ``t_first`` (its first photoelectron, ns). ``waveforms`` (S, n) in the front end's unit, sample k
    at ``t0 + k dt`` (ns), or None.
    """

    pe: np.ndarray
    hits: dict
    waveforms: np.ndarray | None
    t0: float
    dt: float


def _padded(fn, x):
    """Return fn(x) for a numpy array, padded along its first axis to a power of two (few distinct shapes compile)."""
    n = len(x)
    m = 1 << max(int(n - 1).bit_length(), 0)
    pad = np.zeros((m - n,) + x.shape[1:], x.dtype)
    return np.asarray(fn(np.concatenate([x, pad])), np.float64)[:n] if n else np.zeros(x.shape)


def _np(v):
    return {k: _np(x) for k, x in v.items()} if isinstance(v, dict) else np.asarray(v, np.float64)


def photoelectrons(model, v, sensor, t, rng, lam=None, theta=None):  # noqa: PLR0913
    """Return (sensor, t) of the photoelectrons of photon arrivals (sensor ids, times in ns; numpy).

    Each photon is detected with probability efficiency QE(lam) CE(theta) (lam in nm and theta in rad when given,
    else QE and CE are not applied), then gets the time spread, the delay draw and the transit offset.
    """
    v = _np(v)
    sensor, t = np.asarray(sensor), np.asarray(t, np.float64)
    eff = np.broadcast_to(v["efficiency"], (int(sensor.max(initial=-1)) + 1,))[sensor]
    if lam is not None:
        qe = np.interp(lam, model.qe_knots, v["qe"]) if model.qe_knots else v["qe"][np.clip(np.searchsorted(
            model.qe_bands, lam, side="right") - 1, 0, len(v["qe"]) - 1)]  # fmt: skip
        eff = eff * qe
    if theta is not None and model.ce_knots:
        eff = eff * np.interp(theta, model.ce_knots, v["ce"])
    keep = rng.random(len(t)) < eff
    s, tt = sensor[keep], t[keep] + model.tts.sample(rng, v["tts"], int(keep.sum()))
    if model.delay is not None:
        tt = tt + model.delay.sample(rng, v["delay"], len(tt))
    off = np.broadcast_to(v["t_offset"], (int(sensor.max(initial=-1)) + 1,))
    return s, tt + off[s]


def _windows(sensor, t, q, window):
    """Return per window (sensor, first time, charge, photoelectrons) of time-sorted integration (numpy)."""
    order = np.lexsort((t, sensor))
    s, t, q = sensor[order], t[order], q[order]
    free = np.ones(len(t), bool)
    out = []
    while free.any():
        idx = np.flatnonzero(free)
        us, first = np.unique(s[idx], return_index=True)
        start = np.full(int(s.max()) + 1, np.inf)
        start[us] = t[idx[first]]
        inw = free & (t <= start[s] + window)
        out.append((us, start[us], np.bincount(s[inw], q[inw], len(start))[us],
                    np.bincount(s[inw], minlength=len(start))[us]))  # fmt: skip
        free &= ~inw
    return [np.concatenate(c) for c in zip(*out, strict=False)] if out else [np.zeros(0)] * 4


def electronics(model, v, sensor, t, rng, n_sensors: int, gate=None) -> SampledEvent:  # noqa: PLR0913
    """Return the hits and waveforms of given photoelectrons (sensor ids, times in ns) under a model's values."""
    v = _np(v)
    sensor, t = np.asarray(sensor, np.int64), np.asarray(t, np.float64)
    g = model.charge.sample(rng, v["charge"], len(t))
    fe = v["frontend"]
    ws, wt, wq, wn = _windows(sensor, t, g, float(fe["window"]))
    acc = _padded(lambda q: model.threshold.accept(v["threshold"], q), wq) if model.threshold is not None else 1.0
    qm = wq + rng.normal(0.0, float(fe["charge_noise"]), len(wq))
    keep = (rng.random(len(wq)) < acc) & (qm > 0.0)
    ws, wt, qm, wn = ws[keep], wt[keep], qm[keep], wn[keep]
    first = wt.copy()
    if model.time_walk is not None:
        shift = _padded(lambda q: model.time_walk.at(v["time_walk"], q)[0], qm)
        sig = _padded(lambda q: model.time_walk.at(v["time_walk"], q)[1], qm)
        wt = wt + shift + rng.normal(0.0, 1.0, len(wt)) * sig
    if gate is not None:  # a hit outside the readout gate is not recorded
        inside = (wt >= gate[0]) & (wt <= gate[1])
        ws, wt, qm, wn, first = ws[inside], wt[inside], qm[inside], wn[inside], first[inside]
    hits = dict(
        sensor=ws.astype(np.int64), t=wt, q=qm * float(fe["charge_scale"]), n_pe=wn.astype(np.int64), t_first=first
    )
    pe = np.bincount(sensor, minlength=n_sensors).astype(np.float64)
    if model.pulse is None or model.readout != "waveform" or gate is None:
        return SampledEvent(pe, hits, None, 0.0, 0.0)
    return SampledEvent(pe, hits, _waveforms(model, v, (sensor, t, g), n_sensors, gate, rng), float(gate[0]),
                        float(model.sampling))  # fmt: skip


def _waveforms(model, v, pes, n_sensors, gate, rng):
    import jax.numpy as jnp

    sensor, t, g = pes

    n = int(np.ceil((gate[1] - gate[0]) / model.sampling - 1e-9))
    r = {f: model.ranges[f"pulse/{f}"] for f in model.pulse.fields() if f"pulse/{f}" in model.ranges}
    lo, hi = model.pulse.support(r)
    k0 = np.ceil((t + lo - gate[0]) / model.sampling).astype(np.int64)
    width = int(np.ceil((hi - lo) / model.sampling)) + 1
    k = k0[:, None] + np.arange(width)[None, :]
    lag = gate[0] + k * model.sampling - t[:, None]
    pv = {a: jnp.asarray(b, jnp.float32) for a, b in v["pulse"].items()}
    amp = _padded(lambda x: model.pulse.shape(pv, jnp.asarray(x, jnp.float32)), lag)
    gain = np.broadcast_to(v["gain"], (n_sensors,))[sensor]
    ok = (k >= 0) & (k < n)
    wf = np.zeros(n_sensors * n)
    np.add.at(wf, (sensor[:, None] * n + k)[ok], (amp * (g * gain)[:, None])[ok])
    wf = wf.reshape(n_sensors, n) + float(v["frontend"]["baseline"])
    return wf + rng.normal(0.0, float(v["frontend"]["noise"]), wf.shape)


def sample(light, params, *, seed: int) -> SampledEvent:
    """Return one simulated event of the expected light (see the module docstring)."""
    h = histograms(light)
    v = values(h, params)
    rng = np.random.default_rng(seed)
    rho = np.clip(np.asarray(_DENSITY(h, v), np.float64), 0.0, None)
    counts = rng.poisson(rho)
    s, b = np.nonzero(counts)
    reps = counts[s, b]
    s, b = np.repeat(s, reps), np.repeat(b, reps)
    t = h.gate[0] + (b + rng.random(len(b))) * h.grid.dt
    return electronics(h.model, v, s, t, rng, rho.shape[0], h.gate)
