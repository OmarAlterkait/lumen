"""The photoelectron density and the expected waveform, after tracking (exact folds of the time histogram).

    rho_s(t)  = sum_j H_sj [F(t + dt - t_j - T_s) - F(t - t_j - T_s)]  per bin of width dt, plus dark rate x dt
    E[V_s(t)] = baseline + gain_s m1 (rho_s * <h>)(t)                   m1, m2: the SPE charge's mean and mean square
    Var[V_s]  = gain_s^2 m2 (rho_s * <h^2>)(t) + noise^2                shot noise (Campbell) plus white noise

F is the time spread's CDF (with the emission profile, when there is one), T_s the transit offset, H the time
histogram as photoelectrons (``lumen.readout.detection.nodes``) and <h> the single-photoelectron pulse averaged over
each
bin. The bins are the time nodes' spacing, starting at the gate's start; the waveform takes every m-th bin, m =
sampling / node spacing (a whole number by construction of ``lumen.readout.make_grid``). Every step is linear after
the histogram, so derivatives in every value are exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lumen.readout.detection import histograms, nodes, values

SUB = 4  # points per bin averaging the pulse


def _pytree(cls, data, meta):
    def flatten(o):
        return [getattr(o, k) for k in data], tuple(getattr(o, k) for k in meta)

    def unflatten(aux, ch):
        return cls(**dict(zip(data, ch, strict=True)), **dict(zip(meta, aux, strict=True)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


@dataclass(frozen=True)
class PEDensity:
    """Expected photoelectrons per bin ``rate (S, n)`` (dark included); bin k is [t0 + k dt, t0 + (k + 1) dt) ns."""

    rate: Any
    t0: float
    dt: float


@dataclass(frozen=True)
class Waveform:
    """The expected waveform: ``mean`` and ``variance`` (C, n) in ``unit``, sample k at t0 + k dt (ns)."""

    mean: Any
    variance: Any
    channels: Any
    t0: float
    dt: float
    unit: str


_pytree(PEDensity, ("rate",), ("t0", "dt"))
_pytree(Waveform, ("mean", "variance", "channels"), ("t0", "dt", "unit"))


def _n_bins(h):
    m = h.model
    step = int(round(m.sampling / h.grid.dt)) if m.readout == "waveform" else 1
    n = int(math.ceil((h.gate[1] - h.gate[0]) / (h.grid.dt * step) - 1e-9))
    return n * step, step


def _fft_conv(a, k, n_out):
    n = 1 << int(np.ceil(np.log2(a.shape[-1] + k.shape[-1])))
    return jnp.fft.irfft(jnp.fft.rfft(a, n) * jnp.fft.rfft(k, n), n)[..., :n_out]


def _density(h, v):
    """Return the PE per bin (S, nf) from the histograms and split values."""
    g, m = h.grid, h.model
    H, late, early = nodes(h, v)
    nf, _ = _n_bins(h)
    s0 = h.gate[0] - g.t0
    T0 = jnp.reshape(v["t_offset"], (-1, 1))
    lag = (jnp.arange(-g.nt, nf) * g.dt).astype(jnp.float32)[None, :]

    def F(x):  # noqa: N802
        return m.tts.cdf(v["tts"], x, h.profile)

    k = F(s0 + lag + g.dt - T0) - F(s0 + lag - T0)  # k[l + nt]: bin i, node j, l = i - j
    out = _fft_conv(H, k, g.nt + nf)[..., g.nt :]
    e = h.gate[0] + g.dt * jnp.arange(nf + 1, dtype=jnp.float32)
    for cell in (late, early):
        tm = (cell[:, 1] / jnp.where(cell[:, 0] != 0.0, cell[:, 0], 1.0))[:, None]
        out = out + cell[:, :1] * jnp.diff(F(e[None, :] - tm - T0), axis=-1)
    dark = jnp.reshape(v["dark_rate"], (-1, 1)) * 1e-9 * g.dt
    return out + dark


def pe_density(light, params) -> PEDensity:
    """Return the expected photoelectrons per bin of the time nodes' spacing over the gate (dark included)."""
    h = histograms(light)
    return PEDensity(_density(h, values(h, params)), float(h.gate[0]), float(h.grid.dt))


def pulse_kernel(model, p, dt, power=1):
    """Return (first lag index, kernel): the pulse (to ``power``) averaged over bins [(j - 1) dt, j dt]."""
    r = {f: model.ranges[f"pulse/{f}"] for f in model.pulse.fields() if f"pulse/{f}" in model.ranges}
    lo, hi = model.pulse.support(r)
    j0, j1 = int(math.floor(lo / dt)), int(math.ceil(hi / dt)) + 1
    j = jnp.arange(j0, j1 + 1, dtype=jnp.float32)
    t = (j[:, None] - 1.0 + (jnp.arange(SUB, dtype=jnp.float32)[None, :] + 0.5) / SUB) * dt
    return j0, jnp.mean(model.pulse.shape(p, t) ** power, -1)


def _pulse_fold(rho, model, p, dt, power):
    j0, k = pulse_kernel(model, p, dt, power)
    nf = rho.shape[-1]
    if j0 >= 0:  # W_n = sum_j k_j rho_{n - j}: the full convolution shifted by the first lag
        return jnp.concatenate([jnp.zeros(rho.shape[:-1] + (j0,), rho.dtype), _fft_conv(rho, k, nf - j0)], -1)
    return _fft_conv(rho, k, nf - j0)[..., -j0:]


def expected_waveform(light, params) -> Waveform:
    """Return the expected waveform (mean and variance per sensor and sample; see the module docstring)."""
    h = histograms(light)
    m = h.model
    if m.pulse is None or m.readout != "waveform":
        raise ValueError(f"sensor group {h.group!r}: value set {m.set_name} has no waveform readout (readout "
                         f"{m.readout!r}); use a value set with a pulse and readout 'waveform'")  # fmt: skip
    v = values(h, params)
    rho = _density(h, v)
    _, step = _n_bins(h)
    m1, m2 = m.charge.moments(v["charge"])
    gain = jnp.reshape(v["gain"], (-1, 1))
    fe = v["frontend"]
    mean = fe["baseline"] + gain * m1 * _pulse_fold(rho, m, v["pulse"], h.grid.dt, 1)[:, ::step]
    var = gain**2 * m2 * _pulse_fold(rho, m, v["pulse"], h.grid.dt, 2)[:, ::step] + fe["noise"] ** 2
    return Waveform(mean, var, jnp.arange(rho.shape[0]), float(h.gate[0]), float(h.grid.dt * step), m.unit)


def waveform_nll(w: Waveform, data):
    """Return the Gaussian negative log-likelihood of measured waveforms (C, n) under the expected waveform.

    An approximation at low occupancy (the shot noise is not Gaussian there): use ``hits_nll`` for single
    photoelectrons. The Monte Carlo error of the expectation is not included.
    """
    d = jnp.asarray(data, jnp.float32)
    if d.shape != w.mean.shape:
        raise ValueError(f"waveform_nll: data of shape {d.shape}, the expected waveform is {w.mean.shape}")
    return 0.5 * jnp.sum((d - w.mean) ** 2 / w.variance + jnp.log(2.0 * jnp.pi * w.variance))
