"""The photoelectron density and the expected waveform, after tracking (exact convolutions of the time histogram).

    rho_s(t)  = sum_j H_sj [F(t + dt - t_j - T_s) - F(t - t_j - T_s)]  per bin of width dt, plus dark rate x dt
    E[V_s(t)] = baseline + gain_s m1 (rho_s * <h>)(t)                   m1, m2: the SPE charge's mean and mean square
    Var[V_s]  = gain_s^2 m2 (rho_s * <h^2>)(t) + noise^2                shot noise (Campbell) plus white noise

F is the time spread's CDF, T_s the transit offset, H the time histogram
as photoelectrons after the delay (``lumen.readout.detection.masses``; rho is ``detection.density`` plus dark) and
<h> the single-photoelectron pulse averaged over each bin. The bins are the time nodes' spacing, starting at the
gate's start; the waveform takes every m-th bin, m = sampling / node spacing (a whole number by construction of
``lumen.sensors.SensorValuesModel.grid``). Every step is linear after the
histogram, so derivatives in every value are exact. Each runs per block of sensors (``detection.rows_map``).
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
    FFT_ARRAYS,
    block_density,
    density,
    density_row_bytes,
    fft_conv,
    histograms,
    rows_map,
    sensors,
    values,
)
from lumen.readout.detection import n_bins as _n_bins
from lumen.sources._base import bucket
from lumen.units import TWO_PI

SUB = 4  # points per bin averaging the pulse


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


jax.tree_util.register_dataclass(PEDensity, data_fields=["rate"], meta_fields=["t0", "dt"])
jax.tree_util.register_dataclass(
    Waveform, data_fields=["mean", "variance", "channels"], meta_fields=["t0", "dt", "unit"]
)


def pe_density(light, params, *, dark: bool = True) -> PEDensity:
    """Return the expected photoelectron time profile: photoelectrons per bin of the time nodes' spacing over the gate,
    after the whole time response (each delay cell's delay, the time spread, the transit offset), with the dark counts
    (``dark=False``: the light's alone; its bins sum to ``pe`` within the gate)."""  # noqa: D205, D209
    h = histograms(light)
    v = values(h, params)
    return PEDensity(density(h, v, dark), float(h.gate[0]), float(h.grid.dt))


def pulse_kernel(model, p, dt, power=1):
    """Return (first lag index, kernel): the pulse (to ``power``) averaged over bins [(j - 1) dt, j dt]."""
    r = {f: model.ranges[f"pulse/{f}"] for f in model.pulse.fields() if f"pulse/{f}" in model.ranges}
    lo, hi = model.pulse.support(r)
    j0, j1 = int(math.floor(lo / dt)), int(math.ceil(hi / dt)) + 1
    j = np.arange(j0, j1 + 1, dtype=np.float32)  # on the host in float32: a constant (``detection._f32_grid``)
    f32 = np.float32
    t = jnp.asarray((j[:, None] - f32(1.0) + (np.arange(SUB, dtype=f32)[None, :] + f32(0.5)) / f32(SUB)) * f32(dt))
    return j0, jnp.mean(model.pulse.shape(p, t) ** power, -1)


def _pulse_fold(rho, model, p, dt, power):
    j0, k = pulse_kernel(model, p, dt, power)
    nf = rho.shape[-1]
    if j0 >= 0:  # W_n = sum_j k_j rho_{n - j}: the full convolution shifted by the first lag
        return jnp.concatenate([jnp.zeros(rho.shape[:-1] + (j0,), rho.dtype), fft_conv(rho, k, nf - j0)], -1)
    return fft_conv(rho, k, nf - j0)[..., -j0:]


def expected_waveform(light, params) -> Waveform:
    """Return the expected waveform (mean and variance per sensor and sample; see the module docstring)."""
    h = histograms(light)
    m = h.model
    if m.pulse is None or m.readout != "waveform":
        raise ValueError(f"sensor group {h.group!r}: value set {m.set_name} has no waveform readout (readout "
                         f"{m.readout!r}); use a value set with a pulse and readout 'waveform'")  # fmt: skip
    v = values(h, params)
    nf, step = _n_bins(h)
    m1, m2 = m.charge.moments(v["charge"])
    fe = v["frontend"]

    def block(hb, vb):
        rho = block_density(hb, vb, dark=True)
        gain = jnp.reshape(vb["gain"], (-1, 1))
        mean = fe["baseline"] + gain * m1 * _pulse_fold(rho, m, vb["pulse"], h.grid.dt, 1)[:, ::step]
        var = gain**2 * m2 * _pulse_fold(rho, m, vb["pulse"], h.grid.dt, 2)[:, ::step] + fe["noise"] ** 2
        return mean, var

    _, k = pulse_kernel(m, v["pulse"], h.grid.dt)
    fold = 2 * CELL_BYTES * (FFT_ARRAYS * bucket(nf + k.shape[0]) + 2 * nf)  # the pulse's two folds of the density
    n = h.charge.shape[0]
    mean, var = rows_map(lambda s, b: block(*sensors(h, v, s, b)), n, density_row_bytes(h) + fold,
                         2 * CELL_BYTES * (nf // step + 1))  # fmt: skip
    return Waveform(mean, var, jnp.arange(n), float(h.gate[0]), float(h.grid.dt * step), m.unit)


def waveform_nll(w: Waveform, data):
    """Return the Gaussian negative log-likelihood of measured waveforms (C, n) under the expected waveform.

    An approximation at low occupancy (the shot noise is not Gaussian there): use ``hits_nll`` for single
    photoelectrons. The Monte Carlo error of the expectation is not included.
    """
    d = jnp.asarray(data, jnp.float32)
    if d.shape != w.mean.shape:
        raise ValueError(f"waveform_nll: data of shape {d.shape}, the expected waveform is {w.mean.shape}")
    return 0.5 * jnp.sum((d - w.mean) ** 2 / w.variance + jnp.log(TWO_PI * w.variance))
