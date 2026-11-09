"""Expected photoelectrons per sensor: the detection factor applied to the charge histogram.

    pe_s = efficiency_s * sum_{l,a} charge[s, l, a] QE_l CE_a          (summed over replicas)

Linear in the histogram and in every detection value, so exact in QE(lambda) and CE(theta) and differentiable in
both. ``light`` is a ``lumen.readout.Histograms`` or anything with a ``histograms`` attribute holding one; ``params``
is any mapping holding the group's ``sensors/<group>/<field>`` values (a ``lumen.Params`` or a dict of arrays: give a
dict to differentiate).
"""

from __future__ import annotations

import jax.numpy as jnp

from lumen.readout.grid import Histograms


def histograms(light) -> Histograms:
    """Return the Histograms of a light (itself, or its ``histograms``)."""
    h = light if isinstance(light, Histograms) else getattr(light, "histograms", None)
    if not isinstance(h, Histograms):
        raise TypeError(f"readout: expected lumen.readout.Histograms (or a light holding one), got {type(light)}")
    if h.model is None:
        raise ValueError(f"readout: sensor group {h.group!r} has no sensor model")
    return h


def values(h: Histograms, params) -> dict:
    """Return the group's values split by component ({"tts": {"sigma": ...}, "qe": ..., ...}) as float arrays.

    float32 unless 64-bit mode is on and a value is float64 (the float64 reference of the tests).
    """
    v = h.model.split(params, h.group)

    def flt(x):
        if isinstance(x, dict):
            return {k: flt(y) for k, y in x.items()}
        x = jnp.asarray(x)
        return x if x.dtype == jnp.float64 else x.astype(jnp.float32)

    return flt(v)


def expected_pe(h: Histograms, v: dict):
    """Return (S,) expected photoelectrons from the charge histogram and split values."""
    ce = v["ce"] if h.grid.theta_knots is not None else jnp.ones(1, jnp.float32)
    return v["efficiency"] * jnp.einsum("rsla,l,a->s", h.charge, v["qe"], ce)


def pe(light, params):
    """Return the expected photoelectrons per sensor (S,)."""
    h = histograms(light)
    return expected_pe(h, values(h, params))


def pe_error(light, params):
    """Return the Monte Carlo error of ``pe`` per sensor (S,), from the spread of its R >= 2 replicas."""
    h = histograms(light)
    v = values(h, params)
    ce = v["ce"] if h.grid.theta_knots is not None else jnp.ones(1, jnp.float32)
    per = v["efficiency"] * jnp.einsum("rsla,l,a->rs", h.charge, v["qe"], ce)
    r = per.shape[0]
    if r < 2:
        raise ValueError("pe_error: the charge histogram has one replica; simulate with replicas >= 2")
    return jnp.sqrt(r / (r - 1.0) * jnp.sum((per - per.mean(0)) ** 2, 0))


def nodes(h: Histograms, v: dict):
    """Return the time histogram as photoelectrons: (node masses (S, nt), late (S, 2), early (S, 2)).

    The time histogram is scaled per sensor so its total (nodes plus overflow) equals ``pe``: exact for any QE or
    efficiency scale; the time shape keeps the histogram's own wavelength weighting. A fixed delay table is applied
    to the nodes; the overflow cells move by its mean.
    """
    if h.time is None:
        raise ValueError(f"sensor group {h.group!r}: no time histogram (readout 'pe'); simulate with a timed readout")
    H = h.time.sum(0)
    s = H.shape[0]
    late = h.late.sum(0) if h.late is not None else jnp.zeros((s, 2), H.dtype)
    early = h.early.sum(0) if h.early is not None else jnp.zeros((s, 2), H.dtype)
    total = H.sum(-1) + late[:, 0] + early[:, 0]
    scale = expected_pe(h, v) / jnp.where(total != 0.0, total, 1.0)
    m = h.model
    if m.delay is not None:
        H = m.delay.apply(v["delay"], H, h.grid.dt)
        e = jnp.asarray(m.delay.edges, jnp.float32)
        w = v["delay"]["pdf"] / jnp.sum(v["delay"]["pdf"])
        shift = jnp.sum(w * 0.5 * (e[1:] + e[:-1]))
        late, early = late.at[:, 1].add(late[:, 0] * shift), early.at[:, 1].add(early[:, 0] * shift)
    return H * scale[:, None], late * scale[:, None], early * scale[:, None]
