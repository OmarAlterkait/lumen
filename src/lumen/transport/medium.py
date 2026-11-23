"""The bulk medium a photon is in, evaluated at its wavelength, and the direction of a bulk event, per channel.

One ``lax.switch`` branch per medium (a real per-photon branch under tallymap) and one per (medium, channel) for the
event's direction. A channel's lobe is drawn at the reference values and weighted by its live / reference density; a
polarised phase function (Rayleigh) is drawn from the photon's own state. After a scattering the state is projected on
the plane perpendicular to the new direction (G4OpRayleigh's rule, and G4OpMieHG's); a re-emitted photon is
unpolarised and goes on as its channel's emission record gives it (``lumen.media.emission``: the wavelength at or
below the absorbed energy, the delay, the weight; G4OpWLS). Mie
(G4OpMieHG) is a channel like any other: a medium without MIEHG has no Mie branch.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen.media import bulk
from lumen.rules.choices import DEFENSIVE_SHARE
from lumen.surfaces import polar

sg = jax.lax.stop_gradient
F32 = jnp.float32


def medium_at(tables, pm, lam, mi):
    """Return (n, n_g, mu_abs, mu (C_max,)) of medium mi at lam (nm); an opaque medium gives n = n_g = 1."""
    c_max = max([len(m.channels) for m in tables.media] + [1])

    def one(m):
        def f(_):
            e = bulk.evaluate(m, pm[m.name], lam)
            mu = jnp.pad(e["mu"], (0, c_max - e["mu"].shape[-1]))
            return (e.get("n", jnp.ones_like(lam)), e.get("n_g", jnp.ones_like(lam)), e["mu_abs"] + 0.0 * lam,
                    mu.astype(F32))  # fmt: skip

        return f

    return jax.lax.switch(mi, [one(m) for m in tables.media], None)


def index_at(tables, pm, lam, mi):
    """Return n of medium mi at lam (nm); 1 for an opaque medium."""

    def one(m):
        return lambda _: (m.index(pm[m.name]["index"], lam) if m.index is not None else jnp.ones_like(lam))

    return jax.lax.switch(mi, [one(m) for m in tables.media], None)


def _state_after(ch, er, ei, wo):
    """Return the polarisation after an event of channel ch into wo (re-emission: unpolarised)."""
    if ch.emission is not None:
        return jnp.zeros_like(er), jnp.zeros_like(ei)
    return polar.project(er, ei, wo)


class BulkDraw(NamedTuple):
    """What a channel's draw gives.

    The direction, its live and reference log densities, wavelength, delay, yield weight (and the yield weight at the
    reference values), the polarisation after and whether the channel re-emits.
    """

    wo: jax.Array
    lp: jax.Array
    lpr: jax.Array
    lam: jax.Array
    delay: jax.Array
    yield_w: jax.Array
    yield_r: jax.Array
    er: jax.Array
    ei: jax.Array
    reemitted: jax.Array  # the channel re-emits (G4OpWLS)


def _lobe_fns(lobe):
    """Return (sample, logpdf) of a lobe in the polarised signature (p, d, er, ei, ...): a plain one ignores er, ei."""
    if lobe.pol_logpdf is not None:
        return lobe.pol_sample, lobe.pol_logpdf
    return (lambda p, d, er, ei, rand: lobe.sample(p, d, d, rand)), (lambda p, d, er, ei, wo: lobe.logpdf(p, d, d, wo))


def _draw(ch, params, d, pol, rand, shared):
    """Return (wo, lp, lpr) of one channel's lobe (params: live, ref)."""
    p, pr = params
    er, ei = pol
    sample, logpdf = _lobe_fns(ch.lobe)
    w_c = sg(sample(pr, d, sg(er), sg(ei), rand))
    lp = logpdf(p, d, er, ei, w_c)
    lpr = sg(lp) if shared else logpdf(pr, d, sg(er), sg(ei), w_c)
    return w_c, lp, lpr


def channel_draw(model, ops, mi, c, d, pol, lam, rand):  # noqa: PLR0913 -- the event's inputs
    """Return the BulkDraw of channel c of medium mi for direction d, state pol = (er, ei) and wavelength lam.

    A lobe whose parameter is a defensive row (``StepConfig.edge_rows``) is drawn at its cover of the reference
    (``Lobe.cover``: the one defensive mixture), and its weight is the live density over the one drawn from.
    """
    tables = model.tables
    shared = ops.reference is None
    covered = {(m, ch) for m, ch, _ in model.step.edge_rows.lobes}
    pm, pmr = ops.params["media"], (ops.params if shared else ops.reference)["media"]
    branches, offset = [], []
    for med in tables.media:
        offset.append(len(branches))
        for ch in med.channels:

            def f(_, med=med, ch=ch):
                p = pm[med.name][ch.name].get("lobe", {})
                pr = jax.tree_util.tree_map(sg, pmr[med.name][ch.name].get("lobe", {}))
                cov = (med.name, ch.name) in covered and ch.lobe.cover is not None
                if cov:
                    pr = ch.lobe.cover(pr, DEFENSIVE_SHARE)
                w_c, lp, lpr = _draw(ch, (p, pr), d, pol, rand["lobe"], shared and not cov)
                er, ei = _state_after(ch, *pol, w_c)
                if ch.emission is None:
                    return BulkDraw(wo=w_c, lp=lp, lpr=lpr, lam=lam, delay=F32(0.0), yield_w=F32(1.0),
                                    yield_r=F32(1.0), er=er, ei=ei,
                                    reemitted=jnp.bool_(False))  # fmt: skip
                v, vr = pm[med.name][ch.name], jax.tree_util.tree_map(sg, pmr[med.name][ch.name])
                e = ch.emission.sample(v, vr, rand["reemit"], lam, ops.emission[med.name][ch.name])
                return BulkDraw(wo=w_c, lp=lp, lpr=lpr, lam=e.lam, delay=e.delay, yield_w=e.w,
                                yield_r=e.w_r, er=er, ei=ei, reemitted=jnp.bool_(True))  # fmt: skip

            branches.append(f)
    if not branches:
        return BulkDraw(wo=d, lp=F32(0.0), lpr=F32(0.0), lam=lam, delay=F32(0.0), yield_w=F32(1.0),
                        yield_r=F32(1.0), er=pol[0], ei=pol[1], reemitted=jnp.bool_(False))  # fmt: skip
    idx = jnp.asarray(offset, jnp.int32)[mi] + c
    return jax.lax.switch(jnp.clip(idx, 0, len(branches) - 1), branches, None)
