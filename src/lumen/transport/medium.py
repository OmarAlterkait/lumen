"""The bulk medium a photon is in, evaluated at its wavelength, and the direction of a bulk event, per channel.

One ``lax.switch`` branch per medium (a real per-photon branch under tallymap) and one per (medium, channel) for the
event's direction. A channel's lobe is drawn at the reference values and weighted by its live / reference density; a
polarised phase function (Rayleigh) is drawn from the photon's own state. After a scattering the state is projected on
the plane perpendicular to the new direction (G4OpRayleigh's rule, and G4OpMieHG's); a re-emitted photon is
unpolarised, at a wavelength drawn from the channel's spectrum after a delay drawn from Exp(tau) (G4OpWLS).

Notes
-----
Ported from the baseline's lumen.step (_medium_at, _index_at, _channel_lobe, _bulk_state). Mie (G4OpMieHG) is a
channel like any other: a medium without MIEHG has no Mie branch.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen.media import bulk
from lumen.surfaces import polar
from lumen.transport import estimator, moving_source

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
    if ch.reemit is not None:
        return jnp.zeros_like(er), jnp.zeros_like(ei)
    return polar.project(er, ei, wo)


class BulkDraw(NamedTuple):
    """What a channel's draw gives.

    The direction, its live and reference log densities, wavelength, delay, yield weight, the polarisation after and
    (moving source) the live log density's gradient and the state's sensitivity.
    """

    wo: jax.Array
    lp: jax.Array
    lpr: jax.Array
    lam: jax.Array
    delay: jax.Array
    yield_w: jax.Array
    er: jax.Array
    ei: jax.Array
    sens: tuple


def _draw(ch, params, d, pol, rand, shared):
    """Return (wo, lp, lpr, the live log density as a function of wo) of one channel's lobe (params: live, ref)."""
    p, pr = params
    er, ei = pol
    lobe = ch.lobe
    if lobe.pol_logpdf is not None:
        w_c = sg(lobe.pol_sample(pr, d, sg(er), sg(ei), rand))
        lp = lobe.pol_logpdf(p, d, er, ei, w_c)
        lpr = sg(lp) if shared else lobe.pol_logpdf(pr, d, sg(er), sg(ei), w_c)
        return w_c, lp, lpr, lambda wo: lobe.pol_logpdf(sg(p), sg(d), sg(er), sg(ei), wo)
    w_c = sg(lobe.sample(pr, d, d, rand))
    lp = lobe.logpdf(p, d, d, w_c)
    lpr = sg(lp) if shared else lobe.logpdf(pr, d, d, w_c)
    return w_c, lp, lpr, lambda wo: lobe.logpdf(sg(p), sg(d), sg(d), wo)


def channel_draw(ctx, mi, c, d, pol, lam):
    """Return the BulkDraw of channel c of medium mi for direction d, state pol = (er, ei) and wavelength lam.

    ``ctx``: (tables, live params, reference params, uniforms, shared, moving, edge rows, spectrum -> its tables).
    """
    tables, pm, pmr, u, shared, moving, rows, arrays_of = ctx
    branches, offset = [], []
    for med in tables.media:
        offset.append(len(branches))
        for ch in med.channels:

            def f(_, med=med, ch=ch):
                p = pm[med.name][ch.name].get("lobe", {})
                q = pmr[med.name][ch.name].get("lobe", {})
                ref = estimator.mie_reference(q, med.name, rows) if ch.name == "mie" else q
                floored = ref is not q
                pr = jax.tree_util.tree_map(sg, ref)
                w_c, lp, lpr, lp_out = _draw(ch, (p, pr), d, pol, u["lobe"], shared and not floored)
                st = _state_after(ch, *pol, w_c)
                sens = ()
                if moving:
                    E = moving_source.sensitivity(lambda wo: _state_after(ch, *(sg(e) for e in pol), wo), w_c)
                    sens = (sg(jax.jacfwd(lp_out)(w_c)), *E)
                if ch.reemit is None:
                    return BulkDraw(w_c, lp, lpr, lam, F32(0.0), F32(1.0), *st, sens)
                v = pm[med.name][ch.name]
                lam2 = sg(ch.reemit.sample(u["reemit"][0], arrays_of(ch.reemit)))
                return BulkDraw(w_c, lp, lpr, lam2, -v["tau"] * jnp.log(u["reemit"][1]), jnp.asarray(v["yield"], F32),
                                *st, sens)  # fmt: skip

            branches.append(f)
    if not branches:
        z = jnp.zeros((3, 2), F32)
        return BulkDraw(d, F32(0.0), F32(0.0), lam, F32(0.0), F32(1.0), *pol,
                        (jnp.zeros(3, F32), z, z) if moving else ())  # fmt: skip
    idx = jnp.asarray(offset, jnp.int32)[mi] + c
    return jax.lax.switch(jnp.clip(idx, 0, len(branches) - 1), branches, None)
