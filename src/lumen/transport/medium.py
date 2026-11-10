"""The bulk medium a photon is in, evaluated at its wavelength, and the direction of a bulk event, per channel.

One ``lax.switch`` branch per medium (a real per-photon branch under tallymap) and one per (medium, channel) for the
event's direction. A channel's lobe is drawn at the reference values and weighted by its live / reference density; a
polarised phase function (Rayleigh) is drawn from the photon's own state. After a scattering the state is projected on
the plane perpendicular to the new direction (G4OpRayleigh's rule, and G4OpMieHG's); a re-emitted photon is
unpolarised, at a wavelength drawn from the channel's spectrum after a delay drawn from Exp(tau) (G4OpWLS). Mie
(G4OpMieHG) is a channel like any other: a medium without MIEHG has no Mie branch.
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


def spectrum_arrays(model, ops):
    """Return spectrum -> its tables (the operands')."""
    ids = [id(s) for s in model.spectra]
    return lambda spec: ops.tables["spectra"][ids.index(id(spec))]


def reemit_draw(spec, tau, rand, arrays_of):
    """Return (wavelength, delay) of a re-emission: the spectrum's draw and Exp(tau), from two uniforms."""
    return sg(spec.sample(rand[0], arrays_of(spec))), -tau * jnp.log(rand[1])


def _lobe_fns(lobe):
    """Return (sample, logpdf) of a lobe in the polarised signature (p, d, er, ei, ...): a plain one ignores er, ei."""
    if lobe.pol_logpdf is not None:
        return lobe.pol_sample, lobe.pol_logpdf
    return (lambda p, d, er, ei, rand: lobe.sample(p, d, d, rand)), (lambda p, d, er, ei, wo: lobe.logpdf(p, d, d, wo))


def _draw(ch, params, d, pol, rand, shared):
    """Return (wo, lp, lpr, the live log density as a function of wo) of one channel's lobe (params: live, ref)."""
    p, pr = params
    er, ei = pol
    sample, logpdf = _lobe_fns(ch.lobe)
    w_c = sg(sample(pr, d, sg(er), sg(ei), rand))
    lp = logpdf(p, d, er, ei, w_c)
    lpr = sg(lp) if shared else logpdf(pr, d, sg(er), sg(ei), w_c)
    return w_c, lp, lpr, lambda wo: logpdf(sg(p), sg(d), sg(er), sg(ei), wo)


def channel_draw(model, ops, mi, c, d, pol, lam, rand):  # noqa: PLR0913 -- the event's inputs
    """Return the BulkDraw of channel c of medium mi for direction d, state pol = (er, ei) and wavelength lam."""
    tables, moving, rows = model.tables, model.step.moving_source, model.step.edge_rows
    shared = ops.reference is None
    pm, pmr = ops.params["media"], (ops.params if shared else ops.reference)["media"]
    arrays_of = spectrum_arrays(model, ops)
    branches, offset = [], []
    for med in tables.media:
        offset.append(len(branches))
        for ch in med.channels:

            def f(_, med=med, ch=ch):
                p = pm[med.name][ch.name].get("lobe", {})
                q = pmr[med.name][ch.name].get("lobe", {})
                ref = estimator.lobe_reference(q, ch.lobe, (med.name, ch.name), rows)
                floored = ref is not q
                pr = jax.tree_util.tree_map(sg, ref)
                w_c, lp, lpr, lp_out = _draw(ch, (p, pr), d, pol, rand["lobe"], shared and not floored)
                st = _state_after(ch, *pol, w_c)
                sens = ()
                if moving:
                    E = moving_source.sensitivity(lambda wo: _state_after(ch, *(sg(e) for e in pol), wo), w_c)
                    sens = (sg(jax.jacfwd(lp_out)(w_c)), *E)
                if ch.reemit is None:
                    return BulkDraw(w_c, lp, lpr, lam, F32(0.0), F32(1.0), *st, sens)
                v = pm[med.name][ch.name]
                lam2, delay = reemit_draw(ch.reemit, v["tau"], rand["reemit"], arrays_of)
                return BulkDraw(w_c, lp, lpr, lam2, delay, jnp.asarray(v["yield"], F32), *st, sens)

            branches.append(f)
    if not branches:
        sens = moving_source.no_sensitivities() if moving else ()
        return BulkDraw(d, F32(0.0), F32(0.0), lam, F32(0.0), F32(1.0), *pol, sens)
    idx = jnp.asarray(offset, jnp.int32)[mi] + c
    return jax.lax.switch(jnp.clip(idx, 0, len(branches) - 1), branches, None)
