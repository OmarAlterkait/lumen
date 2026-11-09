"""A bulk event: the free path ends before the next surface, in a scattering or re-emitting channel.

The channel is drawn from the reference rates (in Geant4's fixed order: Rayleigh, Mie, WLS) and weighted by the
live / reference density of the free path in that channel (mu_c exp(-mu s)), times the absorption over the path
(exp(-mu_abs s), a weight, never an event: G4OpAbsorption in expectation). The direction comes from the channel's
lobe (``lumen.transport.medium.channel_draw``).

Notes
-----
Ported from the baseline's lumen.step (bulk_event).
"""

from __future__ import annotations

import jax.numpy as jnp

from lumen.transport import medium
from lumen.transport import moving_source as ms
from lumen.transport.estimator import choose, magic, magic_log
from lumen.transport.state import Next


def bulk_event(ph, vx, next_cls, ctx):
    """Return the Next of a bulk event at the free-path end (``step``'s vertex quantities vx).

    ``next_cls``: the photon's class there (before the first invisible crossing the start's, after the last the one
    before the visible crossing, between them unknown); ``ctx``: ``medium.channel_draw``'s context.
    """
    moving = ctx[5]
    c = choose(vx.mu_c_r, vx.u["channel"][0])
    cs = jnp.maximum(c, 0)
    F = vx.free
    w = magic(vx.mu_c[cs] * jnp.exp(-vx.mu * F.s), vx.mu_c_r[cs] * jnp.exp(-vx.mu_r * vx.s)) * jnp.exp(-vx.mu_abs * F.s)
    w = w * F.f if moving else w
    b = medium.channel_draw(ctx, vx.mi, cs, F.wi, (F.er, F.ei), ph.lam)
    chain, free = (ms.free_start(F.x, b.wo, b.sens[0], E=b.sens[1:]), jnp.bool_(True)) if moving else (None, None)
    return Next(x=F.x, d=b.wo, w=ph.w * w * magic_log(b.lp, b.lpr) * b.yield_w, cls=next_cls, tag=ph.tag,
                  ok=jnp.bool_(True), lam=b.lam, dtime=F.dt + F.s / vx.v_g + b.delay, layer=jnp.int32(0),
                  sn=vx.q["normal"].astype(jnp.float32), outcome=jnp.int32(-1), er=b.er, ei=b.ei, chain=chain,
                  free=free)  # fmt: skip
