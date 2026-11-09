"""The photon loop on tallymap: the only module that imports tallymap.

    tally = run(spec, model, ops, rows, key, max_vertices=MAX_VERTICES, replicas=1, diagnostics=(), backend="auto")

Each photon is one tallymap item that loops the same step until it ends; its outputs are tallymap sums (exact int64
fixed point on the GPU: order-independent, so a run is bitwise independent of how photons are scheduled). There is
no vertex cap in the physics: ``max_vertices`` is a guard against a photon that cannot end, and a photon it stops is
reported (``truncated``, ``n_truncated``), never dropped silently. ``backend="auto"``: "cuda" on a GPU (exact sums),
"cpu" elsewhere (plain JAX, inexact sums, ~2.5e-4 from the GPU). Photon ``i`` uses the key ``fold_in(key, first_id +
i)``; replicas split the photons into contiguous blocks with their own cells. tallymap's own names stay in this file.

Notes
-----
Ported from the baseline's lumen.engine_tallymap.run; its environment switches and diagnostics options are gone
(``lumen.diagnostics`` instead).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import tallymap as tm

from lumen import _env
from lumen._core import rng
from lumen.transport.step import step

MAX_VERTICES = 1 << 22  # the guard (vertices per photon), not a physics cut
TALLYMAP_BACKEND = {"cuda": "cuda", "cpu": "lockstep"}  # lumen's backend names -> tallymap's
CLOSED_OVER = "baked_floats"  # the diagnostics key of an item program's closed-over float arrays
OVERFLOW = "exact_overflow"  # the stats key of an exact sum's overflow
SCHEDULE = "step"  # tallymap refills a lane per step, not per photon: 1.09-1.39x faster for lumen (L40S)


class Tally(NamedTuple):
    """A run's outputs ({name: cells}), each photon's ``per_item`` report, and whether no photon was truncated."""

    outputs: dict
    per_item: dict
    complete: jax.Array
    overflow: jax.Array


def run(  # noqa: PLR0913 -- SPEC section 5's signature
    spec, model, ops, rows, key, *, max_vertices=MAX_VERTICES, replicas=1, diagnostics=(), backend="auto", first_id=0,
    n_active=None, form="operands",
):  # fmt: skip
    """Return the Tally of the photons ``rows`` (an Emission batch) under ``key`` (raw key data).

    ``form="operands"`` (the default): one jitted loop whose operands are ``ops``, the key, ``first_id`` and
    ``n_active`` (only rows i < n_active run): changing any value never recompiles, and a loop that closes over a
    float array is refused (``ClosedOverValues``). ``form="closure"``: the baseline's eager form, values baked in
    (comparisons only).
    """
    static = (spec, model, int(max_vertices), int(replicas), tuple(diagnostics), backend, form)
    loop = _loop if form == "closure" else _LOOP
    return loop(static, ops, rows, key, jnp.asarray(first_id, jnp.int32), n_active)


class ClosedOverValues(ValueError):  # noqa: N818 -- the documented error catalogue's name
    """The photon loop closes over a float array: a value baked into the kernel, which recompiles when it changes."""


def _refuse_closed_over(tag):
    from tallymap import _diagnostics

    floats = _diagnostics.PROGRAMS[-1][CLOSED_OVER] if _diagnostics.PROGRAMS else []
    if floats:
        raise ClosedOverValues(f"the photon loop ({tag}) closes over {len(floats)} float array(s) {floats[:4]}: pass "
                               "every value through the Operands (lumen.transport.step.operands)")  # fmt: skip


def _loop(static, ops, rows, key, first_id, n_active):
    spec, model, max_vertices, replicas, diagnostics, backend, form = static
    n = rows.x.shape[0]
    cuda = backend == "cuda" or (backend == "auto" and jax.default_backend() == "gpu")
    if cuda:
        _env.tallymap_cache()  # fail closed before tallymap compiles a kernel into HOME
    shapes = spec.outputs(replicas) | {"truncated": (1, True), "n_truncated": (1, True)}
    for dg in diagnostics:
        shapes |= dg.outputs()
    outputs = {k: tm.Sum((cells,), exact=exact and cuda) for k, (cells, exact) in shapes.items()}

    def item(tl, x, *b):
        o, k, fid = b[0] if b else (ops, key, first_id)
        ph = spec.init(model, o, x["row"])
        pkey = rng.photon_key(k, fid + x["id"])
        slot = x["id"] * replicas // n

        def body(c):
            p, tl, mem = c
            p2, dep = step(model, o, p, pkey)
            adds, mem2 = list(spec.emits(dep, slot, o.grid)), []
            for dg, m in zip(diagnostics, mem, strict=True):
                m2, a = dg.update(m, p, p2, dep, x["id"])
                adds, mem2 = adds + a, mem2 + [m2]
            for name, cell, value in adds:
                tl = tl.add(name, cell, value)
            return p2, tl, tuple(mem2)

        mem0 = tuple(dg.init(ph) for dg in diagnostics)
        ph, tl, _ = jax.lax.while_loop(lambda c: spec.alive(c[0]) & (c[0].vertex < max_vertices), body, (ph, tl, mem0))
        tl = tl.add("truncated", jnp.int32(0), jnp.where(ph.alive, ph.w, 0.0))  # stopped by the guard: reported
        tl = tl.add("n_truncated", jnp.int32(0), jnp.where(ph.alive, 1.0, 0.0))
        return tl, spec.per_item(ph)

    xs = dict(row=rows, id=jnp.arange(n, dtype=jnp.int32))
    closure = form == "closure"
    opts = dict(outputs=outputs, max_steps=max_vertices + 1, backend=TALLYMAP_BACKEND["cuda" if cuda else "cpu"],
                _schedule=SCHEDULE, in_axes=0 if closure else (0, None))  # fmt: skip
    r = tm.vmap(item, **opts)(*((xs,) if closure else (xs, (ops, key, first_id))), n_active=n_active)
    if not closure:
        _refuse_closed_over(form)
    complete = (r.outputs["n_truncated"][0] == 0) & (r.stats["n_truncated"] == 0)
    return Tally(dict(r.outputs), dict(r.per_item), complete, r.stats.get(OVERFLOW, jnp.int32(0)))


_LOOP = jax.jit(_loop, static_argnums=0)
