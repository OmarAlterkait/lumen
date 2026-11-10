"""The photon state: 13 core fields and the optional parts a run asks for, built in one place from an emission row.

- core: x, d (m), t (ns), lam (nm), w, alive, vertex, er, ei (the Jones vector), vol, srf, sn, nz; always;
- cls, tag: the surface lookup's class of the volume (its internal compression, ``lumen.build.surfaces``) and its
  sensor index;
- ``Stack``: layer, svol, scls, stag, when a layered (coated) surface exists;
- ``MovingSource``: free and the chain (``lumen.transport.moving_source``), when wrt holds a moving-path parameter;
- ``z``: the edge-row factor (``lumen.transport.estimator.edge_floor``), when wrt holds a parameter at a domain edge.

``vertex`` counts the photon's steps and is its random-number counter. ``vol`` is the placed volume the photon is in,
``srf`` the surface it sits on (the navigator's surface identity, -1 none), ``sn`` the normal of the event that left
it and ``nz`` the zero-length crossings in a row (the stuck rule). Diagnostics keep their own per-photon memory; they
are never state. Sources fill an ``Emission`` and ``start`` builds the Photon.

Notes
-----
Ported from the baseline's lumen.state (Photon, emit, with_free, chain); its ``last`` field is a diagnostic now
(``lumen.diagnostics``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen.transport import moving_source

F32, I32 = jnp.float32, jnp.int32


class Emission(NamedTuple):
    """One emitted photon (or a batch): what a source gives the loop.

    ``x`` (m), ``d`` (unit), ``t`` (ns), ``lam`` (nm), ``w`` (expected photons it stands for), ``polarisation``
    (..., 2, 3): the Jones vector's real and imaginary parts (zeros: unpolarised), ``drawn_direction``: d was drawn
    from a density in solid angle (the moving-source method may move it).
    """

    x: jax.Array
    d: jax.Array
    t: jax.Array
    lam: jax.Array
    w: jax.Array
    polarisation: jax.Array
    drawn_direction: jax.Array


class Stack(NamedTuple):
    """Inside a layered surface: the layer, and the volume, class and tag beyond the remembered surface.

    layer 0: not in one; 1: about to cross the film outward; 2: at the backing.
    """

    layer: jax.Array
    svol: jax.Array
    scls: jax.Array
    stag: jax.Array


class MovingSource(NamedTuple):
    """The moving-source part: ``free`` (the direction has free coordinates the method can move) and the chain."""

    free: jax.Array
    chain: moving_source.Chain


class Photon(NamedTuple):
    """One photon's state (see the module docstring); engines map it over photons."""

    x: jax.Array
    d: jax.Array
    t: jax.Array
    lam: jax.Array
    w: jax.Array
    alive: jax.Array
    vertex: jax.Array
    er: jax.Array
    ei: jax.Array
    vol: jax.Array
    srf: jax.Array
    sn: jax.Array
    nz: jax.Array
    cls: jax.Array
    tag: jax.Array
    stack: Stack | None = None
    moving: MovingSource | None = None
    z: jax.Array | None = None


class Next(NamedTuple):
    """What an event (a bulk event or a surface outcome) hands on: the state after it, before the roulette.

    ``ok``: the photon continues; ``dtime``: the time elapsed (segment and delay); ``sn``: the normal remembered;
    ``outcome``: the surface outcome (-1 a bulk event); ``chain``, ``free``: the moving-source part (None without).
    """

    x: jax.Array
    d: jax.Array
    w: jax.Array
    cls: jax.Array
    tag: jax.Array
    ok: jax.Array
    lam: jax.Array
    dtime: jax.Array
    layer: jax.Array
    sn: jax.Array
    outcome: jax.Array
    er: jax.Array
    ei: jax.Array
    chain: moving_source.Chain | None
    free: jax.Array | None


def start(row: Emission, vol, cls, *, stacked=False, moving=False, edge=False) -> Photon:
    """Return the Photon of an emission row in volume ``vol`` (-1: located at its first step) and class ``cls``.

    The optional parts exist only when asked for: ``stacked`` (a layered surface exists), ``moving`` (the
    moving-source method), ``edge`` (edge rows are on). The sensor tag starts unknown (-1), as the baseline's.
    """
    x, d = jnp.asarray(row.x, F32), jnp.asarray(row.d, F32)
    pol = jnp.asarray(row.polarisation, F32)
    m1 = jnp.asarray(-1, I32)
    stack = Stack(jnp.asarray(0, I32), m1, m1, m1) if stacked else None
    free = jnp.asarray(row.drawn_direction, bool)
    mv = None
    if moving:
        chain = jax.lax.cond(free, lambda: moving_source.free_start(x, d), lambda: moving_source.none_start(x, d))
        mv = MovingSource(free, chain)
    return Photon(x, d, jnp.asarray(row.t, F32), jnp.asarray(row.lam, F32), jnp.asarray(row.w, F32),
                  jnp.asarray(True), jnp.asarray(0, I32), pol[0], pol[1], jnp.asarray(vol, I32), m1,
                  jnp.zeros(3, F32), jnp.asarray(0, I32), jnp.asarray(cls, I32), m1, stack, mv,
                  jnp.asarray(1.0, F32) if edge else None)  # fmt: skip
