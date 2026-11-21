"""The photon state: the core fields and the optional parts a run asks for, built in one place from an emission row.

- core: x, d (m), t (ns), lam (nm), w, wr, alive, vertex, er, ei (the Jones vector), vol, srf, sn, nz; always;
- cls: the surface lookup's class of the volume (its internal compression, ``lumen.build.lookup``);
- ``Analog``: the photon's analog (truth) record, when a run asks for truth rows (``lumen.transport.step``, "analog");
- ``gen``: the photon's re-emission count (G4OpWLS events: a bulk WLS channel), when a sensor
  feature reads it (``Model.local``);
- ``moving``: a source-position derivative's per-photon state (``StepConfig.moving_source``); dormant, always None in
  this version;

``w`` is the photon's weight (what it deposits, with its derivative), ``wr`` the same weight at the reference values
(the values every draw is made at; stop-gradient), which alone the roulette reads (``lumen.transport.step._after``):
equal to ``w`` in value when the run's values are the reference. ``vertex`` counts the photon's steps and is its
random-number counter. ``vol`` is the placed volume the photon is in,
``srf`` the surface it sits on (the navigator's surface identity, -1 none), ``sn`` the normal of the event that left
it and ``nz`` the zero-length crossings in a row (the stuck rule). Diagnostics keep their own per-photon memory; they
are never state. Sources fill a ``lumen.record.Photons`` row and ``start`` builds the Photon.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from lumen.record import DEPOSIT_FIELDS, Photons

F32, I32 = jnp.float32, jnp.int32


class Analog(NamedTuple):
    """The analog photon riding on a tracked one (the truth record: ``lumen.transport.step``, "analog").

    ``x``: the photon's analog position in its weight, uniform on [0, w) while its analog fate is open (inf once it is
    decided); ``k``: its row's analog weight (the physical photons it stands for, <= 1); at its cathode absorption
    ``sensor`` (-1: none), ``time`` (ns), ``lam`` (nm), ``scale`` (the sensor type's 1 / A_n read per normal
    incidence, else 1: the readout's QE is per photon at normal incidence, ``lumen.surfaces.types.per_normal``) and
    the deposit's local fields ``pos``, ``dir``, ``normal``, ``gen`` (``lumen.transport.step.Deposit``; None unless
    the run reads them).
    """

    x: jax.Array
    k: jax.Array
    sensor: jax.Array
    time: jax.Array
    lam: jax.Array
    scale: jax.Array
    pos: jax.Array | None = None
    dir: jax.Array | None = None
    normal: jax.Array | None = None
    gen: jax.Array | None = None


def analog_fields(local) -> tuple:
    """Return the deposit fields of ``local`` (``Model.local``) the Analog records (not ``inv_an``: its ``scale``)."""
    return tuple(f for f in local if f in Analog._fields)


def analog_start(u, k, local: tuple = ()) -> Analog:
    """Return the Analog part of a photon of analog weight ``k`` (<= 1) at its uniform ``u``: open iff u < k.

    ``local``: the deposit's local fields it records (``Model.local``).
    """
    k = jnp.asarray(k, F32)
    x = jnp.where(u < k, u / jnp.where(k > 0.0, k, 1.0), jnp.inf).astype(F32)
    z = jnp.asarray(0.0, F32)
    loc = {f: jnp.zeros(*DEPOSIT_FIELDS[f]) for f in analog_fields(local)}
    return Analog(x=x, k=k, sensor=jnp.asarray(-1, I32), time=z, lam=z, scale=jnp.asarray(1.0, F32), **loc)


class Photon(NamedTuple):
    """One photon's state (see the module docstring); engines map it over photons."""

    x: jax.Array
    d: jax.Array
    t: jax.Array
    lam: jax.Array
    w: jax.Array
    wr: jax.Array
    alive: jax.Array
    vertex: jax.Array
    er: jax.Array
    ei: jax.Array
    vol: jax.Array
    srf: jax.Array
    sn: jax.Array
    nz: jax.Array
    cls: jax.Array
    analog: Analog | None = None
    gen: jax.Array | None = None
    moving: Any = None  # dormant (StepConfig.moving_source)


class Next(NamedTuple):
    """What an event (a bulk event or a surface outcome) hands on: the state after it, before the roulette.

    ``ok``: the photon continues; ``dtime``: the time elapsed (segment and delay); ``sn``: the normal remembered;
    ``outcome``: the surface outcome (-1 a bulk event); ``reemitted``: the event re-emitted the photon (G4OpWLS: a bulk
    WLS channel); ``fates``: the event's weight accounting, live weights (``lumen.api.Health``): the weight it goes on
    with before the drawn outcome's or direction's live / proposal factor (its expectation over that draw), the weight
    absorbed in the medium (the path's absorption, a WLS yield below 1) and at the surface (1 - the continuing shares).
    Its path is weighted live / reference as the continuing weight is, so at the reference values the photon's weight is
    exactly their sum (+ ``lost``). ``far``: a surface outcome left the photon beyond the surface (in the volume after).
    """

    x: jax.Array
    d: jax.Array
    w: jax.Array
    cls: jax.Array
    ok: jax.Array
    lam: jax.Array
    dtime: jax.Array
    sn: jax.Array
    outcome: jax.Array
    er: jax.Array
    ei: jax.Array
    reemitted: jax.Array
    fates: tuple | None = None
    far: jax.Array | None = None


def start(row: Photons, vol, cls, *, gen=False) -> Photon:
    """Return the Photon of an emitted row in volume ``vol`` (-1: located at its first step) and class ``cls``.

    The row carries ``pol`` (a source's rows always do). The optional part exists only when asked for: ``gen`` (the
    re-emission count, 0 at the start). The reference weight starts at the weight.
    """
    x, d = jnp.asarray(row.x, F32), jnp.asarray(row.d, F32)
    pol = jnp.asarray(row.pol, F32)
    m1 = jnp.asarray(-1, I32)
    w, t = jnp.asarray(row.w, F32), jnp.asarray(row.t, F32)
    return Photon(x=x, d=d, t=t, lam=jnp.asarray(row.lam, F32), w=w, wr=jax.lax.stop_gradient(w),
                  alive=jnp.asarray(True), vertex=jnp.asarray(0, I32), er=pol[0], ei=pol[1], vol=jnp.asarray(vol, I32),
                  srf=m1, sn=jnp.zeros(3, F32), nz=jnp.asarray(0, I32), cls=jnp.asarray(cls, I32),
                  gen=jnp.asarray(0, I32) if gen else None)  # fmt: skip
