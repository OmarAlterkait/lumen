"""Diagnostics: per-photon cells of your own, recorded by a run beside its light.

    light = lumen.simulate(det, source, samples=N, seed=0, chunk=N, diagnostics=(Trace(steps=24, n_traced=N),))
    light.diagnostics["trace"]                                   # the cells, by output name

A diagnostic declares ``outputs()`` {name: (cells, exact)} (names the run does not use itself, nor another
diagnostic) and is one of two kinds:

- a probe of the photon's state: ``init(ph)``, its memory for one photon, and ``update(mem, ph, ph2, dep, pid)`` ->
  (memory, [(output, cell, value)]) after each step, ``pid`` the photon's id in its launch. A probe sees every step,
  so its run is exact and in one chunk (``samples <= chunk``).
- a records diagnostic: ``emits(dep, weights)`` -> [(output, cell, value)] of one deposit record, built from
  the run's deposit records like the readout (``lumen.run.rebuild``: each add times the record's row weight, ``dep`` a
  ``lumen.run.Record``), never inside the loop, summed over the chunks. Its optional ``reads`` names the deposit
  fields it reads beyond sensor, q, time and lam (``lumen.sensors.features.FIELDS``); its optional
  ``moments(dep, weights)`` -> [(output, cell, value)] are adds whose per-photon sum per cell is squared (a per-photon
  second moment, ``lumen.run.rebuild``; outputs it declares too).

``Trace``: each photon's state after each of its first ``steps`` steps, ``TRACE_FIELDS`` per step (float32 cells,
each written once; integers below 2^24; ``srf`` + 1 in two 16-bit halves). ``last`` is the event that started the
current straight path (0 the source, 1 a bulk event, 2 + o a surface outcome o, ``lumen.surfaces.outcomes``; a
STRAIGHT crossing keeps it).
``lost`` is the step's lost plus abandoned weight. A diagnostic's Deposit carries the readout's weights (the row's
weight times the photon's, ``lumen.run.absolute``); the photon state ``ph.w`` is relative to its emission row.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from lumen.surfaces.outcomes import STRAIGHT

TRACE_FIELDS = ("alive", "x0", "x1", "x2", "d0", "d1", "d2", "vol", "cls", "last", "srf_hi", "srf_lo", "lost", "pair")
HALF = 65536  # an integer's two halves: (v + 1) // HALF, (v + 1) % HALF


def halves(s):
    """Return the two float32 halves of an int32 s + 1 (exact below 2^31)."""
    s1 = jnp.asarray(s, jnp.int32) + 1
    return (s1 // HALF).astype(jnp.float32), (s1 % HALF).astype(jnp.float32)


def last_event(last, event):
    """Return the photon's ``last`` after a step whose Deposit.event is ``event`` (-2 a bulk event)."""
    return jnp.where(event == -2, 1, jnp.where(event == STRAIGHT, last, 2 + jnp.maximum(event, 0))).astype(jnp.int32)


@dataclass(frozen=True, kw_only=True)
class Trace:
    """The per-step state of the first ``steps`` steps of ``n_traced`` photons (see the module docstring)."""

    steps: int
    n_traced: int

    def outputs(self):
        """Return {"trace": (cells, exact)}."""
        return {"trace": (self.n_traced * self.steps * len(TRACE_FIELDS), False)}

    def init(self, ph):
        """Return the memory of a fresh photon: its ``last`` (0, the source)."""
        return jnp.int32(0)

    def update(self, mem, ph, ph2, dep, pid):
        """Return (memory, adds) after one step."""
        last = last_event(mem, dep.event)

        def f(v):
            return jnp.asarray(v, jnp.float32)

        row = ([f(ph.alive)] + [f(ph2.x[a]) for a in range(3)] + [f(ph2.d[a]) for a in range(3)]
               + [f(ph2.vol), f(ph2.cls), f(last), *halves(ph2.srf), f(dep.lost + dep.abandoned),
                  f(dep.pair)])  # fmt: skip
        ok = (ph.vertex < self.steps) & (pid < self.n_traced)  # a photon beyond the traced ones writes nothing
        base = jnp.where(ok, (pid * self.steps + ph.vertex) * len(TRACE_FIELDS), -1)
        return last, [("trace", jnp.where(base >= 0, base + i, -1), v) for i, v in enumerate(row)]

    def read(self, cells):
        """Return the trace (n_traced, steps, len(TRACE_FIELDS)) from its output cells."""
        return np.asarray(cells, np.float32).reshape(self.n_traced, self.steps, len(TRACE_FIELDS))
