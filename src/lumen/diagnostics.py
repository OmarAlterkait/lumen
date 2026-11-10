"""Diagnostics: probes with their own per-photon memory, run alongside the photon loop; never photon state.

    run(loop, model, ops, rows, key, ..., diagnostics=(Trace(steps=24, n_photons=N),))

A diagnostic declares ``outputs()`` {name: (cells, exact)}, ``init(ph)`` its memory for one photon and
``update(mem, ph, ph2, dep, pid)`` -> (memory, [(output, cell, value)]) after each step.

``Trace``: each photon's state after each of its first ``steps`` steps, ``TRACE_FIELDS`` per step (float32 cells,
each written once; integers below 2^24; ``srf`` + 1 in two 16-bit halves). ``last`` is the event that started the
current straight path (0 the source, 1 a bulk event, 2 + o a surface outcome o; a STRAIGHT crossing keeps it).
``lost`` is the step's lost plus abandoned weight.
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


def from_halves(hi, lo):
    """Return the int32 that ``halves`` split (numpy)."""
    return (np.asarray(hi, np.int64) * HALF + np.asarray(lo, np.int64) - 1).astype(np.int32)


def last_event(last, event):
    """Return the photon's ``last`` after a step whose Deposit.event is ``event`` (-2 a bulk event)."""
    return jnp.where(event == -2, 1, jnp.where(event == STRAIGHT, last, 2 + jnp.maximum(event, 0))).astype(jnp.int32)


@dataclass(frozen=True)
class Trace:
    """The per-step state of the first ``steps`` steps of ``n_photons`` photons (see the module docstring)."""

    steps: int
    n_photons: int

    def outputs(self):
        """Return {"trace": (cells, exact)}."""
        return {"trace": (self.n_photons * self.steps * len(TRACE_FIELDS), False)}

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
        base = jnp.where(ph.vertex < self.steps, (pid * self.steps + ph.vertex) * len(TRACE_FIELDS), -1)
        return last, [("trace", jnp.where(base >= 0, base + i, -1), v) for i, v in enumerate(row)]

    def read(self, cells):
        """Return the trace (n_photons, steps, len(TRACE_FIELDS)) from its output cells."""
        return np.asarray(cells, np.float32).reshape(self.n_photons, self.steps, len(TRACE_FIELDS))
