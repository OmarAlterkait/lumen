"""The tallymap item: what one photon's loop starts from, how long it runs, and what each step adds to the outputs.

    spec = photons(TallyShape(n_sensors, lam_edges))      # the default: per-sensor charge and time weight, health
    tally = lumen.run.run(spec, model, ops, rows, key, max_vertices=..., replicas=...)

``ItemSpec(init, alive, emits, outputs, per_item)``: ``init(model, row)`` the Photon of an emission row (the row is the
item's own mapped input, so a re-emitted photon keeps its row); ``alive(ph)``; ``emits(dep, slot)`` the (output, cell,
value) adds of one step's Deposit for replica ``slot``; ``outputs(replicas)`` {name: (cells, exact)}; ``per_item(ph)``
what each photon reports when it ends.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from lumen.transport import state

HEALTH = ("lost", "abandoned", "escaped", "late")  # weights; plus the counts "stuck" (pushed, abandoned)


class TallyShape(NamedTuple):
    """The histograms' static shape: the sensors and the wavelength band edges (nm, ascending)."""

    n_sensors: int
    lam_edges: tuple


class ItemSpec(NamedTuple):
    """One photon's loop (see the module docstring)."""

    init: Any
    alive: Any
    emits: Any
    outputs: Any
    per_item: Any


def start_photon(model, row):
    """Return the Photon of an emission row, located (its volume and class from the navigator)."""
    nav = model.geometry.nav
    vol = nav.locate(jnp.asarray(row.x, jnp.float32))
    st = model.step
    return state.start(row, vol, nav.dev()["cls"][vol], stacked=model.stacked, moving=st.moving_source,
                       edge=bool(st.edge_rows))  # fmt: skip


def photons(tally: TallyShape) -> ItemSpec:
    """Return the default ItemSpec: per sensor and band the detected weight "Q" and its time-weighted sum "Tw"."""
    n_lam = len(tally.lam_edges) - 1
    cells = tally.n_sensors * n_lam

    def emits(dep, slot):
        e = jnp.asarray(tally.lam_edges, jnp.float32)
        b = jnp.searchsorted(e, dep.lam, side="right") - 1
        band = jnp.where((b >= 0) & (b < n_lam), b, -1)
        idx = jnp.where((dep.sensor >= 0) & (band >= 0), dep.sensor * n_lam + band, -1)
        idx = jnp.where(idx >= 0, idx + slot * cells, -1)
        adds = [("Q", idx, dep.q), ("Tw", idx, dep.q * dep.time)]
        adds += [(k, jnp.int32(0), getattr(dep, k)) for k in HEALTH]
        return adds + [("stuck", jnp.where(f, i, -1), 1.0) for i, f in enumerate((dep.pushed, dep.abandon))]

    def outputs(replicas):
        out = {"Q": (replicas * cells, True), "Tw": (replicas * cells, True), "stuck": (2, False)}
        return out | {k: (1, True) for k in HEALTH}

    return ItemSpec(start_photon, lambda ph: ph.alive, emits, outputs, lambda ph: dict(vertices=ph.vertex))
