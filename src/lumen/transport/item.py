"""The tallymap item: what one photon's loop starts from, how long it runs, and what each step adds to the outputs.

    loop = photons(HistogramShape(n_sensors, lam_edges))      # the default: per-sensor charge and time weight, health
    tally = lumen.run.run(loop, model, ops, rows, key, max_vertices=..., replicas=...)

``PhotonLoop(init, alive, emits, outputs, per_item)``: ``init(model, ops, row)`` the Photon of an emission row (the row
is the item's own mapped input, so a re-emitted photon keeps its row); ``alive(ph)``; ``emits(dep, replica, grid)`` the
(output, cell, value) adds of one step's Deposit for the replica index ``replica`` (``grid``: ``Operands.grid``);
``outputs(replicas)`` {name: (cells, exact)}; ``per_item(ph)`` what each photon reports when it ends.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax.numpy as jnp

from lumen.transport import state
from lumen.transport.step import geometry

HEALTH = ("lost", "abandoned", "escaped", "late")  # weights; plus the counts "stuck" (pushed, abandoned)


class HistogramShape(NamedTuple):
    """The histograms' static shape: the sensors and the wavelength band edges (nm, ascending)."""

    n_sensors: int
    lam_edges: tuple


class PhotonLoop(NamedTuple):
    """One photon's loop (see the module docstring)."""

    init: Any
    alive: Any
    emits: Any
    outputs: Any
    per_item: Any


def start_photon(model, ops, row):
    """Return the Photon of an emission row, located (its volume and class from the navigator)."""
    nav = geometry(model, ops).nav
    vol = nav.locate(jnp.asarray(row.x, jnp.float32))
    st = model.step
    return state.start(row, vol, nav.dev()["cls"][vol], stacked=model.stacked, moving=st.moving_source,
                       edge=bool(st.edge_rows))  # fmt: skip


def photons(tally: HistogramShape) -> PhotonLoop:
    """Return the default PhotonLoop: per sensor and band the detected weight "Q" and its time-weighted sum "Tw"."""
    n_lam = len(tally.lam_edges) - 1
    cells = tally.n_sensors * n_lam

    def emits(dep, replica, grid):
        e = jnp.asarray(tally.lam_edges, jnp.float32) if grid is None else grid["lam_edges"]
        b = jnp.searchsorted(e, dep.lam, side="right") - 1
        band = jnp.where((b >= 0) & (b < n_lam), b, -1)
        idx = jnp.where((dep.sensor >= 0) & (band >= 0), dep.sensor * n_lam + band, -1)
        idx = jnp.where(idx >= 0, idx + replica * cells, -1)
        return [("Q", idx, dep.q), ("Tw", idx, dep.q * dep.time)] + health_adds(dep)

    def outputs(replicas):
        return {"Q": (replicas * cells, True), "Tw": (replicas * cells, True)} | HEALTH_OUTPUTS

    return PhotonLoop(start_photon, lambda ph: ph.alive, emits, outputs, lambda ph: dict(vertices=ph.vertex))


class HistogramLayout(NamedTuple):
    """The histograms of one sensor group (``lumen.readout``): sensors, the static grid, replicas on the charge
    histogram, half-replicas on the time histogram."""  # noqa: D205, D209

    n_sensors: int
    grid: Any
    replicas: int = 16
    half_replicas: int = 2


@functools.lru_cache(maxsize=32)
def readout(r: HistogramLayout) -> PhotonLoop:
    """Return the PhotonLoop filling a group's histograms through ``lumen.readout.charge_cells`` / ``time_cells``.

    Outputs: "charge" (R S L A), "arrival" (R S: every deposit, whatever its wavelength or angle) and "arrival_t" (S:
    q t), and on a timed grid "time" (H S nt), "over_q" / "over_qt" (H S 2: q and q t beyond the last node, side 0,
    and before the first, side 1), with the health; photon i of n fills replica i R // n and half-replica i H // n
    (``replica`` is the charge replica). One PhotonLoop per HistogramLayout (no recompile for the same layout).
    """
    from lumen.readout import charge_cells, time_cells

    g, s = r.grid, r.n_sensors
    cells, nodes = s * g.n_lam * g.n_theta, s * g.nt

    def emits(dep, replica, _grid):
        idx, w = charge_cells(dep, g)
        adds = [("charge", jnp.where(idx[m] >= 0, idx[m] + replica * cells, -1), dep.q * w[m]) for m in range(len(w))]
        adds += [("arrival", jnp.where(dep.sensor >= 0, dep.sensor + replica * s, -1), dep.q),
                 ("arrival_t", dep.sensor, dep.q * dep.time)]  # fmt: skip
        if g.nt:
            h = replica * r.half_replicas // r.replicas
            ti, tw, oi, ow = time_cells(dep, g)
            adds += [("time", jnp.where(ti[m] >= 0, ti[m] + h * nodes, -1), dep.q * tw[m]) for m in range(4)]
            for m in range(2):
                o = jnp.where(oi[m] >= 0, oi[m] + h * 2 * s, -1)
                adds += [("over_q", o, dep.q * ow[m]), ("over_qt", o, dep.q * ow[m] * dep.time)]
        return adds + health_adds(dep)

    def outputs(replicas):
        if replicas != r.replicas:
            raise ValueError(f"readout: the item was made for {r.replicas} replicas, run with {replicas}")
        out = {"charge": (replicas * cells, True), "arrival": (replicas * s, True), "arrival_t": (s, True)}
        out |= HEALTH_OUTPUTS
        if g.nt:
            hs = r.half_replicas
            out |= {"time": (hs * nodes, True), "over_q": (hs * 2 * s, True), "over_qt": (hs * 2 * s, True)}
        return out

    return PhotonLoop(start_photon, lambda ph: ph.alive, emits, outputs, lambda ph: dict(vertices=ph.vertex))


def histograms(r: HistogramLayout, outputs, model, group: str, gate) -> Any:
    """Return the ``lumen.readout.Histograms`` of a run of ``readout(r)`` (its tally outputs)."""
    from lumen.readout import Histograms

    g, s, hs = r.grid, r.n_sensors, r.half_replicas
    charge = outputs["charge"].reshape(r.replicas, s, g.n_lam, g.n_theta)
    if not g.nt:
        return Histograms(charge, grid=g, model=model, group=group, gate=tuple(gate))
    q, qt = (outputs[k].reshape(hs, s, 2) for k in ("over_q", "over_qt"))
    late, early = (jnp.stack([q[..., i], qt[..., i]], -1) for i in (0, 1))
    return Histograms(charge, outputs["time"].reshape(hs, s, g.nt), late, early, None, g, model, group, tuple(gate))


def health_adds(dep) -> list:
    """Return the health adds of a Deposit: the weights of ``HEALTH`` and the "stuck" counts (pushed, abandoned)."""
    adds = [(k, jnp.int32(0), getattr(dep, k)) for k in HEALTH]
    return adds + [("stuck", jnp.where(f, i, -1), 1.0) for i, f in enumerate((dep.pushed, dep.abandon))]


HEALTH_OUTPUTS = {"stuck": (2, False)} | {k: (1, True) for k in HEALTH}


def grid(tally: HistogramShape) -> dict:
    """Return the default item's grid positions as operands (``Operands.grid``): the band edges (nm)."""
    return {"lam_edges": jnp.asarray(tally.lam_edges, jnp.float32)}
