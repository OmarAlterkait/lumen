"""The histograms tracking fills for the readout: their static grid, the cells a deposit goes to, and their size.

Two histograms per sensor group (the deposit interface between tracking and the readout):

  charge  (R, S, L, A)  replicas x sensors x wavelength cells x hit-angle cells: the expected photons the detection
                        rule counts (before QE and CE). Wavelength cells are hat functions on the QE knots (so
                        sum_k QE_k hat_k(lambda) is Geant4's linear interpolation, constant beyond the end knots) or
                        bands; hit-angle cells are hat functions on the CE knots (theta in rad, A = 1 without CE).
  time    (H, S, nt)    the same deposits on the time nodes t0 + j dt, one wavelength band, H half-replicas (>= 1),
                        split over the four nearest nodes with cubic Lagrange weights (moments to the third kept,
                        continuous in t); ``late`` / ``early`` (H, S, 2) keep (q, q t) of what falls beyond the last
                        / before the first node, so a sensor's total is exact. Any per-photon weight constant in time
                        may multiply the time histogram: the readout scales it per sensor to the charge histogram.

``charge_cells(dep, grid)`` and ``time_cells(dep, grid)`` give a deposit's (index, weight) pairs; tracking adds
``dep.q * weight`` at each index (-1: none). The grid comes from the sensor model's declared ranges and the readout
gate (``make_grid``), never from current values: calibrating a value never changes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

CELL_BYTES = 4  # float32 histogram cells
TIME_SPLITS = ("cubic", "linear")


@dataclass(frozen=True)
class Grid:
    """The static grid of one group's histograms (see the module docstring); times in ns, wavelengths in nm."""

    lam_knots: tuple | None = None
    lam_edges: tuple | None = None
    theta_knots: tuple | None = None
    t0: float = 0.0
    dt: float = 1.0
    nt: int = 0
    split: str = "cubic"

    def __post_init__(self):
        """Check the grid."""
        if self.lam_edges is not None and self.lam_knots is not None:
            raise ValueError("Grid: give lam_edges or lam_knots, not both")
        for name in ("lam_edges", "lam_knots", "theta_knots"):
            v = getattr(self, name)
            if v is not None and (len(v) < 2 or any(b <= a for a, b in zip(v, v[1:], strict=False))):
                raise ValueError(f"Grid.{name}: at least two ascending values")
        if self.nt < 0 or (self.nt and not self.dt > 0):
            raise ValueError("Grid: nt >= 0 and dt > 0")
        if self.split not in TIME_SPLITS:
            raise ValueError(f"Grid.split: one of {TIME_SPLITS}, not {self.split!r}")

    @property
    def n_lam(self) -> int:
        """Return the number of wavelength cells."""
        if self.lam_edges is not None:
            return len(self.lam_edges) - 1
        return len(self.lam_knots) if self.lam_knots is not None else 1

    @property
    def n_theta(self) -> int:
        """Return the number of hit-angle cells."""
        return len(self.theta_knots) if self.theta_knots is not None else 1


def _hats(knots, x):
    """Return [(index, weight)] of x on hat functions over knots (constant beyond the end knots)."""
    k = jnp.asarray(knots, jnp.float32)
    i = jnp.clip(jnp.searchsorted(k, x, side="right") - 1, 0, len(knots) - 2)
    f = jnp.clip((x - k[i]) / (k[i + 1] - k[i]), 0.0, 1.0)
    return [(i, 1.0 - f), (i + 1, f)]


def _lam_cells(grid, lam):
    if grid.lam_knots is not None:
        return _hats(grid.lam_knots, lam)
    if grid.lam_edges is not None:
        b = jnp.searchsorted(jnp.asarray(grid.lam_edges, jnp.float32), lam, side="right") - 1
        return [(jnp.where((b >= 0) & (b < grid.n_lam), b, -1), jnp.float32(1.0))]
    return [(jnp.int32(0), jnp.float32(1.0))]


def charge_cells(dep, grid: Grid):
    """Return (idx (M,), weight (M,)) of a deposit in the charge histogram (S, L, A), M <= 4; idx -1 for none.

    ``dep`` has ``sensor`` (-1: none), ``lam`` (nm) and, when the grid has hit-angle knots, ``hit`` (rad).
    """
    lc = _lam_cells(grid, dep.lam)
    ac = _hats(grid.theta_knots, dep.hit) if grid.theta_knots is not None else [(jnp.int32(0), jnp.float32(1.0))]
    idx, w = [], []
    for li, lw in lc:
        for ai, aw in ac:
            ok = (dep.sensor >= 0) & (li >= 0)
            idx.append(jnp.where(ok, (dep.sensor * grid.n_lam + li) * grid.n_theta + ai, -1))
            w.append(lw * aw)
    return jnp.stack(idx).astype(jnp.int32), jnp.stack([jnp.asarray(x, jnp.float32) for x in w])


def _time_weights(grid, t):
    x = (t - grid.t0) / grid.dt
    j = jnp.floor(jax.lax.stop_gradient(x))
    g = x - j  # in [0, 1): differentiable in t
    j = j.astype(jnp.int32)
    if grid.split == "linear":
        return [(j, 1.0 - g), (j + 1, g)]
    return [(j - 1, -g * (g - 1.0) * (g - 2.0) / 6.0), (j, (g + 1.0) * (g - 1.0) * (g - 2.0) / 2.0),
            (j + 1, -(g + 1.0) * g * (g - 2.0) / 2.0), (j + 2, (g + 1.0) * g * (g - 1.0) / 6.0)]  # fmt: skip


def time_cells(dep, grid: Grid):
    """Return (idx (4,), weight (4,), oidx (2,), oweight (2,)) of a deposit in the time histogram.

    idx indexes (S, nt); oidx the overflow cells (S, 2) as ``sensor * 2 + side`` (0 late, 1 early). A cubic split's
    outer weights are signed: a deposit within two nodes of the grid's ends puts a signed share in an overflow cell
    (the total stays exact). Tracking adds ``q w`` to the overflow's charge and ``q w t`` to its time weight.
    """
    ok = dep.sensor >= 0
    tc = _time_weights(grid, dep.time)
    idx = [jnp.where(ok & (n >= 0) & (n < grid.nt), dep.sensor * grid.nt + n, -1) for n, _ in tc]
    late = sum(jnp.where(n >= grid.nt, f, 0.0) for n, f in tc)
    early = sum(jnp.where(n < 0, f, 0.0) for n, f in tc)
    oidx = [jnp.where(ok, dep.sensor * 2, -1), jnp.where(ok, dep.sensor * 2 + 1, -1)]
    return (jnp.stack(idx).astype(jnp.int32), jnp.stack([f for _, f in tc]).astype(jnp.float32),
            jnp.stack(oidx).astype(jnp.int32), jnp.stack([late, early]).astype(jnp.float32))  # fmt: skip


def node_spacing(sigma_min: float, split: str = "cubic") -> float:
    """Return the largest node spacing (ns) whose discretisation is invisible for a time spread of sigma_min (ns).

    sigma / 4 for the cubic split, sigma / 10 for the linear one (below 0.01 chi^2 per event against Geant4's hit
    times binned, at any readout bin down to 0.1 ns).
    """
    if not sigma_min > 0:
        raise ValueError("node_spacing: the narrowest time spread must be > 0 ns; declare its range in the value set")
    return float(sigma_min) / (4.0 if split == "cubic" else 10.0)


def window_end(t_hi: float, t_offset_min: float = 0.0, reach: float = 0.0) -> float:
    """Return the arrival time (ns) after which a photon never counts in a gate ending at t_hi.

    ``reach`` is how far the time kernels spread a photoelectron earlier (ns, >= 0).
    """
    return float(t_hi) - min(float(t_offset_min), 0.0) + float(reach)


def make_grid(model, gate: tuple[float, float], split: str = "cubic") -> Grid:
    """Return a group's static grid from its model's structure and declared ranges and the readout gate (ns)."""
    lam = dict(lam_knots=model.qe_knots, lam_edges=model.qe_bands, theta_knots=model.ce_knots)
    if model.readout == "pe":
        return Grid(**lam)
    r = {f: model.ranges[f"tts/{f}"] for f in model.tts.fields() if f"tts/{f}" in model.ranges}
    lo, hi = model.tts.support(r)
    dmax = model.delay.support(None)[1] if model.delay is not None else 0.0
    off = model.ranges["t_offset"]
    dt = node_spacing(model.tts.min_width(r), split)
    if model.readout == "waveform":  # a whole number of nodes per front-end sample
        dt = model.sampling / math.ceil(model.sampling / dt - 1e-9)
    t_first = gate[0] - off[1] - hi - dmax
    t_last = gate[1] - off[0] - lo  # a photoelectron's time is arrival + t_offset + delay + spread
    nt = int(math.ceil((t_last - t_first) / dt)) + 1
    return Grid(**lam, t0=float(t_first), dt=dt, nt=nt, split=split)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Histograms:
    """One sensor group's histograms (the deposit interface, see the module docstring) with their static context.

    ``profile``: an optional scintillation emission profile (fractions, decay, rise times in ns), applied after
    tracking; ``None`` for prompt light. ``gate`` (ns) is the readout gate; ``group`` the sensor group's name.
    """

    charge: Any
    time: Any = None
    late: Any = None
    early: Any = None
    profile: Any = None
    grid: Grid = Grid()
    model: Any = None
    group: str = "pmt"
    gate: tuple = (0.0, 0.0)

    def tree_flatten(self):
        """Return the arrays and the static context."""
        return (self.charge, self.time, self.late, self.early, self.profile), (self.grid, self.model, self.group,
                                                                               self.gate)  # fmt: skip

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from the arrays and the static context."""
        return cls(*children, *aux)


class _Photons(NamedTuple):
    sensor: Any
    lam: Any
    hit: Any
    time: Any


def from_photons(model, gate, n_sensors: int, photons: dict, replicas: int = 1, profile=None) -> Histograms:
    """Return the Histograms of a list of photons reaching the sensors (an external photon list, a reference's hits).

    ``photons``: ``sensor`` (ids), ``lam`` (nm), ``time`` (ns), optional ``hit`` (rad) and ``weight`` (default 1);
    photon i goes to replica i mod ``replicas``. Filled with ``charge_cells`` and ``time_cells``, as tracking does.
    """
    grid = make_grid(model, gate)
    n = len(photons["sensor"])
    m = 1 << max(int(n - 1).bit_length(), 0)  # padded to a power of two (no sensor): few shapes compile

    def col(k, d, fill=0.0):
        x = np.asarray(photons.get(k, np.zeros(n)))
        return jnp.asarray(np.concatenate([x, np.full(m - n, fill)]), d)

    dep = _Photons(col("sensor", jnp.int32, -1), col("lam", jnp.float32), col("hit", jnp.float32),
                   col("time", jnp.float32))  # fmt: skip
    w = np.r_[np.asarray(photons.get("weight", np.ones(n)), np.float64), np.zeros(m - n)]
    r = np.arange(m) % replicas
    idx, cw = (np.asarray(a) for a in _CELLS(dep, grid))
    charge = np.zeros((replicas, n_sensors * grid.n_lam * grid.n_theta))
    ok = idx >= 0
    np.add.at(charge, (np.broadcast_to(r[:, None], idx.shape)[ok], idx[ok]), (w[:, None] * cw)[ok])
    out = dict(charge=charge.reshape(replicas, n_sensors, grid.n_lam, grid.n_theta))
    if grid.nt:
        ti, tw, oi, ow = (np.asarray(a) for a in _TIME_CELLS(dep, grid))
        time, over = np.zeros(n_sensors * grid.nt), np.zeros((n_sensors * 2, 2))
        ok = ti >= 0
        np.add.at(time, ti[ok], (w[:, None] * tw)[ok])
        t = np.r_[np.asarray(photons["time"], np.float64), np.zeros(m - n)]
        for c, val in ((0, w[:, None] * ow), (1, (w * t)[:, None] * ow)):
            np.add.at(over[:, c], oi[oi >= 0], val[oi >= 0])
        over = over.reshape(n_sensors, 2, 2)
        out.update(time=time.reshape(1, n_sensors, grid.nt), late=over[None, :, 0], early=over[None, :, 1])
    out = {k: jnp.asarray(v, jnp.float32) for k, v in out.items()}
    return Histograms(**out, profile=profile, grid=grid, model=model, gate=tuple(float(x) for x in gate))


_CELLS = jax.jit(jax.vmap(charge_cells, in_axes=(0, None)), static_argnums=1)
_TIME_CELLS = jax.jit(jax.vmap(time_cells, in_axes=(0, None)), static_argnums=1)


def histogram_bytes(grid: Grid, n_sensors: int, replicas: int, tangents: int = 0, half_replicas: int = 2) -> int:
    """Return the bytes of a group's histograms, each forward tangent adding them again."""
    charge = replicas * n_sensors * grid.n_lam * grid.n_theta
    time = half_replicas * n_sensors * (grid.nt + 4) if grid.nt else 0
    return (charge + time) * CELL_BYTES * (1 + tangents)


def check_memory(grid: Grid, n_sensors: int, replicas: int, tangents: int, budget: float) -> int:
    """Return the histogram bytes, or refuse (MemoryError) before launch when they exceed ``budget`` bytes."""
    need = histogram_bytes(grid, n_sensors, replicas, tangents)
    if need > budget:
        span = grid.nt * grid.dt
        raise MemoryError(f"the histograms need {need / 1e9:.2f} GB, over the device budget of {budget / 1e9:.2f} "
                          f"GB: {n_sensors} sensors x {grid.nt} time nodes ({span:.0f} ns at {grid.dt:.3g} ns), "
                          f"{replicas} replicas, {tangents} derivative directions. Shorten the readout gate, widen "
                          "the time spread's declared lower range (coarser nodes), use fewer replicas or fewer "
                          "derivative directions (wrt).")  # fmt: skip
    return need


def photon_chunks(samples: int, chunk: int) -> list[tuple[int, int]]:
    """Return the (first photon id, count) of each chunk; exact sums make a run independent of the chunk size."""
    if samples < 0 or chunk <= 0:
        raise ValueError(f"photon_chunks: samples >= 0 and chunk > 0, got {samples}, {chunk}")
    return [(i, min(chunk, samples - i)) for i in range(0, samples, chunk)]


__all__ = [
    "Grid", "Histograms", "charge_cells", "check_memory", "from_photons", "histogram_bytes", "make_grid",
    "node_spacing", "photon_chunks", "time_cells", "window_end",
]  # fmt: skip
