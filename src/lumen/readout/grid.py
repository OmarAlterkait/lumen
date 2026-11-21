"""The histograms the readout bins the deposit records into: their static grid and the cells a deposit goes to.

Two histograms per sensor group (what every readout function reads):

  charge  (S, *axes)     sensors x one cell axis per declared axis: the expected photons the detection rule
                         counts (before the sensor's acceptance). An axis (``Axis``) is a feature of the deposit
                         (``lumen.sensors.features``: a plain function of the deposit record) on knots (hat functions,
                         constant beyond the end knots: sum_k T_k hat_k(f) is Geant4's linear interpolation) or between
                         band edges (none outside the outer edges). The sensor model declares the axes its stages read
                         (its acceptance tables, the axes a response of your own reads); only those are binned.
  time    (S, *delay_axes, nt)  the same deposits on the time nodes t0 + j dt, per cell of the delay's axes (one cell
                         without), split over the four nearest nodes with
                         cubic Lagrange weights (moments to the third kept, continuous in t); ``late`` / ``early``
                         (S, *delay_axes, 2) keep (q, q t) of what falls beyond the last / before the first node, so a
                         sensor's total is exact (no light is dropped). Each deposit
                         is weighted by the detection of its charge cells
                         (``weighted``: the acceptance of the values the run is given, an operand of the readout,
                         read as the charge histogram reads it), so the time shape is that of the detected light at
                         those values and differentiable in them; the readout scales it per sensor to the charge
                         histogram.

``charge_cells(dep, grid)`` and ``time_cells(dep, grid)`` give a deposit's (index, weight) pairs; the readout adds
``dep.q * weight`` at each index (-1: none; ``lumen.readout.fill``). Beside the charge histogram, ``charge2`` (S, *axes)
is its per-photon second moment, sum_i x_i^2 with x_i photon i's share of the cell, and ``pe_error`` (S,) the Monte
Carlo error of each sensor's pe at the values the run was given (``moment_error``). The grid comes from the sensor
model's declared axes and ranges and the readout gate (``lumen.sensors.SensorValuesModel.grid``), never from current
values: calibrating a value never changes it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core.safe import literal
from lumen.sources._base import bucket

CELL_BYTES = 4  # float32 histogram cells
NODES_PER_SIGMA = 4  # time nodes per narrowest time spread: the cubic split's spacing sigma / 4 (``node_spacing``)


def moment_error(q2, s, n: int):
    """Return the Monte Carlo error of a sum ``s`` of ``n`` i.i.d. photons from its per-photon second moment ``q2``.

    q2 = sum_i x_i^2, x_i photon i's share of the sum; the error is sqrt(n / (n - 1) (q2 - s^2 / n)), clipped at 0.

    n / (n - 1) (q2 - s^2 / n) is unbiased for the variance of s when the photons are independent and identically
    distributed (a source drawing its rows at random). Rows that are fixed (``FromArrays``) are independent but not
    identically distributed: the estimate then includes the spread between the rows, the error of a run of rows drawn
    like them, and is larger than the spread over seeds of the same rows. It has no derivative (``stop_gradient``).
    """
    n = float(max(int(n), 2))
    q2, s = jax.lax.stop_gradient(q2), jax.lax.stop_gradient(s)
    return jnp.sqrt(jnp.maximum(n / (n - 1.0) * (q2 - s * s / n), 0.0))


class Axis(NamedTuple):
    """One histogram axis: a feature of the deposit (a function with ``reads``) on ``knots`` or between ``edges``.

    Knots: hat functions, constant beyond the end knots. Edges: one cell between each pair (the outer edges may be
    infinite), none outside them. ``name``: how readout code and value sets refer to it.
    """

    name: str
    feature: Any
    knots: tuple | None = None
    edges: tuple | None = None

    @property
    def size(self) -> int:
        """Return the number of cells."""
        return len(self.knots) if self.knots is not None else len(self.edges) - 1

    def check(self) -> None:
        """Refuse an axis without exactly one of knots and edges, or with fewer than two ascending values."""
        v = self.knots if self.knots is not None else self.edges
        if (
            (self.knots is None) == (self.edges is None)
            or len(v) < 2
            or any(b <= a for a, b in zip(v, v[1:], strict=False))
        ):
            raise ValueError(f"Axis {self.name!r}: exactly one of knots or edges, at least two ascending values")

    def cells(self, dep):
        """Return [(index, weight)] of a deposit on this axis (index -1: none)."""
        f = self.feature(dep)
        if self.knots is not None:
            return hats(self.knots, f)
        b = jnp.searchsorted(literal(self.edges), f, side="right") - 1
        return [(jnp.where((b >= 0) & (b < self.size), b, -1), jnp.float32(1.0))]


@dataclass(frozen=True)
class Grid:
    """The static grid of one group's histograms (see the module docstring); times in ns."""

    axes: tuple = ()  # (Axis, ...): the charge histogram's cell axes
    delay_axes: tuple = ()  # (Axis, ...): the time histogram's cell axes (a delay that depends on features)
    t0: float = 0.0
    dt: float = 1.0
    nt: int = 0
    weighted: bool = False  # the time histogram weighted by the run's acceptance per charge cell
    normalise: bool = False  # each deposit divided by its surface's A_n (Deposit.inv_an): a QE at normal incidence

    def __post_init__(self):
        """Check the grid."""
        for a in (*self.axes, *self.delay_axes):
            a.check()
        for axes in (self.axes, self.delay_axes):
            if len({a.name for a in axes}) != len(axes):
                raise ValueError(f"Grid: distinct axis names, got {[a.name for a in axes]}")
        if self.nt < 0 or (self.nt and not self.dt > 0):
            raise ValueError("Grid: nt >= 0 and dt > 0")

    @classmethod
    def bands(cls, edges, **kw) -> Grid:
        """Return a grid of wavelength bands between ``edges`` (nm; the outer ones may be infinite)."""
        from lumen.sensors.features import wavelength

        return cls(axes=(Axis("wavelength", wavelength, edges=tuple(float(e) for e in edges)),), **kw)

    @property
    def shape(self) -> tuple:
        """Return the charge histogram's cell shape (one length per axis)."""
        return tuple(a.size for a in self.axes)

    @property
    def n_cells(self) -> int:
        """Return the charge cells per sensor."""
        return math.prod(self.shape)

    @property
    def delay_shape(self) -> tuple:
        """Return the time histogram's cell shape (one length per delay axis)."""
        return tuple(a.size for a in self.delay_axes)

    @property
    def n_delay(self) -> int:
        """Return the time histogram's cells per sensor (1 without delay axes)."""
        return math.prod(self.delay_shape)

    @property
    def reads(self) -> frozenset:
        """Return the deposit fields the grid's features read."""
        from lumen.sensors.features import reads

        out = {f for a in (*self.axes, *self.delay_axes) for f in reads(a.feature)}
        return frozenset(out | ({"inv_an"} if self.normalise else set()))

    def axis(self, name: str) -> Axis | None:
        """Return the charge axis of that name, or None."""
        return next((a for a in self.axes if a.name == name), None)


def hats(knots, x):
    """Return [(index, weight)] of x on hat functions over knots (constant beyond the end knots)."""
    k = literal(knots)  # built from scalars: never a float array closed over by the photon loop
    i = jnp.clip(jnp.searchsorted(k, x, side="right") - 1, 0, len(knots) - 2)
    f = jnp.clip((x - k[i]) / (k[i + 1] - k[i]), 0.0, 1.0)
    return [(i, 1.0 - f), (i + 1, f)]


def cells(axes, dep):
    """Return [(flat index, weight)] of a deposit over axes (row-major; index -1: none); [(0, 1)] without axes."""
    out = None
    for a in axes:
        cs = a.cells(dep)
        if out is None:
            out = cs
            continue
        out = [(jnp.where((i >= 0) & (j >= 0), i * a.size + j, -1), w * v) for i, w in out for j, v in cs]
    return out if out is not None else [(jnp.int32(0), jnp.float32(1.0))]


def charge_cells(dep, grid: Grid):
    """Return (idx (M,), weight (M,)) of a deposit in the charge histogram (S, *axes), M <= 2^axes; idx -1 for none.

    ``dep`` has ``sensor`` (-1: none) and the fields the grid's features read.
    """
    idx, w = [], []
    for ci, cw in cells(grid.axes, dep):
        ok = (dep.sensor >= 0) & (ci >= 0)
        idx.append(jnp.where(ok, dep.sensor * grid.n_cells + ci, -1))
        w.append(cw)
    return jnp.stack(idx).astype(jnp.int32), jnp.stack([jnp.asarray(x, jnp.float32) for x in w])


def _time_weights(grid, t):
    # clamped before the cast to int32 (a time far past the grid would wrap to an early index): beyond either end
    # every node index is outside, so the clamp moves no weight between the nodes and the overflow
    x = jnp.clip((t - grid.t0) / grid.dt, -4.0, grid.nt + 4.0)
    j = jnp.floor(jax.lax.stop_gradient(x))
    g = x - j  # in [0, 1): differentiable in t
    j = j.astype(jnp.int32)
    return [(j - 1, -g * (g - 1.0) * (g - 2.0) / 6.0), (j, (g + 1.0) * (g - 1.0) * (g - 2.0) / 2.0),
            (j + 1, -(g + 1.0) * g * (g - 2.0) / 2.0), (j + 2, (g + 1.0) * g * (g - 1.0) / 6.0)]  # fmt: skip


def time_cells(dep, grid: Grid):
    """Return (idx (4 B,), weight (4 B,), oidx (2 B,), oweight (2 B,)) of a deposit in the time histogram, B the
    deposit's delay cells (1 without delay axes, up to 2^axes).

    idx indexes (S, *delay_axes, nt); oidx the overflow cells (S, *delay_axes, 2) as ``cell * 2 + side`` (0 late,
    1 early). A cubic split's outer weights are signed: a deposit within two nodes of the grid's ends puts a signed
    share in an overflow cell (the total stays exact). Tracking adds ``q w`` to the overflow's charge and ``q w t`` to
    its time weight.
    """  # noqa: D205
    tc = _time_weights(grid, dep.time)
    late = sum(jnp.where(n >= grid.nt, f, 0.0) for n, f in tc)
    early = sum(jnp.where(n < 0, f, 0.0) for n, f in tc)
    idx, w, oidx, ow = [], [], [], []
    for bi, bw in cells(grid.delay_axes, dep):
        ok = (dep.sensor >= 0) & (bi >= 0)
        row = dep.sensor * grid.n_delay + bi
        idx += [jnp.where(ok & (n >= 0) & (n < grid.nt), row * grid.nt + n, -1) for n, _ in tc]
        w += [bw * f for _, f in tc]
        oidx += [jnp.where(ok, row * 2, -1), jnp.where(ok, row * 2 + 1, -1)]
        ow += [bw * late, bw * early]
    return (jnp.stack(idx).astype(jnp.int32), jnp.stack(w).astype(jnp.float32),
            jnp.stack(oidx).astype(jnp.int32), jnp.stack(ow).astype(jnp.float32))  # fmt: skip


TIME_ULPS = 8  # float32 time steps per node spacing at least: times are float32 and absolute (``check_times``)


def check_times(grids, times) -> None:
    """Refuse times (ns: the readout gates, the source's emission times) whose float32 resolution is coarser than a
    timed grid's node spacing over ``TIME_ULPS``: a photon's time is a float32 counted from the source's own zero, so
    at 1e9 ns it moves in steps of 64 ns and the time spread's structure is gone. Times are given relative to the
    event (its start subtracted from the source's times and the gate): a float32 input has lost the resolution before
    lumen sees it, so lumen cannot restore it."""  # noqa: D205, D209
    dts = [g.dt for g in grids if g.nt]
    if not dts:
        return
    top = float(np.max(np.abs(np.asarray(times, np.float64)))) if np.size(times) else 0.0
    step = float(np.spacing(np.float32(top)))
    if step > min(dts) / TIME_ULPS:
        raise GateError(f"times up to {top:.6g} ns are float32 steps of {step:.3g} ns, coarser than the time nodes' "
                        f"{min(dts):.3g} ns / {TIME_ULPS}: give the source's times and the gate relative to the "
                        "event (subtract its start)")  # fmt: skip


def node_spacing(sigma_min: float) -> float:
    """Return the largest node spacing (ns) whose discretisation is invisible for a time spread of sigma_min (ns).

    sigma / ``NODES_PER_SIGMA`` (4) with the cubic split. The cubic split keeps a deposit's moments to the third, so
    its error is the fourth: a deposit at phase g between nodes gets a fourth moment off by
    -(g + 1) g (g - 1) (g - 2) dt^4 (at most 9/16 dt^4), and a Gaussian spread's density is off by that / 24 times
    its fourth derivative. Per photoelectron that is a chi^2, at any time binning, of at most
    (9/16)^2 (dt / sigma)^8 / 24: 2.0e-7 at sigma / 4 (2.0e-6 at sigma / 3, 5e-5 at sigma / 2), so below 0.01 per
    event up to 5e4 photoelectrons of light at single instants. On SK, WCTE, SBND and Hyper-K light it measured
    2e-9 to 1.1e-7 at sigma / 4, and the hit likelihood of events of 600 to 3,400 photoelectrons moved by at most
    0.006 (up to 0.17 at sigma / 2).
    """
    if not sigma_min > 0:
        raise ValueError("node_spacing: the narrowest time spread must be > 0 ns; declare its range in the value set")
    return float(sigma_min) / NODES_PER_SIGMA


class GateError(ValueError):
    """A readout gate a timed grid cannot read light in (reversed, zero-width, or not finite)."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True)
class Histograms:
    """One sensor group's histograms (the deposit interface, see the module docstring) with their static context.

    ``charge2``: the charge histogram's per-photon second moment (``moment_error`` of it and ``charge``, over the run's
    photons, is a cell's Monte Carlo error); ``pe_error``: each sensor's pe error at the values the histograms were
    built with (``lumen.readout.pe_error``). ``gate`` (ns) is the readout gate; ``group`` the sensor group's name;
    ``sensors`` the detector's sensor id of each of the group's sensors (None: every sensor of the detector, in order).
    """

    charge: Any
    charge2: Any = None
    time: Any = None
    late: Any = None
    early: Any = None
    pe_error: Any = None
    grid: Grid = Grid()
    model: Any = None
    group: str
    gate: tuple = (0.0, 0.0)
    sensors: tuple | None = None

    _ARRAYS = ("charge", "charge2", "time", "late", "early", "pe_error")
    _STATIC = ("grid", "model", "group", "gate", "sensors")

    def tree_flatten(self):
        """Return the arrays and the static context."""
        return tuple(getattr(self, k) for k in self._ARRAYS), tuple(getattr(self, k) for k in self._STATIC)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from the arrays and the static context."""
        return cls(**dict(zip(cls._ARRAYS, children, strict=True)), **dict(zip(cls._STATIC, aux, strict=True)))


def from_photons(model, gate, n_sensors: int, photons: dict, *, group: str) -> Histograms:
    """Return the Histograms of a list of photons reaching the sensors (an external photon list, a reference's hits).

    ``photons``: ``sensor`` (ids), ``lam`` (nm), ``time`` (ns), the deposit fields the grid's features read
    (``pos``, ``dir``, ``normal``: (n, 3) in the sensor's frame; ``gen``), ``inv_an`` (1 / A_n, default 1) and
    ``weight`` (default 1); ``group``: the sensor group whose values (``sensors/<group>/...``) the readout reads. The
    photons are deposit records (row weight 1, each its own photon) built into the histograms by the same readout as a
    run's (``lumen.readout.fill``, ``lumen.run.rebuild``); the time histogram and the pe moment weighted by the value
    set's own acceptance (``model.acceptance`` of ``model.values``), and ``pe_error`` at those values over the n
    photons.
    """
    from lumen import run as runner
    from lumen.readout import detection, fill

    grid = model.grid(gate)
    n = len(photons["sensor"])
    check_sensors(photons["sensor"], n_sensors, "from_photons")
    missing = sorted(k for k in grid.reads if k not in photons and k not in runner.DEFAULTS and k != "weight")
    if missing:  # a field the grid's features read, never filled with a stand-in (pos 0 would read the axis's 90 deg)
        raise ValueError(f"from_photons: the photons lack {missing}, which the readout's features read")
    check_times([grid], np.r_[np.asarray(photons["time"], np.float64).ravel(), list(gate)])
    bad = [k for k in ("lam", "time") if not np.all(np.isfinite(np.asarray(photons[k], np.float64)))]
    if bad:
        raise ValueError(f"from_photons: the photons' {bad} must be finite")
    m = bucket(n)  # padded to a power of two (no sensor): few shapes compile
    fields = runner.record_fields(tuple(sorted(grid.reads)), False)
    own = dict(sensor=photons["sensor"], q=photons.get("weight", 1.0), w0=1.0, photon=np.arange(n))

    def value(k):  # a photon list's column, padded with rows of no sensor
        x = np.asarray(own[k] if k in own else photons.get(k, runner.DEFAULTS.get(k, 0.0)), np.float64)
        x = np.broadcast_to(x, (n, *x.shape[1:]) if x.ndim and len(x) == n else (n, *x.shape))
        pad = -1.0 if k == "sensor" else runner.DEFAULTS.get(k, 0.0)
        return np.concatenate([x, np.full((m - n, *x.shape[1:]), pad)])

    rec = runner.encode(value, fields)
    lay = fill.HistogramLayout(n_sensors, grid)
    cells, nan = runner.rebuild(fill.readout((lay,)), rec, n, (model.acceptance(model.values),))
    if int(nan):
        raise ValueError(f"from_photons: {int(nan)} histogram cells are not finite (a non-finite weight or field)")
    h = fill.histograms(lay, cells, model, group, tuple(float(x) for x in gate), ":0")
    v = detection.values(h, {f"sensors/{group}/{k}": x for k, x in model.values.items()})
    return dataclasses.replace(h, pe_error=detection.pe_moment_error(h, v, cells["pe2"], n))


def check_sensors(sensor, n_sensors: int, what: str) -> None:
    """Refuse sensor ids outside [0, n_sensors): an id past the end would be binned as another sensor's."""
    s = np.asarray(sensor, np.float64)
    if s.size and (not np.all(s == np.floor(s)) or s.min() < 0 or s.max() >= n_sensors):
        raise ValueError(f"{what}: sensor ids are integers in [0, {n_sensors}), got values in [{s.min()}, "
                         f"{s.max()}]")  # fmt: skip


def photon_chunks(samples: int, chunk: int) -> list[tuple[int, int]]:
    """Return the (first photon id, count) of each chunk (each photon the same whatever the chunk size)."""
    if samples < 0 or chunk <= 0:
        raise ValueError(f"photon_chunks: samples >= 0 and chunk > 0, got {samples}, {chunk}")
    return [(i, min(chunk, samples - i)) for i in range(0, samples, chunk)]


__all__ = [
    "Axis", "GateError", "Grid", "Histograms", "cells", "charge_cells", "check_sensors", "from_photons", "hats",
    "check_times", "moment_error", "node_spacing", "photon_chunks", "time_cells",
]  # fmt: skip
