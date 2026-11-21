"""The readout's cells built from the deposit records: the histograms and the arrivals, per block of sensors.

The photon loop only appends deposit records (``lumen.run.trace``); every binned output is built here from them, one
tallymap launch per block of sensors (``lumen.run.rebuild``: each record through ``emits``, into exact sums).

    lay = HistogramLayout(n_sensors, grid)                           # one sensor group's histograms
    blocks = plan((lay,), budget, derivative, backend)               # the launches whose exact cells fit the budget
    cells = None
    for b in blocks:                                                 # every group's cells, the sensors in order
        cells = place((lay,), b, lumen.run.rebuild(readout((lay,), b), t.records, t.count, weights)[0], cells)
    h = histograms(lay, cells, model, group, gate, ":0")             # lumen.readout.Histograms

A record's features (the readout axes: ``lumen.readout.grid``) and the readout stages that act per deposit (the
normalisation over A_n, the time histogram's detection weight) are evaluated here, per record; transport never sees
an axis. A sensor's cells are the same whatever the block: the same adds into the same exact sums.

Beside each charge cell and each sensor's arrival the readout builds a per-photon second moment (``Readout.moments``):
sum_i x_i^2, x_i photon i's adds to the cell summed over its records before the square (``lumen.run.rebuild``). For N
i.i.d. photons N / (N - 1) (sum_i x_i^2 - S^2 / N) is an unbiased estimate of the variance of the cell's sum S
(``lumen.readout.grid.moment_error``); a moment is diagonal (no covariance between cells) and has no derivative.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import tallymap as tm

from lumen.readout.grid import Histograms, charge_cells, time_cells


class HistogramLayout(NamedTuple):
    """The histograms of one sensor group (``lumen.readout``): sensors, the static grid, the event axis."""

    n_sensors: int
    grid: Any
    index: tuple | None = None  # per detector sensor, its position in the group (-1: not in it); None: all, in order
    events: int = 0  # E > 0: every histogram per event, the event a record's label in [0, E)

    # the one statement of the cell layout: each output kind is (lead, sensors, per) flat, lead the event blocks, per
    # the cells of one sensor in one block
    def kinds(self) -> tuple:
        """Return the output kinds of the group: charge, on a timed grid time, over_q and over_qt, and charge2 (the
        charge cells' per-photon second moment)."""  # noqa: D205, D209
        return ("charge", "time", "over_q", "over_qt")[: 4 if self.grid.nt else 1] + ("charge2",)

    def lead(self, kind: str) -> int:
        """Return the leading blocks of an output kind: the events (1 without)."""
        return max(self.events, 1)

    def per(self, kind: str) -> int:
        """Return one sensor's cells of an output kind in one block."""
        g = self.grid
        return g.n_cells if kind in ("charge", "charge2") else g.n_delay * (g.nt if kind == "time" else 2)

    def size(self, kind: str, sensors: int) -> int:
        """Return the cells of an output kind on ``sensors`` of the group's sensors."""
        return self.lead(kind) * sensors * self.per(kind)


class Readout(NamedTuple):
    """What is built from the records: ``emits(dep, weights)`` the (output, cell, value) adds of one record, each
    times its row weight by ``lumen.run.rebuild``; ``outputs()`` {name: (cells, exact)}; ``moments(dep, weights)``
    (None: none) the (output, cell, value) adds whose per-photon sum per cell is squared (``lumen.run.rebuild``)."""  # noqa: D205, D209

    emits: Any
    outputs: Any
    moments: Any = None


class Block(NamedTuple):
    """One launch of the readout: per group its sensors [b0, b1) in the group (None: none of it), and whether it builds
    the detector's arrivals."""  # noqa: D205, D209

    spans: tuple
    arrivals: bool = True


def full(layouts) -> Block:
    """Return the block of every sensor of every group, with the arrivals."""
    return Block(tuple((0, r.n_sensors) for r in layouts), True)


@functools.lru_cache(maxsize=64)
def readout(layouts, block: Block | None = None) -> Readout:  # noqa: C901, PLR0915 -- one closure per output kind
    """Return the Readout of the sensor groups' histograms (``lumen.readout.grid.charge_cells`` / ``time_cells``) on
    the sensors of ``block`` (None: all, ``full``).

    ``layouts``: a tuple of HistogramLayouts, one per modelled group (each with its own grid and index map; a sensor
    is in one group). Outputs of group k on its block of s = b1 - b0 sensors: "charge:<k>" (s *axes), and on a timed
    grid "time:<k>" (s *delay_axes nt), "over_q:<k>" / "over_qt:<k>" (s *delay_axes 2: q and q t beyond the last
    node, side 0, and before the first, side 1). With the block's arrivals, once for the detector's sensors:
    "arrival" (S: every deposit, whatever its wavelength or angle) and "arrival_t" (S: q t). The moments (``moments``:
    each photon's adds to a cell summed, then squared): "charge2:<k>" (s *axes) of the charge cells, and with the
    arrivals "arrival2" (S) and "pe2" (S: the sensor's pe at the acceptance ``weights`` of its group, before the
    efficiency; 0 for a group whose weights are not given). One Readout per (layouts, block) (no recompile for the
    same). With ``events`` E > 0 every output is per event, event-major (charge (E s *axes), time (E s *delay_axes nt),
    over (E s *delay_axes 2), arrival (E S), arrival_t (E S)), the event the record's label; a label outside [0, E)
    fills none.
    """  # noqa: D205
    rs = tuple(layouts)
    if len({(r.n_sensors if r.index is None else len(r.index), r.events) for r in rs}) != 1:
        raise ValueError("readout: the groups' layouts differ in events or in the detector's sensors")
    block = full(rs) if block is None else block
    n_all = rs[0].n_sensors if rs[0].index is None else len(rs[0].index)
    sfx = [f":{k}" for k in range(len(rs))]
    n_ev = rs[0].events

    def ev_block(dep):  # the cell block of the record's event; -1: no event
        if not n_ev:
            return 0
        e = dep.label
        return jnp.where((e >= 0) & (e < n_ev), e, -1)

    def ok(i, b):  # a cell index i of block b is filled (without events b is never -1)
        return (i >= 0) & (b >= 0) if n_ev else i >= 0

    def local(r, span, dep):  # the record's sensor in the group's block (-1: not in it)
        pos = dep.sensor
        if r.index is not None:
            pos = jnp.asarray(r.index, jnp.int32)[jnp.clip(dep.sensor, 0, n_all - 1)]
            pos = jnp.where(dep.sensor >= 0, pos, -1)
        if span != (0, r.n_sensors):
            pos = jnp.where((pos >= span[0]) & (pos < span[1]), pos - span[0], -1)
        return pos

    def charge_adds(r, x, span, dep, kind="charge"):  # the charge cells' adds (q d per cell weight)
        g, cells = r.grid, (span[1] - span[0]) * r.per("charge")
        idx, w = charge_cells(dep._replace(sensor=local(r, span, dep)), g)
        eb = ev_block(dep)
        qd = dep.q * dep.inv_an if g.normalise else dep.q  # the normalisation stage: over A_n (Grid.normalise)
        return [(f"{kind}{x}", jnp.where(ok(idx[m], eb), idx[m] + eb * cells, -1), qd * w[m]) for m in range(len(w))]

    def group_adds(r, x, span, dep, weight):
        g, s = r.grid, span[1] - span[0]
        adds = charge_adds(r, x, span, dep)
        if g.nt:
            nodes, over = (s * r.per(k) for k in ("time", "over_q"))  # one block of each kind
            hdep = dep._replace(sensor=local(r, span, dep))  # the group's sensors, at their position in the block
            idx, w = charge_cells(hdep, g)
            qd = dep.q * dep.inv_an if g.normalise else dep.q
            eb = ev_block(dep)
            ti, tw, oi, ow = time_cells(hdep, g)
            q = qd * detection_weight(g, weight, idx, w)
            adds += [(f"time{x}", jnp.where(ok(ti[m], eb), ti[m] + eb * nodes, -1), q * tw[m])
                     for m in range(len(tw))]  # fmt: skip
            for m in range(len(ow)):
                o = jnp.where(ok(oi[m], eb), oi[m] + eb * over, -1)
                adds += [(f"over_q{x}", o, q * ow[m]), (f"over_qt{x}", o, q * ow[m] * dep.time)]
        return adds

    on = [(r, x, sp) for r, x, sp in zip(rs, sfx, block.spans, strict=True) if sp is not None]

    def arrival_cell(dep):  # the record's sensor in the event's block of the detector's sensors
        eb = ev_block(dep)
        return jnp.where(ok(dep.sensor, eb), dep.sensor + eb * n_all, -1) if n_ev else dep.sensor

    def adds_of(dep, weights=()):
        tws = dict(zip(sfx, _group_weights(rs, weights), strict=True))
        out = [a for r, x, sp in on for a in group_adds(r, x, sp, dep, tws[x])]
        if block.arrivals:
            at = arrival_cell(dep)
            out += [("arrival", at, dep.q), ("arrival_t", at, dep.q * dep.time)]
        return out

    def emits(dep, weights=()):
        names = [a[0] for a in adds_of(dep, weights)]
        # a record of a sensor outside the block adds nothing: one lax.cond around its cells and weights (bit for bit
        # the ungated adds)
        mine = jnp.bool_(block.arrivals)
        for r, _, sp in on:
            mine = mine | (local(r, sp, dep) >= 0)

        def some(_):
            cv = adds_of(dep, weights)
            return [(jnp.asarray(c, jnp.int32), jnp.asarray(v, jnp.float32)) for _, c, v in cv]

        def none(_):
            return [(jnp.int32(-1), jnp.float32(0.0)) for _ in names]

        cv = jax.lax.cond(mine & (dep.sensor >= 0), some, none, None)
        return [(n, c, v) for n, (c, v) in zip(names, cv, strict=True)]

    def moments(dep, weights=()):
        """Return the moment adds of one record: its charge cells' ("charge2:<k>", as "charge:<k>"), and with the
        arrivals its arrival ("arrival2") and its pe at the acceptance ``weights`` ("pe2": q times the acceptance read
        at its charge cells, as ``detection_weight``; the efficiency is the readout's, after)."""  # noqa: D205, D209
        tws = dict(zip(sfx, _group_weights(rs, weights, need=False), strict=True))
        out = [a for r, x, sp in on for a in charge_adds(r, x, sp, dep, "charge2")]
        if block.arrivals:
            pe_v = jnp.float32(0.0)
            for r, x in zip(rs, sfx, strict=True):  # the record's sensor is in at most one group
                if tws[x] is None:
                    continue
                pos = local(r, (0, r.n_sensors), dep)
                idx, w = charge_cells(dep._replace(sensor=pos), r.grid)
                la = jnp.maximum(idx, 0) % r.grid.n_cells
                qd = dep.q * dep.inv_an if r.grid.normalise else dep.q
                pe_v = pe_v + jnp.where(pos >= 0, qd * jnp.sum(jnp.where(idx >= 0, w * jnp.asarray(tws[x])[la], 0.0)),
                                        0.0)  # fmt: skip
            at = arrival_cell(dep)
            out += [("arrival2", at, dep.q), ("pe2", at, pe_v)]
        return out

    def outputs():
        e = max(n_ev, 1)
        once = {k: (e * n_all, True) for k in ("arrival", "arrival_t", "arrival2", "pe2")} if block.arrivals else {}
        return {f"{k}{x}": (r.size(k, sp[1] - sp[0]), True) for r, x, sp in on for k in r.kinds()} | once

    return Readout(emits, outputs, moments)


ARRIVALS = ("arrival", "arrival_t", "arrival2", "pe2")  # the readout's outputs per detector sensor (the first block's)


def sensor_cells(r: HistogramLayout) -> int:
    """Return one sensor's exact cells in a group's readout: the charge cells and their moment, the time nodes and their
    overflow."""  # noqa: D205, D209
    return sum(r.size(k, 1) for k in r.kinds())


def cell_count(layouts) -> int:
    """Return the exact cells of the whole readout of ``layouts``: every group's histograms and the arrivals."""
    return sum(c for c, _ in readout(tuple(layouts)).outputs().values())


MAX_CELLS = (1 << 31) - 1  # a launch's cells of one output are int32 indices (``readout``): below 2^31 each


def plan(layouts, budget: float, derivative: bool, backend) -> tuple:
    """Return the readout's launches (Blocks): the groups' sensors in order, as many in each launch as their exact
    cells (``sensor_cells``, and the arrivals in the first) fit in ``budget`` bytes (``tm.max_cells``: with a
    derivative's tangent cells) and every output's cells stay int32 indices (``MAX_CELLS``). One launch when
    everything fits; MemoryError when one sensor does not."""  # noqa: D205, D209
    from lumen.run import resolve_backend

    rs = tuple(layouts)
    d = "jvp" if derivative else None
    cap = tm.max_cells(tm.Sum((1,), fixed_point=True), resolve_backend(backend)[0], d, memory=max(budget, 0.0))
    n_all = rs[0].n_sensors if rs[0].index is None else len(rs[0].index)
    if max(rs[0].events, 1) * n_all > MAX_CELLS:
        raise MemoryError(f"the arrivals of {n_all} sensors over {rs[0].events} events are more cells than int32 "
                          "indices reach: fewer events per run")  # fmt: skip
    used = len(ARRIVALS) * max(rs[0].events, 1) * n_all  # the arrivals' cells, in the first launch
    blocks, cur = [], [None] * len(rs)
    for k, r in enumerate(rs):
        per, b = sensor_cells(r), 0
        index_room = min(MAX_CELLS // (r.lead(kind) * r.per(kind)) for kind in r.kinds())  # sensors per launch
        if index_room < 1:
            raise MemoryError(f"one sensor's readout is more cells than int32 indices reach ({r.kinds()}): shorten "
                              "the readout gates or give fewer events per run")  # fmt: skip
        while b < r.n_sensors:
            # a group's sensors are one span per launch: once it has one, the rest go to the next launch
            room = 0 if cur[k] is not None else min((cap - used) // per, index_room)
            if room < 1:
                if not any(cur) and blocks:
                    raise MemoryError(f"the readout of one sensor is {per} exact cells, over the {cap} the "
                                      f"{budget / 1e9:.2f} GB left for it hold: shorten the readout gates, widen the "
                                      "time spreads' declared lower ranges (coarser nodes) or give a smaller "
                                      "chunk")  # fmt: skip
                blocks.append(Block(tuple(cur), not blocks))
                cur, used = [None] * len(rs), 0
                continue
            e = min(r.n_sensors, b + room)
            cur[k], used, b = (b, e), used + (e - b) * per, e
    blocks.append(Block(tuple(cur), not blocks))
    return tuple(blocks)


def place(layouts, block: Block, out: dict, cells: dict | None) -> dict:
    """Return the readout's cells of every group (``readout``'s names, its sensors in order; None: none yet, zeros)
    with one launch's outputs ``out`` (of ``block``) added at its sensors, in place: the cells of several blocks, or
    of several chunks, need no copy beside them. A block of everything into no cells is its own output."""  # noqa: D205, D209
    rs = tuple(layouts)
    if cells is None and block == full(rs):
        return dict(out)
    if cells is None:
        cells = {k: jnp.zeros(c, jnp.float32) for k, (c, _) in readout(rs).outputs().items()}
    cells = dict(cells)
    for k, r in enumerate(rs):
        if block.spans[k] is None:
            continue
        x, (b0, b1) = f":{k}", block.spans[k]
        for name in r.kinds():
            cells[name + x] = _add_at(cells[name + x], out[name + x], (r.lead(name), r.n_sensors, b0, b1 - b0))
    for name in ARRIVALS if block.arrivals else ():
        cells[name] = _add_at(cells[name], out[name], (1, 1, 0, 1))
    return cells


@functools.partial(jax.jit, static_argnums=2, donate_argnums=0)
def _add_at(cells, part, shape):
    """Return flat ``cells`` (lead, S, rest) with flat ``part`` (lead, s, rest) added at sensor b0, in place."""
    lead, n, b0, s = shape
    c = cells.reshape(lead, n, -1)
    p = part.reshape(lead, s, -1)
    zero = jnp.int32(0)
    cur = jax.lax.dynamic_slice(c, (zero, jnp.int32(b0), zero), p.shape)
    return jax.lax.dynamic_update_slice(c, cur + p, (zero, jnp.int32(b0), zero)).ravel()


def _group_weights(rs, weights, need: bool = True) -> tuple:
    """Return one acceptance weight per group (None where none is given); refuse a weighted grid without them
    (``need``: the time histogram reads them; the pe moment reads them where given)."""  # noqa: D205, D209
    if need and any(r.grid.weighted for r in rs) and len(weights) != len(rs):
        raise ValueError("readout: a weighted time histogram reads its QE x CE from the run's weights (one per "
                         f"group, lumen.run.run): got {len(weights)} for {len(rs)} groups")  # fmt: skip
    return tuple(weights) if weights else (None,) * len(rs)


def detection_weight(g, tw, idx, w):
    """Return a deposit's detection weight on a weighted grid: ``tw`` (the acceptance per charge cell of the
    run's values, ``lumen.run.run``'s ``weights``) read at its charge cells ``idx`` with their weights ``w``
    (``charge_cells``), as the readout reads the charge histogram; 1 on a grid without."""  # noqa: D205, D209
    if not g.weighted:
        return 1.0
    la = jnp.maximum(idx, 0) % g.n_cells  # the cell of each index within its sensor
    return jnp.sum(jnp.where(idx >= 0, w * jnp.asarray(tw)[la], 0.0))


def histograms(r: HistogramLayout, outputs, model, group: str, gate, suffix: str) -> Any:
    """Return the ``lumen.readout.Histograms`` of a group's cells (``place``; ``suffix``: the group's), no pe_error.

    Its ``pe_error`` is the caller's (``lumen.readout.detection.pe_moment_error`` at the run's values).

    With events, every array has a leading event axis: charge and charge2 (E S L A ...), time (E S nt), late / early
    (E S 2).
    """
    names = [f"{k}{suffix}" for k in r.kinds()]
    arrays = _histogram_arrays(r, *(outputs[k] for k in names))  # one dispatch (the arrays are only moved)
    ids = None
    if r.index is not None:  # the detector's sensor id at each of the group's positions
        ids = [0] * r.n_sensors
        for s, pos in enumerate(r.index):
            if pos >= 0:
                ids[pos] = s
        ids = tuple(ids)
    timed = dict(zip(("time", "late", "early"), arrays[2:], strict=True)) if r.grid.nt else {}
    return Histograms(charge=arrays[0], charge2=arrays[1], **timed, grid=r.grid, model=model, group=group,
                      gate=tuple(gate), sensors=ids)  # fmt: skip


@functools.partial(jax.jit, static_argnums=0)
def _histogram_arrays(r: HistogramLayout, charge, *rest) -> tuple:
    """Return a group's histogram arrays from its cells (``kinds``' order): (charge, charge2), or (charge, charge2,
    time, late, early)."""  # noqa: D205, D209
    g, s = r.grid, r.n_sensors
    ev = (r.events,) if r.events else ()
    charge, charge2 = (a.reshape(*ev, s, *g.shape) for a in (charge, rest[-1]))
    if not g.nt:
        return charge, charge2
    time, over_q, over_qt = rest[:3]
    q, qt = (a.reshape(*ev, s, *g.delay_shape, 2) for a in (over_q, over_qt))
    late, early = (jnp.stack([q[..., i], qt[..., i]], -1) for i in (0, 1))
    return charge, charge2, time.reshape(*ev, s, *g.delay_shape, g.nt), late, early
