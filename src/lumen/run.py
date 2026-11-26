"""The photon loop on tallymap: the only module that imports tallymap.

    t = trace(model, ops, rows, key, fields=record_fields(model.local, labelled), capacity=C)  # deposit records
    out, nan_cells = rebuild(readout, t.records, t.count, weights)                            # the readout's cells
    tally = run(readout, model, ops, rows, key)                                                # both, in one call

Each photon is one tallymap item that loops the same step until it ends. The loop's one output of the light is its
deposit records (``tallymap.Records``): every step that reaches a sensor appends one record, the values the readout's
cells are formed from (``record_fields``: sensor, q relative to the emission row's weight, the row weight, time,
wavelength, the photon's id, its label and the Deposit fields the readout's features read). No histogram, grid or axis
lives in the loop; its sums are the health and the guard's ``truncated`` weight (and a diagnostic's own). The records
are in a deterministic (photon, step) order: tallymap counts each photon's records in a first pass and writes them at
their offsets in the second (count-then-fill). The launch's slots are its own count rounded up to a power of two
(``capacity``: one rule on a GPU and on the host), so the records' memory follows the light and only a few shapes of the
fill compile; a launch that appended more than its slots would poison every float output (NaN) and count what it
dropped (``Traced.dropped``), never silently, and cannot happen when the slots come from the count.

The readout's outputs are built from the records by ``rebuild``: one tallymap item per record runs the readout's emits
into exact sums (each cell the correctly rounded float32 of the exact sum of its adds, tangents and reverse cotangents
included, order-independent; a cell is exact or NaN, the NaN ones counted). Each add is the readout's value of the
record times its row weight, formed as the loop formed its emits, so a cell equals, bit for bit, the cell a loop
filling it in place would have. There is no vertex cap in the physics: ``max_vertices`` is a guard against a photon
that cannot end, and a photon it stops is reported (``truncated``: its weight), never dropped silently.
``backend="auto"``: tallymap's kernel backend, compiled for the device JAX places the loop on (CUDA on a GPU,
C++/OpenMP on the CPU; exact sums), and the GPU is expected where the environment says so (``_env.gpu_expected``:
``JAX_PLATFORMS=cuda``, Slurm's GPU variables); "cpu": the same on the CPU (tallymap's CPU
target; on a GPU machine the loop's inputs are put on the CPU and its outputs back, ``on_device``); "xla": tallymap's
plain-JAX backend, the same exact sums.
Photon ``i`` uses the key ``fold_in(key, first_id + i)``.
tallymap's own names stay in this file.

On the kernel backend the item is one top-level loop, so tallymap refills a lane per step (``SCHEDULE``): a photon's
start (``start_photon``, which locates it) is its first iteration's work, inside the kernel (tallymap's kernel history
records the schedule each kernel was built for). The start is a live photon at vertex 0 (``max_vertices`` >= 1).
Diagnostics: ``lumen.diagnostics``.

The photon carries its weight relative to its emission row (it starts at 1), and every weight the loop adds is
multiplied by the row's weight ``row.w``: the roulette then acts relative to the photon's own start weight, for any
source, and the row weight's derivative enters through that one linear factor (exact). A record keeps q relative and
the row weight apart, so the readout forms each add as (value) x w0, as the loop forms its own. A diagnostic is handed
each step's Deposit with its weights (``DEPOSIT_WEIGHTS``) so multiplied, so what it sums is in the readout's units;
the photon state it is handed carries the relative weight. On unit-weight rows this is bitwise the absolute-weight
loop.

Truth rows (``Model.analog``): each photon also carries its analog photon (``transport.state.Analog``), drawn from one
uniform of its own stream (``ANALOG_STREAM``) and its row's analog weight (``analog``: the physical photons the row
stands for, <= 1); ``per_item`` then reports the analog absorption (sensor -1: none). A row's label
(``Photons.label``) reaches the records as their ``label`` when the rows carry one.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import tallymap as tm

from lumen import _env
from lumen._core import rng
from lumen._core.rng import ANALOG_STREAM
from lumen._core.static import Static
from lumen.record import DEPOSIT_FIELDS, VECTOR_FIELDS
from lumen.transport import state
from lumen.transport.item import HEALTH, fates, start_photon
from lumen.transport.step import Operands, step

MAX_VERTICES = 1 << 22  # the guard (vertices per photon), not a physics cut
BACKENDS = ("auto", "cpu", "xla")  # lumen's backend names
# tallymap's name of its kernel backend (exact sums, one thread per photon), compiled for the device JAX places the
# loop on (CUDA on a GPU, C++/OpenMP on the CPU): lumen's "auto" and "cpu"; lumen's "xla" is tallymap's "xla"
KERNEL = "auto"
OVERFLOW = (
    "fixed_point_overflow"  # the stats key of the exact sums' count of NaN bins (a non-finite emit, a passed top)
)
DEPOSIT_WEIGHTS = ("q", "lost", "abandoned", "escaped", "distance_cut", "absorbed_bulk", "absorbed_surface",
                   "rouletted", "reweighted", "share")  # a Deposit's weights (None: off)  # fmt: skip
RECORDS = "deposits"  # the photon loop's records output (``tallymap.Records``)
SCHEDULE = "step"  # tallymap refills a lane per step, not per photon: 1.09-1.39x faster for lumen (L40S)
MIN_BLOCKS = 8  # tallymap caps the loop kernel's registers so that this many blocks fit an SM (outputs unchanged)


class RunResult(NamedTuple):
    """A run's outputs ({name: cells}), each photon's ``per_item`` report and ``overflow`` (exact sums' NaN cells)."""

    outputs: dict
    per_item: dict
    overflow: jax.Array


def run(  # noqa: PLR0913 -- the signature of docs/dev/architecture.md, "Data flow and signatures"
    readout, model, ops, rows, key, *, max_vertices=MAX_VERTICES, diagnostics=(), backend="auto",
    first_id=0, n_active=None, analog=None, weights=(), records=None,
):  # fmt: skip
    """Return the RunResult of the photons ``rows`` (a ``lumen.record.Photons`` batch) under ``key`` (raw key data):
    ``trace`` and the readout's outputs built from its records (``rebuild``), with the health and the diagnostics'
    sums.

    ``readout``: its ``emits``, ``outputs`` and ``moments`` (``lumen.readout.fill.readout``), built in one launch over
    every record; ``weights``: the readout's operands (per sensor group of ``readout``, its time histogram's QE x CE per
    cell, the sensor model's ``acceptance`` of the run's values). The records' slots follow ``capacity``'s rule (with
    values: one per row, run again at the count's power of two past them; traced: the memory's, ``guard``ed; or
    ``records``). ``lumen.simulate`` builds its readout per block of sensors.
    """  # noqa: D205
    check_device(backend)
    fields = record_fields(model.local, rows.label is not None)
    cells = sum(c for c, _ in readout.outputs().values())
    rows_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(rows))
    free = memory_limit(backend) - rows_bytes - readout_bytes(cells, True, backend)
    inputs = (ops, rows, key, analog)
    concrete = all(jax.core.is_concrete(x) for x in jax.tree_util.tree_leaves(inputs))
    room, n = slots_in(free, fields, True, batch_size(inputs)), rows.x.shape[0]
    cap = capacity(room, rows=n if concrete else None, records=records)

    def go(cap):
        return trace(
            model,
            ops,
            rows,
            key,
            fields=fields,
            capacity=cap,
            max_vertices=max_vertices,
            diagnostics=diagnostics,
            backend=backend,
            first_id=first_id,
            n_active=n_active,
            analog=analog,
        )

    t = go(cap)
    over, need = overflow(t.count, t.dropped, cap) if concrete else (False, cap)
    if over:  # past its slots (poisoned): again at the count's power of two
        if need is None or need > room:
            raise MemoryError(f"a launch appends {int(t.count)} deposit records (int32), over the {room} slots that "
                              "fit: give fewer rows")  # fmt: skip
        cap = need
        t = go(cap)
    if not concrete:
        guard(t, cap, n)
    out, nan = rebuild(readout, t.records, t.count, weights, backend=backend)
    return RunResult(out | t.sums | records_health(t), t.per_item,
                     jnp.asarray(t.overflow, jnp.int32) + jnp.asarray(nan, jnp.int32))  # fmt: skip


def records_health(t) -> dict:
    """Return a launch's record counts for the health: its records ("records"), those past the capacity and photons
    whose two passes differed ("records_dropped", "records_mismatch": each poisons the outputs; 0 in a run that
    fits)."""  # noqa: D205, D209
    i32 = lambda x: jnp.asarray(x, jnp.int32)  # noqa: E731
    return {"records": i32(t.count), "records_dropped": i32(t.dropped), "records_mismatch": i32(t.mismatch)}


def resolve_backend(backend) -> tuple[str, str | None]:
    """Return (tallymap's backend name, why the GPU is expected) of lumen's: the one place backends are decided.

    The name is ``KERNEL`` ("auto" and "cpu": tallymap compiles its kernel for the device JAX places the loop
    on, a GPU or the CPU; ``on_device`` moves a "cpu" run to the CPU) or
    "xla" (tallymap's plain-JAX backend). ``why`` says why the GPU is expected
    (``"auto"`` where the environment expects a GPU, ``lumen._env.gpu_expected``), else None.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    why = _env.gpu_expected() if backend == "auto" else None
    return ("xla" if backend == "xla" else KERNEL), why


def readout_bytes(cells: int, derivative: bool, backend) -> int:
    """Return the device bytes one launch of ``cells`` exact float32 cells holds, a derivative's tangent cells with
    them (``tm.call_memory``: tallymap's accumulators, readout and launch buffers)."""  # noqa: D205, D209
    return tm.call_memory([tm.Sum((max(cells, 1),), fixed_point=True)], resolve_backend(backend)[0],
                          "jvp" if derivative else None)  # fmt: skip


def tallymap_version() -> str:
    """Return the installed tallymap's version (the run records name it)."""
    return getattr(tm, "__version__", "unknown")


def vmap_runner(item, outputs, backend):
    """Return tallymap's vmap of ``item`` over one axis (one thread each) on a resolved backend (``resolve_backend``).

    For checks that run a plain per-item function on the photon loop's engine (``lumen.build.placements``).
    """
    return tm.vmap(item, outputs=outputs, in_axes=0, backend=backend)


def sum_output(cells: int):
    """Return a tallymap float sum of ``cells`` cells (no fixed point), an output of ``vmap_runner``."""
    return tm.Sum((cells,), fixed_point=False)


class DeviceError(RuntimeError):
    """A run that expects the GPU found JAX on another device (a failed CUDA plugin falls back to the CPU silently)."""


def _jax_on_cpu() -> bool:
    """Return whether JAX places new computations on the CPU (``jax.default_device``, else its default backend)."""
    dev = jax.config.jax_default_device
    if dev is None:
        return jax.default_backend() == "cpu"
    return getattr(dev, "platform", str(dev)) == "cpu"


def on_device(backend, f, *args, back=True):
    """Return ``f(*args)``; for ``backend="cpu"`` where JAX places computations elsewhere (a GPU), on the CPU.

    tallymap compiles for the device JAX places the computation on, so a "cpu" run puts ``args`` on the CPU and runs
    ``f`` inside ``jax.default_device`` of it; with ``back``, the outputs are put back on JAX's default device, where
    the caller's other arrays are. ``jax.device_put`` is differentiable: tangents and cotangents move with the values.
    """
    if backend != "cpu" or _jax_on_cpu():
        return f(*args)
    home, cpu = jax.devices()[0], jax.devices("cpu")[0]
    with jax.default_device(cpu):
        out = f(*jax.device_put(args, cpu))
    return jax.device_put(out, home) if back else out


def check_device(backend):
    """Refuse a run that expects the GPU while JAX's default backend is not it (``DeviceError``).

    Expected: ``backend="auto"`` where the environment expects a GPU (``lumen._env.gpu_expected``:
    ``JAX_PLATFORMS=cuda``, Slurm's GPU variables). ``backend="cpu"`` and ``"xla"`` never expect one ("cpu" is moved
    to the CPU, ``on_device``).
    """
    _, why = resolve_backend(backend)
    if why and jax.default_backend() not in ("gpu", "cuda"):
        raise DeviceError(f"the photon loop expects the GPU ({why}) but JAX runs on {jax.default_backend()!r}: the "
                          "CUDA plugin did not load (see JAX's log above); run on a working GPU node, or pass "
                          "backend='cpu'")  # fmt: skip


class ClosedOverValues(ValueError):  # noqa: N818 -- the documented error catalogue's name
    """The photon loop closes over a float array: a value baked into the kernel, which recompiles when it changes."""


def _refuse_closed_over():
    programs = tm.history()["programs"]
    floats = programs[-1]["closed_over_floats"] if programs else []
    if floats:
        raise ClosedOverValues(f"the photon loop closes over {len(floats)} float array(s) {floats[:4]}: pass "
                               "every value through the Operands (lumen.transport.step.operands)")  # fmt: skip


@functools.lru_cache(maxsize=64)
def _jitted(fn):
    return jax.jit(lambda static, *a: fn(static(), *a), static_argnums=0)


def _once(fn, model, *args):
    """Return ``fn(model, *args)`` through a jit of it: its jaxpr is traced once per model structure and shapes
    (``Model.static``), so tallymap's trace of the item at every launch reuses it (a counted launch, ``capacity``
    None, traces the item twice); a float array it closes over is refused (``ClosedOverValues``), as one the item
    closes over is."""  # noqa: D205, D209
    j = _jitted(fn)
    floats = [(tuple(c.shape), str(c.dtype)) for c in j.trace(model.static, *args).jaxpr.consts
              if jnp.issubdtype(getattr(c, "dtype", jnp.int32), jnp.floating)]  # fmt: skip
    if floats:
        raise ClosedOverValues(f"the photon loop closes over {len(floats)} float array(s) {floats[:4]}: pass "
                               "every value through the Operands (lumen.transport.step.operands)")  # fmt: skip
    return j(model.static, *args)


HEALTH_SUMS = (*HEALTH, "truncated")  # a photon's health, as it ends: its fates and the weight the guard stopped


def _photon_health(acc, ph, w0):
    """Return a photon's health as it ends (``HEALTH_SUMS``): its fate totals (``acc``, ``HEALTH``) and the weight the
    vertex guard stopped (``truncated``), each relative to its row's weight ``w0``."""  # noqa: D205, D209
    return jnp.concatenate([acc * w0, jnp.where(ph.alive, ph.w * w0, 0.0)[None]])  # stopped by the guard


def _health_sums(h, backend):
    """Return the health's exact sums ({name: (1,)}, ``HEALTH_SUMS``) of the photons' rows ``h`` (n, len(HEALTH_SUMS)),
    in a launch of their own."""  # noqa: D205, D209

    def item(tl, row):
        for i, k in enumerate(HEALTH_SUMS):
            tl = tl.add(k, jnp.int32(0), row[i])
        return tl

    outputs = {k: tm.Sum((1,), fixed_point=True) for k in HEALTH_SUMS}
    return dict(tm.vmap(item, outputs=outputs, in_axes=0, backend=backend)(h).outputs)


def _loop(static, ops, rows, key, first_id, n_active, analog=None):  # noqa: PLR0913
    model, max_vertices, diagnostics, backend, fields, capacity = static
    model, diagnostics = model(), tuple(dg() for dg in diagnostics)  # by their structures (``trace``)
    n = rows.x.shape[0]
    name = resolve_backend(backend)[0]
    kernel = name == KERNEL
    if kernel:
        _env.tallymap_cache()  # tallymap's kernels under lumen's cache root, not tallymap's own default in HOME
    shapes = {}
    for dg in diagnostics:
        shapes |= dg.outputs()
    outputs = {k: tm.Sum((cells,), fixed_point=exact) for k, (cells, exact) in shapes.items()}
    outputs[RECORDS] = tm.Records(dict(fields), capacity=capacity)

    def item(tl, x, b):
        o, k, fid = b
        pkey = rng.photon_key(k, fid + x["id"])
        w0 = x["row"].w  # the row's weight: the photon carries its weight relative to it (starting at 1)

        # the start's operands are only what it reads (the navigator): the cond that
        # runs it once is differentiated through all its branch closes over, and with the values and the reference
        # closed over the reverse pass added a cotangent into a tally (tallymap refuses it: a grad on SK failed)
        starts = Operands(params=None, reference=None, nav=o.nav)

        def begin():  # the photon's start (it locates it: a loop of its own), once, as the first iteration's work
            ph = _once(start_photon, model, starts, x["row"]._replace(w=jnp.ones_like(w0)))
            if model.analog:  # its analog photon: one uniform of its own stream, open with probability k
                u = rng.uniforms(jax.random.fold_in(pkey, ANALOG_STREAM), 1)[0]
                ph = ph._replace(analog=state.analog_start(u, x["k"], model.local))
            return ph, tuple(dg.init(ph) for dg in diagnostics)

        def body(c):
            p, tl, mem, started, acc = c
            if kernel:  # the plain-JAX backend vmaps the item: a cond there would run both branches at every step
                p, mem = jax.lax.cond(started, lambda: (p, mem), begin)
            p2, dep = _once(step, model, o, p, pkey)
            if x["row"].label is not None:
                dep = dep._replace(label=x["row"].label)
            tl = tl.append(RECORDS, record(dep, w0, fid + x["id"], fields), when=dep.sensor >= 0)
            acc = acc + fates(dep)  # the photon's fate totals: added to the health once, when it ends
            adds, mem2 = [], []
            seen = absolute(dep, w0) if diagnostics else dep
            for dg, m in zip(diagnostics, mem, strict=True):
                m2, a = dg.update(m, p, p2, seen, x["id"])
                adds, mem2 = adds + a, mem2 + [m2]
            for name, cell, value in adds:
                tl = tl.add(name, cell, value)
            return p2, tl, tuple(mem2), jnp.bool_(True), acc

        # one top-level loop (tallymap's step schedule needs it): the carry starts as a placeholder of the photon
        ph0, mem0 = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), jax.eval_shape(begin)) if kernel \
            else begin()  # fmt: skip
        live = lambda c: ~c[3] | (c[0].alive & (c[0].vertex < max_vertices))  # noqa: E731
        acc0 = jnp.zeros(len(HEALTH), jnp.float32)
        ph, tl, _, _, acc = jax.lax.while_loop(live, body, (ph0, tl, mem0, jnp.bool_(False), acc0))
        rec = dict(vertices=ph.vertex, health=_photon_health(acc, ph, w0))
        if model.analog:
            a = ph.analog
            rec = (
                rec
                | dict(sensor=a.sensor, time=a.time, lam=a.lam, scale=a.scale)
                | {f: getattr(a, f) for f in ("pos", "dir", "normal", "gen") if getattr(a, f) is not None}
            )
        return tl, rec

    xs = dict(row=rows, id=jnp.arange(n, dtype=jnp.int32)) | ({} if analog is None else {"k": analog})
    opts = dict(outputs=outputs, max_steps=max_vertices + 1, backend=name,
                schedule=SCHEDULE, min_blocks_per_sm=MIN_BLOCKS, in_axes=(0, None))  # fmt: skip
    r = tm.vmap(item, **opts)(xs, (ops, key, first_id), n_active=n_active)
    _refuse_closed_over()
    per_item = dict(r.per_item)
    sums = {k: v for k, v in r.outputs.items() if k != RECORDS} | _health_sums(per_item.pop("health"), name)
    return Traced(dict(r.outputs[RECORDS]), r.stats["appended"][RECORDS], r.stats["records_dropped"][RECORDS],
                  r.stats["records_mismatch"][RECORDS], sums, per_item,
                  r.stats.get(OVERFLOW, jnp.int32(0)))  # fmt: skip


def absolute(dep, w0):
    """Return a Deposit with its weights (``DEPOSIT_WEIGHTS``) times the row weight ``w0``, as the emits add them."""
    return dep._replace(**{k: getattr(dep, k) * w0 for k in DEPOSIT_WEIGHTS if getattr(dep, k) is not None})


# ------------------------------------------------------------------------------------------------- deposit records
VECTORS = VECTOR_FIELDS  # the Deposit's 3-vectors: a record field per component (fields are scalars)


def record_fields(local: tuple, labelled: bool) -> tuple:
    """Return the deposit record's fields ((name, dtype), ...) of a run carrying the Deposit fields ``local``
    (``Model.local``) and, when ``labelled``, the emission label: sensor, q (relative to the row weight), the row
    weight w0, time, lam, the photon's id in the run, then the label and each carried field (a vector by
    component)."""  # noqa: D205, D209
    f = [("sensor", jnp.int32), ("q", jnp.float32), ("w0", jnp.float32), ("time", jnp.float32),
         ("lam", jnp.float32), ("photon", jnp.int32)]  # fmt: skip
    f += [("label", DEPOSIT_FIELDS["label"][1])] if labelled else []
    for k in sorted(local):
        shape, dtype = DEPOSIT_FIELDS[k]
        f += [(f"{k}{i}", dtype) for i in range(shape[0])] if shape else [(k, dtype)]
    return tuple(f)


def record_bytes(fields) -> int:
    """Return a record's bytes (one slot of every field)."""
    return sum(jnp.dtype(d).itemsize for _, d in fields)


def record(dep, w0, photon, fields) -> dict:
    """Return the record of one step's Deposit (its weights relative to the row weight ``w0``) of photon ``photon``."""
    return encode(lambda k: w0 if k == "w0" else photon if k == "photon" else getattr(dep, k), fields)


DEFAULTS = {"inv_an": 1.0}  # a record field a photon list may leave out: its value (else 0)


def encode(value, fields) -> dict:
    """Return a record ({field: array}, ``record_fields``) from ``value(name)``, a deposit field's value (a vector's
    last axis its components); the one encoding of the photon loop's records and of a photon list's
    (``lumen.readout.grid.from_photons``)."""  # noqa: D205, D209
    out = {}
    for k, d in fields:
        if k[-1].isdigit() and k[:-1] in VECTORS:
            v = value(k[:-1])[..., int(k[-1])]
        else:
            v = value(k)
        out[k] = jnp.asarray(v, d)
    return out


class Record(NamedTuple):
    """A deposit record as the readout's emits read it (the Deposit's fields of the record, each scalar or None)."""

    sensor: jax.Array
    q: jax.Array
    time: jax.Array
    lam: jax.Array
    label: jax.Array | None = None
    pos: jax.Array | None = None
    dir: jax.Array | None = None
    normal: jax.Array | None = None
    gen: jax.Array | None = None
    inv_an: jax.Array | None = None


def deposit(rec: dict) -> Record:
    """Return the Record of one record's fields ({field: scalar}, ``record_fields``)."""
    vec = {k: jnp.stack([rec[f"{k}{i}"] for i in range(3)]) for k in VECTORS if f"{k}0" in rec}
    one = {k: rec[k] for k in ("label", "gen", "inv_an") if k in rec}
    return Record(rec["sensor"], rec["q"], rec["time"], rec["lam"], **one, **vec)


def _rebuild(static, rec, n, weights):
    readout, backend, part = static
    readout = readout()
    name = resolve_backend(backend)[0]
    if name == KERNEL:
        _env.tallymap_cache()
    mine = set(_moment_names(readout, rec, weights))
    outputs = {k: tm.Sum((cells,), fixed_point=exact) for k, (cells, exact) in readout.outputs().items()
               if (k in mine) == (part == "moments")}  # fmt: skip
    size = rec["sensor"].shape[0]
    nn = jnp.minimum(n, size)
    if part == "moments":  # one item per record slot, the records a broadcast operand: it reads its photon's others

        def item(tl, i, b):
            for k, c, v in _moment_adds(readout, b[1], nn, i, b[0]):
                tl = tl.add(k, c, v)
            return tl

        res = tm.vmap(item, outputs=outputs, in_axes=(0, None), backend=name)(jnp.arange(size, dtype=jnp.int32),
                                                                              (weights, rec), n_active=nn)  # fmt: skip
        return dict(res.outputs), res.stats.get(OVERFLOW, jnp.int32(0))

    def item(tl, r, b):
        dep = deposit(r)
        for k, c, v in readout.emits(dep, b):
            tl = tl.add(k, c, v * r["w0"])  # formed as the photon loop formed its emits: (value) x row weight
        return tl

    res = tm.vmap(item, outputs=outputs, in_axes=(0, None), backend=name)(rec, weights, n_active=nn)
    return dict(res.outputs), res.stats.get(OVERFLOW, jnp.int32(0))


def _moment_names(readout, rec, weights) -> list:
    """Return the readout's moment outputs (``readout.moments``' names; none without)."""
    if readout.moments is None:
        return []
    dep = deposit({k: jnp.zeros((), v.dtype) for k, v in rec.items()})
    return [k for k, _, _ in readout.moments(dep, weights)]


def _moment_adds(readout, rec, nn, i, weights) -> list:
    """Return the squares of the (photon, cell) sums of the moment outputs that record i leads.

    A record leads a (photon, cell) of a moment output when no earlier record of its photon, and no earlier add of
    its own, adds to that cell (the records are in (photon, step) order, so the photon's records are its
    neighbours). The leader sums the photon's adds to the cell (each times its record's row weight) in order, its own
    and its later records', in float32, and adds the square.
    """

    def moment_adds(j):  # record j's moment adds, each times its row weight
        rj = {k: v[j] for k, v in rec.items()}
        return [(k, jnp.asarray(c, jnp.int32), jnp.asarray(v, jnp.float32) * rj["w0"])
                for k, c, v in readout.moments(deposit(rj), weights)]  # fmt: skip

    mine = moment_adds(i)

    def joined(adds, a, into):  # ``into`` plus the adds of ``adds`` to add a's cell, in order
        k, c, _ = mine[a]
        for kb, cb, vb in adds:
            if kb == k:
                into = into + jnp.where((cb == c) & (c >= 0), vb, 0.0)
        return into

    def met(adds, a, into):  # ``into``, or whether one of ``adds`` adds to add a's cell
        k, c, _ = mine[a]
        for kb, cb, _ in adds:
            if kb == k:
                into = into | ((cb == c) & (c >= 0))
        return into

    x = [joined(mine[a + 1 :], a, mine[a][2]) for a in range(len(mine))]  # its own later adds to the cell
    dup = [met(mine[:a], a, jnp.bool_(False)) for a in range(len(mine))]  # an own earlier add leads it
    photon, size = rec["photon"][i], rec["sensor"].shape[0]
    ours = lambda j: rec["photon"][jnp.clip(j, 0, size - 1)] == photon  # noqa: E731

    def earlier(c):  # an earlier record of the photon: does it add to record i's cells?
        mj = moment_adds(c[0])
        return c[0] - 1, tuple(met(mj, a, d) for a, d in enumerate(c[1]))

    def later(c):  # a later record of the photon: its adds to record i's cells, in order
        mj = moment_adds(c[0])
        return c[0] + 1, tuple(joined(mj, a, xa) for a, xa in enumerate(c[1]))

    _, dup = jax.lax.while_loop(lambda c: (c[0] >= 0) & ours(c[0]), earlier, (i - 1, tuple(dup)))
    _, x = jax.lax.while_loop(lambda c: (c[0] < nn) & ours(c[0]), later, (i + 1, tuple(x)))
    return [(k, jnp.where(d, -1, c), v * v) for (k, c, _), v, d in zip(mine, x, dup, strict=True)]


_REBUILD = jax.jit(_rebuild, static_argnums=0)


def rebuild(readout, records, n, weights=(), *, backend="auto"):
    """Return (the readout's outputs {name: cells}, their exact sums' NaN cells) built from deposit records.

    ``readout``: ``emits(dep, weights)`` the (output, cell, value) adds of one record (``Record``), ``outputs()``
    {name: (cells, exact)} and ``moments(dep, weights)`` (None: none) the adds whose per-photon sums are squared
    (``lumen.readout.fill``); ``records``: a run's records (``trace``), the first ``n`` of them filled, in (photon,
    step) order; ``weights``: operands of the emits. One tallymap item per record: each add times the record's row
    weight as the photon loop multiplied its emits, into exact sums, so a cell is the correctly rounded float32 of the
    exact sum of the same float32 values whatever the order; derivatives through the records' float fields and the
    weights. A moment output's cell is sum_i x_i^2 over the photons, x_i photon i's adds to it summed over its
    records (float32, in order) before the square (``_moment_adds``), into the same exact sums: a photon's records
    are in one launch, so the moments of several launches add. The moments are a launch of their own on the records
    without a derivative (``stop_gradient``): a moment is no gradient's error, and the differentiated launch carries
    none of their work.
    """
    weights = tuple(jnp.asarray(w, jnp.float32) for w in weights)
    n = jnp.asarray(n, jnp.int32)
    held = Static(readout)  # the readout by its structure (``lumen._core.static``)
    out, nan = on_device(backend, lambda *a: _REBUILD((held, backend, "emits"), *a), records, n, weights)
    if readout.moments is not None:  # a launch of their own, on the records without a derivative: no derived kernel
        still = jax.tree_util.tree_map(jax.lax.stop_gradient, (records, weights))
        mo, mnan = on_device(backend, lambda *a: _REBUILD((held, backend, "moments"), *a), still[0], n, still[1])
        out, nan = out | mo, nan + mnan
    return out, nan


def memory_statistics(backend) -> dict | None:
    """Return JAX's memory statistics of the run's device (a GPU's: its ``bytes_limit``), None for the host (the
    "xla" backend runs where JAX places it: on a GPU, the GPU's)."""  # noqa: D205, D209
    if backend == "cpu" or _jax_on_cpu():
        return None
    stats = jax.devices()[0].memory_stats() or {}
    return stats if "bytes_limit" in stats else None


def memory_limit(backend) -> float:
    """Return the bytes the run's device may hold: the GPU's limit (JAX's memory statistics), else the host memory
    available now (``lumen._env.host_available``: what the process's limit leaves beside what is in use)."""  # noqa: D205, D209
    stats = memory_statistics(backend)
    return float(stats["bytes_limit"]) if stats else _env.host_available()


class RecordsUnknown(ValueError):  # noqa: N818 -- the documented error catalogue's name
    """A launch under jax.jit or jax.vmap whose deposit records cannot have a single slot in the device's memory."""


class RecordsOverflow(RuntimeError):  # noqa: N818 -- the documented error catalogue's name
    """A launch under jax.jit or jax.vmap that appended more deposit records than its static capacity: raised when
    the run executes."""  # noqa: D205, D209


MAX_RECORDS = 1 << 29  # the most slots: tallymap's one pass overflows its int32 slot arithmetic at 2^30 (I7)
# A one-pass Records launch's peak device memory over its record outputs: the raw buffer and each record's slot key,
# then the canonical layout written beside them (tallymap's _records.canon). Measured on SK and SBND (lumen study
# review6/I7, tallymap 94da9c2, the device's peak over the launch's record bytes); the one statement of it.
PEAK = 2.4
KEY_BYTES = 8  # a one-pass record's slot key (its item and its append ordinal, two int32): an output of the launch


def pow2(n: int) -> int:
    """Return the power of two at or above ``n`` (1 for n <= 1)."""
    return 1 << max(int(n) - 1, 0).bit_length()


def slot_bytes(fields, derivative: bool, batch: int = 1) -> float:
    """Return the device bytes one record slot of a launch takes at its peak.

    One launch: ``PEAK`` times the record's bytes r (``record_bytes``; with a tangent per float field for a
    derivative, counted as a second record). Under jax.vmap of B members (``batch_size``) tallymap runs the members
    one after another (its batching rule: ``lax.map``, one launch each) and stacks every output of each: per slot
    the record (r), its slot key (``KEY_BYTES``) and, outside the map, the copy of each integer field that an overflow
    sets to -1 (tallymap ``_api``). So the B members' outputs are held at once beside the one launch running at its
    peak: ``PEAK`` r + B (r + ``KEY_BYTES`` + the integer fields' bytes). (SK's 24 B record, two of them integer
    fields: 40 B per member per slot, as XLA's buffer assignment of the jitted vmap gives: 120, 200 B per slot at 2, 4
    members, of which 40 the launch's own.)
    """
    r = record_bytes(fields) * (1 + derivative)
    ints = sum(jnp.dtype(d).itemsize for _, d in fields if jnp.issubdtype(jnp.dtype(d), jnp.integer))
    return PEAK * r + (batch * (r + KEY_BYTES + ints) if batch > 1 else 0.0)


def slots_in(free: float, fields, derivative: bool, batch: int = 1) -> int:
    """Return the most record slots, a power of two (0: none), whose launch at its peak (``slot_bytes``) fits
    ``free`` bytes; at most ``MAX_RECORDS``."""  # noqa: D205, D209
    room = min(int(max(free, 0.0) // slot_bytes(fields, derivative, batch)), MAX_RECORDS)
    return 1 << room.bit_length() - 1 if room > 0 else 0


def batch_size(tree) -> int:
    """Return the product of the jax.vmap batch sizes ``tree``'s leaves carry (1 outside jax.vmap)."""
    from jax.interpreters import batching

    size = 1
    for x in jax.tree_util.tree_leaves(tree):
        b, y = 1, x
        while isinstance(y, jax.core.Tracer):
            if isinstance(y, batching.BatchTracer) and isinstance(y.batch_dim, int):
                b *= y.val.shape[y.batch_dim]
            y = getattr(y, "val", getattr(y, "primal", None))
        size = max(size, b)
    return size


def capacity(room: int, *, rows: int | None = None, counted: int | None = None, records: int | None = None) -> int:
    """Return a launch's record slots (a power of two), at most ``room``, the most whose launch fits the memory left
    (``lumen.api._record_room``, from ``slot_bytes``).

    ``records`` given (an optional override, e.g. to shrink the memory): its power of two. A run with values
    (``rows``, the launch's rows, given): ``counted``, the power of two of the most records a launch of the run
    appended so far (``Traced.count``, tallymap's ``stats['appended']``, exact also past the slots), else one slot
    per row to start; ``lumen.simulate`` runs a launch whose count exceeds its slots again at the count's power of
    two (its outputs, poisoned, are dropped), or in smaller chunks when that does not fit. Under jax.jit or jax.vmap
    (no ``rows``) the count cannot reach the host before the shapes are fixed: ``room`` itself, so nothing that fits
    the memory overflows; the result does not depend on the slots while nothing overflows (the records' layout is
    canonical, the readout's sums exact), and a launch past them is refused when it executes (``guard``).
    MemoryError when ``records`` does not fit; ``RecordsUnknown`` when not one slot does.
    """  # noqa: D205
    if records is not None:
        slots = pow2(records)
        if slots > room:
            raise MemoryError(f"records={int(records)} need {slots} slots, over the {room} that fit the memory left "
                              "for them: use a smaller chunk")  # fmt: skip
        return slots
    if room < 1:
        raise RecordsUnknown("not one deposit record fits the memory left beside the readout: use a smaller chunk "
                             "(or fewer jax.vmap members)")  # fmt: skip
    if rows is None:
        return int(room)
    return int(min(counted if counted is not None else pow2(rows), room))


def overflow(appended, dropped, slots: int) -> tuple[bool, int | None]:
    """Return (whether a launch with values overflowed its ``slots``, the slots its records need: None when no count
    the slots can hold reports it). tallymap's ``appended`` and ``records_dropped`` are int32: past 2^31 - 1 records
    the count wraps (negative, or anything) while the drop is still counted, so a drop, a count past the slots and a
    negative count are each an overflow, and a need past ``MAX_RECORDS`` is none a launch can have."""  # noqa: D205, D209
    a, d = int(appended), int(dropped)
    if a <= slots and a >= 0 and d == 0:
        return False, slots
    if a < 0 or a <= slots or a > MAX_RECORDS:  # wrapped (or a drop the count does not show): no size holds it
        return True, None
    return True, pow2(a)


def guard(t, slots: int, chunk: int) -> None:
    """Refuse, when the run executes, a traced launch ``t`` (``Traced``) that appended more records than its ``slots``:
    its outputs are already poisoned (NaN, ``health.records_dropped``); this makes it an error under jax.jit, jax.vmap
    and their derivatives alike (a host callback: the count reaches the host only once the launch has run)."""  # noqa: D205, D209

    def refuse(appended, dropped):  # (under jax.vmap, the batch's arrays)
        if int(np.max(dropped)) > 0:
            need = int(np.max(appended))
            below = chunk * slots // max(need, 1)
            raise RecordsOverflow(f"a launch appended {need} deposit records, over its {slots} slots: give "
                                  f"records={need} (or more), or a chunk below {below}")  # fmt: skip

    jax.debug.callback(refuse, t.count, t.dropped)


def trace(model, ops, rows, key, *, fields, capacity=None, max_vertices=MAX_VERTICES, diagnostics=(),  # noqa: PLR0913
          backend="auto", first_id=0, n_active=None, analog=None):  # fmt: skip
    """Return the Traced photons ``rows`` (a ``lumen.record.Photons`` batch) under ``key`` (raw key data): their
    deposit records (``record_fields``: ``fields``), the health and each photon's report.

    ``capacity`` None (``capacity``): the launch's own count. tallymap's count pass runs and returns the count to the
    host, and the fill runs with the power of two at or above it (``Traced.records`` of that length); its program is
    compiled once per power of two (tallymap keeps the compiled count and fill programs). An int: that many slots, in
    one jitted loop (under jax.jit or jax.vmap, where the count cannot reach the host before the shapes are fixed).
    Either way the operands are ``ops``, the key, ``first_id`` and ``n_active`` (only rows i < n_active run): changing
    a value never recompiles, and a loop that closes over a float array is refused (``ClosedOverValues``).
    ``analog`` (N,): the rows' analog weights, given exactly when ``model.analog`` (truth rows).
    """  # noqa: D205
    cap = None if capacity is None else int(capacity)
    # the model and the diagnostics by their structures (``lumen._core.static``): a reload, or a diagnostic built
    # afresh, of the same structure compiles nothing and leaves nothing behind
    static = (model.static, int(max_vertices), tuple(Static(dg) for dg in diagnostics), backend, tuple(fields), cap)
    if int(max_vertices) < 1:
        raise ValueError(f"max_vertices must be >= 1, got {max_vertices}")
    check_device(backend)
    if (analog is None) == model.analog:
        raise ValueError(
            "run: give the rows' analog weights (analog=) exactly when the model carries the analog photon"
        )
    first = jnp.asarray(first_id, jnp.int32)
    loop = _LOOP if cap is not None else _loop  # counted: tallymap's call outside a jit (its own compiled programs)
    return on_device(backend, lambda *a: loop(static, *a), ops, rows, key, first, n_active, analog)


class Traced(NamedTuple):
    """A launch's deposit records ({field: (capacity,)}, ``record_fields``, in (photon, step) order, ``count`` of them;
    ``dropped`` past the capacity, ``mismatch`` photons whose count and fill passes differ: either poisons every float
    output), its sums ({name: cells}: the health, "truncated", the diagnostics'), ``per_item`` and ``overflow``."""  # noqa: D205, D209

    records: dict
    count: jax.Array
    dropped: jax.Array
    mismatch: jax.Array
    sums: dict
    per_item: dict
    overflow: jax.Array


_LOOP = jax.jit(_loop, static_argnums=0)
