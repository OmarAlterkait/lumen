"""The sensor readout's library stages on the binned deposits (``lumen.readout.Histograms``): acceptance, delay, spread.

    pe_s      = efficiency_s * sum_cells charge[s, cells] prod_k T_k[cells]
    masses    = the time histogram scaled to pe_s, each delay cell's slice through its delay table, summed
    spread    = the per-photoelectron time spread (TTS) and the offset
    density   = masses through the spread per node-spacing bin over the gate: the expected photoelectron time profile

Each is an independent function of the histograms and the group's split values ``v`` (``values``); every readout
output (``pe``, ``pe_density``, ``expected_hits``, ``expected_waveform``, ``sample``) composes them, so the time
response (delay x spread) is one definition everywhere. A stage of your own is a function of the same ``(h, v)``
(``lumen.response``). Linear in the histograms and in every acceptance value, so exact on the axes' knots and
differentiable. ``light`` is a ``lumen.readout.Histograms`` or a Light of one modelled group (its ``groups``);
``params`` any mapping holding the group's ``sensors/<group>/<field>`` values (a ``lumen.Params`` or a dict of
arrays: give a dict to differentiate).

Every stage acts on each sensor alone, so the stages after the charge (``masses``, ``density`` and the hit and waveform
readouts built on them) run over blocks of the group's sensors (``rows_map``): as many sensors at once as their working
memory fits in the device's free memory (the one budget, as the readout's cells are built in blocks,
``lumen.readout.fill.plan``), each block's rows written into the output in place, so no stage holds the whole group's
intermediates (Hyper-K's 19,620 sensors x 13,008 time nodes, DUNE FD1's 160,137 nodes). When the whole group fits the
stage runs on it in one piece, as before.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from lumen.readout.grid import CELL_BYTES, Histograms
from lumen.sources._base import bucket
from lumen.units import S_PER_NS

# a block's working memory is half the device's free memory (after the stage's output): the other half is XLA's own
# (its temporaries, the FFT plans' work areas, a derivative's tangents)
BLOCK_SHARE = 2
FFT_ARRAYS = 8  # float32 arrays of an FFT's length one sensor's convolution holds (padded input, transforms, product)


def histograms(light) -> Histograms:
    """Return the Histograms of a light: itself, or the one group's of a Light (``light.groups``)."""
    groups = getattr(light, "groups", None)
    if groups and len(groups) > 1:
        raise TypeError(f"readout: the light holds the histograms of the sensor groups {list(groups)}: give "
                        "light.group(name)")  # fmt: skip
    h = light if isinstance(light, Histograms) else next(iter(groups.values())) if groups else None
    if not isinstance(h, Histograms):
        raise TypeError(f"readout: expected lumen.readout.Histograms (or a light of one modelled group), got "
                        f"{type(light)}")  # fmt: skip
    if h.model is None:
        raise ValueError(
            f"readout: sensor group {h.group!r} has no sensor model: give the group a value set (SensorGroup.values)"
        )
    return h


def values(h: Histograms, params) -> dict:
    """Return the group's values split by component ({"tts": {"sigma": ...}, "qe": ..., ...}) as float arrays: float32,
    or float64 where a value is float64 under jax_enable_x64 (a float64 derivative check, ``lumen.tools``)."""  # noqa: D205, D209
    v = h.model.split(params, h.group)

    def flt(x):
        if isinstance(x, dict):
            return {k: flt(y) for k, y in x.items()}
        x = jnp.asarray(x)
        return x if x.dtype == jnp.float64 else x.astype(jnp.float32)

    return flt(v)


def _einsum(h: Histograms, v: dict, out: str):
    """Return the charge histogram contracted with the acceptance tables (``model.tables``), indices ``out``."""
    letters = "labcdefghijkmnopq"  # one index letter per axis (s: sensor)
    names = [a.name for a in h.grid.axes]
    ix = {n: letters[i] for i, n in enumerate(names)}
    spec = ["s" + "".join(ix[n] for n in names)]
    ops = [h.charge]
    for t, tax in h.model.tables:
        spec.append("".join(ix[n] for n in tax))
        ops.append(v[t])
    return jnp.einsum(",".join(spec) + "->" + out, *ops)


def expected_pe(h: Histograms, v: dict):
    """Return (S,) expected photoelectrons from the charge histogram and split values."""
    return v["efficiency"] * _einsum(h, v, "s")


def pe(light, params):
    """Return the expected photoelectrons per sensor: of a group's Histograms (its sensors), or of a Light (every
    sensor of the detector, in ``Light.arrivals``' order: each modelled group's at its sensors, 0 at a sensor of no
    modelled group)."""  # noqa: D205, D209
    if isinstance(light, Histograms):
        h = histograms(light)
        return expected_pe(h, values(h, params))
    n = jnp.shape(light.arrivals)[-1]
    out = jnp.zeros(n, jnp.float32)
    for h in (light.groups or {}).values():
        ids = jnp.arange(n) if h.sensors is None else jnp.asarray(h.sensors, jnp.int32)
        out = out.at[ids].add(expected_pe(h, values(h, params)))
    return out


def pe_error(light):
    """Return the Monte Carlo error of ``pe`` per sensor at the values the run was given (``simulate``'s ``params``):
    of a group's Histograms (its sensors), or of a Light (every sensor, as ``pe``; 0 at a sensor of no modelled group).

    From each sensor's per-photon second moment of its pe at those values (``lumen.readout.grid.moment_error``,
    unbiased for i.i.d. photons); the error of the pe at other values is not kept: run at them, or compare seeds.
    """  # noqa: D205
    if isinstance(light, Histograms):
        return histograms(light).pe_error
    n = jnp.shape(light.arrivals)[-1]
    out = jnp.zeros(jnp.shape(light.arrivals), jnp.float32)
    for h in (light.groups or {}).values():
        ids = jnp.arange(n) if h.sensors is None else jnp.asarray(h.sensors, jnp.int32)
        out = out.at[..., ids].set(h.pe_error)
    return out


def pe_moment_error(h: Histograms, v: dict, pe2, n: int):
    """Return the pe error per sensor of a group's histograms ``h`` at its split values ``v`` (those its pe moment
    was weighted by, ``lumen.readout.fill.readout``'s "pe2") from that moment ``pe2`` (per detector sensor, the
    detector's order; per event first with events) over ``n`` photons: the efficiency times ``moment_error``."""  # noqa: D205, D209
    from lumen.readout.grid import moment_error

    ids = slice(None) if h.sensors is None else jnp.asarray(h.sensors, jnp.int32)
    q2 = jnp.asarray(pe2)[..., ids]
    eff = jnp.asarray(v["efficiency"], jnp.float32)
    return eff * moment_error(q2, _einsum_events(h, v), n)


def _einsum_events(h: Histograms, v: dict):
    """Return the pe before the efficiency per sensor (``_einsum``), per event first when the histograms have one."""
    if h.charge.ndim == 1 + len(h.grid.axes):
        return _einsum(h, v, "s")
    return jax.vmap(lambda c: _einsum(dataclasses.replace(h, charge=c), v, "s"))(h.charge)


def free_memory() -> float:
    """Return the bytes free on the device JAX computes on: a GPU's limit less what is in use, else the host's."""
    from lumen import run as runner

    stats = runner.memory_statistics("auto")
    if stats and "bytes_in_use" in stats:
        return float(stats["bytes_limit"] - stats["bytes_in_use"])
    return runner.memory_limit("auto")


def rows_map(f, n: int, row_bytes: float, out_row_bytes: float = 0.0):
    """Return ``f(0, n)``: the arrays (each of leading axis n, one row per sensor or hit) of n independent rows,
    computed over blocks of rows. ``f(start, size)`` returns the rows [start, start + size) (``start`` an int when
    the whole fits, else traced); ``row_bytes``: a row's working memory, ``out_row_bytes``: its output's.

    A block is as many rows as fit in the free memory (``free_memory``) left by the output, over ``BLOCK_SHARE``; all
    of them fit: one call ``f(0, n)``. Otherwise equal blocks run in turn (``lax.scan``, the last one ending at row n),
    each written into the output in place (``dynamic_update_slice``) and rematerialised in reverse mode
    (``jax.checkpoint``), so neither a value nor a derivative holds more than one block's intermediates.
    MemoryError when one row does not fit.
    """  # noqa: D205
    budget = (free_memory() - n * out_row_bytes) / BLOCK_SHARE
    b = int(budget // max(row_bytes, 1.0))
    if b >= n:
        return f(0, n)
    if b < 1:
        raise MemoryError(f"the readout of one sensor needs {row_bytes / 1e9:.2f} GB, over the {budget / 1e9:.2f} GB "
                          "free for it: shorten the readout gates, widen the time spreads' declared lower ranges "
                          "(coarser nodes)")  # fmt: skip
    nb = -(-n // b)
    b = -(-n // nb)  # equal blocks
    starts = jnp.minimum(jnp.arange(nb, dtype=jnp.int32) * b, n - b)
    block = jax.checkpoint(lambda s: f(s, b))
    shapes = jax.eval_shape(block, jnp.int32(0))
    out = jax.tree_util.tree_map(lambda x: jnp.zeros((n, *x.shape[1:]), x.dtype), shapes)

    def body(acc, s):
        part = block(s)
        return jax.tree_util.tree_map(lambda o, x: jax.lax.dynamic_update_slice_in_dim(o, x, s, 0), acc, part), None

    return jax.lax.scan(body, out, starts)[0]


def sensors(h: Histograms, v: dict, start, size: int):
    """Return (histograms, values) of the group's sensors [start, start + size): the histograms' sensor axis and the
    per-sensor values (``lumen.sensors.value_sets.PER_SENSOR``, when given per sensor) cut; the whole group itself."""  # noqa: D205, D209
    from lumen.sensors.value_sets import PER_SENSOR

    n = h.charge.shape[0]
    if isinstance(start, int) and start == 0 and size == n:
        return h, v

    def cut(x):
        return None if x is None else jax.lax.dynamic_slice_in_dim(x, start, size, 0)

    hb = dataclasses.replace(h, **{k: cut(getattr(h, k)) for k in Histograms._ARRAYS})
    per = lambda k, x: k in PER_SENSOR and jnp.ndim(x) > 0 and jnp.shape(x)[0] == n  # noqa: E731
    return hb, {k: cut(jnp.asarray(x)) if per(k, x) else x for k, x in v.items()}


def masses_row_bytes(h: Histograms) -> float:
    """Return one sensor's working memory in ``masses``: its time histogram (every delay cell) through the delay."""
    g, m = h.grid, h.model
    work = 4 * g.n_delay * g.nt
    if m.delay is not None:
        off0, ntap = m.delay.taps(g.dt)[:2]
        work += FFT_ARRAYS * g.n_delay * bucket(g.nt + ntap + abs(off0))
    return CELL_BYTES * work


def density_row_bytes(h: Histograms) -> float:
    """Return one sensor's working memory in ``density``: ``masses``' and the convolution with the time spread."""
    g = h.grid
    nf, _ = n_bins(h)
    return masses_row_bytes(h) + CELL_BYTES * (FFT_ARRAYS * bucket(2 * g.nt + nf) + 4 * nf)


def masses(h: Histograms, v: dict):
    """Return the time histogram as photoelectrons after the delay: (node masses (S, nt), late (S, 2), early (S, 2)),
    per block of sensors (``rows_map``: ``block_masses`` of each)."""  # noqa: D205, D209
    if h.time is None:
        raise ValueError(f"sensor group {h.group!r}: no time histogram (readout 'pe'); simulate with a timed readout")
    g = h.grid
    return rows_map(lambda s, b: block_masses(*sensors(h, v, s, b)), h.time.shape[0], masses_row_bytes(h),
                    CELL_BYTES * (g.nt + 4))  # fmt: skip


def block_masses(h: Histograms, v: dict):
    """Return ``masses`` of the sensors of ``h`` in one piece.

    The time histogram is scaled per sensor so its total (nodes plus overflow) equals ``pe``: exact for any acceptance
    or efficiency scale; the time shape is the detected light's at the acceptance the run was given (``Grid.weighted``:
    simulate weights each deposit by it, so the shape is differentiable in it through the run). Each delay cell's
    slice goes through its delay table (``DelayTable.apply``); its overflow cells move by its mean delay.
    """
    if h.time is None:
        raise ValueError(f"sensor group {h.group!r}: no time histogram (readout 'pe'); simulate with a timed readout")
    H = h.time
    s = H.shape[0]
    cell = H.shape[1:-1]  # the delay's cells
    late = h.late if h.late is not None else jnp.zeros((s, *cell, 2), H.dtype)
    early = h.early if h.early is not None else jnp.zeros((s, *cell, 2), H.dtype)
    red = tuple(range(1, H.ndim - 1))
    total = H.sum(-1).sum(red) + late[..., 0].sum(red) + early[..., 0].sum(red)
    scale = expected_pe(h, v) / jnp.where(total != 0.0, total, 1.0)
    m = h.model
    if m.delay is not None:
        H = m.delay.apply(v["delay"], H, h.grid.dt)
        shift = m.delay.mean(v["delay"])
        late, early = late.at[..., 1].add(late[..., 0] * shift), early.at[..., 1].add(early[..., 0] * shift)
        late, early = late.sum(red), early.sum(red)
    return H * scale[:, None], late * scale[:, None], early * scale[:, None]


def spread(h: Histograms, v: dict):
    """Return (cdf, sf) of the per-photoelectron time spread (the TTS) at a lag x (ns)."""
    m = h.model
    return (lambda x: m.tts.cdf(v["tts"], x)), (lambda x: m.tts.sf(v["tts"], x))


def n_bins(h: Histograms):
    """Return (bins over the gate at the node spacing, nodes per front-end sample)."""
    m = h.model
    step = int(round(m.sampling / h.grid.dt)) if m.readout == "waveform" else 1
    n = int(math.ceil((h.gate[1] - h.gate[0]) / (h.grid.dt * step) - 1e-9))
    return n * step, step


def fft_conv(a, k, n_out):
    """Return the first ``n_out`` samples of the linear convolution of ``a`` with ``k`` along the last axis (by FFT)."""
    n = bucket(a.shape[-1] + k.shape[-1])
    return jnp.fft.irfft(jnp.fft.rfft(a, n) * jnp.fft.rfft(k, n), n)[..., :n_out]


def _f32_grid(i0: int, i1: int, dt: float, t0: float, dtype):
    """Return the float32 times t0 + i dt, i in [i0, i1), formed on the host in float32 (the device's own float32
    product and sum), as ``dtype`` (the histograms': float32, or float64 in 64-bit mode, where the times' sums with
    other values are then formed in float64 whether or not XLA keeps a float32 intermediate at a higher precision,
    as it may inside the loop over blocks)."""  # noqa: D205, D209
    import numpy as np

    return jnp.asarray(np.arange(i0, i1, dtype=np.float32) * np.float32(dt) + np.float32(t0), dtype)


def density(h: Histograms, v: dict, dark: bool = False):
    """Return the expected photoelectrons per bin (S, nf) over the gate, bin k [g0 + k dt, g0 + (k + 1) dt): the delayed
    node masses (``masses``) through the spread (``spread``) and the transit offset, plus with ``dark`` the dark
    counts. Per block of sensors (``rows_map``: ``block_density`` of each)."""  # noqa: D205, D209
    if h.time is None:
        raise ValueError(f"sensor group {h.group!r}: no time histogram (readout 'pe'); simulate with a timed readout")
    nf, _ = n_bins(h)
    return rows_map(lambda s, b: block_density(*sensors(h, v, s, b), dark), h.time.shape[0], density_row_bytes(h),
                    CELL_BYTES * nf)  # fmt: skip


def block_density(h: Histograms, v: dict, dark: bool = False):
    """Return ``density`` of the sensors of ``h`` in one piece."""
    g = h.grid
    H, late, early = block_masses(h, v)
    nf, _ = n_bins(h)
    s0 = h.gate[0] - g.t0
    T0 = jnp.reshape(v["t_offset"], (-1, 1))
    lag = _f32_grid(-g.nt, nf, g.dt, 0.0, H.dtype)[None, :]
    F = spread(h, v)[0]
    k = F(s0 + lag + g.dt - T0) - F(s0 + lag - T0)  # k[l + nt]: bin i, node j, l = i - j
    out = fft_conv(H, k, g.nt + nf)[..., g.nt :]
    e = _f32_grid(0, nf + 1, g.dt, h.gate[0], H.dtype)
    for cell in (late, early):
        tm = (cell[:, 1] / jnp.where(cell[:, 0] != 0.0, cell[:, 0], 1.0))[:, None]
        out = out + cell[:, :1] * jnp.diff(F(e[None, :] - tm - T0), axis=-1)
    return out + jnp.reshape(v["dark_rate"], (-1, 1)) * S_PER_NS * g.dt if dark else out
