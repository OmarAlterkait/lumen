"""Truth rows: the analog (Geant4-like) record of a run, one row per detected photon.

    light = lumen.simulate(det, src, p, samples=n, seed=0, truth=lumen.readout.Truth())
    hits = light.truth                                   # TruthHits: numpy columns, in photon order
    hits.label, hits.sensor, hits.time, hits.detected
    n = hits.sum_by(("label", "sensor"))                 # per (label, sensor): photons n and the first time

Each Monte Carlo photon carries one analog photon (one uniform of its own, ``lumen.run``): it is absorbed on a cathode
at most once, at the crossing where its uniform falls in the running sum of the reached shares (Geant4's law: one
cathode absorption per photon, ``lumen.transport.step``, "analog"), and then detected with its group's detection
probability (efficiency x acceptance of a modelled group, ``detection``, of the deposit fields its features
read, recorded at the absorption: the same tables the binned readout reads; 1 on a counting group, whose surface
already holds its efficiency), from a second uniform of the photon's own. A row's ``time`` is the arrival at the
cathode (ns, before the sensor's time spread, delay and offset), ``label`` the emission row's label (-1 without),
``sensor`` the detector's sensor id. Optional columns: ``angle`` (rad, the cathode position angle,
``lumen.sensors.features.theta``) and ``position`` ((n, 3) m, the
crossing point in the sensor placement's frame: its local axes, about the placement's origin). ``Truth(keep=
"absorbed")`` keeps every absorbed photon with its ``detected`` flag.

The rows are an output of the same photon loop as the expected values, at any roulette: the photon's analog fate is
kept by the roulette while it is open (``lumen.transport.step._analog_roulette``), so the record is exact and the
expected values are unchanged in law. A row stands for one physical photon: the source's rows must stand for at most
one photon each (``samples`` >= the photons emitted). Not differentiable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import rng
from lumen._core.rng import DETECT_STREAM
from lumen._core.static import Static
from lumen.sensors.features import FIELDS, polar_angle
from lumen.sources._base import bucket

KEEPS = ("detected", "absorbed")
COLUMNS = ("photon", "label", "sensor", "time", "detected")


@dataclass(frozen=True, kw_only=True)
class Truth:
    """What truth rows a run keeps: ``keep`` "detected" (default) or "absorbed" (every cathode absorption, with its
    ``detected`` flag); ``angle``, ``position``: the hit angle and the hit point columns."""  # noqa: D205, D209

    keep: str = "detected"
    angle: bool = False
    position: bool = False


@dataclass(frozen=True, kw_only=True)
class TruthHits:
    """The truth rows of a run (see the module docstring): numpy columns, one row per photon, in photon order.

    ``photon`` the photon's id in the run, ``label``, ``sensor``, ``time`` (ns), ``detected`` (bool); ``angle`` and
    ``position`` when asked for (else None). ``n_photons``: the photons run; ``groups``: the sensor groups' names and
    ``group_of`` (S,): each sensor's group (its index in ``groups``).
    """

    photon: np.ndarray
    label: np.ndarray
    sensor: np.ndarray
    time: np.ndarray
    detected: np.ndarray
    angle: np.ndarray | None
    position: np.ndarray | None
    n_photons: int
    groups: tuple
    group_of: np.ndarray

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.sensor)

    def select(self, mask) -> TruthHits:
        """Return the rows where ``mask`` is True (a boolean array over the rows)."""
        m = np.asarray(mask, bool)
        cut = lambda a: None if a is None else a[m]  # noqa: E731
        return dataclasses.replace(self, **{k: cut(getattr(self, k)) for k in (*COLUMNS, "angle", "position")})

    def sum_by(self, keys=("label", "sensor"), *, detected: bool = True) -> dict:
        """Return the rows grouped by integer columns: {each key: (G,), "n": photons (G,), "t_first": first time (G,)}.

        Groups are sorted by the keys in order; ``detected``: only the detected rows (default). One int64 key per row
        (mixed radix of the columns), one argsort and ``reduceat``.
        """
        hits = self.select(self.detected) if detected else self
        cols = [np.asarray(getattr(hits, k), np.int64) for k in keys]
        if not len(hits):
            return {k: np.zeros(0, np.int64) for k in keys} | {"n": np.zeros(0, np.int64), "t_first": np.zeros(0)}
        lo = [c.min() for c in cols]
        span = [int(c.max() - m) + 1 for c, m in zip(cols, lo, strict=True)]
        if np.prod(np.asarray(span, np.float64)) >= 2.0**62:
            raise ValueError(f"sum_by: the keys {keys} span too many values for one int64 key")
        key = np.zeros(len(hits), np.int64)
        for c, m, s in zip(cols, lo, span, strict=True):
            key = key * s + (c - m)
        order = np.argsort(key, kind="stable")
        k, t = key[order], np.asarray(hits.time, np.float64)[order]
        start = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        out = {kk: c[order][start] for kk, c in zip(keys, cols, strict=True)}
        return out | {"n": np.diff(np.r_[start, len(k)]), "t_first": np.minimum.reduceat(t, start)}


def tables(det) -> tuple:
    """Return (group names, group of each sensor (S,), position in its group (S,), [(group index, model, name)] of the
    modelled groups) of a detector: what the detection draw reads."""  # noqa: D205, D209
    from lumen.build import build
    from lumen.sensors import group_model

    n = len(det.parts.optics.sensor_pvs)
    names = tuple(g.name for g in det.sensor_set.groups)
    group_of, local = np.full(n, -1, np.int32), np.zeros(n, np.int32)
    modelled = []
    for gi, g in enumerate(det.sensor_set.groups):
        tags = build.group_tags(g, det.parts, det.sensor_set)
        group_of[tags], local[tags] = gi, np.arange(len(tags), dtype=np.int32)
        m = group_model(g)
        if m is not None:
            modelled.append((gi, m, g.name))
    return names, group_of, local, modelled


def detection_values(modelled, params) -> tuple:
    """Return each modelled group's split values as float32 arrays (the detection draw's operands)."""
    f32 = lambda v: {k: f32(x) for k, x in v.items()} if isinstance(v, dict) else jnp.asarray(v, jnp.float32)  # noqa: E731
    return tuple(f32(m.split(params, name)) for _, m, name in modelled)


def local_fields(det, truth, grid_reads) -> tuple:
    """Return the deposit fields a run carries (``Model.local``): what its grids' features read, and with truth rows
    what the modelled groups' acceptance reads and the asked columns (``angle``, ``position``: the point)."""  # noqa: D205, D209
    need = set(grid_reads)
    if truth is not None:
        need |= {f for _, m, _ in tables(det)[3] for f in m.detection_reads}
        need |= {"pos"} if (truth.angle or truth.position) else set()
    return tuple(sorted(need))


def _probability(modelled, values, group_of, local, rec):
    """Return each row's detection probability (1 on a counting group's sensors)."""
    s = jnp.clip(rec["sensor"], 0, group_of.shape[0] - 1)
    g, loc = group_of[s], local[s]
    p = jnp.ones(s.shape, jnp.float32)
    for (gi, m, _), v in zip(modelled, values, strict=True):
        p = jnp.where(g == gi, m.detection(v, loc, {k: x for k, x in rec.items() if k in ("lam", *FIELDS)}), p)
    return p * rec["scale"]  # the sensor type's 1 / A_n: the QE is per photon at normal incidence


def _keep_mask(modelled, values, group_of, local, rec, key, first, count, keep):  # noqa: PLR0913
    modelled = modelled()  # the groups' sensor models by their structure (``lumen._core.static``)
    n = rec["sensor"].shape[0]
    ids = first + jnp.arange(n, dtype=jnp.int32)
    absorbed = (rec["sensor"] >= 0) & (jnp.arange(n) < count)  # rows past the run's photons are not photons

    u = jax.vmap(lambda i: rng.uniforms(jax.random.fold_in(rng.photon_key(key, i), DETECT_STREAM), 1)[0])(ids)
    detected = absorbed & (u < _probability(modelled, values, group_of, local, rec))
    return (detected if keep == "detected" else absorbed), detected


_MASK = jax.jit(_keep_mask, static_argnums=(0, 8))


@jax.jit
def _count(mask):
    return jnp.sum(mask.astype(jnp.int32))


def _gather(cols, mask, m):
    idx = jnp.nonzero(mask, size=m, fill_value=0)[0]
    return idx, {k: v[idx] for k, v in cols.items()}


_GATHER = jax.jit(_gather, static_argnums=2)


def chunk_rows(spec: Truth, tabs, values, rec, labels, key, first: int, count: int) -> dict:  # noqa: PLR0913
    """Return one chunk's truth rows as numpy columns: its photons' analog records (``rec``, ``RunResult.per_item``)
    thinned by detection and compacted on the device, in photon order, the labels gathered from the rows."""  # noqa: D205, D209
    _, group_of, local, modelled = tabs
    mask, detected = _MASK(Static(tuple(modelled)), values, jnp.asarray(group_of), jnp.asarray(local), rec, key,
                           jnp.asarray(first, jnp.int32), jnp.asarray(count, jnp.int32), spec.keep)  # fmt: skip
    c = int(_count(mask))
    cols = dict(sensor=rec["sensor"], time=rec["time"], detected=detected)
    if labels is not None:
        cols["label"] = jnp.asarray(labels, jnp.int32)
    if spec.angle:
        cols["angle"] = jax.vmap(polar_angle)(rec["pos"])
    if spec.position:
        cols["position"] = rec["pos"]
    idx, got = _GATHER(cols, mask, bucket(max(c, 1)))
    out = {k: np.asarray(v)[:c] for k, v in got.items()}
    out["photon"] = first + np.asarray(idx)[:c].astype(np.int64)
    if labels is None:
        out["label"] = np.full(c, -1, np.int32)
    return out


def join(chunks: list, spec: Truth, n_photons: int, tabs) -> TruthHits:
    """Return the TruthHits of a run's chunks (``chunk_rows``)."""
    names, group_of, _, _ = tabs
    col = lambda k: np.concatenate([c[k] for c in chunks]) if chunks else np.zeros(0)  # noqa: E731
    pos = col("position").reshape(-1, 3) if spec.position else None
    return TruthHits(photon=col("photon").astype(np.int64), label=col("label").astype(np.int32),
                     sensor=col("sensor").astype(np.int32), time=col("time").astype(np.float32),
                     detected=col("detected").astype(bool),
                     angle=col("angle").astype(np.float32) if spec.angle else None, position=pos,
                     n_photons=int(n_photons), groups=names, group_of=group_of)  # fmt: skip


__all__ = ["KEEPS", "Truth", "TruthHits"]
