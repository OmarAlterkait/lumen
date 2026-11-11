"""Helpers the unit tests share that the package itself does not use: reports, checks and literal forms of lumen's
own pieces, kept out of src so that the package carries only what it runs."""

import numpy as np

from lumen.build import scene, solids
from lumen.rules import optics as rules
from lumen.rules.optics import CONST_KEYS, G4_10_KEYS, TABLE_KEYS, OpticsRefused, SensorEntry
from lumen.surfaces import polar
from lumen.surfaces import types as stypes

PARTITION_SAMPLES = 20000  # check_partition: points sampled per definition
T_FAR = 1e3  # m: oracle_first_visible's ray length
DN_FLOOR = 1e-3  # mismatch: |d.n| below this is taken as this (grazing crossings)


# ---------------------------------------------------------------------------------------------------------- optics
def key_report(materials: dict) -> list[str]:
    """Return report lines naming the Geant4-10 keys and the keys no Geant4 optical process reads, per material."""
    out = []
    for m, rec in sorted(materials.items()):
        legacy_keys = sorted(k for k in rec.properties if k in G4_10_KEYS)
        user = sorted(k for k in rec.properties if k not in TABLE_KEYS | CONST_KEYS | set(G4_10_KEYS))
        if legacy_keys:
            out.append(
                f"  {m!r}: Geant4-10 keys {legacy_keys}: inert user keys under Geant4 >= 11 (scintillation(...))"
            )
        if user:
            out.append(f"  {m!r}: user keys {user}: read by no Geant4 optical process")
    return out


def rt_sensor(s, eff=None) -> SensorEntry:
    """Return the sensor form of a polished dielectric_dielectric surface (DIELECTRIC_RT_SENSOR).

    Its absorbed share is detected with ``eff``: by default the surface's EFFICIENCY (Geant4's Detection status); a
    sensitive detector that counts every absorption (WCSim's) passes 1.
    """
    e = rules.sensor(s, eff=eff)
    if e.surface_type is not stypes.DIELECTRIC_RT_SENSOR:
        raise OpticsRefused(f"opticalsurface {s.name!r}: {e.surface_type.name}, not a polished dielectric_dielectric")
    return e


def jones(er, ei, wi, n, k, amplitudes):
    """Return the state after an outcome with amplitudes (a_s, a_p) into direction k (n the facing normal)."""
    s = polar.s_axis(wi, n, er)
    e_s, e_p = polar.components(er, ei, wi, s)
    return polar.recombine(e_s, e_p, amplitudes[0], amplitudes[1], s, k)


def safe_norm(v, axis=-1):
    """Return |v|, with derivative 0 (not NaN) at v = 0."""
    import jax.numpy as jnp

    from lumen._core import safe

    return safe.safe_sqrt(jnp.sum(v * v, axis=axis))


# -------------------------------------------------------------------------------------------------------- geometry
def is_single_convex(c: solids.Compiled) -> bool:
    """Return whether a compiled solid is one convex leaf (a convex quadric, or planes only)."""
    lf = c.leaves[0]
    return isinstance(c.expr, int) and lf.kind == "q" and (
        not lf.has_q or bool(np.all(np.linalg.eigvalsh(lf.A) >= -solids.CONVEX_TOL)))  # fmt: skip


def check_partition(sc, n=PARTITION_SAMPLES, seed=0):
    """Count, per definition, sampled points covered by two or more members.

    Returns
    -------
    dict
        definition name -> (ambiguous, ranked): points in two or more members of equal rank and different media
        (must be 0), and points resolved by rank (reported).
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, d in sc.definitions.items():
        mem = [m for m in d.members if not isinstance(m.solid, scene.Mesh)]
        out[name] = (0, 0) if len(mem) < 2 else _overlaps(mem, rng, n)
    return out


def _overlaps(mem, rng, n):
    compiled = [solids.Compiled(m.solid) for m in mem]
    bbs = [solids.node_bbox(m.solid) for m in mem]
    lo = np.min([b[0] for b in bbs], 0)
    hi = np.max([b[1] for b in bbs], 0)
    x = rng.uniform(lo, hi, (n, 3))
    ins = np.stack([c.inside_np(x) for c in compiled], 1)
    ranks = np.array([m.rank for m in mem])
    amb = ranked = 0
    for i in np.flatnonzero(ins.sum(1) >= 2):
        k = np.nonzero(ins[i])[0]
        top = ranks[k].max()
        amb += int(np.sum(ranks[k] == top) >= 2 and len({mem[j].label.medium for j in k if ranks[j] == top}) > 1)
        ranked += 1
    return amb, ranked


def oracle_first_visible(orc, origins, dirs, push, t_min=1e-7, medium=None):
    """Return the oracle's first visible change of each ray, as arrays (hit, t, after, sensor, before, amb, start)."""
    recs = [orc.query(origins[i], dirs[i], t_min, T_FAR, push=push, medium=None if medium is None else int(medium[i]))
            for i in range(len(origins))]  # fmt: skip
    return dict(hit=np.array([r["hit"] for r in recs]), t=np.array([r["t"] if r["hit"] else np.inf for r in recs]),
                after=np.array([r["after"][0] for r in recs]), sensor=np.array([r["after"][1] for r in recs]),
                before=np.array([r["before"][0] for r in recs]), amb=np.array([r["ambiguous"] for r in recs]),
                start=np.array([r["start"] for r in recs]))  # fmt: skip


def mismatch(e, o, t_tol=2e-5, push=0.0, early=None, ulps=8):
    """Return, per ray, whether an engine's first visible crossing disagrees with the oracle's.

    A crossing reported earlier than the oracle's by at most the push, with the same medium and sensor after, is the
    push contract consuming a layer of the starting medium: not a mismatch; ``early`` (a dict) receives its mask. The
    t tolerance is the larger of t_tol * max(1, t) and ``ulps`` float32 ulps of the hit's coordinates along the normal
    (|dt| |d.n|): a grazing crossing's t is only resolved to ulp / |d.n|.
    """
    bad = (e["hit"] != o["hit"]) | (e["before"] != o["before"])
    both = e["hit"] & o["hit"]
    tol = t_tol * np.maximum(1.0, np.where(np.isfinite(o["t"]), o["t"], 1.0))
    if "dn" in e:
        ulp = np.spacing(np.maximum(e["xabs"], 1.0).astype(np.float32)).astype(float)
        tol = np.maximum(tol, ulps * ulp / np.maximum(e["dn"], DN_FLOOR))
    dt = np.where(both, e["t"] - o["t"], 0.0)
    same = (e["after"] == o["after"]) & (e["sensor"] == o["sensor"])
    ok_t = np.abs(dt) <= tol
    early_ok = (dt < -tol) & (dt >= -(push + tol))
    bad |= both & ~(same & (ok_t | early_ok))
    if early is not None:
        early["mask"] = both & same & early_ok
    return bad
