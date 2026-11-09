"""The charge check: lumen's detected light per sensor against a reference simulation, source by source.

    rows = charge.check("sk")                                  # the manifest's "sk" entry: its detector and reference
    rows = charge.check("mydet", extra=["refs/mydet.json"])    # a detector's own manifest

The reference (WCSim or Geant4 through g4ref, reduced into the store) launched isotropic monochromatic photons at
t = 0 from each source of its config and recorded a hit per photon its sensors absorbed. lumen runs the same sources
with ``photons_factor`` times the reference's photons, in ``replicas`` independent sub-runs, and histograms per
sensor and wavelength band the detected weight, and the arrival times pooled over sensors on the reference's time
nodes. Sensors are matched by placement centre (``tol_mm``); placements at one centre count as one sensor.

Per source and band: the total lumen / reference with its error (the reference's Poisson error and lumen's spread
over the replicas) and z; the per-sensor Pearson chi2 / ndf (sensors expecting at least ``MU_MIN`` hits, variance
mu + the Monte Carlo variance); the worst group of sensors (type x distance quantile); the arrival times (mean,
width, KS). Over the whole check: the total ratio and its z. Pass rules: ``lumen.validate.stats``.

A manifest entry may scale parameters to the reference's own physics (``scale``: {key: factor}; ANNIE's WCSim frame
loss), apply a per-sensor detection probability the reference applied (``detection``: "module:function", called with
the detector and the photon energy in eV), split the bands at ``split_ev``, and name the sensor column that types the
sensors (``group_column``).
"""

from __future__ import annotations

import importlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from lumen import run as runner
from lumen.transport import item
from lumen.units import HC
from lumen.validate import photons, records, references, stats

DEFAULTS = dict(replicas=8, per_call=1 << 20, photons_factor=2.0, seed=20261107, tol_mm=5.0, distance_groups=6,
                dt=1.0, nt=700)  # fmt: skip
MU_MIN = 5.0  # the per-sensor chi2 takes sensors expecting at least this many hits
COINCIDENT_MM = 1e-3  # sensors whose centres are closer than this are one placement duplicated
F32, I32 = jnp.float32, jnp.int32


def read_reference(d) -> dict:
    """Return a reduced product: hits (src, sensor, t, E), the sensor table, photons launched, the source config."""
    with np.load(d / "hits.npz") as z:
        hits = {k: z[k] for k in z.files}
    rows = [r.split(",") for r in (d / "sensors.csv").read_text().splitlines()[1:]]
    cfg = np.loadtxt(d / "sources.cfg", comments="#", ndmin=2)
    centres = np.array([[float(v) for v in r[3:6]] for r in rows])
    return dict(hits=hits, lv=np.array([r[2] for r in rows]), centres=centres,
                launched=np.array(json.loads((d / "launched.json").read_text()), np.float64), x_mm=cfg[:, :3],
                n_cfg=cfg[:, 3], energy=cfg[:, 4])  # fmt: skip


def item_spec(tally: item.TallyShape, n_groups: int, nt: int) -> item.ItemSpec:
    """Return the default item plus arrival times on ``nt`` nodes per time group and band ("H"), late and early ("L").

    The node spacing ``dt`` and each sensor's time group enter through the grid operand (``grid["dt"]``,
    ``grid["group"]``), as the band edges do.
    """
    base = item.photons(tally)
    nb = len(tally.lam_edges) - 1

    def emits(dep, slot, grid):
        b = jnp.searchsorted(grid["lam_edges"], dep.lam, side="right") - 1
        ok = (dep.sensor >= 0) & (b >= 0) & (b < nb)
        cell = grid["group"][jnp.maximum(dep.sensor, 0)] * nb + b
        x = dep.time / grid["dt"]
        j = jnp.floor(x).astype(I32)
        g = x - j
        adds = list(base.emits(dep, slot, grid))
        for jj, w in ((j, 1.0 - g), (j + 1, g)):
            node = jnp.where(ok & (jj >= 0) & (jj < nt), cell * nt + jj, -1)
            edge = jnp.where(ok & (jj >= nt), cell * 2, jnp.where(ok & (jj < 0), cell * 2 + 1, -1))
            adds += [("H", node, dep.q * w), ("L", edge, dep.q * w)]
        return adds

    def outputs(replicas):
        return base.outputs(replicas) | {"H": (n_groups * nb * nt, True), "L": (n_groups * nb * 2, True)}

    return base._replace(emits=emits, outputs=outputs)


def run_source(s, run_args, x_m, lam, nph, key):
    """Return one source's lumen run: per replica, sensor and band the detected weight, times, health, photons."""
    spec, ops, k, nc = run_args
    acc, n_calls = None, max(1, int(round(nph / nc)))
    for i in range(n_calls):
        rows = photons.isotropic(jnp.tile(jnp.asarray(x_m, F32), (nc, 1)), jnp.full(nc, lam, F32),
                                 jax.random.fold_in(key, 1 << 20 | i))  # fmt: skip
        t = runner.run(spec, s.model, ops, rows, key, replicas=k, first_id=i * nc)
        out = {name: np.asarray(v, np.float64) for name, v in t.outputs.items()}
        acc = out if acc is None else {name: acc[name] + v for name, v in out.items()}
    acc["photons"] = n_calls * nc
    return acc


def match(ref, centres_mm, tol_mm):
    """Return each reference sensor's lumen sensor (by centre), and each lumen sensor's first coincident one."""
    can = np.arange(len(centres_mm))
    for a, b in sorted(cKDTree(centres_mm).query_pairs(COINCIDENT_MM)):
        can[max(a, b)] = min(can[a], can[b])
    d, m = cKDTree(centres_mm).query(ref["centres"])
    if d.max() > tol_mm:
        raise ValueError(f"sensor match: {int((d > tol_mm).sum())} reference sensors farther than {tol_mm} mm from "
                         f"any lumen sensor (largest {d.max():.1f} mm)")  # fmt: skip
    return can[m], can


def _groups(types, dist, n_groups):
    for ty in sorted(set(types)):
        idx = np.flatnonzero(types == ty)
        for g in np.array_split(idx[np.argsort(dist[idx])], min(n_groups, len(idx))):
            yield ty, g


def compare_band(r, b, n, p, ctx):
    """Return one source and band's comparison (``ctx``: the run's sensor folding, types, distances, settings)."""
    can, types, dist, nn = ctx["can"], ctx["types"], ctx["dist"], ctx["launched"]
    k = r["Q"].shape[0]
    qk = r["Q"][:, :, b] / (r["photons"] / k) * p  # (K, S): detected per emitted photon
    qk = np.stack([np.bincount(can, weights=v, minlength=len(can)) for v in qk])
    mu, var_mc = qk.mean(0) * nn, qk.var(0, ddof=1) / k * nn**2
    tot = qk.sum(1) * nn
    ratio = mu.sum() / max(n.sum(), 1.0)
    rel_mc = tot.std(ddof=1) / np.sqrt(k) / tot.mean() if tot.mean() > 0 else 0.0
    err = ratio * np.hypot(1.0 / np.sqrt(max(n.sum(), 1.0)), rel_mc)
    m = mu >= MU_MIN
    pull = np.where(m, (mu - n) / np.sqrt(np.where(mu + var_mc > 0, mu + var_mc, 1.0)), 0.0)
    worst = None
    for ty, g in _groups(types, dist, ctx["distance_groups"]):
        vg = mu[g].sum() + var_mc[g].sum()
        zg = (mu[g].sum() - n[g].sum()) / np.sqrt(vg) if vg > 0 else 0.0
        if worst is None or abs(zg) > abs(worst["z"]):
            worst = dict(type=str(ty), distance_m=[dist[g].min() / 1e3, dist[g].max() / 1e3], z=zg,
                         ratio=mu[g].sum() / max(n[g].sum(), 1.0), n_ref=int(n[g].sum()))  # fmt: skip
    hl = (r["H"][:, b] * ctx["p_group"][:, None]).sum(0) / r["photons"] * nn
    hr, late_r, _ = stats.node_split(ctx["t_ref"], ctx["dt"], hl.shape[-1])
    tm = stats.times(hl, hr, ctx["dt"], r["H"][:, b].sum())
    return dict(n_ref=n.sum(), mu=mu.sum(), ratio=ratio, err=err, z=(ratio - 1) / err if err > 0 else 0.0,
                mc_var=(rel_mc * mu.sum()) ** 2, chi2=float((pull[m] ** 2).sum()), ndf=int(m.sum()),
                n_pull3=int((np.abs(pull[m]) > 3).sum()), worst_group=worst, late_ref=late_r,
                worst_sensors=[(int(i), float(pull[i]), float(n[i]), float(mu[i])) for i in np.argsort(-abs(pull))[:3]],
                **tm)  # fmt: skip


def loop_order(det, values) -> np.ndarray:
    """Return per-sensor values given in sensor-id order (the id map) in the photon loop's sensor order."""
    where = {p: i for i, p in enumerate(det.sensors["ids"])}
    return np.asarray(values)[[where[p] for p in det.parts.optics.tag_pvs]]


def _detection(spec, det, energy):
    if "detection" not in spec:
        return None
    mod, fn = spec["detection"].split(":")
    return loop_order(det, getattr(importlib.import_module(mod), fn)(det, float(energy)))


def _params(det, spec):
    p = det.params()
    for key, factor in spec.get("scale", {}).items():
        p = p.scale(key, factor)
    return photons.packed(det, p)


def _types(det, s, spec, ref_lv):
    col = spec.get("group_column")
    if col is None:
        return ref_lv.astype(str), np.zeros(s.n_sensors, np.int32)
    types = loop_order(det, det.sensors[col]).astype(str)
    return types, np.unique(types, return_inverse=True)[1].astype(np.int32)


def check(name, *, detector=None, extra=(), **options) -> list:
    """Run the charge check named ``name`` (a manifest entry); return its record rows (``lumen.validate.records``)."""
    import lumen

    spec = references.spec("charge", name, extra)
    o = (
        DEFAULTS
        | {k: v for k, v in spec.items() if k in DEFAULTS}
        | {k: v for k, v in options.items() if v is not None}
    )
    t0 = time.time()
    ref = read_reference(references.product(spec["reference"], spec, extra))
    det = lumen.load(detector or spec["detector"])
    lo, hi = det.file.wavelength_range
    split = spec.get("split_ev")
    s = photons.setup(det, (lo, HC / split, hi) if split else (lo, hi))
    nb = len(s.tally.lam_edges) - 1
    centres = photons.sensor_centres(s) * 1e3
    lumen_of_ref, can = match(ref, centres, o["tol_mm"])
    types_ref = np.full(s.n_sensors, "", object)
    types_ref[lumen_of_ref] = ref["lv"]
    types, group = _types(det, s, spec, types_ref)
    n_groups = int(group.max()) + 1
    grid = item.grid(s.tally) | {"dt": jnp.asarray(o["dt"], F32), "group": jnp.asarray(group)}
    ops = photons.operands(s, _params(det, spec))._replace(grid=grid)
    run_args = (item_spec(s.tally, n_groups, int(o["nt"])), ops, int(o["replicas"]), int(o["per_call"]))
    t_build = time.time() - t0
    out, health = [], dict(truncated=0.0, n_truncated=0.0, lost=0.0, abandoned=0.0, escaped=0.0, photons=0)
    for i, (x, n_cfg, energy) in enumerate(zip(ref["x_mm"], ref["n_cfg"], ref["energy"], strict=True)):
        t1 = time.time()
        r = run_source(s, run_args, x * 1e-3, HC / energy, o["photons_factor"] * n_cfg, photons.raw_key(o["seed"], i))
        r["Q"] = r["Q"].reshape(int(o["replicas"]), s.n_sensors, nb)
        r["H"] = r["H"].reshape(n_groups, nb, -1)
        for k in health:
            health[k] += float(r[k].sum()) if k != "photons" else r["photons"]
        p = _detection(spec, det, energy)
        p = np.ones(s.n_sensors) if p is None else p
        p_group = np.array([p[group == g].mean() for g in range(n_groups)])
        h = ref["hits"]
        mine = h["src"] == i
        band = (np.where(h["E"][mine] > split, 0, 1) if split else np.zeros(int(mine.sum()), int)).astype(int)
        for b in range(nb):
            sel = lumen_of_ref[h["sensor"][mine][band == b]]
            n = np.bincount(sel, minlength=s.n_sensors).astype(np.float64)
            ctx = dict(can=can, types=types, dist=np.linalg.norm(centres - x, axis=1), launched=ref["launched"][i],
                       distance_groups=o["distance_groups"], p_group=p_group, t_ref=h["t"][mine][band == b],
                       dt=float(o["dt"]))  # fmt: skip
            out.append(dict(src=i, band=b, seconds=time.time() - t1, **compare_band(r, b, n, p, ctx)))
    return _rows(name, spec, out, health, dict(build_s=t_build, total_s=time.time() - t0, **o))


def _rows(name, spec, out, health, run):
    ref = spec["reference"]
    rows = []
    for r in out:
        tag = f"source {r['src']}" + (f" band {r['band']}" if any(x["band"] for x in out) else "")
        rows.append(records.row("charge", name, f"ratio {tag}", r["ratio"], r["err"], z=r["z"],
                                passed=abs(r["z"]) <= stats.Z_MAX, reference=ref, **{k: v for k, v in r.items()
                                if k not in ("ratio", "err", "z")}))  # fmt: skip
        c = r["chi2"] / max(r["ndf"], 1)
        rows.append(records.row("charge", name, f"chi2/ndf {tag}", c, passed=stats.chi2_passes(c, r["ndf"]),
                                reference=ref, ndf=r["ndf"]))  # fmt: skip
        rows.append(records.row("charge", name, f"mean time lumen - reference {tag} (ns)", r["dt"], r["dt_err"],
                                reference=ref, ks_p=r["ks_p"]))  # fmt: skip
    nr, mu = sum(r["n_ref"] for r in out), sum(r["mu"] for r in out)
    mc = sum(r["mc_var"] for r in out)
    ratio = mu / nr
    err = ratio * np.sqrt(1.0 / nr + mc / mu**2)
    rows.append(records.row("charge", name, "ratio", ratio, err, z=(ratio - 1) / err, passed=abs(ratio - 1) <= 3 * err,
                            reference=ref, mc_error=ratio * np.sqrt(mc) / mu, n_ref=nr, mu=mu, run=run))  # fmt: skip
    rows.append(records.row("charge", name, "truncated photons", health["n_truncated"],
                            passed=health["n_truncated"] == 0, reference=ref, **health))  # fmt: skip
    return rows
