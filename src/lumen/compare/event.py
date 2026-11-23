"""One Geant4 event through lumen: its charged steps as light sources, its hits as the reference.

    ev = event.extract(run_dir, "mu_ev0.npz", chunk=0, event=0)     # once: one event of a g4ref gun run's chunk files
    ev = event.read("mu_ev0.npz")                                   # steps (m, ns, eV), hits, sensors
    light = event.simulate(det, ev, samples=1 << 18, seed=0)        # every light of every material, summed
    res = event.compare(det, ev, samples=1 << 18, seed=0)           # per sensor: lumen and Geant4 counts; times
    many = event.compare_events(det, paths, samples=1 << 16, seed=0)  # many events: their pulls, pooled

Each material and light (Cherenkov, scintillation) with Geant4 photons in the event is one step source (the
true step length) and one run; materials the detector has no optics for are left out (``Event.outside`` counts their
Geant4 photons). A Geant4 hit is a photon its sensor absorbed: lumen's count of a deposit is the charge check's
(``charge.response`` "absorbed": the deposit is the physical absorbed share, counted as it is; only a sensor model's
readout divides by a cathode's A_n). The comparison's statistics are the charge check's (``charge.compare_band``).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import lumen
from lumen import sources
from lumen.compare import charge, photons
from lumen.tools import stats
from lumen.units import EV_PER_MEV

LIGHTS = ("cherenkov", "scintillation")
MM = 1e-3  # m per mm: g4ref writes mm, MeV
MU_MIN = charge.PEARSON_MIN  # a sensor's expected hits from which its Poisson pull is near Gaussian
STEP_FIELDS = ("start", "end", "t_start", "t_end", "beta_start", "beta_end", "charge", "edep", "step_length")


@dataclass(frozen=True)
class Event:
    """One Geant4 event: its steps, the materials they are in, its hits and the reference's sensors.

    ``steps``: {start, end (n, 3) m; t_start, t_end ns; beta_start, beta_end; charge e; edep eV; step_length m;
    material (index into ``materials``); pdg; photons {"cherenkov", "scintillation"}: Geant4's photons per step}.
    ``hits``: {sensor (index into ``sensors``), t ns, energy eV, origin (2 Cherenkov, 3 scintillation)}.
    ``sensors``: {centre (S, 3) m, name}. ``meta``: what made it (the GDML, seed, particle, ...).
    """

    steps: dict
    materials: tuple
    hits: dict
    sensors: dict
    meta: dict

    def outside(self, params) -> dict:
        """Return {(light, material): Geant4 photons} of the lights lumen has no source for in ``params``."""
        return {k: n for k, (m, n) in self._parts(params, LIGHTS, keep=False).items()}

    def sources(self, params, lights=LIGHTS) -> dict:
        """Return {(light, material): its step source} of every light in ``lights`` the event has photons of."""
        out = {}
        for (light, mat), (sel, _) in self._parts(params, lights).items():
            kw = {k: np.asarray(self.steps[k][sel], np.float32) for k in ("start", "end", "t_start", "t_end",
                                                                           "step_length")}  # fmt: skip
            if light == "cherenkov":
                kw |= {k: np.asarray(self.steps[k][sel], np.float32) for k in ("beta_start", "beta_end", "charge")}
                out[(light, mat)] = sources.CherenkovSteps(mat, **kw)
            else:  # the file's edep is in eV
                edep = np.asarray(self.steps["edep"][sel], np.float64) / EV_PER_MEV
                out[(light, mat)] = sources.ScintillationSteps.of(params, mat, edep=edep.astype(np.float32), **kw)
        return out

    def _parts(self, params, lights, keep=True):
        names = {k.split("/")[1] for k in params if k.startswith("materials/") and k.endswith("/RINDEX")}
        bare = {n.split("0x")[0]: n for n in names}  # a GDML's 0x suffixes: lumen keeps them, Geant4 strips them
        out = {}
        for i, m in enumerate(self.materials):
            sel = self.steps["material"] == i
            mat = m if m in names else bare.get(m, m)
            for light in lights:
                n = int(self.steps["photons"][light][sel].sum())
                if n == 0:
                    continue
                ok = mat in names and (light == "cherenkov" or f"materials/{mat}/SCINTILLATIONYIELD" in params)
                if ok == keep:
                    out[(light, mat)] = (sel, n)
        return out


def extract(run_dir, out, *, chunk: int = 0, event: int = 0, meta: dict | None = None) -> Event:
    """Write one event of a g4ref gun run (``chunk<k>_{steps,hits}.npy``, ``_sensors.csv``, ``_materials.csv``) to
    ``out`` (npz, lumen's units) and return it; ``meta`` joins the run's summary (``chunk<k>_summary.txt``)."""  # noqa: D205, D209
    d, c = Path(run_dir), f"chunk{chunk}"
    st = np.load(d / f"{c}_steps.npy", mmap_mode="r")
    st = np.asarray(st[np.asarray(st["evt"]) == event])
    h = np.load(d / f"{c}_hits.npy", mmap_mode="r")
    h = np.asarray(h[np.asarray(h["evt"]) == event])
    rows = [r.split(",") for r in (d / f"{c}_sensors.csv").read_text().splitlines()[1:]]
    mats = [r.split(",")[1] for r in (d / f"{c}_materials.csv").read_text().splitlines()[1:]]
    summary = dict(r.split(" ", 1) for r in (d / f"{c}_summary.txt").read_text().splitlines()[:12] if " " in r)
    info = {k: summary[k] for k in ("geant4", "gdml", "seed", "sens_regex") if k in summary}
    info |= dict(chunk=chunk, event=event) | (meta or {})
    info["gdml"] = Path(info.get("gdml", "")).name
    xyz = lambda s: np.stack([st[f"x{s}"], st[f"y{s}"], st[f"z{s}"]], 1) * MM  # noqa: E731
    arrays = dict(start=xyz(0), end=xyz(1), t_start=st["t0"], t_end=st["t1"], beta_start=st["b0"],
                  beta_end=st["b1"], charge=st["q"], edep=st["edep"].astype(np.float64) * 1e6,
                  step_length=st["len"] * MM, material=st["mat"], pdg=st["pdg"], n_cherenkov=st["nc"],
                  n_scintillation=st["ns"], hit_sensor=h["sensor"], hit_t=h["t"], hit_energy=h["E"],
                  hit_origin=h["origin"], sensor_centre=np.array([[float(v) for v in r[3:6]] for r in rows]) * MM,
                  sensor_name=np.array([r[1] for r in rows]), materials=np.array(mats),
                  meta=json.dumps(info))  # fmt: skip
    np.savez_compressed(out, **arrays)
    return read(out)


def read(path) -> Event:
    """Return the Event written by ``extract``."""
    with np.load(path) as z:
        a = {k: z[k] for k in z.files}
    steps = {k: a[k] for k in (*STEP_FIELDS, "material", "pdg")}
    steps["photons"] = {"cherenkov": a["n_cherenkov"], "scintillation": a["n_scintillation"]}
    hits = dict(sensor=a["hit_sensor"], t=a["hit_t"], energy=a["hit_energy"], origin=a["hit_origin"])
    return Event(steps, tuple(str(m) for m in a["materials"]), hits,
                 dict(centre=a["sensor_centre"], name=a["sensor_name"]), json.loads(str(a["meta"])))  # fmt: skip


def simulate(det, ev: Event, params=None, *, lights=LIGHTS, samples: int, seed: int, **kw):
    """Return the Light of the event's steps (``lumen.simulate`` of each source, summed); ``kw``: ``simulate``'s."""
    p = det.params() if params is None else params
    runs = [lumen.simulate(det, s, p, samples=samples, seed=seed + 7919 * i, **kw)
            for i, s in enumerate(ev.sources(p, lights).values())]  # fmt: skip
    if not runs:
        raise ValueError(f"event: no light of {lights} lumen can emit (outside: {ev.outside(p)})")
    return add(*runs)


def add(*lights):
    """Return the sum of independent Lights of one detector (arrivals, histograms, health; errors in quadrature)."""
    first = lights[0]
    total = sum(x.arrivals for x in lights)
    tsum = sum(jnp.nan_to_num(x.mean_arrival_time) * x.arrivals for x in lights)
    plus = lambda *a: sum(a)  # noqa: E731
    groups = None
    if first.groups is not None:  # the cells and their moments add; the pe errors in quadrature
        groups = {k: dataclasses.replace(jax.tree_util.tree_map(plus, *(x.groups[k] for x in lights)),
                                         pe_error=jnp.sqrt(sum(x.groups[k].pe_error ** 2 for x in lights)))
                  for k in first.groups}  # fmt: skip
    return type(first)(arrivals=total, arrivals_error=jnp.sqrt(sum(x.arrivals_error**2 for x in lights)),
                       mean_arrival_time=jnp.where(total > 0, tsum / jnp.where(total > 0, total, 1.0), jnp.nan),
                       health=jax.tree_util.tree_map(plus, *(x.health for x in lights)), provenance=first.provenance,
                       groups=groups)  # fmt: skip


def compare(det, ev: Event, params=None, *, lights=LIGHTS, samples: int, seed: int,  # noqa: PLR0913
            dt: float = 1.0, nt: int = 700, tol_mm: float = 5.0, backend: str = "auto") -> dict:  # fmt: skip
    """Return lumen against the event's Geant4 hits: per sensor (the photon loop's order) "lumen" (expected
    hits) and "geant4" (hits), the pooled arrival times on ``nt`` nodes of ``dt`` ns ("t", "times_lumen",
    "times_geant4"), "light" (the summed Light) and the charge check's statistics ("summary": ratio, err, z, chi2,
    ndf, null, times). One exact run per source of ``samples`` photons (``lumen.simulate`` with the diagnostic, on
    ``backend``)."""  # noqa: D205, D209
    p = det.params() if params is None else params
    s = photons.setup(det)
    k = charge.response(dict(name="event"), det)  # the reference's count of a deposit: the charge check's rule
    diag = photons.Detected(s.n_sensors, s.lam_edges, (0,) * s.n_sensors, nt, dt)
    q, q2, qv, h, runs = 0.0, 0.0, 0.0, 0.0, []
    for i, src in enumerate(ev.sources(p, lights).values()):
        light = lumen.simulate(det, src, p, samples=samples, chunk=samples, seed=seed + 7919 * i, backend=backend,
                               speed="exact", diagnostics=(diag,))  # fmt: skip
        r = photons.detected(light, diag)
        # per sensor the count, its moment and variance (the sources are independent runs: their sums and variances add)
        q, q2, qv = q + r["Q"][:, 0] * k, q2 + r["Q2"][:, 0] * k**2, qv + r["Q_var"][:, 0] * k**2
        h, runs = h + r["H"][0, 0], runs + [light]
    if not runs:
        raise ValueError(f"event: no light of {lights} lumen can emit (outside: {ev.outside(p)})")
    centres = photons.sensor_centres(s) * 1e3
    lumen_of_ref, can = charge.match(dict(centres=ev.sensors["centre"] * 1e3), centres, tol_mm)
    origin = {"cherenkov": 2, "scintillation": 3}
    mine = np.isin(ev.hits["origin"], [origin[x] for x in lights])
    n = np.bincount(lumen_of_ref[ev.hits["sensor"][mine]], minlength=s.n_sensors).astype(np.float64)
    tot = float(np.sum(qv))  # the total's variance: the sensors' (a photon's deposits in two sensors not correlated)
    r = dict(charge=q[:, None], Q2=q2[:, None], charge_var=qv[:, None], total_var=np.array([tot]), H=h[None, None],
             photons=1.0)  # the run's light as is: photons 1  # fmt: skip
    first = ev.steps["start"][0] * 1e3
    # one event in one band: one test (charge.check counts its sources x bands)
    ctx = dict(can=can, types=np.zeros(s.n_sensors, int), dist=np.linalg.norm(centres - first, axis=1), launched=1.0,
               distance_groups=6, p_group=np.ones(1), t_ref=ev.hits["t"][mine], dt=dt, tests=1)  # fmt: skip
    summary = charge.compare_band(r, 0, n, np.ones(s.n_sensors), ctx)
    mu = np.bincount(can, weights=q, minlength=s.n_sensors)
    times_g4 = stats.node_split(ev.hits["t"][mine], dt, nt)[0]
    return dict(lumen=mu, lumen_var=np.bincount(can, weights=qv, minlength=s.n_sensors), geant4=n, t=dt * np.arange(nt),
                times_lumen=h, times_geant4=times_g4, light=add(*runs), summary=summary)  # fmt: skip


def compare_events(det, events, params=None, *, lights=LIGHTS, samples: int, seed: int,  # noqa: PLR0913
                   mu_min: float = MU_MIN, toys: int = 1, groups=None, **kw) -> dict:  # fmt: skip
    """Return ``compare`` over many Geant4 events of one detector, and their pooled statistics.

    Each event runs lumen twice, ``samples // 2`` photons per source each, with independent seeds: half A chooses
    which sensors enter a statistic and sets its variance, half B is what Geant4 is compared with, so lumen's own
    noise cannot select its sensors nor correlate a residual with its variance (one run doing both biases the mean
    pull: from lumen's noise, and from Poisson skew where the counts' mean is used). ``events``: Events, or paths
    ``read`` takes. ``kw``: ``compare``'s (``dt``, ``nt``, ``backend``, ...).

    Returns, per event (rows, in order): "lumen" (A and B's mean) and "lumen_var" (its Monte Carlo variance),
    "geant4", "summary" (``compare``'s, half B); pooled over the events: "t", "times_lumen", "times_geant4";
    "pull": (Geant4 - B) / sqrt(A + var B) of every sensor and event where A >= ``mu_min``, with "pull_mu" (A)
    and "pull_toy" (the same of Poisson toys about lumen's mean, ``toys`` per sensor and event: what an exact match
    gives, skewed where counts are few); "bins": Geant4 against lumen summed in bins of the expected hits (bins on A,
    measured with B, and the other way round, averaged: no regression to the mean), each its relative difference,
    error and z; "sensor_z" (each sensor over the events) and "event_z" (each event's total): (Geant4 - B) /
    sqrt(A + var B), N(0, 1) if lumen is right; "groups": the same per group of sensors (``groups``, {name: mask};
    by default ``default_groups``: each surface of the detector's display layout and each sensor type), each its
    total's ratio and z and its per-event pulls' mean and spread: an effect common to a group's sensors, which no
    single sensor shows, shows there (a spread above 1 where the sensors' is 1).
    """
    keep = ("lumen", "lumen_var", "geant4", "t", "times_lumen", "times_geant4")
    rows, summaries = [], []
    for i, e in enumerate(events):  # each event's arrays on the host: the device holds one event at a time
        ev = e if isinstance(e, Event) else read(e)
        half = []
        for j in (0, 1):
            r = compare(det, ev, params, lights=lights, samples=max(samples // 2, 1), seed=seed + 104729 * i + 7 * j,
                        **kw)  # fmt: skip
            half.append({k: np.asarray(r[k]) for k in keep})
            if j == 1:
                summaries.append(r["summary"])
            del r
        rows.append(half)
    if not rows:
        raise ValueError("compare_events: no events")
    a, va, b, vb, n = (np.stack([r[h][k] for r in rows]) for h, k in ((0, "lumen"), (0, "lumen_var"),
                                                                       (1, "lumen"), (1, "lumen_var"),
                                                                       (1, "geant4")))  # fmt: skip
    out = statistics(a, va, b, vb, n, mu_min=mu_min, seed=seed, toys=toys,
                     groups=default_groups(det) if groups is None else groups)  # fmt: skip
    return out | dict(summary=summaries, t=rows[0][1]["t"],
                      times_lumen=sum(0.5 * (r[0]["times_lumen"] + r[1]["times_lumen"]) for r in rows),
                      times_geant4=sum(r[1]["times_geant4"] for r in rows))  # fmt: skip


def statistics(a, va, b, vb, n, *, mu_min: float = MU_MIN, seed: int = 0, toys: int = 1,  # noqa: PLR0913
               groups=None) -> dict:  # fmt: skip
    """Return ``compare_events``' statistics from two independent lumen halves ``a``, ``b`` (events, sensors), their
    variances and Geant4's counts ``n``: half A selects and sets the variance, half B is compared."""  # noqa: D205, D209
    a, va, b, vb, n = (np.asarray(x, np.float64) for x in (a, va, b, vb, n))
    mu, var = 0.5 * (a + b), 0.25 * (va + vb)
    sig = np.maximum(a, 0.0) + vb  # Geant4 - B: Poisson about lumen (estimated by A) and B's own noise
    m = a >= mu_min
    pull = (n[m] - b[m]) / np.sqrt(sig[m])
    rng = np.random.default_rng(seed)
    bm = np.maximum(b[m], 0.0)  # an exact match: Poisson counts about lumen, less lumen's noise of the same size
    toy = np.concatenate([(rng.poisson(bm) - bm + rng.normal(0.0, np.sqrt(vb[m]))) / np.sqrt(sig[m])
                          for _ in range(max(toys, 1))])  # fmt: skip
    s_sig = np.where(m, sig, 0.0).sum(0)
    lit = m.sum(0) > 0
    sensor_z = np.full(n.shape[1], np.nan)
    sensor_z[lit] = np.where(m, n - b, 0.0).sum(0)[lit] / np.sqrt(s_sig[lit])  # sensors some event lights enough
    event_z = (n.sum(1) - b.sum(1)) / np.sqrt(sig.sum(1))
    return dict(lumen=mu, lumen_var=var, geant4=n, pull=pull, pull_mu=a[m], pull_toy=toy, bins=_bins(a, va, b, vb, n),
                sensor_z=sensor_z, event_z=event_z, mu_min=mu_min,
                groups=_groups(b, vb, sig, n, {"all": np.ones(n.shape[1], bool)} | dict(groups or {})))  # fmt: skip


def _groups(b, vb, sig, n, groups) -> list:
    """Each group's total (Geant4 / B, its error and z) and its events' pulls (mean, spread, and their errors)."""
    out = []
    for name, mask in groups.items():
        m = np.asarray(mask, bool)
        if not m.any():
            continue
        tb, tn, ts = b[:, m].sum(1), n[:, m].sum(1), sig[:, m].sum(1)
        z = (tn - tb) / np.sqrt(ts)
        k = len(z)
        out.append(dict(name=str(name), sensors=int(m.sum()), ratio=float(tn.sum() / tb.sum()),
                        err=float(np.sqrt(ts.sum()) / tb.sum()), z=float((tn.sum() - tb.sum()) / np.sqrt(ts.sum())),
                        event_mean=float(z.mean()), event_sd=float(z.std(ddof=1)) if k > 1 else float("nan"),
                        event_sd_err=float(1 / np.sqrt(2 * (k - 1))) if k > 1 else float("nan")))  # fmt: skip
    return out


def default_groups(det) -> dict:
    """Return the groups of sensors ``compare_events`` reports by default: each surface of the detector's display
    layout (a face, the barrel, a cap) and each sensor type (the id map's "type" column, where there are several)."""  # noqa: D205, D209
    from lumen.tools import layout

    out = {}
    n = len(det.sensor_table().position)
    lay = layout.of(det)
    for name, idx in lay.parts:
        if len(lay.parts) > 1:
            m = np.zeros(n, bool)
            m[lay.index[idx]] = True
            out[name] = m
    types = det.sensor_table().columns.get("type")
    if types is not None and len(set(types)) > 1:
        t = np.asarray(types)
        out |= {f"type {u}": t == u for u in sorted(set(types))}
    return out


BIN_EDGES = (0.5, 2.0, 5.0, 20.0, 100.0, 1000.0, np.inf)  # expected hits per sensor and event


def _bins(a, va, b, vb, n, edges=BIN_EDGES):
    """Geant4 against lumen summed per bin of the expected hits: binned on one half, measured with the other."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        d, w, cnt = [], [], 0
        for sel, ref, vref in ((a, b, vb), (b, a, va)):
            m = (sel >= lo) & (sel < hi)
            if m.sum() == 0:
                continue
            e = ref[m].sum()
            d.append((n[m].sum() - e) / e)
            w.append(np.sqrt(e + vref[m].sum()) / e)
            cnt += int(m.sum())
        if d:
            err = float(np.mean(w))  # the two halves' estimates share Geant4's counts: their mean, not combined
            diff = float(np.mean(d))
            out.append(dict(lo=lo, hi=hi, count=cnt // len(d), diff=diff, err=err, z=diff / err))
    return out
