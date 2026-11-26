"""The charge check: lumen's detected light per sensor against a reference simulation, source by source.

    rows = charge.check("sk")                                  # the manifest's "sk" entry: its detector and reference
    rows = charge.check("mydet", extra=["refs/mydet.json"])    # a detector's own manifest

The reference (WCSim or Geant4 through g4ref, reduced into the store) launched isotropic monochromatic photons at
t = 0 from each source of its config and recorded a hit per photon its sensors absorbed. lumen runs the same sources
through ``lumen.simulate`` at its default controls, with ``photons_factor`` times the reference's photons, and
histograms per sensor and wavelength band the detected weight with its per-photon second moment (lumen's Monte Carlo
variance, ``photons.Detected``), and the arrival times pooled over sensors on the reference's time nodes. Sensors are
matched by placement centre (``tol_mm``); placements at one centre count as one sensor.

Per source and band: the total lumen / reference with its error and z (``stats.ratio_test`` at ALPHA over the check's
sources and bands, the per-sensor test's look-elsewhere: the two samples' mean as Poisson variance plus lumen's variance
of the total, from the total's own moment; a reference count too small to reject at the level has no power and is
reported; lumen detecting nothing has a score, never z = 0); the per-sensor agreement (``stats.reference_test``, the
comparisons' one test: each sensor's difference over the two samples' mean as Poisson variance plus lumen's variance,
sensors with too few hits for any difference to show at the level left out, the largest score against the sensors'
count, at ALPHA over the check's per-sensor tests; the Pearson chi2 / ndf kept as information); the worst group of
sensors (type x distance
quantile); the arrival times (mean, width, KS). Over the whole check: the total ratio and its z, no truncated weight,
and the lost and abandoned weight within the total's Monte Carlo error (gated on every entry). Pass rules:
``lumen.tools.stats``.

An entry's rows gate by one rule (``references.gated``): Geant4 through g4ref on the overlap-free export of the
detector's geometry (``lumen.compare.ownership``) gates; another simulation of the detector gates only where it has no
such reference, else it is a reported cross-check. When the reference is such an export run, the check also reads the
run of the same sources on the original GDML (``references.original``) and reports, never gates, lumen / G4(original)
and G4(export) / G4(original): the overlap effect, which Geant4 resolves along each photon's path where volumes
really overlap (MicroBooNE's wire planes and volTPCActive). A reference hit is a photon its sensor absorbed
(``response`` "absorbed", the default): lumen's deposit is the physical absorbed share, so the check counts it as it is.
With ``response`` "qe" the reference also kept each absorbed photon with its sensor's QE (ANNIE's WCSim): lumen's count
is times its sensor group's detection at normal incidence, from the detector's value set (the sensor model's
``detection`` at ``normal_rows``). An
entry may split the bands at ``split_ev``, and name the sensor column that types the sensors (``group_column``).
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.spatial import cKDTree

import lumen
from lumen import sources
from lumen.build import build
from lumen.compare import photons, records, references
from lumen.sensors import group_model
from lumen.tools import stats
from lumen.units import HC

DEFAULTS = dict(per_call=1 << 20, photons_factor=2.0, seed=20261107, tol_mm=5.0, distance_groups=6,
                dt=1.0, nt=700)  # fmt: skip
COINCIDENT_MM = 1e-3  # sensors whose centres are closer than this are one placement duplicated


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


def run_source(det, diag, source, per_call, seed):
    """Return one source's lumen run (``lumen.simulate`` in calls of ``per_call`` photons, the default controls, as
    production runs): per sensor and band the detected weight and its variance, the totals', the times, the health and
    the photons (independent calls: their sums and their variances add). ``source``: (position m, wavelength nm,
    photons)."""  # noqa: D205, D209
    x_m, lam, nph = source
    acc, n_calls = None, max(1, int(round(nph / per_call)))
    for i in range(n_calls):
        src = sources.Point(x_m, n_photons=float(per_call), lam=lam)
        light = lumen.simulate(det, src, samples=per_call, chunk=per_call, seed=seed + i, speed="exact",
                               diagnostics=(diag,))  # fmt: skip
        out = photons.detected(light, diag) | {k: np.asarray(v, np.float64) for k, v in light.health._asdict().items()}
        acc = out if acc is None else {name: acc[name] + v for name, v in out.items()}
    acc["photons"] = n_calls * per_call
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
    per = nn / r["photons"]  # lumen's detected weight to the reference's photons
    fold = lambda v: np.bincount(can, weights=v, minlength=len(can))  # noqa: E731 -- coincident placements as one
    mu, var_mc = fold(r["charge"][:, b] * p) * per, fold(r["charge_var"][:, b] * p * p) * per**2  # (S,)
    # the sensors' variances are each its own photons' (a photon's deposits in two sensors, or in two coincident
    # placements, are not correlated here); the total's is the photons' moment of the total (its own cell)
    mc_var = max(float(r["total_var"][b]), 0.0) * per**2  # absolute (0 where lumen detects nothing)
    total = stats.ratio_test(mu.sum(), mc_var, n.sum(), level=stats.ALPHA / ctx["tests"])  # as the per-sensor test
    # the reference's Poisson variance from both samples' mean (from lumen's alone it would correlate with the
    # numerator), plus lumen's variance (from its photons' moments: the terms' null is Pearson's 1 and 2)
    lam = 0.5 * (mu + n)
    q2 = fold(r["Q2"][:, b] * p * p)  # lumen's effective photons per sensor (Kish)
    kish = np.where(q2 > 0, fold(r["charge"][:, b] * p) ** 2 / np.where(q2 > 0, q2, 1.0), 0.0)
    agree = stats.reference_test(mu, np.maximum(var_mc, 0.0), kish, n, lam, n, tests=ctx["tests"])
    m = agree["power"]
    pull = np.where(m, (mu - n) / np.sqrt(np.where(lam + var_mc > 0, lam + var_mc, 1.0)), 0.0)
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
    return dict(n_ref=n.sum(), mu=mu.sum(), ratio=total["ratio"], err=total["err"], z=total["z"],
                ratio_passed=total["passed"], mc_var=mc_var, chi2=float((pull[m] ** 2).sum()), ndf=int(m.sum()),
                null=1.0, null_var=2.0,
                n_pull3=int((np.abs(pull[m]) > 3).sum()), worst_group=worst, late_ref=late_r,
                agree=dict(stat=agree["stat"], p=agree["p"], level=agree["level"], n=agree["n"],
                           passed=agree["passed"]),
                worst_sensors=[(int(i), float(pull[i]), float(n[i]), float(mu[i])) for i in np.argsort(-abs(pull))[:3]],
                **tm)  # fmt: skip


PEARSON_MIN = 5.0  # Geant4 against Geant4: the per-sensor chi2 (information only) takes sensors with this many hits


def geant4_ratio(a, b, launched_a, launched_b) -> dict:
    """Return one Geant4 run against another of the same sources (``a`` and ``b``: hits per sensor, photons launched
    each): the total ratio, photons normalised, with its Poisson error and z, and the per-sensor Pearson chi2 / ndf
    (sensors with at least ``PEARSON_MIN`` hits in both, normalised; variance a + k^2 b, k = launched_a /
    launched_b)."""  # noqa: D205, D209
    k = float(launched_a) / float(launched_b)
    ta, tb = float(a.sum()), float(b.sum())
    ratio = ta / (k * tb) if tb > 0 else float("nan")
    err = ratio * np.sqrt(1.0 / max(ta, 1.0) + 1.0 / max(tb, 1.0))
    use = (a >= PEARSON_MIN) & (k * b >= PEARSON_MIN)
    chi2 = float(np.sum((a[use] - k * b[use]) ** 2 / (a[use] + k * k * b[use])))
    return dict(ratio=ratio, err=err, z=(ratio - 1.0) / err if err > 0 else 0.0, chi2=chi2, ndf=int(use.sum()),
                n_a=ta, n_b=tb, k=k)  # fmt: skip


RESPONSES = ("absorbed", "qe")


def response(spec, det, energy=None) -> np.ndarray:
    """Return per sensor (the photon loop's order) the factor on lumen's deposit that gives the reference's count of
    a photon of ``energy`` eV (the module docstring: ``response``). The one rule for every comparison with a
    reference's hits (``lumen.compare.event`` too): "absorbed" counts the deposit as it is, 1 on every sensor, at any
    energy."""  # noqa: D205, D209
    kind = spec.get("response", "absorbed")
    if kind not in RESPONSES:
        raise ValueError(f"charge entry {spec['name']!r}: response {kind!r} is not one of {RESPONSES}")
    p = np.ones(len(det.parts.optics.sensor_pvs))
    if kind == "qe":
        if energy is None:
            raise ValueError(f"charge entry {spec['name']!r}: response 'qe' is read at the photon's energy")
        lam = HC / float(energy)
        params = det.params()
        for g in det.sensor_set.groups:
            model, tags = group_model(g), build.group_tags(g, det.parts, det.sensor_set)
            if model is None or not hasattr(model, "detection"):
                raise ValueError(f"charge entry {spec['name']!r}: response 'qe' needs a sensor model with a "
                                 f"detection on group {g.name!r}")  # fmt: skip
            i = np.arange(len(tags))
            rows = model.normal_rows(np.full((len(tags), 1), lam))
            d = model.detection(model.split(params, g.name), i[:, None], rows)
            p[tags] = np.asarray(d, np.float64)[:, 0]
    return p


def _types(det, s, spec, ref_lv):
    col = spec.get("group_column")
    if col is None:
        return ref_lv.astype(str), np.zeros(s.n_sensors, np.int32)
    types = np.asarray(det.sensor_table().columns[col]).astype(str)
    return types, np.unique(types, return_inverse=True)[1].astype(np.int32)


def check(name, *, detector=None, extra=(), **options) -> list:
    """Run the charge check named ``name`` (a manifest entry); return its record rows (``lumen.compare.records``)."""
    spec = references.spec("charge", name, extra)
    o = (
        DEFAULTS
        | {k: v for k, v in spec.items() if k in DEFAULTS}
        | {k: v for k, v in options.items() if v is not None}
    )
    t0 = time.time()
    ref = read_reference(references.product(spec["reference"], spec, extra))
    orig_id = references.original(spec["reference"], extra)  # the same sources on the original GDML, reported
    orig = read_reference(references.product(orig_id, spec, extra)) if orig_id else None
    if orig is not None and not (np.allclose(orig["x_mm"], ref["x_mm"]) and np.allclose(orig["energy"], ref["energy"])):
        raise ValueError(f"charge entry {name!r}: {orig_id} does not run {spec['reference']}'s sources")
    det = lumen.load(detector or spec["detector"], precision=spec.get("precision"))  # None: the detector's own
    split = spec.get("split_ev")  # every wavelength the source emits, in one band or two at the split
    s = photons.setup(det, (-np.inf, HC / split, np.inf) if split else None)
    nb = len(s.lam_edges) - 1
    centres = photons.sensor_centres(s) * 1e3
    lumen_of_ref, can = match(ref, centres, o["tol_mm"])
    lumen_of_orig = match(orig, centres, o["tol_mm"])[0] if orig is not None else None
    types_ref = np.full(s.n_sensors, "", object)
    types_ref[lumen_of_ref] = ref["lv"]
    types, group = _types(det, s, spec, types_ref)
    n_groups = int(group.max()) + 1
    diags = {}  # one Detected per response: its total's moment weights the sensors as the comparison does
    t_build = time.time() - t0
    out, health = (
        [],
        dict(truncated=0.0, lost=0.0, abandoned=0.0, escaped=0.0, photons=0, launched=float(np.sum(ref["launched"]))),
    )  # launched: the reference's, every source's
    for i, (x, n_cfg, energy) in enumerate(zip(ref["x_mm"], ref["n_cfg"], ref["energy"], strict=True)):
        t1 = time.time()
        seed = int(o["seed"]) + (i << 12)  # each source its own seeds, each call one of them
        p = response(spec, det, energy)
        key = tuple(float(v) for v in p)
        if key not in diags:
            diags[key] = photons.Detected(s.n_sensors, s.lam_edges, tuple(int(g) for g in group), int(o["nt"]),
                                          float(o["dt"]), () if np.all(p == 1.0) else key)  # fmt: skip
        r = run_source(det, diags[key], (x * 1e-3, HC / energy, o["photons_factor"] * n_cfg), int(o["per_call"]),
                       seed)  # fmt: skip
        r["charge"], r["charge_var"] = r.pop("Q"), r.pop("Q_var")
        for k in ("truncated", "lost", "abandoned", "escaped", "photons"):
            health[k] += float(np.sum(r[k])) if k != "photons" else r["photons"]
        p_group = np.array([p[group == g].mean() for g in range(n_groups)])
        h = ref["hits"]
        mine = h["src"] == i
        band = (np.where(h["E"][mine] > split, 0, 1) if split else np.zeros(int(mine.sum()), int)).astype(int)
        for b in range(nb):
            sel = lumen_of_ref[h["sensor"][mine][band == b]]
            n = np.bincount(sel, minlength=s.n_sensors).astype(np.float64)
            ctx = dict(can=can, types=types, dist=np.linalg.norm(centres - x, axis=1), launched=ref["launched"][i],
                       distance_groups=o["distance_groups"], p_group=p_group, t_ref=h["t"][mine][band == b],
                       dt=float(o["dt"]), tests=len(ref["x_mm"]) * nb)  # fmt: skip
            row = dict(src=i, band=b, seconds=time.time() - t1, **compare_band(r, b, n, p, ctx))
            if orig is not None:
                ho, mo = orig["hits"], orig["hits"]["src"] == i
                bo = (np.where(ho["E"][mo] > split, 0, 1) if split else np.zeros(int(mo.sum()), int)).astype(int)
                n_o = np.bincount(lumen_of_orig[ho["sensor"][mo][bo == b]], minlength=s.n_sensors).astype(np.float64)
                vs = compare_band(r, b, n_o, p, dict(ctx, launched=orig["launched"][i], t_ref=ho["t"][mo][bo == b]))
                row["original"] = {k: vs[k] for k in ("n_ref", "mu", "ratio", "err", "z", "mc_var", "chi2", "ndf")}
                row["geant4"] = geant4_ratio(n, n_o, ref["launched"][i], orig["launched"][i])
            out.append(row)
    gated = references.gated("charge", spec, extra)
    return _rows(name, spec, out, health, dict(build_s=t_build, total_s=time.time() - t0, gated=gated, **o),
                 orig_id)  # fmt: skip


def _rows(name, spec, out, health, run, orig_id=None):
    ref = spec["reference"]
    rows = []
    gate = (lambda ok: ok) if run.get("gated", True) else (lambda ok: None)  # a reported cross-check: no verdict
    # (a result without power, None, stays None)
    for r in out:
        tag = f"source {r['src']}" + (f" band {r['band']}" if any(x["band"] for x in out) else "")
        rows.append(records.row("charge", name, f"ratio {tag}", r["ratio"], r["err"], z=r["z"],
                                passed=gate(r["ratio_passed"]), reference=ref, **{k: v for k, v in r.items()
                                if k not in ("ratio", "err", "z", "ratio_passed")}))  # fmt: skip
        a = r["agree"]
        rows.append(records.row("charge", name, f"sensors agree {tag}", a["stat"],
                                passed=None if a["passed"] is None else gate(a["passed"]), reference=ref,
                                **{k: v for k, v in a.items() if k != "passed"},
                                chi2_ndf=r["chi2"] / max(r["ndf"], 1), ndf=r["ndf"], null=r["null"],
                                null_var=r["null_var"]))  # fmt: skip
        rows.append(records.row("charge", name, f"mean time lumen - reference {tag} (ns)", r["dt"], r["dt_err"],
                                reference=ref, ks_p=r["ks_p"]))  # fmt: skip
    nr, mu = sum(r["n_ref"] for r in out), sum(r["mu"] for r in out)
    mc = sum(r["mc_var"] for r in out)
    t = stats.ratio_test(mu, mc, nr)
    rows.append(records.row("charge", name, "ratio", t["ratio"], t["err"], z=t["z"],
                            passed=gate(t["passed"]), reference=ref,
                            mc_error=np.sqrt(mc) / nr if nr > 0 else np.nan, n_ref=nr, mu=mu, run=run))  # fmt: skip
    if orig_id is not None:
        rows += _original_rows(name, out, orig_id)
    rows.append(records.row("charge", name, "truncated weight", health["truncated"],
                            passed=health["truncated"] == 0, reference=ref, **health))  # fmt: skip
    # a navigation failure is a correctness failure on every detector: the weight lost or abandoned, in the
    # reference's photons, could not move the total by its Monte Carlo error had it all been detected
    failed = (health["lost"] + health["abandoned"]) * health["launched"] / max(health["photons"], 1)
    rows.append(records.row("charge", name, "lost and abandoned weight", failed, float(np.sqrt(mc)),
                            passed=bool(failed <= np.sqrt(mc)), reference=ref))  # fmt: skip
    return rows


def _original_rows(name, out, orig_id):
    """Return the reported rows against the same sources on the original GDML: G4(export) / G4(original) per source
    and band and in total, and lumen / G4(original) in total (``passed`` None: the overlap effect is shown, not
    gated)."""  # noqa: D205, D209
    rows = []
    for r in out:
        tag = f"source {r['src']}" + (f" band {r['band']}" if any(x["band"] for x in out) else "")
        g = r["geant4"]
        rows.append(records.row("charge", name, f"Geant4 export / original {tag}", g["ratio"], g["err"], z=g["z"],
                                passed=None, reference=orig_id, chi2_ndf=g["chi2"] / max(g["ndf"], 1), ndf=g["ndf"],
                                n_export=g["n_a"], n_original=g["n_b"]))  # fmt: skip
    ta = sum(r["geant4"]["n_a"] for r in out)
    kb = sum(r["geant4"]["k"] * r["geant4"]["n_b"] for r in out)
    k2b = sum(r["geant4"]["k"] ** 2 * r["geant4"]["n_b"] for r in out)
    g4 = ta / kb
    g4_err = g4 * np.sqrt(1.0 / ta + k2b / kb**2)
    nb = sum(r["geant4"]["n_b"] for r in out)
    rows.append(records.row("charge", name, "Geant4 export / original", g4, g4_err, z=(g4 - 1) / g4_err, passed=None,
                            reference=orig_id, n_export=ta, n_original=nb))  # fmt: skip
    nr, mu = sum(r["original"]["n_ref"] for r in out), sum(r["original"]["mu"] for r in out)
    mc = sum(r["original"]["mc_var"] for r in out)
    t = stats.ratio_test(mu, mc, nr)
    rows.append(records.row("charge", name, "ratio to the original GDML's reference", t["ratio"], t["err"], z=t["z"],
                            passed=None, reference=orig_id, n_ref=nr, mu=mu))  # fmt: skip
    return rows
