"""Y1-CE (refs_new/SK_CONSISTENCY.md): the yield with WCSim's non-flat CE, from lumen's own hit angle.

    python tools/readout/check_y1_ce.py DET --ce-runs DIR --table CE_TABLE.json [--photons 4194304] [--out OUT.json]

The reference: WCSim with QE off and the CE table on (refs_new/wcsim_ce/{sk,wcte}_ce, true hits chunk*_hits.npy), the
E1 sources, seeds and chunking, so its hits are E1's thinned by CE. lumen: the detector's border_into group read by
``lumen:pmt`` with QE 1 over the detector's range, efficiency 1 and CE(theta) from the table (theta from the deposit:
``lumen.transport.step.hit_angle``), each E1 source's isotropic monochromatic photons through ``lumen.simulate``.

Per source: the CE-only ratio N_CE / N_E1 against lumen's <CE> =
sum pe / sum arrivals (its Monte Carlo error from the replicas); and N_CE per sensor against mu_i = lumen's pe scaled
to the reference's photons (Pearson chi^2 / ndf over sensors expecting >= 5, variance mu + lumen's Monte Carlo
variance), with the total yield R = N_CE / sum mu. Passes: |z| < 3 on <CE>, and the chi^2/ndf rule of the charge
check (``lumen.validate.stats.chi2_passes``). The ratio's error treats the two runs as independent samples
(PLAN.md's +-0.0016 SK, +-0.0023 WCTE): the seeds are E1's, but each CE draw takes a random number from the event's
engine, so every later photon of the event follows another history; the CE run is not E1 thinned photon by photon.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_wcsim import load_hits

MU_MIN = 5.0


def value_set(path, table, lam_range):
    """Write the Y1-CE value set (QE 1 on the detector's range, CE from the table, readout "pe")."""
    t = json.loads(Path(table).read_text())
    d = dict(schema="lumen-pmt-values/1", model="lumen:pmt", model_version=1, name="y1_ce", version=1, readout="pe",
             provenance=dict(source=f"{table}: {t['provenance']['source']}", status="transcribed"),
             qe=dict(band_edges_nm=[float(lam_range[0]), float(lam_range[1])], values=[1.0]),
             ce=dict(theta_deg=[float(x) for x in t["theta_deg"]], values=[x / 100.0 for x in t["ce_percent"]]),
             efficiency=1.0, t_offset_ns=0.0, tts=None, charge=None, gain=1.0, threshold=None, dark_rate_hz=0.0,
             frontend=dict(sampling_ns=1.0, unit="pe", window_ns=0.0, charge_step_pe=0.25,
                           charge_max_pe=4.0))  # fmt: skip
    Path(path).write_text(json.dumps(d, indent=1))
    return str(path)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("det")
    ap.add_argument("--ce-runs", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--photons", type=int, default=1 << 22)
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    import lumen
    from lumen import readout as ro
    from lumen.build import build
    from lumen.units import HC
    from lumen.validate import charge, photons, references, stats

    t0 = time.time()
    spec = references.spec("charge", a.det)
    ref = charge.read_reference(references.product(spec["reference"], spec))
    det0 = lumen.load(spec["detector"])
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    vs = value_set(work / f"y1_ce_{a.det}.json", a.table, det0.file.wavelength_range)
    g0 = next(g for g in det0.sensor_set.groups if g.rule == "border_into")
    g = dataclasses.replace(g0, model="lumen:pmt", defaults=vs)
    sset = dataclasses.replace(det0.sensor_set, groups=tuple(g if x is g0 else x for x in det0.sensor_set.groups))
    parts = build.build(det0.file, sset)
    params = build.parameters(det0.file, sset, parts)
    det = lumen.Detector(det0.name, det0.path, det0.file, sset, parts, det0.sensors, det0.calibration, params)
    s = photons.setup(det)
    lumen_of_ref, _ = charge.match(ref, photons.sensor_centres(s) * 1e3, spec.get("tol_mm", 5.0))
    _, ce_hits = load_hits(a.ce_runs)
    e1 = ref["hits"]
    rows = []
    for i, (x_mm, energy) in enumerate(zip(ref["x_mm"], ref["energy"], strict=True)):
        x = jnp.asarray(np.asarray(x_mm) * 1e-3, jnp.float32)
        lam = float(HC / energy)

        def source(first, count, x=x, lam=lam, i=i):
            key = jax.random.fold_in(jax.random.PRNGKey(a.seed), i << 16 | first // (1 << 20))
            return photons.isotropic(jnp.tile(x, (count, 1)), jnp.full(count, lam, jnp.float32), key)

        light = lumen.simulate(det, source, samples=a.photons, seed=a.seed + i, replicas=a.replicas)
        h = ro.detection.histograms(light)
        v = ro.detection.values(h, params)
        per = np.asarray(v["efficiency"][None] * jnp.einsum("rsla,l,a->rs", h.charge, v["qe"], v["ce"]), np.float64)
        arr = np.asarray(h.charge.sum((2, 3)), np.float64)  # (R, S): arrivals in the QE band, CE 1
        ce_r = per.sum(1) / arr.sum(1)
        ce_mean = per.sum() / arr.sum()
        ce_err = float(np.std(ce_r, ddof=1) / np.sqrt(len(ce_r)))
        n_e1 = int((e1["src"] == i).sum())
        sel = ce_hits["src"] == i
        n_ce = int(sel.sum())
        ratio = n_ce / n_e1
        ratio_err = float(ratio * np.sqrt(1.0 / n_ce + 1.0 / n_e1))  # independent samples: see the module docstring
        z = (ratio - ce_mean) / np.hypot(ratio_err, ce_err)
        scale = ref["launched"][i] / a.photons
        mu = per.sum(0) * scale
        mc_var = (per.shape[0] / (per.shape[0] - 1) * ((per - per.mean(0)) ** 2).sum(0)) * scale**2
        n_obs = np.bincount(lumen_of_ref[ce_hits["sensor"][sel]], minlength=mu.size).astype(np.float64)
        ok = mu >= MU_MIN
        chi2 = float(np.sum((n_obs[ok] - mu[ok]) ** 2 / (mu[ok] + mc_var[ok])))
        ndf = int(ok.sum()) - 1
        rows.append(dict(source=i, x_mm=[float(c) for c in x_mm], energy_ev=float(energy), n_e1=n_e1, n_ce=n_ce,
                         ratio_wcsim=ratio, ratio_err=ratio_err, ce_lumen=float(ce_mean), ce_lumen_err=ce_err,
                         z=float(z),
                         yield_ratio=float(n_ce / mu.sum()), chi2_ndf=chi2 / max(ndf, 1), ndf=ndf,
                         passed=bool(abs(z) < 3.0 and stats.chi2_passes(chi2 / max(ndf, 1), ndf)),
                         health={k: float(np.asarray(v_).sum()) for k, v_ in light.health.items()}))  # fmt: skip
        print(json.dumps(rows[-1]), flush=True)
    out = dict(det=a.det, ce_runs=a.ce_runs, table=a.table, photons=a.photons, value_set=vs, seconds=time.time() - t0,
               passed=all(r["passed"] for r in rows), rows=rows)  # fmt: skip
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"Y1-CE {a.det}: pass {out['passed']}", flush=True)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
