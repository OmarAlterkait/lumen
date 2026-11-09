"""WCSim digits from WCSim's own true hits: lumen's PMT electronics against WCSim's digitiser, independent of tracking.

    python tools/readout/check_wcsim.py --hits HITS_DIR --digits DIGITS.npz --sets SET.json [--types TYPES.csv]
                                        --out OUT.json [--seed 1]

HITS_DIR holds a WCSim run's true hits (chunk<c>_hits.npy: src, evt, sensor, t, E); DIGITS.npz the same run's digits
(refs_new/scripts/dump_digits.C via to_digits.py). Every true hit is a photoelectron in WCSim (QE is applied, when
on, before the hit is recorded), so lumen's electronics take them as its photoelectrons: SPE charge, integration
windows, threshold, charge noise, time walk and charge scale under the WCSim-emulating value set (tests/references/
wcsim_emulation). WCSim truncates a digit's time toward zero on a 0.4 ns grid (WCSimWCDigitizer.hh:41-47); that is
applied to lumen's hit times here, on the reference side of the comparison (SPEC 15), when the run's digitiser has
it (``--time-precision``, 0.4 for SK and WCTE, 0 for ANNIE).

Compared, each with a two-sample chi^2 (both samples are Monte Carlo of the same true hits) or a z:
  P(digit | n true hits in the tube) for n = 1..5; the photons-per-digit occupancy; the digit charge (all and
  single-photon digits); T - T_true(first) of single-photon digits pooled and per charge bin (mean, rms); the yields
  per source (digits and photons in digits over true hits).
--types: ANNIE's tube type per WCSim tube (sensor,tube_type), with --sets a format string ``annie_{}.json``.
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from lumen.readout.events import electronics
from lumen.sensors import pmt

GATE = (0.0, 1e12)  # ns: NoTrigger's window [0, LongTime] (WCSimWCTrigger.cc GetPre/PostTriggerWindow, the cut at
# digit_time >= lowerbound; annie e88314b :126-155, :534; develop alike): a digit before t = 0 is dropped
Q_BINS = [0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]  # pe: SK_CONSISTENCY.md's timing bins (Q = q_digit / scale)


def load_hits(d):
    out = []
    for f in sorted(glob.glob(str(Path(d) / "chunk*_hits.npy"))):
        m = re.search(r"chunk(\d+)_hits\.npy$", f)
        if m:
            h = np.load(f)
            out.append((int(m.group(1)), h))
    chunk = np.concatenate([np.full(len(h), c, np.int64) for c, h in out])
    h = np.concatenate([h for _, h in out])
    return chunk, h


def key(chunk, src, evt, sensor):
    return (np.int64(1) * chunk << 40) | (src.astype(np.int64) << 32) | (evt.astype(np.int64) << 16) | sensor.astype(
        np.int64)  # fmt: skip


def chi2_two(a, b):
    """Two-sample chi^2 of histograms a, b (counts), ndf = non-empty bins - 1; returns (chi2, ndf, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ka, kb = np.sqrt(b.sum() / a.sum()), np.sqrt(a.sum() / b.sum())
    ok = (a + b) > 0
    c2 = float(np.sum((ka * a[ok] - kb * b[ok]) ** 2 / (a[ok] + b[ok])))
    ndf = int(ok.sum()) - 1
    return dict(chi2=c2, ndf=ndf, chi2_ndf=c2 / max(ndf, 1), p=float(stats.chi2.sf(c2, max(ndf, 1))))


def mean_rms(x):
    n = len(x)
    m, s = float(np.mean(x)), float(np.std(x))
    return dict(n=n, mean=m, mean_err=s / np.sqrt(n), rms=s, rms_err=s / np.sqrt(2.0 * n))


def lumen_digits(model, sensor_key, t, rng, prec):
    """Run lumen's electronics on photoelectrons grouped by tube-event key; return the hits with their keys."""
    uk, inv = np.unique(sensor_key, return_inverse=True)
    vals = model.split(model.defaults("pmt", 1), "pmt")
    ev = electronics(model, vals, inv, t, rng, len(uk), gate=GATE)
    h = ev.hits
    tt = prec * np.trunc(h["t"] / prec) if prec > 0 else h["t"]
    return dict(key=uk[h["sensor"]], q=h["q"], t=tt, nph=h["n_pe"], t_first=h["t_first"])


def compare(ours, ref, hk, scale):
    """Return the comparison statistics of lumen's digits (``ours``) and WCSim's (``ref``) for true-hit keys hk."""
    out = {}
    uk, n_true = np.unique(hk, return_counts=True)
    for name, d in (("lumen", ours), ("wcsim", ref)):
        has = np.isin(uk, d["key"])
        out[f"p_digit_{name}"] = {str(n): [int(has[n_true == n].sum()), int((n_true == n).sum())] for n in range(1, 6)}
    z = {}
    for n in range(1, 6):
        a, na = out["p_digit_lumen"][str(n)]
        b, nb = out["p_digit_wcsim"][str(n)]
        p = (a + b) / (na + nb)
        z[str(n)] = dict(lumen=a / na, wcsim=b / nb, z=(a / na - b / nb) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb)))
    out["p_digit"] = z
    nb = np.arange(0.5, 30.5)
    out["occupancy"] = chi2_two(np.histogram(ours["nph"], nb)[0], np.histogram(ref["nph"], nb)[0])
    out["occupancy_frac_multi"] = dict(lumen=float(np.mean(ours["nph"] > 1)), wcsim=float(np.mean(ref["nph"] > 1)))
    qb = np.r_[np.arange(-0.5, 8.0, 0.05), 1e9]
    out["charge_all"] = chi2_two(np.histogram(ours["q"], qb)[0], np.histogram(ref["q"], qb)[0])
    s1, r1 = ours["nph"] == 1, ref["nph"] == 1
    out["charge_single"] = chi2_two(np.histogram(ours["q"][s1], qb)[0], np.histogram(ref["q"][r1], qb)[0])
    out["charge_mean"] = dict(lumen=mean_rms(ours["q"]), wcsim=mean_rms(ref["q"]))
    out["charge_single_mean"] = dict(lumen=mean_rms(ours["q"][s1]), wcsim=mean_rms(ref["q"][r1]))
    dt_l, dt_w = ours["t"][s1] - ours["t_first"][s1], ref["t"][r1] - ref["t_true_first"][r1]
    tb = np.arange(-20.0, 20.0001, 0.2)
    out["time_single"] = chi2_two(np.histogram(dt_l, tb)[0], np.histogram(dt_w, tb)[0])
    out["time_single"].update(lumen=mean_rms(dt_l), wcsim=mean_rms(dt_w))
    per = []
    ql, qw = ours["q"][s1] / scale, ref["q"][r1] / scale
    for lo, hi in zip(Q_BINS[:-1], Q_BINS[1:], strict=False):
        a, b = dt_l[(ql >= lo) & (ql < hi)], dt_w[(qw >= lo) & (qw < hi)]
        if min(len(a), len(b)) < 100:
            continue
        ma, mb = mean_rms(a), mean_rms(b)
        per.append(dict(Q=[lo, hi], lumen=ma, wcsim=mb, d_mean=ma["mean"] - mb["mean"],
                        d_mean_err=float(np.hypot(ma["mean_err"], mb["mean_err"])),
                        d_rms_rel=ma["rms"] / mb["rms"] - 1.0,
                        d_rms_rel_err=float(np.hypot(ma["rms_err"], mb["rms_err"]) / mb["rms"]),
                        chi2=chi2_two(np.histogram(a, tb)[0], np.histogram(b, tb)[0])))  # fmt: skip
    out["time_by_charge"] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True)
    ap.add_argument("--digits", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--types")
    ap.add_argument("--time-precision", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    chunk, h = load_hits(a.hits)
    hk = key(chunk, h["src"], h["evt"], h["sensor"])
    z = np.load(a.digits)
    d = z["digits"]
    ref = dict(key=key(d["chunk"].astype(np.int64), d["src"], d["evt"], d["sensor"]), q=d["q"].astype(np.float64),
               t=d["t"], nph=d["nph"], t_true_first=d["t_true_first"], src=d["src"])  # fmt: skip
    rng = np.random.default_rng(a.seed)
    if a.types:
        ty = np.genfromtxt(a.types, delimiter=",", names=True, dtype=None, encoding=None)
        tmap = dict(zip(ty["sensor"].tolist(), ty["tube_type"].tolist(), strict=False))
        tube = np.asarray([tmap[s] for s in h["sensor"].tolist()])
        parts = [(t, tube == t) for t in sorted(set(tmap.values()))]
    else:
        parts = [(None, np.ones(len(h), bool))]
    pieces, scales = [], {}
    for t, sel in parts:
        f = a.sets.format(t) if t else a.sets
        model = pmt.model_from_dict(json.loads(Path(f).read_text()))
        scales[t] = float(model.values["frontend/charge_scale"])
        pieces.append(lumen_digits(model, hk[sel], h["t"][sel].astype(np.float64), rng, a.time_precision))
    ours = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    scale = next(iter(scales.values()))
    res = dict(true_hits=int(len(h)), digits_wcsim=int(len(d)), digits_lumen=int(len(ours["q"])))
    res["all"] = compare(ours, ref, hk, scale)
    src_of = (ours["key"] >> 32) & 0xFF
    yields = []
    for s in np.unique(h["src"]):
        nt = int(np.sum(h["src"] == s))
        dl, dw = int(np.sum(src_of == s)), int(np.sum(ref["src"] == s))
        pl, pw = int(ours["nph"][src_of == s].sum()), int(ref["nph"][ref["src"] == s].sum())
        yields.append(dict(src=int(s), true=nt, digits_lumen=dl, digits_wcsim=dw,
                           z_digits=(dl - dw) / np.sqrt(dl + dw), photons_in_digits_lumen=pl / nt,
                           photons_in_digits_wcsim=pw / nt, z_photons=(pl - pw) / np.sqrt(pl + pw)))  # fmt: skip
    res["yields"] = yields
    if a.types:
        res["by_type"] = {}
        for t, sel in parts:
            kk = np.unique(hk[sel])
            o = {k: v[np.isin(ours["key"], kk)] for k, v in ours.items()}
            r = {k: v[np.isin(ref["key"], kk)] for k, v in ref.items()}
            res["by_type"][t] = compare(o, r, hk[sel], scales[t])
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    al = res["all"]
    print(
        json.dumps(
            dict(
                p_digit=al["p_digit"],
                occupancy=al["occupancy"],
                charge_all=al["charge_all"],
                charge_single=al["charge_single"],
                time_single=al["time_single"],
            ),
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
