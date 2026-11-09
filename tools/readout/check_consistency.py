"""The SK (and WCTE) consistency checks of refs_new/SK_CONSISTENCY.md, with lumen's physical value sets.

    python tools/readout/check_consistency.py --det sk --e1 E1_HITS --qe QE_HITS --qe-digits QE_DIGITS.npz
        --reference sk_consistency_reference.json --set src/lumen/sensors/data/sk_r3600.json
        --emulation tests/references/wcsim_emulation/sk.json --convention 1.3333333333 --out OUT.json

lumen's physical QE and threshold against WCSim's QE x 4/3 (x 1/0.73 for WCTE) and SKI threshold, from WCSim's own
true hits (the photons reaching the tubes, from the QE-off run E1), so independent of tracking:
  Y1  yield after QE: the QE-on run's true hits N_s against convention x mu_s, mu = sum of lumen's QE(lambda) over
      E1's true hits (lambda = 2 pi 197.3 / E, WCSim's convention), per source and per sensor (Pearson chi^2/ndf);
  A_n closed form: P(digit | n photoelectrons) from lumen's compound charge machinery under the WCSim-emulating set,
      against the reference's closed form (and, for SK, its measured A_1 = 0.7390);
  T1.2 the digit-yield ratio: WCSim's QE-on digitised tube-events over lumen's (E1's hits thinned by lumen's QE,
      through lumen's
      physical electronics), against the ratio predicted from each model's A_n and the per-tube photon counts;
  T1.3 the accepted single-photon charge spectrum: lumen's over WCSim's, against the predicted ratio
      (WCSim's 0.985 scale; the same SPE, threshold and noise otherwise);
  T2  the digit time offset and spread per charge bin: lumen's physical timing (a per-photoelectron Gaussian) against
      its own prediction sqrt(sigma^2 + 0.4^2 / 12) after the reference-side 0.4 ns truncation, beside WCSim's.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from check_wcsim import Q_BINS, chi2_two, key, load_hits, lumen_digits, mean_rms
from scipy import stats

from lumen.readout import hits as hm
from lumen.sensors import pmt

HC_WCSIM = 2.0 * np.pi * 197.3  # nm eV (WCSimWCSD.cc:202)


def qe_of(model, energy, run_ce):
    """lumen's detection probability of WCSim's true hits: QE(lambda) x efficiency x the run's own (flat) CE.

    The QE-on runs use WCSim's stock collection efficiency (flat: SK 1, WCTE 0.95); the true hits carry no hit
    position, so a non-flat CE is checked through tracking's hit angle instead (Y1-CE).
    """
    lam = HC_WCSIM / np.asarray(energy, np.float64)
    return np.interp(lam, model.qe_knots, model.values["qe"]) * run_ce * float(model.values["efficiency"])


def a_n(model, n_max=4):
    """P(hit | n photoelectrons) for n = 1..n_max from lumen's charge grid: the n-fold SPE, threshold, noise, q > 0."""
    import jax.numpy as jnp

    v = model.split(model.defaults("pmt", 1), "pmt")
    v = {
        k: ({a: jnp.asarray(b, jnp.float32) for a, b in x.items()} if isinstance(x, dict) else x) for k, x in v.items()
    }
    n, n_neg, q = hm.charge_grid(model)
    dq = model.charge_step
    spe = np.roll(np.asarray(model.charge.pmf(v["charge"], np.r_[q - dq / 2, q[-1] + dq / 2]), np.float64), -n_neg)
    acc = np.asarray(model.threshold.accept(v["threshold"], jnp.asarray(q, jnp.float32)), np.float64)
    w = 2.0 * np.pi * np.fft.rfftfreq(n, dq)
    noise = np.exp(-0.5 * (float(model.values["frontend/charge_noise"]) * w) ** 2)
    out = {}
    for k in range(1, n_max + 1):
        pdf = np.roll(np.fft.irfft(np.fft.rfft(spe) ** k, n), n_neg)
        dig = np.roll(np.fft.irfft(np.fft.rfft(np.roll(pdf * acc, -n_neg)) * noise, n), n_neg)
        out[str(k)] = float(np.sum(dig[q > 0]))
    return out


def binom_digits(counts, p, accept):
    """Expected digits: sum over tube-events of sum_k Binom(k; n, p) A_k (A_k = 1 beyond the table)."""
    k = np.arange(0, counts.max() + 1)
    ak = np.r_[0.0, [accept.get(str(i), 1.0) for i in k[1:]]]
    un, mult = np.unique(counts, return_counts=True)
    return float(np.sum(mult * (stats.binom.pmf(k[None, :], un[:, None], np.clip(p, 0, 1)) * ak[None, :]).sum(1)))


def main():
    ap = argparse.ArgumentParser()
    for k in ("det", "e1", "qe", "qe-digits", "reference", "set", "emulation", "out"):
        ap.add_argument(f"--{k}", required=k != "reference")
    ap.add_argument("--convention", type=float, required=True)
    ap.add_argument("--run-ce", type=float, required=True)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    phys = pmt.model_from_dict(json.loads(Path(a.set).read_text()))
    emu = pmt.model_from_dict(json.loads(Path(a.emulation).read_text()))
    c1, h1 = load_hits(a.e1)
    c2, h2 = load_hits(a.qe)
    res = dict(det=a.det, convention=a.convention)
    # Y1: per source and per sensor
    p1 = qe_of(phys, h1["E"], a.run_ce)
    y1 = []
    for s in np.unique(h1["src"]):
        m1, m2 = h1["src"] == s, h2["src"] == s
        mu, var_mu = p1[m1].sum(), (p1[m1] ** 2).sum()
        n = int(m2.sum())
        r = n / (a.convention * mu)
        err = r * np.sqrt(1.0 / n + var_mu / mu**2)
        y1.append(dict(src=int(s), N_wcsim=n, mu_lumen=float(mu), R=r, R_err=err, z=(r - 1) / err))
    res["Y1_sources"] = y1
    ns = int(max(h1["sensor"].max(), h2["sensor"].max())) + 1
    mu_i = np.bincount(h1["sensor"], p1, ns) * a.convention
    var_i = np.bincount(h1["sensor"], p1**2, ns) * a.convention**2 + mu_i
    n_i = np.bincount(h2["sensor"], minlength=ns)
    ok = mu_i >= 5
    pull = (n_i[ok] - mu_i[ok]) / np.sqrt(var_i[ok])
    res["Y1_sensors"] = dict(n=int(ok.sum()), chi2_ndf=float(np.mean(pull**2)), frac_beyond_3=float(np.mean(np.abs(
        pull) > 3)))  # fmt: skip
    # A_n: lumen's machinery on the WCSim-emulating set against WCSim's closed form
    an = a_n(emu)
    res["A_n_lumen_emulation"] = an
    if a.reference:
        ref = json.loads(Path(a.reference).read_text())
        res["A_n_reference"] = ref["closed_form"]["A_n"]
        res["A_1_measured_E1"] = ref["measured"]["E1_qe_off"]["P_digit_given_1_true_hit"]
    an_phys = a_n(phys)
    res["A_n_lumen_physical"] = an_phys
    # T1.2: digit yields, WCSim QE-on run vs lumen physical on E1 thinned by lumen's QE
    rng = np.random.default_rng(a.seed)
    k1 = key(c1, h1["src"], h1["evt"], h1["sensor"])
    keep = rng.random(len(h1)) < p1
    t_pe = h1["t"].astype(np.float64)[keep] + phys.tts.sample(rng, {"sigma": phys.values["tts/sigma"]}, int(keep.sum()))
    ours = lumen_digits(phys, k1[keep], t_pe, rng, 0.4)
    z = np.load(a.qe_digits)["digits"]
    t12 = []
    uk, inv, counts = np.unique(k1, return_inverse=True, return_counts=True)
    src_k = (uk >> 32) & 0xFF
    p_src = {int(s): float(np.mean(p1[h1["src"] == s])) for s in np.unique(h1["src"])}
    src_of = (ours["key"] >> 32) & 0xFF
    for s in np.unique(h1["src"]):
        cs = counts[src_k == s]
        pred = binom_digits(cs, a.convention * p_src[int(s)], an) / binom_digits(cs, p_src[int(s)], an_phys)
        # digitised tube-events (the prediction's quantity: a later window of late light is not in it)
        kw = key(z["chunk"].astype(np.int64), z["src"], z["evt"], z["sensor"])
        dw, dl = len(np.unique(kw[z["src"] == s])), len(np.unique(ours["key"][src_of == s]))
        r = dw / dl
        # the prediction is built on E1's realisation, independent of the QE-on run's: its noise ~ that of dw
        err = r * np.sqrt(2 / dw + 1 / dl)
        t12.append(dict(src=int(s), D_wcsim=dw, D_lumen=dl, ratio=r, ratio_err=err, predicted=pred, z=(r - pred) / err))
    res["T1_2_yield_ratio"] = t12
    # T1.3: accepted single-photon charge spectra, the ratio against the scale transform
    qb = np.arange(0.0, 6.0001, 0.05)
    s1, r1 = ours["nph"] == 1, z["nph"] == 1
    hl, _ = np.histogram(ours["q"][s1], qb)
    hw, _ = np.histogram(z["q"][r1].astype(np.float64) / float(emu.values["frontend/charge_scale"]), qb)
    res["T1_3_charge_shape"] = chi2_two(hl, hw)
    res["T1_3_charge_shape"]["note"] = "WCSim's charges divided by its 0.985: then the spectra are the same model"
    # T2: timing per charge bin, lumen physical vs its own prediction (and WCSim beside)
    tl = _first_times(ours, k1[keep], h1["t"].astype(np.float64)[keep])
    uk1, n_kept = np.unique(k1[keep], return_counts=True)
    one = n_kept[np.searchsorted(uk1, ours["key"])] == 1  # the tube-event's only photoelectron: its true time is known
    s1 = s1 & one
    dt_l = (ours["t"] - tl)[s1]
    dt_w = (z["t"] - z["t_true_first"])[r1]
    sig = float(phys.values["tts/sigma"])
    per = []
    ql, qw = ours["q"][s1], z["q"][r1] / float(emu.values["frontend/charge_scale"])
    for lo, hi in zip(Q_BINS[:-1], Q_BINS[1:], strict=False):
        sel = (ql >= lo) & (ql < hi)
        aa, bb = dt_l[sel], dt_w[(qw >= lo) & (qw < hi)]
        if min(len(aa), len(bb)) < 100:
            continue
        walk = np.zeros(int(sel.sum()))
        if phys.time_walk is not None:
            walk = np.interp(ql[sel], phys.time_walk.knots, phys.values["time_walk/sigma"])
        pred_rms = float(np.sqrt(sig**2 + np.mean(walk**2) + 0.4**2 / 12.0))
        ma = mean_rms(aa)
        per.append(dict(Q=[lo, hi], lumen=ma, wcsim=mean_rms(bb), lumen_pred_mean=-0.2, lumen_pred_rms=pred_rms,
                        d_mean=ma["mean"] + 0.2, d_rms_rel=ma["rms"] / pred_rms - 1.0))  # fmt: skip
    res["T2_timing"] = per
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "T2_timing"}, indent=1, default=float))


def _first_times(d, k1, t):
    """The first true time of each lumen hit's tube-event (before the time spread): for single-photon hits."""
    order = np.lexsort((t, k1))
    uk, first = np.unique(k1[order], return_index=True)
    return t[order][first][np.searchsorted(uk, d["key"])]


if __name__ == "__main__":
    main()
