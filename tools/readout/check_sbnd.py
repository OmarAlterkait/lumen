"""SBND PMT waveforms: lumen's expected waveform from the photon lists against sbndcode OpDetSim's mean and variance.

    python tools/readout/check_sbnd.py --ref REFS_NEW/sbnd_opdetsim --calib sbnd_pmt_calibration_v3r1.npz
                                       --tpb timeTPB.npy --out OUT.json [--occupancies 1 10]

The inputs are OpDetSim's own (refs_new/PLAN.md (b)): the photon lists it digitised (photons/photons.npz: event,
channel, t, band; SimPhotonsLite keeps (int) t), and, as test inputs only (decision 55), the per-channel values of
its PMT calibration database (sbnd_data CalibrationDatabase/pds_calibration.db, tag v3r1: on, SER, SPE amplitude and
its spread). What OpDetSim does with them (sbndcode v10_26_00 DigiPMTSBNDAlg.cc) is emulated by a lumen:pmt value set
per channel, no number tuned:
  detection   per ns bin with c photons, Poisson(c eff) photoelectrons, eff = the fcl efficiency / ScintPreScale
              (0.0392): coated VUV, coated visible, uncoated visible (an uncoated tube reads no VUV light); eff1: 1
              / 0.0392 for every band read;
  time        + Gaussian TTS (2.4 ns FWHM) + the TPB time drawn by CLHEP::RandGeneral, in [0, 1) ns (timeTPB's 1000
              bins on [0, 1)); then the photoelectron is placed on the sample floor(t / 2 ns) (ResolutionIncrease 1),
              so the effective pulse is the SER as a staircase: h(x) = SER[ceil(x / 2 ns)];
  charge      per photoelectron N(1, (spe_amp_std / spe_amp)^2) (PMTGaussianGainFluctuation), gain spe_amp at the
              SER's peak (the SER normalised to its peak, as PMTSinglePEmodel "data");
  noise       eff1_nonoise: none (baseline 14250); nominal and eff1: the data noise templates, whose per-channel
              baseline and variance are taken from the reference's own samples before any light (-1000 to -200 ns).
The variance: lumen's Campbell variance of Poisson photoelectrons given the photons, plus the variance between the
reference's events of their own photon lists: each event's expected waveform is eff x sum over its photons of R(t),
R lumen's expected response to one photoelectron. OpDetSim stores each sample as unsigned short (DigiPMTSBNDAlg.cc
ConstructWaveformLite*: std::vector<short unsigned int>(waves.begin(), waves.end())), truncating it: where a pulse
is present the sample loses its fractional part, so without noise the reference's mean is lowered by 0.5 P(pulse
present) (and its variance raised by P / 3 - P^2 / 4), P from the photon lists; with the data noise, whose values are
fractional, every sample loses it alike, in the measured baseline. Compared at the occupancies where OpDetSim's
non-linearity is off.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from lumen import readout as rd
from lumen.sensors import pmt

PRESCALE = 0.0392  # opticalproperties_sbnd.fcl:33 ScintPreScale
EFF = {"pmt_coated": (0.0392, 0.026), "pmt_uncoated": (0.0, 0.0357)}  # digi_pmt_sbnd.fcl (VUV, visible)
EDGES = (100.0, 200.29757415185782, 900.0)  # nm: the 6.19 eV split (VUV, visible)
LAM = (128.0, 430.0)  # nm: a wavelength inside each band
TTS_FWHM = 2.4  # ns, digi_pmt_sbnd.fcl TTS
SAMPLE = 2.0  # ns
GATE = (-1000.0, 9000.0)  # ns: the fixed 10 us window from -1 us (make_fcl.sh)
BASELINE = 14250.0
SUB = 32  # staircase knots per sample
PRE = slice(0, 400)  # samples before any light (photons arrive after 0 ns)
LOCAL = slice(440, 495)  # just before the light (-120 to -10 ns): the baseline the main lobe is measured from


def value_set(ser_len, tpb):
    """The OpDetSim-emulating lumen:pmt value set (structure; per-channel values are set per call)."""
    edges = (np.arange(len(tpb) + 1) / len(tpb)).tolist()  # RandGeneral: [0, 1) ns
    n = ser_len * SUB + SUB
    return dict(schema=pmt.SCHEMA, model="lumen:pmt", model_version=1, name="sbnd_opdetsim_emulation", version=1,
                readout="waveform", provenance=dict(source="sbndcode v10_26_00 OpDetSim (test reference)",
                                                    status="transcribed"),
                qe=dict(band_edges_nm=list(EDGES), values=[1.0, 1.0]), ce=None, efficiency=1.0, t_offset_ns=0.0,
                tts=dict(form="gaussian", sigma_ns=TTS_FWHM / 2.3548200450309493),
                delay=dict(form="table", edges_ns=edges, pdf=list(np.asarray(tpb, float))),
                charge=dict(form="gauss_exp", sigma_pe=0.4, fraction=0.0, tau_pe=1.0), gain=1.0, threshold=None,
                pulse=dict(form="table", t_first_ns=-SAMPLE, step_ns=SAMPLE / SUB, shape=[0.0] * n),
                dark_rate_hz=0.0,
                frontend=dict(sampling_ns=SAMPLE, unit="ADC", baseline=BASELINE, noise=0.0, window_ns=0.0,
                              charge_noise_pe=0.0, charge_scale=1.0, charge_step_pe=0.25, charge_max_pe=4.0),
                ranges={"tts/sigma": [1.0, 2.0], "t_offset": [0.0, 0.0]})  # fmt: skip


def staircase(ser):
    """The SER as OpDetSim places it: h(x) = SER[ceil(x / 2 ns)] for x in (-2, ...), on knots of 2 ns / SUB."""
    x = -SAMPLE + (SAMPLE / SUB) * np.arange(len(ser) * SUB + SUB)
    j = np.ceil(x / SAMPLE - 1e-9).astype(int)
    return np.where((j >= 0) & (j < len(ser)), ser[np.clip(j, 0, len(ser) - 1)], 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--tpb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--occupancies", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--configs", nargs="+", default=["eff1_nonoise", "eff1", "nominal"])
    a = ap.parse_args()
    cal = np.load(a.calib)
    tpb = np.load(a.tpb)
    ph = np.load(Path(a.ref) / "photons" / "photons.npz")["photons"]
    model = pmt.model_from_dict(value_set(cal["ser"].shape[1], tpb))
    on, rec = cal["on_pmt"] > 0, cal["reconstruct_channel"] > 0
    peak = np.abs(cal["ser"]).max(1)
    avg_raw = cal["ser"][rec].mean(0)  # channels on but uncalibrated: the average SER at the average amplitude
    avg, avg_amp = avg_raw / np.abs(avg_raw).max(), float(cal["spe_amp"][rec].mean())
    params0 = {k: np.asarray(v, np.float32) for k, v in model.defaults("pmt", 1).items()}

    @jax.jit
    def expect(h, p):
        w = rd.expected_waveform(h, p)
        return w.mean[0], w.variance[0]

    res = {}
    for cfg in a.configs:
        ref = np.load(Path(a.ref) / f"waveforms_{cfg}.npz")
        types = ref["channels"]["pd_type"]
        occ = list(ref["occupancies"])
        res[cfg] = {}
        for L in a.occupancies:
            li = occ.index(L)
            for s in ref["sources"]:
                rows = []
                sel = (ph["L"] == L) & (ph["src"] == s)
                for ci, ch in enumerate(ref["pmt_channels"]):
                    k = int(np.flatnonzero(cal["channel"] == ch)[0])
                    pd = types[ch]
                    eff = np.asarray(EFF[pd]) / PRESCALE if cfg == "nominal" else np.asarray(
                        [0.0 if pd == "pmt_uncoated" else 1.0, 1.0]) / PRESCALE  # fmt: skip
                    m_ref, v_ref = ref["mean_pmt"][li, s, ci].astype(np.float64), ref["var_pmt"][li, s, ci]
                    nev = int(ref["n_events"][li, s, ch])
                    p = dict(params0)
                    p["sensors/pmt/qe"] = eff.astype(np.float32)
                    shape = (cal["ser"][k] / peak[k]) if rec[k] else avg
                    p["sensors/pmt/pulse/shape"] = staircase(shape).astype(np.float32)
                    amp = (float(cal["spe_amp"][k]) if rec[k] else avg_amp) if on[k] else 0.0  # an off tube: no pulse
                    p["sensors/pmt/gain"] = np.asarray([amp], np.float32)
                    p["sensors/pmt/charge/sigma"] = np.float32(cal["spe_amp_std"][k] / cal["spe_amp"][k])
                    if cfg != "eff1_nonoise":
                        p["sensors/pmt/frontend/baseline"] = np.float32(m_ref[PRE].mean())
                        p["sensors/pmt/frontend/noise"] = np.float32(np.sqrt(v_ref[PRE].mean()))
                    mine = sel & (ph["opchannel"] == ch)
                    t = np.trunc(ph["t"][mine]).astype(np.float64)  # SimPhotonsLite's (int) t
                    lam = np.asarray(LAM)[ph["band"][mine]]
                    phot = dict(sensor=np.zeros(len(t), int), lam=lam, time=t, weight=np.full(len(t), 1.0 / nev))
                    h = rd.from_photons(model, GATE, 1, phot) if len(t) else None
                    verr = None
                    if h is None:
                        mean, var = np.full(5000, float(p["sensors/pmt/frontend/baseline"])), np.full(
                            5000, float(p["sensors/pmt/frontend/noise"]) ** 2)  # fmt: skip
                    else:
                        mean, var = (np.asarray(x, np.float64) for x in expect(h, p))
                        within = var.copy()
                        between, m2d, m4d = _between(model, p, ph[mine], eff, nev, expect)
                        var = var + between
                        if cfg == "eff1_nonoise":
                            pp = _present(ph[mine], eff, nev)
                            mean, var = mean - 0.5 * pp, var + pp / 3.0 - pp**2 / 4.0
                            within = within + pp / 3.0 - pp**2 / 4.0
                        # the sample variance's error: event values = their mean + Gaussian within-event noise
                        w0 = np.maximum(within, 0.0)
                        verr = np.sqrt(np.maximum(m4d - m2d**2 + 4 * w0 * m2d + 2 * w0**2, 0.0) / nev)
                    sig = np.abs(mean - mean[PRE].mean()) > 1e-3 * max(np.abs(mean - mean[PRE].mean()).max(), 1e-9)
                    err2 = (v_ref + np.maximum(var, 0.0)) / nev / 2.0 + 1e-12  # (FFT round-off: var >= 0)
                    win = sig | (np.abs(m_ref - m_ref[PRE].mean()) > 5 * np.sqrt(err2))
                    area_l = float(np.sum((mean - mean[PRE].mean())[win]))
                    area_r = float(np.sum((m_ref - m_ref[PRE].mean())[win]))
                    area_err = float(np.sqrt(np.sum(v_ref[win]) / nev))
                    # the main lobe (expected pulse below 20 % of its peak), against the local baseline
                    sl = mean - mean[LOCAL].mean()
                    lobe = sl < 0.2 * sl.min() if sl.min() < 0 else np.zeros(len(sl), bool)
                    lobe_l, lobe_r = float(sl[lobe].sum()), float((m_ref - m_ref[LOCAL].mean())[lobe].sum())
                    lobe_err = float(np.sqrt(np.sum(v_ref[lobe]) / nev + lobe.sum() ** 2 * v_ref[LOCAL].mean() / nev
                                             / (LOCAL.stop - LOCAL.start)))  # fmt: skip
                    vsig_l = float(np.sum((var - var[PRE].mean())[win]))
                    vsig_r = float(np.sum((v_ref - v_ref[PRE].mean())[win]))
                    vp = 0.0
                    if verr is not None:  # over the samples carrying light (round-off elsewhere)
                        lit_v = win & (var > 0.01 * var.max()) & (verr > 0)
                        vp = float(np.sqrt(np.mean(((v_ref - var) / verr)[lit_v] ** 2))) if lit_v.any() else 0.0
                    rows.append(dict(channel=int(ch), type=pd, n_photons=int(len(t)), on=bool(on[k]), var_pull_rms=vp,
                                     chi2_mean=float(np.sum((m_ref - mean)[win] ** 2 / err2[win])),
                                     ndf=int(win.sum()), area_lumen=area_l, area_ref=area_r, area_err=area_err,
                                     lobe_lumen=lobe_l, lobe_ref=lobe_r, lobe_err=lobe_err,
                                     var_signal_lumen=vsig_l, var_signal_ref=vsig_r,
                                     var_signal_err=float(np.sqrt(np.sum(2 * v_ref[win] ** 2) / nev)),
                                     peak_sample_lumen=int(np.argmax(np.abs(mean - mean[PRE].mean()))),
                                     peak_sample_ref=int(np.argmax(np.abs(m_ref - m_ref[PRE].mean())))))  # fmt: skip
                res[cfg][f"L{L}_s{int(s)}"] = rows
                Path(a.out).write_text(json.dumps(res, indent=1))  # after each case: a stopped job keeps its part
                lit = [r for r in rows if r["n_photons"] > 0 and r["on"]]
                c2, nd = sum(r["chi2_mean"] for r in lit), sum(r["ndf"] for r in lit)
                ar = np.array([(r["area_lumen"] - r["area_ref"]) / max(r["area_err"], 1e-9) for r in lit])
                vr = np.array([r["var_pull_rms"] for r in lit])
                vs = np.array([r["var_signal_ref"] / r["var_signal_lumen"] for r in lit if r["var_signal_lumen"] != 0])
                print(f"{cfg} L{L} s{int(s)}: mean chi2/ndf {c2:.0f}/{nd} = {c2 / max(nd, 1):.3f}; area pulls rms "
                      f"{np.sqrt(np.mean(ar**2)):.2f} mean {ar.mean():+.2f}; per-sample variance pulls rms "
                      f"{np.sqrt(np.mean(vr**2)):.2f}; summed variance ref/lumen median {np.median(vs):.3f}; "
                      f"channels {len(lit)}", flush=True)  # fmt: skip
    Path(a.out).write_text(json.dumps(res, indent=1))


def _response(model, p, expect):
    """lumen's expected response to one photoelectron arriving at an even and at an odd ns (on the sample grid)."""
    resp = []
    for t0 in (0.0, 1.0):
        h = rd.from_photons(model, GATE, 1, dict(sensor=np.zeros(1, int), lam=np.full(1, LAM[1]), time=np.full(1, t0)))
        q = dict(p)
        q["sensors/pmt/qe"] = np.ones(2, np.float32)
        q["sensors/pmt/frontend/baseline"] = np.float32(0.0)
        q["sensors/pmt/frontend/noise"] = np.float32(0.0)
        resp.append(np.asarray(expect(h, q)[0], np.float64))
    return resp


_RESP: dict = {}


def _between(model, p, phot, eff, nev, expect):
    """The variance between the reference's events of their expected waveforms (each from its own photons)."""
    key = (
        p["sensors/pmt/pulse/shape"].tobytes(),
        float(p["sensors/pmt/gain"][0]),
        float(p["sensors/pmt/charge/sigma"]),
    )
    if key not in _RESP:  # the response depends only on the channel's pulse, gain and spread
        _RESP[key] = _response(model, p, expect)
    resp = _RESP[key]
    ev = np.unique(phot["event"], return_inverse=True)[1]
    ti = np.trunc(phot["t"]).astype(int)
    w = np.zeros((2, len(np.unique(ev)), 5000))
    np.add.at(w, (ti % 2, ev, ti // 2), eff[phot["band"]])
    n = 1 << int(np.ceil(np.log2(10000)))
    mu = sum(np.fft.irfft(np.fft.rfft(w[k], n) * np.fft.rfft(resp[k], n), n)[:, :5000] for k in (0, 1))
    full = np.zeros((nev, 5000))
    full[: len(mu)] = mu  # events without a photon here have no light
    d = full - full.mean(0)
    return full.var(0, ddof=1), (d**2).mean(0), (d**4).mean(0)


def _present(phot, eff, nev):
    """P(a pulse is present) per sample: some photoelectron placed in the 550 samples before it."""
    ti = np.trunc(phot["t"]).astype(int)
    lam = np.zeros(5000)
    np.add.at(lam, (1000 + ti) // 2, (1.0 - np.exp(-eff[phot["band"]])) / nev)
    return 1.0 - np.exp(-np.convolve(lam, np.ones(550))[:5000])


if __name__ == "__main__":
    main()
