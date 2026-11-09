"""Write lumen's shipped PMT value sets (and the WCSim-emulating test sets) from the authorities' sources.

    python tools/readout/transcribe.py --wcsim WCSIM_SRC --annie ANNIE_SRC --spline SPLINE_TXT
        --out src/lumen/sensors/data
        --tests tests/references/wcsim_emulation --sbnd-fcl SBNDCODE_FCL --sbnd-calib CALIB.npz --sbnd-tpb timeTPB.npy

WCSIM_SRC: WCSim develop e2dc7a4; ANNIE_SRC: WCSim annie e88314b; SPLINE_TXT: ROOT's TGraph::Eval(Q, 0, "S") of
PMT3inchR14374_WCTE's resolution graph at Q = 0.5, 0.55, ..., 4.0 (WCSimPMTObject.cc:3811-3823, evaluated with the
hk-software image's ROOT). Every array is parsed from the source (comments removed); every field records file:line.

Rules (DECISIONS 40, 46, 55; SPEC 12):
- physical QE: WCSim's table without WCSim's x4/3 (SensitiveDetector_Only) nor WCTE's 1/0.73 correction factor;
  WCSim interpolates linearly, extrapolates linearly below the first knot and returns 0 at or above 660 nm (and
  below 200 nm): knots are added where that differs from Geant4's constant-beyond-the-ends rule.
- physical threshold: the SKI acceptance polynomial; the dark rate is WCSim's rate times its conversion factor (the
  rate before the threshold, which is what lumen simulates).
- timing: WCSim has no per-photoelectron time spread (its smear is per digit, charge dependent). The shipped sets
  use WCSim's smear at 1 pe as a per-photoelectron Gaussian (status unverified); WCTE's 1.5 ns FWHM is its PMT's
  transit-time spread and its charge-dependent resolution becomes the time walk's spread. The WCSim-emulating test
  sets use no per-photoelectron spread and WCSim's per-digit smear as the time walk.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

FWHM_WCSIM = 2.355  # WCSim's FWHM-to-sigma conversion, literally (WCSimPMTObject.cc:3840)
SKI_POLY = ("include/WCSimWCDigitizer.hh", "Threshold")
HC_WCSIM = 2.0 * np.pi * 197.3  # WCSim's wavelength convention, nm eV (WCSimWCSD.cc:202)


class Source:
    def __init__(self, root, rel, tag):
        self.path, self.tag, self.rel = Path(root) / rel, tag, rel
        self.lines = self.path.read_text().splitlines()
        text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), self.path.read_text(), flags=re.S)
        self.clean = [ln.split("//")[0] for ln in text.splitlines()]  # comments blanked, line numbers kept

    def function(self, signature):
        """(first line, last line, body text without comments) of a C++ function, 1-based lines."""
        i0 = next(i for i, ln in enumerate(self.clean) if signature in ln)
        depth, i, opened = 0, i0, False
        while True:
            depth += self.clean[i].count("{") - self.clean[i].count("}")
            opened = opened or "{" in self.clean[i]
            if opened and depth == 0:
                break
            i += 1
        return i0 + 1, i + 1, "\n".join(self.clean[i0 : i + 1])

    def array(self, signature):
        a, b, body = self.function(signature)
        inner = body.split("{", 1)[1]
        vals = re.search(r"\{([^{}]*)\}", inner).group(1)
        corr = re.search(r"correctionFactor\s*=\s*([0-9./]+)\s*;", body)
        nums = [float(x.replace("*correctionFactor", "")) for x in re.split(r"[,\s]+", vals) if x.strip()]
        return np.asarray(nums), f"{self.tag} {self.rel}:{a}-{b}", corr.group(1) if corr else None

    def scalar(self, signature, name):
        a, b, body = self.function(signature)
        return float(re.search(rf"{name}\s*=\s*([0-9.eE+\-]+)", body).group(1))

    def where(self, signature):
        a, b, _ = self.function(signature)
        return f"{self.tag} {self.rel}:{a}-{b}"


def spe_table(src, cls):
    cdf, where, _ = src.array(f"{cls}::Getqpe()")
    assert len(cdf) == 501, (cls, len(cdf))
    pdf = np.clip(np.diff(np.r_[0.0, cdf]), 0.0, None)  # the 0 sentinel at index 500 is never selected
    edges = (np.arange(502) - 50) / 22.83  # rn1pe: q = (i - 50 + u) / 22.83 (WCSimWCPMT.cc:55-77)
    return dict(form="table", edges_pe=edges.round(10).tolist(), pdf=pdf.round(8).tolist()), where


def ski_threshold(src):
    a, b, body = src.function("static void Threshold(")
    expr = re.search(r"std::min\(1\.0,(.*?)\);", body, re.S).group(1)
    coeffs = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*", expr.replace(" ", "").replace("\n", ""))]
    assert len(coeffs) == 11, coeffs
    return dict(form="poly", coefficients=coeffs, offset_pe=0.1, limit_pe=1.1), f"WCSim {src.rel}:{a}-{b}"


def wcsim_qe(wl, qe, lo_cut=200.0, hi_cut=660.0):
    """Knots reproducing WCSimPMTQE's rule: linear, extrapolated below the first knot, 0 at or above hi_cut."""
    wl, qe = list(wl), list(qe)
    while wl[-1] > hi_cut:  # drop knots past the cut, keeping the interpolated value at the cut
        wl.pop(), qe.pop()
    if qe[0] > 0.0:  # WCSim extrapolates the first segment down to zero
        z = wl[0] - qe[0] * (wl[1] - wl[0]) / (qe[1] - qe[0])
        wl, qe = [max(z, lo_cut)] + wl, [0.0] + qe
    if qe[-1] > 0.0:
        wl, qe = wl + [wl[-1] + 0.01], qe + [0.0]
    return [round(x, 4) for x in wl], [round(x, 8) for x in qe]


def wcsim_sigma_knots(c, q_lo=0.5, q_hi=None, floor=0.58, a=0.33):
    """Charge knots and WCSim's sigma(Q) = max(floor, a + sqrt(c / Q)) on them (Q >= 0.5), dense enough for 1e-3 ns."""
    q_hi = q_hi or c / (floor - a) ** 2
    q = np.unique(np.r_[np.geomspace(q_lo, q_hi, 96), q_hi])
    return q.round(6).tolist(), np.maximum(floor, a + np.sqrt(c / q)).round(6).tolist()


def base(name, source, status, readout="hits"):
    return dict(
        schema="lumen-pmt-values/1",
        model="lumen:pmt",
        model_version=1,
        name=name,
        version=1,
        readout=readout,
        provenance=dict(source=source, status=status, fields={}),
    )


def frontend(window, charge_noise=0.03, scale=1.0, step=1.0 / 45.66, qmax=64.0):
    return dict(
        sampling_ns=1.0,
        unit="pe",
        baseline=0.0,
        noise=0.0,
        window_ns=window,
        charge_noise_pe=charge_noise,
        charge_scale=scale,
        charge_step_pe=step,
        charge_max_pe=qmax,
    )


def sk_sets(src, hh):
    wl, w_where, _ = src.array("PMT20inch::GetQEWavelength()")
    qe, q_where, _ = src.array("PMT20inch::GetQE()")
    kn, qv = wcsim_qe(wl, qe)
    ce_line = next(i + 1 for i, ln in enumerate(src.lines) if "static G4double CE[10]={100,100,99" in ln)
    ce = [float(x) / 100 for x in re.search(r"\{(.*)\}", src.lines[ce_line - 1]).group(1).split(",")]
    charge, c_where = spe_table(src, "PMT20inch")
    thr, t_where = ski_threshold(hh)
    rate, conv = src.scalar("PMT20inch::GetDarkRate()", "rate"), src.scalar("PMT20inch::GetDarkRateConversionFactor()",
                                                                             "factor")  # fmt: skip
    s1 = 0.33 + np.sqrt(10.0)
    d = base(
        "sk_r3600",
        "WCSim develop e2dc7a4: PMT20inch (src/WCSimPMTObject.cc) and the SKI digitizer "
        "(include/WCSimWCDigitizer.hh)",
        "transcribed",
    )
    d.update(qe=dict(wavelength_nm=kn, values=qv), ce=dict(theta_deg=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
             values=ce), efficiency=1.0, t_offset_ns=0.0, tts=dict(form="gaussian", sigma_ns=round(float(s1), 6)),
             charge=charge, gain=1.0, threshold=thr, dark_rate_hz=round(rate * conv * 1e3, 6),
             frontend=frontend(200.0), ranges={"tts/sigma": [2.0, 10.0], "t_offset": [-10.0, 10.0]})  # fmt: skip
    d["provenance"]["fields"] = dict(
        qe=f"{w_where}; {q_where}: physical QE, without SensitiveDetector_Only's x4/3 (WCSimWCSD.cc:213); WCSim's "
        "rule (WCSimPMTQE.cc:41-80): linear, 0 at or above 660 nm",
        ce=f"WCSim develop e2dc7a4 src/WCSimPMTObject.cc:{ce_line}: the SKDetSim table (commented in WCSim), "
        "transcribed "
        "from WCSim src/WCSimPMTObject.cc:70-71; theta = acos(|z_local| / r_local) of the hit point",
        tts=f"{src.where('PMT20inch::HitTimeSmearing(')}: WCSim's per-digit smear 0.33 + sqrt(10 / Q) ns at Q = 1 pe, "
        "used as a per-photoelectron Gaussian (WCSim has no per-photoelectron time spread)",
        charge=f"{c_where} (rn1pe: WCSimWCPMT.cc:55-77)", threshold=f"{t_where}: the SKI acceptance",
        dark_rate=f"{src.where('PMT20inch::GetDarkRate()')} x {src.where('PMT20inch::GetDarkRateConversionFactor()')}",
        frontend="WCSimWCDigitizer.hh:97-100 (window 200 ns, no dead time); charge noise 0.03 pe: Threshold "
        "(WCSimWCDigitizer.hh:115); WCSim's 0.985 charge factor (WCSimWCDigitizer.cc:262) and 0.4 ns time "
        "truncation are its conventions, not physical values")  # fmt: skip
    d["provenance"]["warning"] = ("timing unverified until transcribed: the per-photoelectron time spread is WCSim's "
                                  "per-digit smear at 1 pe")  # fmt: skip
    return d, s1


def wcte_sets(src, hh, spline):
    wl, w_where, _ = src.array("PMT3inchR14374_WCTE::GetQEWavelength()")
    qe, q_where, corr = src.array("PMT3inchR14374_WCTE::GetQE()")
    n = int(src.scalar("PMT3inchR14374_WCTE::GetNbOfQEDefined()", "factor"))
    kn, qv = wcsim_qe(wl[: n + 1], qe[: n + 1])
    charge, c_where = spe_table(src, "PMT3inchR14374_WCTE")
    thr, t_where = ski_threshold(hh)
    rate = src.scalar("PMT3inchR14374_WCTE::GetDarkRate()", "rate")
    conv = src.scalar("PMT3inchR14374_WCTE::GetDarkRateConversionFactor()", "factor")
    q, r = np.loadtxt(spline, unpack=True)
    tw = dict(charge_pe=q.round(4).tolist(), shift_ns=[0.0] * len(q), sigma_ns=(r / FWHM_WCSIM).round(8).tolist())
    ts = src.where("PMT3inchR14374_WCTE::HitTimeSmearing(")
    d = base("wcte_r14374", "WCSim develop e2dc7a4: PMT3inchR14374_WCTE (src/WCSimPMTObject.cc) and the SKI "
             "digitizer", "transcribed")  # fmt: skip
    d.update(qe=dict(wavelength_nm=kn, values=qv), ce=dict(theta_deg=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
             values=[0.95] * 10), efficiency=1.0, t_offset_ns=0.0,
             tts=dict(form="gaussian", sigma_ns=round(1.5 / FWHM_WCSIM, 8)), time_walk=tw, charge=charge, gain=1.0,
             threshold=thr, dark_rate_hz=round(rate * conv * 1e3, 6), frontend=frontend(200.0),
             ranges={"tts/sigma": [0.5, 3.0], "t_offset": [-10.0, 10.0]})  # fmt: skip
    d["provenance"]["fields"] = dict(
        qe=f"{w_where}; {q_where}: the first {n + 1} points (GetNbOfQEDefined = {n}), without the correction factor "
        f"{corr} nor x4/3; WCSim's rule: linear, extrapolated to 0 below 300 nm, 0 at or above 660 nm",
        ce=f"{src.where('PMT3inchR14374_WCTE::GetCollectionEfficiencyArray()')}: 95 % flat",
        tts=f"{ts}: pmt_tts = 1.5 ns FWHM / 2.355", time_walk=f"{ts}: the resolution graph R(min(Q, 4)) "
        "(WCSimPMTObject.cc:3811-3823) / 2.355, ROOT's spline evaluation at 0.05 pe steps; sigma at max(Q, 0.5) "
        "(WCSimWCDigitizer.cc:347)", charge=c_where, threshold=t_where,
        dark_rate=f"{src.where('PMT3inchR14374_WCTE::GetDarkRate()')} x the conversion factor {conv}",
        frontend="as sk_r3600")  # fmt: skip
    return d


ANNIE = {"R7081": ("PMT_R7081", 3.4), "D784KFLB": ("PMT_D784KFLB", 1.98), "R5912HQE": ("PMT_R5912HQE", 2.4),
         "R7081HQE": ("PMT_R7081HQE", 3.4)}  # fmt: skip
ANNIE_QE_RATIO = 1.025  # the reference's tuning (annie tuning_parameters.mac, QERatio; options_s0.txt: QERatio 1.025)


def annie_set(src, ty):
    cls, _ = ANNIE[ty]
    wl, w_where, _ = src.array(f"{cls}::GetQEWavelength()")
    qe, q_where, _ = src.array(f"{cls}::GetQE()")
    a, b, body = src.function(f"{cls}::HitTimeSmearing(")
    c = float(re.search(r"timingConstant\s*=\s*([0-9.]+)", body).group(1))
    charge, c_where = spe_table(src, cls)
    rate = src.scalar(f"{cls}::GetDarkRate()", "rate")
    conv = src.scalar(f"{cls}::GetDarkRateConversionFactor()", "factor")
    s1 = 0.33 + np.sqrt(c)
    d = base(f"wcsim_annie_{ty}", f"WCSim annie e88314b: {cls} (src/WCSimPMTObject.cc); the SKI digitizer without "
             "photon integration or threshold (src/WCSimWCDigitizer.cc:296-345)", "transcribed")  # fmt: skip
    d.update(
        qe=dict(wavelength_nm=[float(x) for x in wl], values=[round(float(x) * ANNIE_QE_RATIO, 8) for x in qe]),
        ce=None,
        efficiency=1.0,
        t_offset_ns=0.0,
        tts=dict(form="gaussian", sigma_ns=round(float(s1), 6)),
        charge=charge,
        gain=1.0,
        threshold=None,
        dark_rate_hz=round(rate * conv * 1e3, 6),
        frontend=frontend(0.0, charge_noise=0.0),
        ranges={"tts/sigma": [1.0, 10.0], "t_offset": [-10.0, 10.0]},
    )
    d["provenance"]["fields"] = dict(
        qe=f"{w_where}; {q_where}; times the tuning's QEratio {ANNIE_QE_RATIO} (Multi_Tank_Types: min(QE QEratio, "
        "maxQE_all QEratio), the cap never binding); 0 outside (280, 660) nm in WCSim",
        tts=f"WCSim annie {src.rel}:{a}-{b}: 0.33 + sqrt({c} / Q) ns at Q = 1 pe as a per-photoelectron Gaussian "
        "(WCSim applies it per photoelectron at that photoelectron's own charge, WCSimWCPMT.cc:172-174)",
        charge=c_where,
        dark_rate=f"{src.where(f'{cls}::GetDarkRate()')} x {conv}",
        frontend="each photoelectron its own hit (DoPhotonIntegration false), no threshold (commented out)",
    )
    d["provenance"]["warning"] = "timing unverified: WCSim's per-photoelectron smear at 1 pe"
    return d, c


def emulation(d, sigma_c=None, tw=None, scale=0.985, window=None):
    """A WCSim-emulating test set: no per-photoelectron spread, WCSim's per-digit smear as the time walk."""
    e = json.loads(json.dumps(d))
    e["name"] = d["name"] + "_wcsim_emulation"
    e["provenance"] = dict(
        source=d["provenance"]["source"],
        status="transcribed",
        note="test reference only: WCSim's conventions (0.985, per-digit smear), not physical",
    )
    e["tts"] = dict(form="gaussian", sigma_ns=0.0)
    e["ranges"]["tts/sigma"] = [0.0, 10.0]
    if sigma_c is not None:
        q, s = wcsim_sigma_knots(sigma_c)
        e["time_walk"] = dict(charge_pe=q, shift_ns=[0.0] * len(q), sigma_ns=s)
    if tw is not None:
        e["time_walk"] = tw
    e["frontend"]["charge_scale"] = scale
    return e


def fcl_values(path, keys):
    """{key: (value, line)} of `key: value` lines of an fcl file."""
    out = {}
    for i, ln in enumerate(Path(path).read_text().splitlines(), 1):
        m = re.match(r"\s*(\w+):\s*([-+0-9.eE]+)", ln.split("#")[0])
        if m and m.group(1) in keys and m.group(1) not in out:
            out[m.group(1)] = (float(m.group(2)), i)
    return out


def sbnd_sets(fcl_dir, calib, tpb_path):
    """sbnd_pmt (coated) and sbnd_pmt_uncoated from digi_pmt_sbnd.fcl, DigiPMTSBNDAlg and sbnd_data."""
    keys = (
        "PMTRiseTime",
        "PMTFallTime",
        "PMTMeanAmplitude",
        "TransitTime",
        "PMTChargeToADC",
        "TTS",
        "CableTime",
        "PMTBaseline",
        "PMTBaselineRMS",
        "PMTDarkNoiseRate",
        "PMTCoatedVUVEff_tpc0",
        "PMTCoatedVISEff_tpc0",
        "PMTUncoatedEff_tpc0",
    )
    f = fcl_values(Path(fcl_dir) / "digi_pmt_sbnd.fcl", keys)
    ln = {k: f"digi_pmt_sbnd.fcl:{v[1]}" for k, v in f.items()}
    v = {k: x[0] for k, x in f.items()}
    width = np.sqrt(2.0) * (np.sqrt(-np.log(0.1)) - np.sqrt(-np.log(0.9)))  # 10-90 % (DigiPMTSBNDAlg.cc:117-118)
    cal = np.load(calib)
    rec = cal["reconstruct_channel"] > 0
    rel = float(np.mean(cal["spe_amp_std"][rec] / cal["spe_amp"][rec]))
    tpb = np.load(tpb_path)
    edges = HC_EDGES
    out = {}
    for name, eff in (
        ("sbnd_pmt", (v["PMTCoatedVUVEff_tpc0"], v["PMTCoatedVISEff_tpc0"])),
        ("sbnd_pmt_uncoated", (0.0, v["PMTUncoatedEff_tpc0"])),
    ):
        d = base(
            name,
            "sbndcode v10_26_00: digi_pmt_sbnd.fcl, DigiPMTSBNDAlg.cc; sbnd_data v01_42_00",
            "transcribed",
            readout="waveform",
        )
        d.update(qe=dict(band_edges_nm=list(edges), values=list(eff)), ce=None, efficiency=1.0,
                 t_offset_ns=v["CableTime"], tts=dict(form="gaussian", sigma_ns=round(v["TTS"] / 2.3548200450309493,
                                                                                       8)),
                 delay=dict(form="table", edges_ns=list(range(len(tpb) + 1)), pdf=[float(x) for x in tpb]),
                 charge=dict(form="gauss_exp", sigma_pe=round(rel, 6), fraction=0.0, tau_pe=1.0),
                 gain=round(v["PMTChargeToADC"] * v["PMTMeanAmplitude"], 6), threshold=None,
                 pulse=dict(form="bigauss", t_peak_ns=v["TransitTime"],
                            sigma_rise_ns=round(v["PMTRiseTime"] / width, 6),
                            sigma_fall_ns=round(v["PMTFallTime"] / width, 6)),
                 dark_rate_hz=v["PMTDarkNoiseRate"],
                 frontend=dict(sampling_ns=2.0, unit="ADC", baseline=v["PMTBaseline"], noise=v["PMTBaselineRMS"],
                               window_ns=0.0, charge_noise_pe=0.0, charge_scale=1.0, charge_step_pe=0.25,
                               charge_max_pe=64.0),
                 ranges={"tts/sigma": [1.0, 3.0], "t_offset": [-10.0, 10.0], "pulse/t_peak": [40.0, 70.0],
                         "pulse/sigma_rise": [0.5, 3.0], "pulse/sigma_fall": [2.0, 12.0]})  # fmt: skip
        d["provenance"]["fields"] = dict(
            qe=f"{ln['PMTCoatedVUVEff_tpc0']}, {ln['PMTCoatedVISEff_tpc0']}, {ln['PMTUncoatedEff_tpc0']}: detection "
            "efficiency per real photon (VUV below 6.19 eV's 200.3 nm, visible above); an uncoated tube reads no VUV",
            tts=f"{ln['TTS']}: 2.4 ns FWHM, Gaussian (DigiPMTSBNDAlg.hh:158 FWHM to sigma)",
            delay="sbnd_data OpDetSim/digi_pmt_sbnd_v2int0.root timeTPB: the measured TPB emission time in 1 ns bins "
            "(Phys. Rev. C 91, 035503), on every PMT photon (DigiPMTSBNDAlg.cc:273, 314, 334)",
            pulse=f"{ln['TransitTime']}, {ln['PMTRiseTime']}, {ln['PMTFallTime']}: the ideal SPE (DigiPMTSBNDAlg.cc "
            "Pulse1PE: two half-Gaussians about the transit time, sigma from the 10-90 % times)",
            gain=f"{ln['PMTChargeToADC']} x {ln['PMTMeanAmplitude']}: ADC at the SPE's peak",
            charge="PMTGaussianGainFluctuation: N(1, (spe_amp_std / spe_amp)^2) per photoelectron; the mean of "
            "sbnd_data CalibrationDatabase/pds_calibration.db (tag v3r1) over its calibrated channels",
            dark_rate=ln["PMTDarkNoiseRate"],
            frontend=f"{ln['PMTBaseline']}, {ln['PMTBaselineRMS']}; 2 ns sampling",
        )
        d["provenance"]["warning"] = ("sbndcode v10_26_00 adds the TPB time in [0, 1) ns (CLHEP::RandGeneral of the "
                                      "table); this set uses the table in ns; production SPE shapes and gains are "
                                      "per-channel calibrations, not these defaults")  # fmt: skip
        out[name] = d
    return out


def microboone_set():
    d = base(
        "microboone_qe",
        "lumen-b1 (rebuild/next 880b220c) lumen/detectors/microboone.py:7-8, 20: MicroBooNE's "
        "electronics simulation QE per physical photon",
        "transcribed",
        readout="pe",
    )
    d.update(
        qe=dict(band_edges_nm=[100.0, 900.0], values=[0.0093]),
        ce=None,
        efficiency=1.0,
        t_offset_ns=0.0,
        tts=None,
        charge=None,
        gain=1.0,
        threshold=None,
        dark_rate_hz=0.0,
        frontend=dict(sampling_ns=1.0, unit="pe", window_ns=0.0, charge_step_pe=0.25, charge_max_pe=4.0),
    )
    return d


HC_EDGES = (100.0, 1239.841984 / 6.19, 900.0)  # nm: the 6.19 eV split of SBND's PhotonHitConverter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wcsim", required=True)
    ap.add_argument("--annie", required=True)
    ap.add_argument("--spline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--sbnd-fcl", required=True)
    ap.add_argument("--sbnd-calib", required=True)
    ap.add_argument("--sbnd-tpb", required=True)
    a = ap.parse_args()
    src, hh = Source(a.wcsim, "src/WCSimPMTObject.cc", "WCSim"), Source(a.wcsim, "include/WCSimWCDigitizer.hh", "")
    asrc = Source(a.annie, "src/WCSimPMTObject.cc", "WCSim annie")
    out, tests = Path(a.out), Path(a.tests)
    out.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    sk, _ = sk_sets(src, hh)
    wcte = wcte_sets(src, hh, a.spline)
    sets = {"sk_r3600": sk, "wcte_r14374": wcte}
    emu = {"sk": emulation(sk, sigma_c=10.0)}
    tw = dict(wcte["time_walk"])
    tw["sigma_ns"] = (np.sqrt(1.5**2 + (np.asarray(tw["sigma_ns"]) * FWHM_WCSIM) ** 2) / FWHM_WCSIM).round(8).tolist()
    emu["wcte"] = emulation(wcte, tw=tw)
    for ty in ANNIE:
        d, c = annie_set(asrc, ty)
        sets[d["name"]] = d
        emu[f"annie_{ty}"] = emulation(d, sigma_c=c)
    sets.update(sbnd_sets(a.sbnd_fcl, a.sbnd_calib, a.sbnd_tpb))
    sets["microboone_qe"] = microboone_set()
    for name, d in sets.items():
        (out / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n")
    for name, d in emu.items():
        (tests / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n")
    print("wrote", sorted(sets), "and", sorted(emu))


if __name__ == "__main__":
    main()
