"""Summarise check_sbnd.py results per case over the bright tubes: lobe ratio, chi^2, signal variance.

python tools/readout/summarize_sbnd.py RESULT.json [...]
"""

import json
import sys

import numpy as np

for f in sys.argv[1:]:
    d = json.load(open(f))
    for cfg, cases in d.items():
        for case, rows in cases.items():
            lit = [r for r in rows if r["on"] and r["n_photons"] >= 50 and r["lobe_lumen"] < 0]
            if not lit:
                continue
            lp = np.array([(r["lobe_ref"] - r["lobe_lumen"]) / r["lobe_err"] for r in lit])
            # photon-weighted mean lobe ratio and its error
            w = np.array([r["lobe_lumen"] for r in lit])
            rr = np.array([r["lobe_ref"] for r in lit])
            e = np.array([r["lobe_err"] for r in lit])
            tot = rr.sum() / w.sum()
            tot_err = np.sqrt((e**2).sum()) / abs(w.sum())
            vs = np.array([r["var_signal_ref"] / r["var_signal_lumen"] for r in lit if r["var_signal_lumen"]])
            c2 = sum(r["chi2_mean"] for r in lit) / sum(r["ndf"] for r in lit)
            vp = np.array([r["var_pull_rms"] for r in lit])
            pk = np.array([r["peak_sample_ref"] - r["peak_sample_lumen"] for r in lit])
            print(
                f"{cfg:13s} {case:7s} n={len(lit):3d} chi2/ndf {c2:5.2f} | lobe ref/lumen {tot:.4f}+-{tot_err:.4f} "
                f"pulls rms {np.sqrt(np.mean(lp**2)):.2f} mean {lp.mean():+.2f} | signal var ref/lumen median "
                f"{np.median(vs):.3f} var pulls rms {np.sqrt(np.mean(vp**2)):.2f} | peak |dk| {np.abs(pk).max()}"
            )
