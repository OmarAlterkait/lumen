"""The readout port's gate: lumen's readout against B1's own readout code on B1's stored deposit histograms.

    python tools/readout/check_port.py GOLDEN_DIR B1_LUMEN_DIR OUT_JSON

GOLDEN_DIR holds a6/<set>/hist.npz (B1's tallies: Q (sensor, band, node), the late / early cells); B1_LUMEN_DIR is
B1's ``lumen`` package (its response.py and readout.py are loaded as the reference). For each set the same inputs
go through both:
- the photoelectrons per sensor (B1 fold_qe over the bands vs lumen.readout.pe);
- the binned photoelectron density over a gate (B1 response.readout vs lumen.readout.pe_density), with the set's
  emission profile, a Gaussian time spread and a transit offset;
- the first-hit window terms (mu, U, W) at a time per sensor, their windowed first-arrival likelihood with dark light,
  and the Poisson charge likelihood (B1 readout.first_hit_terms / first_arrival_nll / poisson_nll_dark).
Reported per output: bitwise equal or not, the largest absolute and relative difference.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lumen import readout as rd
from lumen.readout import hits as hm
from lumen.sensors import components as comp
from lumen.sensors import pmt

SETS = {  # emission profile (fractions, decay, rise ns), TTS sigma (ns), transit offset (ns), dark rate (per ns)
    "sk": (((1.0,), (0.0,), (0.0,)), 1.0, 3.0, 4.2e-6),
    "wcte": (((1.0,), (0.0,), (0.0,)), 0.64, 1.0, 1.0e-6),
    "annie": (((1.0,), (0.0,), (0.0,)), 1.2, 2.0, 3.0e-6),
    "sbnd": (((0.23, 0.77), (6.0, 1300.0), (0.0, 0.0)), 1.02, 55.1, 1.0e-6),
    "microboone": (((0.23, 0.77), (6.0, 1590.0), (0.0, 0.0)), 1.0, 0.0, 0.0),
    "toy_vx3_ABC": (((1.0,), (0.0,), (0.0,)), 0.3, 0.5, 0.0),
}


def load_b1(root):
    """Return B1's response and readout modules, loaded from its package directory under their own names."""
    pkg = types.ModuleType("b1lumen")
    sys.modules["b1lumen"] = pkg
    out = {}
    for name in ("response", "readout"):
        spec = importlib.util.spec_from_file_location(f"b1lumen.{name}", Path(root) / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"b1lumen.{name}"] = mod
        if name == "readout":  # its `from lumen import response`: give it B1's
            src = (
                (Path(root) / "readout.py")
                .read_text()
                .replace("from lumen import response", "from b1lumen import response")
            )
            exec(compile(src, str(Path(root) / "readout.py"), "exec"), mod.__dict__)
        else:
            spec.loader.exec_module(mod)
        setattr(pkg, name, mod)
        out[name] = mod
    sys.modules["b1lumen"] = pkg
    return out["response"], out["readout"]


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    scale = np.maximum(np.abs(b).max(), 1e-30)
    return dict(bitwise=bool(np.array_equal(a, b)), max_abs=float(d.max()), max_rel=float(d.max() / scale),
                shape=list(a.shape))  # fmt: skip


def band_model(edges, sigma, n_out):
    """A lumen:pmt model with QE per band (all 1) and the gate's Gaussian time spread, for the comparison."""
    d = dict(schema=pmt.SCHEMA, model="lumen:pmt", model_version=1, name="port_gate", version=1, readout="hits",
             provenance=dict(source="the port gate", status="measured"),
             qe=dict(band_edges_nm=list(edges), values=[1.0] * (len(edges) - 1)), ce=None, efficiency=1.0,
             t_offset_ns=0.0, tts=dict(form="gaussian", sigma_ns=sigma), charge=dict(form="gauss_exp", sigma_pe=0.3,
             fraction=0.0, tau_pe=1.0), gain=1.0, threshold=None, dark_rate_hz=0.0,
             frontend=dict(sampling_ns=1.0, unit="pe", baseline=0.0, noise=0.0, window_ns=200.0, charge_noise_pe=0.0,
                           charge_scale=1.0, charge_step_pe=0.25, charge_max_pe=4.0),
             ranges={"tts/sigma": [sigma, sigma], "t_offset": [-100.0, 100.0]})  # fmt: skip
    return pmt.model_from_dict(d)


def check_set(rs, ro, z, name):
    profile, sigma, T0, dark = SETS[name]
    meta = json.loads(str(z["meta"]))["grid"]
    Q, ql, tl = (jnp.asarray(z[k], jnp.float32) for k in ("Q", "Q_late", "Tw_late"))
    qe_, te = jnp.asarray(z["Q_early"], jnp.float32), jnp.asarray(z["Tw_early"], jnp.float32)
    S, L, nt = Q.shape
    g1 = rs.Grid(lam_edges=tuple(meta["lam_edges"]), t0=meta["t0"], dt=meta["dt"], nt=meta["nt"], split=meta["split"])
    g2 = rd.Grid(lam_edges=tuple(meta["lam_edges"]), t0=meta["t0"], dt=meta["dt"], nt=meta["nt"], split=meta["split"])
    out = {}
    # photoelectrons: per band charge (nodes + overflow), QE 1 per band
    D = Q.sum(-1) + ql + qe_
    qe_vals = jnp.linspace(0.5, 1.0, L).astype(jnp.float32)
    m = band_model(meta["lam_edges"], sigma, nt)
    b1_pe = rs.fold_qe(D, qe_vals)
    h_all = rd.Histograms(D[None, :, :, None], Q.sum(1)[None], jnp.stack([ql.sum(1), tl.sum(1)], -1)[None],
                         jnp.stack([qe_.sum(1), te.sum(1)], -1)[None], profile, g2, m, "pmt",
                         (meta["t0"], meta["t0"] + nt * meta["dt"]))  # fmt: skip
    p = {k: np.asarray(v, np.float32) for k, v in m.defaults("pmt", S).items()}
    p["sensors/pmt/qe"] = np.asarray(qe_vals)
    out["pe"] = compare(rd.pe(h_all, p), b1_pe)
    # the binned photoelectron density, per band summed (band 0's own nodes as B1 folds them)
    Hb, lateb = Q[:, 0, :], (ql[:, 0], tl[:, 0])
    b1_wave = rs.readout(Hb, g1, profile=profile, sigma=sigma, T0=T0, b0=meta["t0"], width=meta["dt"], n_bins=nt,
                         late=lateb)  # fmt: skip
    tot = Hb.sum(-1) + ql[:, 0] + jnp.zeros_like(ql[:, 0])
    h0 = rd.Histograms(tot[None, :, None, None], Hb[None], jnp.stack([ql[:, 0], tl[:, 0]], -1)[None],
                      jnp.zeros((1, S, 2), jnp.float32), profile, g2, band_model(meta["lam_edges"][:2], sigma, nt),
                      "pmt", (meta["t0"], meta["t0"] + nt * meta["dt"]))  # fmt: skip
    p0 = {k: np.asarray(v, np.float32) for k, v in h0.model.defaults("pmt", S).items()}
    p0["sensors/pmt/t_offset"] = np.full(S, T0, np.float32)
    out["pe_density"] = compare(rd.pe_density(h0, p0).rate, b1_wave)
    # first-hit terms at each sensor's mean arrival time + 2 ns
    mt = (Hb * (g1.t0 + g1.dt * jnp.arange(nt))).sum(-1) / jnp.maximum(Hb.sum(-1), 1e-30)
    tq = mt + T0 + 2.0
    a = ro.first_hit_terms(Hb, g1, tq, 1.0, profile=profile, sigma=sigma, T0=T0, late=lateb)
    b = hm.first_hit_terms(Hb, g2, tq, 1.0, lambda x: comp.profile_cdf(x, profile, sigma),
                          lambda x: comp.profile_sf(x, profile, sigma), T0=T0, late=lateb)  # fmt: skip
    for k, x, y in zip(("mu", "U", "W"), b, a, strict=False):
        out[f"first_hit_{k}"] = compare(x, y)
    rng = np.random.default_rng(0)
    n = jnp.asarray(rng.poisson(np.asarray(a[0])), jnp.float32)
    gate = (float(g1.t0), float(g1.t0 + nt * g1.dt))
    a_nll = ro.first_arrival_nll(*a, n, dark, tq - 0.5, gate, 1.0)
    b_nll = hm.first_arrival_nll(*b, n, dark, tq - 0.5, gate, 1.0)
    out["first_arrival_nll"] = compare(b_nll, a_nll)
    pn = hm.poisson_nll_dark(a[0], n, dark, gate)
    out["poisson_nll_dark"] = compare(pn, ro.poisson_nll_dark(a[0], n, dark, gate))
    return out


def check_cells(rs):
    """B1's response.cells against charge_cells (wavelength knots, no time) and time_cells (one band, nodes)."""
    rng = np.random.default_rng(5)
    n = 4096
    tag = jnp.asarray(rng.integers(-1, 50, n), jnp.int32)
    lam = jnp.asarray(rng.uniform(250.0, 700.0, n), jnp.float32)
    t = jnp.asarray(rng.uniform(-5.0, 130.0, n), jnp.float32)
    knots = (280.0, 300.0, 350.0, 420.0, 500.0, 660.0)
    out = {}
    for split in ("cubic", "linear"):
        g1 = rs.Grid(lam_knots=knots)
        a = jax.vmap(lambda tg, lm, tt: rs.cells(dict(tag=tg, lam=lm, time=tt), g1))(tag, lam, t)
        dep = rd.grid._Photons(tag, lam, jnp.zeros(n), t)
        b = jax.vmap(lambda d: rd.charge_cells(d, rd.Grid(lam_knots=knots)))(dep)
        out[f"charge_cells_idx_{split}"], out[f"charge_cells_w_{split}"] = compare(b[0], a[0]), compare(b[1], a[1])
        g1 = rs.Grid(t0=0.0, dt=0.5, nt=240, split=split)
        a = jax.vmap(lambda tg, lm, tt: rs.cells(dict(tag=tg, lam=lm, time=tt), g1))(tag, lam, t)
        b = jax.vmap(lambda d: rd.time_cells(d, rd.Grid(t0=0.0, dt=0.5, nt=240, split=split)))(dep)
        for k, i in (("idx", 0), ("w", 1), ("oidx", 2), ("ow", 3)):
            out[f"time_cells_{k}_{split}"] = compare(b[i], a[i])
    return out


def main():
    golden, b1, out = sys.argv[1:4]
    rs, ro = load_b1(b1)
    res = {}
    for name in SETS:
        f = Path(golden) / "a6" / name / "hist.npz"
        if f.exists():
            res[name] = check_set(rs, ro, np.load(f), name)
            print(name, {k: (v["bitwise"], f"{v['max_rel']:.2e}") for k, v in res[name].items()})
    res["cells"] = check_cells(rs)
    print("cells", {k: v["bitwise"] for k, v in res["cells"].items()})
    res["_device"] = str(jax.devices()[0])
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
