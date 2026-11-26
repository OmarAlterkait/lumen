"""The overlap check: g4ref's overlap toys through lumen's step against Geant4 10.7.4's runs of the same toys.

    rows = overlaps.check("coinc_sib")

Each toy (the coinc_sib set: boxes of different media placed coincident or partly overlapping in both orders, among
filler daughters, and a daughter overlapping a sibling) is converted from its GDML with no configuration: the world
kept, the slab ``sensor_lv`` counting every photon entering it. A pencil beam along ``direction`` (isotropic photons
for the toys listed as such, as Geant4 ran them) starts at ``source_mm``. The arrival time says which medium filled
which part of the path, the detected fraction which interfaces were met. Per toy: the detected fraction and the mean
arrival time, each against Geant4's within errors (|z| <= 3; lumen's errors from its photons' second moments of the
totals, Geant4's binomial and the spread of its times). A toy Geant4 detects nothing in passes when lumen detects
less than one of Geant4's photons.
"""

from __future__ import annotations

import json

import numpy as np

import lumen
from lumen import _env, api, sources
from lumen.compare import photons, records, references
from lumen.format.io import sha256_file
from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe
from lumen.rules import navigation as rules
from lumen.tools import stats
from lumen.units import HC

PHOTONS = 1 << 21
SEED = 3


def toy_detector(gdml, spec):
    """Return the toy's detector, converted once into the cache (keyed by the GDML's content)."""
    group = SensorGroup("slab", rule="any_entry", lv=spec["sensor_lv"], settings={"entry_efficiency": 1.0})
    recipe = Recipe(gdml.stem, spec["world"], groups=(group,))
    out = _env.cache_dir() / "validate" / "toys" / f"{gdml.stem}-{sha256_file(gdml).removeprefix('sha256:')[:16]}.lumen"
    return lumen.load(api._convert_once(gdml, recipe, out))  # a cache entry: concurrent runs place it once


def geant4(d, toy, launched):
    """Return Geant4's (detected fraction, its error, mean time, its error, hits) of one toy."""
    with np.load(d / f"{toy}.npz") as z:
        t = z["t"].astype(np.float64)
    n = launched[toy]
    f = len(t) / n
    sd = t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else np.nan
    return f, np.sqrt(max(f * (1 - f), 1.0 / n) / n), (t.mean() if len(t) else np.nan), sd, len(t)


def lumen_toy(det, spec, isotropic):
    """Return lumen's detected fraction and its error, mean time and its error, and the detected weight of one toy
    (``lumen.simulate`` at its defaults; the errors from the totals' per-photon moments, the mean time's by the delta
    method of the ratio of its two totals)."""  # noqa: D205, D209
    lam = HC / spec["energy_ev"]
    x = np.asarray(spec["source_mm"]) * 1e-3
    src = sources.Point(x, n_photons=float(PHOTONS), lam=lam, direction=None if isotropic else spec["direction"])
    diag = photons.Detected(len(det.parts.optics.sensor_pvs), (-np.inf, np.inf))
    light = lumen.simulate(det, src, samples=PHOTONS, chunk=PHOTONS, seed=SEED, _jump=False, diagnostics=(diag,))
    c = photons.detected(light, diag)
    q, tw = float(c["total"][0]), float(c["Tw_total"][0])
    f, ef = q / PHOTONS, np.sqrt(max(c["total_var"][0], 0.0)) / PHOTONS
    t = tw / q if q > 0 else np.nan
    vt = (c["Tw_total_var"][0] - 2.0 * t * c["cov_total"][0] + t * t * c["total_var"][0]) / q**2 if q > 0 else 0.0
    health = {k: float(np.sum(getattr(light.health, k))) for k in ("lost", "escaped")}
    return f, ef, t, np.sqrt(max(vt, 0.0)), q, health


def check(name, *, extra=()) -> list:
    """Run the overlap check named ``name`` (a manifest entry); return its record rows."""
    spec = references.spec("overlaps", name, extra)
    d = references.product(spec["reference"], spec, extra)
    launched = json.loads((d / "launched.json").read_text())
    rows = []
    for toy in spec["toys"]:
        det = toy_detector(d / f"{toy}.gdml", spec)
        f, ef, tl, etl, q, health = lumen_toy(det, spec, toy in spec["isotropic"])
        fg, eg, tg, etg, n_g4 = geant4(d, toy, launched)
        z = (f - fg) / np.hypot(ef, eg) if np.hypot(ef, eg) > 0 else 0.0
        rows.append(records.row("overlaps", toy, "detected fraction", f, ef, z=z, passed=abs(z) <= stats.Z_MAX,
                                reference=spec["reference"], geant4=fg, geant4_error=eg, geant4_hits=n_g4,
                                **health))  # fmt: skip
        if n_g4 and q > 0:
            et = np.hypot(etl, etg)
            # no Monte Carlo error on either side (every photon one path): they agree to float32's resolution or not
            zt = (tl - tg) / et if et > 0 else (0.0 if abs(tl - tg) <= rules.BAND_ULPS * rules.FLOAT32.eps * abs(tg)
                                      else np.inf)  # fmt: skip
            rows.append(records.row("overlaps", toy, "mean arrival time (ns)", tl, etl, z=zt,
                                    passed=abs(zt) <= stats.Z_MAX, reference=spec["reference"], geant4=tg,
                                    geant4_error=etg))  # fmt: skip
    return rows
