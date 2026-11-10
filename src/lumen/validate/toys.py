"""The overlap check: g4ref's overlap toys through lumen's step against Geant4 10.7.4's runs of the same toys.

    rows = toys.check("coinc_sib")

Each toy (the coinc_sib set: boxes of different media placed coincident or partly overlapping in both orders, among
filler daughters, and a daughter overlapping a sibling) is converted from its GDML with no configuration: the world
kept, the slab ``sensor_lv`` counting every photon entering it. A pencil beam along ``direction`` (isotropic photons
for the toys listed as such, as Geant4 ran them) starts at ``source_mm``. The arrival time says which medium filled
which part of the path, the detected fraction which interfaces were met. Per toy: the detected fraction and the mean
arrival time, each against Geant4's within errors (|z| <= 3; lumen's errors from replicas, Geant4's binomial and the
spread of its times). A toy Geant4 detects nothing in passes when lumen detects less than one of Geant4's photons.
"""

from __future__ import annotations

import json

import numpy as np

from lumen import run as runner
from lumen import sources
from lumen.transport import step as steps
from lumen.units import HC
from lumen.validate import photons, records, references, stats

PHOTONS = 1 << 21
REPLICAS = 16
BAND = 5.0  # nm: the band about the toys' wavelength
CONTROLS = steps.Controls(t_max=10.0, roulette=0.0)  # the overlap check's settings: 10 m reach, no roulette
GUARD = 2000  # vertices per photon (the toys' photons cross a handful of faces)


def toy_detector(gdml, spec):
    """Return the toy's detector, converted once into the cache (keyed by the GDML's content)."""
    import hashlib

    import lumen
    from lumen import _env
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe

    lam = HC / spec["energy_ev"]
    group = SensorGroup("slab", "any_entry", "lumen:counting", lv=spec["sensor_lv"], settings={"eff": 1.0})
    recipe = Recipe(gdml.stem, spec["world"], (lam - BAND, lam + BAND), (group,))
    out = (
        _env.cache_dir()
        / "validate"
        / "toys"
        / f"{gdml.stem}-{hashlib.sha256(gdml.read_bytes()).hexdigest()[:16]}.lumen"
    )
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        lumen.convert(gdml, recipe, out)
    return lumen.load(out)


def geant4(d, toy, launched):
    """Return Geant4's (detected fraction, its error, mean time, its error, hits) of one toy."""
    with np.load(d / f"{toy}.npz") as z:
        t = z["t"].astype(np.float64)
    n = launched[toy]
    f = len(t) / n
    sd = t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else np.nan
    return f, np.sqrt(max(f * (1 - f), 1.0 / n) / n), (t.mean() if len(t) else np.nan), sd, len(t)


def lumen_toy(det, spec, isotropic, replicas):
    """Return lumen's per-replica detected fraction and mean time of one toy."""
    s = photons.setup(det)
    lam = HC / spec["energy_ev"]
    x = np.asarray(spec["source_mm"]) * 1e-3
    key = photons.raw_key(3)
    src = sources.Point(x, n_photons=float(PHOTONS), lam=lam, direction=None if isotropic else spec["direction"])
    rows = src.emit(None, photons.raw_key(4), PHOTONS)
    ops = photons.operands(s, photons.packed(det), controls=CONTROLS)
    t = runner.run(item_spec(s), s.model, ops, rows, key, max_vertices=GUARD, replicas=replicas)
    q = np.asarray(t.outputs["Q"], np.float64).reshape(replicas, -1).sum(1)
    tw = np.asarray(t.outputs["Tw"], np.float64).reshape(replicas, -1).sum(1)
    return (
        q / (PHOTONS / replicas),
        tw,
        q,
        dict(lost=float(t.outputs["lost"][0]), escaped=float(t.outputs["escaped"][0])),
    )


def item_spec(s):
    """Return the default item of a toy's setup."""
    from lumen.transport import item

    return item.photons(s.tally)


def check(name, *, replicas=None, extra=()) -> list:
    """Run the overlap check named ``name`` (a manifest entry); return its record rows."""
    spec = references.spec("overlaps", name, extra)
    k = int(replicas or REPLICAS)
    d = references.product(spec["reference"], spec, extra)
    launched = json.loads((d / "launched.json").read_text())
    rows = []
    for toy in spec["toys"]:
        det = toy_detector(d / f"{toy}.gdml", spec)
        fk, tw, q, health = lumen_toy(det, spec, toy in spec["isotropic"], k)
        fg, eg, tg, etg, n_g4 = geant4(d, toy, launched)
        f, ef = fk.mean(), stats.sem(fk)
        z = (f - fg) / np.hypot(ef, eg) if np.hypot(ef, eg) > 0 else 0.0
        rows.append(records.row("overlaps", toy, "detected fraction", f, ef, z=z, passed=abs(z) <= stats.Z_MAX,
                                reference=spec["reference"], geant4=fg, geant4_error=eg, geant4_hits=n_g4,
                                **health))  # fmt: skip
        if n_g4 and q.sum() > 0:
            tk = tw / np.where(q > 0, q, 1.0)
            tl, etl = float(tw.sum() / q.sum()), stats.sem(tk)
            zt = (tl - tg) / np.hypot(etl, etg)
            rows.append(records.row("overlaps", toy, "mean arrival time (ns)", tl, etl, z=zt,
                                    passed=abs(zt) <= stats.Z_MAX, reference=spec["reference"], geant4=tg,
                                    geant4_error=etg))  # fmt: skip
    return rows
