"""The Monte Carlo gate: arrivals and photon-loop derivatives on fixed seeds, bit for bit against a reference record.

A change that should not move results moves none of them.

    python -m lumen.compare gate --package PKG [NAME ...] [--out DIR] [--against DIR]
    python -m lumen.compare mc run DET.lumen PHOTONS.npz OUT.npz --keys-from REF.npz [--seed0 0 --k 16 --n 65536]
        [--roulette EPS] [--speed exact|fast]
    python -m lumen.compare mc compare REF.npz OUT.npz          # exit 0 when every array is bitwise identical
    python -m lumen.compare mc agree REF.npz OUT.npz            # exit 0 when the arrays agree statistically

``gate`` runs every ``mc`` entry of a detector package's manifest (its ``REFERENCES``; the entries ``NAME``, else all):
it fetches the entry's reference product (``lumen.compare.references.product``: every file's sha256 checked against
the manifest), converts the package's detector (``lumen.convert("PKG:DETECTOR")``) into OUT/NAME.lumen, writes its
record OUT/NAME.npz on the reference record's keys, seeds and photon count, with the run's environment
(``lumen.compare.records.environment``), prints the environment's differences from the reference record's and then
the comparison; exit 0 when every record is bitwise identical to its reference. ``--against DIR`` compares with
DIR/NAME.npz instead of the product's record (a re-baseline's check, docs/dev/validation.md).

The source is a photon file (``FromArrays``: fixed emission rows ``x, d, t, lam, w, er, ei``), so both sides trace
the same light; each of K seeds is one independent run. Per seed: the arrivals, and for each family key (the
reference record's ``keys``) the forward derivative of the arrivals along p -> p (1 - th) at th = 0. The record holds
``keys``, ``primal`` (K, S), ``tangent`` (keys, K, S), ``names`` (the sensors in arrival order), ``n``, ``seed0``,
``roulette`` (the step's ε of every run: ``lumen.Controls``' default unless the caller passes its own) and
``environment`` (JSON).
``compare`` prints, per array, whether it is bitwise identical, how many cells differ and the totals' relative change
with its paired standard error. ``agree`` (two runs that should agree in expectation, not in bits: fast against exact,
or the same code on other seeds) tests every array with ``lumen.tools.stats.mean_test``, the comparisons' one test, on
``GROUPS`` groups of consecutive sensors and the total (paired on the same seeds, else independent), at ALPHA over the
record's arrays; paired, a difference within the runs' float32 rounding is no failure (the checks' one resolution,
``stats.FD_ROUNDING`` of each output, the floor of ``mean_test``: two builds on the same seeds may differ by
deterministic float32 offsets of a few ulp per vertex); an array without power is reported as such, never as agreement,
and the records agree when some array agrees and none differs (``stats.gate``).
A chi2 / ndf band over the groups would fail heavy tails:
MicroBooNE's LAr RAYLEIGH derivative, exact against exact on 512 + 512 seeds, gave a groups' mean z^2 of 1.55.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import lumen
from lumen.format.calibrations import Table
from lumen.sources.arrays import FromArrays
from lumen.tools import stats
from lumen.transport.step import Controls

GROUPS = 20  # sensor groups of ``agree`` (consecutive sensors in arrival order), as the gradient check's


def record(det, photons, keys, *, seed0: int = 0, k: int = 16, n: int = 65536, controls=None, jump=True,  # noqa: PLR0913 -- the gate's settings
           log=None) -> dict:  # fmt: skip
    """Return the gate's record of a loaded detector on a photon file (see the module docstring)."""
    import jax
    import jax.numpy as jnp

    ctl = controls if controls is not None else Controls()
    p = det.params()
    ph = np.load(photons)
    src = FromArrays(ph["x"], ph["d"], ph["t"], ph["lam"], ph["w"], polarisation=np.stack([ph["er"], ph["ei"]], 1))
    missing = [key for key in keys if key not in p]
    if missing:
        raise KeyError(f"the reference's keys are not this detector's: {missing}")
    prim, tan = [], [[] for _ in keys]
    t0 = time.time()
    for seed in range(seed0, seed0 + k):
        prim.append(np.asarray(lumen.simulate(det, src, p, samples=n, seed=seed, controls=ctl, _jump=jump).arrivals))
        for j, key in enumerate(keys):
            v0 = p[key]

            def f(th, key=key, v0=v0, seed=seed):
                if isinstance(v0, Table):
                    q = p.set(key, Table(jnp.asarray(v0.energy), jnp.asarray(v0.value) * (1.0 - th)))
                else:
                    q = p.set(key, jnp.asarray(v0, jnp.float32) * (1.0 - th))
                return lumen.simulate(det, src, q, samples=n, seed=seed, controls=ctl, _jump=jump).arrivals

            tan[j].append(np.asarray(jax.jvp(f, (jnp.float32(0.0),), (jnp.float32(1.0),))[1]))
        if log:
            log(f"seed {seed} done {time.time() - t0:.0f} s")
    return dict(keys=np.array(keys), primal=np.array(prim), tangent=np.array(tan).reshape(len(keys), k, -1), n=n,
                seed0=seed0, roulette=float(ctl.roulette), names=np.array(det.parts.optics.sensor_pvs))  # fmt: skip


def compare(ref: dict, new: dict) -> tuple[bool, list]:
    """Return (whether every array is bitwise identical, one line per array) of two records on the same seeds."""
    if list(ref["names"]) != list(new["names"]) or list(ref["keys"]) != list(new["keys"]):
        return False, ["the sensors or the keys differ"]
    lines, same = [], True
    pairs = [("arrivals", ref["primal"], new["primal"])]
    pairs += [(f"d/d({key})", ref["tangent"][i], new["tangent"][i]) for i, key in enumerate(ref["keys"])]
    for tag, x, y in pairs:
        eq = x.shape == y.shape and bool(np.array_equal(x, y))
        same &= eq
        tx, ty = x.sum(-1), y.sum(-1)  # per-seed totals
        rel = (ty.mean() - tx.mean()) / abs(tx.mean()) if tx.mean() else 0.0
        se = np.std(ty - tx, ddof=1) / np.sqrt(len(tx)) / abs(tx.mean()) if tx.mean() else 0.0
        nd = int(np.sum(x != y)) if x.shape == y.shape else x.size
        lines.append(f"{tag:60s} {'bitwise identical' if eq else 'DIFFERS'} ({nd} of {x.size} cells); total "
                     f"{tx.mean():.6g} -> {ty.mean():.6g}, rel {rel:+.3e} (paired sem {se:.1e})")  # fmt: skip
    return same, lines


def environment_changes(ref: dict, new: dict) -> list:
    """Return one line per environment entry (``records.environment``'s versions and device) that differs between two
    records (a record without one: none)."""  # noqa: D205, D209
    if "environment" not in ref or "environment" not in new:
        return []
    flat = lambda e: {**e.get("commits", {}), "gpu": e.get("gpu")}  # noqa: E731
    a, b = flat(json.loads(str(ref["environment"]))), flat(json.loads(str(new["environment"])))
    return [f"environment {k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def gate(package: str, names=(), *, out="validation/mc", against=None, data=None, roulette=None,  # noqa: PLR0913
         jump: bool = True) -> int:  # fmt: skip
    """Run the gate of a detector package's ``mc`` entries (see the module docstring); return the exit status."""
    import importlib
    from pathlib import Path

    from lumen.compare import records, references

    extra = [str(importlib.import_module(package).REFERENCES)]
    entries = [references.spec("mc", n, extra) for n in (names or references.names("mc", extra))]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctl = None if roulette is None else lumen.Controls(roulette=roulette)
    same_all = True
    for e in entries:
        d = references.product(e["reference"], e, extra)
        ref = dict(np.load(Path(against) / f"{e['name']}.npz" if against else d / e["record"]))
        det = lumen.load(lumen.convert(f"{package}:{e['detector']}", out=out / f"{e['name']}.lumen", data=data,
                                       overwrite=True))  # fmt: skip
        rec = record(det, d / e["photons"], [str(k) for k in ref["keys"]], seed0=int(ref["seed0"]),
                     k=len(ref["primal"]), n=int(ref["n"]), controls=ctl, jump=jump, log=print)  # fmt: skip
        rec["environment"] = json.dumps(records.environment())
        np.savez(out / f"{e['name']}.npz", **rec)
        same, lines = compare(ref, rec)
        same_all &= same
        sys.stdout.write("\n".join([f"== {e['name']}", *environment_changes(ref, rec), *lines]) + "\n")
    return 0 if same_all else 1


def _groups(x, ng=GROUPS):
    idx = np.array_split(np.arange(x.shape[-1]), min(ng, x.shape[-1]))
    return np.concatenate([np.stack([x[..., i].sum(-1) for i in idx], -1), x.sum(-1, keepdims=True)], -1)


def agree(ref: dict, new: dict) -> tuple[bool, list]:
    """Return (whether every array agrees, one line per array) of two records (``mean_test``; see the docstring)."""
    if list(ref["names"]) != list(new["names"]) or list(ref["keys"]) != list(new["keys"]):
        return False, ["the sensors or the keys differ"]
    paired = int(ref["seed0"]) == int(new["seed0"]) and ref["primal"].shape == new["primal"].shape
    pairs = [("arrivals", ref["primal"], new["primal"])]
    pairs += [(f"d/d({key})", ref["tangent"][i], new["tangent"][i]) for i, key in enumerate(ref["keys"])]
    results, lines = [], []
    for tag, x, y in pairs:
        gx, gy = _groups(np.asarray(x, np.float64)), _groups(np.asarray(y, np.float64))
        floor = stats.FD_ROUNDING * np.maximum(abs(gx.mean(0)), abs(gy.mean(0))) if paired else None
        t = stats.mean_test(gx, gy, paired=paired, tests=len(pairs), floor=floor)
        results.append(t["passed"])
        verdict = "no power" if t["passed"] is None else "agrees" if t["passed"] else "DIFFERS"
        lines.append(f"{tag:60s} {verdict} (largest score {t['stat']:.2f} over {t['n']} outputs, p {t['p']:.3g} "
                     f"against {t['level']:.2g}; {'paired' if paired else 'independent'})")  # fmt: skip
    return stats.gate(results), lines


def main(argv=None) -> int:
    """Run ``python -m lumen.compare mc`` (see the module docstring); return the exit status."""
    ap = argparse.ArgumentParser(prog="python -m lumen.compare mc", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="write a detector's record")
    r.add_argument("det")
    r.add_argument("photons")
    r.add_argument("out")
    r.add_argument("--keys-from", required=True, help="the reference record whose keys to differentiate")
    r.add_argument("--seed0", type=int, default=0)
    r.add_argument("--k", type=int, default=16)
    r.add_argument("--n", type=int, default=65536)
    r.add_argument("--roulette", type=float, help="the step's roulette ε (default: lumen.Controls' own)")
    r.add_argument("--speed", default="fast", choices=("fast", "exact"),
                   help="exact: the plain Rayleigh walk instead of the diffusion jump (validation)")  # fmt: skip
    for name, what in (("compare", "compare a record with its reference, bit for bit"),
                       ("agree", "test two records' agreement in expectation (fast or other seeds)")):  # fmt: skip
        c = sub.add_parser(name, help=what)
        c.add_argument("ref")
        c.add_argument("new")
    a = ap.parse_args(argv)
    if a.command == "run":
        from lumen.compare import records

        keys = [str(x) for x in np.load(a.keys_from)["keys"]]
        ctl = None if a.roulette is None else lumen.Controls(roulette=a.roulette)
        out = record(lumen.load(a.det), a.photons, keys, seed0=a.seed0, k=a.k, n=a.n, controls=ctl,
                     jump=a.speed == "fast", log=lambda s: sys.stdout.write(s + "\n"))  # fmt: skip
        np.savez(a.out, **out, environment=json.dumps(records.environment()))
        sys.stdout.write(f"wrote {a.out}\n")
        return 0
    ref, new = dict(np.load(a.ref)), dict(np.load(a.new))
    ok, lines = (agree if a.command == "agree" else compare)(ref, new)
    sys.stdout.write("\n".join([*environment_changes(ref, new), *lines]) + "\n")
    return 0 if ok else 1
