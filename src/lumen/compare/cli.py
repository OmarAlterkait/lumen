"""``python -m lumen.compare``: the one command of the validation harness; a check writes its JSON record.

    python -m lumen.compare charge sk --detector-file sk.lumen [--out DIR]       # every check: --references MANIFEST
    python -m lumen.compare gradients sbnd [--[no-]reverse]
    python -m lumen.compare geometry sk --g4ref BINARY                           # a manifest entry
    python -m lumen.compare geometry-file DET.lumen --gdml FILE --g4ref BINARY --out DIR   # lumen.compare.geometry
    python -m lumen.compare overlaps coinc_sib
    python -m lumen.compare report DIR [DIR ...] [--out docs/validation/report.md]
    python -m lumen.compare list                                                 # the checks every manifest lists
    python -m lumen.compare charge sk --package lumen_wc                         # a detector package's manifest
    python -m lumen.compare gate --package lumen_wc [sk ...]                     # the MC gate (lumen.compare.mc)
    python -m lumen.compare mc run|compare|agree ...                             # its records (lumen.compare.mc)
    python -m lumen.compare export DET.lumen OUT.gdml                            # lumen.compare.ownership

A check's name is a manifest entry (``lumen.compare.references``): lumen's own detectors, or a manifest given with
``--references`` or by a detector package (``--package PKG``: its ``REFERENCES``). Every check takes its own
flags only (``--help`` per check). Records go to ``--out`` (default
``./validation``), never into a reference store. The exit status is 0 when some row passes and none fails
(``lumen.tools.stats.gate``: a check with nothing gated fails).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lumen.compare import charge, geometry, gradients, mc, overlaps, ownership, records, references, report
from lumen.tools import stats

CHECKS = ("charge", "gradients", "geometry", "overlaps")


def _parser() -> argparse.ArgumentParser:
    """Return the parser of the manifest-driven commands: one sub-parser per check, each with its own flags only."""
    v = argparse.ArgumentParser(prog="python -m lumen.compare", description=__doc__.splitlines()[0],
                                epilog="also: gate, mc, export, geometry-file (each with its own --help)")  # fmt: skip
    sub = v.add_subparsers(dest="check", required=True)

    def check(name, help_):
        c = sub.add_parser(name, help=help_)
        c.add_argument("names", nargs="*", help="manifest entries (default: every entry of the check)")
        c.add_argument("--references", action="append", default=[], help="another reference manifest (JSON)")
        c.add_argument("--package", action="append", default=[],
                       help="a detector package whose manifest (REFERENCES) to read")  # fmt: skip
        c.add_argument("--out", help="the records' directory")
        return c

    c = check("charge", "charge per sensor against a reference run")
    c.add_argument("--detector-file", help="the path of the entry's detector file (lumen.load takes a path)")
    c.add_argument("--photons-factor", type=float, help="lumen's photons over the reference's")

    c = check("gradients", "the photon loop's derivatives against finite differences")
    c.add_argument("--detector-file", help="the path of the entry's detector file (lumen.load takes a path)")
    c.add_argument("--seeds", type=int, help="the independent runs (seeds) that give the errors")
    c.add_argument("--photons", type=int, help="photons per seed")
    c.add_argument("--reverse", action=argparse.BooleanOptionalAction,
                   help="also reverse mode (default: the manifest's)")  # fmt: skip
    c.add_argument("--only", help="comma-separated prefixes of the parameter families to check")
    c.add_argument("--shard", help="I/N, run the families j with j %% N == I (one job each, on many GPUs)")
    c.add_argument("--merge", action="store_true", help="merge the shards' records into the check's")

    c = check("geometry", "points located as Geant4 locates them, on the entry's GDML")
    c.add_argument("--detector-file", help="the path of the entry's detector file (lumen.load takes a path)")
    c.add_argument("--g4ref", help="the g4ref binary (Geant4 10.7.4)")

    check("overlaps", "the overlap toys")

    r = sub.add_parser("report", help="write the report of the records in DIR [DIR ...]")
    r.add_argument("names", nargs="*", help="record files and directories (default: validation)")
    r.add_argument("--references", action="append", default=[], help="another reference manifest (JSON)")
    r.add_argument("--package", action="append", default=[], help="a detector package whose manifest to read")
    r.add_argument("--out", help="the report's file")

    li = sub.add_parser("list", help="the checks every manifest lists")
    li.add_argument("--references", action="append", default=[], help="another reference manifest (JSON)")
    li.add_argument("--package", action="append", default=[], help="a detector package whose manifest to read")
    li.set_defaults(names=[], out=None)
    return v


def _run(a, name):
    common = dict(extra=a.references)
    if a.check == "charge":
        return charge.check(name, detector=a.detector_file, photons_factor=a.photons_factor, **common)
    if a.check == "gradients":
        only = tuple(a.only.split(",")) if a.only else ()
        log = lambda s: sys.stderr.write(s + "\n")  # noqa: E731 -- one line per family, as it finishes
        shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
        return gradients.check(name, detector=a.detector_file, photons=a.photons, seeds=a.seeds,
                               reverse=a.reverse, only=only, log=log, shard=shard, **common)  # fmt: skip
    if a.check == "geometry":
        return geometry.validate(name, detector=a.detector_file, g4ref=a.g4ref, **common)
    return overlaps.check(name, **common)


def validate(a) -> int:
    """Run a check or make the report (see the module docstring); return the exit status."""
    if a.check == "list":
        for c in CHECKS:
            sys.stdout.write(f"{c}: {' '.join(references.names(c, a.references))}\n")
        return 0
    if a.check == "report":
        path = report.write(records.read(a.names or ["validation"]), a.out or "validation/report.md")
        sys.stdout.write(f"{path}\n")
        return 0
    ok = True
    for name in a.names or references.names(a.check, a.references):
        t0 = time.time()
        out, tag = a.out or "validation", f"{a.check}_{name}"
        if getattr(a, "merge", False):
            from lumen.compare import gradients

            files = sorted(Path(out).glob(f"{tag}_shard*.json"))
            rows = gradients.merge(name, [records.read([f]) for f in files])
        else:
            rows = _run(a, name)
            if getattr(a, "shard", None):
                tag += "_shard{}of{}".format(*a.shard.split("/"))
        path = records.write(rows, out, tag, a.references)
        gated = [r for r in rows if r["pass"] is not None]
        bad = [r["quantity"] for r in gated if not r["pass"]]
        ok &= stats.gate(r["pass"] for r in rows)  # nothing gated fails: a check never passes on nothing
        failing = "; failing: " + ", ".join(bad[:8]) if bad else ""
        sys.stdout.write(f"{a.check} {name}: {len(gated) - len(bad)}/{len(gated)} pass ({time.time() - t0:.0f} s) -> "
                         f"{path}{failing}\n")  # fmt: skip
    return 0 if ok else 1


def package_references(packages) -> list:
    """Return the reference manifests of detector packages (each package's ``REFERENCES``), as paths."""
    import importlib

    return [str(importlib.import_module(p).REFERENCES) for p in packages]


def _gate(argv) -> int:
    g = argparse.ArgumentParser(prog="python -m lumen.compare gate", description=mc.__doc__.splitlines()[0])
    g.add_argument("names", nargs="*", help="the package's mc entries (default: all)")
    g.add_argument("--package", required=True, help="the detector package (its REFERENCES and DETECTORS)")
    g.add_argument("--out", default="validation/mc", help="the records' and detector files' directory")
    g.add_argument("--against", help="compare with DIR/NAME.npz instead of the reference product's record")
    g.add_argument("--data", help="the package's data directory (default: LUMEN_DATA)")
    g.add_argument("--roulette", type=float, help="the step's roulette ε (default: lumen.Controls' own)")
    g.add_argument("--speed", default="fast", choices=("fast", "exact"),
                   help="exact: the plain Rayleigh walk instead of the diffusion jump (validation)")  # fmt: skip
    a = g.parse_args(argv)
    return mc.gate(a.package, a.names, out=a.out, against=a.against, data=a.data, roulette=a.roulette,
                   jump=a.speed == "fast")  # fmt: skip


def main(argv=None) -> int:
    """Run ``python -m lumen.compare`` (see the module docstring)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    head = argv[:1]
    if head == ["gate"]:
        return _gate(argv[1:])
    if head == ["mc"]:
        return mc.main(argv[1:])
    if head == ["export"]:
        return ownership.main(argv[1:])
    if head == ["geometry-file"]:  # the check on a detector file and its GDML, not a manifest entry
        return geometry.main(argv[1:])
    a = _parser().parse_args(argv)
    a.references = [*a.references, *package_references(a.package)]
    return validate(a)
