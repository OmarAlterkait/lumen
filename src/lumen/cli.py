"""The ``lumen`` command: ``lumen validate`` runs a check and writes its JSON record; ``report`` makes the report.

    lumen validate charge sk [--detector-file sk.lumen] [--out DIR]          # every check: --references MANIFEST.json
    lumen validate gradients sbnd [--reverse]
    lumen validate geometry sk --g4ref BINARY
    lumen validate overlaps coinc_sib
    lumen validate report DIR [DIR ...] [--out docs/validation/report.md]
    lumen validate list                                                      # the checks every manifest lists

A check's name is a manifest entry (``lumen.validate.references``): lumen's own detectors, or any detector package's
under ``lumen.detectors``, or a manifest given with ``--references``. Records go to ``--out`` (default
``./validation``), never into a reference store. The exit status is 0 when every gated row passes.
"""

from __future__ import annotations

import argparse
import sys
import time

CHECKS = ("charge", "gradients", "geometry", "overlaps")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lumen", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="run a validation check, or make the report")
    v.add_argument("check", choices=(*CHECKS, "report", "list"))
    v.add_argument("names", nargs="*", help="manifest entries (a check), or record files and directories (report)")
    v.add_argument("--references", action="append", default=[], help="another reference manifest (JSON)")
    v.add_argument("--detector-file", help="the detector file to load in place of the entry's detector")
    v.add_argument("--out", help="the records' directory (report: the report's file)")
    v.add_argument("--photons-factor", type=float, help="charge: lumen's photons over the reference's")
    v.add_argument("--replicas", type=int, help="the independent sub-runs that give the errors")
    v.add_argument("--photons", type=int, help="gradients: photons per replica")
    v.add_argument("--reverse", action="store_true", help="gradients: also reverse mode against forward mode")
    v.add_argument("--only", help="gradients: comma-separated prefixes of the parameter families to check")
    v.add_argument("--g4ref", help="geometry: the g4ref binary (Geant4 10.7.4), or a probe CSV made elsewhere")
    return ap


def _run(a, name):
    from lumen.validate import charge, geometry, gradients, toys

    common = dict(extra=a.references)
    if a.check == "charge":
        return charge.check(name, detector=a.detector_file, photons_factor=a.photons_factor, replicas=a.replicas,
                            **common)  # fmt: skip
    if a.check == "gradients":
        only = tuple(a.only.split(",")) if a.only else ()
        log = lambda s: sys.stderr.write(s + "\n")  # noqa: E731 -- one line per family, as it finishes
        return gradients.check(name, detector=a.detector_file, photons=a.photons, replicas=a.replicas,
                               reverse=a.reverse, only=only, log=log, **common)  # fmt: skip
    if a.check == "geometry":
        return geometry.validate(name, detector=a.detector_file, g4ref=a.g4ref, **common)
    return toys.check(name, replicas=a.replicas, **common)


def validate(a) -> int:
    """Run ``lumen validate`` (see the module docstring); return the exit status."""
    from lumen.validate import records, references, report

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
        rows = _run(a, name)
        path = records.write(rows, a.out or "validation", f"{a.check}_{name}")
        gated = [r for r in rows if r["pass"] is not None]
        bad = [r["quantity"] for r in gated if not r["pass"]]
        ok &= not bad
        failing = "; failing: " + ", ".join(bad[:8]) if bad else ""
        sys.stdout.write(f"{a.check} {name}: {len(gated) - len(bad)}/{len(gated)} pass ({time.time() - t0:.0f} s) -> "
                         f"{path}{failing}\n")  # fmt: skip
    return 0 if ok else 1


def main(argv=None) -> int:
    """Run the ``lumen`` command line."""
    a = _parser().parse_args(argv)
    return validate(a)


if __name__ == "__main__":
    sys.exit(main())
