"""The g4ref runs a reference manifest describes: one per product whose producer has a ``run`` block.

    python tools/g4ref/runs.py MANIFEST [NAME ...]     # one line per run (default every run):
    NAME  NCHUNKS  TIME  GDML  CONFIG  SENS_REGEX  SEED  KEEP_GDML  ORIGINAL  [g4ref flags ...]
    python tools/g4ref/runs.py MANIFEST --run NAME     # that run's ``run`` block, as JSON

A product's producer holds how it was made (``nchunks``, ``seed``, ``sens_regex``, ``flags``) and its ``run``: the
GDML and config paths under LUMEN_SITE_DATA and the run's Slurm time, all a rerun needs (and ``flags`` when a rerun
passes other flags than the recorded run did). NAME is the product's id
without its g4ref binary's sha256 prefix (8 digits: a rerun's product carries its own binary's). KEEP_GDML is yes when
the product keeps a copy of the GDML (its ``inputs`` name it), no when another product holds it. ORIGINAL is the run of
the same sources on the original GDML when this run is on a detector's overlap-free export
(``producer.export.original``, without its sha256 prefix), else "-". tools/g4ref/references.sh and make_store.sh read
these lines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _name(pid: str, sha: str) -> str:
    return pid[: -len(sha) - 1] if pid.endswith("-" + sha) else pid


def runs(manifest) -> dict:
    """Return {NAME: (product entry, its producer)} of every product of ``manifest`` with a ``run`` block."""
    out = {}
    for pid, e in json.loads(Path(manifest).read_text())["products"].items():
        p = e.get("producer") or {}
        if p.get("tool") == "g4ref" and "run" in p:
            out[_name(pid, p["binary_sha256"][:8])] = (e, p)
    return out


def row(name: str, e: dict, p: dict) -> str:
    """Return a run's line (see the module docstring)."""
    r = p["run"]
    keep = "yes" if Path(r["gdml"]).name in (e.get("inputs") or {}) else "no"
    export = p.get("export") or {}
    original = _name(export["original"], p["binary_sha256"][:8]) if export.get("original") else "-"
    return " ".join([name, str(p["nchunks"]), r["time"], r["gdml"], r["config"], p["sens_regex"], str(p["seed"]), keep,
                     original, *r.get("flags", p.get("flags", []))])  # fmt: skip


def main(argv=None) -> int:
    """Print the runs (or one run's block) of a manifest (see the module docstring)."""
    args = list(sys.argv[1:] if argv is None else argv)
    table = runs(args.pop(0))
    if args[:1] == ["--run"]:
        print(json.dumps(table[args[1]][1]["run"]))
        return 0
    missing = [n for n in args if n not in table]
    if missing:
        print(f"no run {missing} in the manifest (its runs: {sorted(table)})", file=sys.stderr)
        return 2
    for name, (e, p) in table.items():
        if not args or name in args:
            print(row(name, e, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
