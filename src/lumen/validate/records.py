"""The JSON record every ``lumen validate`` run writes: one row per checked quantity, read back by the report.

    rows = [records.row("charge", "sk", "ratio", 0.999, 0.0005, z=-1.8, passed=True, reference="wcsim-sk-e2dc7a48")]
    path = records.write(rows, out_dir, "charge_sk")      # out_dir/charge_sk.json; never inside a reference store

Each row has the fields of ``FIELDS``: the check, the detector (or the reference's name, e.g. "wcte_g4"), the
quantity, its value, error and z (None where they do not apply), whether it passes, the reference product id, the
commits (lumen, tallymap, jax), the GPU (or "cpu") and the date; ``extra`` holds anything else a check reports.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FIELDS = ("check", "detector", "quantity", "value", "error", "z", "pass", "reference", "commits", "gpu", "date")


def _tree(path) -> str:
    """Return the git commit of the tree lumen is imported from (with "+dirty"), or "" outside a git checkout."""
    import subprocess

    def git(*a):
        r = subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""

    sha = git("rev-parse", "--short=12", "HEAD")
    return sha + ("+dirty" if sha and git("status", "--porcelain", "--untracked-files=no") else "")


def environment() -> dict:
    """Return the commits (lumen's version and source tree, tallymap, jax), the device and the date a run records."""
    import jax

    import lumen

    try:
        import tallymap

        tm = getattr(tallymap, "__version__", "unknown")
    except ImportError:
        tm = None
    dev = jax.devices()[0]
    tree = _tree(Path(lumen.__file__).parent)
    return dict(commits=dict(lumen=lumen.__version__ + (f" (tree {tree})" if tree else ""), tallymap=tm,
                             jax=jax.__version__),
                gpu=dev.device_kind if dev.platform == "gpu" else "cpu",
                date=time.strftime("%Y-%m-%d %H:%M"))  # fmt: skip


def _plain(v):
    if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def row(check, detector, quantity, value, error=None, *, z=None, passed=None, reference=None, **extra) -> dict:  # noqa: PLR0913 -- the record's fields
    """Return one record row (``environment()`` is filled in by ``write``)."""
    return dict(check=check, detector=detector, quantity=quantity, value=_plain(value), error=_plain(error),
                z=_plain(z), **{"pass": None if passed is None else bool(passed)}, reference=reference,
                extra=_plain(extra))  # fmt: skip


def write(rows, out_dir, name) -> Path:
    """Write the rows (with the run's environment) to ``out_dir/<name>.json``; return its path.

    Refuses a directory inside a reference store: check outputs never overwrite reference products.
    """
    from lumen.validate import references

    out = Path(out_dir)
    if references.inside_store(out):
        raise PermissionError(f"{out} is inside a reference store: write check outputs elsewhere (--out)")
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    doc = [dict(r, **env) for r in rows]
    path = out / f"{name}.json"
    tmp = path.with_suffix(f".json.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)
    return path


def read(paths) -> list:
    """Return the rows of record files and directories of them (``*.json``), in name order."""
    rows = []
    for p in map(Path, paths):
        for f in sorted(p.glob("*.json")) if p.is_dir() else [p]:
            rows += json.loads(f.read_text())
    return rows
