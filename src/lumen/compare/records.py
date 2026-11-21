"""The JSON record every ``python -m lumen.compare`` run writes: one row per checked quantity, read back by the report.

    rows = [records.row("charge", "sk", "ratio", 0.999, 0.0005, z=-1.8, passed=True, reference="wcsim-sk-e2dc7a48")]
    path = records.write(rows, out_dir, "charge_sk")      # out_dir/charge_sk.json; never inside a reference store

Each row has the fields of ``FIELDS``: the check, the detector (or the reference's name, e.g. "wcte_g4"), the
quantity, its value, error and z (None where they do not apply), whether it passes, the reference product id, the
commits (lumen, tallymap, jax), the GPU (or "cpu") and the date; ``extra`` holds anything else a check reports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lumen
from lumen import _cache, run
from lumen.compare import references

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
    """Return what a run's numbers depend on beyond the code: the versions (lumen's and its source tree, tallymap, jax,
    jaxlib, the CUDA runtime JAX runs and the NVRTC tallymap compiles with), the device and the date."""  # noqa: D205, D209
    from importlib import metadata

    import jax
    import jaxlib

    dev = jax.devices()[0]
    tree = _tree(Path(lumen.__file__).parent)
    gpu = dev.platform == "gpu"
    nvrtc = next((d.version for d in metadata.distributions() if (d.metadata["Name"] or "").startswith(
        "nvidia-cuda-nvrtc")), "none")  # fmt: skip
    cuda = " ".join(dev.client.platform_version.split()) if gpu else "none"
    return dict(commits=dict(lumen=lumen.__version__ + (f" (tree {tree})" if tree else ""),
                             tallymap=run.tallymap_version(), jax=jax.__version__, jaxlib=jaxlib.__version__,
                             cuda=cuda, nvrtc=nvrtc),
                gpu=dev.device_kind if gpu else "cpu", date=time.strftime("%Y-%m-%d %H:%M"))  # fmt: skip


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


def write(rows, out_dir, name, extra=()) -> Path:
    """Write the rows to ``out_dir/<name>.json``; return its path.

    A row gets the run's environment unless it carries its own: a row read back from a record (as ``gradients.merge``
    combines shards) keeps the environment it ran in.

    Refuses a directory inside a reference store (``LUMEN_DATA``, or the store of lumen's manifest or of an ``extra``
    one): check outputs never overwrite reference products.
    """
    out = Path(out_dir)
    if references.inside_store(out, extra):
        raise PermissionError(f"{out} is inside a reference store: write check outputs elsewhere (--out)")
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    doc = [dict(env, **r) for r in rows]
    path = out / f"{name}.json"
    _cache.write_bytes(path, (json.dumps(doc, indent=1) + "\n").encode())  # a unique temporary name, moved into place
    return path


def read(paths) -> list:
    """Return the rows of record files and directories of them (``*.json``), in name order."""
    rows = []
    for p in map(Path, paths):
        for f in sorted(p.glob("*.json")) if p.is_dir() else [p]:
            rows += json.loads(f.read_text())
    return rows
