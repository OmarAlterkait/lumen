"""Checks a detector package's tests share: its value sets, its recipes' readout gates and its reference manifest.

A detector package (lumen-wc, lumen-tpc, ...) exposes ``DETECTORS`` (name: ``lumen.DetectorEntry``) and
``REFERENCES`` (its manifest's path); its tests call these and keep only their own numbers:

    m = lumen.testing.check_value_set("lumen_wc:sk_r3600")       # loads, with a QE table; returns the sensor model
    timed = lumen.testing.check_readout_gates(lumen_wc.DETECTORS)  # every timed group gated; the names with one
    gate = lumen.testing.check_references_gate(lumen_wc.REFERENCES)   # {charge entry: gated}; no entry scales

``histograms`` gives the histograms of a photon list (a sensor model's grid is ``model.grid(gate)``), for a package's
own readout numbers. ``notebooks``, ``check_notebooks_clean`` and ``run_notebook`` run a package's notebooks:

    for path in lumen.testing.notebooks(ROOT / "notebooks"):          # NN_*.ipynb, in order
        lumen.testing.run_notebook(path, work)                      # a fresh kernel in ``work``; written with outputs
"""

from __future__ import annotations

import warnings

from lumen.api import DetectorEntry
from lumen.compare import references as refs
from lumen.readout.grid import from_photons
from lumen.sensors import group_model, load_model


def load_value_set(name: str):
    """Return the sensor model of a value set (``"package:set"``), its warnings (timing, provenance notes) silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_model(name)


def _check(ok: bool, message) -> None:
    if not ok:
        raise AssertionError(message)


def check_value_set(name: str):
    """Check that a value set loads (its provenance, readout and tables checked by the loader), has a QE table and
    gives a readout grid over a wide gate; return its sensor model."""  # noqa: D205, D209
    m = load_value_set(name)
    _check("qe" in dict(m.tables), f"value set {name!r}: no qe table")
    m.grid((-400.0, 1100.0))
    return m


def check_readout_gates(detectors: dict) -> set:
    """Check that every group with a timed sensor model (hits, waveforms) of each recipe in ``detectors`` (a package's
    ``DETECTORS``) carries its readout gate, 0 <= lo < hi; return the names of the recipes with such a group."""  # noqa: D205, D209
    timed = set()
    for entry in detectors.values():
        recipe = DetectorEntry.of(entry).recipe
        for g in getattr(recipe, "groups", ()):  # a recipe made from its inputs is checked when it is made
            if g.values is None or not group_model(g).grid((0.0, 1.0)).nt:
                continue
            lo, hi = (g.readout or {}).get("gate", (None, None))
            _check(lo is not None and 0.0 <= lo < hi, f"recipe {recipe.name!r}, group {g.name!r}: readout gate "
                   f"{g.readout} (a timed sensor model needs 0 <= lo < hi)")  # fmt: skip
            timed.add(recipe.name)
    return timed


def check_references_gate(references) -> dict:
    """Check that no charge entry of a manifest (a package's ``REFERENCES``) scales a parameter to pass; return
    {entry: whether it gates}."""  # noqa: D205, D209
    m = (references,)
    names = refs.names("charge", m)
    scaled = [n for n in names if "scale" in refs.spec("charge", n, m)]
    _check(not scaled, f"charge entries {scaled} scale a parameter to pass")
    return {n: refs.gated("charge", refs.spec("charge", n, m), m) for n in names}


def notebooks(directory) -> list:
    """Return a package's notebooks (``directory/NN_*.ipynb``) in their order."""
    from pathlib import Path

    return sorted(Path(directory).glob("[0-9][0-9]_*.ipynb"))


def check_notebooks_clean(paths) -> None:
    """Check that notebooks are committed without outputs (read as JSON: no notebook packages needed)."""
    import json
    from pathlib import Path

    for path in paths:
        code = [c for c in json.loads(Path(path).read_text())["cells"] if c["cell_type"] == "code"]
        _check(not any(c.get("outputs") or c.get("execution_count") for c in code), f"{path}: committed with outputs")


def run_notebook(path, work, *, timeout: int = 1800):
    """Run a notebook end to end in a fresh kernel in directory ``work`` (``timeout`` s per cell); write it there with
    its outputs (``<package>_<name>.executed.ipynb``) whether or not it ran through; return the executed notebook."""  # noqa: D205, D209
    from pathlib import Path

    import nbformat
    from nbclient import NotebookClient

    path, work = Path(path), Path(work)
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(work)}}
        ).execute()
    finally:
        nbformat.write(nb, work / f"{path.parents[1].name}_{path.stem}.executed.ipynb")
    return nb


def histograms(model, gate, n_sensors: int, photons: dict, *, group: str):
    """Return the readout histograms of a photon list: ``sensor``, ``lam``, ``time``, the deposit fields the grid's
    features read (``pos``, ``dir``, ``normal``: (n, 3) in the sensor's frame; ``gen``), optional ``inv_an`` (1 / A_n)
    and ``weight``; ``group`` the group whose values the readout reads (``lumen.readout.grid.from_photons``)."""  # noqa: D205, D209
    return from_photons(model, gate, n_sensors, photons, group=group)


__all__ = ["check_notebooks_clean", "check_readout_gates", "check_references_gate", "check_value_set", "histograms",
           "load_value_set", "notebooks", "run_notebook"]  # fmt: skip
