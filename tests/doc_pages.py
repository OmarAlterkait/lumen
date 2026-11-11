"""Run the Python blocks of a documentation page in order, in one namespace, from the repository's top directory."""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def blocks(page):
    """Return the fenced Python blocks of a page (docs/...), in order."""
    return re.findall(r"```python\n(.*?)```", (ROOT / page).read_text(), re.S)


def run(page, tmp_path, monkeypatch):
    """Execute every Python block of ``page`` with the examples' outputs in ``tmp_path``; return the namespace."""
    monkeypatch.setenv("LUMEN_EXAMPLES_OUT", str(tmp_path / "lumen-examples"))
    monkeypatch.setenv("LUMEN_DETECTORS", str(tmp_path / "calibrations"))
    (tmp_path / "calibrations").mkdir(exist_ok=True)
    monkeypatch.chdir(ROOT)
    ns = {}
    for i, code in enumerate(blocks(page)):
        exec(compile(code, f"{page}[block {i}]", "exec"), ns)
    os.chdir(ROOT)
    return ns
