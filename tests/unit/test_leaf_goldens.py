"""The port gate for the leaf modules: their outputs on the stored input grids equal the baseline's bit for bit.

Runs only with ``--port-goldens DIR``, and only on the CPU type the goldens were made on: XLA's CPU code depends on
the instruction set, so the same program gives different last bits on another CPU.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

TOOLS = Path(os.environ.get("LUMEN_PORT_TOOLS", "/nonexistent"))  # noqa: TID251 -- a test option, not lumen's


def _cpu():
    with open("/proc/cpuinfo") as f:
        return next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "?")


def _load(name):
    import sys

    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_leaf_modules_are_bitwise_on_the_stored_grids(request):
    root = request.config.getoption("--port-goldens")
    if not root:
        pytest.skip("no --port-goldens directory given")
    if not TOOLS.is_dir():
        pytest.skip("LUMEN_PORT_TOOLS is not the port's gate tooling")
    man = json.loads((Path(root) / "MANIFEST.json").read_text())
    made_on = man["environment"].get("cpu")
    if made_on and made_on != _cpu():
        pytest.skip(f"goldens made on {made_on}, this is {_cpu()}")
    rep = _load("check_leaf_goldens").check(Path(root))
    bad = {k: v for k, v in rep["cases"].items() if v["status"] != "bitwise"}
    assert not bad, bad
