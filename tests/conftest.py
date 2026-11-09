"""Shared test configuration.

XLA's CPU thunk runtime (jax 0.5.3) runs the navigator's nested loops wrongly and nondeterministically: a query whose
inputs are all constants misses the face it should meet in some processes (the baseline's navigator alike). The tests
run on the classic CPU runtime until the jax pin moves past it.
"""

import os

CPU_FLAG = "--xla_cpu_use_thunk_runtime=false"
_ENV = os.environ  # noqa: TID251 -- XLA's own flag, set before jax starts (not a lumen setting)


def pytest_configure(config):
    # the leaf goldens (--port-goldens) were made on the default runtime and are compared bit for bit on it
    if config.getoption("--port-goldens") is None and CPU_FLAG.split("=")[0] not in _ENV.get("XLA_FLAGS", ""):
        _ENV["XLA_FLAGS"] = (_ENV.get("XLA_FLAGS", "") + " " + CPU_FLAG).strip()


def pytest_addoption(parser):
    parser.addoption(
        "--port-goldens",
        default=None,
        help="directory of the stored port goldens (leaf-module input grids and outputs)",
    )
