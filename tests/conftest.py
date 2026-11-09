"""Shared test configuration.

The tests run on XLA's default CPU runtime. The classic-runtime override that A5 added for jax 0.5.3 is gone with the
jax 0.6.2 pin: 0.5.3's thunk runtime missed navigator faces in programs loaded from the persistent compilation cache;
0.6.2 misses none, cache on or off (tools/port/xla_runtime_probe.py check, port/gates/xla_runtime_062.json).
"""


def pytest_addoption(parser):
    parser.addoption(
        "--port-goldens",
        default=None,
        help="directory of the stored port goldens (leaf-module input grids and outputs)",
    )
