"""Shared test configuration."""


def pytest_addoption(parser):
    parser.addoption(
        "--port-goldens",
        default=None,
        help="directory of the stored port goldens (leaf-module input grids and outputs)",
    )
