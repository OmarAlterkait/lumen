"""The unit tier's fixtures: the navigator toys, built once per session."""

import nav_toys
import pytest


@pytest.fixture(scope="session")
def toys():
    """Return the session's toy builder: toys(name) -> (scene, navigator), cached (read only)."""
    return nav_toys.toy
