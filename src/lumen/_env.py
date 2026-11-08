"""The only place lumen reads the environment.

lumen reads three variables and nothing else:

- ``LUMEN_DATA``: where shared reference data lives.
- ``LUMEN_DETECTORS``: a search path (``os.pathsep``-separated) of directories holding detector files and
  calibrations.
- ``LUMEN_CACHE``: the per-user cache root. Unset, it is ``$XDG_CACHE_HOME/lumen`` (``~/.cache/lumen``).
"""

from __future__ import annotations

import os
from pathlib import Path

VARIABLES = ("LUMEN_DATA", "LUMEN_DETECTORS", "LUMEN_CACHE")


def data_dir() -> Path | None:
    """Return the shared reference-data directory, or None when ``LUMEN_DATA`` is unset."""
    value = os.environ.get("LUMEN_DATA")
    return Path(value) if value else None


def detector_path() -> tuple[Path, ...]:
    """Return the directories searched for detector files and calibrations, in order."""
    value = os.environ.get("LUMEN_DETECTORS", "")
    return tuple(Path(p) for p in value.split(os.pathsep) if p)


def cache_dir() -> Path:
    """Return the per-user cache root: ``LUMEN_CACHE``, else ``$XDG_CACHE_HOME/lumen``, else ``~/.cache/lumen``."""
    value = os.environ.get("LUMEN_CACHE")
    if value:
        return Path(value)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "lumen"
