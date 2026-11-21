"""The per-user cache: its entries' paths, and the atomic writes every writer uses (``lumen.format.atomic``).

An entry is written in full under a unique temporary name beside its place, then renamed into place in one step: a
reader sees no entry or a whole one (the rules, and what a process that finishes second does: ``lumen.format.atomic``).
"""

from __future__ import annotations

from pathlib import Path

from lumen import _env
from lumen.format.atomic import directory, write_bytes

__all__ = ["directory", "entry", "write_bytes"]


def entry(*parts: str) -> Path:
    """Return the path of a cache entry under the cache root (nothing is created)."""
    return _env.cache_dir().joinpath(*parts)
