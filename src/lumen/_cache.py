"""Atomic writes into the per-user cache.

Many processes (a Slurm array building the same detector) may write the same cache entry at once. Every write
goes to a temporary name in the same directory and is then moved into place with ``os.replace``, which is atomic
on one filesystem. A reader therefore sees either no entry or a complete one, never a partial one.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from lumen import _env


def entry(*parts: str) -> Path:
    """Return the path of a cache entry under the cache root (nothing is created)."""
    return _env.cache_dir().joinpath(*parts)


def write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, creating the parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def directory(path: Path) -> Iterator[Path]:
    """Build a cache directory atomically.

    Yields a temporary directory beside ``path``; fill it inside the ``with`` block. On a clean exit it is moved to
    ``path``. If another process finished the same entry first, that entry is kept and this one is discarded (both
    hold the same content, since an entry is keyed by its content).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"))
    try:
        yield tmp
        try:
            os.replace(tmp, path)
        except OSError:
            if not path.is_dir():
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
