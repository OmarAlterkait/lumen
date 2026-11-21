"""Atomic writes into shared directories: the per-user cache, a detector store, a calibration directory.

The one helper every writer of lumen uses (``lumen._cache`` names it for the cache).

Many processes (a Slurm array loading the same detector) may write the same entry at once, and others read it
meanwhile. Every writer goes through this module: an entry is written in full under a unique temporary name beside
its place (so on the same filesystem), then renamed into place in one step. A reader therefore sees either no entry
or a complete one, never a partial one; nothing waits, retries or locks.

- A file (``write_bytes``): ``os.replace`` over the name; with ``exclusive=True`` a hard link instead, which fails
  when the name exists (the file is never replaced).
- A directory (``directory``): ``rename(2)`` of the filled temporary directory. Renaming a directory onto a non-empty
  one fails (ENOTEMPTY or EEXIST) and leaves both as they were, so of N writers exactly one places its entry. What a
  loser does is the caller's choice (``exists``): ``"keep"`` discards its copy and uses the winner's (an entry keyed
  by its content: both hold the same), ``"raise"`` refuses (``FileExistsError``), ``"replace"`` moves the old entry
  aside, renames the new one in and deletes the old one (a reader in between finds no entry, never a partial one).

Measured on WekaFS and on XFS (node-local scratch): a directory renamed onto a non-empty directory fails
(WekaFS ENOTEMPTY, XFS EEXIST) and both are left whole; onto an empty one, or a missing name, it succeeds; a hard
link onto an existing name fails with EEXIST. POSIX requires the same of GPFS and Lustre (not measured here).
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

LOST = (errno.EEXIST, errno.ENOTEMPTY)  # rename(2) onto a non-empty directory: another writer placed it first


def write_bytes(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Write ``data`` to ``path`` atomically, creating the parent directories.

    ``exclusive``: never replace an existing file (``FileExistsError``), also when another process writes the same
    name at the same moment.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if exclusive:
            os.link(tmp, path)  # EEXIST when the name is taken: FileExistsError
        else:
            os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


@contextlib.contextmanager
def directory(path: Path, *, exists: str = "keep") -> Iterator[Path]:
    """Build the directory ``path`` atomically.

    Yields a new temporary directory beside ``path``; fill it inside the ``with`` block. On a clean exit it is renamed
    to ``path``. When ``path`` is already there (placed by another process, before or during this build), ``exists``
    decides: ``"keep"`` keeps it and discards this copy, ``"raise"`` discards this copy and raises
    ``FileExistsError``, ``"replace"`` puts this copy in its place. On an error nothing is placed.
    """
    if exists not in ("keep", "raise", "replace"):
        raise ValueError(f"exists={exists!r}: one of 'keep', 'raise', 'replace'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"))
    try:
        yield tmp
        try:
            os.rename(tmp, path)
            return
        except OSError as e:
            if e.errno not in LOST or not path.is_dir():
                raise
            if exists == "raise":
                raise FileExistsError(errno.EEXIST, "another entry is in place", str(path)) from None
            if exists == "keep":
                return
        old = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".old"))
        os.rename(path, old)  # onto an empty directory of a unique name: allowed
        try:
            os.rename(tmp, path)
        finally:
            shutil.rmtree(old, ignore_errors=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
