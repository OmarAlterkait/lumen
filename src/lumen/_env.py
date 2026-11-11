"""The only place lumen reads the environment (style rule 8).

What it reads, and the function that reads it:

- ``LUMEN_DATA`` (``data_dir``): where shared reference data lives.
- ``LUMEN_DETECTORS`` (``detector_path``): a search path (``os.pathsep``-separated) of detector files and calibrations.
- ``LUMEN_CACHE``, else ``XDG_CACHE_HOME``, else ``HOME`` (``cache_dir``): the per-user cache root,
  ``$XDG_CACHE_HOME/lumen`` (``~/.cache/lumen``) when ``LUMEN_CACHE`` is unset.
- ``JAX_PLATFORMS``, else Slurm's ``SLURM_JOB_GPUS``, ``SLURM_STEP_GPUS``, ``SLURM_GPUS_ON_NODE`` (``gpu_expected``):
  whether a GPU is expected, so a run that expects one never falls back to the CPU silently (``DeviceError``).
- ``TALLYMAP_CUDA_CACHE`` (``tallymap_cache``): tallymap's CUDA kernel cache; unset, ``<LUMEN_CACHE>/tallymap`` (set
  here for tallymap), never under HOME.
- ``JAX_COMPILATION_CACHE_DIR`` and ``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`` (``xla_cache``, at import): JAX's
  persistent compilation cache; unset, and none set in ``jax.config``, ``<LUMEN_CACHE>/xla``, never under HOME (then it
  stays off); every program is cached, not only those slower than JAX's 1 s. Why a cached program is safe on the
  pinned jax: docs/dev/decisions.md, "jax pinned to 0.6.2".
"""

from __future__ import annotations

import os
from pathlib import Path

TALLYMAP_CACHE = "TALLYMAP_CUDA_CACHE"  # read by tallymap itself; set here when unset
XLA_CACHE = "JAX_COMPILATION_CACHE_DIR"  # read by JAX itself; lumen's default set through jax.config when unset
MIN_COMPILE_TIME = "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"  # read by JAX itself; lumen sets 0 when unset


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


GPU_ALLOCATED = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")  # Slurm's GPU ids of the allocation (non-empty: a GPU)
GPU_COUNT = "SLURM_GPUS_ON_NODE"  # Slurm's number of GPUs allocated on this node


def gpu_expected() -> str | None:
    """Return why a GPU is expected here (the environment), or None.

    ``JAX_PLATFORMS`` decides when it is set (naming cuda or gpu: expected; naming only others, e.g. cpu: not);
    otherwise a Slurm GPU allocation (``GPU_ALLOCATED`` non-empty, or ``GPU_COUNT`` above 0) expects one.
    """
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if platforms:
        named = {p.strip() for p in platforms.lower().split(",")}
        return f"JAX_PLATFORMS={platforms}" if named & {"cuda", "gpu"} else None
    for v in GPU_ALLOCATED:
        if os.environ.get(v, "").strip():
            return f"{v}={os.environ[v]}"
    count = os.environ.get(GPU_COUNT, "").strip()
    return f"{GPU_COUNT}={count}" if count.isdigit() and int(count) > 0 else None


def tallymap_cache() -> Path:
    """Return tallymap's CUDA kernel cache, set for tallymap: never under HOME (a shared quota), else fail closed."""
    value = os.environ.get(TALLYMAP_CACHE)
    path = Path(value) if value else cache_dir() / "tallymap"
    home = Path.home().resolve()
    if path.resolve() == home or home in path.resolve().parents:
        raise RuntimeError(f"tallymap's CUDA cache {path} is under HOME ({home}): set LUMEN_CACHE (or "
                           f"{TALLYMAP_CACHE}) to a directory outside it")  # fmt: skip
    os.environ[TALLYMAP_CACHE] = str(path)
    return path


def _under_home(path: Path) -> bool:
    home = Path.home().resolve()
    p = path.resolve()
    return p == home or home in p.parents


def xla_cache() -> Path | None:
    """Turn on JAX's persistent compilation cache; return its directory, or None when it stays off.

    An explicit ``JAX_COMPILATION_CACHE_DIR`` (JAX reads it itself) or a directory already set in ``jax.config`` is
    respected as it is. Otherwise the cache goes to ``<LUMEN_CACHE>/xla``, never under HOME: when that default falls
    under HOME (``LUMEN_CACHE`` unset), the cache stays off. With the cache on, every program is written to it, not
    only those that took JAX's default 1 s to compile (``MIN_COMPILE_TIME``, unless set in the environment): lumen's
    programs each compile in under a second, so with the default nothing was cached and a new process compiled them
    all again.
    """
    import jax

    path: Path | None
    value = os.environ.get(XLA_CACHE)
    current = getattr(jax.config, "jax_compilation_cache_dir", None)
    if value:
        path = Path(value)
    elif current:
        path = Path(current)
    else:
        path = cache_dir() / "xla"
        if _under_home(path):
            return None
        jax.config.update("jax_compilation_cache_dir", str(path))
    if not os.environ.get(MIN_COMPILE_TIME):
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    return path


xla_cache()  # at import: lumen's default for every program it compiles (style rule 8: the environment is read here)
