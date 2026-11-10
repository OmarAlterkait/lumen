"""The only place lumen reads the environment.

lumen reads three variables and nothing else:

- ``LUMEN_DATA``: where shared reference data lives.
- ``LUMEN_DETECTORS``: a search path (``os.pathsep``-separated) of directories holding detector files and
  calibrations.
- ``LUMEN_CACHE``: the per-user cache root. Unset, it is ``$XDG_CACHE_HOME/lumen`` (``~/.cache/lumen``).

It reads whether a GPU is expected (``gpu_expected``: ``JAX_PLATFORMS``, else Slurm's GPU allocation), so a run that
expects one never falls back to the CPU silently (``lumen.run.DeviceError``).

It also resolves the photon loop's CUDA kernel cache for tallymap (``tallymap_cache``): ``TALLYMAP_CUDA_CACHE``,
else ``<LUMEN_CACHE>/tallymap``, never under HOME (tallymap's own default is ``~/.cache/tallymap``).

And it turns on JAX's persistent compilation cache when this module is imported (``xla_cache``): an explicit
``JAX_COMPILATION_CACHE_DIR`` (JAX reads it itself) or a directory already set in ``jax.config`` is respected;
otherwise ``<LUMEN_CACHE>/xla``, and never under HOME (then the cache stays off). On jax 0.6.2 a program loaded from
the cache is bitwise the one a fresh compile gives, on the CPU and the GPU (tests/physics/test_persistent_cache.py);
on jax 0.5.3 it was not.
"""

from __future__ import annotations

import os
from pathlib import Path

VARIABLES = ("LUMEN_DATA", "LUMEN_DETECTORS", "LUMEN_CACHE")
TALLYMAP_CACHE = "TALLYMAP_CUDA_CACHE"  # read by tallymap itself; set here when unset
XLA_CACHE = "JAX_COMPILATION_CACHE_DIR"  # read by JAX itself; lumen's default set through jax.config when unset


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
    under HOME (``LUMEN_CACHE`` unset), the cache stays off.
    """
    import jax

    value = os.environ.get(XLA_CACHE)
    if value:
        return Path(value)
    current = getattr(jax.config, "jax_compilation_cache_dir", None)
    if current:
        return Path(current)
    path = cache_dir() / "xla"
    if _under_home(path):
        return None
    jax.config.update("jax_compilation_cache_dir", str(path))
    return path


xla_cache()  # at import: lumen's default for every program it compiles (style rule 8: the environment is read here)
