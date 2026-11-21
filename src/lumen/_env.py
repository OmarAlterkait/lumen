"""The only place lumen reads the environment (style rule 8).

What it reads, and the function that reads it:

- ``LUMEN_DATA`` (``data_dir``): where shared reference data lives; unset, ``data`` under the cache root.
- ``LUMEN_ASSETS`` (``assets_url``): a mirror of the reference products' release assets, a base URL (``file://``
  too) holding the archives under their own file names; unset, each product is fetched from its manifest's URL.
- ``LUMEN_DETECTORS`` (``detector_path``): a search path (``os.pathsep``-separated) of calibrations: where
  ``save_calibration`` writes and a calibration's parents are found (a detector file is loaded by its path).
- ``LUMEN_CACHE`` (``cache_dir``): the per-user cache root; unset, ``lumen-<user>`` in the system's temporary
  directory. One rule for every cache: lumen's defaults are under this root, never under HOME (a small, shared quota
  on clusters); a directory you name yourself is used as named.
- ``JAX_PLATFORMS``, else Slurm's ``SLURM_JOB_GPUS``, ``SLURM_STEP_GPUS``, ``SLURM_GPUS_ON_NODE`` (``gpu_expected``):
  whether a GPU is expected, so a run that expects one never falls back to the CPU silently (``DeviceError``).
- the process's memory limit (``host_memory``): its cgroup's, else the RAM; what it leaves now (``host_available``):
  a CPU run's memory.
- ``TALLYMAP_CACHE`` (``tallymap_cache``): tallymap's kernel cache (CUDA and CPU kernels); unset,
  ``<cache root>/tallymap`` (set here for tallymap).

lumen sets nothing in JAX's configuration: JAX's persistent compilation cache is JAX's own setting
(``JAX_COMPILATION_CACHE_DIR``, ``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS``, read by JAX; docs/howto/install.md).
"""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path

TALLYMAP_CACHE = "TALLYMAP_CACHE"  # read by tallymap itself; set here when unset


def data_dir() -> Path:
    """Return the shared reference-data directory: ``LUMEN_DATA``, else ``data`` under the cache root."""
    value = os.environ.get("LUMEN_DATA")
    return Path(value) if value else cache_dir() / "data"


def assets_url() -> str | None:
    """Return the base URL of a mirror of the reference products' release assets (``LUMEN_ASSETS``), or None."""
    return os.environ.get("LUMEN_ASSETS") or None


def detector_path() -> tuple[Path, ...]:
    """Return the directories searched for calibrations, in order."""
    value = os.environ.get("LUMEN_DETECTORS", "")
    return tuple(Path(p) for p in value.split(os.pathsep) if p)


def cache_dir() -> Path:
    """Return the per-user cache root: ``LUMEN_CACHE``, else ``lumen-<user>`` in the system's temporary directory."""
    value = os.environ.get("LUMEN_CACHE")
    return Path(value) if value else Path(tempfile.gettempdir()) / f"lumen-{getpass.getuser()}"


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


def _cgroup(v2: str, v1: str) -> float | None:
    """Return a value of the process's own memory cgroup (v2 file ``v2``, v1 file ``v1``), else the root's; None."""
    try:  # the process's own cgroup (v2 "0::/path", v1 "N:memory:/path"), else the root's
        own = [ln.split(":", 2) for ln in Path("/proc/self/cgroup").read_text().splitlines()]
    except OSError:
        own = []
    paths = [f"/sys/fs/cgroup{p}/{v2}" for i, c, p in own if i == "0" and c == ""]
    paths += [f"/sys/fs/cgroup/memory{p}/{v1}" for _, c, p in own if "memory" in c.split(",")]
    for f in (*paths, f"/sys/fs/cgroup/{v2}", f"/sys/fs/cgroup/memory/{v1}"):
        try:
            v = Path(f).read_text().strip()
        except OSError:
            continue
        if v.isdigit():
            return float(v)
    return None


def host_memory() -> float:
    """Return the bytes of host memory this process may use: its cgroup's limit (a batch job's), else the RAM."""
    ram = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    limit = _cgroup("memory.max", "memory.limit_in_bytes")
    return ram if limit is None else min(limit, ram)


def host_available() -> float:
    """Return the bytes of host memory available to this process now: its limit (``host_memory``) less what its
    cgroup uses, and no more than the machine's available memory (``MemAvailable``): the memory of a device without
    memory statistics (the CPU), as a GPU's limit less JAX's use is."""  # noqa: D205, D209
    out = host_memory()
    used = _cgroup("memory.current", "memory.usage_in_bytes")
    if used is not None:
        out = min(out, host_memory() - used)
    try:
        info = dict(ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
        out = min(out, float(info["MemAvailable"].split()[0]) * 1024.0)
    except (OSError, KeyError, ValueError):
        pass
    return max(out, 0.0)


def tallymap_cache() -> Path:
    """Return tallymap's kernel cache (CUDA and CPU): ``TALLYMAP_CACHE``, else ``<cache root>/tallymap``, set for it."""
    value = os.environ.get(TALLYMAP_CACHE)
    path = Path(value) if value else cache_dir() / "tallymap"
    os.environ[TALLYMAP_CACHE] = str(path)
    return path
