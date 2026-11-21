# Installing lumen

## Requirements

- Linux with an **NVIDIA GPU** and a **CUDA 12** driver, for full speed. lumen runs its photon loop through
  [tallymap](https://github.com/OmarAlterkait/tallymap), which compiles it into a CUDA kernel on a GPU and into
  C++ with OpenMP on the CPU.
- Python 3.10 to 3.13.
- Without a GPU, lumen compiles the photon loop for the CPU (`backend="auto"`, the default, or `backend="cpu"`).
  That needs a C++ compiler with OpenMP on the PATH (install g++; set `CXX` to use another); without one, `simulate`
  raises tallymap's `UnsupportedError` saying so. `backend="xla"` runs tallymap's plain-JAX backend instead, which
  needs no compiler but compiles and runs slowly (tests and small photon counts only); it sums exactly too. On a GPU machine,
  `backend="cpu"` still runs on the CPU, and its results come back to the GPU. How fast a full detector runs on the CPU:
  [known limits](../explanation/known-limits.md#a-full-detector-on-the-cpu).

## Install

lumen pins exact versions of JAX and tallymap. Other versions change the photon loop's results bit for bit, or break
its reverse-mode derivatives. One line installs lumen with both, in a fresh environment:

```bash
python -m venv /path/to/envs/lumen && source /path/to/envs/lumen/bin/activate
pip install "lumen[cuda] @ git+https://github.com/OmarAlterkait/lumen.git"
python -c "import jax, lumen; print(lumen.__version__, jax.devices())"
```

That is the latest release: the repository's `main` moves only at releases. For a fixed version, name its tag:
`pip install "lumen[cuda,plot] @ git+https://github.com/OmarAlterkait/lumen.git@v0.2.0"`. The `plot` extra
(matplotlib) adds `lumen.tools.plot` and the tutorial's figures. lumen is not on PyPI: `pip install lumen` installs an
unrelated package.

`jax[cuda12]` (the `cuda` extra) provides everything tallymap needs on a GPU. tallymap is not on PyPI; lumen's
dependency names its tag on GitHub: `tallymap @ git+https://github.com/OmarAlterkait/tallymap.git@v0.8.0`. The first GPU run also builds a
small C++ launch shim, so a C++17 compiler must be on the PATH once.
Without `[cuda]` (plain `jax`), lumen runs on the CPU only.

The last line should list a `CudaDevice`. If it lists only `CpuDevice`, JAX did not find the CUDA driver. lumen still
runs in that case, on the CPU. A job that expects a GPU and does not get one raises `DeviceError` instead
([errors](../reference/errors.md)).

To work on lumen itself, install a clone editable (see `docs/dev/README.md`):

```bash
git clone https://github.com/OmarAlterkait/lumen.git
pip install -e "lumen[cuda,dev]"
```

## Caches and directories

lumen keeps compiled kernels and converted detectors in a cache, by default in the system's temporary directory (gone
after a reboot). Point the cache at a disk with some space to keep it, and name a directory for your calibrations:

```bash
export LUMEN_CACHE=/path/with/space/lumen-cache
export LUMEN_DETECTORS=/path/to/your/calibrations
```

| Variable | What | Default |
|---|---|---|
| `LUMEN_CACHE` | lumen's cache root | `lumen-<user>` in the system's temporary directory (never under HOME) |
| `TALLYMAP_CACHE` | tallymap's compiled kernels, CUDA and CPU | `$LUMEN_CACHE/tallymap` (lumen sets it; tallymap's own default is `~/.cache/tallymap`) |
| `JAX_COMPILATION_CACHE_DIR` | JAX's persistent compilation cache (JAX's own setting; lumen sets nothing). Set it, with `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` (lumen's programs each compile in under JAX's 1 s threshold), to reuse compiled programs in new processes | none (JAX's: no persistent cache) |
| `CUDA_CACHE_PATH` | the NVIDIA driver's own cache | `~/.nv/ComputeCache` (set it if HOME is small) |
| `LUMEN_DETECTORS` | where calibrations are saved and found (a search path; a detector file is loaded by its path) | none |
| `LUMEN_DATA` | reference data of the validation checks and the families' notebooks: fetched there on first use from the release assets, each file checked by its sha256 | `$LUMEN_CACHE/data` |
| `LUMEN_ASSETS` | a mirror of the release assets (a base URL, `file://` too, holding the archives under their own names): reference data is fetched from it instead of GitHub, still checked by its sha256 | none (GitHub) |

## Compile time and photon counts

The first run of a detector compiles its photon loop, which takes a few minutes. Later runs reuse the cache, in new
processes too when JAX's persistent cache is set (`JAX_COMPILATION_CACHE_DIR`, above). Converting a detector with
tens of thousands of sensors takes minutes, once. How many photons you need depends on the light per sensor; see [choosing the photon count](../tutorial/first-hour.md) in the tutorial, and check
`Light.arrivals_error`.

## GPU memory

JAX takes a fixed share of the GPU's memory at its first use (`XLA_PYTHON_CLIENT_MEM_FRACTION`, 0.75 by default), and
lumen's runs must fit in it. Derivatives take the most: several tangents batched with `jax.vmap` over one `jax.jvp`
keep the photon state once per tangent. lumen checks a run's memory before the launch and raises `MemoryError` when it
does not fit; on a small card, raise `XLA_PYTHON_CLIENT_MEM_FRACTION` (for example to 0.8) or use fewer samples per
run.
