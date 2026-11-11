# Installing lumen

## Requirements

- Linux with an **NVIDIA GPU** and a **CUDA 12** driver. lumen runs its photon loop through
  [tallymap](https://github.com/OmarAlterkait/tallymap), which compiles a CUDA kernel for it.
- Python 3.10 or newer.
- Without a GPU, lumen falls back to tallymap's plain-JAX CPU backend. Its results agree with the GPU's to about
  2.5e-4, but it compiles slowly and runs slowly. Use it for tests and for small photon counts (up to about 2^16
  photons); see [known limits](../explanation/known-limits.md#a-full-detector-on-the-cpu).

## Install

lumen pins exact versions of JAX and tallymap. Other versions change the photon loop's results bit for bit, or break
its reverse-mode derivatives. One line installs lumen with both, in a fresh environment:

```bash
python -m venv /path/to/envs/lumen && source /path/to/envs/lumen/bin/activate
pip install "lumen[cuda] @ git+https://github.com/OmarAlterkait/lumen.git"
python -c "import jax, lumen; print(lumen.__version__, jax.devices())"
```

That is the latest release: the repository's `main` moves only at releases. For a fixed version, name its tag:
`pip install "lumen[cuda] @ git+https://github.com/OmarAlterkait/lumen.git@v0.1.2"`. `pip install matplotlib` adds
the tutorial's figures.

`jax[cuda12]` (the `cuda` extra) provides everything tallymap's GPU backend needs. tallymap is not on PyPI; lumen's
dependency names its tag (v0.6.0rc1) on GitHub. The first GPU run also builds a small C++ launch shim, so a C++17
compiler must be on the PATH once. Without `[cuda]` (plain `jax`), lumen runs on the CPU only.

The last line should list a `CudaDevice`. If it lists only `CpuDevice`, JAX did not find the CUDA driver. lumen still
runs in that case, on the CPU backend. A job that expects a GPU and does not get one raises `DeviceError` instead
([errors](../reference/errors.md)).

To work on lumen itself, install a clone editable (see `docs/dev/README.md`):

```bash
git clone https://github.com/OmarAlterkait/lumen.git
pip install -e "lumen[cuda,dev]"
```

## Caches and directories

lumen keeps compiled kernels and converted detectors in a cache. Point the cache at a disk with some space, and name a
directory for your calibrations:

```bash
export LUMEN_CACHE=/path/with/space/lumen-cache
export LUMEN_DETECTORS=/path/to/your/calibrations
```

| Variable | What | Default |
|---|---|---|
| `LUMEN_CACHE` | lumen's cache root | `$XDG_CACHE_HOME/lumen` (`~/.cache/lumen`) |
| `TALLYMAP_CUDA_CACHE` | tallymap's compiled CUDA kernels | `$LUMEN_CACHE/tallymap` (lumen sets it; tallymap's own default is `~/.cache/tallymap`) |
| `JAX_COMPILATION_CACHE_DIR` | JAX's persistent compilation cache | `$LUMEN_CACHE/xla` (never under HOME) |
| `CUDA_CACHE_PATH` | the NVIDIA driver's own cache | `~/.nv/ComputeCache` (set it if HOME is small) |
| `LUMEN_DETECTORS` | where detector files and calibrations are saved and found (a search path) | none |
| `LUMEN_DATA` | shared reference data (the validation checks) | none |

## Compile time and photon counts

The first run of a detector compiles its photon loop, which takes a few minutes. Later runs reuse the cache, in new
processes too. A measured large detector:

| 43,212-sensor detector, one NVIDIA L40S | time |
|---|---|
| convert and first load | about 10 min |
| first `simulate` (compiles) | 31 s |
| later `simulate`, 2^18 photons | 2.2 s |

At 2^18 photons from a point at the centre, 45 % of those sensors saw no light. How many photons you need depends on
the light per sensor; see [choosing the photon count](../tutorial/first-hour.md) in the tutorial, and check
`Light.arrivals_error`.
