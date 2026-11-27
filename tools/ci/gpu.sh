#!/bin/bash
# The GPU part of lumen's checks (GitHub's CI has no GPU): run on a GPU host before a merge to main and before a tag,
# and record the commit and the job in the release checklist (docs/dev/README.md).
#
#   tools/ci/gpu.sh [FAMILY_REPO ...]          e.g. tools/ci/gpu.sh ../lumen-wc ../lumen-tpc
#
# Runs, from lumen's top directory: the physics tier, the unit tests marked slow, the example scripts; then for each
# family repository given: its tests (the notebooks included), and its MC gate, every "mc" check of its manifest
# (converted with its recipe, bitwise against the reference record; the products are fetched like any reference).
# Run it inside a GPU allocation of your site (a Slurm job, for example); it starts no job itself. Outputs go to
# $OUT (default: a new temporary directory).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-$(mktemp -d)}
# the processes below reuse each other's compiled programs, and the next run's: JAX's persistent cache (lumen sets none
# itself), under lumen's cache root unless set (docs/howto/install.md)
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$(python -c 'from lumen import _env; print(_env.cache_dir() / "jax")')}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}
echo "lumen $(git rev-parse --short HEAD), outputs in $OUT, compiled programs in $JAX_COMPILATION_CACHE_DIR"
# the physics tier skips itself without a GPU (tests/physics/conftest.py): on a CPU fallback this script would pass
# with nothing run, so it refuses first
python -c 'import sys, jax; b = jax.default_backend(); sys.exit(0 if b == "gpu" else f"gpu.sh: JAX runs on {b}, not a GPU")'
python -m pytest tests/physics -q -p no:cacheprovider
JAX_PLATFORMS=cpu python -m pytest tests/unit -m slow -n 4 -q -p no:cacheprovider  # the unit tier is CPU
export LUMEN_EXAMPLES_OUT=$OUT/examples LUMEN_DETECTORS=$OUT/calibrations
mkdir -p "$LUMEN_DETECTORS"
for s in examples/first_hour.py examples/your_own_detector.py examples/extending.py; do
  echo "== $s"; python "$s" > "$OUT/$(basename "$s" .py).log"
done
for repo in "$@"; do
  repo=$(cd "$repo" && pwd)
  pkg=$(cd "$repo/src" && ls -d lumen_*/ | head -1); pkg=${pkg%/}
  echo "== $pkg: tests"
  (cd "$repo" && python -m pytest -q -p no:cacheprovider)
  echo "== $pkg: MC gate"
  python -m lumen.compare gate --package "$pkg" --out "$OUT/mc-$pkg"
done
echo "all GPU checks passed"
