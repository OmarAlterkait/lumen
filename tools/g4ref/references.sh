#!/bin/bash
# Maintainer script: runs g4ref (build.sh) as Slurm arrays in a Geant4 10.7.4 container image; it records how the
# reference products were made.
# Regenerate g4ref reference products with the current g4ref, each run's GDML, config, sensor rule, seed and chunking
# as the run it replaces (its summary.txt), into OUT/<name>/:
#
#   references.sh OUT [NAME ...]                  lumen's own checks: NAME overlap_toys (default all)
#   references.sh OUT --runs MANIFEST [NAME ...]  a detector package's runs: the products of its reference manifest
#                                                 whose producer has a run block (tools/g4ref/runs.py; default all)
#
# A run's GDML and config are relative to LUMEN_SITE_DATA; its flags are its own (--split-ev, --wcsim-hits, ...),
# passed to run.sh as given. Then make_store.sh --runs MANIFEST turns each run into a store product listed in the
# manifest.
set -euo pipefail
OUT=$1; shift
RUNS=""
if [ "${1:-}" = --runs ]; then RUNS=$2; shift 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
D=${LUMEN_SITE_DATA:?the directory holding the reference inputs (lumen/g4ref, ...)}
SUB=${JOBSUB:?the Slurm submit command}
LOGS=${LOGS:-$OUT/logs}
read -ra SITE <<< "${SLURM_ARGS:?the sbatch options of your site: partition, account, QoS}"
SLURM=("${SITE[@]}" --cpus-per-task 1 --mem 4G)

array() {   # array NAME NCHUNKS TIME GDML CONFIG SENS SEED [flags]
  local name=$1 n=$2 t=$3; shift 3
  "$SUB" "${SLURM[@]}" --array "0-$((n - 1))" --time "$t" --job-name "g4ref_$name" \
    --output "$LOGS/%x.%A_%a.log" "$HERE/run.sh" "$OUT/$name" "$D/$1" "$D/$2" "$3" "$4" "$n" "${@:5}"
}

if [ -n "$RUNS" ]; then
  found=0
  while read -r name n t gdml cfg sens seed _keep _original flags; do
    read -ra F <<< "${flags:-}"
    array "$name" "$n" "$t" "$gdml" "$cfg" "$sens" "$seed" "${F[@]}"
    found=$((found + 1))
  done < <(python3 "$HERE/runs.py" "$RUNS" "$@")
  [ "$found" -gt 0 ] || { echo "no run of $* in $RUNS" >&2; exit 2; }
  exit 0
fi
for name in ${*:-overlap_toys}; do
  case $name in
    overlap_toys)   # g4ref/geom/coinc_sib run.sh and run_t.sh: no bulk processes; a pencil beam except the cs_ toys
      "$SUB" "${SLURM[@]}" --array 0-17 --time 00:30:00 --job-name g4ref_overlap_toys \
        --output "$LOGS/%x.%A_%a.log" "$HERE/overlap_toys.sh" "$OUT/overlap_toys" "$HERE/run.sh" ;;
    *) echo "unknown reference $name" >&2; exit 2 ;;
  esac
done
