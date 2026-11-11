#!/bin/bash
# Regenerate every g4ref reference product of lumen's checks with the current g4ref (port step A7), each run's GDML,
# config, sensor rule, seed and chunking as the run it replaces (its summary.txt), into OUT/<name>/:
#
#   references.sh OUT [NAME ...]        NAME: wcte_g4 annie_g4 sbnd microboone overlap_toys (default all)
#
# Then refs/reduce.py turns each run into a store product listed in src/lumen/detectors/references.json.
set -euo pipefail
OUT=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
D=${LUMEN_SITE_DATA:?the directory holding lumen/g4ref and lumen/wcsim}
SUB=${JOBSUB:?the Slurm submit command}
LOGS=${LOGS:-$OUT/logs}
SLURM=(--partition milano --account mli:cider-ml --qos preemptable --cpus-per-task 1 --mem 4G)
NAMES=${*:-wcte_g4 annie_g4 sbnd microboone overlap_toys}

array() {   # array NAME NCHUNKS TIME GDML CONFIG SENS SEED [flags]
  local name=$1 n=$2 t=$3; shift 3
  "$SUB" "${SLURM[@]}" --array "0-$((n - 1))" --time "$t" --job-name "a7ref_$name" \
    --output "$LOGS/%x.%A_%a.log" "$HERE/run.sh" "$OUT/$name" "$1" "$2" "$3" "$4" "$n" "${@:5}"
}

for name in $NAMES; do
  case $name in
    wcte_g4) array wcte_g4 100 00:20:00 $D/lumen/wcsim/out/probe_wcte/wcte_probe.gdml $D/lumen/wcsim/out/wcte_main.cfg \
               '^InteriorWCPMT' 20261102 --wcsim-hits ;;
    annie_g4) array annie_g4 16 00:40:00 $D/lumen/wcsim/out/overlaps/annie_v2_g4clean.gdml \
                $D/lumen/wcsim/out/annie_main.cfg '^InteriorWCPMT' 20261106 --wcsim-hits ;;
    sbnd) array sbnd 60 00:30:00 $D/geom_query/gdml/sbnd_v02_00_full_simulation.gdml \
            $D/lumen/g4ref/geom/sbnd_sources_1e7.cfg '^volOpDetSensitive' 20261029 ;;
    microboone) array microboone 100 00:20:00 $D/lumen/g4ref/geom/microboone_optics.gdml \
                  $D/lumen/g4ref/geom/microboone_sources_2e7.cfg '^volOpDetSensitive' 20261101 ;;
    overlap_toys)   # g4ref/geom/coinc_sib run.sh and run_t.sh: no bulk processes; a pencil beam except the cs_ toys
      "$SUB" "${SLURM[@]}" --array 0-17 --time 00:30:00 --job-name a7ref_overlap_toys \
        --output "$LOGS/%x.%A_%a.log" "$HERE/overlap_toys.sh" "$OUT/overlap_toys" "$HERE/run.sh" ;;
    *) echo "unknown reference $name" >&2; exit 2 ;;
  esac
done
