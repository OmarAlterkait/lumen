#!/bin/bash
# Maintainer script: runs boundary (build_boundary.sh) on every facet-path case of tests/data/g4_boundary and reduces
# the runs there (reduce_boundary.py); it records how those references were made.
#
#   tools/g4ref/boundary_cases.sh BOUNDARY WORKDIR [N]       (N photons per case, default 1000000)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$1; WORK=$2; N=${3:-1000000}
DATA="$HERE/../../tests/data/g4_boundary"
mkdir -p "$WORK"
UNIFIED=(--type dielectric_metal --model unified --finish ground --R 1 --spike 0 --lobe 1 --back 0)
GLISUR=(--type dielectric_dielectric --model glisur --finish ground)
# name | arguments
CASES=(
  "unified_metal_s0.3_th60|${UNIFIED[*]} --value 0.3 --theta 60 --seed 11"
  "unified_metal_s1_th30|${UNIFIED[*]} --value 1 --theta 30 --seed 12"
  "unified_metal_s1_th75|${UNIFIED[*]} --value 1 --theta 75 --seed 13"
  "unified_metal_s1_th85|${UNIFIED[*]} --value 1 --theta 85 --seed 14"
)
specs=()
for c in "${CASES[@]}"; do
  name=${c%%|*}; args=${c#*|}
  # shellcheck disable=SC2086
  "$BIN" --out "$WORK/$name.npy" --n "$N" $args
  specs+=("$name=$WORK/$name.npy")
done
python "$HERE/reduce_boundary.py" "$DATA" "${specs[@]}"
