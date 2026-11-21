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
  "glisur_n1_1.67_p0.01_th30|${GLISUR[*]} --n1 1 --n2 1.67 --value 0.01 --theta 30 --seed 21"
  "glisur_n1_1.67_p0.01_th70|${GLISUR[*]} --n1 1 --n2 1.67 --value 0.01 --theta 70 --seed 22"
  "glisur_n1.67_1_p0.01_th20|${GLISUR[*]} --n1 1.67 --n2 1 --value 0.01 --theta 20 --seed 23"
  "glisur_n1.67_1_p0.01_th50|${GLISUR[*]} --n1 1.67 --n2 1 --value 0.01 --theta 50 --seed 24"
  "glisur_n1_1.5_p0.6_th50|${GLISUR[*]} --n1 1 --n2 1.5 --value 0.6 --theta 50 --seed 25"
  "glisur_n1_1.67_p0.3_th60_s|${GLISUR[*]} --n1 1 --n2 1.67 --value 0.3 --theta 60 --pol s --seed 26"
  "glisur_n1_1.67_p0.3_th60_p|${GLISUR[*]} --n1 1 --n2 1.67 --value 0.3 --theta 60 --pol p --seed 27"
  "glisur_n1.67_1.8_p0.01_th45|${GLISUR[*]} --n1 1.67 --n2 1.8 --value 0.01 --theta 45 --seed 28"
)
specs=()
for c in "${CASES[@]}"; do
  name=${c%%|*}; args=${c#*|}
  # shellcheck disable=SC2086
  "$BIN" --out "$WORK/$name.npy" --n "$N" $args
  specs+=("$name=$WORK/$name.npy")
done
python "$HERE/reduce_boundary.py" "$DATA" "${specs[@]}"
