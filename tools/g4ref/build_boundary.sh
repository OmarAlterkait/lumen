#!/bin/bash
# Build boundary, the one-interface Geant4 program the facet-path references come from (boundary.cc; its sha256 is
# the "source_sha256" of tests/data/g4_boundary/MANIFEST.json). Needs Geant4 11.3.0 (geant4-config on the PATH) and a
# C++17 compiler; Geant4's sequential mode.
#
#   tools/g4ref/build_boundary.sh [OUT]        (default: ./boundary)
#   ./boundary --help
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-boundary}
# shellcheck disable=SC2046
g++ -O2 -std=c++17 $(geant4-config --cflags | sed 's/-std=c++1[17]//') -Wno-shadow -o "$OUT" "$HERE/boundary.cc" \
  $(geant4-config --libs)
"$OUT" --help 2>&1 | head -3 || true
