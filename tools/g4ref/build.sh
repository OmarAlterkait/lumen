#!/bin/bash
# Build g4ref, the Geant4 optical-photon reference application lumen's Geant4 references come from (one file,
# g4ref.cc; its sha256 is the "source_sha256" of their manifest entries). Needs Geant4 10.7.4 built with GDML
# (geant4-config on the PATH), xerces-c and a C++17 compiler; Geant4's sequential mode.
#
#   tools/g4ref/build.sh [OUT]        (default: ./g4ref)
#   ./g4ref --help
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-g4ref}
# shellcheck disable=SC2046
g++ -O2 -std=c++17 $(geant4-config --cflags | sed 's/-std=c++11//') -Wno-shadow -o "$OUT" "$HERE/g4ref.cc" \
  $(geant4-config --libs) -lxerces-c
"$OUT" --help | head -3
