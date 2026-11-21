#!/bin/bash
# Store a family's MC-gate inputs as one product: each detector's photon file and its reference record, read-only, sha256
# in MANIFEST.json; prints the manifest entry (the "mc" checks of the family's references.json name its files).
#
#   mc_store.sh STORE ID PACKAGE LUMEN COMMIT DET=PHOTONS.npz,RECORD.npz ...
#
# A record is the MC gate's output (python -m lumen.compare gate --package PACKAGE) on the detector converted with
# the package's recipe, at lumen COMMIT (the arrivals and photon-loop derivatives of lumen.compare.mc record: 16 seeds x
# 65,536 photons from the photon file); a new record (after a change that moves results) is made with
#   python -m lumen.compare gate --package PACKAGE NAME --out OUT_DIR    (OUT_DIR/NAME.npz; docs/dev/validation.md)
# The photon files are fixed emission rows (lumen.sources.arrays.FromArrays: x, d, t, lam, w, er, ei) sampled once from
# each detector's reference event; they are inputs, kept as they are.
set -euo pipefail
STORE=$1; ID=$2; PKG=$3; LUMEN=$4; COMMIT=$5; shift 5
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
items=()
for spec in "$@"; do
  det=${spec%%=*}; files=${spec#*=}
  items+=("${det}_photons.npz=${files%%,*}" "${det}_record.npz=${files#*,}")
done
cat > "$TMP/producer.json" << JSON
{"tool": "python -m lumen.compare gate --package $PKG (lumen.compare.mc record: 16 seeds x 65,536 photons, the arrivals and the derivatives along each family key)",
 "lumen": "$LUMEN", "commit": "$COMMIT",
 "photons": "fixed emission rows per detector (FromArrays), sampled once from the detector's reference event"}
JSON
[ -e "$STORE/$ID" ] && { echo "$ID: in the store already (a product is never overwritten)"; exit 0; }
python3 "$HERE/reduce.py" files "$STORE" "$ID" "${items[@]}" --producer "$TMP/producer.json"
