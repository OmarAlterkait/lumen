#!/bin/bash
# Build the reference store (port step A7): every product lumen's checks read, reduced by refs/reduce.py, read-only.
#
#   make_store.sh STORE G4REF_RUNS        (a milano job: it hashes the WCSim images, 3.6 GB)
#
# The WCSim products reuse the existing runs (refs/README.md: hk-software for SK and WCTE, anniesoft for ANNIE); the
# g4ref products are the runs refs/g4ref/references.sh made into G4REF_RUNS. Prints each product's manifest entry.
set -euo pipefail
STORE=$1; RUNS=$2
HERE=$(cd "$(dirname "$0")" && pwd)
D=${LUMEN_SITE_DATA:?the directory holding lumen/wcsim and lumen/g4ref}
export IMAGE=${IMAGE:?the container image the g4ref runs used} G4REF_SRC=${G4REF_SRC:?the g4ref source file the runs were built from}
W=$D/lumen/wcsim
TMP=$(mktemp -d -p "${STORE_TMP:-/tmp}")
trap 'rm -rf "$TMP"' EXIT
R="python3 $HERE/reduce.py"
have() { [ -e "$STORE/$1" ] && echo "$1: in the store already (a product is never overwritten)"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

wcsim() {   # wcsim FILE DET COMMIT IMAGE GEANT4 RESPONSE
  cat > "$1" << EOF
{"tool": "WCSim", "repository": "$4", "commit": "$3", "image": "$5", "image_sha256": "$(sha "$5")", "geant4": "$6",
 "macro": "refs/wcsim/macros/$2.mac (lumen-b1)", "harness": "refs/wcsim/run.sh hits (lumen-b1)",
 "response": "$7", "hit": "a photon absorbed at WCSim's photocathode boundary (glass to InteriorWCPMT), true hits of trigger 0"}
EOF
}
wcsim "$TMP/sk.json" sk e2dc7a48b1f83c74a541123542606a90d7b67a6f WCSim/WCSim $W/images/hk-software_main.sif 10.3.3 \
  "DoNotApplyQE, collection efficiency off, no dark noise"
wcsim "$TMP/wcte.json" wcte e2dc7a48b1f83c74a541123542606a90d7b67a6f WCSim/WCSim $W/images/hk-software_main.sif 10.3.3 \
  "DoNotApplyQE, collection efficiency off, no dark noise"
wcsim "$TMP/annie.json" annie e88314b31266255ed02a95ce87677f92d4c2db22 ANNIEsoft/WCSim $W/images/anniesoft_wcsim_latest.sif \
  10.1.3 "QE kept (Multi_Tank_Types: no DoNotApplyQE in ANNIE's WCSim), collection efficiency off"

have wcsim-sk-e2dc7a48 || $R hits $W/out/sk_main "$STORE" wcsim-sk-e2dc7a48 --config $W/out/sk_main.cfg --producer "$TMP/sk.json" \
  --input sk.gdml=$W/out/sk.gdml
have wcsim-wcte-e2dc7a48 || $R hits $W/out/wcte_main "$STORE" wcsim-wcte-e2dc7a48 --config $W/out/wcte_main.cfg --producer "$TMP/wcte.json" \
  --input wcte.gdml=$W/out/wcte.gdml
have wcsim-annie-e88314b || $R hits $W/out/annie_main "$STORE" wcsim-annie-e88314b --config $W/out/annie_main.cfg --producer "$TMP/annie.json" \
  --input annie_v2.gdml=$W/out/annie_v2.gdml --input tuning_parameters.mac=$W/src_annie_ref/macros/tuning_parameters.mac

g4() {   # g4 NAME ID [--input ...]: a g4ref run of references.sh
  local name=$1 id=$2; shift 2
  python3 - "$RUNS/$name/g4ref.json" "$TMP/$name.json" << 'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
import os
p.update(tool="g4ref", geant4="10.7.4", image=os.environ["IMAGE"],
         source=os.environ["G4REF_SRC"], physics="G4OpticalPhysics, photons as primaries")
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
  local cfg
  cfg=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['config'])" "$RUNS/$name/g4ref.json")
  have "$id" || $R hits "$RUNS/$name" "$STORE" "$id" --config "$cfg" --producer "$TMP/$name.json" "$@"
}
G4=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['binary_sha256'][:8])" "$RUNS/sbnd/g4ref.json")
g4 wcte_g4 "g4ref-wcte-$G4" --input wcte_probe.gdml=$W/out/probe_wcte/wcte_probe.gdml
g4 annie_g4 "g4ref-annie-$G4" --input annie_v2_g4clean.gdml=$W/out/overlaps/annie_v2_g4clean.gdml
g4 sbnd "g4ref-sbnd-$G4"
g4 microboone "g4ref-microboone-$G4" --input microboone_optics.gdml=$D/lumen/g4ref/geom/microboone_optics.gdml
python3 - "$RUNS/overlap_toys/coinT_AB/g4ref.json" "$TMP/toys.json" << 'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
p = {k: v for k, v in p.items() if k in ("binary", "binary_sha256", "seed", "flags")}
import os
p.update(tool="g4ref", geant4="10.7.4", image=os.environ["IMAGE"], sens_regex="^volS",
         sources="the cs_ toys: isotropic photons (coinc_sib/src.cfg); the others a pencil beam along +x "
                 "(G4REF_PENCIL=1,0,0), from x = -800 mm, 2.95 eV, the photon count of each toy's config")
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
have "g4ref-overlap-toys-$G4" || $R toys "$RUNS/overlap_toys" "$STORE" "g4ref-overlap-toys-$G4" --producer "$TMP/toys.json" \
  --gdml-dir $D/lumen/g4ref/geom/coinc_sib

# the GDMLs the public detector files were converted from (lumen.detectors.register_builtins' dataset paths)
[ -e "$STORE/gdml/MANIFEST.json" ] || mkdir -p "$STORE/gdml"
[ -e "$STORE/gdml/MANIFEST.json" ] || for f in $W/out/sk.gdml $W/out/wcte.gdml $W/out/annie_v2.gdml $D/geom_query/gdml/sbnd_v02_00_full_simulation.gdml \
         $D/geom_query/gdml/sbnd_pds_mapping.json $D/geom_query/gdml/microboone.gdml; do
  [ -e "$STORE/gdml/$(basename "$f")" ] || { cp "$f" "$STORE/gdml/"; chmod 0444 "$STORE/gdml/$(basename "$f")"; }
done
(cd "$STORE/gdml" && sha256sum ./*)
