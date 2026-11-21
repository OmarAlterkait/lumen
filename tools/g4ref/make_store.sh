#!/bin/bash
# Maintainer script: runs g4ref (build.sh) as Slurm arrays in a Geant4 10.7.4 container image; it records how the
# reference products were made.
# Build the reference store from g4ref runs, reduced by tools/g4ref/reduce.py, read-only (a product is never overwritten):
#
#   make_store.sh STORE G4REF_RUNS                    lumen's own checks: the overlap toys
#   make_store.sh STORE G4REF_RUNS --runs MANIFEST    a detector package's runs (its reference manifest, as references.sh)
#
# G4REF_RUNS holds the runs tools/g4ref/references.sh made. With --runs, each run NAME (tools/g4ref/runs.py) becomes the
# product NAME-<the g4ref binary's sha256, 8 digits>, its producer carrying the run block, keeping a copy of its GDML
# when the run's KEEP_GDML is yes. A run with an ORIGINAL is a run on a detector's overlap-free export
# (python -m lumen.compare export DET.lumen GDML, which writes GDML's record beside it as .json): its producer's "export"
# holds that record and names ORIGINAL-<sha256> as "original" (lumen.compare.references.original). Prints each
# product's manifest entry.
set -euo pipefail
STORE=$1; RUNS=$2; shift 2
TABLE=""
if [ "${1:-}" = --runs ]; then TABLE=$2; shift 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
D=${LUMEN_SITE_DATA:?the directory holding the reference inputs (lumen/g4ref, ...)}
export IMAGE=${IMAGE:?the container image the g4ref runs used} G4REF_SRC=${G4REF_SRC:?the g4ref source file the runs were built from}
TMP=$(mktemp -d -p "${STORE_TMP:-/tmp}")
trap 'rm -rf "$TMP"' EXIT
R="python3 $HERE/reduce.py"
have() { [ -e "$STORE/$1" ] && echo "$1: in the store already (a product is never overwritten)"; }

g4() {   # g4 NAME ID ORIGINAL [--input ...]: the g4ref run NAME of references.sh as the store product ID
  local name=$1 id=$2 original=$3; shift 3
  python3 - "$RUNS/$name/g4ref.json" "$TMP/$name.json" "$id" "$original" "$(python3 "$HERE/runs.py" "$TABLE" --run "$name")" << 'EOF'
import json, os, sys
p = json.load(open(sys.argv[1]))
p.update(tool="g4ref", geant4="10.7.4", image=os.environ["IMAGE"],
         source=os.environ["G4REF_SRC"], physics="G4OpticalPhysics, photons as primaries", run=json.loads(sys.argv[5]))
sha = sys.argv[3].rsplit("-", 1)[1]
if sys.argv[4] != "-":  # a run on the overlap-free export: its record, and the run on the original GDML
    rec = json.load(open(os.path.splitext(p["gdml"])[0] + ".json"))
    if rec["sha256"] != "sha256:" + p["gdml_sha256"] or not rec["lossless"]:
        raise SystemExit(f"{p['gdml']}: not the lossless export its record describes")
    p["export"] = dict(rec, original=f"{sys.argv[4]}-{sha}")
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
  local cfg
  cfg=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['config'])" "$RUNS/$name/g4ref.json")
  have "$id" || $R hits "$RUNS/$name" "$STORE" "$id" --config "$cfg" --producer "$TMP/$name.json" "$@"
}

if [ -n "$TABLE" ]; then
  while read -r name _n _t gdml _cfg _sens _seed keep original _flags; do
    G4=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['binary_sha256'][:8])" "$RUNS/$name/g4ref.json")
    if [ "$keep" = yes ]; then g4 "$name" "$name-$G4" "$original" --input "$(basename "$gdml")=$D/$gdml"
    else g4 "$name" "$name-$G4" "$original"; fi
  done < <(python3 "$HERE/runs.py" "$TABLE")
  exit 0
fi

G4=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['binary_sha256'][:8])" "$RUNS/overlap_toys/coinT_AB/g4ref.json")
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
