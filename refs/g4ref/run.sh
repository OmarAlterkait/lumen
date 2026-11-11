#!/bin/bash
# One chunk of a g4ref reference run (a Slurm array task: chunk = SLURM_ARRAY_TASK_ID), in IMAGE (Geant4 10.7.4).
#
#   run.sh OUTDIR GDML CONFIG SENS_REGEX SEED NCHUNKS [g4ref flags ...]
#
# Writes OUTDIR/chunk<k>_{hits.npy,sensors.csv,totals.csv,summary.txt} and chunk<k>.log (g4ref's common format). PENCIL=dx,dy,dz sends every photon along that direction (g4ref's G4REF_PENCIL). The binary is
# G4REF, the binary (required); its sha256 goes into OUTDIR/g4ref.json, which refs/reduce.py copies
# into the reference manifest with the flags.
set -euo pipefail
OUT=$1; GDML=$2; CFG=$3; SENS=$4; SEED=$5; N=$6; shift 6
G4REF=${G4REF:?the g4ref binary}
IMAGE=${IMAGE:?the container image}
K=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT"
cd "$OUT"
if [ "$K" = 0 ]; then
  python3 - "$G4REF" "$GDML" "$CFG" "$SENS" "$SEED" "$N" "${PENCIL:-}" "$@" > g4ref.json << 'EOF'
import hashlib, json, sys
g4, gdml, cfg, sens, seed, n, pencil, *flags = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(json.dumps(dict(binary=g4, binary_sha256=sha(g4), gdml=gdml, gdml_sha256=sha(gdml), config=cfg,
                      config_sha256=sha(cfg), sens_regex=sens, seed=int(seed), nchunks=int(n), pencil=pencil or None,
                      flags=flags), indent=1))
EOF
fi
ENVS=()
[ -n "${PENCIL:-}" ] && ENVS=(env G4REF_PENCIL="$PENCIL")
singularity exec -B "${BIND:?the paths to bind into the container}" "$IMAGE" "${ENVS[@]}" "$G4REF" --gdml "$GDML" --config "$CFG" --sens-regex "$SENS" \
  --seed "$SEED" --batch 1000 --chunk "$K" --nchunks "$N" --split-ev 6.19 --out "chunk$K" "$@" > "chunk$K.log" 2>&1
grep -E "^hits" "chunk${K}_summary.txt"
