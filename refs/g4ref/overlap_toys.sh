#!/bin/bash
# One of the 18 overlap toys (g4ref/geom/coinc_sib), as its run.sh / run_t.sh ran it: SLURM_ARRAY_TASK_ID picks the
# toy. The cs_ toys: isotropic photons (src.cfg); the others a pencil beam along +x, with the photon count of the run
# they replace (its summary.txt's config).
#
#   overlap_toys.sh OUTDIR RUN_SH
set -euo pipefail
OUT=$1; RUN=$2
G=${G4REF_DIR:?the g4ref directory (geom/, build/)}
TOYS=(coinT_AB coinT_BA cs_AB cs_A cs_BA cs_B partT_AB partT_BA partTm_AB partTm_BA single_B3 single_B vx3_ABC vx3_ACB
      vxc_AB vxc_BA xl_AS xl_SA)
toy=${TOYS[${SLURM_ARRAY_TASK_ID:-0}]}
cfg=$(awk '$1 == "config" {print $2}' "$G/out/coinc_sib/${toy}_summary.txt")
case $toy in cs_*) unset PENCIL ;; *) export PENCIL=1,0,0 ;; esac
SLURM_ARRAY_TASK_ID=0 bash "$RUN" "$OUT/$toy" "$G/geom/coinc_sib/$toy.gdml" "$cfg" '^volS' 1 1 \
  --no-rayleigh --no-absorption --no-wls --no-mie
