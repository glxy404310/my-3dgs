#!/usr/bin/env bash
# Root reviews and freezes new code before invoking. Actual score proof only.
set -euo pipefail
if [[ "$#" != 8 ]]; then
  echo 'Usage: run-evaluation.sh SCORE_RESULT_JSON SHA SCORE_SUPERVISOR_JSON SHA PUP_PLY SHA SCORE_EXTERNAL_JSON SHA' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum --check --status <<'FROZEN_SHA256'
6ca7bf915c779913f27237235edb4beb258f16e9098b6ea0bc6e567e856fb4db  /mnt/d/Documents/my 3dgs/scripts/experiments/evaluate_pup_frozen_baseline_v2.py
f8c7d922a798e68d475237ea11bc477525b1862438ce97faf01a30a364b7d2f6  /mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/EVALUATION_CONTRACT.md
cd7c2669a62daaba607e094a7353d229dd1cc6b56d3d8fe7bcc3a18d2b0de668  /mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/evaluation-v1/ASSUMPTIONS.md
9cef1a9aa557e28538785136efe8ff4756cd65b7145d75be68cbaa24f558caec  /mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/evaluation-v1/READONLY_PINS.json
2a089b7d85987b571bb53f22ce9a97912fe690e01fc2d9d950dead39fd3a29a3  /mnt/d/Documents/my 3dgs/scripts/experiments/shared_coordinate_native_v3.py
0c62664a1b90f6581215d2285971462175aae4c753cfe9fad5a0cd207c2525e6  /mnt/d/Documents/my 3dgs/scripts/experiments/evaluate_nanogs_frozen_baseline.py
9014824dce6ac7ae3484869ace002fb4c073d922779a8072c453948ac34442d3  /mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/EVALUATION_V2_TOOL_REPAIR.md
FROZEN_SHA256
exec /usr/bin/timeout --signal=TERM --kill-after=3s 185s \
  /home/administrator/venvs/3dgs/bin/python -B -u \
  '/mnt/d/Documents/my 3dgs/scripts/experiments/evaluate_pup_frozen_baseline_v2.py' --run \
  --score-result "$1" --score-sha256 "$2" --score-supervisor "$3" --supervisor-sha256 "$4" \
  --pup-ply "$5" --pup-sha256 "$6" --score-external "$7" --external-sha256 "$8"
