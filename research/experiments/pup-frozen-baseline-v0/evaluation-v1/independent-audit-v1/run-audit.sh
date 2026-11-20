#!/usr/bin/env bash
# Root must review/freeze this CPU-only one-shot audit before supplying actual pins.
set -euo pipefail
if [[ "$#" != 10 ]]; then
  echo 'Usage: run-audit.sh RESULTS SHA SUPERVISOR SHA RAW_MANIFEST SHA EXTERNAL_V2 SHA PRESERVED_FAILED_EXTERNAL_V1 SHA' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum --check --status <<'FROZEN_SHA256'
8808b5aa1e9bc0ec9ea491decba916298b6ad5522d1b48dacf7845b9dae22f23  /mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/evaluation-v1/independent-audit-v1/audit.py
1d6ff80b5ae4ce56ef31875c9d6fd94d9677c0d6d874fd889902c8e9d3628040  /mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/evaluation-v1/independent-audit-v1/CONTRACT.md
FROZEN_SHA256
exec /usr/bin/timeout --signal=TERM --kill-after=1s 98s \
  /home/administrator/venvs/3dgs/bin/python -B -u \
  '/mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/evaluation-v1/independent-audit-v1/audit.py' --run \
  --result "$1" --result-sha256 "$2" --supervisor "$3" --supervisor-sha256 "$4" \
  --manifest "$5" --manifest-sha256 "$6" --external "$7" --external-sha256 "$8" \
  --prior-failed-external "$9" --prior-failed-external-sha256 "${10}"
