#!/usr/bin/env bash
# CPU-only one-shot audit. Root reviews/freezes before supplying actual artifact pins.
set -euo pipefail
if [[ "$#" != 12 ]]; then
  echo 'Usage: run-audit.sh RESULTS SHA SUPERVISOR SHA RAW_MANIFEST SHA COMBINED_RAW_MANIFEST SHA CONFIG SHA EXTERNAL_RUN SHA' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum --check --status <<'FROZEN_SHA256'
b7f26d2b28c0787cc26385990e2d79cb0bfd67f3663fe6efcdc5d1aec40a24c3  /mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/independent-audit-v1/audit.py
d6e34a2a05aa4123dee521c7bbfaf5b631a9eb3fc5687a7833eb9b24178dca3d  /mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/independent-audit-v1/CONTRACT.md
FROZEN_SHA256
exec /usr/bin/timeout --signal=TERM --kill-after=1s 98s \
  /home/administrator/venvs/3dgs/bin/python -B -u \
  '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/independent-audit-v1/audit.py' --run \
  --result "$1" --result-sha256 "$2" --supervisor "$3" --supervisor-sha256 "$4" \
  --manifest "$5" --manifest-sha256 "$6" --combined-manifest "$7" --combined-manifest-sha256 "$8" \
  --config "$9" --config-sha256 "${10}" --external "${11}" --external-sha256 "${12}"
