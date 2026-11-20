#!/usr/bin/env bash
# Root invokes only after reviewing code and freezing actual passed SH0+SH3 records.
# Scientific output: score-v1/run-v1 must be absent; existing score-v1 docs stay intact.
set -euo pipefail
if [[ "$#" != 3 ]]; then
  echo 'Usage: run-score.sh ACTUAL_KERNEL_CERTIFICATION_JSON CERTIFICATION_SHA256 NATIVE_PARITY_NPZ_SHA256' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
exec /usr/bin/timeout --signal=TERM --kill-after=5s 895s \
  /home/administrator/venvs/3dgs/bin/python -B -u \
  '/mnt/d/Documents/my 3dgs/scripts/experiments/run_pup_frozen_score.py' --run \
  --kernel-certification "$1" --kernel-certification-sha256 "$2" \
  --parity-reference-sha256 "$3"
