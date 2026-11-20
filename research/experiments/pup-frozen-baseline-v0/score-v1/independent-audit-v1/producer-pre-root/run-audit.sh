#!/usr/bin/env bash
set -euo pipefail
[[ "$#" == 0 ]] || exit 2
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
exec /home/administrator/venvs/3dgs/bin/python -B -u \
  '/mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/score-v1/independent-audit-v1/audit.py' --run
