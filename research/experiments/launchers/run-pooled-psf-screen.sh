#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
task_root='/mnt/d/Documents/my 3dgs'
task_exp="$task_root/research/experiments/pooled-psf-screen-v0"
timeout --signal=TERM --kill-after=5s 300s \
  /home/administrator/venvs/3dgs/bin/python -u "$task_root/scripts/experiments/diagnose_pooled_psf_screen.py" \
  --upstream "$task_root/research/experiments/pruned-opacity-transfer-v0/run" \
  --contract "$task_exp/CONTRACT.md" --output "$task_exp/run" \
  2>&1 | tee "$task_exp/run.log"
