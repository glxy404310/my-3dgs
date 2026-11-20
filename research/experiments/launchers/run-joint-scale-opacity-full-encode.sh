#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
task_root='/mnt/d/Documents/my 3dgs'
task_exp="$task_root/research/experiments/joint-scale-opacity-rounding-v0"
/home/administrator/venvs/3dgs/bin/python -u "$task_root/scripts/experiments/encode_joint_rounding_full.py" \
  --model /home/administrator/projects/gaussian-splatting/output/truck_30000_r1 \
  --pilot "$task_exp/run" --contract "$task_exp/FULL_CONTRACT.md" \
  --output "$task_exp/full-codec" 2>&1 | tee "$task_exp/full-encode.log"
