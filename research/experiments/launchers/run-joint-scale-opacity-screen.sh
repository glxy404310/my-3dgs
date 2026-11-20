#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
task_root='/mnt/d/Documents/my 3dgs'
task_exp="$task_root/research/experiments/joint-scale-opacity-rounding-v0"
/home/administrator/venvs/3dgs/bin/python -m py_compile "$task_root/scripts/experiments/screen_joint_scale_opacity_rounding.py"
/home/administrator/venvs/3dgs/bin/python -u "$task_root/scripts/experiments/screen_joint_scale_opacity_rounding.py" \
  --model /home/administrator/projects/gaussian-splatting/output/truck_30000_r1 \
  --previous "$task_root/research/experiments/truck30k-frozen-compression-v0/run-v1" \
  --coverage "$task_root/research/experiments/truck30k-coverage-controls-v0/run" \
  --contract "$task_exp/CONTRACT.md" \
  --output "$task_exp/run" 2>&1 | tee "$task_exp/run.log"
