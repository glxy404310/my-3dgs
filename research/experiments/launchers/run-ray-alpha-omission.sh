#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
task_root='/mnt/d/Documents/my 3dgs'
task_exp="$task_root/research/experiments/ray-alpha-omission-v0"
/home/administrator/venvs/3dgs/bin/python -u "$task_root/scripts/experiments/diagnose_ray_alpha_omission.py" \
  --repo /home/administrator/projects/gaussian-splatting --source /home/administrator/datasets/tandt_db/tandt/truck \
  --model /home/administrator/projects/gaussian-splatting/output/truck_30000_r1 \
  --contract "$task_exp/CONTRACT.md" --output "$task_exp/run" 2>&1 | tee "$task_exp/run.log"
