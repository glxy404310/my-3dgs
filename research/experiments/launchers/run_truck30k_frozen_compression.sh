#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
root='/mnt/d/Documents/my 3dgs'
base="$root/research/experiments/truck30k-frozen-compression-v0"
[[ ! -e "$base/run-v1" ]] || { echo 'Refuse overwrite'; exit 2; }
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv > "$base/gpu_before_v1.csv"
/home/administrator/venvs/3dgs/bin/python "$root/scripts/experiments/diagnose_truck30k_frozen_compression.py" \
  --repo /home/administrator/projects/gaussian-splatting \
  --source /home/administrator/datasets/tandt_db/tandt/truck \
  --model /home/administrator/projects/gaussian-splatting/output/truck_30000_r1 \
  --output "$base/run-v1" 2>&1 | tee "$base/run-v1.log"
echo 0 > "$base/exit_code_v1.txt"
