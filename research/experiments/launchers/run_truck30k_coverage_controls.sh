#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
root='/mnt/d/Documents/my 3dgs'
base="$root/research/experiments/truck30k-coverage-controls-v0"
[[ ! -e "$base/run" ]] || { echo 'Refuse overwrite'; exit 2; }
/home/administrator/venvs/3dgs/bin/python "$root/scripts/experiments/diagnose_truck30k_coverage_controls.py" \
 --repo /home/administrator/projects/gaussian-splatting \
 --source /home/administrator/datasets/tandt_db/tandt/truck \
 --model /home/administrator/projects/gaussian-splatting/output/truck_30000_r1 \
 --previous "$root/research/experiments/truck30k-frozen-compression-v0/run-v1" \
 --output "$base/run" 2>&1 | tee "$base/run.log"
echo 0 > "$base/exit_code.txt"
