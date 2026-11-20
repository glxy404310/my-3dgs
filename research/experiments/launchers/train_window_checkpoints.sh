#!/usr/bin/env bash
set -euo pipefail
repo=/home/administrator/projects/gaussian-splatting
output="$repo/output/truck_windows_3500_11000_20261021"
if [[ -e "$output" ]]; then echo 'Output exists; refusing overwrite'; exit 2; fi
mkdir -p "$output"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
cd "$repo"
date --iso-8601=seconds | tee "$output/started_at.txt"
git rev-parse HEAD | tee "$output/repo_commit.txt"
set +e
/home/administrator/venvs/3dgs/bin/python train.py \
 --source_path /home/administrator/datasets/tandt_db/tandt/truck \
 --model_path "$output" --eval --resolution 1 --data_device cpu \
 --iterations 11000 --test_iterations 3500 11000 --save_iterations 3500 11000 \
 --checkpoint_iterations 3500 11000 2>&1 | tee "$output/train.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$output/exit_code.txt"
date --iso-8601=seconds | tee "$output/finished_at.txt"
exit "$status"
