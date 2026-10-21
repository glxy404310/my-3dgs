#!/usr/bin/env bash
set -uo pipefail

repo=/home/administrator/projects/gaussian-splatting
venv=/home/administrator/venvs/3dgs
source_path=/home/administrator/datasets/tandt_db/tandt/truck
model_path="$repo/output/truck_diag_ckpt7000_20261021"
log_path="$model_path/train.log"

mkdir -p "$model_path"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2

cd "$repo"

printf 'started_at=%s\n' "$(date --iso-8601=seconds)" | tee "$model_path/run_meta.txt"
printf 'repo_commit=%s\n' "$(git rev-parse HEAD)" | tee -a "$model_path/run_meta.txt"
printf 'command=train.py --source_path %s --model_path %s --eval --resolution 1 --iterations 7000 --test_iterations 7000 --save_iterations 7000 --checkpoint_iterations 7000\n' "$source_path" "$model_path" | tee -a "$model_path/run_meta.txt"

set +e
"$venv/bin/python" train.py \
  --source_path "$source_path" \
  --model_path "$model_path" \
  --eval \
  --resolution 1 \
  --iterations 7000 \
  --test_iterations 7000 \
  --save_iterations 7000 \
  --checkpoint_iterations 7000 \
  2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" | tee -a "$model_path/run_meta.txt"
printf 'exit_code=%s\n' "$status" | tee -a "$model_path/run_meta.txt"
printf '%s\n' "$status" > "$model_path/train_exit_code.txt"
exit "$status"
