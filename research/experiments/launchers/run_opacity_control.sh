#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
seed=${1:-0}
variant=${2:-opacity_corrected_split}
result="$root/research/experiments/split-diagnostic-v0/natural_seed$seed"
"$python" "$root/scripts/experiments/split_diagnostic.py" --repo "$repo" \
  --checkpoint "$repo/output/truck_diag_ckpt7000_20261021/chkpnt7000.pth" \
  --source_path /home/administrator/datasets/tandt_db/tandt/truck \
  --eval --resolution 1 --data_device cpu --seed "$seed" --calib_views 8 --eval_views 0 --top_k 0 \
  --recovery_steps 0 1 5 20 100 --variant "$variant" \
  --selection_file "$result/noop/selection.json" --output "$result/$variant" \
  2>&1 | tee "$result/$variant.log"
"$python" "$root/scripts/experiments/compare_split_results.py" --baseline "$result/noop" \
  --variants "$result/original_split" "$result/$variant" --output "$result/comparison_with_$variant.json" \
  2>&1 | tee "$result/comparison_with_$variant.log"
