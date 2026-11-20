#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
seed=${1:-0}
result="$root/research/experiments/split-diagnostic-v0/early3600_seed$seed"
for variant in original_split las_split; do
 "$python" "$root/scripts/experiments/split_diagnostic_moments.py" --repo "$repo" \
  --checkpoint "$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth" \
  --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
  --seed "$seed" --online_interval 100 --event_file "$result/event_state.pth" --eval_views 0 --top_k 0 \
  --recovery_steps 0 1 5 20 100 --variant "$variant" --inherit_moments \
  --selection_file "$result/noop/selection.json" --output "$result/${variant}_moments" \
  2>&1 | tee "$result/${variant}_moments.log"
done
"$python" "$root/scripts/experiments/compare_split_results.py" --baseline "$result/noop" \
 --variants "$result/original_split" "$result/las_split" "$result/original_split_moments" "$result/las_split_moments" \
 --output "$result/comparison_moments.json" > "$result/comparison_moments.log"
