#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
event="$root/research/experiments/split-diagnostic-v0/early3600_seed0"
for target in noop gradient random101 random202; do
 variant=las_split
 selection="$root/research/experiments/matched-count-v0/early/las_split/selection.json"
 if [[ "$target" == noop ]]; then variant=noop; fi
 if [[ "$target" == random101 ]]; then selection="$root/research/experiments/candidate-composition-v0/random_101/planned_selection.json"; fi
 if [[ "$target" == random202 ]]; then selection="$root/research/experiments/candidate-composition-v0/random_202/planned_selection.json"; fi
 result="$root/research/experiments/candidate-composition-long-v0/$target"
 mkdir -p "$result"
 "$python" "$root/scripts/experiments/split_diagnostic.py" --repo "$repo" \
  --checkpoint "$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth" \
  --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
  --seed 0 --online_interval 100 --event_file "$event/event_state.pth" --eval_views 0 --top_k 1481 \
  --selection_file "$selection" --recovery_steps 0 20 100 200 500 --variant "$variant" \
  --output "$result/run" 2>&1 | tee "$result/run.log"
 printf '0\n' > "$result/exit_code.txt"
done
