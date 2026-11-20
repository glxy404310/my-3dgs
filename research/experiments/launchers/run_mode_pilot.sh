#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
event="$root/research/experiments/split-diagnostic-v0/early3600_seed0"
mode=${1:-sanity}
result="$root/research/experiments/mode-state-v0/$mode";mkdir -p "$result"
common=(--repo "$repo" --checkpoint "$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth" --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu --seed 0 --online_interval 100 --event_file "$event/event_state.pth" --variant las_split)
if [[ "$mode" == sanity ]]; then
 policies=(legacy mode_clock);common+=(--eval_views 2 --top_k 4 --recovery_steps 0 1)
elif [[ "$mode" == main ]]; then
 policies=(legacy inherit fresh rotated_fresh rotated_old_difference mode_clock)
 common+=(--eval_views 0 --top_k 1481 --selection_file "$root/research/experiments/matched-count-v0/early/las_split/selection.json" --recovery_steps 0 20 100 200 500)
else exit 2
fi
for policy in "${policies[@]}";do
 "$python" "$root/scripts/experiments/split_mode_pilot.py" "${common[@]}" --child_optimizer "$policy" --output "$result/$policy" 2>&1 | tee "$result/$policy.log"
done
printf '0\n' > "$result/exit_code.txt"
