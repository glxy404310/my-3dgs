#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
event="$root/research/experiments/split-diagnostic-v0/early3600_seed0"
mode=${1:-main}
result="$root/research/experiments/split-refit-v0/$mode"
mkdir -p "$result"
common=(--repo "$repo" --checkpoint "$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth" --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu --seed 0 --online_interval 100 --event_file "$event/event_state.pth" --variant las_split --inherit_moments --refit_steps 4 --refit_lambda .1 --refit_tau .03 --training_time_budget 1.5)
if [[ "$mode" == sanity ]]; then
 targets=(gt);common+=(--eval_views 2 --top_k 4 --recovery_steps 0 1)
elif [[ "$mode" == main ]]; then
 targets=(gt auxiliary_gt parent);common+=(--eval_views 0 --top_k 0 --selection_file "$event/noop/selection.json" --recovery_steps 0 1 5 20 100)
else exit 2
fi
for target in "${targets[@]}"; do
 "$python" "$root/scripts/experiments/split_refit_pilot.py" "${common[@]}" --refit_target "$target" --output "$result/$target" 2>&1 | tee "$result/$target.log"
done
printf '0\n' > "$result/exit_code.txt"
