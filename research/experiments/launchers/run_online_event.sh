#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
seed=${1:-0}
checkpoint=${2:-$repo/output/truck_diag_ckpt7000_20261021/chkpnt7000.pth}
label=${3:-online}
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
result="$root/research/experiments/split-diagnostic-v0/${label}_seed$seed"
mkdir -p "$result"
common=(--repo "$repo" --checkpoint "$checkpoint" --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu --seed "$seed" --online_interval 100 --event_file "$result/event_state.pth" --eval_views 0 --top_k 0 --recovery_steps 0 1 5 20 100)
for variant in noop original_split las_split; do
  extra=()
  if [[ "$variant" == noop ]]; then extra+=(--create_event); else extra+=(--selection_file "$result/noop/selection.json"); fi
  "$python" "$root/scripts/experiments/split_diagnostic.py" "${common[@]}" --variant "$variant" --output "$result/$variant" "${extra[@]}" 2>&1 | tee "$result/$variant.log"
done
"$python" "$root/scripts/experiments/compare_split_results.py" --baseline "$result/noop" --variants "$result/original_split" "$result/las_split" --output "$result/comparison.json" 2>&1 | tee "$result/comparison.log"
printf '0\n' > "$result/exit_code.txt"
