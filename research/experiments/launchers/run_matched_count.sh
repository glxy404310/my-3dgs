#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
window=${1:?early or mid}
if [[ "$window" == early ]]; then
 checkpoint="$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth"
 event="$root/research/experiments/split-diagnostic-v0/early3600_seed0"
elif [[ "$window" == mid ]]; then
 checkpoint="$repo/output/truck_diag_ckpt7000_20261021/chkpnt7000.pth"
 event="$root/research/experiments/split-diagnostic-v0/online_seed0"
else exit 2
fi
result="$root/research/experiments/matched-count-v0/$window"
mkdir -p "$result"
"$python" "$root/scripts/experiments/split_diagnostic.py" --repo "$repo" --checkpoint "$checkpoint" \
 --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
 --seed 0 --online_interval 100 --event_file "$event/event_state.pth" --eval_views 0 --top_k 1481 \
 --recovery_steps 0 1 5 20 100 --variant las_split --output "$result/las_split" 2>&1 | tee "$result/las_split.log"
printf '0\n' > "$result/exit_code.txt"
