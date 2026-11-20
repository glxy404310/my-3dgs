#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
py='/home/administrator/venvs/3dgs/bin/python'
event=${1:?event required}
if [[ "$event" == 3600 ]]; then capture='frozen-prune-v0'
elif [[ "$event" == 3700 ]]; then
 capture='frozen-prune-confirm-v0'
 [[ "$(cat "$root/research/experiments/world-prune-controls-v0/event3600/exit_code.txt")" == 0 ]]
else exit 2; fi
base="$root/research/experiments/world-prune-controls-v0"
mkdir -p "$base"
[[ ! -e "$base/event$event" ]]
"$py" "$root/scripts/experiments/world_prune_strong_controls.py" \
 --repo /home/administrator/projects/gaussian-splatting \
 --capture_root "$root/research/experiments/$capture/capture" \
 --output "$base/event$event" --event "$event" \
 --extent_check "$root/research/experiments/train-independent-v0/SCENE_EXTENT_CHECK.json" \
 2>&1 | tee "$base/event$event.log"
printf '0\n' > "$base/event$event/exit_code.txt"
