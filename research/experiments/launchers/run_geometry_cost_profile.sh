#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
mode=${1:-sanity}
if [[ "$mode" == sanity ]]; then
 sizes=(--views 2 --rounds 2 --repeats 3)
elif [[ "$mode" == main ]]; then
 sizes=(--views 8 --rounds 5 --repeats 30)
else exit 2;fi
out="$root/research/experiments/profile_output/geometry-cost-v0"
mkdir -p "$out"
[[ ! -e "$out/$mode" && ! -e "$out/$mode.log" ]] || { echo 'Existing output; refuse overwrite';exit 2; }
/home/administrator/venvs/3dgs/bin/python "$root/scripts/experiments/profile_split_geometry.py" \
 --repo /home/administrator/projects/gaussian-splatting \
 --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
 --event "$root/research/experiments/split-diagnostic-v0/early3600_seed0/event_state.pth" \
 --reference_root "$root/research/experiments/split-diagnostic-v0/early3600_seed0" \
 --output "$out/$mode" "${sizes[@]}" 2>&1 | tee "$out/$mode.log"
printf '0\n' > "$out/$mode/exit_code.txt"
