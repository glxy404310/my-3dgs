#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
mode=${1:-sanity}
out="$root/research/experiments/native-adc-v0"
mkdir -p "$out"
if [[ "$mode" == sanity ]];then
 branches=(reference observed);iteration=3600
elif [[ "$mode" == main ]];then
 branches=(observed);iteration=4500
elif [[ "$mode" == repeat ]];then
 branches=(reference_repeat);iteration=3600
else exit 2;fi
for branch in "${branches[@]}";do
 result="$out/$mode/$branch"
 [[ ! -e "$result" && ! -e "$result.log" ]] || { echo 'Output exists; refuse overwrite';exit 2; }
 mkdir -p "$out/$mode"
 observe=();if [[ "$branch" == observed ]];then observe=(--observe);fi
 /home/administrator/venvs/3dgs/bin/python "$root/scripts/experiments/observe_native_adc.py" \
  --repo /home/administrator/projects/gaussian-splatting \
  --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
  --checkpoint /home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021/chkpnt3500.pth \
  --iterations "$iteration" --output "$result" "${observe[@]}" 2>&1 | tee "$result.log"
 printf '0\n' > "$result/exit_code.txt"
done
