#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
mode=${1:-sanity}
if [[ "$mode" == sanity ]];then iteration=3622
elif [[ "$mode" == main ]];then iteration=4500
else exit 2;fi
out="$root/research/experiments/native-adc-exact-v0"
mkdir -p "$out"
[[ ! -e "$out/$mode" && ! -e "$out/$mode.log" ]] || { echo 'Existing output; refuse overwrite';exit 2; }
/home/administrator/venvs/3dgs/bin/python "$root/scripts/experiments/observe_native_adc.py" \
 --repo /home/administrator/projects/gaussian-splatting \
 --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
 --checkpoint /home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021/chkpnt3500.pth \
 --iterations "$iteration" --observe --exact_state --operator_stages --output "$out/$mode" 2>&1 | tee "$out/$mode.log"
printf '0\n' > "$out/$mode/exit_code.txt"
