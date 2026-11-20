#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python='/home/administrator/venvs/3dgs/bin/python'
out="$root/research/experiments/train-independent-v0"
stage=${1:-sanity}
if [[ "$stage" == sanity ]];then iterations=650; name=native-sanity
elif [[ "$stage" == main ]];then
 iterations=4500; name=native
 [[ -e "$out/native-sanity/exit_code.txt" ]] || { echo 'Sanity required';exit 2; }
else exit 2;fi
[[ ! -e "$out/$name" ]] || { echo 'Refuse overwrite';exit 2; }
[[ -e "$out/INPUT_CHECK_V2.json" ]] || exit 2
"$python" "$root/scripts/experiments/train_native_census.py" \
 --repo /home/administrator/projects/gaussian-splatting \
 --source_path /home/administrator/datasets/tandt_db/tandt/train \
 --eval --resolution 1 --data_device cpu --iterations "$iterations" --stage "$stage" --output "$out/$name" \
 2>&1 | tee "$out/$name.log"
printf '0\n' > "$out/$name/exit_code.txt"
