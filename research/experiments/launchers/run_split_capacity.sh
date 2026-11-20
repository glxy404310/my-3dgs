#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
root='/mnt/d/Documents/my 3dgs'
repo='/home/administrator/projects/gaussian-splatting'
py='/home/administrator/venvs/3dgs/bin/python'
mode=${1:?sanity or main}
[[ "$mode" == sanity || "$mode" == main ]]
base="$root/research/experiments/split-capacity-v0/$mode"
if [[ "$mode" == main ]]; then
 [[ "$(cat "$root/research/experiments/split-capacity-v0/sanity/exit_code.txt")" == 0 ]]
fi
[[ ! -e "$base" ]]
mkdir -p "$base"
if [[ "$mode" == sanity ]]; then parents=16; candidates=4; births=2; steps=2; evaluation=4
else parents=128; candidates=64; births=16; steps=500; evaluation=0; fi
scenes=(truck train)
if [[ "$mode" == sanity ]]; then scenes=(truck); fi
for scene in "${scenes[@]}"; do
 dir="$base/$scene"; mkdir -p "$dir"
 if [[ "$scene" == truck ]]; then checkpoint="$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth"
 else checkpoint="$root/research/experiments/train-independent-v0/native/chkpnt3500.pth"; fi
 common=(--repo "$repo" --source_path "/home/administrator/datasets/tandt_db/tandt/$scene" --eval --resolution 1 --data_device cpu
 --checkpoint "$checkpoint" --packet "$dir/packet.pth" --parents "$parents" --candidates "$candidates" --births "$births"
 --steps "$steps" --eval_views "$evaluation" --seed 0)
 "$py" "$root/scripts/experiments/split_capacity_pilot.py" "${common[@]}" --stage prepare --output "$dir/prepare" 2>&1 | tee "$dir/prepare.log"
 "$py" "$root/scripts/experiments/split_capacity_pilot.py" "${common[@]}" --stage neighbors --output "$dir/neighbors" 2>&1 | tee "$dir/neighbors.log"
 for arm in noop native random long mean conflict select_projected select_raw select_gradient select_random; do
  "$py" "$root/scripts/experiments/split_capacity_pilot.py" "${common[@]}" --stage run --arm "$arm" --output "$dir/$arm" 2>&1 | tee "$dir/$arm.log"
 done
done
printf '0\n' > "$base/exit_code.txt"
