#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != _timed ]]; then
 exec timeout 7200 bash "$0" _timed "$@"
fi
shift
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
root='/mnt/d/Documents/my 3dgs'
repo='/home/administrator/projects/gaussian-splatting'
py='/home/administrator/venvs/3dgs/bin/python'
mode=${1:?sanity or main}
[[ "$mode" == sanity || "$mode" == main ]]
base="$root/research/experiments/finite-substitution-v1/$mode"
[[ ! -e "$base" ]]
if [[ "$mode" == main ]]; then
 [[ "$(cat "$root/research/experiments/finite-substitution-v1/sanity/exit_code.txt")" == 0 ]]
 cases=12; steps=120
else cases=1; steps=2; fi
mkdir -p "$base"
for scene in truck train; do
 dir="$base/$scene"; mkdir -p "$dir"
 if [[ "$scene" == truck ]]; then checkpoint="$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth"
 else checkpoint="$root/research/experiments/train-independent-v0/native/chkpnt3500.pth"; fi
 common=(--repo "$repo" --source_path "/home/administrator/datasets/tandt_db/tandt/$scene" --eval --resolution 1 --data_device cpu
 --checkpoint "$checkpoint" --prior_packet "$root/research/experiments/split-capacity-v0/main/$scene/packet.pth"
 --packet "$dir/cases.pth" --cases "$cases" --steps "$steps" --train_pool 48)
 "$py" "$root/scripts/experiments/finite_substitution.py" "${common[@]}" --stage prepare --output "$dir/prepare" 2>&1 | tee "$dir/prepare.log"
 "$py" "$root/scripts/experiments/finite_substitution.py" "${common[@]}" --stage run --output "$dir/run" 2>&1 | tee "$dir/run.log"
 if [[ "$mode" == sanity ]]; then
  "$py" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["complete"] and s["cases"]>=1 and s["completed_arms"]==4*s["cases"] and s["steps"]==2, "Sanity requires a real completed case in each scene"' "$dir/run/summary.json"
 fi
done
printf '0\n' > "$base/exit_code.txt"
