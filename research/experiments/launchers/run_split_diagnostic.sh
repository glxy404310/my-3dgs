#!/usr/bin/env bash
set -euo pipefail
mode=${1:-sanity}
seed=${2:-0}
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
checkpoint="$repo/output/truck_diag_ckpt7000_20261021/chkpnt7000.pth"
folder="$mode"
if [[ "$mode" == natural ]]; then folder="natural_seed$seed"; fi
result="$root/research/experiments/split-diagnostic-v0/$folder"
mkdir -p "$result"
common=(--repo "$repo" --checkpoint "$checkpoint" --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu --seed "$seed")
if [[ "$mode" == sanity ]]; then
  common+=(--calib_views 1 --eval_views 2 --top_k 4 --recovery_steps 0 1)
elif [[ "$mode" == main ]]; then
  common+=(--calib_views 8 --eval_views 0 --top_k 128 --recovery_steps 0 1 5 20 100)
elif [[ "$mode" == natural ]]; then
  common+=(--calib_views 8 --eval_views 0 --top_k 0 --recovery_steps 0 1 5 20 100)
else
  echo "Unknown mode $mode"; exit 2
fi
for variant in noop original_split; do
  extra=()
  if [[ "$variant" != noop ]]; then
    extra+=(--selection_file "$result/noop/selection.json")
  fi
  "$python" "$root/scripts/experiments/split_diagnostic.py" "${common[@]}" --variant "$variant" --output "$result/$variant" "${extra[@]}" 2>&1 | tee "$result/$variant.log"
done
"$python" "$root/scripts/experiments/compare_split_results.py" --baseline "$result/noop" --variants "$result/original_split" --output "$result/comparison.json" 2>&1 | tee "$result/comparison.log"
printf '0\n' > "$result/exit_code.txt"
