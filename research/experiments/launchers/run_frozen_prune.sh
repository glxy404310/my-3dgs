#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python='/home/administrator/venvs/3dgs/bin/python'
out="$root/research/experiments/frozen-prune-v0"
[[ ! -e "$out" ]] || { echo 'Refuse overwrite';exit 2; }
mkdir -p "$out"
"$python" "$root/scripts/experiments/run_frozen_prune.py" \
 --repo /home/administrator/projects/gaussian-splatting \
 --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
 --checkpoint /home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021/chkpnt3500.pth \
 --iterations 3622 --seed 0 --method clone_corrected_las \
 --resume_package "$root/research/experiments/prune-lineage-v0/sanity/packages/seed0.pth" \
 --observe --exact_state --operator_stages --output "$out/capture" 2>&1 | tee "$out/capture.log"
printf '0\n' > "$out/capture/exit_code.txt"
"$python" "$root/scripts/experiments/verify_native_exact.py" "$out/capture"
"$python" "$root/scripts/experiments/frozen_prune_counterfactual.py" \
 --repo /home/administrator/projects/gaussian-splatting --capture_root "$out/capture" --output "$out/counterfactual" \
 2>&1 | tee "$out/counterfactual.log"
printf '0\n' > "$out/exit_code.txt"
