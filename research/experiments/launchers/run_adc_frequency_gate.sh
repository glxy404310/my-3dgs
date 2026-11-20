#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
root='/mnt/d/Documents/my 3dgs'
python='/home/administrator/venvs/3dgs/bin/python'
scene=${1:?truck or train}
stage=${2:?sanity or main}
if [[ "$scene" == truck ]]; then
 checkpoint=/home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021/chkpnt3500.pth
 source=/home/administrator/datasets/tandt_db/tandt/truck
elif [[ "$scene" == train ]]; then
 checkpoint="$root/research/experiments/train-independent-v0/native/chkpnt3500.pth"
 source=/home/administrator/datasets/tandt_db/tandt/train
 "$python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASS"' "$root/research/experiments/adc-frequency-gate-v0/main/truck/RESULTS.json"
else exit 2; fi
out="$root/research/experiments/adc-frequency-gate-v0/$stage/$scene"
[[ ! -e "$out" ]] || { echo 'Output exists; refuse overwrite'; exit 2; }
mkdir -p "$out/packages"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv > "$out/gpu_before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/processes_before.csv"
date --iso-8601=seconds > "$out/started_at.txt"
if [[ "$stage" == sanity ]]; then
 runs=(A0 B0); sanity=(--sanity)
elif [[ "$stage" == main ]]; then
 sanity=()
 if [[ "$scene" == truck ]]; then runs=(A0 B0 B1 A1 A0_repeat); else runs=(B0 A0 A1 B1 A0_repeat); fi
else exit 2; fi
for run in "${runs[@]}"; do
 seed=${run:1:1}; interval=100
 [[ ${run:0:1} == B ]] && interval=200
 package="$out/packages/seed$seed.pth"
 create=(); [[ -e "$package" ]] || create=(--create-package)
 "$python" "$root/scripts/experiments/run_adc_frequency_gate.py" \
  --repo /home/administrator/projects/gaussian-splatting --source_path "$source" \
  --checkpoint "$checkpoint" --package "$package" "${create[@]}" \
  --eval --resolution 1 --data_device cpu --iterations 30000 \
  --seed "$seed" --interval "$interval" --output "$out/$run" "${sanity[@]}" \
  2>&1 | tee "$out/$run.log"
 echo 0 > "$out/$run/exit_code.txt"
done
if [[ "$stage" == main ]]; then
 "$python" "$root/scripts/experiments/analyze_adc_frequency_gate.py" "$out" | tee "$out/analysis.log"
fi
date --iso-8601=seconds > "$out/finished_at.txt"
echo 0 > "$out/exit_code.txt"
