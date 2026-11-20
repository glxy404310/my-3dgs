#!/usr/bin/env bash
# Explicit user authorization after Truck FAIL/visual review; original launcher and gate retained.
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
root='/mnt/d/Documents/my 3dgs'
python='/home/administrator/venvs/3dgs/bin/python'
stage=${1:?sanity or main}
out="$root/research/experiments/adc-frequency-gate-v0/$stage/train"
[[ ! -e "$out" ]] || { echo 'Output exists; refuse overwrite'; exit 2; }
mkdir -p "$out/packages"
"$python" - "$root" "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
prior=root/'research/experiments/adc-frequency-gate-v0/main/truck/RESULTS.json'
old=json.loads(prior.read_text())
assert old['status']=='FAIL'
runner=root/'scripts/experiments/run_adc_frequency_gate.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest()=='46dc1636b9e8ad4630579d7af2ea3058e22487318da8fb0eeacb5022b706c840'
(out/'AUTHORIZATION.json').write_text(json.dumps(dict(
 authorization='User explicitly accepted observed Truck visual tradeoff and requested unchanged-parameter Train replication on 2026-10-22.',
 user_request='我看这个画质损失可以接受，先保持参数不变跑Train，看看能不能复现这个省时和画质折中。原来的FAIL保留，别为了过关改结果。',
 original_truck_status=old['status'],original_truck_results_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
 original_truck_gate_overridden=False,launch_basis='New explicit user authorization, not original Truck PASS',
 unchanged_runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
 thresholds_unchanged=True),indent=2,ensure_ascii=False))
PY
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$out/gpu_before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/processes_before.csv"
date --iso-8601=seconds > "$out/started_at.txt"
if [[ "$stage" == sanity ]]; then runs=(A0 B0); sanity=(--sanity)
elif [[ "$stage" == main ]]; then runs=(B0 A0 A1 B1 A0_repeat); sanity=()
else exit 2; fi
for run in "${runs[@]}"; do
 seed=${run:1:1}; interval=100
 [[ ${run:0:1} == B ]] && interval=200
 package="$out/packages/seed$seed.pth"
 create=(); [[ -e "$package" ]] || create=(--create-package)
 "$python" "$root/scripts/experiments/run_adc_frequency_gate.py" \
  --repo /home/administrator/projects/gaussian-splatting \
  --source_path /home/administrator/datasets/tandt_db/tandt/train \
  --checkpoint "$root/research/experiments/train-independent-v0/native/chkpnt3500.pth" \
  --package "$package" "${create[@]}" --eval --resolution 1 --data_device cpu --iterations 30000 \
  --seed "$seed" --interval "$interval" --output "$out/$run" "${sanity[@]}" \
  2>&1 | tee "$out/$run.log"
 echo 0 > "$out/$run/exit_code.txt"
done
if [[ "$stage" == main ]]; then
 "$python" "$root/scripts/experiments/analyze_adc_frequency_gate.py" "$out" | tee "$out/analysis.log"
fi
date --iso-8601=seconds > "$out/finished_at.txt"
echo 0 > "$out/exit_code.txt"
