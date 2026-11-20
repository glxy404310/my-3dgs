#!/usr/bin/env bash
# One independent discovery collection. No retry; TERM295/KILL300.
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
root='/mnt/d/Documents/my 3dgs'
base="$root/research/experiments/teacher-artifact-discovery-v0"
run="$base/run-v1"
log="$base/run-v1.log"
watchdog="$base/WATCHDOG_v1.json"
[[ ! -e "$run" && ! -e "$log" && ! -e "$watchdog" ]] || {
    echo 'Refuse overwrite or retry of the independent discovery collection' >&2
    exit 2
}
started=$(date +%s.%N)
set +e
timeout --signal=TERM --kill-after=5s 295s \
    /home/administrator/venvs/3dgs/bin/python \
    "$root/scripts/experiments/collect_teacher_artifact_discovery.py" \
    --output "$run" 2>&1 | tee "$log"
pipeline_codes=("${PIPESTATUS[@]}")
set -e
finished=$(date +%s.%N)
elapsed=$(awk -v first="$started" -v last="$finished" 'BEGIN {printf "%.9f", last-first}')
process_exit=${pipeline_codes[0]}
tee_exit=${pipeline_codes[1]}
within_budget=$(awk -v elapsed="$elapsed" 'BEGIN {print (elapsed <= 300.0 ? "true" : "false")}')
complete=false
if [[ "$process_exit" == 0 && "$tee_exit" == 0 && "$within_budget" == true ]]; then
    complete=true
fi
printf '{\n  "process_timeout_exit_code": %d,\n  "tee_exit_code": %d,\n  "external_process_pipe_wall_seconds": %s,\n  "wall_scope": "before timeout launch through Python/timeout/tee return; excludes sidecar write",\n  "term_deadline_seconds": 295,\n  "kill_grace_seconds": 5,\n  "hard_budget_seconds": 300,\n  "within_300_seconds": %s,\n  "external_completion_confirmed": %s,\n  "automatic_retry_permitted": false,\n  "planned_new_native_forwards": 10,\n  "planned_reused_teacher_views": 3,\n  "partial_results_path": "run-v1/RESULTS.json"\n}\n' \
    "$process_exit" "$tee_exit" "$elapsed" "$within_budget" "$complete" > "$watchdog"
if [[ "$complete" != true ]]; then
    echo 'Collection nonzero/pipe/budget incomplete; preserve original RESULTS reason; do not retry' >&2
    exit 2
fi
exit 0
