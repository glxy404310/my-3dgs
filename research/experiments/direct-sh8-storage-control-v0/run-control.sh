#!/usr/bin/env bash
# Frozen one-attempt launcher. Do not run during preparation.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py='/home/administrator/venvs/3dgs/bin/python'
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ -e "$here/run-v1" ]]; then
  printf '%s\n' 'Refusing an existing run-v1; no retry or overwrite.' >&2
  exit 2
fi
external_start="$(date +%s)"
set +e
timeout --signal=TERM --kill-after=5s 175s "$py" -B "$here/run_control.py" --output "$here/run-v1"
command_exit=$?
set -e
external_end="$(date +%s)"
launcher_sha="$(sha256sum -- "$here/run-control.sh")"
launcher_sha="${launcher_sha%% *}"
if [[ -d "$here/run-v1" ]]; then
  printf '{"schema":"direct-sh8-storage-control-v0/external-v1","exit_code":%d,"timeout_term_seconds":175,"timeout_kill_after_seconds":5,"maximum_seconds":180,"elapsed_seconds":%d,"launcher_sha256":"%s","automatic_retry":false}\n' \
    "$command_exit" "$((external_end - external_start))" "$launcher_sha" > "$here/run-v1/EXTERNAL.json"
fi
exit "$command_exit"
