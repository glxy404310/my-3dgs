#!/usr/bin/env bash
# Review/freeze first. Five actual producer path/SHA pairs; exactly one attempt.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py='/home/administrator/venvs/3dgs/bin/python'
if [[ "$#" != 10 ]]; then
  printf '%s\n' 'Usage: run-evaluation.sh RESULT_JSON SHA SUPERVISOR_JSON SHA EXTERNAL_JSON SHA CODEC SHA DECODED_PLY SHA' >&2
  exit 2
fi
if [[ -e "$here/actualrun-v1" ]]; then
  printf '%s\n' 'Refusing an existing actualrun-v1; no retry or overwrite.' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
external_start="$(date +%s%N)"
sha256sum --check --status <<'FROZEN_SHA256'
f766439a639afdaea36351e5d35c0fa1a6b66d64cbe6783a9b004fb39b257b6a  /mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/evaluate.py
0f9e2474564b1423869f82240c4178526d960920c43d034c3ffea18e1a1bbcb7  /mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/CONTRACT.md
8afba4389647500bb55a7a7ee5baefead02eecc193db38c65bb92703fc241cf2  /mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/ASSUMPTIONS.md
75dda71c84475286d857e5cfef4c97c81f162c5d0456b5912d78e386f9adf108  /mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/READONLY_PINS.json
FROZEN_SHA256
set +e
/usr/bin/timeout --signal=TERM --kill-after=3s 195s "$py" -B -u "$here/evaluate.py" --run \
  --result "$1" --result-sha256 "$2" --supervisor "$3" --supervisor-sha256 "$4" \
  --external "$5" --external-sha256 "$6" --codec "$7" --codec-sha256 "$8" \
  --decoded "$9" --decoded-sha256 "${10}"
command_exit=$?
set -e
external_end="$(date +%s%N)"
if [[ -d "$here/actualrun-v1" ]]; then
  "$py" -B -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); elapsed=(int(sys.argv[3])-int(sys.argv[2]))/1e9; value=dict(schema="direct-sh8-native-evaluation-v1/external-v1",exit_code=int(sys.argv[4]),attempts=1,actual_host_wall_seconds=elapsed,hard_limit_seconds=200,timeout_term_seconds=195,timeout_kill_after_seconds=3,within_200_seconds=elapsed<=200,combined_main_plus_audit_limit_seconds=300,independent_audit_reserved_seconds=100,launcher_sha256=hashlib.sha256((p/"run-evaluation.sh").read_bytes()).hexdigest()); (p/"actualrun-v1/EXTERNAL.json").write_text(json.dumps(value,indent=2)+"\n")' \
    "$here" "$external_start" "$external_end" "$command_exit"
fi
exit "$command_exit"
