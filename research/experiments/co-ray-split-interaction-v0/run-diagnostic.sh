#!/usr/bin/env bash
# Root must first read all code and freeze both manifests. No implicit defaults.
set -euo pipefail
if [[ "$#" != 5 ]]; then
  printf '%s\n' 'Usage: run-diagnostic.sh CONTRACT SHA256 ARMS SHA256 RUN_DIRECTORY' >&2
  exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py='/home/administrator/venvs/3dgs/bin/python'
if [[ -e "$5" ]]; then
  printf '%s\n' 'Existing output refused; no retry or overwrite.' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
started="$(date +%s%N)"
set +e
/usr/bin/timeout --signal=TERM --kill-after=3s 292s "$py" -B -u "$here/execute.py" --run \
  --contract "$1" --contract-sha256 "$2" --arms "$3" --arms-sha256 "$4" --output "$5"
code=$?
set -e
finished="$(date +%s%N)"
if [[ -d "$5" ]]; then
  "$py" -B -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); record=dict(schema="co-ray-split-interaction-v0/external-child-v1", child_shell_wall_seconds=(int(sys.argv[3])-int(sys.argv[2]))/1e9, exit_code=int(sys.argv[4]), timeout_TERM_seconds=292, timeout_KILL_after_seconds=3, maximum_main_seconds=300, attempts=1, launcher_sha256=hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest(), root_external_wall_required=True); target=root/"EXTERNAL_CHILD.json"; stream=target.open("x",encoding="utf8"); json.dump(record,stream,indent=2); stream.write("\n"); stream.close()' \
    "$5" "$started" "$finished" "$code" "$here/run-diagnostic.sh"
fi
exit "$code"
