#!/usr/bin/env bash
set -euo pipefail
ROOT='/mnt/d/Documents/my 3dgs'
EXPERIMENT="$ROOT/profile_output/nanogs-frozen-baseline-v0"
RUN="$EXPERIMENT/run-v1"
LOG="$EXPERIMENT/launch-v1.log"
WATCHDOG="$EXPERIMENT/WATCHDOG_v1.json"
PYTHON='/home/administrator/venvs/3dgs/bin/python'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[[ ! -e "$RUN" && ! -e "$LOG" && ! -e "$WATCHDOG" ]] || { echo 'Fresh run/log/watchdog required; no retry'; exit 2; }
START_NS=$(date +%s%N)
set +e
timeout --signal=TERM --kill-after=5s 295s "$PYTHON" -B -u \
  "$ROOT/scripts/experiments/benchmark_nanogs_native.py" --output-dir "$RUN" 2>&1 | tee "$LOG"
STATUS=("${PIPESTATUS[@]}")
set -e
END_NS=$(date +%s%N)
"$PYTHON" -B -c 'import json,sys; p,start,end,process,pipe=sys.argv[1:]; elapsed=(int(end)-int(start))/1e9; d=dict(process_exit_code=int(process),tee_exit_code=int(pipe),actual_process_and_pipe_wall_seconds=elapsed,within_300_seconds=elapsed<=300,complete=int(process)==0 and int(pipe)==0 and elapsed<=300); open(p,"w").write(json.dumps(d,indent=2)+"\n")' \
  "$WATCHDOG" "$START_NS" "$END_NS" "${STATUS[0]}" "${STATUS[1]}"
if [[ "${STATUS[0]}" != 0 || "${STATUS[1]}" != 0 ]]; then
  echo 'process nonzero/pipe/budget; original RESULTS cause preserved'
  exit 2
fi
"$PYTHON" -B -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["within_300_seconds"] else 2)' "$WATCHDOG"
