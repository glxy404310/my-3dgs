#!/usr/bin/env bash
set -euo pipefail
# This timestamp precedes both Python process startups, which count toward audit wall.
AUDIT_START_NS=$(date +%s%N)
ROOT='/mnt/d/Documents/my 3dgs'
PYTHON='/home/administrator/venvs/3dgs/bin/python'
for NAME in INDEPENDENT_PREMERGE_VERIFICATION.json INDEPENDENT_PREMERGE.log INDEPENDENT_PREMERGE_WATCHDOG.json INDEPENDENT_PREMERGE_LAUNCH_STATE.json; do
  [[ ! -e "$ROOT/research/experiments/nanogs-opacity-premerge-v0/$NAME" ]] || { echo 'Fresh audit paths required; no retry'; exit 2; }
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
set +e
"$PYTHON" -B - "$AUDIT_START_NS" "$ROOT" <<'PY'
import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
start_ns, root = int(sys.argv[1]), Path(sys.argv[2])
base = root/'research/experiments/nanogs-opacity-premerge-v0'
output = base/'INDEPENDENT_PREMERGE_VERIFICATION.json'
log = base/'INDEPENDENT_PREMERGE.log'
watch = base/'INDEPENDENT_PREMERGE_WATCHDOG.json'
state = base/'INDEPENDENT_PREMERGE_LAUNCH_STATE.json'
if any(p.exists() for p in (output, log, watch, state)):
    raise SystemExit('Fresh audit paths required; no automatic retry')
main_path = base/'WATCHDOG_v1.json'
main_bytes = main_path.read_bytes()
main = json.loads(main_bytes)
main_wall = float(main['actual_process_and_pipe_wall_seconds'])
remaining = float(main['remaining_shared_seconds'])
record = dict(status='INCONCLUSIVE', main_external_wall_seconds=main_wall,
    main_watchdog_sha256=hashlib.sha256(main_bytes).hexdigest(),
    remaining_shared_seconds_at_launch=remaining, hard_shared_limit_seconds=300,
    audit_start_unix_ns=start_ns, process_exit_code=None, timeout=False,
    cleanup_reserve_seconds=2, terminate_to_kill_grace_seconds=5,
    shared_definition='main external wall + audit external wall, including Python startups')
valid = (main.get('complete') is True and main.get('process_exit_code') == 0 and
         main.get('tee_exit_code') == 0 and 0 <= main_wall < 300 and
         abs(remaining - (300-main_wall)) <= 1e-9)
deadline_ns = start_ns + int(max(0., remaining)*1e9)
kill_ns = deadline_ns - 2_000_000_000
term_ns = kill_ns - 5_000_000_000
child = None
if not valid or remaining < 10 or time.time_ns() >= term_ns - 1_000_000_000:
    record['reason'] = 'Incomplete main run, inconsistent shared budget, or insufficient remaining time; no audit started'
else:
    command = [sys.executable, '-B', '-u', str(root/'scripts/experiments/verify_nanogs_premerge_independent.py'),
               '--work-deadline-unix-ns', str(term_ns-1_000_000_000), '--output', str(output)]
    record['command'] = command
    record['term_deadline_unix_ns'], record['kill_deadline_unix_ns'] = term_ns, kill_ns
    with log.open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record['audit_pid'] = child.pid
        try:
            child.wait(timeout=max(0., (term_ns-time.time_ns())/1e9))
        except subprocess.TimeoutExpired:
            record['timeout'] = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=max(0., (kill_ns-time.time_ns())/1e9))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        record['process_exit_code'] = child.returncode
    if output.is_file():
        result = json.loads(output.read_text())
        record['audit_result_status'] = result.get('status')
        record['audit_output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        record['status'] = result.get('status', 'INCONCLUSIVE')
    else:
        record['reason'] = 'Auditor produced no complete verification artifact'
    if record['timeout']:
        record['status'] = 'INCONCLUSIVE'
if not output.exists():
    with output.open('x') as stream:
        json.dump(dict(status='INCONCLUSIVE', reason=record.get('reason', 'Shared-budget timeout'),
            scientific_inputs_audited=False, original_SSIM_audit='FAIL_UNCHANGED'), stream, indent=2)
        stream.write('\n')
record['audit_external_wall_seconds'] = (time.time_ns()-start_ns)/1e9
record['combined_external_wall_seconds'] = main_wall + record['audit_external_wall_seconds']
record['within_shared_300_seconds'] = record['combined_external_wall_seconds'] <= 300
if not record['within_shared_300_seconds']:
    record['status'] = 'INCONCLUSIVE'
record['main_watchdog_unchanged'] = main_path.read_bytes() == main_bytes
if not record['main_watchdog_unchanged']:
    record['status'] = 'FAIL'
with state.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record), flush=True)
raise SystemExit(0 if record['status'] == 'PASS' and record['process_exit_code'] == 0 else 2)
PY
SUPERVISOR_EXIT=$?
AUDIT_END_NS=$(date +%s%N)
set -e
# Endpoint is after the supervisor and auditor Python processes have both exited.
# This final metadata writer is excluded exactly as the main launcher's WATCHDOG writer.
"$PYTHON" -B - "$AUDIT_START_NS" "$AUDIT_END_NS" "$ROOT" "$SUPERVISOR_EXIT" <<'PY'
import hashlib, json, sys
from pathlib import Path
start_ns, end_ns, root, supervisor = sys.argv[1:]
base = Path(root)/'research/experiments/nanogs-opacity-premerge-v0'
watch = base/'INDEPENDENT_PREMERGE_WATCHDOG.json'
if watch.exists():
    raise SystemExit('Existing watchdog preserved; no retry')
state_path = base/'INDEPENDENT_PREMERGE_LAUNCH_STATE.json'
if not state_path.exists():
    raise SystemExit('Supervisor did not create launch state; no verification is claimed')
record = json.loads(state_path.read_text())
if record['audit_start_unix_ns'] != int(start_ns):
    raise SystemExit('Launch state belongs to another attempt; preserved unchanged')
record['supervisor_exit_code'] = int(supervisor)
record['audit_external_wall_seconds'] = (int(end_ns)-int(start_ns))/1e9
record['combined_external_wall_seconds'] = record['main_external_wall_seconds']+record['audit_external_wall_seconds']
record['within_shared_300_seconds'] = record['combined_external_wall_seconds'] <= 300
record['external_scope'] = 'Bash timestamp before supervisor Python startup through both supervisor/auditor Python shutdown; final WATCHDOG metadata writer excluded as in main launcher.'
record['launcher_state_sha256'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
if not record['within_shared_300_seconds']:
    record['status'] = 'INCONCLUSIVE'
if int(supervisor) != 0 and record['status'] == 'PASS':
    record['status'] = 'INCONCLUSIVE'
with watch.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record), flush=True)
raise SystemExit(0 if record['status'] == 'PASS' and record['process_exit_code'] == 0 else 2)
PY
