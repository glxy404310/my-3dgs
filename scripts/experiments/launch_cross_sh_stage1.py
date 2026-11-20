"""One-shot CPU launch with an external 1800-second process deadline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/d/Documents/my 3dgs')
STAGE = ROOT / 'research/experiments/cross-gaussian-sh-rounding-v0'
SCRIPT = ROOT / 'scripts/experiments/diagnose_cross_gaussian_sh_rounding.py'
CONTRACT = STAGE / 'CONTRACT_STAGE1.md'
CONTRACT_SHA = '3387c3502a613ced7c04e1e7e16c6cf2d0707d35cc9fe9e296992ac0e5a0f97c'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = STAGE / 'run-stage1'
    log = STAGE / 'run-stage1.log'
    launch_result = STAGE / 'LAUNCH_STAGE1.json'
    if output.exists() or log.exists() or launch_result.exists():
        raise RuntimeError('One-shot launch output already exists')
    if digest(CONTRACT) != CONTRACT_SHA:
        raise RuntimeError('Contract changed before launch')
    command = [sys.executable, '-B', str(SCRIPT), '--mode', 'stage1',
               '--contract', str(CONTRACT), '--contract-sha256', CONTRACT_SHA,
               '--output', str(output)]
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    info = dict(command=command, script_sha256_before=digest(SCRIPT),
                launcher_sha256=digest(Path(__file__)), contract_sha256=CONTRACT_SHA,
                external_timeout_seconds=1800, timeout=False, CPU_only=True)
    start = time.monotonic()
    try:
        with log.open('x', encoding='utf-8') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, timeout=1800, check=False)
        info['child_exit_code'] = completed.returncode
    except subprocess.TimeoutExpired:
        info.update(timeout=True, child_exit_code=None)
    finally:
        info['external_wall_seconds'] = time.monotonic() - start
        info['script_sha256_after'] = digest(SCRIPT)
        info['script_unchanged'] = info['script_sha256_before'] == info['script_sha256_after']
        launch_result.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    result_path = output / 'RESULTS.json'
    summary = json.loads(result_path.read_text()) if result_path.exists() else {}
    print(json.dumps(dict(launch=info, result={key: summary.get(key) for key in (
        'decision', 'tool_valid', 'integrity_complete', 'stage1_complete',
        'selected_capacity_pass_count', 'failure_type', 'failure')})), flush=True)
    return 124 if info['timeout'] else info['child_exit_code']


if __name__ == '__main__':
    sys.exit(main())
