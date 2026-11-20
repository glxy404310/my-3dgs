#!/usr/bin/env python3
"""Stdlib-only, single-attempt external wall/RSS supervisor for CPU audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WALL_SECONDS, RSS_BYTES = 120, 8 << 30


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rss_tree(root_pid):
    parent, rss = {}, {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            content = (directory/'status').read_text()
            owner = re.search(r'^PPid:\s+(\d+)', content, re.M)
            memory = re.search(r'^VmRSS:\s+(\d+) kB', content, re.M)
            parent[int(directory.name)] = int(owner[1])
            rss[int(directory.name)] = int(memory[1])*1024 if memory else 0
        except (OSError, TypeError):
            continue
    included = {root_pid}
    while True:
        grown = included | {pid for pid, owner in parent.items() if owner in included}
        if grown == included:
            return sum(rss.get(pid, 0) for pid in included), len(included)
        included = grown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New independent-audit-v1/run-* directory')
    parser.add_argument('--run', action='store_true', help='Explicit actual audit; requires root-frozen completed input pins')
    args = parser.parse_args()
    need(args.run and sys.platform == 'linux', 'Explicit actual CPU audit in existing Linux environment required')
    output = args.output.resolve()
    need(output.parent == HERE and re.fullmatch(r'run-[A-Za-z0-9_-]+', output.name), 'New local audit run path required')
    need(not output.exists() and re.fullmatch('[0-9a-f]{64}', args.contract_sha256), 'Existing outputs and placeholder pins are refused')
    need(sha(args.contract) == args.contract_sha256, 'Actual root contract hash differs')
    # No array/Torch imports or child process before root freeze validation.
    contract = json.loads(args.contract.read_text(encoding='utf8'))
    need(contract['frozen_by_root'] is True and contract['reviewed_full_code'] is True and
         contract['reviewed_code_sha256']['run-audit.py'] == sha(__file__) and
         contract['reviewed_code_sha256']['audit.py'] == sha(HERE/'audit.py'), 'Root full code freeze absent')
    output.mkdir()
    started, active, peak, largest_tree = time.monotonic(), None, 0, 1
    report = {'schema': 'co-ray-split-interaction-v0/audit-host-record-v1', 'phase': 'audit',
        'complete': False, 'exit_code': None, 'automatic_retries': 0, 'wall_limit_seconds': WALL_SECONDS,
        'RSS_limit_bytes': RSS_BYTES, 'CUDA_VISIBLE_DEVICES': '', 'cpu_threads': 1,
        'formal_PASS_scope': 'SAVED_ARTIFACT_CONSISTENCY_ONLY', 'training_authorized': False}
    def interrupted(number, _frame):
        raise TimeoutError('Audit supervisor signal '+str(number))
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, interrupted)
    signal.alarm(WALL_SECONDS)
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CO_RAY_CPU_AUDIT_WORKER='1', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1', MALLOC_ARENA_MAX='1')
        env.pop('PYTHONPATH', None)
        with (output/'AUDIT_WORKER.log').open('x', encoding='utf8') as log:
            active = subprocess.Popen([sys.executable, '-B', '-u', str(HERE/'audit.py'), '--contract', str(args.contract.resolve()),
                '--contract-sha256', args.contract_sha256, '--output', str(output)], stdout=log, stderr=subprocess.STDOUT,
                cwd=output, env=env, start_new_session=True)
            while active.poll() is None:
                rss, count = rss_tree(os.getpid())
                peak, largest_tree = max(peak, rss), max(largest_tree, count)
                need(peak <= RSS_BYTES, 'Audit sampled process-tree RSS exceeded8GiB')
                need(time.monotonic()-started < WALL_SECONDS, 'Audit external wall exceeded120seconds')
                time.sleep(.1)
            report['exit_code'] = active.returncode
        need(active.returncode == 0 and (output/'AUDIT.json').is_file() and not (output/'FAILURE.json').exists(), 'CPU audit did not complete')
        audited = json.loads((output/'AUDIT.json').read_text(encoding='utf8'))
        need(audited['status'] == 'SAVED_ARTIFACT_CONSISTENCY_PASS' and
             audited['external_seconds_before_audit']+time.monotonic()-started <= 600, 'Cumulative600second limit exceeded')
        report['complete'] = True
        report['audit_sha256'] = sha(output/'AUDIT.json')
    except BaseException as error:
        report['error_type'], report['error'] = type(error).__name__, str(error)
    finally:
        signal.alarm(0)
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=.5)
        report.update(wall_seconds=time.monotonic()-started, peak_tree_RSS_bytes=peak,
            largest_observed_process_tree=largest_tree, worker_reaped=active is not None and active.poll() is not None,
            supervisor_array_libraries_imported=any(name in sys.modules for name in ('numpy', 'torch', 'scipy')))
        if report['wall_seconds'] > WALL_SECONDS:
            report['complete'] = False
            report['error'] = 'Actual external wall exceeded120seconds; any child PASS is ineligible'
        write(output/'AUDIT_HOST_RECORD.json', report)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
