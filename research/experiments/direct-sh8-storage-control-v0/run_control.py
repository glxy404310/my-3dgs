#!/usr/bin/env python3
"""One fixed CPU scientific attempt. Run only after root review/authorization."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply')
SOURCE_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
POINTS = 2050770
RSS_LIMIT = 8 * 1024 ** 3
WORK_SECONDS = 160
ALARM_SECONDS = 165
HERE = Path(__file__).resolve().parent


class AbortRun(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha256(path, deadline=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise AbortRun('internal deadline during hashing')
            data = stream.read(8 * 1024 * 1024)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def tree_rss():
    """Sample Linux resident memory, not address space; include descendants."""
    processes = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / 'status').read_text()
            ppid = rss = 0
            for line in status.splitlines():
                if line.startswith('PPid:'):
                    ppid = int(line.split()[1])
                elif line.startswith('VmRSS:'):
                    rss = int(line.split()[1]) * 1024
            processes[int(entry.name)] = (ppid, rss)
        except (FileNotFoundError, ProcessLookupError):
            continue
    descendants = {os.getpid()}
    while True:
        added = {pid for pid, (ppid, _) in processes.items() if ppid in descendants} - descendants
        if not added:
            return sum(processes.get(pid, (0, 0))[1] for pid in descendants)
        descendants.update(added)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    work_deadline = started + WORK_SECONDS
    final_deadline = started + ALARM_SECONDS
    child = None
    current_stage = None
    result = {'schema': 'direct-sh8-storage-control-v0/result-v1', 'status': 'partial',
              'started_utc': utc(), 'points': POINTS, 'sh_degree': 3,
              'source': {'path': str(SOURCE), 'expected_sha256': SOURCE_SHA},
              'codec': {'path': str(output / 'control.dsh8')},
              'decoded_ply': {'path': str(output / 'decoded.ply')},
              'code_sha256': {}, 'reports': {}}
    supervisor = {'schema': 'direct-sh8-storage-control-v0/supervisor-v1',
                  'status': 'partial', 'started_utc': result['started_utc'],
                  'rss_limit_bytes': RSS_LIMIT, 'rss_sample_seconds': 0.1,
                  'work_deadline_seconds': WORK_SECONDS, 'alarm_seconds': ALARM_SECONDS,
                  'peak_tree_rss_bytes': 0, 'stages': [], 'automatic_retry': False}

    def alarm(signum, frame):
        raise AbortRun(f'received signal {signum}; cleanup margin active')

    def checkpoint():
        write_json(output / 'RESULT.json', result)
        write_json(output / 'SUPERVISOR.json', supervisor)

    def run_stage(name, arguments):
        nonlocal child, current_stage
        if time.monotonic() >= work_deadline:
            raise AbortRun('160-second work deadline')
        command = [sys.executable, '-B', str(HERE / (name + '.py'))] + arguments
        log_path = output / (name + '.log')
        stage = {'name': name, 'argv': command, 'started_utc': utc(),
                 'exit_code': None, 'peak_tree_rss_bytes': 0, 'log': str(log_path),
                 'start_monotonic': time.monotonic()}
        current_stage = stage
        supervisor['stages'].append(stage)
        checkpoint()
        stage_start = time.monotonic()
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while True:
                rss = tree_rss()
                supervisor['peak_tree_rss_bytes'] = max(supervisor['peak_tree_rss_bytes'], rss)
                stage['peak_tree_rss_bytes'] = max(stage['peak_tree_rss_bytes'], rss)
                if rss > RSS_LIMIT:
                    raise AbortRun('8-GiB aggregate process-tree RSS limit exceeded')
                if time.monotonic() >= work_deadline:
                    raise AbortRun('160-second work deadline')
                code = child.poll()
                if code is not None:
                    stage['exit_code'] = code
                    break
                time.sleep(0.1)
        stage['elapsed_seconds'] = time.monotonic() - stage_start
        stage.pop('start_monotonic', None)
        stage['log_sha256'] = sha256(log_path, final_deadline)
        child = None
        current_stage = None
        checkpoint()
        if stage['exit_code'] != 0:
            raise AbortRun(f'{name} failed with exit code {stage["exit_code"]}')
        report_path = output / (name + '.json')
        data = json.loads(report_path.read_text())
        if data['status'] not in ('complete', 'verified'):
            raise AbortRun(f'{name} report did not complete')
        result['reports'][name] = {'path': str(report_path),
                                   'sha256': sha256(report_path, final_deadline)}
        return data

    signal.signal(signal.SIGTERM, alarm)
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, ALARM_SECONDS)
    error = None
    try:
        checkpoint()
        result['runtime'] = {'python': sys.version, 'executable': sys.executable,
                             'numpy': importlib.metadata.version('numpy'),
                             'plyfile': importlib.metadata.version('plyfile'),
                             'gpu_used': False}
        for name in ('CONTRACT.md', 'ASSUMPTIONS.md', 'encode.py', 'decode.py',
                     'verify.py', 'run_control.py', 'run-control.sh', 'prepare_test.py'):
            result['code_sha256'][name] = sha256(HERE / name, final_deadline)
        result['source']['bytes_before'] = SOURCE.stat().st_size
        result['source']['sha256_before'] = sha256(SOURCE, work_deadline)
        if result['source']['sha256_before'] != SOURCE_SHA:
            raise AbortRun('teacher hash before differs from frozen hash')
        checkpoint()
        codec, decoded = output / 'control.dsh8', output / 'decoded.ply'
        encode = run_stage('encode', ['--source', str(SOURCE), '--output', str(codec),
                           '--source-sha256', SOURCE_SHA, '--expected-points', str(POINTS),
                           '--report', str(output / 'encode.json')])
        decode = run_stage('decode', ['--codec', str(codec), '--output', str(decoded),
                           '--report', str(output / 'decode.json')])
        verify = run_stage('verify', ['--source', str(SOURCE), '--codec', str(codec),
                           '--decoded', str(decoded), '--report', str(output / 'verify.json')])
        if verify['status'] != 'verified' or verify['points'] != POINTS:
            raise AbortRun('raw verification did not pass')
        if not (encode['codec_sha256'] == decode['codec_sha256'] == verify['codec_sha256']
                and decode['decoded_sha256'] == verify['decoded_sha256']
                and decode['source_sha256'] == verify['source_sha256'] == SOURCE_SHA):
            raise AbortRun('artifact hashes disagree across stages')
        result['codec'].update(bytes=codec.stat().st_size, sha256=verify['codec_sha256'],
                               header_sha256=verify['header_sha256'],
                               payload_sha256=verify['payload_sha256'])
        result['decoded_ply'].update(bytes=decoded.stat().st_size, sha256=verify['decoded_sha256'])
        result['fields'] = verify['fields']
        result['checks'] = {'retained_exact': verify['retained_exact'],
                            'q_formula_exact': verify['q_formula_exact'],
                            'decoded_formula_exact': verify['decoded_formula_exact'],
                            'error_bound_violations': verify['error_bound']['violations']}
        result['storage'] = {'actual_codec_bytes': codec.stat().st_size,
                              'source_bytes': result['source']['bytes_before'],
                              'source_over_codec': result['source']['bytes_before'] / codec.stat().st_size,
                              'codec_over_source': codec.stat().st_size / result['source']['bytes_before'],
                              'supplied_half_point_ply_bytes': 254297012,
                              'codec_over_supplied_half_point_ply': codec.stat().st_size / 254297012,
                              'predicted_payload_bytes': POINTS * 117, 'header_bytes': 128,
                              'scope': 'actual single-file storage only; decoded points/VRAM/render work unchanged'}
    except BaseException as failure:
        error = failure
        result.update(status='error', error=f'{type(failure).__name__}: {failure}')
        supervisor.update(status='error', error=result['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # A running stage never survives the coordinator's handled failure.
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=0.5)
            if current_stage is not None:
                current_stage['exit_code'] = child.returncode
                current_stage['elapsed_seconds'] = time.monotonic() - current_stage.pop('start_monotonic', started)
        try:
            result['source']['bytes_after'] = SOURCE.stat().st_size
            result['source']['sha256_after'] = sha256(SOURCE, final_deadline)
            result['source']['unchanged'] = (
                result['source'].get('sha256_before') == result['source']['sha256_after'] == SOURCE_SHA
                and result['source'].get('bytes_before') == result['source']['bytes_after'])
            if not result['source']['unchanged'] and error is None:
                error = AbortRun('teacher changed or hash missing')
                result.update(status='error', error=str(error))
                supervisor.update(status='error', error=str(error))
        except BaseException as failure:
            result['source']['after_error'] = f'{type(failure).__name__}: {failure}'
            if error is None:
                error = failure
                result.update(status='error', error=result['source']['after_error'])
                supervisor.update(status='error', error=result['error'])
        # Record all available artifacts, including failed-stage partial files.
        # An exhausted budget is explicit; it never triggers another attempt.
        result['available_files'] = []
        for path in sorted(output.iterdir()):
            if path.name in ('RESULT.json', 'SUPERVISOR.json') or not path.is_file():
                continue
            entry = {'path': str(path), 'bytes': path.stat().st_size,
                     'partial': path.name.endswith('.partial')}
            try:
                entry['sha256'] = sha256(path, final_deadline)
                for artifact in ('codec', 'decoded_ply'):
                    expected = result[artifact]
                    if (str(path) == expected['path'] and 'sha256' in expected
                            and entry['sha256'] != expected['sha256']):
                        raise AbortRun(f'{artifact} changed after verification')
            except BaseException as failure:
                entry['hash_error'] = f'{type(failure).__name__}: {failure}'
                if error is None:
                    error = failure
                    result.update(status='error', error=entry['hash_error'])
                    supervisor.update(status='error', error=result['error'])
            result['available_files'].append(entry)
        if error is None:
            result['status'] = supervisor['status'] = 'verified'
        result['finished_utc'] = supervisor['finished_utc'] = utc()
        result['elapsed_seconds'] = supervisor['elapsed_seconds'] = time.monotonic() - started
        checkpoint()
    print(json.dumps({'status': result['status'], 'result': str(output / 'RESULT.json')}), flush=True)
    return 0 if error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
