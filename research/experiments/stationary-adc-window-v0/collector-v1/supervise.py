#!/usr/bin/env python3
"""Bounded Linux process supervisor. Unapproved contracts never start a worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SCHEMA = 'stationary-adc-window-v0/collector-v1'
EXPECTED = dict(forward=100, backward=100, scene_optimizer=99,
                exposure_optimizer=99, adc=0, opacity_reset=0)
MAX_LIMITS = dict(external_seconds=300, rss_bytes=8 * 1024**3,
                  torch_allocated_bytes=4 * 1024**3,
                  torch_reserved_bytes=4 * 1024**3, raw_bytes=4 * 1024**3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate(contract):
    # These checks precede platform checks, output creation and process spawning.
    require(contract.get('schema') == SCHEMA, 'Wrong collector contract schema')
    require(contract.get('training_authorized') is True,
            'DENIED: explicit human authorization for new updates is absent')
    require(contract.get('root_review_complete') is True,
            'DENIED: full root code review is incomplete')
    evidence = contract.get('approval_evidence', {})
    require(isinstance(evidence, dict) and evidence.get('user_message') and
            evidence.get('recorded_at'), 'Explicit approval evidence is required')
    require(contract.get('expected_counts') == EXPECTED, 'Unexpected execution counts')
    limits = contract.get('limits', {})
    require(set(limits) == set(MAX_LIMITS), 'All finite resource limits are required')
    for key, maximum in MAX_LIMITS.items():
        require(type(limits[key]) in (int, float) and 0 < limits[key] <= maximum,
                'Invalid resource bound: ' + key)
    require(sys.platform == 'linux', 'Existing WSL Linux runtime is required')
    python = Path(contract['runtime_python'])
    require(python.is_absolute() and python.resolve() == Path(sys.executable).resolve(),
            'Unexpected Python executable')
    require(Path(sys.prefix).resolve() == python.parent.parent.resolve(),
            'Unexpected Python environment')
    code_pins = contract.get('reviewed_code_sha256', {})
    require(set(code_pins) == {'collect.py', 'supervise.py'}, 'Both code hashes required')
    for name, digest in code_pins.items():
        require(sha256(HERE / name) == digest, 'Reviewed code changed: ' + name)
    require(isinstance(contract.get('pins'), list) and contract['pins'], 'Input pins absent')
    for pin in contract['pins']:
        p = Path(pin['path'])
        require(p.is_absolute() and p.is_file(), 'Pinned input is missing')
        require('bytes' not in pin or p.stat().st_size == pin['bytes'], 'Input size changed')
        require(sha256(p) == pin['sha256'], 'Pinned input hash changed: ' + str(p))
    return limits


def process_group_rows(group):
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():
            continue
        try:
            stat = (p / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) != group:
                continue
            rss = 0
            for line in (p / 'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1]) * 1024
                    break
            rows.append(dict(pid=int(p.name), state=fields[0], rss_bytes=rss))
        except (OSError, ValueError, IndexError):
            continue
    return rows


def group_alive(group):
    return any(row['state'] != 'Z' for row in process_group_rows(group))


def stop_group(proc):
    require(proc.pid > 1 and proc.pid != os.getpid(), 'Unsafe process-group target')
    if group_alive(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 1
        while group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if group_alive(proc.pid):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def output_bytes(folder):
    total = 0
    if folder.exists():
        for p in folder.rglob('*'):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except FileNotFoundError:
                pass
    return total


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temp.replace(path)


def run(args):
    started = time.monotonic()
    contract_path = args.contract.resolve()
    cfg = json.loads(contract_path.read_text(encoding='utf8'))
    limits = validate(cfg)
    out = args.output.resolve()
    require(out.parent == HERE and out.name.startswith('run-') and
            out.name.replace('-', '').replace('_', '').isalnum(), 'New local run-* output required')
    require(not out.exists(), 'Output must not already exist')
    log = HERE / ('host-' + out.name + '.log')
    require(not log.exists(), 'Host log must not already exist')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', STATIONARY_ADC_SUPERVISED='1')
    command = [cfg['runtime_python'], str(HERE / 'collect.py'),
               '--contract', str(contract_path), '--output', str(out)]
    require(time.monotonic() - started < limits['external_seconds'],
            'Preflight exhausted the external wall budget; worker was not started')
    peak_rss = peak_processes = peak_output = 0
    stop_reason = None
    proc = None
    record = dict(schema=SCHEMA + '/external', command=command,
                  contract_sha256=sha256(contract_path), worker_reaped=False,
                  output_budget_scope='worker output before EXTERNAL_ROOT.json; host log excluded',
                  full_process_GPU_memory='UNMEASURED; worker allocator limits are separate',
                  GPU_trace_independently_certified=False)
    try:
        with log.open('x', encoding='utf8') as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                    env=env, cwd=HERE, start_new_session=True)
            while proc.poll() is None:
                rows = process_group_rows(proc.pid)
                peak_rss = max(peak_rss, sum(row['rss_bytes'] for row in rows))
                peak_processes = max(peak_processes, len(rows))
                peak_output = max(peak_output, output_bytes(out))
                elapsed = time.monotonic() - started
                if elapsed > limits['external_seconds']:
                    stop_reason = 'EXTERNAL_WALL_LIMIT'
                elif peak_rss > limits['rss_bytes']:
                    stop_reason = 'PROCESS_GROUP_RSS_LIMIT'
                elif peak_output > limits['raw_bytes']:
                    stop_reason = 'OUTPUT_BYTE_LIMIT'
                if stop_reason:
                    break
                time.sleep(0.1)
    except BaseException as exc:
        stop_reason = 'SUPERVISOR_EXCEPTION: ' + type(exc).__name__ + ': ' + str(exc)
    finally:
        if proc is not None:
            # Also reap descendants after an otherwise successful parent exit.
            orphaned = group_alive(proc.pid) and proc.poll() is not None
            if orphaned and stop_reason is None:
                stop_reason = 'WORKER_LEFT_LIVE_DESCENDANTS'
            stop_group(proc)
            record.update(worker_exit_code=proc.returncode,
                          worker_reaped=not group_alive(proc.pid))
        elapsed = time.monotonic() - started
        peak_output = max(peak_output, output_bytes(out))
        if elapsed > limits['external_seconds'] and stop_reason is None:
            stop_reason = 'EXTERNAL_WALL_LIMIT'
        if peak_output > limits['raw_bytes'] and stop_reason is None:
            stop_reason = 'OUTPUT_BYTE_LIMIT'
        if proc is not None and proc.returncode != 0 and stop_reason is None:
            stop_reason = 'WORKER_NONZERO_EXIT'
        record.update(external_wall_seconds=elapsed, peak_group_RSS_bytes=peak_rss,
                      peak_group_processes=peak_processes, output_bytes=peak_output,
                      stop_reason=stop_reason, host_log=str(log),
                      status='INVALID' if stop_reason else 'WORKER_EXIT_RECORDED_GATES_UNEVALUATED')
        if out.is_dir():
            dump(out / 'EXTERNAL_ROOT.json', record)
        else:
            dump(HERE / ('host-' + out.name + '.external.json'), record)
    print(json.dumps(record, allow_nan=False))
    return 0 if proc and proc.returncode == 0 and record['worker_reaped'] and not stop_reason else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps(dict(status='DENIED_BEFORE_WORKER', error=str(exc))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
