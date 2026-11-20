"""External shared 300-second budget for both frozen evaluation invocations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/d/Documents/my 3dgs')
STAGE = ROOT / 'research/experiments/nanogs-frozen-baseline-v0'
SCRIPT = ROOT / 'scripts/experiments/evaluate_nanogs_frozen_baseline.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('identity-only', 'complete'), required=True)
    args = parser.parse_args()
    mode = args.mode
    output = STAGE / ('identity-only' if mode == 'identity-only' else 'evaluation-v1')
    log = STAGE / (mode + '.log')
    metadata = STAGE / ('WATCHDOG_' + mode + '.json')
    if output.exists() or log.exists() or metadata.exists():
        raise RuntimeError('Fresh one-shot evaluation paths required')
    prior_seconds = 0.
    if mode == 'complete':
        previous = json.loads((STAGE / 'WATCHDOG_identity-only.json').read_text())
        if previous['child_exit_code'] != 0 or previous['timeout']:
            raise RuntimeError('Identity invocation did not finish successfully')
        if digest(SCRIPT) != previous['script_sha256_before']:
            raise RuntimeError('Evaluator changed after identity invocation')
        prior_seconds = previous['external_wall_seconds']
    remaining = 300. - prior_seconds
    if remaining <= 0:
        raise RuntimeError('Shared 300-second evaluation budget exhausted')
    command = [sys.executable, '-B', '-u', str(SCRIPT), '--mode', mode,
               '--output-dir', str(output)]
    info = dict(command=command, mode=mode, prior_external_seconds=prior_seconds,
                remaining_external_seconds=remaining, timeout=False,
                script_sha256_before=digest(SCRIPT), launcher_sha256=digest(Path(__file__)))
    started = time.monotonic()
    try:
        with log.open('x', encoding='utf-8') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=remaining, check=False)
        info['child_exit_code'] = completed.returncode
    except subprocess.TimeoutExpired:
        info.update(timeout=True, child_exit_code=None)
    finally:
        info['external_wall_seconds'] = time.monotonic() - started
        info['combined_external_wall_seconds'] = prior_seconds + info['external_wall_seconds']
        info['script_sha256_after'] = digest(SCRIPT)
        metadata.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    result = json.loads((output / 'RESULTS.json').read_text()) if (output / 'RESULTS.json').exists() else {}
    print(json.dumps(dict(watchdog=info, result={key: result.get(key) for key in (
        'decision', 'integrity_complete', 'forward_calls', 'stage_wall_seconds',
        'identity_max_abs_rgb_all33', 'error')})), flush=True)
    if info['timeout']:
        return 124
    if info['combined_external_wall_seconds'] > 300:
        return 2
    return info['child_exit_code']


if __name__ == '__main__':
    sys.exit(main())
