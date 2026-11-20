#!/usr/bin/env python3
"""Tiny synthetic format/arithmetic witness only; never reads scientific data."""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    started = time.monotonic()
    assert not (HERE / 'run-v1').exists(), 'scientific run-v1 must remain absent'
    for name in ('encode.py', 'decode.py', 'verify.py', 'run_control.py', 'prepare_test.py'):
        ast.parse((HERE / name).read_text(), filename=name)
    fields = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
              + [f'f_rest_{i}' for i in range(45)] + ['opacity']
              + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    count = 8
    values = (np.arange(count * 62, dtype=np.float32).reshape(count, 62) / 1024).astype('<f4')
    rest = values[:, 9:54]
    rest[:] = 0
    values[0, [0, 54, 58]] = np.float32(-0.0)
    rest[1] = np.float32(-0.0)
    ties = [-127, -126.5, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 126.5, 127]
    rest[2, :len(ties)] = ties
    tiny = np.nextafter(np.float32(0), np.float32(1))
    rest[3, :2] = [tiny, -tiny]
    maximum = np.finfo(np.float32).max
    rest[4, :3] = [maximum, -maximum, np.float32(maximum / 2)]
    rest[5] = np.random.default_rng(20261023).normal(size=45).astype(np.float32)
    normal = np.finfo(np.float32).tiny
    rest[6, :4] = [normal, -normal, tiny, -tiny]
    rest[7] = np.linspace(-1, 1, 45, dtype=np.float32)
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex 8\n'
              + ''.join(f'property float {name}\n' for name in fields) + 'end_header\n').encode('ascii')
    source_bytes = header + values.tobytes()
    cases = []
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='direct-sh8-tiny-') as temporary:
        folder = Path(temporary)
        source, codec, decoded = folder / 'tiny.ply', folder / 'tiny.dsh8', folder / 'decoded.ply'
        source.write_bytes(source_bytes)

        def run(program, args, should_pass=True, contains=None):
            completed = subprocess.run([sys.executable, '-B', str(HERE / program)] + list(map(str, args)),
                                       capture_output=True, text=True, env=environment, timeout=15)
            assert (completed.returncode == 0) == should_pass, completed.stderr
            if contains:
                assert contains in completed.stderr, completed.stderr
            cases.append({'program': program, 'expected_success': should_pass,
                          'exit_code': completed.returncode, 'error_contains': contains,
                          'stdout_sha256': sha(completed.stdout.encode()),
                          'stderr_sha256': sha(completed.stderr.encode())})

        run('encode.py', ['--source', source, '--output', codec, '--source-sha256', sha(source_bytes),
                          '--expected-points', count, '--report', folder / 'encode.json'])
        assert codec.stat().st_size == 128 + count * 117 == 1064
        # Decoder subprocess receives no teacher, and even its previous path is absent.
        source.unlink()
        run('decode.py', ['--codec', codec, '--output', decoded, '--report', folder / 'decode.json'])
        source.write_bytes(source_bytes)
        run('verify.py', ['--source', source, '--codec', codec, '--decoded', decoded,
                          '--report', folder / 'verify.json'])
        verification = json.loads((folder / 'verify.json').read_text())
        assert verification['status'] == 'verified' and verification['error_bound']['violations'] == 0
        assert verification['retained_exact'] and verification['decoded_formula_exact']
        payload = codec.read_bytes()
        q = np.frombuffer(payload, dtype=np.dtype([('keep', '<f4', (17,)), ('amax', '<f4'), ('q', 'i1', (45,))]),
                          offset=128, count=count)['q']
        assert q[2, :len(ties)].tolist() == [-127, -126, -2, -2, 0, 0, 0, 2, 2, 126, 127]
        assert verification['zero_amax_points'] == 2

        def corrupt(name, mutate, expected_error, refresh_payload_hash=False):
            damaged = bytearray(payload)
            mutate(damaged)
            if refresh_payload_hash:
                damaged[64:96] = hashlib.sha256(damaged[128:]).digest()
            bad = folder / (name + '.dsh8')
            bad.write_bytes(damaged)
            run('decode.py', ['--codec', bad, '--output', folder / (name + '.ply'),
                              '--report', folder / (name + '.json')], False, expected_error)

        corrupt('payload-hash', lambda b: b.__setitem__(128, b[128] ^ 1), 'payload SHA256 mismatch')
        corrupt('reserved', lambda b: b.__setitem__(96, 1), 'reserved bytes')
        corrupt('q-minus128', lambda b: b.__setitem__(128 + 72, 128), 'invalid amax/q', True)
        corrupt('zero-scale-q', lambda b: b.__setitem__(128 + 72, 1), 'noncanonical amax/q', True)
        corrupt('negative-zero-amax', lambda b: b.__setitem__(slice(128 + 68, 128 + 72), struct.pack('<f', -0.0)),
                'invalid amax/q', True)
        corrupt('trailing', lambda b: b.extend(b'x'), 'codec length mismatch')
        nonfinite = bytearray(source_bytes)
        nonfinite[len(header):len(header) + 4] = struct.pack('<f', float('nan'))
        invalid_source = folder / 'nonfinite.ply'
        invalid_source.write_bytes(nonfinite)
        run('encode.py', ['--source', invalid_source, '--output', folder / 'nonfinite.dsh8',
                          '--source-sha256', sha(nonfinite), '--expected-points', count,
                          '--report', folder / 'nonfinite.json'], False, 'nonfinite source value')
        changed = bytearray(decoded.read_bytes())
        changed[len(header)] ^= 1
        changed_ply = folder / 'changed-retained.ply'
        changed_ply.write_bytes(changed)
        run('verify.py', ['--source', source, '--codec', codec, '--decoded', changed_ply,
                          '--report', folder / 'changed.json'], False, 'retained field bits differ')
        output = {'status': 'passed', 'scientific_execution': False, 'synthetic_points': count,
                  'source_sha256': sha(source_bytes), 'codec_sha256': sha(payload),
                  'codec_actual_bytes': codec.stat().st_size, 'cases': cases,
                  'teacher_path_absent_during_decode': True, 'manual_ties_to_even_check': True,
                  'contains_min_subnormal_and_max_finite': True,
                  'raw_verification': verification,
                  'elapsed_seconds': time.monotonic() - started,
                  'python': sys.version, 'numpy': np.__version__}
    assert not (HERE / 'run-v1').exists()
    output['run_v1_absent'] = True
    output['code_sha256'] = {name: sha((HERE / name).read_bytes()) for name in
                             ('CONTRACT.md', 'ASSUMPTIONS.md', 'encode.py', 'decode.py',
                              'verify.py', 'run_control.py', 'run-control.sh', 'prepare_test.py')}
    (HERE / 'PREPARATION_CHECKS.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: output[key] for key in ('status', 'synthetic_points', 'codec_actual_bytes',
                                                  'scientific_execution', 'run_v1_absent', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
