#!/usr/bin/env python3
"""Standalone Direct SH8 v1 decoder; the codec is its only input artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np


def digest_file(path, offset=0):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decode(args, report):
    codec, output = Path(args.codec), Path(args.output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('output or partial output already exists')
    # These constants define the pre-shared format; no encoder import is used.
    header_struct = struct.Struct('<8sHHQHHHHI32s32s32s')
    with codec.open('rb') as stream:
        header = stream.read(128)
    if len(header) != 128:
        raise ValueError('truncated header')
    magic, version, hsize, count, degree, total, kept, rest_count, rsize, source, payload, reserved = header_struct.unpack(header)
    if ((magic, version, hsize, degree, total, kept, rest_count, rsize)
            != (b'DSH8CTL\0', 1, 128, 3, 62, 17, 45, 117)):
        raise ValueError('unknown codec format')
    if count <= 0 or reserved != bytes(32) or source == bytes(32):
        raise ValueError('invalid count, source hash or reserved bytes')
    if codec.stat().st_size != 128 + count * 117:
        raise ValueError('codec length mismatch, including trailing bytes')
    payload_sha = digest_file(codec, 128)
    if payload_sha != payload.hex():
        raise ValueError('payload SHA256 mismatch')
    fields = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
              + [f'f_rest_{i}' for i in range(45)] + ['opacity']
              + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    keep = [name for name in fields if not name.startswith('f_rest_')]
    records = np.memmap(codec, mode='r', offset=128, shape=(count,),
                        dtype=[('keep', '<f4', (17,)), ('amax', '<f4'), ('q', 'i1', (45,))])
    ply_header = ('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(count) + '\n'
                  + ''.join(f'property float {name}\n' for name in fields) + 'end_header\n').encode('ascii')
    zero_amax = 0
    with partial.open('xb') as stream:
        stream.write(ply_header)
        for start in range(0, count, 65536):
            block = records[start:start + 65536]
            scale, q = block['amax'], block['q']
            nonzero = scale != 0
            if (not np.isfinite(scale).all() or np.any(scale < 0)
                    or np.any(np.signbit(scale)) or np.any(q == -128)):
                raise ValueError(f'invalid amax/q in chunk {start}')
            if (np.any(q[~nonzero] != 0)
                    or np.any(np.max(np.abs(q[nonzero].astype(np.int16)), axis=1) != 127)):
                raise ValueError(f'noncanonical amax/q in chunk {start}')
            if not np.isfinite(block['keep']).all():
                raise ValueError('nonfinite retained value')
            decoded = np.empty(len(block), dtype=[(name, '<f4') for name in fields])
            for index, name in enumerate(keep):
                decoded[name].view('<u4')[:] = block['keep'][:, index].view('<u4')
            reconstructed = ((q.astype(np.float64) * scale.astype(np.float64)[:, None])
                             / 127.0).astype('<f4')
            for index in range(45):
                decoded[f'f_rest_{index}'] = reconstructed[:, index]
            stream.write(decoded.tobytes())
            zero_amax += int(np.count_nonzero(~nonzero))
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(partial, output)
    if output.stat().st_size != len(ply_header) + count * 248:
        raise ValueError('decoded PLY stat size mismatch')
    report.update(status='complete', points=count, sh_degree=3, fields=fields,
                  codec=str(codec.resolve()), codec_bytes=codec.stat().st_size,
                  codec_sha256=digest_file(codec), header_sha256=hashlib.sha256(header).hexdigest(),
                  payload_sha256=payload_sha, source_sha256=source.hex(),
                  decoded_ply=str(output.resolve()), decoded_bytes=output.stat().st_size,
                  decoded_sha256=digest_file(output), zero_amax_points=zero_amax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = {'status': 'partial', 'program': 'decode.py', 'numpy': np.__version__}
    started = time.monotonic()
    try:
        decode(args, report)
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
