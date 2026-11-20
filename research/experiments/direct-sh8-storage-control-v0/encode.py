#!/usr/bin/env python3
"""CPU encoder for the pre-shared Direct SH8 v1 format. No model imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

FIELDS = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
          + [f'f_rest_{i}' for i in range(45)] + ['opacity']
          + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
KEEP = [name for name in FIELDS if not name.startswith('f_rest_')]
PLY_DTYPE = np.dtype([(name, '<f4') for name in FIELDS])
RECORD_DTYPE = np.dtype([('keep', '<f4', (17,)), ('amax', '<f4'), ('q', 'i1', (45,))])
HEADER = struct.Struct('<8sHHQHHHHI32s32s32s')
CHUNK = 65536


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_source(path, expected_points):
    properties, count, fmt = [], None, None
    with path.open('rb') as stream:
        if stream.readline() != b'ply\n':
            raise ValueError('expected LF PLY magic')
        while True:
            line = stream.readline(65537)
            if not line or len(line) > 65536 or stream.tell() > 65536:
                raise ValueError('invalid or oversized PLY header')
            words = line.decode('ascii').strip().split()
            if words == ['end_header']:
                offset = stream.tell()
                break
            if not words or words[0] in ('comment', 'obj_info'):
                continue
            if words == ['format', 'binary_little_endian', '1.0'] and fmt is None:
                fmt = True
            elif len(words) == 3 and words[:2] == ['element', 'vertex'] and count is None:
                count = int(words[2])
            elif (len(words) == 3 and words[0] == 'property'
                  and words[1] in ('float', 'float32') and count is not None):
                properties.append(words[2])
            else:
                raise ValueError(f'unsupported PLY header line: {words!r}')
    if not fmt or properties != FIELDS or count != expected_points or count <= 0:
        raise ValueError('source format, schema or point count mismatch')
    if path.stat().st_size != offset + count * 248:
        raise ValueError('source byte length or trailing bytes mismatch')
    return np.memmap(path, mode='r', dtype=PLY_DTYPE, offset=offset, shape=(count,))


def encode(args, report):
    source, output = Path(args.source), Path(args.output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('output or partial output already exists')
    source_sha = sha256(source)
    report.update(source_sha256=source_sha, source_bytes=source.stat().st_size)
    if source_sha != args.source_sha256:
        raise ValueError('source SHA256 differs from frozen expected SHA256')
    vertices = read_source(source, args.expected_points)
    payload_hash = hashlib.sha256()
    zero_amax = finite_fields = zero_fields = 0
    amax_min, amax_max = float('inf'), 0.0
    keep_indices = [FIELDS.index(name) for name in KEEP]
    assert RECORD_DTYPE.itemsize == 117 and HEADER.size == 128
    with partial.open('xb') as stream:
        stream.write(bytes(128))
        for start in range(0, len(vertices), CHUNK):
            raw = vertices[start:start + CHUNK].view('<f4').reshape(-1, 62)
            finite_fields += int(np.count_nonzero(np.isfinite(raw)))
            zero_fields += int(np.count_nonzero(raw == 0))
            if not np.isfinite(raw).all():
                raise ValueError(f'nonfinite source value in chunk starting {start}')
            rest = raw[:, 9:54]
            amax = np.max(np.abs(rest), axis=1)
            nonzero = amax != 0
            values64 = rest.astype(np.float64)
            ratio = np.zeros_like(values64)
            np.divide(values64 * 127.0, amax.astype(np.float64)[:, None],
                      out=ratio, where=nonzero[:, None])
            rounded = np.rint(ratio)
            if np.any(rounded < -127) or np.any(rounded > 127):
                raise ValueError('quantizer out of range; clipping is forbidden')
            records = np.empty(len(raw), dtype=RECORD_DTYPE)
            records['keep'].view('<u4')[:] = raw.view('<u4')[:, keep_indices]
            records['amax'] = amax
            records['q'] = rounded.astype(np.int8)
            data = records.tobytes()
            stream.write(data)
            payload_hash.update(data)
            zero_amax += int(np.count_nonzero(~nonzero))
            amax_min = min(amax_min, float(amax.min()))
            amax_max = max(amax_max, float(amax.max()))
        payload_sha = payload_hash.digest()
        stream.seek(0)
        stream.write(HEADER.pack(b'DSH8CTL\0', 1, 128, len(vertices), 3, 62, 17,
                                 45, 117, bytes.fromhex(source_sha), payload_sha, bytes(32)))
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(partial, output)
    actual_bytes = output.stat().st_size
    if actual_bytes != 128 + 117 * len(vertices):
        raise ValueError('actual codec stat size mismatch')
    report.update(status='complete', points=len(vertices), finite_fields=finite_fields,
                  zero_fields=zero_fields, zero_amax_points=zero_amax,
                  nonzero_amax_points=len(vertices) - zero_amax,
                  amax_min=amax_min, amax_max=amax_max, codec_bytes=actual_bytes,
                  codec_sha256=sha256(output), payload_sha256=payload_sha.hex(),
                  codec=str(output.resolve()), partial=str(partial.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--expected-points', required=True, type=int)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = {'status': 'partial', 'program': 'encode.py', 'numpy': np.__version__}
    started = time.monotonic()
    try:
        encode(args, report)
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
