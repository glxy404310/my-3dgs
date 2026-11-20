#!/usr/bin/env python3
"""Independent raw checker: teacher access starts after independent decode."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

FIELDS = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
          + [f'f_rest_{i}' for i in range(45)] + ['opacity']
          + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
KEEP = [name for name in FIELDS if not name.startswith('f_rest_')]


def sha256(path, offset=0):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(offset)
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def read_ply(path, count):
    names, found_count, found_format = [], None, False
    with path.open('rb') as stream:
        if stream.readline() != b'ply\n':
            raise ValueError('PLY magic mismatch')
        while True:
            line = stream.readline(65537)
            if not line or len(line) > 65536 or stream.tell() > 65536:
                raise ValueError('invalid PLY header')
            tokens = line.decode('ascii').strip().split()
            if tokens == ['end_header']:
                data_offset = stream.tell()
                break
            if not tokens or tokens[0] in ('comment', 'obj_info'):
                continue
            if tokens == ['format', 'binary_little_endian', '1.0'] and not found_format:
                found_format = True
            elif len(tokens) == 3 and tokens[:2] == ['element', 'vertex'] and found_count is None:
                found_count = int(tokens[2])
            elif (len(tokens) == 3 and tokens[0] == 'property'
                  and tokens[1] in ('float', 'float32') and found_count is not None):
                names.append(tokens[2])
            else:
                raise ValueError(f'unsupported PLY header line: {tokens!r}')
    if not found_format or names != FIELDS or found_count != count:
        raise ValueError('PLY schema or point-count mismatch')
    if path.stat().st_size != data_offset + count * 62 * 4:
        raise ValueError('PLY length or trailing bytes mismatch')
    return np.memmap(path, mode='r', offset=data_offset, shape=(count,),
                     dtype=[(name, '<f4') for name in FIELDS])


def raw_check(args, report):
    source, codec, decoded = map(Path, (args.source, args.codec, args.decoded))
    with codec.open('rb') as stream:
        header = stream.read(128)
    magic, version, header_size, count, degree, total, kept, nrest, record_size, source_digest, payload_digest, reserved = struct.unpack('<8sHHQHHHHI32s32s32s', header)
    if ((magic, version, header_size, degree, total, kept, nrest, record_size)
            != (b'DSH8CTL\0', 1, 128, 3, 62, 17, 45, 117)
            or count <= 0 or reserved != bytes(32)):
        raise ValueError('codec header mismatch')
    if codec.stat().st_size != 128 + count * 117:
        raise ValueError('codec size mismatch')
    if sha256(source) != source_digest.hex() or sha256(codec, 128) != payload_digest.hex():
        raise ValueError('source or payload digest mismatch')
    teacher = read_ply(source, count)
    reconstructed = read_ply(decoded, count)
    records = np.memmap(codec, mode='r', offset=128, shape=(count,),
                        dtype=[('keep', '<f4', (17,)), ('amax', '<f4'), ('q', 'i1', (45,))])
    stats = {name: {'source_finite': 0, 'source_zero': 0, 'source_negative_zero': 0,
                    'decoded_finite': 0, 'decoded_zero': 0, 'decoded_negative_zero': 0,
                    'source_min': float('inf'), 'source_max': -float('inf'),
                    'decoded_min': float('inf'), 'decoded_max': -float('inf'),
                    'sum_abs_error': 0.0, 'sum_squared_error': 0.0, 'max_abs_error': 0.0,
                    'bit_mismatches': 0} for name in FIELDS}
    bound = {'violations': 0, 'max_abs_error': 0.0, 'max_half_step': 0.0,
             'max_float32_rounding': 0.0, 'max_float64_allowance': 0.0,
             'max_error_minus_halfstep_and_float32_rounding': -float('inf')}
    zero_amax = nonzero_amax = q_zero = 0
    q_min, q_max = 127, -127
    for start in range(0, count, 65536):
        original = teacher[start:start + 65536]
        decoded_block = reconstructed[start:start + 65536]
        codes = records[start:start + 65536]
        original_rest = original.view('<f4').reshape(-1, 62)[:, 9:54]
        amax_expected = np.max(np.abs(original_rest), axis=1)
        amax, q = codes['amax'], codes['q']
        nonzero = amax != 0
        if (not np.isfinite(amax).all() or np.any(np.signbit(amax))
                or np.any(q < -127) or np.any(q > 127)
                or not np.array_equal(amax.view('<u4'), amax_expected.view('<u4'))
                or np.any(q[~nonzero] != 0)
                or np.any(np.max(np.abs(q[nonzero].astype(np.int16)), axis=1) != 127)):
            raise ValueError(f'amax or q invariant failed at chunk {start}')
        original64 = original_rest.astype(np.float64)
        scale64 = amax.astype(np.float64)[:, None]
        ratio = np.zeros_like(original64)
        np.divide(original64 * 127.0, scale64, out=ratio, where=nonzero[:, None])
        if not np.array_equal(q.astype(np.float64), np.rint(ratio)):
            raise ValueError(f'q differs from frozen formula at chunk {start}')
        ideal64 = (q.astype(np.float64) * scale64) / 127.0
        expected32 = ideal64.astype('<f4')
        observed32 = decoded_block.view('<f4').reshape(-1, 62)[:, 9:54]
        if not np.array_equal(expected32.view('<u4'), observed32.view('<u4')):
            raise ValueError(f'decoded SH differs from frozen formula at chunk {start}')
        error64 = np.abs(observed32.astype(np.float64) - original64)
        cast_rounding = np.abs(expected32.astype(np.float64) - ideal64)
        half_step = scale64 / 254.0
        # Verification arithmetic only: no epsilon/floor enters the quantizer.
        arithmetic_allowance = np.where(scale64 == 0, 0.0, 4.0 * np.spacing(scale64))
        upper = np.nextafter(half_step + cast_rounding + arithmetic_allowance, np.inf)
        violations = int(np.count_nonzero(error64 > upper))
        bound['violations'] += violations
        bound['max_abs_error'] = max(bound['max_abs_error'], float(error64.max()))
        bound['max_half_step'] = max(bound['max_half_step'], float(half_step.max()))
        bound['max_float32_rounding'] = max(bound['max_float32_rounding'], float(cast_rounding.max()))
        bound['max_float64_allowance'] = max(bound['max_float64_allowance'], float(arithmetic_allowance.max()))
        bound['max_error_minus_halfstep_and_float32_rounding'] = max(
            bound['max_error_minus_halfstep_and_float32_rounding'],
            float(np.max(error64 - half_step - cast_rounding)))
        zero_amax += int(np.count_nonzero(~nonzero))
        nonzero_amax += int(np.count_nonzero(nonzero))
        q_zero += int(np.count_nonzero(q == 0))
        q_min, q_max = min(q_min, int(q.min())), max(q_max, int(q.max()))
        for index, name in enumerate(KEEP):
            original_bits = original[name].view('<u4')
            if (not np.array_equal(codes['keep'][:, index].view('<u4'), original_bits)
                    or not np.array_equal(decoded_block[name].view('<u4'), original_bits)):
                raise ValueError(f'retained field bits differ: {name}, chunk {start}')
        for name in FIELDS:
            x, y = original[name], decoded_block[name]
            s = stats[name]
            for prefix, values in (('source', x), ('decoded', y)):
                finite = int(np.count_nonzero(np.isfinite(values)))
                s[prefix + '_finite'] += finite
                s[prefix + '_zero'] += int(np.count_nonzero(values == 0))
                s[prefix + '_negative_zero'] += int(np.count_nonzero((values == 0) & np.signbit(values)))
                if finite != len(values):
                    raise ValueError(f'nonfinite {prefix} field: {name}')
                s[prefix + '_min'] = min(s[prefix + '_min'], float(values.min()))
                s[prefix + '_max'] = max(s[prefix + '_max'], float(values.max()))
            err = y.astype(np.float64) - x.astype(np.float64)
            s['sum_abs_error'] += float(np.sum(np.abs(err), dtype=np.float64))
            s['sum_squared_error'] += float(np.sum(err * err, dtype=np.float64))
            s['max_abs_error'] = max(s['max_abs_error'], float(np.max(np.abs(err))))
            s['bit_mismatches'] += int(np.count_nonzero(x.view('<u4') != y.view('<u4')))
        report['checked_points'] = start + len(original)
        report['per_field_partial_sums'] = stats
    for s in stats.values():
        s['mean_abs_error'] = s.pop('sum_abs_error') / count
        s['rmse'] = (s.pop('sum_squared_error') / count) ** 0.5
    report.pop('per_field_partial_sums', None)
    report.update(points=count, sh_degree=degree, fields=FIELDS, per_field=stats,
                  error_bound=bound, zero_amax_points=zero_amax,
                  nonzero_amax_points=nonzero_amax, zero_q_values=q_zero,
                  q_min=q_min, q_max=q_max, retained_exact=True, decoded_formula_exact=True,
                  q_formula_exact=True, codec_sha256=sha256(codec), codec_bytes=codec.stat().st_size,
                  payload_sha256=payload_digest.hex(), header_sha256=hashlib.sha256(header).hexdigest(),
                  source_sha256=source_digest.hex(), source_bytes=source.stat().st_size,
                  decoded_sha256=sha256(decoded), decoded_bytes=decoded.stat().st_size)
    if bound['violations']:
        raise ValueError('quantization error amplitude bound violated')
    report['status'] = 'verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--codec', required=True)
    parser.add_argument('--decoded', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = {'status': 'partial', 'program': 'verify.py', 'numpy': np.__version__,
              'float64_allowance': '4*spacing(float64(amax)); zero for amax=0',
              'bound': 'nextafter(amax/254 + actual_float32_cast_rounding + float64_allowance, +inf)'}
    started = time.monotonic()
    try:
        raw_check(args, report)
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
