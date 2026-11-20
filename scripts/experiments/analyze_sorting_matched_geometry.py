"""Independent CPU audit of frozen matched-Q evidence; never rerun the producer.

No producer/codec imports, torch, CUDA, training, or edits to raw evidence.
The sole output is run/INDEPENDENT_ANALYSIS.json. All I/O counts against 90 s.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import threading
import time
import traceback

import numpy as np
from plyfile import PlyData


MODES = ('base_float32', 'local_tau025', 'global_tau025', 'local_tau0125',
         'uniform8', 'uniform12', 'uniform16', 'uniform20', 'uniform24', 'uniform32')
RANKS = (0, 31, 62, 93, 124, 155, 186, 218)
SAMPLES = (16, 48, 80, 112, 144, 176, 208, 240)
POINTS = 2050770
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
THRESHOLD = float(np.float32(1) / np.float32(255))
CAP = float(np.float32(.99))
EARLY = float(np.float32(.0001))
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def write_report(path, report):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class Audit:
    def __init__(self, report):
        self.report = report
        self.failures = report['failures']
        self.hash_seconds = 0.
        self.large_io_seconds = 0.
        self.hashed_bytes = 0

    def check(self, condition, label, observed=None):
        if not bool(condition):
            self.failures.append(dict(check=label, observed=observed))
        return bool(condition)

    def digest(self, path):
        began = time.perf_counter()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
                self.hashed_bytes += len(block)
        self.hash_seconds += time.perf_counter() - began
        return digest.hexdigest()

    def archive(self, path):
        began = time.perf_counter()
        with np.load(path, allow_pickle=False) as stream:
            value = {key: stream[key] for key in stream.files}
        self.large_io_seconds += time.perf_counter() - began
        return value


def independent_decode(path, original, expected, audit):
    """Parse all bytes without importing either implementation under audit."""
    began = time.perf_counter()
    raw = path.read_bytes()
    audit.large_io_seconds += time.perf_counter() - began
    magic, n, local, block_size, blocks = struct.unpack_from('<8sQIII', raw, 0)
    assert magic == b'SGSPOS01' and n == POINTS and local in (0, 1)
    assert block_size == (4096 if local else n)
    assert blocks == (n + block_size - 1) // block_size
    offset = 28
    permutation = np.frombuffer(raw, dtype='<u4', offset=offset, count=n).copy() if local else None
    if local:
        assert np.array_equal(np.sort(permutation), np.arange(n, dtype=np.uint32))
        offset += 4 * n
        # Independently reconstruct Morton21 and stable original-ID tie order.
        source = original.astype(np.float64)
        lo, hi = source.min(axis=0), source.max(axis=0)
        fraction = np.divide(source - lo, hi - lo, out=np.zeros_like(source), where=hi != lo)
        quant = np.floor(fraction * 2097151).astype(np.uint64)
        key = np.zeros(n, dtype=np.uint64)
        for bit in range(21):
            for axis in range(3):
                key += ((quant[:, axis] >> bit) & 1) << (3 * bit + axis)
        assert np.array_equal(permutation, np.argsort(key, kind='stable').astype(np.uint32))
        del source, fraction, quant, key
    ordered = np.empty((n, 3), dtype=np.float32)
    counts = {str(bit): 0 for bit in list(range(25)) + [32]}
    payload = escapes = padding = 0
    start = 0
    adaptive_bound_ok = True
    epsilon = expected['epsilon']
    for _ in range(blocks):
        desc = struct.unpack_from('<I6d3B', raw, offset)
        offset += 55
        count, low, high, widths = desc[0], desc[1:4], desc[4:7], desc[7:10]
        assert count == min(block_size, n - start)
        ids = permutation[start:start + count] if local else slice(start, start + count)
        source_block = original[ids].astype(np.float64)
        assert np.array_equal(source_block.min(axis=0), low) and np.array_equal(source_block.max(axis=0), high)
        for axis, width in enumerate(widths):
            assert width in tuple(range(25)) + (32,) and low[axis] <= high[axis]
            counts[str(width)] += count
            if epsilon is not None:
                magnitude = max(abs(low[axis]), abs(high[axis]))
                rounding = abs(float(np.spacing(np.float32(magnitude)))) / 2
                guard = 2 * abs(float(np.spacing(np.float64(magnitude))))
                bound_width = 0 if high[axis] == low[axis] else next(
                    (bit for bit in range(1, 25)
                     if (high[axis] - low[axis]) / (2 * (2**bit - 1)) + rounding + guard <= epsilon), 32)
                adaptive_bound_ok = adaptive_bound_ok and width == bound_width
            else:
                bit = expected['uniform_bits']
                assert width == (0 if low[axis] == high[axis] and bit != 32 else bit)
            if width == 32:
                length = count * 4
                values = np.frombuffer(raw, dtype='<f4', offset=offset, count=count)
                assert np.array_equal(values.view(np.uint32), original[ids, axis].view(np.uint32))
                escapes += length
            elif width == 0:
                assert low[axis] == high[axis]
                length = 0
                values = np.full(count, low[axis], dtype=np.float64)
            else:
                plane = (count + 7) // 8
                length = plane * width
                code = np.zeros(count, dtype=np.uint32)
                for bit in range(width):
                    data = np.frombuffer(raw, dtype=np.uint8, offset=offset + bit * plane, count=plane)
                    unpacked = np.unpackbits(data, bitorder='little')
                    assert not unpacked[count:].any(), 'Nonzero bit-plane padding'
                    code += unpacked[:count].astype(np.uint32) * (2**bit)
                true_code = np.rint((source_block[:, axis] - low[axis]) / (high[axis] - low[axis]) * (2**width - 1)).astype(np.uint32)
                assert np.array_equal(code, true_code), 'Payload does not encode source rounding'
                values = low[axis] + code.astype(np.float64) * ((high[axis] - low[axis]) / (2**width - 1))
                padding += width * (8 * plane - count)
            ordered[start:start + count, axis] = values
            payload += length
            offset += length
        start += count
    assert start == n and offset == len(raw) and np.isfinite(ordered).all()
    decoded = np.empty_like(ordered)
    if local:
        decoded[permutation] = ordered
    else:
        decoded[:] = ordered
    error = np.abs(decoded.astype(np.float64) - original.astype(np.float64)).max(axis=0)
    record = dict(points=n, local=bool(local), block_size=block_size, block_count=blocks,
                  bytes=len(raw), fixed_header_bytes=28, descriptor_bytes=55 * blocks,
                  header_bytes=28 + 55 * blocks, permutation_bytes=4 * n if local else 0,
                  payload_bytes=payload, escape_bytes=escapes, escape_coordinates=counts['32'],
                  bits_counts_coordinates=counts, bitplane_padding_bits=padding,
                  max_axis_error=error.tolist(), max_coordinate_error=float(error.max()),
                  epsilon_compliant=bool(epsilon is None or np.all(error <= epsilon)),
                  adaptive_minimum_width_verified=adaptive_bound_ok,
                  complete_file_decode_verified=True, padding_zero=True,
                  source_quantization_rounding_verified=True)
    assert len(raw) == record['header_bytes'] + record['permutation_bytes'] + payload
    assert record['epsilon_compliant'] and adaptive_bound_ok
    for field in ('points', 'local', 'block_size', 'block_count', 'bytes', 'fixed_header_bytes',
                  'descriptor_bytes', 'header_bytes', 'permutation_bytes', 'payload_bytes',
                  'escape_bytes', 'escape_coordinates', 'bits_counts_coordinates', 'bitplane_padding_bits'):
        if field in expected:
            audit.check(record[field] == expected[field], 'codec metadata/' + expected['mode'] + '/' + field)
    if 'max_axis_error' in expected:
        audit.check(np.allclose(error, expected['max_axis_error'], rtol=0, atol=1e-15), 'codec max coordinate error/' + expected['mode'])
    return decoded, record


def scalar_ray(cache, ids, xy, order):
    """Scalar Python double recurrence with math.exp and no producer functions."""
    mapped = np.searchsorted(cache['unique_ids'], ids)
    t, color, accepted = 1., [0., 0., 0.], 0
    for index in order:
        local = mapped[int(index)]
        dx = float(cache['means2D'][local, 0]) - int(xy[0])
        dy = float(cache['means2D'][local, 1]) - int(xy[1])
        a, b, c, opacity = map(float, cache['conic_opacity'][local])
        exponent = -.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        if exponent > 0:
            continue
        alpha = min(CAP, opacity * math.exp(exponent))
        if alpha < THRESHOLD:
            continue
        following = t * (1. - alpha)
        if following < EARLY:
            break
        for channel in range(3):
            color[channel] += (float(cache['rgb'][local, channel]) * alpha) * t
        t, accepted = following, accepted + 1
    return np.clip(np.array(color), 0., 1.), accepted


def matrix_projection(xyz, view, projection):
    def column(matrix, axis):
        value = xyz[:, 0] * np.float32(matrix[0, axis])
        value = value + xyz[:, 1] * np.float32(matrix[1, axis])
        value = value + xyz[:, 2] * np.float32(matrix[2, axis])
        return value + np.float32(matrix[3, axis])
    depth = column(view, 2)
    h = np.column_stack([column(projection, axis) for axis in (0, 1, 3)])
    ndc = h[:, :2] * (np.float32(1) / (h[:, 2] + np.float32(1e-7)))[:, None]
    pixels = ((ndc + np.float32(1)) * np.array([979, 546], dtype=np.float32) - np.float32(1)) * np.float32(.5)
    return pixels, depth, ndc


def displacement(cache, ref, original, decoded, tau):
    ids = np.union1d(ref['selected_candidate_ids'], cache['unique_ids'])
    before, old_z, ndc = matrix_projection(original[ids], ref['world_view_transform'], ref['full_proj_transform'])
    after, new_z, _ = matrix_projection(decoded[ids], ref['world_view_transform'], ref['full_proj_transform'])
    active_old = ref['active'][ids]
    before[active_old] = ref['means2D'][ids[active_old]]
    where = np.searchsorted(cache['q_active_ids'], ids)
    active_new = where < len(cache['q_active_ids'])
    active_new[active_new] &= cache['q_active_ids'][where[active_new]] == ids[active_new]
    after[active_new] = cache['q_active_means2D'][where[active_new]]
    diff = np.abs(after.astype(np.float64) - before.astype(np.float64))
    axis_max = diff.max(axis=0)
    invalid = (old_z <= .2) | ~np.isfinite(before).all(axis=1)
    inside = (old_z > .2) & (np.abs(ndc) <= 1).all(axis=1)
    maximum = float(axis_max.max())
    outside_values = diff[~inside]
    return dict(selected_support_union_points=len(ids),
                selected_support_invalid_original_projection_points=int(invalid.sum()),
                selected_support_max_axis_displacement_px=maximum,
                selected_support_axis_max_displacement_px=axis_max.tolist(),
                selected_support_original_near_crossing_count=int(((old_z > .2) != (new_z > .2)).sum()),
                selected_support_outside_original_frustum_points=int((~inside).sum()),
                selected_support_outside_max_axis_displacement_px=float(outside_values.max()) if outside_values.size else None,
                displacement_control_valid=bool(tau is None or (maximum <= tau + 2e-4 and not invalid.any())),
                all_selected_support_ids_retained=True)


def typed(value):
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return float(value)
    except ValueError:
        return value


def row_derived(row):
    get = lambda label: np.array([float(row[label + '_' + c]) for c in 'rgb'])
    c, d, base, gt, native = [get(label) for label in ('C', 'D', 'base', 'GT', 'native')]
    mse = lambda a, b: float(np.mean((a - b)**2))
    maximum = float(np.max(np.abs(c - d)))
    return dict(max_channel_difference=maximum, exceeds_1_over_255=maximum > 1/255,
                order_mse=mse(c, d), C_gt_mse=mse(c, gt), D_gt_mse=mse(d, gt),
                base_gt_mse=mse(base, gt), C_base_mse=mse(c, base), D_base_mse=mse(d, base),
                D_native_max_channel_error=float(np.max(np.abs(d - native))))


def summarize(rows, mode, view, byte_count, child=None):
    diff = np.array([r['max_channel_difference'] for r in rows])
    count = int((diff > 1/255).sum())
    mean = lambda field: float(np.mean([r[field] for r in rows]))
    views = int(count >= .01 * len(rows)) if child is None else sum(s['view_gate'] for s in child)
    worst = lambda field: mean(field) if child is None else float(np.mean(sorted([s[field] for s in child], reverse=True)[:2]))
    return dict(mode=mode, view=view, encoded_position_bytes=byte_count, pixels=len(rows),
                exceed_count=count, exceed_fraction=count/len(rows), view_gate=count >= .01*len(rows),
                qualifying_views=views, capacity_gate=bool(count >= .01*len(rows) and views >= 4) if view == 'ALL' else False,
                max_channel_difference=float(diff.max()), p99_channel_difference=float(np.quantile(diff, .99)),
                order_mse=mean('order_mse'), C_gt_mse=mean('C_gt_mse'), D_gt_mse=mean('D_gt_mse'),
                base_gt_mse=mean('base_gt_mse'), C_base_mse=mean('C_base_mse'), D_base_mse=mean('D_base_mse'),
                C_worst2_view_gt_mse=worst('C_gt_mse'), D_worst2_view_gt_mse=worst('D_gt_mse'),
                max_D_native_error=max(r['D_native_max_channel_error'] for r in rows),
                old_inactive_candidate_occurrences=sum(r['old_inactive_candidate_count'] for r in rows),
                accepted_C=sum(r['accepted_C'] for r in rows), accepted_D=sum(r['accepted_D'] for r in rows),
                candidate_order_positions_changed=sum(r['candidate_order_positions_changed'] for r in rows))


def compare(audit, actual, expected, label, tolerance=1e-14):
    for field, value in actual.items():
        other = expected.get(field)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            equal = other is not None and math.isclose(value, float(other), rel_tol=0, abs_tol=tolerance)
        else:
            equal = value == other
        audit.check(equal, label + '/' + field, dict(independent=value, saved=other) if not equal else None)


def audit_run(args, report, audit):
    run = args.run
    config = read_json(run / 'CONFIG.json')
    assert config is not None, 'Missing producer CONFIG.json'
    result = read_json(run / 'RESULTS.json', read_json(run / 'TIMEOUT.json', {}))
    traces = read_json(run / 'TRACES.json', [])
    codecs = read_json(run / 'CODECS.json', [])
    disps = read_json(run / 'DISPLACEMENTS.json', [])
    mode_checks = read_json(run / 'MODE_CHECKS.json', [])
    report['original_result'] = result
    report['original_tool_failed_or_partial'] = not (result.get('tool_valid') is True and result.get('integrity_complete') is True)
    with (run / 'METRICS.csv').open(newline='') as stream:
        rows = [{key: typed(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    with (run / 'SUMMARY.csv').open(newline='') as stream:
        summaries = [{key: typed(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    report['completeness'] = dict(metric_rows=len(rows), expected_metric_rows=20480,
                                  summary_rows=len(summaries), expected_summary_rows=90,
                                  native_cache_records=len(traces), expected_native_cache_records=80,
                                  codecs=len(codecs), expected_codecs=10,
                                  base_reference_files=len(list(run.glob('BASE_REFERENCE_*.npz'))), expected_base_reference_files=8)
    complete = (len(rows), len(summaries), len(traces), len(codecs)) == (20480, 90, 80, 10)
    report['complete_fixed_matrix'] = complete
    if not report['original_tool_failed_or_partial']:
        audit.check(complete, 'complete fixed matrix')
    expected_hashes = dict(config['file_hashes_before'])
    for trace in traces:
        expected_hashes[str(run / trace['filename'])] = trace['sha256']
        expected_hashes[str(run / trace['original_reference_filename'])] = trace['original_reference_sha256']
    for codec in codecs:
        expected_hashes[str(run / codec['filename'])] = codec['sha256']
    for name, expected in result.get('output_hashes', {}).items():
        expected_hashes[str(run / name)] = expected
    hash_checks = {}
    for name, expected in expected_hashes.items():
        path = Path(name)
        actual = audit.digest(path) if path.is_file() else None
        hash_checks[name] = dict(expected=expected, actual=actual, matches=actual == expected)
        audit.check(actual == expected, 'file hash/' + name)
    report['input_output_hashes'] = hash_checks
    report['evidence_file_hashes'] = {name: audit.digest(run / name) for name in
        ('CONFIG.json', 'RESULTS.json', 'TIMEOUT.json', 'TRACES.json', 'CODECS.json',
         'METRICS.csv', 'SUMMARY.csv', 'DISPLACEMENTS.json', 'MODE_CHECKS.json') if (run / name).is_file()}
    if result.get('file_hashes_after'):
        audit.check(result['file_hashes_after'] == config['file_hashes_before'], 'producer before/after file manifest')
    teacher = next(Path(path) for path, expected in config['file_hashes_before'].items() if expected == TEACHER_SHA)
    began = time.perf_counter()
    ply = PlyData.read(str(teacher))['vertex'].data
    audit.large_io_seconds += time.perf_counter() - began
    column = lambda names: np.column_stack([ply[name] for name in names]).astype(np.float32)
    original = column(('x', 'y', 'z'))
    assert original.shape == (POINTS, 3) and np.isfinite(original).all()
    full, non_xyz = hashlib.sha256(), hashlib.sha256()
    for field in FIELDS:
        if field == '_xyz':
            values = original
        elif field == '_features_dc':
            values = column([f'f_dc_{i}' for i in range(3)])
        elif field == '_features_rest':
            values = column([f'f_rest_{i}' for i in range(45)]).reshape(POINTS, 3, 15).transpose(0, 2, 1).copy()
        elif field == '_scaling':
            values = column([f'scale_{i}' for i in range(3)])
        elif field == '_rotation':
            values = column([f'rot_{i}' for i in range(4)])
        else:
            values = column(['opacity'])
        raw = values.tobytes(order='C')
        full.update(raw)
        if field != '_xyz':
            non_xyz.update(field.encode('ascii'))
            non_xyz.update(raw)
        del values, raw
    parameter_hash, other_hash = full.hexdigest(), non_xyz.hexdigest()
    audit.check(parameter_hash == config['parameters_before'], 'PLY reconstructed initial six-parameter hash')
    audit.check(other_hash == config['non_xyz_parameters_before'], 'PLY reconstructed non-xyz hash')
    for key in ('parameters_after', 'parameters_after_final_restore'):
        if key in result:
            audit.check(result[key] == parameter_hash, 'restored parameter hash/' + key)
    audit.check(result.get('original_xyz_object_restored') is True, 'producer original XYZ object restored')
    audit.check(result.get('parameters_restored_after_exit') is True, 'producer final parameter restoration')
    for check in mode_checks:
        audit.check(check['non_xyz_hash'] == other_hash and check['xyz_matches_file_decode'] and check['no_optimizer'], 'mode parameter integrity/' + check['mode'])
    report['parameter_integrity'] = dict(independently_reconstructed_initial_hash=parameter_hash,
        independently_reconstructed_non_xyz_hash=other_hash,
        object_identity_and_live_final_state_evidence='Producer object-identity/restoration records only; CPU audit independently reconstructs PLY hash, not past live objects.')
    audit.check(config['train_ranks'] == list(RANKS) and config['image_size'] == [979, 546], 'fixed cameras/image size')
    epsilons = {str(tau): tau*.2/(math.sqrt(3)*(config['fmax']/math.cos(config['maximum_fov_half_radians'])+tau)) for tau in (.25, .125)}
    audit.check(epsilons == config['epsilons'], 'PLEDGE epsilon algebra')
    report['independent_epsilons'] = epsilons
    references, probe_checks = {}, []
    for rank in RANKS:
        path = run / f'BASE_REFERENCE_{rank:03d}.npz'
        if not path.is_file():
            continue
        ref = audit.archive(path)
        references[rank] = ref
        common = ref['active'] & ref['probe_active']
        equal = np.array_equal(ref['depths'][common].view(np.uint32), ref['probe_depths'][common].view(np.uint32))
        audit.check(equal, 'probe/common active depth bitwise/' + str(rank))
        record = dict(rank=rank, common_active_points=int(common.sum()), common_depth_bitwise_equal=equal,
                      missing_original_active_points=int((ref['active'] & ~ref['probe_active']).sum()))
        base_trace = next((trace for trace in traces if trace['mode'] == 'base_float32' and trace['rank'] == rank), None)
        if base_trace:
            probe = base_trace['depth_probe']
            audit.check(probe['common_depth_bitwise_equal'] and probe['world_view_bitwise_identical'] and
                        probe['original_xyz_bitwise_identical'] and probe['changed_only_projection_and_canvas'], 'probe immutable-state record/' + str(rank))
            audit.check(probe['common_original_active_points'] == record['common_active_points'], 'probe common count/' + str(rank))
        probe_checks.append(record)
    report['probe_checks'] = probe_checks
    by_rows = {}
    max_derived_error = 0.
    for row in rows:
        identity = (row['mode'], row['view'], int(row['sample']))
        audit.check(identity not in by_rows, 'metric unique key/' + str(identity))
        for field, value in row_derived(row).items():
            if isinstance(value, bool):
                audit.check(value == row[field], 'row derived bool/' + str(identity) + '/' + field)
            else:
                error = abs(value - row[field])
                max_derived_error = max(max_derived_error, error)
                audit.check(error <= 1e-14, 'row RGB-derived metric/' + str(identity) + '/' + field)
            row[field] = value
        by_rows[identity] = row
    report['all_saved_RGB_derived_rows_recomputed'] = len(rows)
    report['max_RGB_derived_metric_error'] = max_derived_error
    codec_checks, cache_checks, independent_displacements, scalar_checks = [], [], [], []
    recorded_modes = {item['mode'] for item in codecs}
    orphan_codecs = []
    for path in sorted(run.glob('*.positions.bin')):
        mode = path.name.removesuffix('.positions.bin')
        if mode not in recorded_modes:
            definition = next(item for item in config['modes'] if item['name'] == mode)
            orphan_codecs.append(dict(mode=mode, filename=path.name, sha256=audit.digest(path),
                uniform_bits=definition['bits'], epsilon=epsilons[str(definition['tau'])] if definition['tau'] is not None else None,
                status='Encoded file exists but producer did not complete this mode or commit its codec metadata.'))
    report['uncommitted_encoded_files'] = orphan_codecs
    original_cache_by_rank = {rank: entry for rank, entry in zip(RANKS, config['old_caches'])}
    for codec in codecs + orphan_codecs:
        mode = codec['mode']
        decoded, record = independent_decode(run / codec['filename'], original, codec, audit)
        record['mode'] = mode
        codec_checks.append(record)
        tau = next(item['tau'] for item in config['modes'] if item['name'] == mode)
        if tau is not None:
            audit.check(codec['epsilon'] == epsilons[str(tau)], 'codec declared epsilon/' + mode)
        for trace in [item for item in traces if item['mode'] == mode]:
            rank, view = trace['rank'], trace['view']
            ref = references[rank]
            cache = audit.archive(run / trace['filename'])
            unique = cache['unique_ids']
            assert np.array_equal(np.unique(cache['candidate_ids']), unique)
            assert np.all((unique >= 0) & (unique < POINTS))
            assert np.array_equal(cache['q_active_ids'], np.sort(cache['q_active_ids']))
            audit.check(ref['probe_active'][unique].all(), 'all Q candidates original probe covered/' + mode + '/' + view)
            old_depth = ref['probe_depths'][unique]
            audit.check(np.isfinite(old_depth).all() and (old_depth > .2).all(), 'all Q candidate exact old depths valid/' + mode + '/' + view)
            audit.check(np.array_equal(old_depth.view(np.uint32), cache['old_depth_reference'].view(np.uint32)), 'old depth reference bits/' + mode + '/' + view)
            audit.check(np.array_equal(ref['active'][unique], cache['old_reference_active']), 'original active reference/' + mode + '/' + view)
            native_order, base_c_identity, moved = True, True, 0
            per_tile = {}
            for tile, (lo, hi) in enumerate(zip(cache['tile_offsets'][:-1], cache['tile_offsets'][1:])):
                ids = cache['candidate_ids'][int(lo):int(hi)]
                local = np.searchsorted(unique, ids)
                depths = cache['depths'][local]
                native_order = native_order and bool(np.all((depths[1:] > depths[:-1]) | ((depths[1:] == depths[:-1]) & (ids[1:] >= ids[:-1]))))
                c_order = np.lexsort((ids, old_depth[local]))
                changed = int(np.count_nonzero(c_order != np.arange(len(ids))))
                moved += changed
                if mode in ('base_float32', 'uniform32'):
                    base_c_identity = base_c_identity and changed == 0
                per_tile[tile] = (ids, c_order, changed, int((~ref['active'][ids]).sum()))
            audit.check(native_order, 'Qnative depth ascending tie ID/' + mode + '/' + view)
            if mode in ('base_float32', 'uniform32'):
                audit.check(base_c_identity, 'base general C lexsort restores every native tile item/' + mode + '/' + view)
            xs = ((2*np.arange(16)+1)*979//32).astype(np.int32)
            ys = ((2*np.arange(16)+1)*546//32).astype(np.int32)
            expected_xy = np.array([(x,y) for y in ys for x in xs], dtype=np.int32)
            audit.check(np.array_equal(cache['pixel_xy'], expected_xy), 'fixed pixel grid/' + mode + '/' + view)
            native_errors, scalar_error = [], 0.
            for sample in range(256):
                key = (mode, view, sample)
                if key not in by_rows:
                    continue
                row = by_rows[key]
                tile = int(cache['pixel_tile_index'][sample])
                ids, c_order, changed, old_inactive = per_tile[tile]
                audit.check(row['candidate_count'] == len(ids) and row['candidate_order_positions_changed'] == changed and row['old_inactive_candidate_count'] == old_inactive,
                            'row candidates/order/' + str(key))
                audit.check(row['x'] == int(expected_xy[sample,0]) and row['y'] == int(expected_xy[sample,1]) and row['tile_id'] == int(cache['tile_ids'][tile]), 'row pixel/tile/' + str(key))
                for label, array in (('GT', cache['gt_rgb']), ('native', cache['native_rgb']), ('base', cache['base_native_rgb'])):
                    saved = np.array([row[label+'_'+channel] for channel in 'rgb'])
                    audit.check(np.array_equal(saved, array[sample].astype(np.float64)), 'row cached RGB/' + str(key) + '/' + label)
                native_errors.append(row['D_native_max_channel_error'])
                if sample in SAMPLES:
                    arm_errors = {}
                    for arm, order in (('C', c_order), ('D', range(len(ids)))):
                        rgb, accepted = scalar_ray(cache, ids, cache['pixel_xy'][sample], order)
                        saved = np.array([row[arm+'_'+channel] for channel in 'rgb'])
                        error = float(np.max(np.abs(rgb - saved)))
                        scalar_error = max(scalar_error, error)
                        audit.check(error <= 1e-10, 'independent scalar FP64 RGB/' + str(key) + '/' + arm, error)
                        audit.check(accepted == row['accepted_'+arm], 'independent scalar accepted/' + str(key) + '/' + arm)
                        arm_errors[arm] = error
                    scalar_checks.append(dict(mode=mode, view=view, sample=sample, max_arm_RGB_error=arm_errors))
            audit.check(max(native_errors, default=0.) <= 2e-5, 'D/native parity/' + mode + '/' + view)
            if mode == 'base_float32':
                old = audit.archive(Path(original_cache_by_rank[rank]['path']))
                audit.check(np.array_equal(old['pixel_xy'], cache['pixel_xy']) and np.array_equal(old['gt_rgb'], cache['gt_rgb']), 'old cache grid and GT/' + view)
                audit.check(float(np.max(np.abs(old['native_rgb'] - cache['native_rgb']))) <= 2e-5, 'base/old native parity/' + view)
            disp = displacement(cache, ref, original, decoded, tau)
            saved_disp = next((item for item in disps if item['mode'] == mode and item['view'] == view), None)
            if saved_disp:
                for field in ('selected_support_union_points', 'selected_support_invalid_original_projection_points',
                              'selected_support_max_axis_displacement_px', 'selected_support_original_near_crossing_count', 'displacement_control_valid'):
                    audit.check(disp[field] == saved_disp[field], 'independent displacement/' + mode + '/' + view + '/' + field,
                                dict(independent=disp[field], saved=saved_disp[field]) if disp[field] != saved_disp[field] else None)
            else:
                audit.check(False, 'missing displacement record/' + mode + '/' + view)
            disp.update(mode=mode, view=view)
            independent_displacements.append(disp)
            cache_checks.append(dict(mode=mode, view=view, native_depth_ID_order=native_order,
                zero_perturbation_C_itemwise_identity=base_c_identity if mode in ('base_float32', 'uniform32') else None,
                native_tiles=len(per_tile), unique_candidates=len(unique), all_Q_candidates_original_probe_covered=bool(ref['probe_active'][unique].all()),
                scalar_max_error=scalar_error))
            del cache, per_tile
        del decoded
    report['codec_checks'] = codec_checks
    report['cache_checks'] = cache_checks
    report['independent_displacements'] = independent_displacements
    report['scalar_FP64_fixed_rank_midpoints'] = list(SAMPLES)
    report['scalar_FP64_rays'] = scalar_checks
    report['scalar_FP64_ray_count'] = len(scalar_checks)
    report['scalar_FP64_arm_outputs'] = 2 * len(scalar_checks)
    report['PLEDGE_domain_statement'] = 'Outside-center selected footprint candidates were retained and checked; only center-in-frustum points are covered by the stated theorem. Actual control includes the full selected support union.'
    by_summary = {(item['mode'], item['view']): item for item in summaries}
    rebuilt = []
    for mode in MODES:
        selected = [row for row in rows if row['mode'] == mode]
        if not selected:
            continue
        byte_count = next(item['bytes'] for item in codecs if item['mode'] == mode)
        children = []
        for view in config['views']:
            subset = [row for row in selected if row['view'] == view]
            if subset:
                value = summarize(subset, mode, view, byte_count)
                children.append(value)
                if (mode, view) in by_summary:
                    compare(audit, value, by_summary[(mode, view)], 'summary/' + mode + '/' + view)
        value = summarize(selected, mode, 'ALL', byte_count, children)
        if (mode, 'ALL') in by_summary:
            compare(audit, value, by_summary[(mode, 'ALL')], 'summary/' + mode + '/ALL')
        rebuilt.extend(children + [value])
    report['independent_summaries'] = rebuilt
    if not complete or report['original_tool_failed_or_partial']:
        report['independent_decision'] = 'FAILED_TOOL_METHOD_UNKNOWN'
        report['scientific_gates_adjudicated'] = False
        report['reason'] = 'Producer failed or fixed matrix incomplete; partial rows are consistency evidence only and do not adjudicate the method.'
        report['producer_exact_failure_condition'] = result.get('tool_failure', result.get('phase', 'Missing RESULTS.json'))
        report['failed_Q_cache_evidence_limit'] = 'The failing uniform8 candidate list was not persisted. The exact missing candidate IDs/count cannot be independently reconstructed from saved evidence; no approximate depth, rerender, or geometry reconstruction was used to repair or replace this condition.'
        overall = {item['mode']: item for item in rebuilt if item['view'] == 'ALL'}
        if 'local_tau025' in overall:
            primary = overall['local_tau025']
            report['completed_primary_necessary_conditions_only'] = dict(
                total_exceed_fraction_at_least_1pct=primary['exceed_fraction'] >= .01,
                at_least_4_views_each_1pct=primary['qualifying_views'] >= 4,
                C_GT_mean_at_least_5pct_better=primary['C_gt_mse'] <= .95*primary['D_gt_mse'],
                C_own_worst2_not_worse=primary['C_worst2_view_gt_mse'] <= primary['D_worst2_view_gt_mse'],
                displacement_control_valid=all(item['displacement_control_valid'] for item in independent_displacements if item['mode'] == 'local_tau025'),
                complete_uniform_same_or_lower_byte_capacity_controls_available=False,
                adjudication='Descriptive necessary-condition checks for the completed primary only; no complete experiment method verdict.')
        return
    overall = {item['mode']: item for item in rebuilt if item['view'] == 'ALL'}
    primary = overall['local_tau025']
    uniforms = [overall[mode] for mode in MODES if mode.startswith('uniform')]
    eliminating = [item['mode'] for item in uniforms if item['encoded_position_bytes'] <= primary['encoded_position_bytes'] and not item['capacity_gate']]
    gates = dict(primary_total_exceed_fraction_at_least_1pct=primary['exceed_fraction'] >= .01,
        primary_at_least_4_views_each_1pct=primary['qualifying_views'] >= 4,
        primary_C_GT_mean_at_least_5pct_better=primary['C_gt_mse'] <= .95*primary['D_gt_mse'],
        primary_C_own_worst2_not_worse=primary['C_worst2_view_gt_mse'] <= primary['D_worst2_view_gt_mse'],
        no_same_or_lower_bytes_uniform_control_eliminates_capacity=not eliminating)
    valid_disp = all(item['displacement_control_valid'] for item in independent_displacements if item['mode'] == 'local_tau025')
    decision = 'NEEDS_ACTUAL_ENCODING_BENEFIT_CHECK' if all(gates.values()) else 'STOP_CURRENT_SORTING_QUANTIZATION_ROUTE'
    if not valid_disp:
        decision = 'INCONCLUSIVE_DISPLACEMENT_CONTROL_NOT_MET'
    audit.check(gates == result['gates'], 'all independent scientific gates')
    audit.check(valid_disp == result['primary_displacement_control_valid'], 'primary displacement state')
    audit.check(eliminating == result['eliminating_uniform_controls'], 'same/lower actual-byte eliminating controls')
    audit.check(decision == result['decision'], 'independent final decision')
    report.update(independent_decision=decision, scientific_gates_adjudicated=valid_disp,
                  independent_gates=gates, primary_displacement_control_valid=valid_disp,
                  eliminating_uniform_controls=eliminating,
                  uniform_actual_byte_controls=[dict(mode=item['mode'], bytes=item['encoded_position_bytes'],
                      at_or_below_primary=item['encoded_position_bytes'] <= primary['encoded_position_bytes'], capacity_gate=item['capacity_gate']) for item in uniforms])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    output = args.run / 'INDEPENDENT_ANALYSIS.json'
    assert not output.exists(), 'Independent audit output already exists; do not silently replace an audit.'
    report = dict(audit_valid=False, independent_decision='FAILED_TOOL_METHOD_UNKNOWN', failures=[],
                  producer_rerun=False, GPU_used=False, torch_imported=False, training_performed=False,
                  thread_limit=1, hard_wall_budget_seconds=90, all_IO_included_in_wall_budget=True,
                  independent_script=str(Path(__file__).resolve()), scientific_gates_adjudicated=False)
    audit = Audit(report)

    def timeout():
        report.update(audit_valid=False, independent_decision='FAILED_INDEPENDENT_AUDIT_TIMEOUT',
                      elapsed_wall_seconds=time.perf_counter()-start, hash_wall_seconds=audit.hash_seconds,
                      large_file_read_wall_seconds=audit.large_io_seconds, hashed_bytes=audit.hashed_bytes)
        write_report(output, report)
        os._exit(2)

    watchdog = threading.Timer(90, timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        report['independent_script_sha256'] = audit.digest(Path(__file__))
        audit_run(args, report, audit)
        report['audit_valid'] = not audit.failures
        if audit.failures:
            report['saved_evidence_decision_before_audit_failures'] = report['independent_decision']
            report['independent_decision'] = 'FAILED_TOOL_METHOD_UNKNOWN'
            report['scientific_gates_adjudicated'] = False
    except Exception as exc:
        report.update(audit_valid=False, audit_error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc(), independent_decision='FAILED_TOOL_METHOD_UNKNOWN',
                      scientific_gates_adjudicated=False)
    finally:
        watchdog.cancel()
        report.update(elapsed_wall_seconds=time.perf_counter()-start, hash_wall_seconds=audit.hash_seconds,
                      large_file_read_wall_seconds=audit.large_io_seconds, hashed_bytes=audit.hashed_bytes)
        report['compute_and_small_IO_wall_seconds'] = max(0., report['elapsed_wall_seconds'] - audit.hash_seconds - audit.large_io_seconds)
        write_report(output, report)
    print(json.dumps({key: report.get(key) for key in ('audit_valid', 'independent_decision', 'elapsed_wall_seconds',
          'hash_wall_seconds', 'large_file_read_wall_seconds', 'scalar_FP64_ray_count', 'scalar_FP64_arm_outputs', 'audit_error')}), flush=True)
    return 0 if report['audit_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
