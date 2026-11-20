"""Independent CPU audit of saved order-only evidence; never reruns its producer.

Only output is run/INDEPENDENT_ANALYSIS.json. The original failed run stays failed.
No imports from the producer or original ray replay are used.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from plyfile import PlyData


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def independent_ray(cache, point_ids, xy, permutation):
    """Scalar FP64 recurrence, including native reject/crossing semantics."""
    mapped = np.searchsorted(cache['unique_ids'], point_ids)
    assert np.array_equal(cache['unique_ids'][mapped], point_ids)
    means = cache['means2D'][mapped].astype(np.float64)
    parameters = cache['conic_opacity'][mapped].astype(np.float64)
    colors = cache['rgb'][mapped].astype(np.float64)
    threshold = float(np.float32(1) / np.float32(255))
    cap = float(np.float32(.99))
    stopping = float(np.float32(.0001))
    t = 1.0
    result = [0.0, 0.0, 0.0]
    accepted = []
    for position in permutation:
        position = int(position)
        dx = float(means[position, 0]) - int(xy[0])
        dy = float(means[position, 1]) - int(xy[1])
        a, b, c, opacity = map(float, parameters[position])
        exponent = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        if exponent > 0:
            continue
        alpha = min(cap, opacity * math.exp(exponent))
        if alpha < threshold:
            continue
        next_t = t * (1.0 - alpha)
        if next_t < stopping:
            break
        for channel in range(3):
            result[channel] += (float(colors[position, channel]) * alpha) * t
        accepted.append(position)
        t = next_t
    return np.clip(np.asarray(result, dtype=np.float64), 0, 1), np.asarray(accepted, dtype=np.int64)


def aggregate(rows, differences):
    def mean(field):
        return float(np.mean([float(row[field]) for row in rows]))
    exceeded = int(np.count_nonzero(differences > 1 / 255))
    return dict(pixels=len(rows), exceed_count=exceeded,
                exceed_fraction=exceeded / len(rows),
                max_channel_difference=float(differences.max()),
                p99_channel_difference=float(np.quantile(differences, .99)),
                order_mse=mean('order_mse'), old_gt_mse=mean('old_gt_mse'), new_gt_mse=mean('new_gt_mse'))


def audit(root, report):
    run = root / 'research/experiments/sorting-quantization-capacity-v0/run'
    config = load_json(run / 'CONFIG.json')
    original = load_json(run / 'RESULTS.json')
    post = load_json(run / 'RESULTS_POSTPROCESS.json')
    with (run / 'METRICS.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (run / 'SUMMARY_COMPLETE.csv').open(newline='') as stream:
        completed = list(csv.DictReader(stream))
    producer = Path(config['script']).read_text()
    observed_hashes = {}
    expected_hashes = {
        config['script']: config['script_sha256'],
        config['contract']: config['contract_sha256'],
        str(run / 'METRICS.csv'): post['metrics_sha256'],
        str(run / 'RESULTS.json'): post['source_results_sha256'],
    }
    for entry in [config['inputs'][key] for key in ('ply', 'cameras', 'traces')] + config['inputs']['caches']:
        expected_hashes[entry['path']] = entry['sha256']
    for path, expected in expected_hashes.items():
        actual = digest(path)
        observed_hashes[path] = dict(expected=expected, actual=actual, matches=actual == expected)
    report['input_hashes'] = observed_hashes
    assert all(v['matches'] for v in observed_hashes.values()), 'Input/content hash mismatch'
    for entry in config['inputs']['caches']:
        assert original['hashes_after']['caches'][entry['path']] == entry['sha256']
    for name in ('ply', 'cameras', 'traces'):
        assert original['hashes_after'][name] == config['inputs'][name]['sha256']
    assert original['hashes_after']['script'] == config['script_sha256']
    assert original['hashes_after']['contract'] == config['contract_sha256']
    report['evidence_files_sha256'] = {name: digest(run / name) for name in (
        'CONFIG.json', 'RESULTS.json', 'RESULTS_POSTPROCESS.json', 'METRICS.csv',
        'PREFLIGHT.json', 'SUMMARY.csv', 'SUMMARY_COMPLETE.csv')}
    report['original_run'] = dict(decision=original['decision'], tool_valid=original['tool_valid'],
        error=original['error'], elapsed_seconds=original['elapsed_seconds'],
        external_process_exit_code_recorded_by_postprocess=post['original_process_exit_code'],
        producer_failure_return_code=2,
        note='Producer main returns 2 on tool failure; postprocess records outer process exit 1. Neither is converted to success.')
    assert original['decision'] == 'FAILED_TOOL_METHOD_UNKNOWN' and original['tool_valid'] is False
    assert post['original_run_status'] == original['decision'] and post['original_error'] == original['error']
    assert post['original_process_exit_code'] == 1
    assert len(rows) == 6144 and len(completed) == 27
    traces = load_json(config['inputs']['traces']['path'])
    views = [trace['view'] for trace in traces]
    assert len(views) == 8 and len(set(views)) == 8
    assert [trace['rank'] for trace in traces] == [0, 31, 62, 93, 124, 155, 186, 218]
    keyed = {(int(r['bit']), r['view'], int(r['sample'])): r for r in rows}
    assert len(keyed) == 6144
    assert set(keyed) == {(bit, view, sample) for bit in (8, 12, 16) for view in views for sample in range(256)}
    derived = {}
    maximum_derived_errors = {name: 0.0 for name in ('max_channel_difference', 'order_mse', 'old_gt_mse', 'new_gt_mse')}
    for row in rows:
        old, new, gt = (np.asarray([float(row[prefix + c]) for c in 'rgb']) for prefix in ('old_', 'new_', 'GT_'))
        assert np.isfinite(np.concatenate([old, new, gt])).all()
        assert ((old >= 0) & (old <= 1) & (new >= 0) & (new <= 1)).all()
        calculated = dict(max_channel_difference=float(np.max(np.abs(new - old))),
            order_mse=float(np.mean((new - old) ** 2)),
            old_gt_mse=float(np.mean((old - gt) ** 2)), new_gt_mse=float(np.mean((new - gt) ** 2)))
        for name, value in calculated.items():
            maximum_derived_errors[name] = max(maximum_derived_errors[name], abs(float(row[name]) - value))
        assert (row['exceeds_1_over_255'] == 'True') == (calculated['max_channel_difference'] > 1 / 255)
        derived[(int(row['bit']), row['view'], int(row['sample']))] = calculated
    assert max(maximum_derived_errors.values()) <= 1e-15
    report['all_metrics'] = dict(rows=6144, configurations=3, pixels_per_configuration=2048,
        views=8, pixels_per_view=256, key_coverage_complete=True, finite=True,
        maximum_rgb_derived_field_errors=maximum_derived_errors)
    summaries = {}
    gates = {}
    summary_error = 0.0
    for bit in (8, 12, 16):
        for view in views + ['ALL']:
            selected = [r for r in rows if int(r['bit']) == bit and (view == 'ALL' or r['view'] == view)]
            differences = np.asarray([derived[(bit, r['view'], int(r['sample']))]['max_channel_difference'] for r in selected])
            entry = aggregate(selected, differences)
            entry.update(bit=bit, view=view)
            match = [r for r in completed if int(r['bit']) == bit and r['view'] == view]
            assert len(match) == 1
            for field in ('pixels', 'exceed_count', 'exceed_fraction', 'max_channel_difference',
                          'p99_channel_difference', 'order_mse', 'old_gt_mse', 'new_gt_mse'):
                summary_error = max(summary_error, abs(float(match[0][field]) - entry[field]))
            if view != 'ALL':
                assert (match[0]['capacity_gate'] == 'True') == (entry['exceed_fraction'] >= .01)
                assert match[0]['qualifying_views'] == ''
            summaries[f'{bit}:{view}'] = entry
        overall = summaries[f'{bit}:ALL'].copy()
        overall['qualifying_views'] = sum(summaries[f'{bit}:{v}']['exceed_fraction'] >= .01 for v in views)
        overall['capacity_gate'] = overall['qualifying_views'] >= 4 and overall['exceed_fraction'] >= .01
        for field, value in overall.items():
            if field in ('bit', 'view'):
                continue
            assert abs(float(post['gates'][str(bit)][field]) - float(value)) <= 1e-15
        completed_overall = next(r for r in completed if int(r['bit']) == bit and r['view'] == 'ALL')
        assert int(completed_overall['qualifying_views']) == overall['qualifying_views']
        assert (completed_overall['capacity_gate'] == 'True') == overall['capacity_gate']
        gates[str(bit)] = overall
    assert summary_error <= 1e-15
    passed = [bit for bit in (8, 12) if gates[str(bit)]['capacity_gate']]
    assert passed == post['passed_main_bits'] == original['passed_main_bits']
    assert bool(passed) == post['main_capacity_gate'] == original['main_capacity_gate']
    report['recomputed_view_and_bit_summaries'] = list(summaries.values())
    report['summary_complete_max_field_error'] = summary_error
    report['gates'] = gates
    report['passed_main_bits'] = passed

    # Confirm the narrow late serialization failure without executing the producer.
    assert "fieldnames=list(rows[0])" in producer
    assert "overall['qualifying_views']" in producer and "overall['capacity_gate']" in producer
    assert producer.index("result.update(tool_valid=True") < producer.index("write_csv(args.output / 'SUMMARY.csv', summaries)", producer.index("result.update(tool_valid=True"))
    sample = dict(bit=8, view=views[0], pixels=256)
    sink = io.StringIO()
    writer = csv.DictWriter(sink, fieldnames=list(sample))
    try:
        writer.writerows([sample, {**sample, 'qualifying_views': 8, 'capacity_gate': True}])
        raise AssertionError('Expected heterogeneous summary field error')
    except ValueError as error:
        report['serialization_failure_reproduced'] = str(error)
    report['static_contract_review'] = dict(
        fixed_bits=[8, 12, 16], full_model_bounds=True, ties_to_even_rounding=True,
        native_depth_plus_quantization_delta=True, stable_tile_sort=True,
        unchanged_support_and_projection_and_colors=True, fp64_compositing=True,
        native_float32_alpha_cap_threshold_and_early_T=True,
        crossing_term_excluded=True, final_clamp=True,
        renderer_or_contract_deviation_found=False,
        original_report_serialization_bug=True,
        source_review_note='Read producer source, contract and configuration; independently reproduce the equations and gate below. No producer functions imported.')

    source = PlyData.read(config['inputs']['ply']['path'])
    vertices = source['vertex'].data
    xyz = np.column_stack([vertices[axis] for axis in ('x', 'y', 'z')]).astype(np.float64)
    assert xyz.shape == (2050770, 3) and np.isfinite(xyz).all()
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    assert np.array_equal(lo, config['xyz_min']) and np.array_equal(hi, config['xyz_max'])
    steps = {str(bit): ((hi - lo) / (2 ** bit - 1)).tolist() for bit in (8, 12, 16)}
    assert steps == config['quantization_steps']
    assert config['source_ply_bytes'] == Path(config['inputs']['ply']['path']).stat().st_size
    assert config['position_float32_payload_bytes'] == len(xyz) * 12
    assert config['source_position_payload_fraction'] == len(xyz) * 12 / config['source_ply_bytes']
    cameras = {camera['img_name']: camera for camera in load_json(config['inputs']['cameras']['path'])}
    indices = [int((j + .5) * 256 / 8) for j in range(8)]
    details = []
    max_rgb_error = max_base_error = max_native_error = max_depth_rel = max_depth_abs = 0.0
    max_count_error = 0
    for trace in traces:
        filename = Path(config['inputs']['traces']['path']).parent / trace['filename']
        assert digest(filename) == trace['sha256']
        with np.load(filename, allow_pickle=False) as archive:
            cache = {key: archive[key] for key in archive.files}
        assert int(cache['point_count']) == len(xyz) and cache['pixel_xy'].shape == (256, 2)
        assert np.array_equal(np.unique(cache['candidate_ids']), cache['unique_ids'])
        camera = cameras[trace['view']]
        axis = np.asarray(camera['rotation'], dtype=np.float64)[:, 2]
        center = np.asarray(camera['position'], dtype=np.float64)
        native_depth = cache['depths'].astype(np.float64)
        rebuilt = (xyz[cache['unique_ids']] - center) @ axis
        depth_abs = float(np.max(np.abs(rebuilt - native_depth)))
        depth_rel = float(np.max(np.abs(rebuilt - native_depth) / (1 + np.abs(native_depth))))
        max_depth_abs, max_depth_rel = max(max_depth_abs, depth_abs), max(max_depth_rel, depth_rel)
        assert depth_rel <= 2e-5
        for pi in range(256):
            expected = keyed[(8, trace['view'], pi)]
            assert all(float(expected['GT_' + c]) == float(cache['gt_rgb'][pi, k]) for k, c in enumerate('rgb'))
            assert int(expected['x']) == int(cache['pixel_xy'][pi, 0]) and int(expected['y']) == int(cache['pixel_xy'][pi, 1])
            tile = int(cache['pixel_tile_index'][pi])
            assert int(expected['tile_id']) == int(cache['tile_ids'][tile])
            begin, end = map(int, cache['tile_offsets'][tile:tile + 2])
            for bit in (8, 12, 16):
                other = keyed[(bit, trace['view'], pi)]
                assert all(other['old_' + c] == expected['old_' + c] and other['GT_' + c] == expected['GT_' + c] for c in 'rgb')
                assert int(other['rank']) == trace['rank']
                assert int(other['candidate_count']) == end - begin
                assert int(other['accepted_new']) == int(other['accepted_old']) + int(other['newly_accepted']) - int(other['removed_accepted'])
                assert 0 <= int(other['tile_candidate_positions_changed']) <= end - begin
                assert int(other['newly_accepted']) <= int(other['executed_order_changed_contributions']) <= int(other['accepted_new'])
        view_error = 0.0
        for pi in indices:
            tile = int(cache['pixel_tile_index'][pi])
            begin, end = map(int, cache['tile_offsets'][tile:tile + 2])
            ids = cache['candidate_ids'][begin:end]
            local = np.searchsorted(cache['unique_ids'], ids)
            original_order = np.arange(len(ids))
            zero_order = np.argsort(native_depth[local], kind='stable')
            assert np.array_equal(original_order, zero_order)
            old_rgb, old_accepted = independent_ray(cache, ids, cache['pixel_xy'][pi], original_order)
            zero_rgb, zero_accepted = independent_ray(cache, ids, cache['pixel_xy'][pi], zero_order)
            assert np.array_equal(old_rgb, zero_rgb) and np.array_equal(old_accepted, zero_accepted)
            old_reported = np.asarray([float(keyed[(8, trace['view'], pi)]['old_' + c]) for c in 'rgb'])
            max_base_error = max(max_base_error, float(np.max(np.abs(old_rgb - old_reported))))
            max_native_error = max(max_native_error, float(np.max(np.abs(old_rgb - cache['native_rgb'][pi]))))
            positions = xyz[ids]
            original_ranks = {int(value): rank for rank, value in enumerate(old_accepted)}
            for bit in (8, 12, 16):
                levels = 2 ** bit - 1
                quantized = lo + np.rint((positions - lo) / (hi - lo) * levels) * ((hi - lo) / levels)
                counterfactual_depth = native_depth[local] + (quantized - positions) @ axis
                permutation = np.argsort(counterfactual_depth, kind='stable')
                rgb, accepted = independent_ray(cache, ids, cache['pixel_xy'][pi], permutation)
                row = keyed[(bit, trace['view'], pi)]
                reported = np.asarray([float(row['new_' + c]) for c in 'rgb'])
                rgb_error = float(np.max(np.abs(rgb - reported)))
                max_rgb_error, view_error = max(max_rgb_error, rgb_error), max(view_error, rgb_error)
                changed = sum(original_ranks.get(int(value), -1) != rank for rank, value in enumerate(accepted))
                counts = dict(candidate_count=len(ids), tile_candidate_positions_changed=int(np.count_nonzero(permutation != original_order)),
                    accepted_old=len(old_accepted), accepted_new=len(accepted), executed_order_changed_contributions=changed,
                    newly_accepted=len(set(accepted) - set(old_accepted)), removed_accepted=len(set(old_accepted) - set(accepted)))
                max_count_error = max(max_count_error, max(abs(count - int(row[field])) for field, count in counts.items()))
        details.append(dict(view=trace['view'], sample_indices=indices, rays=8, counterfactual_outputs=24,
            depth_unique_ids_checked=len(cache['unique_ids']), depth_max_absolute_error=depth_abs,
            depth_max_relative_error=depth_rel, counterfactual_max_rgb_error=view_error))
    assert max_rgb_error <= 1e-10 and max_base_error <= 1e-10 and max_native_error <= 2e-5
    assert max_count_error == 0
    report['independent_replay'] = dict(
        implementation='Independent scalar math.exp FP64 recurrence; direct original PLY and C2W camera matrices; no producer import',
        sample_rule='Eight fixed midpoint ranks floor((j+0.5)*256/8) per camera, j=0..7',
        sample_indices=indices, cameras=8, base_rays=64, zero_perturbation_rays=64,
        counterfactual_outputs=192, bits=[8, 12, 16], target_rgb_absolute_error=1e-10,
        maximum_counterfactual_rgb_error=max_rgb_error, maximum_base_rgb_error=max_base_error,
        maximum_native_rgb_parity_error_on_sample=max_native_error,
        maximum_depth_absolute_error_all_cached_unique_ids=max_depth_abs,
        maximum_depth_relative_error_all_cached_unique_ids=max_depth_rel,
        maximum_count_field_error=max_count_error, zero_order_and_rgb_bitwise=True,
        full_model_xyz_bounds_and_steps_verified=True, quantization_steps=steps, views=details)
    # Recheck every original evidence/input after this read-only analysis.
    assert all(digest(path) == value['actual'] for path, value in observed_hashes.items())
    assert all(digest(run / name) == sha for name, sha in report['evidence_files_sha256'].items())
    report.update(status='INDEPENDENT_EXISTING_EVIDENCE_AND_POSTPROCESS_VALIDATED',
        original_run_reclassified_as_success=False, all_source_and_evidence_files_unchanged=True,
        saved_metrics_and_postprocess_support_capacity_gate=True,
        decision='NEEDS_FULL_POSITION_QUANTIZATION_AND_NOVELTY' if passed else 'STOP_CURRENT_ORDER_COMPONENT_AT_FIXED_PRECISIONS',
        limitations=[
            'All 6144 rows and 27 summary rows checked algebraically; rendering independently replayed for 64 of 2048 rays (192 bit outputs), not all rays.',
            'Depth direction verified for every cached unique ID in all eight cameras; quantizer bounds use every original PLY point.',
            'Only order under fixed native support, projections, conic, opacity and RGB is varied. No full position quantization, additive error decomposition, actual codec/rate, speed, training, cross-scene generalization or novelty evidence.',
            'GT is descriptive only: mean GT error increases at 8 and 12 bits; passing is a necessary space/capacity signal, not evidence of improving photographs.',
            'Original failed RESULTS and nonzero exit are preserved. Recovery validates saved evidence after a report serialization failure; it does not make the original process successful.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    report = dict(status='INDEPENDENT_ANALYSIS_IN_PROGRESS', cpu_budget_seconds=30,
        gpu_used=False, training_performed=False, original_experiment_rerun=False,
        audit_script_sha256=digest(__file__))
    exit_code = 0
    try:
        audit(args.root, report)
    except Exception as error:
        report.update(status='INDEPENDENT_ANALYSIS_FAILED', error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc())
        exit_code = 1
    report['elapsed_wall_seconds'] = time.perf_counter() - start_wall
    report['elapsed_cpu_seconds'] = time.process_time() - start_cpu
    output = args.root / 'research/experiments/sorting-quantization-capacity-v0/run/INDEPENDENT_ANALYSIS.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'elapsed_wall_seconds', 'elapsed_cpu_seconds')}, ensure_ascii=False))
    if exit_code:
        print(report['traceback'])
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
