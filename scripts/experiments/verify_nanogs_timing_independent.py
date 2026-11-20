"""One-shot CPU-only timing/provenance audit; never imports the producer or torch."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / 'profile_output/nanogs-frozen-baseline-v0'
RUN = BASE / 'run-v1'
EVAL = PROJECT / 'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
ARMS = ('teacher', 'all219', 'nanogs')
METRICS = ('cuda', 'host_wall')
VIEWS = tuple(f'{8 * rank + 1:06d}.jpg' for rank in range(32))
TIME_TOL = 1e-9
RATIO_FPS_TOL = 1e-9
CONTRACT_SHA = '230f02ae07bae098a6a3679669d7ed019b6701b7fbaee55a8b688416da41363f'
MODEL_PINS = {
    'teacher': 'b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0',
    'all219': '96fcb4e9942c78fd26d9fb386f6f1c9429f66fce7405ac151c84fe529f9b20fd',
    'nanogs': 'cd945d1ec44f959c37137eefaf9a56b5f8db4fb66e5db0cf38afaba274f163ec',
}
EVAL_PINS = {
    'RESULTS.json': '19f229837935d3624f2e179112d58217f479e1f2a572e23ce0cac6700c17f29c',
    'CONFIG.json': '3b53a5c588fbd55d190dc232b61c72ce9998e0cf04d0b26393c3b32b83b368c0',
    'CAMERAS.json': 'd70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67',
    'RAW_MANIFEST.json': '82dd8bfa39548d64b0a2eb55cf3cccc9de7e08d19657dd089585dc8f9bdca0e7',
}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def csv_file(path):
    with Path(path).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def percentile(sorted_values, fraction):
    """Linear interpolation at (n-1)*q, implemented without numpy/statistics."""
    index = (len(sorted_values) - 1) * fraction
    lo = math.floor(index)
    hi = math.ceil(index)
    return sorted_values[lo] + (index - lo) * (sorted_values[hi] - sorted_values[lo])


def stats(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count or not all(math.isfinite(x) for x in ordered):
        raise ValueError('Empty or nonfinite population')
    mean = math.fsum(ordered) / count
    return {'count': count, 'min': ordered[0], 'p05': percentile(ordered, .05),
            'p25': percentile(ordered, .25), 'median': percentile(ordered, .5),
            'p75': percentile(ordered, .75), 'p95': percentile(ordered, .95),
            'max': ordered[-1], 'mean': mean,
            'sample_sd': math.sqrt(math.fsum((v - mean) ** 2 for v in ordered) / (count - 1))
            if count > 1 else None}


def hash_array(value):
    # Reproduce the declared byte-identity format, not any producer statistics.
    import numpy as np
    contiguous = np.ascontiguousarray(value)
    h = hashlib.sha256(f'{contiguous.dtype}:{contiguous.shape}\n'.encode())
    h.update(contiguous.tobytes())
    return h.hexdigest()


def audit(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f'One-shot audit will not overwrite {output}')
    checks, initial_hashes = [], {}
    report = {'status': 'INCOMPLETE', 'audit_pid': os.getpid(), 'audit_script_sha256': digest_file(__file__),
              'audit_python': sys.executable, 'tolerances': {'time_statistics_ms': TIME_TOL,
                  'ratio_and_FPS': RATIO_FPS_TOL, 'integer_and_order': 'exact'},
              'native_forward_calls': 0, 'GPU_used': False, 'training_performed': False,
              'producer_imported': False, 'original_quality_audit_status': 'FAIL_UNCHANGED',
              'scope': 'Independent stored timing arithmetic and provenance; no new live GPU observation.',
              'limitations': [
                  'Cannot reobserve historical live GPU tensors, event timestamps, warmups or render calls.',
                  'Model and replay integrity are checked against recorded witnesses and independently hashed files.',
                  'Replay RGB arrays were not saved separately; their recorded digests can be compared to actual reference arrays.',
                  'Only three cyclic arm permutations were measured; pairwise predecessors are not fully balanced.',
                  'Ascending/descending repeats are 5/4; no thermal or hardware-idle state is independently reobserved.',
                  'Timing PASS does not change the previous SSIM audit FAIL or establish image-quality/novelty PASS.']}

    def check(name, condition, detail=None):
        item = {'name': name, 'pass': bool(condition)}
        if detail is not None:
            item['detail'] = detail
        checks.append(item)

    def same_tree(name, actual, expected, tolerance=TIME_TOL):
        if isinstance(expected, dict):
            check(name + ':keys', isinstance(actual, dict) and set(actual) == set(expected))
            if isinstance(actual, dict):
                for key, value in expected.items():
                    if key in actual:
                        same_tree(name + '/' + key, actual[key], value, tolerance)
        elif isinstance(expected, (int, str, bool)) or expected is None:
            check(name, actual == expected)
        else:
            difference = abs(float(actual) - expected)
            check(name, math.isfinite(float(actual)) and difference <= tolerance, difference)

    def remember(path):
        path = Path(path)
        if time.monotonic() - started > 110:
            raise TimeoutError('110-second audit work limit; external watchdog has 120-second hard limit')
        key = str(path.resolve())
        if key not in initial_hashes:
            initial_hashes[key] = digest_file(path)
        return initial_hashes[key]

    try:
        # No scientific rendering/producer module is imported anywhere in this script.
        import numpy as np
        protected_outputs = list(RUN.iterdir()) + [BASE / 'CONTRACT.md', BASE / 'WATCHDOG_v1.json',
            BASE / 'run-native-benchmark.sh', BASE / 'launch-v1.log',
            PROJECT / 'scripts/experiments/benchmark_nanogs_native.py', Path(__file__),
            PROJECT / 'research/experiments/nanogs-frozen-baseline-v0/INDEPENDENT_PIXEL_DISPOSITION.json']
        for path in protected_outputs:
            if path.is_file():
                remember(path)
        contract_sha = remember(BASE / 'CONTRACT.md')
        check('locked_contract_SHA', contract_sha == CONTRACT_SHA)
        result = json_file(RUN / 'RESULTS.json')
        config = json_file(RUN / 'CONFIG.json')
        manifest = json_file(RUN / 'MANIFEST.json')
        watchdog = json_file(BASE / 'WATCHDOG_v1.json')
        report['benchmark_watchdog'] = watchdog
        report['benchmark_session_id_from_root'] = 87036
        check('benchmark_terminal', result['decision'] == 'BENCHMARK_COMPLETE' and result['stage'] == 'terminal')
        check('benchmark_watchdog_exit', watchdog['process_exit_code'] == watchdog['tee_exit_code'] == 0)
        check('benchmark_watchdog_budget', watchdog['complete'] and watchdog['within_300_seconds'] and
              0 < watchdog['actual_process_and_pipe_wall_seconds'] <= 300)
        for key, expected in {'forward_attempted': 984, 'forward_completed': 984, 'parity_completed': 96,
                              'warmup_completed': 24, 'measured_completed': 864,
                              'backward_calls': 0, 'optimizer_steps': 0}.items():
            check('producer_counter/' + key, result[key] == expected)
        check('producer_integrity_record', result['integrity_complete'] and not result['integrity_errors'])
        check('GPU_peak_limit', result['GPU_memory']['peak_allocated_bytes'] <= 10 * 1024 ** 3)
        report['benchmark_GPU_memory_record'] = result['GPU_memory']
        report['benchmark_hardware_record'] = config['hardware']
        check('config_contract_SHA', config['contract_sha256'] == CONTRACT_SHA)
        check('view_names', tuple(config['test_views']) == VIEWS)
        check('warmup_indices', config['warmup_camera_indices'] == [0, 4, 8, 13, 17, 22, 26, 31])
        check('no_outlier_removal_record', config['no_outlier_removal'] is True)
        check('render_contract', config['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False,
              exposure=False, scaling_modifier=1., separate_sh=False, native_final_clamp=True))

        # Independent ordinal -> repeat/arm/rank reconstruction, not producer timing_plan().
        raw = csv_file(RUN / 'rawtimes.csv')
        expected_plan, rows = [], []
        check('raw_row_count_no_filtering', len(raw) == 864)
        for ordinal in range(864):
            repeat, offset = divmod(ordinal, 96)
            position, camera_position = divmod(offset, 32)
            rank = camera_position if repeat % 2 == 0 else 31 - camera_position
            expected_plan.append({'timed_forward_index': ordinal, 'repeat': repeat,
                'arm': ARMS[(repeat + position) % 3], 'arm_position': position,
                'camera_direction': 'ascending' if repeat % 2 == 0 else 'descending',
                'camera_position': camera_position, 'camera_rank': rank, 'image_name': VIEWS[rank]})
        check('stored_plan_exact', config['complete_plan'] == expected_plan)
        for ordinal, record in enumerate(raw):
            integers = ('timed_forward_index', 'repeat', 'arm_position', 'camera_position', 'camera_rank')
            row = {key: int(value) if key in integers else float(value) if key.endswith('_ms') else value
                   for key, value in record.items()}
            check(f'raw_order/{ordinal}', ordinal < len(expected_plan) and
                  all(row[key] == value for key, value in expected_plan[ordinal].items()))
            check(f'raw_finite_positive/{ordinal}', all(math.isfinite(row[k]) and row[k] > 0
                                                        for k in ('cuda_ms', 'host_wall_ms')))
            rows.append(row)
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm]
            check('arm_frames/' + arm, len(selected) == 288)
            check('per_view_repeats/' + arm, Counter(r['camera_rank'] for r in selected) == Counter({i: 9 for i in range(32)}))
            check('per_repeat_frames/' + arm, Counter(r['repeat'] for r in selected) == Counter({i: 32 for i in range(9)}))
            check('position_repeats/' + arm, all(len({r['repeat'] for r in selected if r['arm_position'] == p}) == 3
                                               for p in range(3)))
            check('direction_5_to_4/' + arm,
                  len({r['repeat'] for r in selected if r['camera_direction'] == 'ascending'}) == 5 and
                  len({r['repeat'] for r in selected if r['camera_direction'] == 'descending'}) == 4)

        per_view, per_repeat, ratios = [], [], []
        summaries = {'arms': {}, 'speedup_ratios': {},
                     'ratio_convention': 'numerator latency / denominator latency; >1 means denominator faster'}
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm]
            for rank, name in enumerate(VIEWS):
                samples = [r for r in selected if r['camera_rank'] == rank]
                entry = {'arm': arm, 'camera_rank': rank, 'image_name': name, 'repeats': len(samples)}
                for metric in METRICS:
                    entry[metric + '_median_ms'] = percentile(sorted(r[metric + '_ms'] for r in samples), .5)
                per_view.append(entry)
            for repeat in range(9):
                samples = [r for r in selected if r['repeat'] == repeat]
                entry = {'arm': arm, 'repeat': repeat, 'frames': len(samples)}
                for metric in METRICS:
                    total = math.fsum(r[metric + '_ms'] for r in samples)
                    entry[metric + '_total_ms'] = total
                    entry[metric + '_FPS'] = 32000 / total
                per_repeat.append(entry)
            summaries['arms'][arm] = {}
            for metric in METRICS:
                summaries['arms'][arm][metric] = {
                    'all_288_frame_latency_ms': stats([r[metric + '_ms'] for r in selected]),
                    'repeat_32view_total_ms': stats([r[metric + '_total_ms'] for r in per_repeat if r['arm'] == arm]),
                    'repeat_32view_FPS': stats([r[metric + '_FPS'] for r in per_repeat if r['arm'] == arm])}
        for numerator, denominator in (('teacher', 'all219'), ('teacher', 'nanogs'), ('all219', 'nanogs')):
            pair = numerator + '_over_' + denominator
            summaries['speedup_ratios'][pair] = {}
            for metric in METRICS:
                paired, views = [], []
                for index in range(9):
                    values = {r['arm']: r[metric + '_total_ms'] for r in per_repeat if r['repeat'] == index}
                    value = values[numerator] / values[denominator]
                    paired.append(value)
                    ratios.append(dict(pair=pair, metric=metric, scope='paired_repeat_32view_total', index=index, ratio=value))
                for index in range(32):
                    values = {r['arm']: r[metric + '_median_ms'] for r in per_view if r['camera_rank'] == index}
                    value = values[numerator] / values[denominator]
                    views.append(value)
                    ratios.append(dict(pair=pair, metric=metric, scope='per_view_9repeat_median', index=index, ratio=value))
                summaries['speedup_ratios'][pair][metric] = {
                    'paired_repeat_total_ratio': stats(paired), 'per_view_median_ratio': stats(views),
                    'pooled_288frame_total_ratio': math.fsum(r[metric + '_ms'] for r in rows if r['arm'] == numerator) /
                    math.fsum(r[metric + '_ms'] for r in rows if r['arm'] == denominator)}
        for filename, expected in [('PER_VIEW_MEDIANS.csv', per_view), ('REPEAT_TOTALS.csv', per_repeat),
                                   ('SPEEDUP_RATIOS.csv', ratios)]:
            actual = csv_file(RUN / filename)
            check(filename + ':row_count', len(actual) == len(expected))
            for index, (stored, calculated) in enumerate(zip(actual, expected)):
                check(f'{filename}/{index}:columns', set(stored) == set(calculated))
                for key, value in calculated.items():
                    converted = int(stored[key]) if isinstance(value, int) else float(stored[key]) if isinstance(value, float) else stored[key]
                    same_tree(f'{filename}/{index}/{key}', converted, value,
                              RATIO_FPS_TOL if key == 'ratio' or key.endswith('FPS') else TIME_TOL)
        same_tree('RESULTS/timing_summary', result['timing_summary'], summaries)
        report['independently_recomputed_summary'] = summaries
        report['independently_recomputed_per_view_medians'] = per_view
        report['independently_recomputed_repeat_totals'] = per_repeat
        report['independently_recomputed_ratios'] = ratios
        report['clock_difference'] = {arm: {
            'cuda_minus_host_wall_ms': stats([r['cuda_ms'] - r['host_wall_ms'] for r in rows if r['arm'] == arm]),
            'cuda_greater_than_host_rows': sum(r['cuda_ms'] > r['host_wall_ms'] for r in rows if r['arm'] == arm),
            'interpretation': 'Different clocks retained unchanged. Cause of positive mean difference is unknown; no post-hoc rejection threshold.'}
            for arm in ARMS}

        # Byte provenance is re-hashed on CPU, separately from the producer's assertions.
        input_before, input_after = manifest['input_hashes_before'], manifest['input_hashes_after']
        check('input_hash_table_keys', set(input_before) == set(input_after))
        check('config_manifest_input_table', config['input_hashes_before'] == input_before)
        input_table = []
        for path, before in input_before.items():
            current = remember(path)
            after = input_after.get(path)
            check('input_file/' + path, current == before == after)
            input_table.append(dict(path=path, producer_before=before, producer_after=after, audit_sha256=current))
        report['input_file_provenance'] = input_table
        source_table = []
        for path, digest in config['source_hashes_before'].items():
            current = remember(path)
            check('source_file/' + path, current == digest == input_before.get(path) == input_after.get(path))
            source_table.append(dict(path=path, source_before=digest, producer_after=input_after.get(path), audit_sha256=current))
        report['source_file_provenance'] = source_table
        output_table = []
        check('output_manifest_complete', set(manifest['outputs']) ==
              {p.name for p in RUN.iterdir() if p.is_file() and p.name not in ('RESULTS.json', 'CONFIG.json', 'MANIFEST.json')})
        for relative, entry in manifest['outputs'].items():
            path = RUN / relative
            current = remember(path)
            check('output_file/' + relative, current == entry['sha256'] and path.stat().st_size == entry['bytes'])
            output_table.append(dict(path=str(path), expected=entry, audit_sha256=current))
        report['output_file_provenance'] = output_table
        for filename, expected in EVAL_PINS.items():
            check('evaluation_pin/' + filename, remember(EVAL / filename) == expected)
        evaluation = json_file(EVAL / 'RESULTS.json')
        old_manifest = {entry['path']: entry for entry in json_file(EVAL / 'RAW_MANIFEST.json')}
        parity = json_file(RUN / 'PARITY.json')
        check('parity_96_rows', len(parity) == 96)
        parity_table = []
        for ordinal, witness in enumerate(parity):
            arm, rank = ARMS[ordinal // 32], ordinal % 32
            view = VIEWS[rank]
            relative = f'raw/{arm}/{view}.npz'
            path = EVAL / relative
            file_sha = remember(path)
            check(f'parity/{ordinal}:order', (witness['arm'], witness['camera_rank'], witness['image_name']) == (arm, rank, view))
            check(f'parity/{ordinal}:file', file_sha == witness['reference_file_sha256'] == old_manifest[relative]['sha256'])
            with np.load(path, allow_pickle=False) as saved:
                value = saved['rgb']
                check(f'parity/{ordinal}:array_shape_dtype', value.shape == (3, 546, 979) and value.dtype == np.float32 and np.isfinite(value).all())
                array_sha = hash_array(value)
            check(f'parity/{ordinal}:actual_reference_array', array_sha == witness['reference_RGB_array_sha256'])
            check(f'parity/{ordinal}:recorded_replay', witness['passed'] and witness['max_abs_RGB_error'] == 0 and
                  witness['replay_RGB_array_sha256'] == array_sha)
            parity_table.append(dict(arm=arm, camera_rank=rank, image_name=view, path=str(path),
                file_sha256=file_sha, independent_array_sha256=array_sha, recorded_replay=witness))
        report['parity_reference_provenance'] = parity_table

        model_table = {}
        for arm in ARMS:
            before, integrity = result['models'][arm]['before'], result['model_integrity'][arm]
            after = integrity['after']
            check('model_records/' + arm, before == after and integrity['loaded'] and integrity['verified'])
            check('model_pin/' + arm, before['sha256'] == MODEL_PINS[arm] == evaluation['models'][arm]['before']['sha256'])
            check('model_point_SH/' + arm, before['point_count'] == (2050770 if arm == 'teacher' else 1025385) and
                  before['active_sh_degree'] == before['max_sh_degree'] == 3)
            check('model_file_metadata/' + arm, config['model_files'][arm] == evaluation['model_files'][arm])
            model_table[arm] = {'before_record': before, 'after_record': after, 'live_tensors_reobserved': False}
        report['model_record_provenance'] = model_table
        for arm in ARMS:
            actual_bytes = config['actual_PLY_bytes'][arm]
            check('actual_PLY_bytes/' + arm, actual_bytes is None if arm == 'all219' else
                  actual_bytes == Path(config['model_files'][arm]['path']).stat().st_size)
        cameras_before, cameras_after = json_file(RUN / 'CAMERAS_BEFORE.json'), json_file(RUN / 'CAMERAS_AFTER.json')
        check('camera_files_identical_bytes', remember(RUN / 'CAMERAS_BEFORE.json') == remember(RUN / 'CAMERAS_AFTER.json'))
        check('camera_records_before_after', cameras_before == cameras_after)
        check('camera_names_32', tuple(c['image_name'] for c in cameras_before) == VIEWS)
        check('camera_canonical_digest', hashlib.sha256(json.dumps(cameras_before, sort_keys=True).encode()).hexdigest() ==
              config['camera_records_sha256_before'])
        old_cameras = {c['image_name']: c for c in json_file(EVAL / 'CAMERAS.json')}
        for camera in cameras_before:
            check('camera_prior_identity/' + camera['image_name'], camera == old_cameras[camera['image_name']])
            for key in ('R', 'T', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
                value = np.asarray(camera[key]['values'], dtype=np.float64 if key in ('R', 'T') else np.float32)
                check('camera_serialized_array/' + camera['image_name'] + '/' + key, hash_array(value) == camera[key]['sha256'])
        check('camera_image_before_after', config['camera_image_array_sha256_before'] ==
              result['camera_integrity']['image_array_sha256_after'] and result['camera_integrity']['verified'])
        check('camera_image_identity_to_prior', all(config['camera_image_array_sha256_before'][view] ==
              evaluation['gt_array_sha256'][view] for view in VIEWS))
        report['camera_provenance'] = {'before_file_sha256': remember(RUN / 'CAMERAS_BEFORE.json'),
            'after_file_sha256': remember(RUN / 'CAMERAS_AFTER.json'),
            'matrix_records_before': cameras_before, 'matrix_records_after': cameras_after,
            'image_array_sha256_before': config['camera_image_array_sha256_before'],
            'image_array_sha256_after': result['camera_integrity']['image_array_sha256_after'],
            'live_camera_tensors_reobserved': False}
        disposition = json_file(PROJECT / 'research/experiments/nanogs-frozen-baseline-v0/INDEPENDENT_PIXEL_DISPOSITION.json')
        check('old_quality_FAIL_preserved', disposition['formal_audit_status'] == 'FAIL_UNCHANGED')
        forbidden = [name for name in sys.modules if name in ('torch', 'scene', 'gaussian_renderer', 'benchmark_nanogs_native')]
        check('no_GPU_or_producer_import', not forbidden)
        report['forbidden_imports'] = forbidden
        # A second hash pass detects modifications during this independent audit.
        audit_after = {}
        for path, before in initial_hashes.items():
            audit_after[path] = digest_file(path)
            check('audit_read_only/' + path, audit_after[path] == before)
        report['audit_input_hashes_before'] = initial_hashes
        report['audit_input_hashes_after'] = audit_after
        report['dimensions'] = dict(raw_rows=len(rows), per_view_rows=len(per_view), per_repeat_rows=len(per_repeat),
                                    ratio_rows=len(ratios), parity_rows=len(parity), input_files=len(input_table))
        report['status'] = 'PASS' if all(item['pass'] for item in checks) else 'FAIL'
    except Exception as error:
        report['status'] = 'FAIL'
        report['exception'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
    report['wall_seconds'] = time.monotonic() - started
    report['checks_count'] = len(checks)
    report['failed_checks'] = [item for item in checks if not item['pass']]
    report['checks'] = checks
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'wall_seconds', 'checks_count')} |
                     {'failed_checks': len(report['failed_checks']), 'output': str(output)}), flush=True)
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'INDEPENDENT_TIMING_VERIFICATION.json')
    raise SystemExit(audit(parser.parse_args().output))
