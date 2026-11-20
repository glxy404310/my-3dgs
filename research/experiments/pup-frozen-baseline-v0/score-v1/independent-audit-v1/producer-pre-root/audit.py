#!/usr/bin/env python3
"""Independent CPU audit of saved PUP scores, mask and PLY export; no producer imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SCORE = HERE.parent
ORIGINAL = SCORE/'run-v1'
OUTPUT = HERE/'run-v1'
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
TEACHER = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply')
N, K, CHUNK = 2050770, 1025385, 65536
WALL, RSS = 120.0, 8*1024**3
ATOL, RTOL = 1e-4, 1e-6
STATUS = 'OFFICIAL_FIXED_SCORE_EXPORT_COMPLETE_NO_QUALITY_CLAIM'
PINS = {
    'SCORE_RESULT.json': '3775aa13372c4af264e384f3996424b94ac64362df3044160358f1f5d05fad68',
    'SUPERVISOR.json': 'fb5bfcb5cd09d4aac53e0571b4726467a0346d0b6b97dfc24db51a047395327a',
    'point_cloud.ply': '7581b48c11ec44f8577028ee77b0bf5c3361d5370f5dee1105c27d20c718107c',
}
ARTIFACTS = {
    'fishers': ('fishers_complete219_resolution1.pt', (N, 6, 6), 'float32'),
    'singular_values': ('singular_values.pt', (N, 6), 'float32'),
    'scores': ('scores_log_svd_sum.pt', (N,), 'float32'),
    'prune_mask': ('original_prune_mask.pt', (N,), 'bool'),
    'survivor_indices': ('survivor_indices.pt', (K,), 'int64'),
}
MATRICES = {'R': ('float64', [3, 3]), 'T': ('float64', [3]),
    'world_view_transform': ('float32', [4, 4]), 'projection_matrix': ('float32', [4, 4]),
    'full_proj_transform': ('float32', [4, 4]), 'camera_center': ('float32', [3])}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return dict(sha256=h.hexdigest(), bytes=path.stat().st_size)


def check_prelaunch():
    manifest = read(HERE/'PRELAUNCH_IDENTITY.json')
    for name, expected in manifest['files'].items():
        need(identity(HERE/name) == expected, 'Audit prelaunch identity changed: '+name)
    for name, expected in manifest['original_records'].items():
        need(identity(ORIGINAL/name) == expected, 'Original record changed since audit preparation: '+name)
    for text, expected in manifest['source_text'].items():
        need(identity(text) == expected, 'Reviewed author source snapshot changed: '+text)
    return manifest


def original_identities():
    result = {str(p): identity(p) for p in sorted(ORIGINAL.iterdir()) if p.is_file()}
    result[str(SCORE/'EXTERNAL_RUN.json')] = identity(SCORE/'EXTERNAL_RUN.json')
    for name, expected in PINS.items():
        need(result[str(ORIGINAL/name)]['sha256'] == expected, 'Root artifact pin changed: '+name)
    return result


def input_identities(inputs):
    result = {}
    for text, expected in inputs['files'].items():
        result[text] = identity(text)
        need(result[text]['sha256'] == expected, 'Original input hash mismatch: '+text)
    heads = {}
    for text, expected in inputs['repos'].items():
        heads[text] = subprocess.check_output(['git', '-C', text, 'rev-parse', 'HEAD'], text=True).strip()
        need(heads[text] == expected, 'Recorded source HEAD mismatch: '+text)
    return dict(files=result, git_heads=heads)


def normalized(name):
    need('/' not in name and '\\' not in name, 'Non-basename camera name')
    value = name[:-4] if name.endswith('.jpg') else name
    need(value.isdigit() and len(value) == 6, 'Unexpected camera basename')
    return value


def record_checks(result, canonical, cameras, inputs, supervisor, external, np):
    need(result['status'] == supervisor['status'] == STATUS and result['phase'] == 'complete', 'Original completion status')
    need(result['GT_used_as_label'] is False and result['optimizer_constructed'] is False and
         result['training_steps'] == 0, 'Recorded frozen/zero-update state')
    need((result['forward_calls'], result['statistical_backward_calls'], result['pool_calls_started']) ==
         (220, 219, 219) and result['active_camera'] is None, 'Recorded call counts')
    need(canonical['status'] == 'CANONICAL_NATIVE_ONLY' and
         canonical['render_calls'] == canonical['backward_calls'] == 0, 'Canonical recorded zero render/backward')
    raw = canonical['raw_fields']
    need(len(raw) == 6 and all(result[key] == raw for key in
         ('raw_fields_before', 'raw_fields_after_score', 'raw_fields_final_teacher')), 'Recorded raw-six identity')
    source = PROJECT/'idea-stage/artifact-pivot-20261023/pup-readonly/sourceTEXT/main/scene/gaussian_model.py.txt'
    need(identity(source)['sha256'] == inputs['files']['/home/administrator/projects/pup-frozen-baseline-e971ea48/scene/gaussian_model.py'],
         'Reviewed author prune/PLY source differs from pinned actual source')
    need(result['file_integrity'] is True and result['camera_identity_unchanged'] is True, 'Original integrity declarations')
    need(cameras['train_count'] == 219 and cameras['test_count'] == 32, 'Camera counts')
    native, pup = canonical['train_cameras'], cameras['train_cameras']
    need(len(native) == len(pup) == 219, 'Complete219 camera records')
    native_map = {row['basename']: row for row in native}
    need(len(native_map) == 219, 'Unique native basenames')
    order = [row['basename'] for row in pup]
    expected_train = [f'{i:06d}' for i in range(1, 252) if (i-1) % 8 != 0]
    expected_test = [f'{i:06d}' for i in range(1, 252) if (i-1) % 8 == 0]
    need(order == expected_train and cameras['scoring_order'] == order == result['scored_views'], 'Exact219 scoring order')
    need([normalized(x) for x in cameras['test_camera_names']] == expected_test, 'Disjoint32 test order')
    for row in pup:
        reference = native_map[row['basename']]
        need(normalized(row['image_name']) == row['basename'] == normalized(reference['image_name']), 'Normalized camera identity')
        need({k: v for k, v in reference.items() if k not in ('uid', 'image_name')} ==
             {k: v for k, v in row.items() if k not in ('uid', 'image_name')}, 'Camera/source-photo exact record match')
        need(inputs['files'][row['source_photo']] == row['source_photo_sha256'], 'Camera photo source binding')
        for key, (dtype, shape) in MATRICES.items():
            for entry in (reference[key], row[key]):
                value = np.asarray(entry['values'], dtype=dtype)
                camera_hash = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
                camera_hash.update(value.tobytes())
                need(entry['dtype'] == dtype and entry['shape'] == shape == list(value.shape) and
                     np.isfinite(value).all() and camera_hash.hexdigest() == entry['sha256'],
                     'Camera value/hash/dtype/shape: '+key)
    parity = result['parity']
    need(parity['passed'] is True and parity['full_image_max_abs'] == 0.0 and parity['limit'] == 2e-5 and
         parity['native_render_calls'] == 0 and inputs['files'][parity['reference_path']] == parity['reference_sha256'],
         'Recorded parity/source binding (not rerendered)')
    need(supervisor['hard_wall_seconds'] == external['hard_limit_seconds'] == inputs['hard_wall_seconds'] == 900 and
         supervisor['automatic_retries'] == 0 and external['attempts'] == 1 and external['exit_code'] == 0 and
         external['actual_host_wall_seconds'] == 79.0357449 and external['within_900_seconds'] is True and
         supervisor['within_900_seconds'] is True and supervisor['wall_seconds'] <= 900 and
         supervisor['input_integrity'] is True and supervisor['preflight_complete'] is True and
         supervisor['supervisor_torch_imported'] is False, 'Original external/supervisor budget and retry records')
    need(supervisor['host_RSS_limit_bytes'] == inputs['host_limit_bytes'] == 10*1024**3 and
         supervisor['peak_sampled_combined_host_RSS_bytes'] <= supervisor['host_RSS_limit_bytes'] and
         result['gpu']['peak_allocated_bytes'] <= inputs['gpu_allocated_limit_bytes'], 'Original recorded memory budgets')
    stages = supervisor['stages']
    need([x['stage'] for x in stages] == ['A', 'B'] and all(x['exit_code'] == 0 and
         x['remaining_live_group_pids'] == [] and x['pid'] == x['pgid'] for x in stages), 'Original stage/no-residual declarations')
    return dict(camera_count=219, test_count=32, camera_arrays_verified=219*2*len(MATRICES),
        scoring_order_exact=True, raw_six_record_identity=True, recorded_counters_consistent=True,
        historical_calls_replayed=False, historical_process_absence_reobserved=False, parity_rerendered=False)


def tensor_summary(value, np):
    flat = value.reshape(-1)
    counts = dict(finite_count=0, nan_count=0, negative_infinity_count=0, positive_infinity_count=0)
    minimum, maximum = None, None
    for start in range(0, len(flat), CHUNK*36):
        piece = flat[start:start+CHUNK*36]
        finite = np.isfinite(piece)
        counts['finite_count'] += int(finite.sum())
        counts['nan_count'] += int(np.isnan(piece).sum())
        counts['negative_infinity_count'] += int(np.isneginf(piece).sum())
        counts['positive_infinity_count'] += int(np.isposinf(piece).sum())
        if finite.any():
            low, high = float(piece[finite].min()), float(piece[finite].max())
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
    return dict(shape=list(value.shape), dtype='torch.'+str(value.dtype), finite_min=minimum,
                finite_max=maximum, **counts)


def ply_raw(vertex, field, np):
    if field == '_xyz':
        array = np.column_stack([vertex[x] for x in ('x', 'y', 'z')])
    elif field == '_features_dc':
        array = np.column_stack([vertex[f'f_dc_{i}'] for i in range(3)]).reshape(-1, 3, 1).transpose(0, 2, 1)
    elif field == '_features_rest':
        array = np.column_stack([vertex[f'f_rest_{i}'] for i in range(45)]).reshape(-1, 3, 15).transpose(0, 2, 1)
    elif field in ('_scaling', '_rotation'):
        prefix, count = ('scale', 3) if field == '_scaling' else ('rot', 4)
        array = np.column_stack([vertex[f'{prefix}_{i}'] for i in range(count)])
    elif field == '_opacity':
        array = vertex['opacity'][:, None]
    else:
        raise RuntimeError('Unexpected raw field '+field)
    return np.ascontiguousarray(array, dtype=np.float32)


def raw_identity(vertex, expected, np):
    records = {}
    for field in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
        array = ply_raw(vertex, field, np)
        need(np.isfinite(array).all(), 'Nonfinite PLY raw field '+field)
        records[field] = dict(dtype=str(array.dtype), shape=list(array.shape),
            sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())
        need(records[field] == expected[field], 'CPU PLY raw field differs from recorded raw identity: '+field)
        del array
    return records


def worker():
    audit = dict(status='AUDIT_IN_PROGRESS', scope='SAVED_SCORE_MASK_EXPORT_INPUT_INTEGRITY_ONLY',
        full_svd_recomputed=False, fisher_regenerated=False, render_calls=0, training_steps=0,
        tolerance=dict(finite_logscore_atol=ATOL, finite_logscore_rtol=RTOL), completed_checks=[])
    def checkpoint(name, value):
        audit[name] = value
        audit['completed_checks'].append(name)
        save(OUTPUT/'AUDIT_RESULT.json', audit)
        print(name, flush=True)
    try:
        need(os.environ.get('PUP_AUDIT_WORKER') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU worker environment')
        resource.setrlimit(resource.RLIMIT_AS, (RSS, RSS))
        check_prelaunch()
        checkpoint('original_artifacts_before', original_identities())
        result, canonical, cameras, inputs, supervisor = [read(ORIGINAL/name) for name in
            ('SCORE_RESULT.json', 'CANONICAL_NATIVE.json', 'PUP_CAMERAS.json', 'INPUTS_BEFORE.json', 'SUPERVISOR.json')]
        external = read(SCORE/'EXTERNAL_RUN.json')
        checkpoint('inputs_before', input_identities(inputs))
        import numpy as np
        import torch
        from plyfile import PlyData
        torch.set_num_threads(1)
        need(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized at CPU audit start')
        checkpoint('records', record_checks(result, canonical, cameras, inputs, supervisor, external, np))
        tensors, summaries = {}, {}
        for key, (name, shape, dtype) in ARTIFACTS.items():
            entry = result['artifacts'][key]
            need(Path(entry['path']) == ORIGINAL/name and identity(ORIGINAL/name) ==
                 dict(sha256=entry['sha256'], bytes=entry['bytes']), 'Saved tensor file hash/size '+key)
            value = torch.load(ORIGINAL/name, map_location='cpu', weights_only=True)
            need(type(value) is torch.Tensor and value.device.type == 'cpu' and tuple(value.shape) == shape and
                 value.dtype == getattr(torch, dtype) and not value.requires_grad, 'CPU tensor schema '+key)
            tensors[key] = value
            summaries[key] = tensor_summary(value.numpy(), np)
            need(summaries[key] == entry['summary'], 'Saved tensor summary mismatch '+key)
        checkpoint('tensor_summaries', summaries)
        fishers, sv, scores = [tensors[key].numpy() for key in ('fishers', 'singular_values', 'scores')]
        need(np.isfinite(fishers).all() and np.isfinite(sv).all() and not np.isnan(scores).any() and
             not np.isposinf(scores).any(), 'Tensor nonfinite policy')
        symmetry_max, diagonal_min, nonnegative, descending = 0.0, float('inf'), True, True
        error_max, error_square, finite_count, beyond, neginf_count = 0.0, 0.0, 0, 0, 0
        for start in range(0, N, CHUNK):
            stop = min(start+CHUNK, N)
            f, s, saved = fishers[start:stop], sv[start:stop], scores[start:stop]
            symmetry_max = max(symmetry_max, float(np.abs(f-f.transpose(0, 2, 1)).max()))
            diagonal_min = min(diagonal_min, float(np.diagonal(f, axis1=1, axis2=2).min()))
            nonnegative = nonnegative and bool((s >= 0).all())
            descending = descending and bool((s[:, :-1] >= s[:, 1:]).all())
            computed = torch.log(tensors['singular_values'][start:stop]).sum(dim=1).numpy()
            need(np.array_equal(np.isneginf(computed), np.isneginf(saved)), 'CPU logsum -inf positions differ')
            finite = np.isfinite(saved)
            need(np.isfinite(computed[finite]).all(), 'CPU logsum finite positions differ')
            difference = np.abs(computed[finite].astype(np.float64)-saved[finite].astype(np.float64))
            error_max = max(error_max, float(difference.max()) if difference.size else 0.0)
            error_square += float(np.dot(difference, difference))
            finite_count += int(finite.sum())
            beyond += int((difference > ATOL+RTOL*np.abs(saved[finite].astype(np.float64))).sum())
            neginf_count += int(np.isneginf(saved).sum())
        checkpoint('fisher_sv_structure', dict(all_blocks=N, symmetry_max_abs=symmetry_max,
            diagonal_min=diagonal_min, singular_values_nonnegative=nonnegative, singular_values_descending=descending,
            positive_semidefinite_verified=False, fisher_to_svd_relation_verified=False))
        need(symmetry_max == 0.0 and diagonal_min >= 0 and nonnegative and descending, 'Fisher/SV structure')
        checkpoint('cpu_float32_logscore', dict(max_abs_difference=error_max,
            rms_difference=(error_square/finite_count)**.5 if finite_count else None,
            finite_count=finite_count, negative_infinity_count=neginf_count,
            negative_infinity_pattern_exact=True, beyond_tolerance_count=beyond, atol=ATOL, rtol=RTOL))
        need(beyond == 0, 'CPU float32 logsum exceeds prelocked tolerance')
        threshold_index = int(.5*(N-1))
        threshold = np.sort(scores)[threshold_index]
        recomputed_mask = scores <= threshold
        indices = np.flatnonzero(~recomputed_mask)
        need(np.array_equal(recomputed_mask, tensors['prune_mask'].numpy()) and
             np.array_equal(indices, tensors['survivor_indices'].numpy()), 'Full mask/indices differ from original author rule')
        pruning = result['pruning']
        threshold_value = float(threshold) if np.isfinite(threshold) else '-inf'
        need(pruning['percent'] == .5 and pruning['threshold'] == threshold_value and
             pruning['kept_count'] == len(indices) == K and pruning['removed_count'] == N-K and
             pruning['at_threshold_count'] == int((scores == threshold).sum()) and
             pruning['natural_ties_unchanged'] is True and pruning['exact_half_count'] is True, 'Recorded natural pruning facts')
        checkpoint('full_mask_and_indices', dict(sort_index=threshold_index, threshold=threshold_value,
            prune_count=int(recomputed_mask.sum()), kept_count=len(indices), exact=True, ties=int((scores == threshold).sum())))
        teacher = PlyData.read(str(TEACHER), mmap='r')['vertex'].data
        exported = PlyData.read(str(ORIGINAL/'point_cloud.ply'), mmap='r')['vertex'].data
        names = ('x', 'y', 'z', 'nx', 'ny', 'nz', *(f'f_dc_{i}' for i in range(3)),
            *(f'f_rest_{i}' for i in range(45)), 'opacity', *(f'scale_{i}' for i in range(3)), *(f'rot_{i}' for i in range(4)))
        need(teacher.dtype.names == exported.dtype.names == names and len(teacher) == N and len(exported) == K,
             'Original62 PLY schema/count')
        properties = {}
        for name in names:
            source, actual = np.ascontiguousarray(teacher[name][indices]), np.ascontiguousarray(exported[name])
            need(source.dtype == actual.dtype == np.dtype('<f4') and source.tobytes() == actual.tobytes(), 'PLY property differs: '+name)
            properties[name] = dict(dtype=str(actual.dtype), sha256=hashlib.sha256(actual.tobytes()).hexdigest(), exact=True)
            need(properties[name] == result['ply']['properties'][name], 'Recorded PLY property hash differs: '+name)
        need(identity(ORIGINAL/'point_cloud.ply') == dict(sha256=result['ply']['sha256'], bytes=result['ply']['bytes']), 'Recorded PLY hash/size')
        checkpoint('all62_ply_properties', dict(point_count=K, properties=properties, byte_exact=True))
        checkpoint('cpu_teacher_raw_fields', raw_identity(teacher, canonical['raw_fields'], np))
        checkpoint('cpu_export_raw_fields', raw_identity(exported, result['raw_fields_exported_subset'], np))
        checkpoint('inputs_after', input_identities(inputs))
        checkpoint('original_artifacts_after', original_identities())
        need(audit['inputs_before'] == audit['inputs_after'] and
             audit['original_artifacts_before'] == audit['original_artifacts_after'], 'Audit changed protected inputs/artifacts')
        check_prelaunch()
        need(not torch.cuda.is_initialized() and not any(name in sys.modules for name in
             ('scene', 'gaussian_renderer', 'fisher_renderer', 'run_pup_frozen_score')), 'Forbidden scientific backend imported')
        audit['cuda_initialized'] = False
        audit['status'] = 'AUDIT_PASS_SCOPE_LIMITED'
    except BaseException as error:
        audit['status'] = 'AUDIT_FAIL_OR_INCOMPLETE'
        audit['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    save(OUTPUT/'AUDIT_RESULT.json', audit)
    return 0 if audit['status'] == 'AUDIT_PASS_SCOPE_LIMITED' else 1


def tree_rss(root):
    entries = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                fields = dict(line.split(':', 1) for line in (p/'status').read_text().splitlines() if ':' in line)
                entries[int(p.name)] = (int(fields['PPid']), int(fields.get('VmRSS', '0 kB').split()[0])*1024)
            except (OSError, KeyError, ValueError):
                pass
    selected = {root}
    while True:
        added = {pid for pid, (parent, _) in entries.items() if parent in selected}-selected
        if not added:
            break
        selected.update(added)
    return sum(entries.get(pid, (0, 0))[1] for pid in selected)


def group_members(pgid):
    result = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                fields = (p/'stat').read_text().rpartition(') ')[2].split()
                if fields[0] != 'Z' and int(fields[2]) == pgid:
                    result.append(int(p.name))
            except (OSError, IndexError, ValueError):
                pass
    return sorted(result)


def supervise():
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned WSL Python required')
    check_prelaunch()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Launcher must hide CUDA')
    OUTPUT.mkdir()
    started, child, closing = time.monotonic(), None, False
    record = dict(status='AUDIT_INCONCLUSIVE', execution_attempt=1, automatic_retries=0,
        wall_limit_seconds=WALL, host_RSS_limit_bytes=RSS, peak_sampled_combined_host_RSS_bytes=0)
    save(OUTPUT/'ATTEMPT_STARTED.json', dict(execution_attempt=1, pid=os.getpid(), unix=time.time()))
    def interrupted(number, _frame):
        if not closing:
            raise TimeoutError('Audit supervisor signal '+str(number))
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, interrupted)
    signal.setitimer(signal.ITIMER_REAL, WALL)
    try:
        with (OUTPUT/'WORKER.log').open('x', encoding='utf8') as log:
            child = subprocess.Popen([str(PYTHON), '-B', '-u', str(HERE/'audit.py'), '--worker'],
                cwd=HERE, env=dict(os.environ, PUP_AUDIT_WORKER='1'), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            record['worker_pid'] = record['worker_pgid'] = child.pid
            while child.poll() is None:
                used = tree_rss(os.getpid())
                record['peak_sampled_combined_host_RSS_bytes'] = max(record['peak_sampled_combined_host_RSS_bytes'], used)
                need(used <= RSS, 'Combined sampled audit RSS exceeded8GiB; no retry')
                time.sleep(.05)
            record['worker_exit_code'] = child.returncode
            need(child.returncode == 0 and read(OUTPUT/'AUDIT_RESULT.json')['status'] == 'AUDIT_PASS_SCOPE_LIMITED', 'Audit worker failed; partial result preserved')
            need(not group_members(child.pid), 'Audit worker left residual process group')
            record['status'] = 'AUDIT_PASS_SCOPE_LIMITED'
    except BaseException as error:
        record['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        closing = True
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(number, signal.SIG_IGN)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=1)
            record['worker_exit_code'] = child.returncode
            record['remaining_live_group_pids'] = group_members(child.pid)
            if record['remaining_live_group_pids']:
                record['status'] = 'AUDIT_INCONCLUSIVE'
        record['wall_seconds'] = time.monotonic()-started
        record['supervisor_torch_imported'] = 'torch' in sys.modules
        record['within_budget'] = record['wall_seconds'] <= WALL and record['peak_sampled_combined_host_RSS_bytes'] <= RSS
        if not record['within_budget']:
            record['status'] = 'AUDIT_INCONCLUSIVE'
        save(OUTPUT/'SUPERVISOR.json', record)
        print(json.dumps(record), flush=True)
    return 0 if record['status'] == 'AUDIT_PASS_SCOPE_LIMITED' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    need(args.run != args.worker, 'Choose exactly one mode; root must authorize --run')
    sys.exit(worker() if args.worker else supervise())
