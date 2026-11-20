#!/usr/bin/env python3
"""Independent CPU artifact audit. Import/help are stdlib only; no producer imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import tempfile
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
EVALUATION = HERE.parent
PROJECT = HERE.parents[4]
RUN = EVALUATION/'actualrun-v1'
BUILD = EVALUATION.parent/'build-v1'
OLD = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
OUTPUT = HERE/'run-v1'
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
TEACHER = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply')
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
MODEL_SHA = 'b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0'
COUNT = 2050770
COMPLETE = 'FROZEN_SPEEDY_CONTROL_EVALUATED_AUDIT_PENDING'
PASS = 'PASS_SAVED_ARTIFACT_CONSISTENCY'
RSS_LIMIT = 8*1024**3
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
FIXED = ('000016.jpg', '000017.jpg', '000105.jpg')
METRICS = ('gt_mse', 'gt_psnr', 'gt_ssim', 'teacher_mse', 'teacher_max_abs_rgb')
TOLERANCES = dict(mse=1e-12, psnr=1e-9, ssim=1e-10, max_abs_rgb=1e-12, pixels=0, timing_ms=1e-9)
GATES = dict(mean_GT_PSNR_loss_dB_max=.1, mean_GT_SSIM_loss_max=.001,
    own_GT_tail4_MSE_ratio_max=1.05, worst_test_GT_PSNR_loss_dB_max=.3,
    mean_perview_median_wall_reduction_min=.10, faster_repeat_blocks_min=9)
COMMITS = dict(main='34c45c6d9b8bd6110231864f2f358b6d3abbf73d',
    rasterizer='b38abb6da7a7217999f8c5ec0e9219284464ffe0', glm='5c46b9c07008ae65cb81ab79cd677ecc1934b903')
PRODUCER_PINS = {'evaluate.py': '2fa4df61730af1c23960cb222e61f1ae60a18ce9a412344f8f019f5f3a1a1669',
    'CONTRACT.md': 'b793aac54aa8a84737188a1dc847fbbfc83a49406cb15d3270d0f7ceef213a36',
    'ASSUMPTIONS.md': 'd0513637ab03c657ab092447210ecd1e24e706d291eafb52f3f28f05c3457113',
    'READONLY_PINS.json': '8016c62f879faaa0876ae5b215bec9369c875f4f64522a09520a1fbba01a81fe',
    'run-evaluation.sh': '9d86fe2aa60047b127394f6040ba03a947514a73a4477584047e83fa4cb8d86f',
    'run-evaluation.ps1': 'a8c27cc2807e3d59b4110614e8f1d9e20264d3f3406b697fad509a5a69a78c85'}
PLY_FIELDS = (['x', 'y', 'z', 'nx', 'ny', 'nz']+[f'f_dc_{i}' for i in range(3)]+
    [f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)])
PIN_FILES = dict(result=RUN/'RESULTS.json', supervisor=RUN/'SUPERVISOR.json',
    manifest=RUN/'RAW_MANIFEST.json', combined_manifest=RUN/'COMBINED_RAW_MANIFEST.json',
    config=RUN/'CONFIG.json', external=EVALUATION/'EXTERNAL_RUN.json')
PROOF_PATHS = {'build-result': BUILD/'actualrun-v1/BUILD_RESULT.json',
    'build-supervisor': BUILD/'actualrun-v1/SUPERVISOR.json', 'build-external': BUILD/'actualrun-v1/HOST_EXTERNAL_RUN.json',
    'witness-result': BUILD/'witness-v1/RESULTS.json', 'witness-supervisor': BUILD/'witness-v1/SUPERVISOR.json',
    'witness-external': BUILD/'witness-v1/HOST_EXTERNAL_RUN.json'}
LAYOUTS = {'R': ('float64', (3, 3)), 'T': ('float64', (3,)), 'world_view_transform': ('float32', (4, 4)),
    'projection_matrix': ('float32', (4, 4)), 'full_proj_transform': ('float32', (4, 4)), 'camera_center': ('float32', (3,))}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_sha(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
    result.update(value.tobytes())
    return result.hexdigest()


def pixels(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        need(data.files == ['rgb'], 'NPZ exactly rgb: '+str(path))
        value = data['rgb']
    need(value.dtype == np.float32 and value.shape == (3, 546, 979) and np.isfinite(value).all()
         and value.min() >= 0 and value.max() <= 1, 'Invalid raw RGB: '+str(path))
    return value


def quantize(value):
    import numpy as np
    return np.rint(value.transpose(1, 2, 0)*np.float32(255)).astype(np.uint8)


def metric_pair(value, truth, teacher, mean_truth, variance_truth):
    import numpy as np
    from scipy.ndimage import convolve1d
    weights = np.exp(-(np.arange(-5, 6, dtype=np.float64)**2)/(2*1.5**2))
    weights /= weights.sum(dtype=np.float64)
    def smooth(image):
        return convolve1d(convolve1d(image, weights, axis=1, mode='constant', cval=0),
                          weights, axis=2, mode='constant', cval=0)
    if value is None:
        mean = smooth(truth)
        return mean, smooth(truth*truth)-mean*mean
    value = value.astype(np.float64)
    difference = value-truth
    mse = float(np.mean(difference*difference, dtype=np.float64))
    mean = smooth(value)
    variance = smooth(value*value)-mean*mean
    covariance = smooth(value*truth)-mean*mean_truth
    luminance = (2*mean*mean_truth+.01**2)/(mean*mean+mean_truth*mean_truth+.01**2)
    structure = (2*covariance+.03**2)/(variance+variance_truth+.03**2)
    fidelity = value-teacher
    return dict(gt_mse=mse, gt_psnr=None if mse == 0 else -10*math.log10(mse),
        zero_MSE_infinite_PSNR=(mse == 0), gt_ssim=float(np.mean(luminance*structure, dtype=np.float64)),
        teacher_mse=float(np.mean(fidelity*fidelity, dtype=np.float64)), teacher_max_abs_rgb=float(np.max(np.abs(fidelity))))


def summarize_quality(rows, test_views, exact):
    by = {(r['arm'], r['view']): r for r in rows}
    arms = {}
    for arm in ('teacher', 'speedy'):
        selected = [by[arm, view] for view in test_views]
        worst = sorted(selected, key=lambda r: (-r['gt_mse'], r['view']))[:4]
        arms[arm] = dict(test_count=32, mean_gt_mse=math.fsum(r['gt_mse'] for r in selected)/32,
            mean_gt_psnr=None if any(r['gt_psnr'] is None for r in selected) else math.fsum(r['gt_psnr'] for r in selected)/32,
            mean_gt_ssim=math.fsum(r['gt_ssim'] for r in selected)/32, own_GT_tail4=worst,
            own_GT_tail4_mean_MSE=math.fsum(r['gt_mse'] for r in worst)/4)
    deltas = []
    for view in test_views+['000016.jpg']:
        a, b = by['teacher', view], by['speedy', view]
        deltas.append(dict(view=view, teacher_minus_speedy_gt_psnr=None if a['gt_psnr'] is None or b['gt_psnr'] is None else a['gt_psnr']-b['gt_psnr'],
            teacher_minus_speedy_gt_ssim=a['gt_ssim']-b['gt_ssim'], speedy_minus_teacher_gt_mse=b['gt_mse']-a['gt_mse']))
    a, b = arms['teacher'], arms['speedy']
    psnr = None if a['mean_gt_psnr'] is None or b['mean_gt_psnr'] is None else a['mean_gt_psnr']-b['mean_gt_psnr']
    ssim = a['mean_gt_ssim']-b['mean_gt_ssim']
    ratio = b['own_GT_tail4_mean_MSE']/a['own_GT_tail4_mean_MSE'] if a['own_GT_tail4_mean_MSE'] > 0 else None
    losses = [d['teacher_minus_speedy_gt_psnr'] for d in deltas[:32]]
    worst = None if any(x is None for x in losses) else max(losses)
    gates = dict(mean_GT_PSNR=psnr is not None and psnr <= .1, mean_GT_SSIM=ssim <= .001,
        own_GT_tail4_MSE=ratio is not None and ratio <= 1.05, worst_test_GT_PSNR=worst is not None and worst <= .3)
    return dict(arms=arms, per_view_signed_differences=deltas, mean_GT_PSNR_loss_dB=psnr,
        mean_GT_SSIM_loss=ssim, own_GT_tail4_MSE_ratio=ratio, worst_test_GT_PSNR_loss_dB=worst,
        quality_gate_conditions=gates, quality_joint_pass=all(gates.values()),
        worst_SSIM_loss_ranking32=sorted(deltas[:32], key=lambda d: (-d['teacher_minus_speedy_gt_ssim'], d['view'])),
        exact_RGB_match_per_view=exact, exact_RGB_match_all33=all(r['exact_RGB_match'] for r in exact))


def plan(views):
    return [dict(timed_forward_index=repeat*64+position*32+rank, repeat=repeat, arm=arm,
        arm_position=position, camera_rank=rank, view=view)
        for repeat in range(10) for position, arm in enumerate(('native', 'speedy') if repeat % 2 == 0 else ('speedy', 'native'))
        for rank, view in enumerate(views)]


def parse_timing(path):
    with Path(path).open(newline='', encoding='utf8') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == ['timed_forward_index', 'repeat', 'arm', 'arm_position', 'camera_rank', 'view', 'wall_ms', 'cuda_ms'], 'Exact timing CSV columns')
        rows = []
        for item in reader:
            rows.append({k: int(v) if k in ('timed_forward_index', 'repeat', 'arm_position', 'camera_rank') else
                float(v) if k in ('wall_ms', 'cuda_ms') else v for k, v in item.items()})
    return rows


def summarize_timing(rows, views):
    expected = plan(views)
    need(len(rows) == 640 and all(all(row[k] == value for k, value in p.items()) for row, p in zip(rows, expected)), 'Exact ordered640 plan')
    need(all(math.isfinite(row[k]) and row[k] > 0 for row in rows for k in ('wall_ms', 'cuda_ms')), 'All timings positive finite')
    medians, blocks, means = [], [], {}
    for arm in ('native', 'speedy'):
        for view in views:
            selected = [r for r in rows if r['arm'] == arm and r['view'] == view]
            need(len(selected) == 10, 'Ten repeats per arm/view')
            medians.append(dict(arm=arm, view=view, count=10, wall_median_ms=statistics.median(r['wall_ms'] for r in selected),
                cuda_median_ms=statistics.median(r['cuda_ms'] for r in selected)))
        for repeat in range(10):
            selected = [r for r in rows if r['arm'] == arm and r['repeat'] == repeat]
            need(len(selected) == 32, 'Full32 blocks')
            blocks.append(dict(arm=arm, repeat=repeat, frames=32, wall_mean_ms=math.fsum(r['wall_ms'] for r in selected)/32,
                cuda_mean_ms=math.fsum(r['cuda_ms'] for r in selected)/32))
        means[arm] = {kind: math.fsum(r[kind+'_median_ms'] for r in medians if r['arm'] == arm)/32 for kind in ('wall', 'cuda')}
    block_by = {(r['arm'], r['repeat']): r for r in blocks}
    faster = sum(block_by['speedy', i]['wall_mean_ms'] < block_by['native', i]['wall_mean_ms'] for i in range(10))
    reduction = 1-means['speedy']['wall']/means['native']['wall']
    return dict(mean_32_perview_10repeat_median_ms=means, per_view=medians, paired_repeat_blocks=blocks,
        synchronized_wall_reduction_fraction=reduction, faster_repeat_blocks=faster,
        timing_joint_pass=reduction >= .10 and faster >= 9, no_outlier_removal=True, official_internal_kernel_times_used=False)


def ply_fields(path, expected_count=COUNT):
    import numpy as np
    with Path(path).open('rb') as stream:
        need(stream.readline() == b'ply\n', 'LF PLY magic')
        names, count, fmt = [], None, False
        while True:
            raw = stream.readline(65537)
            need(raw and len(raw) <= 65536 and stream.tell() <= 65536, 'Bounded PLY header')
            words = raw.decode('ascii').strip().split()
            if words == ['end_header']:
                offset = stream.tell()
                break
            if not words or words[0] in ('comment', 'obj_info'):
                continue
            if words == ['format', 'binary_little_endian', '1.0'] and not fmt:
                fmt = True
            elif len(words) == 3 and words[:2] == ['element', 'vertex'] and count is None:
                count = int(words[2])
            elif len(words) == 3 and words[0] == 'property' and words[1] in ('float', 'float32') and count is not None:
                names.append(words[2])
            else:
                raise RuntimeError('Unsupported PLY header: '+repr(words))
    need(fmt and count == expected_count and names == PLY_FIELDS and Path(path).stat().st_size == offset+count*248, 'Exact62 FP32 PLY schema/count/bytes')
    value = np.memmap(path, dtype='<f4', mode='r', offset=offset, shape=(count, 62))
    hashes = {name: hashlib.sha256() for name in FIELDS}
    for start in range(0, count, 65536):
        block = value[start:start+65536]
        need(np.isfinite(block).all(), 'All62 teacher attributes finite')
        columns = dict(_xyz=block[:, :3], _features_dc=block[:, 6:9].reshape(-1, 1, 3),
            _features_rest=block[:, 9:54].reshape(-1, 3, 15).transpose(0, 2, 1),
            _scaling=block[:, 55:58], _rotation=block[:, 58:62], _opacity=block[:, 54:55])
        for name in FIELDS:
            hashes[name].update(np.ascontiguousarray(columns[name]).tobytes())
    del value
    shapes = dict(_xyz=[count, 3], _features_dc=[count, 1, 3], _features_rest=[count, 15, 3],
        _scaling=[count, 3], _rotation=[count, 4], _opacity=[count, 1])
    return {name: dict(dtype='float32', shape=shapes[name], sha256=hashes[name].hexdigest()) for name in FIELDS}


def tree_rss():
    entries = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                status = dict(line.split(':', 1) for line in (path/'status').read_text().splitlines() if ':' in line)
                entries[int(path.name)] = (int(status['PPid']), int(status.get('VmRSS', '0 kB').split()[0])*1024)
            except (OSError, ValueError, KeyError):
                pass
    family = {os.getpid()}
    while True:
        expanded = family | {pid for pid, (parent, _) in entries.items() if parent in family}
        if expanded == family:
            return sum(entries.get(pid, (0, 0))[1] for pid in family)
        family = expanded


def audit(args):
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned WSL Python required')
    need(not OUTPUT.exists(), 'One attempt only; preserve existing run-v1')
    pins = {}
    for name, expected_path in PIN_FILES.items():
        path, digest = getattr(args, name), getattr(args, name+'_sha256')
        need(path is not None and digest is not None and path.resolve() == expected_path.resolve()
            and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Root actual path/SHA required: '+name)
        pins[str(expected_path)] = digest
    OUTPUT.mkdir()
    started = time.monotonic()
    checks, hashes, rows, comparisons = [], {}, [], []
    report = dict(schema='speedy-frozen-control/independent-audit-v1', status='INCONCLUSIVE',
        technical_tolerances=TOLERANCES, SSIM_definition='CPU_float64_Gaussian11_sigma1.5_constant0_population_CHWmean_C1.01sq_C2.03sq',
        old_formal_Nano_SSIM_audit='FAIL_UNCHANGED', rerenders=0, backward_calls=0, optimizer_updates=0,
        unverified_scope=['GPU forward history and timing boundaries are checked from saved declarations/RPC only; no independent GPU execution trace.',
            'No innovation acceptance or pruning+Speedy frontier claim. Root must measure complete audit wall <=100s and main+audit <=340s.'])
    supervisor = dict(schema='speedy-frozen-control/independent-supervisor-v1', status='INCONCLUSIVE', attempts=1,
        automatic_retry=False, internal_limit_seconds=94, outer_timeout_TERM_seconds=98, outer_timeout_KILL_after_seconds=1,
        external_limit_seconds=100, combined_limit_seconds=340, host_RSS_limit_bytes=RSS_LIMIT,
        peak_sampled_process_tree_RSS_bytes=0, sample_interval_seconds=.05, CPU_only=True, view_threads=4,
        child_processes_spawned=0, timeout=False, memory_stop=False)
    stop_monitor = threading.Event()
    monitor_errors = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    def protect(path, expected=None, size=None):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = dict(before=sha(path), bytes=Path(path).stat().st_size)
        need(expected is None or hashes[path]['before'] == expected, 'Input SHA mismatch: '+path)
        need(size is None or hashes[path]['bytes'] == size, 'Input byte count mismatch: '+path)
    def same(name, got, expected, quantity=None):
        if isinstance(expected, dict):
            check(name+':keys', isinstance(got, dict) and set(got) == set(expected))
            if isinstance(got, dict):
                for key in expected:
                    if key in got:
                        same(name+'/'+key, got[key], expected[key], quantity)
            return
        if isinstance(expected, list):
            check(name+':length', isinstance(got, list) and len(got) == len(expected))
            if isinstance(got, list):
                for i, (a, b) in enumerate(zip(got, expected)):
                    same(name+'/'+str(i), a, b, quantity)
            return
        if isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
            check(name, got == expected and (not isinstance(expected, bool) or isinstance(got, bool)))
            return
        if isinstance(expected, float):
            label = name.lower()
            kind = quantity or ('ssim' if 'ssim' in label else 'psnr' if 'psnr' in label else
                'max_abs_rgb' if 'max_abs' in label else 'timing_ms' if 'timing' in label else 'mse')
            allowed = TOLERANCES[kind]
            error = abs(float(got)-expected) if isinstance(got, (int, float)) else None
            check(name, error is not None and math.isfinite(float(got)) and math.isfinite(expected) and error <= allowed)
            comparisons.append(dict(name=name, absolute_error=error, tolerance=allowed))
            return
        check(name, got == expected)
    def interrupted(number, _frame):
        raise TimeoutError('Audit interrupted '+str(number)+(' '+monitor_errors[0] if monitor_errors else ''))
    def monitor():
        while not stop_monitor.is_set():
            try:
                rss = tree_rss()
                supervisor['peak_sampled_process_tree_RSS_bytes'] = max(supervisor['peak_sampled_process_tree_RSS_bytes'], rss)
                if rss > RSS_LIMIT:
                    supervisor['memory_stop'] = True
                    monitor_errors.append('RSS exceeds8GiB')
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
            except OSError as error:
                monitor_errors.append('RSS monitor: '+repr(error))
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            stop_monitor.wait(.05)
    for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(number, interrupted)
    signal.alarm(94)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    save(OUTPUT/'REQUEST.json', dict(pins=pins, attempts=1, started_unix=time.time(),
        harness_pins={name: sha(HERE/name) for name in ('audit.py', 'CONTRACT.md', 'ASSUMPTIONS.md', 'run-audit.sh')}))
    save(OUTPUT/'AUDIT.json', report)
    save(OUTPUT/'SUPERVISOR.json', supervisor)
    try:
        for path, digest in pins.items():
            protect(path, digest)
        for name, digest in read(OUTPUT/'REQUEST.json')['harness_pins'].items():
            protect(HERE/name, digest)
        cfg, result, main_supervisor, external = read(RUN/'CONFIG.json'), read(RUN/'RESULTS.json'), read(RUN/'SUPERVISOR.json'), read(EVALUATION/'EXTERNAL_RUN.json')
        need(isinstance(cfg['protected_files'], dict) and cfg['protected_files'], 'Full protected table required')
        for path, digest in cfg['protected_files'].items():
            protect(path, digest)
        for name, digest in PRODUCER_PINS.items():
            need(cfg['protected_files'][str(EVALUATION/name)] == digest, 'Reviewed producer source pin: '+name)
            protect(EVALUATION/name, digest)
        # Bind read-only declarations before any scientific array is opened.
        for name in ('CAMERAS.json', 'METRICS.csv', 'TIMING.csv', 'PANEL_MANIFEST.json', 'RPC.jsonl',
            'native_STATE.json', 'speedy_STATE.json', 'native_TIMING.partial.csv', 'speedy_TIMING.partial.csv',
            'native_RAW_MANIFEST.json', 'speedy_RAW_MANIFEST.json'):
            protect(RUN/name)
        for path in (RUN/'rpc').iterdir():
            need(path.is_file(), 'RPC files only')
            protect(path)
        check('schemas', cfg['schema'] == 'speedy-frozen-control/config-v1' and result['schema'] == 'speedy-frozen-control/result-v1' and main_supervisor['schema'] == 'speedy-frozen-control/supervisor-v1')
        check('terminal', result['status'] == main_supervisor['status'] == COMPLETE and result['integrity_complete'] is True)
        check('protected_declarations', result['protected_inputs_unchanged'] is True and result['protected_file_count'] == len(cfg['protected_files']))
        check('old_FAIL_unchanged', result['old_formal_Nano_SSIM_audit'] == cfg['old_formal_Nano_SSIM_audit'] == 'FAIL_UNCHANGED')
        check('no_novelty_training', result['novelty_claim'] is False and result['training_authorized'] is False and cfg['novelty_claim'] is False and cfg['pruning_speed_frontier_claim'] is False)
        same('gate_thresholds', result['gate_thresholds'], GATES)
        same('config/gates', cfg['gates'], GATES)
        check('true_main_external', external['schema'] == 'speedy-frozen-control/host-external-v1' and external['exit_code'] == 0 and external['attempts'] == 1
            and external['automatic_retry'] is False and external['within_limit'] is True and external['hard_limit_seconds'] == 240
            and 0 < external['actual_host_wall_seconds'] <= 240 and external['shared_main_plus_audit_limit_seconds'] == 340
            and external['independent_audit_reserved_seconds'] == 100)
        check('main_resource_budgets', cfg['main_external_limit_seconds'] == 240 and cfg['audit_reserved_seconds'] == 100 and cfg['combined_external_limit_seconds'] == 340
            and cfg['internal_deadline_seconds'] == 225 and cfg['host_RSS_limit_bytes'] == 10*1024**3
            and cfg['each_worker_GPU_allocated_limit_bytes'] == 3*1024**3 and cfg['sum_worker_peak_GPU_limit_bytes'] == 6*1024**3
            and main_supervisor['attempts'] == 1 and main_supervisor['automatic_retry'] is False and main_supervisor['controller_torch_imported'] is False
            and main_supervisor['shared_internal_deadline_seconds'] == 225 and main_supervisor['outer_timeout_TERM_seconds'] == 235
            and main_supervisor['outer_timeout_KILL_after_seconds'] == 3 and main_supervisor['peak_sampled_tree_RSS_bytes'] <= 10*1024**3
            and 0 < main_supervisor['controller_elapsed_seconds'] <= 225 and main_supervisor['all_workers_reaped'] is True
            and main_supervisor['monitor_errors'] == [] and main_supervisor['worker_exit_codes_before_cleanup'] == {'native': 0, 'speedy': 0}
            and main_supervisor['worker_exit_codes_after_cleanup'] == {'native': 0, 'speedy': 0})
        # Actual build / witness proofs, independently bound to their immutable sources.
        need(set(cfg['actual_proofs']) == set(PROOF_PATHS), 'Six build/witness proofs')
        for key, path in PROOF_PATHS.items():
            item = cfg['actual_proofs'][key]
            need(Path(item['path']).resolve() == path.resolve() and cfg['protected_files'][str(path)] == item['sha256'], 'Actual proof binding: '+key)
            protect(path, item['sha256'])
            check('external/proof:'+key, external[key.replace('-', '_')+'_sha256'] == item['sha256'])
        build, witness = read(PROOF_PATHS['build-result']), read(PROOF_PATHS['witness-result'])
        check('actual_build', build['schema'] == 'speedy-private-extension/build-v1' and build['status'] == 'PRIVATE_BUILD_IMPORT_PASS_ONLY'
            and build['tracked_sources_unchanged'] and build['protected_unchanged'] and build['source_sha256_before'] == build['source_sha256_after']
            and [s['name'] for s in build['stages']] == ['hardware', 'nvcc', 'compiler', 'author_build', 'private_import_only']
            and all(s['exit_code'] == 0 and 0 < s['elapsed_seconds'] <= 180 for s in build['stages']))
        check('actual_3F_witness', witness['schema'] == 'speedy-private-extension/witness-v1' and witness['status'] == 'SPEEDY_SYNTHETIC_FORWARD_WITNESS_PASS'
            and witness['forward_calls'] == 3 and all(witness[k] == 0 for k in ('backward_calls', 'optimizer_updates', 'model_loads', 'scene_instances'))
            and witness['synthetic_inputs_unchanged'] and witness['tracked_sources_unchanged'] and witness['protected_unchanged']
            and witness['numerical_verified_local_case'] and witness['c_call_counts'] == dict(forward_attempted=3, forward_completed=3, backward_attempted=0, mark_visible_attempted=0))
        check('witness_numerical_declarations', 0 <= witness['repeat_max_abs'] <= witness['repeat_tolerance'] == 1e-6
            and witness['sh_response_max_abs'] >= witness['sh_response_minimum'] == 1e-6 and 0 < witness['elapsed_seconds'] <= 60)
        for kind, seconds, rss in (('build', 180, 8*1024**3), ('witness', 60, 4*1024**3)):
            s, e = read(PROOF_PATHS[kind+'-supervisor']), read(PROOF_PATHS[kind+'-external'])
            check('actual_supervisor:'+kind, s['schema'] == 'speedy-private-extension/supervisor-v1' and s['kind'] == kind and s['status'] == 'SUPERVISOR_PASS'
                and s['worker_exit_code'] == 0 and s['attempts'] == 1 and s['automatic_retry'] is False and 0 < s['elapsed_seconds'] <= seconds
                and s['peak_sampled_process_tree_rss_bytes'] <= rss and not s['cleanup']['live_after_cleanup'])
            check('actual_external:'+kind, e['schema'] == 'speedy-private-extension/host-external-v1' and e['exit_code'] == 0 and e['attempts'] == 1
                and e['automatic_retry'] is False and e['hard_limit_seconds'] == seconds and e['within_limit'] is True
                and 0 < e['actual_host_wall_seconds'] <= seconds and e['preparation_manifest_sha256'] == build['preparation_manifest_sha256'])
        for path, digest in witness['build_records'].items():
            need(path in cfg['protected_files'] and cfg['protected_files'][path] == digest, 'Witness actual build records')
        protect(BUILD/'PREPARATION_SHA256.json', build['preparation_manifest_sha256'])
        for item in read(BUILD/'PREPARATION_SHA256.json')['files']:
            protect(BUILD/item['name'], item['sha256'], item['bytes'])
        sources = read(BUILD/'SOURCE_MANIFEST.json')['sources']
        check('official_source_manifest', set(sources) == set(COMMITS) and canonical(sources) == build['source_sha256_before'] == witness['source_sha256_before'] == witness['source_sha256_after'])
        for kind, source in sources.items():
            check('official_commit:'+kind, source['head'] == COMMITS[kind])
            for name, item in source['files'].items():
                if 'sha256' in item:
                    path = str(Path(source['path'])/name)
                    need(cfg['protected_files'][path] == item['sha256'], 'Source must be protected: '+path)
                    protect(path, item['sha256'], item['bytes'])
        for path, digest in build['protected_after'].items():
            need(cfg['protected_files'][path] == digest, 'Build protected provenance')
        for key in ('wrapper', 'extension'):
            check('private_actual:'+key, cfg['private_build'][key] == build[key] and Path(build[key]['path']).parent == Path(build['private_lib'])/'diff_gaussian_rasterization')
            protect(build[key]['path'], build[key]['sha256'])
        check('private_lib_identity', cfg['private_build']['private_lib'] == build['private_lib']
            and Path(build['private_lib']).parent == Path('/home/administrator/projects/speedy-rasterizer-b38abb6d-private-build-v1')
            and Path(build['wrapper']['path']).name == '__init__.py'
            and Path(build['extension']['path']).name == '_C.cpython-310-x86_64-linux-gnu.so')
        check('official_render_path', cfg['official_render_path'] == str(Path(sources['main']['path'])/'gaussian_renderer/__init__.py'))
        old_cfg, old_result, old_cameras = read(OLD/'CONFIG.json'), read(OLD/'RESULTS.json'), read(OLD/'CAMERAS.json')
        for rel, digest in read(EVALUATION/'READONLY_PINS.json').items():
            need(cfg['protected_files'][str(PROJECT/rel)] == digest, 'Reviewed readonly pin missing')
        views, test_views = old_cfg['test_views']+['000016.jpg'], old_cfg['test_views']
        check('same_views', cfg['views'] == views and cfg['test_views'] == test_views and len(views) == len(set(views)) == 33
            and len(test_views) == 32 and test_views == sorted(test_views) and old_result['integrity_complete'] is True
            and old_result['decision'] == 'FROZEN_BASELINE_EVALUATED')
        check('same_renderer', cfg['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False, exposure=False, scaling_modifier=1.,
            separate_sh=False, convert_SHs_python=False, compute_cov3D_python=False, native_final_clamp=True))
        check('exact_camera_records', cfg['camera_records'] == old_cameras == read(RUN/'CAMERAS.json') and [r['image_name'] for r in old_cameras] == views)
        import numpy as np
        from PIL import Image, ImageDraw
        camera_proof = []
        for camera in old_cameras:
            check('camera_dimensions:'+camera['image_name'], camera['image_width'] == 979 and camera['image_height'] == 546)
            check('camera_scalars:'+camera['image_name'], all(math.isfinite(float(camera[k])) for k in ('uid', 'FoVx', 'FoVy', 'znear', 'zfar')))
            details = {}
            for name, (dtype, shape) in LAYOUTS.items():
                a = np.asarray(camera[name]['values'], dtype=dtype)
                check('camera_array:'+camera['image_name']+'/'+name, a.dtype == np.dtype(dtype) and a.shape == shape and np.isfinite(a).all() and array_sha(a) == camera[name]['sha256'])
                details[name] = dict(dtype=dtype, shape=list(shape), array_sha256=array_sha(a))
            camera_proof.append(dict(image_name=camera['image_name'], arrays=details, full_record_exact=True))
        check('teacher_fixed', cfg['teacher'] == dict(path=str(TEACHER), sha256=TEACHER_SHA, points=COUNT, native_model_sha256=MODEL_SHA))
        protect(TEACHER, TEACHER_SHA)
        reconstructed = ply_fields(TEACHER)
        report['teacher_native_fields_reconstructed'] = reconstructed
        states = {arm: read(RUN/(arm+'_STATE.json')) for arm in ('native', 'speedy')}
        for arm, state in states.items():
            check('worker_record:'+arm, state == result['workers'][arm] and state['arm'] == arm and state['status'] == 'COMPLETE')
            check('worker_counts:'+arm, state['forward_calls'] == (329 if arm == 'native' else 361) and state['quality_forward_calls'] == (1 if arm == 'native' else 33)
                and state['warmup_forward_calls'] == 8 and state['timed_forward_calls'] == 320 and state['backward_calls'] == state['optimizer_updates'] == state['Scene_instances'] == 0
                and state['optimizer_constructed'] is False)
            check('six_fields_before_after:'+arm, state['fields_before'] == state['fields_after'] == state['original_PLY_fields'] == reconstructed)
            model = state['model_before']
            check('model_before_after:'+arm, model == state['model_after'] and model['sha256'] == MODEL_SHA and model['point_count'] == COUNT
                and model['active_sh_degree'] == model['max_sh_degree'] == 3 and set(model['tensor_ids']) == set(FIELDS))
            check('camera_before_after:'+arm, state['camera_restoration_proof'] == camera_proof and state['camera_after_exact'] is True)
            check('GPU_budget:'+arm, 0 < state['gpu_peak_allocated_bytes'] <= 3*1024**3 and state['gpu_peak_allocated_bytes'] == main_supervisor['worker_peak_GPU_allocated_bytes'][arm])
            if arm == 'speedy':
                check('actual_private_import', state['loaded_wrapper'] == build['wrapper']['path'] and state['loaded_extension'] == build['extension']['path'])
            else:
                check('native_import', state['loaded_wrapper'] == '/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization/__init__.py'
                    and state['loaded_extension'] == '/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so')
            need(state['loaded_wrapper'] in cfg['protected_files'] and state['loaded_extension'] in cfg['protected_files'], 'Loaded ABI pins mandatory')
        check('forward_total', result['forward_calls'] == 690 and result['backward_calls'] == result['optimizer_updates'] == 0)
        check('GPU_sum', sum(s['gpu_peak_allocated_bytes'] for s in states.values()) == main_supervisor['sum_worker_peak_GPU_allocated_bytes'] <= 6*1024**3)
        same('config/timing_plan', cfg['complete_timing_plan'], plan(test_views))
        check('warmup_views', cfg['warmup_views'] == test_views[:8])
        timed = parse_timing(RUN/'TIMING.csv')
        timing = summarize_timing(timed, test_views)
        same('timing', result['timing'], timing, 'timing_ms')
        for arm in ('native', 'speedy'):
            same('worker_timing:'+arm, parse_timing(RUN/(arm+'_TIMING.partial.csv')), [r for r in timed if r['arm'] == arm], 'timing_ms')
        # RPC sequence binds all quality/warmup/timed/finish declarations and ACK order.
        commands = [('native', 'quality', dict(views=['000016.jpg'])), ('speedy', 'quality', dict(views=views))]
        commands += [(a, 'warmup', dict(views=test_views[:8])) for a in ('native', 'speedy')]
        commands += [(a, 'timed_block', dict(repeat=i, rows=[p for p in plan(test_views) if p['repeat'] == i and p['arm'] == a]))
            for i in range(10) for a in (('native', 'speedy') if i % 2 == 0 else ('speedy', 'native'))]
        commands += [(a, 'finish', {}) for a in ('native', 'speedy')]
        rpc_log = [json.loads(line) for line in (RUN/'RPC.jsonl').read_text().splitlines()]
        need(len(rpc_log) == 52, 'Exactly26 RPC requests and26 ACKs')
        seqs, forwards, previous_clock = dict(native=0, speedy=0), dict(native=0, speedy=0), -math.inf
        expected_rpc_names = {a+'.0.response.json' for a in ('native', 'speedy')}
        for arm in ('native', 'speedy'):
            check('ready_ACK:'+arm, read(RUN/'rpc'/f'{arm}.0.response.json') == dict(seq=0, arm=arm, status='ACK', ready=True))
        for index, (arm, action, data) in enumerate(commands):
            seqs[arm] += 1
            seq = seqs[arm]
            command = dict(seq=seq, arm=arm, action=action, **data)
            response = read(RUN/'rpc'/f'{arm}.{seq}.response.json')
            same('RPC_command:'+str(index), read(RUN/'rpc'/f'{arm}.{seq}.request.json'), command)
            expected_rpc_names.update((f'{arm}.{seq}.request.json', f'{arm}.{seq}.response.json'))
            expected_measured = [r for r in timed if r['arm'] == arm and r['repeat'] == data['repeat']] if action == 'timed_block' else []
            same('RPC_response:'+str(index), response, dict(seq=seq, arm=arm, status='ACK', action=action, measured=expected_measured), 'timing_ms')
            forwards[arm] += (1 if arm == 'native' else 33) if action == 'quality' else 8 if action == 'warmup' else 32 if action == 'timed_block' else 0
            req, ack = rpc_log[index*2:index*2+2]
            check('RPC_log_request:'+str(index), {k: v for k, v in req.items() if k not in ('event', 'monotonic')} == command and req['event'] == 'REQUEST')
            check('RPC_log_ACK:'+str(index), {k: v for k, v in ack.items() if k not in ('event', 'monotonic')} == dict(seq=seq, arm=arm, action=action, forward_calls=forwards[arm]) and ack['event'] == 'ACK')
            check('RPC_serial_clock:'+str(index), math.isfinite(req['monotonic']) and math.isfinite(ack['monotonic']) and previous_clock <= req['monotonic'] <= ack['monotonic'])
            previous_clock = ack['monotonic']
        check('RPC_exact_membership', {p.name for p in (RUN/'rpc').iterdir()} == expected_rpc_names)
        reference = [r for r in read(OLD/'RAW_MANIFEST.json') if r['path'].split('/')[1] in ('GT', 'teacher')]
        need(len(reference) == 66 and {r['path'] for r in reference} == {f'raw/{a}/{v}.npz' for a in ('GT', 'teacher') for v in views}, 'Exactly66 old references')
        expected_reference = [dict(r, path=str(OLD/r['path']), origin=str(OLD)) for r in reference]
        same('reference66', cfg['reference_raw_manifest'], expected_reference)
        manifest = read(RUN/'RAW_MANIFEST.json')
        need(len(manifest) == 34 and {r['path'] for r in manifest} == {'raw/vanilla_parity/000016.jpg.npz'} | {f'raw/speedy/{v}.npz' for v in views}, 'Exactly34 new raws')
        same('worker_manifest34', manifest, read(RUN/'native_RAW_MANIFEST.json')+read(RUN/'speedy_RAW_MANIFEST.json'))
        expected_combined = expected_reference+[dict(r, path=str(RUN/r['path']), origin=str(RUN)) for r in manifest]
        same('combined100', read(RUN/'COMBINED_RAW_MANIFEST.json'), expected_combined)
        need(len(expected_combined) == len({r['path'] for r in expected_combined}) == 100, 'Unique100 raw records')
        raw_records = {}
        for root, items in ((OLD, reference), (RUN, manifest)):
            for item in items:
                check('raw_layout:'+item['path'], item['dtype'] == 'float32' and item['shape'] == [3, 546, 979])
                protect(root/item['path'], item['sha256'], item['bytes'])
                arm, view = item['path'].split('/')[1], Path(item['path']).name.removesuffix('.npz')
                raw_records[arm, view] = (root, item)
                if root == RUN:
                    need(item['png_path'] == f'images/{arm}/{view}.png', 'Exact new PNG membership')
                    protect(root/item['png_path'], item['png_sha256'])
        check('new_raw_file_membership', {str(p.relative_to(RUN)) for p in (RUN/'raw').rglob('*') if p.is_file()} == {r['path'] for r in manifest})
        check('new_png_file_membership', {str(p.relative_to(RUN)) for p in (RUN/'images').rglob('*') if p.is_file()} == {r['png_path'] for r in manifest})
        need(set(cfg['source_GT_bindings']) == set(views), 'Exactly33 photo bindings')
        for view, binding in cfg['source_GT_bindings'].items():
            path = '/home/administrator/datasets/tandt_db/tandt/truck/images/'+view
            need(binding == dict(path=path, sha256=old_cfg['input_hashes_before'][path], GT_array_sha256=old_result['gt_array_sha256'][view]), 'Original source photo/GT binding')
            protect(path, binding['sha256'])
        def raw(arm, view):
            root, item = raw_records[arm, view]
            value = pixels(root/item['path'])
            if root == RUN:
                need(array_sha(value) == item['array_sha256'], 'Raw array hash: '+item['path'])
                with Image.open(root/item['png_path']) as png:
                    need(png.mode == 'RGB' and png.size == (979, 546) and np.array_equal(np.asarray(png), quantize(value)), 'Exact PNG pixels: '+item['png_path'])
            return value
        stored = []
        with (RUN/'METRICS.csv').open(newline='', encoding='utf8') as stream:
            reader = csv.DictReader(stream)
            need(reader.fieldnames == ['arm', 'view', 'split', 'gt_mse', 'gt_psnr', 'zero_MSE_infinite_PSNR', 'gt_ssim', 'teacher_mse', 'teacher_max_abs_rgb'], 'Exact metric columns')
            for item in reader:
                need(item['zero_MSE_infinite_PSNR'] in ('True', 'False'), 'Explicit CSV bool')
                stored.append({k: (v == 'True') if k == 'zero_MSE_infinite_PSNR' else (None if v == '' else float(v)) if k in METRICS else v for k, v in item.items()})
        need(len(stored) == len({(r['arm'], r['view']) for r in stored}) == 66, 'Exactly66 unique metric rows')
        def one_view(view):
            gt, teacher, speedy = raw('GT', view), raw('teacher', view), raw('speedy', view)
            with Image.open(cfg['source_GT_bindings'][view]['path']) as photo:
                need(photo.mode == 'RGB' and photo.size == (979, 546), 'Original RGB JPEG layout')
                original = np.ascontiguousarray(np.asarray(photo).astype(np.float32).transpose(2, 0, 1)/np.float32(255))
            need(np.array_equal(original.view(np.uint8), gt.view(np.uint8)) and array_sha(gt) == cfg['source_GT_bindings'][view]['GT_array_sha256'], 'Source JPEG equals GT')
            truth, baseline = gt.astype(np.float64), teacher.astype(np.float64)
            mean, variance = metric_pair(None, truth, baseline, None, None)
            out = [dict(arm=a, view=view, split='train' if view == '000016.jpg' else 'test', **metric_pair(v, truth, baseline, mean, variance)) for a, v in (('teacher', teacher), ('speedy', speedy))]
            return out, dict(view=view, exact_RGB_match=np.array_equal(teacher.view(np.uint8), speedy.view(np.uint8)), mismatching_RGB_elements=int(np.count_nonzero(teacher != speedy)))
        exact = []
        with (OUTPUT/'METRICS.partial.csv').open('x', newline='', encoding='utf8') as stream:
            writer = None
            with ThreadPoolExecutor(max_workers=4) as pool:
                for entries, equality in pool.map(one_view, views):
                    rows.extend(entries)
                    exact.append(equality)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
                        writer.writeheader()
                    writer.writerows(entries)
                    stream.flush()
        same('all66_metrics', stored, rows)
        quality = summarize_quality(rows, test_views, exact)
        same('quality', result['quality'], quality)
        anchor = raw('vanilla_parity', '000016.jpg').astype(np.float64)
        parity = float(np.max(np.abs(anchor-raw('teacher', '000016.jpg').astype(np.float64))))
        same('native_parity_maxabs', states['native']['teacher_parity_max_abs_RGB'], parity, 'max_abs_rgb')
        check('native_parity_gate', 0 <= parity <= 2e-5)
        expected_panels = [(v, h, tag) for v in FIXED for h, tag in ((546, 'fixed.full'), (220, 'fixed.top220'))]
        expected_panels += [(r['view'], 546, f'exploratory_SSIM_loss_rank{i}.full') for i, r in enumerate(quality['worst_SSIM_loss_ranking32'][:4], 1)]
        panel_manifest = read(RUN/'PANEL_MANIFEST.json')
        need(len(panel_manifest) == 10 and {r['path'] for r in panel_manifest} == {f'panels/{v}.{tag}.png' for v, _, tag in expected_panels}, 'Exact10 panels')
        check('panel_file_membership', {p.name for p in (RUN/'panels').iterdir()} == {f'{v}.{tag}.png' for v, _, tag in expected_panels})
        panel_records = {r['path']: r for r in panel_manifest}
        for view, height, tag in expected_panels:
            path = f'panels/{view}.{tag}.png'
            item = panel_records[path]
            protect(RUN/path, item['sha256'], item['bytes'])
            check('panel_metadata:'+path, item['view'] == view and item['height'] == height and item['width'] == 2937 and item['header_height'] == 28
                and item['columns'] == ['GT', 'teacher', 'Speedy'] and item['tag'] == tag and item['image_source'] == 'actual_saved_pixels' and item['AI_generated'] is False)
            with Image.open(RUN/path) as image:
                need(image.mode == 'RGB' and image.size == (2937, height+28), 'Panel dimensions')
                panel = np.asarray(image)
            expected_image = Image.new('RGB', (2937, height+28), 'white')
            draw = ImageDraw.Draw(expected_image)
            for column, arm in enumerate(('GT', 'teacher', 'speedy')):
                expected_pixels = quantize(raw(arm, view)[:, :height])
                check('panel_pixels:'+path+'/'+arm, np.array_equal(panel[28:, column*979:(column+1)*979], expected_pixels))
                expected_image.paste(Image.fromarray(expected_pixels, 'RGB'), (column*979, 28))
                draw.text((column*979+8, 7), f'{view} | '+('Speedy' if arm == 'speedy' else arm)+f' | {tag}', fill='black')
            check('panel_full_pixels_and_labels:'+path, np.array_equal(panel, np.asarray(expected_image)))
        outcome = 'KEEP_KNOWN_ENGINEERING_CONTROL' if quality['quality_joint_pass'] and timing['timing_joint_pass'] else 'STOP_CURRENT_KERNEL_CONTROL'
        check('control_gate_outcome', result['gate_outcome_before_independent_audit'] == outcome)
        check('pending_audit_declaration', result['independent_audit_status'] == 'NOT_RUN')
        report.update(recomputed_quality=quality, recomputed_timing=timing, control_outcome_from_saved_artifacts=outcome,
            timing_speed_ratios={k: timing['mean_32_perview_10repeat_median_ms']['native'][k]/timing['mean_32_perview_10repeat_median_ms']['speedy'][k] for k in ('wall', 'cuda')},
            raw_arrays_checked=100, source_GT_pairs_checked=33, metric_rows_checked=66, timing_rows_checked=640, panel_images_checked=10,
            native_parity_max_abs_RGB=parity, teacher_points_checked=COUNT, teacher_properties_checked=62,
            main_actual_external_wall_seconds=external['actual_host_wall_seconds'])
        need(not monitor_errors, 'RSS monitor error')
        report['status'] = PASS if all(c['passed'] for c in checks) else 'FAIL_SAVED_ARTIFACT_CONSISTENCY'
    except BaseException as error:
        supervisor['timeout'] = isinstance(error, TimeoutError)
        report['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        # Hash every bound file again even after a failed calculation, preserving partial evidence.
        report.update(checks=checks, comparisons=comparisons, protected_files=hashes, partial_metric_rows=rows)
        save(OUTPUT/'AUDIT.json', report)
        save(OUTPUT/'SUPERVISOR.json', supervisor)
        try:
            for path, item in hashes.items():
                item['after'] = sha(path)
                check('unchanged_after:'+path, item['before'] == item['after'] and item['bytes'] == Path(path).stat().st_size)
        except BaseException as error:
            report['final_hash_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            report['status'] = 'INCONCLUSIVE'
        if any(not c['passed'] for c in checks) and report['status'] == PASS:
            report['status'] = 'FAIL_SAVED_ARTIFACT_CONSISTENCY'
        signal.alarm(0)
        stop_monitor.set()
        watcher.join(timeout=.2)
        if monitor_errors:
            report['status'] = 'INCONCLUSIVE'
        report.update(checks=checks, comparisons=comparisons, protected_files=hashes, partial_metric_rows=rows,
            failed_checks=[c['name'] for c in checks if not c['passed']], monitor_errors=monitor_errors,
            actual_internal_wall_seconds=time.monotonic()-started, torch_imported='torch' in sys.modules)
        supervisor.update(status=report['status'], actual_internal_wall_seconds=time.monotonic()-started,
            torch_imported='torch' in sys.modules, monitor_errors=monitor_errors)
        if 'original_exception' in report:
            supervisor['original_exception'] = report['original_exception']
        save(OUTPUT/'AUDIT.json', report)
        save(OUTPUT/'SUPERVISOR.json', supervisor)
    return 0 if report['status'] == PASS else 2


def synthetic_check():
    import numpy as np
    # Only tiny, generated inputs. No file reads, real arrays or actual output directory.
    rng = np.random.default_rng(19)
    truth = rng.random((3, 13, 17), dtype=np.float64)
    mean, variance = metric_pair(None, truth, truth, None, None)
    identity = metric_pair(truth, truth, truth, mean, variance)
    need(identity['gt_mse'] == identity['teacher_mse'] == identity['teacher_max_abs_rgb'] == 0
        and identity['gt_psnr'] is None and identity['zero_MSE_infinite_PSNR'] and abs(identity['gt_ssim']-1) < 1e-14, 'Tiny identity metric')
    shifted = metric_pair(truth+.01, truth, truth, mean, variance)
    need(abs(shifted['gt_mse']-.0001) < 1e-16 and abs(shifted['gt_psnr']-40) < 1e-10, 'Tiny float64 MSE/PSNR')
    # Check the separable convolution against a direct finite 11x11 sum.
    weights = np.exp(-np.arange(-5, 6, dtype=np.float64)**2/(2*1.5**2))
    weights /= weights.sum()
    padded = np.pad(truth, ((0, 0), (5, 5), (5, 5)))
    direct = np.zeros_like(truth)
    for y in range(11):
        for x in range(11):
            direct += weights[y]*weights[x]*padded[:, y:y+13, x:x+17]
    need(np.max(np.abs(direct-mean)) < 1e-14, 'Tiny direct Gaussian reference')
    # Generated two-point PLY exercises all62 fields and independent SH transpose.
    points = (np.arange(124, dtype=np.float32).reshape(2, 62)-17)/np.float32(31)
    with tempfile.TemporaryDirectory(prefix='tiny-ply-', dir=HERE) as directory:
        fixture = Path(directory)/'generated.ply'
        header = 'ply\nformat binary_little_endian 1.0\nelement vertex 2\n'+''.join('property float '+n+'\n' for n in PLY_FIELDS)+'end_header\n'
        fixture.write_bytes(header.encode('ascii')+points.astype('<f4').tobytes())
        fields = ply_fields(fixture, expected_count=2)
        expected_rest = np.array([[[points[p, 9+c*15+k] for c in range(3)] for k in range(15)] for p in range(2)], dtype='<f4')
        need(fields['_features_rest'] == dict(dtype='float32', shape=[2, 15, 3], sha256=hashlib.sha256(expected_rest.tobytes()).hexdigest()), 'Tiny62 PLY SH mapping')
        need(fields['_xyz']['sha256'] == hashlib.sha256(np.ascontiguousarray(points[:, :3]).tobytes()).hexdigest(), 'Tiny PLY XYZ mapping')
    views = [f'{i:06d}.jpg' for i in range(1001, 1033)]
    ordered = plan(views)
    timings = [dict(r, wall_ms=2. if r['arm'] == 'native' else 1., cuda_ms=1.5 if r['arm'] == 'native' else .75) for r in ordered]
    summary = summarize_timing(timings, views)
    need(len(ordered) == 640 and summary['timing_joint_pass'] and summary['faster_repeat_blocks'] == 10
        and summary['synchronized_wall_reduction_fraction'] == .5, 'Tiny full timing protocol')
    rejected = 0
    for broken in (timings[:-1], list(reversed(timings)), [dict(r, wall_ms=0) if i == 0 else r for i, r in enumerate(timings)]):
        try:
            summarize_timing(broken, views)
        except RuntimeError:
            rejected += 1
    need(rejected == 3, 'Tiny incomplete/order/nonpositive rejection')
    synthetic_rows = [dict(arm=a, view=v, split='train' if v == '000016.jpg' else 'test',
        gt_mse=.01 if a == 'teacher' else .009, gt_psnr=20. if a == 'teacher' else 21., gt_ssim=.8 if a == 'teacher' else .9,
        teacher_mse=0. if a == 'teacher' else .001, teacher_max_abs_rgb=0. if a == 'teacher' else .1, zero_MSE_infinite_PSNR=False)
        for v in views+['000016.jpg'] for a in ('teacher', 'speedy')]
    quality = summarize_quality(synthetic_rows, views, [dict(view=v, exact_RGB_match=False, mismatching_RGB_elements=1) for v in views+['000016.jpg']])
    need(quality['quality_joint_pass'] and quality['worst_SSIM_loss_ranking32'][0]['teacher_minus_speedy_gt_ssim'] < 0
        and [r['view'] for r in quality['worst_SSIM_loss_ranking32']] == views, 'Tiny negative loss/tie ordering')
    panel = np.concatenate([quantize(truth.astype(np.float32))]*3, axis=1)
    need(panel.shape == (13, 51, 3) and np.array_equal(panel[:7, 17:34], quantize(truth.astype(np.float32)[:, :7])), 'Tiny pixel crop')
    need('torch' not in sys.modules and not OUTPUT.exists(), 'Preparation must not initialize GPU or actual output')
    return dict(status='TINY_SYNTHETIC_PASS_ONLY', actual_audit_executed=False, scientific_arrays_read=False,
        checks=['float64_identity_MSE_PSNR_SSIM', 'direct_11x11_Gaussian_reference', 'generated_2point_62field_PLY_SH_transpose',
            '640_AB_BA_protocol', 'malformed_timing_rejected', 'signed_negative_ranking', 'three_column_pixel_crop'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--synthetic-check', action='store_true')
    for name in PIN_FILES:
        parser.add_argument('--'+name.replace('_', '-'), type=Path)
        parser.add_argument('--'+name.replace('_', '-')+'-sha256')
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check(), indent=2, allow_nan=False))
        return 0
    return audit(args)


if __name__ == '__main__':
    sys.exit(main())
