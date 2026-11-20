#!/usr/bin/env python3
"""Frozen native three-arm timing; stdlib-only import and CPU synthetic mode."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Only audited freeze/subset/identity helpers; no producer metric summarizer.
from evaluate_nanogs_frozen_baseline import camera_record, freeze, make_subset, model_identity, array_hash
from shared_coordinate_native_v3 import DEFAULT_REPO, DEFAULT_SOURCE, FIELDS, _check_import_path, sha

PROJECT = Path(__file__).resolve().parents[2]
PROFILE = PROJECT/'profile_output/nanogs-frozen-baseline-v0'
EVALUATION = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
NANOGS = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/run-v1/output.ply'
TEACHER = DEFAULT_REPO/'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
INDICES = PROJECT/'research/experiments/truck30k-coverage-controls-v0/run/all219_indices.npy'
ARMS = ('teacher', 'all219', 'nanogs')
TEST_NAMES = tuple(f'{8*i+1:06d}.jpg' for i in range(32))
WARMUP = (0, 4, 8, 13, 17, 22, 26, 31)
GPU_LIMIT = 10*1024**3
COUNT, TARGET = 2050770, 1025385
CONTRACT_SHA = '230f02ae07bae098a6a3679669d7ed019b6701b7fbaee55a8b688416da41363f'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
NANOGS_SHA = '383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8'
INDICES_SHA = 'f6c9f4da05e064db5c94134c8ee241eb2188fb83a5c3f860655d2eaf279f992f'
EVAL_PINS = {
    'RESULTS.json': '19f229837935d3624f2e179112d58217f479e1f2a572e23ce0cac6700c17f29c',
    'CONFIG.json': '3b53a5c588fbd55d190dc232b61c72ce9998e0cf04d0b26393c3b32b83b368c0',
    'CAMERAS.json': 'd70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67',
    'RAW_MANIFEST.json': '82dd8bfa39548d64b0a2eb55cf3cccc9de7e08d19657dd089585dc8f9bdca0e7',
}
MODEL_PINS = {
    'teacher': 'b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0',
    'all219': '96fcb4e9942c78fd26d9fb386f6f1c9429f66fce7405ac151c84fe529f9b20fd',
    'nanogs': 'cd945d1ec44f959c37137eefaf9a56b5f8db4fb66e5db0cf38afaba274f163ec',
}
ROW_FIELDS = ('timed_forward_index', 'repeat', 'arm', 'arm_position', 'camera_direction',
              'camera_position', 'camera_rank', 'image_name', 'cuda_ms', 'host_wall_ms')


class BudgetStop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_csv(path, rows):
    require(bool(rows), f'Empty CSV rows: {path}')
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timing_plan():
    plan = []
    for repeat in range(9):
        order = ARMS[repeat % 3:]+ARMS[:repeat % 3]
        ranks = list(range(32)) if repeat % 2 == 0 else list(range(31, -1, -1))
        for position, arm in enumerate(order):
            for camera_position, rank in enumerate(ranks):
                plan.append(dict(timed_forward_index=len(plan), repeat=repeat, arm=arm,
                    arm_position=position, camera_direction='ascending' if repeat % 2 == 0 else 'descending',
                    camera_position=camera_position, camera_rank=rank, image_name=TEST_NAMES[rank]))
    return plan


def distribution(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    require(len(values) > 0 and np.isfinite(values).all(), 'Invalid timing distribution')
    return dict(count=len(values), min=float(values.min()), p05=float(np.quantile(values, .05)),
                p25=float(np.quantile(values, .25)), median=float(np.median(values)),
                p75=float(np.quantile(values, .75)), p95=float(np.quantile(values, .95)),
                max=float(values.max()), mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def aggregate_timings(rows):
    """Independent timing reductions, with paired and pooled ratios kept separate."""
    import numpy as np
    require(len(rows) == 864, 'Exactly 864 measured frames required')
    expected = timing_plan()
    for row, plan in zip(rows, expected):
        require(all(row[key] == value for key, value in plan.items()), 'Measured sequence differs from frozen plan')
        require(all(math.isfinite(row[key]) and row[key] > 0 for key in ('cuda_ms', 'host_wall_ms')), 'Nonpositive/nonfinite timing')
    per_view, per_repeat, ratios, summary = [], [], [], {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row['arm'] == arm]
        require(len(arm_rows) == 288, f'Incomplete arm: {arm}')
        summary[arm] = {}
        for rank, view in enumerate(TEST_NAMES):
            selected = [row for row in arm_rows if row['camera_rank'] == rank]
            require(len(selected) == 9, 'Each view needs all nine repeats')
            per_view.append(dict(arm=arm, camera_rank=rank, image_name=view, repeats=9,
                cuda_median_ms=float(np.median([row['cuda_ms'] for row in selected])),
                host_wall_median_ms=float(np.median([row['host_wall_ms'] for row in selected]))))
        for repeat in range(9):
            selected = [row for row in arm_rows if row['repeat'] == repeat]
            require(len(selected) == 32, 'Each arm/repeat needs all 32 views')
            totals = {key: sum(row[key] for row in selected) for key in ('cuda_ms', 'host_wall_ms')}
            per_repeat.append(dict(arm=arm, repeat=repeat, frames=32,
                cuda_total_ms=totals['cuda_ms'], host_wall_total_ms=totals['host_wall_ms'],
                cuda_FPS=32000/totals['cuda_ms'], host_wall_FPS=32000/totals['host_wall_ms']))
        for metric in ('cuda', 'host_wall'):
            summary[arm][metric] = dict(
                all_288_frame_latency_ms=distribution([r[metric+'_ms'] for r in arm_rows]),
                repeat_32view_total_ms=distribution([r[metric+'_total_ms'] for r in per_repeat if r['arm'] == arm]),
                repeat_32view_FPS=distribution([r[metric+'_FPS'] for r in per_repeat if r['arm'] == arm]))
    pair_summary = {}
    for numerator, denominator in (('teacher', 'all219'), ('teacher', 'nanogs'), ('all219', 'nanogs')):
        pair_key = numerator+'_over_'+denominator
        pair_summary[pair_key] = {}
        for metric in ('cuda', 'host_wall'):
            for repeat in range(9):
                a = next(r for r in per_repeat if r['arm'] == numerator and r['repeat'] == repeat)
                b = next(r for r in per_repeat if r['arm'] == denominator and r['repeat'] == repeat)
                ratios.append(dict(pair=pair_key, metric=metric, scope='paired_repeat_32view_total', index=repeat,
                                   ratio=a[metric+'_total_ms']/b[metric+'_total_ms']))
            for rank in range(32):
                a = next(r for r in per_view if r['arm'] == numerator and r['camera_rank'] == rank)
                b = next(r for r in per_view if r['arm'] == denominator and r['camera_rank'] == rank)
                ratios.append(dict(pair=pair_key, metric=metric, scope='per_view_9repeat_median', index=rank,
                                   ratio=a[metric+'_median_ms']/b[metric+'_median_ms']))
            num = sum(r[metric+'_ms'] for r in rows if r['arm'] == numerator)
            den = sum(r[metric+'_ms'] for r in rows if r['arm'] == denominator)
            pair_summary[pair_key][metric] = dict(
                paired_repeat_total_ratio=distribution([r['ratio'] for r in ratios if r['pair'] == pair_key and
                    r['metric'] == metric and r['scope'] == 'paired_repeat_32view_total']),
                per_view_median_ratio=distribution([r['ratio'] for r in ratios if r['pair'] == pair_key and
                    r['metric'] == metric and r['scope'] == 'per_view_9repeat_median']),
                pooled_288frame_total_ratio=num/den)
    return dict(arms=summary, speedup_ratios=pair_summary,
                ratio_convention='numerator latency / denominator latency; >1 means denominator faster'), per_view, per_repeat, ratios


def synthetic_check():
    plan = timing_plan()
    require(len(plan) == 864, 'Plan count')
    for arm in ARMS:
        for position in range(3):
            require(len({r['repeat'] for r in plan if r['arm'] == arm and r['arm_position'] == position}) == 3,
                    'Each arm/position must occur in exactly three repeats')
    rows = []
    for row in plan:
        rows.append(dict(row, cuda_ms={'teacher': 10., 'all219': 5., 'nanogs': 4.}[row['arm']],
                         host_wall_ms={'teacher': 12., 'all219': 6., 'nanogs': 5.}[row['arm']]))
    summary, views, repeats, ratios = aggregate_timings(rows)
    require(len(views) == 96 and len(repeats) == 27 and len(ratios) == 246, 'Aggregation dimensions')
    require(summary['speedup_ratios']['teacher_over_all219']['cuda']['paired_repeat_total_ratio']['median'] == 2., 'Paired ratio definition')
    require(summary['speedup_ratios']['teacher_over_nanogs']['host_wall']['pooled_288frame_total_ratio'] == 2.4, 'Pooled ratio definition')
    forbidden = [name for name in ('torch', 'scene', 'gaussian_renderer') if name in sys.modules]
    require(not forbidden, 'Synthetic mode imported a GPU/model module')
    return dict(synthetic_only=True, status='PASS', measured_plan_rows=len(plan), per_view_rows=len(views),
                per_repeat_rows=len(repeats), ratio_rows=len(ratios), positive_direction_repeats=5,
                reverse_direction_repeats=4, forbidden_modules=forbidden, scientific_inputs_read=False)


def driver_record():
    executable = shutil.which('nvidia-smi')
    if executable is None and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        executable = '/usr/lib/wsl/lib/nvidia-smi'
    require(executable is not None, 'nvidia-smi required to record driver identity')
    response = subprocess.run([executable, '--query-gpu=index,name,uuid,driver_version,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=False)
    require(response.returncode == 0, 'Cannot record GPU/driver metadata')
    return dict(command=executable, query=response.stdout.strip(), stderr=response.stderr.strip())


def benchmark(output):
    require(output.resolve() == (PROFILE/'run-v1').resolve(), 'Fixed fresh run-v1 path required')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, parity, protected, models, model_before = [], [], {}, {}, {}
    cameras, camera_before, image_before = [], None, None
    torch, raw_stream, cleanup = None, None, False
    result = dict(decision='INCONCLUSIVE', integrity_complete=False, stage='initialization',
        forward_attempted=0, forward_completed=0, parity_completed=0, warmup_completed=0,
        measured_completed=0, backward_calls=0, optimizer_steps=0, quality_acceptance_gate=None,
        independent_process_budget_seconds=300, internal_compute_deadline_seconds=290,
        gpu_allocated_limit_bytes=GPU_LIMIT, models={}, model_integrity={}, camera_integrity=None)
    config = dict(contract_sha256=CONTRACT_SHA, test_views=TEST_NAMES, warmup_camera_indices=WARMUP,
        complete_plan=timing_plan(), renderer=dict(SH=3, background=[0, 0, 0], AA=False, exposure=False,
        scaling_modifier=1., separate_sh=False, native_final_clamp=True),
        timing_scope='Pre-sync outside wall; wall includes event recording, full render and end-event synchronize. Loading/metrics/I/O outside both timing windows.',
        direction_balance=dict(ascending=5, descending=4), no_outlier_removal=True)

    def save_partial():
        result['runner_wall_seconds_before_partial_save'] = time.monotonic()-started
        dump(output/'RESULTS.json', result)
        dump(output/'CONFIG.json', config)
        dump(output/'PARITY.json', parity)

    def budget():
        if time.monotonic()-started >= 290:
            raise BudgetStop('290-second compute deadline; only cleanup/save may follow')
        if torch is not None:
            require(torch.cuda.max_memory_allocated() <= GPU_LIMIT, 'CUDA allocated peak exceeds 10GiB')

    def on_signal(signum, _):
        result['received_signal'] = signum
        if not cleanup:
            raise BudgetStop(f'Signal {signum}: retain partial run and do not retry')

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, on_signal)
        signal.alarm(290)
    try:
        bootstrap = {PROFILE/'CONTRACT.md': CONTRACT_SHA,
            PROJECT/'scripts/experiments/evaluate_nanogs_frozen_baseline.py': '0c62664a1b90f6581215d2285971462175aae4c753cfe9fad5a0cd207c2525e6',
            PROJECT/'scripts/experiments/shared_coordinate_native_v3.py': '2a089b7d85987b571bb53f22ce9a97912fe690e01fc2d9d950dead39fd3a29a3',
            TEACHER: TEACHER_SHA, NANOGS: NANOGS_SHA, INDICES: INDICES_SHA}
        bootstrap.update({EVALUATION/name: value for name, value in EVAL_PINS.items()})
        for path, digest in bootstrap.items():
            budget()
            protected[str(path)] = sha(path)
            require(protected[str(path)] == digest, f'Pinned input changed: {path}')
        evaluation, old_config = load_json(EVALUATION/'RESULTS.json'), load_json(EVALUATION/'CONFIG.json')
        require(evaluation['decision'] == 'FROZEN_BASELINE_EVALUATED' and evaluation['integrity_complete'], 'Prior complete evaluation required')
        require(tuple(old_config['test_views']) == TEST_NAMES, 'Original 32-camera identity')
        for arm in ARMS:
            require(evaluation['models'][arm]['before']['sha256'] == MODEL_PINS[arm] and
                    evaluation['models'][arm]['before'] == evaluation['models'][arm]['after'], 'Prior frozen model identity')
        native_sources = old_config['source_hashes_before']
        for path, digest in native_sources.items():
            protected[path] = sha(path)
            require(protected[path] == digest, f'Native source/extension changed: {path}')
        extra_paths = [Path(__file__).resolve(), PROFILE/'run-native-benchmark.sh', DEFAULT_SOURCE/'sparse/0/points3D.ply']
        extra_paths += [DEFAULT_SOURCE/'sparse/0'/name for name in ('images.bin', 'cameras.bin')]
        extra_paths += [DEFAULT_SOURCE/'images'/view for view in TEST_NAMES]
        for path in extra_paths:
            protected[str(path)] = sha(path)
            expected = old_config['input_hashes_before'].get(str(path))
            require(expected is None or protected[str(path)] == expected, f'Evaluation camera/data source changed: {path}')
        raw_manifest = {entry['path']: entry for entry in load_json(EVALUATION/'RAW_MANIFEST.json')}
        references = {}
        for arm in ARMS:
            for view in TEST_NAMES:
                relative = f'raw/{arm}/{view}.npz'
                entry, path = raw_manifest[relative], EVALUATION/relative
                protected[str(path)] = sha(path)
                require(protected[str(path)] == entry['sha256'], f'Evaluation RGB file changed: {relative}')
                references[(arm, view)] = dict(path=path, sha256=entry['sha256'])
        config.update(input_hashes_before=protected.copy(), source_hashes_before=native_sources,
                      model_files={arm: evaluation['model_files'][arm] for arm in ARMS},
                      actual_PLY_bytes=dict(teacher=TEACHER.stat().st_size, all219=None, nanogs=NANOGS.stat().st_size))
        save_partial()
        import numpy as np
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        require(torch.cuda.is_available(), 'Original CUDA environment required')
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1., GPU_LIMIT/properties.total_memory))
        torch.cuda.reset_peak_memory_stats()
        config['hardware'] = dict(torch_version=torch.__version__, torch_CUDA_version=torch.version.cuda,
            GPU_name=properties.name, total_memory_bytes=properties.total_memory,
            compute_capability=list(torch.cuda.get_device_capability(0)), driver=driver_record())
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, DEFAULT_REPO)
        if str(DEFAULT_REPO) not in sys.path:
            sys.path.insert(0, str(DEFAULT_REPO))
        from scene import Scene, GaussianModel
        from gaussian_renderer import render
        import diff_gaussian_rasterization as rasterizer
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, DEFAULT_REPO)
        for module in (rasterizer, rasterizer._C):
            path = str(Path(module.__file__).resolve())
            require(path in native_sources and sha(path) == native_sources[path], 'Loaded rasterizer differs from prior evaluation')
        dataset = SimpleNamespace(source_path=str(DEFAULT_SOURCE), model_path=str(TEACHER.parents[2]),
            images='images', depths='', resolution=1, white_background=False, train_test_exp=False,
            data_device='cpu', eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        with torch.no_grad():
            result['stage'] = 'loading_three_resident_models'
            teacher = GaussianModel(3, 'default')
            scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False, resolution_scales=[1.0])
            models['teacher'] = freeze(teacher)
            cameras = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
            require(tuple(c.image_name for c in cameras) == TEST_NAMES and len(scene.getTrainCameras()) == 219, 'Official Scene camera split')
            require(all((c.image_width, c.image_height) == (979, 546) for c in cameras), 'Native image size')
            del scene
            gc.collect()  # Release unused CPU training photos before timing; test Camera objects remain.
            indices = np.load(INDICES, allow_pickle=False)
            require(indices.ndim == 1 and len(indices) == TARGET and np.issubdtype(indices.dtype, np.integer) and
                    len(np.unique(indices)) == TARGET and int(indices.min()) >= 0 and int(indices.max()) < COUNT, 'Frozen all219 indices')
            models['all219'] = make_subset(teacher, np.sort(indices), GaussianModel)
            nanogs = GaussianModel(3, 'default')
            nanogs.load_ply(str(NANOGS), use_train_test_exp=False)
            models['nanogs'] = freeze(nanogs)
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            for arm in ARMS:
                budget()
                model_before[arm] = model_identity(models[arm])
                require(model_before[arm]['sha256'] == MODEL_PINS[arm], f'{arm}: raw model differs from evaluation')
                require(model_before[arm]['point_count'] == (COUNT if arm == 'teacher' else TARGET), 'Point count')
                result['models'][arm] = {'before': model_before[arm]}
            camera_before = [camera_record(c) for c in cameras]
            old_cameras = {record['image_name']: record for record in load_json(EVALUATION/'CAMERAS.json')}
            require(all(record == old_cameras[record['image_name']] for record in camera_before), 'Camera transforms differ from evaluation')
            image_before = {c.image_name: array_hash(c.original_image[:3].detach().cpu().numpy()) for c in cameras}
            dump(output/'CAMERAS_BEFORE.json', camera_before)
            config['camera_image_array_sha256_before'] = image_before
            config['camera_records_sha256_before'] = hashlib.sha256(json.dumps(camera_before, sort_keys=True).encode()).hexdigest()
            save_partial()
            result['stage'] = 'parity_96_before_timing'
            for arm in ARMS:
                for rank, camera in enumerate(cameras):
                    budget()
                    result['forward_attempted'] += 1
                    rendered = render(camera, models[arm], pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)
                    result['forward_completed'] += 1
                    rgb = rendered['render'].detach().cpu().numpy()
                    require(rgb.dtype == np.float32 and rgb.shape == (3, 546, 979) and np.isfinite(rgb).all(), 'Invalid native RGB')
                    reference = references[(arm, camera.image_name)]
                    with np.load(reference['path'], allow_pickle=False) as saved:
                        old_rgb = saved['rgb']
                    require(old_rgb.dtype == np.float32 and old_rgb.shape == rgb.shape, 'Reference RGB ABI')
                    difference = float(np.max(np.abs(rgb-old_rgb)))
                    parity.append(dict(arm=arm, camera_rank=rank, image_name=camera.image_name,
                        reference_file_sha256=reference['sha256'], reference_RGB_array_sha256=array_hash(old_rgb),
                        replay_RGB_array_sha256=array_hash(rgb), max_abs_RGB_error=difference, passed=difference <= 2e-5))
                    result['parity_completed'] += 1
                    del rendered, rgb, old_rgb
                    if difference > 2e-5:
                        save_partial()
                        raise RuntimeError(f'Native replay parity failed: {arm}/{camera.image_name}')
                save_partial()
                print(json.dumps(dict(stage='parity_arm_complete', arm=arm, forwards=result['forward_completed'])), flush=True)
            require(len(parity) == 96 and all(p['passed'] for p in parity), 'All 96 parity checks required')
            plan = timing_plan()
            event_pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in plan]
            for begin, end in event_pairs:
                begin.record()
                end.record()
            torch.cuda.synchronize()  # Materialize all CUDA event handles outside measured frames.
            result['stage'] = 'fixed_24_warmups'
            for arm in ARMS:
                for rank in WARMUP:
                    budget()
                    result['forward_attempted'] += 1
                    rendered = render(cameras[rank], models[arm], pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)
                    torch.cuda.synchronize()
                    result['forward_completed'] += 1
                    result['warmup_completed'] += 1
                    del rendered
            require(result['warmup_completed'] == 24, 'Complete fixed warmups required')
            result['stage'] = 'measuring'
            raw_stream = (output/'rawtimes.csv').open('w', newline='', encoding='utf8')
            writer = csv.DictWriter(raw_stream, fieldnames=ROW_FIELDS)
            writer.writeheader()
            raw_stream.flush()
            for specification, (begin, end) in zip(plan, event_pairs):
                budget()
                arm, rank = specification['arm'], specification['camera_rank']
                result['forward_attempted'] += 1
                torch.cuda.synchronize()
                wall_start = time.perf_counter_ns()
                begin.record()
                rendered = render(cameras[rank], models[arm], pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)
                end.record()
                end.synchronize()
                wall_end = time.perf_counter_ns()
                cuda_ms = float(begin.elapsed_time(end))
                result['forward_completed'] += 1
                row = dict(specification, cuda_ms=cuda_ms, host_wall_ms=(wall_end-wall_start)/1e6)
                require(all(math.isfinite(row[key]) and row[key] > 0 for key in ('cuda_ms', 'host_wall_ms')), 'Invalid measured frame time')
                rows.append(row)
                writer.writerow(row)
                raw_stream.flush()  # Outside timing; retain every completed raw row on interruption.
                result['measured_completed'] += 1
                del rendered
                if specification['camera_position'] == 31:
                    save_partial()
                    print(json.dumps(dict(stage='measured_arm_complete', repeat=specification['repeat'], arm=arm,
                                          measured_rows=len(rows))), flush=True)
            raw_stream.close()
            raw_stream = None
            budget()
            result['timing_summary'], per_view, per_repeat, ratios = aggregate_timings(rows)
            write_csv(output/'PER_VIEW_MEDIANS.csv', per_view)
            write_csv(output/'REPEAT_TOTALS.csv', per_repeat)
            write_csv(output/'SPEEDUP_RATIOS.csv', ratios)
            require(result['forward_completed'] == 984 and len(rows) == 864, 'Incomplete frozen benchmark')
            result['decision'] = 'BENCHMARK_COMPLETE'
    except BaseException as exc:
        result.update(decision='INCONCLUSIVE', failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc())
    finally:
        cleanup = True
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        if raw_stream is not None:
            raw_stream.flush()
            raw_stream.close()
        integrity_errors, after = [], {}
        for path, digest in protected.items():
            try:
                after[path] = sha(path)
                if after[path] != digest:
                    integrity_errors.append(f'File changed: {path}')
            except Exception as exc:
                integrity_errors.append(f'File check unavailable: {path}: {exc}')
        for arm in ARMS:
            if arm not in models:
                result['model_integrity'][arm] = {'loaded': False, 'verified': False}
                continue
            try:
                identity = model_identity(models[arm])
                verified = arm in model_before and identity == model_before[arm]
                result['model_integrity'][arm] = {'loaded': True, 'verified': verified, 'after': identity}
                if not verified:
                    integrity_errors.append(f'Model integrity unknown/changed: {arm}')
            except Exception as exc:
                result['model_integrity'][arm] = {'loaded': True, 'verified': False, 'error': str(exc)}
                integrity_errors.append(f'Model check unavailable: {arm}: {exc}')
        try:
            if camera_before is not None and image_before is not None:
                camera_after = [camera_record(c) for c in cameras]
                image_after = {c.image_name: array_hash(c.original_image[:3].detach().cpu().numpy()) for c in cameras}
                verified = camera_after == camera_before and image_after == image_before
                result['camera_integrity'] = dict(verified=verified, image_array_sha256_after=image_after)
                dump(output/'CAMERAS_AFTER.json', camera_after)
                if not verified:
                    integrity_errors.append('Camera matrix/image tensors changed')
        except Exception as exc:
            result['camera_integrity'] = dict(verified=False, error=str(exc))
            integrity_errors.append(f'Camera check unavailable: {exc}')
        if torch is not None and torch.cuda.is_initialized():
            try:
                result['GPU_memory'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), final_allocated_bytes=torch.cuda.memory_allocated(),
                    final_reserved_bytes=torch.cuda.memory_reserved())
                if result['GPU_memory']['peak_allocated_bytes'] > GPU_LIMIT:
                    integrity_errors.append('GPU allocated peak exceeds limit')
            except Exception as exc:
                integrity_errors.append(f'GPU memory record unavailable: {exc}')
        result['integrity_errors'] = integrity_errors
        result['integrity_complete'] = bool(protected) and len(after) == len(protected) and not integrity_errors and \
            all(result['model_integrity'].get(arm, {}).get('verified', False) for arm in ARMS) and \
            bool(result.get('camera_integrity') and result['camera_integrity']['verified'])
        if not result['integrity_complete']:
            result['original_decision_before_integrity_failure'] = result['decision']
            result['decision'] = 'INCONCLUSIVE'
        result['measured_completed'] = len(rows)
        result['stage'] = 'terminal'
        config['input_hashes_before'] = protected
        outputs = {str(path.relative_to(output)): dict(bytes=path.stat().st_size, sha256=sha(path))
            for path in sorted(output.rglob('*')) if path.is_file() and path.name not in ('MANIFEST.json', 'RESULTS.json', 'CONFIG.json')}
        dump(output/'MANIFEST.json', dict(input_hashes_before=protected, input_hashes_after=after, outputs=outputs))
        result['runner_wall_seconds_before_final_save'] = time.monotonic()-started
        save_partial()
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['decision'] == 'BENCHMARK_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-check', action='store_true', help='CPU plan/reduction witness only; no scientific inputs')
    parser.add_argument('--output-dir', type=Path, default=PROFILE/'run-v1')
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check()), flush=True)
        return 0
    return benchmark(args.output_dir)


if __name__ == '__main__':
    sys.exit(main())
