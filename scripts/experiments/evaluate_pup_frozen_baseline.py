#!/usr/bin/env python3
"""One-shot native PUP evaluation. Import/help are stdlib-only; no implicit launch."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from shared_coordinate_native_v3 import DEFAULT_REPO, DEFAULT_SOURCE, FIELDS, _check_import_path, sha
from evaluate_nanogs_frozen_baseline import array_hash, camera_record, freeze, model_identity

PROJECT = Path(__file__).resolve().parents[2]
STAGE = PROJECT/'research/experiments/pup-frozen-baseline-v0'
EVALUATION = STAGE/'evaluation-v1'
OUTPUT = EVALUATION/'run-v1'
OLD = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
SCORE = STAGE/'score-v1/run-v1'
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
TEACHER = DEFAULT_REPO/'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
NATIVE_COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
SCORE_SHA = '3775aa13372c4af264e384f3996424b94ac64362df3044160358f1f5d05fad68'
SUPERVISOR_SHA = 'fb5bfcb5cd09d4aac53e0571b4726467a0346d0b6b97dfc24db51a047395327a'
PUP_SHA = '7581b48c11ec44f8577028ee77b0bf5c3361d5370f5dee1105c27d20c718107c'
EXTERNAL_SHA = '81ec1d44ff857ec5ebcba221f6c8643b66bf9382e441910a4555b2c61d3007a2'
COMPLETE_SCORE = 'OFFICIAL_FIXED_SCORE_EXPORT_COMPLETE_NO_QUALITY_CLAIM'
HOST_LIMIT, GPU_LIMIT = 8*1024**3, 4*1024**3
MAIN_LIMIT, INTERNAL_LIMIT, AUDIT_RESERVE = 200, 195, 100
ARMS = ('teacher', 'all219', 'nanogs', 'pup')
FIXED = ('000016.jpg', '000017.jpg', '000105.jpg')
SSIM_DEFINITION = 'CPU_float64_SciPy_Gaussian11_sigma1.5_constant0_population_CHWmean_C1.01sq_C2.03sq'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def dump(path, value):
    path = Path(path)
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def protected_hashes(files):
    for path, expected in files.items():
        need(sha(path) == expected, 'Protected input changed: '+path)


def preflight(args):
    mandatory = ((args.score_result, SCORE/'SCORE_RESULT.json', args.score_sha256, SCORE_SHA),
                 (args.score_supervisor, SCORE/'SUPERVISOR.json', args.supervisor_sha256, SUPERVISOR_SHA),
                 (args.pup_ply, SCORE/'point_cloud.ply', args.pup_sha256, PUP_SHA),
                 (args.score_external, STAGE/'score-v1/EXTERNAL_RUN.json', args.external_sha256, EXTERNAL_SHA))
    files = {}
    for actual, expected_path, supplied_sha, expected_sha in mandatory:
        need(actual.resolve() == expected_path.resolve() and supplied_sha == expected_sha,
             'Actual root-supplied score path/SHA required: '+str(expected_path))
        files[str(actual.resolve())] = expected_sha
    for relative, expected in read(EVALUATION/'READONLY_PINS.json').items():
        files[str(PROJECT/relative)] = expected
    for path in (Path(__file__).resolve(), STAGE/'EVALUATION_CONTRACT.md', EVALUATION/'ASSUMPTIONS.md',
                 EVALUATION/'READONLY_PINS.json', EVALUATION/'run-evaluation.sh'):
        files[str(path)] = sha(path)
    protected_hashes(files)
    score, supervisor, external = read(args.score_result), read(args.score_supervisor), read(args.score_external)
    need(score['status'] == supervisor['status'] == COMPLETE_SCORE and score['phase'] == 'complete' and
         score['file_integrity'] and score['camera_identity_unchanged'] and supervisor['input_integrity'] and
         supervisor['within_900_seconds'] and supervisor['automatic_retries'] == 0,
         'Actual full score/export and supervisor success required')
    need(score['forward_calls'] == 220 and score['statistical_backward_calls'] == 219 and
         score['training_steps'] == 0 and score['optimizer_constructed'] is False and
         len(score['scored_views']) == len(set(score['scored_views'])) == 219,
         'Original complete219 score provenance required')
    need(external['exit_code'] == 0 and external['attempts'] == 1 and external['within_900_seconds'] is True and
         external['hard_limit_seconds'] == 900 and 0 < external['actual_host_wall_seconds'] <= 900,
         'Actual external score budget success required')
    ply = score['ply']
    need(ply['status'] == 'ALL62_PROPERTIES_BYTE_EXACT' and len(ply['properties']) == 62 and
         all(item['exact'] for item in ply['properties'].values()) and
         Path(ply['path']).resolve() == args.pup_ply.resolve() and ply['sha256'] == PUP_SHA and
         ply['bytes'] == args.pup_ply.stat().st_size and
         1 <= ply['point_count'] == score['pruning']['kept_count'] <= 1025385,
         'All62 natural-mask survivor proof/count/bytes required')
    cfg, previous = read(OLD/'CONFIG.json'), read(OLD/'RESULTS.json')
    need(previous['decision'] == 'FROZEN_BASELINE_EVALUATED' and previous['integrity_complete'] and
         previous['input_hashes_after'] == cfg['input_hashes_before'] and
         previous['source_hashes_after'] == cfg['source_hashes_before'], 'Prior native cache provenance incomplete')
    views = cfg['test_views']+['000016.jpg']
    need(len(set(cfg['test_views'])) == 32 and len(set(views)) == 33 and
         cfg['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False, exposure=False, native_final_clamp=True),
         'Wrong pinned test split/native renderer settings')
    manifest = read(OLD/'RAW_MANIFEST.json')
    expected_paths = {f'raw/{arm}/{view}.npz' for arm in ('GT', 'teacher', 'all219', 'nanogs') for view in views}
    need(len(manifest) == 132 and {item['path'] for item in manifest} == expected_paths, 'Exact prior132-cache membership required')
    for item in manifest:
        path = OLD/item['path']
        need(item['dtype'] == 'float32' and item['shape'] == [3, 546, 979] and path.stat().st_size == item['bytes'],
             'Prior raw metadata mismatch: '+item['path'])
        files[str(path)] = item['sha256']
    files.update(cfg['source_hashes_before'])
    files[str(TEACHER)] = TEACHER_SHA
    score_inputs = read(SCORE/'INPUTS_BEFORE.json')
    # All251 source photographs and existing COLMAP inputs were pinned by the actual score.
    for path, expected in score_inputs['files'].items():
        if Path(path).is_relative_to(DEFAULT_SOURCE):
            files[path] = expected
    canonical = read(SCORE/'CANONICAL_NATIVE.json')
    need(canonical['status'] == 'CANONICAL_NATIVE_ONLY' and len(canonical['train_cameras']) == 219,
         'Complete canonical train-camera record required')
    head = subprocess.check_output(['git', '-C', str(DEFAULT_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    need(head == NATIVE_COMMIT, 'Wrong native commit')
    subprocess.run(['git', '-C', str(DEFAULT_REPO), 'diff-index', '--quiet', 'HEAD', '--'], check=True)
    protected_hashes(files)
    return dict(protected_files=files, test_views=cfg['test_views'], training_anchor='000016.jpg',
                renderer=cfg['renderer'], image_size=[979, 546], prior_cache=str(OLD),
                old_SSIM_audit_status='FAIL_UNCHANGED', SSIM_definition=SSIM_DEFINITION,
                score_result_sha256=SCORE_SHA, score_supervisor_sha256=SUPERVISOR_SHA,
                score_external_sha256=EXTERNAL_SHA, pup_ply=dict(ply),
                native_commit=NATIVE_COMMIT, main_external_limit_seconds=MAIN_LIMIT,
                audit_external_reserve_seconds=AUDIT_RESERVE, combined_external_limit_seconds=300,
                host_RSS_limit_bytes=HOST_LIMIT, gpu_allocated_limit_bytes=GPU_LIMIT,
                optimizer_steps=0, backward_calls=0, quality_acceptance_gate=None)


def pixels(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        need(saved.files == ['rgb'], 'Exactly one rgb NPZ member required')
        value = saved['rgb']
    need(value.dtype == np.float32 and value.shape == (3, 546, 979) and
         np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1, 'Invalid raw CHW float32 RGB: '+str(path))
    return value


def cpu_metric_functions():
    import numpy as np
    from scipy.ndimage import convolve1d
    weights = np.exp(-np.arange(-5, 6, dtype=np.float64)**2/(2*1.5**2))
    weights /= weights.sum()
    def filt(a):
        return convolve1d(convolve1d(a, weights, axis=1, mode='constant', cval=0),
                          weights, axis=2, mode='constant', cval=0)
    def metric(value, gt, teacher, mb, vb):
        a = value.astype(np.float64)
        mse = float(np.mean((a-gt)**2, dtype=np.float64))
        ma = filt(a)
        va, cov = filt(a*a)-ma*ma, filt(a*gt)-ma*mb
        ssim = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
        return dict(gt_mse=mse, gt_psnr=-10*math.log10(mse) if mse > 0 else None,
                    zero_MSE_infinite_PSNR=mse == 0, gt_ssim=float(np.mean(ssim, dtype=np.float64)),
                    teacher_mse=float(np.mean((a-teacher)**2, dtype=np.float64)),
                    teacher_max_abs_rgb=float(np.max(np.abs(a-teacher))))
    return filt, metric


def summary(rows):
    output = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm and r['split'] == 'test']
        need(len(selected) == 32, 'Incomplete32 test means')
        tail = sorted(selected, key=lambda r: (-r['gt_mse'], r['view']))[:4]
        mean = math.fsum(r['gt_mse'] for r in selected)/32
        output[arm] = dict(test_count=32, mean_gt_mse=mean,
            mean_gt_psnr=math.fsum(r['gt_psnr'] for r in selected)/32 if all(r['gt_psnr'] is not None for r in selected) else None,
            psnr_from_mean_gt_mse=-10*math.log10(mean) if mean > 0 else None,
            mean_gt_ssim=math.fsum(r['gt_ssim'] for r in selected)/32,
            mean_teacher_mse=math.fsum(r['teacher_mse'] for r in selected)/32,
            own_GT_tail4=tail, own_GT_tail4_mean_MSE=math.fsum(r['gt_mse'] for r in tail)/4,
            train_000016=next(r for r in rows if r['arm'] == arm and r['split'] == 'train'))
    return output


def signed_differences(summaries):
    keys = ('mean_gt_mse', 'mean_gt_psnr', 'psnr_from_mean_gt_mse', 'mean_gt_ssim', 'mean_teacher_mse')
    return {f'pup_minus_{reference}': {key: summaries['pup'][key]-summaries[reference][key]
            if summaries['pup'][key] is not None and summaries[reference][key] is not None else None for key in keys}
            for reference in ('all219', 'teacher')}


def rgb_image(array):
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.rint(array.transpose(1, 2, 0)*255).astype(np.uint8))


def panel(arrays, view, height, tag, path):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (979*5, height+28), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(zip(('GT', 'teacher', 'all219', 'NanoGS', 'PUP'), arrays)):
        canvas.paste(rgb_image(array[:, :height, :]), (i*979, 28))
        draw.text((i*979+8, 7), f'{view} | {label} | {tag}', fill='black')
    canvas.save(path)


def worker():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    cfg = read(OUTPUT/'CONFIG.json')
    need(os.environ.get('PUP_EVALUATION_WORKER') == '1' and torch.cuda.is_available(), 'Supervisor-only native CUDA worker required')
    torch.cuda.set_per_process_memory_fraction(min(1., GPU_LIMIT/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    for name in ('scene', 'utils', 'gaussian_renderer'):
        _check_import_path(name, DEFAULT_REPO)
    sys.path.insert(0, str(DEFAULT_REPO))
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    import diff_gaussian_rasterization as rasterizer
    for name in ('scene', 'utils', 'gaussian_renderer'):
        _check_import_path(name, DEFAULT_REPO)
    for module in (rasterizer, rasterizer._C):
        path = str(Path(module.__file__).resolve())
        need(path in cfg['protected_files'] and sha(path) == cfg['protected_files'][path], 'Rasterizer ABI provenance mismatch')
    result = dict(status='INCOMPLETE', integrity_complete=False, forward_calls=0, backward_calls=0,
                  optimizer_steps=0, old_SSIM_audit_status='FAIL_UNCHANGED', quality_acceptance_gate=None,
                  independent_audit_status='NOT_RUN', SSIM_definition=SSIM_DEFINITION)
    rows, raw_manifest = [], []
    def flush():
        dump(OUTPUT/'RESULTS.json', result)
        dump(OUTPUT/'RAW_MANIFEST.json', raw_manifest)
    def gpu_check():
        peak = torch.cuda.max_memory_allocated()
        need(peak <= GPU_LIMIT, 'CUDA allocated4GiB budget exceeded')
        result['gpu_peak_allocated_bytes'] = peak
    try:
        flush()
        dataset = SimpleNamespace(source_path=str(DEFAULT_SOURCE), model_path=str(TEACHER.parents[2]),
            images='images', depths='', resolution=1, white_background=False, train_test_exp=False,
            data_device='cpu', eval=True, sh_degree=3)
        with torch.no_grad():
            model = GaussianModel(3, 'default')
            scene = Scene(dataset, model, load_iteration=30000, shuffle=False, resolution_scales=[1.0])
            need(len(model._xyz) == 2050770, 'Read-only teacher reconstruction count mismatch')
            train = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
            tests = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
            need(len(train) == 219 and len(tests) == 32 and [c.image_name for c in tests] == cfg['test_views'], 'Exact219/32 split required')
            canonical = {r['image_name']: r for r in read(SCORE/'CANONICAL_NATIVE.json')['train_cameras']}
            train_records = [camera_record(c) for c in train]
            for record in train_records:
                expected = canonical[record['image_name']]
                need(all(all(expected[k][subkey] == subvalue for subkey, subvalue in v.items())
                         if isinstance(v, dict) else expected[k] == v for k, v in record.items()),
                     'Full219 canonical camera mismatch: '+record['image_name'])
                need(sha(expected['source_photo']) == expected['source_photo_sha256'], 'Canonical source photo changed')
            anchors = [c for c in train if c.image_name == '000016.jpg']
            need(len(anchors) == 1, 'Fixed train000016 anchor missing')
            cameras = tests+anchors
            records = [camera_record(c) for c in cameras]
            need(records == read(OLD/'CAMERAS.json'), 'Exact33 old camera records differ')
            need(all((c.image_width, c.image_height) == (979, 546) for c in train+tests), 'Native dimensions differ')
            dump(OUTPUT/'CAMERAS.json', dict(train=train_records, evaluation=records, test_views=cfg['test_views']))
            model.load_ply(cfg['pup_ply']['path'], use_train_test_exp=False)
            freeze(model)
            need(len(model._xyz) == cfg['pup_ply']['point_count'] and
                 all(bool(torch.isfinite(getattr(model, name)).all()) for name in FIELDS), 'PUP count/raw fields invalid')
            field_records = {}
            for name in FIELDS:
                array = np.ascontiguousarray(getattr(model, name).detach().cpu().numpy())
                field_records[name] = dict(dtype=str(array.dtype), shape=list(array.shape),
                                           sha256=hashlib.sha256(array.tobytes()).hexdigest())
            need(field_records == read(SCORE/'SCORE_RESULT.json')['raw_fields_exported_subset'],
                 'Native PUP loader changed exported frozen raw fields')
            result['raw_fields_loaded'] = field_records
            before = model_identity(model)
            result['model_before'] = before
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            (OUTPUT/'raw/pup').mkdir(parents=True)
            (OUTPUT/'images/pup').mkdir(parents=True)
            for camera in cameras:
                gpu_check()
                view = camera.image_name
                gt = pixels(OLD/f'raw/GT/{view}.npz')
                need(np.array_equal(gt, camera.original_image[:3].detach().cpu().numpy()), 'Cached GT differs from canonical photo tensor')
                result['forward_calls'] += 1
                value = render(camera, model, pipe, background, scaling_modifier=1., separate_sh=False,
                               use_trained_exp=False)['render']
                need(value.dtype == torch.float32 and tuple(value.shape) == (3, 546, 979) and
                     bool(torch.isfinite(value).all()) and float(value.min()) >= 0 and float(value.max()) <= 1,
                     'Invalid native final clamp RGB')
                rgb = value.detach().cpu().numpy()
                path = OUTPUT/f'raw/pup/{view}.npz'
                np.savez(path, rgb=rgb)
                png = OUTPUT/f'images/pup/{view}.png'
                rgb_image(rgb).save(png)
                raw_manifest.append(dict(path=str(path.relative_to(OUTPUT)), sha256=sha(path), bytes=path.stat().st_size,
                    dtype='float32', shape=[3, 546, 979], array_sha256=array_hash(rgb),
                    png_path=str(png.relative_to(OUTPUT)), png_sha256=sha(png), stage='native_final_clamped_rgb'))
                del value, rgb
                gpu_check()
                flush()
            need(result['forward_calls'] == 33, 'Exactly33 PUP forward calls required')
            result['model_after'] = model_identity(model)
            need(result['model_before'] == result['model_after'], 'PUP parameters or tensor identities changed')
            need(train_records == [camera_record(c) for c in train] and records == [camera_record(c) for c in cameras], 'Camera mutation')
            gpu_check()
        # GPU work is finished. Every model metric below comes only from raw CPU float32 files.
        filt, metric = cpu_metric_functions()
        for view in cfg['test_views']+['000016.jpg']:
            gt = pixels(OLD/f'raw/GT/{view}.npz').astype(np.float64)
            teacher = pixels(OLD/f'raw/teacher/{view}.npz').astype(np.float64)
            mb = filt(gt)
            vb = filt(gt*gt)-mb*mb
            for arm in ARMS:
                root = OUTPUT if arm == 'pup' else OLD
                value = pixels(root/f'raw/{arm}/{view}.npz')
                rows.append(dict(arm=arm, split='train' if view == '000016.jpg' else 'test', view=view,
                                 **metric(value, gt, teacher, mb, vb)))
        with (OUTPUT/'METRICS.csv').open('x', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries = summary(rows)
        result['summary'] = summaries
        result['signed_summary_differences'] = signed_differences(summaries)
        table = {(r['arm'], r['view']): r for r in rows}
        differences = []
        for view in cfg['test_views']+['000016.jpg']:
            for reference in ('all219', 'teacher'):
                current, baseline = table['pup', view], table[reference, view]
                differences.append(dict(view=view, split=current['split'], reference=reference,
                    **{f'pup_minus_reference_{key}': current[key]-baseline[key]
                       if current[key] is not None and baseline[key] is not None else None
                       for key in ('gt_mse', 'gt_psnr', 'gt_ssim', 'teacher_mse')}))
        with (OUTPUT/'PUP_DIFFERENCES.csv').open('x', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(differences[0]))
            writer.writeheader()
            writer.writerows(differences)
        increases = [dict(view=view, pup_minus_all219_gt_mse=table['pup', view]['gt_mse']-table['all219', view]['gt_mse']) for view in cfg['test_views']]
        increases.sort(key=lambda r: (-r['pup_minus_all219_gt_mse'], r['view']))
        result['exploratory_top4_PUP_minus_all219_GT_MSE'] = increases[:4]
        panels = OUTPUT/'panels'
        panels.mkdir()
        for view in dict.fromkeys(FIXED+tuple(r['view'] for r in increases[:4])):
            arrays = [pixels(OLD/f'raw/{arm}/{view}.npz') for arm in ('GT', 'teacher', 'all219', 'nanogs')]
            arrays.append(pixels(OUTPUT/f'raw/pup/{view}.npz'))
            if view in FIXED:
                panel(arrays, view, 546, 'fixed', panels/f'{view}.fixed.full.png')
                panel(arrays, view, 220, 'fixed x0 y0 w979 h220', panels/f'{view}.fixed.top220.png')
            if view in {r['view'] for r in increases[:4]}:
                rank = next(i+1 for i, r in enumerate(increases[:4]) if r['view'] == view)
                panel(arrays, view, 546, f'exploratory GT-MSE increase rank{rank}', panels/f'{view}.exploratory_rank{rank}.full.png')
        previous = read(OLD/'RESULTS.json')
        result['model_files'] = dict(previous['model_files'], pup=cfg['pup_ply'])
        result['equal_point_count_claim_allowed'] = cfg['pup_ply']['point_count'] == previous['model_files']['all219']['point_count']
        protected_hashes(cfg['protected_files'])
        result['protected_hashes_after'] = cfg['protected_files']
        result['integrity_complete'] = True
        result['status'] = 'PUP_FROZEN_NATIVE_EVALUATED_AUDIT_PENDING'
        flush()
        return 0
    except BaseException as error:
        result['status'] = 'INCONCLUSIVE'
        result['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        flush()
        raise


def tree_rss():
    rows = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            data = dict(line.split(':', 1) for line in (entry/'status').read_text().splitlines() if ':' in line)
            rows[int(entry.name)] = (int(data['PPid']), int(data.get('VmRSS', '0 kB').split()[0])*1024)
        except (OSError, ValueError, KeyError, ProcessLookupError):
            pass
    family = {os.getpid()}
    while True:
        expanded = family | {pid for pid, (parent, _) in rows.items() if parent in family}
        if expanded == family:
            return sum(rows.get(pid, (0, 0))[1] for pid in family)
        family = expanded


def supervise(args):
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned WSL Python required')
    need(not OUTPUT.exists(), 'One-shot run-v1 must be absent; never retry')
    started = time.monotonic()
    active, peak_rss = None, 0
    record = dict(status='INCONCLUSIVE', automatic_retries=0, main_limit_seconds=MAIN_LIMIT,
                  audit_reserved_external_seconds=AUDIT_RESERVE, combined_external_limit_seconds=300,
                  host_RSS_limit_bytes=HOST_LIMIT, gpu_allocated_limit_bytes=GPU_LIMIT,
                  independent_audit_status='NOT_RUN', supervisor_torch_imported=False)
    OUTPUT.mkdir()
    dump(OUTPUT/'ATTEMPT_STARTED.json', dict(started_unix=time.time(), execution_attempt=1, pid=os.getpid()))
    def interrupted(number, _frame):
        raise TimeoutError('Main one-shot interrupted by signal '+str(number))
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.alarm(INTERNAL_LIMIT)
    try:
        cfg = preflight(args)
        dump(OUTPUT/'CONFIG.json', cfg)
        env = dict(os.environ, PYTHONPATH=str(DEFAULT_REPO), PYTHONDONTWRITEBYTECODE='1', PUP_EVALUATION_WORKER='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (OUTPUT/'WORKER.log').open('x', encoding='utf8') as log:
            active = subprocess.Popen([str(PYTHON), '-B', '-u', str(Path(__file__).resolve()), '--internal-worker'],
                cwd=OUTPUT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                peak_rss = max(peak_rss, tree_rss())
                need(peak_rss <= HOST_LIMIT, 'Aggregate host RSS exceeds8GiB')
                need(time.monotonic()-started < INTERNAL_LIMIT, 'Main195-second internal deadline')
                time.sleep(.1)
            record['worker_exit_code'] = active.returncode
        need(active.returncode == 0, 'Worker failed; original log/result retained')
        need(read(OUTPUT/'RESULTS.json')['status'] == 'PUP_FROZEN_NATIVE_EVALUATED_AUDIT_PENDING', 'Worker incomplete')
        record['status'] = 'PUP_FROZEN_NATIVE_EVALUATED_AUDIT_PENDING'
    except BaseException as error:
        record['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=1)
        record['main_internal_wall_seconds'] = time.monotonic()-started
        record['peak_sampled_combined_host_RSS_bytes'] = peak_rss
        record['within_200_seconds'] = record['main_internal_wall_seconds'] <= MAIN_LIMIT
        record['supervisor_torch_imported'] = 'torch' in sys.modules
        dump(OUTPUT/'SUPERVISOR.json', record)
    return 0 if record['status'] == 'PUP_FROZEN_NATIVE_EVALUATED_AUDIT_PENDING' else 1


def synthetic_check():
    import numpy as np
    import tempfile
    from PIL import Image
    filt, metric = cpu_metric_functions()
    value = np.linspace(0, 1, 3*13*17, dtype=np.float32).reshape(3, 13, 17)
    gt = value.astype(np.float64)
    mb = filt(gt)
    identical = metric(value, gt, gt, mb, filt(gt*gt)-mb*mb)
    need(identical['gt_mse'] == 0 and identical['gt_psnr'] is None and identical['gt_ssim'] == 1 and
         identical['teacher_mse'] == identical['teacher_max_abs_rgb'] == 0, 'Synthetic identity metrics')
    shifted = np.maximum(value-.1, 0).astype(np.float32)
    changed = metric(shifted, gt, gt, mb, filt(gt*gt)-mb*mb)
    need(changed['gt_mse'] > 0 and changed['gt_psnr'] > 0 and changed['gt_ssim'] < 1, 'Synthetic difference metrics')
    arrays = [np.full((3, 546, 979), i/5, dtype=np.float32) for i in range(5)]
    with tempfile.TemporaryDirectory(prefix='pup-evaluation-synthetic-') as temporary:
        for height in (546, 220):
            path = Path(temporary)/f'panel{height}.png'
            panel(arrays, 'synthetic.jpg', height, 'synthetic', path)
            with Image.open(path) as image:
                need(image.size == (4895, height+28) and
                     all(image.getpixel((i*979+400, 28+height-1)) == (i*51,)*3 for i in range(5)),
                     'Synthetic five-column pixel-preserving panel/crop failed')
    need('torch' not in sys.modules and 'scene' not in sys.modules, 'Synthetic check imported forbidden backend')
    return dict(status='SYNTHETIC_ARRAYS_ONLY_PASS', scientific_inputs_read=False, Scene_constructed=False,
                torch_imported=False, native_forward_calls=0, identical=identical, changed=changed,
                evaluator_sha256=sha(__file__), SSIM_definition=SSIM_DEFINITION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--synthetic-check', action='store_true')
    for name in ('score-result', 'score-supervisor', 'pup-ply', 'score-external'):
        parser.add_argument('--'+name, type=Path)
    for name in ('score-sha256', 'supervisor-sha256', 'pup-sha256', 'external-sha256'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check(), indent=2, allow_nan=False))
        return 0
    if args.internal_worker:
        need(os.environ.get('PUP_EVALUATION_WORKER') == '1', 'Worker has no standalone launch interface')
        return worker()
    need(all(getattr(args, name) is not None for name in ('score_result', 'score_supervisor', 'pup_ply',
         'score_external', 'score_sha256', 'supervisor_sha256', 'pup_sha256', 'external_sha256')),
         'All actual score paths and root-supplied hashes are mandatory')
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
