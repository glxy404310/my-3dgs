"""Frozen native evaluation of the pinned official NanoGS default output.

Stdlib-only import/--help. Running either mode requires the original CUDA venv.
No training, simplification, optimizer, backward, or original-model writes.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from shared_coordinate_native_v3 import (
    DEFAULT_REPO, DEFAULT_SOURCE, FIELDS, _check_import_path, _model_tensors,
    assert_frozen, model_hash, sha,
)

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT / 'research/experiments/nanogs-frozen-baseline-v0'
AUTHOR_COMMIT = 'cc64a752b5def8d17f9992216bfbb9e533f8364f'
AUTHOR_ROOT = PROJECT / 'third_party/NanoGS' / AUTHOR_COMMIT
CONTRACT_SHA = '41140ba8c045b81c1ce54512bbc0bf2f671d35125578e215633c099de9a88062'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
INDICES_SHA = 'f6c9f4da05e064db5c94134c8ee241eb2188fb83a5c3f860655d2eaf279f992f'
NANOGS_SHA = '383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8'
CPU_RESULT_SHA = '5e9f4a1b120a50a708865bb4a242b0c9277eca08f0f2fc4ff86ce85570a27f27'
COUNT, TARGET = 2050770, 1025385
WALL_LIMIT, GPU_LIMIT = 300, 10 * 1024 ** 3
PANELS = ('000016.jpg', '000017.jpg', '000105.jpg')
APP_NAMES = [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def array_hash(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def camera_record(camera):
    record = {name: getattr(camera, name) for name in
              ('image_name', 'uid', 'image_width', 'image_height', 'FoVx', 'FoVy', 'znear', 'zfar')}
    for name in ('R', 'T', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
        value = getattr(camera, name)
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        record[name] = {'sha256': array_hash(value), 'values': value.tolist()}
    return record


def model_identity(model):
    assert_frozen(model)
    return {'sha256': model_hash(model),
            'tensor_ids': {name: id(getattr(model, name)) for name in FIELDS},
            'point_count': int(model._xyz.shape[0]),
            'active_sh_degree': model.active_sh_degree, 'max_sh_degree': model.max_sh_degree}


def freeze(model):
    for tensor in _model_tensors(model):
        tensor.requires_grad_(False)
    assert_frozen(model)
    require(model.active_sh_degree == model.max_sh_degree == 3, 'Expected native SH3')
    return model


def make_identity(teacher, ply, GaussianModel):
    """Exact author's read/store attribute mapping, without serializing identity PLY."""
    import numpy as np
    import torch
    src = (AUTHOR_ROOT / 'src').resolve()
    for name, module in tuple(sys.modules.items()):
        if name == 'nanogs' or name.startswith('nanogs.'):
            filename = getattr(module, '__file__', None)
            if filename is not None:
                require(Path(filename).resolve().is_relative_to(src), 'Foreign NanoGS module already imported')
            else:
                require(all(Path(p).resolve().is_relative_to(src) for p in module.__path__),
                        'Foreign NanoGS namespace already imported')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from nanogs.utils.ply_utils import read_ply
    from nanogs.utils.splat_utils import logit
    header, mu, op, sc, q, sh, app_names = read_ply(str(ply))
    require(header.vertex_count == COUNT and app_names == APP_NAMES, 'Author input schema/SH mapping mismatch')
    arrays = {'_xyz': mu, '_opacity': logit(op).astype(np.float32)[:, None],
              '_scaling': np.log(np.maximum(sc, 1e-12)).astype(np.float32), '_rotation': q,
              '_features_dc': sh[:, :3].reshape(COUNT, 1, 3),
              '_features_rest': sh[:, 3:].reshape(COUNT, 3, 15).transpose(0, 2, 1)}
    model = GaussianModel(3, 'default')
    model.active_sh_degree = 3
    differences = {}
    for name in FIELDS:
        value = torch.from_numpy(np.ascontiguousarray(arrays[name])).to(teacher._xyz.device)
        require(bool(torch.isfinite(value).all()), f'Nonfinite identity field: {name}')
        differences[name] = float((value - getattr(teacher, name)).abs().max())
        if name in ('_xyz', '_features_dc', '_features_rest'):
            require(torch.equal(value, getattr(teacher, name)), f'Author unchanged mapping mismatch: {name}')
        setattr(model, name, torch.nn.Parameter(value, requires_grad=False))
    return freeze(model), {'raw_parameter_max_abs_difference': differences,
                           'opacity_lower_clip_count': int((op < 1e-6).sum()),
                           'opacity_upper_clip_count': int((op > 1 - 1e-6).sum()),
                           'scale_writer_floor_count': int((sc < 1e-12).sum()),
                           'conversion_difference_is_descriptive_not_a_gate': True}


def make_subset(teacher, indices, GaussianModel):
    import torch
    keep = torch.as_tensor(indices, device=teacher._xyz.device, dtype=torch.long)
    model = GaussianModel(3, 'default')
    model.active_sh_degree = 3
    for name in FIELDS:
        value = getattr(teacher, name).index_select(0, keep).detach().clone()
        setattr(model, name, torch.nn.Parameter(value, requires_grad=False))
    return freeze(model)


def summarize(rows):
    result = {}
    for arm in sorted({row['arm'] for row in rows}):
        selected = [row for row in rows if row['arm'] == arm and row['split'] == 'test']
        require(len(selected) == 32, f'Incomplete test set for {arm}')
        mean_mse = sum(row['gt_mse'] for row in selected) / 32
        tail = sorted(selected, key=lambda row: (-row['gt_mse'], row['view']))[:4]
        fidelity_tail = sorted(selected, key=lambda row: (-row['teacher_mse'], row['view']))[:4]
        result[arm] = {
            'test_count': 32, 'mean_gt_mse': mean_mse,
            'mean_gt_psnr': sum(row['gt_psnr'] for row in selected) / 32,
            'psnr_from_mean_gt_mse': -10 * math.log10(max(mean_mse, 1e-30)),
            'mean_gt_ssim': sum(row['gt_ssim'] for row in selected) / 32,
            'gt_tail4': tail, 'gt_tail4_mean_mse': sum(row['gt_mse'] for row in tail) / 4,
            'gt_worst': tail[0],
            'mean_teacher_mse': sum(row['teacher_mse'] for row in selected) / 32,
            'teacher_tail4': fidelity_tail,
            'teacher_tail4_mean_mse': sum(row['teacher_mse'] for row in fidelity_tail) / 4,
            'teacher_worst': fidelity_tail[0],
            'train_000016': next(row for row in rows if row['arm'] == arm and row['split'] == 'train'),
        }
    return result


def write_panels(out):
    import numpy as np
    from PIL import Image, ImageDraw
    folder = out / 'panels'
    folder.mkdir()
    for view in PANELS:
        arrays = []
        for arm in ('GT', 'teacher', 'all219', 'nanogs'):
            with np.load(out / 'raw' / arm / (view + '.npz'), allow_pickle=False) as saved:
                arrays.append(saved['rgb'])
        for tag, height in (('full', 546), ('topband_x0_y0_w979_h220', 220)):
            canvas = Image.new('RGB', (979 * 4, height + 28), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            for i, (arm, array) in enumerate(zip(('GT', 'teacher', 'all219', 'NanoGS'), arrays)):
                pixels = np.rint(np.clip(array[:, :height, :979], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
                canvas.paste(Image.fromarray(pixels), (i * 979, 28))
                draw.text((i * 979 + 8, 7), f'{view} | {arm}', fill=(0, 0, 0))
            canvas.save(folder / f'{view}.{tag}.png')


def evaluate(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'decision': 'INCOMPLETE', 'mode': args.mode, 'integrity_complete': False,
              'forward_calls': 0, 'optimizer_steps': 0, 'backward_calls': 0,
              'stage_wall_limit_seconds': WALL_LIMIT, 'gpu_allocated_limit_bytes': GPU_LIMIT,
              'test_set_previously_used': True, 'quality_acceptance_gate': None}
    rows, manifest, history_checks, models = [], [], [], {}
    timer = None
    torch = None
    prior_wall = 0.0

    def flush():
        result['stage_wall_seconds'] = time.monotonic() - started
        result['combined_evaluation_wall_seconds'] = prior_wall + result['stage_wall_seconds']
        dump(out / 'RESULTS.json', result)
        dump(out / 'RAW_MANIFEST.json', manifest)
        dump(out / 'HISTORICAL_REPLAY.json', history_checks)
        if rows:
            with (out / 'METRICS.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def timeout():
        result['decision'] = 'STOP_WALL_BUDGET'
        result['integrity_complete'] = False
        flush()
        os._exit(2)

    def budget():
        require(time.monotonic() - started + prior_wall <= WALL_LIMIT, 'Combined identity/complete wall budget exceeded')
        if torch is not None:
            require(torch.cuda.max_memory_allocated() <= GPU_LIMIT, 'CUDA allocated memory budget exceeded')

    def save_array(arm, view, value):
        path = out / 'raw' / arm / (view + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, rgb=value)
        manifest.append({'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': sha(path),
                         'dtype': str(value.dtype), 'shape': list(value.shape), 'stage': 'native_final_clamped_rgb'})

    try:
        timer = threading.Timer(WALL_LIMIT, timeout)
        timer.daemon = True
        timer.start()
        contract = EXPERIMENT / 'CONTRACT.md'
        assumptions = EXPERIMENT / 'EVALUATION_ASSUMPTIONS.md'
        root = DEFAULT_REPO / 'output/truck_30000_r1'
        ply = root / 'point_cloud/iteration_30000/point_cloud.ply'
        cache = DEFAULT_SOURCE / 'sparse/0/points3D.ply'
        indices_path = PROJECT / 'research/experiments/truck30k-coverage-controls-v0/run/all219_indices.npy'
        historical = PROJECT / 'research/experiments/pruned-opacity-transfer-v0/run'
        coverage_path = PROJECT / 'research/experiments/truck30k-coverage-controls-v0/run/METRICS.csv'
        inputs = [contract, assumptions, Path(__file__), Path(__file__).with_name('shared_coordinate_native_v3.py'),
                  ply, cache, indices_path, coverage_path, historical / 'CONFIG.json',
                  historical / 'RAW_MANIFEST.json', historical / 'METRICS.csv', AUTHOR_ROOT / 'SOURCE_MANIFEST.json']
        for name in ('cameras.bin', 'images.bin'):
            inputs.append(DEFAULT_SOURCE / 'sparse/0' / name)
        before = {str(path): sha(path) for path in inputs}
        require(before[str(contract)] == CONTRACT_SHA, 'Locked contract hash mismatch')
        require(before[str(ply)] == TEACHER_SHA and before[str(indices_path)] == INDICES_SHA, 'Original input hash mismatch')
        result['model_files'] = {'teacher': {'path': str(ply), 'sha256': TEACHER_SHA,
                                            'file_bytes': ply.stat().st_size, 'point_count': COUNT},
                                 'all219': {'indices_path': str(indices_path), 'indices_sha256': INDICES_SHA,
                                            'point_count': TARGET, 'PLY_written': False}}
        author_manifest = read_json(AUTHOR_ROOT / 'SOURCE_MANIFEST.json')
        require(author_manifest['commit'] == AUTHOR_COMMIT, 'Wrong author commit')
        for entry in author_manifest['downloaded_author_files']:
            path = AUTHOR_ROOT / entry['path']
            require(sha(path) == entry['sha256'] and entry['exact'], f'Author source mismatch: {path}')
            before[str(path)] = entry['sha256']
        old_config = read_json(historical / 'CONFIG.json')
        require(old_config['input_hashes_before']['teacher'] == TEACHER_SHA and
                old_config['input_hashes_before']['indices'] == INDICES_SHA, 'History uses different input')
        source_before = {path: sha(path) for path in old_config['source_hashes_before']}
        require(source_before == old_config['source_hashes_before'], 'Native source/binary differs from historical replay')
        old_manifest = {entry['path']: entry for entry in read_json(historical / 'RAW_MANIFEST.json')}
        old_metrics = {row['view']: row for row in read_csv(historical / 'METRICS.csv') if row['state'] == 'base'}
        coverage = {row['view']: row for row in read_csv(coverage_path) if row['arm'] == 'all219'}
        require(len(old_metrics) == 32, 'Expected 32 historical test rows')
        prior_identity = None
        if args.mode == 'complete':
            require(sha(args.cpu_results) == CPU_RESULT_SHA, 'CPU result differs from completed pinned run')
            cpu = read_json(args.cpu_results)
            require(cpu['decision'] == 'OFFICIAL_SIMPLIFICATION_COMPLETE_QUALITY_UNEVALUATED' and
                    cpu['integrity_complete'] and cpu['child_exit_code'] == 0 and
                    not cpu['timeout'] and not cpu['memory_stop'], 'CPU simplification not valid and complete')
            validity = cpu['output_model']
            require(validity['vertex_count'] == TARGET and validity['property_count'] == 62 and
                    validity['count_equals_nominal_target'] and validity['positive_finite_scale'] and
                    validity['quaternion_norm_max_error'] <= 1e-4 and
                    not any(validity['nonfinite_by_property'].values()), 'CPU output failed validity checks')
            require(sha(args.nanogs_ply) == validity['sha256'] == NANOGS_SHA, 'Independent NanoGS output hash mismatch')
            for path in (args.cpu_results, args.nanogs_ply):
                before[str(path)] = sha(path)
            prior_identity = read_json(args.identity_dir / 'RESULTS.json')
            require(prior_identity['decision'] == 'IDENTITY_EVALUATED' and prior_identity['integrity_complete'],
                    'A complete identity-only result is required')
            require(prior_identity['mode'] == 'identity-only', 'Identity result has wrong mode')
            prior_wall = float(prior_identity['stage_wall_seconds'])
            require(math.isfinite(prior_wall) and 0 < prior_wall < WALL_LIMIT, 'Invalid identity stage wall')
            budget()
            timer.cancel()
            timer = threading.Timer(WALL_LIMIT - prior_wall - (time.monotonic() - started), timeout)
            timer.daemon = True
            timer.start()
            result['prior_identity_stage_wall_seconds'] = prior_wall
            result['current_stage_wall_limit_seconds'] = WALL_LIMIT - prior_wall
            prior_config = read_json(args.identity_dir / 'CONFIG.json')
            for path, digest in prior_config['input_hashes_before'].items():
                require(sha(path) == digest, f'Identity input has changed: {path}')
            for filename in ('RESULTS.json', 'CONFIG.json', 'RAW_MANIFEST.json', 'METRICS.csv'):
                path = args.identity_dir / filename
                before[str(path)] = sha(path)
            for entry in read_json(args.identity_dir / 'RAW_MANIFEST.json'):
                path = args.identity_dir / entry['path']
                require(sha(path) == entry['sha256'], f'Identity raw artifact changed: {path}')
                before[str(path)] = entry['sha256']
            result['identity_reused'] = {'directory': str(args.identity_dir),
                                        'result_sha256': sha(args.identity_dir / 'RESULTS.json'),
                                        'summary': prior_identity['summary'],
                                        'conversion': prior_identity['conversion']}
            result['cpu_output_model'] = validity
            result['model_files']['nanogs'] = {'path': str(args.nanogs_ply), 'sha256': NANOGS_SHA,
                                               'file_bytes': args.nanogs_ply.stat().st_size, 'point_count': TARGET}

        import numpy as np
        import torch as torch_module
        torch = torch_module
        torch.set_grad_enabled(False)
        require(torch.cuda.is_available(), 'Original CUDA native environment required')
        torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_LIMIT / torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, DEFAULT_REPO)
        if str(DEFAULT_REPO) not in sys.path:
            sys.path.insert(0, str(DEFAULT_REPO))
        from scene import Scene, GaussianModel
        from gaussian_renderer import render
        from utils.loss_utils import ssim
        import diff_gaussian_rasterization as rasterizer
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, DEFAULT_REPO)
        for module in (rasterizer, rasterizer._C):
            path = str(Path(module.__file__).resolve())
            require(path in source_before and sha(path) == source_before[path], 'Loaded rasterizer ABI differs from history')
        for filename in ('utils/loss_utils.py', 'utils/graphics_utils.py', 'submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h'):
            path = str(DEFAULT_REPO / filename)
            source_before[path] = sha(path)
        dataset = SimpleNamespace(source_path=str(DEFAULT_SOURCE), model_path=str(root), images='images', depths='',
                                  resolution=1, white_background=False, train_test_exp=False,
                                  data_device='cpu', eval=True, sh_degree=3)
        with torch.no_grad():
            teacher = GaussianModel(3, 'default')
            scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False, resolution_scales=[1.0])
            freeze(teacher)
            require(teacher._xyz.shape[0] == COUNT, 'Wrong teacher count')
            tests = sorted(scene.getTestCameras(), key=lambda camera: camera.image_name)
            train = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
            require(len(tests) == 32 and len(train) == 219, 'Unexpected official camera split')
            require([camera.image_name for camera in tests] == sorted(old_metrics), 'Official test identities differ from historical 32')
            anchor = [camera for camera in train if camera.image_name == '000016.jpg']
            require(len(anchor) == 1, 'Training anchor missing/ambiguous')
            cameras = tests + anchor
            require(all((camera.image_width, camera.image_height) == (979, 546) for camera in cameras), 'Image size mismatch')
            camera_before = [camera_record(camera) for camera in cameras]
            for camera in cameras:
                photo = DEFAULT_SOURCE / 'images' / camera.image_name
                before[str(photo)] = sha(photo)
            camera_digest = hashlib.sha256(json.dumps(camera_before, sort_keys=True).encode()).hexdigest()
            if prior_identity:
                require(prior_identity['camera_identity_sha256'] == camera_digest, 'Identity camera transforms differ')
                require(prior_config['source_hashes_before'] == source_before, 'Identity native sources differ')
            result['camera_identity_sha256'] = camera_digest
            dump(out / 'CAMERAS.json', camera_before)
            config = {'mode': args.mode, 'input_hashes_before': before, 'source_hashes_before': source_before,
                      'camera_identity_sha256': camera_digest, 'test_views': [camera.image_name for camera in tests],
                      'training_anchor': '000016.jpg', 'image_size': [979, 546],
                      'renderer': {'SH': 3, 'background': [0, 0, 0], 'AA': False, 'exposure': False, 'native_final_clamp': True},
                      'panel_views': PANELS, 'crop_xywh': [0, 0, 979, 220],
                      'history_tolerances': {'max_rgb': 2e-5, 'mse': 1e-9, 'psnr': 2e-4, 'ssim': 2e-5},
                      'stage_wall_limit_seconds': WALL_LIMIT, 'gpu_allocated_limit_bytes': GPU_LIMIT}
            dump(out / 'CONFIG.json', config)
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            teacher_cache = {}
            gt_hashes = {}
            indices = np.load(indices_path, allow_pickle=False)
            require(indices.ndim == 1 and len(indices) == TARGET and np.issubdtype(indices.dtype, np.integer) and
                    len(np.unique(indices)) == TARGET and int(indices.min()) >= 0 and int(indices.max()) < COUNT,
                    'Invalid saved all219 subset')
            indices = np.sort(indices)

            def historical_array(arm, view):
                relative = f'raw/{arm}/{view}.npz'
                entry = old_manifest[relative]
                path = historical / relative
                require(sha(path) == entry['sha256'], f'Historical raw hash mismatch: {relative}')
                before[str(path)] = entry['sha256']
                with np.load(path, allow_pickle=False) as saved:
                    return saved['rgb']

            arms = ('teacher', 'identity') if args.mode == 'identity-only' else ('teacher', 'all219', 'nanogs')
            for arm in arms:
                budget()
                if arm == 'teacher':
                    model = teacher
                elif arm == 'identity':
                    model, result['conversion'] = make_identity(teacher, ply, GaussianModel)
                elif arm == 'all219':
                    model = make_subset(teacher, indices, GaussianModel)
                else:
                    model = GaussianModel(3, 'default')
                    model.load_ply(str(args.nanogs_ply), use_train_test_exp=False)
                    freeze(model)
                    require(model._xyz.shape[0] == TARGET, 'Native NanoGS count mismatch')
                    require(all(bool(torch.isfinite(getattr(model, name)).all()) for name in FIELDS), 'Nonfinite native NanoGS field')
                identity_before = model_identity(model)
                if arm == 'teacher' and prior_identity:
                    require(identity_before['sha256'] == prior_identity['models']['teacher']['before']['sha256'],
                            'Teacher raw parameters differ from the identity run')
                models[arm] = {'before': identity_before}
                for camera in cameras:
                    budget()
                    view = camera.image_name
                    gt = camera.original_image[:3].to('cuda')
                    result['forward_calls'] += 1
                    rgb = render(camera, model, pipe, background, scaling_modifier=1.0,
                                 separate_sh=False, use_trained_exp=False)['render']
                    require(rgb.dtype == torch.float32 and tuple(rgb.shape) == (3, 546, 979) and
                            bool(torch.isfinite(rgb).all()) and float(rgb.min()) >= 0 and float(rgb.max()) <= 1,
                            'Native final float RGB invalid')
                    rgb_cpu = rgb.detach().cpu().numpy()
                    gt_cpu = gt.detach().cpu().numpy()
                    if arm == 'teacher':
                        teacher_cache[view] = rgb_cpu.copy()
                        gt_hashes[view] = array_hash(gt_cpu)
                        save_array('GT', view, gt_cpu)
                    else:
                        require(array_hash(gt_cpu) == gt_hashes[view], 'GT changed between arms')
                    target = torch.from_numpy(teacher_cache[view]).to('cuda')
                    mse = (rgb - gt).square().mean()
                    row = {'arm': arm, 'split': 'train' if view == '000016.jpg' else 'test', 'view': view,
                           'gt_mse': float(mse), 'gt_psnr': float(-10 * torch.log10(mse.clamp_min(1e-30))),
                           'gt_ssim': float(ssim(rgb, gt)), 'teacher_mse': float((rgb - target).square().mean()),
                           'teacher_max_abs_rgb': float((rgb - target).abs().max())}
                    rows.append(row)
                    save_array(arm, view, rgb_cpu)
                    if arm in ('teacher', 'all219'):
                        old = coverage[view]
                        prefix = 'teacher' if arm == 'teacher' else 'student'
                        check = {'arm': arm, 'view': view,
                                 'coverage_psnr_abs_error': abs(row['gt_psnr'] - float(old[prefix + '_psnr_gt'])),
                                 'coverage_ssim_abs_error': abs(row['gt_ssim'] - float(old[prefix + '_ssim_gt']))}
                        if arm == 'all219':
                            check['coverage_teacher_mse_abs_error'] = abs(row['teacher_mse'] - float(old['mse_ts']))
                        if row['split'] == 'test':
                            old_rgb = historical_array('teacher' if arm == 'teacher' else 'base', view)
                            check['historical_max_abs_rgb'] = float(np.max(np.abs(rgb_cpu - np.clip(old_rgb, 0, 1))))
                            if arm == 'teacher':
                                require(np.array_equal(gt_cpu, historical_array('GT', view)), 'Historical GT changed')
                            else:
                                check['historical_gt_mse_abs_error'] = abs(row['gt_mse'] - float(old_metrics[view]['gt_mse']))
                                check['historical_teacher_mse_abs_error'] = abs(row['teacher_mse'] - float(old_metrics[view]['teacher_student_mse']))
                        history_checks.append(check)
                        require(check['coverage_psnr_abs_error'] <= 2e-4 and check['coverage_ssim_abs_error'] <= 2e-5,
                                f'Historical coverage metrics mismatch: {arm}/{view}')
                        require(check.get('historical_max_abs_rgb', 0) <= 2e-5, f'Historical RGB replay mismatch: {arm}/{view}')
                        require(all(value <= 1e-9 for key, value in check.items() if key.endswith('mse_abs_error')),
                                f'Historical MSE mismatch: {arm}/{view}')
                    budget()
                models[arm]['after'] = model_identity(model)
                require(models[arm]['before'] == models[arm]['after'], f'Model parameters/tensor identities changed: {arm}')
                result['models'] = models
                flush()
                if arm != 'teacher':
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            require(model_identity(teacher) == models['teacher']['before'], 'Teacher changed while evaluating other arms')
            require([camera_record(camera) for camera in cameras] == camera_before, 'Camera transforms changed')
            require(all(array_hash(camera.original_image[:3].detach().cpu().numpy()) == gt_hashes[camera.image_name]
                        for camera in cameras), 'Source GT tensors changed')
            if args.mode == 'complete':
                write_panels(out)
            budget()
            # Include the historical files actually used, then check all protected inputs again.
            config['input_hashes_before'] = before
            dump(out / 'CONFIG.json', config)
            after = {path: sha(path) for path in before}
            source_after = {path: sha(path) for path in source_before}
            require(after == before, 'Protected source/model/data/artifact file changed')
            require(source_after == source_before, 'Native source/binary changed')
            result['input_hashes_after'] = after
            result['source_hashes_after'] = source_after
            result['gt_array_sha256'] = gt_hashes
            result['summary'] = summarize(rows)
            result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            if args.mode == 'identity-only':
                result['identity_max_abs_rgb_all33'] = max(row['teacher_max_abs_rgb'] for row in rows if row['arm'] == 'identity')
                result['identity_max_abs_rgb_test32'] = max(row['teacher_max_abs_rgb'] for row in rows
                                                         if row['arm'] == 'identity' and row['split'] == 'test')
            result['stage_timing_scope'] = 'Whole invocation including load, hashing, native rendering, metrics, and artifact I/O; not FPS.'
            budget()
            result['integrity_complete'] = True
            result['decision'] = 'IDENTITY_EVALUATED' if args.mode == 'identity-only' else 'FROZEN_BASELINE_EVALUATED'
        flush()
        return 0
    except Exception as error:
        result['decision'] = 'STOP_EVALUATION_ERROR'
        result['integrity_complete'] = False
        result['error'] = f'{type(error).__name__}: {error}'
        result['traceback'] = traceback.format_exc()
        if torch is not None and torch.cuda.is_initialized():
            result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        flush()
        print(result['traceback'], file=sys.stderr)
        return 2
    finally:
        if timer is not None:
            timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('identity-only', 'complete'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--nanogs-ply', type=Path, default=EXPERIMENT / 'run-v1/output.ply')
    parser.add_argument('--cpu-results', type=Path, default=EXPERIMENT / 'run-v1/RESULTS.json')
    parser.add_argument('--identity-dir', type=Path, default=EXPERIMENT / 'identity-only')
    args = parser.parse_args()
    args.nanogs_ply = args.nanogs_ply.resolve()
    args.cpu_results = args.cpu_results.resolve()
    args.identity_dir = args.identity_dir.resolve()
    return evaluate(args)


if __name__ == '__main__':
    raise SystemExit(main())
