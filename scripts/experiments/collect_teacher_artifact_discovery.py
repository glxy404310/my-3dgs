#!/usr/bin/env python3
"""Collect twelve fixed discovery views, reusing three verified teacher caches."""
import time
PROCESS_START = time.perf_counter()

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

CONTRACT_SHA = 'd956919518fae6dbc2331f594a2fe86a01062823dac677c19fb2c3889dd2b07d'
HELPER_SHA = '2a089b7d85987b571bb53f22ce9a97912fe690e01fc2d9d950dead39fd3a29a3'
OLD_RESULTS_SHA = '8f8548dcde8677b1a40fbdd4fd82422b819380bea3970da88825dea39966f131'
PLY_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
PARAMETERS_SHA = 'b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0'
OLD_CAMERA_SHA = '835bb5cd31d88362c64b9136ef2e71cab55f8a884e71ea11597d8232df8d0262'
CACHE_SHA = ('736416b72c637a6d280c84dd48c76ad9f156e6880dc8a98bf1f81b0d48b74e25',
             '3afc9c00c608fc2ac6b61fe1907f46409d945ea64bc4082cf07465537ab173a0',
             'd9ecf4edafb1deec314dfe9d06dde661acf890990fb2fab1b6c1d5e46878aca1')
NAMES = tuple(f'{number:06d}.jpg' for number in (2, 23, 46, 69, 92, 115, 136, 159, 182, 205, 228, 251))
RANKS = (0, 19, 39, 59, 79, 99, 118, 138, 158, 178, 198, 218)
MATRIX_FIELDS = ('R', 'T', 'trans', 'scale', 'FoVx', 'FoVy', 'image_width', 'image_height',
                 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center')
STATE = None
FINALIZING = False
np = native = torch = None


class Inconclusive(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not bool(condition):
        raise Inconclusive(code, message)


def budget():
    require(time.perf_counter() - PROCESS_START < 300., 'INCONCLUSIVE_BUDGET', '300-second process budget exhausted')


def on_signal(signum, _frame):
    failure = {'code': 'INCONCLUSIVE_BUDGET', 'message': f'received signal {signum}'}
    if STATE is not None:
        STATE.setdefault('signals', []).append(failure)
        if STATE['failure'] is None:
            STATE['failure'] = failure
    if not FINALIZING:
        raise Inconclusive(failure['code'], failure['message'])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            budget()
            digest.update(block)
    return digest.hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    require(expected is None or value == expected, 'INCONCLUSIVE_IDENTITY', f'Hash mismatch: {path}')
    STATE['input_hashes_before'][str(path)] = value
    return value


def progress(stage, **fields):
    budget()
    record = {'stage': stage, 'elapsed_before_progress_write_seconds': time.perf_counter() - PROCESS_START, **fields}
    STATE['last_progress'] = record
    with (OUT / 'PROGRESS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(record), flush=True)


def array(camera, field):
    value = getattr(camera, field)
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def camera_hash(camera):
    digest = hashlib.sha256()
    digest.update(f'{camera.image_name}:{camera.uid}:{camera.colmap_id}\n'.encode('utf8'))
    for field in MATRIX_FIELDS + ('original_image', 'alpha_mask'):
        value = getattr(camera, field)
        if value is None:
            digest.update((field + ':None\n').encode('ascii'))
        else:
            values = np.ascontiguousarray(array(camera, field))
            digest.update(f'{field}:{values.dtype}:{values.shape}\n'.encode('ascii'))
            digest.update(values.tobytes())
    return digest.hexdigest()


def save_camera_matrices(cameras, phase):
    budget()
    values = {f'view{view:02d}__{field}': array(camera, field).copy()
              for view, camera in enumerate(cameras) for field in MATRIX_FIELDS}
    np.savez(OUT / 'raw' / f'cameras_{phase}.npz', **values)


def check_rgb(image, label):
    require(image.shape == (546, 979, 3) and image.dtype == np.float32
            and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1,
            'INCONCLUSIVE_RGB_SCHEMA', f'Expected original float32 HWC RGB: {label}')


def save_rgb(path, image):
    budget()
    check_rgb(image, str(path))
    with path.open('xb') as stream:
        np.save(stream, image, allow_pickle=False)
    budget()


def forward(bundle, camera, purpose):
    budget()
    require(STATE['new_forward_attempts'] < 10, 'INCONCLUSIVE_FORWARD_BUDGET', 'No more than ten new forwards permitted')
    STATE['new_forward_attempts'] += 1
    STATE['forward_attempt_log'].append({'view': camera.image_name, 'purpose': purpose})
    image = native.render_rgb(bundle, camera)['rgb'].permute(1, 2, 0).contiguous().numpy()
    STATE['new_forward_completed'] += 1
    check_rgb(image, camera.image_name)
    budget()
    return image


def save_pair(path, teacher, gt):
    from PIL import Image
    pixels = np.concatenate((np.rint(np.clip(teacher, 0, 1) * 255).astype(np.uint8),
                             np.rint(np.clip(gt, 0, 1) * 255).astype(np.uint8)), axis=1)
    Image.fromarray(pixels).save(path, compress_level=4)
    with Image.open(path) as saved:
        require(saved.size == (1958, 546) and np.array_equal(np.asarray(saved), pixels),
                'INCONCLUSIVE_DISPLAY_WRITE', f'PNG pixel/layout mismatch: {path}')


def prepare_sources(root, old_run):
    pin(root / 'research/experiments/teacher-artifact-discovery-v0/CONTRACT.md', CONTRACT_SHA)
    pin(root / 'research/experiments/teacher-artifact-discovery-v0/ASSUMPTIONS.md')
    pin(Path(__file__))
    pin(root / 'research/experiments/launchers/run-teacher-artifact-discovery.sh')
    pin(Path(__file__).parent / 'shared_coordinate_native_v3.py', HELPER_SHA)
    pin(old_run / 'RESULTS.json', OLD_RESULTS_SHA)
    old = json.loads((old_run / 'RESULTS.json').read_text(encoding='utf8'))
    require(old['parameters_before']['30000'] == PARAMETERS_SHA
            and old['parameters_after']['30000'] == PARAMETERS_SHA
            and old['source_ply_pins']['30000']['sha256'] == PLY_SHA
            and old['integrity']['complete'], 'INCONCLUSIVE_OLD_PROVENANCE', 'Old 30k identity/integrity incomplete')
    require(tuple(row['name'] for row in old['camera_metadata']['30000']) == NAMES,
            'INCONCLUSIVE_OLD_PROVENANCE', 'Old twelve camera name order differs')
    old_hashes = old['input_hashes_before']
    repo = Path('/home/administrator/projects/gaussian-splatting')
    source = Path('/home/administrator/datasets/tandt_db/tandt/truck')
    # These are source identity checks, not reused scientific masks or results.
    sources = [repo / name for name in native.NATIVE_SOURCE_FILES]
    sources += [source / 'sparse/0' / name for name in ('cameras.bin', 'images.bin', 'points3D.ply')]
    sources += [source / 'images' / name for name in NAMES]
    extensions = [Path(path) for path in old_hashes if '/site-packages/' in path
                  and ('diff_gaussian_rasterization' in path or 'simple_knn' in path)]
    require(len(extensions) >= 2, 'INCONCLUSIVE_OLD_PROVENANCE', 'Old native extension identity unavailable')
    for path in sources + extensions:
        require(str(path.resolve()) in old_hashes, 'INCONCLUSIVE_OLD_PROVENANCE', f'No old source pin: {path}')
        pin(path, old_hashes[str(path.resolve())])
    ply = repo / 'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
    pin(ply, PLY_SHA)
    for phase in ('before', 'after'):
        pin(old_run / 'raw' / f'cameras_30000_{phase}.npz', OLD_CAMERA_SHA)
    for name, expected in zip(NAMES[:3], CACHE_SHA):
        pin(old_run / 'raw' / f'30000_{name}_base.npy', expected)
    STATE['old_run_provenance'] = {'path': str(old_run), 'old_results_sha256': OLD_RESULTS_SHA,
        'old_terminal_decision': old['decision'], 'old_30k_parameters_before_after_sha256': PARAMETERS_SHA,
        'old_source_PLY_sha256': PLY_SHA, 'cached_names': list(NAMES[:3]),
        'old_camera_archive_sha256': OLD_CAMERA_SHA, 'cached_RGB_sha256': dict(zip(NAMES[:3], CACHE_SHA))}
    return old


def validate_current(bundle, cameras, old, old_run):
    require(bundle.parameters_sha256 == PARAMETERS_SHA and bundle.source_ply_sha256 == PLY_SHA,
            'INCONCLUSIVE_IDENTITY', 'Current teacher tensor/PLY identity differs')
    require(tuple(camera.image_name for camera in cameras) == NAMES, 'INCONCLUSIVE_IDENTITY', 'Current fixed names differ')
    with (np.load(old_run / 'raw/cameras_30000_before.npz', allow_pickle=False) as before,
          np.load(old_run / 'raw/cameras_30000_after.npz', allow_pickle=False) as after):
        for view, camera in enumerate(cameras):
            require((camera.image_width, camera.image_height, camera.colmap_id) == (979, 546, 1),
                    'INCONCLUSIVE_IDENTITY', f'Current image dimensions/id differ: {camera.image_name}')
            for field in MATRIX_FIELDS:
                key = f'view{view:02d}__{field}'
                require(np.array_equal(before[key], after[key]) and np.array_equal(before[key], array(camera, field)),
                        'INCONCLUSIVE_CAMERA_IDENTITY', f'Old/current camera matrix differs: {camera.image_name}/{field}')
            require(camera.FoVx == old['camera_metadata']['30000'][view]['FoVx']
                    and camera.FoVy == old['camera_metadata']['30000'][view]['FoVy'],
                    'INCONCLUSIVE_CAMERA_IDENTITY', f'Current FoV differs: {camera.image_name}')
    STATE['camera_hashes_before'] = {camera.image_name: camera_hash(camera) for camera in cameras}
    save_camera_matrices(cameras, 'before')


def finalize_integrity(bundle, cameras):
    if bundle is not None:
        budget()
        native.assert_frozen(bundle.gaussians)
        after = native.model_hash(bundle.gaussians)
        STATE['parameters_after_sha256'] = after
        require(after == STATE['parameters_before_sha256'] == PARAMETERS_SHA,
                'INCONCLUSIVE_INTEGRITY', 'Original Gaussian tensor state changed')
        STATE['model_integrity_status'] = 'unchanged_before_after'
        require(len(cameras) == 12 and len(STATE['camera_hashes_before']) == 12,
                'INCONCLUSIVE_INTEGRITY', 'Original camera before state incomplete')
        STATE['camera_hashes_after'] = {camera.image_name: camera_hash(camera) for camera in cameras}
        require(STATE['camera_hashes_before'] == STATE['camera_hashes_after'],
                'INCONCLUSIVE_INTEGRITY', 'An original camera or original image/alpha tensor changed')
        STATE['camera_integrity_status'] = 'unchanged_before_after'
        save_camera_matrices(cameras, 'after')
    for path, expected in STATE['input_hashes_before'].items():
        value = sha(path)
        STATE['input_hashes_after'][path] = value
        require(value == expected, 'INCONCLUSIVE_INTEGRITY', f'An input changed: {path}')
    STATE['recorded_source_files_rehashed'] = True
    STATE['integrity_complete'] = bundle is not None and len(STATE['camera_hashes_after']) == 12


def main():
    global STATE, FINALIZING, OUT, np, native, torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    old_run = root / 'research/experiments/shared-coordinate-residual-v0/run-v3'
    STATE = {'status': 'INCONCLUSIVE_NOT_COMPLETED', 'scope': 'artifact discovery collection only',
        'failure': None, 'output': str(OUT), 'selected_names': list(NAMES),
        'contract_sha256': CONTRACT_SHA, 'helper_sha256': HELPER_SHA,
        'planned_new_forwards': 10, 'planned_reused_teacher_views': 3,
        'new_forward_attempts': 0, 'new_forward_completed': 0, 'forward_attempt_log': [],
        'training_performed': False, 'parameter_updates': 0, 'optimizer_created': False,
        'backward_calls': 0, 'SE3_calls': 0, 'matching_calls': 0, 'fitting_calls': 0, 'pruning_calls': 0,
        'artifact_classification': None, 'method_verdict': None, 'causal_or_novelty_claim': None,
        'confirmation_photos_or_ROIs_selected': False,
        'input_hashes_before': {}, 'input_hashes_after': {}, 'output_hashes': {},
        'parameters_before_sha256': None, 'parameters_after_sha256': None,
        'camera_hashes_before': {}, 'camera_hashes_after': {}, 'integrity_complete': False,
        'recorded_source_files_rehashed': False, 'model_loaded': False,
        'model_integrity_status': 'not_loaded_not_checked',
        'camera_integrity_status': 'not_loaded_not_checked',
        'image_index': [], 'external_watchdog_completion_required': True}
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    bundle, cameras = None, []
    try:
        for directory in ('raw', 'raw/teacher', 'raw/GT', 'panels'):
            (OUT / directory).mkdir()
        budget()
        pin(Path(__file__).parent / 'shared_coordinate_native_v3.py', HELPER_SHA)
        import shared_coordinate_native_v3 as native_module
        native = native_module
        progress('verify_old_cache_and_sources')
        old = prepare_sources(root, old_run)
        import numpy as numpy_module
        import torch as torch_module
        np, torch = numpy_module, torch_module
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.reset_peak_memory_stats()
        progress('load_only_frozen_30k_teacher')
        STATE['model_integrity_status'] = 'load_attempt_no_registered_model_final_state_unknown'
        bundle = native.load_truck_checkpoint(30000)
        STATE['model_loaded'] = True
        STATE['model_integrity_status'] = 'loaded_final_check_pending'
        STATE['parameters_before_sha256'] = bundle.parameters_sha256
        require(len(bundle.train_cameras) == 219, 'INCONCLUSIVE_IDENTITY', 'Expected 219 original training cameras')
        STATE['all_training_camera_names'] = [camera.image_name for camera in bundle.train_cameras]
        cameras = [bundle.train_cameras[rank] for rank in RANKS]
        STATE['camera_integrity_status'] = 'selected_originals_final_check_pending'
        bundle = replace(bundle, train_cameras=cameras)
        validate_current(bundle, cameras, old, old_run)
        # Verify actual renderer extension origins against the pinned old inputs.
        for module_name in ('diff_gaussian_rasterization', 'diff_gaussian_rasterization._C', 'simple_knn._C'):
            module = __import__(module_name, fromlist=['__file__'])
            path = str(Path(module.__file__).resolve())
            require(path in STATE['input_hashes_before'], 'INCONCLUSIVE_IDENTITY', f'Unpinned runtime extension: {path}')
        write_json(OUT / 'CONFIG.json', {'contract_sha256': CONTRACT_SHA, 'helper_sha256': HELPER_SHA,
            'selected_names': list(NAMES), 'selected_ranks': list(RANKS), 'model_iteration': 30000,
            'SH_degree': 3, 'resolution': 1, 'black_background': True, 'AA': False, 'exposure': False,
            'raw_shape_HWC': [546, 979, 3], 'raw_dtype': 'float32', 'CPU_threads': 1,
            'new_forward_plan': {'anchor': 1, 'remaining_uncached_views': 9, 'total': 10},
            'cached_views': list(NAMES[:3]), 'old_cache_provenance': STATE['old_run_provenance'],
            'all_input_hashes_before': STATE['input_hashes_before'], 'purpose': 'discovery, no independent confirmation'})
        caches = []
        for name in NAMES[:3]:
            image = np.load(old_run / 'raw' / f'30000_{name}_base.npy', mmap_mode='r', allow_pickle=False)
            check_rgb(image, name)
            caches.append(image)
        progress('single_anchor', view=NAMES[0])
        anchor = forward(bundle, cameras[0], 'cache_identity_anchor')
        save_rgb(OUT / 'raw' / 'anchor_000002_fresh.npy', anchor)
        difference = anchor.astype(np.float64) - caches[0].astype(np.float64)
        STATE['anchor_cache_identity'] = {'view': NAMES[0],
            'max_abs_RGB_difference': float(np.max(np.abs(difference))),
            'RGB_MSE_float64': float(np.mean(difference**2)), 'max_limit': 2e-5, 'MSE_limit': 1e-10,
            'purpose': 'cache identity only; no teacher-versus-GT score'}
        write_json(OUT / 'ANCHOR_IDENTITY.json', STATE['anchor_cache_identity'])
        require(STATE['anchor_cache_identity']['max_abs_RGB_difference'] <= 2e-5
                and STATE['anchor_cache_identity']['RGB_MSE_float64'] <= 1e-10,
                'INCONCLUSIVE_ANCHOR_IDENTITY', 'New unperturbed anchor differs from old cache')
        for view, (name, camera) in enumerate(zip(NAMES, cameras)):
            progress('collect_fixed_discovery_view', selected_index=view, view=name, reused=view < 3)
            teacher_path = OUT / 'raw/teacher' / (name + '.npy')
            if view < 3:
                teacher = caches[view]
                shutil.copyfile(old_run / 'raw' / f'30000_{name}_base.npy', teacher_path)
                require(sha(teacher_path) == CACHE_SHA[view], 'INCONCLUSIVE_CACHE_COPY', 'Reused cache copy is not byte exact')
                origin = 'reused_old_base_byte_for_byte'
            else:
                teacher = forward(bundle, camera, 'uncached_discovery_view_once')
                save_rgb(teacher_path, teacher)
                origin = 'one_new_unperturbed_native_forward'
            gt = camera.original_image.detach().cpu().permute(1, 2, 0).contiguous().numpy().copy()
            gt_path = OUT / 'raw/GT' / (name + '.npy')
            save_rgb(gt_path, gt)
            panel_path = OUT / 'panels' / f'{view:02d}_{name}_teacher_left_GT_right.png'
            save_pair(panel_path, teacher, gt)
            STATE['image_index'].append({'selected_index': view, 'name': name, 'teacher_origin': origin,
                'teacher_float': str(teacher_path.relative_to(OUT)), 'GT_float': str(gt_path.relative_to(OUT)),
                'pair_png': str(panel_path.relative_to(OUT)), 'pair_size_wh': [1958, 546],
                'left': 'frozen mature teacher', 'right': 'current native original GT',
                'role': 'discovery only; not independent confirmation'})
            write_json(OUT / 'IMAGE_INDEX.json', STATE['image_index'])
        require(STATE['new_forward_attempts'] == STATE['new_forward_completed'] == 10
                and len(STATE['image_index']) == 12, 'INCONCLUSIVE_COLLECTION_COMPLETENESS', 'Ten forwards/twelve fixed views incomplete')
        lines = ['# Fixed training-view discovery collection', '',
                 'All twelve are discovery material. Teacher left, GT right; original full pixels, no enhancement.', '']
        lines += [f'- [{row["name"]}]({row["pair_png"]}) — {row["teacher_origin"]}' for row in STATE['image_index']]
        (OUT / 'IMAGE_INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
        STATE['status'] = 'COLLECTED_FOR_DISCOVERY'
        progress('collection_complete_pending_final_integrity', new_forwards=10, reused_teacher_views=3)
    except BaseException as exc:
        failure = {'code': getattr(exc, 'code', 'INCONCLUSIVE_EXCEPTION'), 'type': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc()}
        if STATE['failure'] is None:
            STATE['failure'] = failure
        else:
            STATE.setdefault('secondary_errors', []).append(failure)
    finally:
        FINALIZING = True
        try:
            finalize_integrity(bundle, cameras)
            if torch is not None and torch.cuda.is_initialized():
                STATE['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                STATE['GPU_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            STATE['output_hashes'] = {str(path.relative_to(OUT)): sha(path) for path in sorted(OUT.rglob('*'))
                if path.is_file() and path.name not in ('RESULTS.json', 'PROGRESS.jsonl')}
            budget()
        except BaseException as exc:
            STATE['integrity_complete'] = False
            error = {'code': getattr(exc, 'code', 'INCONCLUSIVE_INTEGRITY'), 'message': str(exc), 'type': type(exc).__name__}
            STATE['final_integrity_error'] = error
            if STATE['failure'] is None:
                STATE['failure'] = error
        if STATE['failure'] is not None:
            STATE['status'] = 'INCONCLUSIVE'
        STATE['elapsed_before_RESULTS_serialization_seconds'] = time.perf_counter() - PROCESS_START
        STATE['time_scope'] = 'Starts before heavy imports; excludes this final RESULTS write and interpreter shutdown. External watchdog sidecar records process/pipe wall time.'
        STATE['unfinished_new_forward_attempts'] = STATE['new_forward_attempts'] - STATE['new_forward_completed']
        try:
            write_json(OUT / 'RESULTS.json', STATE)
            if time.perf_counter() - PROCESS_START >= 300. and STATE['failure'] is None:
                STATE['status'] = 'INCONCLUSIVE'
                STATE['failure'] = {'code': 'INCONCLUSIVE_BUDGET', 'message': 'Final RESULTS serialization crossed budget'}
                write_json(OUT / 'RESULTS.json', STATE)
            print('TEACHER_DISCOVERY_COLLECTION_TERMINATED ' + STATE['status'], flush=True)
        except BaseException as exc:
            print(f'FINAL_SAVE_FAILED {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
    return 0 if STATE['status'] == 'COLLECTED_FOR_DISCOVERY' and STATE['failure'] is None else 2


if __name__ == '__main__':
    sys.exit(main())
