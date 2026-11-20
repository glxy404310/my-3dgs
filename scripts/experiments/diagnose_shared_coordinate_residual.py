#!/usr/bin/env python3
"""Locked frozen Truck shared-coordinate diagnostic. Never train or update 3DGS."""
import time

PROCESS_START = time.perf_counter()
CPU_START = time.process_time()

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import traceback
from dataclasses import replace

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

CONTRACT_SHA = '7536a2641dd54ef84c02b2159ee6ec31fa8fab895e4ef9001c367b33f9a2273c'
HELPER_SHA = {
    'shared_coordinate_native.py': '9592b9d2a47c846acb9a3b1dbe7a2d56eacce166790a89cfa9216a287a84bd21',
    'shared_coordinate_tools.py': '2cfa37834cc44844ccd9b0ffe2064ca12728ae896aa8ff3d3217d556c3684594',
}
NATIVE_SHA = {
    'scene/__init__.py': '895d042f4dd2c6ee521dd985aea5c49d9c6f5c7379f76c4d537d8dc5468f18bf',
    'scene/dataset_readers.py': 'b06f01648fee893c4785ed38eeb934f91b79817b9dcbb694291c9162b0b6ba94',
    'scene/cameras.py': '7344b9a4c95e3649becab1b9e30eb0b351a15a61370f64352dcb231aed916128',
    'scene/gaussian_model.py': '546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310',
    'gaussian_renderer/__init__.py': 'a25cf5fdb22cd8757fb2a1e37b75352042d65b16138a9fa6243f8616ca1369b8',
    'utils/camera_utils.py': '7644d89893f2481a7a9be592227bf11f3e6e5c469c9ce92c3cb25f9ff920c458',
    'utils/graphics_utils.py': 'd6e75de8161e98b72c2691e8a944a842e1b6625d2306035d079c868ac632edfc',
}
RANKS = (0, 19, 39, 59, 79, 99, 118, 138, 158, 178, 198, 218)
NAMES = tuple(f'{index:06d}.jpg' for index in
              (2, 23, 46, 69, 92, 115, 136, 159, 182, 205, 228, 251))
ARMS = ('zero', 'shared_translation', 'per_image_affine', 'pose',
        'pose_translation', 'pose_shuffle', 'pose_brown')
SHARED_POSE = ('pose_translation', 'pose_shuffle', 'pose_brown')
POSE_ARMS = ('pose',) + SHARED_POSE
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
KNOWN_SHIFTS = ((.375, -.625), (-.45, .7), (.6, .3))
WIDTH, HEIGHT, H = 979, 546, 2.5e-4
RCOND, RANK_TOL = 1e-10, 1e-8
TIME_LIMIT, MATCH_BATCH, PANEL_COUNT = 300., 12, 6
np = tools = native = torch = None
STATE = None
FINALIZING = False


class Stop(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not bool(condition):
        raise Stop(code, message)


def budget():
    if time.perf_counter() - PROCESS_START >= TIME_LIMIT:
        raise Stop('INCONCLUSIVE_BUDGET', '300-second scientific process budget exhausted')


def on_signal(signum, _frame):
    reason = {'code': 'INCONCLUSIVE_BUDGET', 'message': f'received signal {signum}',
              'elapsed_before_failure_save_seconds': time.perf_counter() - PROCESS_START}
    if STATE is not None:
        STATE.setdefault('signals', []).append(reason)
    if not FINALIZING:
        raise Stop(reason['code'], reason['message'])


def safe_json(value):
    if hasattr(value, 'tolist'):
        return safe_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): safe_json(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf8')
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
    if expected is not None:
        require(value == expected, 'INCONCLUSIVE_INPUT_IDENTITY', f'Hash mismatch: {path}')
    STATE['input_hashes_before'][str(path)] = value
    return value


def progress(stage, **details):
    budget()
    record = {'stage': stage, 'elapsed_before_progress_write_seconds':
              time.perf_counter() - PROCESS_START, **details}
    STATE['last_progress'] = record
    with (STATE['output'] / 'PROGRESS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(safe_json(record), allow_nan=False) + '\n')
    print(json.dumps(safe_json(record), allow_nan=False), flush=True)
    write_json(STATE['output'] / 'STATUS.json', record)
    budget()


def save_arrays(path, **arrays):
    budget()
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)
    STATE['artifacts'].append({'path': str(path.relative_to(STATE['output'])),
                               'bytes': path.stat().st_size})
    budget()


def save_image_array(path, image):
    budget()
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, image, allow_pickle=False)
    temporary.replace(path)
    STATE['artifacts'].append({'path': str(path.relative_to(STATE['output'])),
                               'bytes': path.stat().st_size})
    budget()


def ls(matrix, target, label):
    """One column-scaled full-design, unregularized solve, with both spectra."""
    budget()
    matrix, target = np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64)
    scales = np.linalg.norm(matrix, axis=0)
    divisor = np.where(scales > 0, scales, 1.)
    normalized = matrix / divisor
    coefficients_scaled, _, rank, singular = np.linalg.lstsq(normalized, target, rcond=RCOND)
    coefficients = coefficients_scaled / (divisor if target.ndim == 1 else divisor[:, None])
    raw_singular = np.linalg.svd(matrix, compute_uv=False)
    raw_rank = int(np.sum(raw_singular > RCOND * raw_singular[0])) if len(raw_singular) else 0
    feasibility_rank = int(np.sum(singular > RANK_TOL * singular[0])) if len(singular) else 0
    full_rank = len(singular) >= matrix.shape[1] and not np.any(scales == 0)
    condition = float(singular[0] / singular[-1]) if full_rank and singular[-1] > 0 else math.inf
    residual = matrix @ coefficients - target
    STATE['linear_lstsq_calls'] += 1
    return {'label': label, 'coefficients': coefficients, 'column_norms': scales,
            'safe_column_divisor': divisor, 'raw_singular_values': raw_singular,
            'raw_rank_rcond': raw_rank, 'scaled_singular_values': singular,
            'scaled_rank_rcond': int(rank), 'scaled_rank_feasibility': feasibility_rank,
            'scaled_condition': condition, 'rcond': RCOND, 'rank_relative_threshold': RANK_TOL,
            'residual': residual, 'sse': float(np.sum(residual**2))}


def compact_fit(fit):
    return {key: value for key, value in fit.items() if key != 'residual'}


def pose_design_check(design, label):
    check = ls(design, np.zeros(len(design)), label)
    if check['scaled_rank_feasibility'] != 6 or check['scaled_condition'] > 1e6:
        write_json(STATE['output'] / 'FAILURE_DESIGN.json', compact_fit(check))
    require(check['scaled_rank_feasibility'] == 6 and check['scaled_condition'] <= 1e6,
            'INCONCLUSIVE_POSE_DESIGN', f'Unusable six-column pose design: {label}')
    return compact_fit(check)


def gradient(image):
    return tools.central_gradient(tools.rgb_to_gray(image))


def feature(grad, centers=None):
    return tools.sample_normalized_gradient_features(grad, GRID if centers is None else centers, PATCH)


def rgb_patches(image, centers, offsets):
    sampled, valid = tools.bilinear_sample(image, centers[:, None, :] + offsets[None, :, :])
    require(valid.all(), 'INCONCLUSIVE_SAMPLING_SUPPORT', 'Required RGB support left the image')
    return sampled.astype(np.float32)


def texture(grad):
    sampled = feature(grad)
    g = sampled['sampled_gradients']
    tensor = np.einsum('npi,npj->nij', g, g) / len(PATCH)
    finite = np.isfinite(tensor).all(axis=(1, 2))
    eigenvalues = np.full((len(GRID), 2), np.nan)
    eigenvalues[finite] = np.linalg.eigvalsh(tensor[finite])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = eigenvalues[:, 0] / eigenvalues[:, 1]
    valid = sampled['valid'] & (eigenvalues[:, 0] >= 1e-4) & (ratio >= .05)
    return {'valid': valid, 'tensor_mean': tensor, 'eigenvalues': eigenvalues, 'ratio': ratio}


MATCH_KEYS = (
    'mse', 'cost', 'ncc', 'candidate_valid', 'valid', 'grid_match_valid', 'best_index',
    'best_offset_yx', 'best_render_center_yx', 'best_center_yx', 'best_mse', 'best_ncc',
    'neighborhood_indices', 'neighborhood_offsets_yx', 'neighborhood_delta_yx',
    'neighborhood_cost', 'grid_boundary', 'has_neighborhood', 'quadratic_coefficients',
    'quadratic_rank', 'quadratic_singular_values', 'quadratic_residual',
    'quadratic_residual_sum_squares', 'hessian', 'hessian_eigenvalues',
    'hessian_condition_number', 'positive_definite', 'within_grid_step',
    'refinement_delta_yx', 'proposed_offset_yx', 'proposed_center_yx',
    'refined_sample_valid', 'refined_offset_yx', 'refined_center_yx', 'refined_feature',
    'refined_norm', 'refined_mse', 'refined_ncc', 'refined_cost', 'reference_valid',
)


def match(reference_grad, target_grad, centers, path):
    """All locked candidates/refinements, batched without changing their order."""
    pieces = {}
    for start in range(0, len(centers), MATCH_BATCH):
        budget()
        result = tools.bidirectional_match(reference_grad, target_grad,
            centers[start:start + MATCH_BATCH], SEARCH, SEARCH, PATCH,
            np.array([.25, .25]), rcond=RCOND)
        for direction in ('forward', 'reverse'):
            for key in MATCH_KEYS:
                if key in result[direction]:
                    pieces.setdefault(f'{direction}__{key}', []).append(result[direction][key])
        for key in ('valid', 'round_trip_error_yx', 'round_trip_error_norm', 'reverse_gt_center_yx'):
            pieces.setdefault(key, []).append(result[key])
        # Each helper match solves one possible six-coefficient quadratic per
        # point/direction. Count actual solves, including later rejected peaks.
        STATE['quadratic_lstsq_calls'] += sum(int(result[d]['has_neighborhood'].sum())
                                               for d in ('forward', 'reverse'))
    merged = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    accepted = (merged['valid'] & (merged['forward__refined_ncc'] >= .8)
                & (merged['reverse__refined_ncc'] >= .8)
                & (merged['round_trip_error_norm'] <= .25))
    merged.update(accepted=accepted, centers_yx=centers, search_offsets_yx=SEARCH,
                  patch_offsets_yx=PATCH, grid_steps_yx=np.array([.25, .25]))
    save_arrays(path, **merged)
    return {'valid': accepted, 'd': merged['forward__refined_offset_yx'],
            'cost': merged['forward__refined_cost'], 'ncc': merged['forward__refined_ncc']}


def camera_snapshot(camera, include_images=True):
    result = {}
    for key in ('R', 'T', 'trans', 'scale', 'FoVx', 'FoVy', 'image_width', 'image_height',
                'world_view_transform', 'projection_matrix', 'full_proj_transform',
                'camera_center', 'original_image', 'alpha_mask'):
        if not include_images and key in ('original_image', 'alpha_mask'):
            continue
        value = getattr(camera, key)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy().copy()
        elif isinstance(value, np.ndarray):
            value = value.copy()
        result[key] = value
    return result


def load_checkpoint(iteration, args):
    progress('load_checkpoint', iteration=iteration)
    bundle = native.load_truck_checkpoint(iteration, repo=args.repo, source=args.source,
                                           model_root=args.model)
    # Register before further validation so every loaded model gets a finally check.
    CHECKPOINTS[iteration] = bundle
    STATE['parameters_before'][str(iteration)] = bundle.parameters_sha256
    require(math.isfinite(bundle.scene_extent) and bundle.scene_extent > 0,
            'INCONCLUSIVE_INPUT_IDENTITY', 'Scene extent must be finite and positive')
    require(len(bundle.train_cameras) == 219, 'INCONCLUSIVE_INPUT_IDENTITY', 'Expected 219 train cameras')
    selected = [bundle.train_cameras[index] for index in RANKS]
    STATE.setdefault('all_training_camera_names', {})[str(iteration)] = [camera.image_name for camera in bundle.train_cameras]
    require(tuple(camera.image_name for camera in selected) == NAMES,
            'INCONCLUSIVE_INPUT_IDENTITY', 'Fixed camera rank/name mismatch')
    SNAPSHOTS[iteration] = [(camera, camera_snapshot(camera)) for camera in selected]
    save_arrays(STATE['output'] / 'raw' / f'cameras_{iteration}_before.npz',
                **{f'view{view:02d}__{key}': value for view, (_, snapshot) in enumerate(SNAPSHOTS[iteration])
                   for key, value in snapshot.items() if key not in ('original_image', 'alpha_mask')})
    # Keep exactly the selected references; no matrix/image is changed. Release
    # the other 207 training images instead of retaining both full Scenes.
    bundle = replace(bundle, train_cameras=selected)
    CHECKPOINTS[iteration] = bundle
    metadata = []
    for camera in selected:
        require((camera.image_width, camera.image_height, camera.colmap_id) == (WIDTH, HEIGHT, 1),
                'INCONCLUSIVE_INPUT_IDENTITY', 'Unexpected native camera dimensions/id')
        fx = WIDTH / (2 * math.tan(camera.FoVx / 2))
        fy = HEIGHT / (2 * math.tan(camera.FoVy / 2))
        require(abs(fx - 1163.2547280302354 * WIDTH / 1957) <= 1e-9 and
                abs(fy - 1156.280404988286 * HEIGHT / 1091) <= 1e-9,
                'INCONCLUSIVE_INPUT_IDENTITY', 'Native FoV does not match scaled fixed metadata')
        metadata.append({'name': camera.image_name, 'FoVx': camera.FoVx, 'FoVy': camera.FoVy,
                         'width': WIDTH, 'height': HEIGHT, 'fx': fx, 'fy': fy,
                         'cx': (WIDTH - 1) / 2., 'cy': (HEIGHT - 1) / 2.,
                         'scaled_colmap_boundary_cx': WIDTH / 2.,
                         'scaled_colmap_boundary_cy': HEIGHT / 2., 'camera_id': camera.colmap_id})
    STATE['camera_metadata'][str(iteration)] = metadata
    STATE['scene_extents'][str(iteration)] = bundle.scene_extent
    write_json(STATE['output'] / 'CONFIG.json', CONFIG)
    return bundle, selected


def forward(bundle, camera, category):
    budget()
    attempts = STATE.setdefault('forward_attempt_counts', {})
    attempts[category] = attempts.get(category, 0) + 1
    started = time.perf_counter()
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    event_start.record()
    result = native.render_rgb(bundle, camera)['rgb']
    event_end.record()
    event_end.synchronize()
    STATE['forward_counts'][category] = STATE['forward_counts'].get(category, 0) + 1
    STATE['gpu_native_render_seconds'] += event_start.elapsed_time(event_end) / 1000.
    STATE['render_call_wall_seconds'] += time.perf_counter() - started
    image = result.permute(1, 2, 0).contiguous().numpy()
    require(image.shape == (HEIGHT, WIDTH, 3) and np.isfinite(image).all(),
            'INCONCLUSIVE_RENDER_SCHEMA', 'Native RGB schema/nonfinite output')
    budget()
    return image


def base_image(bundle, camera):
    key = (bundle.iteration, camera.image_name)
    if key not in BASE_CACHE:
        image = forward(bundle, camera, 'core_base')
        path = STATE['output'] / 'raw' / f'{bundle.iteration}_{camera.image_name}_base.npy'
        save_image_array(path, image)
        BASE_CACHE[key] = (image, gradient(image))
    return BASE_CACHE[key]


def witnesses(bundle, cameras):
    errors, zero_feature_error_squared = [], []
    records = []
    for index, (camera, shift) in enumerate(zip(cameras[:3], KNOWN_SHIFTS)):
        progress('witness', index=index, view=camera.image_name)
        image, grad = base_image(bundle, camera)
        source_texture = texture(grad)
        require(source_texture['valid'].sum() >= 24, 'INCONCLUSIVE_WITNESS_CAPACITY',
                f'Insufficient source texture in known-shift witness {index}')
        yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float64)
        shifted, shifted_valid = tools.bilinear_sample(image,
            np.stack((yy + shift[0], xx + shift[1]), axis=-1))
        save_image_array(STATE['output'] / 'raw' / f'witness_{index}_shifted_target.npy',
                         shifted.astype(np.float32))
        matched = match(gradient(shifted), grad, GRID,
                        STATE['output'] / 'raw' / f'witness_{index}_matches.npz')
        valid = source_texture['valid'] & matched['valid']
        error = np.linalg.norm(matched['d'][valid] - np.asarray(shift), axis=1)
        save_arrays(STATE['output'] / 'raw' / f'witness_{index}_summary.npz',
                    texture_valid=source_texture['valid'], texture_eigenvalues=source_texture['eigenvalues'],
                    valid=valid, recovered=matched['d'], known_shift=np.asarray(shift),
                    displacement_error=error, shifted_image_valid=shifted_valid)
        require(valid.sum() >= 24, 'INCONCLUSIVE_WITNESS_CAPACITY',
                f'Insufficient bidirectional known-shift patches in witness {index}')
        record = {'index': index, 'view': camera.image_name, 'known_shift_yx': shift,
                  'source_texture_count': int(source_texture['valid'].sum()), 'valid_count': int(valid.sum()),
                  'median_error_px': float(np.median(error)), 'p90_error_px': float(np.quantile(error, .9)),
                  'rms_error_px': float(np.sqrt(np.mean(error**2))), 'zero_checks': []}
        records.append(record)
        STATE['witnesses'] = records
        write_json(STATE['output'] / 'WITNESSES.json', records)
        require(record['median_error_px'] <= .1 and record['p90_error_px'] <= .2,
                'FAILED_TOOL_METHOD_UNKNOWN', f'Known off-grid shift not recovered in witness {index}')
        errors.append(record['rms_error_px'])
        original_feature = feature(grad)
        for label, pose_camera in (('repeat', camera), ('zero_copy', native.perturb_camera(camera, np.zeros(6)))):
            replay = forward(bundle, pose_camera, 'witness_repeat_zero')
            save_image_array(STATE['output'] / 'raw' / f'witness_{index}_{label}.npy', replay)
            difference = replay.astype(np.float64) - image
            replay_feature = feature(gradient(replay))
            fv = original_feature['valid'] & replay_feature['valid']
            squared = np.sum((original_feature['features'][fv] - replay_feature['features'][fv])**2, axis=1)
            zero_feature_error_squared.extend(squared.tolist())
            check = {'kind': label, 'max_rgb_difference': float(np.max(np.abs(difference))),
                     'rgb_mse': float(np.mean(difference**2)), 'feature_count': int(fv.sum()),
                     'feature_l2_rms': float(np.sqrt(np.mean(squared))) if len(squared) else None}
            record['zero_checks'].append(check)
            write_json(STATE['output'] / 'WITNESSES.json', records)
            require(len(squared) > 0 and check['max_rgb_difference'] <= 2e-5 and check['rgb_mse'] <= 1e-10,
                    'FAILED_TOOL_METHOD_UNKNOWN', f'Native {label} witness failed for {camera.image_name}')
    STATE['sigma_shift'] = max(errors)
    STATE['sigma_zero'] = float(np.sqrt(np.mean(zero_feature_error_squared)))
    STATE['tools_complete'] = True
    write_json(STATE['output'] / 'WITNESSES.json', {'records': records,
               'sigma_shift': STATE['sigma_shift'], 'sigma_zero': STATE['sigma_zero']})
    progress('witnesses_complete', sigma_shift=STATE['sigma_shift'], sigma_zero=STATE['sigma_zero'])


def capacity_upper_bound(mask, iteration, view, stage):
    record = {'iteration': iteration, 'view': view, 'name': NAMES[view], 'stage': stage,
              'single_checkpoint_superset_count': int(mask.sum()),
              'spatial_fold_upper_bounds': [int((mask & (FOLD == f)).sum()) for f in (0, 1)],
              'quadrant_upper_bounds': [int((mask & (QUADRANT == q)).sum()) for q in range(4)],
              'actual_two_checkpoint_common_mask': 'unknown_until_both_checkpoint_masks_complete',
              'proof': 'The final two-checkpoint common mask is a subset of this mask.'}
    possible = (record['single_checkpoint_superset_count'] >= 48
                and min(record['spatial_fold_upper_bounds']) >= 16
                and min(record['quadrant_upper_bounds']) >= 6)
    if not possible:
        STATE['capacity_upper_bound_failure'] = record
        write_json(STATE['output'] / 'CAPACITY_UPPER_BOUND_FAILURE.json', record)
        raise Stop('INCONCLUSIVE_MATCHING_CAPACITY',
                   f'Fixed view {NAMES[view]} cannot meet common capacity even before remaining gates: {stage}')


def quality_and_matches(bundle, cameras):
    result = []
    for view, camera in enumerate(cameras):
        progress('core_fd_and_matching', iteration=bundle.iteration, view=view, name=camera.image_name)
        prefix = f'{bundle.iteration}_{camera.image_name}'
        base, grad = base_image(bundle, camera)
        gt = camera.original_image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
        if camera.image_name not in GT_CACHE:
            GT_CACHE[camera.image_name] = gt.copy()
            save_image_array(STATE['output'] / 'raw' / f'{camera.image_name}_gt.npy', gt)
        else:
            require(np.array_equal(gt, GT_CACHE[camera.image_name]), 'INCONCLUSIVE_INPUT_IDENTITY',
                    'GT decoding differs across checkpoint loads')
        gt_grad = gradient(gt)
        gt_texture = texture(gt_grad)
        save_arrays(STATE['output'] / 'raw' / f'{prefix}_GT_texture.npz', **gt_texture)
        capacity_upper_bound(gt_texture['valid'], bundle.iteration, view, 'GT_texture')
        matched = match(gt_grad, grad, GRID, STATE['output'] / 'raw' / f'{prefix}_real_matches.npz')
        capacity_upper_bound(gt_texture['valid'] & matched['valid'], bundle.iteration, view,
                             'GT_texture_and_real_matching_before_FD')
        spatial = tools.feature_spatial_jacobian(grad, GRID, PATCH, np.array([.125, .125]))
        A = spatial['A']
        features_fd = np.empty((2, 6, 2, len(GRID), 2 * len(PATCH)), dtype=np.float64)
        for scale_index, scale in enumerate((1., .5)):
            for axis in range(6):
                for sign_index, sign in enumerate((-1., 1.)):
                    twist = np.zeros(6)
                    twist[axis] = sign * H * scale * (1. if axis < 3 else bundle.scene_extent)
                    pose_camera = native.perturb_camera(camera, twist)
                    image = forward(bundle, pose_camera, 'core_fd')
                    sampled = feature(gradient(image))
                    features_fd[scale_index, axis, sign_index] = sampled['features']
                    save_arrays(STATE['output'] / 'raw' /
                        f'{prefix}_fd_s{scale_index}_a{axis}_sign{sign_index}.npz',
                        rgb15=rgb_patches(image, GRID, RAW15), centers_yx=GRID,
                        rgb_offsets_yx=RAW15, twist=twist, features=sampled['features'],
                        feature_valid=sampled['valid'])
        B_h = ((features_fd[0, :, 1] - features_fd[0, :, 0]) / 2.).transpose(1, 2, 0)
        B_half = (features_fd[1, :, 1] - features_fd[1, :, 0]).transpose(1, 2, 0)
        H_h, H_half = np.full((len(GRID), 2, 6), np.nan), np.full((len(GRID), 2, 6), np.nan)
        residual = np.full_like(B_half, np.nan)
        a_singular = np.full((len(GRID), 2), np.nan)
        a_scaled_singular = np.full((len(GRID), 2), np.nan)
        a_column_norms = np.full((len(GRID), 2), np.nan)
        a_ranks = np.full((len(GRID), 2), -1, dtype=np.int64)
        a_condition = np.full(len(GRID), np.inf)
        computational = (np.isfinite(A).all(axis=(1, 2)) & np.isfinite(B_h).all(axis=(1, 2))
                         & np.isfinite(B_half).all(axis=(1, 2)))
        for point in np.flatnonzero(computational):
            full = ls(A[point], B_h[point], f'{prefix}/A/full/{point}')
            half = ls(A[point], B_half[point], f'{prefix}/A/half/{point}')
            H_h[point], H_half[point] = full['coefficients'], half['coefficients']
            residual[point] = half['residual']
            a_singular[point] = half['raw_singular_values']
            a_scaled_singular[point] = half['scaled_singular_values']
            a_column_norms[point] = half['column_norms']
            a_ranks[point] = (half['raw_rank_rcond'], half['scaled_rank_rcond'])
            if a_singular[point, -1] > 0:
                a_condition[point] = a_singular[point, 0] / a_singular[point, -1]
        denominator = np.maximum(np.linalg.norm(B_half, axis=(1, 2)), 10 * STATE['sigma_zero'])
        with np.errstate(divide='ignore', invalid='ignore'):
            response_error = np.linalg.norm(B_h - B_half, axis=(1, 2)) / denominator
            tangent_error = np.linalg.norm(residual, axis=(1, 2)) / denominator
        h_error = np.linalg.norm(H_h - H_half, axis=(1, 2)) / math.sqrt(6)
        h_scale = np.linalg.norm(H_half, axis=(1, 2)) / math.sqrt(6)
        quality = (computational & (a_singular[:, -1] >= 1e-3) & (a_condition <= 30)
                   & (denominator > 0) & (response_error <= .25) & (tangent_error <= .35)
                   & (h_error <= np.maximum(.03, .25 * h_scale)))
        valid = gt_texture['valid'] & matched['valid'] & quality
        save_arrays(STATE['output'] / 'raw' / f'{prefix}_native_quality.npz', A=A,
            B_h=B_h, B_half=B_half, H_h=H_h, H_half=H_half, projection_residual=residual,
            A_raw_singular=a_singular, A_scaled_singular=a_scaled_singular,
            A_column_norms=a_column_norms, A_ranks=a_ranks, A_condition=a_condition,
            denominator=denominator, response_error=response_error, tangent_error=tangent_error,
            H_error=h_error, H_scale=h_scale, native_quality=quality, computational=computational,
            gt_texture_valid=gt_texture['valid'], gt_structure_tensor=gt_texture['tensor_mean'],
            gt_texture_eigenvalues=gt_texture['eigenvalues'], matching_valid=matched['valid'], valid=valid)
        result.append({'camera': camera, 'base': base, 'grad': grad, 'gt': gt,
                       'H': H_half, 'd': matched['d'], 'cost': matched['cost'], 'valid': valid})
        STATE['core_views_complete'].append({'iteration': bundle.iteration, 'view': view,
            'texture': int(gt_texture['valid'].sum()), 'matching': int(matched['valid'].sum()),
            'native_quality': int(quality.sum()), 'intersection': int(valid.sum())})
        write_json(STATE['output'] / 'CORE_CAPACITY_PARTIAL.json', STATE['core_views_complete'])
        capacity_upper_bound(valid, bundle.iteration, view, 'single_checkpoint_complete_mask')
    return result


def common_capacity(data):
    masks = []
    for view in range(12):
        mask = data[7000][view]['valid'] & data[30000][view]['valid']
        record = {'view': view, 'name': NAMES[view], 'common_count': int(mask.sum()),
                  'spatial_fold_counts': [int((mask & (FOLD == fold)).sum()) for fold in (0, 1)],
                  'quadrant_counts': [int((mask & (QUADRANT == q)).sum()) for q in range(4)]}
        STATE['common_capacity'].append(record)
        save_arrays(STATE['output'] / 'raw' / f'{NAMES[view]}_common_mask.npz',
                    common_mask=mask, centers_yx=GRID, spatial_fold=FOLD, quadrant=QUADRANT,
                    checkpoint_7000_valid=data[7000][view]['valid'],
                    checkpoint_30000_valid=data[30000][view]['valid'])
        write_json(STATE['output'] / 'COMMON_CAPACITY.json', STATE['common_capacity'])
        require(record['common_count'] >= 48 and min(record['spatial_fold_counts']) >= 16
                and min(record['quadrant_counts']) >= 6, 'INCONCLUSIVE_MATCHING_CAPACITY',
                f'Fixed view {NAMES[view]} has insufficient common native/matching coverage')
        masks.append(mask)
    return np.asarray(masks)


def field_design(arm, points):
    points = np.asarray(points, dtype=np.float64)
    if arm in ('shared_translation', 'pose_translation'):
        return np.broadcast_to(np.eye(2), (*points.shape[:-1], 2, 2)).copy()
    flattened = points.reshape(-1, 2)
    design = tools.brown_conrady_design(flattened, **INTRINSICS)
    return design.reshape(*points.shape[:-1], 2, 4)


def center_field(arm):
    return field_design(arm, GRID[SHUFFLE] if arm == 'pose_shuffle' else GRID)


def affine_design(points):
    y = (points[:, 0] - INTRINSICS['cy']) / INTRINSICS['fy']
    x = (points[:, 1] - INTRINSICS['cx']) / INTRINSICS['fx']
    basis = np.column_stack((np.ones(len(points)), y, x))
    design = np.zeros((len(points), 2, 6))
    design[:, 0, :3], design[:, 1, 3:] = basis, basis
    return design


def separability(h_blocks, phi_blocks, label):
    original, projected, q_cross = [], [], []
    for hh, pp in zip(h_blocks, phi_blocks):
        norms = np.linalg.norm(hh, axis=0)
        q, _ = np.linalg.qr(hh / np.where(norms > 0, norms, 1.))
        original.append(pp)
        projected.append(pp - q @ (q.T @ pp))
        q_cross.append(q.T @ pp)
    phi, residual = np.vstack(original), np.vstack(projected)
    fit = ls(residual, np.zeros(len(residual)), label)
    original_norm = np.linalg.norm(phi, axis=0)
    retained = np.divide(np.linalg.norm(residual, axis=0), original_norm,
                         out=np.zeros_like(original_norm), where=original_norm > 0)
    eigenvalues, vectors = np.linalg.eigh(phi.T @ phi)
    partial = {**compact_fit(fit), 'column_retained_norm_ratio': retained,
               'original_Phi_gram_eigenvalues': eigenvalues}
    write_json(STATE['output'] / 'fits' / f'{label.replace("/", "_")}_partial.json', partial)
    require(np.all(eigenvalues > 0), 'INCONCLUSIVE_SHARED_POSE_SEPARABILITY',
            f'Original Brown Gram not positive definite: {label}')
    inverse_root = (vectors * (1. / np.sqrt(eigenvalues))) @ vectors.T
    c_singular = np.linalg.svd(residual @ inverse_root, compute_uv=False)
    cross_singular = np.linalg.svd(np.vstack(q_cross) @ inverse_root, compute_uv=False)
    record = {**compact_fit(fit), 'column_retained_norm_ratio': retained,
              'original_Phi_gram_eigenvalues': eigenvalues, 'whitened_complement_singular_values': c_singular,
              'principal_angles_radians': np.arccos(np.clip(cross_singular, 0., 1.))}
    write_json(STATE['output'] / 'fits' / f'{label.replace("/", "_")}_complete.json', record)
    require(fit['scaled_rank_feasibility'] == 4 and fit['scaled_condition'] <= 1e4
            and np.all(retained >= .01) and c_singular[-1] >= .01,
            'INCONCLUSIVE_SHARED_POSE_SEPARABILITY', f'Brown/pose confounding: {label}')
    return record


def fit_shared(arm, views, fit_masks, data, label):
    design = center_field(arm)
    count = design.shape[-1]
    blocks, targets, h_blocks, phi_blocks, checks = [], [], [], [], {}
    with_pose = arm in SHARED_POSE
    for block_index, view in enumerate(views):
        mask = fit_masks[view]
        phi = design[mask].reshape(-1, count)
        target = data[view]['d'][mask].reshape(-1)
        if with_pose:
            hh = data[view]['H'][mask].reshape(-1, 6)
            checks[str(view)] = pose_design_check(hh, f'{label}/train_pose/{view}')
            block = np.zeros((len(target), 6 * len(views) + count))
            block[:, 6 * block_index:6 * block_index + 6] = hh
            block[:, -count:] = phi
            h_blocks.append(hh)
            phi_blocks.append(phi)
        else:
            block = phi
        blocks.append(block)
        targets.append(target)
    record = {'arm': arm, 'pose_designs': checks}
    if arm == 'pose_brown':
        record['separability'] = separability(h_blocks, phi_blocks, f'{label}/Brown_complement')
    matrix, target = np.vstack(blocks), np.concatenate(targets)
    fit = ls(matrix, target, f'{label}/joint')
    record['joint_fit'] = compact_fit(fit)
    save_arrays(STATE['output'] / 'fits' / f'{label}_design.npz',
                design=matrix, target=target, coefficients=fit['coefficients'],
                residual=fit['residual'], column_norms=fit['column_norms'],
                raw_singular_values=fit['raw_singular_values'], scaled_singular_values=fit['scaled_singular_values'])
    theta = fit['coefficients'][-count:] if with_pose else fit['coefficients']
    training_q = {view: fit['coefficients'][6 * k:6 * k + 6] for k, view in enumerate(views)} if with_pose else {}
    return theta, training_q, record


def validate_pose(bundle, entry, common, q, shared, label, category):
    predicted = np.einsum('nij,j->ni', entry['H'], q)
    combined = predicted + shared
    finite_h = np.isfinite(entry['H']).all(axis=(1, 2))
    twist = H * q * np.array([1., 1., 1., bundle.scene_extent, bundle.scene_extent, bundle.scene_extent])
    norms = {'rotation_norm_rad': float(np.linalg.norm(twist[:3])),
             'normalized_translation_norm': float(np.linalg.norm(twist[3:]) / bundle.scene_extent),
             'max_combined_prediction_per_axis_px': np.max(np.abs(combined[common]), axis=0)}
    write_json(STATE['output'] / 'fits' / f'{label}_domain.json', {'q': q, 'twist': twist, **norms})
    require(np.isfinite(q).all() and finite_h[common].all() and norms['rotation_norm_rad'] <= .002
            and norms['normalized_translation_norm'] <= .002
            and np.all(norms['max_combined_prediction_per_axis_px'] <= 2.),
            'INCONCLUSIVE_POSE_LINEARIZATION', f'Pose/complete prediction domain exceeded: {label}')
    image = forward(bundle, native.perturb_camera(entry['camera'], twist), category)
    indices = np.flatnonzero(common)
    save_arrays(STATE['output'] / 'raw' / f'{label}_native_rgb25.npz',
                rgb25=rgb_patches(image, GRID, HALO12), centers_yx=GRID,
                rgb_offsets_yx=HALO12, q=q, twist=twist, common_mask=common,
                expected_Hq=predicted, shared_displacement=shared)
    matched = match(gradient(image), entry['grad'], GRID[indices],
                    STATE['output'] / 'raw' / f'{label}_native_matches.npz')
    valid = matched['valid']
    error = np.linalg.norm(matched['d'][valid] - predicted[indices][valid], axis=1)
    record = {**norms, 'q': q, 'twist': twist, 'common_count': len(indices),
              'native_valid_count': int(valid.sum()), 'native_valid_fraction': float(valid.mean()),
              'median_native_error_px': float(np.median(error)) if len(error) else None,
              'p90_native_error_px': float(np.quantile(error, .9)) if len(error) else None,
              'max_native_displacement_per_axis_px': np.max(np.abs(matched['d'][valid]), axis=0)
                  if valid.any() else None}
    save_arrays(STATE['output'] / 'raw' / f'{label}_native_validation.npz',
                original_grid_indices=indices, original_common_mask=common,
                native_valid=valid, native_displacement=matched['d'], expected_Hq=predicted[indices],
                error_on_valid=error)
    write_json(STATE['output'] / 'fits' / f'{label}_native_validation.json', record)
    require(valid.mean() >= .75 and len(error) > 0 and record['median_native_error_px'] <= .1
            and record['p90_native_error_px'] <= .25
            and np.all(record['max_native_displacement_per_axis_px'] <= 2.),
            'INCONCLUSIVE_POSE_LINEARIZATION', f'Own native pose did not validate: {label}')
    return image, record


def bilinear_replay(image, positions):
    sampled, valid = tools.bilinear_sample(image, positions)
    require(valid.all(), 'INCONCLUSIVE_SAMPLING_SUPPORT', 'OOF RGB query outside native image')
    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, np.array([HEIGHT - 1, WIDTH - 1]))
    corners = np.stack((np.stack((low[..., 0], low[..., 1]), axis=-1),
                        np.stack((low[..., 0], high[..., 1]), axis=-1),
                        np.stack((high[..., 0], low[..., 1]), axis=-1),
                        np.stack((high[..., 0], high[..., 1]), axis=-1)), axis=-2)
    corner_rgb = image[corners[..., 0], corners[..., 1]]
    return sampled.astype(np.float32), corners, corner_rgb


def run_cell(bundle, data, common, mode, cell, oof, fields):
    held_fold, eval_fold = cell
    label_base = f'{mode}_{bundle.iteration}_if{held_fold}' + (f'_sf{eval_fold}' if mode == 'main' else '')
    train_views = [view for view in range(12) if view % 2 != held_fold]
    held_views = [view for view in range(12) if view % 2 == held_fold]
    if mode == 'main':
        fit_masks = common & (FOLD[None, :] != eval_fold)
        eval_masks = common & (FOLD[None, :] == eval_fold)
    else:
        fit_masks, eval_masks = common & CENTER[None, :], common & EDGE[None, :]
    progress('fit_cell', iteration=bundle.iteration, mode=mode, image_fold=held_fold, spatial_eval_fold=eval_fold)
    shared_results = {}
    for arm in ('shared_translation',) + SHARED_POSE:
        label = f'{label_base}_{arm}'
        theta, training_q, record = fit_shared(arm, train_views, fit_masks, data, label)
        shared_results[arm] = theta
        record['shared_theta'] = theta
        record['training_q'] = training_q
        record['training_native_validations'] = {}
        write_json(STATE['output'] / 'fits' / f'{label}_fit.json', record)
        shared = np.einsum('nij,j->ni', center_field(arm), theta)
        if arm == 'pose_brown':
            fields.append(shared.copy())
        for view, q in training_q.items():
            progress('validate_training_q', iteration=bundle.iteration, mode=mode, arm=arm, view=view)
            _, check = validate_pose(bundle, data[view], common[view], q, shared,
                f'{label}_train_v{view:02d}', f'{mode}_train_pose_validation')
            record['training_native_validations'][str(view)] = check
            write_json(STATE['output'] / 'fits' / f'{label}_fit.json', record)
    for view in held_views:
        entry = data[view]
        fit_mask, eval_mask = fit_masks[view], eval_masks[view]
        indices = np.flatnonzero(eval_mask)
        observed = entry['d'][indices]
        gt_rgb = rgb_patches(entry['gt'], GRID[indices], PATCH)
        for arm in ARMS:
            budget()
            label = f'{label_base}_{arm}_held_v{view:02d}'
            theta = shared_results.get(arm)
            shared = np.zeros((len(GRID), 2)) if theta is None else np.einsum('nij,j->ni', center_field(arm), theta)
            q, affine, record = None, None, {'arm': arm, 'view': view}
            if arm == 'per_image_affine':
                fit = ls(affine_design(GRID[fit_mask]).reshape(-1, 6),
                         entry['d'][fit_mask].reshape(-1), label)
                affine = fit['coefficients']
                prediction = np.einsum('nij,j->ni', affine_design(GRID), affine)
                record['fit'] = compact_fit(fit)
            elif arm in POSE_ARMS:
                hh = entry['H'][fit_mask].reshape(-1, 6)
                record['pose_design'] = pose_design_check(hh, label)
                fit = ls(hh, (entry['d'][fit_mask] - shared[fit_mask]).reshape(-1), label)
                q = fit['coefficients']
                prediction = np.einsum('nij,j->ni', entry['H'], q) + shared
                record['fit'] = compact_fit(fit)
            else:
                prediction = shared
            source = entry['base']
            if q is not None:
                source, record['native_validation'] = validate_pose(bundle, entry, common[view], q, shared,
                    label, f'{mode}_heldout_pose_validation')
            points = GRID[indices, None, :] + PATCH[None, :, :]
            if arm == 'per_image_affine':
                pixel_shift = np.einsum('npij,j->npi',
                    affine_design(points.reshape(-1, 2)).reshape(len(indices), len(PATCH), 2, 6), affine)
            elif theta is not None:
                field_points = (GRID[SHUFFLE[indices], None, :] + PATCH[None, :, :]
                                if arm == 'pose_shuffle' else points)
                pixel_shift = np.einsum('npij,j->npi', field_design(arm, field_points), theta)
            else:
                pixel_shift = np.zeros_like(points)
            query = points + pixel_shift
            predicted_rgb, corners, corner_rgb = bilinear_replay(source, query)
            point_rgb_mse = np.mean((predicted_rgb.astype(np.float64) - gt_rgb)**2, axis=(1, 2))
            displacement_sse = np.sum((observed - prediction[indices])**2, axis=1)
            require(not np.isfinite(oof[arm]['prediction'][view, indices]).any(),
                    'INCONCLUSIVE_CROSSFIT_COMPLETENESS', 'An OOF patch was evaluated twice')
            oof[arm]['prediction'][view, indices] = prediction[indices]
            oof[arm]['displacement_sse'][view, indices] = displacement_sse
            oof[arm]['rgb_mse'][view, indices] = point_rgb_mse
            oof[arm]['rgb_prediction'][view, indices] = predicted_rgb
            save_arrays(STATE['output'] / 'raw' / f'{label}_evaluation.npz',
                original_grid_indices=indices, common_mask=common[view], fit_mask=fit_mask, eval_mask=eval_mask,
                observed_d=observed, predicted_d=prediction[indices], displacement_sse=displacement_sse,
                gt_rgb13=gt_rgb, predicted_rgb13=predicted_rgb, point_rgb_mse=point_rgb_mse,
                source_query_yx=query, source_corner_yx=corners, source_corner_rgb=corner_rgb,
                shared_theta=np.array([]) if theta is None else theta,
                pose_q=np.array([]) if q is None else q,
                affine_coefficients=np.array([]) if affine is None else affine)
            write_json(STATE['output'] / 'fits' / f'{label}_fit.json', record)
    STATE['fit_cells_complete'].append(label_base)
    write_json(STATE['output'] / 'FIT_CELLS_COMPLETE.json', STATE['fit_cells_complete'])


def empty_oof():
    return {arm: {'prediction': np.full((12, len(GRID), 2), np.nan),
                  'displacement_sse': np.full((12, len(GRID)), np.nan),
                  'rgb_mse': np.full((12, len(GRID)), np.nan),
                  'rgb_prediction': np.full((12, len(GRID), len(PATCH), 3), np.nan, dtype=np.float32)}
            for arm in ARMS}


def summarize_oof(oof, expected, label):
    summary = {}
    for arm in ARMS:
        arrays = oof[arm]
        require(np.array_equal(np.isfinite(arrays['displacement_sse']), expected)
                and np.array_equal(np.isfinite(arrays['rgb_mse']), expected),
                'INCONCLUSIVE_CROSSFIT_COMPLETENESS', f'Incomplete common OOF mask: {label}/{arm}')
        by_view = [{'name': NAMES[view], 'count': int(expected[view].sum()),
                    'displacement_mse': float(arrays['displacement_sse'][view, expected[view]].mean()),
                    'rgb_mse': float(arrays['rgb_mse'][view, expected[view]].mean())} for view in range(12)]
        summary[arm] = {'per_view': by_view,
                        'mean_displacement_mse': float(np.mean([row['displacement_mse'] for row in by_view])),
                        'mean_rgb_mse': float(np.mean([row['rgb_mse'] for row in by_view]))}
        save_arrays(STATE['output'] / 'raw' / f'{label}_{arm}_OOF.npz',
                    common_eval_mask=expected, centers_yx=GRID, **arrays)
    return summary


def stress_capacity(common):
    rows = []
    for view in range(12):
        center, edge = common[view] & CENTER, common[view] & EDGE
        counts = [int((center & (QUADRANT == quadrant)).sum()) for quadrant in range(4)]
        rows.append({'view': view, 'center_count': int(center.sum()), 'edge_count': int(edge.sum()),
                     'center_quadrant_counts': counts})
    STATE['center_edge_capacity'] = rows
    write_json(STATE['output'] / 'CENTER_EDGE_CAPACITY.json', rows)
    require(all(row['center_count'] >= 16 and row['edge_count'] >= 12
                and min(row['center_quadrant_counts']) >= 3 for row in rows),
            'INCONCLUSIVE_STRESS_CAPACITY', 'Fixed center/edge coverage is insufficient')


def field_rms(field):
    return float(np.sqrt(np.mean(np.sum(np.asarray(field)**2, axis=-1))))


def cosine(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else None


def science_gates(main, fields):
    gates, details = {}, {}
    for iteration in (7000, 30000):
        summary, field = main[iteration], np.asarray(fields[iteration])
        require(field.shape == (4, len(GRID), 2), 'INCONCLUSIVE_CROSSFIT_COMPLETENESS',
                'Expected four separate main Brown fields')
        primary = summary['pose_brown']['mean_displacement_mse']
        comparisons = {}
        for arm in ARMS[:-1]:
            reference = summary[arm]['mean_displacement_mse']
            comparisons[arm] = {'reference': reference, 'main': primary,
                'relative_reduction': 1 - primary / reference if reference > 0 else None,
                'passes_20pct': reference > 0 and primary <= .8 * reference}
        wins = sum(pose['displacement_mse'] > 0 and brown['displacement_mse'] <= .9 * pose['displacement_mse']
                   for pose, brown in zip(summary['pose']['per_view'], summary['pose_brown']['per_view']))
        rms = field_rms(field)
        rgb = summary['pose_brown']['mean_rgb_mse']
        item = {'all_six_controls_20pct': all(row['passes_20pct'] for row in comparisons.values()),
                'at_least_8_views_10pct_vs_pose': wins >= 8,
                'four_cell_concatenated_rms': rms >= max(.1, 3 * STATE['sigma_shift']),
                'RGB_no_worse_zero': rgb <= summary['zero']['mean_rgb_mse'],
                'RGB_no_worse_pose': rgb <= summary['pose']['mean_rgb_mse']}
        gates[str(iteration)] = item
        details[str(iteration)] = {'comparisons': comparisons, 'views_winning_10pct': wins,
            'concatenated_field_rms_px': rms, 'rms_threshold_px': max(.1, 3 * STATE['sigma_shift']),
            'cell_field_rms_px': [field_rms(x) for x in field]}
    cross_cosine = cosine(fields[7000], fields[30000])
    gates['concatenated_checkpoint_field_cosine'] = cross_cosine is not None and cross_cosine >= .8
    details['concatenated_checkpoint_field_cosine'] = cross_cosine
    details['cell_checkpoint_field_cosines'] = [cosine(a, b) for a, b in zip(fields[7000], fields[30000])]
    passed = all(all(item.values()) if isinstance(item, dict) else item for item in gates.values())
    return gates, details, ('NEEDS_CAUSAL_ATTRIBUTION_AND_STRONG_BASELINES' if passed
                           else 'STOP_CURRENT_TRUCK_SHARED_COORDINATE_SCREEN')


def explanatory_figures(iteration, data, common, oof):
    from PIL import Image, ImageDraw

    folder = STATE['output'] / 'figures'
    candidates = []
    for view, entry in enumerate(data):
        budget()
        image = Image.fromarray((np.clip(entry['base'], 0, 1) * 255).round().astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for point in np.flatnonzero(common[view]):
            y, x = GRID[point]
            for value, color in ((entry['d'][point], '#ff9b00'),
                                 (oof['pose_brown']['prediction'][view, point], '#00c8ff')):
                dy, dx = value
                draw.line((float(x), float(y), float(x + 8 * dx), float(y + 8 * dy)), fill=color, width=2)
            candidates.append((float(entry['cost'][point]), view, int(point)))
        draw.rectangle((0, 0, WIDTH, 24), fill='black')
        draw.text((6, 5), f'{iteration} {NAMES[view]} | orange: matched d; cyan: OOF main; arrows x8', fill='white')
        image.save(folder / f'{iteration}_{NAMES[view]}_overlay.png')
    ranked = sorted(candidates)
    selected = [('best', row) for row in ranked[:PANEL_COUNT]] + [('worst', row) for row in ranked[-PANEL_COUNT:]]
    board = Image.new('RGB', (3 * 13 * 8, len(selected) * (13 * 8 + 24)), 'white')
    draw = ImageDraw.Draw(board)
    records = []
    for row_index, (kind, (cost, view, point)) in enumerate(selected):
        entry = data[view]
        center = GRID[point:point + 1]
        gt_patch = rgb_patches(entry['gt'], center, PATCH)[0]
        base_patch = rgb_patches(entry['base'], center + entry['d'][point], PATCH)[0]
        main_patch = oof['pose_brown']['rgb_prediction'][view, point]
        top = row_index * (13 * 8 + 24)
        draw.text((3, top + 4), f'{kind} {NAMES[view]} grid={point} feature cost={cost:.5g}', fill='black')
        for column, patch in enumerate((gt_patch, base_patch, main_patch)):
            image = Image.fromarray((np.clip(patch.reshape(13, 13, 3), 0, 1) * 255).round().astype(np.uint8))
            board.paste(image.resize((104, 104), Image.Resampling.NEAREST), (column * 104, top + 24))
        records.append({'kind': kind, 'view': view, 'name': NAMES[view], 'grid_index': point, 'cost': cost})
    board.save(folder / f'{iteration}_fixed_best_worst_patches.png')
    write_json(folder / f'{iteration}_patch_selection.json', {'columns': ['GT', 'base at matched d', 'OOF native main'],
                                                              'selection': records})
    budget()


def prepare_inputs(args, root):
    contract = root / 'research/experiments/shared-coordinate-residual-v0/CONTRACT.md'
    pin(contract, CONTRACT_SHA)
    for name, expected in HELPER_SHA.items():
        pin(Path(__file__).parent / name, expected)
    for name, expected in NATIVE_SHA.items():
        pin(args.repo / name, expected)
    for package in ('diff-gaussian-rasterization', 'simple-knn'):
        directory = args.repo / 'submodules' / package
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ('.cu', '.cuh', '.h', '.cpp', '.py') and not any(
                    part in ('.git', 'build', '__pycache__') for part in path.relative_to(directory).parts):
                pin(path)
    pin(Path(__file__))
    pin(root / 'research/experiments/launchers/run-shared-coordinate-residual.sh')
    for name in ('ASSUMPTIONS_MAIN.md', 'ASSUMPTIONS_NATIVE.md', 'ASSUMPTIONS_TOOLS.md'):
        pin(root / 'research/experiments/shared-coordinate-residual-v0' / name)
    for path in (args.source / 'sparse/0/cameras.bin', args.source / 'sparse/0/images.bin',
                 args.source / 'sparse/0/points3D.ply'):
        pin(path)
    for name in NAMES:
        pin(args.source / 'images' / name)
    for iteration in (7000, 30000):
        path = args.model / f'point_cloud/iteration_{iteration}/point_cloud.ply'
        STATE['source_ply_pins'][str(iteration)] = {'path': str(path), 'sha256': pin(path)}


def archive_native_runtime():
    for module_name in ('diff_gaussian_rasterization', 'diff_gaussian_rasterization._C', 'simple_knn._C'):
        module = importlib.import_module(module_name)
        pin(Path(module.__file__))
    from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
    cameras = read_intrinsics_binary(str(ARGS.source / 'sparse/0/cameras.bin'))
    require(len(cameras) == 1 and 1 in cameras, 'INCONCLUSIVE_INPUT_IDENTITY', 'Expected one COLMAP camera id 1')
    camera = cameras[1]
    require(camera.model == 'PINHOLE' and camera.width == 1957 and camera.height == 1091
            and np.allclose(camera.params, [1163.2547280302354, 1156.280404988286, 978.5, 545.5], rtol=0, atol=1e-9),
            'INCONCLUSIVE_INPUT_IDENTITY', 'COLMAP PINHOLE metadata mismatch')
    extrinsics = read_extrinsics_binary(str(ARGS.source / 'sparse/0/images.bin'))
    selected_ids = {image.name: image.camera_id for image in extrinsics.values() if image.name in NAMES}
    require(set(selected_ids) == set(NAMES) and set(selected_ids.values()) == {1},
            'INCONCLUSIVE_INPUT_IDENTITY', 'Selected source image camera ids differ')
    STATE['native_runtime'] = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
        'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
        'colmap_selected_camera_ids': selected_ids}


def finalize_integrity():
    record = STATE['integrity']
    for iteration, bundle in CHECKPOINTS.items():
        budget()
        native.assert_frozen(bundle.gaussians)
        after = native.model_hash(bundle.gaussians)
        STATE['parameters_after'][str(iteration)] = after
        require(after == STATE['parameters_before'][str(iteration)],
                'INCONCLUSIVE_INTEGRITY', f'Gaussian parameters changed at {iteration}')
        record['models'][str(iteration)] = 'unchanged'
        require(iteration in SNAPSHOTS, 'INCONCLUSIVE_INTEGRITY', 'Loaded model lacks original camera snapshots')
        for camera, before in SNAPSHOTS[iteration]:
            budget()
            after_camera = camera_snapshot(camera)
            require(all(np.array_equal(np.asarray(before[key]), np.asarray(after_camera[key]), equal_nan=True)
                        for key in before), 'INCONCLUSIVE_INTEGRITY', f'Original camera changed: {camera.image_name}')
        record['cameras'][str(iteration)] = 'exactly_unchanged_no_restore_required'
        save_arrays(STATE['output'] / 'raw' / f'cameras_{iteration}_after.npz',
                    **{f'view{view:02d}__{key}': value for view, (camera, _) in enumerate(SNAPSHOTS[iteration])
                       for key, value in camera_snapshot(camera, include_images=False).items()})
    for path, expected in STATE['input_hashes_before'].items():
        actual = sha(path)
        STATE['input_hashes_after'][path] = actual
        require(actual == expected, 'INCONCLUSIVE_INTEGRITY', f'Source or implementation changed: {path}')
    record['all_recorded_files_rehashed'] = True
    record['loaded_models_and_original_cameras_checked'] = len(record['models']) == len(CHECKPOINTS)
    record['complete'] = True


def main():
    global STATE, FINALIZING, np, tools, native, torch, GRID, PATCH, RAW15, HALO12, SEARCH
    global FOLD, QUADRANT, SHUFFLE, CENTER, EDGE, INTRINSICS, CONFIG, ARGS
    global CHECKPOINTS, SNAPSHOTS, BASE_CACHE, GT_CACHE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = ARGS = parser.parse_args()
    args.repo, args.source, args.model = args.repo.resolve(), args.source.resolve(), args.model.resolve()
    args.output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    CHECKPOINTS, SNAPSHOTS, BASE_CACHE, GT_CACHE = {}, {}, {}, {}
    STATE = {'output': args.output, 'decision': 'INCONCLUSIVE_NOT_COMPLETED', 'failure': None,
             'training_performed': False, 'parameter_updates': 0, 'optimizer_created': False,
             'backward_calls': 0, 'linear_lstsq_calls': 0, 'quadratic_lstsq_calls': 0,
             'input_hashes_before': {}, 'input_hashes_after': {}, 'parameters_before': {}, 'parameters_after': {},
             'source_ply_pins': {}, 'camera_metadata': {}, 'scene_extents': {}, 'artifacts': [],
             'forward_counts': {}, 'gpu_native_render_seconds': 0., 'render_call_wall_seconds': 0.,
             'core_views_complete': [], 'common_capacity': [], 'fit_cells_complete': [], 'tools_complete': False,
             'external_watchdog_completion_required_for_final_validity': True,
             'integrity': {'complete': False, 'models': {}, 'cameras': {}, 'all_recorded_files_rehashed': False}}
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    CONFIG = {'contract_sha256': CONTRACT_SHA, 'repo': str(args.repo), 'source': str(args.source),
              'model_root': str(args.model), 'output': str(args.output),
              'selected_ranks': RANKS, 'selected_names': NAMES,
              'checkpoint_iterations': [7000, 30000], 'cpu_threads': 1, 'process_budget_seconds': TIME_LIMIT,
              'width': WIDTH, 'height': HEIGHT, 'sh_degree': 3, 'white_background': False,
              'antialiasing': False, 'exposure': False, 'arms': ARMS, 'main_cell_order': CELLS,
              'h': H, 'translation_step': 'h * scene_extent', 'spatial_step_px': .125,
              'search_range_px': [-2, 2], 'search_step_px': .25, 'known_shifts_yx': KNOWN_SHIFTS,
              'lstsq_rcond': RCOND, 'rank_relative_threshold': RANK_TOL,
              'seed': 20261023, 'patch_size': 13, 'FD_saved_RGB_size': 15, 'support_halo': 12,
              'training_performed': False, 'Gaussian_parameter_updates': 0,
              'has_2D_matching_quadratic_fits_and_linear_least_squares': True}
    try:
        for folder in ('raw', 'fits', 'figures'):
            (args.output / folder).mkdir()
        write_json(args.output / 'STATUS.json', {'stage': 'STARTED', 'decision': STATE['decision']})
        budget()
        require(args.source == Path('/home/administrator/datasets/tandt_db/tandt/truck')
                and args.model == args.repo / 'output/truck_30000_r1',
                'INCONCLUSIVE_INPUT_IDENTITY', 'Only the locked existing Truck source/model are supported')
        prepare_inputs(args, root)
        import numpy as numpy_module
        import shared_coordinate_tools as tools_module
        import shared_coordinate_native as native_module
        import torch as torch_module
        np, tools, native, torch = numpy_module, tools_module, native_module, torch_module
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.reset_peak_memory_stats()
        GRID = np.array([(32 + 64 * j, 32 + 64 * i) for j in range(8) for i in range(15)], dtype=np.float64)
        PATCH = np.array([(y, x) for y in range(-6, 7) for x in range(-6, 7)], dtype=np.float64)
        RAW15 = np.array([(y, x) for y in range(-7, 8) for x in range(-7, 8)], dtype=np.float64)
        HALO12 = np.array([(y, x) for y in range(-12, 13) for x in range(-12, 13)], dtype=np.float64)
        SEARCH = np.array([(y / 4, x / 4) for y in range(-8, 9) for x in range(-8, 9)], dtype=np.float64)
        ii, jj = np.tile(np.arange(15), 8), np.repeat(np.arange(8), 15)
        FOLD = ((ii // 4 + jj // 4) % 2).astype(np.int64)
        QUADRANT = ((GRID[:, 0] >= (HEIGHT - 1) / 2).astype(int) * 2 + (GRID[:, 1] >= (WIDTH - 1) / 2).astype(int))
        SHUFFLE = np.random.default_rng(20261023).permutation(len(GRID))
        rho = np.sqrt(((GRID[:, 1] - (WIDTH - 1) / 2) / (WIDTH / 2))**2
                      + ((GRID[:, 0] - (HEIGHT - 1) / 2) / (HEIGHT / 2))**2)
        CENTER, EDGE = rho <= .7, rho >= .85
        require(np.all(np.minimum(GRID % 256, 256 - GRID % 256) >= 32),
                'INCONCLUSIVE_SAMPLING_SUPPORT', 'Grid does not meet macroblock support spacing')
        save_arrays(args.output / 'raw' / 'LOCKED_GRID.npz', centers_yx=GRID, grid_i=ii, grid_j=jj,
                    spatial_fold=FOLD, quadrant=QUADRANT, shuffle=SHUFFLE, rho=rho,
                    center=CENTER, edge=EDGE, patch_offsets_yx=PATCH, raw15_offsets_yx=RAW15,
                    halo12_offsets_yx=HALO12, search_offsets_yx=SEARCH)
        bundle30, cameras30 = load_checkpoint(30000, args)
        archive_native_runtime()
        write_json(args.output / 'INPUTS_BEFORE.json', STATE['input_hashes_before'])
        witnesses(bundle30, cameras30)
        # No real GT matching/fitting is reachable before the complete witness stage.
        bundle7, cameras7 = load_checkpoint(7000, args)
        INTRINSICS = {key: STATE['camera_metadata']['30000'][0][key] for key in ('fx', 'fy', 'cx', 'cy')}
        CONFIG.update(intrinsics=INTRINSICS, input_hashes_before=STATE['input_hashes_before'],
                      native_runtime=STATE['native_runtime'])
        write_json(args.output / 'CONFIG.json', CONFIG)
        data = {30000: quality_and_matches(bundle30, cameras30),
                7000: quality_and_matches(bundle7, cameras7)}
        require(STATE['forward_counts'].get('core_base', 0) == 24
                and STATE['forward_counts'].get('core_fd', 0) == 576,
                'INCONCLUSIVE_FORWARD_COMPLETENESS', 'The complete core must contain 600 native forwards')
        common = common_capacity(data)
        main_summary, main_fields, main_oof = {}, {}, {}
        for iteration in (7000, 30000):
            oof, fields = empty_oof(), []
            for cell in CELLS:
                run_cell(CHECKPOINTS[iteration], data[iteration], common, 'main', cell, oof, fields)
            main_oof[iteration], main_fields[iteration] = oof, fields
            main_summary[iteration] = summarize_oof(oof, common, f'main_{iteration}')
            explanatory_figures(iteration, data[iteration], common, oof)
            write_json(args.output / f'MAIN_{iteration}.json', main_summary[iteration])
            save_arrays(args.output / 'raw' / f'main_{iteration}_four_Brown_fields.npz',
                        fields=np.asarray(fields), cell_order=np.asarray(CELLS), centers_yx=GRID)
        gates, details, proposed_decision = science_gates(main_summary, main_fields)
        STATE.update(main_summary=main_summary, scientific_gates=gates, field_diagnostics=details,
                     completed_main_science_decision=proposed_decision)
        write_json(args.output / 'MAIN_PARTIAL_RESULTS.json',
                   {'summary': main_summary, 'gates': gates, 'field_diagnostics': details,
                    'proposed_decision_pending_stress_and_integrity': proposed_decision})
        stress_summary, stress_fields = {}, {}
        try:
            stress_capacity(common)
            for iteration in (7000, 30000):
                oof, fields = empty_oof(), []
                for image_fold in (0, 1):
                    run_cell(CHECKPOINTS[iteration], data[iteration], common, 'stress',
                             (image_fold, None), oof, fields)
                stress_summary[iteration] = summarize_oof(oof, common & EDGE[None, :], f'stress_{iteration}')
                stress_fields[iteration] = fields
                save_arrays(args.output / 'raw' / f'stress_{iteration}_Brown_fields.npz',
                            fields=np.asarray(fields), image_fold_order=np.array([0, 1]), centers_yx=GRID)
                write_json(args.output / f'STRESS_{iteration}.json', stress_summary[iteration])
            STATE['center_edge_stress'] = {'status': 'COMPLETE_DESCRIPTIVE', 'summary': stress_summary,
                'field_rms_per_cell': {str(k): [field_rms(x) for x in v] for k, v in stress_fields.items()},
                'checkpoint_field_cosines_per_cell': [cosine(a, b) for a, b in
                                                       zip(stress_fields[7000], stress_fields[30000])],
                'new_extrapolation_success_gate_or_claim': None}
        except Stop as exc:
            if exc.code in ('INCONCLUSIVE_STRESS_CAPACITY', 'INCONCLUSIVE_POSE_DESIGN',
                            'INCONCLUSIVE_SHARED_POSE_SEPARABILITY', 'INCONCLUSIVE_POSE_LINEARIZATION'):
                STATE['center_edge_stress'] = {'status': 'UNKNOWN', 'code': exc.code,
                    'message': str(exc), 'completed_checkpoint_summaries': stress_summary}
            else:
                raise
        write_json(args.output / 'CENTER_EDGE_STRESS.json', STATE['center_edge_stress'])
        STATE['decision'] = proposed_decision
        progress('science_complete_pending_final_integrity', decision=proposed_decision)
    except BaseException as exc:
        STATE['failure'] = {'code': exc.code if isinstance(exc, Stop) else 'INCONCLUSIVE_EXCEPTION',
            'message': str(exc), 'type': type(exc).__name__, 'traceback': traceback.format_exc(),
            'elapsed_before_failure_save_seconds': time.perf_counter() - PROCESS_START}
        STATE['decision'] = STATE['failure']['code']
    finally:
        FINALIZING = True
        try:
            if torch is not None and torch.cuda.is_initialized():
                STATE['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                STATE['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            finalize_integrity()
            budget()
        except BaseException as exc:
            STATE['integrity']['finalization_error'] = {'code': getattr(exc, 'code', 'INCONCLUSIVE_INTEGRITY'),
                'type': type(exc).__name__, 'message': str(exc)}
            STATE['integrity']['complete'] = False
            if STATE['failure'] is None:
                STATE['failure'] = STATE['integrity']['finalization_error']
                STATE['decision'] = getattr(exc, 'code', 'INCONCLUSIVE_INTEGRITY')
        if STATE.get('signals') and STATE['failure'] is None:
            STATE['failure'] = STATE['signals'][0]
            STATE['decision'] = 'INCONCLUSIVE_BUDGET'
        STATE['native_forward_count_total'] = sum(STATE['forward_counts'].values())
        STATE['native_forward_attempt_count_total'] = sum(STATE.get('forward_attempt_counts', {}).values())
        STATE['unfinished_native_forward_attempts'] = (STATE['native_forward_attempt_count_total']
                                                      - STATE['native_forward_count_total'])
        STATE['elapsed_before_results_serialization_seconds'] = time.perf_counter() - PROCESS_START
        STATE['cpu_process_seconds_before_results_serialization'] = time.process_time() - CPU_START
        STATE['timing_scopes'] = {
            'elapsed_before_results_serialization_seconds': 'Python timer starts before heavy imports; excludes final JSON write and interpreter shutdown; not whole process wall time',
            'cpu_process_seconds_before_results_serialization': 'process_time across loading/render dispatch/CPU matching/fits/IO; excludes final JSON write and shutdown',
            'gpu_native_render_seconds': 'sum of CUDA events enclosing render_rgb including its device-to-CPU transfer; excludes model loading and CPU matching',
            'render_call_wall_seconds': 'sum of native render call wall times including RGB transfer; excludes model loading and fitting',
            'external_process_wall_seconds': 'recorded by launcher WATCHDOG.json after timeout/Python/tee return'}
        if STATE['elapsed_before_results_serialization_seconds'] >= TIME_LIMIT and STATE['failure'] is None:
            STATE['failure'] = {'code': 'INCONCLUSIVE_BUDGET', 'message': 'Budget exhausted before final RESULTS serialization'}
            STATE['decision'] = 'INCONCLUSIVE_BUDGET'
        STATE['output'] = str(STATE['output'])
        out = Path(STATE['output'])
        try:
            write_json(out / 'RESULTS.json', STATE)
            # If saving itself crosses the limit, preserve a budget verdict.
            if time.perf_counter() - PROCESS_START >= TIME_LIMIT and STATE['failure'] is None:
                STATE['failure'] = {'code': 'INCONCLUSIVE_BUDGET', 'message': 'Final RESULTS write crossed scientific budget'}
                STATE['decision'] = 'INCONCLUSIVE_BUDGET'
                write_json(out / 'RESULTS.json', STATE)
            write_json(out / 'STATUS.json', {'stage': 'FINISHED' if STATE['failure'] is None else 'TERMINATED',
                'decision': STATE['decision'], 'original_failure': STATE['failure'],
                'integrity_complete': STATE['integrity']['complete']})
            print('SHARED_COORDINATE_DIAGNOSTIC_TERMINATED ' + STATE['decision'], flush=True)
        except BaseException as exc:
            print(f'FINAL_SAVE_FAILED {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        CHECKPOINTS.clear()
    return 0 if STATE['failure'] is None else 2


if __name__ == '__main__':
    sys.exit(main())
