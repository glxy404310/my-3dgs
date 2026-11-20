"""Frozen matched-geometry sorting diagnostic; never trains or calls backward.

Each position file is independently decoded before native projection/support are
captured. C and D replay exactly the same quantized geometry, colors and support.
Only their tile order differs. See the separately locked experiment contract.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from diagnose_ray_alpha_omission import (
    WIDTH, HEIGHT, RANKS, CAP, ALPHA_THRESHOLD, EARLY_T, PARITY_LIMIT,
    DIFFERENCE_LIMIT, TEACHER_SHA, MEMORY_LIMIT, extract_cache, prefix_views,
    raster_node, sample_pixels,
)
from diagnose_sorting_quantization_capacity import alphas, composite, pixel_inputs
from evaluate_joint_rounding_full import FIELDS, model_hash, sha
from sorting_position_codec import encode_decode


POINTS = 2050770
BUDGET_SECONDS = 600
NEAR = .2
MODES = (
    ('base_float32', 32, None, False),
    ('local_tau025', None, .25, True),
    ('global_tau025', None, .25, False),
    ('local_tau0125', None, .125, True),
    ('uniform8', 8, None, False),
    ('uniform12', 12, None, False),
    ('uniform16', 16, None, False),
    ('uniform20', 20, None, False),
    ('uniform24', 24, None, False),
    ('uniform32', 32, None, False),
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path, value):
    temporary = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    """Require one schema for both view and ALL records; never drop fields."""
    require(bool(rows), 'Cannot write empty metrics')
    fields = list(rows[0])
    require(all(set(row) == set(fields) for row in rows), 'CSV row schema mismatch')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def attributes_hash(model, names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode('ascii'))
        digest.update(getattr(model, name).detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def native_initialized_geometry(image):
    """Only radii>0 geometry is initialized; inactive storage is never read."""
    saved = raster_node(image).saved_tensors
    n, radii, geom = len(saved[1]), saved[5], saved[8]
    views = prefix_views(geom, [
        ('depths', torch.float32, (n,)), ('clamped', torch.uint8, (n, 3)),
        ('internal_radii', torch.int32, (n,)), ('means2D', torch.float32, (n, 2)),
        ('cov3D', torch.float32, (n, 6)), ('conic_opacity', torch.float32, (n, 4)),
        ('rgb', torch.float32, (n, 3)), ('tiles_touched', torch.int32, (n,)),
    ], {'depths', 'means2D'})
    ids_gpu = torch.nonzero(radii > 0, as_tuple=False).flatten()
    ids = ids_gpu.detach().cpu().numpy().astype(np.int64)
    depth = views['depths'].index_select(0, ids_gpu).detach().cpu().numpy().copy()
    means = views['means2D'].index_select(0, ids_gpu).detach().cpu().numpy().copy()
    require(np.isfinite(depth).all() and np.isfinite(means).all(), 'Nonfinite initialized native geometry')
    require((depth > NEAR).all(), 'Native initialized point is behind kernel near plane')
    return dict(ids=ids, depth=depth, means=means)


def matrix_component(xyz, matrix, column):
    """Explicit float32 affine terms in the native storage convention (no BLAS)."""
    value = xyz[:, 0] * np.float32(matrix[0, column])
    value = value + xyz[:, 1] * np.float32(matrix[1, column])
    value = value + xyz[:, 2] * np.float32(matrix[2, column])
    return value + np.float32(matrix[3, column])


def matrix_projection(xyz, view, projection):
    depth = matrix_component(xyz, view, 2)
    homogeneous = np.column_stack([matrix_component(xyz, projection, axis) for axis in (0, 1, 3)])
    reciprocal_w = np.float32(1) / (homogeneous[:, 2] + np.float32(1e-7))
    ndc = homogeneous[:, :2] * reciprocal_w[:, None]
    pixels = ((ndc + np.float32(1)) * np.array([WIDTH, HEIGHT], dtype=np.float32) - np.float32(1)) * np.float32(.5)
    require(np.isfinite(depth).all() and np.isfinite(pixels).all(), 'Nonfinite matrix projection')
    return pixels, depth, ndc


def old_reference(initialized, camera, xyz):
    active = np.zeros(POINTS, dtype=bool)
    depths = np.zeros(POINTS, dtype=np.float32)
    means = np.zeros((POINTS, 2), dtype=np.float32)
    active[initialized['ids']] = True
    depths[initialized['ids']] = initialized['depth']
    means[initialized['ids']] = initialized['means']
    return dict(active=active, depths=depths, means2D=means,
                world_view_transform=camera.world_view_transform.detach().cpu().numpy().copy(),
                full_proj_transform=camera.full_proj_transform.detach().cpu().numpy().copy(),
                point_count=np.int64(len(xyz)))


def old_depths_for_cache(cache, reference, xyz):
    ids = cache['unique_ids']
    require(reference['probe_active'][ids].all(), 'Original native depth probe does not cover every Q candidate')
    depths = reference['probe_depths'][ids].copy()
    missing = ~reference['active'][ids]
    require(np.isfinite(depths).all(), 'Nonfinite old-depth reference')
    return depths, dict(old_inactive_unique_candidate_count=int(missing.sum()),
                        old_inactive_native_probe_depth_min=float(depths[missing].min()) if missing.any() else None,
                        old_inactive_native_probe_depth_max=float(depths[missing].max()) if missing.any() else None,
                        approximate_depth_fallback_count=0)


def capture_depth_probe(camera, model, pipe, bg, render, reference):
    from utils.graphics_utils import getProjectionMatrix
    fov = math.pi - 1e-4
    projection = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=fov, fovY=fov).transpose(0, 1).cuda()
    full_projection = camera.world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    probe_camera = SimpleNamespace(image_width=1, image_height=1, FoVx=fov, FoVy=fov,
                                   world_view_transform=camera.world_view_transform,
                                   full_proj_transform=full_projection, camera_center=camera.camera_center)
    rendered = render(probe_camera, model, pipe, bg, separate_sh=False, use_trained_exp=False)
    probe = native_initialized_geometry(rendered['render'])
    active = np.zeros(POINTS, dtype=bool)
    depth = np.zeros(POINTS, dtype=np.float32)
    active[probe['ids']], depth[probe['ids']] = True, probe['depth']
    common = active & reference['active']
    require(np.array_equal(depth[common].view(np.uint32), reference['depths'][common].view(np.uint32)),
            'Depth probe and original initialized depths differ in float32 bits')
    reference.update(probe_active=active, probe_depths=depth,
                     probe_projection_matrix=projection.detach().cpu().numpy().copy(),
                     probe_full_proj_transform=full_projection.detach().cpu().numpy().copy())
    record = dict(active_points=int(active.sum()), common_original_active_points=int(common.sum()),
                  common_depth_bitwise_equal=True, missing_original_active_points=int((reference['active'] & ~active).sum()),
                  fov_x=fov, fov_y=fov, image_width=1, image_height=1,
                  world_view_bitwise_identical=True, original_xyz_bitwise_identical=True,
                  changed_only_projection_and_canvas=True)
    del rendered, probe
    return record


def displacement_record(mode, view, original, quantized, reference, initialized, tau, q_selected_ids):
    q_active = np.zeros(POINTS, dtype=bool)
    q_active[initialized['ids']] = True
    union_ids = np.flatnonzero(reference['active'] | q_active)
    view_matrix, projection = reference['world_view_transform'], reference['full_proj_transform']
    old_pixels, old_z, old_ndc = matrix_projection(original[union_ids], view_matrix, projection)
    new_pixels, new_z, _ = matrix_projection(quantized[union_ids], view_matrix, projection)
    old_active_local, new_active_local = reference['active'][union_ids], q_active[union_ids]
    old_pixels[old_active_local] = reference['means2D'][union_ids[old_active_local]]
    # Initialized native pixels take precedence over the explicit-matrix fallbacks.
    new_pixels[np.searchsorted(union_ids, initialized['ids'])] = initialized['means']
    difference = np.abs(new_pixels.astype(np.float64) - old_pixels.astype(np.float64))
    maximum = difference.max(axis=1)
    inside = (old_z > NEAR) & (np.abs(old_ndc) <= 1).all(axis=1)
    outside = ~inside
    support_ids = np.union1d(reference['selected_candidate_ids'], q_selected_ids)
    support_mask = np.isin(union_ids, support_ids, assume_unique=True)
    require(int(support_mask.sum()) == len(support_ids), 'Selected support not inside initialized union')
    invalid_original = (old_z <= NEAR) | ~np.isfinite(old_pixels).all(axis=1)
    support_max = float(maximum[support_mask].max()) if support_mask.any() else 0.
    displacement_control = bool(tau is None or (support_max <= tau + 2e-4 and not invalid_original[support_mask].any()))
    record = dict(mode=mode, view=view, tau=tau, union_active_points=len(union_ids),
                  old_active_points=int(old_active_local.sum()), new_active_points=int(new_active_local.sum()),
                  q_only_active_points=int((new_active_local & ~old_active_local).sum()),
                  old_only_active_points=int((old_active_local & ~new_active_local).sum()),
                  original_near_or_behind_count=int((old_z <= NEAR).sum()),
                  quantized_near_or_behind_count=int((new_z <= NEAR).sum()),
                  near_plane_crossing_count=int(((old_z > NEAR) != (new_z > NEAR)).sum()),
                  selected_support_union_points=len(support_ids),
                  selected_support_invalid_original_projection_points=int(invalid_original[support_mask].sum()),
                  selected_support_max_axis_displacement_px=support_max,
                  selected_support_original_near_crossing_count=int(((old_z > NEAR) != (new_z > NEAR))[support_mask].sum()),
                  displacement_tolerance_px=2e-4, displacement_control_valid=displacement_control,
                  actual_projection_definition='Native initialized centers; F32 actual-camera-matrix fallback only for inactive centers. No FMA-equivalence claim for fallback.',
                  theorem_scope='Original center inside +/-1 NDC x/y, native depth>.2; no assertion for outside centers with overlapping footprint.',
                  scene_camera_znear_attribute=float(getattr(view, 'znear', .01)) if not isinstance(view, str) else .01)
    for name, mask in (('all_union', np.ones(len(union_ids), dtype=bool)),
                       ('selected_candidate_support_union', support_mask),
                       ('inside_original_frustum', inside), ('outside_original_frustum_footprint', outside)):
        values = maximum[mask]
        record[name] = dict(points=int(mask.sum()), max_axis_displacement_px=float(values.max()) if len(values) else None,
                            p99_max_axis_displacement_px=float(np.quantile(values, .99)) if len(values) else None,
                            count_above_tau=int((values > tau).sum()) if tau is not None else None)
    return record


def replay_view(cache, reference, xyz, base_pixels, mode, view, budget):
    old_depth, fallback = old_depths_for_cache(cache, reference, xyz)
    rows = []
    max_parity, zero_bitwise = 0., True
    for pi in range(256):
        tile, local = pixel_inputs(cache, pi)
        ids = cache['unique_ids'][local]
        alpha = alphas(cache, local, cache['pixel_xy'][pi])
        colors = cache['rgb'][local].astype(np.float64)
        order_d = np.arange(len(local), dtype=np.int64)
        order_c = np.lexsort((ids, old_depth[local]))
        rgb_c, accepted_c = composite(alpha, colors, order_c)
        rgb_d, accepted_d = composite(alpha, colors, order_d)
        native, gt = cache['native_rgb'][pi].astype(np.float64), cache['gt_rgb'][pi].astype(np.float64)
        base = native if base_pixels is None else base_pixels[pi].astype(np.float64)
        parity = float(np.max(np.abs(rgb_d - native)))
        max_parity = max(max_parity, parity)
        exact = np.array_equal(order_c, order_d) and np.array_equal(rgb_c, rgb_d)
        zero_bitwise = zero_bitwise and exact
        difference = float(np.max(np.abs(rgb_c - rgb_d)))
        row = dict(mode=mode, view=view, sample=pi, x=int(cache['pixel_xy'][pi, 0]), y=int(cache['pixel_xy'][pi, 1]),
                   tile_id=int(cache['tile_ids'][tile]), candidate_count=len(local),
                   old_inactive_candidate_count=int((~reference['active'][ids]).sum()),
                   candidate_order_positions_changed=int(np.count_nonzero(order_c != order_d)),
                   accepted_C=len(accepted_c), accepted_D=len(accepted_d),
                   max_channel_difference=difference, exceeds_1_over_255=difference > DIFFERENCE_LIMIT,
                   order_mse=float(np.mean((rgb_c - rgb_d)**2)),
                   C_gt_mse=float(np.mean((rgb_c - gt)**2)), D_gt_mse=float(np.mean((rgb_d - gt)**2)),
                   base_gt_mse=float(np.mean((base - gt)**2)),
                   C_base_mse=float(np.mean((rgb_c - base)**2)), D_base_mse=float(np.mean((rgb_d - base)**2)),
                   D_native_max_channel_error=parity, zero_order_and_rgb_bitwise=bool(exact))
        for channel, a, b, c, d, e in zip('rgb', rgb_c, rgb_d, base, gt, native):
            row['C_' + channel], row['D_' + channel] = float(a), float(b)
            row['base_' + channel], row['GT_' + channel], row['native_' + channel] = float(c), float(d), float(e)
        require(all(math.isfinite(v) for v in row.values() if isinstance(v, float)), 'Nonfinite replay metric')
        rows.append(row)
        if pi % 16 == 0:
            budget()
    require(max_parity <= PARITY_LIMIT, f'D/native parity failed: {mode}/{view}: {max_parity}')
    if mode in ('base_float32', 'uniform32'):
        require(zero_bitwise, f'Zero-perturbation C/D order or RGB mismatch: {mode}/{view}')
    return rows, dict(fallback, max_D_native_error=max_parity, zero_order_and_rgb_bitwise=bool(zero_bitwise))


def require_native_tile_lexicographic_order(cache):
    """Check the contracted native depth/ID tie rule for every selected tile."""
    for start, end in zip(cache['tile_offsets'][:-1], cache['tile_offsets'][1:]):
        ids = cache['candidate_ids'][int(start):int(end)]
        local = np.searchsorted(cache['unique_ids'], ids)
        require(np.array_equal(np.lexsort((ids, cache['depths'][local])), np.arange(len(ids))),
                'Q native tile list violates (native depth, original ID) ordering')


def summarize(rows, mode, view, encoded_bytes, view_summaries=None):
    differences = np.array([r['max_channel_difference'] for r in rows])
    exceed = int(np.count_nonzero(differences > DIFFERENCE_LIMIT))
    if view_summaries is None:
        qualifying_views = int(exceed >= .01 * len(rows))
        worst_c = float(np.mean([r['C_gt_mse'] for r in rows]))
        worst_d = float(np.mean([r['D_gt_mse'] for r in rows]))
    else:
        qualifying_views = sum(s['view_gate'] for s in view_summaries)
        worst_c = float(np.mean(sorted([s['C_gt_mse'] for s in view_summaries], reverse=True)[:2]))
        worst_d = float(np.mean(sorted([s['D_gt_mse'] for s in view_summaries], reverse=True)[:2]))
    return dict(mode=mode, view=view, encoded_position_bytes=encoded_bytes, pixels=len(rows),
                exceed_count=exceed, exceed_fraction=exceed / len(rows), view_gate=bool(exceed >= .01 * len(rows)),
                qualifying_views=qualifying_views,
                capacity_gate=bool(exceed >= .01 * len(rows) and qualifying_views >= 4) if view == 'ALL' else False,
                max_channel_difference=float(differences.max()), p99_channel_difference=float(np.quantile(differences, .99)),
                order_mse=float(np.mean([r['order_mse'] for r in rows])),
                C_gt_mse=float(np.mean([r['C_gt_mse'] for r in rows])),
                D_gt_mse=float(np.mean([r['D_gt_mse'] for r in rows])),
                base_gt_mse=float(np.mean([r['base_gt_mse'] for r in rows])),
                C_base_mse=float(np.mean([r['C_base_mse'] for r in rows])),
                D_base_mse=float(np.mean([r['D_base_mse'] for r in rows])),
                C_worst2_view_gt_mse=worst_c, D_worst2_view_gt_mse=worst_d,
                max_D_native_error=max(r['D_native_max_channel_error'] for r in rows),
                old_inactive_candidate_occurrences=sum(r['old_inactive_candidate_count'] for r in rows),
                accepted_C=sum(r['accepted_C'] for r in rows), accepted_D=sum(r['accepted_D'] for r in rows),
                candidate_order_positions_changed=sum(r['candidate_order_positions_changed'] for r in rows))


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'source', 'model', 'cache', 'contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out, start = args.output, time.perf_counter()
    result = dict(tool_valid=False, integrity_complete=False, decision='INCONCLUSIVE', training_performed=False,
                  backward_calls=0, optimizer_created=False, optimization_updates=0,
                  fixed_parameter_interventions=True, native_forward_calls=0,
                  phase='initializing', thread_limit=1, hard_process_limit_seconds=BUDGET_SECONDS)
    model = original_parameter = None

    def timeout():
        result.update(tool_valid=False, decision='INCONCLUSIVE_TIMEOUT', phase='hard_watchdog',
                      elapsed_stage_seconds=time.perf_counter() - start)
        dump(out / 'TIMEOUT.json', result)
        os._exit(2)

    watchdog = threading.Timer(BUDGET_SECONDS, timeout)
    watchdog.daemon = True
    watchdog.start()

    def budget():
        require(time.perf_counter() - start < BUDGET_SECONDS, '600 second whole-process budget exceeded')
        require(torch.cuda.max_memory_allocated() <= MEMORY_LIMIT, '10 GiB allocated budget exceeded')

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        teacher = args.model / 'point_cloud/iteration_30000/point_cloud.ply'
        require(sha(teacher) == TEACHER_SHA, 'Teacher PLY SHA mismatch')
        old_config_path, old_traces_path = args.cache / 'CONFIG.json', args.cache / 'TRACES.json'
        old_config = json.loads(old_config_path.read_text())
        old_traces = json.loads(old_traces_path.read_text())
        require([t['rank'] for t in old_traces] == list(RANKS), 'Old cache camera split changed')
        require(old_config['teacher_sha256'] == TEACHER_SHA, 'Old cache teacher identity mismatch')
        require(old_config['train_ranks'] == list(RANKS), 'Old config split mismatch')
        require(old_config['image_size'] == [WIDTH, HEIGHT], 'Old image size mismatch')
        sys.path.insert(0, str(args.repo))
        from scene import Scene, GaussianModel
        from gaussian_renderer import render
        import diff_gaussian_rasterization as native

        relevant = [args.repo / path for path in (
            'gaussian_renderer/__init__.py', 'scene/__init__.py', 'scene/gaussian_model.py',
            'scene/cameras.py', 'scene/dataset_readers.py', 'scene/colmap_loader.py',
            'utils/camera_utils.py', 'utils/graphics_utils.py', 'utils/sh_utils.py',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h',
            'submodules/diff-gaussian-rasterization/rasterize_points.cu',
            'submodules/diff-gaussian-rasterization/rasterize_points.h',
            'submodules/diff-gaussian-rasterization/ext.cpp',
            'submodules/diff-gaussian-rasterization/setup.py',
            'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py')]
        relevant += [Path(__file__), args.contract, args.contract.parent / 'ASSUMPTIONS.md',
                     args.contract.parent.parent / 'launchers/run-sorting-matched-geometry.sh',
                     teacher, args.model / 'cameras.json',
                     Path(native.__file__), Path(native._C.__file__), Path(inspect.getfile(render)),
                     Path(inspect.getfile(GaussianModel)), old_config_path, old_traces_path]
        relevant += [Path(__file__).with_name(name) for name in (
            'evaluate_joint_rounding_full.py', 'diagnose_ray_alpha_omission.py',
            'diagnose_sorting_quantization_capacity.py', 'sorting_position_codec.py')]
        missing_source_paths = [str(path) for path in relevant if not path.is_file()]
        require(not missing_source_paths, 'Missing required source paths: ' + ', '.join(missing_source_paths))
        hashes_before = {str(path.resolve()): sha(path) for path in relevant}
        old_source_checks = {}
        for path, expected in old_config['file_hashes_before'].items():
            if path in hashes_before:
                require(hashes_before[path] == expected, 'Source/binary drift from existing cache: ' + path)
                old_source_checks[path] = expected
        require(str(Path(native._C.__file__).resolve()) in old_source_checks, 'No old native binary anchor')
        old_inputs = []
        for trace in old_traces:
            path = args.cache / trace['filename']
            require(sha(path) == trace['sha256'], 'Old cache hash mismatch')
            old_inputs.append(dict(path=str(path), sha256=trace['sha256']))
            hashes_before[str(path.resolve())] = trace['sha256']
        config = dict(file_hashes_before=hashes_before, old_source_anchor=old_source_checks, old_caches=old_inputs,
                      modes=[dict(name=n, bits=b, tau=t, local=l) for n, b, t, l in MODES],
                      train_ranks=list(RANKS), image_size=[WIDTH, HEIGHT], grid=[16, 16],
                      points=POINTS, expected_native_forwards=88, expected_depth_probe_forwards=8, expected_metric_rows=20480,
                      alpha_cap=CAP, alpha_threshold=ALPHA_THRESHOLD, early_T=EARLY_T,
                      parity_limit=PARITY_LIMIT, channel_difference_limit=DIFFERENCE_LIMIT,
                      native_near_cull=NEAR, primary_mode='local_tau025',
                      primary_gates=dict(exceed_fraction_at_least=.01, views_at_least=4, view_exceed_fraction_at_least=.01,
                                         C_gt_mse_at_most_D_multiple=.95, C_own_worst2_not_worse=True,
                                         no_same_or_lower_bytes_uniform_control_eliminates_capacity=True),
                      C_order='Original native float32 depth from original-XYZ same-world-view 1x1 large-FOV probe; original point ID breaks exact depth ties. No approximate depth fallback.',
                      D_order='Fresh Q native tile candidate list, with fresh Q projection/conic/SH/support.',
                      antialiasing=False, exposure=False, black_background=True, native_SH=True,
                      depth_probe_fov=math.pi - 1e-4, depth_probe_common_depth_required_bitwise=True,
                      actual_displacement_control_scope='Original/Q selected tile candidate support union, including outside centers',
                      actual_displacement_tolerance_px=2e-4, invalid_original_projection_allowed=False,
                      backward_calls=0, optimizer_updates=0,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      numpy_version=np.__version__, python_version=sys.version,
                      hard_process_limit_seconds=BUDGET_SECONDS, memory_limit_bytes=MEMORY_LIMIT)
        dump(out / 'CONFIG.json', config)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., MEMORY_LIMIT / torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        config['gpu_name'] = torch.cuda.get_device_name(0)
        dataset = SimpleNamespace(source_path=str(args.source), model_path=str(args.model), images='images', depths='',
                                  resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        model = GaussianModel(3, 'default')
        # Suppress only repetitive camera-loader progress; compact stage summaries remain visible.
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            scene = Scene(dataset, model, load_iteration=30000, shuffle=False)
        train = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)
        require(len(train) == 219 and model.active_sh_degree == 3, 'Unexpected train split/SH degree')
        require(len(model._xyz) == POINTS and model.optimizer is None, 'Unexpected model state')
        require([train[r].image_name for r in RANKS] == [t['view'] for t in old_traces], 'Camera names differ from old cache')
        original_parameter = model._xyz
        original = original_parameter.detach().cpu().contiguous().numpy().copy()
        full_before, other_before = model_hash(model), attributes_hash(model, FIELDS[1:])
        fmax = max(max(c.image_width / (2 * math.tan(c.FoVx / 2)),
                       c.image_height / (2 * math.tan(c.FoVy / 2))) for c in (train[r] for r in RANKS))
        phi = max(max(train[r].FoVx, train[r].FoVy) / 2 for r in RANKS)
        epsilons = {str(tau): tau * NEAR / (math.sqrt(3) * (fmax / math.cos(phi) + tau)) for tau in (.25, .125)}
        config.update(parameters_before=full_before, non_xyz_parameters_before=other_before, fmax=fmax,
                      maximum_fov_half_radians=phi, epsilons=epsilons, views=[train[r].image_name for r in RANKS],
                      scene_camera_znear_attributes=[float(train[r].znear) for r in RANKS],
                      scope='Eight fixed diagnostic cameras only; actual encoded position stream, not an optimized whole-model compressor. No claim of training, wall-time gain, novel algorithm or continuous-domain image guarantee.')
        dump(out / 'CONFIG.json', config)
        budget()
        pixels, bg = sample_pixels(), torch.zeros(3, device='cuda')
        require(len(pixels) == len(np.unique(pixels, axis=0)) == 256, 'Fixed pixel grid invalid')
        rows, summaries, traces, displacements, codec_records, mode_checks = [], [], [], [], [], []
        base_references, base_rgb = {}, {}
        for mode, bits, tau, local in MODES:
            result['phase'] = 'mode_' + mode
            mode_start = time.perf_counter()
            encoded_path = out / (mode + '.positions.bin')
            epsilon = epsilons[str(tau)] if tau is not None else None
            quantized, codec_meta = encode_decode(original, encoded_path, bits=bits, epsilon=epsilon, local=local)
            budget()
            require(quantized.dtype == np.float32 and quantized.shape == original.shape, 'Codec returned unexpected positions')
            require(codec_meta['bytes'] == encoded_path.stat().st_size, 'Codec actual-byte mismatch')
            if bits == 32:
                require(np.array_equal(quantized.view(np.uint32), original.view(np.uint32)), '32bit did not preserve XYZ bits')
            codec_record = dict(mode=mode, filename=encoded_path.name, sha256=sha(encoded_path), **codec_meta)
            codec_records.append(codec_record)
            model._xyz = torch.nn.Parameter(torch.from_numpy(quantized).cuda(), requires_grad=False)
            require(attributes_hash(model, FIELDS[1:]) == other_before, 'Non-XYZ parameters changed before mode')
            view_summaries, mode_rows = [], []
            for rank, old_trace in zip(RANKS, old_traces):
                camera = train[rank]
                rendered = render(camera, model, pipe, bg, separate_sh=False, use_trained_exp=False)
                result['native_forward_calls'] += 1
                image = rendered['render']
                require(tuple(image.shape) == (3, HEIGHT, WIDTH), 'Unexpected native image size')
                cache = extract_cache(image, camera, pixels)
                require_native_tile_lexicographic_order(cache)
                initialized = native_initialized_geometry(image)
                if mode == 'base_float32':
                    reference = old_reference(initialized, camera, original)
                    reference['selected_candidate_ids'] = cache['unique_ids'].copy()
                    probe_record = capture_depth_probe(camera, model, pipe, bg, render, reference)
                    result['native_forward_calls'] += 1
                    reference_path = out / f'BASE_REFERENCE_{rank:03d}.npz'
                    np.savez(reference_path, **reference)
                    base_references[rank] = reference
                    base_rgb[rank] = cache['native_rgb'].copy()
                    with np.load(args.cache / old_trace['filename'], allow_pickle=False) as archive:
                        require(np.array_equal(archive['pixel_xy'], pixels), 'Old pixel grid mismatch')
                        require(np.array_equal(archive['gt_rgb'], cache['gt_rgb']), 'Old GT differs')
                        old_native_error = float(np.max(np.abs(archive['native_rgb'] - cache['native_rgb'])))
                    require(old_native_error <= PARITY_LIMIT, 'Base native differs from old native cache')
                else:
                    reference = base_references[rank]
                    reference_path = out / f'BASE_REFERENCE_{rank:03d}.npz'
                    old_native_error = None
                    probe_record = None
                depth_reference, fallback = old_depths_for_cache(cache, reference, original)
                cache.update(old_depth_reference=depth_reference, old_reference_active=reference['active'][cache['unique_ids']],
                             base_native_rgb=base_rgb[rank], q_active_ids=initialized['ids'],
                             q_active_depths=initialized['depth'], q_active_means2D=initialized['means'])
                path = out / f'CACHE_{mode}_{rank:03d}.npz'
                np.savez(path, **cache)
                trace = dict(mode=mode, view=camera.image_name, rank=rank, filename=path.name, sha256=sha(path),
                             original_reference_filename=reference_path.name, original_reference_sha256=sha(reference_path),
                             selected_candidate_occurrences=len(cache['candidate_ids']),
                             selected_unique_candidates=len(cache['unique_ids']), new_active_points=len(initialized['ids']),
                             native_num_rendered=int(cache['num_rendered']), old_native_cache_max_error=old_native_error,
                             depth_probe=probe_record,
                             **fallback)
                view_rows, checks = replay_view(cache, reference, original, base_rgb[rank], mode, camera.image_name, budget)
                trace.update(checks)
                traces.append(trace)
                displacement = displacement_record(mode, camera.image_name, original, quantized, reference, initialized, tau, cache['unique_ids'])
                displacement['scene_camera_znear_attribute'] = float(camera.znear)
                displacements.append(displacement)
                view_summary = summarize(view_rows, mode, camera.image_name, codec_meta['bytes'])
                view_summaries.append(view_summary)
                mode_rows.extend(view_rows)
                del image, rendered, cache, initialized
                budget()
            require(attributes_hash(model, FIELDS[1:]) == other_before, 'Non-XYZ parameters changed during mode')
            require(np.array_equal(model._xyz.detach().cpu().numpy().view(np.uint32), quantized.view(np.uint32)), 'Native forward changed decoded Q')
            mode_checks.append(dict(mode=mode, non_xyz_hash=other_before, xyz_matches_file_decode=True,
                                    no_optimizer=model.optimizer is None, elapsed_seconds=time.perf_counter() - mode_start))
            model._xyz = original_parameter
            require(model._xyz is original_parameter, 'Original XYZ object not restored after mode')
            overall = summarize(mode_rows, mode, 'ALL', codec_meta['bytes'], view_summaries)
            summaries.extend(view_summaries + [overall])
            rows.extend(mode_rows)
            write_csv(out / 'METRICS.csv', rows)
            write_csv(out / 'SUMMARY.csv', summaries)
            dump(out / 'TRACES.json', traces)
            dump(out / 'DISPLACEMENTS.json', displacements)
            dump(out / 'CODECS.json', codec_records)
            dump(out / 'MODE_CHECKS.json', mode_checks)
            print(json.dumps(dict(mode=mode, rows=len(rows), encoded_position_bytes=codec_meta['bytes'],
                                  exceed_fraction=overall['exceed_fraction'], qualifying_views=overall['qualifying_views'],
                                  C_gt_mse=overall['C_gt_mse'], D_gt_mse=overall['D_gt_mse'],
                                  elapsed_seconds=round(time.perf_counter() - start, 3))), flush=True)
            budget()
        require(len(rows) == 20480 and len(summaries) == 90 and len(traces) == 80, 'Incomplete fixed experiment')
        require(result['native_forward_calls'] == 88, 'Unexpected native forward count')
        require(model._xyz is original_parameter and model.optimizer is None, 'Original model object/optimizer integrity failed')
        full_after = model_hash(model)
        require(full_after == full_before, 'Final six-parameter hash changed')
        hashes_after = {path: sha(Path(path)) for path in hashes_before}
        require(hashes_after == hashes_before, 'Source/contract/codec/binary/teacher/cache drift')
        for trace in traces:
            require(sha(out / trace['filename']) == trace['sha256'], 'New native cache changed')
        for record in codec_records:
            require(sha(out / record['filename']) == record['sha256'], 'Encoded position file changed')
        overall = {s['mode']: s for s in summaries if s['view'] == 'ALL'}
        primary = overall['local_tau025']
        ordinary_controls = [overall[name] for name, _, _, _ in MODES if name.startswith('uniform')]
        eliminating_controls = [s['mode'] for s in ordinary_controls
                                if s['encoded_position_bytes'] <= primary['encoded_position_bytes'] and not s['capacity_gate']]
        gates = dict(primary_total_exceed_fraction_at_least_1pct=primary['exceed_fraction'] >= .01,
                     primary_at_least_4_views_each_1pct=primary['qualifying_views'] >= 4,
                     primary_C_GT_mean_at_least_5pct_better=primary['C_gt_mse'] <= .95 * primary['D_gt_mse'],
                     primary_C_own_worst2_not_worse=primary['C_worst2_view_gt_mse'] <= primary['D_worst2_view_gt_mse'],
                     no_same_or_lower_bytes_uniform_control_eliminates_capacity=not eliminating_controls)
        primary_displacements = [d for d in displacements if d['mode'] == 'local_tau025']
        displacement_control_valid = all(d['displacement_control_valid'] for d in primary_displacements)
        decision = ('NEEDS_ACTUAL_ENCODING_BENEFIT_CHECK' if all(gates.values()) else 'STOP_CURRENT_SORTING_QUANTIZATION_ROUTE')
        if not displacement_control_valid:
            decision = 'INCONCLUSIVE_DISPLACEMENT_CONTROL_NOT_MET'
        result.update(tool_valid=True, integrity_complete=True,
                      decision=decision, primary_displacement_control_valid=displacement_control_valid,
                      gates=gates, primary=primary, eliminating_uniform_controls=eliminating_controls,
                      ordinary_controls=ordinary_controls, pixels=len(rows), traces=len(traces),
                      parameters_before=full_before, parameters_after=full_after, original_xyz_object_restored=True,
                      non_xyz_parameters_hash=other_before, file_hashes_after=hashes_after,
                      max_D_native_error=max(t['max_D_native_error'] for t in traces),
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      output_hashes={name: sha(out / name) for name in ('CONFIG.json', 'TRACES.json', 'CODECS.json',
                                     'METRICS.csv', 'SUMMARY.csv', 'DISPLACEMENTS.json', 'MODE_CHECKS.json')},
                      scope=config['scope'])
        budget()
    except Exception as exc:
        result.update(tool_valid=False, integrity_complete=False, decision='INCONCLUSIVE',
                      tool_failure=f'{type(exc).__name__}: {exc}')
    finally:
        if model is not None and original_parameter is not None:
            model._xyz = original_parameter
            result['original_xyz_object_restored'] = model._xyz is original_parameter
            if 'full_before' in locals():
                result['parameters_after_final_restore'] = model_hash(model)
                result['parameters_restored_after_exit'] = result['parameters_after_final_restore'] == full_before
        watchdog.cancel()
        result['elapsed_stage_seconds'] = time.perf_counter() - start
        dump(out / 'RESULTS.json', result)
    print(json.dumps({key: result[key] for key in (
        'decision', 'tool_valid', 'integrity_complete', 'native_forward_calls',
        'elapsed_stage_seconds', 'gates', 'primary_displacement_control_valid',
        'eliminating_uniform_controls', 'tool_failure') if key in result}), flush=True)
    return 0 if result['tool_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
