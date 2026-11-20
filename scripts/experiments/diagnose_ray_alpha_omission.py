"""Frozen native-cache replay for #24. Eight forwards; never calls backward or updates parameters."""
import argparse
import csv
import inspect
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_joint_rounding_full import model_hash, sha


WIDTH, HEIGHT = 979, 546
RANKS = (0, 31, 62, 93, 124, 155, 186, 218)
CAP = float(np.float32(.99))
ALPHA_THRESHOLD = float(np.float32(1) / np.float32(255))
EARLY_T = float(np.float32(.0001))
PARITY_LIMIT = 2e-5
DIFFERENCE_LIMIT = 1 / 255
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
MEMORY_LIMIT = 10 * 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, value):
    temporary = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_pixels():
    xs = ((2 * np.arange(16) + 1) * WIDTH // 32).astype(np.int32)
    ys = ((2 * np.arange(16) + 1) * HEIGHT // 32).astype(np.int32)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.int32)


def raster_node(image):
    pending, seen, found = [image.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__ == '_RasterizeGaussiansBackward':
            found.append(node)
        pending.extend(other for other, _ in node.next_functions)
    require(len(found) == 1, f'Expected one native rasterizer node; got {len(found)}')
    return found[0]


def prefix_views(buffer, layout, selected):
    """Mirror obtain<T>(...,128); views stay on GPU and do not copy the buffer."""
    require(buffer.dtype == torch.uint8 and buffer.is_cuda and buffer.is_contiguous(), 'Invalid native byte buffer')
    require(buffer.data_ptr() % 128 == 0, 'Native buffer base is not 128-byte aligned')
    offset, result = 0, {}
    for name, dtype, shape in layout:
        offset = (offset + 127) // 128 * 128
        count = math.prod(shape)
        length = count * torch.empty((), dtype=dtype).element_size()
        require(offset + length <= buffer.numel(), f'Native prefix out of bounds: {name}')
        if name in selected:
            result[name] = buffer[offset:offset + length].view(dtype).reshape(shape)
        offset += length
    return result


def extract_cache(image, camera, pixel_xy):
    node = raster_node(image)
    saved = node.saved_tensors
    require(len(saved) == 11, 'Native saved_tensors ABI changed')
    colors_precomp, means3d, scales, rotations, cov_precomp, radii, sh, opacities, geom, bins, img = saved
    n, rendered = len(means3d), int(node.num_rendered)
    require(colors_precomp.numel() == 0 and cov_precomp.numel() == 0, 'Expected native SH/covariance path')
    require(tuple(means3d.shape) == (n, 3) and tuple(scales.shape) == (n, 3), 'Unexpected geometry shape')
    require(tuple(rotations.shape) == (n, 4) and tuple(sh.shape) == (n, 16, 3), 'Unexpected SH/rotation shape')
    require(opacities.numel() == n and radii.numel() == n and 0 <= rendered < 2**31, 'Unexpected native count')
    geom_views = prefix_views(geom, [
        ('depths', torch.float32, (n,)), ('clamped', torch.uint8, (n, 3)),
        ('internal_radii', torch.int32, (n,)), ('means2D', torch.float32, (n, 2)),
        ('cov3D', torch.float32, (n, 6)), ('conic_opacity', torch.float32, (n, 4)),
        ('rgb', torch.float32, (n, 3)), ('tiles_touched', torch.int32, (n,))
    ], {'depths', 'means2D', 'conic_opacity', 'rgb'})
    image_views = prefix_views(img, [
        ('final_T', torch.float32, (WIDTH * HEIGHT,)),
        ('n_contrib', torch.int32, (WIDTH * HEIGHT,)),
        ('ranges', torch.int32, (WIDTH * HEIGHT, 2))
    ], {'final_T', 'n_contrib', 'ranges'})
    # Native uint32 storage is viewed as int32; all used IDs/ranges are checked below 2**31.
    point_list = prefix_views(bins, [('point_list', torch.int32, (rendered,))], {'point_list'})['point_list']
    tiles_x = (WIDTH + 15) // 16
    pixel_tiles = pixel_xy[:, 1] // 16 * tiles_x + pixel_xy[:, 0] // 16
    tile_ids = np.unique(pixel_tiles).astype(np.int64)
    tile_index = torch.from_numpy(tile_ids).cuda()
    ranges = image_views['ranges'].index_select(0, tile_index).cpu().numpy().copy()
    require(bool(((ranges >= 0) & (ranges <= rendered)).all()), 'Native tile range out of bounds')
    require(bool((ranges[:, 0] <= ranges[:, 1]).all()), 'Reversed native tile range')
    pieces, offsets = [], [0]
    for start, end in ranges:
        ids = point_list[int(start):int(end)].cpu().numpy().copy()
        require(bool(((ids >= 0) & (ids < n)).all()), 'Native point ID out of bounds')
        pieces.append(ids)
        offsets.append(offsets[-1] + len(ids))
    candidate_ids = np.concatenate(pieces).astype(np.int64)
    unique_ids = np.unique(candidate_ids)
    gpu_ids = torch.from_numpy(unique_ids).cuda()
    cache = {name: value.index_select(0, gpu_ids).cpu().numpy().copy() for name, value in geom_views.items()}
    cache['radii'] = radii.index_select(0, gpu_ids).detach().cpu().numpy().copy()
    # Inactive geometry can be uninitialized. Check only IDs occurring in selected tile lists.
    for name in ('depths', 'means2D', 'conic_opacity', 'rgb'):
        require(bool(np.isfinite(cache[name]).all()), f'Nonfinite selected native cache: {name}')
    require(bool((cache['radii'] > 0).all()) and bool((cache['depths'] > 0).all()), 'Inactive selected native Gaussian')
    require(bool((cache['rgb'] >= 0).all()) and bool((cache['conic_opacity'][:, 3] >= 0).all()), 'Negative native RGB/opacity')
    for start, end in zip(offsets[:-1], offsets[1:]):
        local = np.searchsorted(unique_ids, candidate_ids[start:end])
        require(bool((np.diff(cache['depths'][local]) >= 0).all()), 'Native tile list is not depth sorted')
    pixels_flat = pixel_xy[:, 1].astype(np.int64) * WIDTH + pixel_xy[:, 0]
    pixel_index = torch.from_numpy(pixels_flat).cuda()
    native_rgb = image.detach().reshape(3, -1).index_select(1, pixel_index).T.cpu().numpy().copy()
    gt = camera.original_image.detach().cpu().numpy()
    require(tuple(gt.shape) == (3, HEIGHT, WIDTH), 'Unexpected GT shape')
    gt_rgb = gt[:, pixel_xy[:, 1], pixel_xy[:, 0]].T.copy()
    require(bool(np.isfinite(native_rgb).all()) and bool(np.isfinite(gt_rgb).all()), 'Nonfinite sampled image')
    cache.update(pixel_xy=pixel_xy, tile_ids=tile_ids, tile_ranges=ranges, tile_offsets=np.array(offsets, dtype=np.int64),
        pixel_tile_index=np.searchsorted(tile_ids, pixel_tiles), candidate_ids=candidate_ids, unique_ids=unique_ids,
        native_rgb=native_rgb, gt_rgb=gt_rgb,
        native_T=image_views['final_T'].index_select(0, pixel_index).cpu().numpy().copy(),
        native_n_contrib=image_views['n_contrib'].index_select(0, pixel_index).cpu().numpy().copy(),
        image_size=np.array([WIDTH, HEIGHT], dtype=np.int32), point_count=np.int64(n), num_rendered=np.int64(rendered))
    require(bool(np.isfinite(cache['native_T']).all()), 'Nonfinite selected native T')
    require(bool(((cache['native_T'] >= 0) & (cache['native_T'] <= 1)).all()), 'Native T out of range')
    return cache


def ray_alphas(means, conic, xy, dtype):
    d = means.astype(dtype) - np.asarray(xy, dtype=dtype)
    q = conic.astype(dtype)
    power = -dtype(.5) * (q[:, 0] * d[:, 0] * d[:, 0] + q[:, 2] * d[:, 1] * d[:, 1]) - q[:, 1] * d[:, 0] * d[:, 1]
    require(bool(np.isfinite(power).all()), 'Nonfinite replay power')
    alpha = np.zeros(len(power), dtype=dtype)
    valid = power <= 0
    alpha[valid] = np.minimum(dtype(CAP), q[valid, 3] * np.exp(power[valid]))
    require(bool(np.isfinite(alpha).all()), 'Nonfinite replay alpha')
    return alpha, valid


def composite(alpha, colors, dtype, threshold, early_stop):
    indices = np.flatnonzero(alpha >= threshold if threshold else alpha > 0)
    selected = alpha[indices]
    after = np.cumprod(dtype(1) - selected, dtype=dtype)
    cutoff = len(indices)
    stopped_at = 0
    if early_stop:
        crosses = np.flatnonzero(after < dtype(EARLY_T))
        if len(crosses):
            cutoff = int(crosses[0])
            stopped_at = int(indices[cutoff]) + 1
    indices, selected, after = indices[:cutoff], selected[:cutoff], after[:cutoff]
    rgb, final_t = np.zeros(3, dtype=dtype), dtype(1)
    if len(indices):
        before = np.concatenate((np.ones(1, dtype=dtype), after[:-1]))
        terms = (colors[indices].astype(dtype) * selected[:, None]) * before[:, None]
        rgb = np.cumsum(terms, axis=0, dtype=dtype)[-1]
        final_t = after[-1]
    return dict(rgb=np.clip(rgb, 0, 1), unclamped_rgb=rgb, final_T=float(final_t),
                total_alpha=1 - float(final_t), accepted_count=len(indices),
                last_contributor=int(indices[-1]) + 1 if len(indices) else 0, stopped_at=stopped_at)


def replay_cache(cache, view, budget=lambda: None):
    """Pure CPU replay; saved NPZ keys can be passed directly as this mapping."""
    rows = []
    unique = cache['unique_ids']
    for pi, xy in enumerate(cache['pixel_xy']):
        tile = int(cache['pixel_tile_index'][pi])
        start, end = cache['tile_offsets'][tile:tile + 2]
        ids = cache['candidate_ids'][int(start):int(end)]
        local = np.searchsorted(unique, ids)
        means, conic, colors = cache['means2D'][local], cache['conic_opacity'][local], cache['rgb'][local]
        a32, valid32 = ray_alphas(means, conic, xy, np.float32)
        a64, valid64 = ray_alphas(means, conic, xy, np.float64)
        arms = {'A': composite(a32, colors, np.float32, ALPHA_THRESHOLD, True),
                'B': composite(a64, colors, np.float64, ALPHA_THRESHOLD, False),
                'C': composite(a64, colors, np.float64, 0, False)}
        native, gt = cache['native_rgb'][pi].astype(np.float64), cache['gt_rgb'][pi].astype(np.float64)
        positive = a64 > 0
        omitted = positive & (a64 < ALPHA_THRESHOLD)
        max_positive_rgb = float(colors[positive].max()) if positive.any() else 0.
        difference = float(np.max(np.abs(arms['B']['rgb'] - arms['C']['rgb'])))
        row = dict(view=view, sample=pi, x=int(xy[0]), y=int(xy[1]), tile_id=int(cache['tile_ids'][tile]),
            candidate_count=len(ids), power_excluded_A=int((~valid32).sum()), power_excluded_BC=int((~valid64).sum()),
            positive_alpha_count=int(positive.sum()), omitted_count=int(omitted.sum()),
            omitted_alpha_sum=float(a64[omitted].sum(dtype=np.float64)),
            zero_alpha_with_positive_opacity_count=int((valid64 & (conic[:, 3] > 0) & ~positive).sum()),
            candidate_max_rgb=float(colors.max()) if len(colors) else 0., positive_alpha_max_rgb=max_positive_rgb,
            all_positive_alpha_rgb_le_1=max_positive_rgb <= 1,
            native_T=float(cache['native_T'][pi]), native_total_alpha=1 - float(cache['native_T'][pi]),
            native_n_contrib=int(cache['native_n_contrib'][pi]),
            parity_max_channel_error=float(np.max(np.abs(native - arms['A']['rgb']))),
            parity_T_absolute_error=abs(float(cache['native_T'][pi]) - arms['A']['final_T']),
            parity_n_contrib_exact=int(cache['native_n_contrib'][pi]) == arms['A']['last_contributor'],
            B_C_max_channel_difference=difference, B_C_exceeds_1_over_255=difference > DIFFERENCE_LIMIT,
            native_gt_mse=float(np.mean((native - gt)**2)))
        for ci, channel in enumerate('rgb'):
            row['native_' + channel] = float(native[ci])
            row['GT_' + channel] = float(gt[ci])
        for arm, values in arms.items():
            for ci, channel in enumerate('rgb'):
                row[arm + '_' + channel] = float(values['rgb'][ci])
                row[arm + '_unclamped_' + channel] = float(values['unclamped_rgb'][ci])
            for name in ('final_T', 'total_alpha', 'accepted_count', 'last_contributor', 'stopped_at'):
                row[arm + '_' + name] = values[name]
            row[arm + '_gt_mse'] = float(np.mean((values['rgb'].astype(np.float64) - gt)**2))
        require(all(math.isfinite(v) for v in row.values() if isinstance(v, float)), 'Nonfinite replay metric')
        rows.append(row)
        if pi % 16 == 0:
            budget()
    return rows


def summarize(rows, view):
    return dict(view=view, pixels=len(rows),
        max_parity_channel_error=max(r['parity_max_channel_error'] for r in rows),
        max_parity_T_error=max(r['parity_T_absolute_error'] for r in rows),
        n_contrib_mismatch_count=sum(not r['parity_n_contrib_exact'] for r in rows),
        B_C_exceed_count=sum(r['B_C_exceeds_1_over_255'] for r in rows),
        ordinary_rgb_exceed_count=sum(r['B_C_exceeds_1_over_255'] and r['all_positive_alpha_rgb_le_1'] for r in rows),
        max_B_C_difference=max(r['B_C_max_channel_difference'] for r in rows),
        native_gt_mse=float(np.mean([r['native_gt_mse'] for r in rows])),
        A_gt_mse=float(np.mean([r['A_gt_mse'] for r in rows])),
        B_gt_mse=float(np.mean([r['B_gt_mse'] for r in rows])),
        C_gt_mse=float(np.mean([r['C_gt_mse'] for r in rows])))


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'source', 'model', 'contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out, start = args.output, time.perf_counter()
    result = dict(decision='INCONCLUSIVE', tool_valid=False, training_performed=False, backward_calls=0,
                  optimizer_created=False, parameter_updates=0, native_forward_calls=0, integrity_complete=False)

    def timeout():
        dump(out / 'RESULTS.json', dict(result, tool_failure='300 second whole-stage timeout',
                                      elapsed_seconds=time.perf_counter() - start))
        os._exit(2)

    watchdog = threading.Timer(300, timeout)
    watchdog.daemon = True
    watchdog.start()

    def budget():
        require(time.perf_counter() - start < 300, '300 second whole-stage budget exceeded')
        require(torch.cuda.max_memory_allocated() <= MEMORY_LIMIT, '10 GiB allocated budget exceeded')

    try:
        teacher = args.model / 'point_cloud/iteration_30000/point_cloud.ply'
        require(sha(teacher) == TEACHER_SHA, 'Unexpected teacher PLY hash')
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
        relevant += [Path(__file__), args.contract, teacher, Path(native.__file__), Path(native._C.__file__),
                     Path(inspect.getfile(render)), Path(inspect.getfile(GaussianModel)),
                     Path(__file__).with_name('evaluate_joint_rounding_full.py')]
        hashes_before = {str(path.resolve()): sha(path) for path in relevant}
        config = dict(script_sha256=sha(Path(__file__)), contract_sha256=sha(args.contract),
            teacher_sha256=TEACHER_SHA, file_hashes_before=hashes_before,
            runtime_wrapper=str(Path(native.__file__).resolve()), runtime_binary=str(Path(native._C.__file__).resolve()),
            train_ranks=RANKS, image_size=[WIDTH, HEIGHT], grid=[16, 16], total_pixels=2048,
            alpha_cap_float32_value=CAP, alpha_threshold_float32_value=ALPHA_THRESHOLD,
            early_T_float32_value=EARLY_T, native_A_parity_limit=PARITY_LIMIT,
            B_C_clamped_difference_limit=DIFFERENCE_LIMIT,
            gates=dict(minimum_total_fraction=.01, minimum_per_view_fraction=.01,
                       minimum_distinct_views=2, minimum_ordinary_rgb_exceed_fraction=.5, require_C_gt_mse_le_B=True),
            antialiasing=False, exposure=False, black_background=True, native_SH=True,
            torch_version=torch.__version__, torch_cuda=torch.version.cuda, numpy_version=np.__version__,
            python_version=sys.version, stage_budget_seconds=300, memory_budget_bytes=MEMORY_LIMIT)
        dump(out / 'CONFIG.json', config)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., MEMORY_LIMIT / torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        config['gpu_name'] = torch.cuda.get_device_name(0)
        dataset = SimpleNamespace(source_path=str(args.source), model_path=str(args.model), images='images',
            depths='', resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        model = GaussianModel(3, 'default')
        scene = Scene(dataset, model, load_iteration=30000, shuffle=False)
        train = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)
        require(len(train) == 219 and model.active_sh_degree == 3, 'Unexpected cameras/SH degree')
        require(len(model._xyz) == 2050770 and model.optimizer is None, 'Unexpected teacher state')
        before = model_hash(model)
        config['parameters_before'] = before
        config['point_count'] = len(model._xyz)
        config['views'] = [train[rank].image_name for rank in RANKS]
        dump(out / 'CONFIG.json', config)
        bg, pixels = torch.zeros(3, device='cuda'), sample_pixels()
        require(len(pixels) == len(np.unique(pixels, axis=0)) == 256, 'Invalid fixed pixel grid')
        all_rows, summaries, traces = [], [], []
        for rank in RANKS:
            camera, view_start = train[rank], time.perf_counter()
            # Autograd stays enabled solely to retain the native forward cache. No backward is called.
            rendered = render(camera, model, pipe, bg, separate_sh=False, use_trained_exp=False)
            result['native_forward_calls'] += 1
            image = rendered['render']
            require(tuple(image.shape) == (3, HEIGHT, WIDTH), 'Unexpected native image shape')
            cache = extract_cache(image, camera, pixels)
            budget()
            cache_path = out / f'CACHE_{rank:03d}.npz'
            np.savez(cache_path, **cache)
            traces.append(dict(view=camera.image_name, rank=rank, filename=cache_path.name,
                               sha256=sha(cache_path), selected_candidates=len(cache['candidate_ids']),
                               unique_selected_ids=len(cache['unique_ids']), native_num_rendered=int(cache['num_rendered'])))
            del image, rendered
            rows = replay_cache(cache, camera.image_name, budget)
            summary = summarize(rows, camera.image_name)
            summary['elapsed_seconds'] = time.perf_counter() - view_start
            all_rows.extend(rows)
            summaries.append(summary)
            write_csv(out / 'PIXELS.csv', all_rows)
            write_csv(out / 'SUMMARY.csv', summaries)
            dump(out / 'TRACES.json', traces)
            print(json.dumps(summary), flush=True)
            require(summary['max_parity_channel_error'] <= PARITY_LIMIT, 'A/native parity exceeds frozen 2e-5')
            del cache
            budget()
        require(len(all_rows) == 2048 and result['native_forward_calls'] == 8, 'Incomplete fixed sample')
        after = model_hash(model)
        require(after == before and model.optimizer is None, 'Teacher state changed')
        hashes_after = {path: sha(Path(path)) for path in hashes_before}
        require(hashes_after == hashes_before, 'Recorded source/binary/contract/script/teacher file changed')
        for trace in traces:
            require(sha(out / trace['filename']) == trace['sha256'], 'Saved trace changed')
        exceed = sum(s['B_C_exceed_count'] for s in summaries)
        ordinary = sum(s['ordinary_rgb_exceed_count'] for s in summaries)
        views = [s['view'] for s in summaries if s['B_C_exceed_count'] >= .01 * s['pixels']]
        mean_b = float(np.mean([r['B_gt_mse'] for r in all_rows]))
        mean_c = float(np.mean([r['C_gt_mse'] for r in all_rows]))
        gates = dict(total_exceed_fraction_at_least_1pct=exceed >= .01 * len(all_rows),
            at_least_2_views_each_1pct=len(views) >= 2,
            at_least_half_exceed_pixels_all_positive_alpha_rgb_le_1=exceed > 0 and ordinary >= .5 * exceed,
            C_overall_gt_mse_le_B=mean_c <= mean_b)
        budget()
        result.update(tool_valid=True, integrity_complete=True,
            decision='NEEDS_FULL_FROZEN_VALIDATION' if all(gates.values()) else 'STOP_CURRENT_CUMULATIVE_OMISSION_DRIVEN_DEPLOYMENT',
            gates=gates, pixels=len(all_rows), B_C_exceed_count=exceed, ordinary_rgb_exceed_count=ordinary,
            views_each_1pct=views, B_overall_gt_mse=mean_b, C_overall_gt_mse=mean_c,
            max_native_A_parity_error=max(s['max_parity_channel_error'] for s in summaries),
            parameters_before=before, parameters_after=after, file_hashes_after=hashes_after,
            trace_manifest_sha256=sha(out / 'TRACES.json'), pixels_csv_sha256=sha(out / 'PIXELS.csv'),
            summary_csv_sha256=sha(out / 'SUMMARY.csv'), peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            scope='Fixed native tile candidates only. No D algorithm, bbox-exterior tails, training, optimizer impact, or performance advantage tested.')
    except Exception as exc:
        result.update(decision='INCONCLUSIVE', tool_valid=False, integrity_complete=False,
                      tool_failure=f'{type(exc).__name__}: {exc}')
    finally:
        watchdog.cancel()
        result['elapsed_stage_seconds'] = time.perf_counter() - start
        dump(out / 'RESULTS.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['tool_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
