"""One-shot fixed-set opacity proxy transfers. Forward-only, never an optimizer."""
import time
PROCESS_STARTED = time.perf_counter()

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_joint_rounding_full import FIELDS, sha, model_hash
from diagnose_ray_alpha_omission import raster_node, prefix_views


STATES = ('base', 'A_local', 'A_global', 'A_shuffle', 'B_local', 'B_global', 'B_shuffle')
HISTORICAL = ('gain_1.05', 'gain_1.1', 'gain_1.25', 'opacity_1.1', 'opacity_1.25', 'opacity_1.5')
PROJECTION_RANKS = (0, 72, 145, 218)
PANEL_VIEWS = ('000009.jpg', '000017.jpg', '000105.jpg')
MEMORY_BYTES = 10 * 1024**3
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, value):
    temporary = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def geometry_hash(model):
    digest = hashlib.sha256()
    for field in FIELDS:
        if field != '_opacity':
            digest.update(getattr(model, field).detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def quantiles(values):
    require(len(values) > 0 and bool(np.isfinite(values).all()), 'Invalid explanation values')
    return {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)}


def native_area(image, n, save_projection):
    """Read only initialized native entries; local graph references die on return."""
    node = raster_node(image)
    saved = node.saved_tensors
    require(len(saved) == 11, 'Native saved_tensors ABI changed')
    colors, xyz, scales, rotations, covariances, radii, sh, opacities, geom, bins, img = saved
    require(colors.numel() == covariances.numel() == 0, 'Native SH/covariance path changed')
    require(tuple(xyz.shape) == (n, 3) and tuple(scales.shape) == (n, 3), 'Native geometry shape changed')
    require(tuple(rotations.shape) == (n, 4) and tuple(sh.shape) == (n, 16, 3), 'Native SH/rotation shape changed')
    require(radii.numel() == opacities.numel() == n and bool((radii >= 0).all()), 'Native radii/count changed')
    views = prefix_views(geom, [
        ('depths', torch.float32, (n,)), ('clamped', torch.uint8, (n, 3)),
        ('internal_radii', torch.int32, (n,)), ('means2D', torch.float32, (n, 2)),
        ('cov3D', torch.float32, (n, 6)), ('conic_opacity', torch.float32, (n, 4)),
        ('rgb', torch.float32, (n, 3)), ('tiles_touched', torch.int32, (n,))
    ], {'means2D', 'conic_opacity'})
    active = radii > 0
    with torch.no_grad():
        q = views['conic_opacity'][active].double()
        require(bool(torch.isfinite(q).all()), 'Nonfinite initialized conic')
        determinant = q[:, 0] * q[:, 2] - q[:, 1].square()
        require(bool((q[:, 0] > 0).all()) and bool((q[:, 2] > 0).all())
                and bool((determinant > 0).all()) and bool(torch.isfinite(determinant).all()),
                'Initialized native conic is not positive definite')
        # The public binding supplies its own radii tensor. GeometryState's
        # internal_radii is only written when that argument is nullptr.
        area = torch.zeros(n, device='cuda', dtype=torch.float64)
        area[active] = 2 * np.pi / determinant.sqrt()
        require(bool(torch.isfinite(area).all()), 'Nonfinite native area')
        projection = None
        if save_projection:
            means = views['means2D'][active].detach()
            require(bool(torch.isfinite(means).all()), 'Nonfinite initialized means2D')
            projected = np.zeros((n, 2), dtype=np.float32)
            active_cpu = active.detach().cpu().numpy()
            projected[active_cpu] = means.cpu().numpy()
            projection = dict(means2D=projected, radii=radii.detach().cpu().numpy().copy())
        stats = dict(active_count=int(active.sum()), mean_area=float(area.mean()))
    return area.detach(), projection, stats


def receiver_map(xyz, keep, deleted, weights, budget):
    receivers = keep[weights[keep] > 0]
    donors = deleted[weights[deleted] > 0]
    require(len(receivers) >= 2 and len(donors) > 0, 'Insufficient positive-weight receivers/donors')
    tree = cKDTree(xyz[receivers], copy_data=True)
    _, near = tree.query(xyz[donors], k=2, workers=1)
    exact_d2 = np.sum((xyz[receivers[near]] - xyz[donors, None, :])**2, axis=2)
    first = np.argmin(exact_d2, axis=1)
    assigned = receivers[near[np.arange(len(donors)), first]]
    ties = np.flatnonzero(exact_d2[:, 0] == exact_d2[:, 1])
    for index in ties:
        budget()
        radius = np.nextafter(np.sqrt(exact_d2[index, 0]), np.inf)
        choices = receivers[np.asarray(tree.query_ball_point(xyz[donors[index]], radius, workers=1), dtype=np.int64)]
        require(len(choices) > 0, 'Empty exact-tie ball')
        distances = np.sum((xyz[choices] - xyz[donors[index]])**2, axis=1)
        assigned[index] = np.min(choices[distances == distances.min()])
    distance = np.sqrt(np.sum((xyz[assigned] - xyz[donors])**2, axis=1))
    budget()
    return receivers, donors, assigned, distance, len(ties)


def main():
    started = PROCESS_STARTED
    parser = argparse.ArgumentParser()
    for name in ('repo', 'source', 'model', 'coverage', 'history', 'abi-reference', 'contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    result = dict(decision='TOOL_FAILED', tool_valid=False, phase='initializing', training_performed=False,
        optimizer_created=False, optimizer_updates=0, backward_calls=0, native_forward_calls=0,
        completed_train_views=0, completed_eval_rows=0, fixed_parameter_interventions=True)
    rows, raw_manifest, teacher, student, base_opacity = [], [], None, None, None
    original_forward, gaussian_renderer, watchdog = None, None, None

    def timeout():
        partial = dict(result)
        partial.update(decision='TOOL_FAILED', tool_valid=False, tool_failure='600 second whole-process budget exceeded',
                       elapsed_seconds=time.perf_counter() - started, hard_watchdog_terminated=True)
        dump(out / 'RESULTS.json', partial)
        os._exit(2)

    watchdog = threading.Timer(max(.001, 600 - (time.perf_counter() - started)), timeout)
    watchdog.daemon = True
    watchdog.start()

    def budget():
        require(time.perf_counter() - started < 600, '600 second whole-process budget exceeded')
        if torch.cuda.is_initialized():
            require(torch.cuda.max_memory_allocated() <= MEMORY_BYTES, '10 GiB allocated budget exceeded')

    def save_npz(path, **arrays):
        budget()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        raw_manifest.append(dict(path=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path)))
        budget()

    try:
        inputs = dict(teacher=args.model / 'point_cloud/iteration_30000/point_cloud.ply',
            indices=args.coverage / 'all219_indices.npy', history=args.history / 'METRICS.csv',
            abi_reference=args.abi_reference, contract=args.contract, script=Path(__file__),
            hash_helper=Path(__file__).with_name('evaluate_joint_rounding_full.py'),
            abi_helper=Path(__file__).with_name('diagnose_ray_alpha_omission.py'))
        before = {name: sha(path) for name, path in inputs.items()}
        require(before['teacher'] == TEACHER_SHA, 'Unexpected teacher PLY')
        abi_reference = json.loads(args.abi_reference.read_text(encoding='utf-8'))
        sys.path.insert(0, str(args.repo))
        from scene import Scene, GaussianModel
        import gaussian_renderer
        import diff_gaussian_rasterization as native

        source_paths = [args.repo / path for path in (
            'gaussian_renderer/__init__.py', 'scene/__init__.py', 'scene/gaussian_model.py',
            'scene/cameras.py', 'scene/dataset_readers.py', 'utils/camera_utils.py',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
            'submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h')]
        source_paths.extend((Path(native.__file__), Path(native._C.__file__)))
        source_before = {str(path.resolve()): sha(path) for path in source_paths}
        for path in source_paths:
            key = str(path.resolve())
            require(source_before[key] == abi_reference['file_hashes_before'][key], 'Validated ABI source/binary changed: ' + key)
        require(before['abi_helper'] == abi_reference['script_sha256'], 'Validated ABI helper changed')
        config = dict(input_hashes_before=before, source_hashes_before=source_before, states=list(STATES),
            historical_controls=list(HISTORICAL), historical_controls_reused_not_rerendered=True,
            seed=20261023, nearest_neighbor_workers=1, nearest_neighbor_k=2,
            tie_rule='Recompute float64 squared distances; exact equal k=2 distances trigger nextafter(sqrt(d2),inf) ball and minimum original ID among exact minima.',
            main_state='B_local', all_train_views=219, test_views=32, image_size=[979, 546],
            projection_ranks=list(PROJECTION_RANKS), projection_overlap='Distance <= sum of native ceil-three-sigma radii; report all donating pairs and both-visible conditional fractions.',
            panel_views=list(PANEL_VIEWS), stage_budget_seconds=600, memory_budget_bytes=MEMORY_BYTES,
            uses_training_GT=False, uses_test_GT_for_fit_or_selection=False, test_set_previously_used=True,
            weights_npy_row_order=['A', 'B'], opacity_variants_npy_row_order=list(STATES),
            primitive='tau=softplus(logit); w=1 or mean native post-lowpass Gaussian projected area; fixed nearest surviving center.',
            scope='Known simple proxy mechanism check; no strict optical conservation, optimizer, novelty or deployment-speed claim.')
        dump(out / 'CONFIG.json', config)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., MEMORY_BYTES / torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        dataset = SimpleNamespace(source_path=str(args.source), model_path=str(args.model), images='images',
            depths='', resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        teacher = GaussianModel(3, 'default')
        # Library progress is captured rather than emitted as thousands of stdout lines.
        with contextlib.redirect_stdout(io.StringIO()):
            scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False)
        for field in FIELDS:
            getattr(teacher, field).requires_grad_(False)
        n = len(teacher._xyz)
        require(n == 2050770 and teacher.optimizer is None, 'Unexpected teacher model')
        teacher_before = model_hash(teacher)
        result['teacher_parameters_before'] = teacher_before
        train = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
        test = sorted(scene.getTestCameras(), key=lambda camera: camera.image_name)
        require(len(train) == 219 and len(test) == 32, 'Camera split changed')
        require(all((camera.image_width, camera.image_height) == (979, 546) for camera in train + test), 'Image size changed')
        keep = np.sort(np.load(inputs['indices']).astype(np.int64))
        require(len(keep) == 1025385 and len(np.unique(keep)) == len(keep) and keep[0] >= 0 and keep[-1] < n, 'Invalid fixed survivor set')
        deleted = np.flatnonzero(~np.isin(np.arange(n, dtype=np.int64), keep, assume_unique=True))
        xyz = teacher._xyz.detach().cpu().numpy().astype(np.float64)
        max_scale = np.exp(teacher._scaling.detach().cpu().numpy().astype(np.float64)).max(axis=1)
        logits = teacher._opacity.detach().cpu().numpy().reshape(-1).astype(np.float64)
        tau = np.logaddexp(0., logits)
        require(bool(np.isfinite(xyz).all()) and bool(np.isfinite(max_scale).all()) and bool((max_scale > 0).all())
                and bool(np.isfinite(tau).all()) and bool((tau > 0).all()), 'Invalid frozen teacher values')
        black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
        result['phase'] = '219-native-area-forwards'
        area_sum = torch.zeros(n, device='cuda', dtype=torch.float64)
        projections, area_rows = {}, []
        for rank, camera in enumerate(train):
            budget()
            package = gaussian_renderer.render(camera, teacher, pipe, black)
            result['native_forward_calls'] += 1
            area, projection, stats = native_area(package['render'], n, rank in PROJECTION_RANKS)
            area_sum.add_(area)
            del area, package
            if projection is not None:
                projections[rank] = projection
                save_npz(out / 'projections' / f'{rank:03d}.npz', **projection)
            area_rows.append(dict(rank=rank, view=camera.image_name, **stats))
            result['completed_train_views'] = rank + 1
            budget()
        weights = np.stack((np.ones(n, dtype=np.float64), (area_sum / 219).cpu().numpy()))
        del area_sum
        require(bool(np.isfinite(weights).all()) and bool((weights >= 0).all()), 'Invalid projected-area weights')
        np.save(out / 'weights.npy', weights)
        write_csv(out / 'AREA.csv', area_rows)
        keep_gpu = torch.from_numpy(keep).cuda()
        student = GaussianModel(3, 'default')
        student.active_sh_degree = 3
        for field in FIELDS:
            setattr(student, field, torch.nn.Parameter(getattr(teacher, field).index_select(0, keep_gpu).detach().clone(), requires_grad=False))
        base_opacity = student._opacity
        student_before = model_hash(student)
        geometry_before = geometry_hash(student)
        variants = {'base': logits[keep].astype(np.float32)}
        mapping = dict(keep_original_ids=keep, deleted_original_ids=deleted)
        explanations, opacity_stats = {}, {}
        result['phase'] = 'fixed-nearest-neighbor-transfers'
        for arm, weight in zip(('A', 'B'), weights):
            receivers, donors, assigned, distances, ties = receiver_map(xyz, keep, deleted, weight, budget)
            position = np.searchsorted(keep, receivers)
            assigned_position = np.searchsorted(keep, assigned)
            donation = np.bincount(assigned_position, weights=weight[donors] * tau[donors], minlength=len(keep))
            donor_mass = float(np.sum(weight[donors] * tau[donors]))
            survivor_mass = float(np.sum(weight[receivers] * tau[receivers]))
            require(survivor_mass > 0 and np.isfinite(donor_mass), 'Invalid proxy totals')
            changed_tau = {}
            changed_tau['local'] = tau[keep].copy()
            changed_tau['local'][position] += donation[position] / weight[receivers]
            changed_tau['global'] = tau[keep].copy()
            changed_tau['global'][position] *= 1 + donor_mass / survivor_mass
            shuffled = np.random.default_rng(20261023).permutation(donation[position])
            changed_tau['shuffle'] = tau[keep].copy()
            changed_tau['shuffle'][position] += shuffled / weight[receivers]
            receiver_for_deleted = np.full(len(deleted), -1, dtype=np.int64)
            receiver_for_deleted[np.searchsorted(deleted, donors)] = assigned
            mapping[arm + '_receiver_original_ids'] = receiver_for_deleted
            mapping[arm + '_donation_per_survivor'] = donation
            overlap = {}
            for rank in PROJECTION_RANKS:
                projection = projections[rank]
                both = (projection['radii'][donors] > 0) & (projection['radii'][assigned] > 0)
                distance2d = np.linalg.norm(projection['means2D'][donors].astype(np.float64) - projection['means2D'][assigned], axis=1)
                intersects = both & (distance2d <= projection['radii'][donors].astype(np.float64) + projection['radii'][assigned])
                overlap[str(rank)] = dict(view=train[rank].image_name, donating_pairs=len(donors), both_visible=int(both.sum()),
                    overlapping_pairs=int(intersects.sum()), fraction_all_pairs=float(intersects.mean()),
                    fraction_both_visible=float(intersects.sum() / both.sum()) if both.any() else None)
            conservation = {}
            expected_mass = float(np.sum(weight * tau))
            for suffix, transferred in changed_tau.items():
                require(bool(np.isfinite(transferred).all()) and bool((transferred > 0).all()), 'Nonfinite transferred tau')
                new_logit = transferred + np.log(-np.expm1(-transferred))
                encoded = new_logit.astype(np.float32)
                require(bool(np.isfinite(encoded).all()), 'Unrepresentable transferred float32 logit; no clipping allowed')
                require(np.array_equal(encoded[weight[keep] == 0], variants['base'][weight[keep] == 0]), 'Zero-weight survivor changed')
                variants[arm + '_' + suffix] = encoded
                mass = float(np.sum(weight[keep] * transferred))
                encoded_mass = float(np.sum(weight[keep] * np.logaddexp(0., encoded.astype(np.float64))))
                conservation[suffix] = dict(expected_proxy_mass=expected_mass, transferred_proxy_mass=mass,
                    relative_error=abs(mass - expected_mass) / expected_mass,
                    rendered_float32_logit_proxy_mass=encoded_mass,
                    rendered_float32_relative_error=abs(encoded_mass - expected_mass) / expected_mass)
            explanations[arm] = dict(zero_weight_points=int((weight == 0).sum()),
                zero_weight_survivors=int((weight[keep] == 0).sum()), zero_weight_deleted=int((weight[deleted] == 0).sum()),
                receivers=len(receivers), donating_points=len(donors), exact_tie_queries=ties,
                center_distance=quantiles(distances), distance_over_sum_max_3d_scales=quantiles(distances / (max_scale[donors] + max_scale[assigned])),
                projected_native_three_sigma_circle_overlap=overlap, proxy_conservation=conservation)
        save_npz(out / 'receiver_mapping.npz', **mapping)
        np.save(out / 'opacity_variants.npy', np.stack([variants[name] for name in STATES]))
        dump(out / 'EXPLANATIONS.json', explanations)
        del mapping, projections, xyz, tau, max_scale, weights

        history_rows = list(csv.DictReader(inputs['history'].open(encoding='utf-8')))
        historical = {state: [row for row in history_rows if row['carrier'] == 'all219' and row['variant'] == state]
                      for state in ('base',) + HISTORICAL}
        require(all(len(values) == 32 and sorted(row['view'] for row in values) == [camera.image_name for camera in test]
                    for values in historical.values()), 'Historical controls have different views')
        old_base = {row['view']: row for row in historical['base']}
        original_forward = gaussian_renderer.GaussianRasterizer.forward
        captured = {}

        def capture_forward(self, *values, **keywords):
            outputs = original_forward(self, *values, **keywords)
            captured['rgb'] = outputs[0]
            result['native_forward_calls'] += 1
            return outputs

        def raw(camera, model, background):
            with torch.no_grad():
                display = gaussian_renderer.render(camera, model, pipe, background)['render']
            rgb = captured.pop('rgb').detach()
            require(torch.equal(display, rgb.clamp(0, 1)) and bool(torch.isfinite(rgb).all()), 'Raw/display RGB capture mismatch')
            budget()
            return rgb

        max_t_delta, roundoff_pixels = 0., 0

        def color_alpha(camera, model):
            nonlocal max_t_delta, roundoff_pixels
            rgb, rgb_white = raw(camera, model, black), raw(camera, model, white)
            transmission = rgb_white.double() - rgb.double()
            delta = float((transmission - transmission[:1]).abs().max())
            max_t_delta = max(max_t_delta, delta)
            require(delta <= 2e-5, 'Black/white transmission channel disagreement')
            alpha = 1 - transmission.mean(dim=0, keepdim=True)
            require(float(alpha.min()) >= -2e-5 and float(alpha.max()) <= 1 + 2e-5, 'Raw alpha outside roundoff allowance')
            roundoff_pixels += int(((alpha < 0) | (alpha > 1)).sum())
            # Preserve the measured float64 alpha, including logged roundoff.
            return rgb, alpha

        gaussian_renderer.GaussianRasterizer.forward = capture_forward
        teacher_cache, panel_cache = {}, {}
        result['phase'] = 'teacher-raw-and-GT-save'
        for camera in test:
            rgb, alpha = color_alpha(camera, teacher)
            rgb_cpu = rgb.cpu().numpy().copy()
            teacher_cache[camera.image_name] = rgb_cpu
            save_npz(out / 'raw' / 'teacher' / (camera.image_name + '.npz'), rgb=rgb_cpu, alpha=alpha.cpu().numpy())
            gt_cpu = camera.original_image.detach().cpu().numpy().copy()
            require(tuple(gt_cpu.shape) == (3, 546, 979) and bool(np.isfinite(gt_cpu).all()), 'Invalid test GT')
            save_npz(out / 'raw' / 'GT' / (camera.image_name + '.npz'), rgb=gt_cpu)
            if camera.image_name in PANEL_VIEWS:
                panel_cache[('teacher', camera.image_name)] = rgb_cpu
            del rgb, alpha
        state_integrity = []
        maximum_replay_difference = 0.
        for state in STATES:
            result['phase'] = 'evaluate-' + state
            student._opacity = base_opacity if state == 'base' else torch.nn.Parameter(
                torch.from_numpy(variants[state][:, None]).cuda(), requires_grad=False)
            before_geometry = geometry_hash(student)
            require(before_geometry == geometry_before, 'Geometry/SH changed before state')
            rendered_opacity = torch.sigmoid(student._opacity.detach())
            opacity_stats[state] = dict(fraction_opacity_ge_0_99=float((rendered_opacity >= .99).float().mean()),
                fraction_float32_sigmoid_eq_1=float((rendered_opacity == 1).float().mean()),
                minimum_logit=float(student._opacity.min()), maximum_logit=float(student._opacity.max()))
            del rendered_opacity
            for camera in test:
                rgb, alpha = color_alpha(camera, student)
                gt = camera.original_image.cuda()
                teacher_rgb = torch.from_numpy(teacher_cache[camera.image_name]).cuda().clamp(0, 1)
                displayed = rgb.clamp(0, 1)
                gt_mse = float((displayed - gt).square().mean())
                teacher_mse = float((displayed - teacher_rgb).square().mean())
                require(np.isfinite([gt_mse, teacher_mse]).all(), 'Nonfinite image metric')
                if state == 'base':
                    difference = max(abs(teacher_mse - float(old_base[camera.image_name]['teacher_student_mse'])),
                                     abs(gt_mse - float(old_base[camera.image_name]['gt_mse'])))
                    maximum_replay_difference = max(maximum_replay_difference, difference)
                    require(difference <= 1e-9, 'Historical base per-view MSE replay failed')
                rows.append(dict(state=state, view=camera.image_name, gt_mse=gt_mse,
                    teacher_student_mse=teacher_mse, gt_psnr=float(-10 * torch.log10((displayed - gt).square().mean().clamp_min(1e-30)))))
                rgb_cpu = rgb.cpu().numpy().copy()
                save_npz(out / 'raw' / state / (camera.image_name + '.npz'), rgb=rgb_cpu, alpha=alpha.cpu().numpy())
                if state in ('base', 'B_local') and camera.image_name in PANEL_VIEWS:
                    panel_cache[(state, camera.image_name)] = rgb_cpu
                result['completed_eval_rows'] = len(rows)
                del rgb, alpha, gt, teacher_rgb, displayed
            after_geometry = geometry_hash(student)
            require(after_geometry == geometry_before, 'Geometry/SH changed after state')
            state_integrity.append(dict(state=state, geometry_before=before_geometry, geometry_after=after_geometry))
            write_csv(out / 'METRICS.csv', rows)
        student._opacity = base_opacity
        require(model_hash(student) == student_before and student._opacity is base_opacity, 'Original student not restored')
        gaussian_renderer.GaussianRasterizer.forward = original_forward
        original_forward = None
        require(len(rows) == 224 and result['native_forward_calls'] == 731, 'Incomplete state or forward count')

        from PIL import Image, ImageDraw
        for view in PANEL_VIEWS:
            panel = Image.new('RGB', (979 * 3, 546 + 28), 'white')
            drawing = ImageDraw.Draw(panel)
            for column, state in enumerate(('teacher', 'base', 'B_local')):
                pixels = np.rint(np.clip(panel_cache[(state, view)], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
                panel.paste(Image.fromarray(pixels), (column * 979, 28))
                drawing.text((column * 979 + 8, 7), state, fill='black')
            (out / 'panels').mkdir(exist_ok=True)
            panel.save(out / 'panels' / (view + '_teacher_base_B_local.png'))
            budget()
        summary = {}
        base_rows = {row['view']: row for row in rows if row['state'] == 'base'}
        for state in STATES:
            selected = [row for row in rows if row['state'] == state]
            tail = sorted(selected, key=lambda row: (-row['gt_mse'], row['view']))[:4]
            summary[state] = dict(mean_gt_mse=float(np.mean([row['gt_mse'] for row in selected])),
                tail4_gt_mse=float(np.mean([row['gt_mse'] for row in tail])),
                mean_teacher_mse=float(np.mean([row['teacher_student_mse'] for row in selected])),
                mean_gt_psnr=float(np.mean([row['gt_psnr'] for row in selected])),
                nonworse_GT_views_vs_base=sum(row['gt_mse'] <= base_rows[row['view']]['gt_mse'] for row in selected),
                worst4_views=[row['view'] for row in tail], source='current-fixed-state')
        for state in HISTORICAL:
            selected = historical[state]
            summary['historical_' + state] = dict(mean_gt_mse=float(np.mean([float(row['gt_mse']) for row in selected])),
                tail4_gt_mse=float(np.mean(sorted([float(row['gt_mse']) for row in selected], reverse=True)[:4])),
                mean_teacher_mse=float(np.mean([float(row['teacher_student_mse']) for row in selected])),
                mean_gt_psnr=float(np.mean([float(row['gt_psnr']) for row in selected])),
                nonworse_GT_views_vs_base=sum(float(row['gt_mse']) <= base_rows[row['view']]['gt_mse'] for row in selected),
                worst4_views=[row['view'] for row in sorted(selected, key=lambda row: (-float(row['gt_mse']), row['view']))[:4]],
                source='historical-pruned-alpha-oracle-v0')
        main_state, base = summary['B_local'], summary['base']
        base_gates = dict(teacher_MSE_at_least_20pct_lower=main_state['mean_teacher_mse'] <= .8 * base['mean_teacher_mse'],
            GT_MSE_at_least_5pct_lower=main_state['mean_gt_mse'] <= .95 * base['mean_gt_mse'],
            own_tail4_GT_MSE_nonworse=main_state['tail4_gt_mse'] <= base['tail4_gt_mse'],
            at_least_24_nonworse_GT_views=main_state['nonworse_GT_views_vs_base'] >= 24)
        control_gates = {state: dict(GT_mean_at_least_5pct_lower=main_state['mean_gt_mse'] <= .95 * values['mean_gt_mse'],
                                    own_tail4_GT_MSE_nonworse=main_state['tail4_gt_mse'] <= values['tail4_gt_mse'])
                         for state, values in summary.items() if state not in ('base', 'B_local')}
        passed = all(base_gates.values()) and all(all(gates.values()) for gates in control_gates.values())
        budget()
        after = {name: sha(path) for name, path in inputs.items()}
        source_after = {str(path.resolve()): sha(path) for path in source_paths}
        require(before == after and source_before == source_after, 'Recorded input/source hashes changed')
        require(model_hash(teacher) == teacher_before and teacher.optimizer is None and student.optimizer is None, 'Teacher changed or optimizer exists')
        write_csv(out / 'SUMMARY.csv', [dict(state=state, **{key: value for key, value in values.items() if key != 'worst4_views'},
                                               worst4_views=';'.join(values['worst4_views'])) for state, values in summary.items()])
        dump(out / 'RAW_MANIFEST.json', raw_manifest)
        dump(out / 'OPACITY_STATS.json', opacity_stats)
        result.update(tool_valid=True, phase='complete',
            decision='NEEDS_CROSS_SCENE_AND_STRONG_RECOVERY_BASELINES' if passed else 'STOP_CURRENT_NEAREST_NEIGHBOR_PROXY_TRANSFER',
            base_gates=base_gates, control_gates=control_gates, main_mechanism_passed=passed,
            summary=summary, rows=len(rows), original_student_parameters_before=student_before,
            original_student_parameters_after=model_hash(student), original_opacity_object_restored=student._opacity is base_opacity,
            teacher_parameters_after=model_hash(teacher), state_geometry_integrity=state_integrity,
            input_hashes_before=before, input_hashes_after=after, source_hashes_before=source_before, source_hashes_after=source_after,
            historical_base_maximum_MSE_replay_difference=maximum_replay_difference, historical_controls_source_sha256=before['history'],
            max_background_T_channel_delta=max_t_delta, alpha_roundoff_pixels=roundoff_pixels,
            raw_NPZ_files=len(raw_manifest), raw_NPZ_bytes=sum(item['bytes'] for item in raw_manifest),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            ABI_scope='Validated source/binary hashes, saved tensor shapes, aligned geometry prefixes and initialized radii/conic entries; no claim of source-to-binary build provenance.',
            scope=config['scope'])
    except Exception as exc:
        result.update(decision='TOOL_FAILED', tool_valid=False, tool_failure=f'{type(exc).__name__}: {exc}')
    finally:
        if student is not None and base_opacity is not None:
            student._opacity = base_opacity
            result['original_opacity_object_restored'] = student._opacity is base_opacity
        if gaussian_renderer is not None and original_forward is not None:
            gaussian_renderer.GaussianRasterizer.forward = original_forward
        write_csv(out / 'METRICS.csv', rows)
        if not (out / 'RAW_MANIFEST.json').exists():
            dump(out / 'RAW_MANIFEST.json', raw_manifest)
        result['elapsed_seconds'] = time.perf_counter() - started
        if result['elapsed_seconds'] >= 600:
            result.update(decision='TOOL_FAILED', tool_valid=False, tool_failure='600 second whole-process budget exceeded')
        dump(out / 'RESULTS.json', result)
        watchdog.cancel()
    print(json.dumps({key: result.get(key) for key in ('decision', 'tool_valid', 'tool_failure', 'elapsed_seconds', 'rows')}), flush=True)
    return 0 if result['tool_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
