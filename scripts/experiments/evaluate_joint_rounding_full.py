"""Decode all six frozen full-model codecs and evaluate official test RGB only.

No training, optimizer, code selection, or parameter fitting occurs here. The
teacher is loaded from its original iteration-30000 PLY. PNGs are visualization
copies; all metrics and repeat checks use the original floating-point renders.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
COMMON_FIELDS = (['x', 'y', 'z'] + [f'rot_{i}' for i in range(4)]
                 + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)])
CONFIGS = {'8x8': 8, '10x8': 10}
ARMS = ('A_independent', 'B_opacity_only', 'C_joint27')
BUDGET_SECONDS = 300
MEMORY_BYTES = 10 * 1024**3
HISTORICAL_PSNR_TOLERANCE = 2e-4
HISTORICAL_SSIM_TOLERANCE = 2e-5


class ToolInvalid(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ToolInvalid(message)


def dump(path, value):
    temporary = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for field in FIELDS:
        h.update(getattr(model, field).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def summarize(rows, key, higher_is_better=False):
    ranked = sorted(rows, key=lambda r: ((r[key] if higher_is_better else -r[key]), r['view']))
    values = np.asarray([r[key] for r in rows], dtype=np.float64)
    count = math.ceil(len(rows) * .1)
    return dict(n=len(rows), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                worst10_count=count, worst10_mean=float(np.mean([r[key] for r in ranked[:count]])),
                worst10_direction='lowest' if higher_is_better else 'highest',
                worst10_views=[r['view'] for r in ranked[:count]])


def decode_arm(path, point_count, scale_bits, low, high, budget=lambda: None):
    """Read little-endian packed point records, including records crossing bytes."""
    bits_per_point = 3 * scale_bits + 8
    total_bits = point_count * bits_per_point
    require(path.stat().st_size == (total_bits + 7) // 8, f'Wrong payload byte count: {path}')
    packed = np.memmap(path, dtype=np.uint8, mode='r')
    if total_bits % 8:
        require(int(packed[-1]) >> (total_bits % 8) == 0, f'Nonzero padding bits: {path}')
    scales = np.empty((point_count, 3), dtype=np.float32)
    logits = np.empty((point_count, 1), dtype=np.float32)
    endpoints = {'zero': 0, 'one': 0}
    scale_weights = (1 << np.arange(scale_bits)).astype(np.uint16)
    alpha_weights = (1 << np.arange(8)).astype(np.uint16)
    # A multiple of eight points makes every chunk begin on a byte boundary.
    for start in range(0, point_count, 32768):
        budget()
        stop = min(point_count, start + 32768)
        first, last = start * bits_per_point, stop * bits_per_point
        bits = np.unpackbits(packed[first // 8:(last + 7) // 8], bitorder='little')
        bits = bits[:last-first].reshape(stop-start, bits_per_point)
        codes = np.stack([bits[:, i*scale_bits:(i+1)*scale_bits] @ scale_weights
                          for i in range(3)], axis=1)
        alpha_codes = bits[:, 3*scale_bits:] @ alpha_weights
        scales[start:stop] = low + codes.astype(np.float64) * ((high-low) / (2**scale_bits-1))
        alpha = alpha_codes.astype(np.float64) / 255.
        # Legal endpoint codes remain exact zero/one after sigmoid(+/-inf).
        with np.errstate(divide='ignore'):
            logits[start:stop, 0] = np.log(alpha) - np.log1p(-alpha)
        endpoints['zero'] += int(np.count_nonzero(alpha_codes == 0))
        endpoints['one'] += int(np.count_nonzero(alpha_codes == 255))
    require(np.isfinite(scales).all() and not np.isnan(logits).any(), 'Invalid decoded scale/opacity')
    return scales, logits, endpoints


def environment(args, model):
    if args.repo is not None:
        repo = args.repo.resolve()
    else:
        candidates = [*model.parents, Path(__file__).resolve().parents[2] / 'gaussian-splatting']
        repo = next((p for p in candidates if (p/'scene/gaussian_model.py').is_file()
                     and (p/'gaussian_renderer').is_dir()), None)
        require(repo is not None, 'Cannot locate official renderer; supply --repo')
    source = args.source
    if source is None:
        cfg = model/'cfg_args'
        require(cfg.is_file(), 'Missing cfg_args; supply --source')
        node = ast.parse(cfg.read_text(encoding='utf-8').strip(), mode='eval').body
        require(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'Namespace', 'Unsupported cfg_args format; supply --source')
        keywords = {k.arg: k.value for k in node.keywords}
        require('source_path' in keywords, 'cfg_args has no source_path; supply --source')
        source = Path(ast.literal_eval(keywords['source_path']))
    require(Path(source).is_dir(), f'Dataset source does not exist: {source}')
    return repo, Path(source).resolve()


def evaluate(args, out, started, result):
    import torch
    from PIL import Image

    def budget():
        require(time.perf_counter()-started < BUDGET_SECONDS, '300 second full-stage budget exceeded')
        if torch.cuda.is_initialized():
            require(torch.cuda.max_memory_allocated() <= MEMORY_BYTES, '10 GiB torch allocated budget exceeded')

    budget()
    model = args.model.resolve()
    if model.is_file():
        ply = model
        require(ply.parent.name == 'iteration_30000' and ply.parent.parent.name == 'point_cloud',
                '--model PLY must be the original point_cloud/iteration_30000/point_cloud.ply')
        model = ply.parents[2]
    else:
        ply = model/'point_cloud/iteration_30000/point_cloud.ply'
    repo, source = environment(args, model)
    sys.path.insert(0, str(repo))
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim

    io_seconds = 0.

    def save_image(image, path):
        nonlocal io_seconds
        t = time.perf_counter()
        array = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')
        Image.fromarray(array).save(path, compress_level=1)
        io_seconds += time.perf_counter()-t
        budget()

    previous_config = json.loads((args.previous/'CONFIG.json').read_text(encoding='utf-8'))
    with (args.previous/'METRICS.csv').open(newline='', encoding='utf-8') as f:
        previous_rows = {r['view']: r for r in csv.DictReader(f) if r['group'] == 'real_test'}
    cpu_results = json.loads(args.cpu_results.read_text(encoding='utf-8'))
    require(cpu_results.get('decision') == 'PROCEED_TO_FULL_FROZEN_RGB'
            and cpu_results.get('numerical_valid') is True
            and cpu_results.get('both_proxy_and_utility_configs_pass') is True
            and cpu_results.get('cap_skip_check_pass') is True, 'CPU prerequisite gate did not pass')
    header_path, common_path = args.codec/'HEADER.json', args.codec/'COMMON.bin'
    header = json.loads(header_path.read_text(encoding='utf-8'))
    n = header['point_count']
    require(isinstance(n, int) and n > 0, 'Invalid point count')
    require(header['common_fields'] == COMMON_FIELDS, 'COMMON field order differs from frozen contract')
    require(header['alpha_decode'] == 'j/255 including endpoints', 'Unexpected opacity decode rule')
    require(set(header['configs']) == set(CONFIGS), 'Both fixed bit configurations are required')
    low, high = np.asarray(header['scale_min'], np.float64), np.asarray(header['scale_max'], np.float64)
    require(low.shape == high.shape == (3,) and np.isfinite([low, high]).all()
            and np.all(high > low), 'Invalid full-teacher scale grid')
    require(common_path.stat().st_size == n*55*4, 'COMMON byte count is not N*55*4')
    inputs = dict(teacher=ply, header=header_path, common=common_path, contract=args.contract,
                  cpu_results=args.cpu_results, previous_config=args.previous/'CONFIG.json',
                  previous_metrics=args.previous/'METRICS.csv', script=Path(__file__))
    byte_report = {}
    for config, bits in CONFIGS.items():
        require(header['configs'][config]['scale_bits'] == bits
                and header['configs'][config]['opacity_bits'] == 8, f'Wrong grid: {config}')
        byte_report[config] = {}
        for arm in ARMS:
            name = f'{config}_{arm}'
            payload = args.codec/f'{name}.bin'
            inputs[name] = payload
            expected = (n*(3*bits+8)+7)//8
            actual = payload.stat().st_size
            require(actual == expected, f'Wrong packed byte count for {name}: {actual} != {expected}')
            full_bytes = common_path.stat().st_size + header_path.stat().st_size + actual
            byte_report[config][arm] = dict(payload_bytes=actual, expected_payload_bytes=expected,
                common_bytes=common_path.stat().st_size, header_bytes=header_path.stat().st_size,
                full_model_codec_bytes=full_bytes, full_model_bits_per_point=8*full_bytes/n,
                teacher_ply_bytes=ply.stat().st_size,
                teacher_ply_to_codec_container_ratio=ply.stat().st_size/full_bytes,
                scope='COMMON + HEADER + one arm; uncompressed float32 SH/position/rotation; no entropy coding')
        require(len({v['full_model_codec_bytes'] for v in byte_report[config].values()}) == 1,
                f'Arms have unequal complete byte counts: {config}')
    hash_start = time.perf_counter()
    hashes_before = {name: sha(path) for name, path in inputs.items()}
    io_seconds += time.perf_counter()-hash_start
    require(hashes_before['teacher'] == previous_config['teacher_sha256']
            == cpu_results['teacher_sha256_before'] == cpu_results['teacher_sha256_after'], 'Teacher source hash mismatch')
    if 'teacher_sha256' in header:
        require(header['teacher_sha256'] == hashes_before['teacher'], 'HEADER teacher hash mismatch')
    result['source_hashes_before'] = hashes_before
    result['actual_bytes'] = byte_report
    result['point_count'] = n
    result['contract_sha256'] = hashes_before['contract']
    require(torch.cuda.is_available(), 'Official CUDA renderer unavailable')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(min(1., MEMORY_BYTES/torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(20261023)
    config_record = dict(repo=str(repo), source=str(source), teacher=str(ply), codec=str(args.codec),
        previous=str(args.previous), cpu_results=str(args.cpu_results), contract=str(args.contract),
        source_hashes=hashes_before, configs={k: dict(scale_bits=v, opacity_bits=8) for k, v in CONFIGS.items()},
        arms=list(ARMS), training_performed=False, optimizer_created=False, parameter_updates=0,
        resolution=1, image_width=979, image_height=546, sh_degree=3, black_background=True,
        antialiasing=False, train_test_exp=False, convert_SHs_python=False, compute_cov3D_python=False,
        camera_domain='all 32 official test cameras sorted by image_name; no other evaluation cameras',
        primary_metric='student_gt_mse: arithmetic mean of per-image means over RGB pixels',
        gate=dict(minimum_C_relative_mean_gt_mse_reduction=.05, comparisons=['A_independent', 'B_opacity_only'],
                  worst10_rule='each arm sorts its own GT MSE; C worst ceil(32*.1)=4 mean <= comparator',
                  both_configs_required=True),
        teacher_repeat_mse_required=0., historical_teacher_psnr_tolerance=HISTORICAL_PSNR_TOLERANCE,
        historical_teacher_ssim_tolerance=HISTORICAL_SSIM_TOLERANCE,
        full_stage_wall_seconds_limit=BUDGET_SECONDS, peak_torch_allocated_bytes_limit=MEMORY_BYTES,
        metrics_use_native_float_renders=True,
        metric_render_rule='native renderer clamp preserved; no PNG quantization is used for metrics',
        pngs_clamp_only_for_visualization=True,
        opacity_endpoints='0 -> -inf logit; 255 -> +inf logit; no clamp',
        full_codec_accounting='COMMON.bin + HEADER.json + that arm bitstream',
        novelty_claim=False)
    dump(out/'CONFIG.json', config_record)
    budget()
    dataset = SimpleNamespace(source_path=str(source), model_path=str(model), images='images', depths='',
        resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    teacher = GaussianModel(3, 'default')
    student = None
    teacher_before = None
    gpu_render_seconds = 0.
    decode_seconds = 0.
    try:
        scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False)
        for field in FIELDS:
            getattr(teacher, field).requires_grad_(False)
        require(teacher.optimizer is None, 'Teacher unexpectedly has an optimizer')
        require(len(teacher._xyz) == n and teacher.active_sh_degree == 3, 'Teacher point count/SH degree mismatch')
        teacher_before = model_hash(teacher)
        result['teacher_parameters_sha256_before'] = teacher_before
        test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
        names = [c.image_name for c in test]
        require(len(test) == 32 and len(set(names)) == 32 and names == sorted(previous_rows),
                'Official test camera names do not match the original 32-camera run')
        require(all(Path(name).name == name for name in names), 'Unsafe image name')
        require(all((c.image_width, c.image_height) == (979, 546) for c in test), 'Render resolution mismatch')
        require(all(hasattr(c, 'original_image') and tuple(c.original_image.shape) == (3, 546, 979)
                    for c in test), 'All 32 official test GT photographs are required')
        train_names = {c.image_name for c in scene.getTrainCameras()}
        require(not train_names.intersection(names), 'Official train/test cameras overlap')
        require(set(previous_config['calibration']).issubset(train_names), 'Prior selection cameras missing')
        config_record['test_views'] = names
        dump(out/'CONFIG.json', config_record)
        common = np.memmap(common_path, dtype='<f4', mode='r', shape=(n, 55))
        student = GaussianModel(3, 'default')
        student.active_sh_degree = 3
        common_arrays = (('_xyz', common[:, :3]), ('_rotation', common[:, 3:7]),
                         ('_features_dc', common[:, 7:10].reshape(n, 1, 3)),
                         ('_features_rest', common[:, 10:55].reshape(n, 3, 15).transpose(0, 2, 1)))
        for field, array in common_arrays:
            budget()
            data = np.array(array, dtype=np.float32, order='C', copy=True)
            require(np.isfinite(data).all(), f'Nonfinite COMMON values: {field}')
            require(np.array_equal(data, getattr(teacher, field).detach().cpu().numpy()),
                    f'COMMON is not bit-exact with original teacher: {field}')
            setattr(student, field, torch.nn.Parameter(torch.from_numpy(data).cuda(), requires_grad=False))
        del common_arrays, data, common
        teacher_scales = teacher._scaling.detach().cpu().numpy()
        require(np.array_equal(teacher_scales.min(axis=0), low)
                and np.array_equal(teacher_scales.max(axis=0), high), 'Grid endpoints are not full-teacher min/max')
        del teacher_scales
        result['common_float32_parameters_exact'] = True
        bg = torch.zeros(3, device='cuda')
        teacher_cache, teacher_rows = {}, []
        (out/'images'/'teacher').mkdir(parents=True)
        (out/'images'/'gt').mkdir()

        def render_checked(cam, model_to_render):
            nonlocal gpu_render_seconds
            budget()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                image = render(cam, model_to_render, pipe, bg)['render']
            torch.cuda.synchronize()
            gpu_render_seconds += time.perf_counter()-t
            require(tuple(image.shape) == (3, 546, 979) and bool(torch.isfinite(image).all()),
                    f'Invalid RGB output: {cam.image_name}')
            budget()
            return image

        def gt_metrics(image, gt):
            mse = float((image-gt).square().mean())
            psnr = float(-10*torch.log10((image-gt).square().mean().clamp_min(1e-30)))
            similarity = float(ssim(image, gt))
            require(np.isfinite([mse, psnr, similarity]).all(), 'Nonfinite RGB metric')
            return mse, psnr, similarity

        max_psnr_delta, max_ssim_delta = 0., 0.
        for cam in test:
            a = render_checked(cam, teacher)
            gt = cam.original_image.cuda()
            mse, psnr, similarity = gt_metrics(a, gt)
            teacher_rows.append(dict(view=cam.image_name, teacher_gt_mse=mse,
                                     teacher_gt_psnr=psnr, teacher_gt_ssim=similarity))
            teacher_cache[cam.image_name] = a.detach().cpu()
            old = previous_rows[cam.image_name]
            max_psnr_delta = max(max_psnr_delta, abs(psnr-float(old['teacher_psnr_gt'])))
            max_ssim_delta = max(max_ssim_delta, abs(similarity-float(old['teacher_ssim_gt'])))
            save_image(a, out/'images'/'teacher'/f'{cam.image_name}.png')
            save_image(gt, out/'images'/'gt'/f'{cam.image_name}.png')
            del a, gt
        result['historical_teacher_parity'] = dict(max_abs_psnr_delta=max_psnr_delta,
            max_abs_ssim_delta=max_ssim_delta, passed=max_psnr_delta <= HISTORICAL_PSNR_TOLERANCE
            and max_ssim_delta <= HISTORICAL_SSIM_TOLERANCE)
        require(result['historical_teacher_parity']['passed'], 'Historical teacher GT metric parity failed')
        repeat = render_checked(test[0], teacher)
        repeat_mse = float((repeat-teacher_cache[names[0]].cuda()).square().mean())
        result['teacher_repeat_mse_before_arms'] = repeat_mse
        require(repeat_mse == 0., 'Teacher repeat float RGB MSE is not strictly zero')
        del repeat
        teacher_by_view = {r['view']: r for r in teacher_rows}
        result['teacher_gt'] = {key: summarize(teacher_rows, key, key != 'teacher_gt_mse')
                                for key in ('teacher_gt_mse', 'teacher_gt_psnr', 'teacher_gt_ssim')}
        dump(out/'TEACHER_METRICS.json', teacher_rows)
        rows = []
        fieldnames = ['config', 'arm', 'group', 'view', 'teacher_student_mse', 'student_gt_mse',
                      'student_gt_psnr', 'student_gt_ssim', 'teacher_gt_mse', 'teacher_gt_psnr',
                      'teacher_gt_ssim', 'delta_psnr_gt', 'delta_ssim_gt']
        with (out/'METRICS.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for config, bits in CONFIGS.items():
                for arm in ARMS:
                    budget()
                    key = f'{config}_{arm}'
                    payload = inputs[key]
                    t = time.perf_counter()
                    scales, logits, endpoints = decode_arm(payload, n, bits, low, high, budget)
                    decode_seconds += time.perf_counter()-t
                    student._scaling = torch.nn.Parameter(torch.from_numpy(scales).cuda(), requires_grad=False)
                    student._opacity = torch.nn.Parameter(torch.from_numpy(logits).cuda(), requires_grad=False)
                    del scales, logits
                    require(student.optimizer is None and all(not getattr(student, k).requires_grad for k in FIELDS),
                            f'Student is not frozen: {key}')
                    activated = student.get_opacity
                    require(bool(torch.isfinite(activated).all()) and int((activated == 0).sum()) == endpoints['zero']
                            and int((activated == 1).sum()) == endpoints['one'], f'Opacity endpoint decode failed: {key}')
                    del activated
                    before = model_hash(student)
                    directory = out/'images'/key
                    directory.mkdir()
                    arm_rows = []
                    arm_started = time.perf_counter()
                    with torch.no_grad():
                        for cam in test:
                            image = render_checked(cam, student)
                            a, gt = teacher_cache[cam.image_name].cuda(), cam.original_image.cuda()
                            mse_ts = float((a-image).square().mean())
                            mse, psnr, similarity = gt_metrics(image, gt)
                            require(math.isfinite(mse_ts), f'Nonfinite teacher/student MSE: {key}')
                            baseline = teacher_by_view[cam.image_name]
                            row = dict(config=config, arm=arm, group='real_test',
                                teacher_student_mse=mse_ts, student_gt_mse=mse, student_gt_psnr=psnr,
                                student_gt_ssim=similarity, **baseline,
                                delta_psnr_gt=psnr-baseline['teacher_gt_psnr'],
                                delta_ssim_gt=similarity-baseline['teacher_gt_ssim'])
                            writer.writerow(row)
                            f.flush()
                            rows.append(row)
                            arm_rows.append(row)
                            save_image(image, directory/f'{cam.image_name}.png')
                            del image, a, gt
                    after = model_hash(student)
                    require(before == after, f'Student parameters changed during evaluation: {key}')
                    metrics = {metric: summarize(arm_rows, metric, metric.endswith(('psnr', 'ssim')))
                        for metric in ('teacher_student_mse', 'student_gt_mse', 'student_gt_psnr',
                                       'student_gt_ssim', 'teacher_gt_mse', 'teacher_gt_psnr', 'teacher_gt_ssim')}
                    result['arms'][key] = dict(status='COMPLETE', config=config, arm=arm, metrics=metrics,
                        decoded_opacity_endpoint_counts=endpoints, parameters_sha256_before=before,
                        parameters_sha256_after=after, parameters_unchanged=True, optimizer_created=False,
                        parameter_updates=0, elapsed_arm_seconds=time.perf_counter()-arm_started,
                        saved_student_png_count=len(arm_rows),
                        selected_views={label: [r['view'] for r in sorted(arm_rows, key=lambda r: (-r[metric], r['view']))[:3]]
                            for label, metric in (('largest_gt_error', 'student_gt_mse'),
                                                  ('largest_teacher_difference', 'teacher_student_mse'))})
                    dump(out/'PROGRESS.json', result)
                    print(f'EVALUATED {key} 32 official test views', flush=True)
        repeat = render_checked(test[0], teacher)
        result['teacher_repeat_mse_after_arms'] = float((repeat-teacher_cache[names[0]].cuda()).square().mean())
        del repeat
        require(result['teacher_repeat_mse_after_arms'] == 0., 'Final teacher repeat float MSE is not strictly zero')
        gates = {}
        for config in CONFIGS:
            c = result['arms'][f'{config}_C_joint27']['metrics']['student_gt_mse']
            comparisons = {}
            for arm in ARMS[:2]:
                other = result['arms'][f'{config}_{arm}']['metrics']['student_gt_mse']
                reduction = (other['mean']-c['mean'])/other['mean'] if other['mean'] > 0 else None
                mean_pass = other['mean'] > 0 and c['mean'] <= .95*other['mean']
                tail_pass = c['worst10_mean'] <= other['worst10_mean']
                comparisons[arm] = dict(relative_mean_gt_mse_reduction=reduction,
                    mean_gt_mse_reduction_at_least_5pct=bool(mean_pass),
                    worst10_gt_mse_nonworse=bool(tail_pass), passed=bool(mean_pass and tail_pass))
            gates[config] = dict(comparisons=comparisons, passed=all(v['passed'] for v in comparisons.values()))
        result['decision_gate'] = dict(configs=gates, both_configs_pass=all(v['passed'] for v in gates.values()),
                                      primary_metric='student_gt_mse', no_configuration_selection=True)
        result['decision'] = ('PROCEED_TO_STRONGER_CONTROLS_AND_SECOND_SCENE'
                              if result['decision_gate']['both_configs_pass'] else 'STOP_CURRENT_IMPLEMENTATION')
        result['selected_teacher_gt_views'] = [r['view'] for r in sorted(teacher_rows,
            key=lambda r: (-r['teacher_gt_mse'], r['view']))[:3]]
        result['completed_views_per_arm'] = 32
        result['all_six_arms_reported'] = len(rows) == 192
        budget()
    finally:
        result['timing'] = dict(cpu_decode_seconds=decode_seconds, disk_io_and_png_seconds=io_seconds,
            gpu_render_seconds=gpu_render_seconds, elapsed_full_stage_seconds=time.perf_counter()-started,
            note='CPU encoding is a separate prior stage; decode includes packed payload reads; wall stage includes all work')
        result['peak_torch_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        if teacher_before is not None:
            result['teacher_parameters_sha256_after'] = model_hash(teacher)
            result['teacher_parameters_unchanged'] = teacher_before == result['teacher_parameters_sha256_after']
        result['optimizer_created'] = teacher.optimizer is not None or (student is not None and student.optimizer is not None)
        t = time.perf_counter()
        result['source_hashes_after'] = {name: sha(path) for name, path in inputs.items()}
        result['timing']['disk_io_and_png_seconds'] += time.perf_counter()-t
        result['inputs_unchanged'] = hashes_before == result['source_hashes_after']
        result['timing']['elapsed_full_stage_seconds'] = time.perf_counter()-started
    require(result.get('teacher_parameters_unchanged') is True and result['inputs_unchanged'],
            'Teacher parameters or immutable input files changed')
    require(not result['optimizer_created'], 'An optimizer unexpectedly exists')
    budget()
    result['tool_valid'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'codec', 'previous', 'output', 'cpu-results', 'contract'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--repo', type=Path, help='Official gaussian-splatting checkout; otherwise infer from model ancestors')
    parser.add_argument('--source', type=Path, help='Original dataset; otherwise read source_path from model/cfg_args')
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(decision='INCONCLUSIVE', tool_valid=False, training_performed=False,
        optimizer_created=False, parameter_updates=0,
        arms={f'{config}_{arm}': dict(status='PENDING') for config in CONFIGS for arm in ARMS},
        scope='Full frozen official-test RGB evaluation. No novelty, mature codec, speed, or cross-scene claim.')

    def timeout():
        failure = dict(decision='INCONCLUSIVE', tool_valid=False, training_performed=False,
            optimizer_created=False, parameter_updates=0, tool_failure='300 second hard wall budget exceeded',
            elapsed_full_stage_seconds=time.perf_counter()-started,
            partial_artifacts='METRICS.csv, images/, CONFIG.json and PROGRESS.json are preserved; no automatic rerun')
        dump(args.output/'RESULTS.json', failure)
        print('INCONCLUSIVE: 300 second hard wall budget exceeded', file=sys.stderr, flush=True)
        os._exit(2)

    watchdog = threading.Timer(BUDGET_SECONDS, timeout)
    watchdog.daemon = True
    watchdog.start()
    exit_code = 0
    try:
        evaluate(args, args.output, started, result)
    except Exception as exc:
        result['decision'] = 'INCONCLUSIVE'
        result['tool_valid'] = False
        result['tool_failure'] = f'{type(exc).__name__}: {exc}'
        exit_code = 2
    finally:
        result['elapsed_full_stage_seconds'] = time.perf_counter()-started
        dump(args.output/'RESULTS.json', result)
        watchdog.cancel()
    print(json.dumps(dict(decision=result['decision'], tool_valid=result['tool_valid'],
                          output=str(args.output)), allow_nan=False), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
