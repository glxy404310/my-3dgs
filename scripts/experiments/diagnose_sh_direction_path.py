"""Measure the frozen xyz -> SH direction gradient; never update parameters."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
ITERATIONS = (7000, 30000)
RESOLUTION = 2
TIME_LIMIT = 300
MEMORY_LIMIT = 10_000_000_000


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def file_hash(path):
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


def norm(x):
    return torch.linalg.vector_norm(x)


def dot(x, y):
    return (x.double() * y.double()).sum()


def cosine(x, y):
    denominator = float(norm(x)) * float(norm(y))
    return float(dot(x, y)) / denominator if denominator else 0.0


def ratio_stats(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), median=float(np.median(x)), p90=float(np.quantile(x, .9)),
                maximum=float(x.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    assert args.model.resolve() == (args.repo / 'output/truck_30000_r1').resolve()
    assert args.source.resolve() == Path('/home/administrator/datasets/tandt_db/tandt/truck').resolve()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'STATUS.json', dict(status='STARTED', training_performed=False, parameter_updates=0))
    started = None
    try:
        sys.path.insert(0, str(args.repo))
        from scene import Scene, GaussianModel
        from gaussian_renderer import render
        from utils.sh_utils import eval_sh
        from fused_ssim import fused_ssim

        paths = {i: args.model / f'point_cloud/iteration_{i}/point_cloud.ply' for i in ITERATIONS}
        hashes = {i: file_hash(path) for i, path in paths.items()}
        config = dict(repo=str(args.repo), source=str(args.source), model=str(args.model),
                      input_kind='PLY snapshots from the same truck_30000_r1 run',
                      script_sha256=file_hash(Path(__file__)),
                      ply_sha256=hashes, iterations=ITERATIONS, resolution=RESOLUTION,
                      native_python_max_image_difference=2e-5,
                      native_python_relative_gradient_difference=1e-3,
                      tiny_median_ratio=.01, tiny_p90_ratio=.05,
                      time_limit_seconds=TIME_LIMIT, allocator_limit_bytes=MEMORY_LIMIT,
                      training_performed=False, optimizer_created=False, parameter_updates=0)
        dump(out / 'CONFIG.json', config)
        torch.manual_seed(20261022)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def budget():
            torch.cuda.synchronize()
            assert time.perf_counter() - started < TIME_LIMIT, 'GPU stage exceeded 300 seconds'
            assert torch.cuda.max_memory_allocated() < MEMORY_LIMIT, 'Allocated peak exceeded 10 GB'
            assert torch.cuda.max_memory_reserved() < MEMORY_LIMIT, 'Reserved peak exceeded 10 GB'

        dataset = SimpleNamespace(source_path=str(args.source), model_path=str(args.model),
            images='images', depths='', resolution=RESOLUTION, white_background=False,
            train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False, antialiasing=False)
        model = GaussianModel(3, 'default')
        scene = Scene(dataset, model, load_iteration=ITERATIONS[0], shuffle=False)
        train = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
        test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
        assert len(train) == 219 and len(test) == 32
        config['full_training_camera_count'] = len(train)
        config['full_heldout_camera_count'] = len(test)
        train = [train[int(i)] for i in np.linspace(0, len(train)-1, 8, dtype=int)]
        test = [test[int(i)] for i in np.linspace(0, len(test)-1, 8, dtype=int)]
        assert not set(c.image_name for c in train) & set(c.image_name for c in test)
        config['training_views'] = [c.image_name for c in train]
        config['heldout_views'] = [c.image_name for c in test]
        dump(out / 'CONFIG.json', config)
        bg = torch.zeros(3, device='cuda')
        budget()

        def image_and_gradient(camera, loss_kind, route):
            colors = None
            if route != 'native':
                direction = model.get_xyz - camera.camera_center[None]
                direction = direction / direction.norm(dim=1, keepdim=True)
                if route == 'detached':
                    direction = direction.detach()
                features = model.get_features.transpose(1, 2).reshape(-1, 3, 16)
                colors = (eval_sh(model.active_sh_degree, features, direction) + .5).clamp_min(0)
            image = render(camera, model, pipe, bg, override_color=colors,
                           separate_sh=False, use_trained_exp=False)['render']
            if camera.alpha_mask is not None:
                image = image * camera.alpha_mask.cuda()
            gt = camera.original_image.cuda()
            if loss_kind == 'mse':
                loss = (image - gt).square().mean()
            else:
                loss = .8 * (image - gt).abs().mean() + .2 * (1 - fused_ssim(image[None], gt[None]))
            gradient = torch.autograd.grad(loss, model._xyz)[0].detach()
            image, loss_value = image.detach(), float(loss.detach())
            assert bool(torch.isfinite(image).all()) and bool(torch.isfinite(gradient).all())
            assert np.isfinite(loss_value)
            budget()
            return image, gradient, loss_value

        writer = None
        records, states = [], []
        with (out / 'METRICS.csv').open('w', newline='') as metrics_file:
            def measure(camera, loss_kind, iteration, group):
                native_image, native_gradient, native_loss = image_and_gradient(camera, loss_kind, 'native')
                full_image, full, full_loss = image_and_gradient(camera, loss_kind, 'full')
                image_difference = float((native_image - full_image).abs().max())
                gradient_difference = float(norm(native_gradient - full) / norm(native_gradient).clamp_min(1e-12))
                detached_image, geometry, detached_loss = image_and_gradient(camera, loss_kind, 'detached')
                detach_image_difference = float((full_image - detached_image).abs().max())
                sh = full - geometry
                sh_norm, full_norm, geometry_norm = float(norm(sh)), float(norm(full)), float(norm(geometry))
                local_sh = torch.linalg.vector_norm(sh, dim=1)
                local_geometry = torch.linalg.vector_norm(geometry, dim=1)
                local_full = torch.linalg.vector_norm(full, dim=1)
                paired = (local_sh > 0) & (local_geometry > 0)
                products = (sh.double() * geometry.double()).sum(1)
                negative = products < 0
                local_ratios = local_sh[local_full > 0] / local_full[local_full > 0]
                quantiles = torch.quantile(local_ratios, torch.tensor([.5, .9, .99], device='cuda')) if len(local_ratios) else torch.zeros(3, device='cuda')
                parity_pass = image_difference <= 2e-5 and gradient_difference <= 1e-3 and detach_image_difference <= 2e-5
                row = dict(iteration=iteration, group=group, view=camera.image_name, loss=loss_kind,
                    native_loss=native_loss, python_full_loss=full_loss, detached_loss=detached_loss,
                    native_python_image_maxdiff=image_difference,
                    native_python_gradient_relative_difference=gradient_difference,
                    full_detached_image_maxdiff=detach_image_difference, parity_pass=parity_pass,
                    full_norm=full_norm, geometry_norm=geometry_norm, sh_norm=sh_norm,
                    sh_full_ratio=sh_norm / max(full_norm, 1e-30),
                    sh_geometry_cosine=cosine(sh, geometry), sh_geometry_dot=float(dot(sh, geometry)),
                    active_both_count=int(paired.sum()), active_full_count=int((local_full > 0).sum()),
                    negative_dot_fraction=float((negative & paired).sum() / paired.sum().clamp_min(1)),
                    negative_dot_absolute_energy_fraction=float((-products.clamp_max(0)).sum() / products.abs().sum().clamp_min(1e-30)),
                    local_sh_full_ratio_median=float(quantiles[0]), local_sh_full_ratio_p90=float(quantiles[1]),
                    local_sh_full_ratio_p99=float(quantiles[2]),
                    heldout_mean_geometry_mse_dot_full='', heldout_mean_geometry_mse_dot_geometry='',
                    heldout_mean_geometry_mse_dot_sh='', heldout_mean_geometry_mse_cos_full='',
                    heldout_mean_geometry_mse_cos_geometry='', heldout_mean_geometry_mse_cos_sh='')
                assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
                return row, full, geometry, sh

            def record(row):
                nonlocal writer
                if writer is None:
                    writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                metrics_file.flush()
                records.append(row)
                print(f'MEASURED {row["iteration"]} {row["group"]} {row["view"]} {row["loss"]} ratio={row["sh_full_ratio"]:.6g} parity={row["parity_pass"]}', flush=True)
                assert row['parity_pass'], 'Native/Python parity gate failed; do not interpret this diagnostic'
                budget()

            for iteration in ITERATIONS:
                if iteration != ITERATIONS[0]:
                    model.load_ply(str(paths[iteration]), False)
                for field in FIELDS:
                    getattr(model, field).requires_grad_(field == '_xyz')
                assert model.optimizer is None and model.active_sh_degree == 3
                before = model_hash(model)
                reference = torch.zeros_like(model._xyz)
                for camera in test:
                    for loss_kind in ('mse', 'training'):
                        row, full, geometry, sh = measure(camera, loss_kind, iteration, 'heldout')
                        record(row)
                        if loss_kind == 'mse':
                            reference += geometry / len(test)
                        del full, geometry, sh
                np.save(out / f'iteration_{iteration}_heldout_mean_geometry_mse_gradient.npy', reference.cpu().numpy())
                for camera in train:
                    for loss_kind in ('mse', 'training'):
                        row, full, geometry, sh = measure(camera, loss_kind, iteration, 'training')
                        for name, gradient in (('full', full), ('geometry', geometry), ('sh', sh)):
                            row[f'heldout_mean_geometry_mse_dot_{name}'] = float(dot(reference, gradient))
                            row[f'heldout_mean_geometry_mse_cos_{name}'] = cosine(reference, gradient)
                        record(row)
                        del full, geometry, sh, gradient
                after = model_hash(model)
                assert before == after, 'Model tensors changed during frozen measurement'
                assert hashes[iteration] == file_hash(paths[iteration]), 'Input PLY changed'
                states.append(dict(iteration=iteration, points=len(model._xyz),
                    parameters_sha256_before=before, parameters_sha256_after=after,
                    source_sha256_after=file_hash(paths[iteration]),
                    heldout_mean_geometry_mse_gradient_norm=float(norm(reference))))
                del reference
                torch.cuda.empty_cache()
                budget()
        summaries = {f'{i}_{loss_kind}': ratio_stats([r['sh_full_ratio'] for r in records
            if r['iteration'] == i and r['group'] == 'training' and r['loss'] == loss_kind])
            for i in ITERATIONS for loss_kind in ('mse', 'training')}
        assert len(records) == 64 and all(s['n'] == 8 for s in summaries.values())
        tiny = all(s['median'] < .01 and s['p90'] < .05 for s in summaries.values())
        assert all(hashes[i] == file_hash(paths[i]) for i in ITERATIONS)
        result = dict(training_performed=False, optimizer_created=False, parameter_updates=0,
            resolution=RESOLUTION, states=states, training_ratio_summaries=summaries,
            decision='STOP_TINY_PATH' if tiny else 'NONZERO_REQUIRES_CAUSAL_TEST',
            decision_is_method_success=False, geometry_improvement_demonstrated=False,
            all_parity_gates_passed=all(r['parity_pass'] for r in records), rows=len(records),
            elapsed_gpu_stage_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            limits='Unpreconditioned gradients, one scene, two PLY snapshots, half resolution; no causal intervention or training.')
        budget()
        dump(out / 'RESULTS.json', result)
        dump(out / 'STATUS.json', dict(status='COMPLETED', decision=result['decision'],
            training_performed=False, parameter_updates=0))
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        status = dict(status='FAILED', error_type=type(error).__name__, error=str(error),
                      training_performed=False, parameter_updates=0)
        if started is not None:
            status['elapsed_gpu_stage_seconds'] = time.perf_counter()-started
        dump(out / 'STATUS.json', status)
        raise


if __name__ == '__main__':
    main()
