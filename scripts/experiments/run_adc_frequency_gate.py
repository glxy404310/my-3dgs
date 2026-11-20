"""Locked ADC100/200 cost gate. Uses existing operators; no algorithm search."""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', required=True)
    known, _ = pre.parse_known_args()
    sys.path.insert(0, known.repo)
    import numpy as np
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from utils.general_utils import build_rotation
    from fused_ssim import fused_ssim
    from lpipsPyTorch.modules.lpips import LPIPS
    from known_adc_ops import clone_with_checks, las_with_checks
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        separate_sh = True
    except ImportError:
        separate_sh = False

    p = argparse.ArgumentParser(parents=[pre])
    mp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--package', required=True)
    p.add_argument('--create-package', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--interval', type=int, choices=[100, 200], required=True)
    p.add_argument('--seed', type=int, choices=[0, 1], required=True)
    p.add_argument('--sanity', action='store_true')
    a = p.parse_args()
    assert a.eval and a.resolution == 1 and a.optimizer_type == 'default'
    assert not (a.depths or a.random_background or a.train_test_exp or a.white_background)
    assert a.iterations == 30000 and a.position_lr_max_steps == 30000
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    process_start = time.perf_counter()
    a.model_path = str(out)
    a.densify_until_iter = 6500
    a.densification_interval = a.interval
    dataset, opt, pipe = mp.extract(a), op.extract(a), pp.extract(a)
    end = 3622 if a.sanity else 7000
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, g)
    train = scene.getTrainCameras()
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    assert len(test) in (32, 38)
    assert not ({c.image_name for c in train} & {c.image_name for c in test})
    background = torch.zeros(3, device='cuda')
    criterion = LPIPS('vgg', '0.1').cuda().eval()
    for param in criterion.parameters():
        param.requires_grad_(False)

    def rng():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

    def set_rng(r):
        random.setstate(r[0])
        np.random.set_state(r[1])
        torch.set_rng_state(r[2])
        torch.cuda.set_rng_state_all(r[3])

    def copy_tree(x):
        if torch.is_tensor(x):
            return x.detach().clone().requires_grad_(x.requires_grad)
        if isinstance(x, dict):
            return {k: copy_tree(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return type(x)(copy_tree(v) for v in x)
        return x

    def equal(x, y):
        if torch.is_tensor(x):
            assert x.dtype == y.dtype and x.device == y.device and torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                equal(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert type(x) == type(y) and len(x) == len(y)
            for v, w in zip(x, y):
                equal(v, w)
        else:
            assert x == y

    package_path = Path(a.package)
    contract = dict(checkpoint=str(Path(a.checkpoint).resolve()), seed=a.seed,
                    train_order=[c.image_name for c in train], test_order=[c.image_name for c in test],
                    resolution=1, schedule_horizon=30000, cutoff=6500,
                    source_sha256=sha256(a.checkpoint))
    if a.create_package:
        assert not package_path.exists()
        model, iteration = torch.load(a.checkpoint, weights_only=False)
        assert iteration == 3500
        g.restore(model, opt)
        del model
        camera_rng = random.Random(a.seed)
        indices, stack = [], []
        for _ in range(3500):
            if not stack:
                stack = list(range(len(train)))
            indices.append(stack.pop(camera_rng.randrange(len(stack))))
        package_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(contract=contract, model=g.capture(), exposure=g._exposure.detach(),
                        exposure_optimizer=g.exposure_optimizer.state_dict(),
                        exposure_mapping=g.exposure_mapping, pretrained_exposures=g.pretrained_exposures,
                        rng=rng(), camera_indices=indices, initial_camera_stack=list(range(len(train)))), package_path)
    package = torch.load(package_path, weights_only=False)
    assert package['contract'] == contract
    assert package['initial_camera_stack'] == list(range(len(train)))
    g.restore(package['model'], opt)
    g._exposure.data.copy_(package['exposure'])
    g.exposure_optimizer.load_state_dict(package['exposure_optimizer'])
    g.exposure_mapping = package['exposure_mapping']
    g.pretrained_exposures = package['pretrained_exposures']
    equal(g.capture(), package['model'])
    equal(g._exposure, package['exposure'])
    equal(g.exposure_optimizer.state_dict(), package['exposure_optimizer'])
    set_rng(package['rng'])
    camera_indices = package['camera_indices'][:end - 3500]
    assert len(camera_indices) == end - 3500
    write_json(out / 'resume_checks.json', dict(pass_checks=True, exact_model_and_adam=True,
        exact_exposure=True, rng_restored=True, camera_sequence_from_shared_package=True,
        initial_stack_verified=True, checkpoint_iteration=3500, contract=contract))
    camera_names = [train[i].image_name for i in camera_indices]
    write_json(out / 'cameras.json', camera_names)
    package_hash = sha256(package_path)
    del package

    original_clone = GaussianModel.densify_and_clone
    original_adc = GaussianModel.densify_and_prune
    original_prune = GaussianModel.prune_points
    counts = {}

    @torch.no_grad()
    def fast_clone(model, grads, threshold, extent):
        ids = torch.where((torch.norm(grads, dim=-1) >= threshold) &
                          (model.get_scaling.max(dim=1).values <= model.percent_dense * extent))[0]
        n = len(model.get_xyz)
        target = (1 - torch.sqrt(1 - model.get_opacity[ids])).clamp(1e-7, 1 - 1e-7)
        original_clone(model, grads, threshold, extent)
        model._opacity[ids] = torch.logit(target)
        model._opacity[n:] = torch.logit(target)
        counts['clone'] = len(ids)

    @torch.no_grad()
    def fast_split(model, grads, threshold, extent, N=2):
        assert N == 2
        n = len(model.get_xyz)
        padded = torch.zeros(n, device='cuda')
        padded[:len(grads)] = grads.squeeze()
        scales = model.get_scaling
        ids = torch.where((padded >= threshold) & (scales.max(dim=1).values > model.percent_dense * extent))[0]
        k = len(ids)
        parent_scale = scales[ids]
        values, axes = parent_scale.max(dim=1, keepdim=True)
        offsets = parent_scale * torch.zeros_like(parent_scale).scatter(1, axes, 1.) * (3 * .45)
        offsets = torch.cat([offsets, -offsets])
        rotation = model._rotation[ids]
        xyz = torch.bmm(build_rotation(rotation).repeat(2, 1, 1), offsets.unsqueeze(-1)).squeeze(-1)
        xyz = xyz + model.get_xyz[ids].repeat(2, 1)
        minor = math.sqrt(1 - .45 ** 2)
        child_scale = parent_scale.scatter(1, axes, values * (1 - .45) / minor).repeat(2, 1) * minor
        raw_alpha = torch.logit((model.get_opacity[ids] * .6).clamp(1e-7, 1 - 1e-7).repeat(2, 1))
        model.densification_postfix(xyz, model._features_dc[ids].repeat(2, 1, 1),
            model._features_rest[ids].repeat(2, 1, 1), raw_alpha, torch.log(child_scale),
            rotation.repeat(2, 1), model.tmp_radii[ids].repeat(2))
        remove = torch.zeros(n + 2 * k, device='cuda', dtype=torch.bool)
        remove[ids] = True
        original_prune(model, remove)
        counts['split'] = k

    g.densify_and_clone = MethodType(fast_clone, g)
    g.densify_and_split = MethodType(fast_split, g)
    metrics, events, timing = [], [], []
    operator_verified = False

    @torch.no_grad()
    def verify_adc(radii):
        nonlocal operator_verified
        # Sanity only: compare the light operator to the existing checked implementation.
        before = copy_tree(g.capture())
        exposure_opt = copy_tree(g.exposure_optimizer.state_dict())
        original_adc(g, opt.densify_grad_threshold, .005, scene.cameras_extent, 20, radii)
        expected = copy_tree(g.capture())
        g.restore(before, opt)
        g.exposure_optimizer.load_state_dict(exposure_opt)
        def checked_clone(model, *args):
            return clone_with_checks(model, *args, native_clone=original_clone, corrected=True)[0]
        def checked_split(model, *args):
            return las_with_checks(model, *args, build_rotation=build_rotation)
        g.densify_and_clone = MethodType(checked_clone, g)
        g.densify_and_split = MethodType(checked_split, g)
        original_adc(g, opt.densify_grad_threshold, .005, scene.cameras_extent, 20, radii)
        equal(expected, g.capture())
        g.densify_and_clone = MethodType(fast_clone, g)
        g.densify_and_split = MethodType(fast_split, g)
        operator_verified = True
        write_json(out / 'operator_equivalence.json', dict(pass_checks=True,
            full_capture_and_optimizer_exact=True, reference='known_adc_ops.py', event=3600))

    @torch.no_grad()
    def evaluate(iteration):
        state = rng()
        before_n = len(g.get_xyz)
        selected = test[:2] if a.sanity else test
        for camera in selected:
            prediction = render(camera, g, pipe, background, separate_sh=separate_sh,
                                use_trained_exp=False)['render']
            if camera.alpha_mask is not None:
                prediction = prediction * camera.alpha_mask.cuda()
            gt = camera.original_image.cuda()
            mse = (prediction - gt).square().mean().clamp_min(1e-30)
            row = dict(iteration=iteration, view=camera.image_name,
                       psnr=float(-10 * torch.log10(mse)), l1=float((prediction - gt).abs().mean()),
                       ssim=float(ssim(prediction, gt)), lpips=float(criterion(prediction[None], gt[None]).mean()))
            assert all(math.isfinite(row[k]) for k in ('psnr', 'l1', 'ssim', 'lpips'))
            metrics.append(row)
        assert len(g.get_xyz) == before_n
        set_rng(state)
        with (out / 'metrics.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0]))
            w.writeheader()
            w.writerows(metrics)
        print(f'EVALUATED {iteration} views={len(selected)}', flush=True)

    repo_commit = subprocess.check_output(['git', '-C', a.repo, 'rev-parse', 'HEAD'], text=True).strip()
    assert repo_commit == '54c035f7834b564019656c3e3fcc3646292f727d'
    config = dict(args=vars(a), optimization=vars(opt), pipeline=vars(pipe), contract=contract,
                  repo_commit=repo_commit, script_sha256=sha256(__file__),
                  known_ops_sha256=sha256(Path(__file__).with_name('known_adc_ops.py')),
                  plan_sha256=sha256(Path(__file__).resolve().parents[2] / 'refine-logs/EXPERIMENT_PLAN.md'),
                  package_sha256=package_hash, torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(), torch_threads=torch.get_num_threads(),
                  test_views=len(test), separate_sh=separate_sh,
                  schedule_horizon=30000, optimizer_steps=end - 3500)
    write_json(out / 'config.json', config)
    evaluate(3500)
    # Warm the actual training forward/backward kernels without taking an update.
    warm_rng = rng()
    camera = train[camera_indices[0]]
    pkg = render(camera, g, pipe, background, separate_sh=separate_sh, use_trained_exp=False)
    pred, gt = pkg['render'], camera.original_image.cuda()
    if camera.alpha_mask is not None:
        pred = pred * camera.alpha_mask.cuda()
    ((1 - opt.lambda_dssim) * l1_loss(pred, gt) + opt.lambda_dssim *
     (1 - fused_ssim(pred[None], gt[None]))).backward()
    g.optimizer.zero_grad(set_to_none=True)
    g.exposure_optimizer.zero_grad(set_to_none=True)
    del pkg, pred, gt
    set_rng(warm_rng)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    block_start = 3501
    training_seconds = 0.
    peak_allocated, peak_reserved = 0, 0
    for iteration, camera_index in zip(range(3501, end + 1), camera_indices):
        g.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            g.oneupSHdegree()
        camera = train[camera_index]
        pkg = render(camera, g, pipe, background, separate_sh=separate_sh, use_trained_exp=False)
        image, viewspace, visible, radii = (pkg[k] for k in ('render', 'viewspace_points', 'visibility_filter', 'radii'))
        if camera.alpha_mask is not None:
            image *= camera.alpha_mask.cuda()
        gt = camera.original_image.cuda()
        loss = ((1 - opt.lambda_dssim) * l1_loss(image, gt) + opt.lambda_dssim *
                (1 - fused_ssim(image[None], gt[None])))
        loss.backward()
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                g.max_radii2D[visible] = torch.max(g.max_radii2D[visible], radii[visible])
                g.add_densification_stats(viewspace, visible)
                if iteration > opt.densify_from_iter and iteration % a.interval == 0:
                    n_before = len(g.get_xyz)
                    counts.clear()
                    if a.sanity and iteration == 3600:
                        verify_adc(radii)
                    else:
                        original_adc(g, opt.densify_grad_threshold, .005, scene.cameras_extent, 20, radii)
                    n_after = len(g.get_xyz)
                    events.append(dict(iteration=iteration, n_before=n_before, n_after=n_after,
                                       cloned=counts['clone'], split=counts['split'],
                                       pruned=n_before + counts['clone'] + counts['split'] - n_after))
                if iteration % opt.opacity_reset_interval == 0:
                    g.reset_opacity()
            g.exposure_optimizer.step()
            g.exposure_optimizer.zero_grad(set_to_none=True)
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
        if iteration % 100 == 0 or iteration in (6750, end):
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            training_seconds += seconds
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            timing.append(dict(first=block_start, last=iteration, seconds=seconds,
                               cumulative_seconds=training_seconds, n_gaussians=len(g.get_xyz),
                               allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved()))
            print(f'PROGRESS {iteration} cumulative_train_seconds={training_seconds:.3f} n={len(g.get_xyz)}', flush=True)
            if iteration in (6500, 6750, 7000) or iteration == end:
                evaluate(iteration)
            block_start = iteration + 1
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
    expected_events = list(range(3600, min(end, 6499) + 1, a.interval))
    assert [e['iteration'] for e in events] == expected_events
    assert all(e['pruned'] >= 0 for e in events)
    assert not a.sanity or operator_verified
    write_json(out / 'events.json', events)
    write_json(out / 'timing.json', timing)
    torch.save(dict(model=g.capture(), iteration=end, exposure=g._exposure.detach(),
                    exposure_optimizer=g.exposure_optimizer.state_dict(),
                    exposure_mapping=g.exposure_mapping, pretrained_exposures=g.pretrained_exposures,
                    rng=rng(), camera_position=end - 3500), out / f'checkpoint_{end}.pth')
    write_json(out / 'summary.json', dict(complete=True, interval=a.interval, seed=a.seed,
        sanity=a.sanity, source_iteration=3500, final_iteration=end, updates=end - 3500,
        actual_events=expected_events, n_gaussians=len(g.get_xyz), training_seconds=training_seconds,
        process_seconds=time.perf_counter() - process_start,
        peak_training_allocated_bytes=peak_allocated, peak_training_reserved_bytes=peak_reserved,
        measurement='Synchronized block wall time including forward/backward/ADC/optimizer; excludes evaluation, loading, audit, saving. Sanity timing includes equivalence audit and is not a cost result.',
        scope='Fixed restored checkpoint and precomputed camera sequence; continuation seeds are not independent training seeds.'))
    print('ADC_FREQUENCY_GATE_RUN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
