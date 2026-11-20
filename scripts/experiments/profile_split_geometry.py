"""TRAIN-view render-cost diagnostic; no training, tuning, or quality claims."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', required=True)
    known, _ = pre.parse_known_args()
    sys.path.insert(0, known.repo)
    import numpy as np
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from gaussian_renderer import render
    from scene import Scene, GaussianModel
    from utils.general_utils import build_rotation

    parser = argparse.ArgumentParser(parents=[pre])
    mp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--event', required=True)
    parser.add_argument('--reference_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--views', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    assert args.eval and args.optimizer_type == 'default'
    assert not args.train_test_exp and not args.random_background and not args.depths
    assert args.views > 0 and args.rounds >= 2 and args.repeats >= 2
    out, reference = Path(args.output), Path(args.reference_root)
    out.mkdir(parents=True, exist_ok=False)
    args.model_path = str(out)
    dataset, opt, pipe = mp.extract(args), op.extract(args), pp.extract(args)
    torch.manual_seed(0)
    scene_model = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, scene_model, shuffle=False)
    train = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
    test_names = {camera.image_name for camera in scene.getTestCameras()}
    cameras = [train[int(i)] for i in np.linspace(0, len(train)-1, min(args.views, len(train)), dtype=int)]
    assert not ({camera.image_name for camera in cameras} & test_names)
    selection = json.loads((reference / 'noop/selection.json').read_text())
    policies = ('noop', 'original_split', 'las_split')
    models, counts, checks = {}, {}, {}
    background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device='cuda')
    with torch.no_grad():
        for policy in policies:
            event = torch.load(args.event, map_location='cuda', weights_only=False)
            assert event['seed'] == 0 and event['event_iteration'] == 3600
            assert event['source_path'] == str(Path(args.source_path).resolve())
            assert event['resolution'] == args.resolution
            model = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            # restore() calls training_setup(), which expects Scene's exposure fields.
            # Exposure training/rendering is disabled; give each model an independent copy.
            model._exposure = torch.nn.Parameter(scene_model._exposure.detach().clone())
            model.exposure_mapping = dict(scene_model.exposure_mapping)
            model.pretrained_exposures = None
            model.restore(event['model'], opt)
            del event
            n_before = len(model.get_xyz)
            assert n_before == 1058297
            ids = torch.tensor(selection['parent_ids'], device='cuda', dtype=torch.long)
            k = len(ids)
            assert k == 12404 and ids.unique().numel() == k
            if policy != 'noop':
                grads = torch.zeros(n_before, device='cuda')
                grads[ids] = 1.
                model.tmp_radii = torch.zeros(n_before, device='cuda')
                if policy == 'original_split':
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(10000)
                        model.densify_and_split(grads, .5, scene.cameras_extent, N=2)
                else:
                    scales = model.get_scaling[ids]
                    values, axes = scales.max(dim=1, keepdim=True)
                    offsets = scales * torch.zeros_like(scales).scatter(1, axes, 1.) * 1.35
                    offsets = torch.cat([offsets, -offsets])
                    xyz = torch.bmm(build_rotation(model._rotation[ids]).repeat(2, 1, 1), offsets.unsqueeze(-1)).squeeze(-1)
                    xyz = xyz + model.get_xyz[ids].repeat(2, 1)
                    minor = math.sqrt(1-.45**2)
                    scale = scales.scatter(1, axes, values*.55/minor).repeat(2, 1)*minor
                    opacity = torch.logit((model.get_opacity[ids]*.6).clamp(1e-7, 1-1e-7)).repeat(2, 1)
                    model.densification_postfix(xyz, model._features_dc[ids].repeat(2,1,1),
                        model._features_rest[ids].repeat(2,1,1), opacity, torch.log(scale),
                        model._rotation[ids].repeat(2,1), model.tmp_radii[ids].repeat(2))
                    model.prune_points(torch.cat([grads.bool(), torch.zeros(2*k, device='cuda', dtype=torch.bool)]))
                model.tmp_radii = None
                expected = torch.load(reference / policy / 'split_children.pt', map_location='cpu', weights_only=True)
                assert torch.equal(ids.cpu(), expected['parent_ids'])
                assert torch.equal(model.get_xyz[-2*k:].cpu(), expected['child_xyz'])
                assert torch.equal(model.get_scaling[-2*k:].cpu(), expected['child_scaling'])
                checks[policy] = 'exact_saved_child_geometry_match'
            assert len(model.get_xyz) == n_before + (0 if policy == 'noop' else k)
            model.optimizer = None  # Rendering only; release unused Adam state.
            models[policy] = model
            counts[policy] = len(model.get_xyz)
        assert counts['original_split'] == counts['las_split']
        torch.cuda.empty_cache()
        records, quality = [], []
        ordering = random.Random(20261022)
        for camera in cameras:
            gt = camera.original_image.cuda()
            for policy in policies:
                for _ in range(5):
                    render(camera, models[policy], pipe, background, separate_sh=False, use_trained_exp=False)
                prediction = render(camera, models[policy], pipe, background, separate_sh=False, use_trained_exp=False)['render']
                if camera.alpha_mask is not None:
                    prediction = prediction * camera.alpha_mask.cuda()
                error = prediction - gt
                mse = float(error.square().mean())
                quality.append(dict(view=camera.image_name, policy=policy,
                    train_l1=float(error.abs().mean()), train_psnr=-10*math.log10(max(mse, 1e-30))))
            for round_index in range(args.rounds):
                order = list(policies)
                ordering.shuffle(order)
                for position, policy in enumerate(order):
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    tick = time.perf_counter()
                    start.record()
                    for _ in range(args.repeats):
                        render(camera, models[policy], pipe, background, separate_sh=False, use_trained_exp=False)
                    end.record()
                    end.synchronize()
                    wall_ms = (time.perf_counter()-tick)*1000/args.repeats
                    gpu_ms = start.elapsed_time(end)/args.repeats
                    assert math.isfinite(gpu_ms) and gpu_ms > 0 and math.isfinite(wall_ms)
                    records.append(dict(view=camera.image_name, round=round_index, order=position,
                        policy=policy, n_gaussians=counts[policy], gpu_stream_ms=gpu_ms, wall_ms=wall_ms))
            del gt, prediction, error
    for name, rows in (('timing.csv', records), ('train_quality.csv', quality)):
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(complete=True, scope='Train-view pre/post split rendering only. No training, held-out selection, budget policy, or novel method tested.',
        n_gaussians=counts, checks=checks, views=[camera.image_name for camera in cameras],
        heldout_overlap=False, args=vars(args), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        means={policy: {metric: statistics.mean(row[metric] for row in records if row['policy'] == policy)
                       for metric in ('gpu_stream_ms', 'wall_ms')} for policy in policies},
        timing_scope='CUDA stream elapsed and synchronized wall time, averaged over repeated render calls; excludes GT transfer, mask and quality metrics. CUDA intervals may contain CPU launch gaps.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary['means'], indent=2), flush=True)


if __name__ == '__main__':
    main()
