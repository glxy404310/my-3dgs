"""One isolated split, then paired GT-only recovery; reuse official 3DGS code."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def save_image(path, tensor):
    arr = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', required=True)
    known, _ = pre.parse_known_args()
    sys.path.insert(0, known.repo)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    try:
        from fused_ssim import fused_ssim
    except ImportError:
        fused_ssim = None

    parser = argparse.ArgumentParser(parents=[pre])
    mp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=['noop', 'original_split', 'opacity_corrected_split', 'las_split'], required=True)
    parser.add_argument('--inherit_moments', action='store_true',
                        help='Known control: inherit .2m and .04v with unchanged scalar Adam step')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--calib_views', type=int, default=8)
    parser.add_argument('--eval_views', type=int, default=0, help='0 means all held-out views')
    parser.add_argument('--top_k', type=int, default=128, help='0 means all eligible parents')
    parser.add_argument('--online_interval', type=int, default=0,
                        help='Use gradients along this many online train steps; hold last optimizer step at event boundary')
    parser.add_argument('--event_file', help='Canonical online event snapshot shared by all branches')
    parser.add_argument('--create_event', action='store_true', help='Save a new event_file after online warmup')
    parser.add_argument('--recovery_steps', type=int, nargs='+', default=[0, 1, 5, 20, 100])
    parser.add_argument('--selection_file', help='Reuse a previous run selection after verifying view IDs')
    args = parser.parse_args()
    if args.inherit_moments and args.variant == 'noop':
        raise ValueError('Moment inheritance requires splitting')
    run_variant=args.variant+('_moments' if args.inherit_moments else '')
    if not args.eval or args.train_test_exp or args.random_background or args.depths:
        raise ValueError('Protocol requires --eval, fixed background, no exposure or depth supervision')
    if args.optimizer_type != 'default':
        raise ValueError('Only the installed default Adam is supported')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    args.model_path = str(out)
    dataset, opt, pipe = mp.extract(args), op.extract(args), pp.extract(args)
    seed_all(args.seed)
    started = time.perf_counter()
    model = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, model, shuffle=False)
    event = None
    if args.event_file and not args.create_event:
        event = torch.load(args.event_file, map_location='cuda', weights_only=False)
        for key,value in dict(seed=args.seed,online_interval=args.online_interval,
                              checkpoint=str(Path(args.checkpoint).resolve()),resolution=args.resolution,
                              source_path=str(Path(args.source_path).resolve())).items():
            if event[key] != value:
                raise ValueError(f'Canonical event mismatch: {key}')
        payload, iteration = event['model'], event['event_iteration']
    else:
        if args.create_event and (not args.event_file or Path(args.event_file).exists()):
            raise ValueError('create_event requires a new event_file')
        payload, iteration = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    model.restore(payload, opt)
    del payload
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    if not train or not test:
        raise ValueError('Both train and held-out cameras are required')
    eval_ids = np.linspace(0, len(test)-1, min(args.eval_views, len(test)), dtype=int) if args.eval_views else range(len(test))
    test = [test[int(i)] for i in eval_ids]
    calib_ids = np.linspace(0, len(train)-1, min(args.calib_views, len(train)), dtype=int)
    calib = [train[int(i)] for i in calib_ids]
    bg = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device='cuda')
    n_before = len(model.get_xyz)
    print(f'LOADED checkpoint={iteration} gaussians={n_before} train={len(train)} test={len(test)}', flush=True)

    def rendering(cam):
        pkg = render(cam, model, pipe, bg, separate_sh=False, use_trained_exp=False)
        if cam.alpha_mask is not None:
            pkg['render'] = pkg['render'] * cam.alpha_mask.cuda()
        return pkg

    def loss_for(image, gt):
        ss = fused_ssim(image[None], gt[None]) if fused_ssim is not None else ssim(image, gt)
        return (1-opt.lambda_dssim)*(image-gt).abs().mean() + opt.lambda_dssim*(1-ss)

    # Frozen mode leaves parameters/state unchanged; online mode follows the documented train trajectory.
    grad_sum = torch.zeros(n_before, device='cuda')
    counts = torch.zeros(n_before, device='cuda')
    online_sequence=[]
    online_rng=random.Random(args.seed+20000)
    online_stack=[]
    if event is not None:
        by_name={c.image_name:c for c in train}
        calibration=[by_name[n] for n in event['calib_names']]
        grad_sum,counts=event['grad_sum'],event['counts']
        online_rng.setstate(event['camera_rng_state'])
        online_stack=event['remaining_camera_stack']
        online_sequence=event['online_camera_sequence']
    elif args.online_interval:
        if args.online_interval < 2:
            raise ValueError('Online interval must be at least 2')
        calibration=[]
        for i in range(args.online_interval):
            if not online_stack:
                online_stack=list(range(len(train)))
            calibration.append(train[online_stack.pop(online_rng.randrange(len(online_stack)))])
    else:
        calibration=calib
    start_iteration=event['start_iteration'] if event is not None else iteration
    for index,cam in enumerate([] if event is not None else calibration):
        if args.online_interval:
            model.update_learning_rate(start_iteration+index+1)
            if (start_iteration+index+1)%1000==0:
                model.oneupSHdegree()
            online_sequence.append(cam.image_name)
        model.optimizer.zero_grad(set_to_none=True)
        pkg = rendering(cam)
        loss = loss_for(pkg['render'], cam.original_image.cuda())
        loss.backward()
        visible = pkg['radii'] > 0
        grad_sum[visible] += pkg['viewspace_points'].grad[visible, :2].norm(dim=-1)
        counts[visible] += 1
        if args.online_interval and index<len(calibration)-1:
            # Last backward is held at the ADC boundary. All branches discard its
            # parameter gradients, matching the installed split's rebind behavior.
            model.optimizer.step()
        del pkg, loss
    if args.online_interval:
        iteration=start_iteration+args.online_interval
    model.optimizer.zero_grad(set_to_none=True)
    score = grad_sum / counts.clamp_min(1)
    size_ok = model.get_scaling.detach().max(dim=1).values > opt.percent_dense * scene.cameras_extent
    eligible = size_ok & (counts > 0) & (score >= opt.densify_grad_threshold)
    eligible_ids = eligible.nonzero().flatten()
    if args.selection_file:
        prior = json.loads(Path(args.selection_file).read_text())
        if prior['checkpoint'] != str(Path(args.checkpoint).resolve()) or prior['n_before'] != n_before or prior['iteration']!=iteration:
            raise ValueError('Selection does not belong to this checkpoint')
        if prior['calib_names'] != [c.image_name for c in calibration]:
            raise ValueError('Calibration cameras differ from selection')
        ids = torch.tensor(prior['parent_ids'], dtype=torch.long, device='cuda')
        if not eligible[ids].all():
            raise ValueError('Shared parents are no longer eligible')
    else:
        k = min(args.top_k, len(eligible_ids)) if args.top_k else len(eligible_ids)
        order = torch.argsort(score[eligible_ids], descending=True, stable=True)
        ids = eligible_ids[order[:k]].sort().values
    if len(ids) == 0:
        raise ValueError(f'No eligible split parents: size_eligible={int(size_ok.sum())}, grad_max={float(score.max())}')
    selection = dict(checkpoint=str(Path(args.checkpoint).resolve()), iteration=iteration,
                     n_before=n_before, eligible_count=len(eligible_ids), size_eligible_count=int(size_ok.sum()),
                     parent_ids=ids.cpu().tolist(), parent_scores=score[ids].cpu().tolist(),
                     scene_extent=float(scene.cameras_extent), grad_threshold=opt.densify_grad_threshold,
                     calib_names=[c.image_name for c in calibration], split_seed=args.seed+10000)
    dump(out/'selection.json', selection)
    if args.create_event:
        torch.save(dict(model=model.capture(),event_iteration=iteration,start_iteration=start_iteration,
            grad_sum=grad_sum,counts=counts,calib_names=[c.image_name for c in calibration],
            camera_rng_state=online_rng.getstate(),remaining_camera_stack=online_stack,
            online_camera_sequence=online_sequence,seed=args.seed,online_interval=args.online_interval,
            checkpoint=str(Path(args.checkpoint).resolve()),resolution=args.resolution,
            source_path=str(Path(args.source_path).resolve())),args.event_file)
    del event
    dump(out/'config.json', dict(args=vars(args), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), code_commit=subprocess.check_output(
        ['git','-C',args.repo,'rev-parse','HEAD'], text=True).strip(),
        test_names=[c.image_name for c in test], checkpoint_iteration=start_iteration,
        intervention_iteration=iteration,
        scope='Isolated split; subsequent ADC disabled. Online mode holds final backward at ADC boundary and omits its optimizer step in every branch; otherwise checkpoint boundary with fixed-view calibration.',
        start_checkpoint_iteration=start_iteration, online_warmup_camera_sequence=online_sequence))
    print(f'SELECTED {len(ids)}/{len(eligible_ids)} eligible parents', flush=True)
    del grad_sum, counts, score, size_ok, eligible, eligible_ids

    before = {}
    rows = []
    def evaluate(stage, step, update_seconds):
        current = []
        with torch.no_grad():
            for i, cam in enumerate(test):
                image = rendering(cam)['render']
                gt = cam.original_image.cuda()
                ref = before.get(cam.image_name)
                row = dict(variant=run_variant, stage=stage, step=step, image_name=cam.image_name,
                           l1=float((image-gt).abs().mean()), psnr=float(psnr(image,gt).mean()),
                           ssim=float(ssim(image,gt)), n_gaussians=len(model.get_xyz),
                           render_change_l1=0. if ref is None else float((image-ref.cuda()).abs().mean()),
                           update_seconds=update_seconds)
                if not np.isfinite([row['l1'], row['psnr'], row['ssim']]).all():
                    raise RuntimeError('Non-finite evaluation metric')
                if stage == 'before':
                    before[cam.image_name] = image.cpu()
                if i < 2 and stage in ('before','recovery'):
                    save_image(out/'images'/f'{i:02d}_{stage}_{step}.png', image)
                    if stage == 'before':
                        save_image(out/'images'/f'{i:02d}_gt.png', gt)
                current.append(row)
        rows.extend(current)
        with (out/'recovery.csv').open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        avg={key: float(np.mean([r[key] for r in current])) for key in ('l1','psnr','ssim','render_change_l1')}
        print(f'EVAL {stage} step={step} {avg}', flush=True)
        return avg

    pre_metrics = evaluate('before', -1, 0.)
    torch.cuda.synchronize()
    split_start = time.perf_counter()
    with torch.no_grad():
        if args.variant != 'noop':
            previous_states={}
            if args.inherit_moments:
                for group in model.optimizer.param_groups:
                    state=model.optimizer.state[group['params'][0]]
                    previous_states[group['name']]={key:state[key] for key in ('exp_avg','exp_avg_sq')}
                    previous_states[group['name']]['step']=state['step'].clone()
            parent_alpha = model.get_opacity[ids].clone()
            grads = torch.zeros(n_before, device='cuda'); grads[ids] = 1.
            model.tmp_radii = torch.zeros(n_before, device='cuda')
            if args.variant == 'las_split':
                # Isolated LAS operator from authors' Improved-GS code, not their full pipeline.
                # Their ellipsoid contour is 3 sigma, hence the 3.0 displacement factor.
                from utils.general_utils import build_rotation
                scales = model.get_scaling[ids]
                max_values, axes = scales.max(dim=1, keepdim=True)
                axis_mask = torch.zeros_like(scales).scatter(1, axes, 1.)
                offsets = scales * axis_mask * (3. * .45)
                offsets = torch.cat([offsets, -offsets])
                rotations = build_rotation(model._rotation[ids]).repeat(2, 1, 1)
                xyz = torch.bmm(rotations, offsets.unsqueeze(-1)).squeeze(-1) + model.get_xyz[ids].repeat(2, 1)
                minor_rate = (1.-.45**2)**.5
                child_scale = scales.scatter(1, axes, max_values*(1.-.45)/minor_rate).repeat(2,1)*minor_rate
                opacity = torch.logit((parent_alpha*.6).clamp(1e-7,1-1e-7)).repeat(2,1)
                model.densification_postfix(xyz, model._features_dc[ids].repeat(2,1,1),
                    model._features_rest[ids].repeat(2,1,1), opacity, torch.log(child_scale),
                    model._rotation[ids].repeat(2,1), model.tmp_radii[ids].repeat(2))
                remove=torch.cat([grads.bool(),torch.zeros(2*len(ids),dtype=torch.bool,device='cuda')])
                model.prune_points(remove)
            else:
                # Official operator verbatim; a fork prevents split RNG affecting recovery cameras.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.seed+10000)
                    model.densify_and_split(grads, .5, scene.cameras_extent, N=2)
            if len(model.get_xyz) != n_before + len(ids):
                raise RuntimeError('Unexpected split cardinality')
            if args.variant == 'opacity_corrected_split':
                corrected=1-torch.sqrt(1-parent_alpha)
                model._opacity[-2*len(ids):].copy_(torch.logit(corrected.clamp(1e-7,1-1e-7)).repeat(2,1))
            if args.inherit_moments:
                keep=torch.ones(n_before,dtype=torch.bool,device='cuda');keep[ids]=False
                checked=[]
                for group in model.optimizer.param_groups:
                    state=model.optimizer.state[group['params'][0]]
                    old=previous_states[group['name']]
                    assert torch.equal(state['step'],old['step']), 'Adam step changed'
                    for key,factor in (('exp_avg',.2),('exp_avg_sq',.04)):
                        assert torch.equal(state[key][:-2*len(ids)],old[key][keep]), 'Survivor state changed'
                        expected=old[key][ids].repeat((2,)+(1,)*(old[key].ndim-1))*factor
                        state[key][-2*len(ids):].copy_(expected)
                        assert torch.equal(state[key][-2*len(ids):],expected), 'Child state mismatch'
                    checked.append(dict(group=group['name'],step=float(state['step'])))
                dump(out/'moment_checks.json',dict(status='pass',first_moment_factor=.2,
                    second_moment_factor=.04,scalar_step_unchanged=True,survivors_unchanged=True,groups=checked))
                del previous_states,old,state,expected,keep
            torch.save(dict(parent_ids=ids.cpu(), child_xyz=model.get_xyz[-2*len(ids):].cpu(),
                            child_scaling=model.get_scaling[-2*len(ids):].cpu()), out/'split_children.pt')
            model.tmp_radii = None
            del grads, parent_alpha
    torch.cuda.synchronize()
    split_seconds = time.perf_counter()-split_start
    immediate = evaluate('recovery', 0, 0.)
    steps=sorted(set(args.recovery_steps) | {0})
    rng=online_rng if args.online_interval else random.Random(args.seed+20000)
    # Camera sampling is identical for every branch and does not depend on selection/evaluation calls.
    stack=online_stack if args.online_interval else []
    sequence=[]
    update_seconds=0.
    for step in range(1,max(steps)+1):
        if not stack:
            stack=list(range(len(train)))
        cam=train[stack.pop(rng.randrange(len(stack)))]; sequence.append(cam.image_name)
        torch.cuda.synchronize(); tick=time.perf_counter()
        model.update_learning_rate(iteration+step)
        if (iteration+step)%1000 == 0:
            model.oneupSHdegree()
        pkg=rendering(cam)
        loss=loss_for(pkg['render'],cam.original_image.cuda())
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        del pkg, loss
        torch.cuda.synchronize(); update_seconds+=time.perf_counter()-tick
        if step in steps:
            evaluate('recovery',step,update_seconds)
    summary=dict(variant=run_variant, selected_parents=len(ids), n_before=n_before,
                 n_after=len(model.get_xyz), before=pre_metrics, immediate=immediate,
                 split_seconds=split_seconds, recovery_update_seconds=update_seconds,
                 total_seconds=time.perf_counter()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), camera_sequence=sequence,
                 complete=True)
    dump(out/'summary.json',summary)
    print('COMPLETE '+json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
