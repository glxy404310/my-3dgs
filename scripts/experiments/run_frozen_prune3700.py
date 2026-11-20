"""Capture one real pre-prune render state, then let normal training continue."""
import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', required=True)
    known, _ = pre.parse_known_args()
    sys.path.insert(0, known.repo)
    import numpy as np
    import torch
    import inspect
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from utils.loss_utils import ssim
    from utils.general_utils import build_rotation
    from known_adc_ops import clone_with_checks, las_with_checks

    parser = argparse.ArgumentParser(parents=[pre])
    mp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--exact_state', action='store_true')
    parser.add_argument('--operator_stages', action='store_true')
    parser.add_argument('--method', choices=('native', 'clone_corrected', 'clone_corrected_las'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume_package', required=True)
    parser.add_argument('--create_resume', action='store_true')
    args = parser.parse_args()
    assert args.method == 'clone_corrected_las' and args.seed == 0 and args.iterations == 4500
    assert not args.create_resume
    assert args.eval and args.optimizer_type == 'default'
    assert not args.train_test_exp and not args.random_background and not args.depths
    assert not args.operator_stages or (args.observe and args.exact_state)
    assert args.observe and args.exact_state and args.operator_stages
    assert not args.create_resume or args.method == 'native'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    args.model_path = str(out)
    (out/'operator_witness').mkdir()
    dataset, optimization, pipeline = mp.extract(args), op.extract(args), pp.extract(args)
    # This is our trusted local training checkpoint, not an external pickle.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    spec = importlib.util.spec_from_file_location('installed_native_train', Path(args.repo)/'train.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original_render = native.render
    model_class = native.GaussianModel
    original_adc = model_class.densify_and_prune
    original_clone = model_class.densify_and_clone
    original_split = model_class.densify_and_split
    original_prune = model_class.prune_points
    original_scene = native.Scene
    original_update_lr = model_class.update_learning_rate
    method_checks = []
    rows, cameras, events, invariant_checks = [], [], [], []
    context = dict(iteration=None, evaluating=False, in_split=False, counts=None, scene=None, render_args=None)
    resume_checks = {}

    def scene_factory(*positional, **keywords):
        scene = original_scene(*positional, **keywords)
        context['scene'] = scene
        return scene

    def compare_exact(first, second):
        if torch.is_tensor(first):
            assert torch.is_tensor(second) and first.dtype == second.dtype and first.shape == second.shape
            assert first.device == second.device and torch.equal(first, second)
        elif isinstance(first, dict):
            assert first.keys() == second.keys()
            for key in first:
                compare_exact(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            assert type(first) == type(second) and len(first) == len(second)
            for x, y in zip(first, second):
                compare_exact(x, y)
        else:
            assert first == second

    @torch.no_grad()
    def update_lr(model, iteration):
        if not resume_checks:
            assert iteration == 3501
            scene = context['scene']
            train_order = [camera.image_name for camera in scene.getTrainCameras()]
            test_order = [camera.image_name for camera in scene.getTestCameras()]
            contract = dict(dataset={key:value for key,value in vars(dataset).items() if key != 'model_path'},
                            optimization=vars(optimization), pipeline=vars(pipeline),
                            source_checkpoint=str(Path(args.checkpoint).resolve()), seed=args.seed)
            package_path = Path(args.resume_package)
            if args.create_resume:
                assert not package_path.exists()
                package_path.parent.mkdir(parents=True, exist_ok=True)
                package = dict(model=model.capture(), iteration=3500, contract=contract,
                    exposure=model._exposure.detach().clone(), exposure_optimizer=model.exposure_optimizer.state_dict(),
                    exposure_mapping=dict(model.exposure_mapping), pretrained_exposures=model.pretrained_exposures,
                    train_order=train_order, test_order=test_order,
                    camera_stack=train_order, viewpoint_indices=list(range(len(train_order))),
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(package, package_path)
                del package
            # Preserve CPU Adam step and RNG tensor devices saved by torch; no map_location override.
            package = torch.load(package_path, weights_only=False)
            assert package['iteration'] == 3500 and package['contract'] == contract
            assert package['train_order'] == train_order and package['test_order'] == test_order
            assert package['camera_stack'] == train_order
            assert package['viewpoint_indices'] == list(range(len(train_order)))
            model.restore(package['model'], optimization)
            model._exposure.copy_(package['exposure'])
            model.exposure_optimizer.load_state_dict(package['exposure_optimizer'])
            model.exposure_mapping = dict(package['exposure_mapping'])
            model.pretrained_exposures = package['pretrained_exposures']
            compare_exact(model.capture(), package['model'])
            compare_exact(model._exposure, package['exposure'])
            compare_exact(model.exposure_optimizer.state_dict(), package['exposure_optimizer'])
            assert all(parameter.grad is None for optimizer in (model.optimizer, model.exposure_optimizer)
                       for group in optimizer.param_groups for parameter in group['params'])
            random.setstate(package['python_rng'])
            np.random.set_state(package['numpy_rng'])
            torch.set_rng_state(package['torch_rng'])
            torch.cuda.set_rng_state_all(package['cuda_rng'])
            assert random.getstate() == package['python_rng']
            restored_np = np.random.get_state()
            assert restored_np[0] == package['numpy_rng'][0] and np.array_equal(restored_np[1], package['numpy_rng'][1])
            assert restored_np[2:] == package['numpy_rng'][2:]
            assert torch.equal(torch.get_rng_state(), package['torch_rng'])
            restored_cuda = torch.cuda.get_rng_state_all()
            assert len(restored_cuda) == len(package['cuda_rng'])
            assert all(torch.equal(x,y) for x,y in zip(restored_cuda, package['cuda_rng']))
            resume_checks.update(status='pass', package_loaded=True, package=str(package_path), seed=args.seed,
                create_requested=args.create_resume, source_iteration=3500,
                all_capture_values_exact=True, exposure_exact=True, all_rng_restored=True,
                scene_order_match=True, initial_stack_and_indices_match=True,
                fresh_stack_scope='Native training already created full stack from verified Scene order before first update_lr.',
                n_gaussians=len(model.get_xyz))
            (out/'resume_checks.json').write_text(json.dumps(resume_checks, indent=2))
            del package
        return original_update_lr(model, iteration)

    def versions(model):
        tensors = []
        for group in model.optimizer.param_groups:
            for parameter in group['params']:
                tensors.append(parameter)
                if parameter.grad is not None:
                    tensors.append(parameter.grad)
                tensors.extend(value for value in model.optimizer.state[parameter].values() if torch.is_tensor(value))
        tensors.extend([model.max_radii2D, model.xyz_gradient_accum, model.denom, model._exposure])
        return tuple((id(t), t.data_ptr(), t._version, tuple(t.shape)) for t in tensors)

    def rng_state():
        return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()

    def same_rng(first, second):
        return (first[0] == second[0] and first[1][0] == second[1][0]
                and np.array_equal(first[1][1], second[1][1]) and first[1][2:] == second[1][2:]
                and torch.equal(first[2], second[2])
                and all(torch.equal(a, b) for a, b in zip(first[3], second[3])))

    def live_state(model):
        tensors, metadata = {}, [model.active_sh_degree, model.spatial_lr_scale, model.percent_dense]
        for optimizer_name, optimizer in (('scene', model.optimizer), ('exposure', model.exposure_optimizer)):
            for index, group in enumerate(optimizer.param_groups):
                metadata.append((optimizer_name, index, tuple((key, repr(value)) for key, value in group.items() if key != 'params')))
                for p_index, parameter in enumerate(group['params']):
                    name = f'{optimizer_name}/{index}/{p_index}'
                    tensors[name] = parameter
                    if parameter.grad is not None:
                        tensors[name+'/grad'] = parameter.grad
                    for key, value in optimizer.state.get(parameter, {}).items():
                        if torch.is_tensor(value):
                            tensors[name+'/state/'+key] = value
                        else:
                            metadata.append((name, key, repr(value)))
        for name in ('max_radii2D', 'xyz_gradient_accum', 'denom', 'tmp_radii'):
            value = getattr(model, name, None)
            if torch.is_tensor(value):
                tensors[name] = value
            else:
                metadata.append((name, repr(value)))
        return tensors, metadata

    @torch.no_grad()
    def evaluate(model, stage, event):
        if not args.observe:
            return
        scene = context['scene']
        test = sorted(scene.getTestCameras(), key=lambda camera: camera.image_name)
        selected = [test[int(i)] for i in np.linspace(0, len(test)-1, 8, dtype=int)]
        assert len({camera.image_name for camera in selected}) == 8
        assert not ({camera.image_name for camera in selected} & {camera.image_name for camera in scene.getTrainCameras()})
        before_rng, before_versions = rng_state(), versions(model)
        if args.exact_state:
            current, state_metadata = live_state(model)
            snapshots = {name: tensor.detach().clone() for name, tensor in current.items()}
            assert all(tensor.data_ptr() != current[name].data_ptr() for name, tensor in snapshots.items() if tensor.numel())
        context['evaluating'] = True
        for camera in selected:
            prediction = original_render(camera, model, *context['render_args'])['render']
            if camera.alpha_mask is not None:
                prediction = prediction * camera.alpha_mask.cuda()
            gt = camera.original_image.cuda()
            error = prediction - gt
            mse = error.square().mean().clamp_min(1e-30)
            values = dict(l1=float(error.abs().mean()), psnr=float(-10*torch.log10(mse)),
                          ssim=float(ssim(prediction, gt)))
            assert all(np.isfinite(value) for value in values.values())
            rows.append(dict(iteration=context['iteration'], event=event, stage=stage,
                view=camera.image_name, n_gaussians=len(model.get_xyz), **values))
        context['evaluating'] = False
        assert before_versions == versions(model), 'Evaluation mutated parameter/gradient/state tensors'
        assert same_rng(before_rng, rng_state()), 'Evaluation consumed training RNG'
        if args.exact_state:
            current, after_metadata = live_state(model)
            assert state_metadata == after_metadata and current.keys() == snapshots.keys()
            for name, tensor in current.items():
                assert torch.equal(tensor, snapshots[name]), f'Evaluation changed exact state: {name}'
        invariant_checks.append(dict(iteration=context['iteration'], stage=stage, event=event,
            pass_checks=True, exact_state=args.exact_state,
            exact_tensor_count=len(snapshots) if args.exact_state else 0))
        with (out/'observations.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'OBSERVED iteration={context["iteration"]} stage={stage} event={event}', flush=True)

    def render(camera, *positional, **keywords):
        if not context['evaluating']:
            cameras.append(camera.image_name)
        return original_render(camera, *positional, **keywords)

    def report(tb_writer, iteration, l1, loss, loss_func, elapsed, testing_iterations,
               scene, render_func, render_args, train_test_exp):
        context.update(iteration=iteration, scene=scene, render_args=render_args)
        # The official report has no scheduled evaluations in either branch.
        if args.observe and events and iteration == events[-1]['iteration'] + 21:
            evaluate(scene.gaussians, 'after20', events[-1]['iteration'])

    def clone(model, *positional, **keywords):
        n = len(model.get_xyz)
        assert not keywords
        value, witness = clone_with_checks(model, *positional, native_clone=original_clone,
                                          corrected=args.method != 'native')
        witness_path = out/'operator_witness'/f'clone_{context["iteration"]}.pth'
        torch.save(witness, witness_path)
        method_checks.append(dict(iteration=context['iteration'], operator='clone',
            selected=len(witness['ids']), corrected=witness['corrected'], status=witness['status'], witness=str(witness_path)))
        context['counts']['cloned_parents'] = len(model.get_xyz)-n
        ids = witness['ids'].to(model.get_xyz.device)
        context['clone_parent'][ids] = True
        context['source_id'] = torch.cat((context['source_id'], context['source_id'][ids]))
        context['clone_parent'] = torch.cat((context['clone_parent'], torch.zeros(len(ids), dtype=torch.bool, device=ids.device)))
        context['lineage'] = torch.cat((context['lineage'], torch.ones(
            len(model.get_xyz)-n, dtype=torch.uint8, device=model.get_xyz.device)))
        assert len(context['lineage']) == len(model.get_xyz)
        if args.operator_stages:
            evaluate(model, 'after_clone', context['iteration'])
        return value

    def split(model, *positional, **keywords):
        n = len(model.get_xyz)
        context['in_split'] = True
        if args.method == 'clone_corrected_las':
            witness = las_with_checks(model, *positional, build_rotation=build_rotation, **keywords)
            witness_path = out/'operator_witness'/f'las_{context["iteration"]}.pth'
            torch.save(witness, witness_path)
            method_checks.append(dict(iteration=context['iteration'], operator='las',
                selected=len(witness['ids']), status=witness['status'], witness=str(witness_path)))
            value = None
        else:
            value = original_split(model, *positional, **keywords)
        context['in_split'] = False
        # Installed default N=2: net increase equals the number of removed parents.
        context['counts']['split_parents'] = len(model.get_xyz)-n
        if args.operator_stages:
            evaluate(model, 'after_split', context['iteration'])
        return value

    def prune(model, *positional, **keywords):
        n = len(model.get_xyz)
        frozen = None
        assert not keywords and len(positional) == 1
        mask = positional[0]
        assert mask.dtype == torch.bool and mask.shape == (n,)
        if context['in_split']:
            # Both installed native split and isolated LAS append two blocks of
            # children, then delete selected parents through this exact mask.
            prior_n = len(context['lineage'])
            assert n >= prior_n and not mask[prior_n:].any()
            assert n-prior_n == 2*int(mask[:prior_n].sum())
            parents = mask[:prior_n]
            torch.save(dict(event=context['iteration'], removed_source_id=context['source_id'][parents].cpu(),
                removed_lineage=context['lineage'][parents].cpu(), actual_internal_mask=mask.cpu(),
                scope='Split-internal parent removal, separate from final prune.'),
                out/'operator_witness'/f'split_parent_remove_{context["iteration"]}.pth')
            context['source_id'] = torch.cat((context['source_id'], context['source_id'][parents].repeat(2)))
            context['clone_parent'] = torch.cat((context['clone_parent'], torch.zeros(n-prior_n, dtype=torch.bool, device=mask.device)))
            context['lineage'] = torch.cat((context['lineage'], torch.full(
                (n-prior_n,), 2, dtype=torch.uint8, device=mask.device)))
        elif context['counts'] is not None:
            assert len(context['lineage']) == n
            contract = context['prune_contract']
            opacity = model.get_opacity.detach().flatten()
            scale = model.get_scaling.detach().max(dim=1).values
            opacity_reason = opacity < contract['min_opacity']
            screen_reason = torch.zeros_like(mask)
            world_reason = torch.zeros_like(mask)
            if contract['max_screen_size']:
                screen_reason = model.max_radii2D > contract['max_screen_size']
                world_reason = scale > .1*contract['extent']
            assert torch.equal(mask, opacity_reason | screen_reason | world_reason)
            witness = dict(event=context['iteration'], n_before=n,
                lineage=context['lineage'].cpu(), actual_prune_mask=mask.cpu(),
                source_id=context['source_id'].cpu(), clone_parent=context['clone_parent'].cpu(),
                birth_event=torch.where(context['lineage'] == 0, 0, context['iteration']).cpu(),
                opacity=opacity.cpu(), max_scale=scale.cpu(), max_radii2D=model.max_radii2D.detach().cpu(),
                opacity_reason=opacity_reason.cpu(), screen_reason=screen_reason.cpu(),
                world_reason=world_reason.cpu(), pruned_alpha=opacity[mask].cpu(),
                pruned_max_scale=scale[mask].cpu(), contract=contract,
                codes={'0':'pre-event survivor','1':'clone born in this event','2':'split child born in this event'},
                scope='Actual final prune membership; overlapping reasons; event-local birth groups only. No counterfactual quality attribution.')
            torch.save(witness, out/'operator_witness'/f'prune_{context["iteration"]}.pth')
            if context['iteration'] == 3700:
                assert context['iteration'] == 3700
                before_rng, before_versions = rng_state(), versions(model)
                live, metadata = live_state(model)
                # Independent CPU copies include live gradients, Adam, ADC and exposure.
                snapshots = {name: tensor.detach().cpu().clone() for name, tensor in live.items()}
                fields = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
                test = sorted(context['scene'].getTestCameras(), key=lambda c: c.image_name)
                selected = [test[int(i)].image_name for i in np.linspace(0, len(test)-1, 8, dtype=int)]
                render_args = context['render_args']
                assert len(render_args) == 6 and render_args[4] is None and render_args[5] is False
                frozen = dict(event=context['iteration'], method=args.method, seed=args.seed,
                    raw={name: getattr(model, name).detach().cpu().clone() for name in fields},
                    active_sh_degree=model.active_sh_degree,
                    exposure=model._exposure.detach().cpu().clone(),
                    exposure_mapping=dict(model.exposure_mapping), pretrained_exposures=model.pretrained_exposures,
                    dataset=vars(dataset), optimization=vars(optimization),
                    pipeline=vars(render_args[0]), background=render_args[1].detach().cpu().clone(),
                    scaling_modifier=render_args[2], separate_sh=render_args[3],
                    override_color=None, use_trained_exp=render_args[5],
                    views=selected, actual_prune_mask=mask.cpu().clone(),
                    lineage=context['lineage'].cpu().clone(), source_id=context['source_id'].cpu().clone(),
                    n_before=n, scope='Rendering-only frozen state; not a resumable training checkpoint.')
                torch.save(frozen, out/'frozen_prune3700.pth')
                after, after_metadata = live_state(model)
                assert metadata == after_metadata and snapshots.keys() == after.keys()
                assert all(torch.equal(t.detach().cpu(), snapshots[name]) for name,t in after.items())
                assert same_rng(before_rng, rng_state()) and before_versions == versions(model)
                del snapshots, live, after
        value = original_prune(model, *positional, **keywords)
        if frozen is not None:
            keep = ~frozen['actual_prune_mask']
            assert len(model.get_xyz) == int(keep.sum())
            for name, raw in frozen['raw'].items():
                assert torch.equal(getattr(model, name).detach().cpu(), raw[keep]), name
            (out/'FROZEN_CAPTURE_CHECK.json').write_text(json.dumps(dict(status='pass',
                event=3700, n_before=n, n_after=len(model.get_xyz),
                capture_full_state_and_rng_unchanged=True,
                actual_post_prune_raw_parameters_exact_subset=True,
                scope='Same-run capture nonmutation and actual row-order subset; rendering snapshot only.'), indent=2))
            del frozen
        if context['counts'] is not None:
            context['lineage'] = context['lineage'][~mask]
            context['source_id'] = context['source_id'][~mask]
            context['clone_parent'] = context['clone_parent'][~mask]
            assert len(context['lineage']) == len(model.get_xyz)
        if not context['in_split'] and context['counts'] is not None:
            context['counts']['pruned_points'] += n-len(model.get_xyz)
            if args.operator_stages:
                evaluate(model, 'after_prune', context['iteration'])
        return value

    def adc(model, *positional, **keywords):
        iteration = context['iteration']
        evaluate(model, 'before', iteration)
        counts = dict(iteration=iteration, n_before=len(model.get_xyz), cloned_parents=0,
                      split_parents=0, pruned_points=0)
        context['counts'] = counts
        bound = inspect.signature(original_adc).bind(model, *positional, **keywords)
        context['prune_contract'] = {key: float(bound.arguments[key]) if bound.arguments[key] is not None else None
            for key in ('min_opacity', 'extent', 'max_screen_size')}
        context['lineage'] = torch.zeros(len(model.get_xyz), dtype=torch.uint8, device=model.get_xyz.device)
        context['source_id'] = torch.arange(len(model.get_xyz), device=model.get_xyz.device)
        context['clone_parent'] = torch.zeros(len(model.get_xyz), dtype=torch.bool, device=model.get_xyz.device)
        value = original_adc(model, *positional, **keywords)
        counts['n_after'] = len(model.get_xyz)
        assert counts['n_after'] == counts['n_before']+counts['cloned_parents']+counts['split_parents']-counts['pruned_points']
        context['counts'] = None
        context['lineage'] = None
        events.append(counts)
        evaluate(model, 'after', iteration)
        (out/'events.json').write_text(json.dumps(events, indent=2))
        return value

    native.render, native.training_report = render, report
    native.Scene = scene_factory
    native.TENSORBOARD_FOUND = False
    native.network_gui.try_connect = lambda: None
    model_class.densify_and_prune = adc
    model_class.densify_and_clone = clone
    model_class.densify_and_split = split
    model_class.prune_points = prune
    model_class.update_learning_rate = update_lr
    native.safe_state(False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    native.training(dataset, optimization, pipeline, [], [],
                    [args.iterations], args.checkpoint, -1)
    assert len(cameras) == args.iterations-3500
    (out/'summary.json').write_text(json.dumps(dict(complete=True, observed=args.observe,
        iterations=args.iterations, source_iteration=3500, camera_sequence=cameras,
        events=events, invariant_checks=invariant_checks,
        resume_checks=resume_checks, method=args.method, seed=args.seed, method_checks=method_checks,
        final_checkpoint=str(out/f'chkpnt{args.iterations}.pth'), args=vars(args),
        scope='Installed native training loop, resumed fixed seed; all normal ADC enabled. Exact-state checks prove immediate observational non-mutation, not reproduction of an unobserved trajectory. Sequential operator deltas are conditional on preceding operations. Final iteration skips optimizer step exactly as native. RNG at checkpoint is reconstructed, not historical replay.'), indent=2))
    print('NATIVE_OBSERVATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
