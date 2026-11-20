"""Wrap installed training() without editing its loop; observe normal ADC."""
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
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from utils.loss_utils import ssim

    parser = argparse.ArgumentParser(parents=[pre])
    mp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--exact_state', action='store_true')
    parser.add_argument('--operator_stages', action='store_true')
    args = parser.parse_args()
    assert args.eval and args.optimizer_type == 'default'
    assert not args.train_test_exp and not args.random_background and not args.depths
    assert not args.operator_stages or (args.observe and args.exact_state)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    args.model_path = str(out)
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
    rows, cameras, events, invariant_checks = [], [], [], []
    context = dict(iteration=None, evaluating=False, in_split=False, counts=None, scene=None, render_args=None)

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
        value = original_clone(model, *positional, **keywords)
        context['counts']['cloned_parents'] = len(model.get_xyz)-n
        if args.operator_stages:
            evaluate(model, 'after_clone', context['iteration'])
        return value

    def split(model, *positional, **keywords):
        n = len(model.get_xyz)
        context['in_split'] = True
        value = original_split(model, *positional, **keywords)
        context['in_split'] = False
        # Installed default N=2: net increase equals the number of removed parents.
        context['counts']['split_parents'] = len(model.get_xyz)-n
        if args.operator_stages:
            evaluate(model, 'after_split', context['iteration'])
        return value

    def prune(model, *positional, **keywords):
        n = len(model.get_xyz)
        value = original_prune(model, *positional, **keywords)
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
        value = original_adc(model, *positional, **keywords)
        counts['n_after'] = len(model.get_xyz)
        assert counts['n_after'] == counts['n_before']+counts['cloned_parents']+counts['split_parents']-counts['pruned_points']
        context['counts'] = None
        events.append(counts)
        evaluate(model, 'after', iteration)
        (out/'events.json').write_text(json.dumps(events, indent=2))
        return value

    native.render, native.training_report = render, report
    native.TENSORBOARD_FOUND = False
    native.network_gui.try_connect = lambda: None
    model_class.densify_and_prune = adc
    model_class.densify_and_clone = clone
    model_class.densify_and_split = split
    model_class.prune_points = prune
    native.safe_state(False)
    native.training(mp.extract(args), op.extract(args), pp.extract(args), [], [],
                    [args.iterations], args.checkpoint, -1)
    assert len(cameras) == args.iterations-3500
    (out/'summary.json').write_text(json.dumps(dict(complete=True, observed=args.observe,
        iterations=args.iterations, source_iteration=3500, camera_sequence=cameras,
        events=events, invariant_checks=invariant_checks,
        final_checkpoint=str(out/f'chkpnt{args.iterations}.pth'), args=vars(args),
        scope='Installed native training loop, resumed fixed seed; all normal ADC enabled. Exact-state checks prove immediate observational non-mutation, not reproduction of an unobserved trajectory. Sequential operator deltas are conditional on preceding operations. Final iteration skips optimizer step exactly as native. RNG at checkpoint is reconstructed, not historical replay.'), indent=2))
    print('NATIVE_OBSERVATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
