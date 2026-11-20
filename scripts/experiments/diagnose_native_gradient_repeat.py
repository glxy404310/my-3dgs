"""Native-only fixed-input gradient repeatability. Never updates model parameters."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_joint_rounding_full import sha, model_hash


FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    for key in ('repo', 'source', 'model', 'contract', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    process_start = time.perf_counter()
    result = dict(decision='INCONCLUSIVE', training_performed=False, optimizer_created=False,
                  parameter_updates=0, tool_valid=False)
    paths = {i: args.model/f'point_cloud/iteration_{i}/point_cloud.ply' for i in (7000, 30000)}
    before = {str(i): sha(path) for i, path in paths.items()}
    assert before['30000'] == '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
    config = dict(source_hashes=before, script_sha256=sha(Path(__file__)), contract_sha256=sha(args.contract),
        iterations=[7000, 30000], resolution=1, image_size=[979, 546], train_ranks=[0, 72, 145, 218],
        repeats=8, losses=['mse', 'training'], fields=FIELDS, native_renderer_only=True,
        high_contribution_mass=.9, minimum_repeat_ratio=.01, minimum_large_noise_mass=.1,
        minimum_noise_to_neighbor_ratio=.1, minimum_distinct_views=2, zero_gradient_floor_multiplier=1e-6,
        precision_bias_checked=False, kappa_checked=False, optimizer_update_impact_checked=False)
    dump(out/'CONFIG.json', config)
    sys.path.insert(0, str(args.repo))
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from fused_ssim import fused_ssim

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(min(1., 10*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(20261023)
    gpu_stage_start = time.perf_counter()

    def timeout():
        dump(out/'RESULTS.json', dict(result, tool_failure='300 second stage timeout',
            integrity_complete=False, elapsed_seconds=time.perf_counter()-process_start))
        os._exit(2)

    watchdog = threading.Timer(300, timeout)
    watchdog.daemon = True
    watchdog.start()

    def budget():
        assert time.perf_counter()-gpu_stage_start < 300, 'GPU process stage budget exceeded'
        assert torch.cuda.max_memory_allocated() <= 10*1024**3, '10 GiB allocated budget exceeded'

    try:
        dataset = SimpleNamespace(source_path=str(args.source), model_path=str(args.model), images='images',
            depths='', resolution=1, white_background=False, train_test_exp=False, data_device='cpu',
            eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        model = GaussianModel(3, 'default')
        load_start = time.perf_counter()
        scene = Scene(dataset, model, load_iteration=7000, shuffle=False)
        load_seconds = time.perf_counter()-load_start
        train = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
        assert len(train) == 219
        ranks = [0, 72, 145, 218]
        config['main_views'] = [train[r].image_name for r in ranks]
        config['neighbor_views'] = [train[r+1 if r < 218 else r-1].image_name for r in ranks]
        dump(out/'CONFIG.json', config)
        bg = torch.zeros(3, device='cuda')
        all_summary, repeat_rows, states, contribution_rows = [], [], [], []

        def render_image(cam, colors=None):
            im = render(cam, model, pipe, bg, override_color=colors,
                        separate_sh=False, use_trained_exp=False)['render']
            assert tuple(im.shape) == (3, 546, 979) and bool(torch.isfinite(im).all())
            if cam.alpha_mask is not None:
                im = im*cam.alpha_mask.cuda()
            return im

        def gradients(cam, kind):
            im = render_image(cam)
            gt = cam.original_image.cuda()
            loss = (im-gt).square().mean() if kind == 'mse' else .8*(im-gt).abs().mean()+.2*(1-fused_ssim(im[None], gt[None]))
            g = tuple(x.detach() for x in torch.autograd.grad(loss, [getattr(model, f) for f in FIELDS]))
            assert all(bool(torch.isfinite(x).all()) for x in g)
            budget()
            return im.detach(), float(loss.detach()), g

        def squared_per_point(g):
            return g.double().reshape(len(g), -1).square().sum(dim=1)

        def difference_per_point(g, ref):
            return (g.double()-ref.double()).reshape(len(g), -1).square().sum(dim=1)

        for iteration in (7000, 30000):
            if iteration != 7000:
                model.load_ply(str(paths[iteration]), use_train_test_exp=False)
            for field in FIELDS:
                getattr(model, field).requires_grad_(True)
            assert model.optimizer is None
            initial_parameters = model_hash(model)
            n = len(model._xyz)
            for rank in ranks:
                cam, neighbor = train[rank], train[rank+1 if rank < 218 else rank-1]
                colors = torch.full((n, 3), .1, device='cuda', requires_grad=True)
                im = render_image(cam, colors)
                contribution = torch.autograd.grad(im.sum()/3, colors)[0].detach().sum(dim=1).double()
                assert bool(torch.isfinite(contribution).all()) and bool((contribution >= 0).all())
                order = torch.argsort(contribution, descending=True, stable=True)
                total = contribution.sum()
                assert float(total) > 0
                count = int(torch.searchsorted(contribution[order].cumsum(0), .9*total))+1
                keep = order[:count]
                mass = contribution[keep].sum()/total
                weights = contribution[keep]/contribution[keep].sum()
                selected_path = out/f'{iteration}_{cam.image_name}_mass90_indices.npy'
                np.save(selected_path, keep.cpu().numpy())
                contribution_rows.append(dict(iteration=iteration, view=cam.image_name, point_count=n,
                    mass90_point_count=count, actual_selected_mass=float(mass), integrated_Talpha=float(total),
                    indices_sha256=sha(selected_path)))
                del colors, im, order, contribution, total
                for kind in ('mse', 'training'):
                    ref_image, ref_loss, refs = gradients(cam, kind)
                    ref_energies = [squared_per_point(g) for g in refs]
                    selected_ref = [e[keep].clone() for e in ref_energies]
                    global_ref = [float(e.sum()) for e in ref_energies]
                    del ref_energies
                    noise = [torch.zeros(count, device='cuda', dtype=torch.float64) for _ in FIELDS]
                    for repeat in range(1, 8):
                        image, loss, repeated = gradients(cam, kind)
                        assert torch.equal(image, ref_image) and loss == ref_loss, 'Fixed-input forward/loss changed'
                        for fi, (field, g, ref) in enumerate(zip(FIELDS, repeated, refs)):
                            diffs = difference_per_point(g, ref)
                            local = diffs[keep]
                            noise[fi] += local
                            repeat_rows.append(dict(iteration=iteration, view=cam.image_name, loss=kind,
                                field=field, repeat=repeat, global_reference_energy=global_ref[fi],
                                global_difference_energy=float(diffs.sum()),
                                weighted_selected_reference_energy=float(torch.dot(weights, selected_ref[fi])),
                                weighted_selected_difference_energy=float(torch.dot(weights, local)),
                                forward_exact=True, scalar_loss=loss))
                            del diffs, local
                        del repeated, image
                    neighbor_image, neighbor_loss, neighbor_gradients = gradients(neighbor, kind)
                    for fi, (field, ref, other) in enumerate(zip(FIELDS, refs, neighbor_gradients)):
                        reference_energy = float(torch.dot(weights, selected_ref[fi]))
                        repeat_energy = float(torch.dot(weights, noise[fi]/7))
                        neighbor_diff = difference_per_point(other, ref)[keep]
                        neighbor_energy = float(torch.dot(weights, neighbor_diff))
                        denominator = torch.maximum(selected_ref[fi], torch.full_like(selected_ref[fi], reference_energy*1e-12))
                        point_ratio = torch.sqrt((noise[fi]/7)/denominator.clamp_min(1e-300))
                        fraction = float(weights[point_ratio >= .01].sum())
                        repeat_ratio = float(np.sqrt(repeat_energy/max(reference_energy, 1e-300)))
                        neighbor_ratio = float(np.sqrt(repeat_energy/max(neighbor_energy, 1e-300)))
                        row = dict(iteration=iteration, view=cam.image_name, neighbor=neighbor.image_name, loss=kind,
                            field=field, weighted_reference_energy=reference_energy,
                            weighted_mean_repeat_difference_energy=repeat_energy,
                            weighted_neighbor_difference_energy=neighbor_energy,
                            weighted_rms_repeat_relative_to_reference=repeat_ratio,
                            weighted_rms_repeat_relative_to_neighbor_change=neighbor_ratio,
                            selected_mass_fraction_point_noise_ge_1pct=fraction,
                            mass90_point_count=count, total_point_count=n, reference_loss=ref_loss,
                            neighbor_loss=neighbor_loss,
                            per_view_signal=repeat_ratio >= .01 and fraction >= .1 and neighbor_ratio >= .1)
                        assert all(np.isfinite(x) for x in row.values() if isinstance(x, float))
                        all_summary.append(row)
                        del neighbor_diff, denominator, point_ratio
                    del refs, ref_image, neighbor_gradients, neighbor_image, noise, selected_ref
                    print(json.dumps({'iteration': iteration, 'view': cam.image_name, 'loss': kind,
                        'largest_weighted_repeat_ratio': max(r['weighted_rms_repeat_relative_to_reference'] for r in all_summary[-6:])}), flush=True)
                    budget()
                del keep, weights
            final_parameters = model_hash(model)
            assert initial_parameters == final_parameters and model.optimizer is None
            states.append(dict(iteration=iteration, parameters_before=initial_parameters, parameters_after=final_parameters,
                               point_count=n, optimizer_created=False))
        grouped = []
        for iteration in (7000, 30000):
            for kind in ('mse', 'training'):
                for field in FIELDS:
                    signals = [r['view'] for r in all_summary if r['iteration'] == iteration and r['loss'] == kind
                               and r['field'] == field and r['per_view_signal']]
                    grouped.append(dict(iteration=iteration, loss=kind, field=field, signal_views=signals,
                                        gate=len(signals) >= 2))
        after = {str(i): sha(path) for i, path in paths.items()}
        assert after == before
        for name, rows in [('SUMMARY.csv', all_summary), ('REPEATS.csv', repeat_rows), ('CONTRIBUTION.csv', contribution_rows)]:
            with (out/name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        result.update(tool_valid=True, decision='NEEDS_HIGH_PRECISION_CAUSAL_CHECK' if any(r['gate'] for r in grouped)
            else 'STOP_CURRENT_NONDETERMINISM_DRIVEN_IMPLEMENTATION',
            checkpoint_states=states, source_hashes_after=after, grouped_gate=grouped,
            summary_rows=len(all_summary), repeat_rows=len(repeat_rows),
            largest_weighted_repeat_ratio=max(r['weighted_rms_repeat_relative_to_reference'] for r in all_summary),
            largest_noise_to_neighbor_ratio=max(r['weighted_rms_repeat_relative_to_neighbor_change'] for r in all_summary),
            largest_selected_mass_fraction_noise_ge_1pct=max(r['selected_mass_fraction_point_noise_ge_1pct'] for r in all_summary),
            scope='Only repeat-order nondeterminism on these inputs. Systematic precision bias, kappa, optimizer updates, and quality untested.',
            camera_load_seconds=load_seconds, gpu_process_stage_wall_seconds=time.perf_counter()-gpu_stage_start,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), integrity_complete=True)
    except Exception as exc:
        result.update(decision='INCONCLUSIVE', tool_valid=False, tool_failure=f'{type(exc).__name__}: {exc}',
                      integrity_complete=False)
    finally:
        watchdog.cancel()
        result['elapsed_process_seconds'] = time.perf_counter()-process_start
        dump(out/'RESULTS.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['tool_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
