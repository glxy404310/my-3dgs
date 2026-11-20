"""Frozen pixel oracles and fixed controls; no learned repair or optimizer."""
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
from evaluate_joint_rounding_full import FIELDS, sha, model_hash


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    for key in ('repo', 'source', 'model', 'previous', 'coverage', 'contract', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(decision='INCONCLUSIVE', tool_valid=False, training_performed=False,
                  optimizer_created=False, optimizer_updates=0, fixed_control_interventions=True)
    inputs = dict(teacher=args.model/'point_cloud/iteration_30000/point_cloud.ply',
        original24=args.previous/'keep_indices.npy', all219=args.coverage/'all219_indices.npy',
        contract=args.contract, script=Path(__file__))
    before = {k: sha(v) for k, v in inputs.items()}
    assert before['teacher'] == '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
    dump(out/'CONFIG.json', dict(input_hashes=before, alpha_threshold=1e-4,
        fixed_gain=[1.05, 1.1, 1.25], fixed_opacity=[1.1, 1.25, 1.5], selection_uses_GT=False))

    def timeout():
        dump(out/'RESULTS.json', dict(result, tool_failure='300 second process budget exceeded'))
        os._exit(2)

    watchdog = threading.Timer(300, timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        sys.path.insert(0, str(args.repo))
        from scene import Scene, GaussianModel
        import gaussian_renderer
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., 10*1024**3/torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(20261023)
        dataset = SimpleNamespace(source_path=str(args.source), model_path=str(args.model), images='images',
            depths='', resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        teacher = GaussianModel(3, 'default')
        scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False)
        for field in FIELDS:
            getattr(teacher, field).requires_grad_(False)
        initial_teacher = model_hash(teacher)
        cameras = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
        old_rows = list(csv.DictReader((args.previous/'METRICS.csv').open()))
        coverage_rows = list(csv.DictReader((args.coverage/'METRICS.csv').open()))
        previous_base = {(r['arm'], r['view']): r for r in coverage_rows if r['group'] == 'real_test'}
        assert len(cameras) == 32 and [c.image_name for c in cameras] == sorted(r['view'] for r in old_rows if r['group'] == 'real_test')
        assert all((c.image_width, c.image_height) == (979, 546) for c in cameras)
        bg_black, bg_white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
        original_forward = gaussian_renderer.GaussianRasterizer.forward
        captured = {}

        def capture_forward(self, *a, **kw):
            outputs = original_forward(self, *a, **kw)
            captured['rgb'] = outputs[0]
            return outputs

        def raw(cam, model, background):
            with torch.no_grad():
                display = gaussian_renderer.render(cam, model, pipe, background)['render']
            rgb = captured.pop('rgb').detach()
            assert torch.equal(display, rgb.clamp(0, 1)) and bool(torch.isfinite(rgb).all())
            assert time.perf_counter()-started < 300 and torch.cuda.max_memory_allocated() <= 10*1024**3
            return rgb

        max_channel_T_delta = 0.
        alpha_roundoff_pixels = 0

        def color_alpha(cam, model):
            nonlocal max_channel_T_delta, alpha_roundoff_pixels
            black, white = raw(cam, model, bg_black), raw(cam, model, bg_white)
            transmission = white.double()-black.double()
            delta = float((transmission-transmission[:1]).abs().max())
            max_channel_T_delta = max(max_channel_T_delta, delta)
            assert delta <= 2e-5, 'Black/white transmission differs across channels'
            alpha = 1-transmission.mean(dim=0, keepdim=True)
            assert float(alpha.min()) >= -2e-5 and float(alpha.max()) <= 1+2e-5
            alpha_roundoff_pixels += int(((alpha < 0) | (alpha > 1)).sum())
            return black.double(), alpha.clamp(0, 1)

        rows, mask_rows, parameter_states = [], [], []
        gaussian_renderer.GaussianRasterizer.forward = capture_forward
        try:
            teacher_cache = {}
            for cam in cameras:
                c, a = color_alpha(cam, teacher)
                teacher_cache[cam.image_name] = (c.cpu(), a.cpu())
                del c, a
            for carrier in ('original24', 'all219'):
                keep = torch.tensor(np.load(inputs[carrier]), device='cuda', dtype=torch.long)
                assert len(keep) == 1025385 and len(torch.unique(keep)) == len(keep)
                student = GaussianModel(3, 'default')
                student.active_sh_degree = 3
                for field in FIELDS:
                    setattr(student, field, torch.nn.Parameter(getattr(teacher, field)[keep].detach().clone(), requires_grad=False))
                state_before = model_hash(student)
                base_opacity = student._opacity

                def record(cam, variant, output, teacher_rgb, gt):
                    display = output.float().clamp(0, 1)
                    mse = float((display-gt).square().mean())
                    teacher_mse = float((display-teacher_rgb.float().clamp(0, 1)).square().mean())
                    assert np.isfinite([mse, teacher_mse]).all()
                    if variant == 'base':
                        old = previous_base[(carrier, cam.image_name)]
                        assert abs(teacher_mse-float(old['mse_ts'])) <= 1e-9, 'Prior frozen student replay mismatch'
                    rows.append(dict(carrier=carrier, view=cam.image_name, variant=variant,
                        gt_mse=mse, teacher_student_mse=teacher_mse,
                        gt_psnr=float(-10*torch.log10((display-gt).square().mean().clamp_min(1e-30)))))

                for cam in cameras:
                    ct, at = (v.cuda() for v in teacher_cache[cam.image_name])
                    cs, a_s = color_alpha(cam, student)
                    gt = cam.original_image.cuda()
                    valid_s, valid_t = a_s > 1e-4, at > 1e-4
                    alpha_oracle = torch.where(valid_s, at*cs/a_s.clamp_min(1e-300), cs)
                    color_oracle = torch.where(valid_t, a_s*ct/at.clamp_min(1e-300), cs)
                    raw_sse = (cs-ct).square().sum(dim=0, keepdim=True)
                    mask_rows.append(dict(carrier=carrier, view=cam.image_name,
                        student_near_zero_fraction=float((~valid_s).double().mean()),
                        teacher_near_zero_fraction=float((~valid_t).double().mean()),
                        student_near_zero_teacher_difference_energy_fraction=float(raw_sse[~valid_s].sum()/raw_sse.sum().clamp_min(1e-300))))
                    for name, image in [('base', cs), ('alpha_oracle', alpha_oracle), ('color_oracle', color_oracle)]:
                        record(cam, name, image, ct, gt)
                    for gain in (1.05, 1.1, 1.25):
                        record(cam, 'gain_'+str(gain), gain*cs, ct, gt)
                    del ct, at, cs, a_s, gt, alpha_oracle, color_oracle
                for multiplier in (1.1, 1.25, 1.5):
                    alpha = (torch.sigmoid(base_opacity)*multiplier).clamp(0, 1)
                    student._opacity = torch.nn.Parameter(torch.log(alpha)-torch.log1p(-alpha), requires_grad=False)
                    for cam in cameras:
                        rgb = raw(cam, student, bg_black)
                        ct = teacher_cache[cam.image_name][0].cuda()
                        record(cam, 'opacity_'+str(multiplier), rgb, ct, cam.original_image.cuda())
                    student._opacity = base_opacity
                assert state_before == model_hash(student) and student.optimizer is None
                parameter_states.append(dict(carrier=carrier, before=state_before, after=model_hash(student), opacity_restored_exactly=True))
                del student, keep, base_opacity
                print('COMPLETE '+carrier, flush=True)
        finally:
            gaussian_renderer.GaussianRasterizer.forward = original_forward
        summary = {}
        for carrier in ('original24', 'all219'):
            summary[carrier] = {}
            for variant in sorted({r['variant'] for r in rows}):
                r = [x for x in rows if x['carrier'] == carrier and x['variant'] == variant]
                assert len(r) == 32
                summary[carrier][variant] = dict(mean_gt_mse=float(np.mean([x['gt_mse'] for x in r])),
                    tail4_gt_mse=float(np.mean(sorted((x['gt_mse'] for x in r), reverse=True)[:4])),
                    mean_teacher_mse=float(np.mean([x['teacher_student_mse'] for x in r])))
        carrier_summary = summary['all219']
        oracle, base = carrier_summary['alpha_oracle'], carrier_summary['base']
        baseline_rows = {r['view']: r for r in rows if r['carrier'] == 'all219' and r['variant'] == 'base'}
        nonworse = sum(r['gt_mse'] <= baseline_rows[r['view']]['gt_mse'] for r in rows
                       if r['carrier'] == 'all219' and r['variant'] == 'alpha_oracle')
        room = (oracle['mean_teacher_mse'] <= .8*base['mean_teacher_mse'] and oracle['mean_gt_mse'] <= .95*base['mean_gt_mse']
                and oracle['tail4_gt_mse'] <= base['tail4_gt_mse'] and nonworse >= 24)
        controls = {k: dict(mean_gt_5pct_better=oracle['mean_gt_mse'] <= .95*v['mean_gt_mse'],
                            tail4_nonworse=oracle['tail4_gt_mse'] <= v['tail4_gt_mse'])
                    for k, v in carrier_summary.items() if k.startswith(('gain_', 'opacity_'))}
        independent = all(v['mean_gt_5pct_better'] and v['tail4_nonworse'] for v in controls.values())
        assert model_hash(teacher) == initial_teacher and teacher.optimizer is None
        after = {k: sha(v) for k, v in inputs.items()}
        assert before == after
        for name, values in [('METRICS.csv', rows), ('MASKS.csv', mask_rows)]:
            with (out/name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        result.update(tool_valid=True, decision='NEEDS_PRACTICAL_REPAIR_AND_STRONGER_BASELINES' if room and independent else 'STOP_CURRENT_ALPHA_RECOVERY_MOTIVATION',
            alpha_room_gate=bool(room), independent_room_gate=bool(independent), all219_nonworse_views=nonworse,
            controls=controls, summary=summary, rows=len(rows), input_hashes_before=before, input_hashes_after=after,
            teacher_parameters_before=initial_teacher, teacher_parameters_after=model_hash(teacher), student_states=parameter_states,
            max_background_T_channel_delta=max_channel_T_delta, alpha_roundoff_pixels=alpha_roundoff_pixels,
            original_student_replay_mse_tolerance=1e-9,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            scope='Pixel oracle only, simple unrecovered pruning; no deployable repair, material opacity, or novelty claim.')
    except Exception as exc:
        result.update(decision='INCONCLUSIVE', tool_failure=f'{type(exc).__name__}: {exc}', tool_valid=False)
    finally:
        watchdog.cancel()
        result['elapsed_seconds'] = time.perf_counter()-started
        dump(out/'RESULTS.json', result)
    print(json.dumps({'decision':result['decision'], 'tool_valid':result['tool_valid'], 'tool_failure':result.get('tool_failure'),
                      'elapsed_seconds':result['elapsed_seconds']}), flush=True)
    return 0 if result['tool_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
