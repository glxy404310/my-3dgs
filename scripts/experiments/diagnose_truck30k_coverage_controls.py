"""Fixed coverage controls for frozen contribution pruning. No optimizer."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diagnose_truck30k_frozen_compression import FIELDS, camera_sets, dump, model_hash, sha, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, args.repo)
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    from PIL import Image, ImageDraw

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    dataset = SimpleNamespace(source_path=args.source, model_path=str(args.model), images='images', depths='',
        resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    torch.manual_seed(20261022)
    teacher = GaussianModel(3, 'default')
    scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False)
    for field in FIELDS:
        getattr(teacher, field).requires_grad_(False)
    assert teacher.optimizer is None
    before = model_hash(teacher)
    ply = args.model/'point_cloud/iteration_30000/point_cloud.ply'
    ply_hash = sha(ply)
    previous_config = json.loads((args.previous/'CONFIG.json').read_text())
    assert ply_hash == previous_config['teacher_sha256']
    train = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    calibration, virtual, meta = camera_sets(train, scene.cameras_extent)
    assert [train[i].image_name for i in calibration] == previous_config['calibration']
    n = len(teacher._xyz)
    selections = dict(uniform24=np.linspace(0, len(train)-1, 24, dtype=int).tolist(),
        random24=np.random.default_rng(20261022).choice(len(train), 24, replace=False).tolist(),
        all219=list(range(len(train))))
    dump(out/'CONFIG.json', dict(teacher_sha256=ply_hash, script_sha256=sha(Path(__file__)),
        training_performed=False, parameter_updates=0, keep_fraction=.5,
        selection={k: [train[i].image_name for i in v] for k, v in selections.items()},
        previous_run=str(args.previous), virtual_views=meta, resolution=1))
    bg = torch.zeros(3, device='cuda')
    scores = {key: torch.zeros(n, device='cuda', dtype=torch.float64) for key in selections}
    arm_time = {key: 0. for key in selections}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def deadline():
        assert time.perf_counter()-started < 900, 'Budget exceeded'

    torch.cuda.synchronize()
    score_start = time.perf_counter()
    for i, cam in enumerate(train):
        deadline()
        torch.cuda.synchronize()
        t = time.perf_counter()
        color = torch.full((n, 3), .25, device='cuda', requires_grad=True)
        image = render(cam, teacher, pipe, bg, override_color=color)['render']
        weights = torch.autograd.grad(image.sum()/3, color)[0].sum(1).detach()
        rel = abs(float(weights.sum())-float(image.detach().sum()/.75))/max(1., float(weights.sum()))
        assert rel < 2e-5 and float(weights.min()) >= -1e-7
        for key, indices in selections.items():
            if i in indices:
                scores[key] += weights.double()/(cam.image_height*cam.image_width)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-t
        for key, indices in selections.items():
            if i in indices:
                arm_time[key] += elapsed
        del color, image, weights
        if i % 40 == 0:
            print(f'SCORED {i+1}/{len(train)}', flush=True)
    torch.cuda.synchronize()
    score_seconds = time.perf_counter()-score_start
    indices = {'original24': np.load(args.previous/'keep_indices.npy')}
    for key, score in scores.items():
        keep = torch.sort(torch.argsort(score, descending=True, stable=True)[:n//2]).values
        indices[key] = keep.cpu().numpy()
        np.save(out/f'{key}_indices.npy', indices[key])
        np.save(out/f'{key}_scores.npy', score.cpu().numpy())
    del scores
    pairs = [('calibration' if i in calibration else 'real_noncalibration_train', c) for i, c in enumerate(train)]
    pairs += [('real_test', c) for c in test] + virtual
    rows = []
    images = {}
    saved_names = {'000017.jpg', '000016.jpg', 'virtual_check_e05_t75'}
    previous_rows = {r['view']: r for r in csv.DictReader((args.previous/'METRICS.csv').open())}
    evaluation_times = {}
    with (out/'METRICS.csv').open('w', newline='') as f:
        writer = None
        for arm, keep_np in indices.items():
            deadline()
            keep = torch.tensor(keep_np, device='cuda', dtype=torch.long)
            student = GaussianModel(3, 'default')
            student.active_sh_degree = teacher.active_sh_degree
            for field in FIELDS:
                setattr(student, field, torch.nn.Parameter(getattr(teacher, field)[keep].clone(), requires_grad=False))
                assert torch.equal(getattr(student, field), getattr(teacher, field)[keep])
            assert student.optimizer is None
            student_hash = model_hash(student)
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                for group, cam in pairs:
                    deadline()
                    a = render(cam, teacher, pipe, bg)['render']
                    b = render(cam, student, pipe, bg)['render']
                    mse = float((a-b).square().mean())
                    row = dict(arm=arm, group=group, view=cam.image_name, mse_ts=mse,
                        teacher_psnr_gt='', student_psnr_gt='', delta_psnr_gt='',
                        teacher_ssim_gt='', student_ssim_gt='', delta_ssim_gt='')
                    if arm == 'original24':
                        assert abs(mse-float(previous_rows[cam.image_name]['mse_ts'])) < 1e-9
                    if hasattr(cam, 'original_image'):
                        gt = cam.original_image.cuda()
                        for label, im in [('teacher', a), ('student', b)]:
                            row[f'{label}_psnr_gt'] = float(-10*torch.log10((im-gt).square().mean().clamp_min(1e-30)))
                            row[f'{label}_ssim_gt'] = float(ssim(im, gt))
                        row['delta_psnr_gt'] = row['student_psnr_gt']-row['teacher_psnr_gt']
                        row['delta_ssim_gt'] = row['student_ssim_gt']-row['teacher_ssim_gt']
                    if cam.image_name in saved_names:
                        def png(im):
                            return Image.fromarray((im.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8'))
                        images[(cam.image_name, arm)] = png(b)
                        images[(cam.image_name, 'teacher')] = png(a)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    rows.append(row)
            torch.cuda.synchronize()
            evaluation_times[arm] = time.perf_counter()-t
            assert student_hash == model_hash(student)
            del student, keep
            f.flush()
            print(f'EVALUATED {arm} {len(pairs)} views', flush=True)
    assert before == model_hash(teacher) and ply_hash == sha(ply)
    agg = {}
    for arm in indices:
        arm_rows = [r for r in rows if r['arm'] == arm]
        by_view = {r['view']: r for r in arm_rows}
        groups = {g: summary([r['mse_ts'] for r in arm_rows if r['group'] == g]) for g, _ in pairs}
        real = [r for r in arm_rows if r['group'] == 'real_test']
        spikes = [v['view'] for v in meta if by_view[v['view']]['mse_ts'] > 1.25*max(by_view[e]['mse_ts'] for e in v['endpoints'])]
        agg[arm] = dict(groups=groups, virtual_above_125pct_endpoints=spikes,
            mean_delta_psnr_gt=float(np.mean([r['delta_psnr_gt'] for r in real])),
            mean_delta_ssim_gt=float(np.mean([r['delta_ssim_gt'] for r in real])),
            original24_indices_jaccard=float(len(set(keep_np for keep_np in indices[arm]) & set(indices['original24']))/len(set(indices[arm]) | set(indices['original24']))))
    for name in saved_names:
        labels = ['teacher', *indices]
        w, h = images[(name, 'teacher')].size
        panel = Image.new('RGB', (w*len(labels), h+32), 'white')
        draw = ImageDraw.Draw(panel)
        for i, label in enumerate(labels):
            panel.paste(images[(name, label)], (i*w, 32))
            draw.text((i*w+8, 8), label, fill='black')
        panel.save(out/f'{name}_coverage.png')
    dump(out/'RESULTS.json', dict(arms=agg, training_performed=False, parameter_updates=0,
        teacher_unchanged=True, exact_student_subsets=True, original_replay_matches=True,
        actual_shared_score_seconds=score_seconds, attributed_score_seconds=arm_time,
        evaluation_seconds=evaluation_times, elapsed_gpu_stage_seconds=time.perf_counter()-started,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated()))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
